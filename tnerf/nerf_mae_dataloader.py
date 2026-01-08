"""
NeRF-MAE Data Loader for T-NeRF Training.

This module loads NeRF-MAE pretrain data (rgbsigma voxel grids) and renders
multi-view images for training VGGT to predict NeRF MLP parameters.

Data Flow:
1. Load rgbsigma voxel grid from .npz files
2. Generate camera poses around the scene
3. Render images using volume rendering from the voxel grid (based on render.py)
4. Return images and voxel data for training

Rendering Logic:
- Adapted from NeRF's render.py, replacing MLP queries with voxel grid sampling
- Uses trilinear interpolation for smooth voxel queries
- Supports stratified sampling, noise perturbation, and hierarchical sampling
"""

import os
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


def raw2outputs_grid(rgb, sigma, z_vals, rays_d, raw_noise_std=0., white_bkgd=True):
    """
    Adapted from render.py raw2outputs.
    
    Args:
        rgb: [num_rays, num_samples, 3]. RGB values from voxel grid.
        sigma: [num_rays, num_samples]. Density values from voxel grid.
        z_vals: [num_rays, num_samples]. Integration distances.
        rays_d: [num_rays, 3]. Direction of each ray.
        raw_noise_std: float. Standard deviation of noise added to density.
        white_bkgd: bool. If True, assume white background.
    
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map (inverse of depth).
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    # Compute distances between adjacent samples
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    # Add infinite distance for last sample (same as render.py)
    dists = torch.cat([dists, torch.tensor([1e10], device=dists.device).expand(dists[..., :1].shape)], -1)
    
    # Scale by ray direction magnitude (accounts for non-unit direction vectors)
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)
    
    # Add noise to density for regularization during training
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(sigma.shape, device=sigma.device) * raw_noise_std
    
    # Convert density to alpha using the formula: alpha = 1 - exp(-sigma * dist)
    # Using ReLU to ensure non-negative density (same as render.py)
    alpha = 1. - torch.exp(-F.relu(sigma + noise) * dists)
    
    # Compute transmittance and weights
    # T_i = prod_{j=1}^{i-1} (1 - alpha_j)
    # weights_i = T_i * alpha_i
    transmittance = torch.cumprod(
        torch.cat([torch.ones((alpha.shape[0], 1), device=alpha.device), 1. - alpha + 1e-10], -1),
        dim=-1
    )[:, :-1]
    weights = alpha * transmittance
    
    # Compute final RGB color as weighted sum
    rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)
    
    # Compute depth as weighted sum of distances
    depth_map = torch.sum(weights * z_vals, dim=-1)
    
    # Compute disparity (inverse depth)
    disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / (torch.sum(weights, -1) + 1e-10))
    
    # Accumulated opacity
    acc_map = torch.sum(weights, dim=-1)
    
    # White background composition
    if white_bkgd:
        rgb_map = rgb_map + (1. - acc_map[..., None])
    
    return rgb_map, disp_map, acc_map, weights, depth_map


def sample_pdf_grid(bins, weights, N_samples, det=False):
    """
    Hierarchical sampling based on weights (adapted from render.py/sampling.py).
    
    Sample from a piecewise-constant PDF defined by weights.
    
    Args:
        bins: [num_rays, num_bins]. Bin edges.
        weights: [num_rays, num_bins-1]. Weights for each bin.
        N_samples: int. Number of samples to draw.
        det: bool. If True, use deterministic sampling.
    
    Returns:
        samples: [num_rays, N_samples]. Sampled positions.
    """
    device = bins.device
    
    # Add small value to weights to prevent NaN
    weights = weights + 1e-5
    
    # Compute PDF and CDF
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)
    
    # Generate sample positions
    if det:
        u = torch.linspace(0., 1., steps=N_samples, device=device)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples], device=device)
    
    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.clamp(inds - 1, min=0)
    above = torch.clamp(inds, max=cdf.shape[-1] - 1)
    inds_g = torch.stack([below, above], dim=-1)
    
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)
    
    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])
    
    return samples


class NeRFMAEDataset(Dataset):
    """
    Dataset for loading NeRF-MAE pretrain data and rendering multi-view images.
    
    Each sample contains:
    - images: Rendered multi-view images [S, 3, H, W]
    - depth_maps: Rendered depth maps [S, H, W]
    - rgbsigma: Original voxel grid [D, H, W, 4]
    - camera_poses: Camera extrinsics for each view [S, 4, 4]
    - camera_intrinsics: Camera intrinsics [3, 3]
    - bbox_min, bbox_max: Scene bounding box
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        image_size: Tuple[int, int] = (256, 256),  # (H, W)
        num_views: int = 8,
        num_samples: int = 128,  # coarse samples per ray
        num_importance: int = 64,  # fine samples per ray (hierarchical sampling)
        perturb: float = 1.0,  # stratified sampling perturbation
        raw_noise_std: float = 0.0,  # noise added to density
        white_bkgd: bool = True,  # assume white background
        chunk: int = 1024 * 8,  # chunk size for batch rendering
        split_file: Optional[str] = None,
    ):
        """
        Initialize NeRF-MAE dataset.
        
        Args:
            data_dir: Directory containing features/*.npz files
            split: "train", "val", or "test"
            image_size: Target image size (H, W)
            num_views: Number of views to render per scene
            num_samples: Number of coarse samples per ray during rendering
            num_importance: Number of fine samples for hierarchical sampling
            perturb: Stratified sampling perturbation (0.0 = uniform, 1.0 = full random)
            raw_noise_std: Standard deviation of noise added to density
            white_bkgd: If True, composite over white background
            chunk: Chunk size for batched ray rendering
            split_file: Path to nerfmae_split.npz file
        """
        self.data_dir = Path(data_dir)
        self.features_dir = self.data_dir / "features"
        self.split = split
        self.image_size = image_size
        self.num_views = num_views
        self.num_samples = num_samples
        self.num_importance = num_importance
        self.perturb = perturb
        self.raw_noise_std = raw_noise_std
        self.white_bkgd = white_bkgd
        self.chunk = chunk
        
        # Load split file
        if split_file is None:
            split_file = Path(data_dir) / "nerfmae_split.npz"
        
        self.samples = self._load_samples(split_file)
        print(f"Loaded {len(self.samples)} samples for {split} split")
        
    def _load_samples(self, split_file: Path) -> List[str]:
        """Load sample list from split file or scan directory."""
        samples = []
        
        if Path(split_file).exists():
            split_data = np.load(split_file)
            split_key = f"{self.split}_scenes"
            
            if split_key in split_data:
                scene_names = split_data[split_key]
                # Match scene names to actual files in features directory
                available_files = list(self.features_dir.glob("*.npz"))
                
                for npz_file in available_files:
                    samples.append(str(npz_file))
        else:
            print(f"Warning: Split file {split_file} not found")
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample with rendered multi-view images."""
        npz_path = self.samples[idx]
        
        # Load voxel grid data
        data = np.load(npz_path)
        rgbsigma = data['rgbsigma']  # [X, Y, Z, 4] - 3D voxel grid
        bbox_min = data['bbox_min']
        bbox_max = data['bbox_max']
        
        # Convert to torch tensors
        rgbsigma = torch.from_numpy(rgbsigma).float()
        bbox_min = torch.from_numpy(bbox_min).float()
        bbox_max = torch.from_numpy(bbox_max).float()
        
        # Process sigma values: -10000 or negative means empty space
        # Clamp sigma to be non-negative (empty regions become 0)
        rgbsigma[..., 3] = torch.clamp(rgbsigma[..., 3], min=0.0)
        
        # Clamp RGB to [0, 1] range
        rgbsigma[..., :3] = torch.clamp(rgbsigma[..., :3], 0.0, 1.0)
        
        # Generate camera poses
        camera_poses = self._generate_camera_poses(bbox_min, bbox_max)
        
        # Render images from each camera pose using render.py-style volume rendering
        images = []
        depth_maps = []
        for i in range(self.num_views):
            pose = camera_poses[i]
            img, depth = self._render_from_voxel(rgbsigma, pose, bbox_min, bbox_max)
            images.append(img)
            depth_maps.append(depth)
        
        images = torch.stack(images, dim=0)  # [S, 3, H, W]
        depth_maps = torch.stack(depth_maps, dim=0)  # [S, H, W]
        
        # Create camera intrinsics (simple pinhole model)
        focal = self.image_size[0]  # Approximate focal length
        intrinsics = torch.tensor([
            [focal, 0, self.image_size[1] / 2],
            [0, focal, self.image_size[0] / 2],
            [0, 0, 1]
        ], dtype=torch.float32)
        
        return {
            "images": images,                    # [S, 3, H, W]
            "depth_maps": depth_maps,            # [S, H, W]
            "rgbsigma": rgbsigma,               # [H, W, D, 4]
            "camera_poses": camera_poses,        # [S, 4, 4]
            "camera_intrinsics": intrinsics,     # [3, 3]
            "bbox_min": bbox_min,                # [3]
            "bbox_max": bbox_max,                # [3]
            "scene_path": npz_path,
        }
    
    def _generate_camera_poses(
        self, 
        bbox_min: torch.Tensor, 
        bbox_max: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate camera poses looking at the scene center from different angles.
        
        Uses a hemisphere sampling strategy around the scene.
        
        Args:
            bbox_min: Minimum corner of bounding box [3]
            bbox_max: Maximum corner of bounding box [3]
            
        Returns:
            Camera poses [num_views, 4, 4]
        """
        center = (bbox_min + bbox_max) / 2
        size = bbox_max - bbox_min
        radius = torch.norm(size) * 0.8  # Camera distance from center
        
        poses = []
        for i in range(self.num_views):
            # Sample angles on a hemisphere
            theta = 2 * np.pi * i / self.num_views  # azimuth angle
            phi = np.pi / 3  # elevation angle (~60 degrees from vertical)
            
            # Camera position in spherical coordinates
            # x, y are horizontal plane, z is up
            x = center[0].item() + radius.item() * np.cos(theta) * np.sin(phi)
            y = center[1].item() + radius.item() * np.sin(theta) * np.sin(phi)
            z = center[2].item() + radius.item() * np.cos(phi)
            
            cam_pos = torch.tensor([x, y, z], dtype=torch.float32)
            
            # Look-at matrix: camera looks at center, Z-up
            pose = self._look_at(cam_pos, center, torch.tensor([0, 0, 1], dtype=torch.float32))
            poses.append(pose)
        
        return torch.stack(poses, dim=0)  # [S, 4, 4]
    
    def _look_at(
        self, 
        eye: torch.Tensor, 
        center: torch.Tensor, 
        up: torch.Tensor
    ) -> torch.Tensor:
        """
        Create look-at camera matrix (camera-to-world transform).
        
        Args:
            eye: Camera position [3]
            center: Target point [3]
            up: Up vector [3]
            
        Returns:
            Camera pose matrix [4, 4]
        """
        forward = center - eye
        forward = forward / (torch.norm(forward) + 1e-8)
        
        right = torch.linalg.cross(forward, up)
        right = right / (torch.norm(right) + 1e-8)
        
        new_up = torch.linalg.cross(right, forward)
        new_up = new_up / (torch.norm(new_up) + 1e-8)
        
        # Camera-to-world matrix (OpenGL convention: -Z is forward)
        pose = torch.eye(4)
        pose[:3, 0] = right
        pose[:3, 1] = new_up
        pose[:3, 2] = -forward  # Camera looks along -Z in camera space
        pose[:3, 3] = eye
        
        return pose
    
    def _render_from_voxel(
        self,
        rgbsigma: torch.Tensor,
        camera_pose: torch.Tensor,
        bbox_min: torch.Tensor,
        bbox_max: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Render an image from the voxel grid using volume rendering.
        
        Args:
            rgbsigma: Voxel grid [X, Y, Z, 4]
            camera_pose: Camera-to-world matrix [4, 4]
            bbox_min: Minimum corner of bounding box [3]
            bbox_max: Maximum corner of bounding box [3]
            
        Returns:
            image: Rendered image [3, H, W]
            depth: Rendered depth map [H, W]
        """
        H, W = self.image_size
        device = rgbsigma.device
        
        # Generate rays (similar to render.py's get_rays)
        rays_o, rays_d = self._generate_rays(camera_pose, H, W)
        
        # Reshape for batch processing
        rays_o_flat = rays_o.reshape(-1, 3)  # [H*W, 3]
        rays_d_flat = rays_d.reshape(-1, 3)  # [H*W, 3]
        N_rays = rays_o_flat.shape[0]
        
        # Render in chunks to avoid OOM (like batchify_rays in render.py)
        rgb_chunks = []
        depth_chunks = []
        
        for i in range(0, N_rays, self.chunk):
            chunk_rays_o = rays_o_flat[i:i + self.chunk]
            chunk_rays_d = rays_d_flat[i:i + self.chunk]
            
            rgb_chunk, depth_chunk = self._render_rays_chunk(
                chunk_rays_o, chunk_rays_d, rgbsigma, bbox_min, bbox_max
            )
            rgb_chunks.append(rgb_chunk)
            depth_chunks.append(depth_chunk)
        
        # Concatenate chunks
        colors = torch.cat(rgb_chunks, dim=0)  # [H*W, 3]
        depths = torch.cat(depth_chunks, dim=0)  # [H*W]
        
        # Reshape to image
        image = colors.reshape(H, W, 3).permute(2, 0, 1)  # [3, H, W]
        depth = depths.reshape(H, W)  # [H, W]
        
        return image, depth
    
    def _render_rays_chunk(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        rgbsigma: torch.Tensor,
        bbox_min: torch.Tensor,
        bbox_max: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Render a chunk of rays using the voxel grid.
        
        Implements the core rendering logic from render.py's render_rays,
        but queries a voxel grid instead of an MLP network.
        
        Args:
            rays_o: Ray origins [N, 3]
            rays_d: Ray directions [N, 3]
            rgbsigma: Voxel grid [X, Y, Z, 4]
            bbox_min, bbox_max: Bounding box corners [3]
            
        Returns:
            rgb_map: Rendered colors [N, 3]
            depth_map: Rendered depths [N]
        """
        device = rays_o.device
        N_rays = rays_o.shape[0]
        
        # Compute near and far intersection with bbox
        near, far = self._intersect_bbox(rays_o, rays_d, bbox_min, bbox_max)
        
        # === Coarse sampling (like render.py) ===
        t_vals = torch.linspace(0., 1., steps=self.num_samples, device=device)
        z_vals = near[:, None] * (1. - t_vals) + far[:, None] * t_vals  # [N, num_samples]
        
        # Stratified sampling with perturbation (from render.py)
        if self.perturb > 0.:
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            t_rand = torch.rand(z_vals.shape, device=device)
            z_vals = lower + (upper - lower) * t_rand
        
        # Sample points along rays: pts = o + t * d
        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[:, :, None]  # [N, num_samples, 3]
        
        # Query voxel grid (replaces network_query_fn in render.py)
        rgb, sigma = self._query_voxel_grid(pts, rgbsigma, bbox_min, bbox_max)
        sigma = sigma.squeeze(-1)  # [N, num_samples]
        
        # Volume rendering (using raw2outputs_grid, adapted from render.py)
        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs_grid(
            rgb, sigma, z_vals, rays_d,
            raw_noise_std=self.raw_noise_std,
            white_bkgd=self.white_bkgd
        )
        
        # === Hierarchical sampling (from render.py) ===
        if self.num_importance > 0:
            # Save coarse results
            rgb_map_0 = rgb_map
            depth_map_0 = depth_map
            
            # Sample from PDF based on coarse weights
            z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            z_samples = sample_pdf_grid(
                z_vals_mid, 
                weights[..., 1:-1],  # Exclude first and last weights
                self.num_importance, 
                det=(self.perturb == 0.)
            )
            z_samples = z_samples.detach()
            
            # Combine coarse and fine samples, then sort
            z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
            
            # Sample points at new locations
            pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[:, :, None]
            
            # Query voxel grid at fine sample points
            rgb, sigma = self._query_voxel_grid(pts, rgbsigma, bbox_min, bbox_max)
            sigma = sigma.squeeze(-1)
            
            # Final volume rendering with combined samples
            rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs_grid(
                rgb, sigma, z_vals, rays_d,
                raw_noise_std=self.raw_noise_std,
                white_bkgd=self.white_bkgd
            )
        
        return rgb_map, depth_map
    
    def _generate_rays(
        self, 
        camera_pose: torch.Tensor, 
        H: int, 
        W: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate rays for each pixel.
        
        Args:
            camera_pose: Camera-to-world matrix [4, 4]
            H, W: Image dimensions
            
        Returns:
            rays_o: Ray origins [H, W, 3]
            rays_d: Ray directions [H, W, 3]
        """
        focal = H  # Simple approximation
        
        # Pixel coordinates
        i, j = torch.meshgrid(
            torch.arange(H, dtype=torch.float32),
            torch.arange(W, dtype=torch.float32),
            indexing='ij'
        )
        
        # Camera space directions
        dirs = torch.stack([
            (j - W * 0.5) / focal,
            -(i - H * 0.5) / focal,
            -torch.ones_like(i)
        ], dim=-1)  # [H, W, 3]
        
        # Transform to world space
        rotation = camera_pose[:3, :3]
        rays_d = torch.sum(dirs[..., None, :] * rotation, dim=-1)  # [H, W, 3]
        rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
        
        # Ray origins (camera position)
        rays_o = camera_pose[:3, 3].expand(H, W, 3)
        
        return rays_o, rays_d
    
    def _intersect_bbox(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        bbox_min: torch.Tensor,
        bbox_max: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute ray-bbox intersection.
        
        Args:
            rays_o: Ray origins [N, 3]
            rays_d: Ray directions [N, 3]
            bbox_min, bbox_max: Bounding box corners [3]
            
        Returns:
            near, far: Intersection distances [N]
        """
        inv_d = 1.0 / (rays_d + 1e-10)
        
        t_min = (bbox_min - rays_o) * inv_d
        t_max = (bbox_max - rays_o) * inv_d
        
        t1 = torch.minimum(t_min, t_max)
        t2 = torch.maximum(t_min, t_max)
        
        near = torch.max(t1, dim=-1)[0]
        far = torch.min(t2, dim=-1)[0]
        
        # Clamp to valid range
        near = torch.clamp(near, min=0.1)
        far = torch.clamp(far, min=near + 0.1)
        
        return near, far
    
    def _query_voxel_grid(
        self,
        pts: torch.Tensor,
        rgbsigma: torch.Tensor,
        bbox_min: torch.Tensor,
        bbox_max: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Query RGB and sigma values from voxel grid using trilinear interpolation.
        
        Args:
            pts: Query points [N, M, 3] in world coordinates (x, y, z)
            rgbsigma: Voxel grid [X, Y, Z, 4] where X, Y, Z are grid dimensions
            bbox_min, bbox_max: Bounding box [3]
            
        Returns:
            rgb: Color values [N, M, 3]
            sigma: Density values [N, M, 1]
        """
        N, M, _ = pts.shape
        
        # Normalize points to [-1, 1] for grid_sample
        # pts are in world coordinates (x, y, z)
        pts_normalized = 2.0 * (pts - bbox_min) / (bbox_max - bbox_min + 1e-10) - 1.0
        
        # grid_sample expects input in [batch, C, D, H, W] format
        # and grid coordinates in (x, y, z) order mapping to (W, H, D)
        # Our rgbsigma is [X, Y, Z, 4] in world coordinate order
        # We need to rearrange to [1, 4, Z, Y, X] for grid_sample (D=Z, H=Y, W=X)
        voxel = rgbsigma.permute(3, 2, 1, 0).unsqueeze(0)  # [1, 4, Z, Y, X]
        
        # Grid sample expects grid in (x, y, z) order where x->W, y->H, z->D
        # pts_normalized is already in (x, y, z) order
        pts_grid = pts_normalized.reshape(1, N * M, 1, 1, 3)  # [1, N*M, 1, 1, 3]
        
        # Sample from grid using trilinear interpolation
        sampled = F.grid_sample(
            voxel, pts_grid, 
            mode='bilinear', 
            padding_mode='zeros',
            align_corners=True
        )  # [1, 4, N*M, 1, 1]
        
        # Reshape result
        sampled = sampled.view(4, N * M).T.reshape(N, M, 4)  # [N, M, 4]
        
        rgb = sampled[..., :3]
        sigma = sampled[..., 3:4]
        
        # Ensure non-negative sigma
        sigma = torch.clamp(sigma, min=0.0)
        
        return rgb, sigma


def create_nerf_mae_dataloaders(
    data_dir: str,
    batch_size: int = 2,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (256, 256),
    num_views: int = 8,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders for NeRF-MAE data.
    
    Args:
        data_dir: Directory containing pretrain data
        batch_size: Batch size
        num_workers: Number of data loading workers
        image_size: Target image size
        num_views: Number of views per sample
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_dataset = NeRFMAEDataset(
        data_dir=data_dir,
        split="train",
        image_size=image_size,
        num_views=num_views,
    )
    
    val_dataset = NeRFMAEDataset(
        data_dir=data_dir,
        split="val",
        image_size=image_size,
        num_views=num_views,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    
    return train_loader, val_loader


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function to handle variable-sized data."""
    result = {}
    
    # Stack tensor fields
    for key in ['images', 'depth_maps', 'camera_poses', 'camera_intrinsics', 'bbox_min', 'bbox_max']:
        if key in batch[0]:
            result[key] = torch.stack([item[key] for item in batch], dim=0)
    
    # rgbsigma may have different sizes, so we keep them as a list
    if 'rgbsigma' in batch[0]:
        result['rgbsigma'] = [item['rgbsigma'] for item in batch]
    
    # Keep scene paths as list
    if 'scene_path' in batch[0]:
        result['scene_path'] = [item['scene_path'] for item in batch]
    
    return result


if __name__ == "__main__":
    # Test the dataloader
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, 
                        default="d:/lecture/2.0_xk/CV/finalproject/NeRF-MAE_pretrain/NeRF-MAE/pretrain")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_views", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=128)
    args = parser.parse_args()
    
    print("Testing NeRF-MAE dataloader...")
    
    dataset = NeRFMAEDataset(
        data_dir=args.data_dir,
        split="train",
        image_size=(args.image_size, args.image_size),
        num_views=args.num_views,
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Get a sample
    sample = dataset[1]
    print(f"Sample keys: {sample.keys()}")
    print(f"Images shape: {sample['images'].shape}")
    print(f"Depth maps shape: {sample['depth_maps'].shape}")
    print(f"rgbsigma shape: {sample['rgbsigma'].shape}")
    print(f"Camera poses shape: {sample['camera_poses'].shape}")
    
    # Print rendering statistics
    print(f"\nRendering statistics:")
    print(f"  RGB min: {sample['images'].min():.4f}, max: {sample['images'].max():.4f}")
    print(f"  Depth min: {sample['depth_maps'].min():.4f}, max: {sample['depth_maps'].max():.4f}")
    
    # Save rendered images and depth maps for visualization
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, min(4, args.num_views), figsize=(16, 8))
    for i in range(min(4, args.num_views)):
        # RGB image
        img = sample['images'][i].permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        axes[0, i].imshow(img)
        axes[0, i].set_title(f"RGB View {i}")
        axes[0, i].axis('off')
        
        # Depth map
        depth = sample['depth_maps'][i].numpy()
        im = axes[1, i].imshow(depth, cmap='viridis')
        axes[1, i].set_title(f"Depth View {i}")
        axes[1, i].axis('off')
        plt.colorbar(im, ax=axes[1, i], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig("test_rendered_views.png")
    print("Saved test_rendered_views.png")
