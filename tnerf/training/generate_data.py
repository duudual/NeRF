"""
Data Generation Pipeline for T-NeRF Training.

This script generates multi-view images from NeRF-MAE pretrain data (voxel grids)
and saves them to disk for training. This pre-processing step allows the dataloader
to simply load images instead of rendering them on-the-fly.

Data Flow:
1. Load nerfmae_split.npz to get train/val/test splits
2. For each sample in the split, load corresponding rgbsigma voxel grid
3. Generate camera poses around the scene  
4. Render multi-view images using volume rendering from the voxel grid
5. Save images and metadata to output directory

Output Structure:
    output_dir/
        train_samples.json   # list of train sample info
        val_samples.json     # list of val sample info
        test_samples.json    # list of test sample info
        config.json
        samples/
            3dfront_2140_00/
                images/
                    view_000.png
                    view_001.png
                    ...
                cameras.npz  (camera poses, intrinsics, bbox)
                rgbsigma.npz (optional: original voxel grid for loss computation)
            3dfront_2140_01/
                ...

Usage:
    python generate_data.py --input_dir /path/to/pretrain --output_dir /path/to/rendered_data
    python generate_data.py --input_dir ../data/nerf-mae --output_dir ../data/tnerf
    python generate_data.py --input_dir ../../NeRF-MAE_pretrain/NeRF-MAE/pretrain --output_dir ../data/tnerf
    python generate_data.py --input_dir E:/code/cv_finalproject/data/NeRF-MAE_pretrain --output_dir E:/code/cv_finalproject/data/tnerf --max_train 1000 --max_val 100 --max_test 100
    python generate_data.py \
    --input_dir "/media/fengwu/ZX1 1TB/code/cv_finalproject/data/NeRF-MAE_pretrain" \
    --output_dir "/media/fengwu/ZX1 1TB/code/cv_finalproject/data/tnerf" \
    --max_train 1000 \
    --max_val 100 \
    --max_test 100

/media/fengwu/ZX1 1TB/code/cv_finalproject
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import numpy as np

import torch
import torch.nn.functional as F
from PIL import Image


def sample_pdf(bins, weights, N_samples, det=False):
    """
    Hierarchical sampling based on weights.
    
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


class VoxelRenderer:
    """
    Volume renderer for voxel grids.
    
    Renders images from voxel grids using volume rendering,
    adapted from NeRF's render.py with voxel grid sampling.
    """
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (256, 256),
        num_samples: int = 128,
        num_importance: int = 64,
        perturb: float = 1.0,
        raw_noise_std: float = 0.0,
        white_bkgd: bool = True,
        chunk: int = 1024 * 8,
        device: str = "cuda",
    ):
        """
        Initialize VoxelRenderer.
        
        Args:
            image_size: (H, W) target image size
            num_samples: Number of coarse samples per ray
            num_importance: Number of fine samples for hierarchical sampling
            perturb: Stratified sampling perturbation
            raw_noise_std: Noise added to density
            white_bkgd: Use white background
            chunk: Chunk size for batched rendering
            device: Device to use
        """
        self.image_size = image_size
        self.num_samples = num_samples
        self.num_importance = num_importance
        self.perturb = perturb
        self.raw_noise_std = raw_noise_std
        self.white_bkgd = white_bkgd
        self.chunk = chunk
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    def generate_camera_poses(
        self, 
        bbox_min: torch.Tensor, 
        bbox_max: torch.Tensor,
        rgbsigma: torch.Tensor,
        num_views: int = 8,
        camera_height_ratio: float = 0.6,
        min_distance: float = 0.3,
    ) -> torch.Tensor:
        """
        Generate camera poses inside the scene for indoor scene reconstruction.
        
        Strategy for NeRF-friendly camera placement:
        1. Place cameras at different positions inside the scene
        2. Cameras look at various points covering the scene
        3. Ensure good coverage while avoiding empty regions
        4. Use realistic camera height (like human eye level)
        
        Args:
            bbox_min: Minimum corner of bounding box [3]
            bbox_max: Maximum corner of bounding box [3]
            rgbsigma: Voxel grid [X, Y, Z, 4] for checking empty regions
            num_views: Number of views to generate
            camera_height_ratio: Height ratio within scene (0-1), 0.6 = 60% from bottom
            min_distance: Minimum distance ratio between camera and look-at point
            
        Returns:
            Camera poses [num_views, 4, 4]
        """
        center = (bbox_min + bbox_max) / 2
        size = bbox_max - bbox_min
        
        # Get sigma grid to find valid (non-empty) regions
        sigma_grid = rgbsigma[..., 3]
        empty_value = -10000.0
        valid_mask = sigma_grid != empty_value
        
        # Find valid voxel positions for camera placement
        valid_indices = torch.nonzero(valid_mask, as_tuple=False).float()  # [N, 3]
        
        if valid_indices.shape[0] < 10:
            # Fallback to center-based if not enough valid voxels
            return self._generate_orbit_poses(bbox_min, bbox_max, num_views)
        
        # Convert voxel indices to world coordinates
        grid_size = torch.tensor(sigma_grid.shape, dtype=torch.float32)
        valid_world = bbox_min + (valid_indices / (grid_size - 1)) * size
        
        # Strategy: Generate viewing positions that cover the scene well
        poses = []
        
        # Compute scene layout
        x_min, y_min, z_min = bbox_min.tolist()
        x_max, y_max, z_max = bbox_max.tolist()
        
        # Camera height: typically at human eye level (~1.6m from floor)
        # Use ratio of scene height for flexibility
        camera_z = z_min + size[2].item() * camera_height_ratio
        
        # Generate camera positions using different strategies
        for i in range(num_views):
            if i < num_views // 2:
                # Strategy 1: Corner-based views looking towards center
                # Place cameras near corners looking inward
                corner_idx = i % 4
                corners = [
                    (x_min + size[0] * 0.2, y_min + size[1] * 0.2),  # Bottom-left
                    (x_max - size[0] * 0.2, y_min + size[1] * 0.2),  # Bottom-right
                    (x_max - size[0] * 0.2, y_max - size[1] * 0.2),  # Top-right
                    (x_min + size[0] * 0.2, y_max - size[1] * 0.2),  # Top-left
                ]
                cam_x, cam_y = corners[corner_idx]
                
                # Look towards opposite side or center
                look_corners = [
                    (center[0].item() + size[0] * 0.2, center[1].item() + size[1] * 0.2),
                    (center[0].item() - size[0] * 0.2, center[1].item() + size[1] * 0.2),
                    (center[0].item() - size[0] * 0.2, center[1].item() - size[1] * 0.2),
                    (center[0].item() + size[0] * 0.2, center[1].item() - size[1] * 0.2),
                ]
                look_x, look_y = look_corners[corner_idx]
                look_z = camera_z - size[2].item() * 0.1  # Look slightly downward
                
            else:
                # Strategy 2: Edge-based views for wall coverage
                # Place cameras along edges looking at opposite walls
                edge_idx = (i - num_views // 2) % 4
                t = 0.3 + 0.4 * ((i - num_views // 2) // 4)  # Vary position along edge
                
                if edge_idx == 0:  # Left edge
                    cam_x = x_min + size[0] * 0.15
                    cam_y = y_min + size[1] * t
                    look_x = x_max - size[0] * 0.15
                    look_y = cam_y
                elif edge_idx == 1:  # Right edge
                    cam_x = x_max - size[0] * 0.15
                    cam_y = y_min + size[1] * t
                    look_x = x_min + size[0] * 0.15
                    look_y = cam_y
                elif edge_idx == 2:  # Bottom edge
                    cam_x = x_min + size[0] * t
                    cam_y = y_min + size[1] * 0.15
                    look_x = cam_x
                    look_y = y_max - size[1] * 0.15
                else:  # Top edge
                    cam_x = x_min + size[0] * t
                    cam_y = y_max - size[1] * 0.15
                    look_x = cam_x
                    look_y = y_min + size[1] * 0.15
                
                look_z = camera_z
            
            cam_pos = torch.tensor([cam_x, cam_y, camera_z], dtype=torch.float32)
            look_at = torch.tensor([look_x, look_y, look_z], dtype=torch.float32)
            
            # Add some randomness to avoid perfectly aligned views
            cam_pos += torch.randn(3) * size * 0.02
            look_at += torch.randn(3) * size * 0.02
            
            # Create camera pose
            pose = self._look_at(cam_pos, look_at, torch.tensor([0, 0, 1], dtype=torch.float32))
            poses.append(pose)
        
        return torch.stack(poses, dim=0)
    
    def _generate_orbit_poses(
        self,
        bbox_min: torch.Tensor,
        bbox_max: torch.Tensor,
        num_views: int,
        elevation_range: Tuple[float, float] = (30, 60),
    ) -> torch.Tensor:
        """
        Fallback: Generate orbit camera poses around scene (for object-centric scenes).
        
        Args:
            bbox_min: Minimum corner of bounding box [3]
            bbox_max: Maximum corner of bounding box [3]
            num_views: Number of views to generate
            elevation_range: Range of elevation angles in degrees
            
        Returns:
            Camera poses [num_views, 4, 4]
        """
        center = (bbox_min + bbox_max) / 2
        size = bbox_max - bbox_min
        radius = torch.norm(size) * 0.8
        
        poses = []
        for i in range(num_views):
            theta = 2 * np.pi * i / num_views
            phi_deg = elevation_range[0] + (elevation_range[1] - elevation_range[0]) * (i % 3) / 2
            phi = np.deg2rad(90 - phi_deg)
            
            x = center[0].item() + radius.item() * np.cos(theta) * np.sin(phi)
            y = center[1].item() + radius.item() * np.sin(theta) * np.sin(phi)
            z = center[2].item() + radius.item() * np.cos(phi)
            
            cam_pos = torch.tensor([x, y, z], dtype=torch.float32)
            pose = self._look_at(cam_pos, center, torch.tensor([0, 0, 1], dtype=torch.float32))
            poses.append(pose)
        
        return torch.stack(poses, dim=0)
    
    def _look_at(
        self, 
        eye: torch.Tensor, 
        center: torch.Tensor, 
        up: torch.Tensor
    ) -> torch.Tensor:
        """Create look-at camera matrix (camera-to-world transform)."""
        forward = center - eye
        forward = forward / (torch.norm(forward) + 1e-8)
        
        right = torch.linalg.cross(forward, up)
        right = right / (torch.norm(right) + 1e-8)
        
        new_up = torch.linalg.cross(right, forward)
        new_up = new_up / (torch.norm(new_up) + 1e-8)
        
        pose = torch.eye(4)
        pose[:3, 0] = right
        pose[:3, 1] = new_up
        pose[:3, 2] = -forward
        pose[:3, 3] = eye
        
        return pose
    
    def get_intrinsics(self, fov_deg: float = 90.0) -> torch.Tensor:
        """
        Get camera intrinsic matrix.
        
        Args:
            fov_deg: Field of view in degrees (default 90° for wide indoor coverage)
        """
        H, W = self.image_size
        # focal = H / (2 * tan(fov/2))
        fov_rad = np.deg2rad(fov_deg)
        focal = H / (2 * np.tan(fov_rad / 2))
        intrinsics = torch.tensor([
            [focal, 0, W / 2],
            [0, focal, H / 2],
            [0, 0, 1]
        ], dtype=torch.float32)
        return intrinsics
    
    def get_rays(
        self,
        camera_pose: torch.Tensor,
        intrinsics: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate rays for a camera pose.
        
        Args:
            camera_pose: [4, 4] camera-to-world matrix
            intrinsics: [3, 3] camera intrinsic matrix
            
        Returns:
            rays_o: [H*W, 3] ray origins
            rays_d: [H*W, 3] ray directions
        """
        H, W = self.image_size
        
        # Create pixel coordinates on the correct device
        i, j = torch.meshgrid(
            torch.arange(W, dtype=torch.float32, device=camera_pose.device),
            torch.arange(H, dtype=torch.float32, device=camera_pose.device),
            indexing='xy'
        )
        
        # Convert to camera coordinates
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        
        dirs = torch.stack([
            (i - cx) / fx,
            -(j - cy) / fy,  # Negative because y is down in image
            -torch.ones_like(i)  # Looking along -Z in camera space
        ], dim=-1)  # [H, W, 3]
        
        # Transform to world coordinates
        rays_d = torch.sum(dirs[..., None, :] * camera_pose[:3, :3], dim=-1)
        rays_o = camera_pose[:3, 3].expand(rays_d.shape)
        
        rays_o = rays_o.reshape(-1, 3)
        rays_d = rays_d.reshape(-1, 3)
        
        return rays_o, rays_d
    
    def sample_voxel_grid(
        self,
        rgbsigma: torch.Tensor,
        points: torch.Tensor,
        bbox_min: torch.Tensor,
        bbox_max: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sample voxel grid at given 3D points using trilinear interpolation.
        
        Args:
            rgbsigma: [X, Y, Z, 4] voxel grid (RGB + sigma)
            points: [N, 3] query points in world coordinates (x, y, z)
            bbox_min: [3] minimum corner of bounding box
            bbox_max: [3] maximum corner of bounding box
            
        Returns:
            values: [N, 4] interpolated RGB + sigma values
        """
        N = points.shape[0]
        
        # Normalize points to [-1, 1] for grid_sample
        # pts are in world coordinates (x, y, z)
        pts_normalized = 2.0 * (points - bbox_min) / (bbox_max - bbox_min + 1e-10) - 1.0
        
        # grid_sample expects input in [batch, C, D, H, W] format
        # and grid coordinates in (x, y, z) order mapping to (W, H, D)
        # Our rgbsigma is [X, Y, Z, 4] in world coordinate order
        # We need to rearrange to [1, 4, Z, Y, X] for grid_sample (D=Z, H=Y, W=X)
        voxel = rgbsigma.permute(3, 2, 1, 0).unsqueeze(0)  # [1, 4, Z, Y, X]
        
        # Grid sample expects grid in (x, y, z) order where x->W, y->H, z->D
        # pts_normalized is already in (x, y, z) order
        pts_grid = pts_normalized.reshape(1, N, 1, 1, 3)  # [1, N, 1, 1, 3]
        
        # Sample from grid using trilinear interpolation
        sampled = F.grid_sample(
            voxel, pts_grid, 
            mode='bilinear', 
            padding_mode='zeros',
            align_corners=True
        )  # [1, 4, N, 1, 1]
        
        # Reshape result
        values = sampled.view(4, N).T  # [N, 4]
        
        # Ensure non-negative sigma
        values = values.clone()
        values[:, 3] = torch.clamp(values[:, 3], min=0.0)
        
        return values
    
    def raw2outputs(
        self,
        raw: torch.Tensor,
        z_vals: torch.Tensor,
        rays_d: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert raw predictions to RGB using volume rendering.
        
        Args:
            raw: [N_rays, N_samples, 4] - RGB + density
            z_vals: [N_rays, N_samples] - sample positions along rays
            rays_d: [N_rays, 3] - ray directions
            
        Returns:
            rgb_map: [N_rays, 3]
            weights: [N_rays, N_samples]
            depth_map: [N_rays]
        """
        raw2alpha = lambda raw, dists: 1. - torch.exp(-F.relu(raw) * dists)
        
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.full_like(dists[..., :1], 1e10)], dim=-1)
        dists = dists * torch.norm(rays_d[..., None, :], dim=-1)
        
        # RGB values are already in [0,1] range from voxel grid, no need for sigmoid
        rgb = torch.clamp(raw[..., :3], 0.0, 1.0)
        
        noise = 0.
        if self.raw_noise_std > 0:
            noise = torch.randn_like(raw[..., 3]) * self.raw_noise_std
        
        alpha = raw2alpha(raw[..., 3] + noise, dists)
        
        # Transmittance
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones_like(alpha[..., :1]), 1. - alpha + 1e-10], dim=-1),
            dim=-1
        )[..., :-1]
        
        rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)
        depth_map = torch.sum(weights * z_vals, dim=-1)
        acc_map = torch.sum(weights, dim=-1)
        
        if self.white_bkgd:
            rgb_map = rgb_map + (1. - acc_map[..., None])
        
        return rgb_map, weights, depth_map
    
    def intersect_bbox(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        bbox_min: torch.Tensor,
        bbox_max: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute ray-bbox intersection for accurate near/far bounds.
        
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
    
    def render_image(
        self,
        rgbsigma: torch.Tensor,
        camera_pose: torch.Tensor,
        bbox_min: torch.Tensor,
        bbox_max: torch.Tensor,
    ) -> torch.Tensor:
        """
        Render an image from voxel grid.
        
        Args:
            rgbsigma: [X, Y, Z, 4] voxel grid
            camera_pose: [4, 4] camera pose
            bbox_min: [3] bounding box min
            bbox_max: [3] bounding box max
            
        Returns:
            image: [3, H, W] rendered image
        """
        H, W = self.image_size
        intrinsics = self.get_intrinsics()
        
        # Move to device
        rgbsigma = rgbsigma.to(self.device)
        camera_pose = camera_pose.to(self.device)
        bbox_min = bbox_min.to(self.device)
        bbox_max = bbox_max.to(self.device)
        intrinsics = intrinsics.to(self.device)
        
        # Get rays
        rays_o, rays_d = self.get_rays(camera_pose, intrinsics)
        rays_o = rays_o.to(self.device)
        rays_d = rays_d.to(self.device)
        
        # Compute near and far bounds using ray-bbox intersection
        near, far = self.intersect_bbox(rays_o, rays_d, bbox_min, bbox_max)
        
        # Render in chunks
        all_rgb = []
        
        for i in range(0, rays_o.shape[0], self.chunk):
            chunk_rays_o = rays_o[i:i+self.chunk]
            chunk_rays_d = rays_d[i:i+self.chunk]
            chunk_near = near[i:i+self.chunk]
            chunk_far = far[i:i+self.chunk]
            
            # Sample points along rays with per-ray near/far
            t_vals = torch.linspace(0., 1., self.num_samples, device=self.device)
            z_vals = chunk_near[:, None] * (1. - t_vals) + chunk_far[:, None] * t_vals
            
            # Perturb sampling
            if self.perturb > 0:
                mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
                upper = torch.cat([mids, z_vals[..., -1:]], dim=-1)
                lower = torch.cat([z_vals[..., :1], mids], dim=-1)
                t_rand = torch.rand_like(z_vals)
                z_vals = lower + (upper - lower) * t_rand
            
            # Get sample points
            pts = chunk_rays_o[..., None, :] + chunk_rays_d[..., None, :] * z_vals[..., :, None]
            pts_flat = pts.reshape(-1, 3)
            
            # Query voxel grid
            raw_flat = self.sample_voxel_grid(rgbsigma, pts_flat, bbox_min, bbox_max)
            raw = raw_flat.reshape(chunk_rays_o.shape[0], self.num_samples, 4)
            
            # Volume rendering
            rgb_map, weights, _ = self.raw2outputs(raw, z_vals, chunk_rays_d)
            
            # Hierarchical sampling
            if self.num_importance > 0:
                z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
                z_samples = sample_pdf(
                    z_vals_mid,
                    weights[..., 1:-1],
                    self.num_importance,
                    det=(self.perturb == 0),
                ).detach().to(self.device)
                
                z_vals_combined, _ = torch.sort(torch.cat([z_vals, z_samples], dim=-1), dim=-1)
                
                pts = chunk_rays_o[..., None, :] + chunk_rays_d[..., None, :] * z_vals_combined[..., :, None]
                pts_flat = pts.reshape(-1, 3)
                
                raw_flat = self.sample_voxel_grid(rgbsigma, pts_flat, bbox_min, bbox_max)
                raw = raw_flat.reshape(chunk_rays_o.shape[0], -1, 4)
                
                rgb_map, _, _ = self.raw2outputs(raw, z_vals_combined, chunk_rays_d)
            
            all_rgb.append(rgb_map)
        
        rgb_map = torch.cat(all_rgb, dim=0)
        image = rgb_map.reshape(H, W, 3).permute(2, 0, 1)  # [3, H, W]
        
        return image.cpu()

def process_sample(
    npz_path: str,
    output_dir: Path,
    renderer: VoxelRenderer,
    num_views: int = 8,
    save_voxel: bool = True,
) -> Dict:
    """
    Process a single sample: load voxel grid, render views, save outputs.
    
    Args:
        npz_path: Path to .npz file containing voxel grid
        output_dir: Output directory for this sample
        renderer: VoxelRenderer instance
        num_views: Number of views to render
        save_voxel: Whether to save voxel grid (for training loss computation)
        
    Returns:
        Metadata dictionary
    """
    # Load voxel grid
    data = np.load(npz_path)
    rgbsigma = torch.from_numpy(data['rgbsigma']).float()
    bbox_min = torch.from_numpy(data['bbox_min']).float()
    bbox_max = torch.from_numpy(data['bbox_max']).float()
    
    # Process sigma values - but keep original for camera pose generation
    rgbsigma_original = rgbsigma.clone()
    rgbsigma[..., 3] = torch.clamp(rgbsigma[..., 3], min=0.0)
    rgbsigma[..., :3] = torch.clamp(rgbsigma[..., :3], 0.0, 1.0)
    
    # Create output directories
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate camera poses (use original rgbsigma to detect empty regions)
    camera_poses = renderer.generate_camera_poses(bbox_min, bbox_max, rgbsigma_original, num_views)
    intrinsics = renderer.get_intrinsics()
    
    # Render views
    rendered_images = []
    for view_idx in range(num_views):
        pose = camera_poses[view_idx]
        image = renderer.render_image(rgbsigma, pose, bbox_min, bbox_max)
        rendered_images.append(image)
        
        # Save image (ensure tensor on CPU before numpy)
        img_np = (
            image.detach().permute(1, 2, 0).cpu().numpy() * 255
        ).clip(0, 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np)
        img_pil.save(images_dir / f"view_{view_idx:03d}.png")
    
    # Save camera data
    np.savez(
        output_dir / "cameras.npz",
        extrinsics=camera_poses.cpu().numpy(),
        intrinsics=intrinsics.cpu().numpy(),
        bbox_min=bbox_min.cpu().numpy(),
        bbox_max=bbox_max.cpu().numpy(),
    )
    
    # Optionally save voxel grid
    if save_voxel:
        np.savez_compressed(
            output_dir / "rgbsigma.npz",
            rgbsigma=data['rgbsigma'],
            bbox_min=data['bbox_min'],
            bbox_max=data['bbox_max'],
        )
    
    # Return metadata
    metadata = {
        "sample_id": output_dir.name,
        "source_file": str(npz_path),
        "num_views": num_views,
        "image_size": list(renderer.image_size),
        "voxel_shape": list(rgbsigma.shape[:3]),
    }
    
    return metadata


def load_split_info(split_file: Path, features_dir: Path) -> Dict[str, List[str]]:
    """
    Load split information from nerfmae_split.npz.
    
    Args:
        split_file: Path to nerfmae_split.npz
        features_dir: Path to features directory containing .npz files
        
    Returns:
        Dictionary mapping split name to list of npz file paths
    """
    split_data = np.load(split_file, allow_pickle=True)
    
    # Build a set of available files for quick lookup
    available_files = {f.stem: str(f) for f in features_dir.glob("*.npz")}
    
    splits = {}
    for split_name in ['train', 'val', 'test']:
        key = f"{split_name}_scenes"
        if key in split_data:
            sample_names = split_data[key]
            # Match sample names to actual files
            split_files = []
            for name in sample_names:
                name_str = str(name)
                if name_str in available_files:
                    split_files.append(available_files[name_str])
            splits[split_name] = split_files
            print(f"  {split_name}: {len(split_files)} samples (from {len(sample_names)} in split file)")
        else:
            splits[split_name] = []
    
    return splits


def main():
    parser = argparse.ArgumentParser(description="Generate multi-view images from NeRF-MAE voxel grids")
    
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Path to NeRF-MAE pretrain data (contains features/ and nerfmae_split.npz)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for rendered data")
    parser.add_argument("--num_views", type=int, default=8,
                        help="Number of views to render per sample")
    parser.add_argument("--image_size", type=int, default=256,
                        help="Image size (height and width)")
    parser.add_argument("--num_samples", type=int, default=64,
                        help="Number of coarse samples per ray")
    parser.add_argument("--num_importance", type=int, default=32,
                        help="Number of fine samples for hierarchical sampling")
    parser.add_argument("--save_voxel", action="store_true", default=True,
                        help="Save voxel grid for training loss computation")
    parser.add_argument("--no_save_voxel", dest="save_voxel", action="store_false",
                        help="Don't save voxel grid")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use for rendering")
    parser.add_argument("--max_samples", type=int, default=10,
                        help="Maximum number of samples to process per split (default: 10, set to -1 for all samples)")
    parser.add_argument("--max_train", type=int, default=None,
                        help="Maximum number of train samples (overrides --max_samples for train)")
    parser.add_argument("--max_val", type=int, default=None,
                        help="Maximum number of val samples (overrides --max_samples for val)")
    parser.add_argument("--max_test", type=int, default=None,
                        help="Maximum number of test samples (overrides --max_samples for test)")
    parser.add_argument("--splits", type=str, nargs="+", default=["train", "val", "test"],
                        help="Which splits to process")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    features_dir = input_dir / "features"
    split_file = input_dir / "nerfmae_split.npz"
    
    # Check input directory
    if not features_dir.exists():
        print(f"Error: Features directory not found: {features_dir}")
        return
    
    if not split_file.exists():
        print(f"Error: Split file not found: {split_file}")
        return
    
    # Load split information
    print("Loading split information...")
    splits = load_split_info(split_file, features_dir)
    
    # Create renderer
    renderer = VoxelRenderer(
        image_size=(args.image_size, args.image_size),
        num_samples=args.num_samples,
        num_importance=args.num_importance,
        device=args.device,
    )
    
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(exist_ok=True)
    
    # Process each split
    all_results = {}
    
    for split_name in args.splits:
        if split_name not in splits:
            print(f"Warning: Split '{split_name}' not found, skipping")
            continue
        
        npz_files = splits[split_name]
        
        # Determine max samples for this split
        max_for_split = args.max_samples
        if split_name == "train" and args.max_train is not None:
            max_for_split = args.max_train
        elif split_name == "val" and args.max_val is not None:
            max_for_split = args.max_val
        elif split_name == "test" and args.max_test is not None:
            max_for_split = args.max_test
        
        if max_for_split and max_for_split > 0:
            npz_files = npz_files[:max_for_split]
        
        print(f"\nProcessing {split_name} split: {len(npz_files)} samples")
        
        split_metadata = []
        failed_samples = []
        
        for npz_path in tqdm(npz_files, desc=f"Processing {split_name}"):
            sample_name = Path(npz_path).stem
            sample_output_dir = samples_dir / sample_name
            
            try:
                metadata = process_sample(
                    npz_path,
                    sample_output_dir,
                    renderer,
                    num_views=args.num_views,
                    save_voxel=args.save_voxel,
                )
                split_metadata.append(metadata)
            except Exception as e:
                print(f"\nError processing {sample_name}: {e}")
                failed_samples.append(sample_name)
                continue
        
        # Save split index file
        split_info = {
            "samples": split_metadata,
            "failed": failed_samples,
        }
        with open(output_dir / f"{split_name}_samples.json", "w") as f:
            json.dump(split_info, f, indent=2)
        
        all_results[split_name] = {
            "processed": len(split_metadata),
            "failed": len(failed_samples),
        }
        
        print(f"  Processed: {len(split_metadata)}, Failed: {len(failed_samples)}")
    
    # Save config
    config = {
        "num_views": args.num_views,
        "image_size": [args.image_size, args.image_size],
        "num_samples": args.num_samples,
        "num_importance": args.num_importance,
        "save_voxel": args.save_voxel,
        "results": all_results,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"\nData generation complete!")
    for split_name, result in all_results.items():
        print(f"  {split_name}: {result['processed']} processed, {result['failed']} failed")
    print(f"  Output directory: {output_dir}")


if __name__ == "__main__":
    main()
