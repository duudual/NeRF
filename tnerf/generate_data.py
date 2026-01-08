"""
Data Generation Pipeline for T-NeRF Training.

This script generates multi-view images from NeRF-MAE pretrain data (voxel grids)
and saves them to disk for training. This pre-processing step allows the dataloader
to simply load images instead of rendering them on-the-fly.

Data Flow:
1. Load rgbsigma voxel grid from .npz files
2. Generate camera poses around the scene  
3. Render multi-view images using volume rendering from the voxel grid
4. Save images and metadata to output directory

Output Structure:
    output_dir/
        scene_xxx/
            images/
                view_000.png
                view_001.png
                ...
            cameras.npz  (camera poses, intrinsics, bbox)
            rgbsigma.npz (optional: original voxel grid for loss computation)

Usage:
    python generate_data.py --input_dir ./data/nerf-mae --output_dir ./data/tnerf
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
        num_views: int = 8,
        elevation_range: Tuple[float, float] = (30, 60),
    ) -> torch.Tensor:
        """
        Generate camera poses looking at the scene center from different angles.
        
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
            # Sample angles
            theta = 2 * np.pi * i / num_views  # azimuth
            # Vary elevation for diversity
            phi_deg = elevation_range[0] + (elevation_range[1] - elevation_range[0]) * (i % 3) / 2
            phi = np.deg2rad(90 - phi_deg)  # Convert to radians from vertical
            
            # Camera position in spherical coordinates
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
    
    def get_intrinsics(self) -> torch.Tensor:
        """Get camera intrinsic matrix."""
        H, W = self.image_size
        focal = H  # Approximate focal length
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
            points: [N, 3] query points in world coordinates
            bbox_min: [3] minimum corner of bounding box
            bbox_max: [3] maximum corner of bounding box
            
        Returns:
            values: [N, 4] interpolated RGB + sigma values
        """
        # Ensure all inputs are on the same device
        device = rgbsigma.device
        points = points.to(device)
        bbox_min = bbox_min.to(device)
        bbox_max = bbox_max.to(device)
        
        # Normalize points to [-1, 1] for grid_sample
        normalized = 2 * (points - bbox_min) / (bbox_max - bbox_min + 1e-8) - 1
        
        # grid_sample expects [N, D, H, W, 3] grid and [N, D_out, H_out, W_out, 3] points
        # We reshape for single batch
        grid = rgbsigma.permute(3, 0, 1, 2).unsqueeze(0)  # [1, 4, X, Y, Z]
        sample_points = normalized.view(1, 1, 1, -1, 3)  # [1, 1, 1, N, 3]
        
        # Sample using trilinear interpolation
        sampled = F.grid_sample(
            grid,
            sample_points,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        )  # [1, 4, 1, 1, N]
        
        values = sampled.squeeze().T  # [N, 4]
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
        
        rgb = torch.sigmoid(raw[..., :3])
        
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
        # TODO:


def process_scene(
    npz_path: str,
    output_dir: Path,
    renderer: VoxelRenderer,
    num_views: int = 8,
    save_voxel: bool = True,
) -> Dict:
    """
    Process a single scene: load voxel grid, render views, save outputs.
    
    Args:
        npz_path: Path to .npz file containing voxel grid
        output_dir: Output directory for this scene
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
    
    # Process sigma values
    rgbsigma[..., 3] = torch.clamp(rgbsigma[..., 3], min=0.0)
    rgbsigma[..., :3] = torch.clamp(rgbsigma[..., :3], 0.0, 1.0)
    
    # Create output directories
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate camera poses
    camera_poses = renderer.generate_camera_poses(bbox_min, bbox_max, num_views)
    intrinsics = renderer.get_intrinsics()
    
    # Render views
    rendered_images = []
    for view_idx in range(num_views):
        pose = camera_poses[view_idx]
        image = renderer.render_image(rgbsigma, pose, bbox_min, bbox_max)
        rendered_images.append(image)
        
        # Save image
        img_np = (image.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np)
        img_pil.save(images_dir / f"view_{view_idx:03d}.png")
    
    # Save camera data
    np.savez(
        output_dir / "cameras.npz",
        extrinsics=camera_poses.numpy(),
        intrinsics=intrinsics.numpy(),
        bbox_min=bbox_min.numpy(),
        bbox_max=bbox_max.numpy(),
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
        "scene_id": output_dir.name,
        "source_file": str(npz_path),
        "num_views": num_views,
        "image_size": list(renderer.image_size),
        "voxel_shape": list(rgbsigma.shape[:3]),
    }
    
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Generate multi-view images from NeRF-MAE voxel grids")
    
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Path to NeRF-MAE pretrain data (contains features/ and nerfmae_split.npz)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for rendered data")
    parser.add_argument("--num_views", type=int, default=8,
                        help="Number of views to render per scene")
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
    parser.add_argument("--max_scenes", type=int, default=None,
                        help="Maximum number of scenes to process (for testing)")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="Ratio of scenes for training set")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    features_dir = input_dir / "features"
    
    # Check input directory
    if not features_dir.exists():
        print(f"Error: Features directory not found: {features_dir}")
        return
    
    # Get all .npz files
    npz_files = sorted(features_dir.glob("*.npz"))
    if args.max_scenes:
        npz_files = npz_files[:args.max_scenes]
    
    print(f"Found {len(npz_files)} scene files")
    
    # Create renderer
    renderer = VoxelRenderer(
        image_size=(args.image_size, args.image_size),
        num_samples=args.num_samples,
        num_importance=args.num_importance,
        device=args.device,
    )
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process scenes
    all_metadata = []
    failed_scenes = []
    
    for npz_file in tqdm(npz_files, desc="Processing scenes"):
        scene_name = npz_file.stem
        scene_output_dir = output_dir / f"scene_{scene_name}"
        
        try:
            metadata = process_scene(
                str(npz_file),
                scene_output_dir,
                renderer,
                num_views=args.num_views,
                save_voxel=args.save_voxel,
            )
            all_metadata.append(metadata)
        except Exception as e:
            print(f"\nError processing {scene_name}: {e}")
            failed_scenes.append(scene_name)
            continue
    
    print(f"\nProcessed {len(all_metadata)} scenes, {len(failed_scenes)} failed")
    
    # Split into train/val
    num_train = int(len(all_metadata) * args.train_ratio)
    train_scenes = all_metadata[:num_train]
    val_scenes = all_metadata[num_train:]
    
    # Save index files
    with open(output_dir / "train_scenes.json", "w") as f:
        json.dump(train_scenes, f, indent=2)
    
    with open(output_dir / "val_scenes.json", "w") as f:
        json.dump(val_scenes, f, indent=2)
    
    # Save config
    config = {
        "num_views": args.num_views,
        "image_size": [args.image_size, args.image_size],
        "num_samples": args.num_samples,
        "num_importance": args.num_importance,
        "save_voxel": args.save_voxel,
        "num_train_scenes": len(train_scenes),
        "num_val_scenes": len(val_scenes),
        "failed_scenes": failed_scenes,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"\nData generation complete!")
    print(f"  Train scenes: {len(train_scenes)}")
    print(f"  Val scenes: {len(val_scenes)}")
    print(f"  Output directory: {output_dir}")


if __name__ == "__main__":
    main()
