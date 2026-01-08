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
        for i in range(self.num_views):
            pose = camera_poses[i]
            img = self._render_from_voxel(rgbsigma, pose, bbox_min, bbox_max)
            images.append(img)
        
        images = torch.stack(images, dim=0)  # [S, 3, H, W]
        
        # Create camera intrinsics (simple pinhole model)
        focal = self.image_size[0]  # Approximate focal length
        intrinsics = torch.tensor([
            [focal, 0, self.image_size[1] / 2],
            [0, focal, self.image_size[0] / 2],
            [0, 0, 1]
        ], dtype=torch.float32)
        
        return {
            "images": images,                    # [S, 3, H, W]
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
    ) -> torch.Tensor:
        """
        Render an image from the voxel grid using volume rendering.
        
        Adapted from render.py's render_rays, replacing MLP network queries
        with trilinear voxel grid sampling.
        
        Args:
            rgbsigma: Voxel grid [X, Y, Z, 4]
            camera_pose: Camera-to-world matrix [4, 4]
            bbox_min: Minimum corner of bounding box [3]
            bbox_max: Maximum corner of bounding box [3]
            
        Returns:
            image: Rendered image [3, H, W]
        """
        #TODO:
    

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
    for key in ['images', 'camera_poses', 'camera_intrinsics', 'bbox_min', 'bbox_max']:
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
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Images shape: {sample['images'].shape}")
    print(f"rgbsigma shape: {sample['rgbsigma'].shape}")
    print(f"Camera poses shape: {sample['camera_poses'].shape}")
    
    # Print rendering statistics
    print(f"\nRendering statistics:")
    print(f"  RGB min: {sample['images'].min():.4f}, max: {sample['images'].max():.4f}")
    print(f"  RGB mean: {sample['images'].mean():.4f}")
    
    # Save rendered images for visualization
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, min(4, args.num_views), figsize=(16, 4))
    if min(4, args.num_views) == 1:
        axes = [axes]
    
    for i in range(min(4, args.num_views)):
        img = sample['images'][i].permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        axes[i].imshow(img)
        axes[i].set_title(f"View {i}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig("test_rendered_views.png")
    print("Saved test_rendered_views.png")
