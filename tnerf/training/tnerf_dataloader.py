# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
T-NeRF Data Loaders for NLPHead training.

TNerfDataset:
- Loads pre-rendered multi-view images from generate_data.py output
- Each sample contains: multi-view images + camera parameters + voxel grid (optional)
- Designed for training VGGT to predict NeRF MLP parameters

Data Structure Expected (from generate_data.py):
    data_dir/
        train_samples.json
        val_samples.json
        test_samples.json
        config.json
        samples/
            3dfront_2140_00/
                images/
                    view_000.png
                    view_001.png
                    ...
                cameras.npz
                rgbsigma.npz (optional)
            3dfront_2140_01/
                ...
"""

import os
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T


class TNerfDataset(Dataset):
    """
    Dataset for T-NeRF (Transformer-based NeRF) training.
    
    Each sample contains:
    - images: Multi-view images [S, 3, H, W]
    - camera_extrinsics: Camera poses [S, 4, 4]
    - camera_intrinsics: Camera intrinsic matrix [3, 3]
    - bbox_min, bbox_max: Scene bounding box [3]
    - rgbsigma: Voxel grid [X, Y, Z, 4] (when training, for loss computation)
    """
    
    def __init__(
        self,
        data_dir: str,
        voxel_dir: str,
        split: str = "train",
        image_size: Tuple[int, int] = (518, 518),
        num_views: int = 8,
        transform: Optional[T.Compose] = None,
        load_voxel: bool = True,
    ):
        """
        Initialize T-NeRF dataset.
        
        Args:
            data_dir: Directory containing rendered data from generate_data.py
            split: "train", "val", or "test"
            image_size: Target image size (H, W) for resizing
            num_views: Number of views to load per sample
            transform: Optional image transforms
            load_voxel: Whether to load voxel grid (for loss computation)
            voxel_dir: Directory containing voxel grids
        """
        self.data_dir = Path(data_dir)
        self.samples_dir = self.data_dir / "samples"  # Containing img and camera data
        self.voxel_dir = Path(voxel_dir) # Containing voxel grids  \path\to\features
        self.split = split
        self.image_size = image_size
        self.num_views = num_views
        self.load_voxel = load_voxel
        
        if transform is None:
            self.transform = T.Compose([
                T.Resize(image_size),
                T.ToTensor(),
            ])
        else:
            self.transform = transform
        
        self.samples = self._load_index()
        print(f"Loaded {len(self.samples)} samples for {split} split")
    
    def _load_index(self) -> List[Dict]:
        """Load dataset index from JSON file."""
        # Try new format first (from generate_data.py with split support)
        index_path = self.data_dir / f"{self.split}_samples.json"
        
        if index_path.exists():
            with open(index_path, "r") as f:
                data = json.load(f)
                samples = data["samples"]
            return samples
        
        else:
            print(f"Warning: Index file {index_path} not found.")
            return []
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def _get_sample_path(self, sample_info: Dict) -> Path:
        """Get the path to a sample directory."""
        sample_id = sample_info.get("sample_id", "")
        
        # Try new structure first
        sample_path = self.samples_dir / sample_id
        if sample_path.exists():
            return sample_path
        else:        
            raise FileNotFoundError(f"Sample directory not found for: {sample_id}")
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load a single sample."""
        sample_info = self.samples[idx]
        sample_path = self._get_sample_path(sample_info)
        sample_id = sample_info.get("sample_id", "")
        
        images_dir = sample_path / "images"
        
        # Load images
        images = []
        image_files = sorted(images_dir.glob("view_*.png"))
        
        for i, img_path in enumerate(image_files[:self.num_views]):
            img = Image.open(img_path).convert("RGB")
            img = self.transform(img)
            images.append(img)
        
        # pad with last image or zeros
        while len(images) < self.num_views:
            if images:
                images.append(images[-1].clone())
            else:
                images.append(torch.zeros(3, *self.image_size))
        
        images = torch.stack(images[:self.num_views], dim=0)  # [S, 3, H, W]
        
        result = {
            "images": images,
            "sample_id": sample_id,
        }
        
        # Load camera parameters
        camera_path = sample_path / "cameras.npz"
        if camera_path.exists():
            camera_data = np.load(camera_path)
            
            if "extrinsics" in camera_data:
                extrinsics = torch.from_numpy(camera_data["extrinsics"]).float()
                # Only use first num_views cameras
                result["camera_extrinsics"] = extrinsics[:self.num_views]
            
            if "intrinsics" in camera_data:
                result["camera_intrinsics"] = torch.from_numpy(camera_data["intrinsics"]).float()
            
            if "bbox_min" in camera_data:
                result["bbox_min"] = torch.from_numpy(camera_data["bbox_min"]).float()
            
            if "bbox_max" in camera_data:
                result["bbox_max"] = torch.from_numpy(camera_data["bbox_max"]).float()
        
        # Load voxel grid for loss computation
        if self.load_voxel:
            voxel_path = self.voxel_dir / f"{sample_id}.npz"
            if voxel_path.exists():
                voxel_data = np.load(voxel_path)
                rgbsigma = torch.from_numpy(voxel_data["rgbsigma"]).float()
                # Clamp values
                rgbsigma[..., 3] = torch.clamp(rgbsigma[..., 3], min=0.0)
                rgbsigma[..., :3] = torch.clamp(rgbsigma[..., :3], 0.0, 1.0)
                result["rgbsigma"] = rgbsigma
            else:
                raise FileNotFoundError(f"Voxel grid not found for sample: {sample_id} in {voxel_path}")
        
        return result


import random


def tnerf_collate_fn(batch: List[Dict], min_views: int = 2, max_views: int = 8) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for T-NeRF dataset.
    
    Randomly selects a number of views, and all samples in the batch use the same number of views.
    This helps the model learn from varying numbers of input views.
    
    Args:
        batch: List of samples from dataset
        min_views: Minimum number of views to use
        max_views: Maximum number of views to use
        
    Returns:
        Collated batch with randomly selected number of views
    """
    # Randomly select number of views for this batch
    # All samples in the batch will have the same number of views
    available_views = batch[0]['images'].shape[0]
    max_views = min(max_views, available_views)
    min_views = min(min_views, max_views)
    num_views = random.randint(min_views, max_views)
    
    # Randomly select which views to use (same indices for all samples in batch)
    view_indices = sorted(random.sample(range(available_views), num_views))
    
    result = {}
    
    # Stack tensor fields with selected views
    if 'images' in batch[0]:
        # Select subset of views: [S, 3, H, W] -> [num_views, 3, H, W]
        result['images'] = torch.stack([
            item['images'][view_indices] for item in batch
        ], dim=0)  # [B, num_views, 3, H, W]
    
    if 'camera_extrinsics' in batch[0]:
        # Select subset of camera poses: [S, 4, 4] -> [num_views, 4, 4]
        result['camera_extrinsics'] = torch.stack([
            item['camera_extrinsics'][view_indices] for item in batch
        ], dim=0)  # [B, num_views, 4, 4]
    
    # These don't depend on number of views
    if 'camera_intrinsics' in batch[0]:
        result['camera_intrinsics'] = torch.stack([
            item['camera_intrinsics'] for item in batch
        ], dim=0)
    
    if 'bbox_min' in batch[0]:
        result['bbox_min'] = torch.stack([
            item['bbox_min'] for item in batch
        ], dim=0)
    
    if 'bbox_max' in batch[0]:
        result['bbox_max'] = torch.stack([
            item['bbox_max'] for item in batch
        ], dim=0)
    
    # Keep rgbsigma as list (variable sizes)
    if 'rgbsigma' in batch[0]:
        result['rgbsigma'] = [item['rgbsigma'] for item in batch]
    
    # Keep sample_id as list
    if 'sample_id' in batch[0]:
        result['sample_id'] = [item['sample_id'] for item in batch]
    
    # Store the number of views and indices used for this batch
    result['num_views'] = num_views
    result['view_indices'] = view_indices
    
    return result


def create_tnerf_collate_fn(min_views: int = 2, max_views: int = 8):
    """
    Create a collate function with specific min/max view settings.
    
    Args:
        min_views: Minimum number of views per batch
        max_views: Maximum number of views per batch
        
    Returns:
        Collate function with the specified settings
    """
    def collate_fn(batch):
        return tnerf_collate_fn(batch, min_views=min_views, max_views=max_views)
    return collate_fn


def create_tnerf_dataloaders(
    data_dir: str,
    voxel_dir: str,
    batch_size: int = 2,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (518, 518),
    num_views: int = 8,
    load_voxel: bool = True,
    min_views: int = 2,
    max_views: int = 8,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Create training, validation, and test dataloaders for T-NeRF.
    
    Args:
        data_dir: Directory containing rendered data from generate_data.py
        batch_size: Batch size
        num_workers: Number of data loading workers
        image_size: Target image size (for VGGT input)
        num_views: Number of views per sample (max views to load from disk)
        load_voxel: Whether to load voxel grids for loss computation
        min_views: Minimum number of views per batch (for random view sampling)
        max_views: Maximum number of views per batch (defaults to num_views)
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
        test_loader may be None if no test data exists
    """
    if max_views is None:
        max_views = num_views
    
    train_dataset = TNerfDataset(
        data_dir=data_dir,
        voxel_dir=voxel_dir,
        split="train",
        image_size=image_size,
        num_views=num_views,
        load_voxel=load_voxel,
    )
    
    val_dataset = TNerfDataset(
        data_dir=data_dir,
        voxel_dir=voxel_dir,
        split="val",
        image_size=image_size,
        num_views=num_views,
        load_voxel=load_voxel,
    )
    
    # Create collate function with random view sampling for training
    train_collate_fn = create_tnerf_collate_fn(min_views=min_views, max_views=max_views)
    # Use fixed max views for validation (consistent evaluation)
    val_collate_fn = create_tnerf_collate_fn(min_views=max_views, max_views=max_views)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=train_collate_fn,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=val_collate_fn,
    )
    
    # Try to create test loader
    test_loader = None
    try:
        test_dataset = TNerfDataset(
            data_dir=data_dir,
            voxel_dir=voxel_dir,
            split="test",
            image_size=image_size,
            num_views=num_views,
            load_voxel=load_voxel,
        )
        if len(test_dataset) > 0:
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
                collate_fn=val_collate_fn,  # Use fixed views for test
            )
    except Exception as e:
        print(f"Note: No test data available ({e})")
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    # Test the dataloader
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to rendered data directory from generate_data.py")
    parser.add_argument("--voxel_dir", type=str, required=True,
                        help="Path to voxel data directory")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_views", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=256)
    args = parser.parse_args()
    
    print("Testing T-NeRF dataloader...")
    
    train_loader, val_loader, test_loader = create_tnerf_dataloaders(
        data_dir=args.data_dir,
        voxel_dir=args.voxel_dir,
        batch_size=args.batch_size,
        num_views=args.num_views,
        image_size=(args.image_size, args.image_size),
        load_voxel=True,
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    if test_loader:
        print(f"Test batches: {len(test_loader)}")
    
    # Get a sample batch
    batch = next(iter(train_loader))
    print(f"\nBatch keys: {batch.keys()}")
    print(f"Images shape: {batch['images'].shape}")
    
    if 'camera_extrinsics' in batch:
        if isinstance(batch['camera_extrinsics'], torch.Tensor):
            print(f"Camera extrinsics shape: {batch['camera_extrinsics'].shape}")
        else:
            print(f"Camera extrinsics: {len(batch['camera_extrinsics'])} items")
    
    if 'rgbsigma' in batch:
        print(f"Voxel grids: {len(batch['rgbsigma'])} items")
        print(f"First voxel shape: {batch['rgbsigma'][0].shape}")
    
    print("\nDataloader test passed!")
