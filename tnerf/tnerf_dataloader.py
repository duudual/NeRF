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
        """
        self.data_dir = Path(data_dir)
        self.samples_dir = self.data_dir / "samples"
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
            # Handle both formats: list or dict with "samples" key
            if isinstance(data, dict) and "samples" in data:
                samples = data["samples"]
            else:
                samples = data
            return samples
        
        # Fallback: try old format (train_scenes.json)
        old_index_path = self.data_dir / f"{self.split}_scenes.json"
        if old_index_path.exists():
            with open(old_index_path, "r") as f:
                return json.load(f)
        
        # Fallback: scan for sample directories
        print(f"Warning: Index file not found, scanning directories...")
        samples = []
        
        if self.samples_dir.exists():
            sample_dirs = sorted(self.samples_dir.glob("*"))
        else:
            # Try old structure
            sample_dirs = sorted(self.data_dir.glob("scene_*"))
        
        for sample_dir in sample_dirs:
            if sample_dir.is_dir() and (sample_dir / "images").exists():
                samples.append({
                    "sample_id": sample_dir.name,
                })
        
        # Simple split if no split file
        if self.split == "train":
            return samples[:int(len(samples) * 0.8)]
        elif self.split == "val":
            return samples[int(len(samples) * 0.8):int(len(samples) * 0.9)]
        else:  # test
            return samples[int(len(samples) * 0.9):]
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def _get_sample_path(self, sample_info: Dict) -> Path:
        """Get the path to a sample directory."""
        sample_id = sample_info.get("sample_id", sample_info.get("scene_id", ""))
        
        # Try new structure first
        sample_path = self.samples_dir / sample_id
        if sample_path.exists():
            return sample_path
        
        # Try old structure
        sample_path = self.data_dir / sample_id
        if sample_path.exists():
            return sample_path
        
        # Try with scene_ prefix (old format)
        sample_path = self.data_dir / f"scene_{sample_id}"
        if sample_path.exists():
            return sample_path
        
        raise FileNotFoundError(f"Sample directory not found for: {sample_id}")
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load a single sample."""
        sample_info = self.samples[idx]
        sample_path = self._get_sample_path(sample_info)
        sample_id = sample_info.get("sample_id", sample_info.get("scene_id", sample_path.name))
        
        images_dir = sample_path / "images"
        
        # Load images
        images = []
        image_files = sorted(images_dir.glob("view_*.png"))
        
        if len(image_files) == 0:
            # Fallback to any image format
            image_files = sorted(images_dir.glob("*.png")) + sorted(images_dir.glob("*.jpg"))
        
        for i, img_path in enumerate(image_files[:self.num_views]):
            img = Image.open(img_path).convert("RGB")
            img = self.transform(img)
            images.append(img)
        
        # Pad with last image if necessary
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
            voxel_path = sample_path / "rgbsigma.npz"
            if voxel_path.exists():
                voxel_data = np.load(voxel_path)
                rgbsigma = torch.from_numpy(voxel_data["rgbsigma"]).float()
                # Clamp values
                rgbsigma[..., 3] = torch.clamp(rgbsigma[..., 3], min=0.0)
                rgbsigma[..., :3] = torch.clamp(rgbsigma[..., :3], 0.0, 1.0)
                result["rgbsigma"] = rgbsigma
        
        return result


def tnerf_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for T-NeRF dataset.
    
    Handles variable-sized voxel grids by keeping them as a list.
    """
    result = {}
    
    # Stack tensor fields that have consistent shapes
    tensor_keys = ['images', 'camera_extrinsics', 'camera_intrinsics', 'bbox_min', 'bbox_max']
    for key in tensor_keys:
        if key in batch[0]:
            try:
                result[key] = torch.stack([item[key] for item in batch], dim=0)
            except RuntimeError:
                # If shapes don't match, keep as list
                result[key] = [item[key] for item in batch]
    
    # Keep rgbsigma as list (variable sizes)
    if 'rgbsigma' in batch[0]:
        result['rgbsigma'] = [item['rgbsigma'] for item in batch]
    
    # Keep sample_id as list
    if 'sample_id' in batch[0]:
        result['sample_id'] = [item['sample_id'] for item in batch]
    
    return result


def create_tnerf_dataloaders(
    data_dir: str,
    batch_size: int = 2,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (518, 518),
    num_views: int = 8,
    load_voxel: bool = True,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Create training, validation, and test dataloaders for T-NeRF.
    
    Args:
        data_dir: Directory containing rendered data from generate_data.py
        batch_size: Batch size
        num_workers: Number of data loading workers
        image_size: Target image size (for VGGT input)
        num_views: Number of views per sample
        load_voxel: Whether to load voxel grids for loss computation
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
        test_loader may be None if no test data exists
    """
    train_dataset = TNerfDataset(
        data_dir=data_dir,
        split="train",
        image_size=image_size,
        num_views=num_views,
        load_voxel=load_voxel,
    )
    
    val_dataset = TNerfDataset(
        data_dir=data_dir,
        split="val",
        image_size=image_size,
        num_views=num_views,
        load_voxel=load_voxel,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=tnerf_collate_fn,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=tnerf_collate_fn,
    )
    
    # Try to create test loader
    test_loader = None
    try:
        test_dataset = TNerfDataset(
            data_dir=data_dir,
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
                collate_fn=tnerf_collate_fn,
            )
    except Exception as e:
        print(f"Note: No test data available ({e})")
    
    return train_loader, val_loader, test_loader


# Legacy aliases for backward compatibility
NLPDataset = TNerfDataset
create_dataloaders = lambda *args, **kwargs: create_tnerf_dataloaders(*args, **kwargs)[:2]


if __name__ == "__main__":
    # Test the dataloader
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to rendered data directory from generate_data.py")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_views", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=256)
    args = parser.parse_args()
    
    print("Testing T-NeRF dataloader...")
    
    train_loader, val_loader, test_loader = create_tnerf_dataloaders(
        data_dir=args.data_dir,
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
