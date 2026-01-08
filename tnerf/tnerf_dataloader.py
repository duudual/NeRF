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

Data Structure Expected:
    data_dir/
        train_scenes.json
        val_scenes.json
        scene_xxx/
            images/
                view_000.png
                view_001.png
                ...
            cameras.npz
            rgbsigma.npz (optional)
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
    
    Loads pre-rendered multi-view images from generate_data.py output.
    
    Each sample contains:
    - images: Multi-view images [S, 3, H, W]
    - camera_extrinsics: Camera poses [S, 4, 4]
    - camera_intrinsics: Camera intrinsic matrix [3, 3]
    - bbox_min, bbox_max: Scene bounding box [3]
    - rgbsigma: Voxel grid [X, Y, Z, 4] (optional, for loss computation)
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
            data_dir: Directory containing rendered multi-view data
            split: "train" or "val"
            image_size: Target image size (H, W) for resizing
            num_views: Number of views to load per sample
            transform: Optional image transforms
            load_voxel: Whether to load voxel grid (for loss computation)
        """
        self.data_dir = Path(data_dir)
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
        index_path = self.data_dir / f"{self.split}_scenes.json"
        
        if index_path.exists():
            with open(index_path, "r") as f:
                scenes = json.load(f)
            # Convert relative paths to absolute
            for scene in scenes:
                if "scene_id" in scene:
                    scene["path"] = str(self.data_dir / scene["scene_id"])
            return scenes
        else:
            # Fallback: scan for scene directories
            print(f"Warning: Index file {index_path} not found, scanning directories...")
            samples = []
            scene_dirs = sorted(self.data_dir.glob("scene_*"))
            
            # Simple split: first 80% train, rest val
            num_train = int(len(scene_dirs) * 0.8)
            if self.split == "train":
                scene_dirs = scene_dirs[:num_train]
            else:
                scene_dirs = scene_dirs[num_train:]
            
            for scene_dir in scene_dirs:
                if scene_dir.is_dir() and (scene_dir / "images").exists():
                    samples.append({
                        "scene_id": scene_dir.name,
                        "path": str(scene_dir)
                    })
            return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load a single sample."""
        sample_info = self.samples[idx]
        scene_path = Path(sample_info.get("path", self.data_dir / sample_info["scene_id"]))
        images_dir = scene_path / "images"
        
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
            "scene_id": sample_info.get("scene_id", scene_path.name),
        }
        
        # Load camera parameters
        camera_path = scene_path / "cameras.npz"
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
            voxel_path = scene_path / "rgbsigma.npz"
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
    
    # Keep scene_id as list
    if 'scene_id' in batch[0]:
        result['scene_id'] = [item['scene_id'] for item in batch]
    
    return result


def create_tnerf_dataloaders(
    data_dir: str,
    batch_size: int = 2,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (518, 518),
    num_views: int = 8,
    load_voxel: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders for T-NeRF.
    
    Args:
        data_dir: Directory containing rendered data from generate_data.py
        batch_size: Batch size
        num_workers: Number of data loading workers
        image_size: Target image size (for VGGT input)
        num_views: Number of views per sample
        load_voxel: Whether to load voxel grids for loss computation
    
    Returns:
        Tuple of (train_loader, val_loader)
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
    
    return train_loader, val_loader


# Legacy alias for backward compatibility
NLPDataset = TNerfDataset
create_dataloaders = create_tnerf_dataloaders


if __name__ == "__main__":
    # Test the dataloader
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to rendered data directory")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_views", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=256)
    args = parser.parse_args()
    
    print("Testing T-NeRF dataloader...")
    
    train_loader, val_loader = create_tnerf_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_views=args.num_views,
        image_size=(args.image_size, args.image_size),
        load_voxel=True,
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Get a sample batch
    batch = next(iter(train_loader))
    print(f"\nBatch keys: {batch.keys()}")
    print(f"Images shape: {batch['images'].shape}")
    
    if 'camera_extrinsics' in batch:
        print(f"Camera extrinsics shape: {batch['camera_extrinsics'].shape}")
    
    if 'rgbsigma' in batch:
        print(f"Voxel grids: {len(batch['rgbsigma'])} items")
        print(f"First voxel shape: {batch['rgbsigma'][0].shape}")
    
    print("\nDataloader test passed!")
