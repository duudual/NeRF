# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
T-NeRF Data Loaders for NLPHead training.

NLPDataset:
- Loads multi-view image data for NLP training
- Each sample contains: multi-view images for NeRF MLP parameter prediction
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


class NLPDataset(Dataset):
    """
    Dataset for NLP (NeRF MLP Parameters) training.
    
    Each sample contains:
    - images: Multi-view images [S, 3, H, W]
    - target_render: Target rendered images for supervision (optional)
    - ray_origins: Ray origins for volume rendering (optional)
    - ray_directions: Ray directions for volume rendering (optional)
    
    This dataset loads multi-view images for NLP training.
    The loss is computed by rendering with predicted MLP parameters
    and comparing with the input images.
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        image_size: Tuple[int, int] = (518, 518),
        num_views: int = 8,
        transform: Optional[T.Compose] = None,
    ):
        """
        Initialize NLP dataset.
        
        Args:
            data_dir: Directory containing multi-view image data
            split: "train" or "val"
            image_size: Target image size (H, W)
            num_views: Number of views per sample
            transform: Optional image transforms
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.image_size = image_size
        self.num_views = num_views
        
        if transform is None:
            self.transform = T.Compose([
                T.Resize(image_size),
                T.ToTensor(),
            ])
        else:
            self.transform = transform
        
        self.samples = self._load_index()
    
    def _load_index(self) -> List[Dict]:
        """Load dataset index."""
        index_path = self.data_dir / f"{self.split}_scenes.json"
        
        if index_path.exists():
            with open(index_path, "r") as f:
                return json.load(f)
        else:
            # Scan for scene directories
            samples = []
            scene_dirs = sorted(self.data_dir.glob("scene_*"))
            for scene_dir in scene_dirs:
                if scene_dir.is_dir():
                    samples.append({"path": str(scene_dir)})
            return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample_info = self.samples[idx]
        sample_path = Path(sample_info["path"])
        
        # Load images
        images = []
        image_files = sorted(sample_path.glob("*.png")) + sorted(sample_path.glob("*.jpg"))
        
        for i, img_path in enumerate(image_files[:self.num_views]):
            img = Image.open(img_path).convert("RGB")
            img = self.transform(img)
            images.append(img)
        
        # Pad if necessary
        while len(images) < self.num_views:
            images.append(images[-1] if images else torch.zeros(3, *self.image_size))
        
        images = torch.stack(images[:self.num_views], dim=0)  # [S, 3, H, W]
        
        result = {"images": images}
        
        # Load camera parameters if available
        camera_path = sample_path / "cameras.npz"
        if camera_path.exists():
            camera_data = np.load(camera_path)
            if "extrinsics" in camera_data:
                result["camera_extrinsics"] = torch.from_numpy(camera_data["extrinsics"]).float()
            if "intrinsics" in camera_data:
                result["camera_intrinsics"] = torch.from_numpy(camera_data["intrinsics"]).float()
        
        return result


def create_dataloaders(
    data_dir: str,
    batch_size: int = 2,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (518, 518),
    num_views: int = 8,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders.
    
    Args:
        data_dir: Main data directory
        batch_size: Batch size
        num_workers: Number of data loading workers
        image_size: Target image size
        num_views: Number of views per sample
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_dataset = NLPDataset(
        data_dir=data_dir,
        split="train",
        image_size=image_size,
        num_views=num_views,
    )
    val_dataset = NLPDataset(
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
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader
