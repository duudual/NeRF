# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
NColor Data Generator

This script generates training data for NColorHead.

Pipeline:
1. Load multi-view images
2. For each image, predict 3D points using VGGT point_head
3. Stack all predicted point clouds together
4. Add random "air points" (not on surfaces) with alpha=1 (transparent) and white color
5. Query NeRF MLP for color values at each 3D point from 6 directions
6. Generate colormap ground truth for each 2D pixel

The output format:
- target_colors: [S, H, W, 18] - 6 directions * 3 RGB
- target_alpha: [S, H, W, 1] - opacity (0 for surface, 1 for air)
- point_masks: [S, H, W] - valid point masks
- is_air_point: [S, H, W] - air point masks

Usage:
    python ncolor_data_generator.py --input_dir /path/to/images --output_dir /path/to/output
"""

import os
import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T


def parse_args():
    parser = argparse.ArgumentParser(description="NColor Data Generator")
    
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing input images or scene folders")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save generated data")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to pretrained VGGT model (optional)")
    parser.add_argument("--nerf_mlp_path", type=str, default=None,
                        help="Path to pretrained NeRF MLP weights (optional)")
    
    # Data generation settings
    parser.add_argument("--image_size", type=int, default=518,
                        help="Target image size")
    parser.add_argument("--num_views", type=int, default=8,
                        help="Number of views per scene")
    parser.add_argument("--air_point_ratio", type=float, default=0.1,
                        help="Ratio of air points to add (relative to surface points)")
    parser.add_argument("--air_point_range", type=float, default=2.0,
                        help="Range for random air point sampling (in normalized coordinates)")
    
    # Split settings
    parser.add_argument("--train_ratio", type=float, default=0.9,
                        help="Ratio of training samples")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    
    return parser.parse_args()


class NeRFMLPInterface:
    """
    Interface for querying NeRF MLP to get color values at 3D points.
    
    This is a placeholder implementation. The actual implementation should:
    1. Load pretrained NeRF MLP weights
    2. Apply positional encoding to 3D points
    3. Query the MLP for color and density at each point
    """
    
    # 6 directions: up, down, left, right, front, back
    DIRECTIONS = torch.tensor([
        [0, 1, 0],   # up
        [0, -1, 0],  # down
        [-1, 0, 0],  # left
        [1, 0, 0],   # right
        [0, 0, 1],   # front
        [0, 0, -1],  # back
    ], dtype=torch.float32)
    
    def __init__(self, mlp_weights_path: Optional[str] = None, device: str = "cuda"):
        """
        Initialize NeRF MLP interface.
        
        Args:
            mlp_weights_path: Path to pretrained MLP weights
            device: Device to use
        """
        self.device = torch.device(device)
        self.directions = self.DIRECTIONS.to(self.device)
        
        # Load MLP if path provided
        if mlp_weights_path and os.path.exists(mlp_weights_path):
            self._load_mlp(mlp_weights_path)
        else:
            self.mlp = None
    
    def _load_mlp(self, path: str):
        """Load pretrained NeRF MLP weights."""
        # TODO: Implement actual MLP loading
        # This would load weights for the NeRFMLP class from nlp_head.py
        pass
    
    def query_colors_6dir(
        self, 
        points_3d: torch.Tensor,
        epsilon: float = 0.01,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Query color values at 3D points from 6 directions.
        
        For each point, we query colors by looking at the point from 6 orthogonal
        directions (up, down, left, right, front, back).
        
        Args:
            points_3d: 3D point coordinates [N, 3]
            epsilon: Small offset for sampling neighboring points
        
        Returns:
            colors: RGB colors from 6 directions [N, 18] (6 * 3)
            densities: Density values [N, 1]
        """
        N = points_3d.shape[0]
        device = points_3d.device
        
        if self.mlp is None:
            # Placeholder: return random colors
            # In practice, this should query the actual NeRF MLP
            colors = torch.rand(N, 18, device=device)
            densities = torch.rand(N, 1, device=device)
            return colors, densities
        
        # Query colors from 6 directions
        all_colors = []
        
        for i, direction in enumerate(self.directions):
            # Direction vector normalized
            dir_vec = direction.unsqueeze(0).expand(N, -1)  # [N, 3]
            
            # Apply positional encoding to points and direction
            pos_encoded = self._positional_encode(points_3d, L=10)  # [N, 63]
            dir_encoded = self._positional_encode(dir_vec, L=4)     # [N, 27]
            
            # Query MLP
            sigma, color = self._query_mlp(pos_encoded, dir_encoded)
            all_colors.append(color)
        
        # Concatenate colors from all directions
        colors = torch.cat(all_colors, dim=-1)  # [N, 18]
        
        # Get density (use average from all queries)
        densities = sigma
        
        return colors, densities
    
    def _positional_encode(self, x: torch.Tensor, L: int) -> torch.Tensor:
        """
        Apply positional encoding to input.
        
        Args:
            x: Input tensor [N, D]
            L: Number of frequency bands
        
        Returns:
            Encoded tensor [N, D * (2*L + 1)]
        """
        encoded = [x]
        for i in range(L):
            freq = 2.0 ** i
            encoded.append(torch.sin(freq * np.pi * x))
            encoded.append(torch.cos(freq * np.pi * x))
        return torch.cat(encoded, dim=-1)
    
    def _query_mlp(
        self, 
        pos_encoded: torch.Tensor, 
        dir_encoded: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Query NeRF MLP with encoded position and direction.
        
        Args:
            pos_encoded: Positional encoded position [N, 63]
            dir_encoded: Positional encoded direction [N, 27]
        
        Returns:
            sigma: Density [N, 1]
            color: RGB color [N, 3]
        """
        if self.mlp is not None:
            return self.mlp(pos_encoded, dir_encoded)
        else:
            # Placeholder
            N = pos_encoded.shape[0]
            sigma = torch.rand(N, 1, device=pos_encoded.device)
            color = torch.rand(N, 3, device=pos_encoded.device)
            return sigma, color


class NColorDataGenerator:
    """
    Generator for NColor training data.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        nerf_mlp_path: Optional[str] = None,
        image_size: int = 518,
        device: str = "cuda",
    ):
        """
        Initialize data generator.
        
        Args:
            model_path: Path to VGGT model for 3D point prediction
            nerf_mlp_path: Path to NeRF MLP for color queries
            image_size: Target image size
            device: Device to use
        """
        self.device = torch.device(device)
        self.image_size = image_size
        
        # Image transforms
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
        ])
        
        # Initialize VGGT model for point prediction
        self.vggt_model = None
        if model_path and os.path.exists(model_path):
            self._load_vggt_model(model_path)
        
        # Initialize NeRF MLP interface
        self.nerf_interface = NeRFMLPInterface(nerf_mlp_path, device)
    
    def _load_vggt_model(self, path: str):
        """Load VGGT model for 3D point prediction."""
        try:
            from vggt.models.vggt import VGGT
            
            self.vggt_model = VGGT(
                enable_camera=False,
                enable_point=True,
                enable_depth=False,
                enable_track=False,
                enable_ncolor=False,
                enable_nlp=False,
            )
            
            state_dict = torch.load(path, map_location="cpu")
            if "model" in state_dict:
                state_dict = state_dict["model"]
            
            self.vggt_model.load_state_dict(state_dict, strict=False)
            self.vggt_model = self.vggt_model.to(self.device)
            self.vggt_model.eval()
            
            print(f"Loaded VGGT model from {path}")
        except Exception as e:
            print(f"Failed to load VGGT model: {e}")
            self.vggt_model = None
    
    def generate_sample(
        self,
        image_paths: List[str],
        air_point_ratio: float = 0.1,
        air_point_range: float = 2.0,
    ) -> Dict[str, np.ndarray]:
        """
        Generate NColor training sample from multi-view images.
        
        Args:
            image_paths: List of paths to input images
            air_point_ratio: Ratio of air points to add
            air_point_range: Range for air point sampling
        
        Returns:
            Dictionary containing:
            - target_colors: [S, H, W, 18]
            - target_alpha: [S, H, W, 1]
            - point_masks: [S, H, W]
            - is_air_point: [S, H, W]
        """
        # Load and preprocess images
        images = []
        for path in image_paths:
            img = Image.open(path).convert("RGB")
            img_tensor = self.transform(img)
            images.append(img_tensor)
        
        images = torch.stack(images, dim=0)  # [S, 3, H, W]
        images = images.unsqueeze(0).to(self.device)  # [1, S, 3, H, W]
        
        S, H, W = images.shape[1], self.image_size, self.image_size
        
        # Predict 3D points using VGGT
        with torch.no_grad():
            if self.vggt_model is not None:
                predictions = self.vggt_model(images)
                world_points = predictions["world_points"]  # [1, S, H, W, 3]
                world_points_conf = predictions["world_points_conf"]  # [1, S, H, W]
            else:
                # Placeholder: random points
                world_points = torch.randn(1, S, H, W, 3, device=self.device)
                world_points_conf = torch.ones(1, S, H, W, device=self.device)
        
        # Create point mask (valid points with high confidence)
        point_masks = (world_points_conf > 0.5).squeeze(0)  # [S, H, W]
        
        # Initialize outputs
        target_colors = torch.zeros(S, H, W, 18, device=self.device)
        target_alpha = torch.zeros(S, H, W, 1, device=self.device)
        is_air_point = torch.zeros(S, H, W, device=self.device)
        
        # For each view, query colors at predicted 3D points
        for s in range(S):
            # Get valid points for this view
            valid_mask = point_masks[s]  # [H, W]
            points_3d = world_points[0, s]  # [H, W, 3]
            
            # Flatten for batch processing
            points_flat = points_3d.view(-1, 3)  # [H*W, 3]
            
            # Query colors from 6 directions
            colors, densities = self.nerf_interface.query_colors_6dir(points_flat)
            
            # Reshape back to image dimensions
            colors = colors.view(H, W, 18)  # [H, W, 18]
            densities = densities.view(H, W, 1)  # [H, W, 1]
            
            # Set target colors and alpha
            target_colors[s] = colors
            target_alpha[s] = 1.0 - densities  # High density = low alpha (opaque)
        
        # Add random air points
        num_air_points = int(point_masks.sum().item() * air_point_ratio)
        if num_air_points > 0:
            air_points = self._generate_air_points(
                world_points[0], 
                num_air_points, 
                air_point_range
            )
            
            # Mark air point locations
            for s in range(S):
                # Randomly select pixels to mark as air points
                flat_indices = torch.randperm(H * W)[:num_air_points // S]
                h_indices = flat_indices // W
                w_indices = flat_indices % W
                
                is_air_point[s, h_indices, w_indices] = 1.0
                target_alpha[s, h_indices, w_indices, 0] = 1.0  # Air = fully transparent
                target_colors[s, h_indices, w_indices, :] = 1.0  # White color for air
        
        return {
            "target_colors": target_colors.cpu().numpy(),
            "target_alpha": target_alpha.cpu().numpy(),
            "point_masks": point_masks.cpu().numpy(),
            "is_air_point": is_air_point.cpu().numpy(),
        }
    
    def _generate_air_points(
        self,
        surface_points: torch.Tensor,
        num_points: int,
        range_scale: float,
    ) -> torch.Tensor:
        """
        Generate random air points (not on surfaces).
        
        Args:
            surface_points: Surface point cloud [S, H, W, 3]
            num_points: Number of air points to generate
            range_scale: Scale for random sampling range
        
        Returns:
            Air points [N, 3]
        """
        # Compute bounding box of surface points
        all_points = surface_points.reshape(-1, 3)
        valid_mask = ~torch.isnan(all_points).any(dim=-1)
        valid_points = all_points[valid_mask]
        
        if valid_points.shape[0] == 0:
            # Return random points in unit cube
            return torch.rand(num_points, 3, device=surface_points.device) * range_scale
        
        min_coords = valid_points.min(dim=0).values
        max_coords = valid_points.max(dim=0).values
        center = (min_coords + max_coords) / 2
        extent = (max_coords - min_coords) * range_scale
        
        # Generate random points within extended bounding box
        air_points = torch.rand(num_points, 3, device=surface_points.device)
        air_points = (air_points - 0.5) * extent + center
        
        return air_points


def process_scene(
    generator: NColorDataGenerator,
    scene_dir: Path,
    output_dir: Path,
    num_views: int,
    air_point_ratio: float,
    air_point_range: float,
) -> bool:
    """
    Process a single scene directory.
    
    Args:
        generator: Data generator instance
        scene_dir: Path to scene directory
        output_dir: Path to output directory
        num_views: Number of views to use
        air_point_ratio: Ratio of air points
        air_point_range: Range for air points
    
    Returns:
        True if successful, False otherwise
    """
    # Find image files
    image_extensions = [".png", ".jpg", ".jpeg"]
    image_paths = []
    
    for ext in image_extensions:
        image_paths.extend(sorted(scene_dir.glob(f"*{ext}")))
    
    if len(image_paths) < num_views:
        print(f"Not enough images in {scene_dir}: {len(image_paths)}")
        return False
    
    # Select images
    image_paths = image_paths[:num_views]
    
    # Generate sample
    try:
        sample = generator.generate_sample(
            [str(p) for p in image_paths],
            air_point_ratio=air_point_ratio,
            air_point_range=air_point_range,
        )
    except Exception as e:
        print(f"Error generating sample for {scene_dir}: {e}")
        return False
    
    # Save sample
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy images
    for i, src_path in enumerate(image_paths):
        dst_path = output_dir / f"image_{i:04d}.png"
        img = Image.open(src_path).convert("RGB")
        img.save(dst_path)
    
    # Save arrays
    np.save(output_dir / "target_colors.npy", sample["target_colors"])
    np.save(output_dir / "target_alpha.npy", sample["target_alpha"])
    np.save(output_dir / "point_masks.npy", sample["point_masks"])
    np.save(output_dir / "is_air_point.npy", sample["is_air_point"])
    
    return True


def main():
    args = parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Initialize generator
    generator = NColorDataGenerator(
        model_path=args.model_path,
        nerf_mlp_path=args.nerf_mlp_path,
        image_size=args.image_size,
        device=args.device,
    )
    
    # Find scene directories
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Look for scene directories or treat input as single scene
    scene_dirs = list(input_dir.glob("scene_*"))
    if not scene_dirs:
        # Check for image files directly
        image_files = list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg"))
        if image_files:
            scene_dirs = [input_dir]
        else:
            # Look for any subdirectories with images
            scene_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
    
    print(f"Found {len(scene_dirs)} scenes to process")
    
    # Split into train/val
    np.random.shuffle(scene_dirs)
    num_train = int(len(scene_dirs) * args.train_ratio)
    train_scenes = scene_dirs[:num_train]
    val_scenes = scene_dirs[num_train:]
    
    # Process training scenes
    train_index = []
    print("Processing training scenes...")
    for i, scene_dir in enumerate(tqdm(train_scenes)):
        sample_output_dir = output_dir / f"train_{i:06d}"
        success = process_scene(
            generator, scene_dir, sample_output_dir,
            args.num_views, args.air_point_ratio, args.air_point_range
        )
        if success:
            train_index.append({"path": str(sample_output_dir)})
    
    # Process validation scenes
    val_index = []
    print("Processing validation scenes...")
    for i, scene_dir in enumerate(tqdm(val_scenes)):
        sample_output_dir = output_dir / f"val_{i:06d}"
        success = process_scene(
            generator, scene_dir, sample_output_dir,
            args.num_views, args.air_point_ratio, args.air_point_range
        )
        if success:
            val_index.append({"path": str(sample_output_dir)})
    
    # Save indices
    with open(output_dir / "train_index.json", "w") as f:
        json.dump(train_index, f, indent=2)
    
    with open(output_dir / "val_index.json", "w") as f:
        json.dump(val_index, f, indent=2)
    
    print(f"Generated {len(train_index)} training samples and {len(val_index)} validation samples")
    print(f"Output saved to {output_dir}")


if __name__ == "__main__":
    main()
