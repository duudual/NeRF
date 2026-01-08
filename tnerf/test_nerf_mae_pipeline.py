"""
Test script for NeRF-MAE data loading and rendering pipeline.

This script tests:
1. Loading NeRF-MAE voxel data
2. Rendering multi-view images from voxel grid
3. Volume rendering with predicted MLP parameters

Usage:
    python test_nerf_mae_pipeline.py --data_dir /path/to/pretrain
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nerf_mae_dataloader import NeRFMAEDataset
from volume_renderer import VolumeRenderer, VoxelRenderer, BatchedNeRFMLP


def test_data_loading(args):
    """Test data loading."""
    print("=" * 50)
    print("Testing Data Loading")
    print("=" * 50)
    
    dataset = NeRFMAEDataset(
        data_dir=args.data_dir,
        split="train",
        image_size=(args.image_size, args.image_size),
        num_views=args.num_views,
        num_samples=64,
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    if len(dataset) == 0:
        print("ERROR: No samples found!")
        return None
    
    # Load first sample
    sample = dataset[0]
    
    print("\nSample keys:", list(sample.keys()))
    print(f"Images shape: {sample['images'].shape}")
    print(f"rgbsigma shape: {sample['rgbsigma'].shape}")
    print(f"Camera poses shape: {sample['camera_poses'].shape}")
    print(f"bbox_min: {sample['bbox_min']}")
    print(f"bbox_max: {sample['bbox_max']}")
    
    return sample


def test_rendering(sample, args):
    """Test volume rendering."""
    print("\n" + "=" * 50)
    print("Testing Volume Rendering")
    print("=" * 50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Test voxel renderer
    print("\nTesting VoxelRenderer...")
    voxel_renderer = VoxelRenderer(num_samples=64).to(device)
    
    rgbsigma = sample['rgbsigma'].unsqueeze(0).to(device)  # [1, H, W, D, 4]
    camera_pose = sample['camera_poses'][0].unsqueeze(0).to(device)  # [1, 4, 4]
    camera_intrinsics = sample['camera_intrinsics'].unsqueeze(0).to(device)  # [1, 3, 3]
    bbox_min = sample['bbox_min'].unsqueeze(0).to(device)
    bbox_max = sample['bbox_max'].unsqueeze(0).to(device)
    
    with torch.no_grad():
        result = voxel_renderer(
            rgbsigma, camera_pose, camera_intrinsics,
            bbox_min, bbox_max, (args.image_size, args.image_size)
        )
    
    rendered_voxel = result['rgb']  # [1, 3, H, W]
    print(f"Voxel rendered shape: {rendered_voxel.shape}")
    print(f"Voxel rendered range: [{rendered_voxel.min():.3f}, {rendered_voxel.max():.3f}]")
    
    # Test MLP renderer with random parameters
    print("\nTesting VolumeRenderer (MLP)...")
    mlp_renderer = VolumeRenderer(num_samples=32).to(device)
    
    # Random MLP parameters
    num_params = BatchedNeRFMLP.TOTAL_PARAMS
    print(f"MLP parameter count: {num_params}")
    
    random_params = torch.randn(1, num_params, device=device) * 0.01
    
    with torch.no_grad():
        result = mlp_renderer(
            params=random_params,
            camera_poses=camera_pose,
            camera_intrinsics=camera_intrinsics,
            image_size=(args.image_size, args.image_size),
            bbox_min=bbox_min,
            bbox_max=bbox_max,
        )
    
    rendered_mlp = result['rgb']  # [1, 3, H, W]
    print(f"MLP rendered shape: {rendered_mlp.shape}")
    print(f"MLP rendered range: [{rendered_mlp.min():.3f}, {rendered_mlp.max():.3f}]")
    
    return {
        'voxel': rendered_voxel.cpu(),
        'mlp': rendered_mlp.cpu(),
        'dataset': sample['images'],
    }


def visualize_results(sample, rendered, args):
    """Visualize results."""
    print("\n" + "=" * 50)
    print("Visualizing Results")
    print("=" * 50)
    
    num_views = min(4, args.num_views)
    
    fig, axes = plt.subplots(3, num_views, figsize=(4 * num_views, 12))
    
    # Row 1: Dataset rendered images
    for i in range(num_views):
        img = sample['images'][i].permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        axes[0, i].imshow(img)
        axes[0, i].set_title(f"Dataset View {i}")
        axes[0, i].axis('off')
    
    axes[0, 0].set_ylabel("Dataset", fontsize=12)
    
    # Row 2: Voxel renderer output
    voxel_img = rendered['voxel'][0].permute(1, 2, 0).numpy()
    voxel_img = np.clip(voxel_img, 0, 1)
    axes[1, 0].imshow(voxel_img)
    axes[1, 0].set_title("Voxel Renderer")
    axes[1, 0].set_ylabel("Voxel", fontsize=12)
    for i in range(1, num_views):
        axes[1, i].axis('off')
    
    # Row 3: MLP renderer output (random params)
    mlp_img = rendered['mlp'][0].permute(1, 2, 0).numpy()
    mlp_img = np.clip(mlp_img, 0, 1)
    axes[2, 0].imshow(mlp_img)
    axes[2, 0].set_title("MLP Renderer (random)")
    axes[2, 0].set_ylabel("MLP", fontsize=12)
    for i in range(1, num_views):
        axes[2, i].axis('off')
    
    plt.tight_layout()
    
    save_path = os.path.join(os.path.dirname(__file__), "test_pipeline_output.png")
    plt.savefig(save_path, dpi=150)
    print(f"Saved visualization to: {save_path}")
    plt.close()


def test_gradient_flow(args):
    """Test that gradients flow through the pipeline."""
    print("\n" + "=" * 50)
    print("Testing Gradient Flow")
    print("=" * 50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    mlp_renderer = VolumeRenderer(num_samples=16).to(device)
    
    # Create parameters that require grad
    num_params = BatchedNeRFMLP.TOTAL_PARAMS
    params = torch.randn(1, num_params, device=device, requires_grad=True) * 0.01
    
    # Dummy camera
    camera_pose = torch.eye(4, device=device).unsqueeze(0)
    camera_pose[0, 2, 3] = -3.0  # Move camera back
    
    focal = args.image_size
    camera_intrinsics = torch.tensor([
        [focal, 0, args.image_size / 2],
        [0, focal, args.image_size / 2],
        [0, 0, 1]
    ], device=device, dtype=torch.float32).unsqueeze(0)
    
    # Render
    result = mlp_renderer(
        params=params,
        camera_poses=camera_pose,
        camera_intrinsics=camera_intrinsics,
        image_size=(args.image_size // 4, args.image_size // 4),
    )
    
    rgb = result['rgb']
    target = torch.ones_like(rgb)
    loss = ((rgb - target) ** 2).mean()
    
    print(f"Rendered shape: {rgb.shape}")
    print(f"Loss: {loss.item():.6f}")
    
    loss.backward()
    
    grad_norm = params.grad.norm().item()
    print(f"Gradient norm: {grad_norm:.6f}")
    
    if grad_norm > 0:
        print("✓ Gradients flow successfully!")
    else:
        print("✗ No gradients!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str,
                        default="d:/lecture/2.0_xk/CV/finalproject/NeRF-MAE_pretrain/NeRF-MAE/pretrain")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--num_views", type=int, default=4)
    args = parser.parse_args()
    
    print("NeRF-MAE Pipeline Test")
    print("=" * 50)
    print(f"Data dir: {args.data_dir}")
    print(f"Image size: {args.image_size}")
    print(f"Num views: {args.num_views}")
    
    # Test data loading
    sample = test_data_loading(args)
    
    if sample is not None:
        # Test rendering
        rendered = test_rendering(sample, args)
        
        # Visualize
        visualize_results(sample, rendered, args)
        
        # Test gradient flow
        test_gradient_flow(args)
    
    print("\n" + "=" * 50)
    print("Test Complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
