"""
Test Script for Evaluating VGGT MLP Rendering Quality.

This script evaluates the rendering quality of VGGT-predicted MLP parameters 
by loading images, using VGGT to predict NeRF MLP parameters, and performing 
volume rendering.

Pipeline:
1. Load multi-view images
2. Use VGGT to predict MLP parameters from the images
3. Render novel views using the predicted MLP with volume rendering
4. Compare with ground truth (GT) images
5. Compute metrics (PSNR, SSIM) and visualize results

Usage:
    # Test with custom multi-view images
    python test_mlp_rendering.py --images path/to/images --output_dir ./results
    
    # Test with specific camera parameters
    python test_mlp_rendering.py --images path/to/images --near 0.1 --far 5.0
    
    # Full example
    python test_mlp_rendering.py --images ../data/test_scene/images --output_dir ./test_results
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from datetime import datetime
from PIL import Image

# Add parent directories to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))  # NeRF/
sys.path.insert(0, os.path.dirname(os.path.dirname(SCRIPT_DIR)))  # finalproject/

from model.heads.nlp_head import NeRFMLP
from rays import get_rays
from positional_encoding import get_embedder
from utils import to8b


# ============================================================================
# Metrics
# ============================================================================

def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute PSNR between two images."""
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute SSIM between two images (simplified version)."""
    try:
        from skimage.metrics import structural_similarity as ssim
        # Convert to grayscale for SSIM if color
        if len(img1.shape) == 3:
            return float(ssim(img1, img2, channel_axis=2, data_range=255))
        return float(ssim(img1, img2, data_range=255))
    except ImportError:
        # Fallback: simple correlation-based similarity
        img1_flat = img1.flatten().astype(np.float32)
        img2_flat = img2.flatten().astype(np.float32)
        corr = np.corrcoef(img1_flat, img2_flat)[0, 1]
        return max(0, corr)


# ============================================================================
# Volume Rendering with NeRFMLP
# ============================================================================

def raw2outputs(sigma, rgb, z_vals, rays_d, white_bkgd=False):
    """
    Transform model predictions (sigma, rgb) to final RGB and depth.
    
    Args:
        sigma: [N_rays, N_samples, 1] density predictions
        rgb: [N_rays, N_samples, 3] color predictions
        z_vals: [N_rays, N_samples] sample positions along rays
        rays_d: [N_rays, 3] ray directions
        white_bkgd: whether to use white background
        
    Returns:
        rgb_map: [N_rays, 3] rendered color
        depth_map: [N_rays] rendered depth
        acc_map: [N_rays] accumulated opacity
    """
    raw2alpha = lambda sigma_val, dists: 1. - torch.exp(-F.relu(sigma_val) * dists)
    
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.full_like(dists[..., :1], 1e10)], -1)
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)
    
    # sigma is [N_rays, N_samples, 1], squeeze last dim
    sigma_squeezed = sigma.squeeze(-1)  # [N_rays, N_samples]
    
    alpha = raw2alpha(sigma_squeezed, dists)  # [N_rays, N_samples]
    
    # Compute transmittance and weights
    weights = alpha * torch.cumprod(
        torch.cat([torch.ones((alpha.shape[0], 1), device=alpha.device), 1. - alpha + 1e-10], -1), -1
    )[:, :-1]  # [N_rays, N_samples]
    
    # rgb is already [N_rays, N_samples, 3]
    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]
    depth_map = torch.sum(weights * z_vals, -1)  # [N_rays]
    acc_map = torch.sum(weights, -1)  # [N_rays]
    
    if white_bkgd:
        rgb_map = rgb_map + (1. - acc_map[..., None])
    
    return rgb_map, depth_map, acc_map


def render_rays_with_nerf_mlp(
    rays_o: torch.Tensor, 
    rays_d: torch.Tensor, 
    near: float, 
    far: float, 
    nerf_mlp: NeRFMLP, 
    embed_fn, 
    embeddirs_fn,
    N_samples: int = 64,
    white_bkgd: bool = True,
    chunk: int = 1024 * 16
) -> torch.Tensor:
    """
    Render rays using NeRFMLP model.
    
    Args:
        rays_o: [N_rays, 3] ray origins
        rays_d: [N_rays, 3] ray directions
        near: near plane distance
        far: far plane distance
        nerf_mlp: NeRFMLP model with loaded parameters
        embed_fn: positional encoding for position (output dim=63)
        embeddirs_fn: positional encoding for direction (output dim=27)
        N_samples: number of samples per ray
        white_bkgd: use white background
        chunk: chunk size for memory efficiency
        
    Returns:
        rgb_map: [N_rays, 3] rendered RGB values
    """
    device = rays_o.device
    N_rays = rays_o.shape[0]
    
    # Sample points along rays
    t_vals = torch.linspace(0., 1., steps=N_samples, device=device)
    z_vals = near * (1. - t_vals) + far * t_vals
    z_vals = z_vals.expand([N_rays, N_samples])
    
    # Get sample points: [N_rays, N_samples, 3]
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
    
    # Process in chunks to avoid OOM
    all_rgb_maps = []
    
    for i in range(0, N_rays, chunk):
        pts_chunk = pts[i:i+chunk]
        rays_d_chunk = rays_d[i:i+chunk]
        z_vals_chunk = z_vals[i:i+chunk]
        
        chunk_n_rays = pts_chunk.shape[0]
        
        # Flatten points for positional encoding
        pts_flat = pts_chunk.reshape(-1, 3)  # [chunk*N_samples, 3]
        
        # Apply positional encoding to positions
        pos_encoded = embed_fn(pts_flat)  # [chunk*N_samples, 63]
        
        # Get viewing directions (normalized)
        viewdirs = rays_d_chunk / torch.norm(rays_d_chunk, dim=-1, keepdim=True)
        viewdirs = viewdirs[:, None].expand(pts_chunk.shape)  # [chunk, N_samples, 3]
        viewdirs_flat = viewdirs.reshape(-1, 3)  # [chunk*N_samples, 3]
        
        # Apply positional encoding to directions
        dir_encoded = embeddirs_fn(viewdirs_flat)  # [chunk*N_samples, 27]
        
        # Run through NeRF MLP
        sigma, rgb = nerf_mlp(pos_encoded, dir_encoded)
        
        # Reshape outputs
        sigma = sigma.reshape(chunk_n_rays, N_samples, 1)  # [chunk, N_samples, 1]
        rgb = rgb.reshape(chunk_n_rays, N_samples, 3)  # [chunk, N_samples, 3]
        
        # Volume rendering
        rgb_map, _, _ = raw2outputs(sigma, rgb, z_vals_chunk, rays_d_chunk, white_bkgd)
        all_rgb_maps.append(rgb_map)
    
    rgb_map = torch.cat(all_rgb_maps, 0)
    return rgb_map


def render_image(
    H: int, 
    W: int, 
    K: torch.Tensor, 
    c2w: torch.Tensor, 
    nerf_mlp: NeRFMLP, 
    embed_fn, 
    embeddirs_fn,
    near: float = 0.1, 
    far: float = 10., 
    N_samples: int = 64, 
    chunk: int = 1024 * 16,
    white_bkgd: bool = True
) -> torch.Tensor:
    """
    Render a full image from a camera pose using NeRFMLP.
    
    Args:
        H: image height
        W: image width
        K: camera intrinsic matrix
        c2w: camera-to-world transform
        nerf_mlp: NeRFMLP model
        embed_fn: position encoding function
        embeddirs_fn: direction encoding function
        near: near plane
        far: far plane
        N_samples: samples per ray
        chunk: chunk size
        white_bkgd: use white background
        
    Returns:
        rgb_map: [H, W, 3] rendered image
    """
    rays_o, rays_d = get_rays(H, W, K, c2w)
    rays_o = rays_o.reshape(-1, 3)
    rays_d = rays_d.reshape(-1, 3)
    
    rgb_map = render_rays_with_nerf_mlp(
        rays_o, rays_d, near, far, nerf_mlp, embed_fn, embeddirs_fn,
        N_samples=N_samples, white_bkgd=white_bkgd, chunk=chunk
    )
    
    rgb_map = rgb_map.reshape(H, W, 3)
    return rgb_map


# ============================================================================
# Camera Generation
# ============================================================================

def generate_orbit_cameras(
    center: np.ndarray = np.array([0., 0., 0.]),
    radius: float = 2.0,
    num_views: int = 8,
    elevation_deg: float = 30.
) -> List[np.ndarray]:
    """
    Generate orbit camera poses around a center point.
    
    Args:
        center: scene center
        radius: camera distance from center
        num_views: number of views to generate
        elevation_deg: elevation angle in degrees
        
    Returns:
        List of 4x4 camera-to-world matrices
    """
    poses = []
    elevation_rad = np.deg2rad(elevation_deg)
    
    for i in range(num_views):
        theta = 2 * np.pi * i / num_views
        
        # Camera position on orbit
        x = center[0] + radius * np.cos(theta) * np.cos(elevation_rad)
        y = center[1] + radius * np.sin(theta) * np.cos(elevation_rad)
        z = center[2] + radius * np.sin(elevation_rad)
        cam_pos = np.array([x, y, z])
        
        # Look at center
        forward = center - cam_pos
        forward = forward / (np.linalg.norm(forward) + 1e-8)
        
        # Up vector
        up = np.array([0., 0., 1.])
        right = np.cross(forward, up)
        right = right / (np.linalg.norm(right) + 1e-8)
        up = np.cross(right, forward)
        up = up / (np.linalg.norm(up) + 1e-8)
        
        # Build c2w matrix
        c2w = np.eye(4)
        c2w[:3, 0] = right
        c2w[:3, 1] = up
        c2w[:3, 2] = -forward  # OpenGL convention: -Z is forward
        c2w[:3, 3] = cam_pos
        
        poses.append(c2w)
    
    return poses


def generate_intrinsics(H: int, W: int, fov_deg: float = 60.) -> np.ndarray:
    """Generate camera intrinsic matrix from field of view."""
    fov_rad = np.deg2rad(fov_deg)
    focal = H / (2 * np.tan(fov_rad / 2))
    K = np.array([
        [focal, 0, W / 2],
        [0, focal, H / 2],
        [0, 0, 1]
    ])
    return K


# ============================================================================
# Visualization
# ============================================================================

def create_comparison_figure(
    gt_images: List[np.ndarray],
    pred_images: List[np.ndarray],
    metrics: List[Dict],
    output_path: str,
    title: str = "GT vs MLP Prediction"
):
    """
    Create a comparison figure showing GT and predicted images side by side.
    """
    n_views = len(gt_images)
    fig, axes = plt.subplots(2, n_views, figsize=(4*n_views, 8))
    
    if n_views == 1:
        axes = axes.reshape(2, 1)
    
    for i in range(n_views):
        # GT image
        axes[0, i].imshow(gt_images[i])
        axes[0, i].set_title(f"GT View {i}")
        axes[0, i].axis('off')
        
        # Predicted image
        axes[1, i].imshow(pred_images[i])
        psnr = metrics[i].get('psnr', 0)
        ssim = metrics[i].get('ssim', 0)
        axes[1, i].set_title(f"Pred View {i}\nPSNR: {psnr:.2f} | SSIM: {ssim:.3f}")
        axes[1, i].axis('off')
    
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison figure to {output_path}")


def create_rendering_grid(
    images: List[np.ndarray],
    output_path: str,
    title: str = "MLP Rendered Views"
):
    """Create a grid of rendered images."""
    n_images = len(images)
    cols = min(4, n_images)
    rows = (n_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    axes = np.atleast_2d(axes)
    
    for idx, img in enumerate(images):
        row, col = idx // cols, idx % cols
        ax = axes[row, col]
        ax.imshow(img)
        ax.set_title(f"View {idx}")
        ax.axis('off')
    
    # Hide empty subplots
    for idx in range(n_images, rows * cols):
        row, col = idx // cols, idx % cols
        axes[row, col].axis('off')
    
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved rendering grid to {output_path}")


# ============================================================================
# Load VGGT Model
# ============================================================================

def load_vggt_model(model_path: str, device):
    """
    Load VGGT model for MLP prediction.
    
    Args:
        model_path: Path to VGGT checkpoint
        device: Device to use
        
    Returns:
        VGGT model instance
    """
    from model.models.vggt import VGGT as TNeRFVGGT
    
    model = TNeRFVGGT(
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        enable_camera=True,
        enable_point=True,
        enable_depth=False,
        enable_track=False,
        enable_nlp=True,  # Enable NLP head for MLP prediction
        enable_latent=False
    )
    
    if model_path and os.path.exists(model_path):
        print(f"Loading VGGT weights from {model_path}")
        state_dict = torch.load(model_path, map_location="cpu")
        if "model" in state_dict:
            state_dict = state_dict["model"]
        elif "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict, strict=False)
    else:
        print(f"Warning: Model path not found: {model_path}")
        print("Using randomly initialized model (results will be random)")
    
    model = model.to(device)
    model.eval()
    return model


# ============================================================================
# Main Test Function
# ============================================================================

def test_mlp_rendering(args):
    """
    Main test function: Load images, predict MLP, render views.
    """
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load images
    images_dir = Path(args.images)
    image_paths = sorted(
        [str(p) for p in images_dir.glob("*.png")] + 
        [str(p) for p in images_dir.glob("*.jpg")] +
        [str(p) for p in images_dir.glob("*.jpeg")]
    )
    
    if len(image_paths) < 2:
        raise ValueError(f"Need at least 2 images, found {len(image_paths)} in {images_dir}")
    
    print(f"Found {len(image_paths)} images in {images_dir}")
    
    # Limit number of images for VGGT input
    max_input_views = min(args.max_views, len(image_paths))
    input_paths = image_paths[:max_input_views]
    
    # Load and preprocess images for VGGT
    from model.utils.load_fn import load_and_preprocess_images
    images_tensor = load_and_preprocess_images(input_paths).to(device)
    print(f"Input tensor shape: {images_tensor.shape}")  # [S, 3, H, W]
    
    # Load VGGT model
    print("\nLoading VGGT model...")
    vggt_model = load_vggt_model(args.model_path, device)
    
    # Forward pass to get MLP parameters
    print("Running VGGT forward pass...")
    with torch.no_grad():
        predictions = vggt_model(images_tensor)
    
    if "nmlp" not in predictions:
        raise RuntimeError("VGGT model did not produce MLP parameters. Make sure enable_nlp=True")
    
    mlp_params = predictions["nmlp"]  # [B, num_params]
    print(f"Predicted MLP params shape: {mlp_params.shape}")
    print(f"Expected params: {NeRFMLP.TOTAL_PARAMS}")
    
    # Create NeRFMLP and load predicted parameters
    nerf_mlp = NeRFMLP().to(device)
    nerf_mlp.load_from_params(mlp_params)
    nerf_mlp.eval()
    print("NeRFMLP loaded with predicted parameters")
    
    # Get positional encoding functions
    # NeRFMLP expects: pos_encoded [63], dir_encoded [27]
    # pos: 3 + 3*2*10 = 63 (multires=10)
    # dir: 3 + 3*2*4 = 27 (multires_views=4)
    embed_fn, pos_dim = get_embedder(args.multires, 0)
    embeddirs_fn, dir_dim = get_embedder(args.multires_views, 0)
    print(f"Position encoding dim: {pos_dim} (expected 63)")
    print(f"Direction encoding dim: {dir_dim} (expected 27)")
    
    assert pos_dim == 63, f"Position encoding dim mismatch: {pos_dim} vs 63"
    assert dir_dim == 27, f"Direction encoding dim mismatch: {dir_dim} vs 27"
    
    # Get camera parameters from VGGT predictions or use defaults
    if "extrinsic" in predictions and args.use_predicted_cameras:
        from model.utils.pose_enc import pose_encoding_to_extri_intri
        extrinsic, intrinsic = pose_encoding_to_extri_intri(
            predictions["pose_enc"], images_tensor.shape[-2:]
        )
        print("Using VGGT predicted camera parameters")
    else:
        print("Using generated orbit camera parameters")
        extrinsic = None
        intrinsic = None
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_dir / f"mlp_test_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Rendering parameters
    H, W = args.H, args.W
    K = generate_intrinsics(H, W, args.fov)
    K = torch.from_numpy(K).float().to(device)
    
    # Get scene bounds from world points if available
    if "world_points" in predictions and args.use_predicted_bounds:
        world_pts = predictions["world_points"]  # [B, S, H, W, 3]
        pts_flat = world_pts.reshape(-1, 3)
        valid_mask = ~(torch.isnan(pts_flat).any(dim=-1) | torch.isinf(pts_flat).any(dim=-1))
        pts_valid = pts_flat[valid_mask]
        if pts_valid.shape[0] > 0:
            center = pts_valid.mean(dim=0).cpu().numpy()
            radius = pts_valid.std().item() * 3
            near = max(0.1, radius * 0.1)
            far = radius * 3
            print(f"Scene bounds from VGGT: center={center}, radius={radius:.2f}, near={near:.2f}, far={far:.2f}")
        else:
            center = np.array([0., 0., 0.])
            near, far = args.near, args.far
            radius = 2.0
    else:
        center = np.array([0., 0., 0.])
        near, far = args.near, args.far
        radius = 2.0
    
    # Generate camera poses for rendering
    num_render_views = args.num_render_views
    render_poses = generate_orbit_cameras(
        center=center,
        radius=radius,
        num_views=num_render_views,
        elevation_deg=args.elevation
    )
    
    # Render views
    print(f"\nRendering {num_render_views} views...")
    rendered_images = []
    
    for i, pose in enumerate(tqdm(render_poses)):
        c2w = torch.from_numpy(pose[:3, :4]).float().to(device)
        
        with torch.no_grad():
            rgb = render_image(
                H, W, K, c2w, nerf_mlp, embed_fn, embeddirs_fn,
                near=near, far=far, N_samples=args.n_samples,
                chunk=args.chunk, white_bkgd=args.white_bkgd
            )
        
        # Convert to uint8 image
        img_np = (rgb.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        rendered_images.append(img_np)
        
        # Save individual image
        Image.fromarray(img_np).save(output_dir / f"rendered_{i:03d}.png")
        
        # Print stats for first image
        if i == 0:
            print(f"  RGB range: [{rgb.min().item():.3f}, {rgb.max().item():.3f}]")
            print(f"  RGB mean: {rgb.mean().item():.3f}")
    
    # Create visualization grid
    create_rendering_grid(
        rendered_images,
        str(output_dir / "rendering_grid.png"),
        title=f"VGGT MLP Rendered Views ({len(input_paths)} input views)"
    )
    
    # Load GT images for comparison (if available)
    gt_images = []
    for path in input_paths[:min(4, len(input_paths))]:
        img = np.array(Image.open(path).resize((W, H)))
        if img.shape[-1] == 4:  # RGBA
            img = img[:, :, :3]
        gt_images.append(img)
    
    # Compare with some rendered views (first few)
    n_compare = min(len(gt_images), len(rendered_images))
    if n_compare > 0:
        metrics = []
        for i in range(n_compare):
            psnr = compute_psnr(gt_images[i], rendered_images[i])
            ssim = compute_ssim(gt_images[i], rendered_images[i])
            metrics.append({'view_idx': i, 'psnr': psnr, 'ssim': ssim})
            print(f"View {i}: PSNR={psnr:.2f}, SSIM={ssim:.4f}")
        
        create_comparison_figure(
            gt_images[:n_compare],
            rendered_images[:n_compare],
            metrics,
            str(output_dir / "comparison.png"),
            title="Input Views vs MLP Rendered Views"
        )
        
        avg_psnr = np.mean([m['psnr'] for m in metrics])
        avg_ssim = np.mean([m['ssim'] for m in metrics])
    else:
        avg_psnr, avg_ssim = 0, 0
        metrics = []
    
    # Save results
    results = {
        'input_images': len(input_paths),
        'rendered_views': num_render_views,
        'mlp_params_shape': list(mlp_params.shape),
        'image_size': [H, W],
        'near': near,
        'far': far,
        'n_samples': args.n_samples,
        'avg_psnr': float(avg_psnr) if metrics else None,
        'avg_ssim': float(avg_ssim) if metrics else None,
        'per_view_metrics': metrics,
        'timestamp': datetime.now().isoformat(),
        'model_path': args.model_path,
    }
    
    with open(output_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*50}")
    print(f"Results Summary:")
    print(f"  Input images: {len(input_paths)}")
    print(f"  Rendered views: {num_render_views}")
    print(f"  MLP params: {mlp_params.numel()}")
    if metrics:
        print(f"  Average PSNR: {avg_psnr:.2f}")
        print(f"  Average SSIM: {avg_ssim:.4f}")
    print(f"  Output directory: {output_dir}")
    print(f"{'='*50}")
    
    return results


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Test VGGT MLP Rendering Quality')
    
    # Input/Output
    parser.add_argument('--images', type=str, required=True,
                        help='Directory containing input multi-view images')
    parser.add_argument('--output_dir', type=str, default='./mlp_test_results',
                        help='Output directory for results')
    parser.add_argument('--model_path', type=str, 
                        default='../../vggt/model_weights/model.pt',
                        help='Path to VGGT model checkpoint')
    
    # Rendering parameters
    parser.add_argument('--H', type=int, default=256, help='Rendered image height')
    parser.add_argument('--W', type=int, default=256, help='Rendered image width')
    parser.add_argument('--fov', type=float, default=60., help='Field of view in degrees')
    parser.add_argument('--near', type=float, default=0.1, help='Near plane distance')
    parser.add_argument('--far', type=float, default=10., help='Far plane distance')
    parser.add_argument('--n_samples', type=int, default=64, help='Samples per ray')
    parser.add_argument('--chunk', type=int, default=1024*16, help='Chunk size for rendering')
    parser.add_argument('--white_bkgd', action='store_true', default=True,
                        help='Use white background')
    parser.add_argument('--no_white_bkgd', dest='white_bkgd', action='store_false',
                        help='Use black background')
    
    # Camera parameters
    parser.add_argument('--num_render_views', type=int, default=8,
                        help='Number of views to render')
    parser.add_argument('--elevation', type=float, default=30.,
                        help='Camera elevation angle in degrees')
    parser.add_argument('--max_views', type=int, default=8,
                        help='Maximum number of input views for VGGT')
    parser.add_argument('--use_predicted_cameras', action='store_true',
                        help='Use VGGT predicted camera parameters')
    parser.add_argument('--use_predicted_bounds', action='store_true',
                        help='Use VGGT predicted scene bounds')
    
    # Positional encoding (must match NeRFMLP expectations)
    parser.add_argument('--multires', type=int, default=10,
                        help='Position encoding frequencies (output dim = 3 + 3*2*L)')
    parser.add_argument('--multires_views', type=int, default=4,
                        help='Direction encoding frequencies')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    
    args = parser.parse_args()
    
    # Run test
    test_mlp_rendering(args)
    print("\nDone!")


if __name__ == '__main__':
    main()
