"""
Test Script for Evaluating VGGT Latent Head (Tri-plane) Rendering Quality.

This script evaluates the rendering quality of VGGT-predicted Tri-plane representation
by loading images, using VGGT to predict tri-planes, and performing volume rendering.

Pipeline:
1. Load multi-view images
2. Use VGGT to predict Tri-plane representation from the images
3. Render novel views using the predicted tri-planes with volume rendering
4. Compare with ground truth (GT) images
5. Compute metrics (PSNR, SSIM) and visualize results
6. Print sample point RGB/sigma values (GT vs predicted)

Usage:
    python test_latent_rendering.py --images path/to/images --output_dir ./results
    
    python test_latent_rendering.py --images "/media/fengwu/ZX1 1TB/code/cv_finalproject/data/tnerf/samples/3dfront_2006_01/images" 
    --output_dir ./results
    --model_path "/media/fengwu/ZX1 1TB/code/cv_finalproject/tnerf/checkpoints_tnerf/latent_checkpoint_best.pt"
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
sys.path.insert(0, os.path.dirname(os.path.dirname(SCRIPT_DIR)))  # finalproject/

# Import from tnerf's model package
from model.heads.latent_head import LatentHead

# Import from NeRF package with explicit path to avoid conflicts
from NeRF.rays import get_rays
from NeRF.positional_encoding import get_embedder
from NeRF.utils import to8b


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
        if len(img1.shape) == 3:
            return float(ssim(img1, img2, channel_axis=2, data_range=255))
        return float(ssim(img1, img2, data_range=255))
    except ImportError:
        img1_flat = img1.flatten().astype(np.float32)
        img2_flat = img2.flatten().astype(np.float32)
        corr = np.corrcoef(img1_flat, img2_flat)[0, 1]
        return max(0, corr)


# ============================================================================
# Volume Rendering with Tri-plane
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
    
    sigma_squeezed = sigma.squeeze(-1)  # [N_rays, N_samples]
    alpha = raw2alpha(sigma_squeezed, dists)
    
    weights = alpha * torch.cumprod(
        torch.cat([torch.ones((alpha.shape[0], 1), device=alpha.device), 1. - alpha + 1e-10], -1), -1
    )[:, :-1]
    
    rgb_map = torch.sum(weights[..., None] * rgb, -2)
    depth_map = torch.sum(weights * z_vals, -1)
    acc_map = torch.sum(weights, -1)
    
    if white_bkgd:
        rgb_map = rgb_map + (1. - acc_map[..., None])
    
    return rgb_map, depth_map, acc_map


def render_rays_with_triplane(
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    near: float,
    far: float,
    xy_plane: torch.Tensor,
    xz_plane: torch.Tensor,
    yz_plane: torch.Tensor,
    latent_head: LatentHead,
    embed_fn,
    embeddirs_fn,
    N_samples: int = 64,
    white_bkgd: bool = True,
    chunk: int = 1024 * 16,
    scene_bounds: Tuple[float, float] = (-1.0, 1.0),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Render rays using Tri-plane representation.
    
    Args:
        rays_o: [N_rays, 3] ray origins
        rays_d: [N_rays, 3] ray directions
        near: near plane distance
        far: far plane distance
        xy_plane: [1, 32, 64, 64] XY plane features
        xz_plane: [1, 32, 64, 64] XZ plane features
        yz_plane: [1, 32, 64, 64] YZ plane features
        latent_head: LatentHead model (for query_points)
        embed_fn: positional encoding for position
        embeddirs_fn: positional encoding for direction
        N_samples: number of samples per ray
        white_bkgd: use white background
        chunk: chunk size for memory efficiency
        scene_bounds: (min, max) scene bounds for normalizing to [-1, 1]
        
    Returns:
        rgb_map: [N_rays, 3] rendered RGB values
        depth_map: [N_rays] depth values
        all_sigmas: [N_rays, N_samples, 1] sigma values (for debugging)
    """
    device = rays_o.device
    N_rays = rays_o.shape[0]
    
    # Sample points along rays
    t_vals = torch.linspace(0., 1., steps=N_samples, device=device)
    z_vals = near * (1. - t_vals) + far * t_vals
    z_vals = z_vals.expand([N_rays, N_samples])
    
    # Get sample points: [N_rays, N_samples, 3]
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
    
    # Normalize points to [-1, 1] for tri-plane query
    bound_min, bound_max = scene_bounds
    pts_normalized = 2.0 * (pts - bound_min) / (bound_max - bound_min) - 1.0
    pts_normalized = pts_normalized.clamp(-1, 1)
    
    # Process in chunks
    all_rgb_maps = []
    all_depth_maps = []
    all_sigmas = []
    all_rgbs = []
    
    for i in range(0, N_rays, chunk):
        pts_chunk = pts[i:i+chunk]
        pts_norm_chunk = pts_normalized[i:i+chunk]
        rays_d_chunk = rays_d[i:i+chunk]
        z_vals_chunk = z_vals[i:i+chunk]
        
        chunk_n_rays = pts_chunk.shape[0]
        
        # Get viewing directions (normalized)
        viewdirs = rays_d_chunk / torch.norm(rays_d_chunk, dim=-1, keepdim=True)
        viewdirs = viewdirs[:, None].expand(pts_chunk.shape)  # [chunk, N_samples, 3]
        
        # Reshape for batch query: [1, chunk*N_samples, 3]
        pts_flat = pts_norm_chunk.reshape(1, -1, 3)
        viewdirs_flat = viewdirs.reshape(1, -1, 3)
        
        # Query tri-planes
        # latent_head.query_points expects: points [B, N, 3], directions [B, N, 3]
        sigma, rgb = latent_head.query_points(
            xy_plane, xz_plane, yz_plane,
            pts_flat, viewdirs_flat,
            pos_enc_fn=lambda x: embed_fn(x.reshape(-1, 3)).reshape(x.shape[0], x.shape[1], -1),
            dir_enc_fn=lambda x: embeddirs_fn(x.reshape(-1, 3)).reshape(x.shape[0], x.shape[1], -1),
        )
        
        # Reshape outputs: [1, chunk*N_samples, C] -> [chunk, N_samples, C]
        sigma = sigma.reshape(chunk_n_rays, N_samples, 1)
        rgb = rgb.reshape(chunk_n_rays, N_samples, 3)
        
        all_sigmas.append(sigma)
        all_rgbs.append(rgb)
        
        # Volume rendering
        rgb_map, depth_map, _ = raw2outputs(sigma, rgb, z_vals_chunk, rays_d_chunk, white_bkgd)
        all_rgb_maps.append(rgb_map)
        all_depth_maps.append(depth_map)
    
    rgb_map = torch.cat(all_rgb_maps, 0)
    depth_map = torch.cat(all_depth_maps, 0)
    sigmas = torch.cat(all_sigmas, 0)
    rgbs = torch.cat(all_rgbs, 0)
    
    return rgb_map, depth_map, sigmas, rgbs


def render_image(
    H: int,
    W: int,
    K: torch.Tensor,
    c2w: torch.Tensor,
    xy_plane: torch.Tensor,
    xz_plane: torch.Tensor,
    yz_plane: torch.Tensor,
    latent_head: LatentHead,
    embed_fn,
    embeddirs_fn,
    near: float = 0.1,
    far: float = 10.,
    N_samples: int = 64,
    chunk: int = 1024 * 16,
    white_bkgd: bool = True,
    scene_bounds: Tuple[float, float] = (-1.0, 1.0),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Render a full image from a camera pose using Tri-plane.
    
    Returns:
        rgb_map: [H, W, 3] rendered image
        depth_map: [H, W] depth map
    """
    rays_o, rays_d = get_rays(H, W, K, c2w)
    rays_o = rays_o.reshape(-1, 3)
    rays_d = rays_d.reshape(-1, 3)
    
    rgb_map, depth_map, _, _ = render_rays_with_triplane(
        rays_o, rays_d, near, far,
        xy_plane, xz_plane, yz_plane,
        latent_head, embed_fn, embeddirs_fn,
        N_samples=N_samples, white_bkgd=white_bkgd, chunk=chunk,
        scene_bounds=scene_bounds,
    )
    
    rgb_map = rgb_map.reshape(H, W, 3)
    depth_map = depth_map.reshape(H, W)
    return rgb_map, depth_map


# ============================================================================
# Camera Generation
# ============================================================================

def generate_orbit_cameras(
    center: np.ndarray = np.array([0., 0., 0.]),
    radius: float = 2.0,
    num_views: int = 8,
    elevation_deg: float = 30.
) -> List[np.ndarray]:
    """Generate orbit camera poses around a center point."""
    poses = []
    elevation_rad = np.deg2rad(elevation_deg)
    
    for i in range(num_views):
        theta = 2 * np.pi * i / num_views
        
        x = center[0] + radius * np.cos(theta) * np.cos(elevation_rad)
        y = center[1] + radius * np.sin(theta) * np.cos(elevation_rad)
        z = center[2] + radius * np.sin(elevation_rad)
        cam_pos = np.array([x, y, z])
        
        forward = center - cam_pos
        forward = forward / (np.linalg.norm(forward) + 1e-8)
        
        up = np.array([0., 0., 1.])
        right = np.cross(forward, up)
        right = right / (np.linalg.norm(right) + 1e-8)
        up = np.cross(right, forward)
        up = up / (np.linalg.norm(up) + 1e-8)
        
        c2w = np.eye(4)
        c2w[:3, 0] = right
        c2w[:3, 1] = up
        c2w[:3, 2] = -forward
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
    title: str = "GT vs Tri-plane Prediction"
):
    """Create a comparison figure showing GT and predicted images side by side."""
    n_views = len(gt_images)
    fig, axes = plt.subplots(2, n_views, figsize=(4*n_views, 8))
    
    if n_views == 1:
        axes = axes.reshape(2, 1)
    
    for i in range(n_views):
        axes[0, i].imshow(gt_images[i])
        axes[0, i].set_title(f"GT View {i}")
        axes[0, i].axis('off')
        
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
    title: str = "Tri-plane Rendered Views"
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
    
    for idx in range(n_images, rows * cols):
        row, col = idx // cols, idx % cols
        axes[row, col].axis('off')
    
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved rendering grid to {output_path}")


def create_triplane_visualization(
    xy_plane: torch.Tensor,
    xz_plane: torch.Tensor,
    yz_plane: torch.Tensor,
    output_path: str,
):
    """Visualize the tri-plane features."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Take mean across feature dimension for visualization
    xy_vis = xy_plane[0].mean(dim=0).cpu().numpy()
    xz_vis = xz_plane[0].mean(dim=0).cpu().numpy()
    yz_vis = yz_plane[0].mean(dim=0).cpu().numpy()
    
    # Normalize for visualization
    def normalize(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-8)
    
    axes[0].imshow(normalize(xy_vis), cmap='viridis')
    axes[0].set_title('XY Plane (mean features)')
    axes[0].axis('off')
    
    axes[1].imshow(normalize(xz_vis), cmap='viridis')
    axes[1].set_title('XZ Plane (mean features)')
    axes[1].axis('off')
    
    axes[2].imshow(normalize(yz_vis), cmap='viridis')
    axes[2].set_title('YZ Plane (mean features)')
    axes[2].axis('off')
    
    plt.suptitle('Tri-plane Feature Visualization', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved tri-plane visualization to {output_path}")


def print_sample_predictions(
    xy_plane: torch.Tensor,
    xz_plane: torch.Tensor,
    yz_plane: torch.Tensor,
    latent_head: LatentHead,
    embed_fn,
    embeddirs_fn,
    n_samples: int = 10,
    device: torch.device = None,
):
    """
    Print sample point predictions (RGB, sigma) for debugging.
    
    Samples random 3D points and queries their RGB/sigma values.
    """
    print("\n" + "="*60)
    print("Sample Point Predictions (Tri-plane Query)")
    print("="*60)
    
    # Generate random sample points in normalized [-1, 1] space
    torch.manual_seed(42)
    sample_points = torch.rand(1, n_samples, 3, device=device) * 2 - 1  # [-1, 1]
    sample_dirs = torch.randn(1, n_samples, 3, device=device)
    sample_dirs = sample_dirs / torch.norm(sample_dirs, dim=-1, keepdim=True)
    
    # Query tri-planes
    with torch.no_grad():
        sigma, rgb = latent_head.query_points(
            xy_plane, xz_plane, yz_plane,
            sample_points, sample_dirs,
            pos_enc_fn=lambda x: embed_fn(x.reshape(-1, 3)).reshape(x.shape[0], x.shape[1], -1),
            dir_enc_fn=lambda x: embeddirs_fn(x.reshape(-1, 3)).reshape(x.shape[0], x.shape[1], -1),
        )
    
    print(f"\n{'Idx':<5} {'Point (x,y,z)':<30} {'Dir (x,y,z)':<30} {'Sigma':<12} {'RGB (r,g,b)':<20}")
    print("-" * 100)
    
    for i in range(n_samples):
        pt = sample_points[0, i].cpu().numpy()
        d = sample_dirs[0, i].cpu().numpy()
        s = sigma[0, i, 0].item()
        c = rgb[0, i].cpu().numpy()
        
        pt_str = f"({pt[0]:+.3f}, {pt[1]:+.3f}, {pt[2]:+.3f})"
        dir_str = f"({d[0]:+.3f}, {d[1]:+.3f}, {d[2]:+.3f})"
        rgb_str = f"({c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f})"
        
        print(f"{i:<5} {pt_str:<30} {dir_str:<30} {s:<12.4f} {rgb_str:<20}")
    
    print("\nStatistics:")
    print(f"  Sigma - min: {sigma.min().item():.4f}, max: {sigma.max().item():.4f}, mean: {sigma.mean().item():.4f}")
    print(f"  RGB   - min: {rgb.min().item():.4f}, max: {rgb.max().item():.4f}, mean: {rgb.mean().item():.4f}")
    print("="*60 + "\n")


# ============================================================================
# Load VGGT Model
# ============================================================================

def load_vggt_model(model_path: str, device):
    """
    Load VGGT model for Latent (tri-plane) prediction.
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
        enable_nlp=False,
        enable_latent=True,  # Enable Latent head for tri-plane prediction
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

def test_latent_rendering(args):
    """
    Main test function: Load images, predict tri-planes, render views.
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
    print("\nLoading VGGT model with Latent head...")
    vggt_model = load_vggt_model(args.model_path, device)
    
    # Forward pass to get tri-plane predictions
    print("Running VGGT forward pass...")
    with torch.no_grad():
        predictions = vggt_model(images_tensor)
    
    xy_plane = predictions["xy_plane"]  # [B, 32, 64, 64]
    xz_plane = predictions["xz_plane"]
    yz_plane = predictions["yz_plane"]
    
    print(f"\nTri-plane shapes:")
    print(f"  XY plane: {xy_plane.shape}")
    print(f"  XZ plane: {xz_plane.shape}")
    print(f"  YZ plane: {yz_plane.shape}")
    
    # Get the latent head reference for query_points
    latent_head = vggt_model.latent_head
    
    # Get positional encoding functions
    embed_fn, pos_dim = get_embedder(args.multires, 0)
    embeddirs_fn, dir_dim = get_embedder(args.multires_views, 0)
    print(f"Position encoding dim: {pos_dim} (expected 63)")
    print(f"Direction encoding dim: {dir_dim} (expected 27)")
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_dir / f"latent_test_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Print sample point predictions
    print_sample_predictions(
        xy_plane, xz_plane, yz_plane,
        latent_head, embed_fn, embeddirs_fn,
        n_samples=args.n_print_samples,
        device=device,
    )
    
    # Visualize tri-planes
    create_triplane_visualization(
        xy_plane, xz_plane, yz_plane,
        str(output_dir / "triplane_features.png")
    )
    
    # Rendering parameters
    H, W = args.H, args.W
    K = generate_intrinsics(H, W, args.fov)
    K = torch.from_numpy(K).float().to(device)
    
    # Scene bounds
    center = np.array([0., 0., 0.])
    near, far = args.near, args.far
    radius = args.camera_radius
    scene_bounds = (args.scene_bound_min, args.scene_bound_max)
    
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
    depth_images = []
    
    for i, pose in enumerate(tqdm(render_poses)):
        c2w = torch.from_numpy(pose[:3, :4]).float().to(device)
        
        with torch.no_grad():
            rgb, depth = render_image(
                H, W, K, c2w,
                xy_plane, xz_plane, yz_plane,
                latent_head, embed_fn, embeddirs_fn,
                near=near, far=far, N_samples=args.n_samples,
                chunk=args.chunk, white_bkgd=args.white_bkgd,
                scene_bounds=scene_bounds,
            )
        
        img_np = (rgb.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        rendered_images.append(img_np)
        depth_images.append(depth.cpu().numpy())
        
        Image.fromarray(img_np).save(output_dir / f"rendered_{i:03d}.png")
        
        if i == 0:
            print(f"  RGB range: [{rgb.min().item():.3f}, {rgb.max().item():.3f}]")
            print(f"  RGB mean: {rgb.mean().item():.3f}")
            print(f"  Depth range: [{depth.min().item():.3f}, {depth.max().item():.3f}]")
    
    # Create visualization grid
    create_rendering_grid(
        rendered_images,
        str(output_dir / "rendering_grid.png"),
        title=f"Tri-plane Rendered Views ({len(input_paths)} input views)"
    )
    
    # Create depth visualization
    depth_vis = []
    for d in depth_images:
        d_norm = (d - d.min()) / (d.max() - d.min() + 1e-8)
        d_vis = plt.cm.viridis(d_norm)[:, :, :3]
        d_vis = (d_vis * 255).astype(np.uint8)
        depth_vis.append(d_vis)
    
    create_rendering_grid(
        depth_vis,
        str(output_dir / "depth_grid.png"),
        title="Tri-plane Depth Maps"
    )
    
    # Load GT images for comparison
    gt_images = []
    for path in input_paths[:min(4, len(input_paths))]:
        img = np.array(Image.open(path).resize((W, H)))
        if img.shape[-1] == 4:
            img = img[:, :, :3]
        gt_images.append(img)
    
    # Compare with rendered views
    n_compare = min(len(gt_images), len(rendered_images))
    metrics = []
    if n_compare > 0:
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
            title="Input Views vs Tri-plane Rendered Views"
        )
        
        avg_psnr = np.mean([m['psnr'] for m in metrics])
        avg_ssim = np.mean([m['ssim'] for m in metrics])
    else:
        avg_psnr, avg_ssim = 0, 0
    
    # Save results
    results = {
        'input_images': len(input_paths),
        'rendered_views': num_render_views,
        'triplane_shapes': {
            'xy': list(xy_plane.shape),
            'xz': list(xz_plane.shape),
            'yz': list(yz_plane.shape),
        },
        'image_size': [H, W],
        'near': near,
        'far': far,
        'scene_bounds': list(scene_bounds),
        'n_samples': args.n_samples,
        'avg_psnr': float(avg_psnr) if metrics else None,
        'avg_ssim': float(avg_ssim) if metrics else None,
        'per_view_metrics': metrics,
        'timestamp': datetime.now().isoformat(),
        'model_path': args.model_path,
    }
    
    with open(output_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results Summary:")
    print(f"  Input images: {len(input_paths)}")
    print(f"  Rendered views: {num_render_views}")
    print(f"  Tri-plane resolution: {xy_plane.shape[-1]}x{xy_plane.shape[-1]}")
    print(f"  Tri-plane feature dim: {xy_plane.shape[1]}")
    if metrics:
        print(f"  Average PSNR: {avg_psnr:.2f}")
        print(f"  Average SSIM: {avg_ssim:.4f}")
    print(f"  Output directory: {output_dir}")
    print(f"{'='*60}")
    
    return results


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Test VGGT Latent (Tri-plane) Rendering Quality')
    
    # Input/Output
    parser.add_argument('--images', type=str, required=True,
                        help='Directory containing input multi-view images')
    parser.add_argument('--output_dir', type=str, default='./latent_test_results',
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
    
    # Scene bounds for tri-plane normalization
    parser.add_argument('--scene_bound_min', type=float, default=-2.0,
                        help='Minimum scene bound for tri-plane normalization')
    parser.add_argument('--scene_bound_max', type=float, default=2.0,
                        help='Maximum scene bound for tri-plane normalization')
    
    # Camera parameters
    parser.add_argument('--num_render_views', type=int, default=8,
                        help='Number of views to render')
    parser.add_argument('--elevation', type=float, default=30.,
                        help='Camera elevation angle in degrees')
    parser.add_argument('--camera_radius', type=float, default=2.0,
                        help='Camera orbit radius')
    parser.add_argument('--max_views', type=int, default=8,
                        help='Maximum number of input views for VGGT')
    
    # Positional encoding
    parser.add_argument('--multires', type=int, default=10,
                        help='Position encoding frequencies')
    parser.add_argument('--multires_views', type=int, default=4,
                        help='Direction encoding frequencies')
    
    # Debug/Visualization
    parser.add_argument('--n_print_samples', type=int, default=10,
                        help='Number of sample points to print for debugging')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    
    args = parser.parse_args()
    
    # Run test
    test_latent_rendering(args)
    print("\nDone!")


if __name__ == '__main__':
    main()
