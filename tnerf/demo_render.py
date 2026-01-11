"""
T-NeRF Demo: Render video using MLP parameters predicted by VGGT NLP head.

支持两种模式：
1. 静态模式 (static): 传入一组多视角图像，直接预测MLP参数
2. 动态模式 (dynamic): 传入两组多视角图像，通过插值预测中间时刻的MLP参数

用法:
    静态模式: python demo_render.py --mode static --images path/to/images --model_path model.pth
    动态模式: python demo_render.py --mode dynamic --images_a path/to/t0 --images_b path/to/t1 --alpha 0.5
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import imageio
from tqdm import tqdm
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.dynamic_tnerf import create_dynamic_vggt
from network import NeRF
from positional_encoding import get_embedder
from rays import get_rays
from utils import to8b


def raw2outputs(raw, z_vals, rays_d, white_bkgd=False):
    """Transform model predictions to RGB and depth."""
    raw2alpha = lambda raw, dists: 1. - torch.exp(-F.relu(raw) * dists)
    
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).to(dists.device).expand(dists[..., :1].shape)], -1)
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)
    
    rgb = torch.sigmoid(raw[..., :3])
    alpha = raw2alpha(raw[..., 3], dists)
    weights = alpha * torch.cumprod(
        torch.cat([torch.ones((alpha.shape[0], 1), device=alpha.device), 1. - alpha + 1e-10], -1), -1
    )[:, :-1]
    
    rgb_map = torch.sum(weights[..., None] * rgb, -2)
    depth_map = torch.sum(weights * z_vals, -1)
    acc_map = torch.sum(weights, -1)
    
    if white_bkgd:
        rgb_map = rgb_map + (1. - acc_map[..., None])
    
    return rgb_map, depth_map, acc_map


def render_rays_with_mlp(rays_o, rays_d, near, far, nerf_model, embed_fn, embeddirs_fn, 
                          N_samples=64, use_viewdirs=True, white_bkgd=False, chunk=1024*32):
    """Render rays using NeRF model."""
    N_rays = rays_o.shape[0]
    
    # Sample points along rays
    t_vals = torch.linspace(0., 1., steps=N_samples, device=rays_o.device)
    z_vals = near * (1. - t_vals) + far * t_vals
    z_vals = z_vals.expand([N_rays, N_samples])
    
    # Get sample points
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples, 3]
    
    # Run network in chunks to avoid OOM
    all_rgb_maps = []
    all_depth_maps = []
    all_acc_maps = []
    
    for i in range(0, N_rays, chunk):
        pts_chunk = pts[i:i+chunk]
        rays_d_chunk = rays_d[i:i+chunk]
        z_vals_chunk = z_vals[i:i+chunk]
        
        # Flatten for embedding
        pts_flat = pts_chunk.reshape(-1, 3)
        embedded = embed_fn(pts_flat)
        
        if use_viewdirs:
            viewdirs = rays_d_chunk / torch.norm(rays_d_chunk, dim=-1, keepdim=True)
            viewdirs = viewdirs[:, None].expand(pts_chunk.shape)
            viewdirs_flat = viewdirs.reshape(-1, 3)
            embedded_dirs = embeddirs_fn(viewdirs_flat)
            embedded = torch.cat([embedded, embedded_dirs], -1)
        
        # Forward pass through NeRF
        raw_flat = nerf_model(embedded)
        raw = raw_flat.reshape(list(pts_chunk.shape[:-1]) + [raw_flat.shape[-1]])
        
        # Volume rendering
        rgb_map, depth_map, acc_map = raw2outputs(raw, z_vals_chunk, rays_d_chunk, white_bkgd)
        
        all_rgb_maps.append(rgb_map)
        all_depth_maps.append(depth_map)
        all_acc_maps.append(acc_map)
    
    rgb_map = torch.cat(all_rgb_maps, 0)
    depth_map = torch.cat(all_depth_maps, 0)
    acc_map = torch.cat(all_acc_maps, 0)
    
    return rgb_map, depth_map, acc_map


def render_image(H, W, K, c2w, nerf_model, embed_fn, embeddirs_fn, 
                 near=2., far=6., N_samples=64, chunk=1024*32, 
                 use_viewdirs=True, white_bkgd=False):
    """Render a full image from a camera pose."""
    rays_o, rays_d = get_rays(H, W, K, c2w)
    rays_o = rays_o.reshape(-1, 3)
    rays_d = rays_d.reshape(-1, 3)
    
    rgb_map, depth_map, acc_map = render_rays_with_mlp(
        rays_o, rays_d, near, far, nerf_model, embed_fn, embeddirs_fn,
        N_samples=N_samples, use_viewdirs=use_viewdirs, white_bkgd=white_bkgd, chunk=chunk
    )
    
    rgb_map = rgb_map.reshape(H, W, 3)
    depth_map = depth_map.reshape(H, W)
    acc_map = acc_map.reshape(H, W)
    
    return rgb_map, depth_map, acc_map


def generate_spiral_poses(n_poses=120, radius=4., height_range=(-0.5, 0.5), n_rotations=2):
    """Generate spiral camera trajectory."""
    poses = []
    for i in range(n_poses):
        theta = 2 * np.pi * n_rotations * i / n_poses
        height = height_range[0] + (height_range[1] - height_range[0]) * i / n_poses
        
        # Camera position
        cam_pos = np.array([
            radius * np.cos(theta),
            radius * np.sin(theta),
            height
        ])
        
        # Look at origin
        look_at = np.array([0., 0., 0.])
        forward = look_at - cam_pos
        forward = forward / np.linalg.norm(forward)
        
        # Up vector
        up = np.array([0., 0., 1.])
        right = np.cross(up, forward)
        right = right / np.linalg.norm(right)
        up = np.cross(forward, right)
        
        # Build camera-to-world matrix
        c2w = np.eye(4)
        c2w[:3, 0] = right
        c2w[:3, 1] = up
        c2w[:3, 2] = forward
        c2w[:3, 3] = cam_pos
        
        poses.append(c2w)
    
    return np.stack(poses, 0)


def apply_mlp_parameters_to_model(nerf_model, mlp_params):
    """Apply predicted MLP parameters to NeRF model.
    
    Note: This is a simplified version. The actual parameter mapping
    depends on how the NLP head outputs are structured.
    """
    # TODO: Implement proper parameter mapping from NLP head output to NeRF weights
    # This is a placeholder - you need to define how NLP head outputs map to NeRF layers
    pass


def main():
    parser = argparse.ArgumentParser(description='Render video using T-NeRF with VGGT')
    
    # 模式选择
    parser.add_argument('--mode', type=str, default='static', choices=['static', 'dynamic'],
                       help='Rendering mode: static (one image set, constant MLP) or dynamic (two image sets, MLP interpolates over time)')
    
    # 模型路径
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to VGGT model checkpoint')
    
    # 静态模式：单组图像
    parser.add_argument('--images', type=str, default='',
                       help='[Static mode] Directory with multi-view images')
    
    # 动态模式：两组图像
    parser.add_argument('--images_a', type=str, default='',
                       help='[Dynamic mode] Directory with first time step multi-view images')
    parser.add_argument('--images_b', type=str, default='',
                       help='[Dynamic mode] Directory with second time step multi-view images')
    
    # 输出设置
    parser.add_argument('--output', type=str, default='output_video.mp4',
                       help='Output video path')
    
    # 渲染参数
    parser.add_argument('--H', type=int, default=400,
                       help='Image height')
    parser.add_argument('--W', type=int, default=400,
                       help='Image width')
    parser.add_argument('--focal', type=float, default=555.0,
                       help='Focal length')
    parser.add_argument('--n_poses', type=int, default=120,
                       help='Number of poses in spiral trajectory')
    parser.add_argument('--n_samples', type=int, default=64,
                       help='Number of samples per ray')
    parser.add_argument('--chunk', type=int, default=1024*8,
                       help='Chunk size for rendering')
    parser.add_argument('--white_bkgd', action='store_true',
                       help='Use white background')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--multires', type=int, default=10,
                       help='Positional encoding frequency for positions')
    parser.add_argument('--multires_views', type=int, default=4,
                       help='Positional encoding frequency for views')
    
    args = parser.parse_args()
    
    # 验证参数
    if args.mode == 'static':
        if not args.images:
            parser.error("Static mode requires --images argument")
        print(f"\n{'='*50}")
        print("Running in STATIC mode")
        print("- Single time step")
        print("- MLP parameters predicted once")
        print("- Same NeRF used for all frames")
        print(f"Images: {args.images}")
        print(f"{'='*50}\n")
    else:  # dynamic
        if not args.images_a or not args.images_b:
            parser.error("Dynamic mode requires --images_a and --images_b arguments")
        print(f"\n{'='*50}")
        print("Running in DYNAMIC mode")
        print("- Two time steps interpolated")
        print("- MLP parameters predicted per frame")
        print("- Alpha varies from 0→1 across video")
        print(f"Images A (t=0): {args.images_a}")
        print(f"Images B (t=1): {args.images_b}")
        print(f"{'='*50}\n")
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Step 1: 准备模型和数据
    print("\n=== Step 1: Loading VGGT model ===")
    print("Loading VGGT model...")
    dynamic_model = create_dynamic_vggt(
        config="nerf",
        device=str(device),
        model_path=args.model_path
    )
    
    # 准备图像路径
    if args.mode == 'static':
        image_paths = sorted([
            os.path.join(args.images, f) 
            for f in os.listdir(args.images) 
            if f.endswith(('.png', '.jpg', '.jpeg'))
        ])
        print(f"Found {len(image_paths)} images in {args.images}")
    else:  # dynamic
        image_paths_a = sorted([
            os.path.join(args.images_a, f) 
            for f in os.listdir(args.images_a) 
            if f.endswith(('.png', '.jpg', '.jpeg'))
        ])
        image_paths_b = sorted([
            os.path.join(args.images_b, f) 
            for f in os.listdir(args.images_b) 
            if f.endswith(('.png', '.jpg', '.jpeg'))
        ])
        print(f"Found {len(image_paths_a)} images in {args.images_a}")
        print(f"Found {len(image_paths_b)} images in {args.images_b}")
    
    # Step 2: Create NeRF model with positional encoding
    print("\n=== Step 2: Creating NeRF model ===")
    embed_fn, input_ch = get_embedder(args.multires, 0)
    embeddirs_fn, input_ch_views = get_embedder(args.multires_views, 0)
    
    nerf_model = NeRF(
        D=8, W=256,
        input_ch=input_ch,
        input_ch_views=input_ch_views,
        output_ch=4,
        skips=[4],
        use_viewdirs=True
    ).to(device)
    
    # Step 3: Generate camera trajectory
    print("\n=== Step 3: Generating camera trajectory ===")
    render_poses = generate_spiral_poses(
        n_poses=args.n_poses,
        radius=4.,
        height_range=(-0.5, 0.5),
        n_rotations=2
    )
    
    # Camera intrinsics
    K = np.array([
        [args.focal, 0, 0.5 * args.W],
        [0, args.focal, 0.5 * args.H],
        [0, 0, 1]
    ])
    K_tensor = torch.from_numpy(K).float().to(device)
    
    # Step 4: Render video
    print("\n=== Step 4: Rendering video ===")
    frames = []
    
    with torch.no_grad():
        for i, pose in enumerate(tqdm(render_poses, desc="Rendering")):
            # 动态模式：每一帧使用不同的alpha值重新预测MLP
            if args.mode == 'dynamic':
                # alpha从0到1线性变化
                alpha = i / (len(render_poses) - 1) if len(render_poses) > 1 else 0.5
                
                # 为当前帧预测MLP参数
                outputs = dynamic_model.interpolate(
                    images_a=image_paths_a,
                    images_b=image_paths_b,
                    alpha=alpha,
                    heads=["nlp"]
                )
                mlp_params = outputs['nmlp']
                
                # 应用MLP参数到模型
                apply_mlp_parameters_to_model(nerf_model, mlp_params)
            
            # 静态模式：第一帧时预测一次MLP
            elif args.mode == 'static' and i == 0:
                outputs = dynamic_model.interpolate(
                    images_a=image_paths,
                    images_b=image_paths,  # 静态模式：两组相同
                    alpha=0.0,  # alpha无影响
                    heads=["nlp"]
                )
                mlp_params = outputs['nmlp']
                apply_mlp_parameters_to_model(nerf_model, mlp_params)
            
            nerf_model.eval()
            
            c2w = torch.from_numpy(pose[:3, :4]).float().to(device)
            
            rgb, depth, acc = render_image(
                H=args.H,
                W=args.W,
                K=K_tensor,
                c2w=c2w,
                nerf_model=nerf_model,
                embed_fn=embed_fn,
                embeddirs_fn=embeddirs_fn,
                near=2.,
                far=6.,
                N_samples=args.n_samples,
                chunk=args.chunk,
                use_viewdirs=True,
                white_bkgd=args.white_bkgd
            )
            
            # Convert to uint8
            rgb_np = rgb.cpu().numpy()
            rgb8 = to8b(rgb_np)
            frames.append(rgb8)
            
            if i == 0:
                print(f"First frame - RGB range: [{rgb_np.min():.3f}, {rgb_np.max():.3f}]")
            
            # 动态模式：显示当前alpha值
            if args.mode == 'dynamic' and i % 10 == 0:
                tqdm.write(f"Frame {i}/{len(render_poses)}, alpha={alpha:.3f}")
    
    # Step 5: Save video
    print("\n=== Step 5: Saving video ===")
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    imageio.mimwrite(args.output, frames, fps=30, quality=8)
    print(f"Video saved to {args.output}")
    
    # Also save some sample frames
    samples_dir = args.output.replace('.mp4', '_frames')
    os.makedirs(samples_dir, exist_ok=True)
    for i in [0, len(frames)//4, len(frames)//2, 3*len(frames)//4, len(frames)-1]:
        imageio.imwrite(os.path.join(samples_dir, f'frame_{i:03d}.png'), frames[i])
    print(f"Sample frames saved to {samples_dir}")


if __name__ == '__main__':
    main()
