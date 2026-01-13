"""
render a 360-degree video from a trained Dynamic NeRF model.

Usage:
    python render_video.py --ckpt logs/dnerf_deformation/200000.tar \
                           --datadir ../../D_NeRF_Dataset/data/bouncingballs \
                           --output_dir ./videos
"""

import os
import sys
import argparse
import numpy as np
import torch
import imageio
from tqdm import tqdm

# Handle imports for both module and standalone execution
try:
    from .utils import device, to8b
    from .render import render_dnerf
    from .model import create_dnerf
    from .config import config_parser
    from .load_dnerf import load_dnerf_data, get_360_poses, pose_spherical
except ImportError:
    from utils import device, to8b
    from render import render_dnerf
    from model import create_dnerf
    from config import config_parser
    from load_dnerf import load_dnerf_data, get_360_poses, pose_spherical


def render_360_video(args, output_path, n_frames=120, fps=30,
                     time_mode='cycle', fixed_time=0.5):
    """Render a 360 video from trained Dynamic NeRF.
    
    Args:
        args: model configuration arguments
        output_path: path to save the video
        n_frames: number of frames in the video
        fps: frames per second
        time_mode: 'cycle' (animate time) or 'fixed' (static time)
        fixed_time: time value when time_mode='fixed'
    """
    # Load data to get camera parameters
    images, poses, times, render_poses, render_times, hwf, i_split = load_dnerf_data(
        args.datadir, args.half_res, testskip=1
    )
    
    H, W, focal = hwf
    H, W = int(H), int(W)
    
    K = np.array([
        [focal, 0, 0.5 * W],
        [0, focal, 0.5 * H],
        [0, 0, 1]
    ])
    
    # Create model and load checkpoint
    render_kwargs_train, render_kwargs_test, start, _, _ = create_dnerf(args)
    
    # Set bounds
    near = 2.
    far = 6.
    render_kwargs_test.update({'near': near, 'far': far})
    
    # Generate camera poses for 360 rotation
    angles = np.linspace(-180, 180, n_frames + 1)[:-1]
    camera_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in angles], 0)
    
    # Generate time values
    if time_mode == 'cycle':
        # Time cycles from 0 to 1 and back to 0
        time_vals = np.concatenate([
            np.linspace(0, 1, n_frames // 2),
            np.linspace(1, 0, n_frames - n_frames // 2)
        ])
    elif time_mode == 'linear':
        # Time goes linearly from 0 to 1
        time_vals = np.linspace(0, 1, n_frames)
    else:  # fixed
        time_vals = np.ones(n_frames) * fixed_time
    
    print(f"Rendering 360 video with {n_frames} frames...")
    print(f"Time mode: {time_mode}")
    print(f"Output: {output_path}")
    
    # Render frames
    frames = []
    
    for i, (pose, time_val) in enumerate(tqdm(zip(camera_poses, time_vals))):
        pose = pose.to(device)
        
        with torch.no_grad():
            rgb, disp, acc, extras = render_dnerf(
                H, W, K, time_val,
                chunk=args.chunk,
                c2w=pose[:3, :4],
                **render_kwargs_test
            )
        
        frame = to8b(rgb.cpu().numpy())
        frames.append(frame)
    
    frames = np.stack(frames, 0)
    
    # Save video
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    imageio.mimwrite(output_path, frames, fps=fps, quality=8)
    print(f"Saved video to {output_path}")
    
    # Also save individual frames if requested
    frames_dir = output_path.replace('.mp4', '_frames')
    if os.path.exists(frames_dir) or True:  # Always save frames
        os.makedirs(frames_dir, exist_ok=True)
        for i, frame in enumerate(frames):
            imageio.imwrite(os.path.join(frames_dir, f'{i:04d}.png'), frame)
        print(f"Saved {len(frames)} frames to {frames_dir}")
    
    return frames


def render_multi_time_video(args, output_dir, n_angles=40, n_times=5, fps=30):
    """Render videos at multiple fixed times for comparison.
    
    Args:
        args: model configuration arguments
        output_dir: directory to save videos
        n_angles: number of angles for 360 rotation
        n_times: number of time samples
        fps: frames per second
    """
    os.makedirs(output_dir, exist_ok=True)
    
    time_samples = np.linspace(0, 1, n_times)
    
    for i, t in enumerate(time_samples):
        output_path = os.path.join(output_dir, f'time_{t:.2f}.mp4')
        print(f"\nRendering at time t={t:.2f}...")
        render_360_video(
            args, output_path, n_frames=n_angles, fps=fps,
            time_mode='fixed', fixed_time=t
        )


def render_comparison_video(args_sf, args_df, output_path, n_frames=120, fps=30):
    """Render side-by-side comparison video of both approaches.
    
    Args:
        args_sf: args for straightforward model
        args_df: args for deformation model
        output_path: path to save comparison video
        n_frames: number of frames
        fps: frames per second
    """
    # Load data
    images, poses, times, render_poses, render_times, hwf, i_split = load_dnerf_data(
        args_sf.datadir, args_sf.half_res, testskip=1
    )
    
    H, W, focal = hwf
    H, W = int(H), int(W)
    
    K = np.array([
        [focal, 0, 0.5 * W],
        [0, focal, 0.5 * H],
        [0, 0, 1]
    ])
    
    # Create both models
    print("Loading straightforward model...")
    render_kwargs_train_sf, render_kwargs_test_sf, _, _, _ = create_dnerf(args_sf)
    render_kwargs_test_sf.update({'near': 2., 'far': 6.})
    
    print("Loading deformation model...")
    render_kwargs_train_df, render_kwargs_test_df, _, _, _ = create_dnerf(args_df)
    render_kwargs_test_df.update({'near': 2., 'far': 6.})
    
    # Generate camera poses and times
    angles = np.linspace(-180, 180, n_frames + 1)[:-1]
    camera_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in angles], 0)
    time_vals = np.linspace(0, 1, n_frames)
    
    print(f"Rendering comparison video with {n_frames} frames...")
    
    frames = []
    
    for i, (pose, time_val) in enumerate(tqdm(zip(camera_poses, time_vals))):
        pose = pose.to(device)
        
        with torch.no_grad():
            # Render straightforward
            rgb_sf, _, _, _ = render_dnerf(
                H, W, K, time_val, chunk=args_sf.chunk,
                c2w=pose[:3, :4], **render_kwargs_test_sf
            )
            
            # Render deformation
            rgb_df, _, _, _ = render_dnerf(
                H, W, K, time_val, chunk=args_df.chunk,
                c2w=pose[:3, :4], **render_kwargs_test_df
            )
        
        # Create side-by-side frame
        rgb_sf = rgb_sf.cpu().numpy()
        rgb_df = rgb_df.cpu().numpy()
        
        # Add labels
        comparison = np.concatenate([rgb_sf, rgb_df], axis=1)
        frame = to8b(comparison)
        frames.append(frame)
    
    frames = np.stack(frames, 0)
    
    # Save video
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    imageio.mimwrite(output_path, frames, fps=fps, quality=8)
    print(f"Saved comparison video to {output_path}")
    
    return frames


def parse_render_args():
    """Parse rendering arguments."""
    parser = argparse.ArgumentParser(description='Render 360 video from Dynamic NeRF')
    
    parser.add_argument('--ckpt', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--datadir', type=str, required=True,
                        help='Path to D-NeRF dataset')
    parser.add_argument('--output_dir', type=str, default='./videos',
                        help='Directory to save videos')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file')
    
    # Video options
    parser.add_argument('--n_frames', type=int, default=120,
                        help='Number of frames in video')
    parser.add_argument('--fps', type=int, default=30,
                        help='Frames per second')
    parser.add_argument('--time_mode', type=str, default='cycle',
                        choices=['cycle', 'linear', 'fixed'],
                        help='How time changes in video')
    parser.add_argument('--fixed_time', type=float, default=0.5,
                        help='Time value when time_mode=fixed')
    
    # Model options
    parser.add_argument('--network_type', type=str, default='deformation',
                        choices=['straightforward', 'deformation'],
                        help='Type of Dynamic NeRF model')
    parser.add_argument('--half_res', action='store_true', default=True,
                        help='Render at half resolution')
    parser.add_argument('--chunk', type=int, default=1024*32,
                        help='Chunk size for rendering')
    
    return parser.parse_args()


def main():
    """Main rendering function."""
    render_args = parse_render_args()
    
    # Parse model config
    config_parser_fn = config_parser()
    if render_args.config is not None:
        args = config_parser_fn.parse_args(['--config', render_args.config])
    else:
        args = config_parser_fn.parse_args([])
    
    # Override with render args
    args.network_type = render_args.network_type
    args.ft_path = render_args.ckpt
    args.datadir = render_args.datadir
    args.half_res = render_args.half_res
    args.chunk = render_args.chunk
    
    # Create output directory
    os.makedirs(render_args.output_dir, exist_ok=True)
    
    # Generate output filename
    basename = os.path.basename(render_args.datadir)
    output_path = os.path.join(
        render_args.output_dir,
        f'{basename}_{args.network_type}_{render_args.time_mode}.mp4'
    )
    
    # Render video
    render_360_video(
        args, output_path,
        n_frames=render_args.n_frames,
        fps=render_args.fps,
        time_mode=render_args.time_mode,
        fixed_time=render_args.fixed_time
    )


if __name__ == '__main__':
    main()
