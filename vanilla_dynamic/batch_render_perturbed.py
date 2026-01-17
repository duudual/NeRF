"""
Batch render videos for all perturbed weights.

Usage:
    python batch_render_perturbed.py
"""

import os
import sys
import glob
import torch
import numpy as np
import imageio
from tqdm import tqdm

from utils import device, to8b
from render import render_path_dnerf
from model import create_dnerf
from config import config_parser
from load_dnerf import load_dnerf_data


# Dataset paths
DATASETS = {
    'bouncingballs': 'D:/lecture/2.0_xk/CV/finalproject/D_NeRF_Dataset/data/bouncingballs',
    'hellwarrior': 'D:/lecture/2.0_xk/CV/finalproject/D_NeRF_Dataset/data/hellwarrior',
    'hook': 'D:/lecture/2.0_xk/CV/finalproject/D_NeRF_Dataset/data/hook',
    'jumpingjacks': 'D:/lecture/2.0_xk/CV/finalproject/D_NeRF_Dataset/data/jumpingjacks',
    'lego': 'D:/lecture/2.0_xk/CV/finalproject/D_NeRF_Dataset/data/lego',
    'mutant': 'D:/lecture/2.0_xk/CV/finalproject/D_NeRF_Dataset/data/mutant',
    'standup': 'D:/lecture/2.0_xk/CV/finalproject/D_NeRF_Dataset/data/standup',
    'trex': 'D:/lecture/2.0_xk/CV/finalproject/D_NeRF_Dataset/data/trex',
}

PERTURBED_WEIGHTS_DIR = './perturbed_weights'


def render_video_from_weights(ckpt_path, datadir, scene_name, output_dir, half_res=True):
    """Render video from a checkpoint file."""
    
    print(f"\n{'='*60}")
    print(f"Rendering: {scene_name}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"{'='*60}")
    
    # Load data
    images, poses, times, render_poses, render_times, hwf, i_split = load_dnerf_data(
        datadir, half_res=half_res, testskip=4
    )
    
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]
    
    K = np.array([
        [focal, 0, 0.5 * W],
        [0, focal, 0.5 * H],
        [0, 0, 1]
    ])
    
    near, far = 2., 6.
    
    # Create args for model creation
    # IMPORTANT: use_layernorm=False because perturbed weights are from official D-NeRF
    class Args:
        def __init__(self):
            self.network_type = 'deformation'
            self.netdepth = 8
            self.netwidth = 256
            self.netdepth_fine = 8
            self.netwidth_fine = 256
            self.netdepth_deform = 8
            self.netwidth_deform = 256
            self.N_importance = 64
            self.N_samples = 64
            self.use_viewdirs = True
            self.i_embed = 0
            self.multires = 10
            self.multires_views = 4
            self.multires_time = 10
            self.no_include_input = False  # Official D-NeRF uses include_input=True
            self.use_layernorm = False  # No LayerNorm in official weights!
            self.perturb = 0.0  # No perturbation during rendering
            self.raw_noise_std = 0.0
            self.white_bkgd = True
            self.lindisp = False
            self.chunk = 1024 * 16
            self.netchunk = 1024 * 32
            self.zero_canonical = True
            self.ft_path = ckpt_path  # Load this checkpoint
            self.no_reload = False
            self.basedir = output_dir
            self.expname = scene_name
            self.lrate = 5e-4  # Not used for rendering but required
            self.lrate_decay = 250
    
    args = Args()
    
    # Create model and load weights
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_dnerf(args)
    
    # Update bounds
    render_kwargs_test.update({'near': near, 'far': far})
    
    # Move poses to device
    render_poses = render_poses.to(device)
    render_times = render_times.to(device)
    
    # Render
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Rendering {len(render_poses)} frames...")
    with torch.no_grad():
        rgbs, disps = render_path_dnerf(
            render_poses, render_times, hwf, K, args.chunk,
            render_kwargs_test, savedir=output_dir
        )
    
    # Save video
    video_path = os.path.join(output_dir, 'video.mp4')
    imageio.mimwrite(video_path, to8b(rgbs), fps=30, quality=8)
    print(f"Saved video to {video_path}")
    
    # Clean up GPU memory
    del render_kwargs_train, render_kwargs_test
    torch.cuda.empty_cache()
    
    return video_path


def main():
    # Find all perturbed weight directories
    scene_dirs = sorted(glob.glob(os.path.join(PERTURBED_WEIGHTS_DIR, '*')))
    
    print(f"Found {len(scene_dirs)} scenes to render")
    
    results = []
    
    for scene_dir in scene_dirs:
        scene_name = os.path.basename(scene_dir)
        
        # Find checkpoint file
        ckpt_path = os.path.join(scene_dir, 'best.tar')
        if not os.path.exists(ckpt_path):
            print(f"Skipping {scene_name}: no best.tar found")
            continue
        
        # Get dataset path
        if scene_name not in DATASETS:
            print(f"Skipping {scene_name}: no dataset path configured")
            continue
        
        datadir = DATASETS[scene_name]
        if not os.path.exists(datadir):
            print(f"Skipping {scene_name}: dataset not found at {datadir}")
            continue
        
        # Output directory
        output_dir = os.path.join(scene_dir, 'videos')
        
        try:
            video_path = render_video_from_weights(
                ckpt_path, datadir, scene_name, output_dir
            )
            results.append((scene_name, 'success', video_path))
        except Exception as e:
            print(f"Error rendering {scene_name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((scene_name, 'failed', str(e)))
    
    # Print summary
    print("\n" + "="*60)
    print("RENDER SUMMARY")
    print("="*60)
    for scene_name, status, info in results:
        print(f"{scene_name}: {status} - {info}")


if __name__ == '__main__':
    main()
