"""
Quantitative Evaluation for Dynamic NeRF

This script evaluates trained Dynamic NeRF models and compares
the two approaches (straightforward vs deformation) quantitatively.

Metrics computed:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- LPIPS (Learned Perceptual Image Patch Similarity)

Usage:
    python evaluate.py --ckpt_straightforward logs/dnerf_straightforward/200000.tar \
                       --ckpt_deformation logs/dnerf_deformation/200000.tar \
                       --datadir ../../D_NeRF_Dataset/data/bouncingballs
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import imageio
from tqdm import tqdm
from collections import defaultdict

# Handle imports for both module and standalone execution
try:
    from .utils import device, to8b, compute_psnr, compute_ssim, compute_lpips, evaluate_images
    from .render import render_dnerf
    from .model import create_dnerf, load_checkpoint
    from .config import config_parser
    from .load_dnerf import load_dnerf_data
except ImportError:
    from utils import device, to8b, compute_psnr, compute_ssim, compute_lpips, evaluate_images
    from render import render_dnerf
    from model import create_dnerf, load_checkpoint
    from config import config_parser
    from load_dnerf import load_dnerf_data


def evaluate_model(args, images, poses, times, i_test, hwf, K, save_dir=None):
    """Evaluate a single Dynamic NeRF model on test set.
    
    Args:
        args: model configuration arguments
        images: all images [N, H, W, 3]
        poses: all camera poses [N, 4, 4]
        times: all time values [N]
        i_test: indices for test set
        hwf: [H, W, focal]
        K: camera intrinsic matrix
        save_dir: directory to save rendered images (optional)
    
    Returns:
        metrics: dict with evaluation metrics
        pred_images: [N_test, H, W, 3] predicted images
    """
    H, W, focal = hwf
    H, W = int(H), int(W)
    
    # Create model and load checkpoint
    render_kwargs_train, render_kwargs_test, start, _, _ = create_dnerf(args)
    
    # Set bounds
    near = 2.
    far = 6.
    render_kwargs_test.update({'near': near, 'far': far})
    
    # Evaluate on test set
    pred_images = []
    gt_images = images[i_test]
    
    print(f"Evaluating {args.network_type} model on {len(i_test)} test images...")
    
    for idx, test_idx in enumerate(tqdm(i_test)):
        pose = torch.Tensor(poses[test_idx]).to(device)
        time_val = times[test_idx]
        
        with torch.no_grad():
            rgb, disp, acc, extras = render_dnerf(
                H, W, K, time_val,
                chunk=args.chunk,
                c2w=pose[:3, :4],
                **render_kwargs_test
            )
        
        pred_img = rgb.cpu().numpy()
        pred_images.append(pred_img)
        
        if save_dir is not None:
            # Save predicted and GT images
            pred_path = os.path.join(save_dir, f'pred_{idx:03d}.png')
            gt_path = os.path.join(save_dir, f'gt_{idx:03d}.png')
            imageio.imwrite(pred_path, to8b(pred_img))
            imageio.imwrite(gt_path, to8b(gt_images[idx]))
    
    pred_images = np.stack(pred_images, 0)
    
    # Compute metrics
    metrics = evaluate_images(pred_images, gt_images, compute_lpips_flag=True)
    
    return metrics, pred_images


def compare_models(args_straightforward, args_deformation, datadir, output_dir):
    """Compare two Dynamic NeRF approaches.
    
    Args:
        args_straightforward: args for straightforward model
        args_deformation: args for deformation model
        datadir: path to dataset
        output_dir: directory to save comparison results
    
    Returns:
        comparison: dict with comparison results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    images, poses, times, render_poses, render_times, hwf, i_split = load_dnerf_data(
        datadir, half_res=True, testskip=1
    )
    i_train, i_val, i_test = i_split
    
    H, W, focal = hwf
    K = np.array([
        [focal, 0, 0.5 * W],
        [0, focal, 0.5 * H],
        [0, 0, 1]
    ])
    
    comparison = {}
    
    # Evaluate straightforward model
    if args_straightforward is not None:
        sf_save_dir = os.path.join(output_dir, 'straightforward')
        os.makedirs(sf_save_dir, exist_ok=True)
        
        print("\n" + "=" * 60)
        print("Evaluating STRAIGHTFORWARD (6D) model")
        print("=" * 60)
        
        sf_metrics, sf_pred = evaluate_model(
            args_straightforward, images, poses, times, i_test, hwf, K, sf_save_dir
        )
        comparison['straightforward'] = sf_metrics
        
        print(f"\nStraightforward Results:")
        print(f"  PSNR: {sf_metrics['psnr']:.2f} ± {sf_metrics['psnr_std']:.2f}")
        print(f"  SSIM: {sf_metrics['ssim']:.4f} ± {sf_metrics['ssim_std']:.4f}")
        if 'lpips' in sf_metrics:
            print(f"  LPIPS: {sf_metrics['lpips']:.4f} ± {sf_metrics['lpips_std']:.4f}")
    
    # Evaluate deformation model
    if args_deformation is not None:
        df_save_dir = os.path.join(output_dir, 'deformation')
        os.makedirs(df_save_dir, exist_ok=True)
        
        print("\n" + "=" * 60)
        print("Evaluating DEFORMATION NETWORK model")
        print("=" * 60)
        
        df_metrics, df_pred = evaluate_model(
            args_deformation, images, poses, times, i_test, hwf, K, df_save_dir
        )
        comparison['deformation'] = df_metrics
        
        print(f"\nDeformation Network Results:")
        print(f"  PSNR: {df_metrics['psnr']:.2f} ± {df_metrics['psnr_std']:.2f}")
        print(f"  SSIM: {df_metrics['ssim']:.4f} ± {df_metrics['ssim_std']:.4f}")
        if 'lpips' in df_metrics:
            print(f"  LPIPS: {df_metrics['lpips']:.4f} ± {df_metrics['lpips_std']:.4f}")
    
    # Print comparison summary
    if args_straightforward is not None and args_deformation is not None:
        print("\n" + "=" * 60)
        print("COMPARISON SUMMARY")
        print("=" * 60)
        
        psnr_diff = comparison['deformation']['psnr'] - comparison['straightforward']['psnr']
        ssim_diff = comparison['deformation']['ssim'] - comparison['straightforward']['ssim']
        
        print(f"\n{'Metric':<15} {'Straightforward':>15} {'Deformation':>15} {'Difference':>15}")
        print("-" * 60)
        print(f"{'PSNR (↑)':<15} {comparison['straightforward']['psnr']:>15.2f} {comparison['deformation']['psnr']:>15.2f} {psnr_diff:>+15.2f}")
        print(f"{'SSIM (↑)':<15} {comparison['straightforward']['ssim']:>15.4f} {comparison['deformation']['ssim']:>15.4f} {ssim_diff:>+15.4f}")
        
        if 'lpips' in comparison['straightforward'] and 'lpips' in comparison['deformation']:
            lpips_diff = comparison['deformation']['lpips'] - comparison['straightforward']['lpips']
            print(f"{'LPIPS (↓)':<15} {comparison['straightforward']['lpips']:>15.4f} {comparison['deformation']['lpips']:>15.4f} {lpips_diff:>+15.4f}")
        
        print("\n(↑ higher is better, ↓ lower is better)")
        
        # Determine winner
        print("\n" + "-" * 60)
        if psnr_diff > 0:
            print("Winner based on PSNR: DEFORMATION NETWORK")
        elif psnr_diff < 0:
            print("Winner based on PSNR: STRAIGHTFORWARD (6D)")
        else:
            print("TIE based on PSNR")
    
    # Save comparison to JSON
    comparison_file = os.path.join(output_dir, 'comparison.json')
    with open(comparison_file, 'w') as f:
        # Convert numpy values to Python floats for JSON serialization
        comparison_json = {}
        for model_name, metrics in comparison.items():
            comparison_json[model_name] = {k: float(v) for k, v in metrics.items()}
        json.dump(comparison_json, f, indent=2)
    print(f"\nSaved comparison results to {comparison_file}")
    
    return comparison


def parse_eval_args():
    """Parse evaluation arguments."""
    parser = argparse.ArgumentParser(description='Evaluate and compare Dynamic NeRF models')
    
    parser.add_argument('--datadir', type=str, required=True,
                        help='Path to D-NeRF dataset')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                        help='Directory to save evaluation results')
    
    # Straightforward model
    parser.add_argument('--ckpt_straightforward', type=str, default=None,
                        help='Path to straightforward model checkpoint')
    parser.add_argument('--config_straightforward', type=str, default=None,
                        help='Config file for straightforward model')
    
    # Deformation model
    parser.add_argument('--ckpt_deformation', type=str, default=None,
                        help='Path to deformation model checkpoint')
    parser.add_argument('--config_deformation', type=str, default=None,
                        help='Config file for deformation model')
    
    # Common options
    parser.add_argument('--half_res', action='store_true', default=True,
                        help='Evaluate at half resolution')
    parser.add_argument('--chunk', type=int, default=1024*32,
                        help='Chunk size for rendering')
    
    return parser.parse_args()


def main():
    """Main evaluation function."""
    eval_args = parse_eval_args()
    
    # Prepare args for each model
    config_parser_fn = config_parser()
    
    args_sf = None
    args_df = None
    
    if eval_args.ckpt_straightforward is not None:
        if eval_args.config_straightforward is not None:
            args_sf = config_parser_fn.parse_args(['--config', eval_args.config_straightforward])
        else:
            args_sf = config_parser_fn.parse_args([])
        args_sf.network_type = 'straightforward'
        args_sf.ft_path = eval_args.ckpt_straightforward
        args_sf.datadir = eval_args.datadir
        args_sf.half_res = eval_args.half_res
        args_sf.chunk = eval_args.chunk
    
    if eval_args.ckpt_deformation is not None:
        if eval_args.config_deformation is not None:
            args_df = config_parser_fn.parse_args(['--config', eval_args.config_deformation])
        else:
            args_df = config_parser_fn.parse_args([])
        args_df.network_type = 'deformation'
        args_df.ft_path = eval_args.ckpt_deformation
        args_df.datadir = eval_args.datadir
        args_df.half_res = eval_args.half_res
        args_df.chunk = eval_args.chunk
    
    if args_sf is None and args_df is None:
        print("Error: Please provide at least one checkpoint to evaluate.")
        print("Use --ckpt_straightforward and/or --ckpt_deformation")
        return
    
    # Run comparison
    comparison = compare_models(args_sf, args_df, eval_args.datadir, eval_args.output_dir)
    
    return comparison


if __name__ == '__main__':
    main()
