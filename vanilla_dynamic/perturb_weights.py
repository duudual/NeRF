"""
Load official D-NeRF weights and create a new checkpoint with small random perturbations.

This can be used to:
1. Create slightly different initializations for ensemble training
2. Test model robustness to weight perturbations
3. Start fine-tuning from a perturbed version of official weights

Usage:
    python perturb_weights.py --input_ckpt "D:/lecture/2.0_xk/CV/finalproject/D-NeRF/logs/bouncingballs/800000.tar" --output_base "./perturbed_weights/bouncingballs" --noise_std 0.01 --expname bouncingballs   
python perturb_weights.py --input_ckpt "D:/lecture/2.0_xk/CV/finalproject/D-NeRF/logs/hook/800000.tar" --output_base "./perturbed_weights/hook" --noise_std 0.01 --expname hook
python perturb_weights.py --input_ckpt "D:/lecture/2.0_xk/CV/finalproject/D-NeRF/logs/jumpingjacks/800000.tar" --output_base "./perturbed_weights/jumpingjacks" --noise_std 0.01 --expname jumpingjacks
python perturb_weights.py --input_ckpt "D:/lecture/2.0_xk/CV/finalproject/D-NeRF/logs/lego/800000.tar" --output_base "./perturbed_weights/lego" --noise_std 0.01 --expname lego
python perturb_weights.py --input_ckpt "D:/lecture/2.0_xk/CV/finalproject/D-NeRF/logs/mutant/800000.tar" --output_base "./perturbed_weights/mutant" --noise_std 0.01 --expname mutant
python perturb_weights.py --input_ckpt "D:/lecture/2.0_xk/CV/finalproject/D-NeRF/logs/standup/800000.tar" --output_base "./perturbed_weights/standup" --noise_std 0.01 --expname standup
python perturb_weights.py --input_ckpt "D:/lecture/2.0_xk/CV/finalproject/D-NeRF/logs/trex/800000.tar" --output_base "./perturbed_weights/trex" --noise_std 0.01 --expname trex
    
    """

import os
import sys
import argparse
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from network import DNeRF_Deformation
from positional_encoding import get_embedder, get_time_embedder
from utils import device


def perturb_weights(state_dict, noise_std=0.01, noise_type='gaussian', seed=None):
    """
    Add small random perturbations to model weights.
    
    Args:
        state_dict: model state dictionary
        noise_std: standard deviation of noise (relative to weight std)
        noise_type: 'gaussian' or 'uniform'
        seed: random seed for reproducibility
    
    Returns:
        perturbed_state_dict: state dictionary with perturbed weights
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    perturbed_state_dict = {}
    
    for key, value in state_dict.items():
        if value.dtype in [torch.float32, torch.float64]:
            # Calculate perturbation scale based on weight statistics
            # Use mean absolute value for single-element tensors (std would be NaN)
            if value.numel() <= 1:
                # For single element, use absolute value as scale
                weight_scale = value.abs().item()
                if weight_scale == 0:
                    weight_scale = 1.0
            else:
                weight_scale = value.std().item()
                if weight_scale == 0 or np.isnan(weight_scale):
                    weight_scale = value.abs().mean().item()
                    if weight_scale == 0:
                        weight_scale = 1.0
            
            # Generate noise
            if noise_type == 'gaussian':
                noise = torch.randn_like(value) * noise_std * weight_scale
            elif noise_type == 'uniform':
                noise = (torch.rand_like(value) * 2 - 1) * noise_std * weight_scale
            else:
                raise ValueError(f"Unknown noise type: {noise_type}")
            
            perturbed_state_dict[key] = value + noise
        else:
            # Keep non-float tensors unchanged
            perturbed_state_dict[key] = value
    
    return perturbed_state_dict


def main():
    parser = argparse.ArgumentParser(description='Perturb official D-NeRF weights')
    parser.add_argument('--input_ckpt', type=str, required=True,
                        help='Path to official D-NeRF checkpoint')
    parser.add_argument('--output_base', type=str, default=None,
                        help='Output path for perturbed checkpoint')
    parser.add_argument('--noise_std', type=float, default=0.01,
                        help='Noise standard deviation (relative to weight std)')
    parser.add_argument('--noise_type', type=str, default='gaussian',
                        choices=['gaussian', 'uniform'],
                        help='Type of noise to add')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--verify', action='store_true',
                        help='Verify by loading the perturbed model')
    parser.add_argument('--expname', type=str, default='')
    
    args = parser.parse_args()
    
    # Default output path
    if args.output_base is None:
        base, ext = os.path.splitext(args.input_ckpt)
        args.output_base = f"{base}_perturbed_{args.noise_std}{ext}"
    
    print("=" * 60)
    print("D-NeRF Weight Perturbation")
    print("=" * 60)
    print(f"Input checkpoint: {args.input_ckpt}")
    print(f"Output checkpoint: {args.output_base}")
    print(f"Noise std: {args.noise_std}")
    print(f"Noise type: {args.noise_type}")
    print(f"Seed: {args.seed}")
    print("=" * 60)
    
    # Load checkpoint
    print("\nLoading checkpoint...")
    ckpt = torch.load(args.input_ckpt, map_location='cpu')
    
    if 'network_fn_state_dict' in ckpt:
        state_dict = ckpt['network_fn_state_dict']
        is_full_ckpt = True
    else:
        state_dict = ckpt
        is_full_ckpt = False
    
    print(f"Found {len(state_dict)} weight tensors")
    
    # Calculate original weight statistics
    total_params = 0
    for key, value in state_dict.items():
        if value.dtype in [torch.float32, torch.float64]:
            total_params += value.numel()
    print(f"Total parameters: {total_params:,}")
    
    # Perturb weights
    print("\nPerturbing weights...")
    perturbed_state_dict = perturb_weights(
        state_dict, 
        noise_std=args.noise_std,
        noise_type=args.noise_type,
        seed=args.seed
    )
    
    # Calculate perturbation statistics
    print("\nPerturbation statistics:")
    for key in list(state_dict.keys())[:5]:  # Show first 5 layers
        if state_dict[key].dtype in [torch.float32, torch.float64]:
            orig = state_dict[key]
            pert = perturbed_state_dict[key]
            diff = (pert - orig).abs()
            print(f"  {key}:")
            print(f"    Original: mean={orig.mean():.6f}, std={orig.std():.6f}")
            print(f"    Perturbed: mean={pert.mean():.6f}, std={pert.std():.6f}")
            print(f"    Diff: mean={diff.mean():.6f}, max={diff.max():.6f}")
    print("  ...")
    
    # Create new checkpoint
    if is_full_ckpt:
        new_ckpt = {
            'network_fn_state_dict': perturbed_state_dict,
            'global_step': 0,  # Reset step since this is a new initialization
        }
        # Copy fine model if exists
        if 'network_fine_state_dict' in ckpt:
            new_ckpt['network_fine_state_dict'] = perturb_weights(
                ckpt['network_fine_state_dict'],
                noise_std=args.noise_std,
                noise_type=args.noise_type,
                seed=args.seed + 1 if args.seed else None
            )
    else:
        new_ckpt = perturbed_state_dict
    save_path = os.path.join(args.output_base , f"best.tar")
    # Save perturbed checkpoint
    print(f"\nSaving perturbed checkpoint to {save_path}...")
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    torch.save(new_ckpt, save_path)
    print("✓ Saved successfully!")
    
    # Verify by loading
    if args.verify:
        print("\nVerifying perturbed model...")
        embed_fn, input_ch = get_embedder(10, 0)
        embeddirs_fn, input_ch_views = get_embedder(4, 0)
        embed_time_fn, input_ch_time = get_time_embedder(10)
        
        model = DNeRF_Deformation(
            D=8, W=256,
            D_deform=8, W_deform=256,
            input_ch=input_ch, output_ch=4, skips=[4],
            skips_deform=[4],
            input_ch_views=input_ch_views, input_ch_time=input_ch_time,
            use_viewdirs=True,
            zero_canonical=True
        )
        model.embed_fn = embed_fn
        
        model.load_state_dict(perturbed_state_dict, strict=True)
        print("✓ Model loaded successfully with perturbed weights!")
        
        # Test forward pass
        test_pts = torch.randn(10, 3)
        test_pts_embed = embed_fn(test_pts)
        test_views = torch.randn(10, 3)
        test_views_embed = embeddirs_fn(test_views)
        test_time = torch.ones(10, 1) * 0.5
        test_time_embed = embed_time_fn(test_time)
        
        x = torch.cat([test_pts_embed, test_views_embed], dim=-1)
        ts = [test_time_embed, test_time_embed]
        
        model.eval()
        with torch.no_grad():
            out, dx = model(x, ts)
        
        print(f"✓ Forward pass successful!")
        print(f"  Output shape: {out.shape}")
        print(f"  Output range: [{out.min():.4f}, {out.max():.4f}]")
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == '__main__':
    main()
