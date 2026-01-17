"""
Dynamic NeRF Model Creation and Checkpoint Management

This module handles the creation of Dynamic NeRF models (both straightforward
and deformation approaches) and manages checkpoint saving/loading.
"""

import os
import torch
import torch.optim as optim

# Handle imports for both module and standalone execution
try:
    from .utils import device
    from .positional_encoding import get_embedder, get_time_embedder
    from .network import DNeRF_Straightforward, DNeRF_Deformation
    from .render import run_network_dnerf
except ImportError:
    from utils import device
    from positional_encoding import get_embedder, get_time_embedder
    from network import DNeRF_Straightforward, DNeRF_Deformation
    from render import run_network_dnerf


def create_dnerf(args):
    """Instantiate Dynamic NeRF's MLP model.
    
    Args:
        args: parsed arguments containing model configuration
    
    Returns:
        render_kwargs_train: dictionary of training render arguments
        render_kwargs_test: dictionary of test render arguments
        start: starting iteration (from checkpoint)
        grad_vars: list of trainable parameters
        optimizer: optimizer
    """
    # Create embedders for position and view directions
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)
    
    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    
    # Create embedder for time
    embed_time_fn, input_ch_time = get_time_embedder(args.multires_time)
    
    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]
    skips_deform = [3]
    
    # Select network type based on args
    if args.network_type == 'straightforward':
        # Straightforward 6D approach
        model = DNeRF_Straightforward(
            D=args.netdepth, W=args.netwidth,
            input_ch=input_ch, output_ch=output_ch, skips=skips,
            input_ch_views=input_ch_views, input_ch_time=input_ch_time,
            use_viewdirs=args.use_viewdirs
        ).to(device)
        
        model_fine = None
        if args.N_importance > 0:
            model_fine = DNeRF_Straightforward(
                D=args.netdepth_fine, W=args.netwidth_fine,
                input_ch=input_ch, output_ch=output_ch, skips=skips,
                input_ch_views=input_ch_views, input_ch_time=input_ch_time,
                use_viewdirs=args.use_viewdirs
            ).to(device)
    else:
        # Deformation network approach - MATCHES OFFICIAL D-NeRF EXACTLY
        # Official D-NeRF architecture:
        # - Canonical network (_occ): D=8, W=256, skips=[4]
        # - Time/deformation network (_time): D=8, W=256, skips=[4] (same as canonical!)
        # The DNeRF_Deformation class uses self.skips_deform = skips internally
        model = DNeRF_Deformation(
            D=args.netdepth, W=args.netwidth,
            D_deform=8, W_deform=256,  # Match official D-NeRF
            input_ch=input_ch, output_ch=output_ch, skips=skips,
            skips_deform=skips,  # Official D-NeRF uses same skips for time network
            input_ch_views=input_ch_views, input_ch_time=input_ch_time,
            use_viewdirs=args.use_viewdirs,
            zero_canonical=args.zero_canonical
        ).to(device)
        model.embed_fn = embed_fn
        
        # IMPORTANT: Official D-NeRF does NOT use separate fine model
        # It reuses the same model for both coarse and fine sampling
        # Only create fine model if explicitly requested via use_two_models_for_fine
        model_fine = None
        use_two_models = getattr(args, 'use_two_models_for_fine', False)
        if args.N_importance > 0 and use_two_models:
            model_fine = DNeRF_Deformation(
                D=args.netdepth_fine, W=args.netwidth_fine,
                D_deform=8, W_deform=256,  # Match official D-NeRF
                input_ch=input_ch, output_ch=output_ch, skips=skips,
                skips_deform=skips,  # Official D-NeRF uses same skips for time network
                input_ch_views=input_ch_views, input_ch_time=input_ch_time,
                use_viewdirs=args.use_viewdirs,
                zero_canonical=args.zero_canonical
            ).to(device)
            model_fine.embed_fn = embed_fn
    
    # Collect trainable parameters
    grad_vars = list(model.parameters())
    if model_fine is not None:
        grad_vars += list(model_fine.parameters())
    
    # Create network query function
    network_query_fn = lambda inputs, viewdirs, times, network_fn: run_network_dnerf(
        inputs, viewdirs, times, network_fn,
        embed_fn=embed_fn,
        embeddirs_fn=embeddirs_fn,
        embed_time_fn=embed_time_fn,
        netchunk=args.netchunk,
        network_type=args.network_type
    )
    
    # Create optimizer
    optimizer = optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))
    
    start = 0
    basedir = args.basedir
    expname = args.expname
    
    # Load checkpoints
    if args.ft_path is not None and args.ft_path != 'None':
        # If checkpoint path is specified, use it
        ckpts = [args.ft_path]
    else:
        # If not specified, prioritize best.tar, otherwise use the latest checkpoint
        ckpt_dir = os.path.join(basedir, expname)
        best_ckpt = os.path.join(ckpt_dir, 'best.tar')
        # best_ckpt = os.path.join(ckpt_dir, 'latest.tar')
        
        
        if os.path.exists(best_ckpt):
            ckpts = [best_ckpt]
            print(f'Using best checkpoint: {best_ckpt}')
        elif os.path.exists(ckpt_dir):
            ckpts = [os.path.join(ckpt_dir, f) for f in sorted(os.listdir(ckpt_dir)) if 'tar' in f]
        else:
            ckpts = []
            print(f'Warning: Checkpoint directory {ckpt_dir} does not exist.')
    
    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=device)
        
        # Check if this is an official D-NeRF checkpoint
        # Official checkpoints have keys like '_occ.pts_linears.0.weight' directly in network_fn_state_dict
        # Our checkpoints might have different structure
        state_dict = ckpt.get('network_fn_state_dict', {})
        is_official_format = any(k.startswith('_occ.') or k.startswith('_time.') for k in state_dict.keys())
        
        if is_official_format and args.network_type == 'deformation':
            # Official D-NeRF checkpoint format - can load directly since our architecture now matches
            print('Detected official D-NeRF checkpoint format - loading directly')
            model.load_state_dict(state_dict, strict=True)
            start = ckpt.get('global_step', 0)
            print(f'✓ Official D-NeRF weights loaded successfully! (step {start})')
            # Don't try to load optimizer - it won't match
        else:
            # Our own checkpoint format
            start = ckpt['global_step']
            # Only load optimizer if it matches (skip for inference/rendering)
            try:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            except ValueError as e:
                print(f'Warning: Could not load optimizer state (this is OK for rendering): {e}')
            
            # Load model
            model.load_state_dict(ckpt['network_fn_state_dict'])
            if model_fine is not None and 'network_fine_state_dict' in ckpt:
                model_fine.load_state_dict(ckpt['network_fine_state_dict'])
    elif args.load_official_weights and args.network_type == 'deformation':
        # Load official D-NeRF pre-trained weights
        official_path = args.official_ckpt_path
        if official_path is None:
            # Try to find official checkpoint from D-NeRF logs
            scene_name = os.path.basename(os.path.normpath(args.datadir))
            official_logs = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'D-NeRF', 'logs', scene_name)
            if os.path.exists(official_logs):
                ckpts_found = sorted([f for f in os.listdir(official_logs) if f.endswith('.tar')])
                if ckpts_found:
                    official_path = os.path.join(official_logs, ckpts_found[-1])
        
        if official_path and os.path.exists(official_path):
            print(f'\n{"=" * 60}')
            print(f'Loading official D-NeRF weights from: {official_path}')
            print(f'{"=" * 60}')
            if load_official_dnerf_weights(model, official_path):
                print('✓ Official weights loaded successfully!')
                start = 0  # Start from iteration 0
            else:
                print('✗ Failed to load official weights, starting from scratch')
        else:
            print(f'Warning: Official weights path not found: {official_path}')
            if args.official_ckpt_path:
                print(f'  Searched at: {args.official_ckpt_path}')
    
    # Prepare render kwargs
    render_kwargs_train = {
        'network_query_fn': network_query_fn,
        'perturb': args.perturb,
        'N_importance': args.N_importance,
        'network_fine': model_fine,
        'N_samples': args.N_samples,
        'network_fn': model,
        'use_viewdirs': args.use_viewdirs,
        'white_bkgd': args.white_bkgd,
        'raw_noise_std': args.raw_noise_std,
        'network_type': args.network_type,
    }
    
    # For D-NeRF data, we don't use NDC
    render_kwargs_train['ndc'] = False
    render_kwargs_train['lindisp'] = args.lindisp
    
    render_kwargs_test = {k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.
    
    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


def save_checkpoint(path, global_step, network_fn, network_fine, optimizer, args):
    """Save model checkpoint.
    
    Args:
        path: path to save checkpoint
        global_step: current training step
        network_fn: coarse network
        network_fine: fine network (can be None)
        optimizer: optimizer
        args: training arguments
    """
    save_dict = {
        'global_step': global_step,
        'network_fn_state_dict': network_fn.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'args': vars(args),
    }
    
    if network_fine is not None:
        save_dict['network_fine_state_dict'] = network_fine.state_dict()
    
    torch.save(save_dict, path)
    print(f'Saved checkpoint at {path}')


def load_checkpoint(path, model, model_fine=None, optimizer=None):
    """Load model checkpoint.
    
    Args:
        path: path to checkpoint
        model: coarse network
        model_fine: fine network (optional)
        optimizer: optimizer (optional)
    
    Returns:
        global_step: training step from checkpoint
        args: training arguments from checkpoint
    """
    ckpt = torch.load(path, map_location=device)
    
    model.load_state_dict(ckpt['network_fn_state_dict'])
    
    if model_fine is not None and 'network_fine_state_dict' in ckpt:
        model_fine.load_state_dict(ckpt['network_fine_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    
    global_step = ckpt.get('global_step', 0)
    args = ckpt.get('args', {})
    
    return global_step, args


def load_official_dnerf_weights(model, ckpt_path):
    """Load official D-NeRF pre-trained weights.
    
    Since our DNeRF_Deformation now uses the EXACT SAME architecture as official D-NeRF
    (with _occ, _time, _time_out naming), we can load weights directly.
    
    Official checkpoint structure:
    - _occ.pts_linears.{0-7}.{weight,bias}
    - _occ.views_linears.0.{weight,bias}
    - _occ.feature_linear.{weight,bias}
    - _occ.alpha_linear.{weight,bias}
    - _occ.rgb_linear.{weight,bias}
    - _time.{0-7}.{weight,bias}
    - _time_out.{weight,bias}
    
    Args:
        model: DNeRF_Deformation model (architecture matches official D-NeRF)
        ckpt_path: Path to official D-NeRF checkpoint
    
    Returns:
        True if load successful, False otherwise
    """
    import sys
    try:
        ckpt = torch.load(ckpt_path, map_location=device)
        print(f"Loading official D-NeRF weights from {ckpt_path}")
        
        # Extract official model state
        if 'network_fn_state_dict' in ckpt:
            official_state = ckpt['network_fn_state_dict']
        else:
            official_state = ckpt
        
        # Our architecture now matches official exactly, so direct load should work
        # The only difference is our canonical network is named _occ (same as official)
        # and deformation network is _time/_time_out (same as official)
        
        model_state = model.state_dict()
        loaded_count = 0
        
        for key in official_state.keys():
            if key in model_state:
                if model_state[key].shape == official_state[key].shape:
                    model_state[key] = official_state[key]
                    loaded_count += 1
                else:
                    print(f"  Shape mismatch for {key}: "
                          f"model {model_state[key].shape} vs ckpt {official_state[key].shape}")
            else:
                print(f"  Key not found in model: {key}")
        
        model.load_state_dict(model_state, strict=False)
        
        print(f"\n✓ Successfully loaded {loaded_count}/{len(official_state)} weights from official D-NeRF")
        
        return True
        
    except Exception as e:
        print(f"Error loading official D-NeRF weights: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return False
        return False
