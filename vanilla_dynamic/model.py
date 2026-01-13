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
        # Deformation network approach
        model = DNeRF_Deformation(
            D=args.netdepth, W=args.netwidth,
            D_deform=args.netdepth_deform, W_deform=args.netwidth_deform,
            input_ch=input_ch, output_ch=output_ch, skips=skips,
            skips_deform=skips_deform,
            input_ch_views=input_ch_views, input_ch_time=input_ch_time,
            use_viewdirs=args.use_viewdirs,
            zero_canonical=args.zero_canonical
        ).to(device)
        model.embed_fn = embed_fn  # type: ignore
        
        model_fine = None
        if args.N_importance > 0:
            model_fine = DNeRF_Deformation(
                D=args.netdepth_fine, W=args.netwidth_fine,
                D_deform=args.netdepth_deform, W_deform=args.netwidth_deform,
                input_ch=input_ch, output_ch=output_ch, skips=skips,
                skips_deform=skips_deform,
                input_ch_views=input_ch_views, input_ch_time=input_ch_time,
                use_viewdirs=args.use_viewdirs,
                zero_canonical=args.zero_canonical
            ).to(device)
            model_fine.embed_fn = embed_fn  # type: ignore
    
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
        ckpts = [args.ft_path]
    else:
        ckpt_dir = os.path.join(basedir, expname)
        if os.path.exists(ckpt_dir):
            ckpts = [os.path.join(ckpt_dir, f) for f in sorted(os.listdir(ckpt_dir)) if 'tar' in f]
        else:
            ckpts = []
            print(f'Warning: Checkpoint directory {ckpt_dir} does not exist.')
    
    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=device)
        
        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        
        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None and 'network_fine_state_dict' in ckpt:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])
    
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
