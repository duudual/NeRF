"""
Dynamic NeRF (D-NeRF) Implementation
two approaches:
1. Straightforward 6D Extension: MLP with (x, y, z, t, θ, φ) input
2. Deformation Network: Canonical network + Deformation network
"""

from .network import (
    NeRF,
    DNeRF_Straightforward,
    DNeRF_Deformation,
)

from .model import (
    create_dnerf,
    save_checkpoint,
    load_checkpoint,
)

from .render import (
    render_dnerf,
    render_path_dnerf,
    run_network_dnerf,
)

from .load_dnerf import (
    load_dnerf_data,
    load_dnerf_data_for_video,
    get_360_poses,
)

from .utils import (
    device,
    compute_psnr,
    compute_ssim,
    compute_lpips,
    evaluate_images,
)

from .positional_encoding import (
    get_embedder,
    get_time_embedder,
)

from .config import config_parser

__all__ = [
    # Networks
    'NeRF',
    'DNeRF_Straightforward',
    'DNeRF_Deformation',
    
    # Model management
    'create_dnerf',
    'save_checkpoint',
    'load_checkpoint',
    
    # Rendering
    'render_dnerf',
    'render_path_dnerf',
    'run_network_dnerf',
    
    # Data loading
    'load_dnerf_data',
    'load_dnerf_data_for_video',
    'get_360_poses',
    
    # Utilities
    'device',
    'compute_psnr',
    'compute_ssim',
    'compute_lpips',
    'evaluate_images',
    
    # Encoding
    'get_embedder',
    'get_time_embedder',
    
    # Config
    'config_parser',
]
