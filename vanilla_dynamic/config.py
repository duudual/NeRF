
import configargparse


def config_parser():
    """Create argument parser for Dynamic NeRF training and rendering.
    
    Returns:
        parser: ArgumentParser instance
    """
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument("--expname", type=str, 
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/', 
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, 
                        default='../../D_NeRF_Dataset/data/bouncingballs',
                        help='input data directory')

    # Network type selection
    parser.add_argument("--network_type", type=str, default='deformation',
                        choices=['straightforward', 'deformation'],
                        help='Type of dynamic NeRF: straightforward (6D) or deformation')
    
    # Network architecture options
    parser.add_argument("--netdepth", type=int, default=8, 
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256, 
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8, 
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256, 
                        help='channels per layer in fine network')
    
    # Deformation network architecture (only used for deformation type)
    parser.add_argument("--netdepth_deform", type=int, default=6,
                        help='layers in deformation network')
    parser.add_argument("--netwidth_deform", type=int, default=128,
                        help='channels per layer in deformation network')
    parser.add_argument("--zero_canonical", action='store_true', default=True,
                        help='use zero deformation at canonical time (t=0)')
    
    # Training options
    parser.add_argument("--N_rand", type=int, default=1024,
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4, 
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250, 
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*32, 
                        help='number of rays processed in parallel')
    parser.add_argument("--netchunk", type=int, default=1024*64, 
                        help='number of pts sent through network in parallel')
    parser.add_argument("--no_batching", action='store_true', 
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true', 
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None, 
                        help='specific weights file to reload')
    parser.add_argument("--N_iters", type=int, default=200000,
                        help='number of training iterations')

    # Rendering options
    parser.add_argument("--N_samples", type=int, default=64, 
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=128,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true', default=True,
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0, 
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10, 
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4, 
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--multires_time", type=int, default=10,
                        help='log2 of max freq for positional encoding (time)')
    parser.add_argument("--no_include_input", action='store_true',
                        help='do not include raw input in positional encoding (may help with numerical stability)')
    parser.add_argument("--raw_noise_std", type=float, default=0., 
                        help='std dev of noise added to regularize sigma')

    parser.add_argument("--render_only", action='store_true', 
                        help='do not optimize, reload weights and render')
    parser.add_argument("--render_test", action='store_true', 
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0, 
                        help='downsampling factor for rendering')
    parser.add_argument("--render_time", type=float, default=0.5,
                        help='time value for rendering (0-1)')

    # Training options (additional)
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float, default=.5,
                        help='fraction of img taken for central crops')

    # Dataset options
    parser.add_argument("--testskip", type=int, default=1, 
                        help='will load 1/N images from test/val sets')
    parser.add_argument("--half_res", action='store_true', 
                        help='load data at 400x400 instead of 800x800')
    parser.add_argument("--white_bkgd", action='store_true', default=True,
                        help='render on white background (D-NeRF default)')
    parser.add_argument("--lindisp", action='store_true',
                        help='sample linearly in disparity rather than depth')

    # Loss options
    parser.add_argument("--deform_reg_weight", type=float, default=0.0,
                        help='weight for deformation regularization loss')
    
    # Gradient clipping (to prevent numerical explosion)
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help='gradient clipping max norm (0 to disable)')
    
    # Official weights loading
    parser.add_argument("--load_official_weights", action='store_true', 
                        help='load official D-NeRF pre-trained weights')
    parser.add_argument("--official_ckpt_path", type=str, default=None,
                        help='path to official D-NeRF checkpoint')

    # Logging/saving options
    parser.add_argument("--i_print", type=int, default = 500, 
                        help='frequency of console printout')
    parser.add_argument("--i_img", type=int, default=500, 
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000, 
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=50000, 
                        help='frequency of testset saving')
    parser.add_argument("--i_video", type=int, default=50000, 
                        help='frequency of render_poses video saving')

    # 360 Video options
    parser.add_argument("--video_n_frames", type=int, default=120,
                        help='number of frames in 360 video')
    parser.add_argument("--video_fps", type=int, default=30,
                        help='frames per second for video')

    return parser
