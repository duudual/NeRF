import os
import numpy as np
import torch
import imageio
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import device, img2mse, mse2psnr, to8b
from rays import get_rays, get_rays_np
from render import render, render_path
from model import create_nerf
from config import config_parser
from load_llff import load_llff_data
from load_blender import load_blender_data

def evaluate():
    """Evaluate a pre-trained NeRF model on the test set using PSNR metric."""
    parser = config_parser()
    args = parser.parse_args()
    
    # Set up output directory
    output_dir = os.path.join(args.basedir, args.expname, 'evaluation')
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    K = None
    if args.dataset_type == 'llff':
        images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, args.factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=args.spherify)
        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                        (i not in i_test and i not in i_val)])

        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.
            
        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)

    elif args.dataset_type == 'blender':
        images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res, args.testskip)
        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near = 2.
        far = 6.

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return
    
    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if K is None:
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])
    
    # Load the pre-trained model
    print('Loading pre-trained model...')
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    
    # Check if model was loaded successfully
    if start == 0 and len([f for f in os.listdir(os.path.join(args.basedir, args.expname)) if 'tar' in f] if os.path.exists(os.path.join(args.basedir, args.expname)) else []) == 0:
        print('ERROR: No checkpoints found!')
        print(f'Please specify a checkpoint with --ft_path or train the model first')
        return
    
    print(f"Model loaded successfully. Start step: {start}")
    
    # Set up bounds
    bds_dict = {
        'near' : near,
        'far' : far,
    }
    render_kwargs_test.update(bds_dict)
    
    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).to(device)
    poses = torch.Tensor(poses).to(device)
    images = torch.Tensor(images).to(device)
    
    # Render test images and compute PSNR
    print('Rendering test images...')
    test_poses = poses[i_test]
    test_gt = images[i_test]
    
    with torch.no_grad():
        rgbs, disps = render_path(test_poses, hwf, K, args.chunk, render_kwargs_test, gt_imgs=test_gt, savedir=None, render_factor=args.render_factor)
    
    # Compute PSNR for each image
    psnr_values = []
    for i, (rgb, gt) in enumerate(zip(rgbs, test_gt.cpu().numpy())):
        # Ensure both images are in [0, 1] range
        rgb = np.clip(rgb, 0, 1)
        gt = np.clip(gt, 0, 1)
        
        # Compute MSE and PSNR
        mse = img2mse(torch.Tensor(rgb).to(device), torch.Tensor(gt).to(device)).item()
        psnr = mse2psnr(torch.Tensor([mse]).to(device)).item()
        psnr_values.append(psnr)
        
        print(f'Test image {i+1}/{len(psnr_values)}: PSNR = {psnr:.2f} dB')
    
    # Compute average PSNR
    avg_psnr = np.mean(psnr_values)
    print(f'\nAverage PSNR over test set: {avg_psnr:.2f} dB')
    
    # Save PSNR values to file
    psnr_file = os.path.join(output_dir, 'psnr_values.txt')
    with open(psnr_file, 'w') as f:
        f.write(f'Average PSNR: {avg_psnr:.2f} dB\n\n')
        for i, psnr in enumerate(psnr_values):
            f.write(f'Image {i+1}: {psnr:.2f} dB\n')
    
    print(f'PSNR values saved to {psnr_file}')
    
    # Visualization
    print('Creating visualizations...')
    
    # 1. Save rendered test images
    rendered_dir = os.path.join(output_dir, 'rendered_test_images')
    os.makedirs(rendered_dir, exist_ok=True)
    
    for i, rgb in enumerate(rgbs):
        rgb8 = to8b(rgb)
        imageio.imwrite(os.path.join(rendered_dir, f'r_{i}.png'), rgb8)
    
    # 2. Create PSNR histogram
    plt.figure(figsize=(10, 6))
    plt.hist(psnr_values, bins=20, edgecolor='black')
    plt.xlabel('PSNR (dB)')
    plt.ylabel('Frequency')
    plt.title(f'Test Set PSNR Distribution (Avg: {avg_psnr:.2f} dB)')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'psnr_histogram.png'))
    
    # 3. Create comparison grid for first few images
    num_comparisons = min(5, len(rgbs))
    fig, axes = plt.subplots(num_comparisons, 2, figsize=(12, 3*num_comparisons))
    
    for i in range(num_comparisons):
        # Rendered image
        axes[i, 0].imshow(rgbs[i])
        axes[i, 0].set_title(f'Rendered (PSNR: {psnr_values[i]:.2f} dB)')
        axes[i, 0].axis('off')
        
        # Ground truth image
        axes[i, 1].imshow(test_gt[i].cpu().numpy())
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_grid.png'))
    
    # 4. Save all rendered images as a video
    if len(rgbs) > 1:
        video_path = os.path.join(output_dir, 'test_render.mp4')
        imageio.mimwrite(video_path, to8b(rgbs), fps=10, quality=8)
        print(f'Test render video saved to {video_path}')
    
    plt.close('all')
    print(f'All evaluations completed! Results saved to {output_dir}')

if __name__ == '__main__':
    np.random.seed(0)
    if torch.cuda.is_available():
        torch.set_default_device('cuda')
    torch.set_default_dtype(torch.float32)
    evaluate()
