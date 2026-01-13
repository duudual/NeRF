"""
Dynamic NeRF Training Script

Usage:
    python train.py --config configs/bouncingballs_deform.txt
    python train.py --datadir ../../D_NeRF_Dataset/data/bouncingballs --network_type straightforward
"""

import os
import sys
import time
import numpy as np
import torch
import imageio
from tqdm import tqdm, trange

# Handle imports for both module and standalone execution
try:
    from .utils import device, img2mse, mse2psnr, to8b, get_rays, get_rays_np
    from .render import render_dnerf, render_path_dnerf
    from .model import create_dnerf, save_checkpoint
    from .config import config_parser
    from .load_dnerf import load_dnerf_data
except ImportError:
    from utils import device, img2mse, mse2psnr, to8b, get_rays, get_rays_np
    from render import render_dnerf, render_path_dnerf
    from model import create_dnerf, save_checkpoint
    from config import config_parser
    from load_dnerf import load_dnerf_data


def train():
    """Main training function for Dynamic NeRF."""
    parser = config_parser()
    args = parser.parse_args()
    
    print("=" * 60)
    print(f"Dynamic NeRF Training")
    print(f"Network type: {args.network_type}")
    print(f"Data directory: {args.datadir}")
    print("=" * 60)

    # Load D-NeRF data
    images, poses, times, render_poses, render_times, hwf, i_split = load_dnerf_data(
        args.datadir, args.half_res, args.testskip
    )
    print(f'Loaded D-NeRF data: {images.shape}, {poses.shape}, {times.shape}')
    
    i_train, i_val, i_test = i_split
    
    # Set near/far bounds for D-NeRF (similar to blender dataset)
    near = 2.
    far = 6.

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]
    
    # Camera intrinsic matrix
    K = np.array([
        [focal, 0, 0.5 * W],
        [0, focal, 0.5 * H],
        [0, 0, 1]
    ])

    # Create log dir and copy config file
    basedir = args.basedir
    expname = args.expname if args.expname else f"dnerf_{args.network_type}"
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    
    # Save args
    args_file = os.path.join(basedir, expname, 'args.txt')
    with open(args_file, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write(f'{arg} = {attr}\n')
    
    if args.config is not None:
        config_file = os.path.join(basedir, expname, 'config.txt')
        with open(config_file, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create Dynamic NeRF model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_dnerf(args)
    global_step = start
    
    # Update render kwargs with bounds
    bds_dict = {'near': near, 'far': far}
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)
    
    # Move data to GPU
    render_poses = render_poses.to(device)
    render_times = render_times.to(device)

    # Render only mode
    if args.render_only:
        print('RENDER ONLY')
        with torch.no_grad():
            if args.render_test:
                # Render test set
                test_times = torch.from_numpy(times[i_test]).to(device)
                test_poses = torch.from_numpy(poses[i_test]).to(device)
                testsavedir = os.path.join(basedir, expname, f'testset_{start:06d}')
            else:
                # Render 360 path
                test_times = render_times
                test_poses = render_poses
                testsavedir = os.path.join(basedir, expname, f'renderonly_{start:06d}')
            
            os.makedirs(testsavedir, exist_ok=True)
            print(f'Rendering to {testsavedir}')
            
            rgbs, _ = render_path_dnerf(
                test_poses, test_times, hwf, K, args.chunk, 
                render_kwargs_test, savedir=testsavedir
            )
            
            print('Done rendering')
            video_path = os.path.join(testsavedir, 'video.mp4')
            imageio.mimwrite(video_path, to8b(rgbs), fps=args.video_fps, quality=8)
            print(f'Saved video to {video_path}')
        return

    # Prepare training data
    N_rand = args.N_rand
    use_batching = not args.no_batching
    
    if use_batching:
        # Precompute rays for all training images
        print('Precomputing rays for batched training...')
        rays = np.stack([get_rays_np(H, W, K, p[:3, :4]) for p in poses[i_train]], 0)
        rays_rgb = np.concatenate([rays, images[i_train, None]], 1)
        rays_rgb = np.transpose(rays_rgb, [0, 2, 3, 1, 4])
        
        # Add time information
        train_times = times[i_train]
        # Broadcast time to match rays_rgb shape: [N_train, H, W, 3, 1]
        rays_time = train_times[:, None, None, None, None]  # [N_train, 1, 1, 1, 1]
        rays_time = np.broadcast_to(rays_time, [len(i_train), H, W, 3, 1])
        rays_rgb = np.concatenate([rays_rgb, rays_time], -1)  # [N_train, H, W, 3, 4]
        
        rays_rgb = rays_rgb.reshape(-1, 3, 4)  # [N, ro+rd+rgb, 4] (last dim: x,y,z,t)
        rays_rgb = rays_rgb.astype(np.float32)
        
        print('Shuffling rays...')
        np.random.shuffle(rays_rgb)
        print(f'Rays shape: {rays_rgb.shape}')
        
        i_batch = 0
        rays_rgb = torch.Tensor(rays_rgb).to(device)
        
    images_tensor = torch.Tensor(images).to(device)
    poses_tensor = torch.Tensor(poses).to(device)
    times_tensor = torch.Tensor(times).to(device)

    # Training loop
    N_iters = args.N_iters + 1
    print('Begin training')
    print(f'TRAIN views: {i_train}')
    print(f'TEST views: {i_test}')
    print(f'VAL views: {i_val}')
    
    start = start + 1
    for i in trange(start, N_iters):
        time0 = time.time()
        
        if use_batching:
            # Random batch of rays from all images
            batch = rays_rgb[i_batch:i_batch + N_rand]
            batch = torch.transpose(batch, 0, 1)  # [3, N_rand, 4] -> (ro, rd, rgb), each with (x,y,z,t)
            batch_rays = batch[:2, :, :3]  # [2, N_rand, 3] - only xyz for rays
            target_s = batch[2, :, :3]     # [N_rand, 3] - rgb values
            batch_times = batch[0, :, 3]   # [N_rand] - time from rays_o (same for all)
            
            i_batch += N_rand
            if i_batch >= rays_rgb.shape[0]:
                print("Shuffle data after an epoch!")
                rand_idx = torch.randperm(rays_rgb.shape[0])
                rays_rgb = rays_rgb[rand_idx]
                i_batch = 0
        else:
            # Random rays from one image
            img_i = np.random.choice(i_train)
            target = images_tensor[img_i]
            pose = poses_tensor[img_i, :3, :4]
            time_val = times_tensor[img_i].item()
            
            rays_o, rays_d = get_rays(H, W, K, pose)
            
            if i < args.precrop_iters:
                dH = int(H // 2 * args.precrop_frac)
                dW = int(W // 2 * args.precrop_frac)
                coords = torch.stack(
                    torch.meshgrid(
                        torch.linspace(H // 2 - dH, H // 2 + dH - 1, 2 * dH),
                        torch.linspace(W // 2 - dW, W // 2 + dW - 1, 2 * dW),
                        indexing='ij'
                    ), -1
                )
                if i == start:
                    print(f"[Config] Center cropping {2*dH} x {2*dW} until iter {args.precrop_iters}")
            else:
                coords = torch.stack(
                    torch.meshgrid(
                        torch.linspace(0, H - 1, H),
                        torch.linspace(0, W - 1, W),
                        indexing='ij'
                    ), -1
                )
            
            coords = torch.reshape(coords, [-1, 2])
            select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)
            select_coords = coords[select_inds].long()
            
            rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]
            rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]
            batch_rays = torch.stack([rays_o, rays_d], 0)
            target_s = target[select_coords[:, 0], select_coords[:, 1]]
            batch_times = time_val

        # Render
        rgb, disp, acc, extras = render_dnerf(
            H, W, K, batch_times if not use_batching else batch_times.mean().item(),
            chunk=args.chunk, rays=batch_rays,
            verbose=i < 10, retraw=True,
            **render_kwargs_train
        )

        # Compute losses
        optimizer.zero_grad()
        img_loss = img2mse(rgb, target_s)
        loss = img_loss
        psnr = mse2psnr(img_loss)
        
        # Deformation regularization loss
        if args.deform_reg_weight > 0 and 'dx' in extras:
            dx_loss = torch.mean(extras['dx'] ** 2)
            loss = loss + args.deform_reg_weight * dx_loss
        
        # Fine network loss
        if 'rgb0' in extras:
            img_loss0 = img2mse(extras['rgb0'], target_s)
            loss = loss + img_loss0
            psnr0 = mse2psnr(img_loss0)
        
        loss.backward()
        optimizer.step()

        # Update learning rate
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate

        dt = time.time() - time0

        # Logging
        if i % args.i_print == 0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item():.5f} PSNR: {psnr.item():.2f} LR: {new_lrate:.2e}")

        # Save checkpoint
        if i % args.i_weights == 0:
            path = os.path.join(basedir, expname, f'{i:06d}.tar')
            save_checkpoint(
                path, global_step,
                render_kwargs_train['network_fn'],
                render_kwargs_train['network_fine'],
                optimizer, args
            )

        # Render test video
        if i % args.i_video == 0 and i > 0:
            with torch.no_grad():
                rgbs, disps = render_path_dnerf(
                    render_poses, render_times, hwf, K, args.chunk, render_kwargs_test
                )
            print(f'Done, saving video {rgbs.shape}')
            moviebase = os.path.join(basedir, expname, f'{expname}_spiral_{i:06d}_')
            imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=args.video_fps, quality=8)
            imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=args.video_fps, quality=8)

        # Render test set
        if i % args.i_testset == 0 and i > 0:
            testsavedir = os.path.join(basedir, expname, f'testset_{i:06d}')
            os.makedirs(testsavedir, exist_ok=True)
            print(f'Rendering test set to {testsavedir}')
            
            with torch.no_grad():
                test_poses = torch.Tensor(poses[i_test]).to(device)
                test_times = torch.Tensor(times[i_test]).to(device)
                render_path_dnerf(
                    test_poses, test_times, hwf, K, args.chunk,
                    render_kwargs_test, gt_imgs=images[i_test], savedir=testsavedir
                )
            print('Saved test set')

        global_step += 1


if __name__ == '__main__':
    train()
