"""
D-NeRF Dataset Loader

This module loads the D-NeRF dataset which contains dynamic scenes with
time-stamped images and camera poses.

Dataset structure:
    data/
        <scene_name>/
            train/          # Training images
            val/            # Validation images
            test/           # Test images
            transforms_train.json
            transforms_val.json
            transforms_test.json
"""

import os
import json
import numpy as np
import imageio.v2 as imageio
import torch
import cv2


trans_t = lambda t: torch.Tensor([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, t],
    [0, 0, 0, 1]
]).float()

rot_phi = lambda phi: torch.Tensor([
    [1, 0, 0, 0],
    [0, np.cos(phi), -np.sin(phi), 0],
    [0, np.sin(phi), np.cos(phi), 0],
    [0, 0, 0, 1]
]).float()

rot_theta = lambda th: torch.Tensor([
    [np.cos(th), 0, -np.sin(th), 0],
    [0, 1, 0, 0],
    [np.sin(th), 0, np.cos(th), 0],
    [0, 0, 0, 1]
]).float()


def pose_spherical(theta, phi, radius):
    """Generate a spherical camera pose.
    
    Args:
        theta: azimuth angle in degrees
        phi: elevation angle in degrees
        radius: distance from origin
    
    Returns:
        c2w: [4, 4] camera-to-world transformation matrix
    """
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180. * np.pi) @ c2w
    c2w = rot_theta(theta / 180. * np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])) @ c2w
    return c2w


def load_dnerf_data(basedir, half_res=False, testskip=1):
    """Load D-NeRF dataset.
    
    Args:
        basedir: path to dataset directory
        half_res: if True, load at half resolution
        testskip: skip every N-th image in test/val sets
    
    Returns:
        images: [N, H, W, 3] RGB images (float32, 0-1)
        poses: [N, 4, 4] camera poses
        times: [N] normalized time values (0-1)
        render_poses: [N, 4, 4] poses for rendering
        render_times: [N] times for rendering
        hwf: [H, W, focal] image dimensions and focal length
        i_split: [i_train, i_val, i_test] index arrays for data splits
    """
    splits = ['train', 'val', 'test']
    metas = {}
    
    for s in splits:
        json_path = os.path.join(basedir, f'transforms_{s}.json')
        with open(json_path, 'r') as fp:
            metas[s] = json.load(fp)
    
    all_imgs = []
    all_poses = []
    all_times = []
    counts = [0]
    
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        times = []
        
        if s == 'train' or testskip == 0:
            skip = 1
        else:
            skip = testskip
        
        for frame in meta['frames'][::skip]:
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame['transform_matrix']))
            
            # Get time value (normalized to 0-1)
            time_val = frame.get('time', 0.0)
            times.append(time_val)
        
        imgs = (np.array(imgs) / 255.).astype(np.float32)
        poses = np.array(poses).astype(np.float32)
        times = np.array(times).astype(np.float32)
        
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)
        all_times.append(times)
    
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]
    
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    times = np.concatenate(all_times, 0)
    
    H, W = imgs[0].shape[:2]
    camera_angle_x = float(metas['train']['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    
    # Generate render poses for 360 video (at multiple times)
    render_poses = torch.stack([
        pose_spherical(angle, -30.0, 4.0) 
        for angle in np.linspace(-180, 180, 40 + 1)[:-1]
    ], 0)
    
    # For dynamic scenes, we'll render at different times too
    # Default: render at time 0.5 (middle of sequence)
    render_times = torch.ones(render_poses.shape[0]) * 0.5
    
    # Handle RGBA images - composite onto white background
    if imgs.shape[-1] == 4:
        imgs = imgs[..., :3] * imgs[..., -1:] + (1. - imgs[..., -1:])
    
    if half_res:
        H = H // 2
        W = W // 2
        focal = focal / 2.
        
        imgs_half_res = np.zeros((imgs.shape[0], H, W, 3), dtype=np.float32)
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
    
    return imgs, poses, times, render_poses, render_times, [H, W, focal], i_split


def load_dnerf_data_for_video(basedir, half_res=False, n_frames=120, n_rotations=1):
    """Load D-NeRF data and generate poses for 360 video rendering.
    
    Args:
        basedir: path to dataset directory
        half_res: if True, load at half resolution
        n_frames: number of frames in video
        n_rotations: number of full rotations around object
    
    Returns:
        render_poses: [N, 4, 4] poses for 360 video
        render_times: [N] time values (cycling through dynamic sequence)
        hwf: [H, W, focal]
        K: [3, 3] camera intrinsic matrix
    """
    json_path = os.path.join(basedir, 'transforms_train.json')
    with open(json_path, 'r') as fp:
        meta = json.load(fp)
    
    # Get image dimensions
    first_frame = meta['frames'][0]
    fname = os.path.join(basedir, first_frame['file_path'] + '.png')
    img = imageio.imread(fname)
    H, W = img.shape[:2]
    
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    
    if half_res:
        H = H // 2
        W = W // 2
        focal = focal / 2.
    
    # Generate render poses
    angles = np.linspace(-180, 180 * (2 * n_rotations - 1) / (2 * n_rotations), n_frames)
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in angles], 0)
    
    # Generate time values (cycle through time as camera rotates)
    render_times = torch.linspace(0, 1, n_frames)
    
    # Camera intrinsics
    K = np.array([
        [focal, 0, W / 2.],
        [0, focal, H / 2.],
        [0, 0, 1]
    ])
    
    return render_poses, render_times, [H, W, focal], K


def get_360_poses(n_frames=40, phi=-30.0, radius=4.0):
    """Generate camera poses for 360 rotation around object.
    
    Args:
        n_frames: number of frames
        phi: elevation angle in degrees
        radius: distance from origin
    
    Returns:
        poses: [N, 4, 4] camera poses
    """
    poses = torch.stack([
        pose_spherical(angle, phi, radius)
        for angle in np.linspace(-180, 180, n_frames + 1)[:-1]
    ], 0)
    return poses
