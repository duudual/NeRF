"""
Dynamic NeRF Rendering

This module implements volumetric rendering for dynamic scenes,
extending the standard NeRF rendering pipeline to handle time-varying content.
"""

import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import imageio
from tqdm import tqdm

# Handle imports for both module and standalone execution
try:
    from .utils import DEBUG, to8b, device
except ImportError:
    from utils import DEBUG, to8b, device


def batchify(fn, chunk, is_dynamic=True):
    """Constructs a version of 'fn' that applies to smaller batches.
    
    Args:
        fn: function to batchify
        chunk: chunk size (None for no batching)
        is_dynamic: if True, fn is a dynamic network that returns (output, dx)
    
    Returns:
        batched function
    """
    if chunk is None:
        return fn
    
    def ret(inputs):
        results = [fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)]
        if is_dynamic:
            outputs = torch.cat([r[0] for r in results], 0)
            dxs = torch.cat([r[1] for r in results], 0)
            return outputs, dxs
        else:
            return torch.cat(results, 0)
    
    return ret


def run_network_dnerf(inputs, viewdirs, times, fn, embed_fn, embeddirs_fn, 
                      embed_time_fn, netchunk=1024*64, network_type='deformation'):
    """Prepares inputs and applies dynamic network 'fn'.
    
    Args:
        inputs: [..., 3] input points
        viewdirs: [..., 3] viewing directions (can be None)
        times: [..., 1] time values for each point
        fn: network function
        embed_fn: positional encoding function for points
        embeddirs_fn: positional encoding function for directions
        embed_time_fn: positional encoding function for time
        netchunk: chunk size for network evaluation
        network_type: 'straightforward' or 'deformation'
    
    Returns:
        outputs: [..., output_ch] network outputs
        dx: [..., 3] deformations (for deformation network)
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    times_flat = torch.reshape(times, [-1, 1])
    
    # Store raw points for deformation network
    pts_raw = inputs_flat.clone()
    
    # Embed positions
    embedded = embed_fn(inputs_flat)
    
    # Embed view directions
    if viewdirs is not None:
        input_dirs = viewdirs[:, None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)
    
    # Embed time
    embedded_time = embed_time_fn(times_flat)
    
    # Apply network
    def network_fn(x):
        # Split embedded positions and directions
        if viewdirs is not None:
            split_size = embedded.shape[-1] - embedded_dirs.shape[-1]
            pts_embed = x[:, :split_size]
            views_embed = x[:, split_size:]
            x_combined = torch.cat([pts_embed, views_embed], -1)
        else:
            x_combined = x
            pts_embed = x
        
        # Get time embedding for this batch
        batch_size = x.shape[0]
        t_embed = embedded_time[:batch_size]
        
        if network_type == 'deformation':
            # For deformation network, we need raw points
            raw_pts = pts_raw[:batch_size]
            return fn(x_combined, t_embed, raw_pts)
        else:
            return fn(x_combined, t_embed)
    
    # Process in chunks
    all_outputs = []
    all_dx = []
    
    for i in range(0, embedded.shape[0], netchunk):
        chunk_embedded = embedded[i:i+netchunk]
        chunk_time = embedded_time[i:i+netchunk]
        chunk_raw = pts_raw[i:i+netchunk]
        
        if viewdirs is not None:
            # For view-dependent rendering
            if network_type == 'deformation':
                out, dx = fn(chunk_embedded, chunk_time, chunk_raw)
            else:
                out, dx = fn(chunk_embedded, chunk_time)
        else:
            if network_type == 'deformation':
                out, dx = fn(chunk_embedded, chunk_time, chunk_raw)
            else:
                out, dx = fn(chunk_embedded, chunk_time)
        
        all_outputs.append(out)
        all_dx.append(dx)
    
    outputs_flat = torch.cat(all_outputs, 0)
    dx_flat = torch.cat(all_dx, 0)
    
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    dx = torch.reshape(dx_flat, list(inputs.shape[:-1]) + [3])
    
    return outputs, dx


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    """Transforms model's predictions to semantically meaningful values.
    
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
        raw_noise_std: standard deviation of noise added to density
        white_bkgd: if True, assume white background
        pytest: if True, use fixed random numbers for testing
    
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1. - torch.exp(-act_fn(raw) * dists)

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).to(dists.device).expand(dists[..., :1].shape)], -1)

    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    rgb = torch.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[..., 3].shape, device=raw.device) * raw_noise_std
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[..., 3].shape)) * raw_noise_std
            noise = torch.Tensor(noise).to(raw.device)

    alpha = raw2alpha(raw[..., 3] + noise, dists)
    weights = alpha * torch.cumprod(
        torch.cat([torch.ones((alpha.shape[0], 1), device=alpha.device), 1. - alpha + 1e-10], -1), -1
    )[:, :-1]
    rgb_map = torch.sum(weights[..., None] * rgb, -2)

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1. - acc_map[..., None])

    return rgb_map, disp_map, acc_map, weights, depth_map


def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    """Hierarchical sampling (section 5.2 of NeRF paper)."""
    weights = weights + 1e-5
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)

    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    u = u.contiguous().to(cdf.device)
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


def render_rays_dnerf(ray_batch, time_val,
                      network_fn,
                      network_query_fn,
                      N_samples,
                      retraw=False,
                      lindisp=False,
                      perturb=0.,
                      N_importance=0,
                      network_fine=None,
                      white_bkgd=False,
                      raw_noise_std=0.,
                      verbose=False,
                      pytest=False,
                      network_type='deformation',
                      ndc=False,
                      use_viewdirs=False,
                      **kwargs):
    """Volumetric rendering for dynamic scenes.
    
    Args:
        ray_batch: [batch_size, ...] ray batch with origins, directions, bounds, viewdirs
        time_val: float, time value for this batch (0 to 1)
        network_fn: coarse network function
        network_query_fn: function for querying network
        N_samples: number of coarse samples per ray
        retraw: if True, include model's raw predictions
        lindisp: if True, sample linearly in inverse depth
        perturb: perturbation factor
        N_importance: number of fine samples per ray
        network_fine: fine network function
        white_bkgd: if True, assume white background
        raw_noise_std: std dev of noise added to regularize sigma
        verbose: if True, print debug info
        pytest: if True, use fixed random numbers
        network_type: 'straightforward' or 'deformation'
    
    Returns:
        ret: dictionary of results
    """
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]
    viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])
    near, far = bounds[..., 0], bounds[..., 1]

    t_vals = torch.linspace(0., 1., steps=N_samples, device=rays_o.device)
    if not lindisp:
        z_vals = near * (1. - t_vals) + far * t_vals
    else:
        z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * t_vals)

    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        t_rand = torch.rand(z_vals.shape, device=z_vals.device)
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand).to(z_vals.device)
        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
    
    # Create time tensor with same shape as points
    times = torch.ones_like(pts[..., :1]) * time_val

    raw, dx = network_query_fn(pts, viewdirs, times, network_fn)
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
        raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest
    )

    if N_importance > 0:
        rgb_map_0, disp_map_0, acc_map_0, dx_0 = rgb_map, disp_map, acc_map, dx

        z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1], N_importance, det=(perturb == 0.), pytest=pytest)
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
        times = torch.ones_like(pts[..., :1]) * time_val

        run_fn = network_fn if network_fine is None else network_fine
        raw, dx = network_query_fn(pts, viewdirs, times, run_fn)

        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
            raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest
        )

    ret = {'rgb_map': rgb_map, 'disp_map': disp_map, 'acc_map': acc_map, 'dx': dx}
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['dx0'] = dx_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)

    for k in ret:
        if isinstance(ret[k], torch.Tensor) and (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret


def batchify_rays_dnerf(rays_flat, time_val, chunk=1024*32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    
    Args:
        rays_flat: [N_rays, ...] flattened ray batch
        time_val: time value for this batch
        chunk: chunk size
        **kwargs: arguments to pass to render_rays_dnerf
    
    Returns:
        all_ret: dictionary of results
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays_dnerf(rays_flat[i:i+chunk], time_val, **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def get_rays(H, W, K, c2w):
    """Get ray origins and directions from camera parameters."""
    i, j = torch.meshgrid(
        torch.linspace(0, W-1, W, device=c2w.device), 
        torch.linspace(0, H-1, H, device=c2w.device),
        indexing='ij'
    )
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d


def render_dnerf(H, W, K, time_val, chunk=1024*32, rays=None, c2w=None, 
                 near=0., far=1., use_viewdirs=False, c2w_staticcam=None, **kwargs):
    """Render function for dynamic NeRF.
    
    Args:
        H: image height
        W: image width
        K: camera intrinsic matrix
        time_val: time value for rendering (0 to 1)
        chunk: chunk size for rendering
        rays: optional ray batch
        c2w: camera-to-world transformation
        near: near plane distance
        far: far plane distance
        use_viewdirs: whether to use view directions
        c2w_staticcam: optional static camera for view dirs
        **kwargs: additional arguments for render_rays_dnerf
    
    Returns:
        rgb_map: [H, W, 3] or [batch, 3] rendered RGB
        disp_map: [H, W] or [batch] disparity map
        acc_map: [H, W] or [batch] accumulation map
        extras: dict of additional outputs
    """
    if c2w is not None:
        rays_o, rays_d = get_rays(H, W, K, c2w)
    else:
        rays_o, rays_d = rays
    if use_viewdirs:
        viewdirs = rays_d
        if c2w_staticcam is not None:
            rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1, 3]).float()

    sh = rays_d.shape

    rays_o = torch.reshape(rays_o, [-1, 3]).float()
    rays_d = torch.reshape(rays_d, [-1, 3]).float()

    near_t, far_t = near * torch.ones_like(rays_d[..., :1]), far * torch.ones_like(rays_d[..., :1])
    rays = torch.cat([rays_o, rays_d, near_t, far_t], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    all_ret = batchify_rays_dnerf(rays, time_val, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def render_path_dnerf(render_poses, render_times, hwf, K, chunk, render_kwargs, 
                      gt_imgs=None, savedir=None, render_factor=0):
    """Render a path of poses with corresponding times.
    
    Args:
        render_poses: [N, 3, 4] camera poses
        render_times: [N] time values for each pose
        hwf: [H, W, focal]
        K: camera intrinsic matrix
        chunk: chunk size for rendering
        render_kwargs: render arguments
        gt_imgs: optional ground truth images
        savedir: optional save directory
        render_factor: downsampling factor
    
    Returns:
        rgbs: [N, H, W, 3] rendered images
        disps: [N, H, W] disparity maps
    """
    H, W, focal = hwf

    if render_factor != 0:
        H = H // render_factor
        W = W // render_factor
        focal = focal / render_factor
        K = K.copy() if isinstance(K, np.ndarray) else K.clone()
        K[0, 0] = focal
        K[1, 1] = focal
        K[0, 2] = W / 2.
        K[1, 2] = H / 2.

    rgbs = []
    disps = []

    t = time.time()
    for i, (c2w, time_val) in enumerate(tqdm(zip(render_poses, render_times))):
        if isinstance(time_val, torch.Tensor):
            time_val = time_val.item()
        
        rgb, disp, acc, extras = render_dnerf(
            H, W, K, time_val, chunk=chunk, c2w=c2w[:3, :4], **render_kwargs
        )
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())
        
        if i == 0:
            print(f"Render time: {time.time() - t:.2f}s")
            print(f"RGB shape: {rgb.shape}, Disp shape: {disp.shape}")
        
        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)

    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps
