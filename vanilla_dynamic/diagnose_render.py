"""
Detailed diagnosis of deformation rendering
Step by step check of the full rendering pipeline
"""
import torch
import numpy as np
import os
import sys

# Add current dir to path
sys.path.insert(0, '.')

from network import DNeRF_Deformation
from positional_encoding import get_embedder, get_time_embedder
from utils import device, to8b
from load_dnerf import load_dnerf_data
from render import render_dnerf, raw2outputs
import imageio

print("=" * 60)
print("STEP 1: Load checkpoint")
print("=" * 60)

ckpt_path = "./test_deform/best.tar"
ckpt = torch.load(ckpt_path, map_location=device)
print(f"Checkpoint loaded from: {ckpt_path}")
print(f"Global step: {ckpt['global_step']}")
print(f"Keys in checkpoint: {list(ckpt.keys())}")

# Check state dict keys
state_dict = ckpt['network_fn_state_dict']
print(f"\nNetwork state dict keys (first 10):")
for i, key in enumerate(list(state_dict.keys())[:10]):
    print(f"  {key}: {state_dict[key].shape}")

print("\n" + "=" * 60)
print("STEP 2: Create network with correct config")
print("=" * 60)

# Get config from args if available
if 'args' in ckpt:
    args = ckpt['args']
    print(f"Args from checkpoint: {args}")
else:
    print("No args in checkpoint, using default config")

# Create embedders - use default config
embed_fn, input_ch = get_embedder(10, 0)  # multires=10
embeddirs_fn, input_ch_views = get_embedder(4, 0)  # multires_views=4
embed_time_fn, input_ch_time = get_time_embedder(10)  # multires_time=10

print(f"Embedding dimensions: pos={input_ch}, views={input_ch_views}, time={input_ch_time}")

# Create network matching checkpoint structure
model = DNeRF_Deformation(
    D=8, W=256,
    D_deform=8, W_deform=256,
    input_ch=input_ch,
    input_ch_views=input_ch_views,
    input_ch_time=input_ch_time,
    output_ch=4,
    skips=[4],
    use_viewdirs=True,
    zero_canonical=True
).to(device)
model.embed_fn = embed_fn

# Load weights
model.load_state_dict(state_dict)
model.eval()
print("Model loaded successfully")

print("\n" + "=" * 60)
print("STEP 3: Load data")
print("=" * 60)

datadir = "D:/lecture/2.0_xk/CV/finalproject/D_NeRF_Dataset/data/bouncingballs"
images, poses, times, render_poses, render_times, hwf, i_split = load_dnerf_data(
    datadir, half_res=True, testskip=1
)
H, W, focal = hwf
H, W = int(H), int(W)
K = np.array([
    [focal, 0, 0.5 * W],
    [0, focal, 0.5 * H],
    [0, 0, 1]
])
print(f"Image size: {H}x{W}, focal: {focal}")
print(f"Render poses shape: {render_poses.shape}")

print("\n" + "=" * 60)
print("STEP 4: Test single ray rendering")
print("=" * 60)

# Get one test pose
pose_np = np.array(render_poses[0])
time_val = 0.5

# Generate rays using numpy version
from utils import get_rays_np
rays_o_np, rays_d_np = get_rays_np(H, W, K, pose_np[:3, :4])
rays_o = torch.Tensor(rays_o_np).to(device)
rays_d = torch.Tensor(rays_d_np).to(device)
print(f"Rays origin shape: {rays_o.shape}")
print(f"Rays direction shape: {rays_d.shape}")

# Take center pixel
center_h, center_w = H // 2, W // 2
ray_o = rays_o[center_h, center_w]  # [3]
ray_d = rays_d[center_h, center_w]  # [3]
print(f"Center ray origin: {ray_o}")
print(f"Center ray direction: {ray_d}")

print("\n" + "=" * 60)
print("STEP 5: Sample points along ray")
print("=" * 60)

near, far = 2., 6.
N_samples = 64
t_vals = torch.linspace(0., 1., N_samples, device=device)
z_vals = near * (1. - t_vals) + far * t_vals
print(f"Z values range: [{z_vals[0]:.3f}, {z_vals[-1]:.3f}]")

# Compute 3D points
pts = ray_o[None, :] + ray_d[None, :] * z_vals[:, None]  # [N_samples, 3]
print(f"Points shape: {pts.shape}")
print(f"Points range: x=[{pts[:,0].min():.3f}, {pts[:,0].max():.3f}], "
      f"y=[{pts[:,1].min():.3f}, {pts[:,1].max():.3f}], "
      f"z=[{pts[:,2].min():.3f}, {pts[:,2].max():.3f}]")

print("\n" + "=" * 60)
print("STEP 6: Embed inputs and run network")
print("=" * 60)

# Embed position
pts_embed = embed_fn(pts)
print(f"Embedded points shape: {pts_embed.shape}")

# Embed view direction (same for all points on ray)
viewdir = ray_d / ray_d.norm()
viewdirs = viewdir.unsqueeze(0).expand(N_samples, -1)
viewdirs_embed = embeddirs_fn(viewdirs)
print(f"Embedded viewdirs shape: {viewdirs_embed.shape}")

# Embed time
t_tensor = torch.full((N_samples, 1), time_val, device=device)
t_embed = embed_time_fn(t_tensor)
print(f"Embedded time shape: {t_embed.shape}")

# Concatenate position and view embeddings
x = torch.cat([pts_embed, viewdirs_embed], dim=-1)
print(f"Network input shape: {x.shape}")

# Run network
with torch.no_grad():
    out, dx = model(x, [t_embed, t_embed])

print(f"\nNetwork output shape: {out.shape}")
print(f"Deformation dx shape: {dx.shape}")

rgb_raw = out[:, :3]
alpha_raw = out[:, 3]

print(f"\nRGB raw: mean={rgb_raw.mean():.4f}, range=[{rgb_raw.min():.4f}, {rgb_raw.max():.4f}]")
print(f"Alpha raw: mean={alpha_raw.mean():.4f}, range=[{alpha_raw.min():.4f}, {alpha_raw.max():.4f}]")
print(f"Deformation dx: mean={dx.abs().mean():.4f}, max={dx.abs().max():.4f}")

print("\n" + "=" * 60)
print("STEP 7: Apply activation functions")
print("=" * 60)

rgb = torch.sigmoid(rgb_raw)
alpha = torch.relu(alpha_raw)

print(f"RGB (sigmoid): mean={rgb.mean():.4f}, range=[{rgb.min():.4f}, {rgb.max():.4f}]")
print(f"Alpha (ReLU): mean={alpha.mean():.4f}, range=[{alpha.min():.4f}, {alpha.max():.4f}]")
print(f"Alpha > 0 count: {(alpha > 0).sum().item()} / {N_samples}")

print("\n" + "=" * 60)
print("STEP 8: Volume rendering")
print("=" * 60)

# Compute distances
dists = z_vals[1:] - z_vals[:-1]
dists = torch.cat([dists, torch.tensor([1e10], device=device)])
dists = dists * ray_d.norm()
print(f"Distances: mean={dists[:-1].mean():.4f}")

# Compute transmittance and weights
# alpha_i = 1 - exp(-sigma * dist)
alpha_i = 1. - torch.exp(-alpha * dists)
print(f"Alpha_i (opacity): mean={alpha_i.mean():.4f}, range=[{alpha_i.min():.4f}, {alpha_i.max():.4f}]")

# T_i = prod_{j<i}(1 - alpha_j)
transmittance = torch.cumprod(torch.cat([torch.ones(1, device=device), 1. - alpha_i[:-1]]), dim=0)
print(f"Transmittance: first={transmittance[0]:.4f}, last={transmittance[-1]:.4f}")

# weights = T_i * alpha_i
weights = transmittance * alpha_i
print(f"Weights: sum={weights.sum():.4f}, max={weights.max():.4f}")
print(f"Weights > 0.01: {(weights > 0.01).sum().item()} samples")

# Final RGB
rgb_final = (weights[:, None] * rgb).sum(dim=0)
print(f"\nFinal RGB: {rgb_final.tolist()}")

# Accumulated opacity (should be close to 1 if object is visible)
acc = weights.sum()
print(f"Accumulated opacity: {acc.item():.4f}")

if acc < 0.1:
    print("\n⚠️  WARNING: Very low accumulated opacity!")
    print("This means the ray doesn't hit any surface with significant density.")
    print("Possible causes:")
    print("  1. Alpha values are too small/negative")
    print("  2. Network hasn't learned proper density distribution")
    print("  3. Wrong camera pose or ray generation")

print("\n" + "=" * 60)
print("STEP 9: Render full image")
print("=" * 60)

# Use render_dnerf to render full image
from render import render_dnerf
from model import create_dnerf
from config import config_parser

# Create a simple args object
class Args:
    def __init__(self):
        self.network_type = 'deformation'
        self.multires = 10
        self.multires_views = 4
        self.multires_time = 10
        self.i_embed = 0
        self.use_viewdirs = True
        self.N_samples = 64
        self.N_importance = 128
        self.netdepth = 8
        self.netwidth = 256
        self.netdepth_fine = 8
        self.netwidth_fine = 256
        self.chunk = 32768
        self.netchunk = 65536
        self.perturb = 0.0
        self.raw_noise_std = 0.0
        self.white_bkgd = True
        self.zero_canonical = True
        self.basedir = './'
        self.expname = 'test_deform'
        self.ft_path = './test_deform/best.tar'
        self.no_reload = False
        self.load_official_weights = False
        self.official_ckpt_path = None
        self.lrate = 5e-4
        self.lrate_decay = 250
        self.datadir = datadir

args = Args()

# Create model with proper render kwargs
render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_dnerf(args)
render_kwargs_test.update({'near': near, 'far': far})

# Render center crop of image (faster)
crop_size = 50
h_start = H // 2 - crop_size // 2
w_start = W // 2 - crop_size // 2

print(f"Rendering {crop_size}x{crop_size} crop from center...")

# Full render for a small region
with torch.no_grad():
    rgb_img, disp_img, acc_img, extras = render_dnerf(
        H, W, K, time_val,
        chunk=args.chunk,
        c2w=pose[:3, :4],
        **render_kwargs_test
    )

print(f"Rendered image shape: {rgb_img.shape}")
print(f"RGB range: [{rgb_img.min():.4f}, {rgb_img.max():.4f}]")
print(f"RGB mean: {rgb_img.mean():.4f}")
print(f"Disparity range: [{disp_img.min():.4f}, {disp_img.max():.4f}]")
print(f"Accumulation range: [{acc_img.min():.4f}, {acc_img.max():.4f}]")
print(f"Accumulation mean: {acc_img.mean():.4f}")

# Save debug image
debug_img = to8b(rgb_img.cpu().numpy())
imageio.imwrite('./test_deform/debug_render.png', debug_img)
print(f"\nSaved debug image to ./test_deform/debug_render.png")

# Check if mostly white
if rgb_img.mean() > 0.9:
    print("\n⚠️  WARNING: Image is mostly white!")
    print(f"Accumulation mean: {acc_img.mean():.4f}")
    if acc_img.mean() < 0.1:
        print("Low accumulation suggests rays aren't hitting the object properly.")
