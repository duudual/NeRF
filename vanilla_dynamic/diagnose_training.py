"""
Detailed diagnosis of deformation network training
Check RGB and alpha outputs at each stage
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
sys.path.insert(0, '.')

from network import DNeRF_Deformation, NeRF
from positional_encoding import get_embedder, get_time_embedder
from utils import device

print("=" * 60)
print("STEP 1: Check Initial Network State (Before Training)")
print("=" * 60)

# Create network
embed_fn, input_ch = get_embedder(10, 0)
embeddirs_fn, input_ch_views = get_embedder(4, 0)
embed_time_fn, input_ch_time = get_time_embedder(10)

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

# Check canonical network initialization
occ = model._occ
print("\n--- Canonical Network (_occ) Initialization ---")
print(f"Alpha linear weight: norm={occ.alpha_linear.weight.norm():.4f}, mean={occ.alpha_linear.weight.mean():.6f}")
print(f"Alpha linear bias: {occ.alpha_linear.bias.item():.4f}")
print(f"RGB linear weight: norm={occ.rgb_linear.weight.norm():.4f}, mean={occ.rgb_linear.weight.mean():.6f}")
print(f"RGB linear bias: {occ.rgb_linear.bias.tolist()}")

# Check feature linear
print(f"Feature linear weight: norm={occ.feature_linear.weight.norm():.4f}")
print(f"Views linear weight: norm={occ.views_linears[0].weight.norm():.4f}")

# Test with random inputs
n_points = 1000
pts = torch.randn(n_points, 3, device=device) * 0.5
dirs = torch.randn(n_points, 3, device=device)
dirs = dirs / dirs.norm(dim=-1, keepdim=True)
t = torch.full((n_points, 1), 0.5, device=device)

pts_embed = embed_fn(pts)
dirs_embed = embeddirs_fn(dirs)
t_embed = embed_time_fn(t)

x = torch.cat([pts_embed, dirs_embed], dim=-1)
ts = [t_embed, t_embed]

print("\n--- Initial Network Output (Before Training) ---")
with torch.no_grad():
    out, dx = model(x, ts)

rgb_raw = out[:, :3]
alpha_raw = out[:, 3]
rgb_sigmoid = torch.sigmoid(rgb_raw)
alpha_relu = F.relu(alpha_raw)

print(f"RGB raw: mean={rgb_raw.mean():.4f}, std={rgb_raw.std():.4f}, range=[{rgb_raw.min():.4f}, {rgb_raw.max():.4f}]")
print(f"RGB sigmoid: mean={rgb_sigmoid.mean():.4f}, range=[{rgb_sigmoid.min():.4f}, {rgb_sigmoid.max():.4f}]")
print(f"Alpha raw: mean={alpha_raw.mean():.4f}, std={alpha_raw.std():.4f}, range=[{alpha_raw.min():.4f}, {alpha_raw.max():.4f}]")
print(f"Alpha ReLU: mean={alpha_relu.mean():.4f}")

# Trace through the network manually
print("\n--- Tracing Through Network Layers ---")
input_pts = pts_embed
input_views = dirs_embed

h = input_pts
for i, l in enumerate(occ.pts_linears):
    h = l(h)
    h = F.relu(h)
    if i in occ.skips:
        h = torch.cat([input_pts, h], -1)
print(f"After all pts_linears: mean={h.mean():.4f}, std={h.std():.4f}, max={h.max():.4f}")

feature = occ.feature_linear(h)
print(f"After feature_linear: mean={feature.mean():.4f}, std={feature.std():.4f}, max={feature.abs().max():.4f}")

h_view = torch.cat([feature, input_views], -1)
for l in occ.views_linears:
    h_view = l(h_view)
    h_view = F.relu(h_view)
print(f"After views_linear: mean={h_view.mean():.4f}, std={h_view.std():.4f}, max={h_view.max():.4f}")

rgb_out = occ.rgb_linear(h_view)
print(f"RGB output (before sigmoid): mean={rgb_out.mean():.4f}, std={rgb_out.std():.4f}, range=[{rgb_out.min():.4f}, {rgb_out.max():.4f}]")

print("\n" + "=" * 60)
print("STEP 2: Load Trained Model and Check")
print("=" * 60)

# Try to load trained model
import os
ckpt_path = "./test_deform/best.tar"
if os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['network_fn_state_dict'])
    model.eval()
    
    print(f"\nLoaded checkpoint: {ckpt_path}")
    print(f"Global step: {ckpt['global_step']}")
    
    # Check weights after training
    print("\n--- Canonical Network After Training ---")
    print(f"Alpha linear weight: norm={occ.alpha_linear.weight.norm():.4f}, mean={occ.alpha_linear.weight.mean():.6f}")
    print(f"Alpha linear bias: {occ.alpha_linear.bias.item():.4f}")
    print(f"RGB linear weight: norm={occ.rgb_linear.weight.norm():.4f}, mean={occ.rgb_linear.weight.mean():.6f}")
    print(f"RGB linear bias: {occ.rgb_linear.bias.tolist()}")
    print(f"Feature linear weight: norm={occ.feature_linear.weight.norm():.4f}")
    print(f"Views linear weight: norm={occ.views_linears[0].weight.norm():.4f}")
    
    # Test output
    print("\n--- Trained Network Output ---")
    with torch.no_grad():
        out, dx = model(x, ts)
    
    rgb_raw = out[:, :3]
    alpha_raw = out[:, 3]
    rgb_sigmoid = torch.sigmoid(rgb_raw)
    alpha_relu = F.relu(alpha_raw)
    
    print(f"RGB raw: mean={rgb_raw.mean():.4f}, std={rgb_raw.std():.4f}, range=[{rgb_raw.min():.4f}, {rgb_raw.max():.4f}]")
    print(f"RGB sigmoid: mean={rgb_sigmoid.mean():.4f}, range=[{rgb_sigmoid.min():.4f}, {rgb_sigmoid.max():.4f}]")
    print(f"Alpha raw: mean={alpha_raw.mean():.4f}, std={alpha_raw.std():.4f}, range=[{alpha_raw.min():.4f}, {alpha_raw.max():.4f}]")
    print(f"Alpha ReLU: mean={alpha_relu.mean():.4f}")
    
    # Trace through layers
    print("\n--- Tracing Through Trained Network Layers ---")
    h = input_pts
    for i, l in enumerate(occ.pts_linears):
        h = l(h)
        h = F.relu(h)
        if i in occ.skips:
            h = torch.cat([input_pts, h], -1)
    print(f"After all pts_linears: mean={h.mean():.4f}, std={h.std():.4f}, max={h.max():.4f}")
    
    feature = occ.feature_linear(h)
    print(f"After feature_linear: mean={feature.mean():.4f}, std={feature.std():.4f}, max={feature.abs().max():.4f}")
    
    h_view = torch.cat([feature, input_views], -1)
    for l in occ.views_linears:
        h_view = l(h_view)
        h_view = F.relu(h_view)
    print(f"After views_linear: mean={h_view.mean():.4f}, std={h_view.std():.4f}, max={h_view.max():.4f}")
    
    rgb_out = occ.rgb_linear(h_view)
    print(f"RGB output (before sigmoid): mean={rgb_out.mean():.4f}, std={rgb_out.std():.4f}, range=[{rgb_out.min():.4f}, {rgb_out.max():.4f}]")
else:
    print(f"Checkpoint not found: {ckpt_path}")

print("\n" + "=" * 60)
print("STEP 3: Compare with Official D-NeRF NeRFOriginal")
print("=" * 60)

# Check what official D-NeRF does for initialization
print("\nOfficial D-NeRF uses PyTorch default initialization:")
print("- Linear layers: Kaiming uniform (fan_in mode)")
print("- This gives weight ~ U(-1/sqrt(fan_in), 1/sqrt(fan_in))")
print(f"- For W=256: range ~ [-0.0625, 0.0625]")
print(f"- For rgb_linear (in=128, out=3): range ~ [-0.0884, 0.0884]")

# Create a reference network with default init (no custom init)
class NeRF_NoInit(nn.Module):
    def __init__(self, D=8, W=256, input_ch=63, input_ch_views=27, skips=[4]):
        super().__init__()
        self.skips = skips
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + 
            [nn.Linear(W, W) if i not in skips else nn.Linear(W + input_ch, W) for i in range(D-1)]
        )
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])
        self.feature_linear = nn.Linear(W, W)
        self.alpha_linear = nn.Linear(W, 1)
        self.rgb_linear = nn.Linear(W//2, 3)

ref_net = NeRF_NoInit(D=8, W=256, input_ch=input_ch, input_ch_views=input_ch_views).to(device)
print("\n--- Reference Network (PyTorch Default Init) ---")
print(f"Alpha linear weight: norm={ref_net.alpha_linear.weight.norm():.4f}, mean={ref_net.alpha_linear.weight.mean():.6f}")
print(f"Alpha linear bias: {ref_net.alpha_linear.bias.item():.4f}")
print(f"RGB linear weight: norm={ref_net.rgb_linear.weight.norm():.4f}, mean={ref_net.rgb_linear.weight.mean():.6f}")
print(f"RGB linear bias: {ref_net.rgb_linear.bias.tolist()}")

# Test reference network output
h = input_pts
for i, l in enumerate(ref_net.pts_linears):
    h = l(h)
    h = F.relu(h)
    if i in ref_net.skips:
        h = torch.cat([input_pts, h], -1)

feature = ref_net.feature_linear(h)
h_view = torch.cat([feature, input_views], -1)
for l in ref_net.views_linears:
    h_view = l(h_view)
    h_view = F.relu(h_view)

rgb_out = ref_net.rgb_linear(h_view)
alpha_out = ref_net.alpha_linear(h)

print(f"\n--- Reference Network Initial Output ---")
print(f"After views_linear: mean={h_view.mean():.4f}, std={h_view.std():.4f}, max={h_view.max():.4f}")
print(f"RGB output: mean={rgb_out.mean():.4f}, std={rgb_out.std():.4f}, range=[{rgb_out.min():.4f}, {rgb_out.max():.4f}]")
print(f"Alpha output: mean={alpha_out.mean():.4f}, std={alpha_out.std():.4f}, range=[{alpha_out.min():.4f}, {alpha_out.max():.4f}]")
print(f"RGB sigmoid: mean={torch.sigmoid(rgb_out).mean():.4f}")

print("\n" + "=" * 60)
print("DIAGNOSIS SUMMARY")
print("=" * 60)
print("""
If RGB raw output is very large (>10), sigmoid will saturate to 1.0 (white).
This can happen if:
1. views_linear output is too large (should be ~0-2 on average)
2. feature_linear amplifies values too much
3. Weight initialization is wrong (Xavier may be too large for this architecture)

Solution: Use smaller initialization for rgb_linear, or add normalization.
""")
