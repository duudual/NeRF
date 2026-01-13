"""
Dynamic NeRF Network Models

This module implements two approaches for dynamic scene representation:
1. Straightforward 6D approach: MLP with (x, y, z, t, θ, φ) input
2. Deformation Network approach: Canonical network + Deformation network
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class NeRF(nn.Module):
    """Standard NeRF MLP model (used as canonical network)."""
    
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, 
                 skips=[4], use_viewdirs=False):
        """
        Args:
            D: number of layers in network
            W: number of channels per layer
            input_ch: input channel dimension (after positional encoding)
            input_ch_views: input channel dimension for views (after positional encoding)
            output_ch: output channel dimension
            skips: layers to add skip connections
            use_viewdirs: whether to use viewing directions
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        
        # Position encoding layers
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + 
            [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) 
             for i in range(D-1)]
        )
        
        # View-dependent layers
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])
        
        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W//2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)
        
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs


class DNeRF_Straightforward(nn.Module):
    """
    Straightforward Dynamic NeRF (6D approach)
    
    Input: (x, y, z, t, θ, φ) - 3D position + time + viewing direction
    Output: (c, σ) - color and density
    
    This approach directly extends NeRF by adding time as an additional input
    dimension to the MLP.
    """
    
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, input_ch_time=1,
                 output_ch=4, skips=[4], use_viewdirs=True):
        """
        Args:
            D: number of layers in network
            W: number of channels per layer
            input_ch: input channel dimension for position (after positional encoding)
            input_ch_views: input channel dimension for views (after positional encoding)
            input_ch_time: input channel dimension for time (after positional encoding)
            output_ch: output channel dimension
            skips: layers to add skip connections
            use_viewdirs: whether to use viewing directions
        """
        super(DNeRF_Straightforward, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.input_ch_time = input_ch_time
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        
        # Combined position + time input
        input_ch_pts_time = input_ch + input_ch_time
        
        # Position + time encoding layers
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch_pts_time, W)] + 
            [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch_pts_time, W) 
             for i in range(D-1)]
        )
        
        # View-dependent layers
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])
        
        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W//2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x, t_embed):
        """
        Args:
            x: [N, input_ch + input_ch_views] embedded position + view directions
            t_embed: [N, input_ch_time] embedded time
        
        Returns:
            outputs: [N, 4] (rgb, alpha)
            dx: [N, 3] deformation (zeros for this model)
        """
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        
        # Concatenate position and time
        input_pts_time = torch.cat([input_pts, t_embed], dim=-1)
        
        h = input_pts_time
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts_time, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)
        
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        # Return zeros for dx since this model doesn't predict deformation
        dx = torch.zeros_like(input_pts[:, :3]) if input_pts.shape[-1] >= 3 else torch.zeros(input_pts.shape[0], 3, device=input_pts.device)
        
        return outputs, dx


class DeformationNetwork(nn.Module):
    """
    Deformation Network Ψ_t(x, t) → Δx
    
    Predicts a deformation field that transforms points from time t
    to the canonical configuration.
    """
    
    def __init__(self, D=6, W=128, input_ch=3, input_ch_time=1, skips=[3]):
        """
        Args:
            D: number of layers
            W: number of channels per layer
            input_ch: input channel dimension for position (after positional encoding)
            input_ch_time: input channel dimension for time (after positional encoding)
            skips: layers to add skip connections
        """
        super(DeformationNetwork, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_time = input_ch_time
        self.skips = skips
        
        input_ch_all = input_ch + input_ch_time
        
        # Deformation layers
        self.layers = nn.ModuleList(
            [nn.Linear(input_ch_all, W)] + 
            [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch_all, W) 
             for i in range(D-1)]
        )
        
        # Output layer: predict 3D deformation
        self.output_linear = nn.Linear(W, 3)
        
        # Initialize output to small values for stable training
        nn.init.xavier_uniform_(self.output_linear.weight, gain=0.01)
        nn.init.zeros_(self.output_linear.bias)

    def forward(self, x_embed, t_embed):
        """
        Args:
            x_embed: [N, input_ch] embedded position
            t_embed: [N, input_ch_time] embedded time
        
        Returns:
            dx: [N, 3] predicted deformation
        """
        h = torch.cat([x_embed, t_embed], dim=-1)
        input_all = h
        
        for i, l in enumerate(self.layers):
            h = self.layers[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_all, h], -1)
        
        dx = self.output_linear(h)
        return dx


class DNeRF_Deformation(nn.Module):
    """
    Dynamic NeRF with Deformation Network
    
    Two network modules:
    1. Canonical Network Ψ_x(x, d) → (c, σ): Encodes scene in canonical configuration
    2. Deformation Network Ψ_t(x, t) → Δx: Predicts deformation from time t to canonical
    
    Pipeline:
    - For t != 0: Compute dx = Ψ_t(x, t), then query Ψ_x(x + dx, d)
    - For t == 0 (canonical): Directly query Ψ_x(x, d)
    """
    
    def __init__(self, D=8, W=256, D_deform=6, W_deform=128,
                 input_ch=3, input_ch_views=3, input_ch_time=1,
                 output_ch=4, skips=[4], skips_deform=[3],
                 use_viewdirs=True, zero_canonical=True):
        """
        Args:
            D: number of layers in canonical network
            W: number of channels per layer in canonical network
            D_deform: number of layers in deformation network
            W_deform: number of channels per layer in deformation network
            input_ch: input channel dimension for position (after positional encoding)
            input_ch_views: input channel dimension for views (after positional encoding)
            input_ch_time: input channel dimension for time (after positional encoding)
            output_ch: output channel dimension
            skips: layers to add skip connections in canonical network
            skips_deform: layers to add skip connections in deformation network
            use_viewdirs: whether to use viewing directions
            zero_canonical: if True, use zero deformation at t=0
        """
        super(DNeRF_Deformation, self).__init__()
        
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.input_ch_time = input_ch_time
        self.zero_canonical = zero_canonical
        
        # Canonical Network
        self.canonical_net = NeRF(
            D=D, W=W, 
            input_ch=input_ch, 
            input_ch_views=input_ch_views,
            output_ch=output_ch, 
            skips=skips, 
            use_viewdirs=use_viewdirs
        )
        
        # Deformation Network
        self.deform_net = DeformationNetwork(
            D=D_deform, W=W_deform,
            input_ch=input_ch,
            input_ch_time=input_ch_time,
            skips=skips_deform
        )
        
        # Store embed_fn for re-embedding deformed points (set externally)
        self.embed_fn = None  # type: ignore

    def forward(self, x, t_embed, pts_raw=None):
        """
        Args:
            x: [N, input_ch + input_ch_views] embedded position + view directions
            t_embed: [N, input_ch_time] embedded time
            pts_raw: [N, 3] raw 3D positions (before encoding) - needed for deformation
        
        Returns:
            outputs: [N, 4] (rgb, alpha)
            dx: [N, 3] predicted deformation
        """
        input_pts_embed, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        
        # Check if we're at canonical time (t=0)
        # t_embed is already encoded, so we check if all time values are the same
        # and approximately zero in original space
        if pts_raw is not None:
            t_original = t_embed[:, 0] if t_embed.shape[-1] > 0 else t_embed
            is_canonical = self.zero_canonical and torch.all(torch.abs(t_original) < 1e-6)
        else:
            is_canonical = False
        
        if is_canonical:
            # At canonical time, no deformation
            dx = torch.zeros(input_pts_embed.shape[0], 3, device=input_pts_embed.device)
            canonical_pts_embed = input_pts_embed
        else:
            # Predict deformation
            dx = self.deform_net(input_pts_embed, t_embed)
            
            # Apply deformation to raw points and re-embed
            if pts_raw is not None and self.embed_fn is not None:
                deformed_pts = pts_raw + dx
                canonical_pts_embed = self.embed_fn(deformed_pts)
            else:
                # Fallback: just use the embedded points (less accurate)
                canonical_pts_embed = input_pts_embed
        
        # Query canonical network
        canonical_input = torch.cat([canonical_pts_embed, input_views], dim=-1)
        outputs = self.canonical_net(canonical_input)
        
        return outputs, dx


class DNeRF_Fine(nn.Module):
    """
    Fine network wrapper for Dynamic NeRF.
    Supports both straightforward and deformation approaches.
    """
    
    def __init__(self, network_type='deformation', **kwargs):
        """
        Args:
            network_type: 'straightforward' or 'deformation'
            **kwargs: arguments passed to the underlying network
        """
        super(DNeRF_Fine, self).__init__()
        
        self.network_type = network_type
        
        if network_type == 'straightforward':
            self.network = DNeRF_Straightforward(**kwargs)
        else:
            self.network = DNeRF_Deformation(**kwargs)
    
    def forward(self, x, t_embed, pts_raw=None):
        if self.network_type == 'straightforward':
            return self.network(x, t_embed)
        else:
            return self.network(x, t_embed, pts_raw)
    
    def set_embed_fn(self, embed_fn):
        """Set embedding function for deformation network."""
        if hasattr(self.network, 'embed_fn'):
            self.network.embed_fn = embed_fn
