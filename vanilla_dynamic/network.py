"""
Dynamic NeRF Network Models

This module implements two approaches for dynamic scene representation:
1. Straightforward 6D approach: MLP with (x, y, z, t, θ, φ) input
2. Deformation Network approach: Canonical network + Deformation network

The deformation network (DirectTemporalNeRF / DNeRF_Deformation) is designed to be
FULLY COMPATIBLE with official D-NeRF pretrained weights.
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
            # NOTE: Official D-NeRF uses PyTorch default initialization
            # No custom bias initialization - let the network learn naturally
        else:
            self.output_linear = nn.Linear(W, output_ch)
        
        # NOTE: Official D-NeRF does NOT use any custom weight initialization
        # PyTorch's default Kaiming initialization works well for ReLU networks

    def forward(self, x, ts=None):
        """
        Args:
            x: [N, input_ch + input_ch_views] embedded position + view directions
            ts: ignored, for API compatibility with DirectTemporalNeRF
        
        Returns:
            outputs: [N, 4] (rgb, alpha)
            dx: [N, 3] zeros (no deformation for static NeRF)
        """
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

        return outputs, torch.zeros_like(input_pts[:, :3])


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
            # NOTE: Official D-NeRF uses PyTorch default initialization
            # No custom initialization needed
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x, ts):
        """
        Args:
            x: [N, input_ch + input_ch_views] embedded position + view directions
            ts: [N, input_ch_time] embedded time OR [t_embed, t_embed] list (for compatibility)
        
        Returns:
            outputs: [N, 4] (rgb, alpha)
            dx: [N, 3] deformation (zeros for this model)
        """
        # Handle ts format: can be a list [t, t] (from run_network_dnerf) or a tensor
        if isinstance(ts, (list, tuple)):
            t_embed = ts[0]  # Use first element
        else:
            t_embed = ts
        
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
    
    NOTE: This class is kept for backward compatibility but is NOT used
    by DNeRF_Deformation which follows the official D-NeRF architecture.
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
    Dynamic NeRF with Deformation Network - OFFICIAL D-NeRF COMPATIBLE
    
    This implementation exactly matches the official D-NeRF (DirectTemporalNeRF) architecture,
    allowing direct loading of official pretrained weights.
    
    Architecture:
    - _occ: Canonical NeRF network (NeRFOriginal in official code)
    - _time: Deformation network layers (ModuleList)
    - _time_out: Deformation output layer (Linear -> 3D displacement)
    
    Weight mapping (official -> this):
    - _occ.* -> _occ.*
    - _time.* -> _time.*
    - _time_out.* -> _time_out.*
    """
    
    def __init__(self, D=8, W=256, D_deform=8, W_deform=256,
                 input_ch=3, input_ch_views=3, input_ch_time=1,
                 output_ch=4, skips=[4], skips_deform=[4],
                 use_viewdirs=True, zero_canonical=True):
        """
        Args:
            D: number of layers in canonical network
            W: number of channels per layer in canonical network
            D_deform: number of layers in deformation network (default 8 to match official)
            W_deform: number of channels in deformation network (default 256 to match official)
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
        
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.input_ch_time = input_ch_time
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        self.zero_canonical = zero_canonical
        
        # Store embed_fn for re-embedding deformed points (set externally)
        self.embed_fn = None
        
        # Canonical Network (_occ in official D-NeRF)
        # This is NeRFOriginal - a standard NeRF that takes position+views
        self._occ = NeRF(
            D=D, W=W, 
            input_ch=input_ch, 
            input_ch_views=input_ch_views,
            output_ch=output_ch, 
            skips=skips, 
            use_viewdirs=use_viewdirs
        )
        
        # Deformation Network (_time and _time_out in official D-NeRF)
        # Input: embedded position + embedded time
        # Output: 3D displacement dx
        # IMPORTANT: Official D-NeRF uses self.skips (same as canonical network) for time network
        self.skips_deform = skips  # Use same skips as canonical network (official behavior)
        self._time, self._time_out = self._create_time_net(
            D=D_deform, W=W_deform,
            input_ch=input_ch, input_ch_time=input_ch_time,
            skips=self.skips_deform  # Official uses self.skips
        )
        
        # NOTE: Official D-NeRF uses PyTorch default initialization
        # No custom initialization needed - Kaiming uniform works well for ReLU networks
    
    def _create_time_net(self, D=8, W=256, input_ch=63, input_ch_time=9, skips=[]):
        """
        Create deformation network matching official D-NeRF architecture.
        
        Official structure:
        - layers[0]: Linear(input_ch + input_ch_time, W)
        - layers[1..D-1]: Linear(W, W) or Linear(W + input_ch, W) if skip
        - output: Linear(W, 3)
        
        Returns:
            _time: ModuleList of layers
            _time_out: Linear output layer
        """
        layers = [nn.Linear(input_ch + input_ch_time, W)]
        for i in range(D - 1):
            in_channels = W
            if i in skips:
                in_channels += input_ch
            layers.append(nn.Linear(in_channels, W))
        
        return nn.ModuleList(layers), nn.Linear(W, 3)
    
    def _query_time(self, new_pts, t, net, net_final):
        """
        Query deformation network.
        
        Exactly matches official D-NeRF query_time method.
        
        Args:
            new_pts: [N, input_ch] embedded position
            t: [N, input_ch_time] embedded time
            net: ModuleList of deformation layers
            net_final: output Linear layer
        
        Returns:
            dx: [N, 3] deformation
        """
        h = torch.cat([new_pts, t], dim=-1)
        for i, l in enumerate(net):
            h = net[i](h)
            h = F.relu(h)
            if i in self.skips_deform:
                h = torch.cat([new_pts, h], -1)
        
        return net_final(h)

    def forward(self, x, ts, pts_raw=None, t_raw=None):
        """
        Forward pass matching official D-NeRF (DirectTemporalNeRF).
        
        IMPORTANT: This implementation now matches official D-NeRF exactly.
        The official code extracts raw points from embedded positions using input_pts[:, :3]
        because positional encoding includes the original input (include_input=True).
        
        Args:
            x: [N, input_ch + input_ch_views] embedded position + view directions
            ts: tuple/list of (embedded_time, embedded_time) - following official API
                 OR just embedded_time tensor
            pts_raw: [N, 3] raw 3D positions (OPTIONAL - extracted from x if not provided)
            t_raw: [N, 1] raw time values (OPTIONAL - extracted from ts if not provided)
        
        Returns:
            outputs: [N, 4] (rgb, alpha)
            dx: [N, 3] predicted deformation
        """
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        
        # Handle ts format (official uses tuple [t, t])
        if isinstance(ts, (list, tuple)):
            t = ts[0]
        else:
            t = ts
        
        # Get raw time for canonical check
        # Use t_raw if provided (for include_input=False case)
        # Otherwise assume include_input=True and use t[0, 0]
        if t_raw is not None:
            cur_time = t_raw[0, 0].item() if t_raw.numel() > 0 else 1.0
        else:
            cur_time = t[0, 0].item() if t.numel() > 0 else 1.0
        
        # At canonical time (t=0), no deformation
        # Official: if cur_time == 0. and self.zero_canonical:
        if cur_time == 0.0 and self.zero_canonical:
            dx = torch.zeros(input_pts.shape[0], 3, device=input_pts.device)
            # At t=0, use original embedded points directly
            canonical_input = torch.cat([input_pts, input_views], dim=-1)
        else:
            # Predict deformation: dx = _query_time(embed(x), embed(t))
            dx = self._query_time(input_pts, t, self._time, self._time_out)
            
            # Get raw points - use pts_raw if provided, otherwise fall back to input_pts[:, :3]
            # (input_pts[:, :3] only works if include_input=True in positional encoding)
            if pts_raw is not None:
                input_pts_orig = pts_raw
            else:
                input_pts_orig = input_pts[:, :3]
            
            # Apply deformation and re-embed
            # Official: input_pts = self.embed_fn(input_pts_orig + dx)
            deformed_pts = input_pts_orig + dx
            if self.embed_fn is not None:
                canonical_pts_embed = self.embed_fn(deformed_pts)
            else:
                # Fallback - this shouldn't happen in normal operation
                raise RuntimeError("embed_fn must be set for deformation network")
            
            canonical_input = torch.cat([canonical_pts_embed, input_views], dim=-1)
        
        # Query canonical network
        out, _ = self._occ(canonical_input, t)
        
        return out, dx


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
    
    def forward(self, x, ts, pts_raw=None, t_raw=None):
        """
        Forward pass - matches official D-NeRF API.
        
        Args:
            x: [N, input_ch + input_ch_views] embedded position + view directions
            ts: tuple/list of (embedded_time, embedded_time) or just embedded_time tensor
            pts_raw: optional raw points (not needed if include_input=True in embedding)
            t_raw: optional raw time values
        """
        if self.network_type == 'straightforward':
            return self.network(x, ts)
        else:
            return self.network(x, ts, pts_raw, t_raw)
    
    def set_embed_fn(self, embed_fn):
        """Set embedding function for deformation network."""
        if hasattr(self.network, 'embed_fn'):
            self.network.embed_fn = embed_fn
