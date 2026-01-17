"""
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
                 skips=[4], use_viewdirs=False, use_layernorm=False):
        """
        Args:
            D: number of layers in network
            W: number of channels per layer
            input_ch: input channel dimension (after positional encoding)
            input_ch_views: input channel dimension for views (after positional encoding)
            output_ch: output channel dimension
            skips: layers to add skip connections
            use_viewdirs: whether to use viewing directions
            use_layernorm: whether to use LayerNorm for training stability
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        self.use_layernorm = use_layernorm
        
        # Position encoding layers
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + 
            [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) 
             for i in range(D-1)]
        )
        
        # LayerNorm layers for training stability
        if use_layernorm:
            self.pts_layernorms = nn.ModuleList([nn.LayerNorm(W) for _ in range(D)])
            self.views_layernorm = nn.LayerNorm(W//2)
        
        # View-dependent layers
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])
        
        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W//2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

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
            if self.use_layernorm:
                h = self.pts_layernorms[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)
        
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                if self.use_layernorm:
                    h = self.views_layernorm(h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs, torch.zeros_like(input_pts[:, :3])


class DNeRF_Straightforward(nn.Module):
    """
    Input: (x, y, z, t, θ, φ) - 3D position + time + viewing direction
    Output: (c, sigma) - color and density
    
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


class DNeRF_Deformation(nn.Module):
    
    def __init__(self, D=8, W=256, D_deform=8, W_deform=256,
                 input_ch=3, input_ch_views=3, input_ch_time=1,
                 output_ch=4, skips=[4], skips_deform=[4],
                 use_viewdirs=True, zero_canonical=True, use_layernorm=False):
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
            use_layernorm: if True, use LayerNorm for training stability
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
        self.use_layernorm = use_layernorm
        
        self.embed_fn = None
        
        self._occ = NeRF(
            D=D, W=W, 
            input_ch=input_ch, 
            input_ch_views=input_ch_views,
            output_ch=output_ch, 
            skips=skips, 
            use_viewdirs=use_viewdirs,
            use_layernorm=use_layernorm
        )
        
        self.skips_deform = skips  # Use same skips as canonical network (official behavior)
        self._time, self._time_out, self._time_layernorms = self._create_time_net(
            D=D_deform, W=W_deform,
            input_ch=input_ch, input_ch_time=input_ch_time,
            skips=self.skips_deform,  # Official uses self.skips
            use_layernorm=use_layernorm
        )

    
    def _create_time_net(self, D=8, W=256, input_ch=63, input_ch_time=9, skips=[], use_layernorm=False):
        """
        Create deformation network matching official D-NeRF architecture.
        
        Official structure:
        - layers[0]: Linear(input_ch + input_ch_time, W)
        - layers[1..D-1]: Linear(W, W) or Linear(W + input_ch, W) if skip
        - output: Linear(W, 3)
        
        Returns:
            _time: ModuleList of layers
            _time_out: Linear output layer
            _time_layernorms: ModuleList of LayerNorm layers (or None)
        """
        layers = [nn.Linear(input_ch + input_ch_time, W)]
        for i in range(D - 1):
            in_channels = W
            if i in skips:
                in_channels += input_ch
            layers.append(nn.Linear(in_channels, W))
        
        layernorms = None
        if use_layernorm:
            layernorms = nn.ModuleList([nn.LayerNorm(W) for _ in range(D)])
        
        return nn.ModuleList(layers), nn.Linear(W, 3), layernorms
    
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
            if self.use_layernorm and self._time_layernorms is not None:
                h = self._time_layernorms[i](h)
            h = F.relu(h)
            if i in self.skips_deform:
                h = torch.cat([new_pts, h], -1)
        
        return net_final(h)

    def forward(self, x, ts, pts_raw=None, t_raw=None):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        
        if isinstance(ts, (list, tuple)):
            t = ts[0]
        else:
            t = ts
        
        if t_raw is not None:
            cur_time = t_raw[0, 0].item() if t_raw.numel() > 0 else 1.0
        else:
            cur_time = t[0, 0].item() if t.numel() > 0 else 1.0
        
        if cur_time == 0.0 and self.zero_canonical:
            dx = torch.zeros(input_pts.shape[0], 3, device=input_pts.device)
            canonical_input = torch.cat([input_pts, input_views], dim=-1)
        else:
            dx = self._query_time(input_pts, t, self._time, self._time_out)
            
            if pts_raw is not None:
                input_pts_orig = pts_raw
            else:
                input_pts_orig = input_pts[:, :3]
            
            deformed_pts = input_pts_orig + dx
            if self.embed_fn is not None:
                canonical_pts_embed = self.embed_fn(deformed_pts)
            else:
                raise RuntimeError("embed_fn must be set for deformation network")
            
            canonical_input = torch.cat([canonical_pts_embed, input_views], dim=-1)
        
        out, _ = self._occ(canonical_input, t)
        
        return out, dx
