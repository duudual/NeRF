"""
Latent Head for predicting Tri-plane representation.

This head predicts a Tri-plane representation consisting of three 64x64 feature grids
(XY, XZ, YZ planes) with 32-dimensional features per position, plus a shared MLP
for querying RGB and sigma values.

Tri-plane Representation:
- XY plane: 64x64 grid with 32-dim features
- XZ plane: 64x64 grid with 32-dim features  
- YZ plane: 64x64 grid with 32-dim features
- Shared MLP: Takes concatenated features (96-dim) + positional encoding + direction
              and outputs RGB (3) + sigma (1)
"""

import os
from typing import List, Dict, Tuple, Union, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import create_uv_grid, position_grid_to_embed


# ============== Helper functions for DPT-style architecture ==============

def _make_scratch(in_shape, out_shape, expand=False):
    """Create scratch layers for feature fusion."""
    scratch = nn.Module()
    
    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    out_shape4 = out_shape
    
    if expand:
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        out_shape4 = out_shape * 8

    scratch.layer1_rn = nn.Conv2d(in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False)
    scratch.layer2_rn = nn.Conv2d(in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False)
    scratch.layer3_rn = nn.Conv2d(in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False)
    scratch.layer4_rn = nn.Conv2d(in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False)

    return scratch


class ResidualConvUnit(nn.Module):
    """Residual convolution module."""
    def __init__(self, features):
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        return out + x


class FeatureFusionBlock(nn.Module):
    """Feature fusion block."""
    def __init__(self, features, has_residual=True):
        super().__init__()
        self.has_residual = has_residual
        self.resConfUnit = ResidualConvUnit(features)
        self.out_conv = nn.Conv2d(features, features, kernel_size=1, stride=1, padding=0)

    def forward(self, x, residual=None):
        output = x
        if self.has_residual and residual is not None:
            # Interpolate residual to match x's spatial size if needed
            if residual.shape[-2:] != x.shape[-2:]:
                residual = F.interpolate(residual, size=x.shape[-2:], mode="bilinear", align_corners=True)
            output = output + residual
        output = self.resConfUnit(output)
        output = F.interpolate(output, scale_factor=2, mode="bilinear", align_corners=True)
        output = self.out_conv(output)
        return output
        return output


def _make_fusion_block(features, has_residual=True):
    return FeatureFusionBlock(features, has_residual)


# ============== Tri-plane Query MLP ==============

class TriplaneMLP(nn.Module):
    """
    Shared MLP for querying the Tri-plane representation.
    
    Takes concatenated plane features (96-dim) + positional encoding + direction
    and outputs RGB (3) + sigma (1).
    
    Architecture:
        Input: plane_features (96) + pos_enc (63) + dir_enc (27) = 186
        Hidden: 128 -> 128
        Output: 4 (RGB + sigma)
    """
    
    # Positional encoding dimensions
    POS_ENC_DIM = 63   # 3 + 3 * 2 * 10
    DIR_ENC_DIM = 27   # 3 + 3 * 2 * 4
    PLANE_FEAT_DIM = 96  # 32 * 3 planes
    
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        
        input_dim = self.PLANE_FEAT_DIM + self.POS_ENC_DIM
        
        # Position processing network
        self.pos_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        
        # Sigma output (density)
        self.sigma_net = nn.Linear(hidden_dim, 1)
        
        # Color output (with view direction)
        self.color_net = nn.Sequential(
            nn.Linear(hidden_dim + self.DIR_ENC_DIM, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 3),
            nn.Sigmoid(),  # RGB in [0, 1]
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Initialize sigma output with small weights
        nn.init.normal_(self.sigma_net.weight, std=0.01)
        nn.init.zeros_(self.sigma_net.bias)
    
    def forward(
        self, 
        plane_features: torch.Tensor, 
        pos_encoded: torch.Tensor, 
        dir_encoded: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            plane_features: Concatenated features from tri-planes [B, N, 96]
            pos_encoded: Positional encoded position [B, N, 63]
            dir_encoded: Positional encoded direction [B, N, 27]
            
        Returns:
            sigma: Density [B, N, 1]
            rgb: Color [B, N, 3]
        """
        # Concatenate plane features with position encoding
        x = torch.cat([plane_features, pos_encoded], dim=-1)
        
        # Process through position network
        h = self.pos_net(x)
        
        # Sigma output
        sigma = self.sigma_net(h)
        
        # Color output with view direction
        h_with_dir = torch.cat([h, dir_encoded], dim=-1)
        rgb = self.color_net(h_with_dir)
        
        return sigma, rgb


# ============== Latent Head ==============

class LatentHead(nn.Module):
    """
    Latent Head for predicting Tri-plane representation.
    
    This head processes features from a vision transformer backbone and produces:
    1. Three 64x64 feature grids (XY, XZ, YZ planes) with 32-dim features
    2. A shared MLP for querying RGB and sigma values
    
    The output can be used for efficient NeRF-style rendering by:
    1. Bilinear interpolation on each plane using corresponding coordinates
    2. Concatenating the three 32-dim feature vectors (-> 96-dim)
    3. Passing through the shared MLP with positional/directional encoding
    
    Args:
        dim_in (int): Input dimension from transformer (2 * embed_dim).
        patch_size (int): Patch size. Default is 14.
        features (int): Feature channels for intermediate representations. Default is 256.
        plane_size (int): Size of each plane grid. Default is 64.
        plane_feat_dim (int): Feature dimension per plane position. Default is 32.
        out_channels (List[int]): Output channels for each intermediate layer.
        intermediate_layer_idx (List[int]): Indices of layers from aggregated tokens.
        pos_embed (bool): Whether to use positional embedding. Default is True.
        mlp_hidden_dim (int): Hidden dimension for the shared MLP. Default is 128.
    """
    
    def __init__(
        self,
        dim_in: int,
        patch_size: int = 14,
        features: int = 256,
        plane_size: int = 64,
        plane_feat_dim: int = 32,
        out_channels: List[int] = [256, 512, 1024, 1024],
        intermediate_layer_idx: List[int] = [4, 11, 17, 23],
        pos_embed: bool = True,
        mlp_hidden_dim: int = 128,
    ) -> None:
        super(LatentHead, self).__init__()
        
        self.patch_size = patch_size
        self.pos_embed = pos_embed
        self.intermediate_layer_idx = intermediate_layer_idx
        self.plane_size = plane_size
        self.plane_feat_dim = plane_feat_dim
        
        self.norm = nn.LayerNorm(dim_in)
        
        # Projection layers for each output channel from tokens
        self.projects = nn.ModuleList([
            nn.Conv2d(in_channels=dim_in, out_channels=oc, kernel_size=1, stride=1, padding=0) 
            for oc in out_channels
        ])
        
        # Resize layers for feature maps (different scales)
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0], out_channels=out_channels[0], 
                kernel_size=4, stride=4, padding=0
            ),
            nn.ConvTranspose2d(
                in_channels=out_channels[1], out_channels=out_channels[1], 
                kernel_size=2, stride=2, padding=0
            ),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3], out_channels=out_channels[3], 
                kernel_size=3, stride=2, padding=1
            ),
        ])
        
        # Scratch layers for feature fusion
        self.scratch = _make_scratch(out_channels, features, expand=False)
        
        # Fusion blocks
        self.scratch.refinenet1 = _make_fusion_block(features)
        self.scratch.refinenet2 = _make_fusion_block(features)
        self.scratch.refinenet3 = _make_fusion_block(features)
        self.scratch.refinenet4 = _make_fusion_block(features, has_residual=False)
        
        # Output convolution for fused features
        self.fused_conv = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1)
        
        # Global pooling for aggregating across frames
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Plane generators: from pooled features to each plane
        # Each plane is [plane_size, plane_size, plane_feat_dim]
        plane_total_dim = plane_size * plane_size * plane_feat_dim
        
        self.xy_plane_generator = nn.Sequential(
            nn.Linear(features, features * 2),
            nn.ReLU(inplace=True),
            nn.Linear(features * 2, plane_total_dim),
        )
        
        self.xz_plane_generator = nn.Sequential(
            nn.Linear(features, features * 2),
            nn.ReLU(inplace=True),
            nn.Linear(features * 2, plane_total_dim),
        )
        
        self.yz_plane_generator = nn.Sequential(
            nn.Linear(features, features * 2),
            nn.ReLU(inplace=True),
            nn.Linear(features * 2, plane_total_dim),
        )
        
        # Shared MLP for querying
        self.query_mlp = TriplaneMLP(hidden_dim=mlp_hidden_dim)
        
        self._init_plane_generators()
    
    def _init_plane_generators(self):
        """Initialize plane generators with small weights."""
        for generator in [self.xy_plane_generator, self.xz_plane_generator, self.yz_plane_generator]:
            # Initialize last layer with small weights
            if isinstance(generator[-1], nn.Linear):
                nn.init.normal_(generator[-1].weight, std=0.01)
                nn.init.zeros_(generator[-1].bias)
    
    def forward(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        images: torch.Tensor,
        patch_start_idx: int,
        frames_chunk_size: int = 8,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the Latent head.
        
        Args:
            aggregated_tokens_list (List[Tensor]): [B, S, P, 2C]
            images (Tensor): Input images [B, S, 3, H, W]
            patch_start_idx (int): Starting index for patch tokens.
            frames_chunk_size (int): Number of frames per chunk.
            
        Returns:
            Dict containing:
                - xy_plane: [B, plane_feat_dim, plane_size, plane_size]
                - xz_plane: [B, plane_feat_dim, plane_size, plane_size]
                - yz_plane: [B, plane_feat_dim, plane_size, plane_size]
                - query_mlp: Reference to the query MLP module
        """
        B, S, _, H, W = images.shape
        
        # Process frames in chunks and aggregate
        if frames_chunk_size is None or frames_chunk_size >= S:
            return self._forward_impl(aggregated_tokens_list, images, patch_start_idx)
        
        assert frames_chunk_size > 0
        
        all_xy = []
        all_xz = []
        all_yz = []
        
        for frames_start_idx in range(0, S, frames_chunk_size):
            frames_end_idx = min(frames_start_idx + frames_chunk_size, S)
            chunk_output = self._forward_impl(
                aggregated_tokens_list, images, patch_start_idx, 
                frames_start_idx, frames_end_idx
            )
            all_xy.append(chunk_output['xy_plane'])
            all_xz.append(chunk_output['xz_plane'])
            all_yz.append(chunk_output['yz_plane'])
        
        # Average across chunks
        return {
            'xy_plane': torch.stack(all_xy, dim=0).mean(dim=0),
            'xz_plane': torch.stack(all_xz, dim=0).mean(dim=0),
            'yz_plane': torch.stack(all_yz, dim=0).mean(dim=0),
        }
    
    def _forward_impl(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        images: torch.Tensor,
        patch_start_idx: int,
        frames_start_idx: int = None,
        frames_end_idx: int = None,
    ) -> Dict[str, torch.Tensor]:
        """Implementation of forward pass."""
        B, S, _, H, W = images.shape
        
        if frames_start_idx is None:
            frames_start_idx = 0
        if frames_end_idx is None:
            frames_end_idx = S
        
        # Get patch dimensions
        patch_h = H // self.patch_size
        patch_w = W // self.patch_size
        
        # Process intermediate layer tokens
        layer_outputs = []
        for layer_idx, layer_tokens in enumerate(aggregated_tokens_list):
            if layer_idx not in self.intermediate_layer_idx:
                continue
                
            # layer_tokens: [B, S, P, 2C]
            tokens = layer_tokens[:, frames_start_idx:frames_end_idx]
            
            # Extract patch tokens
            patch_tokens = tokens[:, :, patch_start_idx:, :]  # [B, S_chunk, num_patches, 2C]
            
            # Normalize
            patch_tokens = self.norm(patch_tokens)
            
            # Reshape for conv: [B*S_chunk, 2C, H, W]
            S_chunk = frames_end_idx - frames_start_idx
            patch_tokens = patch_tokens.reshape(B * S_chunk, patch_h, patch_w, -1)
            patch_tokens = patch_tokens.permute(0, 3, 1, 2)
            
            # Project
            idx = self.intermediate_layer_idx.index(layer_idx)
            projected = self.projects[idx](patch_tokens)
            resized = self.resize_layers[idx](projected)
            
            layer_outputs.append(resized)
        
        # Apply scratch layers
        layer_1 = self.scratch.layer1_rn(layer_outputs[0])
        layer_2 = self.scratch.layer2_rn(layer_outputs[1])
        layer_3 = self.scratch.layer3_rn(layer_outputs[2])
        layer_4 = self.scratch.layer4_rn(layer_outputs[3])
        
        # Feature fusion
        path_4 = self.scratch.refinenet4(layer_4)
        path_3 = self.scratch.refinenet3(layer_3, path_4)
        path_2 = self.scratch.refinenet2(layer_2, path_3)
        path_1 = self.scratch.refinenet1(layer_1, path_2)
        
        # Final convolution
        fused = self.fused_conv(path_1)  # [B*S_chunk, features, H', W']
        
        # Global pooling across spatial dimensions
        pooled = self.global_pool(fused)  # [B*S_chunk, features, 1, 1]
        pooled = pooled.view(B, S_chunk, -1)  # [B, S_chunk, features]
        
        # Average across frames
        pooled = pooled.mean(dim=1)  # [B, features]
        
        # Generate tri-planes
        xy_plane = self.xy_plane_generator(pooled)  # [B, plane_size*plane_size*plane_feat_dim]
        xz_plane = self.xz_plane_generator(pooled)
        yz_plane = self.yz_plane_generator(pooled)
        
        # Reshape to [B, plane_feat_dim, plane_size, plane_size]
        xy_plane = xy_plane.view(B, self.plane_size, self.plane_size, self.plane_feat_dim)
        xy_plane = xy_plane.permute(0, 3, 1, 2)
        
        xz_plane = xz_plane.view(B, self.plane_size, self.plane_size, self.plane_feat_dim)
        xz_plane = xz_plane.permute(0, 3, 1, 2)
        
        yz_plane = yz_plane.view(B, self.plane_size, self.plane_size, self.plane_feat_dim)
        yz_plane = yz_plane.permute(0, 3, 1, 2)
        
        return {
            'xy_plane': xy_plane,
            'xz_plane': xz_plane,
            'yz_plane': yz_plane,
        }
    
    def query_points(
        self,
        xy_plane: torch.Tensor,
        xz_plane: torch.Tensor,
        yz_plane: torch.Tensor,
        points: torch.Tensor,
        directions: torch.Tensor,
        pos_enc_fn: callable = None,
        dir_enc_fn: callable = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Query RGB and sigma at given 3D points.
        
        Args:
            xy_plane: [B, 32, 64, 64] - XY plane features
            xz_plane: [B, 32, 64, 64] - XZ plane features
            yz_plane: [B, 32, 64, 64] - YZ plane features
            points: [B, N, 3] - 3D points in normalized coordinates [-1, 1]
            directions: [B, N, 3] - View directions (unit vectors)
            pos_enc_fn: Positional encoding function for positions
            dir_enc_fn: Positional encoding function for directions
            
        Returns:
            sigma: [B, N, 1] - Density
            rgb: [B, N, 3] - Color
        """
        B, N, _ = points.shape
        
        # Extract coordinates
        x = points[..., 0:1]  # [B, N, 1]
        y = points[..., 1:2]
        z = points[..., 2:3]
        
        # Create grid coordinates for each plane
        # grid_sample expects [B, N, 1, 2] with values in [-1, 1]
        xy_coords = torch.cat([x, y], dim=-1).unsqueeze(2)  # [B, N, 1, 2]
        xz_coords = torch.cat([x, z], dim=-1).unsqueeze(2)
        yz_coords = torch.cat([y, z], dim=-1).unsqueeze(2)
        
        # Bilinear interpolation on each plane
        # grid_sample input: [B, C, H, W], grid: [B, N, 1, 2]
        xy_feat = F.grid_sample(xy_plane, xy_coords, mode='bilinear', 
                                padding_mode='border', align_corners=True)
        xz_feat = F.grid_sample(xz_plane, xz_coords, mode='bilinear', 
                                padding_mode='border', align_corners=True)
        yz_feat = F.grid_sample(yz_plane, yz_coords, mode='bilinear', 
                                padding_mode='border', align_corners=True)
        
        # Reshape: [B, C, N, 1] -> [B, N, C]
        xy_feat = xy_feat.squeeze(-1).permute(0, 2, 1)  # [B, N, 32]
        xz_feat = xz_feat.squeeze(-1).permute(0, 2, 1)
        yz_feat = yz_feat.squeeze(-1).permute(0, 2, 1)
        
        # Concatenate plane features
        plane_features = torch.cat([xy_feat, xz_feat, yz_feat], dim=-1)  # [B, N, 96]
        
        # Apply positional encoding
        if pos_enc_fn is not None:
            pos_encoded = pos_enc_fn(points)  # [B, N, 63]
        else:
            # Default: simple identity (will need proper encoding)
            pos_encoded = self._default_pos_enc(points)
        
        if dir_enc_fn is not None:
            dir_encoded = dir_enc_fn(directions)  # [B, N, 27]
        else:
            dir_encoded = self._default_dir_enc(directions)
        
        # Query MLP
        sigma, rgb = self.query_mlp(plane_features, pos_encoded, dir_encoded)
        
        return sigma, rgb
    
    def _default_pos_enc(self, x: torch.Tensor, L: int = 10) -> torch.Tensor:
        """Default positional encoding for positions."""
        # x: [B, N, 3]
        freqs = 2.0 ** torch.linspace(0, L-1, L, device=x.device, dtype=x.dtype)
        x_expanded = x.unsqueeze(-1) * freqs  # [B, N, 3, L]
        encoded = torch.cat([x, torch.sin(x_expanded).flatten(-2), 
                            torch.cos(x_expanded).flatten(-2)], dim=-1)
        return encoded  # [B, N, 3 + 3*L*2] = [B, N, 63]
    
    def _default_dir_enc(self, x: torch.Tensor, L: int = 4) -> torch.Tensor:
        """Default positional encoding for directions."""
        freqs = 2.0 ** torch.linspace(0, L-1, L, device=x.device, dtype=x.dtype)
        x_expanded = x.unsqueeze(-1) * freqs
        encoded = torch.cat([x, torch.sin(x_expanded).flatten(-2), 
                            torch.cos(x_expanded).flatten(-2)], dim=-1)
        return encoded  # [B, N, 3 + 3*L*2] = [B, N, 27]
    
    def get_plane_regularization_loss(
        self,
        xy_plane: torch.Tensor,
        xz_plane: torch.Tensor,
        yz_plane: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute regularization loss for tri-plane features.
        
        Encourages:
        1. Spatial smoothness (TV loss)
        2. Feature sparsity (L1)
        
        Args:
            xy_plane, xz_plane, yz_plane: [B, 32, 64, 64]
            
        Returns:
            reg_loss: Scalar regularization loss
        """
        def tv_loss(plane):
            """Total variation loss for spatial smoothness."""
            diff_h = torch.abs(plane[:, :, 1:, :] - plane[:, :, :-1, :])
            diff_w = torch.abs(plane[:, :, :, 1:] - plane[:, :, :, :-1])
            return diff_h.mean() + diff_w.mean()
        
        def l1_loss(plane):
            """L1 sparsity regularization."""
            return torch.abs(plane).mean()
        
        # TV loss for smoothness
        tv = tv_loss(xy_plane) + tv_loss(xz_plane) + tv_loss(yz_plane)
        
        # L1 for sparsity
        l1 = l1_loss(xy_plane) + l1_loss(xz_plane) + l1_loss(yz_plane)
        
        # Weighted combination
        reg_loss = 0.1 * tv + 0.01 * l1
        
        return reg_loss
