"""
Volume Rendering Module for T-NeRF Training.

This module provides differentiable volume rendering functions that can be used
with predicted NeRF MLP parameters. It supports:
1. Ray generation from camera parameters
2. Point sampling along rays (uniform and hierarchical)
3. Positional encoding
4. Volume rendering with predicted MLP
5. Loss computation against ground truth

The rendering is designed to be differentiable so gradients can flow back
to the NLPHead during training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional, List
import math


class PositionalEncoding(nn.Module):
    """
    Positional encoding for NeRF (Fourier features).
    
    Maps input coordinates to higher dimensional space using sin/cos functions.
    """
    
    def __init__(self, input_dim: int = 3, num_freqs: int = 10, include_input: bool = True):
        """
        Args:
            input_dim: Input dimension (3 for position, 3 for direction)
            num_freqs: Number of frequency bands (L in paper)
            include_input: Whether to include original input in output
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_freqs = num_freqs
        self.include_input = include_input
        
        # Compute output dimension
        self.out_dim = 0
        if include_input:
            self.out_dim += input_dim
        self.out_dim += 2 * input_dim * num_freqs  # sin and cos for each freq
        
        # Precompute frequency bands
        freq_bands = 2.0 ** torch.linspace(0, num_freqs - 1, num_freqs)
        self.register_buffer('freq_bands', freq_bands)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply positional encoding.
        
        Args:
            x: Input tensor [..., input_dim]
            
        Returns:
            Encoded tensor [..., out_dim]
        """
        encoded = []
        
        if self.include_input:
            encoded.append(x)
        
        for freq in self.freq_bands:
            encoded.append(torch.sin(x * freq * math.pi))
            encoded.append(torch.cos(x * freq * math.pi))
        
        return torch.cat(encoded, dim=-1)


class BatchedNeRFMLP(nn.Module):
    """
    Batched NeRF MLP that supports different parameters for each batch element.
    
    This module takes flattened MLP parameters and applies them to compute
    RGB and sigma values. Supports batched inference with different parameters
    per batch.
    
    Structure:
        - pos_linear: (63, 128) - processes positional encoded position
        - sigma_linear: (128, 1) - outputs density
        - color_linear: (128 + 27, 3) - outputs RGB color with direction encoding
    """
    
    # Parameter dimensions (matching nlp_head.py NeRFMLP)
    POS_IN_DIM = 63      # 3 + 3 * 2 * 10
    HIDDEN_DIM = 128
    DIR_IN_DIM = 27      # 3 + 3 * 2 * 4
    SIGMA_OUT_DIM = 1
    COLOR_OUT_DIM = 3
    
    # Parameter counts
    POS_WEIGHT_SIZE = POS_IN_DIM * HIDDEN_DIM       # 8064
    POS_BIAS_SIZE = HIDDEN_DIM                       # 128
    SIGMA_WEIGHT_SIZE = HIDDEN_DIM * SIGMA_OUT_DIM  # 128
    SIGMA_BIAS_SIZE = SIGMA_OUT_DIM                  # 1
    COLOR_IN_DIM = HIDDEN_DIM + DIR_IN_DIM          # 155
    COLOR_WEIGHT_SIZE = COLOR_IN_DIM * COLOR_OUT_DIM # 465
    COLOR_BIAS_SIZE = COLOR_OUT_DIM                  # 3
    
    TOTAL_PARAMS = (POS_WEIGHT_SIZE + POS_BIAS_SIZE + 
                    SIGMA_WEIGHT_SIZE + SIGMA_BIAS_SIZE +
                    COLOR_WEIGHT_SIZE + COLOR_BIAS_SIZE)  # 8789
    
    def __init__(self):
        super().__init__()
        # Position and direction encoders
        self.pos_encoder = PositionalEncoding(input_dim=3, num_freqs=10, include_input=True)
        self.dir_encoder = PositionalEncoding(input_dim=3, num_freqs=4, include_input=True)
    
    def forward(
        self, 
        positions: torch.Tensor, 
        directions: torch.Tensor,
        params: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with batched parameters.
        
        Args:
            positions: Position coordinates [B, N, 3]
            directions: View directions [B, N, 3]
            params: MLP parameters [B, TOTAL_PARAMS]
            
        Returns:
            sigma: Density values [B, N, 1]
            rgb: Color values [B, N, 3]
        """
        B, N, _ = positions.shape
        device = positions.device
        
        # Positional encoding
        pos_encoded = self.pos_encoder(positions)  # [B, N, 63]
        dir_encoded = self.dir_encoder(directions)  # [B, N, 27]
        
        # Extract parameters for each layer
        idx = 0
        
        # pos_linear weights and bias
        pos_weight = params[:, idx:idx + self.POS_WEIGHT_SIZE]
        pos_weight = pos_weight.view(B, self.HIDDEN_DIM, self.POS_IN_DIM)
        idx += self.POS_WEIGHT_SIZE
        
        pos_bias = params[:, idx:idx + self.POS_BIAS_SIZE]
        idx += self.POS_BIAS_SIZE
        
        # sigma_linear weights and bias
        sigma_weight = params[:, idx:idx + self.SIGMA_WEIGHT_SIZE]
        sigma_weight = sigma_weight.view(B, self.SIGMA_OUT_DIM, self.HIDDEN_DIM)
        idx += self.SIGMA_WEIGHT_SIZE
        
        sigma_bias = params[:, idx:idx + self.SIGMA_BIAS_SIZE]
        idx += self.SIGMA_BIAS_SIZE
        
        # color_linear weights and bias
        color_weight = params[:, idx:idx + self.COLOR_WEIGHT_SIZE]
        color_weight = color_weight.view(B, self.COLOR_OUT_DIM, self.COLOR_IN_DIM)
        idx += self.COLOR_WEIGHT_SIZE
        
        color_bias = params[:, idx:idx + self.COLOR_BIAS_SIZE]
        
        # Forward pass through layers (batched linear operations)
        # h = ReLU(pos_linear(pos_encoded))
        h = torch.einsum('bnd,bhd->bnh', pos_encoded, pos_weight)  # [B, N, HIDDEN_DIM]
        h = h + pos_bias.unsqueeze(1)
        h = F.relu(h)
        
        # sigma = sigma_linear(h)
        sigma = torch.einsum('bnh,boh->bno', h, sigma_weight)  # [B, N, 1]
        sigma = sigma + sigma_bias.unsqueeze(1)
        
        # color = sigmoid(color_linear([h, dir_encoded]))
        h_with_dir = torch.cat([h, dir_encoded], dim=-1)  # [B, N, HIDDEN_DIM + DIR_IN_DIM]
        rgb = torch.einsum('bnc,boc->bno', h_with_dir, color_weight)  # [B, N, 3]
        rgb = rgb + color_bias.unsqueeze(1)
        rgb = torch.sigmoid(rgb)
        
        return sigma, rgb


class VolumeRenderer(nn.Module):
    """
    Differentiable volume renderer for NeRF.
    
    Renders images from 3D scenes using predicted MLP parameters.
    Supports:
    - Ray generation from camera intrinsics/extrinsics
    - Stratified sampling along rays
    - Volume rendering integration
    """
    
    def __init__(
        self,
        num_samples: int = 64,
        num_importance_samples: int = 0,
        near: float = 0.1,
        far: float = 10.0,
        white_background: bool = True,
    ):
        """
        Args:
            num_samples: Number of coarse samples per ray
            num_importance_samples: Number of fine samples (hierarchical sampling)
            near: Near plane distance
            far: Far plane distance
            white_background: Whether to use white background
        """
        super().__init__()
        self.num_samples = num_samples
        self.num_importance_samples = num_importance_samples
        self.near = near
        self.far = far
        self.white_background = white_background
        
        self.nerf_mlp = BatchedNeRFMLP()
    
    def forward(
        self,
        params: torch.Tensor,
        camera_poses: torch.Tensor,
        camera_intrinsics: torch.Tensor,
        image_size: Tuple[int, int],
        bbox_min: Optional[torch.Tensor] = None,
        bbox_max: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Render images using predicted MLP parameters.
        
        Args:
            params: MLP parameters [B, TOTAL_PARAMS]
            camera_poses: Camera-to-world matrices [B, S, 4, 4] or [B, 4, 4]
            camera_intrinsics: Camera intrinsics [B, 3, 3] or [B, S, 3, 3]
            image_size: Output image size (H, W)
            bbox_min, bbox_max: Scene bounding box (optional)
            
        Returns:
            Dictionary containing:
                - 'rgb': Rendered images [B, S, 3, H, W] or [B, 3, H, W]
                - 'depth': Depth maps
                - 'acc': Accumulated alpha
        """
        B = params.shape[0]
        H, W = image_size
        
        # Handle single view vs multi-view
        if camera_poses.dim() == 3:  # [B, 4, 4]
            camera_poses = camera_poses.unsqueeze(1)  # [B, 1, 4, 4]
        
        if camera_intrinsics.dim() == 2:  # [3, 3]
            camera_intrinsics = camera_intrinsics.unsqueeze(0).expand(B, -1, -1)
        if camera_intrinsics.dim() == 3:  # [B, 3, 3]
            camera_intrinsics = camera_intrinsics.unsqueeze(1)  # [B, 1, 3, 3]
        
        S = camera_poses.shape[1]
        device = params.device
        
        all_rgb = []
        all_depth = []
        all_acc = []
        
        for s in range(S):
            pose = camera_poses[:, s]  # [B, 4, 4]
            K = camera_intrinsics[:, min(s, camera_intrinsics.shape[1] - 1)]  # [B, 3, 3]
            
            # Generate rays
            rays_o, rays_d = self._generate_rays(pose, K, H, W)  # [B, H*W, 3]
            
            # Compute near/far based on bbox if provided
            if bbox_min is not None and bbox_max is not None:
                near, far = self._compute_near_far(rays_o, rays_d, bbox_min, bbox_max)
            else:
                near = torch.ones(B, H * W, 1, device=device) * self.near
                far = torch.ones(B, H * W, 1, device=device) * self.far
            
            # Sample points along rays
            z_vals, pts = self._sample_points(rays_o, rays_d, near, far)
            
            # Expand directions for all sample points
            dirs = rays_d.unsqueeze(2).expand(-1, -1, self.num_samples, -1)  # [B, H*W, num_samples, 3]
            dirs = dirs.reshape(B, -1, 3)  # [B, H*W*num_samples, 3]
            pts_flat = pts.reshape(B, -1, 3)  # [B, H*W*num_samples, 3]
            
            # Query MLP
            sigma, rgb = self.nerf_mlp(pts_flat, dirs, params)
            
            # Reshape back
            sigma = sigma.view(B, H * W, self.num_samples, 1)
            rgb = rgb.view(B, H * W, self.num_samples, 3)
            
            # Volume rendering
            rendered = self._volume_render(rgb, sigma, z_vals)
            
            rgb_map = rendered['rgb'].view(B, H, W, 3).permute(0, 3, 1, 2)  # [B, 3, H, W]
            depth_map = rendered['depth'].view(B, H, W, 1).permute(0, 3, 1, 2)
            acc_map = rendered['acc'].view(B, H, W, 1).permute(0, 3, 1, 2)
            
            all_rgb.append(rgb_map)
            all_depth.append(depth_map)
            all_acc.append(acc_map)
        
        # Stack across views
        rgb = torch.stack(all_rgb, dim=1)  # [B, S, 3, H, W]
        depth = torch.stack(all_depth, dim=1)
        acc = torch.stack(all_acc, dim=1)
        
        if S == 1:
            rgb = rgb.squeeze(1)
            depth = depth.squeeze(1)
            acc = acc.squeeze(1)
        
        return {
            'rgb': rgb,
            'depth': depth,
            'acc': acc,
        }
    
    def _generate_rays(
        self,
        camera_pose: torch.Tensor,
        intrinsics: torch.Tensor,
        H: int,
        W: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate rays for all pixels.
        
        Args:
            camera_pose: Camera-to-world matrix [B, 4, 4]
            intrinsics: Camera intrinsics [B, 3, 3]
            H, W: Image dimensions
            
        Returns:
            rays_o: Ray origins [B, H*W, 3]
            rays_d: Ray directions [B, H*W, 3]
        """
        B = camera_pose.shape[0]
        device = camera_pose.device
        
        # Pixel coordinates
        i, j = torch.meshgrid(
            torch.arange(H, dtype=torch.float32, device=device),
            torch.arange(W, dtype=torch.float32, device=device),
            indexing='ij'
        )
        
        # Flatten and batch
        i = i.reshape(-1).unsqueeze(0).expand(B, -1)  # [B, H*W]
        j = j.reshape(-1).unsqueeze(0).expand(B, -1)
        
        # Camera space ray directions
        fx = intrinsics[:, 0, 0].unsqueeze(1)
        fy = intrinsics[:, 1, 1].unsqueeze(1)
        cx = intrinsics[:, 0, 2].unsqueeze(1)
        cy = intrinsics[:, 1, 2].unsqueeze(1)
        
        dirs = torch.stack([
            (j - cx) / fx,
            -(i - cy) / fy,
            -torch.ones_like(i)
        ], dim=-1)  # [B, H*W, 3]
        
        # Transform to world space
        rotation = camera_pose[:, :3, :3]  # [B, 3, 3]
        rays_d = torch.einsum('bij,bnj->bni', rotation, dirs)  # [B, H*W, 3]
        rays_d = F.normalize(rays_d, dim=-1)
        
        # Ray origins (camera position)
        rays_o = camera_pose[:, :3, 3].unsqueeze(1).expand(-1, H * W, -1)  # [B, H*W, 3]
        
        return rays_o, rays_d
    
    def _compute_near_far(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        bbox_min: torch.Tensor,
        bbox_max: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute near/far from bbox intersection."""
        B, N, _ = rays_o.shape
        
        inv_d = 1.0 / (rays_d + 1e-10)
        
        # Ensure bbox has batch dimension
        if bbox_min.dim() == 1:
            bbox_min = bbox_min.unsqueeze(0).expand(B, -1)
            bbox_max = bbox_max.unsqueeze(0).expand(B, -1)
        
        # Expand bbox to [B, 1, 3] for broadcasting with rays
        bbox_min = bbox_min.unsqueeze(1)  # [B, 1, 3]
        bbox_max = bbox_max.unsqueeze(1)
        
        t_min = (bbox_min - rays_o) * inv_d
        t_max = (bbox_max - rays_o) * inv_d
        
        t1 = torch.minimum(t_min, t_max)
        t2 = torch.maximum(t_min, t_max)
        
        near = torch.max(t1, dim=-1, keepdim=True)[0]
        far = torch.min(t2, dim=-1, keepdim=True)[0]
        
        near = torch.clamp(near, min=0.1)
        far = torch.clamp(far, min=near + 0.1)
        
        return near, far
    
    def _sample_points(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        near: torch.Tensor,
        far: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample points along rays.
        
        Args:
            rays_o: Ray origins [B, N, 3]
            rays_d: Ray directions [B, N, 3]
            near, far: Near/far planes [B, N, 1]
            
        Returns:
            z_vals: Sample distances [B, N, num_samples]
            pts: Sample points [B, N, num_samples, 3]
        """
        B, N, _ = rays_o.shape
        device = rays_o.device
        
        # Stratified sampling
        t_vals = torch.linspace(0, 1, self.num_samples, device=device)
        z_vals = near + (far - near) * t_vals  # [B, N, num_samples]
        
        # Add noise for training
        if self.training:
            mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], dim=-1)
            lower = torch.cat([z_vals[..., :1], mids], dim=-1)
            z_vals = lower + (upper - lower) * torch.rand_like(z_vals)
        
        # Compute 3D points
        pts = rays_o.unsqueeze(2) + rays_d.unsqueeze(2) * z_vals.unsqueeze(-1)
        
        return z_vals, pts
    
    def _volume_render(
        self,
        rgb: torch.Tensor,
        sigma: torch.Tensor,
        z_vals: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Perform volume rendering.
        
        Args:
            rgb: Color values [B, N, num_samples, 3]
            sigma: Density values [B, N, num_samples, 1]
            z_vals: Sample distances [B, N, num_samples]
            
        Returns:
            Dictionary with 'rgb', 'depth', 'acc'
        """
        # Compute distances
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.ones_like(dists[..., :1]) * 1e10], dim=-1)
        
        # Alpha from sigma
        sigma = sigma.squeeze(-1)  # [B, N, num_samples]
        alpha = 1.0 - torch.exp(-F.relu(sigma) * dists)
        
        # Transmittance
        ones = torch.ones_like(alpha[..., :1])
        transmittance = torch.cumprod(torch.cat([ones, 1.0 - alpha + 1e-10], dim=-1), dim=-1)[..., :-1]
        
        # Weights
        weights = alpha * transmittance
        
        # RGB
        rgb_map = torch.sum(weights.unsqueeze(-1) * rgb, dim=-2)
        
        # Depth
        depth_map = torch.sum(weights * z_vals, dim=-1)
        
        # Accumulated alpha
        acc_map = torch.sum(weights, dim=-1)
        
        # White background
        if self.white_background:
            rgb_map = rgb_map + (1.0 - acc_map.unsqueeze(-1))
        
        return {
            'rgb': rgb_map,
            'depth': depth_map,
            'acc': acc_map,
        }


class VoxelRenderer(nn.Module):
    """
    Differentiable renderer that directly samples from voxel grids.
    
    Used for computing ground truth images from NeRF-MAE voxel data
    and for supervision during training.
    """
    
    def __init__(
        self,
        num_samples: int = 128,
        white_background: bool = True,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.white_background = white_background
    
    def forward(
        self,
        rgbsigma: torch.Tensor,
        camera_pose: torch.Tensor,
        camera_intrinsics: torch.Tensor,
        bbox_min: torch.Tensor,
        bbox_max: torch.Tensor,
        image_size: Tuple[int, int],
    ) -> Dict[str, torch.Tensor]:
        """
        Render image from voxel grid.
        
        Args:
            rgbsigma: Voxel grid [Hv, Wv, Dv, 4] or [B, Hv, Wv, Dv, 4]
            camera_pose: Camera-to-world [4, 4] or [B, 4, 4]
            camera_intrinsics: Camera K [3, 3] or [B, 3, 3]
            bbox_min, bbox_max: Bounding box [3] or [B, 3]
            image_size: (H, W)
            
        Returns:
            'rgb': [3, H, W] or [B, 3, H, W]
        """
        # Add batch dimension if needed
        batched = camera_pose.dim() == 3
        if not batched:
            rgbsigma = rgbsigma.unsqueeze(0)
            camera_pose = camera_pose.unsqueeze(0)
            camera_intrinsics = camera_intrinsics.unsqueeze(0)
            bbox_min = bbox_min.unsqueeze(0)
            bbox_max = bbox_max.unsqueeze(0)
        
        B = camera_pose.shape[0]
        H, W = image_size
        device = camera_pose.device
        
        # Generate rays
        rays_o, rays_d = self._generate_rays(camera_pose, camera_intrinsics, H, W)
        
        # Compute near/far
        near, far = self._compute_near_far(rays_o, rays_d, bbox_min, bbox_max)
        
        # Sample points
        z_vals, pts = self._sample_points(rays_o, rays_d, near, far)
        
        # Query voxel grid
        rgb, sigma = self._query_voxel(pts, rgbsigma, bbox_min, bbox_max)
        
        # Volume render
        result = self._volume_render(rgb, sigma, z_vals)
        
        rgb_map = result['rgb'].view(B, H, W, 3).permute(0, 3, 1, 2)
        
        if not batched:
            rgb_map = rgb_map.squeeze(0)
        
        return {'rgb': rgb_map}
    
    def _generate_rays(self, pose, K, H, W):
        """Generate rays similar to VolumeRenderer."""
        B = pose.shape[0]
        device = pose.device
        
        i, j = torch.meshgrid(
            torch.arange(H, dtype=torch.float32, device=device),
            torch.arange(W, dtype=torch.float32, device=device),
            indexing='ij'
        )
        
        i = i.reshape(-1).unsqueeze(0).expand(B, -1)
        j = j.reshape(-1).unsqueeze(0).expand(B, -1)
        
        fx = K[:, 0, 0].unsqueeze(1)
        fy = K[:, 1, 1].unsqueeze(1)
        cx = K[:, 0, 2].unsqueeze(1)
        cy = K[:, 1, 2].unsqueeze(1)
        
        dirs = torch.stack([
            (j - cx) / fx,
            -(i - cy) / fy,
            -torch.ones_like(i)
        ], dim=-1)
        
        rotation = pose[:, :3, :3]
        rays_d = torch.einsum('bij,bnj->bni', rotation, dirs)
        rays_d = F.normalize(rays_d, dim=-1)
        
        rays_o = pose[:, :3, 3].unsqueeze(1).expand(-1, H * W, -1)
        
        return rays_o, rays_d
    
    def _compute_near_far(self, rays_o, rays_d, bbox_min, bbox_max):
        B, N, _ = rays_o.shape
        inv_d = 1.0 / (rays_d + 1e-10)
        
        # Ensure bbox has batch dimension
        if bbox_min.dim() == 1:
            bbox_min = bbox_min.unsqueeze(0).expand(B, -1)
            bbox_max = bbox_max.unsqueeze(0).expand(B, -1)
        
        bbox_min = bbox_min.unsqueeze(1)
        bbox_max = bbox_max.unsqueeze(1)
        
        t_min = (bbox_min - rays_o) * inv_d
        t_max = (bbox_max - rays_o) * inv_d
        
        t1 = torch.minimum(t_min, t_max)
        t2 = torch.maximum(t_min, t_max)
        
        near = torch.max(t1, dim=-1, keepdim=True)[0]
        far = torch.min(t2, dim=-1, keepdim=True)[0]
        
        near = torch.clamp(near, min=0.1)
        far = torch.clamp(far, min=near + 0.1)
        
        return near, far
    
    def _sample_points(self, rays_o, rays_d, near, far):
        B, N, _ = rays_o.shape
        device = rays_o.device
        
        t_vals = torch.linspace(0, 1, self.num_samples, device=device)
        z_vals = near + (far - near) * t_vals
        
        pts = rays_o.unsqueeze(2) + rays_d.unsqueeze(2) * z_vals.unsqueeze(-1)
        
        return z_vals, pts
    
    def _query_voxel(self, pts, rgbsigma, bbox_min, bbox_max):
        """Query voxel grid using trilinear interpolation."""
        B, N, M, _ = pts.shape
        
        # Normalize to [-1, 1]
        bbox_min = bbox_min.view(B, 1, 1, 3)
        bbox_max = bbox_max.view(B, 1, 1, 3)
        pts_norm = 2.0 * (pts - bbox_min) / (bbox_max - bbox_min + 1e-10) - 1.0
        
        # Reshape for grid_sample
        pts_grid = pts_norm.view(B, N * M, 1, 1, 3)
        
        # Prepare voxel grid [B, 4, D, H, W]
        voxel = rgbsigma.permute(0, 4, 3, 1, 2)
        
        # Sample
        sampled = F.grid_sample(
            voxel, pts_grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        )
        
        sampled = sampled.view(B, 4, N, M).permute(0, 2, 3, 1)  # [B, N, M, 4]
        
        rgb = sampled[..., :3]
        sigma = sampled[..., 3:4]
        
        return rgb, sigma
    
    def _volume_render(self, rgb, sigma, z_vals):
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.ones_like(dists[..., :1]) * 1e10], dim=-1)
        
        sigma = sigma.squeeze(-1)
        alpha = 1.0 - torch.exp(-F.relu(sigma) * dists)
        
        ones = torch.ones_like(alpha[..., :1])
        transmittance = torch.cumprod(torch.cat([ones, 1.0 - alpha + 1e-10], dim=-1), dim=-1)[..., :-1]
        
        weights = alpha * transmittance
        rgb_map = torch.sum(weights.unsqueeze(-1) * rgb, dim=-2)
        
        acc_map = torch.sum(weights, dim=-1)
        if self.white_background:
            rgb_map = rgb_map + (1.0 - acc_map.unsqueeze(-1))
        
        return {'rgb': rgb_map}
