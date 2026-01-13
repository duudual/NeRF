"""
Patch Matching Interpolation for Dynamic T-NeRF (v2).

This module implements a novel interpolation scheme that operates at the 
DINOv2 patch embedding level:

1. Separately extract DINOv2 patch tokens for t0 and t1 images
2. Find corresponding patches between t0 and t1 using cosine similarity in local neighborhood
3. Linearly interpolate both 2D positions and features based on alpha
4. Use bilinear splatting to redistribute features to grid positions
5. Run the aggregator's remaining blocks on the interpolated patch tokens
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import math


class PatchEmbeddingInterpolator:
    """
    Patch matching based interpolation at the DINOv2 embedding level.
    
    Key idea:
    - Extract DINOv2 patch embeddings for both t0 and t1
    - For each patch at t0, find the most similar patch in t1's neighborhood (cosine similarity)
    - Interpolate both 2D positions and features linearly based on alpha
    - Use bilinear splatting to place features on a regular grid
    - Normalize by accumulated weights
    
    Args:
        patch_size: Size of each patch (default 14 for DINOv2)
        neighborhood_size: Size of the search neighborhood in t1 (default 5, meaning 5x5)
        embed_dim: Dimension of patch embeddings
    """
    
    def __init__(
        self,
        patch_size: int = 14,
        neighborhood_size: int = 5,
        embed_dim: int = 1024,
        device: str = 'cuda'
    ):
        self.patch_size = patch_size
        self.neighborhood_size = neighborhood_size
        self.embed_dim = embed_dim
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Cache for patch embeddings and metadata
        # Using Dict with Any type to avoid type checking issues
        self._cache: Dict[str, Optional[torch.Tensor]] = {
            'patches_t0': None,       # DINOv2 patch tokens at t0: (B*S, P, C)
            'patches_t1': None,       # DINOv2 patch tokens at t1: (B*S, P, C)
            'B': None,  # type: ignore
            'S': None,  # type: ignore
            'H': None,  # type: ignore
            'W': None,  # type: ignore
            'grid_h': None,  # type: ignore
            'grid_w': None,  # type: ignore
            'positions_t0': None,     # Base patch positions
            'correspondences': None,  # Matched indices
            'offset_y': None,         # Y offsets from matching
            'offset_x': None,         # X offsets from matching
        }
    
    def clear_cache(self):
        """Clear the cached features."""
        for key in self._cache:
            self._cache[key] = None
        torch.cuda.empty_cache()
    
    def _compute_patch_positions(self, grid_h: int, grid_w: int) -> torch.Tensor:
        """
        Compute the 2D center positions of each patch in normalized coordinates [0, 1].
        
        Returns:
            positions: Tensor of shape (grid_h * grid_w, 2) with (y, x) coordinates
        """
        y_coords = (torch.arange(grid_h, device=self.device) + 0.5) / grid_h
        x_coords = (torch.arange(grid_w, device=self.device) + 0.5) / grid_w
        
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        positions = torch.stack([yy.flatten(), xx.flatten()], dim=-1)  # (H*W, 2)
        
        return positions
    
    def _find_correspondences(
        self,
        patches_t0: torch.Tensor,
        patches_t1: torch.Tensor,
        grid_h: int,
        grid_w: int,
        num_views: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        For each patch in t0, find the most similar patch in t1's neighborhood.
        
        Args:
            patches_t0: Patch tokens at t0, shape (B*S, P, C)
            patches_t1: Patch tokens at t1, shape (B*S, P, C)
            grid_h: Number of patches in height per image
            grid_w: Number of patches in width per image
            num_views: Number of views (S)
            
        Returns:
            correspondences: Index of best matching patch in t1 for each t0 patch
            similarities: Cosine similarity scores
            offset_y: Y offset in patch units
            offset_x: X offset in patch units
        """
        BS, P, C = patches_t0.shape
        B = BS // num_views
        S = num_views
        
        half_neigh = self.neighborhood_size // 2
        
        # Reshape to (B, S, H, W, C) for neighborhood access
        patches_t0_spatial = patches_t0.reshape(B, S, grid_h, grid_w, C)
        patches_t1_spatial = patches_t1.reshape(B, S, grid_h, grid_w, C)
        
        # Pad t1 for neighborhood access
        patches_t1_padded = F.pad(
            patches_t1_spatial.permute(0, 1, 4, 2, 3),  # (B, S, C, H, W)
            (half_neigh, half_neigh, half_neigh, half_neigh,0,0),
            mode='replicate'
        ).permute(0, 1, 3, 4, 2)  # (B, S, H+pad, W+pad, C)
        
        # Initialize outputs
        corr_spatial = torch.zeros(B, S, grid_h, grid_w, dtype=torch.long, device=self.device)
        sim_spatial = torch.zeros(B, S, grid_h, grid_w, device=self.device)
        offset_y = torch.zeros(B, S, grid_h, grid_w, device=self.device)
        offset_x = torch.zeros(B, S, grid_h, grid_w, device=self.device)
        
        for i in range(grid_h):
            for j in range(grid_w):
                # Extract neighborhood from t1
                neighborhood = patches_t1_padded[:, :, i:i+self.neighborhood_size, j:j+self.neighborhood_size, :]
                # Shape: (B, S, neigh_size, neigh_size, C)
                
                # Get t0 patch at this position
                patch_t0 = patches_t0_spatial[:, :, i, j, :]  # (B, S, C)
                patch_t0_norm = F.normalize(patch_t0, dim=-1)
                
                # Reshape neighborhood for batch similarity
                neigh_flat = neighborhood.reshape(B, S, -1, C)  # (B, S, neigh_size^2, C)
                neigh_norm = F.normalize(neigh_flat, dim=-1)
                
                # Compute cosine similarity
                sims = torch.einsum('bsc,bsnc->bsn', patch_t0_norm, neigh_norm)  # (B, S, neigh_size^2)
                
                # Find best match
                best_idx = sims.argmax(dim=-1)  # (B, S)
                best_sim = sims.max(dim=-1).values  # (B, S)
                
                # Convert local index to offset
                local_y = best_idx // self.neighborhood_size
                local_x = best_idx % self.neighborhood_size
                
                # Offset from current position (in patch units)
                dy = (local_y.float() - half_neigh)
                dx = (local_x.float() - half_neigh)
                
                # Global index in t1
                global_y = torch.clamp(i + local_y - half_neigh, 0, grid_h - 1)
                global_x = torch.clamp(j + local_x - half_neigh, 0, grid_w - 1)
                global_idx = global_y * grid_w + global_x
                
                corr_spatial[:, :, i, j] = global_idx
                sim_spatial[:, :, i, j] = best_sim
                offset_y[:, :, i, j] = dy
                offset_x[:, :, i, j] = dx
        
        # Flatten to (B, S*grid_h*grid_w)
        correspondences = corr_spatial.reshape(B, S * grid_h * grid_w)
        similarities = sim_spatial.reshape(B, S * grid_h * grid_w)
        
        return correspondences, similarities, offset_y, offset_x
    
    def _bilinear_splat(
        self,
        features: torch.Tensor,
        positions: torch.Tensor,
        grid_h: int,
        grid_w: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Bilinear splatting: distribute features to the 4 nearest grid positions.
        
        Args:
            features: Features to splat, shape (B, S, N, C)
            positions: Normalized 2D positions in [0, 1], shape (B, S, N, 2)
            grid_h: Height of output grid
            grid_w: Width of output grid
            
        Returns:
            splatted_features: Accumulated features on grid, shape (B, S, grid_h*grid_w, C)
            weight_grid: Accumulated weights, shape (B, S, grid_h*grid_w, 1)
        """
        B, S, N, C = features.shape
        
        # Convert normalized positions to grid coordinates
        pos_y = positions[:, :, :, 0] * (grid_h - 1)  # (B, S, N)
        pos_x = positions[:, :, :, 1] * (grid_w - 1)  # (B, S, N)
        
        # Get 4 corner indices
        y0 = pos_y.floor().long().clamp(0, grid_h - 1)
        y1 = (y0 + 1).clamp(0, grid_h - 1)
        x0 = pos_x.floor().long().clamp(0, grid_w - 1)
        x1 = (x0 + 1).clamp(0, grid_w - 1)
        
        # Compute weights
        wy1 = pos_y - y0.float()
        wy0 = 1.0 - wy1
        wx1 = pos_x - x0.float()
        wx0 = 1.0 - wx1
        
        # Four corner weights
        w00 = wy0 * wx0  # (B, S, N)
        w01 = wy0 * wx1
        w10 = wy1 * wx0
        w11 = wy1 * wx1
        
        # Initialize output grids
        splatted = torch.zeros(B, S, grid_h * grid_w, C, device=self.device)
        weights = torch.zeros(B, S, grid_h * grid_w, 1, device=self.device)
        
        # Convert 2D indices to 1D
        idx00 = y0 * grid_w + x0  # (B, S, N)
        idx01 = y0 * grid_w + x1
        idx10 = y1 * grid_w + x0
        idx11 = y1 * grid_w + x1
        
        # Weighted features
        feat_w00 = features * w00.unsqueeze(-1)  # (B, S, N, C)
        feat_w01 = features * w01.unsqueeze(-1)
        feat_w10 = features * w10.unsqueeze(-1)
        feat_w11 = features * w11.unsqueeze(-1)
        
        # Scatter add to grid
        for b in range(B):
            for s in range(S):
                splatted[b, s].scatter_add_(0, idx00[b, s].unsqueeze(-1).expand(-1, C), feat_w00[b, s])
                splatted[b, s].scatter_add_(0, idx01[b, s].unsqueeze(-1).expand(-1, C), feat_w01[b, s])
                splatted[b, s].scatter_add_(0, idx10[b, s].unsqueeze(-1).expand(-1, C), feat_w10[b, s])
                splatted[b, s].scatter_add_(0, idx11[b, s].unsqueeze(-1).expand(-1, C), feat_w11[b, s])
                
                weights[b, s].scatter_add_(0, idx00[b, s].unsqueeze(-1), w00[b, s].unsqueeze(-1))
                weights[b, s].scatter_add_(0, idx01[b, s].unsqueeze(-1), w01[b, s].unsqueeze(-1))
                weights[b, s].scatter_add_(0, idx10[b, s].unsqueeze(-1), w10[b, s].unsqueeze(-1))
                weights[b, s].scatter_add_(0, idx11[b, s].unsqueeze(-1), w11[b, s].unsqueeze(-1))
        
        return splatted, weights
    
    def cache_patch_embeddings(
        self,
        patches_t0: torch.Tensor,
        patches_t1: torch.Tensor,
        B: int,
        S: int,
        H: int,
        W: int
    ):
        """
        Cache the DINOv2 patch embeddings for both time points.
        
        Args:
            patches_t0: DINOv2 patch tokens at t0, shape (B*S, P, C)
            patches_t1: DINOv2 patch tokens at t1, shape (B*S, P, C)
            B: Batch size
            S: Number of views
            H: Image height
            W: Image width
        """
        grid_h = H // self.patch_size
        grid_w = W // self.patch_size
        
        self._cache['patches_t0'] = patches_t0
        self._cache['patches_t1'] = patches_t1
        self._cache['B'] = B
        self._cache['S'] = S
        self._cache['H'] = H
        self._cache['W'] = W
        self._cache['grid_h'] = grid_h
        self._cache['grid_w'] = grid_w
        
        # Compute base positions
        positions = self._compute_patch_positions(grid_h, grid_w)
        self._cache['positions_t0'] = positions
        
        # Find correspondences
        correspondences, similarities, offset_y, offset_x = self._find_correspondences(
            patches_t0, patches_t1, grid_h, grid_w, S
        )
        self._cache['correspondences'] = correspondences
        self._cache['similarities'] = similarities
        self._cache['offset_y'] = offset_y
        self._cache['offset_x'] = offset_x
        
        print(f"[PatchEmbedInterpolator] Cached: {S} views, {grid_h}x{grid_w} patches", flush=True)
        print(f"[PatchEmbedInterpolator] Mean similarity: {similarities.mean().item():.4f}", flush=True)
    
    def interpolate(self, alpha: float) -> torch.Tensor:
        """
        Interpolate patch embeddings between cached t0 and t1.
        
        Args:
            alpha: Interpolation factor in [0, 1], where 0 = t0, 1 = t1
            
        Returns:
            patches_interp: Interpolated patch tokens, shape (B*S, P, C)
        """
        if self._cache['patches_t0'] is None:
            raise RuntimeError("Must call cache_patch_embeddings() before interpolate()")
        
        alpha = max(0.0, min(1.0, alpha))
        
        patches_t0 = self._cache['patches_t0']
        patches_t1 = self._cache['patches_t1']
        B = self._cache['B']
        S = self._cache['S']
        grid_h = self._cache['grid_h']
        grid_w = self._cache['grid_w']
        correspondences = self._cache['correspondences']
        offset_y = self._cache['offset_y']
        offset_x = self._cache['offset_x']
        positions_t0 = self._cache['positions_t0']
        
        BS, P, C = patches_t0.shape
        num_patches_per_view = grid_h * grid_w
        
        # Reshape to (B, S, P, C)
        patches_t0_view = patches_t0.reshape(B, S, P, C)
        patches_t1_view = patches_t1.reshape(B, S, P, C)
        
        # Get corresponding t1 features for each t0 patch
        corr_view = correspondences.reshape(B, S, num_patches_per_view)
        
        # Gather corresponding t1 features
        patches_t1_matched = torch.zeros_like(patches_t0_view)
        for b in range(B):
            for v in range(S):
                patches_t1_matched[b, v] = patches_t1_view[b, v][corr_view[b, v]]
        
        # Linearly interpolate features
        features_interp = (1 - alpha) * patches_t0_view + alpha * patches_t1_matched
        
        # Compute interpolated positions
        # offset_y, offset_x shape: (B, S, grid_h, grid_w)
        offset_y_flat = offset_y.reshape(B, S, num_patches_per_view)
        offset_x_flat = offset_x.reshape(B, S, num_patches_per_view)
        offsets = torch.stack([offset_y_flat / grid_h, offset_x_flat / grid_w], dim=-1)  # (B, S, N, 2)
        
        # Expand base positions
        pos_t0_expanded = positions_t0.unsqueeze(0).unsqueeze(0).expand(B, S, -1, -1)  # (B, S, N, 2)
        
        # Interpolated position = t0_pos + alpha * offset
        positions_interp = pos_t0_expanded + alpha * offsets
        positions_interp = positions_interp.clamp(0, 1)
        
        # Bilinear splatting
        splatted, weights = self._bilinear_splat(features_interp, positions_interp, grid_h, grid_w)
        
        # Normalize by weights
        zero_weight_mask = (weights < 1e-6).squeeze(-1)  # (B, S, num_patches)
        
        weights_safe = weights.clamp(min=1e-6)
        patches_interp = splatted / weights_safe
        
        # Fill zero-weight positions with fallback
        if alpha < 0.5:
            fill_values = patches_t0_view
        else:
            fill_values = patches_t1_view
        
        patches_interp = torch.where(zero_weight_mask.unsqueeze(-1), fill_values, patches_interp)
        
        # Reshape back to (B*S, P, C)
        patches_interp = patches_interp.reshape(BS, P, C)
        
        return patches_interp
    
    def interpolate_linear(self, alpha: float) -> torch.Tensor:
        """
        Simple linear interpolation of patch embeddings (baseline).
        
        Args:
            alpha: Interpolation factor in [0, 1]
            
        Returns:
            patches_interp: Interpolated patch tokens, shape (B*S, P, C)
        """
        if self._cache['patches_t0'] is None:
            raise RuntimeError("Must call cache_patch_embeddings() before interpolate_linear()")
        
        patches_t0 = self._cache['patches_t0']
        patches_t1 = self._cache['patches_t1']
        
        alpha = max(0.0, min(1.0, alpha))
        return (1 - alpha) * patches_t0 + alpha * patches_t1
