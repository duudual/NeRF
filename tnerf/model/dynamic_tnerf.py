"""
Dynamic T-NeRF helper for temporal interpolation with VGGT.

Given two time steps (multi-view image sets), this module encodes each time
step with VGGT, interpolates the DINOv2 patch embeddings using various schemes,
and runs the aggregator + heads to produce intermediate outputs.

Interpolation Modes:
- 'linear': Simple linear interpolation at the patch embedding level
- 'patch_matching': Novel scheme using cosine similarity for patch correspondence,
                   with bilinear splatting for position-aware interpolation

Key Architecture Change (v2):
The interpolation now happens BEFORE the aggregator's alternating attention,
at the DINOv2 patch embedding level. This means:
1. Extract DINOv2 patch tokens for both t0 and t1
2. Interpolate patch tokens (linear or patch_matching)
3. Run aggregator's forward_from_patches() on interpolated tokens
4. Run heads on the aggregated tokens
"""

import os
from typing import Dict, List, Union, Optional
import numpy as np

import torch
import torch.nn as nn

from .models.vggt import VGGT as TNeRFVGGT
from .utils.load_fn import load_and_preprocess_images
from .utils.pose_enc import pose_encoding_to_extri_intri
from .patch_matching_interpolation_v2 import PatchEmbeddingInterpolator


class DynamicVGGT:
    """
    Lightweight wrapper around VGGT for temporal interpolation at patch embedding level.
    
    Supports multiple interpolation modes:
    - 'linear': Simple linear interpolation of DINOv2 patch embeddings
    - 'patch_matching': Patch correspondence + bilinear splatting at embedding level
    
    Key difference from v1: Interpolation happens at DINOv2 patch embedding level,
    BEFORE the aggregator's alternating attention blocks run.
    
    Args:
        device: Device to use
        interpolation_mode: 'linear' or 'patch_matching'
        neighborhood_size: Size of neighborhood for patch matching (default 5)
        enable_*: Head configuration flags
        model_path: Path to pretrained weights
    """

    def __init__(
        self, 
        device: str = '',
        interpolation_mode: str = 'patch_matching',
        neighborhood_size: int = 5,
        enable_camera: bool = True,
        enable_point: bool = True, 
        enable_depth: bool = True,
        enable_track: bool = False,
        enable_nlp: bool = True,
        enable_latent: bool = False,
        model_path: str = '',
        img_size: int = 518,
        patch_size: int = 14,
        embed_dim: int = 1024
    ):

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        
        # Store interpolation configuration
        self.interpolation_mode = interpolation_mode
        self.neighborhood_size = neighborhood_size
        self.patch_size = patch_size
        self.img_size = img_size
        
        # Store head configuration
        self.enable_camera = enable_camera
        self.enable_point = enable_point
        self.enable_depth = enable_depth
        self.enable_track = enable_track
        self.enable_nlp = enable_nlp
        self.enable_latent = enable_latent
        
        # Create model with specified heads
        self.model = TNeRFVGGT(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            enable_camera=enable_camera,
            enable_point=enable_point,
            enable_depth=enable_depth,
            enable_track=enable_track,
            enable_nlp=enable_nlp,
            enable_latent=enable_latent
        )
        
        # Load pretrained weights if provided
        if model_path and os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location="cpu")
            # Handle different checkpoint formats
            if "model" in state_dict:
                state_dict = state_dict["model"]
            elif "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            self.model.load_state_dict(state_dict, strict=False)
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Initialize patch embedding interpolator (lazy loading)
        self._patch_interpolator = None
        
        # Cache for endpoint patch embeddings (for efficient repeated interpolation)
        # Note: We now cache DINOv2 patch embeddings, not aggregator outputs
        self._cache = {
            'patches_t0': None,       # DINOv2 patch embeddings at t0: (B*S, P, C)
            'patches_t1': None,       # DINOv2 patch embeddings at t1: (B*S, P, C)
            'images_t0': None,        # Reference images for head computation
            'B': None,
            'S': None,
            'H': None,
            'W': None,
        }
    
    def _get_patch_interpolator(self):
        """Lazy initialization of patch embedding interpolator."""
        if self._patch_interpolator is None:
            self._patch_interpolator = PatchEmbeddingInterpolator(
                patch_size=self.patch_size,
                neighborhood_size=self.neighborhood_size,
                embed_dim=1024,
                device=str(self.device)
            )
        return self._patch_interpolator
    
    def set_interpolation_mode(self, mode: str):
        """
        Switch interpolation mode.
        
        Args:
            mode: 'linear' or 'patch_matching'
        """
        if mode not in ['linear', 'patch_matching']:
            raise ValueError(f"Unknown mode: {mode}. Use 'linear' or 'patch_matching'")
        self.interpolation_mode = mode
        print(f"[DynamicVGGT] Interpolation mode set to: {mode}", flush=True)
    
    def clear_cache(self):
        """Clear cached endpoint data."""
        for key in self._cache:
            self._cache[key] = None
        if self._patch_interpolator is not None:
            self._patch_interpolator.clear_cache()
        torch.cuda.empty_cache()

    @torch.no_grad()
    def _encode_patch_embeddings(self, images: torch.Tensor):
        """
        Extract DINOv2 patch embeddings using aggregator.encode_patches().
        
        This is the first phase - just the DINOv2 patch embedding extraction,
        before the aggregator's alternating attention.
        
        Args:
            images: Input images [B, S, 3, H, W]
            
        Returns:
            patch_tokens: DINOv2 patch embeddings (B*S, P, C)
            B, S, H, W: Dimensions for later use
        """
        if images.dim() == 4:
            images = images.unsqueeze(0)  # [1, S, 3, H, W]
        
        # Use aggregator's encode_patches method (Phase 1 only)
        patch_tokens, B, S, H, W = self.model.aggregator.encode_patches(images)
        return patch_tokens, B, S, H, W

    @torch.no_grad()
    def _encode_tokens(self, images: torch.Tensor):
        """Run VGGT aggregator only and return token list + patch idx."""
        if images.dim() == 4:
            images = images.unsqueeze(0)  # [1, S, 3, H, W]
        tokens, patch_start_idx = self.model.aggregator(images)
        return tokens, patch_start_idx

    @torch.no_grad()
    def _run_heads(self, tokens_list, images: torch.Tensor, patch_start_idx: int) -> Dict[str, torch.Tensor]:
        """
        Run specified heads with given token list.
        
        Args:
            tokens_list: List of interpolated token features
            images: Reference images for spatial dimensions
            patch_start_idx: Patch start index from aggregator
            heads: List of head names to run. If None, run all enabled heads.
                   Valid names: "camera", "depth", "point", "nlp"
        
        Returns:
            Dictionary containing predictions from specified heads
        """
        preds: Dict[str, torch.Tensor] = {}

        with torch.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', enabled=False):
            # Camera head
            if self.enable_camera and self.model.camera_head is not None:
                pose_enc_list = self.model.camera_head(tokens_list)
                preds["pose_enc"] = pose_enc_list[-1]
                preds["pose_enc_list"] = pose_enc_list

            # Depth head
            if self.enable_depth and self.model.depth_head is not None:
                depth, depth_conf = self.model.depth_head(tokens_list, images=images, patch_start_idx=patch_start_idx)
                preds["depth"] = depth
                preds["depth_conf"] = depth_conf

            # Point head
            if self.enable_point and self.model.point_head is not None:
                pts3d, pts3d_conf = self.model.point_head(tokens_list, images=images, patch_start_idx=patch_start_idx)
                preds["world_points"] = pts3d
                preds["world_points_conf"] = pts3d_conf

            # NLP head (T-NeRF extension for NeRF MLP parameter prediction)
            if self.enable_nlp and self.model.nmlp_head is not None:
                nmlp = self.model.nmlp_head(tokens_list, images=images, patch_start_idx=patch_start_idx)
                preds["nmlp"] = nmlp
            
            # Latent head (Tri-plane representation)
            if self.enable_latent and self.model.latent_head is not None:
                latent_output = self.model.latent_head(tokens_list, images=images, patch_start_idx=patch_start_idx)
                preds["xy_plane"] = latent_output["xy_plane"]
                preds["xz_plane"] = latent_output["xz_plane"]
                preds["yz_plane"] = latent_output["yz_plane"]

        return preds

    @torch.no_grad()
    def cache_endpoints(self, images_a: torch.Tensor, images_b: torch.Tensor):
        """
        Cache the DINOv2 patch embeddings for both endpoints for efficient repeated interpolation.
        
        This caches at the patch embedding level (after DINOv2, before aggregator attention),
        so interpolation can happen at the right stage.
        
        Call this once, then use interpolate_cached() multiple times.
        
        Args:
            images_a: Multi-view images at time A, shape [S, 3, H, W] or [B, S, 3, H, W]
            images_b: Multi-view images at time B, shape [S, 3, H, W] or [B, S, 3, H, W]
        """
        images_a = images_a.to(self.device)
        images_b = images_b.to(self.device)
        
        if images_a.dim() == 4:
            images_a = images_a.unsqueeze(0)
        if images_b.dim() == 4:
            images_b = images_b.unsqueeze(0)
        
        # Extract DINOv2 patch embeddings for both endpoints (Phase 1 only)
        patches_a, B, S, H, W = self._encode_patch_embeddings(images_a)
        patches_b, _, _, _, _ = self._encode_patch_embeddings(images_b)
        
        # Store in cache
        self._cache['patches_t0'] = patches_a
        self._cache['patches_t1'] = patches_b
        self._cache['images_t0'] = images_a
        self._cache['B'] = B
        self._cache['S'] = S
        self._cache['H'] = H
        self._cache['W'] = W
        
        # Cache in patch interpolator (needed for both modes for efficient access)
        interpolator = self._get_patch_interpolator()
        interpolator.cache_patch_embeddings(patches_a, patches_b, B, S, H, W)
        
        print(f"[DynamicVGGT] Cached patch embeddings: B={B}, S={S} views, {H//self.patch_size}x{W//self.patch_size} patches, mode={self.interpolation_mode}", flush=True)

    @torch.no_grad()
    def cache_endpoints_from_paths(self, paths_a: List[str], paths_b: List[str]):
        """
        Load images and cache endpoints.
        
        Args:
            paths_a: Image paths for time A
            paths_b: Image paths for time B
        """
        paths_a = sorted(paths_a)
        paths_b = sorted(paths_b)
        k = min(len(paths_a), len(paths_b))
        
        if k == 0:
            raise ValueError("No images found in one of the time-step groups.")
        
        images_a = load_and_preprocess_images(paths_a[:k]).to(self.device)
        images_b = load_and_preprocess_images(paths_b[:k]).to(self.device)
        
        self.cache_endpoints(images_a, images_b)

    @torch.no_grad()
    def interpolate_cached(self, alpha: float) -> Dict[str, torch.Tensor]:
        """
        Interpolate using cached patch embeddings.
        
        Pipeline:
        1. Interpolate DINOv2 patch embeddings (at cache level)
        2. Run aggregator's forward_from_patches() on interpolated embeddings
        3. Run heads on the aggregated tokens
        
        Much faster than interpolate() for repeated calls.
        
        Args:
            alpha: Interpolation factor in [0, 1], where 0 = A, 1 = B
            
        Returns:
            Dictionary containing predictions from heads
        """
        if self._cache['patches_t0'] is None:
            raise RuntimeError("Must call cache_endpoints() first")
        
        alpha = max(0.0, min(1.0, alpha))
        
        images = self._cache['images_t0']
        B = self._cache['B']
        S = self._cache['S']
        H = self._cache['H']
        W = self._cache['W']
        
        interpolator = self._get_patch_interpolator()
        
        # Step 1: Interpolate DINOv2 patch embeddings
        if self.interpolation_mode == 'linear':
            patches_interp = interpolator.interpolate_linear(alpha)
        else:
            patches_interp = interpolator.interpolate(alpha)
        
        # Step 2: Run aggregator's Phase 2 (forward_from_patches) on interpolated embeddings
        tokens_list, patch_start_idx = self.model.aggregator.forward_from_patches(
            patches_interp, B, S, H, W
        )
        
        # Step 3: Run heads on aggregated tokens
        preds = self._run_heads(tokens_list, images, patch_start_idx)
        
        # Convert pose encoding to extrinsic/intrinsic
        if "pose_enc" in preds:
            extrinsic, intrinsic = pose_encoding_to_extri_intri(preds["pose_enc"], images.shape[-2:])
            preds["extrinsic"] = extrinsic
            preds["intrinsic"] = intrinsic
        
        return preds

    @torch.no_grad()
    def interpolate_sequence(
        self,
        images_a: torch.Tensor,
        images_b: torch.Tensor,
        num_samples: int = 10,
        include_endpoints: bool = True
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Generate a sequence of interpolated outputs from t0 to t1.
        
        Caches endpoints internally for efficiency.
        
        Args:
            images_a: Multi-view images at time A
            images_b: Multi-view images at time B
            num_samples: Number of samples to generate
            include_endpoints: Whether to include t0 (alpha=0) and t1 (alpha=1)
            
        Returns:
            List of prediction dictionaries, one for each sample
        """
        # Cache endpoints
        self.cache_endpoints(images_a, images_b)
        
        # Generate alpha values
        if include_endpoints:
            alphas = torch.linspace(0, 1, num_samples)
        else:
            alphas = torch.linspace(0, 1, num_samples + 2)[1:-1]
        
        results = []
        for alpha in alphas:
            preds = self.interpolate_cached(float(alpha))
            # Clone to avoid reference issues
            preds_copy = {k: v.clone() if torch.is_tensor(v) else v for k, v in preds.items()}
            preds_copy['alpha'] = float(alpha)
            results.append(preds_copy)
        
        print(f"[DynamicVGGT] Generated {len(results)} samples", flush=True)
        return results

    @torch.no_grad()
    def interpolate_sequence_from_paths(
        self,
        paths_a: List[str],
        paths_b: List[str],
        num_samples: int = 10,
        include_endpoints: bool = True
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Load images and generate interpolation sequence.
        
        Args:
            paths_a: Image paths for time A
            paths_b: Image paths for time B
            num_samples: Number of samples
            include_endpoints: Whether to include endpoints
            
        Returns:
            List of prediction dictionaries
        """
        paths_a = sorted(paths_a)
        paths_b = sorted(paths_b)
        k = min(len(paths_a), len(paths_b))
        
        images_a = load_and_preprocess_images(paths_a[:k]).to(self.device)
        images_b = load_and_preprocess_images(paths_b[:k]).to(self.device)
        
        return self.interpolate_sequence(images_a, images_b, num_samples, include_endpoints)

    @torch.no_grad()
    def interpolate(self, images_a: torch.Tensor, images_b: torch.Tensor, alpha: float):
        """
        Interpolate between two multi-view image groups and run specified heads.
        
        Uses the configured interpolation_mode ('linear' or 'patch_matching').
        
        Pipeline (v2 - patch embedding level interpolation):
        1. Extract DINOv2 patch embeddings for both image groups
        2. Interpolate patch embeddings based on mode and alpha
        3. Run aggregator's forward_from_patches() on interpolated embeddings
        4. Pass aggregated tokens through heads
        5. Return head outputs
        
        Args:
            images_a: Multi-view images at time A, shape [S, 3, H, W] or [B, S, 3, H, W]
            images_b: Multi-view images at time B, shape [S, 3, H, W] or [B, S, 3, H, W]
            alpha: Interpolation factor in [0, 1], where 0 = A, 1 = B
            
        Returns:
            Dictionary containing predictions from specified heads:
                - pose_enc: Camera pose encoding (if "camera" in heads)
                - depth, depth_conf: Depth maps (if "depth" in heads)
                - world_points, world_points_conf: 3D points (if "point" in heads)
                - nmlp: NeRF MLP parameters (if "nlp" in heads)
                - extrinsic, intrinsic: Camera matrices (derived from pose_enc)
        """
        alpha = float(alpha)
        alpha = max(0.0, min(1.0, alpha))

        images_a = images_a.to(self.device)
        images_b = images_b.to(self.device)
        
        if images_a.dim() == 4:
            images_a = images_a.unsqueeze(0)
        if images_b.dim() == 4:
            images_b = images_b.unsqueeze(0)

        # Step 1: Extract DINOv2 patch embeddings
        patches_a, B, S, H, W = self._encode_patch_embeddings(images_a)
        patches_b, _, _, _, _ = self._encode_patch_embeddings(images_b)

        # Step 2: Interpolate patch embeddings
        interpolator = self._get_patch_interpolator()
        interpolator.cache_patch_embeddings(patches_a, patches_b, B, S, H, W)
        
        if self.interpolation_mode == 'linear':
            patches_interp = interpolator.interpolate_linear(alpha)
        else:
            patches_interp = interpolator.interpolate(alpha)

        # Step 3: Run aggregator's Phase 2 on interpolated patch embeddings
        tokens_list, patch_idx = self.model.aggregator.forward_from_patches(
            patches_interp, B, S, H, W
        )

        # Step 4: Run heads with aggregated tokens
        preds = self._run_heads(tokens_list, images_a, patch_idx)

        # Convert pose enc to extrinsic/intrinsic if available
        if "pose_enc" in preds:
            extrinsic, intrinsic = pose_encoding_to_extri_intri(preds["pose_enc"], images_a.shape[-2:])
            preds["extrinsic"] = extrinsic
            preds["intrinsic"] = intrinsic

        return preds

    @torch.no_grad()
    def interpolate_from_paths(self, paths_a: List[str], paths_b: List[str], alpha: float) -> Dict[str, torch.Tensor]:
        """
        Utility: load images from disk, then interpolate.
        
        Args:
            paths_a: List of image file paths for time step A
            paths_b: List of image file paths for time step B
            alpha: Interpolation factor in [0, 1]
            
        Returns:
            Dictionary containing interpolated predictions
        """
        # Ensure equal number of views by uniform sampling to the smaller count
        paths_a = sorted(paths_a)
        paths_b = sorted(paths_b)
        na, nb = len(paths_a), len(paths_b)
        if na == 0 or nb == 0:
            raise ValueError("No images found in one of the time-step groups.")
        k = min(na, nb)
        if na != k:
            print(f"num of a views is larger, use topk instead")
        if nb != k:
            print(f"num of b views is larger, use topk instead")

        imgs_a = load_and_preprocess_images(paths_a[:k]).to(self.device)
        imgs_b = load_and_preprocess_images(paths_b[:k]).to(self.device)
        
        return self.interpolate(imgs_a, imgs_b, alpha)

    @torch.no_grad()
    def forward(self, images: torch.Tensor, heads: List[str] = None) -> Dict[str, torch.Tensor]:
        """
        Run specified heads on a single multi-view image group (no interpolation).
            
        Args:
            images: Input images, shape [S, 3, H, W] or [B, S, 3, H, W]
            heads: List of head names to run. Valid: "camera", "depth", "point", "nlp"
            
        Returns:
            Dictionary containing predictions from specified heads
        """
        images = images.to(self.device)
        tokens, patch_idx = self._encode_tokens(images)
        
        preds = self._run_heads(
            tokens,
            images if images.dim() == 5 else images.unsqueeze(0),
            patch_idx
        )
        
        # Convert pose enc to extrinsic/intrinsic if available
        if "pose_enc" in preds:
            extrinsic, intrinsic = pose_encoding_to_extri_intri(preds["pose_enc"], images.shape[-2:])
            preds["extrinsic"] = extrinsic
            preds["intrinsic"] = intrinsic
        
        return preds

    @torch.no_grad()
    def forward_single(self, images: Union[torch.Tensor, List[str]], heads: List[str] = None) -> Dict[str, torch.Tensor]:
        """
        静态模式：从一组多视角图像预测输出（无插值）
        
        Args:
            images: 输入图像，可以是:
                - torch.Tensor: shape [S, 3, H, W] or [B, S, 3, H, W]
                - List[str]: 图像文件路径列表
            heads: 要运行的头列表。有效值: "camera", "depth", "point", "nlp"
            
        Returns:
            包含指定头预测结果的字典
        """
        # 如果是路径列表，加载图像
        if isinstance(images, list) and len(images) > 0 and isinstance(images[0], str):
            images = load_and_preprocess_images(images)
        
        return self.forward(images, heads)


def create_dynamic_vggt(
    config: str = "full",
    device: str = '',
    model_path: str = '',
    interpolation_mode: str = 'patch_matching',
    neighborhood_size: int = 11,
    **kwargs
) -> DynamicVGGT:
    """
    Create a DynamicVGGT instance with predefined configurations.
    
    Args:
        config: Predefined configuration name:
            - "full": All heads enabled (including latent)
            - "nerf": Only point + nlp heads for NeRF applications
            - "nerf_latent": Point + latent heads for Tri-plane NeRF
            - "3d": Camera + point + depth heads for 3D reconstruction
            - "minimal": Only point head
            - "latent": Only latent head for Tri-plane representation
        device: Device to use
        model_path: Path to pretrained model weights
        interpolation_mode: 'linear' or 'patch_matching' (default 'linear')
        neighborhood_size: Size of neighborhood for patch matching (default 5)
        **kwargs: Additional arguments passed to DynamicVGGT
        
    Returns:
        DynamicVGGT instance
    """
    configs = {
        "full": {
            "enable_camera": True,
            "enable_point": True,
            "enable_depth": True,
            "enable_track": False,
            "enable_nlp": True,
            "enable_latent": True
        },
        "nerf": {
            "enable_camera": False,
            "enable_point": True,
            "enable_depth": False,
            "enable_track": False,
            "enable_nlp": True,
            "enable_latent": False
        },
        "nerf_latent": {
            "enable_camera": False,
            "enable_point": True,
            "enable_depth": False,
            "enable_track": False,
            "enable_nlp": False,
            "enable_latent": True
        },
        "3d": {
            "enable_camera": True,
            "enable_point": True,
            "enable_depth": True,
            "enable_track": False,
            "enable_nlp": False,
            "enable_latent": False
        },
        "minimal": {
            "enable_camera": False,
            "enable_point": True,
            "enable_depth": False,
            "enable_track": False,
            "enable_nlp": False,
            "enable_latent": False
        },
        "latent": {
            "enable_camera": False,
            "enable_point": False,
            "enable_depth": False,
            "enable_track": False,
            "enable_nlp": False,
            "enable_latent": True
        }
    }
    
    if config not in configs:
        raise ValueError(f"Unknown config '{config}'. Available configs: {list(configs.keys())}")
    
    # Merge config with user kwargs
    params = configs[config].copy()
    params.update(kwargs)
    
    return DynamicVGGT(
        device=device,
        model_path=model_path,
        interpolation_mode=interpolation_mode,
        neighborhood_size=neighborhood_size,
        **params
    )


def save_uploaded_group(files, root_dir: str, tag: str) -> List[str]:
    """
    Save uploaded file objects to disk and return sorted paths.
    
    Args:
        files: List of uploaded file objects (from Gradio)
        root_dir: Root directory for saving
        tag: Tag for subdirectory name (e.g., 'a' or 'b')
        
    Returns:
        List of saved file paths, sorted
    """
    target_dir = os.path.join(root_dir, f"time_{tag}")
    os.makedirs(target_dir, exist_ok=True)
    saved = []
    if files is None:
        return saved
    for f in files:
        src = f["name"] if isinstance(f, dict) and "name" in f else f.name if hasattr(f, "name") else f
        dst = os.path.join(target_dir, os.path.basename(src))
        with open(dst, "wb") as out_f:
            with open(src, "rb") as in_f:
                out_f.write(in_f.read())
        saved.append(dst)
    saved.sort()
    return saved
