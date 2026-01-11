"""
Dynamic T-NeRF helper for temporal interpolation with VGGT.

Given two time steps (multi-view image sets), this module encodes each time
step with VGGT, linearly interpolates the aggregated tokens, and runs the
existing heads to produce intermediate outputs (poses, depth, points, nmlp).

"""

import os
from typing import Dict, List, Union

import torch
import torch.nn as nn

from .models.vggt import VGGT as TNeRFVGGT
from .utils.load_fn import load_and_preprocess_images
from .utils.pose_enc import pose_encoding_to_extri_intri


class DynamicVGGT:
    """
    Lightweight wrapper around VGGT for temporal interpolation.
    
    Creates a VGGT model instance based on specified head configurations.
    """

    def __init__(
        self, 
        device: str = '',
        enable_camera: bool = True,
        enable_point: bool = True, 
        enable_depth: bool = True,
        enable_track: bool = False,
        enable_nlp: bool = True,
        model_path: str = '',
        img_size: int = 518,
        patch_size: int = 14,
        embed_dim: int = 1024
    ):

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        
        # Store head configuration
        self.enable_camera = enable_camera
        self.enable_point = enable_point
        self.enable_depth = enable_depth
        self.enable_track = enable_track
        self.enable_nlp = enable_nlp
        
        # Create model with specified heads
        self.model = TNeRFVGGT(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            enable_camera=enable_camera,
            enable_point=enable_point,
            enable_depth=enable_depth,
            enable_track=enable_track,
            enable_nlp=enable_nlp
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

        return preds

    @torch.no_grad()
    def interpolate(self, images_a: torch.Tensor, images_b: torch.Tensor, alpha: float):
        """
        Interpolate between two multi-view image groups and run specified heads.
        
        Pipeline:
        1. Encode each image group with VGGT aggregator -> get token features
        2. Linearly interpolate token features based on alpha
        3. Pass interpolated features through specified heads
        4. Return head outputs
        
        Args:
            images_a: Multi-view images at time A, shape [S, 3, H, W] or [B, S, 3, H, W]
            images_b: Multi-view images at time B, shape [S, 3, H, W] or [B, S, 3, H, W]
            alpha: Interpolation factor in [0, 1], where 0 = A, 1 = B
            heads: List of head names to run. 
            
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

        tokens_a, patch_idx = self._encode_tokens(images_a)
        tokens_b, _ = self._encode_tokens(images_b)

        # Linear interpolation for every layer output
        tokens_interp = [(1 - alpha) * ta + alpha * tb for ta, tb in zip(tokens_a, tokens_b)]

        # Run specified heads with interpolated features
        preds = self._run_heads(
            tokens_interp, 
            images_a if images_a.dim() == 5 else images_a.unsqueeze(0), 
            patch_idx
        )

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
        imgs_a = load_and_preprocess_images(paths_a).to(self.device)
        imgs_b = load_and_preprocess_images(paths_b).to(self.device)
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
    **kwargs
) -> DynamicVGGT:
    """
    Create a DynamicVGGT instance with predefined configurations.
    
    Args:
        config: Predefined configuration name:
            - "full": All heads enabled
            - "nerf": Only point + nlp heads for NeRF applications
            - "3d": Camera + point + depth heads for 3D reconstruction
            - "minimal": Only point head
        device: Device to use
        model_path: Path to pretrained model weights
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
            "enable_nlp": True
        },
        "nerf": {
            "enable_camera": False,
            "enable_point": True,
            "enable_depth": False,
            "enable_track": False,
            "enable_nlp": True
        },
        "3d": {
            "enable_camera": True,
            "enable_point": True,
            "enable_depth": True,
            "enable_track": False,
            "enable_nlp": False
        },
        "minimal": {
            "enable_camera": False,
            "enable_point": True,
            "enable_depth": False,
            "enable_track": False,
            "enable_nlp": False
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
