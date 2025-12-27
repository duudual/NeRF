# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
T-NeRF Loss Functions for NColorHead and NLPHead training.

NColorHead Loss:
- Computes the loss between predicted color maps and ground truth color values
- Uses MSE loss for color prediction
- Optionally includes confidence-weighted loss

NLPHead Loss:
- Computes the loss for NeRF MLP parameter prediction
- Uses rendering loss to compare rendered images with ground truth
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple


class TNeRFLoss(nn.Module):
    """
    Combined loss function for T-NeRF training.
    
    Supports:
    - NColor loss: Color map prediction loss
    - NLP loss: NeRF MLP parameter prediction loss
    """
    
    def __init__(
        self,
        mode: str = "both",
        ncolor_weight: float = 1.0,
        nlp_weight: float = 1.0,
        # NColor loss settings
        ncolor_mse_weight: float = 1.0,
        ncolor_conf_weight: float = 0.1,
        ncolor_smooth_weight: float = 0.01,
        # NLP loss settings
        nlp_render_weight: float = 1.0,
        nlp_reg_weight: float = 0.001,
    ):
        super().__init__()
        self.mode = mode
        self.ncolor_weight = ncolor_weight
        self.nlp_weight = nlp_weight
        
        # NColor loss weights
        self.ncolor_mse_weight = ncolor_mse_weight
        self.ncolor_conf_weight = ncolor_conf_weight
        self.ncolor_smooth_weight = ncolor_smooth_weight
        
        # NLP loss weights
        self.nlp_render_weight = nlp_render_weight
        self.nlp_reg_weight = nlp_reg_weight
    
    def forward(
        self, 
        predictions: Dict[str, torch.Tensor], 
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the combined loss.
        
        Args:
            predictions: Dictionary containing model predictions
                - "ncolor": [B, S, H, W, 20] - 6*3 colors + 1 alpha + 1 confidence
                - "ncolor_conf": [B, S, H, W] - confidence scores
                - "nmlp": [B, nerf_param_count] - NeRF MLP parameters
            batch: Dictionary containing ground truth data
                - For NColor: "target_colors", "target_alpha", "point_masks"
                - For NLP: "target_images", "camera_params"
        
        Returns:
            Dictionary containing individual losses and total loss
        """
        loss_dict = {}
        total_loss = torch.tensor(0.0, device=self._get_device(predictions))
        
        # NColor loss
        if self.mode in ["ncolor", "both"] and "ncolor" in predictions:
            ncolor_loss_dict = self._compute_ncolor_loss(predictions, batch)
            for key, value in ncolor_loss_dict.items():
                loss_dict[f"ncolor_{key}"] = value
            
            ncolor_total = (
                ncolor_loss_dict["mse_loss"] * self.ncolor_mse_weight +
                ncolor_loss_dict.get("conf_loss", 0) * self.ncolor_conf_weight +
                ncolor_loss_dict.get("smooth_loss", 0) * self.ncolor_smooth_weight
            )
            loss_dict["ncolor_total"] = ncolor_total
            total_loss = total_loss + ncolor_total * self.ncolor_weight
        
        # NLP loss
        if self.mode in ["nlp", "both"] and "nmlp" in predictions:
            nlp_loss_dict = self._compute_nlp_loss(predictions, batch)
            for key, value in nlp_loss_dict.items():
                loss_dict[f"nlp_{key}"] = value
            
            nlp_total = (
                nlp_loss_dict["render_loss"] * self.nlp_render_weight +
                nlp_loss_dict.get("reg_loss", 0) * self.nlp_reg_weight
            )
            loss_dict["nlp_total"] = nlp_total
            total_loss = total_loss + nlp_total * self.nlp_weight
        
        loss_dict["total_loss"] = total_loss
        return loss_dict
    
    def _get_device(self, predictions: Dict) -> torch.device:
        """Get device from predictions."""
        for key, value in predictions.items():
            if torch.is_tensor(value):
                return value.device
        return torch.device("cpu")
    
    def _compute_ncolor_loss(
        self, 
        predictions: Dict[str, torch.Tensor], 
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute NColor loss.
        
        The ncolor output has shape [B, S, H, W, 20]:
        - Channels 0-17: 6 RGB colors (up, down, left, right, front, back) = 6*3 = 18
        - Channel 18: alpha (opacity)
        - Channel 19: confidence
        
        Args:
            predictions: Model predictions containing "ncolor"
            batch: Ground truth containing "target_colors" and optionally "target_alpha", "point_masks"
        
        Returns:
            Dictionary of loss components
        """
        ncolor = predictions["ncolor"]  # [B, S, H, W, 20]
        B, S, H, W, C = ncolor.shape
        
        loss_dict = {}
        
        # Parse ncolor output
        pred_colors = ncolor[..., :18]  # [B, S, H, W, 18] - 6 directions * 3 RGB
        pred_alpha = ncolor[..., 18:19]  # [B, S, H, W, 1]
        pred_conf = ncolor[..., 19:20]  # [B, S, H, W, 1]
        
        # Get ground truth
        if "target_colors" in batch:
            target_colors = batch["target_colors"]  # [B, S, H, W, 18] or sparse
            target_alpha = batch.get("target_alpha", None)  # [B, S, H, W, 1]
            point_masks = batch.get("point_masks", None)  # [B, S, H, W]
            
            # Apply mask if available
            if point_masks is not None:
                mask = point_masks.unsqueeze(-1).expand_as(pred_colors)
                valid_pred_colors = pred_colors[mask > 0]
                valid_target_colors = target_colors[mask > 0]
                
                if valid_pred_colors.numel() > 0:
                    loss_dict["mse_loss"] = F.mse_loss(valid_pred_colors, valid_target_colors)
                else:
                    loss_dict["mse_loss"] = torch.tensor(0.0, device=ncolor.device)
            else:
                loss_dict["mse_loss"] = F.mse_loss(pred_colors, target_colors)
            
            # Alpha loss
            if target_alpha is not None:
                loss_dict["alpha_loss"] = F.mse_loss(pred_alpha, target_alpha)
            
            # Confidence loss - encourage high confidence on valid predictions
            if point_masks is not None:
                # Flatten for BCE loss
                pred_conf_flat = pred_conf.squeeze(-1)  # [B, S, H, W]
                target_conf = point_masks.float()  # [B, S, H, W]
                loss_dict["conf_loss"] = F.binary_cross_entropy_with_logits(
                    pred_conf_flat, target_conf
                )
        else:
            # If no ground truth, compute dummy loss
            loss_dict["mse_loss"] = torch.tensor(0.0, device=ncolor.device)
        
        # Smoothness loss - encourage smooth color predictions
        if self.ncolor_smooth_weight > 0:
            # Compute gradient in spatial dimensions
            dx = pred_colors[:, :, :, 1:, :] - pred_colors[:, :, :, :-1, :]
            dy = pred_colors[:, :, 1:, :, :] - pred_colors[:, :, :-1, :, :]
            loss_dict["smooth_loss"] = torch.mean(dx.abs()) + torch.mean(dy.abs())
        
        return loss_dict
    
    def _compute_nlp_loss(
        self, 
        predictions: Dict[str, torch.Tensor], 
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute NLP (NeRF MLP Parameters) loss.
        
        This loss encourages the predicted MLP parameters to produce
        renderings that match the target images.
        
        Args:
            predictions: Model predictions containing "nmlp" (MLP parameters)
            batch: Ground truth containing "target_images" and "camera_params"
        
        Returns:
            Dictionary of loss components
        """
        nmlp_params = predictions["nmlp"]  # [B, nerf_param_count]
        
        loss_dict = {}
        
        if "target_render" in batch:
            # If we have pre-rendered target images
            target_render = batch["target_render"]  # [B, H, W, 3]
            
            # Render using predicted MLP parameters
            # This requires the NeRF rendering pipeline
            rendered = self._render_with_params(
                nmlp_params, 
                batch.get("ray_origins"),
                batch.get("ray_directions"),
            )
            
            if rendered is not None:
                loss_dict["render_loss"] = F.mse_loss(rendered, target_render)
            else:
                # Fallback: use parameter regularization
                loss_dict["render_loss"] = torch.tensor(0.0, device=nmlp_params.device)
        else:
            # Without target renders, use parameter regularization
            loss_dict["render_loss"] = torch.tensor(0.0, device=nmlp_params.device)
        
        # Parameter regularization - encourage small parameter values
        loss_dict["reg_loss"] = torch.mean(nmlp_params ** 2)
        
        return loss_dict
    
    def _render_with_params(
        self,
        params: torch.Tensor,
        ray_origins: Optional[torch.Tensor],
        ray_directions: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """
        Render images using predicted NeRF MLP parameters.
        
        This is a placeholder for the actual rendering pipeline.
        The rendering requires:
        1. Loading MLP parameters into a NeRF MLP
        2. Ray marching with positional encoding
        3. Volume rendering
        
        Args:
            params: Predicted MLP parameters [B, nerf_param_count]
            ray_origins: Ray origins [B, H, W, 3]
            ray_directions: Ray directions [B, H, W, 3]
        
        Returns:
            Rendered images [B, H, W, 3] or None if inputs are missing
        """
        if ray_origins is None or ray_directions is None:
            return None
        
        # TODO: Implement actual rendering
        # This would use the NeRFMLP class from nlp_head.py
        # and perform volume rendering
        
        return None


class NColorLoss(nn.Module):
    """
    Standalone NColor loss for training NColorHead.
    
    This loss computes:
    1. MSE loss between predicted and target colors
    2. Alpha (opacity) loss
    3. Confidence loss
    4. Smoothness regularization
    """
    
    def __init__(
        self,
        mse_weight: float = 1.0,
        alpha_weight: float = 1.0,
        conf_weight: float = 0.1,
        smooth_weight: float = 0.01,
        air_alpha_weight: float = 0.5,  # Weight for air points (should have alpha=1)
    ):
        super().__init__()
        self.mse_weight = mse_weight
        self.alpha_weight = alpha_weight
        self.conf_weight = conf_weight
        self.smooth_weight = smooth_weight
        self.air_alpha_weight = air_alpha_weight
    
    def forward(
        self,
        pred_ncolor: torch.Tensor,
        target_colors: torch.Tensor,
        target_alpha: torch.Tensor,
        point_mask: torch.Tensor,
        is_air_point: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute NColor loss.
        
        Args:
            pred_ncolor: Predicted ncolor output [B, S, H, W, 20]
            target_colors: Target colors [B, S, H, W, 18]
            target_alpha: Target alpha [B, S, H, W, 1]
            point_mask: Valid point mask [B, S, H, W]
            is_air_point: Mask for air points [B, S, H, W] (optional)
        
        Returns:
            Dictionary of loss components
        """
        # Parse predictions
        pred_colors = pred_ncolor[..., :18]
        pred_alpha = pred_ncolor[..., 18:19]
        pred_conf = pred_ncolor[..., 19:20]
        
        loss_dict = {}
        
        # Color MSE loss (only on valid points)
        mask_expanded = point_mask.unsqueeze(-1).expand_as(pred_colors)
        valid_pred = pred_colors[mask_expanded > 0]
        valid_target = target_colors[mask_expanded > 0]
        
        if valid_pred.numel() > 0:
            loss_dict["mse_loss"] = F.mse_loss(valid_pred, valid_target)
        else:
            loss_dict["mse_loss"] = torch.tensor(0.0, device=pred_ncolor.device)
        
        # Alpha loss
        alpha_mask = point_mask.unsqueeze(-1)
        valid_pred_alpha = pred_alpha[alpha_mask > 0]
        valid_target_alpha = target_alpha[alpha_mask > 0]
        
        if valid_pred_alpha.numel() > 0:
            loss_dict["alpha_loss"] = F.mse_loss(valid_pred_alpha, valid_target_alpha)
        else:
            loss_dict["alpha_loss"] = torch.tensor(0.0, device=pred_ncolor.device)
        
        # Air point loss - encourage alpha=1 for air points (transparent)
        if is_air_point is not None:
            air_mask = is_air_point.unsqueeze(-1)
            air_pred_alpha = pred_alpha[air_mask > 0]
            # Air points should have alpha=1 (fully transparent)
            air_target_alpha = torch.ones_like(air_pred_alpha)
            
            if air_pred_alpha.numel() > 0:
                loss_dict["air_alpha_loss"] = F.mse_loss(air_pred_alpha, air_target_alpha)
            else:
                loss_dict["air_alpha_loss"] = torch.tensor(0.0, device=pred_ncolor.device)
        
        # Confidence loss
        pred_conf_flat = pred_conf.squeeze(-1)
        target_conf = point_mask.float()
        loss_dict["conf_loss"] = F.binary_cross_entropy_with_logits(pred_conf_flat, target_conf)
        
        # Smoothness loss
        dx = pred_colors[:, :, :, 1:, :] - pred_colors[:, :, :, :-1, :]
        dy = pred_colors[:, :, 1:, :, :] - pred_colors[:, :, :-1, :, :]
        loss_dict["smooth_loss"] = torch.mean(dx.abs()) + torch.mean(dy.abs())
        
        # Total loss
        total = (
            loss_dict["mse_loss"] * self.mse_weight +
            loss_dict["alpha_loss"] * self.alpha_weight +
            loss_dict.get("air_alpha_loss", 0) * self.air_alpha_weight +
            loss_dict["conf_loss"] * self.conf_weight +
            loss_dict["smooth_loss"] * self.smooth_weight
        )
        loss_dict["total"] = total
        
        return loss_dict


class NLPLoss(nn.Module):
    """
    Standalone NLP loss for training NLPHead.
    
    This loss computes:
    1. Rendering loss - MSE between rendered and target images
    2. Parameter regularization loss
    3. Perceptual loss (optional)
    """
    
    def __init__(
        self,
        render_weight: float = 1.0,
        reg_weight: float = 0.001,
        perceptual_weight: float = 0.0,
    ):
        super().__init__()
        self.render_weight = render_weight
        self.reg_weight = reg_weight
        self.perceptual_weight = perceptual_weight
    
    def forward(
        self,
        pred_params: torch.Tensor,
        rendered_images: torch.Tensor,
        target_images: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute NLP loss.
        
        Args:
            pred_params: Predicted MLP parameters [B, nerf_param_count]
            rendered_images: Images rendered using pred_params [B, H, W, 3]
            target_images: Ground truth images [B, H, W, 3]
        
        Returns:
            Dictionary of loss components
        """
        loss_dict = {}
        
        # Rendering loss
        loss_dict["render_loss"] = F.mse_loss(rendered_images, target_images)
        
        # Parameter regularization
        loss_dict["reg_loss"] = torch.mean(pred_params ** 2)
        
        # SSIM loss (optional, for better perceptual quality)
        # loss_dict["ssim_loss"] = 1 - ssim(rendered_images, target_images)
        
        # Total loss
        total = (
            loss_dict["render_loss"] * self.render_weight +
            loss_dict["reg_loss"] * self.reg_weight
        )
        loss_dict["total"] = total
        
        return loss_dict
