# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
T-NeRF Loss Functions for NLPHead training.

NLPHead Loss:
- Computes the loss for NeRF MLP parameter prediction
- Uses rendering loss to compare rendered images with ground truth
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple


class TNeRFLoss(nn.Module):
    """Loss function for T-NeRF NLPHead."""
    
    def __init__(
        self,
        nlp_weight: float = 1.0,
        nlp_render_weight: float = 1.0,
        nlp_reg_weight: float = 0.001,
    ):
        super().__init__()
        self.nlp_weight = nlp_weight
        self.nlp_render_weight = nlp_render_weight
        self.nlp_reg_weight = nlp_reg_weight
    
    def forward(
        self, 
        predictions: Dict[str, torch.Tensor], 
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        # TODO: 验证这里的batch与dataloader的一致
        
        Compute NLP loss only.
        
        Args:
            predictions: Dictionary containing model predictions
                - "nmlp": [B, nerf_param_count] - NeRF MLP parameters
            batch: Dictionary containing ground truth data
                - For NLP: "target_images", "camera_params"
        
        Returns:
            Dictionary containing individual losses and total loss
        """
        loss_dict = {}
        total_loss = torch.tensor(0.0, device=self._get_device(predictions))
        
        if "nmlp" in predictions:
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
