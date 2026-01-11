from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class TNerfLoss(nn.Module):
    """
    Loss function for training NLPHead with point sampling.
    
    Computes:
    1. RGB loss at sampled points
    2. Sigma (density) loss at sampled points
    3. Parameter regularization loss
    """
    
    def __init__(
        self,
        rgb_weight: float = 1.0,
        sigma_weight: float = 0.1,
        reg_weight: float = 0.001,
    ):
        super().__init__()
        self.rgb_weight = rgb_weight
        self.sigma_weight = sigma_weight
        self.reg_weight = reg_weight
    
    def forward(
        self,
        pred_rgb: torch.Tensor,
        pred_sigma: torch.Tensor,
        gt_rgb: torch.Tensor,
        gt_sigma: torch.Tensor,
        mlp_params: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss.
        
        Args:
            pred_rgb: Predicted RGB at sample points [B, N, 3]
            pred_sigma: Predicted sigma at sample points [B, N]
            gt_rgb: Ground truth RGB at sample points [B, N, 3]
            gt_sigma: Ground truth sigma at sample points [B, N]
            mlp_params: Predicted MLP parameters [B, num_params]
            
        Returns:
            Loss dictionary
        """
        # RGB L2 loss
        rgb_loss = F.mse_loss(pred_rgb, gt_rgb)
        
        # Sigma L2 loss (could also try L1 or log-space)
        sigma_loss = F.mse_loss(pred_sigma, gt_sigma)
        
        # Parameter regularization
        reg_loss = torch.mean(mlp_params ** 2)
        
        # Total loss
        total_loss = (
            self.rgb_weight * rgb_loss + 
            self.sigma_weight * sigma_loss + 
            self.reg_weight * reg_loss
        )
        
        return {
            'rgb_loss': rgb_loss,
            'sigma_loss': sigma_loss,
            'reg_loss': reg_loss,
            'total_loss': total_loss,
        }


class TriplaneLoss(nn.Module):
    """
    Loss function for training LatentHead with Tri-plane representation.
    
    Computes:
    1. RGB loss at sampled points
    2. Sigma (density) loss at sampled points
    3. Tri-plane regularization (TV + L1 sparsity)
    """
    
    def __init__(
        self,
        rgb_weight: float = 1.0,
        sigma_weight: float = 0.1,
        tv_weight: float = 0.01,
        l1_weight: float = 0.001,
        perceptual_weight: float = 0.0,
    ):
        """
        Args:
            rgb_weight: Weight for RGB reconstruction loss
            sigma_weight: Weight for density loss
            tv_weight: Weight for Total Variation regularization (smoothness)
            l1_weight: Weight for L1 sparsity regularization
            perceptual_weight: Weight for perceptual loss (if using)
        """
        super().__init__()
        self.rgb_weight = rgb_weight
        self.sigma_weight = sigma_weight
        self.tv_weight = tv_weight
        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight
    
    def compute_tv_loss(self, plane: torch.Tensor) -> torch.Tensor:
        """
        Compute Total Variation loss for a feature plane.
        
        Args:
            plane: Feature plane [B, C, H, W]
            
        Returns:
            TV loss scalar
        """
        # Horizontal TV
        tv_h = torch.mean(torch.abs(plane[:, :, :, 1:] - plane[:, :, :, :-1]))
        # Vertical TV
        tv_v = torch.mean(torch.abs(plane[:, :, 1:, :] - plane[:, :, :-1, :]))
        return tv_h + tv_v
    
    def compute_l1_loss(self, plane: torch.Tensor) -> torch.Tensor:
        """
        Compute L1 sparsity loss for a feature plane.
        
        Args:
            plane: Feature plane [B, C, H, W]
            
        Returns:
            L1 loss scalar
        """
        return torch.mean(torch.abs(plane))
    
    def compute_plane_regularization(
        self,
        xy_plane: torch.Tensor,
        xz_plane: torch.Tensor,
        yz_plane: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute regularization losses for all three planes.
        
        Args:
            xy_plane: XY feature plane [B, C, H, W]
            xz_plane: XZ feature plane [B, C, H, W]
            yz_plane: YZ feature plane [B, C, H, W]
            
        Returns:
            Dictionary with tv_loss and l1_loss
        """
        # TV loss for smoothness
        tv_loss = (
            self.compute_tv_loss(xy_plane) +
            self.compute_tv_loss(xz_plane) +
            self.compute_tv_loss(yz_plane)
        ) / 3.0
        
        # L1 loss for sparsity
        l1_loss = (
            self.compute_l1_loss(xy_plane) +
            self.compute_l1_loss(xz_plane) +
            self.compute_l1_loss(yz_plane)
        ) / 3.0
        
        return {
            'tv_loss': tv_loss,
            'l1_loss': l1_loss,
        }
    
    def forward(
        self,
        pred_rgb: torch.Tensor,
        pred_sigma: torch.Tensor,
        gt_rgb: torch.Tensor,
        gt_sigma: torch.Tensor,
        xy_plane: torch.Tensor,
        xz_plane: torch.Tensor,
        yz_plane: torch.Tensor,
        pred_rendered: Optional[torch.Tensor] = None,
        gt_image: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total loss.
        
        Args:
            pred_rgb: Predicted RGB at sample points [B, N, 3]
            pred_sigma: Predicted sigma at sample points [B, N]
            gt_rgb: Ground truth RGB at sample points [B, N, 3]
            gt_sigma: Ground truth sigma at sample points [B, N]
            xy_plane: XY feature plane [B, C, H, W]
            xz_plane: XZ feature plane [B, C, H, W]
            yz_plane: YZ feature plane [B, C, H, W]
            pred_rendered: Optional rendered image [B, 3, H, W] for perceptual loss
            gt_image: Optional ground truth image [B, 3, H, W] for perceptual loss
            
        Returns:
            Loss dictionary
        """
        # RGB L2 loss
        rgb_loss = F.mse_loss(pred_rgb, gt_rgb)
        
        # Sigma L2 loss
        sigma_loss = F.mse_loss(pred_sigma, gt_sigma)
        
        # Plane regularization
        reg_losses = self.compute_plane_regularization(xy_plane, xz_plane, yz_plane)
        tv_loss = reg_losses['tv_loss']
        l1_loss = reg_losses['l1_loss']
        
        # Total loss
        total_loss = (
            self.rgb_weight * rgb_loss + 
            self.sigma_weight * sigma_loss + 
            self.tv_weight * tv_loss +
            self.l1_weight * l1_loss
        )
        
        # Optional perceptual loss on rendered images
        perceptual_loss = torch.tensor(0.0, device=pred_rgb.device)
        if self.perceptual_weight > 0 and pred_rendered is not None and gt_image is not None:
            perceptual_loss = F.l1_loss(pred_rendered, gt_image)
            total_loss = total_loss + self.perceptual_weight * perceptual_loss
        
        return {
            'rgb_loss': rgb_loss,
            'sigma_loss': sigma_loss,
            'tv_loss': tv_loss,
            'l1_loss': l1_loss,
            'perceptual_loss': perceptual_loss,
            'total_loss': total_loss,
        }