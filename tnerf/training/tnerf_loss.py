from typing import Dict
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