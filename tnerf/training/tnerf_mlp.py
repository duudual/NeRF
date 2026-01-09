from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchedNeRFMLP(nn.Module):
    """
    Batched NeRF MLP that can be initialized with predicted parameters.
    Supports batched query of 3D points.
    
    Structure:
        - pos_linear: (63, 128) - processes positional encoded position
        - sigma_linear: (128, 1) - outputs density
        - color_linear: (128 + 27, 3) - outputs RGB color
    """
    
    # Parameter dimensions
    POS_IN_DIM = 63      # 3 + 3 * 2 * 10 (position + positional encoding)
    HIDDEN_DIM = 128
    DIR_IN_DIM = 27      # 3 + 3 * 2 * 4 (direction + positional encoding)
    SIGMA_OUT_DIM = 1
    COLOR_OUT_DIM = 3
    
    # Parameter counts
    POS_LINEAR_PARAMS = POS_IN_DIM * HIDDEN_DIM + HIDDEN_DIM  # 63*128 + 128 = 8192
    SIGMA_LINEAR_PARAMS = HIDDEN_DIM * SIGMA_OUT_DIM + SIGMA_OUT_DIM  # 128*1 + 1 = 129
    COLOR_LINEAR_PARAMS = (HIDDEN_DIM + DIR_IN_DIM) * COLOR_OUT_DIM + COLOR_OUT_DIM  # 155*3 + 3 = 468
    
    TOTAL_PARAMS = POS_LINEAR_PARAMS + SIGMA_LINEAR_PARAMS + COLOR_LINEAR_PARAMS  # 8789
    
    def __init__(self, params: torch.Tensor):
        """
        Initialize with predicted parameters.
        
        Args:
            params: Flattened parameters tensor [B, TOTAL_PARAMS]
        """
        super().__init__()
        
        if params.dim() == 1:
            params = params.unsqueeze(0)
        
        self.batch_size = params.shape[0]
        self.device = params.device
        
        # Parse parameters
        idx = 0
        
        # pos_linear: weight (B, 128, 63), bias (B, 128)
        pos_weight_size = self.POS_IN_DIM * self.HIDDEN_DIM
        self.pos_weight = params[:, idx:idx + pos_weight_size].view(
            self.batch_size, self.HIDDEN_DIM, self.POS_IN_DIM
        )
        idx += pos_weight_size
        self.pos_bias = params[:, idx:idx + self.HIDDEN_DIM]
        idx += self.HIDDEN_DIM
        
        # sigma_linear: weight (B, 1, 128), bias (B, 1)
        sigma_weight_size = self.HIDDEN_DIM * self.SIGMA_OUT_DIM
        self.sigma_weight = params[:, idx:idx + sigma_weight_size].view(
            self.batch_size, self.SIGMA_OUT_DIM, self.HIDDEN_DIM
        )
        idx += sigma_weight_size
        self.sigma_bias = params[:, idx:idx + self.SIGMA_OUT_DIM]
        idx += self.SIGMA_OUT_DIM
        
        # color_linear: weight (B, 3, 155), bias (B, 3)
        color_in_dim = self.HIDDEN_DIM + self.DIR_IN_DIM
        color_weight_size = color_in_dim * self.COLOR_OUT_DIM
        self.color_weight = params[:, idx:idx + color_weight_size].view(
            self.batch_size, self.COLOR_OUT_DIM, color_in_dim
        )
        idx += color_weight_size
        self.color_bias = params[:, idx:idx + self.COLOR_OUT_DIM]
    
    def forward(
        self, 
        points: torch.Tensor, 
        dirs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Query the MLP at given points.
        
        Args:
            points: Query points [B, N, 3] in normalized [-1, 1] coordinates
            dirs: View directions [B, N, 3] (optional, uses default if not provided)
            
        Returns:
            Output tensor [B, N, 4] (RGB + sigma)
        """
        B, N, _ = points.shape
        
        # Apply positional encoding to points (10 frequencies)
        pos_encoded = positional_encoding(points, num_freqs=10)  # [B, N, 63]
        
        # Create default view direction (towards camera) if not provided
        if dirs is None:
            dirs = torch.zeros_like(points)
            dirs[..., 2] = -1  # Looking towards -z
        
        # Apply positional encoding to directions (4 frequencies)
        dir_encoded = positional_encoding(dirs, num_freqs=4)  # [B, N, 27]
        
        # Forward through MLP layers (batched matrix multiplication)
        # h = ReLU(pos_encoded @ pos_weight.T + pos_bias)
        h = torch.bmm(pos_encoded, self.pos_weight.transpose(1, 2))  # [B, N, 128]
        h = h + self.pos_bias.unsqueeze(1)  # [B, N, 128]
        h = F.relu(h)
        
        # sigma = h @ sigma_weight.T + sigma_bias
        sigma = torch.bmm(h, self.sigma_weight.transpose(1, 2))  # [B, N, 1]
        sigma = sigma + self.sigma_bias.unsqueeze(1)
        sigma = sigma.squeeze(-1)  # [B, N]
        
        # color = sigmoid(cat(h, dir_encoded) @ color_weight.T + color_bias)
        h_with_dir = torch.cat([h, dir_encoded], dim=-1)  # [B, N, 155]
        color = torch.bmm(h_with_dir, self.color_weight.transpose(1, 2))  # [B, N, 3]
        color = color + self.color_bias.unsqueeze(1)
        color = torch.sigmoid(color)  # [B, N, 3]
        
        # Combine output
        output = torch.cat([color, sigma.unsqueeze(-1)], dim=-1)  # [B, N, 4]
        
        return output
    
def positional_encoding(x: torch.Tensor, num_freqs: int = 10) -> torch.Tensor:
    """
    Apply positional encoding to input tensor.
    
    Args:
        x: Input tensor [..., D]
        num_freqs: Number of frequency bands
        
    Returns:
        Encoded tensor [..., D + D * 2 * num_freqs]
    """
    freqs = 2.0 ** torch.arange(num_freqs, device=x.device, dtype=x.dtype)
    x_exp = x.unsqueeze(-1) * freqs  # [..., D, num_freqs]
    
    encoded = torch.cat([
        x,
        torch.sin(x_exp).flatten(start_dim=-2),
        torch.cos(x_exp).flatten(start_dim=-2),
    ], dim=-1)
    
    return encoded