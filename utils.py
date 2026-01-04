"""
Utility functions for the NeRF training pipeline.
"""

import torch
import numpy as np

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEBUG = False

# Utility functions
img2mse = lambda x, y: torch.mean((x - y) ** 2)
log10 = torch.log(torch.tensor([10.0], device=device))
mse2psnr = lambda x: -10.0 * torch.log(x) / log10
to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)