"""
Main entry point for the NeRF training pipeline.
"""


import torch
import numpy as np

from train import train

if __name__=='__main__':
    np.random.seed(0)
    # Use new API instead of deprecated set_default_tensor_type
    if torch.cuda.is_available():
        torch.set_default_device('cuda')
        print("Using CUDA")
    torch.set_default_dtype(torch.float32)
    train()