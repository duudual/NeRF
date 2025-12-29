"""
Main entry point for the NeRF training pipeline.
"""


import torch
import numpy as np

from train import train

if __name__=='__main__':
    np.random.seed(0)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    train()