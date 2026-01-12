"""
T-NeRF Training Script for NLPHead.

This script trains the VGGT model with NLPHead to predict NeRF MLP parameters
from multi-view images. Training uses pre-rendered images from generate_data.py.

Training Pipeline:
1. Load pre-rendered multi-view images from generate_data.py output
2. VGGT + NLPHead predicts MLP parameters from input images
3. Sample 3D points (both valid and empty regions from voxel grid)
4. Query predicted MLP at sampled points
5. Compare with ground truth voxel values at same points
6. Backpropagate loss to train NLPHead

Usage:
    python tnerf_training.py --data_dir /path/to/rendered_data --voxel_dir /path/to/features --pretrained_path /path/to/vggt.pt
    python tnerf_training.py --data_dir ../data/tnerf --voxel_dir ../data/nerf-mae/features --pretrained_path ../../../vggt/model_weights/model.pt
    python tnerf_training.py --data_dir E:/code/cv_finalproject/data/tnerf --voxel_dir E:/code/cv_finalproject/data/NeRF-MAE_pretrain/features --pretrained_path E:/code/cv_finalproject/vggt/model_weights/model.pt
    python tnerf_training.py \
    --voxel_dir "/media/fengwu/ZX1 1TB/code/cv_finalproject/data/NeRF-MAE_pretrain/features" \
    --data_dir "/media/fengwu/ZX1 1TB/code/cv_finalproject/data/tnerf" \
    --pretrained_path "/media/fengwu/ZX1 1TB/code/cv_finalproject/vggt/model_weights/model.pt" \
    --checkpoint_dir "/media/fengwu/ZX1 1TB/code/cv_finalproject/tnerf/checkpoints_tnerf" \
    --num_epochs 100 \
    
    # 从checkpoint继续训练
    python tnerf_training.py \
    --data_dir /path/to/tnerf/data \
    --voxel_dir /path/to/features \
    --head_type nlp \
    --resume /path/to/checkpoints/nlp_checkpoint_latest.pt \
    --num_epochs 100
        
    
    # latent head训练
    python tnerf_training.py --head_type latent --data_dir ../data/tnerf --voxel_dir ../data/nerf-mae/features --pretrained_path ../../../vggt/model_weights/model.pt
"""

import os
import sys
import argparse
import logging
from typing import Dict, Optional, Tuple, List

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from tnerf_loss import TNerfLoss, TriplaneLoss
from tnerf_mlp import BatchedNeRFMLP

# Add relevant project directories to path for local imports
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..'))
VGGT_REPO = os.path.join(PROJECT_ROOT, 'vggt')

sys.path.insert(0, CURRENT_DIR)
sys.path.insert(0, os.path.abspath(os.path.join(CURRENT_DIR, '..')))
sys.path.insert(0, os.path.abspath(os.path.join(CURRENT_DIR, '..', '..')))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, VGGT_REPO)

from tnerf_dataloader import TNerfDataset, create_tnerf_dataloaders
from tnerf.model.models.vggt import VGGT


def parse_args():
    parser = argparse.ArgumentParser(description="Train NLPHead with T-NeRF data")
    
    # Data paths
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to rendered data from generate_data.py")
    parser.add_argument("--voxel_dir", type=str, required=True,
                        help="Path to voxel features directory (e.g., pretrain/features)")
    
    # Model paths
    parser.add_argument("--pretrained_path", type=str, default=None,
                        help="Path to pretrained VGGT model weights")
    parser.add_argument("--checkpoint_dir", type=str, default="../checkpoints_tnerf",
                        help="Directory to save checkpoints")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    
    # Head type selection (nlp or latent)
    parser.add_argument("--head_type", type=str, default="nlp", choices=["nlp", "latent"],
                        help="Which head to train: 'nlp' for NLPHead, 'latent' for LatentHead")
    
    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=10,
                        help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=5,
                        help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="Weight decay")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Gradient clipping norm")
    
    # Model settings
    parser.add_argument("--image_size", type=int, default=336,
                        help="Image size for input")
    parser.add_argument("--num_views", type=int, default=5,
                        help="Number of views to load per sample")
    parser.add_argument("--min_views", type=int, default=2,
                        help="Minimum views for training (random sampling)")
    parser.add_argument("--max_views", type=int, default=5,
                        help="Maximum views for training (random sampling)")
    
    # Point sampling settings for loss computation
    parser.add_argument("--num_sample_points", type=int, default=4096,
                        help="Number of 3D points to sample for loss computation")
    parser.add_argument("--valid_point_ratio", type=float, default=0.7,
                        help="Ratio of valid (non-empty) points in sampling")
    parser.add_argument("--empty_value", type=float, default=-10000.0,
                        help="Sigma value marking empty voxels (default: -10000)")
    
    # Freeze settings
    parser.add_argument("--freeze_backbone", action="store_true", default=True,
                        help="Freeze VGGT backbone")
    parser.add_argument("--freeze_other_heads", action="store_true", default=True,
                        help="Freeze other heads")
    
    # Loss weights
    parser.add_argument("--rgb_weight", type=float, default=1.0,
                        help="Weight for RGB loss")
    parser.add_argument("--sigma_weight", type=float, default=0.1,
                        help="Weight for sigma loss")
    parser.add_argument("--reg_weight", type=float, default=0.001,
                        help="Weight for parameter regularization (NLPHead)")
    parser.add_argument("--tv_weight", type=float, default=0.01,
                        help="Weight for TV regularization (LatentHead)")
    parser.add_argument("--l1_weight", type=float, default=0.001,
                        help="Weight for L1 sparsity regularization (LatentHead)")
    
    # Logging
    parser.add_argument("--log_dir", type=str, default="./logs_tnerf",
                        help="TensorBoard log directory")
    parser.add_argument("--log_freq", type=int, default=50,
                        help="Logging frequency (steps)")
    parser.add_argument("--save_freq", type=int, default=5,
                        help="Checkpoint save frequency (epochs)")
    parser.add_argument("--vis_freq", type=int, default=200,
                        help="Visualization frequency (steps)")
    
    # Hardware
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Data loading workers")
    
    # Early stopping
    parser.add_argument("--early_stop_patience", type=int, default=10,
                        help="Early stopping patience (epochs)")
    parser.add_argument("--early_stop_min_delta", type=float, default=1e-4,
                        help="Minimum change to qualify as improvement")
    
    # Warmup settings
    parser.add_argument("--warmup_epochs", type=int, default=5,
                        help="Number of warmup epochs")
    parser.add_argument("--warmup_start_lr", type=float, default=1e-6,
                        help="Starting learning rate for warmup")
    
    return parser.parse_args()

class PointSampler:
    """
    Sample 3D points from voxel grid for loss computation.
    
    Samples a mix of valid (occupied) and empty points according to the specified ratio.
    This ensures the model learns both to predict density where objects exist
    and to predict empty space where there's nothing.
    
    In NeRF-MAE pretrain data, empty voxels are marked with sigma = -10000.
    """
    
    def __init__(
        self,
        num_points: int = 4096,
        valid_ratio: float = 0.7,
        empty_value: float = -10000.0,
    ):
        """
        Args:
            num_points: Total number of points to sample
            valid_ratio: Ratio of valid (occupied) points
            empty_value: Sigma value that marks empty voxels (default: -10000.0)
        """
        self.num_points = num_points
        self.valid_ratio = valid_ratio
        self.empty_value = empty_value
    
    def sample_points(
        self,
        rgbsigma: torch.Tensor,
        bbox_min: torch.Tensor,
        bbox_max: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample points from a single voxel grid.
        
        Args:
            rgbsigma: Voxel grid [X, Y, Z, 4] (RGB + sigma)
            bbox_min: Minimum bounds [3]
            bbox_max: Maximum bounds [3]
            
        Returns:
            points: Sampled 3D points [N, 3] in world coordinates
            gt_rgb: Ground truth RGB at sampled points [N, 3]
            gt_sigma: Ground truth sigma at sampled points [N]
        """
        device = rgbsigma.device
        X, Y, Z, _ = rgbsigma.shape
        
        # Separate sigma from RGB
        sigma_grid = rgbsigma[..., 3]  # [X, Y, Z]
        rgb_grid = rgbsigma[..., :3]   # [X, Y, Z, 3]
        
        # Find valid (occupied) voxels and empty voxels
        # In NeRF-MAE data, empty voxels are marked with sigma = -10000
        valid_mask = sigma_grid != self.empty_value  # Valid points (occupied regions)
        empty_mask = sigma_grid == self.empty_value  # Empty points (free space)
        
        # Get indices
        valid_indices = torch.nonzero(valid_mask, as_tuple=False)  # [N_valid, 3]
        empty_indices = torch.nonzero(empty_mask, as_tuple=False)  # [N_empty, 3]
        
        # Calculate number of samples for each type
        num_valid = int(self.num_points * self.valid_ratio)
        num_empty = self.num_points - num_valid
        
        # Clamp to available samples
        num_valid = min(num_valid, valid_indices.shape[0])
        num_empty = min(num_empty, empty_indices.shape[0])
        
        # If we don't have enough of one type, supplement with the other
        if num_valid < int(self.num_points * self.valid_ratio):
            num_empty = min(self.num_points - num_valid, empty_indices.shape[0])
        if num_empty < self.num_points - num_valid:
            num_valid = min(self.num_points - num_empty, valid_indices.shape[0])
        
        # Random sample from each set
        sampled_points = []
        
        if num_valid > 0:
            valid_perm = torch.randperm(valid_indices.shape[0], device=device)[:num_valid]
            valid_samples = valid_indices[valid_perm]  # [num_valid, 3]
            sampled_points.append(valid_samples)
        
        if num_empty > 0:
            empty_perm = torch.randperm(empty_indices.shape[0], device=device)[:num_empty]
            empty_samples = empty_indices[empty_perm]  # [num_empty, 3]
            sampled_points.append(empty_samples)
        
        if len(sampled_points) == 0:
            raise RuntimeError("Wrong: No points sampled; check voxel grid and empty_value setting.")
        else:
            sampled_indices = torch.cat(sampled_points, dim=0)  # [N_total, 3]
        
        # Add random offset within voxel for sub-voxel sampling
        noise = torch.rand_like(sampled_indices.float()) - 0.5  # [-0.5, 0.5]
        sampled_coords = sampled_indices.float() + noise
        
        # Convert to world coordinates
        voxel_size = (bbox_max - bbox_min) / torch.tensor([X, Y, Z], device=device, dtype=torch.float32)
        world_points = bbox_min + sampled_coords * voxel_size  # [N, 3]
        
        # Get ground truth values by trilinear interpolation
        # Normalize to [-1, 1] for grid_sample
        grid_coords = sampled_coords / torch.tensor([X-1, Y-1, Z-1], device=device, dtype=torch.float32)
        grid_coords = grid_coords * 2 - 1  # [-1, 1]
        
        # Reshape for grid_sample: [1, 1, 1, N, 3] -> sample from [1, 4, X, Y, Z]
        grid_coords = grid_coords.view(1, 1, 1, -1, 3)
        
        # Prepare voxel grid for grid_sample: [X, Y, Z, 4] -> [1, 4, Z, Y, X]
        voxel_for_sample = rgbsigma.permute(3, 2, 1, 0).unsqueeze(0)  # [1, 4, Z, Y, X]
        
        # Flip grid coords to match grid_sample convention (z, y, x order)
        grid_coords = grid_coords[..., [2, 1, 0]]
        
        # Sample
        sampled_values = F.grid_sample(
            voxel_for_sample,
            grid_coords,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True,
        )  # [1, 4, 1, 1, N]
        
        sampled_values = sampled_values.squeeze()  # [4, N]
        
        gt_rgb = sampled_values[:3].T  # [N, 3]
        gt_sigma = sampled_values[3]   # [N]
        
        return world_points, gt_rgb, gt_sigma


class EarlyStopping:
    """
    Early stopping to stop training when validation loss doesn't improve.
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-4, mode: str = 'min'):
        """
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss (lower is better), 'max' for metrics (higher is better)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        
    def __call__(self, score: float, epoch: int) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current validation score (loss or metric)
            epoch: Current epoch number
            
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False
        
        # Check for improvement
        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def state_dict(self) -> Dict:
        """Return state for checkpointing."""
        return {
            'counter': self.counter,
            'best_score': self.best_score,
            'best_epoch': self.best_epoch,
            'early_stop': self.early_stop,
        }
    
    def load_state_dict(self, state_dict: Dict):
        """Load state from checkpoint."""
        self.counter = state_dict['counter']
        self.best_score = state_dict['best_score']
        self.best_epoch = state_dict['best_epoch']
        self.early_stop = state_dict['early_stop']


class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    """
    Cosine learning rate scheduler with warmup.
    """
    
    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        warmup_start_lr: float = 1e-6,
        eta_min: float = 1e-6,
        last_epoch: int = -1,
    ):
        """
        Args:
            optimizer: Optimizer
            warmup_epochs: Number of warmup epochs
            max_epochs: Total number of epochs
            warmup_start_lr: Starting learning rate for warmup
            eta_min: Minimum learning rate
            last_epoch: Last epoch number
        """
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        
        # Store base learning rates
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """Calculate learning rate for current epoch."""
        if self.last_epoch < self.warmup_epochs:
            # Warmup phase: linear increase
            alpha = self.last_epoch / self.warmup_epochs
            return [
                self.warmup_start_lr + (base_lr - self.warmup_start_lr) * alpha
                for base_lr in self.base_lrs
            ]
        else:
            # Cosine annealing phase
            progress = (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            return [
                self.eta_min + (base_lr - self.eta_min) * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
                for base_lr in self.base_lrs
            ]


class TNerfTrainer:
    """
    Trainer for NLPHead using T-NeRF data (pre-rendered images + voxel grids).
    Uses point sampling for loss computation instead of image-based rendering.
    """
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        self.global_step = 0
        self.current_epoch = 0
        
        self._setup_logging()
        self._build_model()
        self._freeze_layers()
        self._build_optimizer()
        self._build_scheduler()
        self._build_early_stopping()
        self._build_point_sampler()
        self._build_loss()
        self._build_dataloaders()
        
        if args.resume:
            self._load_checkpoint(args.resume)
    
    def _setup_logging(self):
        """Setup logging and tensorboard."""
        os.makedirs(self.args.log_dir, exist_ok=True)
        os.makedirs(self.args.checkpoint_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(os.path.join(self.args.log_dir, "training.log")),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.writer = SummaryWriter(log_dir=self.args.log_dir)
        
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Args: {vars(self.args)}")
    
    def _build_model(self):
        """Build VGGT model with selected head (NLPHead or LatentHead)."""
        self.logger.info(f"Building VGGT model with {self.args.head_type} head...")
        
        # Enable only the selected head
        enable_nlp = (self.args.head_type == "nlp")
        enable_latent = (self.args.head_type == "latent")
        
        self.model = VGGT(
            enable_camera=False,
            enable_point=False,
            enable_depth=False,
            enable_track=False,
            enable_nlp=enable_nlp,
            enable_latent=enable_latent,
        )
        
        # Load pretrained weights
        if self.args.pretrained_path and os.path.exists(self.args.pretrained_path):
            self.logger.info(f"Loading pretrained weights from {self.args.pretrained_path}")
            state_dict = torch.load(self.args.pretrained_path, map_location="cpu")
            
            if "model" in state_dict:
                state_dict = state_dict["model"]
            elif "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            self.logger.info(f"Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        
        self.model = self.model.to(self.device)
    
    def _freeze_layers(self):
        """Freeze backbone and other heads."""
        if self.args.freeze_backbone:
            self.logger.info("Freezing backbone...")
            for param in self.model.aggregator.parameters():
                param.requires_grad = False
        
        if self.args.freeze_other_heads:
            self.logger.info("Freezing other heads...")
            # Freeze heads that are not being trained
            heads_to_freeze = ['camera_head', 'point_head', 'depth_head', 'track_head']
            # Also freeze the other head (nlp or latent) that we're not training
            if self.args.head_type == "nlp":
                heads_to_freeze.append('latent_head')
            else:
                heads_to_freeze.append('nmlp_head')
            
            for head_name in heads_to_freeze:
                head = getattr(self.model, head_name, None)
                if head is not None:
                    for param in head.parameters():
                        param.requires_grad = False
        
        # Count trainable parameters
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"Trainable: {trainable:,} / {total:,}")
    
    def _build_optimizer(self):
        """Build optimizer based on head type."""
        params = []
        
        # Add parameters for the selected head
        if self.args.head_type == "nlp" and self.model.nmlp_head is not None:
            params.append({
                "params": self.model.nmlp_head.parameters(),
                "lr": self.args.lr,
                "name": "nmlp_head"
            })
        elif self.args.head_type == "latent" and self.model.latent_head is not None:
            params.append({
                "params": self.model.latent_head.parameters(),
                "lr": self.args.lr,
                "name": "latent_head"
            })
        
        if not self.args.freeze_backbone:
            params.append({
                "params": self.model.aggregator.parameters(),
                "lr": self.args.lr * 0.01,
                "name": "aggregator"
            })
        
        if len(params) == 0:
            raise ValueError(f"No parameters to optimize for head_type={self.args.head_type}")
        
        self.optimizer = optim.AdamW(params, weight_decay=self.args.weight_decay)
    
    def _build_scheduler(self):
        """Build learning rate scheduler with warmup."""
        self.scheduler = CosineWarmupScheduler(
            optimizer=self.optimizer,
            warmup_epochs=self.args.warmup_epochs,
            max_epochs=self.args.num_epochs,
            warmup_start_lr=self.args.warmup_start_lr,
            eta_min=self.args.lr * 0.01,
        )
        
        self.logger.info(
            f"Scheduler: warmup_epochs={self.args.warmup_epochs}, "
            f"warmup_start_lr={self.args.warmup_start_lr:.2e}"
        )
    
    def _build_early_stopping(self):
        """Build early stopping."""
        self.early_stopping = EarlyStopping(
            patience=self.args.early_stop_patience,
            min_delta=self.args.early_stop_min_delta,
            mode='min',
        )
        
        self.logger.info(
            f"Early stopping: patience={self.args.early_stop_patience}, "
            f"min_delta={self.args.early_stop_min_delta}"
        )
    
    def _build_point_sampler(self):
        """Build point sampler for loss computation."""
        self.point_sampler = PointSampler(
            num_points=self.args.num_sample_points,
            valid_ratio=self.args.valid_point_ratio,
            empty_value=self.args.empty_value,
        )
    
    def _build_loss(self):
        """Build loss function based on head type."""
        if self.args.head_type == "nlp":
            self.loss_fn = TNerfLoss(
                rgb_weight=self.args.rgb_weight,
                sigma_weight=self.args.sigma_weight,
                reg_weight=self.args.reg_weight,
            )
            self.logger.info("Using TNerfLoss for NLPHead training")
        else:  # latent
            self.loss_fn = TriplaneLoss(
                rgb_weight=self.args.rgb_weight,
                sigma_weight=self.args.sigma_weight,
                tv_weight=self.args.tv_weight,
                l1_weight=self.args.l1_weight,
            )
            self.logger.info("Using TriplaneLoss for LatentHead training")
    
    def _build_dataloaders(self):
        """Build data loaders using T-NeRF dataloader."""
        self.logger.info(f"Loading data from {self.args.data_dir}")
        self.logger.info(f"Voxel data from {self.args.voxel_dir}")
        
        self.train_loader, self.val_loader, self.test_loader = create_tnerf_dataloaders(
            data_dir=self.args.data_dir,
            voxel_dir=self.args.voxel_dir,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            image_size=(self.args.image_size, self.args.image_size),
            num_views=self.args.num_views,
            load_voxel=True,  # Need voxel for point sampling loss
            min_views=self.args.min_views,
            max_views=self.args.max_views,
        )
        
        self.logger.info(f"Train batches: {len(self.train_loader)}")
        self.logger.info(f"Val batches: {len(self.val_loader)}")
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save checkpoint with head type information."""
        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "head_type": self.args.head_type,  # Save which head was trained
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "early_stopping_state_dict": self.early_stopping.state_dict(),
            "args": vars(self.args),
        }
        
        # Use head_type in filename for clarity
        path = os.path.join(self.args.checkpoint_dir, f"{self.args.head_type}_checkpoint_{epoch:04d}.pt")
        torch.save(checkpoint, path)
        
        latest = os.path.join(self.args.checkpoint_dir, f"{self.args.head_type}_checkpoint_latest.pt")
        torch.save(checkpoint, latest)
        
        if is_best:
            best = os.path.join(self.args.checkpoint_dir, f"{self.args.head_type}_checkpoint_best.pt")
            torch.save(checkpoint, best)
        
        self.logger.info(f"Saved checkpoint: {path}")
    
    def _load_checkpoint(self, path: str):
        """Load checkpoint and verify head type compatibility."""
        self.logger.info(f"Loading checkpoint: {path}")
        ckpt = torch.load(path, map_location=self.device)
        
        # Verify head type matches
        saved_head_type = ckpt.get("head_type", "nlp")  # Default to nlp for old checkpoints
        if saved_head_type != self.args.head_type:
            self.logger.warning(
                f"Checkpoint was trained with head_type='{saved_head_type}', "
                f"but current training uses head_type='{self.args.head_type}'. "
                f"Loading compatible weights only."
            )
        
        # Load model state dict with strict=False to handle head type differences
        missing, unexpected = self.model.load_state_dict(ckpt["model_state_dict"], strict=False)
        if missing:
            self.logger.info(f"Missing keys (expected for different head types): {len(missing)}")
        if unexpected:
            self.logger.info(f"Unexpected keys: {len(unexpected)}")
        
        # Only load optimizer/scheduler state if head types match
        if saved_head_type == self.args.head_type:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            
            if "early_stopping_state_dict" in ckpt:
                self.early_stopping.load_state_dict(ckpt["early_stopping_state_dict"])
            
            self.current_epoch = ckpt["epoch"] + 1  # Resume from next epoch
            self.global_step = ckpt["global_step"]
            self.logger.info(f"Resuming training from epoch {self.current_epoch}")
        else:
            # Different head type - start fresh training for the new head
            self.logger.info("Different head type - starting fresh training with backbone weights loaded")
    
    def _move_to_device(self, batch: Dict) -> Dict:
        """Move batch to device."""
        result = {}
        for key, value in batch.items():
            if torch.is_tensor(value):
                result[key] = value.to(self.device)
            elif isinstance(value, list) and len(value) > 0 and torch.is_tensor(value[0]):
                result[key] = [v.to(self.device) for v in value]
            else:
                result[key] = value
        return result
    
    def _train_step(self, batch: Dict) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Single training step. Supports both NLPHead and LatentHead.
        
        Args:
            batch: Dictionary from dataloader
            
        Returns:
            Loss dictionary with pred_rgb/pred_sigma for visualization
        """
        batch = self._move_to_device(batch)
        
        # Get input images - all views are input (no separate target views)
        images = batch["images"]  # [B, num_views, 3, H, W]
        
        # Forward through VGGT
        predictions = self.model(images)
        
        # Get voxel grids and bounding boxes
        rgbsigma_list = batch["rgbsigma"]  # List of [X, Y, Z, 4]
        bbox_min = batch["bbox_min"]  # [B, 3]
        bbox_max = batch["bbox_max"]  # [B, 3]
        
        # Sample points and compute loss for each sample in batch
        batch_size = len(rgbsigma_list)
        
        all_pred_rgb = []
        all_pred_sigma = []
        all_gt_rgb = []
        all_gt_sigma = []
        
        if self.args.head_type == "nlp":
            # === NLPHead Training ===
            mlp_params = predictions["nmlp"]  # [B, num_params]
            
            for b in range(batch_size):
                voxel = rgbsigma_list[b].to(self.device)
                b_min = bbox_min[b]
                b_max = bbox_max[b]
                
                # Sample points from voxel grid
                points, gt_rgb, gt_sigma = self.point_sampler.sample_points(
                    voxel, b_min, b_max
                )
                
                # Normalize points to [-1, 1] for MLP query
                points_normalized = (points - b_min) / (b_max - b_min) * 2 - 1
                
                # Query MLP at sampled points
                points_query = points_normalized.unsqueeze(0)  # [1, N, 3]
                
                # Get the b-th MLP's output
                single_mlp = BatchedNeRFMLP(mlp_params[b:b+1])
                pred_output = single_mlp(points_query)  # [1, N, 4]
                
                pred_rgb_b = pred_output[0, :, :3]  # [N, 3]
                pred_sigma_b = pred_output[0, :, 3]  # [N]
                
                all_pred_rgb.append(pred_rgb_b)
                all_pred_sigma.append(pred_sigma_b)
                all_gt_rgb.append(gt_rgb)
                all_gt_sigma.append(gt_sigma)
            
            # Stack for loss computation
            pred_rgb = torch.stack(all_pred_rgb, dim=0)    # [B, N, 3]
            pred_sigma = torch.stack(all_pred_sigma, dim=0)  # [B, N]
            gt_rgb = torch.stack(all_gt_rgb, dim=0)        # [B, N, 3]
            gt_sigma = torch.stack(all_gt_sigma, dim=0)    # [B, N]
            
            # Compute NLP loss
            loss_dict = self.loss_fn(pred_rgb, pred_sigma, gt_rgb, gt_sigma, mlp_params)
            
        else:
            # === LatentHead Training ===
            xy_plane = predictions["xy_plane"]  # [B, 32, 64, 64]
            xz_plane = predictions["xz_plane"]  # [B, 32, 64, 64]
            yz_plane = predictions["yz_plane"]  # [B, 32, 64, 64]
            
            for b in range(batch_size):
                voxel = rgbsigma_list[b].to(self.device)
                b_min = bbox_min[b]
                b_max = bbox_max[b]
                
                # Sample points from voxel grid
                points, gt_rgb, gt_sigma = self.point_sampler.sample_points(
                    voxel, b_min, b_max
                )
                
                # Normalize points to [-1, 1] for triplane query
                points_normalized = (points - b_min) / (b_max - b_min) * 2 - 1
                
                # Generate random view directions (for training, we use random directions)
                directions = torch.randn_like(points_normalized)
                directions = F.normalize(directions, dim=-1)
                
                # Query triplane at sampled points using latent_head's query function
                points_query = points_normalized.unsqueeze(0)  # [1, N, 3]
                directions_query = directions.unsqueeze(0)  # [1, N, 3]
                
                # Latent head returns (sigma, rgb)
                pred_sigma_b, pred_rgb_b = self.model.latent_head.query_points(
                    xy_plane[b:b+1], xz_plane[b:b+1], yz_plane[b:b+1],
                    points_query, directions_query
                )
                
                all_pred_rgb.append(pred_rgb_b[0])  # [N, 3]
                all_pred_sigma.append(pred_sigma_b[0, :, 0])  # [N]
                all_gt_rgb.append(gt_rgb)
                all_gt_sigma.append(gt_sigma)
            
            # Stack for loss computation
            pred_rgb = torch.stack(all_pred_rgb, dim=0)    # [B, N, 3]
            pred_sigma = torch.stack(all_pred_sigma, dim=0)  # [B, N]
            gt_rgb = torch.stack(all_gt_rgb, dim=0)        # [B, N, 3]
            gt_sigma = torch.stack(all_gt_sigma, dim=0)    # [B, N]
            
            # Compute Triplane loss with regularization
            loss_dict = self.loss_fn(
                pred_rgb, pred_sigma, gt_rgb, gt_sigma,
                xy_plane, xz_plane, yz_plane
            )
        
        return loss_dict, pred_rgb, gt_rgb
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        epoch_losses = {}
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            self.optimizer.zero_grad()
            
            try:
                loss_dict, pred_rgb, gt_rgb = self._train_step(batch)
                total_loss = loss_dict["total_loss"]
                
                total_loss.backward()
                
                if self.args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.args.grad_clip
                    )
                
                self.optimizer.step()
                
                # Accumulate losses
                for key, value in loss_dict.items():
                    if key not in epoch_losses:
                        epoch_losses[key] = 0.0
                    epoch_losses[key] += value.item()
                
                # Logging
                if self.global_step % self.args.log_freq == 0:
                    for key, value in loss_dict.items():
                        self.writer.add_scalar(f"train/{key}", value.item(), self.global_step)
                    
                    # Log learning rate
                    lr = self.optimizer.param_groups[0]['lr']
                    self.writer.add_scalar("train/lr", lr, self.global_step)
                
                self.global_step += 1
                pbar.set_postfix({
                    "loss": total_loss.item(),
                    "rgb": loss_dict["rgb_loss"].item(),
                    "sigma": loss_dict["sigma_loss"].item(),
                })
                
            except Exception as e:
                self.logger.warning(f"Error in batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Average losses
        num_batches = len(self.train_loader)
        for key in epoch_losses:
            epoch_losses[key] /= max(num_batches, 1)
        
        return epoch_losses
    
    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate."""
        self.model.eval()
        
        val_losses = {}
        
        for batch in tqdm(self.val_loader, desc="Validation"):
            try:
                loss_dict, _, _ = self._train_step(batch)
                
                for key, value in loss_dict.items():
                    if key not in val_losses:
                        val_losses[key] = 0.0
                    val_losses[key] += value.item()
            except Exception as e:
                self.logger.warning(f"Val error: {e}")
                continue
        
        num_batches = len(self.val_loader)
        for key in val_losses:
            val_losses[key] /= max(num_batches, 1)
        
        return val_losses
    
    def train(self):
        """Main training loop."""
        self.logger.info("Starting training...")
        best_val_loss = float("inf")
        
        for epoch in range(self.current_epoch, self.args.num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_losses = self.train_epoch(epoch)
            self.logger.info(
                f"Epoch {epoch} - Train: {train_losses.get('total_loss', 0):.6f}"
            )
            
            # Log epoch losses
            for key, value in train_losses.items():
                self.writer.add_scalar(f"epoch_train/{key}", value, epoch)
            
            # Validate
            if len(self.val_loader) > 0:
                val_losses = self.validate(epoch)
                val_total_loss = val_losses.get('total_loss', float('inf'))
                
                self.logger.info(
                    f"Epoch {epoch} - Val: rgb={val_losses.get('rgb_loss', 0):.6f}, "
                    f"sigma={val_losses.get('sigma_loss', 0):.6f}, "
                    f"total={val_total_loss:.6f}"
                )
                
                for key, value in val_losses.items():
                    self.writer.add_scalar(f"epoch_val/{key}", value, epoch)
                
                # Check if best model
                is_best = val_total_loss < best_val_loss
                if is_best:
                    best_val_loss = val_total_loss
                
                # Early stopping check
                should_stop = self.early_stopping(val_total_loss, epoch)
                
                # Log early stopping metrics
                self.writer.add_scalar("early_stopping/counter", self.early_stopping.counter, epoch)
                self.writer.add_scalar("early_stopping/best_score", self.early_stopping.best_score, epoch)
                
                if should_stop:
                    self.logger.info(
                        f"Early stopping triggered at epoch {epoch}. "
                        f"Best validation loss: {self.early_stopping.best_score:.6f} "
                        f"at epoch {self.early_stopping.best_epoch}"
                    )
                    self._save_checkpoint(epoch, is_best)
                    break
            else:
                is_best = False
            
            # Update scheduler
            self.scheduler.step()
            
            # Log learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar("learning_rate", current_lr, epoch)
            self.logger.info(f"Learning rate: {current_lr:.2e}")
            
            # Save checkpoint
            if (epoch + 1) % self.args.save_freq == 0 or is_best:
                self._save_checkpoint(epoch, is_best)
        
        self.logger.info("Training completed!")
        if self.early_stopping.best_score is not None:
            self.logger.info(
                f"Best validation loss: {self.early_stopping.best_score:.6f} "
                f"at epoch {self.early_stopping.best_epoch}"
            )
        self.writer.close()

def main():
    args = parse_args()
    trainer = TNerfTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()