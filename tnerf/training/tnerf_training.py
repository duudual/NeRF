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

from tnerf_loss import TNerfLoss
from tnerf_mlp import BatchedNeRFMLP

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tnerf_dataloader import TNerfDataset, create_tnerf_dataloaders
from model.models.vggt import VGGT




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
        """Build VGGT model with NLPHead."""
        self.logger.info("Building VGGT model...")
        
        self.model = VGGT(
            enable_camera=False,
            enable_point=False,
            enable_depth=False,
            enable_track=False,
            enable_nlp=True,
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
            for head_name in ['camera_head', 'point_head', 'depth_head', 'track_head']:
                head = getattr(self.model, head_name, None)
                if head is not None:
                    for param in head.parameters():
                        param.requires_grad = False
        
        # Count trainable parameters
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"Trainable: {trainable:,} / {total:,}")
    
    def _build_optimizer(self):
        """Build optimizer."""
        params = []
        
        if self.model.nmlp_head is not None:
            params.append({
                "params": self.model.nmlp_head.parameters(),
                "lr": self.args.lr,
                "name": "nmlp_head"
            })
        
        if not self.args.freeze_backbone:
            params.append({
                "params": self.model.aggregator.parameters(),
                "lr": self.args.lr * 0.01,
                "name": "aggregator"
            })
        
        self.optimizer = optim.AdamW(params, weight_decay=self.args.weight_decay)
    
    def _build_scheduler(self):
        """Build learning rate scheduler."""
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.args.num_epochs,
            eta_min=self.args.lr * 0.01,
        )
    
    def _build_point_sampler(self):
        """Build point sampler for loss computation."""
        self.point_sampler = PointSampler(
            num_points=self.args.num_sample_points,
            valid_ratio=self.args.valid_point_ratio,
            empty_value=self.args.empty_value,
        )
    
    def _build_loss(self):
        """Build loss function."""
        self.loss_fn = TNerfLoss(
            rgb_weight=self.args.rgb_weight,
            sigma_weight=self.args.sigma_weight,
            reg_weight=self.args.reg_weight,
        )
    
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
        """Save checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "args": vars(self.args),
        }
        
        path = os.path.join(self.args.checkpoint_dir, f"checkpoint_{epoch:04d}.pt")
        torch.save(checkpoint, path)
        
        latest = os.path.join(self.args.checkpoint_dir, "checkpoint_latest.pt")
        torch.save(checkpoint, latest)
        
        if is_best:
            best = os.path.join(self.args.checkpoint_dir, "checkpoint_best.pt")
            torch.save(checkpoint, best)
        
        self.logger.info(f"Saved checkpoint: {path}")
    
    def _load_checkpoint(self, path: str):
        """Load checkpoint."""
        self.logger.info(f"Loading checkpoint: {path}")
        ckpt = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.current_epoch = ckpt["epoch"]
        self.global_step = ckpt["global_step"]
    
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
        Single training step.
        
        Args:
            batch: Dictionary from dataloader
            
        Returns:
            Loss dictionary with pred_rgb/pred_sigma for visualization
        """
        batch = self._move_to_device(batch)
        
        # Get input images - all views are input (no separate target views)
        images = batch["images"]  # [B, num_views, 3, H, W]
        
        # Forward through VGGT to get MLP parameters
        predictions = self.model(images)
        mlp_params = predictions["nmlp"]  # [B, num_params]
        
        # Build batched MLP from predicted parameters
        nerf_mlp = BatchedNeRFMLP(mlp_params)
        
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
            # BatchedNeRFMLP expects [B, N, 3] so we need to unsqueeze and select
            points_query = points_normalized.unsqueeze(0)  # [1, N, 3]
            
            # Get the b-th MLP's output
            # First, get single sample's MLP params
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
        
        # Compute loss
        loss_dict = self.loss_fn(pred_rgb, pred_sigma, gt_rgb, gt_sigma, mlp_params)
        
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
                self.logger.info(
                    f"Epoch {epoch} - Val: rgb={val_losses.get('rgb_loss', 0):.6f}, "
                    f"sigma={val_losses.get('sigma_loss', 0):.6f}, "
                    f"total={val_losses.get('total_loss', 0):.6f}"
                )
                
                for key, value in val_losses.items():
                    self.writer.add_scalar(f"epoch_val/{key}", value, epoch)
                
                is_best = val_losses.get("total_loss", float("inf")) < best_val_loss
                if is_best:
                    best_val_loss = val_losses["total_loss"]
            else:
                is_best = False
            
            # Update scheduler
            self.scheduler.step()
            
            # Save checkpoint
            if (epoch + 1) % self.args.save_freq == 0 or is_best:
                self._save_checkpoint(epoch, is_best)
        
        self.logger.info("Training completed!")
        self.writer.close()

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
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints_tnerf",
                        help="Directory to save checkpoints")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    
    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=100,
                        help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="Weight decay")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Gradient clipping norm")
    
    # Model settings
    parser.add_argument("--image_size", type=int, default=256,
                        help="Image size for input")
    parser.add_argument("--num_views", type=int, default=8,
                        help="Number of views to load per sample")
    parser.add_argument("--min_views", type=int, default=2,
                        help="Minimum views for training (random sampling)")
    parser.add_argument("--max_views", type=int, default=8,
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
                        help="Weight for parameter regularization")
    
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
    
    return parser.parse_args()


def main():
    args = parse_args()
    trainer = TNerfTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()