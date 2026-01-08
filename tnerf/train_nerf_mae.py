"""
T-NeRF Training Script for NLPHead.

This script trains the VGGT model with NLPHead to predict NeRF MLP parameters
from multi-view images. Training uses NeRF-MAE pretrain data (voxel grids)
to generate supervision.

Training Pipeline:
1. Load NeRF-MAE voxel grid data
2. Render multi-view images from voxel grid (input to VGGT)
3. VGGT + NLPHead predicts MLP parameters from images
4. Use predicted MLP to render new views
5. Compare rendered views with voxel-rendered ground truth
6. Backpropagate loss to train NLPHead

Usage:
    python train_nerf_mae.py --data_dir /path/to/pretrain --pretrained_path /path/to/vggt.pt
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nerf_mae_dataloader import NeRFMAEDataset, create_nerf_mae_dataloaders
from volume_renderer import VolumeRenderer, VoxelRenderer, BatchedNeRFMLP
from vggt.models.vggt import VGGT


def parse_args():
    parser = argparse.ArgumentParser(description="Train NLPHead with NeRF-MAE data")
    
    # Data paths
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to NeRF-MAE pretrain data directory")
    
    # Model paths
    parser.add_argument("--pretrained_path", type=str, default=None,
                        help="Path to pretrained VGGT model weights")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints_nerf_mae",
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
                        help="Image size for rendering")
    parser.add_argument("--num_input_views", type=int, default=4,
                        help="Number of input views for VGGT")
    parser.add_argument("--num_target_views", type=int, default=2,
                        help="Number of target views for loss computation")
    parser.add_argument("--num_samples", type=int, default=64,
                        help="Number of samples per ray")
    
    # Freeze settings
    parser.add_argument("--freeze_backbone", action="store_true", default=True,
                        help="Freeze VGGT backbone")
    parser.add_argument("--freeze_other_heads", action="store_true", default=True,
                        help="Freeze other heads")
    
    # Loss weights
    parser.add_argument("--rgb_weight", type=float, default=1.0,
                        help="Weight for RGB loss")
    parser.add_argument("--reg_weight", type=float, default=0.001,
                        help="Weight for parameter regularization")
    
    # Logging
    parser.add_argument("--log_dir", type=str, default="./logs_nerf_mae",
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


class NeRFMAELoss(nn.Module):
    """
    Loss function for training NLPHead with NeRF-MAE data.
    
    Computes:
    1. RGB reconstruction loss between predicted MLP renders and voxel renders
    2. Parameter regularization loss
    """
    
    def __init__(
        self,
        rgb_weight: float = 1.0,
        reg_weight: float = 0.001,
        perceptual_weight: float = 0.0,
    ):
        super().__init__()
        self.rgb_weight = rgb_weight
        self.reg_weight = reg_weight
        self.perceptual_weight = perceptual_weight
    
    def forward(
        self,
        pred_rgb: torch.Tensor,
        target_rgb: torch.Tensor,
        mlp_params: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss.
        
        Args:
            pred_rgb: Predicted RGB images [B, S, 3, H, W] or [B, 3, H, W]
            target_rgb: Target RGB images (same shape)
            mlp_params: Predicted MLP parameters [B, num_params]
            
        Returns:
            Loss dictionary
        """
        # RGB L2 loss
        rgb_loss = F.mse_loss(pred_rgb, target_rgb)
        
        # Parameter regularization
        reg_loss = torch.mean(mlp_params ** 2)
        
        # Total loss
        total_loss = self.rgb_weight * rgb_loss + self.reg_weight * reg_loss
        
        return {
            'rgb_loss': rgb_loss,
            'reg_loss': reg_loss,
            'total_loss': total_loss,
        }


class NeRFMAETrainer:
    """
    Trainer for NLPHead using NeRF-MAE data.
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
        self._build_renderers()
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
    
    def _build_renderers(self):
        """Build volume renderers."""
        self.mlp_renderer = VolumeRenderer(
            num_samples=self.args.num_samples,
            white_background=True,
        ).to(self.device)
        
        self.voxel_renderer = VoxelRenderer(
            num_samples=self.args.num_samples * 2,
            white_background=True,
        ).to(self.device)
    
    def _build_loss(self):
        """Build loss function."""
        self.loss_fn = NeRFMAELoss(
            rgb_weight=self.args.rgb_weight,
            reg_weight=self.args.reg_weight,
        )
    
    def _build_dataloaders(self):
        """Build data loaders."""
        self.logger.info(f"Loading data from {self.args.data_dir}")
        
        train_dataset = NeRFMAEDataset(
            data_dir=self.args.data_dir,
            split="train",
            image_size=(self.args.image_size, self.args.image_size),
            num_views=self.args.num_input_views + self.args.num_target_views,
            num_samples=self.args.num_samples * 2,
        )
        
        val_dataset = NeRFMAEDataset(
            data_dir=self.args.data_dir,
            split="val",
            image_size=(self.args.image_size, self.args.image_size),
            num_views=self.args.num_input_views + self.args.num_target_views,
            num_samples=self.args.num_samples * 2,
        )
        
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        
        self.val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True,
        )
        
        self.logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
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
    
    def _train_step(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """
        Single training step.
        
        Args:
            batch: Dictionary from dataloader
            
        Returns:
            Loss dictionary
        """
        batch = self._move_to_device(batch)
        
        # Split views into input and target
        all_images = batch["images"]  # [B, S, 3, H, W]
        input_images = all_images[:, :self.args.num_input_views]  # [B, num_input, 3, H, W]
        target_view_indices = list(range(self.args.num_input_views, 
                                         self.args.num_input_views + self.args.num_target_views))
        
        # Normalize images for VGGT (expects [0, 1])
        input_images_norm = input_images
        
        # Forward through VGGT to get MLP parameters
        predictions = self.model(input_images_norm)
        mlp_params = predictions["nmlp"]  # [B, num_params]
        
        # Get target camera poses and intrinsics
        camera_poses = batch["camera_poses"]  # [B, S, 4, 4]
        camera_intrinsics = batch["camera_intrinsics"]  # [B, 3, 3]
        
        # Render target views using predicted MLP
        target_poses = camera_poses[:, target_view_indices]  # [B, num_target, 4, 4]
        
        rendered = self.mlp_renderer(
            params=mlp_params,
            camera_poses=target_poses,
            camera_intrinsics=camera_intrinsics,
            image_size=(self.args.image_size, self.args.image_size),
            bbox_min=batch["bbox_min"],
            bbox_max=batch["bbox_max"],
        )
        pred_rgb = rendered["rgb"]  # [B, num_target, 3, H, W]
        
        # Get ground truth target images (rendered from voxel or from dataset)
        target_rgb = all_images[:, target_view_indices]  # [B, num_target, 3, H, W]
        
        # Compute loss
        loss_dict = self.loss_fn(pred_rgb, target_rgb, mlp_params)
        
        return loss_dict, pred_rgb, target_rgb
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        self.mlp_renderer.train()
        
        epoch_losses = {}
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            self.optimizer.zero_grad()
            
            try:
                loss_dict, pred_rgb, target_rgb = self._train_step(batch)
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
                
                # Visualization
                if self.global_step % self.args.vis_freq == 0:
                    self._visualize(pred_rgb, target_rgb, self.global_step)
                
                self.global_step += 1
                pbar.set_postfix({"loss": total_loss.item()})
                
            except Exception as e:
                self.logger.warning(f"Error in batch {batch_idx}: {e}")
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
        self.mlp_renderer.eval()
        
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
    
    def _visualize(
        self, 
        pred_rgb: torch.Tensor, 
        target_rgb: torch.Tensor, 
        step: int
    ):
        """Log visualization to tensorboard."""
        # Take first batch element
        if pred_rgb.dim() == 5:  # [B, S, 3, H, W]
            pred = pred_rgb[0, 0]  # [3, H, W]
            target = target_rgb[0, 0]
        else:  # [B, 3, H, W]
            pred = pred_rgb[0]
            target = target_rgb[0]
        
        # Clamp to valid range
        pred = torch.clamp(pred, 0, 1)
        target = torch.clamp(target, 0, 1)
        
        # Concatenate for comparison
        comparison = torch.cat([pred, target], dim=-1)  # [3, H, 2*W]
        
        self.writer.add_image("train/pred_vs_target", comparison, step)
    
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
                    f"Epoch {epoch} - Val: {val_losses.get('total_loss', 0):.6f}"
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


def main():
    args = parse_args()
    trainer = NeRFMAETrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
