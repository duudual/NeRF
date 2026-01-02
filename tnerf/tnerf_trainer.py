"""
T-NeRF Trainer: Fine-tuning script for NLPHead only.

Usage:
    python tnerf_trainer.py --pretrained_path /path/to/model.pt --data_dir /path/to/data
"""

import os
import argparse
import logging
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Import model and heads
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vggt.models.vggt import VGGT
from tnerf_loss import TNeRFLoss
from tnerf_dataloader import create_dataloaders


def parse_args():
    parser = argparse.ArgumentParser(description="T-NeRF Trainer for NLPHead")
    
    # Model paths
    parser.add_argument("--pretrained_path", type=str, default=None,
                        help="Path to pretrained VGGT model weights")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from")
    
    # Data paths
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing training data")
    
    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--lr_nlp", type=float, default=None,
                        help="Learning rate for NLPHead (default: same as --lr)")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="Weight decay for optimizer")
    
    # Loss weight
    parser.add_argument("--nlp_weight", type=float, default=1.0,
                        help="Weight for NLP loss")
    
    # Logging
    parser.add_argument("--log_dir", type=str, default="./logs",
                        help="Directory for tensorboard logs")
    parser.add_argument("--log_freq", type=int, default=100,
                        help="Logging frequency (steps)")
    parser.add_argument("--save_freq", type=int, default=5,
                        help="Checkpoint save frequency (epochs)")
    
    # Hardware
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use for training")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers")
    
    # Freeze settings
    parser.add_argument("--freeze_backbone", action="store_true", default=True,
                        help="Freeze the backbone (aggregator) during training")
    parser.add_argument("--freeze_other_heads", action="store_true", default=True,
                        help="Freeze other heads (camera, point, depth, track) during training")
    
    return parser.parse_args()


class TNeRFTrainer:
    """
    Trainer class for fine-tuning NLPHead.
    """
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        self.global_step = 0
        self.current_epoch = 0
        
        # Setup logging
        self._setup_logging()
        
        # Build model
        self.model = self._build_model()
        
        # Freeze layers
        self._freeze_layers()
        
        # Build optimizer
        self.optimizer = self._build_optimizer()
        
        # Build scheduler
        self.scheduler = self._build_scheduler()
        
        # Build loss
        self.loss_fn = TNeRFLoss(
            nlp_weight=args.nlp_weight,
        )
        
        # Build dataloaders
        self.train_loader, self.val_loader = self._build_dataloaders()
        
        # Resume if specified
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
        
        self.logger.info("Training mode: nlp")
        self.logger.info(f"Device: {self.device}")
    
    def _build_model(self) -> VGGT:
        """Build and initialize the VGGT model."""
        self.logger.info("Building VGGT model...")
        
        model = VGGT(
            enable_camera=False,
            enable_point=True,  # Needed for 3D point predictions
            enable_depth=False,
            enable_track=False,
            enable_nlp=True,
        )
        
        # Load pretrained weights if provided
        if self.args.pretrained_path:
            self.logger.info(f"Loading pretrained weights from {self.args.pretrained_path}")
            state_dict = torch.load(self.args.pretrained_path, map_location="cpu")
            
            # Handle different checkpoint formats
            if "model" in state_dict:
                state_dict = state_dict["model"]
            elif "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            
            # Load with strict=False to allow missing keys for new heads
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            self.logger.info(f"Loaded pretrained weights. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
        
        model = model.to(self.device)
        return model
    
    def _freeze_layers(self):
        """Freeze layers based on training mode."""
        # Freeze backbone if specified
        if self.args.freeze_backbone:
            self.logger.info("Freezing backbone (aggregator)...")
            for param in self.model.aggregator.parameters():
                param.requires_grad = False
        
        # Freeze other heads if specified
        if self.args.freeze_other_heads:
            self.logger.info("Freezing other heads...")
            heads_to_freeze = []
            
            if self.model.camera_head is not None:
                heads_to_freeze.append(("camera_head", self.model.camera_head))
            if self.model.point_head is not None:
                heads_to_freeze.append(("point_head", self.model.point_head))
            if self.model.depth_head is not None:
                heads_to_freeze.append(("depth_head", self.model.depth_head))
            if self.model.track_head is not None:
                heads_to_freeze.append(("track_head", self.model.track_head))
            
            for name, head in heads_to_freeze:
                for param in head.parameters():
                    param.requires_grad = False
                self.logger.info(f"  Frozen: {name}")
        
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,}")
    
    def _build_optimizer(self) -> optim.Optimizer:
        """Build optimizer with potentially different learning rates for different heads."""
        param_groups = []
        
        # NLPHead parameters
        if self.model.nmlp_head is not None:
            lr_nlp = self.args.lr_nlp if self.args.lr_nlp else self.args.lr
            param_groups.append({
                "params": self.model.nmlp_head.parameters(),
                "lr": lr_nlp,
                "name": "nlp_head"
            })
        
        # Backbone parameters (if not frozen)
        if not self.args.freeze_backbone:
            param_groups.append({
                "params": self.model.aggregator.parameters(),
                "lr": self.args.lr * 0.005,  # Lower LR for backbone
                "name": "aggregator"
            })
        
        optimizer = optim.AdamW(
            param_groups,
            weight_decay=self.args.weight_decay,
        )
        
        return optimizer
    
    def _build_scheduler(self):
        """Build learning rate scheduler."""
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.args.num_epochs,
            eta_min=self.args.lr * 0.01,
        )
        return scheduler
    
    def _build_dataloaders(self):
        """Build training and validation dataloaders."""
        return create_dataloaders(
            data_dir=self.args.data_dir,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
        )
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save training checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "args": vars(self.args),
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(
            self.args.checkpoint_dir, 
            f"checkpoint_epoch_{epoch:04d}.pt"
        )
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        # Save latest checkpoint
        latest_path = os.path.join(self.args.checkpoint_dir, "checkpoint_latest.pt")
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint if applicable
        if is_best:
            best_path = os.path.join(self.args.checkpoint_dir, "checkpoint_best.pt")
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best checkpoint: {best_path}")
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        self.logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        
        self.logger.info(f"Resumed from epoch {self.current_epoch}, step {self.global_step}")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = {}
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = self._move_to_device(batch)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(batch["images"])
            
            # Compute loss
            loss_dict = self.loss_fn(predictions, batch)
            total_loss = loss_dict["total_loss"]
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update epoch losses
            for key, value in loss_dict.items():
                if key not in epoch_losses:
                    epoch_losses[key] = 0.0
                epoch_losses[key] += value.item() if torch.is_tensor(value) else value
            
            # Logging
            if self.global_step % self.args.log_freq == 0:
                self._log_step(loss_dict, epoch)
            
            self.global_step += 1
            pbar.set_postfix({"loss": total_loss.item()})
        
        # Average epoch losses
        num_batches = len(self.train_loader)
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """Run validation."""
        self.model.eval()
        val_losses = {}
        
        for batch in tqdm(self.val_loader, desc="Validation"):
            batch = self._move_to_device(batch)
            predictions = self.model(batch["images"])
            loss_dict = self.loss_fn(predictions, batch)
            
            for key, value in loss_dict.items():
                if key not in val_losses:
                    val_losses[key] = 0.0
                val_losses[key] += value.item() if torch.is_tensor(value) else value
        
        # Average validation losses
        num_batches = len(self.val_loader)
        for key in val_losses:
            val_losses[key] /= num_batches
        
        return val_losses
    
    def _move_to_device(self, batch: Dict) -> Dict:
        """Move batch tensors to device."""
        for key, value in batch.items():
            if torch.is_tensor(value):
                batch[key] = value.to(self.device)
        return batch
    
    def _log_step(self, loss_dict: Dict, epoch: int):
        """Log training step to tensorboard."""
        for key, value in loss_dict.items():
            scalar_value = value.item() if torch.is_tensor(value) else value
            self.writer.add_scalar(f"train/{key}", scalar_value, self.global_step)
        
        # Log learning rates
        for i, param_group in enumerate(self.optimizer.param_groups):
            name = param_group.get("name", f"group_{i}")
            self.writer.add_scalar(f"lr/{name}", param_group["lr"], self.global_step)
    
    def train(self):
        """Main training loop."""
        self.logger.info("Starting training...")
        best_val_loss = float("inf")
        
        for epoch in range(self.current_epoch, self.args.num_epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_losses = self.train_epoch(epoch)
            self.logger.info(f"Epoch {epoch} - Train Loss: {train_losses['total_loss']:.6f}")
            
            # Log epoch losses
            for key, value in train_losses.items():
                self.writer.add_scalar(f"epoch_train/{key}", value, epoch)
            
            # Validation
            val_losses = self.validate(epoch)
            self.logger.info(f"Epoch {epoch} - Val Loss: {val_losses['total_loss']:.6f}")
            
            for key, value in val_losses.items():
                self.writer.add_scalar(f"epoch_val/{key}", value, epoch)
            
            # Update scheduler
            self.scheduler.step()
            
            # Save checkpoint
            is_best = val_losses["total_loss"] < best_val_loss
            if is_best:
                best_val_loss = val_losses["total_loss"]
            
            if (epoch + 1) % self.args.save_freq == 0 or is_best:
                self._save_checkpoint(epoch, is_best)
        
        self.logger.info("Training completed!")
        self.writer.close()


def main():
    args = parse_args()
    trainer = TNeRFTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
