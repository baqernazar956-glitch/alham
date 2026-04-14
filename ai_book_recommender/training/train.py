# -*- coding: utf-8 -*-
"""
🎓 Model Trainer
==================

Training pipeline with experiment tracking and versioning.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration."""
    
    # Model
    model_name: str = "two_tower"
    model_version: str = "1.0.0"
    
    # Training
    epochs: int = 10
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    warmup_steps: int = 1000
    
    # Optimization
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    grad_clip: float = 1.0
    
    # Regularization
    dropout: float = 0.1
    label_smoothing: float = 0.0
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every: int = 1
    keep_best_k: int = 3
    
    # Logging
    log_every: int = 100
    eval_every: int = 500
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class TrainingState:
    """Current training state."""
    
    epoch: int = 0
    global_step: int = 0
    best_metric: float = 0.0
    best_epoch: int = 0
    
    # Metrics history
    train_losses: List[float] = field(default_factory=list)
    val_metrics: List[Dict[str, float]] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
        }


class Trainer:
    """
    🎓 Model Trainer
    
    Features:
    - Experiment tracking
    - Checkpointing with versioning
    - Learning rate scheduling
    - Early stopping
    - Metric logging
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        loss_fn: Optional[Callable] = None,
        metrics_fn: Optional[Callable] = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model to train
            config: Training configuration
            train_loader: Training data loader
            val_loader: Validation data loader
            loss_fn: Custom loss function
            metrics_fn: Function to compute metrics
        """
        self.model = model.to(config.device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn or nn.BCEWithLogitsLoss()
        self.metrics_fn = metrics_fn
        
        # Optimizer
        self.optimizer = self._create_optimizer()
        
        # Scheduler
        self.scheduler = self._create_scheduler()
        
        # State
        self.state = TrainingState()
        
        # Checkpointing
        self.checkpoint_dir = Path(config.checkpoint_dir) / config.model_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Trainer initialized: {config.model_name} on {config.device}")
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer."""
        params = self.model.parameters()
        
        if self.config.optimizer == "adam":
            return optim.Adam(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == "adamw":
            return optim.AdamW(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == "sgd":
            return optim.SGD(
                params,
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
    
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler."""
        total_steps = len(self.train_loader) * self.config.epochs
        
        if self.config.scheduler == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps,
                eta_min=self.config.learning_rate * 0.01
            )
        elif self.config.scheduler == "linear":
            return optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.01,
                total_iters=total_steps
            )
        elif self.config.scheduler == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=len(self.train_loader) * 2,
                gamma=0.5
            )
        else:
            return None
    
    def train(self) -> Dict[str, Any]:
        """
        Run full training loop.
        
        Returns:
            Training results dictionary
        """
        logger.info(f"Starting training for {self.config.epochs} epochs")
        
        for epoch in range(self.config.epochs):
            self.state.epoch = epoch
            
            # Train epoch
            train_loss = self._train_epoch()
            self.state.train_losses.append(train_loss)
            
            # Validate
            if self.val_loader:
                val_metrics = self._validate()
                self.state.val_metrics.append(val_metrics)
                
                # Check if best
                primary_metric = val_metrics.get("ndcg", val_metrics.get("accuracy", 0))
                if primary_metric > self.state.best_metric:
                    self.state.best_metric = primary_metric
                    self.state.best_epoch = epoch
                    self._save_checkpoint(is_best=True)
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_every == 0:
                self._save_checkpoint()
            
            logger.info(
                f"Epoch {epoch + 1}/{self.config.epochs} - "
                f"Loss: {train_loss:.4f}"
            )
        
        # Final save
        self._save_checkpoint(final=True)
        
        return {
            "best_metric": self.state.best_metric,
            "best_epoch": self.state.best_epoch,
            "final_loss": self.state.train_losses[-1],
            "history": {
                "train_losses": self.state.train_losses,
                "val_metrics": self.state.val_metrics,
            }
        }
    
    def _train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in self.train_loader:
            # Move to device
            batch = {
                k: v.to(self.config.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # Model expects different inputs based on architecture
            if hasattr(batch, "user_embedding") and hasattr(batch, "item_embedding"):
                outputs = self.model(
                    batch["user_embedding"],
                    batch["item_embedding"]
                )
            else:
                outputs = self.model(**batch)
            
            # Compute loss
            if isinstance(outputs, dict):
                loss = outputs.get("loss", self.loss_fn(outputs["logits"], batch["label"]))
            else:
                loss = self.loss_fn(outputs, batch["label"])
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.grad_clip
                )
            
            self.optimizer.step()
            
            if self.scheduler:
                self.scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
            self.state.global_step += 1
            
            # Logging
            if self.state.global_step % self.config.log_every == 0:
                logger.debug(f"Step {self.state.global_step}: loss={loss.item():.4f}")
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def _validate(self) -> Dict[str, float]:
        """Run validation."""
        self.model.eval()
        
        all_preds = []
        all_labels = []
        
        for batch in self.val_loader:
            batch = {
                k: v.to(self.config.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            
            if hasattr(batch, "user_embedding") and hasattr(batch, "item_embedding"):
                outputs = self.model(
                    batch["user_embedding"],
                    batch["item_embedding"]
                )
            else:
                outputs = self.model(**batch)
            
            if isinstance(outputs, dict):
                preds = outputs.get("logits", outputs.get("scores"))
            else:
                preds = outputs
            
            all_preds.append(preds.cpu())
            all_labels.append(batch["label"].cpu())
        
        preds = torch.cat(all_preds)
        labels = torch.cat(all_labels)
        
        # Compute metrics
        if self.metrics_fn:
            return self.metrics_fn(preds, labels)
        else:
            # Default metrics
            preds_binary = (torch.sigmoid(preds) > 0.5).float()
            accuracy = (preds_binary == labels).float().mean().item()
            return {"accuracy": accuracy}
    
    def _save_checkpoint(self, is_best: bool = False, final: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config.__dict__,
            "state": self.state.to_dict(),
            "timestamp": datetime.now().isoformat(),
        }
        
        if self.scheduler:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        
        # Save paths
        if final:
            path = self.checkpoint_dir / f"model_final.pt"
        elif is_best:
            path = self.checkpoint_dir / f"model_best.pt"
        else:
            path = self.checkpoint_dir / f"model_epoch_{self.state.epoch}.pt"
        
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint: {path}")
        
        # Save metadata
        metadata = {
            "model_name": self.config.model_name,
            "version": self.config.model_version,
            "best_metric": self.state.best_metric,
            "epochs_trained": self.state.epoch + 1,
            "timestamp": datetime.now().isoformat(),
        }
        
        with open(self.checkpoint_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
    
    def load_checkpoint(self, path: str):
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.config.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if self.scheduler and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        state_dict = checkpoint.get("state", {})
        self.state.epoch = state_dict.get("epoch", 0)
        self.state.global_step = state_dict.get("global_step", 0)
        self.state.best_metric = state_dict.get("best_metric", 0.0)
        
        logger.info(f"Loaded checkpoint: {path}")
