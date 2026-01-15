"""
Main training loop.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from typing import Dict, Optional
from pathlib import Path
import logging
from tqdm import tqdm
import json

from .losses import ClassificationLoss
from .metrics import ClassificationMetrics, MetricTracker

logger = logging.getLogger(__name__)


class Trainer:
    """
    Training loop for 4D Gaussian classifier.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        device: str = 'cuda',
        checkpoint_dir: Optional[Path] = None,
    ):
        """
        Args:
            model: The classifier model
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration dictionary
            device: Device to train on
            checkpoint_dir: Directory to save checkpoints
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path('checkpoints')
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Loss
        self.criterion = ClassificationLoss(
            loss_type=config.get('loss_type', 'focal'),
            focal_alpha=config.get('focal_alpha', 0.25),
            focal_gamma=config.get('focal_gamma', 2.0),
            label_smoothing=config.get('label_smoothing', 0.0),
        )

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 1e-5),
            betas=config.get('betas', (0.9, 0.999)),
        )

        # Scheduler
        total_steps = len(train_loader) * config.get('epochs', 100)
        scheduler_type = config.get('scheduler', 'onecycle')

        if scheduler_type == 'onecycle':
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=config.get('learning_rate', 1e-4),
                total_steps=total_steps,
                pct_start=0.1,
                anneal_strategy='cos',
            )
        elif scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps,
                eta_min=config.get('min_lr', 1e-6),
            )
        else:
            self.scheduler = None

        # Metrics
        self.metrics = ClassificationMetrics()

        # Best model tracking
        self.best_val_acc = 0.0
        self.best_val_f1 = 0.0
        self.best_epoch = 0

        # History
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': [],
            'lr': [],
        }

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        tracker = MetricTracker()

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            # Move to device
            positions = batch['positions'].to(self.device)
            eigenvectors = batch['eigenvectors'].to(self.device)
            scalars = batch['scalars'].to(self.device)
            mask = batch['mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            # Forward
            self.optimizer.zero_grad()
            outputs = self.model(positions, eigenvectors, scalars, mask)

            # Loss
            loss = self.criterion(outputs['logits'], labels)

            # Backward
            loss.backward()

            # Gradient clipping
            if self.config.get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['grad_clip']
                )

            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            # Track metrics
            preds = (outputs['probabilities'] > 0.5).long().cpu().flatten().tolist()
            probs = outputs['probabilities'].cpu().flatten().tolist()
            tracker.update(
                preds,
                labels.cpu().tolist(),
                probs,
                loss.item()
            )

            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        return tracker.compute()

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validation loop."""
        self.model.eval()
        tracker = MetricTracker()

        for batch in tqdm(self.val_loader, desc="Validating"):
            positions = batch['positions'].to(self.device)
            eigenvectors = batch['eigenvectors'].to(self.device)
            scalars = batch['scalars'].to(self.device)
            mask = batch['mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            outputs = self.model(positions, eigenvectors, scalars, mask)

            loss = self.criterion(outputs['logits'], labels)

            preds = (outputs['probabilities'] > 0.5).long().cpu().flatten().tolist()
            probs = outputs['probabilities'].cpu().flatten().tolist()
            tracker.update(
                preds,
                labels.cpu().tolist(),
                probs,
                loss.item()
            )

        return tracker.compute()

    def train(self, epochs: int = None):
        """Full training loop."""
        epochs = epochs or self.config.get('epochs', 100)

        for epoch in range(1, epochs + 1):
            # Train
            train_metrics = self.train_epoch(epoch)
            logger.info(f"Epoch {epoch} Train: {train_metrics}")

            # Validate
            val_metrics = self.validate()
            logger.info(f"Epoch {epoch} Val: {val_metrics}")

            # Update history
            self.history['train_loss'].append(train_metrics.get('loss', 0))
            self.history['train_acc'].append(train_metrics.get('accuracy', 0))
            self.history['val_loss'].append(val_metrics.get('loss', 0))
            self.history['val_acc'].append(val_metrics.get('accuracy', 0))
            self.history['val_f1'].append(val_metrics.get('f1', 0))
            self.history['lr'].append(
                self.optimizer.param_groups[0]['lr']
            )

            # Save best model (by F1)
            if val_metrics.get('f1', 0) > self.best_val_f1:
                self.best_val_f1 = val_metrics['f1']
                self.best_val_acc = val_metrics.get('accuracy', 0)
                self.best_epoch = epoch
                self.save_checkpoint(epoch, is_best=True)
                logger.info(f"New best model! F1: {self.best_val_f1:.4f}")

            # Regular checkpoint
            if epoch % self.config.get('save_every', 10) == 0:
                self.save_checkpoint(epoch)

            # Early stopping
            patience = self.config.get('patience', 0)
            if patience > 0 and epoch - self.best_epoch >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        logger.info(
            f"Training complete. Best F1: {self.best_val_f1:.4f} "
            f"(accuracy: {self.best_val_acc:.4f}) at epoch {self.best_epoch}"
        )

        # Save training history
        self.save_history()

        return self.history

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_f1': self.best_val_f1,
            'best_val_acc': self.best_val_acc,
            'config': self.config,
        }

        if self.scheduler is not None:
            state['scheduler_state_dict'] = self.scheduler.state_dict()

        path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(state, path)

        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(state, best_path)
            logger.info(f"Saved best model to {best_path}")

    def load_checkpoint(self, path: Path):
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.best_val_f1 = checkpoint.get('best_val_f1', 0)
        self.best_val_acc = checkpoint.get('best_val_acc', 0)

        logger.info(f"Loaded checkpoint from {path}")
        return checkpoint.get('epoch', 0)

    def save_history(self):
        """Save training history to JSON."""
        history_path = self.checkpoint_dir / "history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)


def create_trainer(
    model: nn.Module,
    train_dataset,
    val_dataset,
    config: Dict,
    **kwargs
) -> Trainer:
    """
    Create trainer with data loaders.

    Args:
        model: Model to train
        train_dataset: Training dataset
        val_dataset: Validation dataset
        config: Configuration dictionary
        **kwargs: Additional arguments for Trainer

    Returns:
        Configured Trainer instance
    """
    from ..data.gaussian_dataset import GaussianCollator

    collator = GaussianCollator()

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 8),
        shuffle=True,
        num_workers=config.get('num_workers', 4),
        collate_fn=collator,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('batch_size', 8),
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        collate_fn=collator,
        pin_memory=True,
    )

    return Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        **kwargs
    )
