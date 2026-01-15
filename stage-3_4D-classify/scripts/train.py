#!/usr/bin/env python3
"""
Training script for 4D Gaussian Classifier.

Usage:
    python scripts/train.py --config configs/train_full.yaml
    python scripts/train.py --config configs/train_lite.yaml --data-root /path/to/data
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
import random
import numpy as np

from utils.config import Config, load_config, merge_configs
from utils.logging import setup_logging, get_logger
from models.full_model import Gaussian4DClassifier
from data.gaussian_dataset import GaussianSequenceDataset, GaussianCollator, SyntheticGaussianDataset
from data.augmentations import get_train_augmentation
from training.trainer import Trainer

logger = get_logger(__name__)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description='Train 4D Gaussian Classifier')

    parser.add_argument(
        '--config', type=Path, required=True,
        help='Path to config file (YAML or JSON)'
    )
    parser.add_argument(
        '--data-root', type=Path, default=None,
        help='Override data root directory'
    )
    parser.add_argument(
        '--checkpoint-dir', type=Path, default=Path('checkpoints'),
        help='Directory to save checkpoints'
    )
    parser.add_argument(
        '--resume', type=Path, default=None,
        help='Resume training from checkpoint'
    )
    parser.add_argument(
        '--epochs', type=int, default=None,
        help='Override number of epochs'
    )
    parser.add_argument(
        '--batch-size', type=int, default=None,
        help='Override batch size'
    )
    parser.add_argument(
        '--lr', type=float, default=None,
        help='Override learning rate'
    )
    parser.add_argument(
        '--device', type=str, default=None,
        help='Device to train on (cuda/cpu)'
    )
    parser.add_argument(
        '--synthetic', action='store_true',
        help='Use synthetic data for testing'
    )
    parser.add_argument(
        '--debug', action='store_true',
        help='Enable debug mode (smaller data, more logging)'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    config = Config.from_file(args.config)
    config_dict = config.to_dict()

    # Override config with command line arguments
    if args.data_root:
        config_dict['data']['root'] = str(args.data_root)
    if args.epochs:
        config_dict['training']['epochs'] = args.epochs
    if args.batch_size:
        config_dict['data']['batch_size'] = args.batch_size
    if args.lr:
        config_dict['training']['learning_rate'] = args.lr
    if args.device:
        config_dict['device'] = args.device

    # Debug mode adjustments
    if args.debug:
        config_dict['data']['batch_size'] = 2
        config_dict['data']['sequence_length'] = 10
        config_dict['data']['max_gaussians'] = 1000
        config_dict['training']['epochs'] = 5
        config_dict['logging'] = config_dict.get('logging', {})
        config_dict['logging']['level'] = 'DEBUG'

    # Setup logging
    log_level = config_dict.get('logging', {}).get('level', 'INFO')
    log_file = args.checkpoint_dir / 'train.log'
    setup_logging(level=log_level, log_file=log_file)

    logger.info(f"Config: {config_dict}")

    # Set seed
    seed = config_dict.get('seed', 42)
    set_seed(seed)
    logger.info(f"Set random seed: {seed}")

    # Device
    device = config_dict.get('device', 'cuda')
    if device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = 'cpu'
    logger.info(f"Using device: {device}")

    # Create model
    model_config = config_dict.get('model', {})
    model = Gaussian4DClassifier(**model_config)

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model created with {num_params:,} parameters")

    # Create datasets
    data_config = config_dict.get('data', {})

    if args.synthetic:
        logger.info("Using synthetic data")
        train_dataset = SyntheticGaussianDataset(
            num_samples=100,
            sequence_length=data_config.get('sequence_length', 30),
            num_gaussians=data_config.get('max_gaussians', 1000),
        )
        val_dataset = SyntheticGaussianDataset(
            num_samples=20,
            sequence_length=data_config.get('sequence_length', 30),
            num_gaussians=data_config.get('max_gaussians', 1000),
        )
    else:
        data_root = Path(data_config.get('root', 'data'))
        train_root = data_root / 'train'
        val_root = data_root / 'val'

        # Augmentation for training
        aug_config = data_config.get('augmentation', {})
        transform = None
        if aug_config.get('enabled', True):
            transform = get_train_augmentation(
                temporal=True,
                spatial=True,
                gaussian=True,
            )

        train_dataset = GaussianSequenceDataset(
            data_root=train_root,
            sequence_length=data_config.get('sequence_length', 30),
            stride=data_config.get('stride', 1),
            max_gaussians=data_config.get('max_gaussians', 50000),
            transform=transform,
            cache_sequences=data_config.get('cache_sequences', False),
        )

        val_dataset = GaussianSequenceDataset(
            data_root=val_root,
            sequence_length=data_config.get('sequence_length', 30),
            stride=data_config.get('stride', 1),
            max_gaussians=data_config.get('max_gaussians', 50000),
            transform=None,  # No augmentation for validation
            cache_sequences=data_config.get('cache_sequences', False),
        )

    logger.info(f"Train dataset: {len(train_dataset)} samples")
    logger.info(f"Val dataset: {len(val_dataset)} samples")

    # Create data loaders
    collator = GaussianCollator()

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=data_config.get('batch_size', 8),
        shuffle=True,
        num_workers=data_config.get('num_workers', 4),
        collate_fn=collator,
        pin_memory=(device == 'cuda'),
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=data_config.get('batch_size', 8),
        shuffle=False,
        num_workers=data_config.get('num_workers', 4),
        collate_fn=collator,
        pin_memory=(device == 'cuda'),
    )

    # Create trainer
    training_config = config_dict.get('training', {})
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
    )

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume)
        logger.info(f"Resumed from epoch {start_epoch}")

    # Save config
    config_save_path = args.checkpoint_dir / 'config.yaml'
    from utils.config import save_config
    save_config(config_dict, config_save_path)
    logger.info(f"Saved config to {config_save_path}")

    # Train
    logger.info("Starting training...")
    history = trainer.train(epochs=training_config.get('epochs', 100))

    logger.info("Training complete!")
    logger.info(f"Best F1: {trainer.best_val_f1:.4f} at epoch {trainer.best_epoch}")


if __name__ == '__main__':
    main()
