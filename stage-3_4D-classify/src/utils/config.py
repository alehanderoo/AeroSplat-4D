"""
Configuration management utilities.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import json
from copy import deepcopy


def load_config(path: Path) -> Dict[str, Any]:
    """
    Load configuration from YAML or JSON file.

    Args:
        path: Path to config file

    Returns:
        Configuration dictionary
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        if path.suffix in ['.yaml', '.yml']:
            config = yaml.safe_load(f)
        elif path.suffix == '.json':
            config = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")

    return config or {}


def save_config(config: Dict[str, Any], path: Path):
    """
    Save configuration to file.

    Args:
        config: Configuration dictionary
        path: Output path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w') as f:
        if path.suffix in ['.yaml', '.yml']:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        elif path.suffix == '.json':
            json.dump(config, f, indent=2)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")


def merge_configs(
    base: Dict[str, Any],
    override: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Recursively merge override config into base config.

    Args:
        base: Base configuration
        override: Override configuration

    Returns:
        Merged configuration
    """
    result = deepcopy(base)

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = deepcopy(value)

    return result


def get_default_config() -> Dict[str, Any]:
    """Get default configuration."""
    return {
        # Model configuration
        'model': {
            # Stage 1: Spatial encoder
            'spatial_dim': 128,
            'spatial_depth': 4,
            'spatial_heads': 8,
            'spatial_dim_feat': 64,
            'spatial_output_dim': 256,

            # Stage 2: Temporal encoder
            'temporal_hidden_dim': 256,
            'temporal_num_layers': 4,
            'temporal_d_state': 16,
            'temporal_bidirectional': False,

            # Classification
            'num_classes': 1,
            'dropout': 0.1,

            # Features
            'use_eigenvectors': True,
        },

        # Data configuration
        'data': {
            'sequence_length': 30,
            'stride': 1,
            'max_gaussians': 50000,
            'subsample_strategy': 'random',
            'batch_size': 8,
            'num_workers': 4,
            'cache_sequences': False,
        },

        # Training configuration
        'training': {
            'epochs': 100,
            'learning_rate': 1e-4,
            'weight_decay': 1e-5,
            'grad_clip': 1.0,
            'loss_type': 'focal',
            'focal_alpha': 0.25,
            'focal_gamma': 2.0,
            'label_smoothing': 0.0,
            'scheduler': 'onecycle',
            'save_every': 10,
            'patience': 20,
        },

        # Hardware
        'device': 'cuda',
        'seed': 42,
    }


class Config:
    """Configuration class with attribute access."""

    def __init__(self, config_dict: Dict[str, Any] = None):
        """
        Args:
            config_dict: Configuration dictionary
        """
        self._config = config_dict or get_default_config()

    def __getattr__(self, name: str) -> Any:
        if name.startswith('_'):
            return super().__getattribute__(name)

        if name in self._config:
            value = self._config[name]
            if isinstance(value, dict):
                return Config(value)
            return value
        raise AttributeError(f"Config has no attribute '{name}'")

    def __getitem__(self, key: str) -> Any:
        return self._config[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self._config.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        return deepcopy(self._config)

    @classmethod
    def from_file(cls, path: Path) -> 'Config':
        """Load config from file."""
        config_dict = load_config(path)
        default = get_default_config()
        merged = merge_configs(default, config_dict)
        return cls(merged)

    def save(self, path: Path):
        """Save config to file."""
        save_config(self._config, path)
