"""
Configuration management for the DepthSplat inference backend.

This module provides configuration classes and utilities for
setting up the inference service.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple
import os


@dataclass
class ModelConfig:
    """
    Configuration for the DepthSplat model.

    Attributes:
        checkpoint_path: Path to the model checkpoint (.ckpt file)
        config_name: Name of the experiment config (without .yaml extension)
        device: Device to run inference on ('cuda' or 'cpu')
        image_shape: Expected input image size (H, W)
        near: Near plane distance for depth bounds
        far: Far plane distance for depth bounds
    """
    checkpoint_path: str
    config_name: str = "objaverse_white"
    device: str = "cuda"
    image_shape: Tuple[int, int] = (256, 256)
    near: float = 0.35
    far: float = 4.21

    def validate(self):
        """Validate configuration."""
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")


@dataclass
class InferenceConfig:
    """
    Configuration for inference behavior.

    Attributes:
        output_dir: Base directory for outputs (defaults to /tmp)
        enable_depth_analysis: Whether to run depth analysis
        enable_ply_export: Whether to export PLY files
        gaussian_trim_border: Pixels to trim from Gaussian borders (removes edge artifacts)
        target_radius: Normalized camera distance for pose normalization
        training_fx_norm: Training-matched focal length for wild frame processing
    """
    output_dir: str = "/tmp/depthsplat_output"
    enable_depth_analysis: bool = True
    enable_ply_export: bool = True
    gaussian_trim_border: int = 8
    target_radius: float = 2.0
    training_fx_norm: float = 1.0723  # 50 degree FOV


@dataclass
class ServiceConfig:
    """
    Complete configuration for the inference service.

    Combines model and inference configuration.
    """
    model: ModelConfig
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        config_name: str = "objaverse_white",
        device: str = "cuda",
        output_dir: str = "/tmp/depthsplat_output",
    ) -> "ServiceConfig":
        """
        Create configuration from checkpoint path.

        Args:
            checkpoint_path: Path to model checkpoint
            config_name: Experiment config name
            device: Compute device
            output_dir: Output directory

        Returns:
            ServiceConfig instance
        """
        return cls(
            model=ModelConfig(
                checkpoint_path=checkpoint_path,
                config_name=config_name,
                device=device,
            ),
            inference=InferenceConfig(output_dir=output_dir),
        )


# Default paths relative to the depthsplat directory
def get_default_checkpoint_path() -> str:
    """Get the default checkpoint path."""
    backend_dir = Path(__file__).parent
    depthsplat_root = backend_dir.parent
    return str(depthsplat_root / "outputs" / "objaverse_white" / "checkpoints" / "epoch_0-step_65000.ckpt")


def get_default_config() -> ServiceConfig:
    """Get default configuration."""
    return ServiceConfig.from_checkpoint(
        checkpoint_path=get_default_checkpoint_path(),
        config_name="objaverse_white",
    )
