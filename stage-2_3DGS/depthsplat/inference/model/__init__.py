"""
PyTorch Model Wrapper for DepthSplat.

This package provides the model wrapper for running DepthSplat inference
directly with PyTorch.
"""

from .depthsplat_wrapper import (
    DepthSplatWrapper,
    ModelConfig,
    load_model,
    analyze_model,
)

__all__ = [
    # Wrapper
    "DepthSplatWrapper",
    "ModelConfig",
    "load_model",
    "analyze_model",
]
