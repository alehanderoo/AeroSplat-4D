"""
pixel2voxel package

Reimagined voxel overlap builder leveraging CUDA-accelerated computation
to recover the shared viewing volume of a multi-camera rig.
"""

from .config import VoxelizerConfig
from .app import build_and_view

__all__ = ["VoxelizerConfig", "build_and_view"]
