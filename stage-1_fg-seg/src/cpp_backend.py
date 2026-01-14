"""
C++ accelerated backend module for pixel2voxel.

This module provides drop-in replacements for PyTorch implementations
using optimized C++ code. Falls back gracefully if not compiled.
"""

try:
    from . import voxel_ops as _voxel_ops
    AVAILABLE = True
except ImportError:
    _voxel_ops = None
    AVAILABLE = False


def is_available() -> bool:
    """Check if C++ backend is available."""
    return AVAILABLE


def info() -> str:
    """Get information about C++ backend."""
    if not AVAILABLE:
        return "C++ backend not available. Build with: python setup.py build_ext --inplace"
    return f"C++ backend available (module: {_voxel_ops})"


# Re-export functions if available
if AVAILABLE:
    VoxelGrid = _voxel_ops.VoxelGrid
    Vec3 = _voxel_ops.Vec3
    create_voxel_grid = _voxel_ops.create_voxel_grid
    set_voxel_grid_data = _voxel_ops.set_voxel_grid_data
    cast_rays_batch = _voxel_ops.cast_rays_batch
    unproject_pixels = _voxel_ops.unproject_pixels
    count_voxel_visibility = _voxel_ops.count_voxel_visibility
    find_max_voxel = _voxel_ops.find_max_voxel
    voxel_indices_to_world = _voxel_ops.voxel_indices_to_world
    find_crossing_rays = _voxel_ops.find_crossing_rays
else:
    VoxelGrid = None
    Vec3 = None
    create_voxel_grid = None
    set_voxel_grid_data = None
    cast_rays_batch = None
    unproject_pixels = None
    count_voxel_visibility = None
    find_max_voxel = None
    voxel_indices_to_world = None
    find_crossing_rays = None


__all__ = [
    'AVAILABLE',
    'is_available',
    'info',
    'VoxelGrid',
    'Vec3',
    'create_voxel_grid',
    'set_voxel_grid_data',
    'cast_rays_batch',
    'unproject_pixels',
    'count_voxel_visibility',
    'find_max_voxel',
    'voxel_indices_to_world',
    'find_crossing_rays',
]
