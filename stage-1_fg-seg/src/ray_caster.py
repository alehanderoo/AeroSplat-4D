"""Ray casting utilities using C++ backend."""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple
import numpy as np

try:
    from . import voxel_ops
except ImportError:
    raise ImportError("C++ backend required. Build with: python setup.py build_ext --inplace")


class RayCaster:
    """Cast rays from camera pixels through voxel grid using C++."""

    def __init__(self, grid_shape: Tuple[int, int, int], voxel_size: float, grid_min: np.ndarray, device=None) -> None:
        self.grid_shape = grid_shape
        self.voxel_size = voxel_size
        self.grid_min = grid_min if isinstance(grid_min, np.ndarray) else np.array(grid_min)
        self.cpp_grid = voxel_ops.create_voxel_grid(
            np.array(grid_shape, dtype=np.int32), voxel_size, self.grid_min.astype(np.float32)
        )

    def unproject_pixels_to_rays(self, pixel_coords: np.ndarray, camera_model) -> Tuple[np.ndarray, np.ndarray]:
        if pixel_coords.shape[0] == 0:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
        return voxel_ops.unproject_pixels(
            pixel_coords.astype(np.int32),
            camera_model.extrinsics.rotation_camera_to_world.astype(np.float32),
            camera_model.extrinsics.position.astype(np.float32),
            float(camera_model.intrinsics.fx), float(camera_model.intrinsics.fy),
            float(camera_model.intrinsics.cx), float(camera_model.intrinsics.cy),
        )

    def cast_rays_and_accumulate(self, ray_origins: np.ndarray, ray_dirs: np.ndarray, pixel_weights: np.ndarray, accumulation_grid=None, max_distance: float = 100.0) -> np.ndarray:
        if ray_origins.shape[0] == 0:
            return self.cpp_grid.data().reshape(self.grid_shape)
        if pixel_weights.shape[0] != ray_origins.shape[0]:
            raise ValueError(f"pixel_weights shape {pixel_weights.shape} doesn't match ray count {ray_origins.shape[0]}")
        voxel_ops.cast_rays_batch(
            self.cpp_grid,
            ray_origins.astype(np.float32) if ray_origins.dtype != np.float32 else ray_origins,
            ray_dirs.astype(np.float32) if ray_dirs.dtype != np.float32 else ray_dirs,
            pixel_weights.astype(np.float32) if pixel_weights.dtype != np.float32 else pixel_weights,
            max_distance,
        )
        return self.cpp_grid.data().reshape(self.grid_shape)


def localize_from_accumulation(accumulation_grid: np.ndarray, grid_min: np.ndarray, voxel_size: float, method: str = "max") -> np.ndarray:
    if method == "max":
        cpp_grid = voxel_ops.create_voxel_grid(
            np.array(accumulation_grid.shape, dtype=np.int32), voxel_size, grid_min.astype(np.float32)
        )
        voxel_ops.set_voxel_grid_data(cpp_grid, accumulation_grid.astype(np.float32) if accumulation_grid.dtype != np.float32 else accumulation_grid)
        ix, iy, iz, _ = voxel_ops.find_max_voxel(cpp_grid)
        return grid_min + np.array([ix + 0.5, iy + 0.5, iz + 0.5]) * voxel_size
    elif method == "com":
        nx, ny, nz = accumulation_grid.shape
        ix_grid, iy_grid, iz_grid = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
        total_weight = accumulation_grid.sum()
        if total_weight > 0:
            cx = (accumulation_grid * ix_grid).sum() / total_weight
            cy = (accumulation_grid * iy_grid).sum() / total_weight
            cz = (accumulation_grid * iz_grid).sum() / total_weight
            return grid_min + np.array([cx + 0.5, cy + 0.5, cz + 0.5]) * voxel_size
        return grid_min + np.array([nx / 2, ny / 2, nz / 2]) * voxel_size
    raise ValueError(f"Unknown localization method: {method}")
