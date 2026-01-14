from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

from .cameras import CameraModel, load_camera_models
from .config import VoxelizerConfig


@dataclass
class DatasetPaths:
    root: Path
    cameras: Dict[str, Path]


class MultiCameraDataset:
    """
    Thin convenience wrapper around the rendered cam_0X assets accompanied by
    drone_camera_observations.json.
    """

    def __init__(self, config: VoxelizerConfig) -> None:
        self.config = config
        self._camera_models: Optional[List[CameraModel]] = None
        self._drone_positions: Optional[np.ndarray] = None
        self.paths = self._discover_dataset_paths(config.dataset_root)

    @staticmethod
    def _discover_dataset_paths(root: Path) -> DatasetPaths:
        cams: Dict[str, Path] = {}
        for entry in root.iterdir():
            if not entry.is_dir():
                continue
            if entry.name.startswith("cam_"):
                cams[entry.name] = entry
        if not cams:
            raise FileNotFoundError(f"No cam_XX folders under {root}")
        return DatasetPaths(root=root, cameras=cams)

    @property
    def camera_models(self) -> List[CameraModel]:
        if self._camera_models is None:
            models, drone_positions = load_camera_models(self.config.metadata_json, self.config.frame_stride)
            # Filter to those with actual data on disk.
            models = [m for m in models if m.name in self.paths.cameras]
            if not models:
                raise RuntimeError("No camera models matched the dataset folders")
            self._camera_models = models
            self._drone_positions = drone_positions
        return self._camera_models

    @property
    def drone_positions(self) -> np.ndarray:
        if self._drone_positions is None:
            _ = self.camera_models
        assert self._drone_positions is not None
        return self._drone_positions

    def rgb_path(self, camera_name: str, frame_index: int) -> Path:
        cam_root = self.paths.cameras[camera_name]
        return cam_root / "rgb" / f"rgb_{frame_index:04d}.png"

    def depth_path(self, camera_name: str, frame_index: int) -> Path:
        cam_root = self.paths.cameras[camera_name]
        return cam_root / "depth" / f"distance_to_image_plane_{frame_index:04d}.npy"

    def mask_path(self, camera_name: str, frame_index: int) -> Path:
        cam_root = self.paths.cameras[camera_name]
        return cam_root / "mask" / f"mask_{frame_index:04d}.png"

    def load_depth(self, camera_name: str, frame_index: int) -> np.ndarray:
        path = self.depth_path(camera_name, frame_index)
        if not path.exists():
            raise FileNotFoundError(path)
        return np.load(path).astype(np.float32)

    def load_rgb(self, camera_name: str, frame_index: int) -> np.ndarray:
        path = self.rgb_path(camera_name, frame_index)
        if not path.exists():
            raise FileNotFoundError(path)
        try:
            import imageio.v3 as iio  # type: ignore
        except ModuleNotFoundError:  # pragma: no cover - optional dependency
            raise RuntimeError("imageio is required to load RGB frames. Install with `pip install imageio`.")
        return iio.imread(path)

    def load_mask(self, camera_name: str, frame_index: int) -> np.ndarray:
        path = self.mask_path(camera_name, frame_index)
        if not path.exists():
            raise FileNotFoundError(path)
        try:
            import imageio.v3 as iio  # type: ignore
        except ModuleNotFoundError:  # pragma: no cover - optional dependency
            raise RuntimeError("imageio is required to load masks. Install with `pip install imageio`.")
        return iio.imread(path)

    def bounding_box(self, margin: float) -> np.ndarray:
        """
        Compute axis-aligned bounding box covering all recorded drone positions
        with extra padding (margin). Returns np.array shape (2,3) [min,max].
        """
        positions = self.drone_positions
        if positions.size == 0:
            raise RuntimeError("No drone positions available in metadata")
        min_corner = positions.min(axis=0) - margin
        max_corner = positions.max(axis=0) + margin
        return np.stack([min_corner, max_corner], axis=0)
    
    def frustum_bounding_box(self, margin: float) -> np.ndarray:
        """
        Compute a bounding box based on camera frustum sampling.
        This creates a volume that covers the region where cameras are actually looking.
        Returns np.array shape (2,3) [min,max].
        """
        cameras = self.camera_models
        
        # Sample points from each camera frustum
        all_points = []
        
        for cam in cameras:
            # Sample the frustum at near, middle, and far planes
            depths = [cam.near, (cam.near + cam.far) * 0.5, cam.far]
            
            # Sample a grid of points across the image
            w = cam.intrinsics.width
            h = cam.intrinsics.height
            
            # Sample 5x5 grid across image (corners, edges, center)
            us = np.linspace(0, w, 5)
            vs = np.linspace(0, h, 5)
            
            for depth in depths:
                for u in us:
                    for v in vs:
                        # Unproject pixel to 3D in camera space
                        x_cam = (u - cam.intrinsics.cx) * depth / cam.intrinsics.fx
                        y_cam = (v - cam.intrinsics.cy) * depth / cam.intrinsics.fy
                        z_cam = depth
                        
                        # Transform to world space
                        pt_cam = np.array([x_cam, y_cam, z_cam, 1.0], dtype=np.float32)
                        pt_world = cam.extrinsics.camera_to_world @ pt_cam
                        all_points.append(pt_world[:3])
        
        all_points = np.array(all_points, dtype=np.float32)
        
        # Compute bounding box with margin
        min_corner = all_points.min(axis=0) - margin
        max_corner = all_points.max(axis=0) + margin
        
        return np.stack([min_corner, max_corner], axis=0)

