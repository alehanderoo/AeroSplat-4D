from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch

from .cameras import CameraModel
from .config import VoxelizerConfig
from .dataset import MultiCameraDataset

try:
    from . import voxel_ops
    HAS_CPP_BACKEND = True
except ImportError:
    HAS_CPP_BACKEND = False


@dataclass
class VoxelizerResult:
    occupancy: torch.Tensor
    visibility_counts: torch.Tensor
    grid_shape: Tuple[int, int, int]
    voxel_size: float
    grid_min: np.ndarray
    axes: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    device: torch.device

    def occupied_points(self) -> torch.Tensor:
        mask = self.occupancy
        if mask.ndim != 3:
            raise RuntimeError("Occupancy tensor must be 3D")
        if mask.device != self.axes[0].device:
            mask = mask.to(self.axes[0].device)
        coords = torch.stack(torch.meshgrid(*self.axes, indexing="ij"), dim=-1)
        pts = coords[mask]
        return pts

    def occupied_points_cpu(self) -> np.ndarray:
        return self.occupied_points().detach().cpu().numpy()


class CudaVoxelizer:
    """
    Compute the overlapping viewing volume of multiple calibrated cameras using
    CUDA-accelerated ray inclusion tests.
    """

    def __init__(self, dataset: MultiCameraDataset, config: VoxelizerConfig) -> None:
        self.dataset = dataset
        self.config = config
        self.device = self._select_device(config.use_cuda)
        self.camera_tensors = self._prepare_camera_tensors(dataset.camera_models, self.device)
        self.grid_min, self.grid_shape, self.voxel_size, self.axes = self._prepare_grid(dataset, config)

    @staticmethod
    def _select_device(force_cuda: bool | None) -> torch.device:
        has_cuda = torch.cuda.is_available()
        if force_cuda is True and not has_cuda:
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
        if force_cuda is False:
            return torch.device("cpu")
        return torch.device("cuda" if has_cuda else "cpu")

    @staticmethod
    def _prepare_camera_tensors(cameras: Sequence[CameraModel], device: torch.device) -> List[dict]:
        tensors: List[dict] = []
        for cam in cameras:
            tensors.append(
                {
                    "name": cam.name,
                    "world_to_camera": torch.tensor(cam.extrinsics.world_to_camera, dtype=torch.float32, device=device),
                    "fx": torch.tensor(cam.intrinsics.fx, dtype=torch.float32, device=device),
                    "fy": torch.tensor(cam.intrinsics.fy, dtype=torch.float32, device=device),
                    "cx": torch.tensor(cam.intrinsics.cx, dtype=torch.float32, device=device),
                    "cy": torch.tensor(cam.intrinsics.cy, dtype=torch.float32, device=device),
                    "width": torch.tensor(cam.intrinsics.width, dtype=torch.float32, device=device),
                    "height": torch.tensor(cam.intrinsics.height, dtype=torch.float32, device=device),
                    "near": torch.tensor(cam.near, dtype=torch.float32, device=device),
                    "far": torch.tensor(cam.far, dtype=torch.float32, device=device),
                }
            )
        return tensors

    @staticmethod
    def _prepare_grid(dataset: MultiCameraDataset, config: VoxelizerConfig) -> Tuple[np.ndarray, Tuple[int, int, int], float, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        # Use frustum-based bounding box instead of drone positions
        bbox = dataset.frustum_bounding_box(config.margin_meters)
        min_corner = bbox[0]
        max_corner = bbox[1]
        extents = max_corner - min_corner
        max_extent = float(extents.max())
        if max_extent <= 0.0:
            raise RuntimeError("Invalid bounding box extents")

        resolution = int(config.resolution)
        if resolution < 4:
            raise ValueError("resolution must be >= 4")

        voxel_size = max_extent / float(resolution - 1)
        dims = []
        for extent in extents:
            steps = int(np.ceil(extent / voxel_size)) + 1
            dims.append(max(2, steps))
        grid_shape = tuple(dims)  # type: ignore

        axes = tuple(
            torch.linspace(min_corner[i], min_corner[i] + voxel_size * (grid_shape[i] - 1), grid_shape[i])
            for i in range(3)
        )
        return min_corner, grid_shape, voxel_size, axes  # type: ignore

    def build(self) -> VoxelizerResult:
        print(f"\n=== Voxelizer Build Info ===")
        print(f"Device: {self.device}")
        print(f"Grid shape: {self.grid_shape}")
        print(f"Voxel size: {self.voxel_size:.4f}m")
        print(f"Grid bounds: min={self.grid_min}, max={self.grid_min + self.voxel_size * np.array(self.grid_shape)}")
        print(f"Total voxels: {np.prod(self.grid_shape):,}")
        print(f"Cameras: {len(self.camera_tensors)}")
        for i, cam in enumerate(self.camera_tensors):
            print(f"  {i+1}. {cam['name']}: near={float(cam['near']):.2f}, far={float(cam['far']):.2f}")
        
        x_axis, y_axis, z_axis = (axis.to(self.device) for axis in self.axes)
        grid = torch.stack(torch.meshgrid(x_axis, y_axis, z_axis, indexing="ij"), dim=-1)
        flat = grid.reshape(-1, 3)
        visibility_counts = torch.zeros(flat.shape[0], dtype=torch.int16, device=self.device)

        # Use C++ backend for better performance
        if not HAS_CPP_BACKEND:
            raise RuntimeError("C++ backend required. Build with: python setup.py build_ext --inplace")
        
        print(f"\nUsing C++ backend for visibility computation")
        flat_np = flat.cpu().numpy().astype(np.float32)
        
        # Convert camera tensors to CPU numpy for C++ backend
        camera_tensors_cpu = []
        for cam in self.camera_tensors:
            cam_cpu = {k: v.cpu().numpy() if torch.is_tensor(v) else v for k, v in cam.items()}
            camera_tensors_cpu.append(cam_cpu)
        
        counts_np = voxel_ops.count_voxel_visibility(flat_np, camera_tensors_cpu)
        visibility_counts = torch.from_numpy(counts_np).to(torch.int16).to(self.device)

        visibility_grid = visibility_counts.reshape(self.grid_shape)
        occupancy = visibility_grid >= self.config.min_cameras
        
        num_occupied = occupancy.sum().item()
        print(f"\n=== Results ===")
        print(f"Occupied voxels (>= {self.config.min_cameras} cameras): {num_occupied:,}")
        print(f"Occupancy rate: {100.0 * num_occupied / flat.shape[0]:.2f}%")
        
        # Show visibility histogram
        vis_counts_cpu = visibility_counts.detach().cpu().numpy()
        for n_cams in range(len(self.camera_tensors) + 1):
            count = (vis_counts_cpu == n_cams).sum()
            if count > 0:
                print(f"  {n_cams} cameras: {count:,} voxels ({100.0 * count / flat.shape[0]:.2f}%)")

        return VoxelizerResult(
            occupancy=occupancy,
            visibility_counts=visibility_grid,
            grid_shape=self.grid_shape,
            voxel_size=self.voxel_size,
            grid_min=self.grid_min,
            axes=(x_axis, y_axis, z_axis),
            device=self.device,
        )
