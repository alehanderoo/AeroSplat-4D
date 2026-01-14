from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

from .geometry import quat_wxyz_to_rotation_matrix


@dataclass
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    matrix: np.ndarray


@dataclass
class CameraExtrinsics:
    world_to_camera: np.ndarray
    camera_to_world: np.ndarray
    position: np.ndarray
    orientation_wxyz: np.ndarray
    rotation_from_quat: np.ndarray

    @property
    def rotation_world_to_camera(self) -> np.ndarray:
        return self.world_to_camera[:3, :3]

    @property
    def rotation_camera_to_world(self) -> np.ndarray:
        return self.camera_to_world[:3, :3]


@dataclass
class CameraModel:
    name: str
    intrinsics: CameraIntrinsics
    extrinsics: CameraExtrinsics
    near: float
    far: float

    def project_world_points(self, points_world: np.ndarray) -> np.ndarray:
        """
        Project (N, 3) world-space points into image coordinates.
        Returns (N, 3) array with (u, v, z_cam).
        """
        if points_world.ndim != 2 or points_world.shape[1] != 3:
            raise ValueError("points_world must have shape (N, 3)")

        world_h = np.concatenate([points_world, np.ones((points_world.shape[0], 1), dtype=points_world.dtype)], axis=1)
        cam = world_h @ self.extrinsics.world_to_camera.T
        x = cam[:, 0]
        y = cam[:, 1]
        z = cam[:, 2]
        u = self.intrinsics.fx * (x / z) + self.intrinsics.cx
        v = self.intrinsics.fy * (y / z) + self.intrinsics.cy
        return np.stack([u, v, z], axis=1)


def _compute_near_far(depth_samples: Iterable[float]) -> Tuple[float, float]:
    depth_arr = np.array(list(depth_samples), dtype=np.float32)
    if depth_arr.size == 0 or not np.isfinite(depth_arr).any():
        return 0.05, 40.0
    min_depth = float(np.nanmin(depth_arr))
    max_depth = float(np.nanmax(depth_arr))
    near = max(0.05, 0.5 * min_depth)
    far = max(near + 0.5, 1.5 * max_depth)
    return near, far


def load_camera_models(metadata_path: Path, frame_stride: int = 1) -> Tuple[List[CameraModel], np.ndarray]:
    """
    Parse the drone_camera_observations.json and return camera models plus
    the tracked drone positions used to bound the voxel grid.
    """
    with metadata_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    camera_static = meta["cameras"]
    frames = meta.get("frames", [])

    # Index depth samples per camera for far/near heuristics.
    depth_samples: Dict[str, List[float]] = {}
    drone_positions: List[List[float]] = []
    for frame in frames[::frame_stride]:
        drone_positions.append(frame["drone_position_3d"])
        for cam in frame.get("cameras", []):
            if not cam.get("visible", True):
                continue
            depth_samples.setdefault(cam["name"], []).append(cam.get("depth", np.nan))

    models: List[CameraModel] = []
    for cam in camera_static:
        intr = cam["intrinsics"]
        extr = cam["extrinsics"]

        intrinsics = CameraIntrinsics(
            width=cam["resolution"]["width"],
            height=cam["resolution"]["height"],
            fx=intr["fx"],
            fy=intr["fy"],
            cx=intr["cx"],
            cy=intr["cy"],
            matrix=np.array(intr["matrix"], dtype=np.float32),
        )

        camera_to_world = np.array(extr["camera_to_world_matrix"], dtype=np.float32)
        world_to_camera = np.array(extr["world_to_camera_matrix"], dtype=np.float32)
        orientation = np.array(extr["orientation_quat_wxyz"], dtype=np.float32)
        position = np.array(extr["position_world"], dtype=np.float32)

        # If the metadata stored zero position, derive it from the transform.
        if np.linalg.norm(position) < 1e-6:
            position = camera_to_world[:3, 3]

        # FIX: The cameras in the metadata point outward from the center (for observing 
        # a drone flying in the middle). For voxel overlap detection, we need them
        # pointing inward. Flip the viewing direction by rotating 180° around Y-axis.
        flip_180_y = np.array([
            [-1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        camera_to_world = camera_to_world @ flip_180_y
        world_to_camera = np.linalg.inv(camera_to_world)

        rotation_from_quat = quat_wxyz_to_rotation_matrix(orientation)

        extrinsics = CameraExtrinsics(
            world_to_camera=world_to_camera,
            camera_to_world=camera_to_world,
            position=position,
            orientation_wxyz=orientation,
            rotation_from_quat=rotation_from_quat,
        )

        depth_list = depth_samples.get(cam["name"], [])
        near, far = _compute_near_far(depth_list)

        models.append(
            CameraModel(
                name=cam["name"],
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                near=near,
                far=far,
            )
        )

    drone_positions_np = np.array(drone_positions, dtype=np.float32) if drone_positions else np.zeros((0, 3), dtype=np.float32)
    return models, drone_positions_np
