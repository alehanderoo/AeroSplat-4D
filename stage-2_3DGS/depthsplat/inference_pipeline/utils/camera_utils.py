"""
Camera utilities for DepthSplat Inference Pipeline.

This module provides camera intrinsics/extrinsics handling and
projection utilities.

Usage:
    from utils.camera_utils import CameraIntrinsics, load_camera_metadata
    
    metadata = load_camera_metadata("/path/to/renders/cam_01/metadata.txt")
    intrinsics = metadata.intrinsics
    
    # Project a 3D point to image coordinates
    pixel = intrinsics.project(point_3d)
"""

import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Union, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class CameraIntrinsics:
    """
    Camera intrinsic parameters.
    
    Represents the 3x3 camera matrix K:
        [fx  0  cx]
        [0  fy  cy]
        [0   0   1]
    """
    fx: float  # Focal length x
    fy: float  # Focal length y
    cx: float  # Principal point x
    cy: float  # Principal point y
    width: int = 0
    height: int = 0
    
    # Distortion coefficients (optional)
    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    k3: float = 0.0
    
    @property
    def matrix(self) -> np.ndarray:
        """Get the 3x3 intrinsic matrix."""
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float32)
    
    @property
    def distortion_coeffs(self) -> np.ndarray:
        """Get distortion coefficients [k1, k2, p1, p2, k3]."""
        return np.array([self.k1, self.k2, self.p1, self.p2, self.k3], dtype=np.float32)
    
    def project(self, point_3d: np.ndarray) -> np.ndarray:
        """
        Project a 3D point to image coordinates.
        
        Args:
            point_3d: 3D point(s) [3] or [N, 3]
            
        Returns:
            2D pixel coordinates [2] or [N, 2]
        """
        point_3d = np.asarray(point_3d)
        single = point_3d.ndim == 1
        if single:
            point_3d = point_3d.reshape(1, 3)
        
        # Normalize by z
        z = point_3d[:, 2:3]
        z = np.where(z == 0, 1e-10, z)  # Avoid division by zero
        normalized = point_3d[:, :2] / z
        
        # Apply intrinsics
        pixels = np.stack([
            normalized[:, 0] * self.fx + self.cx,
            normalized[:, 1] * self.fy + self.cy
        ], axis=1)
        
        return pixels[0] if single else pixels
    
    def unproject(self, pixel: np.ndarray, depth: float = 1.0) -> np.ndarray:
        """
        Unproject a pixel to 3D ray direction.
        
        Args:
            pixel: 2D pixel coordinates [2] or [N, 2]
            depth: Depth value (default 1.0 for unit ray)
            
        Returns:
            3D point(s) [3] or [N, 3]
        """
        pixel = np.asarray(pixel)
        single = pixel.ndim == 1
        if single:
            pixel = pixel.reshape(1, 2)
        
        # Normalize coordinates
        x = (pixel[:, 0] - self.cx) / self.fx
        y = (pixel[:, 1] - self.cy) / self.fy
        
        # Create 3D points
        points = np.stack([x * depth, y * depth, np.full_like(x, depth)], axis=1)
        
        return points[0] if single else points
    
    @classmethod
    def from_matrix(cls, K: np.ndarray, width: int = 0, height: int = 0) -> "CameraIntrinsics":
        """Create from 3x3 matrix."""
        return cls(
            fx=K[0, 0],
            fy=K[1, 1],
            cx=K[0, 2],
            cy=K[1, 2],
            width=width,
            height=height,
        )


@dataclass
class CameraExtrinsics:
    """
    Camera extrinsic parameters (pose).
    
    Represents the 4x4 transformation matrix from world to camera:
        [R | t]
        [0 | 1]
    """
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))
    translation: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    def __post_init__(self):
        self.rotation = np.asarray(self.rotation, dtype=np.float32)
        self.translation = np.asarray(self.translation, dtype=np.float32)
    
    @property
    def matrix(self) -> np.ndarray:
        """Get the 4x4 extrinsic matrix (world to camera)."""
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = self.rotation
        T[:3, 3] = self.translation
        return T
    
    @property
    def matrix_inv(self) -> np.ndarray:
        """Get the inverse 4x4 matrix (camera to world)."""
        T_inv = np.eye(4, dtype=np.float32)
        R_inv = self.rotation.T
        T_inv[:3, :3] = R_inv
        T_inv[:3, 3] = -R_inv @ self.translation
        return T_inv
    
    @property
    def camera_position(self) -> np.ndarray:
        """Get the camera position in world coordinates."""
        return -self.rotation.T @ self.translation
    
    @property
    def camera_direction(self) -> np.ndarray:
        """Get the camera forward direction (negative Z in camera space)."""
        return self.rotation.T @ np.array([0, 0, 1], dtype=np.float32)
    
    def transform_point(self, point_world: np.ndarray) -> np.ndarray:
        """
        Transform a point from world to camera coordinates.
        
        Args:
            point_world: 3D point(s) in world coords [3] or [N, 3]
            
        Returns:
            3D point(s) in camera coords [3] or [N, 3]
        """
        point_world = np.asarray(point_world)
        single = point_world.ndim == 1
        if single:
            point_world = point_world.reshape(1, 3)
        
        point_cam = (self.rotation @ point_world.T).T + self.translation
        
        return point_cam[0] if single else point_cam
    
    @classmethod
    def from_matrix(cls, T: np.ndarray) -> "CameraExtrinsics":
        """Create from 4x4 matrix."""
        return cls(
            rotation=T[:3, :3],
            translation=T[:3, 3],
        )
    
    @classmethod
    def look_at(
        cls,
        camera_pos: np.ndarray,
        target: np.ndarray,
        up: np.ndarray = None,
    ) -> "CameraExtrinsics":
        """
        Create extrinsics from look-at parameters.
        
        Args:
            camera_pos: Camera position in world coordinates
            target: Point to look at
            up: Up direction (default [0, 1, 0])
        """
        if up is None:
            up = np.array([0, 1, 0], dtype=np.float32)
        
        camera_pos = np.asarray(camera_pos, dtype=np.float32)
        target = np.asarray(target, dtype=np.float32)
        up = np.asarray(up, dtype=np.float32)
        
        # Camera coordinate axes
        z = target - camera_pos
        z = z / np.linalg.norm(z)
        
        x = np.cross(up, z)
        x = x / np.linalg.norm(x)
        
        y = np.cross(z, x)
        
        # Rotation matrix (camera to world)
        R_cw = np.stack([x, y, z], axis=1)
        
        # We need world to camera
        R = R_cw.T
        t = -R @ camera_pos
        
        return cls(rotation=R, translation=t)


@dataclass
class CameraMetadata:
    """Complete camera metadata."""
    camera_id: str
    intrinsics: CameraIntrinsics
    extrinsics: CameraExtrinsics
    near: float = 0.1
    far: float = 100.0
    
    @property
    def projection_matrix(self) -> np.ndarray:
        """Get combined 3x4 projection matrix P = K @ [R|t]."""
        K = self.intrinsics.matrix
        T = self.extrinsics.matrix[:3, :]
        return K @ T


def load_camera_metadata(
    path: Union[str, Path],
    camera_id: Optional[str] = None,
) -> CameraMetadata:
    """
    Load camera metadata from file.
    
    Supports:
        - JSON format (from IsaacSim/Blender export)
        - Text format (simple key-value)
        
    Args:
        path: Path to metadata file
        camera_id: Camera ID (inferred from path if not provided)
        
    Returns:
        CameraMetadata instance
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")
    
    # Infer camera ID from path
    if camera_id is None:
        camera_id = path.parent.parent.name  # e.g., cam_01
    
    # Try JSON format first
    if path.suffix == ".json":
        return _load_json_metadata(path, camera_id)
    else:
        return _load_text_metadata(path, camera_id)


def _load_json_metadata(path: Path, camera_id: str) -> CameraMetadata:
    """Load metadata from JSON file."""
    with open(path, 'r') as f:
        data = json.load(f)
    
    # Handle different JSON formats
    if "intrinsics" in data:
        # Nested format
        K = np.array(data["intrinsics"], dtype=np.float32)
        intrinsics = CameraIntrinsics.from_matrix(K)
        
        if "image_size" in data:
            intrinsics.width = data["image_size"][0]
            intrinsics.height = data["image_size"][1]
        
        if "extrinsics" in data:
            T = np.array(data["extrinsics"], dtype=np.float32)
            extrinsics = CameraExtrinsics.from_matrix(T)
        else:
            extrinsics = CameraExtrinsics()
        
        near = data.get("near", 0.1)
        far = data.get("far", 100.0)
        
    elif "fx" in data:
        # Flat format
        intrinsics = CameraIntrinsics(
            fx=data["fx"],
            fy=data["fy"],
            cx=data["cx"],
            cy=data["cy"],
            width=data.get("width", 0),
            height=data.get("height", 0),
        )
        
        if "rotation" in data:
            extrinsics = CameraExtrinsics(
                rotation=np.array(data["rotation"], dtype=np.float32),
                translation=np.array(data.get("translation", [0, 0, 0]), dtype=np.float32),
            )
        else:
            extrinsics = CameraExtrinsics()
        
        near = data.get("near", 0.1)
        far = data.get("far", 100.0)
    else:
        raise ValueError(f"Unknown JSON format in {path}")
    
    return CameraMetadata(
        camera_id=camera_id,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        near=near,
        far=far,
    )


def _load_text_metadata(path: Path, camera_id: str) -> CameraMetadata:
    """Load metadata from text file."""
    data = {}
    
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            
            if ":" in line:
                key, value = line.split(":", 1)
                data[key.strip().lower()] = value.strip()
            elif "=" in line:
                key, value = line.split("=", 1)
                data[key.strip().lower()] = value.strip()
    
    # Parse values
    def parse_float(key: str, default: float = 0.0) -> float:
        return float(data.get(key, default))
    
    def parse_int(key: str, default: int = 0) -> int:
        return int(float(data.get(key, default)))
    
    intrinsics = CameraIntrinsics(
        fx=parse_float("fx", 1000),
        fy=parse_float("fy", 1000),
        cx=parse_float("cx", 320),
        cy=parse_float("cy", 240),
        width=parse_int("width", 640),
        height=parse_int("height", 480),
    )
    
    # Default extrinsics (identity)
    extrinsics = CameraExtrinsics()
    
    return CameraMetadata(
        camera_id=camera_id,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        near=parse_float("near", 0.1),
        far=parse_float("far", 100.0),
    )


def load_all_cameras(
    render_dir: Union[str, Path],
    camera_ids: Optional[List[str]] = None,
) -> Dict[str, CameraMetadata]:
    """
    Load metadata for all cameras in a render directory.
    
    Args:
        render_dir: Path to render directory
        camera_ids: List of camera IDs (auto-detected if not provided)
        
    Returns:
        Dictionary mapping camera ID to CameraMetadata
    """
    render_dir = Path(render_dir)
    
    if camera_ids is None:
        camera_ids = [d.name for d in render_dir.iterdir() if d.is_dir() and d.name.startswith("cam_")]
        camera_ids.sort()
    
    cameras = {}
    for cam_id in camera_ids:
        cam_dir = render_dir / cam_id
        
        # Try different metadata file names
        for name in ["metadata.json", "metadata.txt", "camera.json"]:
            meta_path = cam_dir / name
            if meta_path.exists():
                try:
                    cameras[cam_id] = load_camera_metadata(meta_path, cam_id)
                    break
                except Exception as e:
                    logger.warning(f"Failed to load metadata for {cam_id}: {e}")
        
        if cam_id not in cameras:
            logger.warning(f"No metadata found for {cam_id}, using defaults")
            cameras[cam_id] = CameraMetadata(
                camera_id=cam_id,
                intrinsics=CameraIntrinsics(fx=1000, fy=1000, cx=1280, cy=720),
                extrinsics=CameraExtrinsics(),
            )
    
    return cameras


if __name__ == "__main__":
    # Test camera utilities
    intrinsics = CameraIntrinsics(fx=500, fy=500, cx=320, cy=240, width=640, height=480)
    print(f"Intrinsics matrix:\n{intrinsics.matrix}")
    
    # Test projection
    point_3d = np.array([1.0, 0.5, 2.0])
    pixel = intrinsics.project(point_3d)
    print(f"3D point {point_3d} projects to pixel {pixel}")
    
    # Test unprojection
    point_back = intrinsics.unproject(pixel, depth=2.0)
    print(f"Pixel {pixel} unprojects to {point_back}")
    
    # Test extrinsics
    extrinsics = CameraExtrinsics.look_at(
        camera_pos=np.array([0, 0, 5]),
        target=np.array([0, 0, 0]),
    )
    print(f"Camera position: {extrinsics.camera_position}")
    print(f"Camera direction: {extrinsics.camera_direction}")
