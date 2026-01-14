"""
Camera utilities for the DepthSplat Gradio demo.

Provides functions for:
- Creating camera matrices from various inputs
- Generating camera trajectories for video
- Converting between coordinate systems
"""

import json
from typing import List, Tuple, Optional, Union

import numpy as np


def create_intrinsics_from_fov(
    fov_degrees: float,
    image_size: Tuple[int, int] = (256, 256),
    normalized: bool = True,
) -> np.ndarray:
    """
    Create intrinsics matrix from field of view.

    Args:
        fov_degrees: Horizontal field of view in degrees
        image_size: (H, W) image dimensions
        normalized: If True, return normalized intrinsics (fx, fy in [0,1] relative to image size)

    Returns:
        [3, 3] intrinsics matrix
    """
    h, w = image_size
    fov_rad = np.deg2rad(fov_degrees)

    if normalized:
        # Normalized focal length
        fx = fy = 0.5 / np.tan(fov_rad / 2)
        cx, cy = 0.5, 0.5
    else:
        # Pixel focal length
        fx = fy = (w / 2) / np.tan(fov_rad / 2)
        cx, cy = w / 2, h / 2

    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1],
    ], dtype=np.float32)

    return K


def create_intrinsics_from_focal(
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    image_size: Tuple[int, int] = None,
    normalize: bool = True,
) -> np.ndarray:
    """
    Create intrinsics matrix from focal length and principal point.

    Args:
        fx, fy: Focal lengths (in pixels if normalize=True, normalized otherwise)
        cx, cy: Principal point (in pixels if normalize=True, normalized otherwise)
        image_size: (H, W) image dimensions (required if normalize=True)
        normalize: If True, normalize the intrinsics to [0, 1] range

    Returns:
        [3, 3] intrinsics matrix
    """
    if normalize and image_size is not None:
        h, w = image_size
        fx = fx / w
        fy = fy / h
        cx = cx / w
        cy = cy / h

    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1],
    ], dtype=np.float32)

    return K


def look_at(
    eye: np.ndarray,
    target: np.ndarray,
    up: np.ndarray = np.array([0, 0, 1]),
) -> np.ndarray:
    """
    Create camera-to-world matrix looking from eye to target.

    Uses OpenCV convention (+X right, +Y down, +Z forward).

    Args:
        eye: Camera position [3]
        target: Look-at target position [3]
        up: World up vector [3]

    Returns:
        [4, 4] camera-to-world matrix
    """
    eye = np.asarray(eye, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    up = np.asarray(up, dtype=np.float32)

    # Forward direction (camera looks along +Z in OpenCV)
    forward = target - eye
    forward = forward / np.linalg.norm(forward)

    # Right direction
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)

    # Recompute up to be orthogonal
    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)

    # Build rotation matrix (columns are right, -up, forward for OpenCV)
    # Actually for camera-to-world: columns are world directions of camera axes
    R = np.stack([right, -up, forward], axis=1)

    # Build 4x4 matrix
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = R
    c2w[:3, 3] = eye

    return c2w


def create_orbit_cameras(
    num_views: int,
    radius: float = 2.0,
    elevation: float = 0.3,
    azimuth_start: float = 0.0,
    look_at_point: np.ndarray = np.array([0, 0, 0]),
) -> np.ndarray:
    """
    Create camera extrinsics orbiting around a point.

    Args:
        num_views: Number of camera views
        radius: Distance from look_at_point in XY plane
        elevation: Height offset (Z coordinate)
        azimuth_start: Starting azimuth angle in degrees
        look_at_point: Point to look at [3]

    Returns:
        [V, 4, 4] camera-to-world matrices
    """
    azimuth_start_rad = np.deg2rad(azimuth_start)
    angles = np.linspace(0, 2 * np.pi, num_views, endpoint=False) + azimuth_start_rad

    extrinsics = []
    for angle in angles:
        # Camera position
        x = radius * np.cos(angle) + look_at_point[0]
        y = radius * np.sin(angle) + look_at_point[1]
        z = elevation + look_at_point[2]

        eye = np.array([x, y, z])
        c2w = look_at(eye, look_at_point)
        extrinsics.append(c2w)

    return np.stack(extrinsics)


def create_spiral_cameras(
    num_views: int,
    radius_range: Tuple[float, float] = (1.5, 2.5),
    elevation_range: Tuple[float, float] = (-0.2, 0.8),
    num_rotations: float = 1.0,
    look_at_point: np.ndarray = np.array([0, 0, 0]),
) -> np.ndarray:
    """
    Create camera extrinsics following a spiral path.

    Args:
        num_views: Number of camera views
        radius_range: (min_radius, max_radius) for spiral
        elevation_range: (min_z, max_z) for spiral
        num_rotations: Number of full rotations
        look_at_point: Point to look at [3]

    Returns:
        [V, 4, 4] camera-to-world matrices
    """
    t = np.linspace(0, 1, num_views)

    # Interpolate radius and elevation
    radii = radius_range[0] + (radius_range[1] - radius_range[0]) * t
    elevations = elevation_range[0] + (elevation_range[1] - elevation_range[0]) * t

    # Angles
    angles = 2 * np.pi * num_rotations * t

    extrinsics = []
    for angle, radius, elevation in zip(angles, radii, elevations):
        x = radius * np.cos(angle) + look_at_point[0]
        y = radius * np.sin(angle) + look_at_point[1]
        z = elevation + look_at_point[2]

        eye = np.array([x, y, z])
        c2w = look_at(eye, look_at_point)
        extrinsics.append(c2w)

    return np.stack(extrinsics)


def interpolate_cameras(
    c2w_start: np.ndarray,
    c2w_end: np.ndarray,
    num_steps: int,
) -> np.ndarray:
    """
    Interpolate between two camera poses.

    Uses spherical linear interpolation for rotation.

    Args:
        c2w_start: [4, 4] starting camera-to-world matrix
        c2w_end: [4, 4] ending camera-to-world matrix
        num_steps: Number of interpolation steps

    Returns:
        [N, 4, 4] interpolated camera-to-world matrices
    """
    from scipy.spatial.transform import Rotation, Slerp

    # Extract rotations and translations
    R_start = Rotation.from_matrix(c2w_start[:3, :3])
    R_end = Rotation.from_matrix(c2w_end[:3, :3])
    t_start = c2w_start[:3, 3]
    t_end = c2w_end[:3, 3]

    # Create slerp interpolator
    key_times = [0, 1]
    key_rots = Rotation.concatenate([R_start, R_end])
    slerp = Slerp(key_times, key_rots)

    # Interpolate
    t_values = np.linspace(0, 1, num_steps)
    interpolated_rots = slerp(t_values)
    interpolated_trans = t_start[None, :] + t_values[:, None] * (t_end - t_start)[None, :]

    # Build matrices
    c2w_interp = np.zeros((num_steps, 4, 4), dtype=np.float32)
    c2w_interp[:, :3, :3] = interpolated_rots.as_matrix()
    c2w_interp[:, :3, 3] = interpolated_trans
    c2w_interp[:, 3, 3] = 1.0

    return c2w_interp


def parse_camera_json(json_str: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse camera parameters from JSON string.

    Expected format:
    {
        "extrinsics": [[4x4 matrix], ...],  // camera-to-world
        "intrinsics": [[3x3 matrix], ...],  // normalized
    }

    or per-image format:
    [
        {"extrinsic": [4x4], "intrinsic": [3x3]},
        ...
    ]

    Args:
        json_str: JSON string with camera parameters

    Returns:
        (extrinsics, intrinsics) as numpy arrays [V, 4, 4] and [V, 3, 3]
    """
    data = json.loads(json_str)

    if isinstance(data, dict):
        extrinsics = np.array(data['extrinsics'], dtype=np.float32)
        intrinsics = np.array(data['intrinsics'], dtype=np.float32)
    elif isinstance(data, list):
        extrinsics = np.array([d['extrinsic'] for d in data], dtype=np.float32)
        intrinsics = np.array([d['intrinsic'] for d in data], dtype=np.float32)
    else:
        raise ValueError("Unknown JSON format")

    return extrinsics, intrinsics


def blender_to_opencv(c2w_blender: np.ndarray) -> np.ndarray:
    """
    Convert Blender camera-to-world matrix to OpenCV convention.

    Blender: +X right, +Y forward, +Z up
    OpenCV:  +X right, +Y down, +Z forward

    Args:
        c2w_blender: [4, 4] or [N, 4, 4] camera-to-world in Blender convention

    Returns:
        Camera-to-world in OpenCV convention
    """
    # Transformation matrix from Blender to OpenCV
    # Swaps Y and Z axes, negates new Y
    flip = np.array([
        [1, 0, 0, 0],
        [0, 0, -1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ], dtype=np.float32)

    if c2w_blender.ndim == 2:
        return c2w_blender @ flip
    else:
        return np.einsum('nij,jk->nik', c2w_blender, flip)


def opencv_to_blender(c2w_opencv: np.ndarray) -> np.ndarray:
    """
    Convert OpenCV camera-to-world matrix to Blender convention.

    Args:
        c2w_opencv: [4, 4] or [N, 4, 4] camera-to-world in OpenCV convention

    Returns:
        Camera-to-world in Blender convention
    """
    flip = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, -1, 0, 0],
        [0, 0, 0, 1],
    ], dtype=np.float32)

    if c2w_opencv.ndim == 2:
        return c2w_opencv @ flip
    else:
        return np.einsum('nij,jk->nik', c2w_opencv, flip)


def invert_extrinsics(c2w: np.ndarray) -> np.ndarray:
    """
    Invert camera-to-world to get world-to-camera matrix.

    Args:
        c2w: [4, 4] or [N, 4, 4] camera-to-world matrix

    Returns:
        World-to-camera matrix (same shape)
    """
    if c2w.ndim == 2:
        return np.linalg.inv(c2w)
    else:
        return np.linalg.inv(c2w)


def normalize_intrinsics(K: np.ndarray, image_size: Tuple[int, int]) -> np.ndarray:
    """
    Normalize intrinsics matrix to [0, 1] range.

    Args:
        K: [3, 3] or [N, 3, 3] intrinsics with pixel coordinates
        image_size: (H, W) image dimensions

    Returns:
        Normalized intrinsics
    """
    h, w = image_size
    K_norm = K.copy()

    if K.ndim == 2:
        K_norm[0, :] /= w
        K_norm[1, :] /= h
    else:
        K_norm[:, 0, :] /= w
        K_norm[:, 1, :] /= h

    return K_norm


def denormalize_intrinsics(K: np.ndarray, image_size: Tuple[int, int]) -> np.ndarray:
    """
    Denormalize intrinsics matrix from [0, 1] to pixel coordinates.

    Args:
        K: [3, 3] or [N, 3, 3] normalized intrinsics
        image_size: (H, W) image dimensions

    Returns:
        Intrinsics in pixel coordinates
    """
    h, w = image_size
    K_pixel = K.copy()

    if K.ndim == 2:
        K_pixel[0, :] *= w
        K_pixel[1, :] *= h
    else:
        K_pixel[:, 0, :] *= w
        K_pixel[:, 1, :] *= h

    return K_pixel
