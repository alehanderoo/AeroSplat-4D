"""
Camera utilities for the DepthSplat inference backend.

Provides functions for:
- Creating camera matrices from various inputs
- Generating camera trajectories for video
- Converting between coordinate systems
- Creating target cameras from azimuth/elevation/distance
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
        fx = fy = 0.5 / np.tan(fov_rad / 2)
        cx, cy = 0.5, 0.5
    else:
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
    up: np.ndarray = None,
) -> np.ndarray:
    """
    Create camera-to-world matrix looking from eye to target.

    Uses OpenCV convention (+X right, +Y down, +Z forward).

    Args:
        eye: Camera position [3]
        target: Look-at target position [3]
        up: World up vector [3] (defaults to [0, 0, 1])

    Returns:
        [4, 4] camera-to-world matrix
    """
    if up is None:
        up = np.array([0, 0, 1], dtype=np.float32)

    eye = np.asarray(eye, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    up = np.asarray(up, dtype=np.float32)

    # Forward direction (camera looks along +Z in OpenCV)
    forward = target - eye
    forward = forward / np.linalg.norm(forward)

    # Right direction
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-6:
        # Camera looking straight up/down, use alternative up
        up = np.array([0, 1, 0], dtype=np.float32)
        right = np.cross(forward, up)
    right = right / np.linalg.norm(right)

    # Recompute up to be orthogonal
    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)

    # Build rotation matrix (columns are right, -up, forward for OpenCV)
    R = np.stack([right, -up, forward], axis=1)

    # Build 4x4 matrix
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = R
    c2w[:3, 3] = eye

    return c2w


def create_target_camera(
    azimuth: float,
    elevation: float,
    distance: float,
    base_radius: float = 2.0,
) -> np.ndarray:
    """
    Create a target camera extrinsics from spherical coordinates.

    Uses OpenCV convention where:
    - +X is right
    - +Y is down
    - +Z is forward (optical axis)

    Args:
        azimuth: Horizontal angle in degrees (0=+X, 90=+Y)
        elevation: Vertical angle in degrees (-90=below, 0=horizontal, 90=above)
        distance: Distance factor (multiplied by base_radius)
        base_radius: Base camera distance from origin

    Returns:
        [4, 4] camera-to-world matrix
    """
    radius = base_radius * distance

    azimuth_rad = np.deg2rad(azimuth)
    elevation_rad = np.deg2rad(elevation)

    # Spherical coordinates
    x = radius * np.cos(elevation_rad) * np.cos(azimuth_rad)
    y = radius * np.cos(elevation_rad) * np.sin(azimuth_rad)
    z = radius * np.sin(elevation_rad)

    cam_pos = np.array([x, y, z], dtype=np.float32)
    target = np.array([0, 0, 0], dtype=np.float32)

    return look_at(cam_pos, target)


def create_orbit_cameras(
    num_views: int,
    radius: float = 2.0,
    elevation: float = 0.3,
    azimuth_start: float = 0.0,
    look_at_point: np.ndarray = None,
) -> np.ndarray:
    """
    Create camera extrinsics orbiting around a point.

    Args:
        num_views: Number of camera views
        radius: Distance from look_at_point in XY plane
        elevation: Height offset (Z coordinate)
        azimuth_start: Starting azimuth angle in degrees
        look_at_point: Point to look at [3] (defaults to origin)

    Returns:
        [V, 4, 4] camera-to-world matrices
    """
    if look_at_point is None:
        look_at_point = np.array([0, 0, 0], dtype=np.float32)

    azimuth_start_rad = np.deg2rad(azimuth_start)
    angles = np.linspace(0, 2 * np.pi, num_views, endpoint=False) + azimuth_start_rad

    extrinsics = []
    for angle in angles:
        x = radius * np.cos(angle) + look_at_point[0]
        y = radius * np.sin(angle) + look_at_point[1]
        z = elevation + look_at_point[2]

        eye = np.array([x, y, z])
        c2w = look_at(eye, look_at_point)
        extrinsics.append(c2w)

    return np.stack(extrinsics)


def create_360_video_cameras(
    num_frames: int,
    radius: float = 1.4,
    elevation_angle: float = 30.0,
) -> np.ndarray:
    """
    Create camera extrinsics for a 360-degree rotation video.

    Args:
        num_frames: Number of frames
        radius: Distance from origin
        elevation_angle: Elevation above horizontal in degrees

    Returns:
        [N, 4, 4] camera-to-world matrices
    """
    angles = np.linspace(0, 2 * np.pi, num_frames, endpoint=False)
    elevation_rad = np.deg2rad(elevation_angle)

    extrinsics = []
    for angle in angles:
        # Camera position on sphere
        x = radius * np.cos(elevation_rad) * np.cos(angle)
        y = radius * np.cos(elevation_rad) * np.sin(angle)
        z = radius * np.sin(elevation_rad)

        cam_pos = np.array([x, y, z], dtype=np.float32)
        target = np.array([0, 0, 0], dtype=np.float32)

        c2w = look_at(cam_pos, target)
        extrinsics.append(c2w)

    return np.stack(extrinsics)


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
    K_norm = K.copy().astype(np.float32)

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
    K_pixel = K.copy().astype(np.float32)

    if K.ndim == 2:
        K_pixel[0, :] *= w
        K_pixel[1, :] *= h
    else:
        K_pixel[:, 0, :] *= w
        K_pixel[:, 1, :] *= h

    return K_pixel


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


def opengl_to_opencv(c2w_opengl: np.ndarray) -> np.ndarray:
    """
    Convert OpenGL camera-to-world matrix to OpenCV convention.

    OpenGL: +X right, +Y up, +Z backward (out of screen)
    OpenCV: +X right, +Y down, +Z forward

    Args:
        c2w_opengl: [4, 4] or [N, 4, 4] camera-to-world in OpenGL convention

    Returns:
        Camera-to-world in OpenCV convention
    """
    # Flip Y and Z axes
    flip = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)

    if c2w_opengl.ndim == 2:
        c2w_opencv = c2w_opengl.copy()
        c2w_opencv[:3, :3] = c2w_opengl[:3, :3] @ flip[:3, :3]
        return c2w_opencv
    else:
        c2w_opencv = c2w_opengl.copy()
        for i in range(c2w_opengl.shape[0]):
            c2w_opencv[i, :3, :3] = c2w_opengl[i, :3, :3] @ flip[:3, :3]
        return c2w_opencv


def invert_extrinsics(c2w: np.ndarray) -> np.ndarray:
    """
    Invert camera-to-world to get world-to-camera matrix.

    Args:
        c2w: [4, 4] or [N, 4, 4] camera-to-world matrix

    Returns:
        World-to-camera matrix (same shape)
    """
    return np.linalg.inv(c2w)


def compute_camera_elevation(c2w: np.ndarray) -> float:
    """
    Compute elevation angle of a camera looking at origin.

    Args:
        c2w: [4, 4] camera-to-world matrix

    Returns:
        Elevation in degrees
    """
    position = c2w[:3, 3]
    distance = np.linalg.norm(position)
    if distance < 1e-6:
        return 0.0
    return np.rad2deg(np.arcsin(position[2] / distance))


def compute_mean_camera_elevation(c2w_array: np.ndarray) -> float:
    """
    Compute mean elevation angle across multiple cameras.

    Args:
        c2w_array: [V, 4, 4] camera-to-world matrices

    Returns:
        Mean elevation in degrees
    """
    elevations = []
    for i in range(c2w_array.shape[0]):
        elevations.append(compute_camera_elevation(c2w_array[i]))
    return float(np.mean(elevations))
