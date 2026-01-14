"""
Render Utilities: Helper functions for Isaac Sim rendering
Contains position calculations, transformations, and utility functions.
"""

import json
import os

import numpy as np
from PIL import Image
from pxr import Gf, Usd, UsdGeom

try:
    from isaacsim.sensors.camera import Camera as IsaacSimCamera
except ModuleNotFoundError:
    IsaacSimCamera = None


def reshape_to_matrix(matrix_values):
    """Convert a flat iterable into a 4x4 numpy matrix."""
    array = np.array(matrix_values, dtype=np.float64).reshape(-1)
    if array.size != 16:
        raise ValueError(f"Expected 16 values to reshape into 4x4 matrix, got {array.size}")
    matrix = array.reshape(4, 4)

    # Replicator / USD matrices are provided in column-major order. Detect this by checking whether the
    # translation components live in the last row rather than the last column and transpose if needed.
    translation_column_norm = np.linalg.norm(matrix[:3, 3])
    translation_row_norm = np.linalg.norm(matrix[3, :3])
    if translation_column_norm < translation_row_norm:
        matrix = matrix.T

    return matrix


def matrix_to_list(matrix: np.ndarray) -> list[list[float]]:
    """Convert numpy matrix to a nested Python list of floats."""
    return matrix.astype(np.float64).tolist()


def extract_camera_pose(view_matrix: np.ndarray):
    """Return camera-to-world matrix, translation, and quaternion (w, x, y, z)."""
    camera_to_world = np.linalg.inv(view_matrix)
    transform = Gf.Transform()
    transform.SetMatrix(Gf.Matrix4d(camera_to_world.tolist()))
    translation = transform.GetTranslation()
    rotation = transform.GetRotation().GetQuat()

    position = [float(translation[0]), float(translation[1]), float(translation[2])]
    quaternion = [
        float(rotation.GetReal()),
        float(rotation.GetImaginary()[0]),
        float(rotation.GetImaginary()[1]),
        float(rotation.GetImaginary()[2]),
    ]
    return camera_to_world, position, quaternion


def compute_camera_intrinsics(camera_params: dict) -> dict:
    """Compute intrinsic matrix and related parameters from camera annotator output."""
    resolution = camera_params["renderProductResolution"]
    width = int(resolution[0])
    height = int(resolution[1])

    aperture = np.asarray(camera_params["cameraAperture"], dtype=np.float64)
    aperture_offset = np.asarray(camera_params["cameraApertureOffset"], dtype=np.float64)
    focal_length = float(camera_params["cameraFocalLength"])

    if width == 0 or aperture[0] == 0.0:
        raise ValueError("Invalid camera aperture or resolution for intrinsic computation")

    pixel_size = aperture[0] / width
    fx = focal_length / pixel_size
    fy = focal_length / pixel_size
    cx = width / 2.0 + aperture_offset[0]
    cy = height / 2.0 + aperture_offset[1]

    intrinsic_matrix = [
        [float(fx), 0.0, float(cx)],
        [0.0, float(fy), float(cy)],
        [0.0, 0.0, 1.0],
    ]

    return {
        "matrix": intrinsic_matrix,
        "fx": float(fx),
        "fy": float(fy),
        "cx": float(cx),
        "cy": float(cy),
        "focal_length_mm": focal_length,
        "aperture": aperture.astype(float).tolist(),
        "aperture_offset": aperture_offset.astype(float).tolist(),
    }


def build_camera_metadata(camera_name, camera_prim, render_product, camera_params):
    """Assemble static camera metadata including intrinsics and extrinsics."""
    try:
        view_matrix = reshape_to_matrix(camera_params["cameraViewTransform"])
        projection_matrix = reshape_to_matrix(camera_params["cameraProjection"])
        intrinsics = compute_camera_intrinsics(camera_params)
        camera_to_world, position, quaternion = extract_camera_pose(view_matrix)
    except Exception as exc:
        print(f"[RENDER] Warning: Unable to build metadata for {camera_name}: {exc}")
        return None

    render_product_path = getattr(render_product, "path", None)
    if render_product_path is not None:
        render_product_path = str(render_product_path)

    camera_path = str(camera_prim.GetPath()) if camera_prim and camera_prim.IsValid() else None
    resolution = camera_params["renderProductResolution"]
    width = int(resolution[0])
    height = int(resolution[1])

    metadata = {
        "name": camera_name,
        "camera_prim_path": camera_path,
        "render_product_path": render_product_path,
        "resolution": {"width": width, "height": height},
        "intrinsics": intrinsics,
        "extrinsics": {
            "world_to_camera_matrix": matrix_to_list(view_matrix),
            "camera_to_world_matrix": matrix_to_list(camera_to_world),
            "position_world": position,
            "orientation_quat_wxyz": quaternion,
        },
        "projection_matrix": matrix_to_list(projection_matrix),
        "meters_per_scene_unit": float(camera_params["metersPerSceneUnit"]),
    }
    return metadata


def flatten_mapping_values(value):
    """Recursively flatten nested dict/list/tuple structures to extract all string values."""
    if isinstance(value, dict):
        for sub_value in value.values():
            yield from flatten_mapping_values(sub_value)
    elif isinstance(value, (list, tuple, set)):
        for sub_value in value:
            yield from flatten_mapping_values(sub_value)
    elif value is not None:
        yield str(value)


def extract_instance_drone_ids(id_to_labels):
    """Extract drone instance IDs from id_to_labels mapping."""
    if not isinstance(id_to_labels, dict):
        return []
    
    drone_ids = []
    for raw_id, payload in id_to_labels.items():
        # Flatten all values in the payload to check for 'drone'
        entries = [entry.lower() for entry in flatten_mapping_values(payload)]
        if any("drone" in entry for entry in entries):
            try:
                drone_ids.append(int(raw_id))
            except (TypeError, ValueError):
                continue
    
    return drone_ids


def compute_world_bbox_3d(prim, seconds=None):
    """
    Compute the 3D bounding box of a prim in world coordinates.
    
    Args:
        prim: USD Prim to compute bounding box for
        seconds: Time in seconds (will be converted to time codes), or None for default
        
    Returns:
        dict with 'center', 'extents', 'corners_world' (8 corners in world space), or None if failed
    """
    if not prim or not prim.IsValid():
        return None
    
    try:
        stage = prim.GetStage()
        tcps = float(stage.GetTimeCodesPerSecond() or 30.0)
        time_code = Usd.TimeCode.Default() if seconds is None else Usd.TimeCode(seconds * tcps)
        
        purposes = [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy]
        bbox_cache = UsdGeom.BBoxCache(time_code, purposes, useExtentsHint=True)
        
        bound = bbox_cache.ComputeWorldBound(prim)
        bound_range = bound.ComputeAlignedBox()
        
        min_point = bound_range.GetMin()
        max_point = bound_range.GetMax()
        
        # Calculate center and extents
        center = [
            (min_point[0] + max_point[0]) / 2.0,
            (min_point[1] + max_point[1]) / 2.0,
            (min_point[2] + max_point[2]) / 2.0,
        ]
        
        extents = [
            max_point[0] - min_point[0],
            max_point[1] - min_point[1],
            max_point[2] - min_point[2],
        ]
        
        # Calculate all 8 corners of the bounding box
        corners = [
            [min_point[0], min_point[1], min_point[2]],  # 0: min corner
            [min_point[0], min_point[1], max_point[2]],  # 1
            [min_point[0], max_point[1], min_point[2]],  # 2
            [min_point[0], max_point[1], max_point[2]],  # 3
            [max_point[0], min_point[1], min_point[2]],  # 4
            [max_point[0], min_point[1], max_point[2]],  # 5
            [max_point[0], max_point[1], min_point[2]],  # 6
            [max_point[0], max_point[1], max_point[2]],  # 7: max corner
        ]
        
        return {
            "center": center,
            "extents": extents,
            "min": [float(min_point[0]), float(min_point[1]), float(min_point[2])],
            "max": [float(max_point[0]), float(max_point[1]), float(max_point[2])],
            "corners_world": corners,
        }
    except Exception as exc:
        print(f"[RENDER] Warning: Failed to compute world bbox for {prim.GetPath()}: {exc}")
        return None


def project_point_to_screen(world_point, view_matrix, projection_matrix, screen_width, screen_height):
    """
    Project a 3D world point to 2D screen coordinates.
    
    Args:
        world_point: [x, y, z] in world space
        view_matrix: 4x4 camera view (world-to-camera) matrix
        projection_matrix: 4x4 camera projection matrix
        screen_width: Screen width in pixels
        screen_height: Screen height in pixels
        
    Returns:
        dict with 'pixel': [x, y], 'ndc': [x, y], 'depth': z, 'in_front': bool, or None if behind camera
    """
    try:
        # Convert to homogeneous coordinates
        world_point_h = np.array([world_point[0], world_point[1], world_point[2], 1.0], dtype=np.float64)

        # Transform to camera space
        camera_point = view_matrix @ world_point_h

        # In Omniverse / OpenGL convention the camera looks down the -Z axis. A negative
        # Z value therefore indicates the point lies in front of the camera.
        in_front = bool(camera_point[2] < 0)

        # Project to clip space
        clip_point = projection_matrix @ camera_point

        # Perspective divide to get NDC
        if abs(clip_point[3]) < 1e-8:
            return None

        ndc = clip_point[:3] / clip_point[3]

        # Convert NDC to pixel coordinates
        # NDC: [-1, 1] -> pixel: [0, width] and [0, height]
        pixel_x = (ndc[0] + 1.0) * screen_width * 0.5
        pixel_y = (1.0 - ndc[1]) * screen_height * 0.5  # Flip Y axis

        depth = float(-camera_point[2])

        return {
            "pixel": [float(pixel_x), float(pixel_y)],
            "ndc": [float(ndc[0]), float(ndc[1])],
            "depth": depth,
            "in_front": in_front,
        }
    except Exception as exc:
        return None


def compute_mask_2d_stats(mask: np.ndarray, screen_width: int, screen_height: int):
    """Compute centroid and bounding box statistics from a boolean mask."""
    if mask is None:
        return None

    try:
        mask_bool = mask.astype(bool, copy=False)
        non_zero = np.argwhere(mask_bool)
        if non_zero.size == 0:
            return None

        ys = non_zero[:, 0].astype(np.float64)
        xs = non_zero[:, 1].astype(np.float64)

        min_x = xs.min()
        max_x = xs.max()
        min_y = ys.min()
        max_y = ys.max()

        center_x = xs.mean()
        center_y = ys.mean()

        width = max_x - min_x
        height = max_y - min_y

        # Clamp values into image bounds for convenience
        if screen_width is not None and screen_width > 0:
            min_x_clamped = float(np.clip(min_x, 0.0, screen_width - 1))
            max_x_clamped = float(np.clip(max_x, 0.0, screen_width - 1))
            ndc_x = float((center_x / screen_width) * 2.0 - 1.0)
        else:
            min_x_clamped = float(min_x)
            max_x_clamped = float(max_x)
            ndc_x = 0.0

        if screen_height is not None and screen_height > 0:
            min_y_clamped = float(np.clip(min_y, 0.0, screen_height - 1))
            max_y_clamped = float(np.clip(max_y, 0.0, screen_height - 1))
            ndc_y = float(1.0 - (center_y / screen_height) * 2.0)
        else:
            min_y_clamped = float(min_y)
            max_y_clamped = float(max_y)
            ndc_y = 0.0

        return {
            "pixel_center": [float(center_x), float(center_y)],
            "ndc": [ndc_x, ndc_y],
            "bbox": {
                "min": [float(min_x), float(min_y)],
                "max": [float(max_x), float(max_y)],
                "min_clamped": [min_x_clamped, min_y_clamped],
                "max_clamped": [max_x_clamped, max_y_clamped],
                "center": [float((min_x + max_x) * 0.5), float((min_y + max_y) * 0.5)],
                "width": float(width),
                "height": float(height),
                "visible": True,
            },
            "pixel_indices": non_zero,
        }
    except Exception as exc:
        print(f"[RENDER] Warning: Failed to compute mask stats: {exc}")
        return None


def find_drone_prim(stage, preferred_path=None):
    """
    Find the drone prim in the stage.
    
    Args:
        stage: USD Stage
        preferred_path: Optional prim path to check first
        
    Returns:
        Prim object or None
    """
    if not stage:
        return None
    
    try:
        # 1. Check preferred path first if provided
        if preferred_path:
            prim = stage.GetPrimAtPath(preferred_path)
            if prim and prim.IsValid():
                return prim
        
        # 2. Common drone/bird paths to check
        common_paths = [
            "/World/drone",
            "/World/Drone",
            "/World/bird",
            "/World/Bird",
            "/World/Eagle",
            "/drone",
            "/Drone",
        ]
        
        for path in common_paths:
            prim = stage.GetPrimAtPath(path)
            if prim and prim.IsValid():
                return prim
        
        # 3. Search through stage hierarchy for asset keywords
        keywords = ["drone", "bird", "eagle", "seeker", "copternode"]
        for prim in stage.Traverse():
            path_str = str(prim.GetPath()).lower()
            if any(kw in path_str for kw in keywords):
                return prim
        
        return None
    except Exception:
        return None


def safe_timeline_call(timeline, method_name, *args, **kwargs):
    """Call a timeline method if available."""
    try:
        method = getattr(timeline, method_name)
        return method(*args, **kwargs)
    except (AttributeError, Exception) as exc:
        return None


def create_camera_api_helper(camera_path, camera_name, render_product_path=None):
    """Instantiate an isaacsim.sensors.camera.Camera helper when available."""
    if IsaacSimCamera is None:
        return None

    try:
        return IsaacSimCamera(
            prim_path=camera_path,
            name=f"_render_helper_{camera_name}",
            render_product_path=render_product_path,
        )
    except Exception:
        return None
