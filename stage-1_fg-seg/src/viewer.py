from __future__ import annotations

import time
from typing import Sequence

import numpy as np

from .cameras import CameraModel
from .voxelizer import VoxelizerResult


def _counts_to_colors(counts: np.ndarray, max_cameras: int) -> np.ndarray:
    if counts.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    max_cameras = max(1, max_cameras)
    normalised = np.clip(counts.astype(np.float32) / float(max_cameras), 0.0, 1.0)
    # Simple gold -> red -> purple gradient
    r = normalised
    g = 1.0 - normalised * 0.8
    b = 0.2 + 0.6 * (1.0 - normalised)
    colors = np.stack([r, g, b], axis=1)
    return np.clip(colors, 0.0, 1.0).astype(np.float32)


def _add_camera_frustum(server, cam: CameraModel, frustum_depth: float = 2.0) -> None:
    """Add a visual frustum for a camera to show its field of view."""
    # Compute the 4 corners of the image plane at frustum_depth
    w = cam.intrinsics.width
    h = cam.intrinsics.height
    fx = cam.intrinsics.fx
    fy = cam.intrinsics.fy
    cx = cam.intrinsics.cx
    cy = cam.intrinsics.cy
    
    # Image corners in pixel coordinates
    corners_2d = np.array([
        [0, 0],
        [w, 0],
        [w, h],
        [0, h]
    ], dtype=np.float32)
    
    # Unproject to 3D in camera space at depth=frustum_depth
    corners_cam = []
    for u, v in corners_2d:
        x = (u - cx) * frustum_depth / fx
        y = (v - cy) * frustum_depth / fy
        z = frustum_depth
        corners_cam.append([x, y, z])
    
    corners_cam = np.array(corners_cam, dtype=np.float32)
    
    # Transform to world space
    corners_cam_h = np.concatenate([corners_cam, np.ones((4, 1), dtype=np.float32)], axis=1)
    corners_world = corners_cam_h @ cam.extrinsics.camera_to_world.T
    corners_world = corners_world[:, :3]
    
    # Get camera position directly from the camera_to_world matrix (translation column)
    cam_pos = cam.extrinsics.camera_to_world[:3, 3]
    
    # Create lines from camera center to each corner
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    for i, corner in enumerate(corners_world):
        line_points = np.stack([cam_pos, corner], axis=0)
        server.scene.add_spline_catmull_rom(
            f"/camera_frustums/{cam.name}/line_{i}",
            positions=line_points,
            color=colors[i],
            line_width=2.0,
            segments=1,
        )
    
    # Add lines connecting the corners (forming a rectangle)
    for i in range(4):
        j = (i + 1) % 4
        line_points = np.stack([corners_world[i], corners_world[j]], axis=0)
        server.scene.add_spline_catmull_rom(
            f"/camera_frustums/{cam.name}/rect_{i}",
            positions=line_points,
            color=(128, 128, 128),
            line_width=1.5,
            segments=1,
        )


def launch_viewer(result: VoxelizerResult, cameras: Sequence[CameraModel]) -> None:
    try:
        import viser
    except ModuleNotFoundError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("viser is required for visualisation. Install with `pip install viser`.") from exc

    server = viser.ViserServer()

    points = result.occupied_points_cpu()
    visibility = result.visibility_counts.detach().cpu().numpy()
    occupied_counts = visibility[result.occupancy.detach().cpu().numpy()]
    colors = _counts_to_colors(occupied_counts, max_cameras=len(cameras))

    print(f"\n=== Viewer Info ===")
    print(f"Occupied points to display: {points.shape[0]:,}")
    
    if points.shape[0] == 0:
        print("WARNING: No occupied voxels found! Check your camera parameters and grid bounds.")
        print("The viewer will still show cameras and bounding box for debugging.")
    else:
        print(f"Point cloud color range: {occupied_counts.min()}-{occupied_counts.max()} cameras")

    # Add point cloud only if we have points
    if points.shape[0] > 0:
        server.scene.add_point_cloud(
            "/voxel_overlap",
            points=points,
            colors=colors,
            point_size=0.03,
        )
        print("✓ Added voxel point cloud")
    
    # Add bounding box helper
    extent = result.voxel_size * (np.array(result.grid_shape) - 1)
    center = result.grid_min + 0.5 * extent
    server.scene.add_box(
        "/bounding_box",
        position=tuple(center.tolist()),
        dimensions=tuple(extent.tolist()),
        opacity=0.1,
        color=(51, 153, 255),
        wireframe=True,
    )
    print(f"✓ Added bounding box at {center} with extent {extent}")

    # Add camera frames and frustums
    for cam in cameras:
        server.scene.add_frame(
            f"/cameras/{cam.name}",
            wxyz=tuple(float(v) for v in cam.extrinsics.orientation_wxyz.tolist()),
            position=tuple(float(v) for v in cam.extrinsics.position.tolist()),
            axes_length=0.5,
            axes_radius=0.01,
        )
        
        # Add frustum visualization
        frustum_depth = min(cam.far * 0.3, 5.0)  # Show 30% of far plane or 5m max
        _add_camera_frustum(server, cam, frustum_depth=frustum_depth)
        
    print(f"✓ Added {len(cameras)} cameras with frustums")

    # Print local URL using supported methods
    url = f"http://{server.get_host()}:{server.get_port()}"
    print(f"\n{'='*50}")
    print(f"Viser running at: {url}")
    print(f"{'='*50}")
    print("Press Ctrl+C to exit.")
    try:
        server.sleep_forever()
    except KeyboardInterrupt:
        pass
