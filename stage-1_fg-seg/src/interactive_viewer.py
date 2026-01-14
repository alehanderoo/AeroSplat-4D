"""
Interactive 3D viewer for drone localization with temporal animation.

This module provides an interactive Viser-based viewer with GUI controls
to visualize the drone trajectory, accumulation grids, and camera positions.
"""

from __future__ import annotations

import time
from typing import Optional, Sequence

import numpy as np

from .cameras import CameraModel
from .localizer import DroneTrajectory, DroneLocalization
from .voxelizer import VoxelizerResult

try:
    from .cpp_backend import find_crossing_rays
except ImportError:
    raise ImportError("C++ backend required for find_crossing_rays. Build with: python setup.py build_ext --inplace")


def _add_camera_frustum(server, cam: CameraModel, frustum_depth: float = 2.0, name_suffix: str = ""):
    """Add a visual frustum for a camera and return handles."""
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
    
    # Camera position
    cam_pos = cam.extrinsics.camera_to_world[:3, 3]
    
    # Create lines from camera center to each corner and store handles
    handles = []
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    for i, corner in enumerate(corners_world):
        line_points = np.stack([cam_pos, corner], axis=0)
        handle = server.scene.add_spline_catmull_rom(
            f"/camera_frustums/{cam.name}{name_suffix}/line_{i}",
            positions=line_points,
            color=colors[i],
            line_width=2.0,
            segments=1,
        )
        handles.append(handle)
    
    # Add lines connecting the corners
    for i in range(4):
        j = (i + 1) % 4
        line_points = np.stack([corners_world[i], corners_world[j]], axis=0)
        handle = server.scene.add_spline_catmull_rom(
            f"/camera_frustums/{cam.name}{name_suffix}/rect_{i}",
            positions=line_points,
            color=(128, 128, 128),
            line_width=1.5,
            segments=1,
        )
        handles.append(handle)
    
    return handles


def _accumulation_to_colors(values: np.ndarray, colormap: str = "hot") -> np.ndarray:
    """Convert accumulation values to RGB colors."""
    if values.size == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    
    # Normalize to [0, 1]
    min_val = values.min()
    max_val = values.max()
    if max_val - min_val > 1e-6:
        normalized = (values - min_val) / (max_val - min_val)
    else:
        normalized = np.ones_like(values) * 0.5
    
    # Apply colormap
    if colormap == "hot":
        # Red -> Yellow -> White
        r = normalized
        g = np.clip(normalized * 2 - 0.5, 0, 1)
        b = np.clip(normalized * 3 - 2, 0, 1)
    elif colormap == "viridis":
        # Purple -> Blue -> Green -> Yellow
        r = np.clip(1.5 * (normalized - 0.6), 0, 1)
        g = np.sin(np.pi * normalized) ** 2
        b = np.clip(1.5 * (0.5 - normalized), 0, 1)
    else:
        # Grayscale
        r = g = b = normalized
    
    colors = np.stack([r, g, b], axis=1)
    return (colors * 255).astype(np.uint8)


def launch_interactive_localizer(
    trajectory: DroneTrajectory,
    cameras: Sequence[CameraModel],
    voxelizer_result: Optional[VoxelizerResult] = None,
    host: str = "0.0.0.0",
    port: int = 8080,
) -> None:
    """
    Launch interactive viewer for drone trajectory.

    Args:
        trajectory: DroneTrajectory with localization results
        cameras: List of camera models
        voxelizer_result: Optional voxelizer result to show occupancy grid
        host: Server host
        port: Server port
    """
    try:
        import viser
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "viser is required for visualization. Install with `pip install viser`."
        ) from exc

    server = viser.ViserServer(host=host, port=port)
    
    # Enable dark mode
    server.gui.configure_theme(dark_mode=True)
    
    print(f"\n{'='*60}")
    print(f"Interactive Localizer Viewer")
    print(f"URL: http://{host}:{port}")
    print(f"{'='*60}\n")

    # Get trajectory data
    positions = trajectory.get_positions()
    confidences = trajectory.get_confidences()
    
    if positions.shape[0] == 0:
        print("WARNING: No localizations to display!")
        return

    num_frames = positions.shape[0]
    
    print(f"Trajectory: {num_frames} frames")
    
    # Check if positions are all identical (no actual motion detected)
    pos_min = positions.min(axis=0)
    pos_max = positions.max(axis=0)
    if np.allclose(pos_min, pos_max):
        print(f"WARNING: All positions are identical at {pos_min}")
        print("This indicates no motion was detected. Check frame differences with --save-diffs.")
    else:
        print(f"Position range: {pos_min} to {pos_max}")

    # Add GUI controls
    with server.gui.add_folder("Playback"):
        gui_frame_slider = server.gui.add_slider(
            "Frame",
            min=0,
            max=num_frames - 1,
            step=1,
            initial_value=0,
        )
        gui_play_button = server.gui.add_button("Play", icon=viser.Icon.PLAYER_PLAY)
        gui_speed_slider = server.gui.add_slider(
            "Speed (FPS)",
            min=1,
            max=30,
            step=1,
            initial_value=10,
        )
        gui_is_playing = server.gui.add_checkbox(
            "Playing",
            initial_value=False,
            disabled=True,
        )

    with server.gui.add_folder("Visualization"):
        gui_show_trajectory = server.gui.add_checkbox(
            "Show Trajectory",
            initial_value=True,
        )
        gui_show_current_grid = server.gui.add_checkbox(
            "Show Current Accumulation",
            initial_value=False,
        )
        gui_show_cameras = server.gui.add_checkbox(
            "Show Cameras",
            initial_value=True,
        )
        gui_show_frustums = server.gui.add_checkbox(
            "Show Frustums",
            initial_value=False,
        )
        gui_show_occupancy = server.gui.add_checkbox(
            "Show Occupancy Grid",
            initial_value=voxelizer_result is not None,
            disabled=voxelizer_result is None,
        )
        gui_show_rays = server.gui.add_checkbox(
            "Show Cast Rays",
            initial_value=False,
        )
        gui_ray_length = server.gui.add_slider(
            "Ray Length (m)",
            min=0.5,
            max=20.0,
            step=0.5,
            initial_value=5.0,
        )
        gui_ray_subsample = server.gui.add_slider(
            "Ray Subsample",
            min=1,
            max=100,
            step=1,
            initial_value=10,
            hint="Show every Nth ray",
        )
        gui_drone_size = server.gui.add_slider(
            "Drone Marker Size",
            min=0.05,
            max=2.0,
            step=0.05,
            initial_value=0.3,
        )
        gui_confidence_threshold = server.gui.add_slider(
            "Min Confidence (%)",
            min=0,
            max=100,
            step=1,
            initial_value=0,
        )

    with server.gui.add_folder("Ground Truth"):
        gui_show_gt = server.gui.add_checkbox(
            "Show Ground Truth",
            initial_value=trajectory.ground_truth_positions is not None,
            disabled=trajectory.ground_truth_positions is None,
        )

    with server.gui.add_folder("Info", expand_by_default=False):
        gui_info_frame = server.gui.add_text(
            "Frame",
            initial_value="0",
            disabled=True,
        )
        gui_info_position = server.gui.add_text(
            "Position",
            initial_value="(0, 0, 0)",
            disabled=True,
        )
        gui_info_confidence = server.gui.add_text(
            "Confidence",
            initial_value="0.0",
            disabled=True,
        )
        gui_info_cameras = server.gui.add_text(
            "Cameras Detected",
            initial_value="0",
            disabled=True,
        )

    # State for playback
    is_playing = False

    @gui_play_button.on_click
    def _(_) -> None:
        nonlocal is_playing
        is_playing = not is_playing
        gui_is_playing.value = is_playing
        if is_playing:
            gui_play_button.name = "Pause"
            gui_play_button.icon = viser.Icon.PLAYER_PAUSE
        else:
            gui_play_button.name = "Play"
            gui_play_button.icon = viser.Icon.PLAYER_PLAY

    # Add static elements and store handles
    # Cameras (always add, control visibility later)
    camera_handles = {}
    for cam in cameras:
        handle = server.scene.add_frame(
            f"/cameras/{cam.name}",
            wxyz=tuple(float(v) for v in cam.extrinsics.orientation_wxyz.tolist()),
            position=tuple(float(v) for v in cam.extrinsics.position.tolist()),
            axes_length=0.5,
            axes_radius=0.01,
            visible=gui_show_cameras.value,
        )
        camera_handles[cam.name] = handle

    # Frustums (always add, control visibility later)
    frustum_handles = {}
    for cam in cameras:
        frustum_depth = min(cam.far * 0.3, 5.0)
        handles = _add_camera_frustum(server, cam, frustum_depth=frustum_depth)
        frustum_handles[cam.name] = handles
        # Set initial visibility
        for handle in handles:
            handle.visible = gui_show_frustums.value

    # Occupancy grid (if available)
    occupancy_handle = None
    if voxelizer_result is not None:
        occ_points = voxelizer_result.occupied_points_cpu()
        if occ_points.shape[0] > 0:
            occ_colors = np.full((occ_points.shape[0], 3), [100, 100, 100], dtype=np.uint8)
            occupancy_handle = server.scene.add_point_cloud(
                "/occupancy_grid",
                points=occ_points,
                colors=occ_colors,
                point_size=0.02,
                visible=gui_show_occupancy.value,
            )

    # Full trajectory line
    trajectory_handle = None
    if num_frames > 1:
        trajectory_handle = server.scene.add_spline_catmull_rom(
            "/trajectory",
            positions=positions,
            color=(0, 255, 255),
            line_width=3.0,
            segments=num_frames - 1,
            visible=gui_show_trajectory.value,
        )

    # Ground truth
    gt_handle = None
    if trajectory.ground_truth_positions is not None:
        gt_pos = trajectory.ground_truth_positions
        if gt_pos.shape[0] > 1:
            gt_handle = server.scene.add_spline_catmull_rom(
                "/ground_truth",
                positions=gt_pos,
                color=(255, 0, 255),
                line_width=2.0,
                segments=gt_pos.shape[0] - 1,
                visible=gui_show_gt.value,
            )

    # Set up callbacks for visibility toggles
    @gui_show_trajectory.on_update
    def _(_) -> None:
        if trajectory_handle is not None:
            trajectory_handle.visible = gui_show_trajectory.value
    
    @gui_show_cameras.on_update
    def _(_) -> None:
        for handle in camera_handles.values():
            handle.visible = gui_show_cameras.value
    
    @gui_show_frustums.on_update
    def _(_) -> None:
        for handles in frustum_handles.values():
            for handle in handles:
                handle.visible = gui_show_frustums.value
    
    @gui_show_occupancy.on_update
    def _(_) -> None:
        if occupancy_handle is not None:
            occupancy_handle.visible = gui_show_occupancy.value
    
    @gui_show_gt.on_update
    def _(_) -> None:
        if gt_handle is not None:
            gt_handle.visible = gui_show_gt.value

    # Main update loop
    last_update_time = time.time()
    
    while True:
        current_time = time.time()
        
        # Playback logic
        if is_playing:
            dt = current_time - last_update_time
            fps = gui_speed_slider.value
            if dt >= 1.0 / fps:
                current_frame = gui_frame_slider.value
                next_frame = (current_frame + 1) % num_frames
                gui_frame_slider.value = next_frame
                last_update_time = current_time

        # Get current frame
        current_frame_idx = gui_frame_slider.value
        localization = trajectory.localizations[current_frame_idx]

        # Filter by confidence
        min_confidence = gui_confidence_threshold.value * confidences.max() / 100.0

        # Update info
        gui_info_frame.value = str(localization.frame_index)
        gui_info_position.value = f"({localization.position[0]:.2f}, {localization.position[1]:.2f}, {localization.position[2]:.2f})"
        gui_info_confidence.value = f"{localization.confidence:.2f}"
        gui_info_cameras.value = str(localization.num_cameras_detected)

        # Update current drone position
        if localization.confidence >= min_confidence:
            server.scene.add_icosphere(
                "/drone_current",
                radius=gui_drone_size.value,
                color=(255, 255, 0),
                position=tuple(localization.position.tolist()),
            )
        else:
            # Remove if below threshold
            try:
                server.scene.remove("/drone_current")
            except:
                pass

        # Update ray visualization
        if gui_show_rays.value and localization.ray_origins is not None:
            ray_origins = localization.ray_origins
            ray_dirs = localization.ray_dirs
            ray_weights = localization.ray_weights
            
            # Subsample rays
            subsample = gui_ray_subsample.value
            indices = np.arange(0, len(ray_origins), subsample)
            
            ray_origins_sub = ray_origins[indices]
            ray_dirs_sub = ray_dirs[indices]
            ray_weights_sub = ray_weights[indices]
            
            # Find rays that cross (come close to each other in 3D)
            crossing_mask = find_crossing_rays(ray_origins_sub, ray_dirs_sub, threshold=0.5)
            
            if np.any(crossing_mask):
                ray_origins_cross = ray_origins_sub[crossing_mask]
                ray_dirs_cross = ray_dirs_sub[crossing_mask]
                ray_weights_cross = ray_weights_sub[crossing_mask]
                
            else:
                print(f"No crossing rays found among {len(ray_origins_sub)} rays")
                ray_origins_cross = ray_origins_sub[:0]  # Empty array
                ray_dirs_cross = ray_dirs_sub[:0]
                ray_weights_cross = ray_weights_sub[:0]
            
            # Compute ray endpoints
            ray_length = gui_ray_length.value
            ray_ends = ray_origins_cross + ray_dirs_cross * ray_length
            
            # Create line segments (N, 2, 3) format for viser
            ray_segments = np.stack([ray_origins_cross, ray_ends], axis=1)
            
            # Color by weight (normalize to 0-1 range)
            ray_weights_flat = ray_weights_cross.flatten()
            if len(ray_weights_flat) > 0 and ray_weights_flat.max() > 0:
                weight_normalized = ray_weights_flat / ray_weights_flat.max()
            else:
                weight_normalized = np.zeros_like(ray_weights_flat)
            
            # Create colors (cyan to yellow gradient based on weight)
            # Shape (N, 2, 3) - color for each endpoint of each segment
            ray_colors = np.empty((len(ray_origins_cross), 2, 3), dtype=np.uint8)
            for i in range(len(ray_origins_cross)):
                w = weight_normalized[i]
                # Cyan (low weight) to yellow (high weight)
                color = np.array([int(w * 255), 255, int((1 - w) * 255)], dtype=np.uint8)
                ray_colors[i, 0] = color
                ray_colors[i, 1] = color
            
            # Add as line segments
            server.scene.add_line_segments(
                "/rays",
                points=ray_segments,
                colors=ray_colors,
                line_width=1.0,
            )
        else:
            # Remove rays if not showing
            if not gui_show_rays.value:
                pass  # Checkbox is off
            elif localization.ray_origins is None:
                # Debug: print once per frame change
                if current_frame_idx == 0 or gui_frame_slider.value != getattr(gui_frame_slider, '_last_debug_frame', -1):
                    print(f"Frame {localization.frame_index}: No ray data available (ray_origins is None)")
                    gui_frame_slider._last_debug_frame = current_frame_idx
            
            try:
                server.scene.remove("/rays")
            except:
                pass

        # Update current accumulation grid
        if gui_show_current_grid.value and localization.accumulation_grid is not None:
            grid = localization.accumulation_grid.numpy()
            
            # Threshold to show only significant voxels
            threshold_val = grid.max() * 0.1
            mask = grid > threshold_val
            
            if mask.any():
                ix, iy, iz = np.nonzero(mask)
                values = grid[ix, iy, iz]
                
                # Convert to world coordinates
                from .voxelizer import VoxelizerResult  # Local import to avoid circular dependency
                
                # Compute grid parameters (should match localizer)
                grid_min = localization.position - np.array(grid.shape) * 0.5 * 0.1  # Approximate
                voxel_size = 0.1  # Approximate
                
                x = grid_min[0] + (ix + 0.5) * voxel_size
                y = grid_min[1] + (iy + 0.5) * voxel_size
                z = grid_min[2] + (iz + 0.5) * voxel_size
                
                points = np.column_stack([x, y, z]).astype(np.float32)
                colors = _accumulation_to_colors(values, colormap="hot")
                
                server.scene.add_point_cloud(
                    "/accumulation_current",
                    points=points,
                    colors=colors,
                    point_size=0.05,
                )
            else:
                try:
                    server.scene.remove("/accumulation_current")
                except:
                    pass
        else:
            try:
                server.scene.remove("/accumulation_current")
            except:
                pass

        time.sleep(0.033)  # ~30 FPS update rate


def launch_viewer_with_trajectory(
    trajectory: DroneTrajectory,
    cameras: Sequence[CameraModel],
    voxelizer_result: Optional[VoxelizerResult] = None,
) -> None:
    """
    Simplified viewer launcher (matches existing API).
    """
    launch_interactive_localizer(trajectory, cameras, voxelizer_result)
