"""
Drone localization from multi-camera frame differences.

This module processes temporal sequences of frames from multiple cameras,
detects motion, casts rays through a voxel grid, and localizes the drone.
"""

from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .cameras import CameraModel
from .config import VoxelizerConfig
from .dataset import MultiCameraDataset
from .frame_diff import FrameDifferencer
from .ray_caster import RayCaster, localize_from_accumulation


def _process_single_frame(args: Tuple) -> Optional[Dict]:
    """
    Process a single frame for localization (module-level for pickling).
    
    Args:
        args: Tuple of (frame_idx, start_frame, dataset_root, metadata_json, 
              camera_names, grid_shape, voxel_size, grid_min, motion_threshold, save_diffs, store_rays)
    
    Returns:
        Dict with frame data including position, confidence, and optionally rays, or None
    """
    (frame_idx, start_frame, dataset_root, metadata_json, camera_names, 
     grid_shape, voxel_size, grid_min, motion_threshold, save_diffs, store_rays) = args
    
    # Recreate necessary objects in worker process
    from .config import VoxelizerConfig
    from .dataset import MultiCameraDataset
    from .frame_diff import FrameDifferencer
    from .ray_caster import RayCaster
    import numpy as np
    
    # Create minimal config for dataset loading
    config = VoxelizerConfig(
        dataset_root=dataset_root,
        metadata_json=metadata_json,
        resolution=160,  # dummy value
        margin_meters=0.5,  # dummy value
    )
    dataset = MultiCameraDataset(config)
    
    frame_differencer = FrameDifferencer(
        threshold=motion_threshold,
        blur_kernel_size=5,
        min_change_area=10,
    )
    
    ray_caster = RayCaster(
        grid_shape=grid_shape,
        voxel_size=voxel_size,
        grid_min=grid_min,
    )
    
    num_cameras_detected = 0
    accumulation_grid = None  # Will be updated by ray_caster
    
    # Collect rays for visualization
    all_ray_origins = []
    all_ray_dirs = []
    all_weights = []
    
    # Process each camera
    for cam_name in camera_names:
        try:
            curr_rgb = dataset.load_rgb(cam_name, frame_idx)
            
            if frame_idx == start_frame:
                continue
            
            prev_rgb = dataset.load_rgb(cam_name, frame_idx - 1)
            
            # Compute frame difference
            save_path = None
            if save_diffs is not None:
                save_path = f"{save_diffs}/frame_{frame_idx:04d}_{cam_name}"
            
            diff_map, changed_mask = frame_differencer.compute_difference(
                curr_rgb, prev_rgb, save_path=save_path
            )
            
            changed_coords = frame_differencer.get_changed_pixel_coords(changed_mask)
            
            if changed_coords.shape[0] == 0:
                continue
            
            num_cameras_detected += 1
            
            # Get camera model
            cam_model = next(c for c in dataset.camera_models if c.name == cam_name)
            
            # Cast rays
            ray_origins, ray_dirs = ray_caster.unproject_pixels_to_rays(
                changed_coords, cam_model
            )
            
            weights = diff_map[changed_coords[:, 1], changed_coords[:, 0]]
            
            # Print diagnostic info
            if save_diffs is not None:
                print(f"  Frame {frame_idx} {cam_name}: {changed_coords.shape[0]} changed pixels, "
                      f"weight range [{weights.min():.1f}, {weights.max():.1f}]")
            
            # Store rays for visualization (controlled by store_rays parameter)
            if store_rays:
                all_ray_origins.append(ray_origins)
                all_ray_dirs.append(ray_dirs)
                all_weights.append(weights)
            
            # Accumulate (grid managed by C++ backend)
            accumulation_grid = ray_caster.cast_rays_and_accumulate(
                ray_origins,
                ray_dirs,
                weights,
                max_distance=50.0,
            )
            
        except FileNotFoundError:
            break
        except Exception:
            continue
    
    if num_cameras_detected == 0:
        return None
    
    # Get final accumulation grid from C++ backend
    if accumulation_grid is None:
        accumulation_grid = ray_caster.cpp_grid.data().reshape(grid_shape)

    
    # Localize
    position = localize_from_accumulation(
        accumulation_grid,
        grid_min,
        voxel_size,
        method="max",
    )
    
    confidence = float(accumulation_grid.max())
    
    # Prepare return data
    result = {
        'frame_idx': frame_idx,
        'position': position,
        'confidence': confidence,
        'num_cameras_detected': num_cameras_detected,
    }
    
    # Include ray data if requested
    if store_rays and len(all_ray_origins) > 0:
        result['ray_origins'] = np.vstack(all_ray_origins)
        result['ray_dirs'] = np.vstack(all_ray_dirs)
        result['ray_weights'] = np.concatenate(all_weights)
        if save_diffs is not None:
            print(f"  Frame {frame_idx}: Stored {len(result['ray_origins'])} rays for visualization")
    
    return result


@dataclass
class DroneLocalization:
    """Single frame drone localization result."""
    frame_index: int
    position: np.ndarray  # (3,) world position
    confidence: float  # Sum of ray votes
    num_cameras_detected: int  # Number of cameras that detected motion
    accumulation_grid: Optional[np.ndarray] = None  # Optional: full grid for visualization
    ray_origins: Optional[np.ndarray] = None  # (N, 3) ray start points
    ray_dirs: Optional[np.ndarray] = None  # (N, 3) ray directions
    ray_weights: Optional[np.ndarray] = None  # (N,) ray weights


@dataclass
class DroneTrajectory:
    """Complete drone trajectory over time."""
    localizations: List[DroneLocalization]
    ground_truth_positions: Optional[np.ndarray] = None  # (N, 3) if available
    
    def get_positions(self) -> np.ndarray:
        """Get all localized positions as (N, 3) array."""
        return np.array([loc.position for loc in self.localizations])
    
    def get_confidences(self) -> np.ndarray:
        """Get confidence values for each localization."""
        return np.array([loc.confidence for loc in self.localizations])


class DroneLocalizer:
    """
    Localize a moving drone from multi-camera video streams.
    """

    def __init__(
        self,
        dataset: MultiCameraDataset,
        config: VoxelizerConfig,
        motion_threshold: float = 30.0,
        accumulation_decay: float = 0.9,
        device = None,
    ) -> None:
        """
        Initialize drone localizer.

        Args:
            dataset: Multi-camera dataset
            config: Voxelizer configuration
            motion_threshold: Pixel difference threshold for motion detection
            accumulation_decay: Decay factor for temporal accumulation (0-1)
            device: Ignored, kept for API compatibility
        """
        self.dataset = dataset
        self.config = config
        self.motion_threshold = motion_threshold
        self.accumulation_decay = accumulation_decay

        # Initialize components
        self.frame_differencer = FrameDifferencer(
            threshold=motion_threshold,
            blur_kernel_size=5,
            min_change_area=10,
        )

        # Get voxel grid parameters from dataset
        bbox = dataset.bounding_box(config.margin_meters)
        self.grid_min = bbox[0]
        extents = bbox[1] - bbox[0]
        max_extent = float(extents.max())
        
        resolution = int(config.resolution)
        self.voxel_size = max_extent / float(resolution - 1)
        
        self.grid_shape = tuple(
            int(np.ceil(extent / self.voxel_size)) + 1
            for extent in extents
        )

        self.ray_caster = RayCaster(
            grid_shape=self.grid_shape,
            voxel_size=self.voxel_size,
            grid_min=self.grid_min,
        )

        print(f"\n=== Localizer Initialized ===")
        print(f"Grid shape: {self.grid_shape}")
        print(f"Voxel size: {self.voxel_size:.4f}m")
        print(f"Grid bounds: {self.grid_min} to {self.grid_min + np.array(self.grid_shape) * self.voxel_size}")
        print(f"Motion threshold: {motion_threshold}")

    def localize_sequence(
        self,
        start_frame: int = 0,
        num_frames: Optional[int] = None,
        store_grids: bool = False,
        num_workers: int = 24,
        save_diffs: Optional[str] = None,
        store_rays: bool = True,
    ) -> DroneTrajectory:
        """
        Localize drone across a sequence of frames using parallel processing.

        Args:
            start_frame: Starting frame index
            num_frames: Number of frames to process (None = all available)
            store_grids: Whether to store full accumulation grids (memory intensive)
            num_workers: Number of parallel workers (default: 24)
            save_diffs: Optional directory to save frame difference images
            store_rays: Whether to store ray data for visualization (default: True)

        Returns:
            DroneTrajectory with localization results
        """
        cameras = self.dataset.camera_models
        camera_names = [cam.name for cam in cameras]

        # Determine frame range
        max_frame = start_frame + (num_frames if num_frames is not None else 1000)
        frame_indices = list(range(start_frame, max_frame))

        print(f"\n=== Processing {len(frame_indices)} frames in parallel with {num_workers} workers ===")

        # Prepare args for parallel processing
        args_list = [
            (
                frame_idx,
                start_frame,
                self.config.dataset_root,
                self.config.metadata_json,
                camera_names,
                self.grid_shape,
                self.voxel_size,
                self.grid_min,
                self.motion_threshold,
                save_diffs,
                store_rays,
            )
            for frame_idx in frame_indices
        ]

        localizations: List[DroneLocalization] = []

        # Process frames in parallel using 'spawn' method (required for CUDA)
        mp_context = mp.get_context('spawn')
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=mp_context) as executor:
            futures = {executor.submit(_process_single_frame, args): args[0] for args in args_list}
            
            for future in as_completed(futures):
                frame_idx = futures[future]
                try:
                    result = future.result()
                    if result is not None:
                        localization = DroneLocalization(
                            frame_index=result['frame_idx'],
                            position=result['position'],
                            confidence=result['confidence'],
                            num_cameras_detected=result['num_cameras_detected'],
                            accumulation_grid=None,
                            ray_origins=result.get('ray_origins'),
                            ray_dirs=result.get('ray_dirs'),
                            ray_weights=result.get('ray_weights'),
                        )
                        localizations.append(localization)
                        
                        # Debug: check if rays were stored
                        ray_info = ""
                        if localization.ray_origins is not None:
                            ray_info = f", rays={len(localization.ray_origins)}"
                        
                        print(f"Frame {result['frame_idx']}: position={result['position']}, confidence={result['confidence']:.2f}, cameras={result['num_cameras_detected']}{ray_info}")
                    else:
                        print(f"Frame {frame_idx}: No motion detected")
                except Exception as e:
                    print(f"Frame {frame_idx}: Error - {e}")

        # Sort by frame index
        localizations.sort(key=lambda loc: loc.frame_index)

        # Create trajectory
        trajectory = DroneTrajectory(localizations=localizations)

        # Try to load ground truth if available
        try:
            trajectory.ground_truth_positions = self.dataset.drone_positions
        except Exception:
            pass

        print(f"\n=== Localization Complete ===")
        print(f"Successfully processed {len(localizations)} frames")

        return trajectory

    def localize_single_frame(
        self,
        frame_idx: int,
        prev_frame_idx: Optional[int] = None,
    ) -> Optional[DroneLocalization]:
        """
        Localize drone in a single frame pair.

        Args:
            frame_idx: Current frame index
            prev_frame_idx: Previous frame index (frame_idx - 1 if None)

        Returns:
            DroneLocalization or None if no motion detected
        """
        if prev_frame_idx is None:
            prev_frame_idx = frame_idx - 1

        cameras = self.dataset.camera_models

        num_cameras_detected = 0
        accumulation_grid = None

        # Process each camera
        for cam_model in cameras:
            try:
                curr_rgb = self.dataset.load_rgb(cam_model.name, frame_idx)
                prev_rgb = self.dataset.load_rgb(cam_model.name, prev_frame_idx)

                # Compute frame difference
                diff_map, changed_mask = self.frame_differencer.compute_difference(
                    curr_rgb, prev_rgb
                )

                # Get changed pixel coordinates
                changed_coords = self.frame_differencer.get_changed_pixel_coords(changed_mask)

                if changed_coords.shape[0] == 0:
                    continue

                num_cameras_detected += 1

                # Cast rays
                ray_origins, ray_dirs = self.ray_caster.unproject_pixels_to_rays(
                    changed_coords, cam_model
                )

                weights = diff_map[changed_coords[:, 1], changed_coords[:, 0]]

                # Accumulate (grid managed by C++ backend)
                accumulation_grid = self.ray_caster.cast_rays_and_accumulate(
                    ray_origins,
                    ray_dirs,
                    weights,
                    max_distance=50.0,
                )

            except Exception as e:
                print(f"Error processing {cam_model.name}: {e}")
                continue

        if num_cameras_detected == 0:
            return None

        # Get final accumulation grid from C++ backend
        if accumulation_grid is None:
            accumulation_grid = self.ray_caster.cpp_grid.data().reshape(self.grid_shape)

        # Localize
        position = localize_from_accumulation(
            accumulation_grid,
            self.grid_min,
            self.voxel_size,
            method="max",
        )

        confidence = float(accumulation_grid.max())

        return DroneLocalization(
            frame_index=frame_idx,
            position=position,
            confidence=confidence,
            num_cameras_detected=num_cameras_detected,
            accumulation_grid=accumulation_grid,
        )
