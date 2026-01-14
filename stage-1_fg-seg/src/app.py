from __future__ import annotations

import argparse
from typing import Optional

from .config import VoxelizerConfig
from .dataset import MultiCameraDataset
from .voxelizer import CudaVoxelizer
from .viewer import launch_viewer


def build_and_view(config: VoxelizerConfig) -> None:
    dataset = MultiCameraDataset(config)
    voxelizer = CudaVoxelizer(dataset, config)
    result = voxelizer.build()
    launch_viewer(result, dataset.camera_models)


def localize_and_view(
    config: VoxelizerConfig,
    start_frame: int = 0,
    num_frames: Optional[int] = None,
    motion_threshold: float = 30.0,
    store_grids: bool = False,
    build_occupancy: bool = True,
    save_diffs: Optional[str] = None,
) -> None:
    """
    Run drone localization pipeline and visualize results.
    
    Args:
        config: Voxelizer configuration
        start_frame: Starting frame index
        num_frames: Number of frames to process (None = all)
        motion_threshold: Pixel difference threshold for motion detection
        store_grids: Whether to store full accumulation grids
        build_occupancy: Whether to build occupancy grid first
    """
    from .localizer import DroneLocalizer
    from .interactive_viewer import launch_interactive_localizer
    import os
    
    # Create save directory if specified
    if save_diffs is not None:
        os.makedirs(save_diffs, exist_ok=True)
        print(f"\n=== Saving frame differences to: {save_diffs} ===")
    
    dataset = MultiCameraDataset(config)
    
    # Optionally build occupancy grid first
    voxelizer_result = None
    if build_occupancy:
        print("\n=== Building Occupancy Grid ===")
        voxelizer = CudaVoxelizer(dataset, config)
        voxelizer_result = voxelizer.build()
        print("Occupancy grid built successfully")
    
    # Initialize localizer
    print("\n=== Initializing Drone Localizer ===")
    localizer = DroneLocalizer(
        dataset=dataset,
        config=config,
        motion_threshold=motion_threshold,
    )
    
    # Run localization
    print("\n=== Localizing Drone ===")
    trajectory = localizer.localize_sequence(
        start_frame=start_frame,
        num_frames=num_frames,
        store_grids=store_grids,
        save_diffs=save_diffs,
    )
    
    # Launch interactive viewer
    print("\n=== Launching Interactive Viewer ===")
    launch_interactive_localizer(
        trajectory=trajectory,
        cameras=dataset.camera_models,
        voxelizer_result=voxelizer_result,
    )


def _parse_args(argv: Optional[list[str]] = None):
    parser = argparse.ArgumentParser(
        description="CUDA voxel overlap builder and drone localizer with interactive visualization."
    )
    
    # Add subcommands
    subparsers = parser.add_subparsers(dest="mode", help="Operation mode")
    
    # Occupancy mode (original functionality)
    occupancy_parser = subparsers.add_parser(
        "occupancy",
        help="Build voxel occupancy grid from camera frustums"
    )
    _add_common_args(occupancy_parser)
    
    # Localization mode (new functionality)
    localize_parser = subparsers.add_parser(
        "localize",
        help="Localize drone from frame differences"
    )
    _add_common_args(localize_parser)
    localize_parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Starting frame index for localization",
    )
    localize_parser.add_argument(
        "--num-frames",
        type=int,
        default=None,
        help="Number of frames to process (None = all available)",
    )
    localize_parser.add_argument(
        "--motion-threshold",
        type=float,
        default=30.0,
        help="Pixel intensity difference threshold for motion detection",
    )
    localize_parser.add_argument(
        "--store-grids",
        action="store_true",
        help="Store full accumulation grids (memory intensive)",
    )
    localize_parser.add_argument(
        "--no-occupancy",
        action="store_true",
        help="Skip building occupancy grid (faster startup)",
    )
    localize_parser.add_argument(
        "--save-diffs",
        type=str,
        default=None,
        help="Directory to save frame difference images for debugging",
    )

    args = parser.parse_args(argv)
    
    # Default to occupancy mode if no mode specified
    if args.mode is None:
        args.mode = "occupancy"
        # Re-parse with occupancy mode defaults
        return _parse_args((argv or []) + ["occupancy"])

    use_cuda: Optional[bool]
    if args.cuda:
        use_cuda = True
    elif args.cpu:
        use_cuda = False
    else:
        use_cuda = None

    config = VoxelizerConfig(
        dataset_root=args.dataset_root,
        metadata_json=args.metadata_json,
        resolution=args.resolution,
        margin_meters=args.margin,
        min_cameras=args.min_cameras,
        frame_stride=args.frame_stride,
        chunk_size=args.chunk_size,
        use_cuda=use_cuda,
        color_sample_frame=args.color_frame,
    )
    
    return args, config


def _add_common_args(parser):
    """Add arguments common to all modes."""
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="~/thesis/renders/5cams_26-10-25",
        help="Path containing cam_0X folders and drone_camera_observations.json",
    )
    parser.add_argument(
        "--metadata-json",
        type=str,
        default=None,
        help="Optional explicit metadata JSON path",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=160,
        help="Voxels along longest axis",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.5,
        help="Padding (meters) around drone bounding box",
    )
    parser.add_argument(
        "--min-cameras",
        type=int,
        default=3,
        help="Minimum cameras that must see a voxel",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=4,
        help="Temporal stride when sampling metadata frames",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2_000_000,
        help="Voxel batch size for CUDA evaluation",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU mode (for debugging)",
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        help="Force CUDA usage (error if unavailable)",
    )
    parser.add_argument(
        "--color-frame",
        type=int,
        default=0,
        help="Frame index to sample when computing voxel colours",
    )


def main(argv: Optional[list[str]] = None) -> None:
    args, config = _parse_args(argv)
    
    if args.mode == "occupancy":
        build_and_view(config)
    elif args.mode == "localize":
        localize_and_view(
            config=config,
            start_frame=args.start_frame,
            num_frames=args.num_frames,
            motion_threshold=args.motion_threshold,
            store_grids=args.store_grids,
            build_occupancy=not args.no_occupancy,
            save_diffs=args.save_diffs,
        )
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()

