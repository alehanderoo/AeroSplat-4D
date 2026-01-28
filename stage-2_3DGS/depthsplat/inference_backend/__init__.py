"""
DepthSplat Inference Backend.

A standalone inference service for 3D Gaussian Splatting reconstruction
from multi-view images. Designed to be used as a backend for various
frontends (Gradio, REST API, CLI, etc.).

Example usage:
    from inference_backend import InferenceService, RenderSettings, VideoSettings

    # Initialize service
    service = InferenceService.from_checkpoint("/path/to/checkpoint.ckpt")

    # Run inference
    result = service.reconstruct(
        images=[img1, img2, img3, img4, img5],
        extrinsics=camera_extrinsics,  # [5, 4, 4] camera-to-world
        intrinsics=camera_intrinsics,  # [5, 3, 3] normalized
        render_settings=RenderSettings(azimuth=45, elevation=30, distance=1.0),
        video_settings=VideoSettings(enabled=True, num_frames=60),
    )

    # Access outputs
    print(f"PLY saved to: {result.ply_path}")
    print(f"Rendered image: {result.rendered_image_path}")
    print(f"Video: {result.video_rgb_path}")
"""

# Core service
from .service import InferenceService

# Type definitions
from .types import (
    CameraParameters,
    InputContext,
    RenderSettings,
    VideoSettings,
    InferenceResult,
    DepthAnalysisResult,
)

# Configuration
from .config import (
    ServiceConfig,
    ModelConfig,
    InferenceConfig,
    get_default_config,
    get_default_checkpoint_path,
)

# Camera utilities
from .camera_utils import (
    create_intrinsics_from_fov,
    create_intrinsics_from_focal,
    create_target_camera,
    create_orbit_cameras,
    create_360_video_cameras,
    look_at,
    normalize_intrinsics,
    denormalize_intrinsics,
    blender_to_opencv,
    opengl_to_opencv,
    compute_camera_elevation,
    compute_mean_camera_elevation,
)

# Depth analysis utilities
from .depth_analysis import (
    compute_depth_metrics,
    scale_and_shift_pred,
    apply_colormap,
    normalize_depth_for_display,
    format_metrics_for_display,
)

# Data loaders
from .data_loader import (
    get_random_example,
    get_example_by_uuid,
    list_available_examples,
    save_example_images,
    HARDCODED_UUIDS,
)

# Wild frame loading
from .wild_frame_loader import (
    WildFrameLoader,
    load_wild_frame,
)

__all__ = [
    # Core
    "InferenceService",
    # Types
    "CameraParameters",
    "InputContext",
    "RenderSettings",
    "VideoSettings",
    "InferenceResult",
    "DepthAnalysisResult",
    # Config
    "ServiceConfig",
    "ModelConfig",
    "InferenceConfig",
    "get_default_config",
    "get_default_checkpoint_path",
    # Camera utils
    "create_intrinsics_from_fov",
    "create_intrinsics_from_focal",
    "create_target_camera",
    "create_orbit_cameras",
    "create_360_video_cameras",
    "look_at",
    "normalize_intrinsics",
    "denormalize_intrinsics",
    "blender_to_opencv",
    "opengl_to_opencv",
    "compute_camera_elevation",
    "compute_mean_camera_elevation",
    # Depth analysis
    "compute_depth_metrics",
    "scale_and_shift_pred",
    "apply_colormap",
    "normalize_depth_for_display",
    "format_metrics_for_display",
    # Data loaders
    "get_random_example",
    "get_example_by_uuid",
    "list_available_examples",
    "save_example_images",
    "HARDCODED_UUIDS",
    # Wild frame loading
    "WildFrameLoader",
    "load_wild_frame",
]

__version__ = "1.0.0"
