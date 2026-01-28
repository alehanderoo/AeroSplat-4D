# DepthSplat Inference Backend

A standalone inference service for 3D Gaussian Splatting reconstruction from multi-view images. This backend provides a clean, reusable API for 3D reconstruction that can be used with various frontends (Gradio, REST API, CLI, etc.).

## Features

- **3D Gaussian Splatting Reconstruction**: Convert multi-view images to 3D Gaussian Splat models
- **Novel View Rendering**: Render the reconstructed scene from any camera viewpoint
- **360° Video Generation**: Generate rotating videos of the reconstructed object
- **PLY Export**: Export 3D Gaussian models in standard PLY format
- **Depth Analysis**: Compare different depth estimation methods (monocular vs multi-view)
- **Wild Frame Support**: Process in-the-wild frames from Isaac Sim with automatic pose normalization

## Installation

The inference backend is part of the DepthSplat project. Ensure you have the main DepthSplat environment set up:

```bash
cd stage-2_3DGS/depthsplat
# Activate your environment (conda/venv)
```

## Quick Start

### Basic Usage

```python
from inference_backend import InferenceService, RenderSettings, VideoSettings
import numpy as np

# Initialize the service
service = InferenceService.from_checkpoint(
    checkpoint_path="/path/to/checkpoint.ckpt",
    config_name="objaverse_white",
    device="cuda",
)

# Prepare inputs
images = [np.array(img) for img in your_images]  # List of [H, W, 3] uint8 arrays
extrinsics = your_camera_extrinsics  # [V, 4, 4] camera-to-world matrices
intrinsics = your_camera_intrinsics  # [V, 3, 3] normalized intrinsics

# Run reconstruction
result = service.reconstruct(
    images=images,
    extrinsics=extrinsics,
    intrinsics=intrinsics,
    render_settings=RenderSettings(azimuth=45, elevation=30, distance=1.0),
    video_settings=VideoSettings(enabled=True, num_frames=60),
)

# Access outputs
print(f"PLY file: {result.ply_path}")
print(f"Rendered image: {result.rendered_image_path}")
print(f"RGB video: {result.video_rgb_path}")
```

### Loading Objaverse Examples

```python
from inference_backend import (
    InferenceService,
    get_example_by_uuid,
    save_example_images,
    HARDCODED_UUIDS,
)

# Load a specific example
example = get_example_by_uuid(
    data_dir="/path/to/objaverse/test",
    uuid=HARDCODED_UUIDS[0],
    num_context_views=5,
)

# Run inference
result = service.reconstruct(
    images=example['images'],
    extrinsics=example['extrinsics'],
    intrinsics=example['intrinsics'],
)
```

### Loading Wild Frames (Isaac Sim)

```python
from inference_backend import WildFrameLoader, InferenceService

# Load wild frame with automatic pose normalization
loader = WildFrameLoader(
    render_dir="/path/to/renders/5cams_drone_50m",
    use_virtual_cameras=True,
)

frame_data = loader.load_frame(frame_id=60)

# Load images
images = loader.load_images(frame_data['image_paths'])

# Run inference
result = service.reconstruct(
    images=images,
    extrinsics=frame_data['extrinsics'],
    intrinsics=frame_data['intrinsics'],
)
```

## API Reference

### InferenceService

The main service class for 3D reconstruction.

```python
class InferenceService:
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        config_name: str = "objaverse_white",
        device: str = "cuda",
    ) -> "InferenceService":
        """Create service from checkpoint path."""

    def reconstruct(
        self,
        images: List[np.ndarray],
        extrinsics: np.ndarray,
        intrinsics: np.ndarray,
        render_settings: RenderSettings = None,
        video_settings: VideoSettings = None,
        output_dir: str = None,
    ) -> InferenceResult:
        """Run 3D Gaussian Splatting reconstruction."""
```

### RenderSettings

Settings for novel view rendering.

```python
@dataclass
class RenderSettings:
    azimuth: float = 0.0      # Horizontal angle (0-360 degrees)
    elevation: float = 30.0   # Vertical angle (-90 to 90 degrees)
    distance: float = 1.0     # Distance factor (0.6-1.4)
```

### VideoSettings

Settings for 360° video generation.

```python
@dataclass
class VideoSettings:
    enabled: bool = True
    num_frames: int = 60
    fps: int = 30
    elevation: float = 30.0
    distance: float = 1.0
    include_depth: bool = True
    include_silhouette: bool = True
```

### InferenceResult

Result of reconstruction containing paths to all outputs.

```python
@dataclass
class InferenceResult:
    ply_path: Optional[str]              # Path to PLY file
    rendered_image_path: Optional[str]   # Path to rendered image
    depth_image_path: Optional[str]      # Path to depth visualization
    silhouette_image_path: Optional[str] # Path to silhouette
    video_rgb_path: Optional[str]        # Path to RGB video
    video_depth_path: Optional[str]      # Path to depth video
    video_silhouette_path: Optional[str] # Path to silhouette video
    output_dir: Optional[str]            # Output directory
    depth_analysis: Optional[Dict]       # Depth analysis results
    metadata: Dict                       # Additional metadata
```

## Camera Conventions

### Coordinate System
The backend uses **OpenCV camera convention**:
- +X: Right
- +Y: Down
- +Z: Forward (optical axis)

### Extrinsics
Camera-to-world transformation matrices `[4, 4]`:
- The rotation matrix `R = c2w[:3, :3]` transforms camera directions to world
- The translation `t = c2w[:3, 3]` is the camera position in world coordinates

### Intrinsics
Normalized intrinsics matrices `[3, 3]` with values in `[0, 1]`:
```
K = [[fx, 0, cx],
     [0, fy, cy],
     [0,  0,  1]]
```
Where `fx`, `fy`, `cx`, `cy` are normalized by image width/height.

## Camera Utilities

```python
from inference_backend import (
    create_target_camera,      # Create camera from azimuth/elevation/distance
    create_360_video_cameras,  # Create cameras for 360° video
    create_intrinsics_from_fov, # Create intrinsics from FOV
    normalize_intrinsics,      # Normalize pixel intrinsics to [0,1]
    look_at,                   # Create camera looking at target
)

# Create a target camera
c2w = create_target_camera(
    azimuth=90.0,      # degrees
    elevation=30.0,    # degrees
    distance=1.0,      # factor
    base_radius=2.0,   # meters
)

# Create video trajectory
video_cameras = create_360_video_cameras(
    num_frames=60,
    radius=2.0,
    elevation_angle=30.0,
)
```

## Directory Structure

```
inference_backend/
├── __init__.py          # Public API exports
├── service.py           # Main InferenceService class
├── types.py             # Type definitions (RenderSettings, etc.)
├── config.py            # Configuration classes
├── camera_utils.py      # Camera transformation utilities
├── depth_analysis.py    # Depth analysis utilities
├── data_loader.py       # Objaverse data loading
├── wild_frame_loader.py # Isaac Sim frame loading
├── services/            # Detection services
│   ├── detection_service.py
│   └── gt_detection_service.py
└── README.md            # This file
```

## Integration Example: Gradio Frontend

See `../inference_gradio/app.py` for a complete example of integrating this backend with a Gradio web interface.

```python
from inference_backend import InferenceService, RenderSettings, VideoSettings
import gradio as gr

# Initialize once
service = InferenceService.from_checkpoint("path/to/checkpoint.ckpt")

def process_images(images, azimuth, elevation, distance):
    result = service.reconstruct(
        images=images,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        render_settings=RenderSettings(azimuth, elevation, distance),
    )
    return result.rendered_image_path, result.ply_path

# Build Gradio interface using the backend...
```

## License

This code is part of the aeroSplat-4D thesis project.
