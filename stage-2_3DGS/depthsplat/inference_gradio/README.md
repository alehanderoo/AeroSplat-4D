# DepthSplat Gradio Demo

A Gradio web interface for the DepthSplat object-centric 3D Gaussian Splatting model. This is a frontend application that uses the `inference_backend` service for all inference operations.

## Architecture

```
inference_gradio/          <- This directory (Gradio frontend)
    └── app.py             <- Web interface, uses inference_backend

inference_backend/         <- Standalone backend service
    ├── service.py         <- Core InferenceService
    ├── types.py           <- RenderSettings, VideoSettings, etc.
    ├── camera_utils.py    <- Camera transformations
    └── ...
```

## Quick Start

```bash
# Activate the depthsplat environment
conda activate depthsplat

# Run the demo
python app.py \
    --checkpoint /path/to/checkpoint.ckpt \
    --config objaverse_white \
    --port 7860
```

The demo will be available at `http://localhost:7860`.

## Usage

### Command Line Arguments

```bash
python app.py \
    --checkpoint /path/to/checkpoint.ckpt \
    --config objaverse_white \
    --port 7860 \
    --share  # Optional: create public link
```

### Features

1. **Load Examples from Objaverse Test Set**
   - Select specific UUIDs for consistent testing
   - Load random examples

2. **Load In-the-Wild Frames (Isaac Sim)**
   - Process frames 0-119 from simulation renders
   - Automatic pose normalization and FOV matching

3. **Novel View Rendering**
   - Control azimuth, elevation, and distance
   - Real-time camera parameter preview

4. **360° Video Generation**
   - RGB, depth, and silhouette videos
   - Configurable frame count

5. **Flight Tracking Mode**
   - Process all 120 frames with rotating viewpoint
   - Generate continuous tracking video

6. **Depth Analysis**
   - Compare standalone Depth Anything V2, coarse MV, DPT residual, and final fused depth
   - Metrics comparison with ground truth (for Isaac Sim frames)

7. **PLY Export**
   - Download 3D Gaussian Splat models

### Outputs

- **Rendered Image**: Novel view synthesis from target viewpoint
- **Depth Map**: Rendered depth visualization
- **Silhouette**: Object alpha/mask
- **360 Videos**: RGB, depth, silhouette rotation videos
- **PLY File**: 3D Gaussian Splat model

## File Structure

```
inference_gradio/
├── app.py              # Main Gradio application (uses inference_backend)
├── README.md           # This file
├── requirements.txt    # Python dependencies
└── debugging/          # Development/debugging scripts
```

## Backend Service

All inference logic has been abstracted into `inference_backend/`. See `../inference_backend/README.md` for details on:

- `InferenceService` - Main reconstruction API
- `RenderSettings` - Novel view parameters
- `VideoSettings` - Video generation settings
- `WildFrameLoader` - Isaac Sim frame loading
- Camera utilities and depth analysis tools

## Camera Conventions

- **Coordinate System**: OpenCV convention (+X right, +Y down, +Z forward)
- **Extrinsics**: Camera-to-world transformation matrices (4x4)
- **Intrinsics**: Normalized (focal length and principal point in [0, 1] range)
