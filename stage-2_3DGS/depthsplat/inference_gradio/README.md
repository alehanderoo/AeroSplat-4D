# DepthSplat Gradio Demo

A Gradio web interface for the DepthSplat object-centric 3D Gaussian Splatting model. Upload images of an object from different viewpoints to generate a 3D Gaussian Splat representation and render novel views.

## Quick Start

```bash
# Activate the depthsplat environment
conda activate depthsplat

# Run the demo
python app.py \
    --checkpoint /home/sandro/aeroSplat-4D/stage-2_3DGS/depthsplat/outputs/objaverse_white_small_gauss/checkpoints/epoch_0-step_100000.ckpt \
    --config objaverse_white_small_gauss \
    --port 7860
```

The demo will be available at `http://localhost:7860`.

## Usage

### Command Line Arguments

```bash
python app.py \
    --checkpoint /home/sandro/aeroSplat-4D/stage-2_3DGS/depthsplat/outputs/objaverse_white_small_gauss/checkpoints/epoch_0-step_100000.ckpt \
    --config objaverse_white_small_gauss \
    --port 7860 \
    --share
```

### Input Requirements

1. **Images**: Upload 5 images of an object from different viewpoints
   - Best results with white/clean backgrounds
   - Object should be centered in each image
   - Images will be resized to 256x256

2. **Camera Settings**:
   - **Default orbital cameras**: Assumes images are taken from viewpoints orbiting around the object at equal angular intervals
   - **Camera distance**: Distance from object center (default: 2.0)
   - **Camera elevation**: Height offset from ground plane (default: 0.3)
   - **Field of view**: Camera FOV in degrees (default: 50)

3. **Target View**:
   - **Azimuth**: Horizontal angle for novel view (0-360 degrees)
   - **Elevation**: Height offset for novel view

### Outputs

- **Rendered Image**: Novel view synthesis from the target viewpoint
- **360 Video**: Rotating video around the reconstructed object
- **PLY File**: 3D Gaussian Splat model downloadable for viewing in external tools

## File Structure

```
inference_gradio/
├── app.py           # Main Gradio application
├── runner.py        # Model loading and inference logic
├── camera_utils.py  # Camera transformation utilities
├── requirements.txt # Python dependencies
└── README.md        # This file
```


## Camera Conventions

- **Coordinate System**: OpenCV convention (+X right, +Y down, +Z forward)
- **Extrinsics**: Camera-to-world transformation matrices (4x4)
- **Intrinsics**: Normalized (focal length and principal point in [0, 1] range)

