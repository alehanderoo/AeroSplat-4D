pixel2voxel — CUDA overlap voxelizer & drone localizer
======================================================

This is a refreshed implementation of the original PixelToVoxel projector with
the following upgrades:

- CUDA-accelerated voxel visibility evaluation (RTX 5090 / CUDA 12.8 ready).
- **NEW: Drone localization from multi-camera frame differences**.
- **NEW: Interactive 3D viewer with temporal animation and GUI controls**.
- Direct support for the rendered capture located at
  `~/thesis/renders/5cams_26-10-25`.
- Camera intrinsics/extrinsics read from `drone_camera_observations.json`
  (quaternion-based transforms).
- Shared-view voxel grid derived from the overlapping camera frustums.
- Interactive web visualisation powered by [viser](https://github.com/nerfstudio-project/viser).


Quick start
-----------

### Occupancy Grid Mode (Original)

Build the voxel occupancy grid from camera frustums:

```bash
conda activate p2v  # or ensure torch, numpy, viser, imageio are available
python -m src occupancy \
  --dataset-root ~/thesis/renders/5cams_26-10-25 \
  --resolution 192 \
  --min-cameras 4
```

### Drone Localization Mode (NEW)

Localize the drone from frame-to-frame differences:

```bash
python -m src localize \
  --dataset-root ~/thesis/renders/5cams_26-10-25 \
  --resolution 128 \
  --start-frame 0 \
  --num-frames 50 \
  --motion-threshold 30.0
```

Once processing finishes, an interactive Viser server link is printed. Open it
in your browser to:
- Scrub through time with the frame slider
- Play/pause animation at different speeds
- Toggle trajectory, cameras, frustums, and accumulation grids
- Compare with ground truth drone positions

Key options
-----------

### Common Options

- `--dataset-root`: Root directory containing cam_0X folders and metadata JSON
- `--resolution`: target voxel count along the longest scene axis (default 160)
- `--margin`: padding (metres) around the drone trajectory bounding box
- `--min-cameras`: minimum number of cameras that must see a voxel
- `--frame-stride`: stride for sampling metadata frames when estimating near/far
- `--chunk-size`: voxels processed per CUDA batch (tune for memory)
- `--cuda/--cpu`: force device selection (auto-detect by default)

### Localization-Specific Options

- `--start-frame`: Starting frame index (default 0)
- `--num-frames`: Number of frames to process (None = all available)
- `--motion-threshold`: Pixel intensity difference threshold (default 30.0)
- `--store-grids`: Store full accumulation grids for visualization (memory intensive)
- `--no-occupancy`: Skip building occupancy grid for faster startup


Implementation notes
--------------------

### Occupancy Grid

- Voxel volume bounds automatically derive from the recorded drone positions.
- Camera depth limits are heuristically estimated from per-frame measurements in
  the metadata; adjust `min_cameras` or `margin` if you need a thicker volume.
- RGB/colour sampling hooks are in place for future work.
- All heavy lifting happens inside `pixel2voxel/voxelizer.py` using PyTorch.

### Drone Localization

- Frame differencing detects motion between consecutive frames per camera.
- Changed pixels are unprojected to 3D rays in world space.
- Rays are cast through a voxel grid using DDA-like traversal.
- Accumulation of ray votes localizes the drone position.
- Inspired by the C++ implementation in `Pixeltovoxelprojector/`.

### Interactive Viewer

- Built with Viser for real-time 3D visualization.
- GUI controls for playback, visualization modes, and parameters.
- Shows drone trajectory, camera positions/frustums, and voxel grids.
- Compares localized positions with ground truth.


Architecture
------------

```
pixel2voxel/
├── app.py              # CLI entry point with occupancy & localize modes
├── cameras.py          # Camera intrinsics/extrinsics loading
├── config.py           # Configuration dataclass
├── dataset.py          # Multi-camera dataset abstraction
├── frame_diff.py       # Frame differencing for motion detection
├── geometry.py         # 3D geometry utilities
├── localizer.py        # Drone localization pipeline
├── ray_caster.py       # Ray casting through voxel grid
├── viewer.py           # Static viewer (original)
├── interactive_viewer.py  # Interactive animated viewer (new)
└── voxelizer.py        # CUDA voxel overlap computation
```

