# DepthSplat Real-Time Inference Pipeline

Real-time inference pipeline for DepthSplat 3D Gaussian Splatting with multi-camera RTSP support.

## Overview

| Phase | Description | Status |
|-------|-------------|--------|
| **Phase 1** | RTSP Stream Simulation | ✅ Complete |
| **Phase 2** | Multi-Camera Pipeline | ✅ Complete |
| **Phase 3** | PyTorch Model Integration | ✅ Complete |
| **Phase 4** | End-to-End Integration | ✅ Complete |

## Quick Start

### Install Dependencies

```bash
# Python dependencies (from depthsplat conda env)
conda activate depthsplat
pip install -r requirements.txt

# GStreamer RTSP support
sudo apt install gstreamer1.0-rtsp gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
    gir1.2-gst-rtsp-server-1.0 python3-gi
```

### 1. Start RTSP Simulator (Terminal 1)

```bash
./scripts/start_rtsp_simulator.sh
```

### 2. Run Inference Pipeline (Terminal 2)

```bash
python main.py --config config/pipeline_config.yaml --mode dev
```

Or use the helper script:

```bash
./scripts/run_pipeline.sh --mode dev
```

### 3. Test Streams (Optional)

```bash
ffplay rtsp://192.168.3.166:8554/cam_01
```

---

## Phase 1: RTSP Stream Simulation

Simulates RTSP streams from pre-rendered IsaacSim frames.

### Features

- 5 synchronized camera streams
- GStreamer backend
- Frame-accurate playback control

### Stream URLs

| Camera | URL |
|--------|-----|
| cam_01 | `rtsp://localhost:8554/cam_01` |
| cam_02 | `rtsp://localhost:8554/cam_02` |
| cam_03 | `rtsp://localhost:8554/cam_03` |
| cam_04 | `rtsp://localhost:8554/cam_04` |
| cam_05 | `rtsp://localhost:8554/cam_05` |

---

## Phase 2: Multi-Camera Pipeline

Multi-camera pipeline using OpenCV and GStreamer for synchronized frame capture.

---

## Phase 3: PyTorch Model Integration

The pipeline loads the DepthSplat model directly from a PyTorch checkpoint using Hydra configuration.

### Configuration

Edit `config/pipeline_config.yaml`:

```yaml
model:
  # PyTorch checkpoint for direct inference
  checkpoint_path: "/path/to/checkpoint.ckpt"

sources:
  rtsp:
    urls:
      - "rtsp://localhost:8554/cam_01"
      - "rtsp://localhost:8554/cam_02"
      # ...
```

### Performance

Current benchmarks (RTX GPU, 5 cameras @ 256x256):

- PyTorch backend: ~11 FPS, ~73ms latency

---

## Directory Structure

```
inference/
├── config/                      # Configuration files
│   └── pipeline_config.yaml     # Main pipeline configuration
├── model/                       # Model wrapper
├── pipeline/                    # Inference pipeline
├── scripts/                     # CLI tools
├── stream_simulator/            # RTSP simulation
├── tests/                       # Unit tests
└── utils/                       # Utility modules
```

---

## Testing

```bash
# All tests
pytest tests/ -v

# Specific modules
pytest tests/test_stream_simulator.py -v
pytest tests/test_pipeline.py -v
pytest tests/test_end_to_end.py -v
```

---

## Troubleshooting

### GStreamer not found

```bash
sudo apt install gstreamer1.0-tools gstreamer1.0-rtsp \
    gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
    gir1.2-gst-rtsp-server-1.0 python3-gi
```

### CUDA not found

Make sure you have the depthsplat conda environment activated:

```bash
conda activate depthsplat
```

---

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  RTSP Streams   │────▶│    OpenCV/      │────▶│     PyTorch     │
│  (5 cameras)    │     │   GStreamer     │     │    Inference    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                                                        ▼
                                              ┌─────────────────┐
                                              │   Gaussian      │
                                              │   Parameters    │
                                              └─────────────────┘
```
