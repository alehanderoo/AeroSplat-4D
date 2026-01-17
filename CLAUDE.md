# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**aeroSplat-4D** is a multi-stage ML pipeline for classifying flying objects (drones vs birds) using 4D Gaussian Splatting reconstructions from multi-camera networks. This is a thesis implementation combining Isaac Sim simulation, 3D object reconstruction, and deep learning classification.

## Build and Environment Setup

```bash
# Set for RTX 5090
export TORCH_CUDA_ARCH_LIST="9.0"

# Create and activate environment
conda env create -f environment.yml
conda activate aeroSplat
```

**Key dependencies:** Python 3.12, PyTorch 2.9, CUDA 12.8, PyTorch Lightning

## Pipeline Architecture

The system follows a **4-stage pipeline**:

```
IsaacSim Simulation → Stage 1: Pixel2Voxel → Stage 2: DepthSplat → Stage 3: 4D Classification
     (data gen)         (foreground seg)      (3D Gaussians)         (drone/bird)
```

### Stage 1: Foreground Segmentation (`stage-1_fg-seg/`)

CUDA-accelerated voxel grid projection for extracting flying objects from multi-camera video.

```bash
# Build C++ extensions first
cd stage-1_fg-seg && python setup.py build_ext --inplace

# Occupancy grid
python -m src occupancy --dataset-root /path/to/renders --resolution 192 --min-cameras 4

# Drone localization
python -m src localize --dataset-root /path/to/renders --resolution 128 --motion-threshold 30.0
```

**Key files:** `src/voxelizer.py` (CUDA voxel), `src/localizer.py` (trajectory), `src/interactive_viewer.py` (Viser 3D viz)

### Stage 2: 3D Gaussian Splatting (`stage-2_3DGS/depthsplat/`)

Feed-forward 3D Gaussian reconstruction using dual-branch architecture (cost volumes + monocular depth).

```bash
cd stage-2_3DGS/depthsplat

# Download pretrained weights
wget https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/gmflow-scale1-things-e9887eda.pth -P pretrained/
wget https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth -P pretrained/

# Train (Hydra-based config)
python -m src.main +experiment=objaverse wandb.name=run1 trainer.max_steps=10000
```

**Config:** `config/experiment/objaverse.yaml`
**Key modifications from original DepthSplat:** Object-centric depth bounds (0.35-4.21m), smaller Gaussian scales (0.1), 360° view sampling, foreground-masked losses

### Stage 3: 4D Classification (`stage-3_4D-classify/`)

Two-stage classifier: VN-Transformer (rotation-invariant spatial encoding) → Mamba (temporal modeling).

```bash
cd stage-3_4D-classify

# Train with synthetic data (testing)
python scripts/train.py --config configs/train_lite.yaml --synthetic

# Train with real data
python scripts/train.py --config configs/train_full.yaml --data-root /path/to/data

# Run tests
python tests/test_pipeline.py
```

**Key files:**
- `src/models/full_model.py` - Gaussian4DClassifier main model
- `src/models/stage1_spatial/vn_transformer.py` - VN-Transformer (rotation invariance via Frobenius inner products)
- `src/models/stage2_temporal/mamba_temporal.py` - Mamba encoder (O(T) complexity)
- `src/data/ply_parser.py` - PLY → GaussianCloud conversion

**PLY format expected:** position (x,y,z), scale_0/1/2, rot_0/1/2/3 (quaternion wxyz), opacity, f_dc_0/1/2 (SH DC), f_rest_* (higher-order SH)

### Isaac Sim Simulation (`isaacsim/`)

Generate synthetic training data with ground truth.

```bash
# Edit isaacsim/config.yaml for scene/camera/rendering settings
# Run in Isaac Sim GUI Script Editor:
exec(open("/home/sandro/aeroSplat-4D/isaacsim/runner.py").read())
run_workflow()
```

**Output:** `/renders/5cams_{object}_{distance}/` containing RGB, depth, masks per camera + metadata JSON

## Data Locations

- **Synthetic renders:** `/renders/5cams_drone_10m/`, `5cams_drone_50m/`, `5cams_drone_100m/`, `5cams_bird_10m/`, etc.
- **3D assets:** `/assets/drones/`, `/assets/birds/` (excluded from git)
- **Pretrained models:** `stage-2_3DGS/depthsplat/pretrained/`

## Configuration Files

- **Root environment:** `environment.yml`
- **IsaacSim:** `isaacsim/config.yaml` (centralized scene, camera, rendering settings)
- **Stage 2:** `stage-2_3DGS/depthsplat/config/experiment/objaverse.yaml` (Hydra)
- **Stage 3:** `stage-3_4D-classify/configs/train_full.yaml`, `train_lite.yaml` (YAML)

## Key Design Decisions

1. **VN-Transformer spatial encoding:** Achieves rotation invariance via Frobenius inner products - critical since drones/birds appear at any orientation
2. **Mamba temporal modeling:** O(T) complexity vs O(T²) for Transformers - essential for variable-length sequences
3. **Attention pooling:** Handles varying Gaussian counts and provides permutation invariance
4. **Object-centric DepthSplat adaptations:** Smaller Gaussian scales, 360° view sampling, foreground-masked losses, silhouette loss to prevent floating Gaussians

## Thesis Documentation

- **Methodology:** `/report/paper/05_method.tex`
- **Architecture selection rationale:** `/report/paper/09_appendix.tex`
- **Per-stage docs:** `stage-1_fg-seg/src/readme.md`, `stage-2_3DGS/depthsplat/README_thesis.md`, `stage-3_4D-classify/README.md`

## Known Issues

- **Stage 2 inference:** Does not work in main aeroSplat env (works in separate depthsplat env) - possible dependency conflict
- **Pipeline integration:** Stages not yet fully automated end-to-end
