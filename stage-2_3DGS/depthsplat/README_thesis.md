
# Object-Centric DepthSplat for Drone/Bird Classification

This repository extends [DepthSplat](https://github.com/cvg/depthsplat) for object-centric 3D Gaussian Splatting, designed for reconstructing flying objects (drones and birds) from multi-view camera networks.

**Thesis**: *Multi-Camera 4D Classification of Flying Objects*

## Quick Start

### 1. Prerequisites

```bash
# Activate environment
conda activate depthsplat

# Install OpenEXR for depth loading
pip install openexr

# Download pretrained weights (recommended)
wget https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/gmflow-scale1-things-e9887eda.pth -P pretrained/
wget https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth -P pretrained/
wget https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth -P pretrained/

```

### 2. Convert Blender Renders to Training Format

```bash
python scripts/convert_objaverse_to_depthsplat.py \
    --input-dir ~/.objaverse/renders \
    --output-dir datasets/objaverse \
    --train-split 0.9 \
    --scenes-per-chunk 100
```

**Expected input structure:**
```
renders/
├── object_001/
│   ├── metadata.json    # Camera poses, intrinsics
│   ├── 000.png          # RGB
│   ├── 000_depth.exr    # Depth
│   ├── 000_mask.png     # Segmentation mask
│   └── ...
└── object_002/
    └── ...
```

### 3. Generate evaluation index

```bash
python scripts/generate_objaverse_eval_index.py
```

### 4. Train

```bash
python -m src.main +experiment=objaverse \
    wandb.entity=a-a-f-verdiesen-tu-delft \
    wandb.name=objaverse_run1 \
    trainer.max_steps=100000
```

**Fine-tune from pretrained:**
```bash
python -m src.main +experiment=objaverse \
    checkpointing.pretrained_model=pretrained/depthsplat-gs-base-re10kdl3dv-448x768-randview2-6-f8ddd845.pth \
    optimizer.lr=5.e-5
```

## Changes from Original DepthSplat

| Component | Original (Scene-Level) | Modified (Object-Centric) |
|-----------|------------------------|---------------------------|
| **Depth bounds** | 0.1 - 1000m | 0.35 - 4.21m (data-driven) |
| **Gaussian scale max** | 3.0 | 0.1 |
| **View sampling** | Sequential frames | Farthest-point sampling |
| **Loss** | MSE + LPIPS | MSE + LPIPS + Depth + Silhouette |
| **Camera assumption** | Forward-facing | 360° around object |
| **Inference masks** | None | GT masks for compositing |
| **Softmax temperature** | 1.0 | 2.0 (prevents depth collapse) |
| **Multi-view matching** | 2 views | 5 views (full coverage) |
| **Background handling** | None | Depth regularization to far plane |

### New Files

| File | Purpose |
|------|---------|
| `src/dataset/dataset_objaverse.py` | Dataset loader with mask/depth support |
| `src/dataset/view_sampler/view_sampler_object_centric.py` | 360° view sampling via farthest-point |
| `config/experiment/objaverse.yaml` | Training configuration |
| `scripts/convert_objaverse_to_depthsplat.py` | Blender → DepthSplat format converter (with depth validation) |
| `scripts/diagnose_objaverse_data.py` | Data quality diagnostic tool |
| `scripts/compute_depth_bounds.py` | Compute optimal near/far bounds from dataset statistics |
| `src/loss/loss_depth.py` | GT depth supervision loss for training stability |

### Key Modifications

**1. Object-Centric View Sampling** (`src/dataset/view_sampler/view_sampler_object_centric.py`)

Maximizes view coverage using farthest-point sampling on camera positions instead of sequential frame selection.

**2. Coordinate Conversion** (`scripts/convert_objaverse_to_depthsplat.py`)

Converts Blender cameras (Y-forward, Z-up) to OpenCV convention (Z-forward, Y-down) with normalized intrinsics.

## Thesis Context

This work implements Section 3.3 of the thesis methodology:

> **Feed-Forward 3D Gaussian Reconstruction**: DepthSplat's dual-branch architecture combines multi-view cost volumes with monocular depth priors (Depth Anything V2), providing robustness in textureless regions where standard photometric matching fails.

### Why DepthSplat?

From Appendix A (Architecture Selection):
- **Dual-branch design**: Monocular priors + multi-view geometry
- **Feed-forward**: Single inference pass (no per-scene optimization)
- **Open-source**: Official implementation from ETH Zurich
- **Modular**: Supports targeted fine-tuning

### Training Pipeline

```
Objaverse Objects (35k) → Blender Renders (32 views/object)
                                    ↓
                        DepthSplat Training (this repo)
                                    ↓
                        Object-Centric 3DGS Model
                                    ↓
                        Fine-tune on Drone/Bird Assets
                                    ↓
                        4D Classification (VN-Transformer + Mamba4D)
```

## Configuration Reference

Key parameters in `config/experiment/objaverse.yaml`:

```yaml
dataset:
  near: 0.35                   # Data-driven (5th percentile of dataset min depths)
  far: 4.21                    # Data-driven (95th percentile of dataset max depths)
  use_masks: true              # Load masks for depth cost volume masking
  view_sampler:
    num_context_views: 5       # Input views
    num_target_views: 2        # Supervision views

model:
  encoder:
    local_mv_match: 5          # Compare all context views for stereo matching
    softmax_temperature: 2.0   # Prevents depth collapse (smoother distributions)
    gaussian_adapter:
      gaussian_scale_max: 0.1  # Smaller for objects vs scenes

loss:
  mse:
    weight: 1.0                # Pixel reconstruction (foreground-masked)
  lpips:
    weight: 0.05               # Perceptual quality (foreground-masked)
  depth:
    weight: 0.5                # GT depth supervision
    min_depth: 0.35            # Match dataset.near
    max_depth: 4.21            # Match dataset.far
    background_weight: 0.1     # Push background depth to far plane
  silhouette:
    weight: 0.2                # Penalize Gaussians in background (prevents floaters)
```

## Evaluation Metrics & Loss Functions

This experiment uses a specific set of metrics for both training supervision and quality evaluation.

### 1. Mean Squared Error (MSE)
**Formula:** $MSE = \frac{1}{N} \sum (I - \hat{I})^2$
- **Intuition**: Pixel-perfect accuracy. Penalizes large errors heavily.
- **Usage**: 
  - **Training**: Used as the primary reconstruction loss. Crucially, it is **Foreground-Masked** in this object-centric setup. The loss is computed only on pixels where the object exists (using `valid_depth_mask`), preventing the model from wasting capacity on the empty background.
  - **Evaluation**: Reported as PSNR.

### 2. Peak Signal-to-Noise Ratio (PSNR)
**Formula:** $PSNR = 10 \cdot \log_{10}(\frac{MAX_I^2}{MSE})$
- **Intuition**: Standardized quality score in decibels (dB). Higher is better (>30dB is high quality). 
- **Usage**: Used strictly for **evaluation** benchmarks. Calculated on the full image.

### 3. SSIM (Structural Similarity Index)
**Formula:** Measures similarity in Luminance ($l$), Contrast ($c$), and Structure ($s$).
- **Intuition**: Perceives structural information (edges, textures) rather than just absolute pixel differences. Values range -1 to 1 (1 is identical).
- **Usage**: **Evaluation** metric. computed with an $11 \times 11$ window to assess structural fidelity.

### 4. LPIPS (Learned Perceptual Image Patch Similarity)
**Formula:** Computes distance in VGG feature space.
- **Intuition**: "Perceptual" metric that aligns with human vision. It understands that a slightly blurry texture is worse than a shifted but sharp texture. Lower is better.
- **Usage**:
  - **Training**: Auxiliary loss (weight=0.05) to enforce sharpness. Like MSE, it is **Masked** (background zeroed out) before computation.
  - **Evaluation**: Benchmarks perceptual quality.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| OOM | Reduce `batch_size` or `num_context_views` |
| Floating artifacts | Enable silhouette loss (weight=0.2); reduce `gaussian_scale_max` |
| Poor reconstruction | Increase training steps or fine-tune from pretrained |
| Depth not loading | Install OpenEXR: `pip install openexr` |
| Training collapse (black output) | Ensure foreground-masked loss is enabled; add depth supervision |
| Depth shows few discrete levels | Increase `softmax_temperature` (try 2.0-3.0) |
| Poor stereo matching | Increase `local_mv_match` to match `num_context_views` |
| Grid pattern in background depth | Increase `background_weight` in depth loss; already fixed upsampling to bilinear |
| Projections appear small | Uses percentile cropping by default; increase `percentile_crop` if needed |

## Recent Modifications

- **Added depth validation** - Filters scenes with broken Blender depth renders (e.g., uniform 100m depth)
- **Added depth cleaning** - Background/invalid depths set to -1 sentinel value
- **Diagnostic scripts** - `scripts/diagnose_objaverse_data.py` for verifying camera poses and depth ranges

### Training Collapse Fix
- **Foreground-masked loss** - MSE/LPIPS only computed on foreground pixels (prevents black output collapse)
- **Depth supervision loss** - New `loss_depth.py` supervises predicted depth with GT Blender depth
- **Fixed mask key bug** - Changed `"masks"` → `"mask"` in encoder so masks are passed to depth predictor
- **Updated loss config** - Now uses MSE + LPIPS + Depth (all foreground-masked)
- **Fixed background depth artifacts** - Modified `mv_unimatch.py` to mask the final residual depth prediction, ensuring background pixels are forced to the far plane.

### Depth Resolution Improvements
- **Data-driven depth bounds** - New `scripts/compute_depth_bounds.py` analyzes dataset to compute optimal near/far bounds. Updated config to use near=0.35m, far=4.21m (was 0.5-5.0m), improving depth candidate resolution by ~14%.
- **Increased multi-view matching** - Changed `local_mv_match` from 3 to 5, ensuring all context views are compared for better stereo matching coverage.

### Floating Gaussian Fix
- **Re-enabled silhouette loss** - Added `silhouette` loss back to config (weight=0.2) to penalize Gaussians in background regions. Uses BCE loss between predicted opacity (from rendered color) and GT mask. This prevents "floating Gaussians" that appear when background is excluded from RGB loss.
- **Background depth regularization** - Added `background_weight` parameter to depth loss (default=0.1) that pushes background pixels toward the far plane. Eliminates grid artifacts and arbitrary depth predictions in background.
- **Fixed match_prob upsampling** - Changed `F.interpolate` mode from `nearest` to `bilinear` in `encoder_depthsplat.py:222` to eliminate grid pattern artifacts in depth predictions.

### Visualization Improvements
- **Larger projection visualizations** - Added `percentile_crop` parameter to `render_projections()` that uses 2nd-98th percentile bounds instead of min/max. Excludes outlier Gaussians so objects fill more of the frame.
- **Silhouette visualization** - New "silhouette" panel in wandb validation showing side-by-side: context views, GT masks, and predicted silhouettes. Helps monitor silhouette loss effectiveness during training. Auto-detects white/black background.

### White Background Training (Recommended)
Based on analysis of successful feed-forward 3DGS methods (LGM, GRM, TriplaneGaussian, Splatter Image), white backgrounds with full-image losses are the standard approach:

- **New config**: `config/experiment/objaverse_white.yaml` for white background training
- **Full-image MSE/LPIPS**: Added `use_foreground_mask` parameter (default: True for backwards compatibility). Set to `false` for white background training.
- **Why white backgrounds work**: Predicting uniform white is heavily penalized because object pixels have diverse colors. Model must learn correct colors AND transparency.
- **Separate alpha supervision**: Silhouette loss provides independent mask supervision, preventing RGB/alpha trade-offs.
- **Reduced Gaussian scale**: Lowered `gaussian_scale_max` from 0.1 to 0.05 to prevent large floaters.

**Usage**:
```bash
# Re-render Objaverse with white backgrounds, then:
python -m src.main +experiment=objaverse_white
```

## References

- **DepthSplat**: Xu et al., "DepthSplat: Connecting Gaussian Splatting and Depth", CVPR 2025
- **Depth Anything V2**: Yang et al., 2024
- **Thesis**: See `report/05_method.tex` (Methodology) and `report/09_appendix.tex` (Architecture Selection)
