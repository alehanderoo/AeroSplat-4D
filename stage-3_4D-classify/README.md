# 4D Gaussian Classifier

A two-stage neural network for classifying flying objects (drones vs birds) from temporal sequences of 3D Gaussian Splatting reconstructions. The architecture achieves **rotation invariance** through Vector Neuron processing and efficient **temporal modeling** via Selective State Space Models.

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│  INPUT: Temporal sequence of .ply files (T frames × M Gaussians)     │
└──────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────┐
│  STAGE 1: VN-Transformer (per-frame)                                 │
│  ────────────────────────────────────                                │
│  • Separates features into equivariant (positions, axes) and        │
│    invariant (opacity, eigenvalues, SH energies)                     │
│  • VN-Attention with Frobenius inner products                        │
│  • Attention pooling → rotation-invariant frame embedding            │
│                                                                      │
│  Output: z^(t) ∈ ℝ^256 per frame                                     │
└──────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────┐
│  STAGE 2: Mamba Temporal Encoder                                     │
│  ────────────────────────────────                                    │
│  • Stacked Mamba blocks with selective SSM                           │
│  • O(T) complexity (vs O(T²) for transformers)                       │
│  • Sequence aggregation (mean/attention pooling)                     │
│                                                                      │
│  Output: y_temporal ∈ ℝ^256                                          │
└──────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────┐
│  CLASSIFICATION HEAD: MLP → σ(logit) → P(drone)                      │
└──────────────────────────────────────────────────────────────────────┘
```

## Components from Reference Projects

### From VN-Transformer

| Component | Purpose |
|-----------|---------|
| `VNLinear` | Channel mixing while preserving 3D vector structure |
| `VNReLU` | Equivariant nonlinearity using learned projection directions |
| `VNLayerNorm` | Normalizes vector norms, applies LayerNorm to magnitudes |
| `VNAttention` | Rotation-invariant attention via Frobenius inner products |
| `VNInvariant` | Projects equivariant features to invariant scalars |
| `Attend` | Flash/standard attention with optional L2 distance mode |

### From Mamba4D

| Component | Purpose |
|-----------|---------|
| `MixerModel` | Stacked Mamba blocks with residual connections |
| `Block` | Prenorm block with fused add-norm optimization |
| `MambaFallback` | GRU-based fallback when mamba-ssm unavailable |

## Project Structure

```
stage-3_4D-classify/
├── configs/
│   ├── train_full.yaml       # Production config
│   └── train_lite.yaml       # Debug/fast iteration config
├── scripts/
│   ├── train.py              # Training entrypoint
│   └── evaluate.py           # Evaluation entrypoint
├── tests/
│   └── test_pipeline.py      # Unit tests
└── src/
    ├── models/
    │   ├── stage1_spatial/   # VN-Transformer
    │   ├── stage2_temporal/  # Mamba encoder
    │   ├── classification_head.py
    │   └── full_model.py     # Gaussian4DClassifier
    ├── data/
    │   ├── ply_parser.py     # PLY → GaussianCloud
    │   ├── feature_extractor.py
    │   ├── gaussian_dataset.py
    │   └── augmentations.py
    ├── training/
    │   ├── trainer.py
    │   ├── losses.py         # BCE, Focal loss
    │   └── metrics.py
    ├── inference/
    │   └── predictor.py
    └── utils/
        ├── config.py
        └── rotation_utils.py
```

## Data Flow

### Training Pipeline

```
1. Dataset Discovery
   data/train/{sequence_001/, sequence_002/, ...}
        └── frame_0001.ply, frame_0002.ply, ...

2. Per-Sequence Loading
   ┌─────────────────────────────────────────────────────────────┐
   │  GaussianSequenceDataset                                    │
   │  ├── parse_ply() → GaussianCloud (positions, scales,       │
   │  │                  rotations, opacities, SH coeffs)        │
   │  ├── GaussianFeatureExtractor.extract_batch()               │
   │  │   ├── Compute covariance eigendecomposition              │
   │  │   ├── Extract SH band energies (rotation-invariant)      │
   │  │   └── Subsample if > max_gaussians (FPS/random)          │
   │  └── Apply augmentations (rotation, jitter, temporal flip)  │
   └─────────────────────────────────────────────────────────────┘
                              │
                              ▼
   Batch: {positions: (B,T,M,3), scalars: (B,T,M,16),
           eigenvectors: (B,T,M,3,3), mask: (B,T,M), labels: (B,)}

3. Forward Pass
   ┌─────────────────────────────────────────────────────────────┐
   │  Gaussian4DClassifier.forward()                             │
   │  ├── For each frame t in T:                                 │
   │  │   └── GaussianSpatialEncoder(pos_t, scalars_t, mask_t)   │
   │  │       ├── VNTransformer: equivariant processing          │
   │  │       └── VNAttentionPooling: → frame_emb (B, 256)       │
   │  ├── Stack → frame_embeddings (B, T, 256)                   │
   │  ├── TemporalMambaEncoder(frame_embeddings)                 │
   │  │   └── → temporal_embedding (B, 256)                      │
   │  └── ClassificationHead → logits (B, 1)                     │
   └─────────────────────────────────────────────────────────────┘

4. Loss & Optimization
   FocalLoss(logits, labels) → backward → AdamW step
```

### Inference Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│  Predictor.predict_sequence(ply_paths)                      │
│  ├── Load PLY files → List[GaussianCloud]                   │
│  ├── Extract features → batch tensors                       │
│  ├── model.forward() → probabilities                        │
│  └── Return: {probability, prediction, confidence, latency} │
└─────────────────────────────────────────────────────────────┘
```

## Installation

```bash
# Core dependencies
pip install torch einops tqdm pyyaml

# Optional: Fast Mamba kernels (recommended for training)
pip install mamba-ssm

# Optional: PLY file support
pip install plyfile

# Optional: Metrics
pip install scikit-learn
```

## Usage

### Training

```bash
# With real data
python scripts/train.py \
    --config configs/train_full.yaml \
    --data-root /path/to/sequences \
    --checkpoint-dir checkpoints/

# With synthetic data (testing)
python scripts/train.py \
    --config configs/train_lite.yaml \
    --synthetic
```

### Evaluation

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/best_model.pt \
    --data-root /path/to/test \
    --output results.json
```

### Inference (Python API)

```python
from src.inference import Predictor

predictor = Predictor(checkpoint_path="checkpoints/best_model.pt")

# Predict from PLY files
result = predictor.predict_sequence([
    "frame_001.ply", "frame_002.ply", ..., "frame_030.ply"
])

print(f"Class: {'drone' if result['prediction'] == 1 else 'bird'}")
print(f"Confidence: {result['confidence']:.2%}")
```

## Data Format

### Expected PLY Structure

Each `.ply` file should contain Gaussian primitives with properties:
- `x, y, z` — position
- `scale_0, scale_1, scale_2` — log-scale
- `rot_0, rot_1, rot_2, rot_3` — quaternion (wxyz)
- `opacity` — logit form
- `f_dc_0, f_dc_1, f_dc_2` — DC spherical harmonics
- `f_rest_0 ... f_rest_44` — higher-order SH (optional)

### Directory Structure

```
data/
├── train/
│   ├── sequence_001/
│   │   ├── frame_0001.ply
│   │   ├── frame_0002.ply
│   │   └── ...
│   └── sequence_002/
│       └── ...
├── val/
│   └── ...
└── labels.json  # {"sequence_001": 0, "sequence_002": 1, ...}
```

## Configuration

Key parameters in `configs/train_full.yaml`:

```yaml
model:
  spatial_dim: 128        # VN-Transformer hidden dim
  spatial_depth: 4        # Number of VN-Transformer layers
  temporal_num_layers: 4  # Number of Mamba layers

data:
  sequence_length: 30     # Frames per sample
  max_gaussians: 50000    # Subsample threshold

training:
  loss_type: focal        # 'bce' or 'focal'
  learning_rate: 1e-4
```

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| VN-Transformer for spatial | Rotation invariance without data augmentation dependency |
| Invariance at Stage 1 | Decouples geometric invariance from temporal modeling |
| Mamba over Transformer | O(T) vs O(T²) — critical for long sequences |
| Attention pooling | Handles variable Gaussian counts, permutation invariant |
| ε-approximate equivariance | Numerical stability with minimal accuracy loss |

## Testing

```bash
python tests/test_pipeline.py
```

Verifies: VN-Transformer, Mamba encoder, full model, dataset, feature extraction, losses, metrics, and rotation invariance.
