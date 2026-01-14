# Rotation-Robust 4D Classification Architecture: Design Plan

## Executive Summary

This document provides a complete implementation plan for a two-stage neural network architecture that classifies flying objects (drones vs. birds) from temporal sequences of 3D Gaussian Splatting reconstructions. The architecture combines:

1. **Stage 1**: VN-Transformer for rotation-invariant spatial encoding of per-frame Gaussian primitives
2. **Stage 2**: Mamba4D State Space Model for temporal dynamics modeling across frame sequences
3. **Classification Head**: MLP for final binary classification

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Project Structure](#2-project-structure)
3. [Dependencies & Environment](#3-dependencies--environment)
4. [Data Pipeline](#4-data-pipeline)
5. [Stage 1: VN-Transformer Spatial Encoder](#5-stage-1-vn-transformer-spatial-encoder)
6. [Stage 2: Mamba4D Temporal Encoder](#6-stage-2-mamba4d-temporal-encoder)
7. [Classification Head](#7-classification-head)
8. [Full Model Integration](#8-full-model-integration)
9. [Training Pipeline](#9-training-pipeline)
10. [Inference Pipeline](#10-inference-pipeline)
11. [Configuration System](#11-configuration-system)
12. [Testing Strategy](#12-testing-strategy)
13. [Implementation Phases](#13-implementation-phases)

---

## 1. System Overview

### 1.1 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           INPUT: Temporal Sequence of .ply Files                 │
│                           {frame_001.ply, frame_002.ply, ..., frame_T.ply}       │
└─────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              PLY LOADER & PREPROCESSOR                           │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │ For each frame t:                                                        │    │
│  │   • Parse .ply → Extract Gaussian attributes                             │    │
│  │   • μ (position), Σ (covariance), α (opacity), c (SH coefficients)      │    │
│  │   • Decompose Σ → scale s, rotation quaternion q                         │    │
│  │   • Compute invariant features (eigenvalues, SH band energies)           │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    STAGE 1: VN-TRANSFORMER (Per-Frame Processing)                │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │ Input: G^(t) = {g_i}_{i=1}^{M_t} (100k-300k Gaussians per frame)        │    │
│  │                                                                          │    │
│  │ Feature Preparation:                                                     │    │
│  │   • Type-1 (Equivariant): positions μ, covariance principal axes        │    │
│  │   • Scalars (Invariant): opacity α, eigenvalues λ, SH band energies     │    │
│  │                                                                          │    │
│  │ VN-Transformer Blocks:                                                   │    │
│  │   • VN-Linear layers (channel mixing, spatial-preserving)               │    │
│  │   • VN-ReLU (equivariant nonlinearity)                                  │    │
│  │   • Frobenius Attention (rotation-invariant attention weights)          │    │
│  │   • ε-approximate equivariance (bias_epsilon ≈ 1e-6)                    │    │
│  │                                                                          │    │
│  │ Output: VN-Invariant layer → Attention pooling                          │    │
│  │         z^(t) ∈ ℝ^D (rotation-invariant frame embedding)                │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         FRAME EMBEDDING SEQUENCE                                 │
│                    {z^(1), z^(2), ..., z^(T)} ∈ ℝ^{T×D}                         │
└─────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    STAGE 2: MAMBA4D TEMPORAL ENCODER                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │ Input: Sequence of frame embeddings {z^(t)}_{t=1}^T                      │    │
│  │                                                                          │    │
│  │ Mamba Block Processing:                                                  │    │
│  │   • Depth-wise convolution for local temporal features                   │    │
│  │   • Selective State Space Model (SSM):                                   │    │
│  │       x_k = Ā·x_{k-1} + B̄·u_k                                           │    │
│  │       y_k = C·x_k + D·u_k                                               │    │
│  │   • Data-dependent A, B, C parameters (selective mechanism)              │    │
│  │   • SiLU activation with gating                                         │    │
│  │                                                                          │    │
│  │ Stacked Mamba Layers → Temporal embedding y_temporal                     │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         CLASSIFICATION HEAD (MLP)                                │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │ Input: y_temporal ∈ ℝ^D                                                  │    │
│  │                                                                          │    │
│  │ MLP: Linear → ReLU → Dropout → Linear                                    │    │
│  │                                                                          │    │
│  │ Output: ŷ = σ(W_c · y_temporal + b_c) ∈ [0,1]                           │    │
│  │         Probability of drone class                                       │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              OUTPUT: Classification                              │
│                        0 = Bird, 1 = Drone (or vice versa)                      │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| VN-Transformer over full equivariant processing | Simpler implementation, better training stability |
| Invariance established early (Stage 1) | Separates geometric invariance from temporal modeling |
| Mamba over Transformer for temporal | O(T) vs O(T²) complexity, better for long sequences |
| Attention pooling for frame aggregation | Permutation invariant, handles variable Gaussian counts |
| ε-approximate equivariance | Numerical stability on accelerators, minimal accuracy loss |

### 1.3 Input/Output Specification

**Input:**
- Sequence of T `.ply` files (T ≥ 3, typically 30-60 for 1-2 seconds at 30fps)
- Each `.ply` contains M_t Gaussians (100k-300k) with attributes:
  - Position μ ∈ ℝ³
  - Covariance Σ (as scale s ∈ ℝ³ and rotation quaternion q ∈ ℝ⁴)
  - Opacity α ∈ [0,1]
  - Spherical Harmonics c ∈ ℝ⁴⁸ (degree 0-3, 16 coefficients × 3 RGB)

**Output:**
- Binary classification probability ŷ ∈ [0,1]
- Class: Drone (1) or Bird (0)

---

## 2. Project Structure

```
4d_gaussian_classifier/
├── README.md
├── requirements.txt
├── setup.py
├── pyproject.toml
│
├── configs/
│   ├── default.yaml              # Default hyperparameters
│   ├── train_config.yaml         # Training configuration
│   └── model_config.yaml         # Model architecture config
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── ply_parser.py         # PLY file loading and parsing
│   │   ├── gaussian_dataset.py   # PyTorch Dataset for sequences
│   │   ├── feature_extractor.py  # Gaussian attribute preprocessing
│   │   ├── augmentations.py      # Data augmentation (temporal, spatial)
│   │   └── collate.py            # Custom collate for variable-length sequences
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── full_model.py         # Complete 2-stage architecture
│   │   │
│   │   ├── stage1_spatial/
│   │   │   ├── __init__.py
│   │   │   ├── vn_transformer.py     # Main VN-Transformer module
│   │   │   ├── vn_layers.py          # VN-Linear, VN-ReLU, VN-Invariant
│   │   │   ├── vn_attention.py       # Frobenius attention implementation
│   │   │   ├── feature_preparation.py # Gaussian → VN feature conversion
│   │   │   └── frame_pooling.py      # Attention-weighted aggregation
│   │   │
│   │   ├── stage2_temporal/
│   │   │   ├── __init__.py
│   │   │   ├── mamba_temporal.py     # Temporal Mamba encoder
│   │   │   ├── mamba_block.py        # Individual Mamba block
│   │   │   └── ssm.py                # Selective State Space core
│   │   │
│   │   └── classification_head.py    # MLP classification head
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py            # Main training loop
│   │   ├── losses.py             # Loss functions (BCE, focal, etc.)
│   │   ├── metrics.py            # Accuracy, F1, AUC metrics
│   │   ├── optimizer.py          # Optimizer factory
│   │   ├── scheduler.py          # LR scheduler factory
│   │   └── callbacks.py          # Checkpointing, early stopping, logging
│   │
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── predictor.py          # Inference wrapper
│   │   └── export.py             # ONNX/TorchScript export
│   │
│   └── utils/
│       ├── __init__.py
│       ├── config.py             # Configuration management
│       ├── logging.py            # Logging utilities
│       ├── distributed.py        # Multi-GPU utilities
│       └── rotation_utils.py     # Quaternion and rotation helpers
│
├── scripts/
│   ├── train.py                  # Training entry point
│   ├── evaluate.py               # Evaluation script
│   ├── predict.py                # Single-sequence inference
│   └── prepare_data.py           # Data preprocessing script
│
├── tests/
│   ├── __init__.py
│   ├── test_ply_parser.py
│   ├── test_vn_equivariance.py   # Verify rotation equivariance/invariance
│   ├── test_mamba_block.py
│   ├── test_full_pipeline.py
│   └── fixtures/                 # Test data fixtures
│       └── sample_sequence/
│
└── notebooks/
    ├── 01_data_exploration.ipynb
    ├── 02_model_debugging.ipynb
    └── 03_results_analysis.ipynb
```

---

## 3. Dependencies & Environment

### 3.1 Core Dependencies

```txt
# requirements.txt

# Core ML
torch>=2.1.0
torchvision>=0.16.0
einops>=0.6.0

# VN-Transformer (install from pip)
VN-transformer>=0.1.0

# Mamba SSM (requires special installation)
mamba-ssm>=1.2.0
causal-conv1d>=1.2.0

# Point Cloud Processing
plyfile>=0.9
open3d>=0.17.0

# KNN for point operations
# KNN_CUDA - install from: https://github.com/unlimblue/KNN_CUDA
# Requires manual installation

# Pointnet2 operations
# Install from: https://github.com/erikwijmans/Pointnet2_PyTorch
# Requires manual installation

# Configuration & Logging
pyyaml>=6.0
omegaconf>=2.3.0
wandb>=0.15.0
tensorboard>=2.13.0

# Utilities
numpy>=1.24.0
scipy>=1.10.0
tqdm>=4.65.0
rich>=13.0.0

# Testing
pytest>=7.3.0
pytest-cov>=4.1.0
```

### 3.2 Environment Setup Script

```bash
#!/bin/bash
# setup_environment.sh

# Create conda environment
conda create -n gaussian4d python=3.10 -y
conda activate gaussian4d

# Install PyTorch with CUDA
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118

# Install VN-Transformer
pip install VN-transformer

# Install Mamba SSM (requires CUDA)
pip install mamba-ssm>=1.2.0
pip install causal-conv1d>=1.2.0

# Install KNN_CUDA
git clone https://github.com/unlimblue/KNN_CUDA.git
cd KNN_CUDA
pip install .
cd ..

# Install Pointnet2 PyTorch
git clone https://github.com/erikwijmans/Pointnet2_PyTorch.git
cd Pointnet2_PyTorch
pip install -r requirements.txt
pip install -e .
cd ..

# Install remaining dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "from VN_transformer import VNTransformer; print('VN-Transformer: OK')"
python -c "from mamba_ssm import Mamba; print('Mamba: OK')"
```

### 3.3 Docker Alternative (Recommended for Mamba)

```dockerfile
# Dockerfile
FROM nvcr.io/nvidia/pytorch:23.10-py3

# Install Mamba dependencies
RUN pip install mamba-ssm>=1.2.0 causal-conv1d>=1.2.0

# Install VN-Transformer
RUN pip install VN-transformer einops

# Install KNN_CUDA
RUN git clone https://github.com/unlimblue/KNN_CUDA.git && \
    cd KNN_CUDA && pip install . && cd ..

# Install Pointnet2
RUN git clone https://github.com/erikwijmans/Pointnet2_PyTorch.git && \
    cd Pointnet2_PyTorch && pip install -r requirements.txt && pip install -e . && cd ..

# Copy project
COPY . /workspace/gaussian4d
WORKDIR /workspace/gaussian4d

RUN pip install -e .
```

---

## 4. Data Pipeline

### 4.1 PLY Parser Module

```python
# src/data/ply_parser.py
"""
PLY file parser for 3D Gaussian Splatting reconstructions.

Expected PLY format from feed-forward reconstruction:
- vertex properties: x, y, z (position)
- scale_0, scale_1, scale_2 (log-scale)
- rot_0, rot_1, rot_2, rot_3 (quaternion wxyz or xyzw - verify!)
- opacity (logit form)
- f_dc_0, f_dc_1, f_dc_2 (DC spherical harmonics)
- f_rest_0 ... f_rest_44 (higher-order SH coefficients)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import torch
from plyfile import PlyData


@dataclass
class GaussianCloud:
    """Container for parsed Gaussian attributes."""
    positions: torch.Tensor      # (N, 3) - xyz coordinates
    scales: torch.Tensor         # (N, 3) - scale in each axis (exp applied)
    rotations: torch.Tensor      # (N, 4) - quaternion (normalized)
    opacities: torch.Tensor      # (N, 1) - opacity [0,1] (sigmoid applied)
    sh_coeffs: torch.Tensor      # (N, 48) - spherical harmonics
    
    # Derived features (computed lazily)
    _covariance: Optional[torch.Tensor] = None
    _eigenvalues: Optional[torch.Tensor] = None
    _eigenvectors: Optional[torch.Tensor] = None
    _sh_band_energies: Optional[torch.Tensor] = None
    
    @property
    def num_gaussians(self) -> int:
        return self.positions.shape[0]
    
    @property
    def covariance(self) -> torch.Tensor:
        """Compute 3x3 covariance matrices from scale and rotation."""
        if self._covariance is None:
            self._covariance = self._compute_covariance()
        return self._covariance
    
    @property
    def eigenvalues(self) -> torch.Tensor:
        """Eigenvalues of covariance (rotation-invariant shape descriptor)."""
        if self._eigenvalues is None:
            self._compute_eigen_decomposition()
        return self._eigenvalues
    
    @property
    def eigenvectors(self) -> torch.Tensor:
        """Principal axes (rotation-equivariant)."""
        if self._eigenvectors is None:
            self._compute_eigen_decomposition()
        return self._eigenvectors
    
    @property
    def sh_band_energies(self) -> torch.Tensor:
        """SH band energies (rotation-invariant)."""
        if self._sh_band_energies is None:
            self._sh_band_energies = self._compute_sh_band_energies()
        return self._sh_band_energies
    
    def _compute_covariance(self) -> torch.Tensor:
        """Σ = R @ S @ S^T @ R^T where R is rotation matrix, S is diagonal scale."""
        # Convert quaternion to rotation matrix
        R = quaternion_to_rotation_matrix(self.rotations)  # (N, 3, 3)
        
        # Scale matrix (diagonal)
        S = torch.diag_embed(self.scales)  # (N, 3, 3)
        
        # Covariance
        RS = R @ S
        cov = RS @ RS.transpose(-1, -2)
        return cov  # (N, 3, 3)
    
    def _compute_eigen_decomposition(self):
        """Compute eigenvalues and eigenvectors of covariance matrices."""
        # torch.linalg.eigh for symmetric matrices (covariance is symmetric)
        eigenvalues, eigenvectors = torch.linalg.eigh(self.covariance)
        
        # Sort in descending order
        idx = torch.argsort(eigenvalues, dim=-1, descending=True)
        self._eigenvalues = torch.gather(eigenvalues, -1, idx)
        self._eigenvectors = torch.gather(
            eigenvectors, -1, 
            idx.unsqueeze(-2).expand(-1, 3, -1)
        )
    
    def _compute_sh_band_energies(self) -> torch.Tensor:
        """
        Compute rotation-invariant SH band energies.
        E_l = Σ_{m=-l}^{l} |c_l^m|² for each degree l ∈ {0,1,2,3}
        
        SH coefficients layout (per channel):
        - l=0: 1 coefficient (index 0)
        - l=1: 3 coefficients (indices 1-3)
        - l=2: 5 coefficients (indices 4-8)
        - l=3: 7 coefficients (indices 9-15)
        Total: 16 per channel × 3 channels = 48
        """
        # Reshape to (N, 3, 16) for per-channel processing
        sh = self.sh_coeffs.view(-1, 3, 16)
        
        # Band indices
        l0 = sh[:, :, 0:1]           # (N, 3, 1)
        l1 = sh[:, :, 1:4]           # (N, 3, 3)
        l2 = sh[:, :, 4:9]           # (N, 3, 5)
        l3 = sh[:, :, 9:16]          # (N, 3, 7)
        
        # Compute band energies (sum of squared coefficients)
        e0 = (l0 ** 2).sum(dim=-1)   # (N, 3)
        e1 = (l1 ** 2).sum(dim=-1)   # (N, 3)
        e2 = (l2 ** 2).sum(dim=-1)   # (N, 3)
        e3 = (l3 ** 2).sum(dim=-1)   # (N, 3)
        
        # Concatenate: (N, 12) - 4 bands × 3 channels
        band_energies = torch.cat([e0, e1, e2, e3], dim=-1)
        return band_energies


def parse_ply(filepath: Path, device: str = 'cpu') -> GaussianCloud:
    """
    Parse a PLY file containing 3D Gaussian Splatting data.
    
    Args:
        filepath: Path to .ply file
        device: Target device for tensors
        
    Returns:
        GaussianCloud with all attributes
    """
    plydata = PlyData.read(filepath)
    vertex = plydata['vertex']
    
    # Extract positions
    positions = np.stack([
        vertex['x'], vertex['y'], vertex['z']
    ], axis=-1).astype(np.float32)
    
    # Extract scales (stored as log-scale, need exp)
    scales_log = np.stack([
        vertex['scale_0'], vertex['scale_1'], vertex['scale_2']
    ], axis=-1).astype(np.float32)
    scales = np.exp(scales_log)
    
    # Extract rotations (quaternion - verify order from your reconstruction!)
    # Common orders: wxyz or xyzw - adjust as needed
    rotations = np.stack([
        vertex['rot_0'], vertex['rot_1'], vertex['rot_2'], vertex['rot_3']
    ], axis=-1).astype(np.float32)
    # Normalize quaternions
    rotations = rotations / (np.linalg.norm(rotations, axis=-1, keepdims=True) + 1e-8)
    
    # Extract opacity (stored as logit, need sigmoid)
    opacity_logit = vertex['opacity'].astype(np.float32)
    opacities = 1.0 / (1.0 + np.exp(-opacity_logit))
    opacities = opacities[:, np.newaxis]
    
    # Extract spherical harmonics
    sh_dc = np.stack([
        vertex['f_dc_0'], vertex['f_dc_1'], vertex['f_dc_2']
    ], axis=-1).astype(np.float32)
    
    # Higher order SH (indices 0-44 for f_rest)
    sh_rest_names = [f'f_rest_{i}' for i in range(45)]
    sh_rest = np.stack([
        vertex[name] for name in sh_rest_names if name in vertex.dtype.names
    ], axis=-1).astype(np.float32)
    
    # Combine SH coefficients
    # Layout: reshape to (N, 3, 16) then flatten to (N, 48)
    # DC is first, then rest interleaved per channel
    if sh_rest.shape[-1] == 45:
        # Full SH up to degree 3
        sh_rest = sh_rest.reshape(-1, 15, 3).transpose(0, 2, 1)  # (N, 3, 15)
        sh_dc = sh_dc.reshape(-1, 3, 1)  # (N, 3, 1)
        sh_full = np.concatenate([sh_dc, sh_rest], axis=-1)  # (N, 3, 16)
        sh_coeffs = sh_full.reshape(-1, 48)  # (N, 48)
    else:
        # Handle partial SH (pad with zeros)
        sh_coeffs = np.zeros((positions.shape[0], 48), dtype=np.float32)
        sh_coeffs[:, :3] = sh_dc  # DC term only
    
    # Convert to tensors
    return GaussianCloud(
        positions=torch.from_numpy(positions).to(device),
        scales=torch.from_numpy(scales).to(device),
        rotations=torch.from_numpy(rotations).to(device),
        opacities=torch.from_numpy(opacities).to(device),
        sh_coeffs=torch.from_numpy(sh_coeffs).to(device),
    )


def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion to rotation matrix.
    
    Args:
        q: Quaternion tensor (N, 4) in wxyz order
        
    Returns:
        Rotation matrices (N, 3, 3)
    """
    # Normalize
    q = q / (q.norm(dim=-1, keepdim=True) + 1e-8)
    
    w, x, y, z = q.unbind(-1)
    
    # Rotation matrix from quaternion
    R = torch.stack([
        1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y,
        2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x,
        2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y
    ], dim=-1).view(-1, 3, 3)
    
    return R
```

### 4.2 Gaussian Dataset

```python
# src/data/gaussian_dataset.py
"""
PyTorch Dataset for temporal sequences of Gaussian reconstructions.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
from torch.utils.data import Dataset
import json

from .ply_parser import parse_ply, GaussianCloud
from .feature_extractor import GaussianFeatureExtractor


class GaussianSequenceDataset(Dataset):
    """
    Dataset for loading temporal sequences of 3D Gaussian reconstructions.
    
    Directory structure expected:
        data_root/
            sequence_001/
                frame_0001.ply
                frame_0002.ply
                ...
                metadata.json  # optional: contains label
            sequence_002/
                ...
    
    Or with separate label file:
        data_root/
            sequences/
                ...
            labels.json  # {"sequence_001": 0, "sequence_002": 1, ...}
    """
    
    def __init__(
        self,
        data_root: Path,
        sequence_length: int = 30,
        stride: int = 1,
        max_gaussians: Optional[int] = None,
        subsample_strategy: str = 'random',  # 'random', 'fps', 'grid'
        labels_file: Optional[Path] = None,
        transform: Optional[callable] = None,
        feature_extractor: Optional[GaussianFeatureExtractor] = None,
        cache_sequences: bool = False,
    ):
        """
        Args:
            data_root: Root directory containing sequences
            sequence_length: Number of frames per sample (T)
            stride: Frame stride for sampling
            max_gaussians: Maximum Gaussians per frame (subsample if exceeded)
            subsample_strategy: How to reduce Gaussians ('random', 'fps', 'grid')
            labels_file: Path to labels JSON (if not in per-sequence metadata)
            transform: Optional transform to apply to features
            feature_extractor: Feature extraction module
            cache_sequences: Whether to cache parsed sequences in memory
        """
        self.data_root = Path(data_root)
        self.sequence_length = sequence_length
        self.stride = stride
        self.max_gaussians = max_gaussians
        self.subsample_strategy = subsample_strategy
        self.transform = transform
        self.feature_extractor = feature_extractor or GaussianFeatureExtractor()
        self.cache_sequences = cache_sequences
        
        # Discover sequences
        self.sequences = self._discover_sequences()
        
        # Load labels
        self.labels = self._load_labels(labels_file)
        
        # Cache storage
        self._cache: Dict[str, List[GaussianCloud]] = {}
    
    def _discover_sequences(self) -> List[Path]:
        """Find all sequence directories."""
        sequences = []
        for d in sorted(self.data_root.iterdir()):
            if d.is_dir():
                ply_files = list(d.glob("*.ply"))
                if len(ply_files) >= self.sequence_length:
                    sequences.append(d)
        return sequences
    
    def _load_labels(self, labels_file: Optional[Path]) -> Dict[str, int]:
        """Load labels from file or per-sequence metadata."""
        labels = {}
        
        if labels_file and labels_file.exists():
            with open(labels_file) as f:
                labels = json.load(f)
        else:
            # Try per-sequence metadata
            for seq_dir in self.sequences:
                meta_file = seq_dir / "metadata.json"
                if meta_file.exists():
                    with open(meta_file) as f:
                        meta = json.load(f)
                        if 'label' in meta:
                            labels[seq_dir.name] = meta['label']
        
        return labels
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            Dictionary with:
                - 'features': (T, M, D) tensor of Gaussian features per frame
                - 'positions': (T, M, 3) tensor of positions (equivariant)
                - 'eigenvectors': (T, M, 3, 3) principal axes (equivariant)
                - 'scalars': (T, M, S) invariant scalar features
                - 'mask': (T, M) boolean mask for valid Gaussians
                - 'label': int (0=bird, 1=drone)
                - 'sequence_name': str
        """
        seq_dir = self.sequences[idx]
        seq_name = seq_dir.name
        
        # Load or retrieve from cache
        if self.cache_sequences and seq_name in self._cache:
            frames = self._cache[seq_name]
        else:
            frames = self._load_sequence(seq_dir)
            if self.cache_sequences:
                self._cache[seq_name] = frames
        
        # Sample frames according to stride
        frame_indices = list(range(0, len(frames), self.stride))[:self.sequence_length]
        sampled_frames = [frames[i] for i in frame_indices]
        
        # Extract features for each frame
        batch = self.feature_extractor.extract_batch(
            sampled_frames, 
            max_gaussians=self.max_gaussians,
            subsample_strategy=self.subsample_strategy
        )
        
        # Apply transforms if any
        if self.transform:
            batch = self.transform(batch)
        
        # Add label
        label = self.labels.get(seq_name, -1)
        batch['label'] = torch.tensor(label, dtype=torch.long)
        batch['sequence_name'] = seq_name
        
        return batch
    
    def _load_sequence(self, seq_dir: Path) -> List[GaussianCloud]:
        """Load all frames in a sequence."""
        ply_files = sorted(seq_dir.glob("*.ply"))
        frames = [parse_ply(f) for f in ply_files]
        return frames


class GaussianCollator:
    """
    Custom collator for variable-length Gaussian sequences.
    Pads sequences to maximum length in batch.
    """
    
    def __init__(self, pad_value: float = 0.0):
        self.pad_value = pad_value
    
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate batch with padding."""
        # Get max dimensions
        max_T = max(b['positions'].shape[0] for b in batch)
        max_M = max(b['positions'].shape[1] for b in batch)
        
        # Determine feature dimensions from first item
        D_pos = batch[0]['positions'].shape[-1]
        D_scalar = batch[0]['scalars'].shape[-1]
        
        B = len(batch)
        
        # Initialize padded tensors
        positions = torch.zeros(B, max_T, max_M, D_pos)
        scalars = torch.zeros(B, max_T, max_M, D_scalar)
        eigenvectors = torch.zeros(B, max_T, max_M, 3, 3)
        mask = torch.zeros(B, max_T, max_M, dtype=torch.bool)
        labels = torch.zeros(B, dtype=torch.long)
        
        # Fill tensors
        for i, b in enumerate(batch):
            T, M = b['positions'].shape[:2]
            positions[i, :T, :M] = b['positions']
            scalars[i, :T, :M] = b['scalars']
            eigenvectors[i, :T, :M] = b['eigenvectors']
            mask[i, :T, :M] = True
            labels[i] = b['label']
        
        return {
            'positions': positions,
            'scalars': scalars,
            'eigenvectors': eigenvectors,
            'mask': mask,
            'labels': labels,
        }
```

### 4.3 Feature Extractor

```python
# src/data/feature_extractor.py
"""
Extract and prepare features from GaussianCloud for the network.
Separates equivariant (Type-1) and invariant (scalar) features.
"""

from typing import Dict, List, Optional
import torch
from dataclasses import dataclass

from .ply_parser import GaussianCloud


@dataclass
class GaussianFeatures:
    """Prepared features for VN-Transformer."""
    # Type-1 (Equivariant) features
    positions: torch.Tensor        # (M, 3) - Gaussian positions
    eigenvectors: torch.Tensor     # (M, 3, 3) - Principal axes of covariance
    
    # Scalar (Invariant) features
    eigenvalues: torch.Tensor      # (M, 3) - Shape descriptors
    opacities: torch.Tensor        # (M, 1) - Transparency
    sh_band_energies: torch.Tensor # (M, 12) - Color invariants
    
    @property
    def num_gaussians(self) -> int:
        return self.positions.shape[0]
    
    @property
    def scalar_dim(self) -> int:
        """Total dimension of scalar features."""
        return 3 + 1 + 12  # eigenvalues + opacity + sh_band_energies = 16


class GaussianFeatureExtractor:
    """
    Extract VN-compatible features from GaussianCloud.
    """
    
    def __init__(
        self,
        center_positions: bool = True,
        normalize_positions: bool = True,
        scale_eigenvalues: bool = True,
    ):
        """
        Args:
            center_positions: Subtract centroid from positions
            normalize_positions: Scale to unit sphere
            scale_eigenvalues: Log-scale eigenvalues for numerical stability
        """
        self.center_positions = center_positions
        self.normalize_positions = normalize_positions
        self.scale_eigenvalues = scale_eigenvalues
    
    def extract(self, cloud: GaussianCloud) -> GaussianFeatures:
        """Extract features from a single GaussianCloud."""
        positions = cloud.positions.clone()
        
        # Center positions (translation invariance)
        if self.center_positions:
            centroid = positions.mean(dim=0, keepdim=True)
            positions = positions - centroid
        
        # Normalize to unit sphere
        if self.normalize_positions:
            max_dist = positions.norm(dim=-1).max() + 1e-8
            positions = positions / max_dist
        
        # Get eigendecomposition (lazy computed in GaussianCloud)
        eigenvalues = cloud.eigenvalues  # (M, 3)
        eigenvectors = cloud.eigenvectors  # (M, 3, 3)
        
        # Log-scale eigenvalues for better numerical properties
        if self.scale_eigenvalues:
            eigenvalues = torch.log(eigenvalues + 1e-8)
        
        # Get other invariants
        opacities = cloud.opacities  # (M, 1)
        sh_band_energies = cloud.sh_band_energies  # (M, 12)
        
        return GaussianFeatures(
            positions=positions,
            eigenvectors=eigenvectors,
            eigenvalues=eigenvalues,
            opacities=opacities,
            sh_band_energies=sh_band_energies,
        )
    
    def extract_batch(
        self,
        clouds: List[GaussianCloud],
        max_gaussians: Optional[int] = None,
        subsample_strategy: str = 'random',
    ) -> Dict[str, torch.Tensor]:
        """
        Extract features from a sequence of clouds.
        
        Args:
            clouds: List of T GaussianClouds
            max_gaussians: Maximum Gaussians per frame
            subsample_strategy: How to subsample if needed
            
        Returns:
            Dictionary with batched tensors (T, M, ...)
        """
        features_list = []
        
        for cloud in clouds:
            # Subsample if needed
            if max_gaussians and cloud.num_gaussians > max_gaussians:
                cloud = self._subsample(cloud, max_gaussians, subsample_strategy)
            
            features = self.extract(cloud)
            features_list.append(features)
        
        # Stack into sequence tensors
        # Need to pad to same M across frames
        max_M = max(f.num_gaussians for f in features_list)
        T = len(features_list)
        
        # Initialize tensors
        positions = torch.zeros(T, max_M, 3)
        eigenvectors = torch.zeros(T, max_M, 3, 3)
        scalars = torch.zeros(T, max_M, features_list[0].scalar_dim)
        mask = torch.zeros(T, max_M, dtype=torch.bool)
        
        for t, f in enumerate(features_list):
            M = f.num_gaussians
            positions[t, :M] = f.positions
            eigenvectors[t, :M] = f.eigenvectors
            scalars[t, :M] = torch.cat([
                f.eigenvalues,
                f.opacities,
                f.sh_band_energies
            ], dim=-1)
            mask[t, :M] = True
        
        return {
            'positions': positions,
            'eigenvectors': eigenvectors,
            'scalars': scalars,
            'mask': mask,
        }
    
    def _subsample(
        self, 
        cloud: GaussianCloud, 
        n: int, 
        strategy: str
    ) -> GaussianCloud:
        """Subsample cloud to n Gaussians."""
        M = cloud.num_gaussians
        
        if strategy == 'random':
            indices = torch.randperm(M)[:n]
        elif strategy == 'fps':
            # Farthest Point Sampling - requires KNN_CUDA
            indices = self._fps_subsample(cloud.positions, n)
        elif strategy == 'grid':
            # Voxel grid subsampling
            indices = self._grid_subsample(cloud.positions, n)
        else:
            raise ValueError(f"Unknown subsample strategy: {strategy}")
        
        return GaussianCloud(
            positions=cloud.positions[indices],
            scales=cloud.scales[indices],
            rotations=cloud.rotations[indices],
            opacities=cloud.opacities[indices],
            sh_coeffs=cloud.sh_coeffs[indices],
        )
    
    def _fps_subsample(self, positions: torch.Tensor, n: int) -> torch.Tensor:
        """Farthest Point Sampling."""
        # Implementation using KNN_CUDA or fallback
        try:
            from knn_cuda import KNN
            # FPS implementation
            device = positions.device
            B, N = 1, positions.shape[0]
            positions_batch = positions.unsqueeze(0)  # (1, N, 3)
            
            # Start with random point
            indices = [torch.randint(0, N, (1,), device=device)]
            
            for _ in range(n - 1):
                # Find farthest point from current set
                current_points = positions_batch[:, torch.cat(indices)]
                knn = KNN(k=1, transpose_mode=True)
                _, idx = knn(current_points, positions_batch)
                distances = ((positions_batch - positions_batch[:, :, idx.squeeze()]) ** 2).sum(-1)
                farthest = distances.argmax()
                indices.append(farthest.unsqueeze(0))
            
            return torch.cat(indices)
        except ImportError:
            # Fallback to random
            return torch.randperm(positions.shape[0])[:n]
    
    def _grid_subsample(self, positions: torch.Tensor, n: int) -> torch.Tensor:
        """Voxel grid subsampling - keep one point per voxel."""
        # Compute voxel size to get approximately n points
        M = positions.shape[0]
        pmin = positions.min(dim=0).values
        pmax = positions.max(dim=0).values
        extent = (pmax - pmin).max()
        
        # Estimate voxel size
        voxel_size = extent / (n ** (1/3))
        
        # Quantize positions
        grid_coords = ((positions - pmin) / (voxel_size + 1e-8)).long()
        
        # Hash voxels
        hash_vals = (grid_coords[:, 0] * 73856093) ^ (grid_coords[:, 1] * 19349663) ^ (grid_coords[:, 2] * 83492791)
        
        # Keep first point in each voxel
        unique_hashes, inverse_indices = torch.unique(hash_vals, return_inverse=True)
        
        # Get first occurrence of each hash
        indices = []
        for i, h in enumerate(unique_hashes[:n]):
            idx = (hash_vals == h).nonzero(as_tuple=True)[0][0]
            indices.append(idx)
        
        return torch.stack(indices) if indices else torch.arange(n)
```

---

## 5. Stage 1: VN-Transformer Spatial Encoder

### 5.1 VN-Transformer Architecture Overview

The VN-Transformer processes 3D geometric data while maintaining rotation equivariance. Key components:

1. **Vector Neuron Representation**: Features as (N, C, 3) where each channel is a 3D vector
2. **VN-Linear**: Channel mixing that preserves spatial dimensions
3. **VN-ReLU**: Equivariant nonlinearity using learned projection directions
4. **Frobenius Attention**: Rotation-invariant attention weights via matrix inner products

### 5.2 Implementation

```python
# src/models/stage1_spatial/vn_transformer.py
"""
VN-Transformer wrapper adapted for 3D Gaussian classification.
Based on: https://github.com/lucidrains/VN-transformer
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from einops import rearrange

# Import from pip package
from VN_transformer import VNTransformer


class GaussianSpatialEncoder(nn.Module):
    """
    Stage 1: VN-Transformer-based spatial encoder for Gaussians.
    
    Takes per-frame Gaussian features and produces rotation-invariant frame embeddings.
    """
    
    def __init__(
        self,
        dim: int = 128,
        depth: int = 4,
        dim_head: int = 64,
        heads: int = 8,
        dim_feat: int = 64,  # Scalar feature dimension
        bias_epsilon: float = 1e-6,
        dropout: float = 0.1,
        output_dim: int = 256,  # Final frame embedding dimension
    ):
        """
        Args:
            dim: VN-Transformer hidden dimension (per vector channel)
            depth: Number of VN-Transformer layers
            dim_head: Dimension per attention head
            heads: Number of attention heads
            dim_feat: Dimension for scalar (non-spatial) features
            bias_epsilon: Small bias for numerical stability (ε-approximate equivariance)
            dropout: Dropout rate
            output_dim: Dimension of output frame embedding
        """
        super().__init__()
        
        self.dim = dim
        self.output_dim = output_dim
        
        # Feature preparation: project scalar features to VN-compatible dimension
        self.scalar_proj = nn.Linear(16, dim_feat)  # 16 = eigenvals(3) + opacity(1) + sh_bands(12)
        
        # Project 3D positions to VN input (expand to dim channels)
        # Position is already 3D, we expand channel dimension
        self.position_expand = nn.Linear(1, dim)
        
        # Main VN-Transformer
        self.vn_transformer = VNTransformer(
            dim=dim,
            depth=depth,
            dim_head=dim_head,
            heads=heads,
            dim_feat=dim_feat,
            bias_epsilon=bias_epsilon,
        )
        
        # Frame-level aggregation (attention pooling)
        self.attention_pool = VNAttentionPooling(dim=dim, output_dim=output_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        positions: torch.Tensor,
        scalars: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Process a single frame's Gaussians.
        
        Args:
            positions: (B, M, 3) Gaussian positions
            scalars: (B, M, D_scalar) Scalar features
            mask: (B, M) Boolean mask for valid Gaussians
            
        Returns:
            frame_embedding: (B, output_dim) rotation-invariant frame embedding
        """
        B, M, _ = positions.shape
        
        # Prepare scalar features
        scalar_feats = self.scalar_proj(scalars)  # (B, M, dim_feat)
        
        # VN-Transformer expects:
        #   coors: (B, M, 3) - spatial coordinates
        #   feats: (B, M, dim_feat) - scalar features
        coors_out, feats_out = self.vn_transformer(
            positions,
            feats=scalar_feats
        )
        # coors_out: (B, M, 3) - equivariant
        # feats_out: (B, M, dim) - mixed equivariant/invariant
        
        # Aggregate to frame-level embedding
        frame_embedding = self.attention_pool(coors_out, feats_out, mask)
        
        return self.dropout(frame_embedding)
    
    def forward_sequence(
        self,
        positions: torch.Tensor,
        scalars: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Process a sequence of frames.
        
        Args:
            positions: (B, T, M, 3) Gaussian positions per frame
            scalars: (B, T, M, D_scalar) Scalar features per frame
            mask: (B, T, M) Boolean mask for valid Gaussians
            
        Returns:
            frame_embeddings: (B, T, output_dim) sequence of frame embeddings
        """
        B, T, M, _ = positions.shape
        
        # Process each frame
        embeddings = []
        for t in range(T):
            pos_t = positions[:, t]  # (B, M, 3)
            scalar_t = scalars[:, t]  # (B, M, D_scalar)
            mask_t = mask[:, t] if mask is not None else None
            
            emb_t = self.forward(pos_t, scalar_t, mask_t)
            embeddings.append(emb_t)
        
        # Stack: (B, T, output_dim)
        return torch.stack(embeddings, dim=1)


class VNAttentionPooling(nn.Module):
    """
    Attention-weighted pooling to aggregate per-Gaussian features to frame-level.
    Produces rotation-invariant output through VN-Invariant operation.
    """
    
    def __init__(self, dim: int, output_dim: int):
        super().__init__()
        
        # Attention weights (computed from invariant features)
        self.attention_mlp = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 1),
        )
        
        # VN-Invariant: learn local coordinate frame
        self.coord_frame = nn.Linear(dim, 3)  # Output 3 vectors for local frame
        
        # Final projection
        self.output_proj = nn.Linear(dim * 3 + dim, output_dim)
    
    def forward(
        self,
        coors: torch.Tensor,
        feats: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            coors: (B, M, 3) equivariant coordinates from VN-Transformer
            feats: (B, M, dim) features from VN-Transformer
            mask: (B, M) validity mask
            
        Returns:
            embedding: (B, output_dim) rotation-invariant embedding
        """
        B, M, _ = coors.shape
        
        # Compute attention weights
        attn_logits = self.attention_mlp(feats).squeeze(-1)  # (B, M)
        
        if mask is not None:
            attn_logits = attn_logits.masked_fill(~mask, float('-inf'))
        
        attn_weights = torch.softmax(attn_logits, dim=-1)  # (B, M)
        
        # Weighted aggregation of coordinates
        coors_agg = torch.einsum('bm,bmd->bd', attn_weights, coors)  # (B, 3)
        
        # Weighted aggregation of features
        feats_agg = torch.einsum('bm,bmd->bd', attn_weights, feats)  # (B, dim)
        
        # VN-Invariant: create local coordinate frame from aggregated features
        # Learn 3 directions that form a basis
        frame_dirs = self.coord_frame(feats_agg)  # (B, 3)
        
        # Gram-Schmidt to orthonormalize (makes it rotation-equivariant basis)
        # For simplicity, we use a simpler approach: express coors_agg in 
        # feature-derived directions
        
        # Simple invariant: take norms and relative angles
        coors_norm = coors_agg.norm(dim=-1, keepdim=True)  # (B, 1)
        
        # Invariant features: concatenate
        invariant = torch.cat([
            feats_agg,  # (B, dim) - already mixed invariant
            coors_norm.expand(-1, feats_agg.shape[-1]),  # spatial magnitude info
        ], dim=-1)
        
        # Note: The VN_transformer package handles VN-Invariant internally
        # For our purposes, feats_agg from VN-Transformer already contains
        # the properly processed features. We just aggregate and project.
        
        output = self.output_proj(invariant)
        return output
```

### 5.3 Enhanced VN-Transformer with Eigenvector Features

```python
# src/models/stage1_spatial/feature_preparation.py
"""
Prepare Gaussian features for VN-Transformer input.
Handles both equivariant (vector) and invariant (scalar) features.
"""

import torch
import torch.nn as nn
from typing import Tuple


class GaussianFeaturePreparation(nn.Module):
    """
    Prepare Gaussian attributes for VN-Transformer.
    
    Separates features into:
    - Type-1 (equivariant): positions, principal axes
    - Scalars (invariant): opacity, eigenvalues, SH band energies
    """
    
    def __init__(
        self,
        use_eigenvectors: bool = True,
        use_positions: bool = True,
        scalar_dim: int = 16,  # eigenvals(3) + opacity(1) + sh_bands(12)
    ):
        super().__init__()
        
        self.use_eigenvectors = use_eigenvectors
        self.use_positions = use_positions
        
        # Number of 3D vector features per Gaussian
        # positions (1) + eigenvectors (3) = 4 vectors max
        self.num_vectors = int(use_positions) + 3 * int(use_eigenvectors)
        
        # Project multiple vectors to single coordinate representation
        if self.num_vectors > 1:
            self.vector_compress = nn.Linear(self.num_vectors, 1)
    
    def forward(
        self,
        positions: torch.Tensor,
        eigenvectors: torch.Tensor,
        scalars: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            positions: (B, M, 3) Gaussian positions
            eigenvectors: (B, M, 3, 3) Principal axes (columns are eigenvectors)
            scalars: (B, M, D_scalar) Scalar features
            
        Returns:
            coors: (B, M, 3) Combined equivariant coordinates
            feats: (B, M, D_scalar) Scalar features (unchanged)
        """
        vectors = []
        
        if self.use_positions:
            vectors.append(positions)  # (B, M, 3)
        
        if self.use_eigenvectors:
            # Extract the 3 principal axes
            for i in range(3):
                vectors.append(eigenvectors[..., i])  # (B, M, 3)
        
        if len(vectors) == 1:
            coors = vectors[0]
        else:
            # Stack and compress
            stacked = torch.stack(vectors, dim=-1)  # (B, M, 3, num_vectors)
            coors = self.vector_compress(stacked).squeeze(-1)  # (B, M, 3)
        
        return coors, scalars
```

---

## 6. Stage 2: Mamba4D Temporal Encoder

### 6.1 Mamba Block Architecture

```python
# src/models/stage2_temporal/mamba_block.py
"""
Mamba block for temporal sequence processing.
Based on: https://github.com/state-spaces/mamba
Adapted from: https://github.com/IRMVLab/Mamba4D
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("Warning: mamba-ssm not installed. Using fallback implementation.")


class MambaBlock(nn.Module):
    """
    Single Mamba block for sequence modeling.
    
    Architecture:
        Input → LayerNorm → Linear → DWConv → SSM → Linear → Residual
                           ↘ Linear → SiLU → Gate ↗
    """
    
    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        """
        Args:
            dim: Input/output dimension
            d_state: SSM state dimension
            d_conv: Depth-wise convolution kernel size
            expand: Expansion factor for inner dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.dim = dim
        self.d_inner = dim * expand
        
        self.norm = nn.LayerNorm(dim)
        
        if MAMBA_AVAILABLE:
            self.mamba = Mamba(
                d_model=dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
        else:
            # Fallback: simple RNN-like processing
            self.mamba = MambaFallback(dim, d_state, d_conv, expand)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) input sequence
            
        Returns:
            y: (B, T, D) output sequence
        """
        residual = x
        x = self.norm(x)
        x = self.mamba(x)
        x = self.dropout(x)
        return x + residual


class MambaFallback(nn.Module):
    """
    Fallback implementation when mamba-ssm is not available.
    Uses GRU as approximation (not truly selective SSM).
    """
    
    def __init__(self, dim: int, d_state: int, d_conv: int, expand: int):
        super().__init__()
        d_inner = dim * expand
        
        self.in_proj = nn.Linear(dim, d_inner * 2)
        self.conv = nn.Conv1d(d_inner, d_inner, d_conv, padding=d_conv-1, groups=d_inner)
        self.gru = nn.GRU(d_inner, d_inner, batch_first=True)
        self.out_proj = nn.Linear(d_inner, dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        
        # Input projection with gating
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        
        # Depthwise conv
        x = x.transpose(1, 2)  # (B, D, T)
        x = self.conv(x)[:, :, :T]  # Causal padding
        x = x.transpose(1, 2)  # (B, T, D)
        
        # GRU (approximates SSM)
        x = F.silu(x)
        x, _ = self.gru(x)
        
        # Gate and project
        x = x * F.silu(z)
        return self.out_proj(x)
```

### 6.2 Temporal Encoder

```python
# src/models/stage2_temporal/mamba_temporal.py
"""
Temporal encoder using stacked Mamba blocks.
Processes sequence of frame embeddings for classification.
"""

import torch
import torch.nn as nn
from typing import Optional

from .mamba_block import MambaBlock


class TemporalMambaEncoder(nn.Module):
    """
    Stage 2: Mamba-based temporal encoder.
    
    Processes sequence of frame embeddings and outputs sequence-level representation.
    """
    
    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 256,
        output_dim: int = 256,
        num_layers: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = False,
    ):
        """
        Args:
            input_dim: Dimension of input frame embeddings
            hidden_dim: Hidden dimension of Mamba blocks
            output_dim: Output dimension
            num_layers: Number of stacked Mamba blocks
            d_state: SSM state dimension
            d_conv: Depthwise conv kernel size
            expand: Expansion factor
            dropout: Dropout rate
            bidirectional: If True, process sequence in both directions
        """
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        
        # Stacked Mamba blocks
        self.layers = nn.ModuleList([
            MambaBlock(
                dim=hidden_dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        
        # Optional bidirectional processing
        self.bidirectional = bidirectional
        if bidirectional:
            self.reverse_layers = nn.ModuleList([
                MambaBlock(
                    dim=hidden_dim,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ])
        
        # Output projection
        out_channels = hidden_dim * 2 if bidirectional else hidden_dim
        self.output_proj = nn.Linear(out_channels, output_dim)
        
        self.norm = nn.LayerNorm(output_dim)
    
    def forward(
        self,
        frame_embeddings: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            frame_embeddings: (B, T, D) sequence of frame embeddings
            mask: (B, T) optional mask for valid frames
            
        Returns:
            output: (B, output_dim) sequence-level embedding
        """
        x = self.input_proj(frame_embeddings)
        
        # Forward pass through Mamba layers
        for layer in self.layers:
            x = layer(x)
        
        if self.bidirectional:
            # Reverse sequence processing
            x_rev = torch.flip(frame_embeddings, dims=[1])
            x_rev = self.input_proj(x_rev)
            
            for layer in self.reverse_layers:
                x_rev = layer(x_rev)
            
            x_rev = torch.flip(x_rev, dims=[1])
            x = torch.cat([x, x_rev], dim=-1)
        
        x = self.output_proj(x)
        x = self.norm(x)
        
        # Aggregate sequence to single embedding
        # Option 1: Last frame
        # output = x[:, -1]
        
        # Option 2: Mean pooling (respecting mask)
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()
            output = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            output = x.mean(dim=1)
        
        return output
```

---

## 7. Classification Head

```python
# src/models/classification_head.py
"""
MLP classification head for binary drone/bird classification.
"""

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """
    MLP classification head.
    
    Architecture: Linear → ReLU → Dropout → Linear → Sigmoid
    """
    
    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        num_classes: int = 1,  # Binary classification
        dropout: float = 0.3,
    ):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_dim) temporal embedding
            
        Returns:
            logits: (B, num_classes) raw logits (apply sigmoid for probabilities)
        """
        return self.mlp(x)
```

---

## 8. Full Model Integration

```python
# src/models/full_model.py
"""
Complete 2-stage 4D Gaussian classification model.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from .stage1_spatial.vn_transformer import GaussianSpatialEncoder
from .stage1_spatial.feature_preparation import GaussianFeaturePreparation
from .stage2_temporal.mamba_temporal import TemporalMambaEncoder
from .classification_head import ClassificationHead


class Gaussian4DClassifier(nn.Module):
    """
    Full 2-stage architecture for 4D Gaussian classification.
    
    Stage 1: VN-Transformer for rotation-invariant spatial encoding
    Stage 2: Mamba for temporal dynamics modeling
    Head: MLP for classification
    """
    
    def __init__(
        self,
        # Stage 1 config
        spatial_dim: int = 128,
        spatial_depth: int = 4,
        spatial_heads: int = 8,
        spatial_dim_feat: int = 64,
        spatial_output_dim: int = 256,
        
        # Stage 2 config
        temporal_hidden_dim: int = 256,
        temporal_num_layers: int = 4,
        temporal_d_state: int = 16,
        temporal_bidirectional: bool = False,
        
        # Classification config
        num_classes: int = 1,
        dropout: float = 0.1,
        
        # Feature config
        use_eigenvectors: bool = True,
    ):
        super().__init__()
        
        # Feature preparation
        self.feature_prep = GaussianFeaturePreparation(
            use_eigenvectors=use_eigenvectors,
            use_positions=True,
        )
        
        # Stage 1: Spatial encoder
        self.spatial_encoder = GaussianSpatialEncoder(
            dim=spatial_dim,
            depth=spatial_depth,
            heads=spatial_heads,
            dim_feat=spatial_dim_feat,
            dropout=dropout,
            output_dim=spatial_output_dim,
        )
        
        # Stage 2: Temporal encoder
        self.temporal_encoder = TemporalMambaEncoder(
            input_dim=spatial_output_dim,
            hidden_dim=temporal_hidden_dim,
            output_dim=temporal_hidden_dim,
            num_layers=temporal_num_layers,
            d_state=temporal_d_state,
            dropout=dropout,
            bidirectional=temporal_bidirectional,
        )
        
        # Classification head
        self.classifier = ClassificationHead(
            input_dim=temporal_hidden_dim,
            hidden_dim=temporal_hidden_dim // 2,
            num_classes=num_classes,
            dropout=dropout,
        )
    
    def forward(
        self,
        positions: torch.Tensor,
        eigenvectors: torch.Tensor,
        scalars: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through full model.
        
        Args:
            positions: (B, T, M, 3) Gaussian positions per frame
            eigenvectors: (B, T, M, 3, 3) Principal axes per frame
            scalars: (B, T, M, D_scalar) Scalar features per frame
            mask: (B, T, M) Boolean mask for valid Gaussians
            
        Returns:
            Dictionary containing:
                - 'logits': (B, num_classes) classification logits
                - 'probabilities': (B, num_classes) sigmoid probabilities
                - 'frame_embeddings': (B, T, D) intermediate frame embeddings
                - 'temporal_embedding': (B, D) final temporal embedding
        """
        B, T, M, _ = positions.shape
        
        # Stage 1: Process each frame through spatial encoder
        frame_embeddings = []
        for t in range(T):
            # Prepare features for this frame
            coors_t, scalars_t = self.feature_prep(
                positions[:, t],
                eigenvectors[:, t],
                scalars[:, t],
            )
            
            mask_t = mask[:, t] if mask is not None else None
            
            # Encode frame
            emb_t = self.spatial_encoder(coors_t, scalars_t, mask_t)
            frame_embeddings.append(emb_t)
        
        # Stack frame embeddings: (B, T, D)
        frame_embeddings = torch.stack(frame_embeddings, dim=1)
        
        # Stage 2: Temporal encoding
        # Create frame-level mask if needed
        frame_mask = None
        if mask is not None:
            frame_mask = mask.any(dim=-1)  # (B, T)
        
        temporal_embedding = self.temporal_encoder(frame_embeddings, frame_mask)
        
        # Classification
        logits = self.classifier(temporal_embedding)
        probabilities = torch.sigmoid(logits)
        
        return {
            'logits': logits,
            'probabilities': probabilities,
            'frame_embeddings': frame_embeddings,
            'temporal_embedding': temporal_embedding,
        }
    
    def predict(self, *args, **kwargs) -> torch.Tensor:
        """Convenience method returning only class probabilities."""
        return self.forward(*args, **kwargs)['probabilities']
```

---

## 9. Training Pipeline

### 9.1 Loss Functions

```python
# src/training/losses.py
"""
Loss functions for training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryFocalLoss(nn.Module):
    """
    Focal loss for binary classification.
    Handles class imbalance by down-weighting easy examples.
    
    FL(p) = -α(1-p)^γ log(p)  for positive class
    FL(p) = -(1-α)p^γ log(1-p)  for negative class
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, 1) raw logits
            targets: (B,) binary labels {0, 1}
        """
        probs = torch.sigmoid(logits).squeeze(-1)
        
        # Focal weights
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        
        # Alpha weighting
        alpha_weight = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        
        # BCE loss
        bce = F.binary_cross_entropy_with_logits(
            logits.squeeze(-1), targets.float(), reduction='none'
        )
        
        loss = alpha_weight * focal_weight * bce
        return loss.mean()


class ClassificationLoss(nn.Module):
    """
    Combined loss for classification.
    """
    
    def __init__(
        self,
        loss_type: str = 'bce',  # 'bce', 'focal'
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        
        if loss_type == 'bce':
            self.criterion = nn.BCEWithLogitsLoss()
        elif loss_type == 'focal':
            self.criterion = BinaryFocalLoss(focal_alpha, focal_gamma)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        self.label_smoothing = label_smoothing
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Apply label smoothing
        if self.label_smoothing > 0:
            targets = targets.float()
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        
        return self.criterion(logits, targets)
```

### 9.2 Trainer

```python
# src/training/trainer.py
"""
Main training loop.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from typing import Dict, Optional
from pathlib import Path
import logging
from tqdm import tqdm

from ..models.full_model import Gaussian4DClassifier
from .losses import ClassificationLoss
from .metrics import ClassificationMetrics


class Trainer:
    """
    Training loop for 4D Gaussian classifier.
    """
    
    def __init__(
        self,
        model: Gaussian4DClassifier,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        device: str = 'cuda',
        checkpoint_dir: Optional[Path] = None,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path('checkpoints')
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Loss
        self.criterion = ClassificationLoss(
            loss_type=config.get('loss_type', 'bce'),
            focal_alpha=config.get('focal_alpha', 0.25),
            focal_gamma=config.get('focal_gamma', 2.0),
        )
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 1e-5),
            betas=config.get('betas', (0.9, 0.999)),
        )
        
        # Scheduler
        total_steps = len(train_loader) * config.get('epochs', 100)
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=config.get('learning_rate', 1e-4),
            total_steps=total_steps,
            pct_start=0.1,
            anneal_strategy='cos',
        )
        
        # Metrics
        self.metrics = ClassificationMetrics()
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
        # Best model tracking
        self.best_val_acc = 0.0
        self.best_epoch = 0
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            # Move to device
            positions = batch['positions'].to(self.device)
            eigenvectors = batch['eigenvectors'].to(self.device)
            scalars = batch['scalars'].to(self.device)
            mask = batch['mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward
            self.optimizer.zero_grad()
            outputs = self.model(positions, eigenvectors, scalars, mask)
            
            # Loss
            loss = self.criterion(outputs['logits'], labels)
            
            # Backward
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config.get('grad_clip', 1.0)
            )
            
            self.optimizer.step()
            self.scheduler.step()
            
            # Track metrics
            total_loss += loss.item()
            preds = (outputs['probabilities'] > 0.5).long().cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().tolist())
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        # Compute epoch metrics
        metrics = self.metrics.compute(all_preds, all_labels)
        metrics['loss'] = total_loss / len(self.train_loader)
        
        return metrics
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validation loop."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        for batch in tqdm(self.val_loader, desc="Validating"):
            positions = batch['positions'].to(self.device)
            eigenvectors = batch['eigenvectors'].to(self.device)
            scalars = batch['scalars'].to(self.device)
            mask = batch['mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            outputs = self.model(positions, eigenvectors, scalars, mask)
            
            loss = self.criterion(outputs['logits'], labels)
            total_loss += loss.item()
            
            probs = outputs['probabilities'].cpu()
            preds = (probs > 0.5).long()
            
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().tolist())
            all_probs.extend(probs.tolist())
        
        metrics = self.metrics.compute(all_preds, all_labels, all_probs)
        metrics['loss'] = total_loss / len(self.val_loader)
        
        return metrics
    
    def train(self, epochs: int):
        """Full training loop."""
        for epoch in range(1, epochs + 1):
            # Train
            train_metrics = self.train_epoch(epoch)
            self.logger.info(f"Epoch {epoch} Train: {train_metrics}")
            
            # Validate
            val_metrics = self.validate()
            self.logger.info(f"Epoch {epoch} Val: {val_metrics}")
            
            # Save best model
            if val_metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['accuracy']
                self.best_epoch = epoch
                self.save_checkpoint(epoch, is_best=True)
            
            # Regular checkpoint
            if epoch % self.config.get('save_every', 10) == 0:
                self.save_checkpoint(epoch)
        
        self.logger.info(f"Training complete. Best accuracy: {self.best_val_acc:.4f} at epoch {self.best_epoch}")
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'config': self.config,
        }
        
        path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(state, path)
        
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(state, best_path)
```

### 9.3 Metrics

```python
# src/training/metrics.py
"""
Classification metrics.
"""

from typing import Dict, List, Optional
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


class ClassificationMetrics:
    """Compute classification metrics."""
    
    def compute(
        self,
        predictions: List[int],
        labels: List[int],
        probabilities: Optional[List[float]] = None,
    ) -> Dict[str, float]:
        """
        Compute classification metrics.
        
        Returns:
            Dictionary with accuracy, precision, recall, f1, and optionally AUC
        """
        preds = np.array(predictions)
        labels = np.array(labels)
        
        metrics = {
            'accuracy': accuracy_score(labels, preds),
            'precision': precision_score(labels, preds, zero_division=0),
            'recall': recall_score(labels, preds, zero_division=0),
            'f1': f1_score(labels, preds, zero_division=0),
        }
        
        if probabilities is not None:
            probs = np.array(probabilities)
            try:
                metrics['auc'] = roc_auc_score(labels, probs)
            except ValueError:
                metrics['auc'] = 0.0
        
        return metrics
```

---

## 10. Inference Pipeline

```python
# src/inference/predictor.py
"""
Inference wrapper for deployment.
"""

import torch
from pathlib import Path
from typing import List, Dict, Union
import time

from ..data.ply_parser import parse_ply
from ..data.feature_extractor import GaussianFeatureExtractor
from ..models.full_model import Gaussian4DClassifier


class Predictor:
    """
    Inference wrapper for 4D Gaussian classifier.
    """
    
    def __init__(
        self,
        checkpoint_path: Path,
        device: str = 'cuda',
        max_gaussians: int = 50000,
    ):
        """
        Args:
            checkpoint_path: Path to model checkpoint
            device: Device to run inference on
            max_gaussians: Maximum Gaussians per frame (subsample if exceeded)
        """
        self.device = device
        self.max_gaussians = max_gaussians
        
        # Load model
        checkpoint = torch.load(checkpoint_path, map_location=device)
        config = checkpoint['config']
        
        self.model = Gaussian4DClassifier(**config['model'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        
        # Feature extractor
        self.feature_extractor = GaussianFeatureExtractor()
    
    @torch.no_grad()
    def predict_sequence(
        self,
        ply_paths: List[Path],
        return_embeddings: bool = False,
    ) -> Dict[str, Union[float, torch.Tensor]]:
        """
        Predict class for a sequence of PLY files.
        
        Args:
            ply_paths: List of paths to PLY files (ordered temporally)
            return_embeddings: Whether to return intermediate embeddings
            
        Returns:
            Dictionary with:
                - 'probability': float, probability of drone class
                - 'prediction': int, 0=bird, 1=drone
                - 'latency_ms': float, inference time in milliseconds
                - 'frame_embeddings': optional, (T, D) tensor
                - 'temporal_embedding': optional, (D,) tensor
        """
        start_time = time.time()
        
        # Load and extract features
        clouds = [parse_ply(p, device=self.device) for p in ply_paths]
        batch = self.feature_extractor.extract_batch(
            clouds,
            max_gaussians=self.max_gaussians,
            subsample_strategy='random',
        )
        
        # Add batch dimension and move to device
        positions = batch['positions'].unsqueeze(0).to(self.device)
        eigenvectors = batch['eigenvectors'].unsqueeze(0).to(self.device)
        scalars = batch['scalars'].unsqueeze(0).to(self.device)
        mask = batch['mask'].unsqueeze(0).to(self.device)
        
        # Forward pass
        outputs = self.model(positions, eigenvectors, scalars, mask)
        
        latency_ms = (time.time() - start_time) * 1000
        
        result = {
            'probability': outputs['probabilities'].item(),
            'prediction': int(outputs['probabilities'].item() > 0.5),
            'latency_ms': latency_ms,
        }
        
        if return_embeddings:
            result['frame_embeddings'] = outputs['frame_embeddings'].squeeze(0).cpu()
            result['temporal_embedding'] = outputs['temporal_embedding'].squeeze(0).cpu()
        
        return result
    
    @torch.no_grad()
    def predict_batch(
        self,
        sequences: List[List[Path]],
    ) -> List[Dict]:
        """Batch prediction for multiple sequences."""
        return [self.predict_sequence(seq) for seq in sequences]
```

---

## 11. Configuration System

```yaml
# configs/default.yaml

# Model configuration
model:
  # Stage 1: Spatial encoder
  spatial_dim: 128
  spatial_depth: 4
  spatial_heads: 8
  spatial_dim_feat: 64
  spatial_output_dim: 256
  
  # Stage 2: Temporal encoder
  temporal_hidden_dim: 256
  temporal_num_layers: 4
  temporal_d_state: 16
  temporal_bidirectional: false
  
  # Classification
  num_classes: 1
  dropout: 0.1
  
  # Features
  use_eigenvectors: true

# Data configuration
data:
  train_root: data/train
  val_root: data/val
  test_root: data/test
  labels_file: data/labels.json
  
  sequence_length: 30
  stride: 1
  max_gaussians: 50000
  subsample_strategy: random
  
  batch_size: 8
  num_workers: 4
  cache_sequences: false

# Training configuration
training:
  epochs: 100
  learning_rate: 0.0001
  weight_decay: 0.00001
  grad_clip: 1.0
  
  loss_type: focal  # bce, focal
  focal_alpha: 0.25
  focal_gamma: 2.0
  label_smoothing: 0.1
  
  save_every: 10
  
  # Early stopping
  patience: 20
  min_delta: 0.001

# Hardware
device: cuda
seed: 42
deterministic: true

# Logging
logging:
  level: INFO
  wandb:
    enabled: true
    project: gaussian4d-classifier
    entity: null
```

---

## 12. Testing Strategy

### 12.1 Equivariance/Invariance Tests

```python
# tests/test_vn_equivariance.py
"""
Test rotation equivariance and invariance properties.
"""

import torch
import pytest
from scipy.spatial.transform import Rotation

from src.models.stage1_spatial.vn_transformer import GaussianSpatialEncoder


def random_rotation_matrix(device='cpu'):
    """Generate random rotation matrix."""
    rot = Rotation.random()
    R = torch.from_numpy(rot.as_matrix()).float().to(device)
    return R


def test_frame_embedding_rotation_invariance():
    """
    Test that frame embeddings are invariant to input rotation.
    
    z(R·G) = z(G) for any rotation R
    """
    torch.manual_seed(42)
    
    model = GaussianSpatialEncoder(dim=64, depth=2, heads=4)
    model.eval()
    
    # Random input
    B, M = 1, 100
    positions = torch.randn(B, M, 3)
    scalars = torch.randn(B, M, 16)
    
    # Original embedding
    with torch.no_grad():
        emb_original = model(positions, scalars)
    
    # Rotate positions
    R = random_rotation_matrix()
    positions_rotated = positions @ R.T
    
    # Rotated embedding
    with torch.no_grad():
        emb_rotated = model(positions_rotated, scalars)
    
    # Should be approximately equal
    diff = (emb_original - emb_rotated).abs().max()
    assert diff < 1e-4, f"Embeddings differ by {diff}"


def test_output_consistency():
    """Test model produces consistent output shapes."""
    model = GaussianSpatialEncoder(dim=64, depth=2, output_dim=128)
    
    B, M = 4, 500
    positions = torch.randn(B, M, 3)
    scalars = torch.randn(B, M, 16)
    
    output = model(positions, scalars)
    
    assert output.shape == (B, 128)
```

### 12.2 Full Pipeline Test

```python
# tests/test_full_pipeline.py
"""
End-to-end pipeline tests.
"""

import torch
import pytest

from src.models.full_model import Gaussian4DClassifier


def test_full_model_forward():
    """Test full model forward pass."""
    model = Gaussian4DClassifier(
        spatial_dim=64,
        spatial_depth=2,
        spatial_heads=4,
        spatial_output_dim=128,
        temporal_hidden_dim=128,
        temporal_num_layers=2,
    )
    
    B, T, M = 2, 10, 100
    positions = torch.randn(B, T, M, 3)
    eigenvectors = torch.randn(B, T, M, 3, 3)
    scalars = torch.randn(B, T, M, 16)
    mask = torch.ones(B, T, M, dtype=torch.bool)
    
    outputs = model(positions, eigenvectors, scalars, mask)
    
    assert 'logits' in outputs
    assert 'probabilities' in outputs
    assert outputs['logits'].shape == (B, 1)
    assert outputs['probabilities'].shape == (B, 1)
    assert (outputs['probabilities'] >= 0).all()
    assert (outputs['probabilities'] <= 1).all()


def test_model_gradient_flow():
    """Test gradients flow through entire model."""
    model = Gaussian4DClassifier(
        spatial_dim=32,
        spatial_depth=1,
        temporal_num_layers=1,
    )
    
    B, T, M = 1, 5, 50
    positions = torch.randn(B, T, M, 3, requires_grad=True)
    eigenvectors = torch.randn(B, T, M, 3, 3)
    scalars = torch.randn(B, T, M, 16)
    mask = torch.ones(B, T, M, dtype=torch.bool)
    
    outputs = model(positions, eigenvectors, scalars, mask)
    loss = outputs['logits'].sum()
    loss.backward()
    
    assert positions.grad is not None
    assert not torch.isnan(positions.grad).any()
```

---

## 13. Implementation Phases

### Phase 1: Foundation (Week 1-2)

**Goals:**
- Set up project structure and environment
- Implement PLY parser and data pipeline
- Create basic tests

**Deliverables:**
1. `ply_parser.py` - Complete and tested
2. `feature_extractor.py` - Feature preparation working
3. `gaussian_dataset.py` - Dataset class loading sequences
4. Environment setup scripts working

**Verification:**
- [ ] Can load PLY file and extract all Gaussian attributes
- [ ] Can parse sequence of PLYs into batched tensors
- [ ] Unit tests pass for data modules

### Phase 2: Stage 1 - Spatial Encoder (Week 3-4)

**Goals:**
- Integrate VN-Transformer package
- Implement feature preparation
- Implement frame-level pooling

**Deliverables:**
1. `vn_transformer.py` - VN-Transformer wrapper
2. `feature_preparation.py` - Equivariant/invariant feature separation
3. `frame_pooling.py` - Attention pooling

**Verification:**
- [ ] VN-Transformer produces output of correct shape
- [ ] Frame embeddings are rotation-invariant (test passes)
- [ ] Memory usage acceptable for 100k+ Gaussians

### Phase 3: Stage 2 - Temporal Encoder (Week 5-6)

**Goals:**
- Implement Mamba blocks
- Implement temporal encoder
- Integrate stages

**Deliverables:**
1. `mamba_block.py` - Single Mamba block
2. `mamba_temporal.py` - Stacked temporal encoder
3. `full_model.py` - Combined architecture

**Verification:**
- [ ] Mamba blocks run without errors
- [ ] Full model forward pass works
- [ ] Gradients flow through entire model

### Phase 4: Training Pipeline (Week 7-8)

**Goals:**
- Implement training loop
- Set up logging and checkpointing
- Run initial experiments

**Deliverables:**
1. `trainer.py` - Full training loop
2. `losses.py` - Loss functions
3. `metrics.py` - Evaluation metrics
4. Training scripts

**Verification:**
- [ ] Training loop runs without errors
- [ ] Checkpoints save/load correctly
- [ ] Metrics are computed correctly

### Phase 5: Integration & Optimization (Week 9-10)

**Goals:**
- End-to-end testing
- Performance optimization
- Documentation

**Deliverables:**
1. `predictor.py` - Inference wrapper
2. Optimized data loading
3. Complete documentation
4. Performance benchmarks

**Verification:**
- [ ] End-to-end inference works
- [ ] Latency targets met or measured
- [ ] All tests pass

---

## Appendix A: Key Hyperparameters

| Parameter | Recommended Value | Notes |
|-----------|-------------------|-------|
| `spatial_dim` | 128 | VN-Transformer hidden dim |
| `spatial_depth` | 4 | Number of VN-Transformer layers |
| `spatial_heads` | 8 | Attention heads |
| `temporal_num_layers` | 4 | Number of Mamba blocks |
| `d_state` | 16 | SSM state dimension |
| `sequence_length` | 30 | Frames per sample |
| `max_gaussians` | 50,000 | Subsample threshold |
| `learning_rate` | 1e-4 | Initial LR |
| `batch_size` | 8 | Adjust based on GPU memory |

## Appendix B: Expected Memory Usage

For reference, approximate memory estimates:

| Component | Memory (GB) | Notes |
|-----------|-------------|-------|
| VN-Transformer (50k points) | ~4-6 | Single frame |
| Mamba Temporal (30 frames) | ~1-2 | Sequence |
| Gradients | 2× forward | During training |
| **Total Training** | ~12-16 | With batch size 8 |

## Appendix C: References

1. VN-Transformer: [arXiv:2206.04176](https://arxiv.org/abs/2206.04176)
2. Vector Neurons: [arXiv:2104.12229](https://arxiv.org/abs/2104.12229)
3. Mamba: [arXiv:2312.00752](https://arxiv.org/abs/2312.00752)
4. Mamba4D: [arXiv:2405.14338](https://arxiv.org/abs/2405.14338)
5. 3D Gaussian Splatting: [arXiv:2308.04079](https://arxiv.org/abs/2308.04079)

---

*Document Version: 1.0*
*Generated for: Rotation-Robust 4D Gaussian Classification Project*
