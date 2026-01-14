Multi-View Flying Object Pipeline
├── Detection & Tracking
│   ├── DBT / TBD Paradigms
│   │   ├── DBT Detect-Before-Track
│   │   │   ├── Faster R-CNN
│   │   │   ├── YOLO
│   │   │   └── DETR
│   │   └── TBD Track-Before-Detect
│   │       ├── Particle Filter
│   │       └── Dynamic Programming
│   ├── Foreground Segmentation
│   │   ├── Background Subtraction
│   │   │   └── ViBe
│   │   └── Motion-Based Detection
│   │       ├── Frame Differencing
│   │       └── Optical Flow
│   │           ├── Lucas-Kanade
│   │           ├── Farneback
│   │           └── RAFT
│   ├── Multi-View Fusion
│   │   ├── Early Fusion 3D-First
│   │   │   ├── BEV Representations
│   │   │   └── Occupancy Grids
│   │   └── Late Fusion 2D-First
│   │       ├── Epipolar Constraints
│   │       └── Triangulation
│   ├── Volumetric Approaches
│   │   ├── Ray-Marching Voting
│   │   └── Space Carving
│   │       ├── Octrees
│   │       └── Voxel Grids
│   └── Multi-Object Tracking
│       ├── SORT Family
│       │   ├── SORT
│       │   ├── DeepSORT
│       │   └── ByteTrack
│       └── State Estimation
│           ├── Kalman Filter
│           └── Particle Filter
├── 3D Reconstruction
│   ├── Classical Methods
│   │   ├── SfM COLMAP
│   │   │   ├── Bundle Adjustment
│   │   │   └── Feature Matching
│   │   ├── MVS Dense Stereo
│   │   │   ├── Patch-based
│   │   │   └── Point Clouds
│   │   └── Visual Hull Silhouette-based
│   ├── Neural Radiance Fields
│   │   ├── Original NeRF 2020
│   │   │   ├── Volume Rendering
│   │   │   └── Mip-NeRF
│   │   ├── Generalizable PixelNeRF
│   │   └── Accelerated Instant-NGP
│   │       └── Hash Encoding
│   ├── Gaussian Splatting
│   │   ├── 4D / Dynamic Temporal GS
│   │   │   └── 4DGS 2024
│   │   ├── Optimization 3DGS (2023)
│   │   │   ├── Differentiable Rasterization
│   │   │   └── Adaptive Density
│   │   └── Feed-Forward Generalizable
│   │       ├── pixelSplat 2024
│   │       └── MVSplat 2024
│   └── Depth Estimation
│       ├── Monocular Learned Prior
│       └── Cost Volume Plane-Sweep
└── Classification (4D Temporal)
    ├── Point Cloud Processing
    │   ├── PointNet Family
    │   │   ├── PointNet 2017
    │   │   │   ├── Max Pool Symmetry
    │   │   │   └── T-Net Alignment
    │   │   └── PointNet++ Hierarchical
    │   │       ├── Set Abstraction
    │   │       └── FPS + Ball Query
    │   ├── Point Transformers
    │   │   ├── PCT 2021
    │   │   └── Point-MAE Self-supervised
    │   └── Graph Networks
    │       └── DGCNN EdgeConv
    ├── Gaussian Tokenization
    │   ├── Feature Extraction
    │   │   ├── Appearance SH coeffs
    │   │   └── Geometry (μ, Σ)
    │   ├── Set Aggregation
    │   │   ├── Attention Pooling
    │   │   └── DeepSets Symmetric Fn
    │   │       └── Sum Pool Symmetry
    │   └── Permutation Invariance
    ├── Temporal Modeling
    │   ├── RNN/LSTM
    │   ├── Transformers
    │   │   ├── TimeSformer
    │   │   └── ViViT
    │   └── Spatio-Temporal GNNs
    ├── SE(3) Equivariance
    │   ├── SO(3) Representations
    │   ├── Spherical CNNs
    │   └── Alternatives to Equivariance
    │       ├── Data Augmentation
    │       └── Canonical-ization
    └── Motion Signatures
        ├── Velocity Δμ/Δt
        ├── Acceleration Δ²μ/Δt²
        └── 4D Space- Time Volume
            └── Drones: Rigid, High-freq; Birds: Deformable

Orphans
└── GMM/MOG2
