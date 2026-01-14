# Background and Related Work: Chapter Outline

## Chapter 2: Background (Foundations)
**Purpose:** Equip readers with technical prerequisites  
**Length:** 5–8 pages | **Tone:** Tutorial-style

### 2.1 Multi-View Geometry Fundamentals
- Camera models (pinhole, intrinsic/extrinsic parameters)
- Epipolar geometry & the fundamental matrix
- Triangulation and 3D point recovery
- Visual hulls and silhouette-based reconstruction
- *Connection:* Why N≥3 cameras with overlapping FOV enables robust 3D inference

### 2.2 3D Gaussian Splatting Formulation
- Gaussian primitives: position μ, covariance Σ (scale s, rotation q), opacity α, color (SH coefficients)
- Differentiable rasterization via tile-based splatting
- EWA (Elliptical Weighted Average) volume splatting projection
- Alpha-compositing equation with transmittance
- Loss function: L1 + D-SSIM (λ=0.2)
- Optimization-based vs. feed-forward paradigms (brief contrast)
- *Connection:* Why Gaussians suit real-time reconstruction of small distant objects

### 2.3 Deep Learning Primitives
- Self-attention and Transformer architecture (brief)
- SE(3) group and equivariance concepts
- Permutation invariance for set/point cloud processing
- **State Space Models (SSMs):** Mamba's selective state space formulation
  - Linear complexity O(N) vs Transformer O(N²)
  - *Relevance:* Foundation for edge-deployable temporal modeling
- *Connection:* Foundation for understanding the 4D classifier architecture

### 2.4 Synthetic Data for Domain Transfer
- Simulation-to-real gap
- Domain randomization principles
- *Connection:* Justifies Isaac Sim pipeline design choices

---

## Chapter 3: Related Work (Literature Review)
**Purpose:** Position thesis; justify methods; identify gaps  
**Length:** 12–18 pages | **Tone:** Critical, comparative

---

### 3.1 Detection & Tracking of Flying Objects (Pillar I)
**~4 pages** | *Supports RQ1*

#### 3.1.1 Detection Paradigms: DBT vs. TBD
- **Detect-Before-Track (DBT):** Faster R-CNN, YOLO series (YOLOv8-10), DETR
  - Strengths: mature, real-time, rich features
  - Weakness: depends on training input (drones change daily), fails on low-SNR, few-pixel targets
- **Track-Before-Detect (TBD):** Particle filters, dynamic programming approaches
  - DP-TBD: detection under signal-to-clutter ratios below 1.5
  - PF-TBD: non-linear motion handling, probabilistic state estimation
  - GLMB TBD: multi-target trajectories with lower computational complexity
  - Strengths: accumulates weak evidence over time
  - Weakness: computational cost, tuning complexity
- **Gap:** Neither paradigm inherently leverages multi-view 3D constraints

#### 3.1.2 Foreground Segmentation
- **Background subtraction:** GMM/MOG2, ViBe
  - ViBe: sample-based modeling, 200 fps, conservative update policies
  - Challenge: sky background lacks stable statistics
- **Motion-based:** Frame differencing, optical flow (Lucas-Kanade → Farneback → RAFT)
  - SEA-RAFT (ECCV 2024): 2.3× faster inference, 22.9% error reduction on Spring benchmark
  - Trade-off: accuracy vs. speed
- **Promptable segmentation:** SAM1/2/3 (https://ai.meta.com/sam3/)– 1+ billion masks training, needs prompt and is still an estimate; might miss critical features (propellors / bird feet)
- **Gap:** 2D segmentation propagates errors to 3D; need multi-view-aware approach

#### 3.1.3 Multi-View Fusion Strategies
- **Early Fusion (3D-first):** BEV representations, occupancy grids, voxel projections
  - Lift-Splat-Shoot (LSS): discrete depth distributions, scatter to BEV
  - BEVFormer (Li et al., ECCV 2022): spatiotemporal transformers, 56.9% NDS on nuScenes
  - BEVFusion (ICRA 2023): unified multi-modal features in shared BEV space
  - RCBEVDet (CVPR 2024): radar-camera fusion, 21-28 FPS real-time
  - Pro: geometry-consistent; Con: memory-heavy
- **Late Fusion (2D-first):** Per-view detection → triangulation/epipolar matching
  - Pro: leverages mature 2D detectors; Con: error accumulation
- **Volumetric approaches:** Ray-marching voting, space carving
- **Gap:** Most methods assume textured scenes or dense views

#### 3.1.4 Multi-Object Tracking (MOT)
- Classical: Kalman filter + Hungarian algorithm (SORT baseline, 260 Hz)
- Modern: DeepSORT (appearance + motion fusion), ByteTrack (low-confidence association)
- OC-SORT (CVPR 2023): observation-centric momentum, 62.1 HOTA on MOT20
- MVTrajecter (ICCV 2025): trajectory motion + appearance costs, 94.3 MOTA on Wildtrack
- Transformer-based trackers (higher computational cost)
- 3D MOT extensions
- **Gap:** Limited work on 3D MOT for aerial targets from ground cameras

#### 3.1.5 Section Summary & Positioning
- Thesis approach: TBD-inspired ray-marching voting with temporal accumulation
- Bridge to next section: Detection provides input to reconstruction

---

### 3.2 3D Reconstruction Methods (Pillar II)
**~5 pages** | *Supports RQ2*

#### 3.2.1 Classical Multi-View Reconstruction
- Structure from Motion: COLMAP, bundle adjustment (reprojection error minimization)
- Multi-View Stereo: dense stereo, PatchMatch
- Visual hull / silhouette-based methods
- **Limitation:** Require texture; slow; not real-time

#### 3.2.2 Neural Radiance Fields (NeRF)
- Original NeRF: volume rendering, positional encoding
- Quality improvements: Mip-NeRF, anti-aliasing
- Generalizable NeRF: PixelNeRF (feed-forward, pixel-aligned CNN features)
- MVSNeRF: MVS-style cost volumes from 3 input views
- Acceleration: Instant-NGP (multiresolution hash encoding, seconds-level training)
- **Limitation:** Slow rendering; per-scene optimization; struggles with sparse views

#### 3.2.3 3D Gaussian Splatting Revolution
- **Optimization-based 3DGS:** Kerbl et al. (2023)
  - Differentiable rasterization, adaptive densification
  - Real-time rendering (≥100 FPS vs NeRF's 1-5 FPS), but per-scene optimization
- **Feed-forward / Generalizable GS:**
  - pixelSplat (CVPR 2024 Best Paper Runner-Up): epipolar transformer, probabilistic depth
  - MVSplat (ECCV 2024 Oral): cost-volume depth, 10× fewer parameters, 22 FPS
  - Splatter Image (CVPR 2024): single-view, 38 FPS, purely 2D operators
  - GPS-Gaussian (CVPR 2024 Highlight): Gaussian parameter maps, stereo matching
  - GS-LRM: transformer-based per-pixel prediction, ~0.23s on A100
  - FreeSplatter: GS-LRM but without camera intrinsics & extrinsics input requirement
  - UFV-Splatter: Followup from FreeSplatter uses LoRa and adaption module for reconstruction. Gave best results in experiments with 5-view input photos of a drone
  - SAM 3D: SOTA with exceptional performance - single image input & needs prompted input (premask/click/text)
- **Trade-offs:** Speed vs. quality vs. view requirements

#### 3.2.3.1 Textureless and Sparse-View Limitations
- **GaussianPro:** "SfM fails to produce 3D points in textureless regions... densification struggles with poor initialization"
- **ProSplat:** "Performance significantly degrades in wide-baseline scenarios due to limited texture details"
- **VolSplat:** "When depth cues are weak, 3D predictions suffer from depth ambiguities"
- **Geometric foundation models (VGGT, Depth Anything 3):**
  - Trained on large-scale scene data for dense reconstruction/depth
  - Struggle with isolated objects against featureless backgrounds (sky)
  - Produce fragmented, noisy geometry for small flying objects
  - *Contrast:* Object-centric methods (GS-LRM) handle isolated objects better but assume close-range, textured subjects
- **MASt3R:** geometric foundation model for dense initialization in textureless regions, fail on low SNR symmetric objects for feature matching
- **Gap:** Feed-forward GS designed for indoor/object-centric scenes; struggle with:
  - Textureless sky backgrounds
  - Wide baselines
  - Small distant objects (few pixels)

#### 3.2.4 Dynamic and 4D Gaussian Splatting
- 4DGS (Wu et al., CVPR 2024): HexPlane-inspired encoding, deformation field (Δμ, Δq, Δs), 82 FPS at 800×800
- Deformable 3D Gaussians: canonical space + deformation field with annealing smoothing
- Spacetime Gaussians: temporal opacity, polynomial motion trajectories
- **Relevance:** Enables temporal consistency in reconstruction
- **Gap:** All 4D methods focus on rendering quality—none extract motion as discriminative features for classification

#### 3.2.5 Depth Estimation as Auxiliary
- Monocular depth priors (MiDaS, DPT)
- Cost-volume / plane-sweep stereo
- Role in feed-forward GS initialization: using depth in gaussian splatting is proven to improve PSNR in reconstructed NVS

#### 3.2.6 Section Summary & Positioning
- Thesis approach: Feed-forward GS adapted for sparse wide-baseline aerial surveillance
- Bridge: Reconstructed Gaussians become input tokens for classification

---

### 3.3 Classification from 3D Representations (Pillar III) ⭐
**1 page per subsection** | *Supports RQ3 (main contribution)*

#### 3.3.1 Point Cloud Classification (Precursor Methods)
- **PointNet:** Per-point MLP + global pooling; permutation invariance
- **PointNet++:** Hierarchical set abstraction, local features
- **DGCNN / EdgeConv:** Dynamic graph construction, local geometry
- **Point Transformers:** PCT, self-attention on points
- **Point Transformer V3** (CVPR 2024): serialized neighborhood mapping (space-filling curves), 3.3× faster, 10.2× lower memory, 1st place Waymo Challenge 2024
- **Point-MAE:** Self-supervised pre-training
- **Relevance:** Gaussian "clouds" share structure with point clouds
- **Gap:** Treat geometry only; ignore appearance (SH) and temporal dynamics

#### 3.3.2 Set Processing and Gaussian Tokenization
- **DeepSets:** Permutation-invariant set functions: f(X) = ρ(Σᵢφ(xᵢ))
- **Set Transformer:** Attention-based set aggregation with inducing points, O(nm) complexity
- **Perceiver** (Jaegle et al., ICML 2021): fixed-size latent arrays for variable input sizes
  - *Relevance:* Directly applicable to variable Gaussian counts per object
- **Gaussians as tokens:** Each Gaussian → feature vector (μ, s, q, α, SH)
- **Aggregation strategies:** FPS sampling, attention pooling, learned queries
- **Gap:** No established "Gaussian classification" paradigm exists

#### 3.3.3 Temporal Sequence Modeling
- **RNN/LSTM:** Sequential hidden state; vanishing gradients
- **Temporal Convolutional Networks (TCN):** Causal convolutions; fixed receptive field
- **Transformers for sequences:** TimeSformer, Video Transformers (ViViT)
  - Factorized attention (space-time)
- **State Space Models:**
  - Mamba (Gu & Dao, Dec 2023): selective SSM, input-dependent transitions, linear-time inference
  - PointMamba (NeurIPS 2024): first Mamba backbone for point clouds, space-filling curve tokenization
  - Mamba3D (ACM MM 2024): Local Norm Pooling + bidirectional SSM, 92.6% accuracy on ScanObjectNN
  - Mamba4D (2024): first 4D point cloud backbone, disentangled spatial-temporal blocks
  - *Relevance:* O(N) complexity critical for Jetson edge deployment
- **Application to 4D Gaussians:** Temporal attention over Gaussian sequences
- **Gap:** No prior work on temporal Gaussian sequences for classification

#### 3.3.4 SE(3) Equivariance for 3D Data
- **Why equivariance matters:** Drones/birds can appear in arbitrary orientations
- **Tensor Field Networks (TFN):** Spherical harmonic convolutions (Clebsch-Gordan products)
- **SE(3)-Transformers:** Attention with SE(3)-invariant weights, equivariant value embeddings
- **Vector Neurons** (Deng et al., ICCV 2021): SO(3)-equivariant linear layers
  - VN-ReLU: projection along learned directions
- **EGNN** (Satorras et al., ICML 2021): Message passing with coordinate updates
  - Only squared distances (E(n)-invariant) and radial directions preserve equivariance
  - Dramatically reduced cost compared to TFN
- **Equiformer** (ICLR 2023 Spotlight): equivariant attention + irreps features, SOTA on QM9/OC20
- **Alternatives:** Data augmentation, pose canonicalization (PCA alignment)
- **Trade-off:** Equivariance vs. computational cost vs. expressivity

| Method | Complexity | Expressivity | Edge Suitability |
|--------|------------|--------------|------------------|
| TFN | High | Very High | No |
| SE(3)-Transformer | High | Very High | No |
| EGNN | Low | Moderate | **Yes** |
| Vector Neurons | Low | Moderate | **Yes** |

- **Thesis relevance:** Critical for orientation-invariant drone classification

#### 3.3.5 Motion Signatures for Flying Object Discrimination
- **Position dynamics:** Velocity profiles, acceleration patterns, trajectory smoothness
- **Trajectory analysis with SVM** (Srigrarom et al., 2020): heading angle, curvature, acceleration → 85% accuracy
- **Shape dynamics:** Scale/rotation changes over time; articulation (wings)
- **Frequency analysis:** FFT of motion signals
  - Flapping frequency: birds 2.5–30 Hz (typically 4–6 Hz)
  - Rotor frequency: drones >50 Hz (>6000 RPM)
  - Key discriminator: drone signatures **symmetric** (blades toward/away simultaneously), bird signatures **asymmetric** (wings move together)
  - Micro-Doppler literature (radar domain, 96% accuracy) → RGB analogue
- **Biometric flight patterns:** Gliding vs. hovering vs. maneuvering
- **Gap:** Most work uses radar/acoustic; limited RGB-based motion classification

#### 3.3.6 Existing Drone vs. Bird Classification Systems
- Radar-based approaches (micro-Doppler spectrograms, 96% accuracy)
- Acoustic signature methods
- Thermal imaging systems
- RGB single-camera approaches (YOLO + tracking heuristics)
- **Gaussian-based classification (only existing work):**
  - "Mitigating Ambiguities in 3D Classification with Gaussian Splatting" (CVPR 2025): first GS classification, scale/rotation characterize surfaces, opacity represents transparency—**static objects only**
  - ShapeSplat (3DV 2025 Oral): 206K Gaussian objects, 87 categories, Gaussian-MAE pretraining—**static objects only**
  - Finding: "distribution of optimized GS centroids significantly differs from point clouds"—naïve centroid use degrades performance (flat surfaces use less gaussians -> could be mitigated by making the feedforward GS model to generate gaussians with a max size?)
- **Semantic Gaussian methods:**
  - SceneSplat, Locate3D: scene-level segmentation and localization from Gaussian representations
  - Focus: dense scene understanding, indoor/outdoor environments
  - **Gap:** Scene-level semantics, not object-level classification; no temporal dynamics
- **Gap:** No prior work combining:
  - Multi-view 3D reconstruction (Gaussians)
  - Temporal dynamics of 3D representations
  - SE(3)-equivariant classification

#### 3.3.7 Section Summary & Positioning
- **Thesis contribution:** Novel 4D Gaussian classification architecture
  - Gaussians tokenized with geometric + appearance features
  - SE(3)-equivariant processing for orientation invariance
  - Temporal Transformer/SSM for motion dynamics
  - End-to-end trainable on synthetic data with sim-to-real transfer

---

### 3.4 Synthetic Data and Sim-to-Real Transfer (Supporting Topic)
**~2 pages** | *Supports training pipeline*

#### 3.4.1 Synthetic Data Generation for Vision
- Simulation platforms: Isaac Sim, AirSim, Blender
- Domain randomization strategies
- Photorealistic rendering vs. stylization

#### 3.4.2 Sim-to-Real Gap Mitigation
- Domain adaptation techniques
- Style transfer approaches
- Curriculum learning from sim to real

#### 3.4.3 Flying Object Datasets
- Existing drone/bird datasets (limitations)
- Drone-vs-Bird Challenge findings: moving cameras and distant objects remain challenging
- Gap: No multi-view synchronized dataset with 3D ground truth

---

### 3.5 Chapter Summary and Research Gaps
**~1 page**

| Gap | How Thesis Addresses |
|-----|---------------------|
| TBD paradigms lack 3D awareness | Ray-marching voting across views |
| Feed-forward GS fails on sky/sparse views | Adapted architecture + training |
| No Gaussian-based classification exists | Novel 4D tokenization + classifier |
| Existing GS classification limited to static objects | Temporal Gaussian dynamics as discriminative features |
| Temporal dynamics unexploited for classification | SE(3)-equivariant temporal Transformer/SSM |
| No multi-view flying object benchmark | Isaac Sim synthetic dataset |

**Transition:** "The following chapter presents our methodology, addressing each identified gap..."

---

## Visual Aids to Include

| Location | Suggested Figure/Table |
|----------|----------------------|
| §2.2 | Diagram: Gaussian primitive parameters + EWA projection |
| §3.1.3 | Figure: Early vs. Late fusion pipelines |
| §3.2.3 | Table: Comparison of feed-forward GS methods (views, speed, quality) |
| §3.2.3.1 | Table: Documented textureless/sparse-view failure modes |
| §3.2.3.1 | Figure: VGGT/DA3 failure on isolated flying objects (fragmented geometry) |
| §3.3.1 | Figure: PointNet → PointNet++ → Point Transformer V3 evolution |
| §3.3.3 | Table: Temporal modeling approaches (Transformer vs. Mamba complexity) |
| §3.3.4 | Diagram: SE(3) equivariance concept + method comparison table |
| §3.3.5 | Figure: Flapping vs. rotor frequency spectra (symmetric vs. asymmetric) |
| §3.5 | Table: Research gaps summary (as above) |

---

This structure ensures your main contribution (4D temporal classification) receives the deepest treatment while providing comprehensive context for the full pipeline. Each section explicitly identifies gaps that your thesis addresses.