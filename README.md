# Installation
export TORCH_CUDA_ARCH_LIST="9.0"
conda env create -f environment.yml
conda activate aeroSplat



# TODO

Priority: **Stage 3 (paper)** > Stage 2 (feeds stage 3) > Stage 1 & IsaacSim (thesis demo)

---

## stage-3_4D-classify

### Data & Training (blocking)
- [ ] Generate synthetic training dataset from IsaacSim renders — *need labeled data before any training*
- [ ] Create train/val/test splits with class balance — *prevent data leakage, ensure fair evaluation*
- [ ] Train baseline model, verify convergence — *prove architecture works before comparisons*
- [ ] Export stage-2 .ply sequences as real-world test set — *validate on non-synthetic data*

### SOTA Comparisons (paper requirement)
- [ ] Implement 2D CNN baseline (ResNet/EfficientNet on crops) — *standard vision baseline*
- [ ] Implement YOLO-based detector+classifier — *common real-time approach*
- [ ] Implement PointNet++ on raw point clouds — *3D point cloud baseline*
- [ ] Reproduce Drone vs Bird challenge winner method — *direct competitor*
- [ ] Multi-view 2D fusion baseline (late fusion) — *alternative to 3D reconstruction*
- [ ] Report accuracy, precision, recall, F1, confusion matrices — *standard classification metrics*

### Ablation Studies (paper requirement)
- [ ] VN-Transformer vs standard Transformer — *justify rotation-invariance design*
- [ ] Mamba vs LSTM vs Transformer temporal encoder — *justify O(T) complexity choice*
- [ ] With/without SH color features — *quantify contribution of appearance*
- [ ] With/without scale/opacity features — *quantify contribution of geometry detail*
- [ ] Sequence length ablation (T=8,16,32,64) — *temporal modeling capacity*
- [ ] Number of Gaussians ablation — *reconstruction fidelity vs efficiency*
- [ ] With/without 3D position from stage-1 — *value of explicit localization*

### Novel Contributions (paper differentiator)
- [ ] Analyze learned attention patterns — *interpretability, what does model focus on*
- [ ] Visualize Gaussian dynamics (flapping vs rotating) — *qualitative evidence of 4D encoding*
- [ ] FFT analysis on Gaussian trajectories — *quantify periodic motion signatures*
- [ ] t-SNE/UMAP of learned embeddings — *show class separability*

### Runtime Analysis (paper + thesis)
- [ ] Benchmark inference: latency, throughput, memory — *prove real-time feasibility*
- [ ] Profile per-component cost breakdown — *identify bottlenecks*
- [ ] Compare to 2D baselines on same hardware — *fair efficiency comparison*

---

## stage-2_3DGS (feeds stage-3)

### Critical Path (blocking stage-3)
- [ ] Fix dependency conflicts in aeroSplat env — *blocking: currently only works in depthsplat env*
- [ ] Run inference on IsaacSim drone/bird renders — *validate pipeline before generating dataset*
- [ ] Export .ply per frame with consistent format — *stage-3 requires: xyz, scale, rot, opacity, SH coeffs*
- [ ] Batch export script for full sequences — *need thousands of .ply files for training*
- [ ] Validate Gaussian quality visually (Polycam/WebGL viewer) — *ensure shape captured, no floaters*

### Integration with Stage-1
- [ ] Accept stage-1 crops as input (masked RGB + depth) — *connect pipeline stages*
- [ ] Handle variable input resolutions — *stage-1 crops vary with distance*
- [ ] Pass through 3D centroid from stage-1 — *stage-3 may use position features*

### Thesis Completeness
- [ ] Benchmark: fps, memory, latency on RTX 5090 — *thesis requires profiling*
- [ ] Report PSNR/SSIM/LPIPS on novel view synthesis — *standard 3DGS metrics*
- [ ] Qualitative comparison: input views vs rendered novel views — *thesis figures*

### Baseline Comparisons (thesis)
- [ ] Compare to LGM (Large Multi-View Gaussian) — *similar feed-forward approach*
- [ ] Compare to Splatter Image — *single-view baseline*
- [ ] Compare to GRM/TriplaneGaussian if time permits — *more baselines strengthen thesis*
- [ ] Report reconstruction quality per method — *justify DepthSplat choice*

### Optional Improvements
- [ ] Fine-tune Depth Anything on aerial objects (Objaverse subset) — *may improve depth for flying objects*
- [ ] Tune Gaussian scale bounds for drone/bird size — *current: 0.005 max, may need adjustment*
- [ ] ONNX/TensorRT export — *deployment optimization, lower priority*

---

## stage-1_fg-seg (thesis pipeline)

### Critical Path (blocking stage-2)
- [ ] Run localizer on IsaacSim renders — *validate detection before pipeline integration*
- [ ] Output standardized format per frame: — *stage-2 needs consistent input*
  - RGB crops (masked or full)
  - 2D bounding boxes per camera view
  - 3D world centroid position
  - Confidence score
- [ ] Export camera intrinsics/extrinsics with crops — *stage-2 needs pose info*

### Validation
- [ ] Compare localized 3D position vs IsaacSim ground truth — *quantify accuracy in meters*
- [ ] Measure detection rate (TP/FP/FN) across frames — *ensure reliable detection*
- [ ] Test at multiple distances (10m, 50m, 100m) — *verify range robustness*
- [ ] Test with motion blur / fast movement — *edge case handling*

### Thesis Completeness
- [ ] Benchmark: fps, memory, latency on RTX 5090 — *thesis requires profiling*
- [ ] Profile: frame differencing vs ray casting vs accumulation — *identify bottlenecks*
- [ ] Report localization error statistics (mean, std, max) — *quantitative results*

### Baseline Comparisons (thesis)
- [ ] Simple background subtraction (cv2.createBackgroundSubtractorMOG2) — *standard baseline*
- [ ] Optical flow magnitude thresholding — *motion-based baseline*
- [ ] Multi-view triangulation from 2D detections — *geometry baseline*
- [ ] CNN detector (YOLO) + 3D lifting — *learning-based baseline*
- [ ] Report precision/recall/F1 per method — *justify voxel approach*

### Optional Improvements
- [ ] Adaptive motion threshold per camera — *handle varying lighting*
- [ ] Temporal smoothing of 3D trajectory (Kalman filter) — *reduce jitter*
- [ ] Handle multiple objects simultaneously — *future extension*

---

## data generation

### IsaacSim

#### Critical Path
- [ ] Generate drone dataset (10m, 50m, 100m distances) — *training data for stage-3*
- [ ] Generate bird dataset (matching distances) — *balanced classes*
- [x] Export camera intrinsics/extrinsics per scene — *required for 3D reconstruction*

#### Paper Robustness Experiments
- [ ] Vary lighting conditions (dawn, noon, dusk) — *generalization ablation*
- [ ] Vary weather (clear, cloudy, fog) — *generalization ablation*
- [ ] Multiple drone/bird models — *prevent overfitting to single asset*
- [ ] Varied flight patterns (hover, linear, erratic) — *temporal diversity*

#### Thesis Demo
- [ ] Headless batch rendering script — *reproducible data generation*
- [ ] Document render settings in config — *thesis methodology section*


### Real World Data

- [ ] check if https://github.com/CenekAlbl/drone-tracking-datasets works (set 3 & 5 have cam/3D drone positions)
- [ ] Specify exact 2k cameras
- [ ] Get 5 camera's + switch to handle traffic and setup camera capture live stream
- [ ] Generate dataset (fly a drone through the camera's field of view, get an RC bird?)
- [ ] annotate dataset (ground truth) using stage 1 fg-seg and verify manually
- [ ] 

---

## Pipeline Integration (thesis demo)

- [ ] End-to-end script: video → stage-1 → stage-2 → stage-3 → classification — *prove full system works*
- [ ] Latency measurement across full pipeline — *real-time feasibility claim*
- [ ] pareto front of #cameras vs accuracy (classification & tracking)
- [ ] Docker compose for reproducibility — *nice-to-have, lower priority*

---

## Paper Writing Checklist

- [ ] Related work: 2D drone detection, 3D reconstruction, 4D understanding — *positioning*
- [ ] Method figure: full architecture diagram — *paper clarity*
- [ ] Quantitative results table: all baselines + ablations — *main results*
- [ ] Qualitative figures: attention maps, Gaussian visualizations — *interpretability*
- [ ] Failure case analysis — *honest evaluation, reviewer expectation*
- [ ] Limitations section — *academic integrity* 
