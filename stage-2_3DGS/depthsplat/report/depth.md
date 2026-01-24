# Depth Estimation and Fusion in DepthSplat

This document explains how depth is estimated, fused, and used for 3D Gaussian reconstruction in DepthSplat.

---

## 1. High-Level Dataflow

```
Input Context Views [B, V, 3, H, W]
         │
         ├──────────────────────────────────────┐
         │                                      │
         ▼                                      ▼
   CNN Backbone                          DINOv2 Backbone
   (multi-scale)                         (monocular features)
         │                                      │
         ▼                                      │
Multi-View Transformer                          │dataset
  (cross-view attention)                        │
         │                                      │
         ▼                                      ▼
┌────────────────────────────────────────────────────────┐
│              Cost Volume Construction                   │
│  (warp features across views at depth candidates)      │
└────────────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────┐
│     Concatenate: [CostVol, CNN, MV, Mono] features     │
└────────────────────────────────────────────────────────┘
         │
         ▼
    UNet Regressor
         │
         ▼
    Depth Head (softmax over candidates)
         │
         ▼
   Coarse Depth [B, V, H/8, W/8]
         │
         ▼
┌────────────────────────────────────────────────────────┐
│              DPT Upsampler                              │
│  Input: mono features + CNN + MV + coarse depth        │
│  Output: residual depth refinement                     │
└────────────────────────────────────────────────────────┘
         │
         ▼
   Final Depth = Bilinear(coarse) + Residual
         │
         ▼
   Unproject to 3D Gaussian Means
```

---

## 2. Component Overview

### 2.1 Dual-Branch Feature Extraction

**CNN Branch** (`CNNEncoder`):
- Extracts multi-scale features at 1/2, 1/4, 1/8 resolution
- Captures local texture and edges
- Output: `features_cnn` at multiple scales

**Monocular Branch** (`DINOv2`):
- Pretrained vision transformer (ViT-S/B/L)
- Extracts semantic features at 1/14 resolution (patch size 14)
- Interpolated to 1/8 resolution
- Output: `features_mono` - rich semantic priors about depth

### 2.2 Multi-View Transformer

- Takes CNN features and applies cross-view self-attention
- Allows views to exchange information about correspondences
- Output: `features_mv` - correspondence-aware features

### 2.3 Cost Volume

- Warps target view features to reference view at discrete depth candidates
- Measures photometric consistency via dot product
- Output: `[B*V, D, H/8, W/8]` - matching probability at each depth

### 2.4 Residual Depth (DPT Upsampler)

The **residual depth** is a learned refinement that corrects the coarse multi-view depth:
- The coarse depth from cost volume matching has limited resolution (1/8)
- Bilinear upsampling introduces artifacts at edges
- The DPT head learns to predict a residual correction using monocular features
- Monocular features capture edges and semantic boundaries that improve depth discontinuities

**Critically**: The DPT head is initialized to output zero, meaning it starts as pure bilinear upsampling and learns corrections during training.

---

## 3. Mathematical Formulation

### 3.1 Feature Extraction

Given input images $\mathbf{I} \in \mathbb{R}^{B \times V \times 3 \times H \times W}$:

**CNN Features:**
$$\mathbf{F}_{cnn}^{(s)} = \text{CNN}(\mathbf{I}) \in \mathbb{R}^{BV \times C_s \times \frac{H}{2^s} \times \frac{W}{2^s}}, \quad s \in \{1, 2, 3\}$$

**Monocular Features:**
$$\mathbf{F}_{mono} = \text{DINOv2}(\mathbf{I}) \in \mathbb{R}^{BV \times C_{vit} \times \frac{H}{8} \times \frac{W}{8}}$$

where $C_{vit} \in \{384, 768, 1024\}$ for ViT-S/B/L respectively.

### 3.2 Multi-View Transformer

The multi-view transformer applies cross-view attention. For features $\mathbf{F} = \{\mathbf{f}_1, ..., \mathbf{f}_V\}$ from $V$ views:

$$\mathbf{F}_{mv} = \text{CrossViewAttn}(\mathbf{F}_{cnn}^{(3)})$$

The attention mechanism allows each spatial location to attend to corresponding locations in other views:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

where queries come from one view and keys/values come from all views (including self).

### 3.3 Cost Volume Construction

For each reference view $r$ and target views $t \in \{1,...,V\} \setminus \{r\}$:

**Step 1: Generate depth candidates**

At the coarsest scale, depth candidates are uniformly sampled in inverse depth space:
$$d_i = d_{min} + \frac{i}{D-1}(d_{max} - d_{min}), \quad i \in \{0, ..., D-1\}$$

where $d$ represents inverse depth (disparity), and $D=128$ candidates.

**Step 2: Warp target features to reference frame**

For each depth candidate $d_i$, compute the warped coordinates:

$$\mathbf{p}_t = K_t \cdot T_{t \leftarrow r} \cdot d_i \cdot K_r^{-1} \cdot \mathbf{p}_r$$

where:
- $\mathbf{p}_r = (u, v, 1)^T$ is the homogeneous pixel coordinate in reference view
- $K_r, K_t$ are intrinsic matrices
- $T_{t \leftarrow r} = T_t^{-1} \cdot T_r$ is the relative pose (camera-to-world matrices)
- $d_i$ is the depth hypothesis

The warped features are obtained via bilinear sampling:
$$\mathbf{F}_t^{warp}(d_i) = \text{BilinearSample}(\mathbf{F}_t, \mathbf{p}_t)$$

**Step 3: Compute matching cost**

The cost volume measures feature similarity:
$$\mathbf{C}(d_i) = \frac{1}{|T|} \sum_{t \in T} \frac{\mathbf{F}_r \cdot \mathbf{F}_t^{warp}(d_i)}{\sqrt{C}}$$

where $C$ is the feature channel dimension. This produces:
$$\mathbf{C} \in \mathbb{R}^{BV \times D \times \frac{H}{8} \times \frac{W}{8}}$$

### 3.4 Depth Regression

**Feature Concatenation:**

All features are concatenated channel-wise:
$$\mathbf{F}_{concat} = [\mathbf{C}, \mathbf{F}_{cnn}^{(3)}, \mathbf{F}_{mv}, \mathbf{F}_{mono}]$$

**UNet Processing:**

The concatenated features pass through a UNet with cross-view attention:
$$\mathbf{F}_{out} = \text{UNet}(\mathbf{F}_{concat})$$

**Soft Argmax Depth:**

A depth head predicts logits over depth candidates:
$$\mathbf{P} = \text{softmax}(\text{DepthHead}(\mathbf{F}_{out})) \in \mathbb{R}^{BV \times D \times H' \times W'}$$

The coarse depth is computed as the expected value:
$$\hat{d}_{coarse} = \sum_{i=0}^{D-1} P_i \cdot d_i$$

This soft argmax is differentiable, unlike hard argmax.

### 3.5 Depth Upsampling with Residual Refinement

**Bilinear Upsampling:**
$$\hat{d}_{bilinear} = \text{Upsample}_{8\times}(\hat{d}_{coarse})$$

**DPT Residual:**

The DPT head takes multi-scale monocular features $\{\mathbf{F}_{mono}^{(l)}\}_{l=1}^{4}$ from intermediate ViT layers, plus CNN and MV features:

$$\Delta d = \text{DPT}(\mathbf{F}_{mono}^{(1:4)}, \mathbf{F}_{cnn}, \mathbf{F}_{mv}, \hat{d}_{coarse})$$

**Final Depth:**
$$\hat{d} = \text{clamp}(\hat{d}_{bilinear} + \Delta d, d_{min}, d_{max})$$

The residual $\Delta d$ learns to:
1. Sharpen depth edges using monocular semantic boundaries
2. Correct systematic biases in cost volume matching
3. Fill in details lost during downsampling

### 3.6 Inverse Depth to Metric Depth

The network operates in inverse depth (disparity) space for numerical stability. Final conversion:
$$D_{metric} = \frac{1}{\hat{d}}$$

Inverse depth has advantages:
- Uniform sampling covers both near and far regions appropriately
- Numerical precision is better for distant objects
- Matches the projective geometry of cameras

---

## 4. From Depth to 3D Gaussians

### 4.1 Unprojection

Each pixel $(u, v)$ with depth $D$ is unprojected to a 3D point:

$$\mathbf{X}_{cam} = D \cdot K^{-1} \cdot \begin{pmatrix} u \\ v \\ 1 \end{pmatrix} = D \cdot \begin{pmatrix} (u - c_x) / f_x \\ (v - c_y) / f_y \\ 1 \end{pmatrix}$$

Transform to world coordinates:
$$\mathbf{X}_{world} = T_{c2w} \cdot \begin{pmatrix} \mathbf{X}_{cam} \\ 1 \end{pmatrix}$$

This $\mathbf{X}_{world}$ becomes the Gaussian mean $\boldsymbol{\mu}$.

### 4.2 Gaussian Parameters

For each pixel, the network also predicts:

**Opacity:** $\alpha = \sigma(\text{raw}_\alpha)$ (sigmoid activation)

**Scales:** $\mathbf{s} = \exp(\text{raw}_s)$ (3D log-scale)

**Rotation:** $\mathbf{q} = \text{normalize}(\text{raw}_q)$ (quaternion)

**Color (Spherical Harmonics):** $\mathbf{c}_{SH}$ (DC + higher-order terms)

### 4.3 Covariance Construction

The 3D Gaussian covariance is constructed from scale and rotation:

$$\Sigma = R \cdot S \cdot S^T \cdot R^T$$

where:
- $S = \text{diag}(s_x, s_y, s_z)$ is the scale matrix
- $R$ is the rotation matrix from quaternion $\mathbf{q}$

---

## 5. Why This Architecture?

### Multi-View Stereo (Cost Volume)
- **Strength:** Geometrically grounded, exploits multi-view consistency
- **Weakness:** Fails in textureless regions, limited resolution

### Monocular Depth (DINOv2 + DPT)
- **Strength:** Semantic understanding, works on any single image
- **Weakness:** Scale ambiguity, less geometrically accurate

### Fusion Strategy
DepthSplat combines both:
1. Cost volume provides geometrically consistent depth at coarse resolution
2. Monocular features provide semantic boundaries for upsampling
3. The residual learning allows the network to learn when to trust each source

The soft argmax over the cost volume is particularly important:
- Provides uncertainty estimate via the distribution width
- The matching probability (max of softmax) indicates confidence
- Low-confidence regions rely more on monocular refinement

---

## 6. Summary Table

| Stage | Input | Output | Resolution | Key Operation |
|-------|-------|--------|------------|---------------|
| CNN Backbone | Images | Multi-scale features | 1/2, 1/4, 1/8 | Conv layers |
| DINOv2 | Images | Semantic features | 1/8 | ViT attention |
| MV Transformer | CNN features | Correspondence features | 1/8 | Cross-view attention |
| Cost Volume | MV features + poses | Matching costs | 1/8 × D | Feature warping |
| Depth Head | Fused features | Coarse depth | 1/8 | Soft argmax |
| DPT Upsampler | Mono + coarse depth | Residual | 1/1 | Feature pyramid |
| Final Depth | Bilinear + residual | Metric depth | 1/1 | Addition + clamp |
| Unprojection | Depth + intrinsics | 3D Gaussians | Per-pixel | Inverse projection |
