# Multi-Camera 4D Classification of Flying Objects

## Thesis Summary: 3D Gaussian Splatting for Drone and Bird Detection

---

## 1. Introduction

### 1.1 Motivation and Context

The proliferation of small UAVs poses significant security challenges for airports, critical infrastructure, and public events. Distinguishing authorized drones from unauthorized ones and natural flyers (birds) is critical. Traditional radar and thermal solutions are expensive and often struggle with the low radar cross-sections and similar thermal signatures of small drones and birds.

**This thesis** proposes a cost-effective alternative: a network of ground-based RGB cameras. By leveraging multi-view geometric constraints and the **4D temporal dynamics** of 3D reconstructions, we aim to achieve robust classification on commodity hardware, outperforming 2D-only or static 3D approaches.

> The intersection of **feed-forward 3D Gaussian Splatting** and **temporal State Space Models (Mamba)** for flying object classification represents an unexplored research frontier.

---

### 1.2 Problem Statement

**Given:** A network of $N \geq 3$ (typically $N=5$) fixed RGB cameras with overlapping fields of view monitoring a central airspace volume.

**Core Question:** How can we detect, reconstruct, classify, and track flying objects (drones and birds) in 3D space using only synchronized RGB camera feeds, while leveraging the temporal dynamics of 3D Gaussian representations?

**System Requirements:**

| Aspect | Description |
|--------|-------------|
| **Input** | Synchronized RGB frames from $N$ cameras with known intrinsics/extrinsics |
| **Output** | 3D occupancy, object class (drone/bird/background), 3D tracks |
| **Constraints** | Real-time processing (< 500ms latency) |
| **Challenges** | Distant objects (50-300m), textureless sky backgrounds, low pixel resolution (< 20px) |

---

### 1.3 Research Questions

| ID | Question |
|----|----------|
| **RQ1** | *Foreground Segmentation:* How to segment dynamic objects in 3D without prior appearance knowledge, specifically against textureless skies? |
| **RQ2** | *3D Reconstruction:* How to adapt feed-forward Gaussian splatting for small, distant objects where standard photometric matching fails? |
| **RQ3** | *Temporal Dynamics:* Can temporal changes in 4D Gaussian parameters (deformation, rotation) distinguish mechanical vs. biological motion? |
| **RQ4** | *End-to-End Performance:* What accuracy and latency can be achieved compared to 2D baselines? |

---

## 2. Problem Setting and Assumptions

### 2.1 Monitored Volume & Configuration
- **Volume:** Central airspace (50-500m radius, up to 200m altitude).
- **Cameras:** >5 upward-looking RGB cameras, synchronized (<10ms error), connected to a central edge server.


### 2.2 Key Constraints
- **Textureless Background:** Open sky lacks features for standard SfM/MVS.
- **Sparse Views:** Only ~5 views available, insufficient for NeRF without overfitting.
- **Low Resolution:** Objects often occupy < 50 pixels, making 2D texture features less reliable for 2D classifiers.

---

## 3. Proposed Approach

The pipeline integrates four major components:

### 3.1 Synthetic Data Generation (NVIDIA Isaac Sim)
- Comprehensive dataset of drones and birds flying through diverse scenes.
- Includes realistic camera effects (motion blur, rolling shutter) and environmental variations.
- Provides **perfect ground truth**: depth maps, segmentation masks, and 3D trajectories for training.

### 3.2 Multi-Camera Foreground Segmentation (Track-Before-Detect)
- **Volumetric Voting:** A ray-marching approach that identifies 3D regions visible in $\geq 3$ cameras.
- **3D-First:** Accumulates evidence in 3D space to resolve ambiguities that plague 2D background subtraction.
- **Robustness:** Handles dynamic objects against textureless skies where optical flow fails.

### 3.3 Feed-Forward 3D Gaussian Reconstruction
- **Architecture:** **DepthSplat** (CVPR 2025).
- **Rationale:** Dual-branch design combines **multi-view cost volumes** (geometric consistency) with **monocular depth priors** (robustness in textureless regions).
- **Domain** Depthsplat will be retrained on objaverse to enable their model architecture to reconstruction of small objects. It will then be finetuned on the synthetic dataset (models of drones and birds)to enable it to work with real data.
- **Augmentation:** Retrained with a **Silhouette Consistency Loss** to anchor geometry using segmentation masks when photometric gradients vanish in the sky.
- **Output:** Explicit 3D Gaussian primitives (position, scale, rotation, opacity, color) per frame.

### 3.4 4D Temporal Dynamics Classification
- **Architecture:**
    - Input:
        - Temporal sequence: 𝒢 = {G^(1), G^(2), ..., G^(T)}
        where G^(t) = {g_i^(t)}_{i=1}^{M_t} is a set of M_t Gaussians at frame t

    - Each Gaussian g_i:
        - Position:    μ_i ∈ ℝ^3        (SE(3)-equivariant)
        - Rotation:    r_i ∈ ℝ^6        (6D continuous representation)
        - Scale:       s_i ∈ ℝ^3        (log-scale for stability)
        - Opacity:     α_i ∈ ℝ          (logit form)
        - Color/SH:    c_i ∈ ℝ^{48}     (optional, may truncate to DC term) (optional)

    - Stage 1: Per-frame embedding (Should incorporate  **SE(3)-equivariant** layers to ensure classification is robust to object orientation.)
    - Stage 2: Inter-frame Temporal embedding/modelling (sequence of N frames) Extracts discriminative motion signatures (e.g., rigid rotor vs. flapping wing) from the evolution of Gaussian parameters.
    - Stage 3: 
        - Classification: ŷ = σ(W·[y_temporal; y_freq] + b) ∈ [0,1]
        - where y_freq ∈ ℝ^F is optional frequency auxiliary features

---

## 4. Key Contributions

1.  **Novel 4D Classification Architecture:** First application using temporal sequences of 3D Gaussian tokens, exploiting dynamic motion signatures for classification.
2.  **Robust 3D Reconstruction Strategy:** Adaptation of **DepthSplat** to become an object centric 3D reconstruction method.
3.  **Volumetric Track-Before-Detect:** A geometry-aware segmentation scheme that outperforms 2D-first detection for small, low-SNR targets.
4.  **Synthetic Data Framework:** Open-source Isaac Sim tools for generating physically accurate multi-view aerial datasets.

---

## 5. Scope and Limitations

| In Scope | Out of Scope |
|----------|--------------|
| Detection, reconstruction, classification (Drone vs. Bird) | Fine-grained model/species recognition |
| $N \geq 3$ overlapping cameras | Single-camera or non-overlapping setups |
| Central monitored volume | Long-range tracking beyond volume |

---

## Summary

This thesis presents a novel end-to-end system for airspace monitoring. By combining **volumetric segmentation**, **DepthSplat-based reconstruction**, and **Mamba-based temporal classification**, it overcomes the limitations of traditional 2D surveillance. The system explicitly models the **4D dynamics** of flying objects, allowing it to distinguish between mechanical drones and biological birds even when spatial resolution is low.
