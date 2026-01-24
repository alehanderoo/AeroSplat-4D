# Mask Integration Strategies for DepthSplat

This document describes the mask integration modifications made to the DepthSplat architecture based on research literature review. All modifications are **disabled by default** for backward compatibility with existing experiments.

## Overview

Object segmentation masks represent an underutilized prior in the original DepthSplat architecture. The research literature demonstrates that masks should be integrated at multiple pipeline stages beyond simple background zeroing, with soft weighting and learned integration consistently outperforming binary operations.

**Key insight from literature:** Soft mask integration with preserved gradients outperforms hard binary masking at every stage.

## Modifications

### 1. Soft Cost Volume Masking

**Files modified:**
- `src/model/encoder/unimatch/mv_unimatch.py`
- `src/model/encoder/encoder_depthsplat.py`

**Problem addressed:**
Binary cost volume zeroing (`cost_volume = cost_volume * mask`) creates gradient discontinuities and discards valid texture information near object boundaries.

**Solution:**
Replaced binary masking with soft confidence weighting using temperature-scaled sigmoid:

```python
mask_confidence = torch.sigmoid((mask - 0.5) / temperature)
cost_volume = cost_volume * mask_confidence
```

**Configuration options:**
- `soft_mask_enabled` (bool, default: false): Enable soft confidence weighting
- `soft_mask_temperature` (float, default: 1.0): Temperature for sigmoid sharpness (lower = sharper, 0 approaches binary)
- `soft_mask_edge_weight` (float, default: 0.0): Optional edge-aware weighting boost using Sobel filters

**Research basis:** Vis-MVSNet, DCV-MVSNet, PatchmatchNet demonstrate that pixel-wise view weights preserve gradients at boundaries.

---

### 2. Gradient Matching Loss

**Files added:**
- `src/loss/loss_gradient.py`
- `config/loss/gradient.yaml`

**Files modified:**
- `src/loss/__init__.py`

**Problem addressed:**
Standard depth losses (L1/L2) don't explicitly enforce sharp depth edges at object boundaries.

**Solution:**
New loss function that supervises depth gradients (via Sobel operators) to match ground truth depth gradients, with optional extra weighting at mask boundaries:

```python
L_gradient = ||grad(depth_pred) - grad(depth_gt)||_1 * boundary_weight
```

**Configuration options:**
- `weight` (float, default: 0.1): Loss weight
- `boundary_weight` (float, default: 2.0): Multiplier at mask boundaries
- `boundary_dilation` (int, default: 3): Dilation radius for boundary detection
- `multi_scale` (bool, default: true): Use multi-scale gradient matching
- `min_depth`, `max_depth` (float): Valid depth range

**Usage in config:**
```yaml
loss:
  gradient:
    weight: 0.1
    boundary_weight: 2.0
```

**Research basis:** Depth Anything V2, EG-MVSNet demonstrate gradient matching at mask boundaries yields sharper reconstructions.

---

### 3. Mask-Guided DPT Fusion

**Files modified:**
- `src/model/encoder/unimatch/dpt_head.py`
- `src/model/encoder/unimatch/mv_unimatch.py`
- `src/model/encoder/encoder_depthsplat.py`

**Problem addressed:**
The DPT feature fusion blocks don't leverage mask information, missing opportunities to emphasize foreground boundaries during depth refinement.

**Solution:**
Added `MaskGuidedGate` module that applies soft attention gating in skip connections:

```python
class MaskGuidedGate(nn.Module):
    def forward(self, features, mask):
        mask_features = self.mask_proj(mask)
        gate = self.gate(torch.cat([features, mask_features], dim=1))
        return features * gate + features * 0.1  # residual
```

The gate is learned end-to-end and applied in `FeatureFusionBlock` skip connections.

**Configuration option:**
- `mask_guided_dpt_enabled` (bool, default: false): Enable mask-guided attention gates

**Research basis:** Pixel-Adaptive Convolution (PAC), attention gates in skip connections improve boundary preservation.

---

### 4. Mask Boundary-Guided Upsampling

**Files modified:**
- `src/model/encoder/unimatch/dpt_head.py`
- `src/model/encoder/unimatch/mv_unimatch.py`
- `src/model/encoder/encoder_depthsplat.py`

**Problem addressed:**
The final 8x depth upsampling doesn't account for mask boundaries, potentially causing soft depth edges at object boundaries.

**Solution:**
Enhanced the DPT output convolution to concatenate mask edge features (computed via Sobel) before the final depth prediction:

```python
mask_edges = sobel_magnitude(mask)
path_1_with_mask = torch.cat([path_1, mask, mask_edges], dim=1)
out = output_conv(output_conv_input(path_1_with_mask))
```

**Configuration option:**
- `mask_guided_upsample_enabled` (bool, default: false): Enable mask boundary-guided upsampling

**Research basis:** Depth boundary sharpness directly impacts Gaussian placement quality in downstream 3D reconstruction.

---

## Configuration

All options are in `config/model/encoder/depthsplat.yaml`:

```yaml
# Mask integration options (all disabled by default)
soft_mask_enabled: false
soft_mask_temperature: 1.0
soft_mask_edge_weight: 0.0
mask_guided_dpt_enabled: false
mask_guided_upsample_enabled: false
```

For experiments, override in your experiment config:

```yaml
model:
  encoder:
    soft_mask_enabled: true
    soft_mask_temperature: 0.5
    mask_guided_dpt_enabled: true
    mask_guided_upsample_enabled: true
```

To enable gradient loss, add to your experiment config:

```yaml
defaults:
  - override /loss: [mse, lpips, depth, mask, gradient]

loss:
  gradient:
    weight: 0.1
    boundary_weight: 2.0
```

---

## Recommended Ablation Study

Based on the research literature, the following ablation order is recommended (by expected impact):

| Ablation | Description | Expected Impact |
|----------|-------------|-----------------|
| Baseline | All disabled | - |
| +soft_mask | Enable soft cost volume masking | Medium-High |
| +gradient_loss | Add gradient matching loss | Medium |
| +dpt_fusion | Enable mask-guided DPT fusion | Medium |
| +boundary_upsample | Enable mask boundary upsampling | Low-Medium |
| Full | All enabled | Highest |

Metrics to track:
- Depth L1 error (overall and at boundaries)
- Depth edge sharpness (gradient magnitude at mask edges)
- Downstream Gaussian reconstruction quality (PSNR, LPIPS)

---

## Architecture Diagram

```
Input Images + Masks
        |
        v
+-------------------+
| CNN Feature       |
| Extraction        |
+-------------------+
        |
        v
+-------------------+
| Multi-View        |
| Transformer       |
+-------------------+
        |
        v
+-------------------+      +---------------------------+
| Cost Volume       |<-----|  [NEW] Soft Mask Weighting |
| Construction      |      |  (soft_mask_enabled)       |
+-------------------+      +---------------------------+
        |
        v
+-------------------+
| Depth Regression  |
| (UNet)            |
+-------------------+
        |
        v
+-------------------+      +---------------------------+
| DPT Feature       |<-----|  [NEW] Mask-Guided Gates   |
| Fusion            |      |  (mask_guided_dpt_enabled) |
+-------------------+      +---------------------------+
        |
        v
+-------------------+      +---------------------------+
| Depth Upsampling  |<-----|  [NEW] Boundary Features   |
| (8x)              |      |  (mask_guided_upsample)    |
+-------------------+      +---------------------------+
        |
        v
+-------------------+      +---------------------------+
| Loss Computation  |<-----|  [NEW] Gradient Loss       |
|                   |      |  (gradient loss)           |
+-------------------+      +---------------------------+
```

---

## References

Based on research synthesis from:
- Vis-MVSNet, DCV-MVSNet: Soft confidence weighting in MVS
- Depth Anything V2: Gradient matching loss
- EG-MVSNet, EPNet: Edge-aware cost regularization
- CamoFormer: Masked separable attention
- ObjectSDF++: Occlusion-aware mask supervision
- GRM, LGM: Alpha mask supervision for Gaussian reconstruction
- SAGD: Gaussian decomposition at boundaries

---

## Changelog

- 2024-01: Initial implementation of mask integration strategies
  - Added soft cost volume masking
  - Added gradient matching loss
  - Added mask-guided DPT fusion gates
  - Added mask boundary-guided upsampling
  - All options disabled by default for backward compatibility
