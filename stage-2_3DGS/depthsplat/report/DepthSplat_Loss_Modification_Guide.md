# DepthSplat Loss Modification Guide

*Object-Centric 3D Gaussian Splatting with White Backgrounds*

## Executive Summary

This document outlines the required modifications to the DepthSplat training loss functions for training an object-centric feed-forward 3D Gaussian Splatting model on Objaverse data with white backgrounds. The recommendations are based on analysis of 12 major feed-forward 3DGS papers and address the specific issue of floating Gaussian artifacts observed in training.

**Key Finding:** The current implementation computes RGB losses (MSE, LPIPS) only on foreground pixels, which allows the model to produce arbitrary outputs in background regions—manifesting as the red floating Gaussians visible in the projection images. The literature consensus is to compute full-image losses combined with separate mask supervision.

## Problem Analysis

### Current Training Behavior

The uploaded visualization images reveal several diagnostic patterns. The comparison image shows the model producing blurry, washed-out reconstructions with significant artifacts around object boundaries. The projection image is particularly telling—it shows red-colored Gaussian artifacts dispersed in the 3D space around the object, visible in all three projection planes (YZ, ZX, XY). These artifacts occur because the model has no loss signal constraining what happens in background regions.

The silhouette comparison shows that while the ground truth masks are clean binary silhouettes, the predicted silhouettes have soft, blurry boundaries with visible texture bleeding into the background. This indicates the alpha/opacity prediction is not being properly supervised.

### Root Cause in Current Loss Implementation

Examining the provided loss files reveals the core issue. In **loss_mse.py**, the configuration has `use_foreground_mask: bool = True` as the default. When this flag is enabled, the MSE loss excludes background pixels from the loss computation entirely. Similarly, **loss_lpips.py** sets background pixels to zero in both prediction and target before computing perceptual loss.

This approach creates a critical problem: the model receives no gradient signal for pixels outside the foreground mask. Gaussians can freely exist in background regions—they contribute to the rendered image but face no penalty for incorrect colors, positions, or opacities. The red artifacts are likely Gaussians that have learned to exist in 3D space but received no training signal to constrain their appearance or removal.

## Literature Consensus

Analysis of successful feed-forward 3DGS methods reveals a consistent approach that differs fundamentally from the current implementation.

### Universal Approach: Full-Image Loss + Separate Mask Supervision

Every major Objaverse-trained method (LGM, GRM, TriplaneGaussian, AGG, Splatter Image, Gamba) computes RGB losses on the **complete rendered image including background**, not just foreground pixels. The mask/alpha is supervised as a **separate, independent loss term**.

The LGM formulation exemplifies this approach:

```
L_rgb = L_MSE(I_rgb, I^GT_rgb) + λ·L_LPIPS(I_rgb, I^GT_rgb)

L_α = L_MSE(I_α, I^GT_α)  # Separate term

L_total = L_rgb + λ_mask · L_α
```

This decoupling is critical because it prevents the model from trading off alpha accuracy for RGB accuracy—both must be correct independently. When losses are computed only on masked regions, incorrect alpha predictions can actually *reduce* loss by excluding difficult pixels from consideration.

### Why White Backgrounds Are Universal

All successful methods use white backgrounds during training, never black. The reasoning is straightforward: white backgrounds create maximum contrast with most object textures, making it harder for models to "cheat" by learning a uniform background color. With black backgrounds (as in the original DepthSplat training), predicting black everywhere has minimal penalty—the model discovered that outputting near-black in difficult regions minimized loss.

With white backgrounds and full-image loss computation, predicting uniform white would produce *maximum* error on object pixels (which have diverse colors). This forces the model to learn proper foreground/background separation rather than exploiting a shortcut.

### Method Comparison Summary

| **Method**           | **RGB Loss**              | **Alpha/Mask Loss**            | **Background** |
|----------------------|---------------------------|--------------------------------|----------------|
| **LGM**              | MSE + LPIPS (full)        | Separate MSE on alpha          | White          |
| **GRM**              | L2 + perceptual (full)    | Separate L2 on mask            | White          |
| **TriplaneGaussian** | MSE (full image)          | Explicit L_MASK                | White          |
| **AGG**              | L_rgba (joint)            | Implicit via RGBA              | White          |
| **Splatter Image**   | MSE + LPIPS               | Regularization losses          | Configurable   |
| **Gamba**            | MSE + LPIPS               | Radial mask constraints        | White          |

## Required Modifications

### Modification 1: Disable Foreground Masking in RGB Losses

**Priority: Critical**

The most important change is to compute MSE and LPIPS losses on the full rendered image, not just foreground pixels. This ensures the model receives gradient signal for background regions, forcing it to either predict the correct background color (white) or predict zero opacity for background Gaussians.

**File: loss_mse.py**

```python
@dataclass
class LossMseCfg:
    weight: float
    # CHANGE: Set to False for white background training
    use_foreground_mask: bool = False  # Was True
```

**File: loss_lpips.py**

```python
@dataclass
class LossLpipsCfg:
    weight: float
    apply_after_step: int
    # CHANGE: Set to False for white background training
    use_foreground_mask: bool = False  # Was True
```

**Rationale:** With white backgrounds in both rendered output and ground truth, the full-image MSE will penalize any non-white pixels in background regions. This creates explicit pressure for the model to either (a) not place Gaussians in background regions, or (b) predict zero opacity for any Gaussians that do exist there.

### Modification 2: Refactor Silhouette Loss to Direct Mask Supervision

**Priority: High**

The current silhouette loss attempts to *estimate* alpha from rendered color (via distance from background color). This is indirect and error-prone. The literature approach is simpler: directly supervise the rendered alpha/opacity channel against the ground truth mask.

**New Implementation: loss_mask.py**

```python
"""
Direct mask supervision loss for object-centric 3DGS.
Supervises rendered alpha directly against GT mask.
"""

from dataclasses import dataclass
import torch
import torch.nn.functional as F
from .loss import Loss

@dataclass
class LossMaskCfg:
    weight: float = 0.1
    use_bce: bool = False  # MSE is standard

class LossMask(Loss):
    def forward(self, prediction, batch, ...):
        # Get rendered alpha from decoder output
        pred_alpha = prediction.alpha  # [B, V, H, W]
        gt_mask = batch["target"]["mask"]

        # Direct MSE supervision (standard approach)
        if self.cfg.use_bce:
            loss = F.binary_cross_entropy(
                pred_alpha.clamp(1e-7, 1-1e-7),
                gt_mask
            )
        else:
            loss = F.mse_loss(pred_alpha, gt_mask)

        return self.cfg.weight * loss
```

**Implementation Note:** This requires the Gaussian renderer to output an alpha channel. If the current decoder does not provide `prediction.alpha`, you will need to modify the decoder to accumulate and output the per-pixel alpha values during splatting. Most 3DGS renderers compute this anyway—it's the sum of Gaussian contributions weighted by their opacities.

### Modification 3: Remove or Reconfigure Silhouette Loss

**Priority: Medium**

If direct alpha supervision is implemented (Modification 2), the current `_alpha_from_color` silhouette loss becomes redundant and potentially harmful. The color-based alpha estimation introduces noise because it cannot distinguish between genuinely dark foreground pixels and background—this ambiguity may be contributing to the floating Gaussians.

- **Option A (Recommended):** Replace loss_silhouette.py entirely with the new loss_mask.py that uses direct alpha supervision

- **Option B:** Keep silhouette loss but only use the `alpha_from_depth` method, which is more reliable

- **Option C:** Disable silhouette loss entirely and rely on full-image RGB loss + direct mask loss

### Modification 4: Add Gaussian Scale Constraints (Optional but Recommended)

**Priority: Medium-Low**

GRM demonstrated that bounded Gaussian scale parameters prevent individual Gaussians from growing large enough to cover entire background regions. This architectural constraint complements the loss modifications.

```python
# In Gaussian parameter prediction head
s_min, s_max = 0.005, 0.02  # GRM values
scale = s_min + (s_max - s_min) * torch.sigmoid(scale_raw)
```

This prevents "bloated" Gaussians that might otherwise fill in large regions arbitrarily. The floating artifacts you're seeing may be Gaussians with uncontrolled scales.

## Updated Loss Configuration

The following shows the recommended loss configuration after modifications. The key changes are highlighted.

```yaml
# config.yaml or equivalent
loss:
  mse:
    weight: 1.0
    use_foreground_mask: false  # CHANGED from true

  lpips:
    weight: 0.05
    apply_after_step: 1000
    use_foreground_mask: false  # CHANGED from true

  mask:  # NEW - replaces silhouette
    weight: 0.1
    use_bce: false

  depth:
    weight: 0.1
    min_depth: 0.1
    max_depth: 10.0
    background_weight: 0.1
```

### Total Loss Formulation

The final loss should be a weighted sum of independent terms:

```
L_total = λ_mse · L_MSE(I_pred, I_gt)       # Full image
        + λ_lpips · L_LPIPS(I_pred, I_gt)   # Full image
        + λ_mask · L_MSE(α_pred, M_gt)      # Separate alpha supervision
        + λ_depth · L_depth(D_pred, D_gt)   # Depth supervision (foreground only)
```

**Critical:** The RGB losses (MSE, LPIPS) must operate on the *complete image* while the mask loss supervises alpha *separately*. This decoupling prevents the model from gaming one loss by manipulating the other.

## Evaluation Protocol

For fair evaluation that focuses on object reconstruction quality, follow the Splatter Image protocol:

1. Set background pixels to the **same color** (typically black) in *both* prediction and ground truth before computing metrics

2. Use the ground truth mask to identify background pixels in both images

3. Then compute PSNR, SSIM, LPIPS on the masked images

```python
def evaluate_foreground_only(pred, gt, mask):
    # Set background to same color in BOTH images
    bg_color = 0.0  # black
    pred_masked = pred * mask + bg_color * (1 - mask)
    gt_masked = gt * mask + bg_color * (1 - mask)
    return compute_metrics(pred_masked, gt_masked)
```

**Important:** Never compute metrics where only one image has background substitution—this is exactly the "metric gaming" problem the literature warns against.

## Implementation Checklist

| **Task**                                              | **Priority**   |
|-------------------------------------------------------|----------------|
| ☐ Set `use_foreground_mask=False` in LossMseCfg       | **Critical**   |
| ☐ Set `use_foreground_mask=False` in LossLpipsCfg     | **Critical**   |
| ☐ Verify training data uses white backgrounds         | **Critical**   |
| ☐ Implement LossMask with direct alpha supervision    | **High**       |
| ☐ Modify decoder to output alpha channel if needed    | **High**       |
| ☐ Remove or disable loss_silhouette.py                | **Medium**     |
| ☐ Add bounded sigmoid for Gaussian scales             | **Medium-Low** |
| ☐ Update evaluation to use foreground-only protocol   | **Medium**     |

## Expected Outcomes

After implementing these modifications, you should observe the following changes:

1. **Elimination of floating Gaussians:** The red artifacts in projection images should disappear because background regions now receive loss signal forcing either correct color (white) or zero opacity

2. **Sharper silhouettes:** The blurry predicted masks should become cleaner with direct alpha supervision

3. **Better view synthesis:** The washed-out reconstructions should improve as the model can no longer hide errors by excluding them from loss computation

4. **GRM-equivalent metrics:** Based on GRM's ablation study, proper alpha regularization should yield approximately 0.1 dB PSNR improvement and 0.003 SSIM improvement

## Conclusion

The core issue is architectural rather than hyperparameter-related: the current implementation's foreground-only loss computation creates a training regime where the model receives no signal for background regions, allowing arbitrary Gaussian placement and appearance in those areas. The literature consensus—white backgrounds plus separate alpha supervision with full-image RGB loss—directly addresses this issue.

The modifications outlined in this document align DepthSplat's training losses with the approach validated by LGM, GRM, TriplaneGaussian, AGG, Splatter Image, and Gamba. The changes are relatively straightforward (primarily configuration changes plus one new loss term) but address the fundamental cause of the floating Gaussian artifacts.

**Priority of implementation:** Modification 1 (disable foreground masking) is critical and should be implemented first. Modification 2 (direct mask supervision) should follow. Modifications 3 and 4 are refinements that can be added after validating the primary changes.
