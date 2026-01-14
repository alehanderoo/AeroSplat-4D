# Loss Modifications for White Background Training

**Date:** 2025-12-23
**Reference:** `report/DepthSplat_Loss_Modification_Guide.md`

## Summary

Implemented the loss modifications outlined in the modification guide to address floating Gaussian artifacts and improve object-centric reconstruction quality. The changes align DepthSplat's training with the literature consensus from LGM, GRM, TriplaneGaussian, AGG, Splatter Image, and Gamba.

## Problem Addressed

The original implementation computed RGB losses (MSE, LPIPS) only on foreground pixels, which allowed the model to produce arbitrary outputs in background regions—manifesting as floating Gaussian artifacts visible in projection images.

## Changes Made

### 1. Disabled Foreground Masking in RGB Losses (Critical)

**Files Modified:**
- `src/loss/loss_mse.py`: Changed `use_foreground_mask` default from `True` to `False`
- `src/loss/loss_lpips.py`: Changed `use_foreground_mask` default from `True` to `False`

**Rationale:** With white backgrounds, full-image MSE will penalize any non-white pixels in background regions. This creates explicit pressure to either not place Gaussians in background regions, or predict zero opacity for any that exist there.

### 2. Added Alpha Channel Output to Decoder (High Priority)

**Files Modified:**
- `src/model/decoder/decoder.py`: Added `alpha` field to `DecoderOutput` dataclass
- `src/model/decoder/cuda_splatting.py`: Added `render_alpha_cuda()` function
- `src/model/decoder/decoder_splatting_cuda.py`:
  - Added import for `render_alpha_cuda`
  - Added `render_alpha()` method
  - Modified `forward()` to compute and return alpha

**Implementation:** Alpha is rendered by splatting white (1,1,1) Gaussians against a black (0,0,0) background. The resulting intensity equals the accumulated opacity.

### 3. Implemented Direct Mask Supervision Loss (High Priority)

**Files Created:**
- `src/loss/loss_mask.py`: New `LossMask` class that directly supervises rendered alpha against GT mask
- `config/loss/mask.yaml`: Configuration file for mask loss

**Files Modified:**
- `src/loss/__init__.py`: Registered `LossMask` and `LossMaskCfgWrapper`

**Implementation:** Uses MSE loss (standard approach) or optional BCE loss to supervise `prediction.alpha` against `batch["target"]["mask"]`.

### 4. Updated Experiment Configuration

**Files Modified:**
- `config/experiment/objaverse_white.yaml`:
  - Changed loss list from `[mse, lpips, depth, silhouette]` to `[mse, lpips, depth, mask]`
  - Replaced `silhouette:` config with `mask:` config
  - Updated comments to reflect the new approach

## Total Loss Formulation

After these changes, the loss is:

```
L_total = λ_mse · L_MSE(I_pred, I_gt)       # Full image (1.0)
        + λ_lpips · L_LPIPS(I_pred, I_gt)   # Full image (0.05)
        + λ_mask · L_MSE(α_pred, M_gt)      # Separate alpha supervision (0.1)
        + λ_depth · L_depth(D_pred, D_gt)   # Depth supervision (0.5)
```

## Expected Outcomes

1. **Elimination of floating Gaussians:** Background regions now receive loss signal
2. **Sharper silhouettes:** Direct alpha supervision produces cleaner masks
3. **Better view synthesis:** Model cannot hide errors by excluding them from loss

## Backward Compatibility

- All changes are backward compatible
- `use_foreground_mask` can be set to `True` in config for old behavior
- `LossSilhouette` remains available for experiments that need it
- Alpha output is always computed but `LossMask` handles `alpha=None` gracefully

## Files Changed Summary

| File | Change Type |
|------|-------------|
| `src/loss/loss_mse.py` | Modified default |
| `src/loss/loss_lpips.py` | Modified default |
| `src/model/decoder/decoder.py` | Added alpha field |
| `src/model/decoder/cuda_splatting.py` | Added render_alpha_cuda |
| `src/model/decoder/decoder_splatting_cuda.py` | Added render_alpha method |
| `src/loss/loss_mask.py` | New file |
| `src/loss/__init__.py` | Registered new loss |
| `config/loss/mask.yaml` | New file |
| `config/experiment/objaverse_white.yaml` | Updated loss config |
