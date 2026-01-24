# Monocular Depth Export

**Date:** 2026-01-10  
**Purpose:** Expose monocular depth (DPT upsampler output) for visualization

## Files Modified

### `src/model/encoder/unimatch/mv_unimatch.py`

Added `mono_depth` to `results_dict` after the DPT upsampler computes `residual_depth`:

```python
# Line ~583 (after depth clamping)
mono_depth = residual_depth.squeeze(1)  # [BV, H, W]
mono_depth = rearrange(mono_depth, "(b v) ... -> b v ...", b=b, v=v)  # [B, V, H, W]
results_dict.update({"mono_depth": mono_depth})
```

### `src/model/encoder/encoder_depthsplat.py`

Added `mono_depth` to `visualization_dump` when available:

```python
# Line ~346 (in visualization_dump section)
if "mono_depth" in results_dict:
    visualization_dump["mono_depth"] = results_dict["mono_depth"]
```

## What is Monocular Depth?

The `mono_depth` is the **residual depth** output from the DPT upsampler head. It represents the depth refinement predicted from monocular features (DINOv2) before being combined with the MVS (multi-view stereo) depth estimate.

- **Not** a standalone monocular depth prediction
- Represents the monocular contribution to the final depth
- Can be positive or negative (it's a residual)
- Final depth = MVS depth + mono_depth

## Usage

Pass `visualization_dump={}` to encoder forward call:

```python
visualization_dump = {}
result = model.encoder(context, global_step=0, visualization_dump=visualization_dump)
mono_depth = visualization_dump.get("mono_depth")  # [B, V, H, W]
```

## Updates (2026-01-24)

**Purpose:** Expose coarse MVS depth and clarify monocular depth variable naming.

### `src/model/encoder/unimatch/mv_unimatch.py`

- Renamed local variable `mono_depth` to `mono_depth_residual`.
- Exposed `coarse_mv_depth` (bilinearly upsampled MVS depth).

```python
# Store monocular depth component (DPT upsampler output)
mono_depth_residual = residual_depth.squeeze(1)  # [BV, H, W]
mono_depth_residual = rearrange(mono_depth_residual, "(b v) ... -> b v ...", b=b, v=v)  # [B, V, H, W]
results_dict.update({"mono_depth": mono_depth_residual})

# Store coarse cost-volume depth (before DPT upsampling)
coarse_mv_depth = depth_bilinear.squeeze(1)  # [BV, H, W]
coarse_mv_depth = rearrange(coarse_mv_depth, "(b v) ... -> b v ...", b=b, v=v)  # [B, V, H, W]
results_dict.update({"coarse_mv_depth": coarse_mv_depth})
```

### `src/model/encoder/encoder_depthsplat.py`

- Added `coarse_mv_depth` to `visualization_dump`.

```python
# Add monocular depth residual (DPT upsampler output) if available
if "mono_depth" in results_dict:
    visualization_dump["mono_depth"] = results_dict["mono_depth"]
# Add coarse cost-volume depth (before DPT upsampling) if available
if "coarse_mv_depth" in results_dict:
    visualization_dump["coarse_mv_depth"] = results_dict["coarse_mv_depth"]
```
