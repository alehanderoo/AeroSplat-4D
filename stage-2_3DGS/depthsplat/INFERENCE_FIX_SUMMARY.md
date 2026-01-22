# Inference Fix Summary

## Problem
The streaming inference project (`inference/`) produced white reconstructions while the Gradio demo (`inference_gradio/`) worked perfectly. Both systems process in-the-wild data streams with cropped dynamic objects.

## Root Causes Identified

### 1. **Incorrect Object Position Coordinate System**
**Issue**: The object position from Isaac Sim JSON was in OpenGL coordinates (Y-up, Z-back), but the camera extrinsics were being converted to OpenCV coordinates (Y-down, Z-forward). The object position was used for pose normalization (centering) AFTER the coordinate flip was applied to extrinsics, causing misalignment.

**Fix**: Apply the same coordinate flip to the object position before setting it on the calibration service:
```python
# Flip Y and Z axes to convert OpenGL → OpenCV
object_position_opencv = object_position.copy()
object_position_opencv[1] *= -1.0  # Flip Y
object_position_opencv[2] *= -1.0  # Flip Z
```

### 2. **Incorrect Crop Region Handling for Extrinsics**
**Issue**: The streaming inference was passing `crop_regions` to `get_extrinsics_tensor()`, but the Gradio demo uses the ORIGINAL (uncropped) camera extrinsics. Crop regions only affect image preprocessing, not camera positions.

**Why This Matters**:
- When you crop an image, you're NOT moving the camera closer
- You're just extracting a sub-region of the original view
- The camera position stays the same
- Only the intrinsics change (focal length relative to crop size)
- But those are then overridden with training-matched intrinsics anyway!

**Fix**: Don't pass crop_regions when getting extrinsics:
```python
ext_tensor = self.calibration_service.get_extrinsics_tensor(
    device='cpu',
    crop_regions=None,  # Use original extrinsics like Gradio
    use_virtual_camera=False,
)
```

## What Was Already Correct

1. ✅ **Training-matched intrinsics** (fx=1.0723, cx=cy=0.5) - Already set via `use_training_intrinsics=True`
2. ✅ **Coordinate flip** (OpenGL→OpenCV) - Already applied via `apply_coordinate_flip=True`
3. ✅ **Pose normalization** (center + scale to radius 2.0) - Already enabled via `apply_pose_normalization=True`
4. ✅ **Data shim** - Already being applied via GradioReconstructor
5. ✅ **Render camera** - Already using training-matched intrinsics and proper orbit setup

## How the Gradio Demo Works

The working Gradio demo (`load_wild_frame` method):
1. Loads full-resolution images from cameras
2. Crops to region around object (tight crop based on bbox size)
3. Adjusts intrinsics for crop (fx, fy, cx, cy)
4. Loads extrinsics from JSON (original camera positions)
5. **Applies coordinate flip** to extrinsics (OpenGL→OpenCV)
6. **Applies pose normalization** (center at object, scale to radius 2.0)
7. **Overrides ALL intrinsics** with training-matched (fx=1.0723, cx=cy=0.5)
8. Preprocesses images → tensor
9. Builds batch with context + dummy target
10. Applies data_shim
11. Runs encoder → Gaussians
12. Renders with orbit camera (also using training-matched intrinsics)

## How the Streaming Inference NOW Works (After Fix)

The fixed streaming inference:
1. Loads frames from file source or RTSP
2. Gets detections (bbox, object position)
3. Crops to region around object (same tight crop logic)
4. **Applies coordinate flip** to object position (NEW FIX)
5. Sets flipped object position on calibration service
6. Gets intrinsics from calibration service (returns training-matched)
7. Gets extrinsics from calibration service **WITHOUT crop_regions** (NEW FIX)
   - Loads original camera positions from JSON
   - Applies coordinate flip
   - Applies pose normalization using flipped object position
8. Calls GradioReconstructor.reconstruct() with images + extrinsics
9. GradioReconstructor applies same pipeline as Gradio demo
10. Renders with orbit camera (training-matched intrinsics)

## Files Modified

- `inference/pipeline/visualization_pipeline.py`:
  - Added coordinate flip to object position before `set_object_position()`
  - Changed `crop_regions=self._last_crop_regions` → `crop_regions=None` for extrinsics (3 locations)
  - Added debug logging to verify camera positions

## Testing Checklist

- [ ] Verify reconstruction is no longer white
- [ ] Check that Gaussians are centered at origin
- [ ] Verify camera distances are ~2.0 meters from origin
- [ ] Confirm object appears correctly oriented
- [ ] Test with multiple in-the-wild datasets
- [ ] Compare quality with Gradio demo output

## Technical Deep Dive: Why Crop Regions Don't Affect Extrinsics

When you crop an image:
- **Intrinsics change**: The focal length and principal point shift because you're working with a smaller image region
- **Extrinsics DON'T change**: The camera hasn't moved in 3D space; you're just looking at part of its view

Example:
```
Original camera: fx=1000px, cx=960px (on 1920x1080 image)
After crop [400:1400, 300:800]:
  - New image size: 1000x500
  - New cx = 960 - 400 = 560px
  - New fx = 1000px (same physical FOV, but relative to new image size it's different)

But camera position in 3D space? UNCHANGED.
```

The Gradio demo gets this right by:
1. Computing cropped intrinsics
2. Then overriding them with training-matched intrinsics
3. Using original (uncropped) extrinsics

This approach works because the model was trained with ~50° FOV cameras at ~2.0m distance. As long as we match those conditions (via training intrinsics + pose normalization), the model generalizes well.
