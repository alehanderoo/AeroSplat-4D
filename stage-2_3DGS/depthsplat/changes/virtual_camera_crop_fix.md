# Virtual Camera Transformation for Cropped Regions

**Date:** 2026-01-11
**Author:** Claude Opus 4.5 (assisted)
**Files Modified:**
- `inference/services/camera_calibration_service.py`
- `inference/services/detection_service.py`
- `inference/pipeline/visualization_pipeline.py`
- `inference/config/pipeline_config.yaml`
- `inference/tests/test_virtual_camera.py` (new)

## Problem Statement

The inference pipeline was failing to produce meaningful depth predictions and silhouettes when cropping input images around detected objects (birds). Two critical issues were identified:

### Issue 1: Extremely High Normalized Focal Length

When cropping a 256x256 region from a 2560x1440 image:
- Original: `fx_norm = 1946.6 / 2560 = 0.76`
- After crop: `fx_norm = 1946.6 / 256 = 7.6`

The normalized focal length became ~10x higher than the model's training distribution (typically 0.5-1.5), causing the depth cost volume matching to fail.

### Issue 2: Broken Multi-View Epipolar Geometry

Each camera was cropped independently at different pixel locations (centered on where the bird appeared in each view). This broke the epipolar geometry:
- Camera 1 crops at (1200, 600)
- Camera 2 crops at (800, 700)
- etc.

The extrinsics remained unchanged, so the model received inconsistent camera poses that didn't match the actual viewing directions of the cropped images.

## Solution: Virtual Camera Transformation

When cropping a region that's not centered on the principal point, we create a "virtual camera" that has been rotated so the crop center lies on the optical axis.

### Mathematical Approach

1. **Compute crop center ray**: The center of the crop region corresponds to a 3D ray in camera coordinates:
   ```
   ray = [(crop_center_x - cx) / fx, (crop_center_y - cy) / fy, 1]
   ray = ray / ||ray||
   ```

2. **Compute rotation**: Using Rodrigues' formula, compute the rotation matrix `R` that aligns this ray with the optical axis `[0, 0, 1]`:
   ```
   v = ray × [0, 0, 1]
   R = I + [v]_x + [v]_x^2 * (1 - cos(θ)) / sin²(θ)
   ```

3. **Transform extrinsics**: Apply the rotation to the camera-to-world matrix:
   ```
   R_virtual = R_c2w @ R_crop.T
   ```
   The translation remains unchanged (camera position stays the same).

4. **Center principal point**: With the virtual rotation applied, the principal point becomes the image center:
   ```
   cx_norm = 0.5
   cy_norm = 0.5
   ```

## Code Changes

### `camera_calibration_service.py`

Added new function `compute_crop_rotation()`:
```python
def compute_crop_rotation(cx, cy, fx, fy, crop_center_x, crop_center_y) -> np.ndarray:
    """Compute rotation matrix that aligns crop center with optical axis."""
```

Added new method `get_virtual_extrinsics()`:
```python
def get_virtual_extrinsics(self, camera_name, crop_region) -> np.ndarray:
    """Get virtual camera extrinsics for a cropped region."""
```

Updated `get_normalized_intrinsics()`:
- Added `use_virtual_camera` parameter (default: `True`)
- When True, centers the principal point at (0.5, 0.5)

Updated `get_intrinsics_tensor()` and `get_extrinsics_tensor()`:
- Added `use_virtual_camera` parameter
- Both methods now accept `crop_regions` to compute virtual camera transforms

### `visualization_pipeline.py`

Updated `_run_inference_with_gaussians()` and `_get_gaussians_object()`:
- Now pass `crop_regions` to both intrinsics and extrinsics tensor methods
- Enable `use_virtual_camera=True` for correct multi-view geometry

## Testing

A test file `tests/test_virtual_camera.py` verifies:
1. `compute_crop_rotation()` returns identity for centered crops
2. Off-center crops produce valid rotation matrices
3. Virtual intrinsics have centered principal point (0.5, 0.5)
4. Virtual extrinsics preserve translation but update rotation
5. Tensor methods work correctly with virtual camera mode

Run tests with:
```bash
conda activate depthsplat
python tests/test_virtual_camera.py
```

## Additional Fix: Larger Crop Regions

### Problem

Even with correct virtual camera geometry, a 256x256 crop from a 2560x1440 image results in a normalized focal length of ~7.6, which is far outside the model's training distribution (typically 0.5-1.5).

### Solution

Increased the default crop size from 256 to 1024 pixels:
- `crop_size=256` → `fx_norm = 1946.6/256 = 7.6` (too high)
- `crop_size=1024` → `fx_norm = 1946.6/1024 = 1.9` (reasonable)

The larger crop region is then resized to the model's input size (256x256), preserving the correct FOV relationship.

### Code Changes

**`config/pipeline_config.yaml`:**
```yaml
detection:
  crop_size: 1024  # Was 256
```

**`pipeline/visualization_pipeline.py`:**
```python
crop_size: int = 1024  # Was 256
```

**`services/detection_service.py`:**
Modified `get_crop_region()` to use `crop_size` as a MINIMUM:
```python
# Start with configured crop size (ensures consistent focal length)
half_size = crop_size / 2

# Expand if object is larger than crop_size
if use_bbox and self.bbox is not None:
    bbox_half_size = max(bbox_width, bbox_height) * 1.5 / 2
    half_size = max(half_size, bbox_half_size)  # Use larger of the two
```

This ensures:
1. Crops are at least 1024x1024 (consistent focal length ~1.9)
2. Crops expand if the object is larger
3. Objects remain centered in the crop

## Remaining Considerations

### Near/Far Planes

The current near/far planes are hardcoded for Objaverse training data:
```python
"near": 0.55
"far": 2.54
```

For birds at longer distances, these may need adjustment based on:
- Estimated object depth from detection
- Scene-specific depth ranges

### Crop Size vs Object Size Trade-off

With `crop_size=1024`, the bird will appear smaller in the cropped view compared to `crop_size=256`. This is intentional - it gives the model a more "normal" FOV that matches its training distribution. The bird should still be visible and centered, just with more context around it.

## Usage Example

```python
from services import CameraCalibrationService

service = CameraCalibrationService(json_path)

# Get virtual camera parameters for crops
intrinsics = service.get_intrinsics_tensor(
    device=device,
    crop_regions=[(x1, y1, x2, y2), ...],  # Per-camera crops
    use_virtual_camera=True,
)
extrinsics = service.get_extrinsics_tensor(
    device=device,
    crop_regions=[(x1, y1, x2, y2), ...],
    use_virtual_camera=True,
)
```
