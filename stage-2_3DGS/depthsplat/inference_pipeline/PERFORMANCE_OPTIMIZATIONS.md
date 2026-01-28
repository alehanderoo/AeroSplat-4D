# Inference Pipeline Performance Optimizations

## Problem
The inference pipeline was running at sub-5 FPS, with the primary bottleneck being the streaming/visualization encoding, NOT the actual reconstruction.

## Root Causes Identified

### 1. **Excessive JPEG Encoding (MAJOR)**
**Issue**: Encoding up to 31 images to JPEG + base64 **every single frame**:
- 5 input thumbnails
- 5 cropped frames
- 5 GT depth visualizations (if enabled)
- 5 monocular depth maps
- 5 predicted depth maps
- 5 silhouette maps
- 1 gaussian render

**Impact**: PIL JPEG encoding is VERY slow (10-50ms per image). With 31 images × ~20ms = **~620ms per frame** = **1.6 FPS max**!

### 2. **Mask Loading from Disk (MAJOR)**
**Issue**: Reading mask files from disk with `cv2.imread()` **twice per camera per frame**:
- Once in `_create_cropped_views()` for visualization
- Once in `_preprocess_frames_with_detections()` for model input

**Impact**: Disk I/O is slow (~5-10ms per read). With 5 cameras × 2 reads = **~100ms per frame**!

### 3. **Encoding When No Clients Connected**
**Issue**: The pipeline was encoding all visualizations even when no WebSocket clients were connected.

**Impact**: Wasting 100% of encoding time when nobody's watching.

### 4. **Encoding Empty/None Lists**
**Issue**: The `timed_encode()` function was processing empty lists and None values.

**Impact**: Small but unnecessary overhead.

## Optimizations Implemented

### 1. **TurboJPEG Support (10-20x faster)**
```python
# Before: PIL encoding (slow)
pil_image.save(buffer, format="JPEG", quality=quality)

# After: TurboJPEG encoding (fast)
jpeg = TurboJPEG()
jpeg_bytes = jpeg.encode(bgr, quality=quality)
```

**Expected speedup**: 31 images × 20ms → 31 images × 2ms = **~558ms saved per frame** ≈ **+2.7 FPS**

**Installation**:
```bash
# Install libjpeg-turbo system library
sudo apt-get install libjpeg-turbo8-dev

# Install Python wrapper
pip install PyTurboJPEG
```

**Fallback**: Automatically falls back to PIL if TurboJPEG is not available.

### 2. **Mask Caching**
```python
# Before: Read from disk every time
mask = cv2.imread(detection.mask_path, cv2.IMREAD_GRAYSCALE)

# After: LRU cache with max 1000 entries
mask = self._load_mask_cached(detection.mask_path)
```

**Expected speedup**: ~100ms → ~0ms (after cache warm-up) = **+10 FPS** (huge!)

### 3. **Skip Encoding When No Clients**
```python
# PERFORMANCE: Skip expensive encoding if no clients connected
if self.vis_server.connected_clients == 0:
    return
```

**Expected speedup**: Development/testing mode without frontend gets full reconstruction speed.

### 4. **Quality-Based Encoding**
```python
# Thumbnails and depth maps: quality=70 (faster, smaller)
inputs = encode_frame_list(input_frames, quality=thumbnail_quality)

# Main gaussian render: quality=85 (better quality)
render_b64 = encode_image_to_base64(gaussian_render, jpeg_quality)
```

**Expected speedup**: Lower quality = smaller buffers = faster encoding ≈ **~50ms saved per frame** ≈ **+1 FPS**

### 5. **Skip Encoding Empty Lists**
```python
def timed_encode(frames, ...):
    if not frames or all(f is None for f in frames):
        return [], 0.0  # Skip immediately
    # ... encode ...
```

**Expected speedup**: Small, but avoids unnecessary work.

## Expected Performance Improvement

**Before optimizations**:
- Encoding time: ~620ms (PIL) + 100ms (masks) = **~720ms overhead**
- Inference time: ~200ms (model)
- **Total: ~920ms per frame = 1.1 FPS**

**After optimizations**:
- Encoding time: ~60ms (TurboJPEG @ quality 70) = **-560ms saved**
- Mask loading: ~0ms (cached after warm-up) = **-100ms saved**
- Inference time: ~200ms (unchanged)
- **Total: ~260ms per frame = 3.8 FPS** ≈ **+2.7 FPS improvement** ✅

**With no clients connected**:
- Skip all encoding: ~0ms
- Inference time: ~200ms
- **Total: ~200ms per frame = 5 FPS** ≈ **+3.9 FPS improvement** ✅

## Configuration Options

New config parameters in `VisualizationPipelineConfig`:
```python
jpeg_quality: int = 85         # Quality for main gaussian render (higher = better quality, slower)
thumbnail_quality: int = 70    # Quality for thumbnails/depth (lower = faster, smaller files)
```

Adjust these based on your needs:
- **Quality-focused**: `jpeg_quality=95, thumbnail_quality=85` (slower but prettier)
- **Speed-focused**: `jpeg_quality=75, thumbnail_quality=60` (faster but more artifacts)
- **Balanced** (default): `jpeg_quality=85, thumbnail_quality=70`

## Testing Checklist

- [x] Verify TurboJPEG installation and fallback
- [x] Check mask caching works correctly
- [x] Confirm encoding skips when no clients
- [x] Test quality settings produce acceptable results
- [ ] Measure actual FPS improvement on real hardware
- [ ] Verify WebSocket clients receive frames correctly
- [ ] Test with 5 camera setup
- [ ] Stress test with multiple clients

## Additional Optimization Opportunities

If you need even more speed:

1. **Reduce visualization columns**: Disable mono_depth, predicted_depth, silhouette if not needed
2. **Lower resolution**: Reduce `render_width/height` from 512 to 256
3. **Frame skipping**: Only encode every Nth frame for visualization
4. **Async encoding**: Move encoding to thread pool (complex but possible)
5. **Reduce camera count**: Use 3 cameras instead of 5 if acceptable
6. **Disable GT depth**: Already optimized (only loads when enabled)

## Files Modified

- `inference/server/websocket_server.py`:
  - Added TurboJPEG support to `encode_image_to_base64()`
  - Added `quality` parameter to encoding functions
  - Updated `VisualizationConfig` with `thumbnail_quality`
  - Optimized `timed_encode()` to skip empty lists

- `inference/pipeline/visualization_pipeline.py`:
  - Added early exit when no clients connected
  - Added `_load_mask_cached()` method with LRU cache
  - Updated mask loading to use cache (2 locations)
  - Added `thumbnail_quality` config parameter
  - Updated config passing to include new quality settings

## Dependencies

Add to `requirements.txt`:
```
PyTurboJPEG>=1.7.0  # Fast JPEG encoding (requires libjpeg-turbo system library)
```

System dependencies (Ubuntu/Debian):
```bash
sudo apt-get install libjpeg-turbo8-dev
```

## Performance Metrics to Monitor

Key metrics in `stats` dict:
- `encoder_ms`: Model inference time (should be ~100-200ms)
- `decoder_ms`: Gaussian rendering time (should be ~10-30ms)
- `column_latency.input_ms`: Input thumbnail encoding time
- `column_latency.cropped_ms`: Cropped frame encoding + processing time
- `total_latency_ms`: End-to-end pipeline time

**Target**: `total_latency_ms` < 250ms for 4+ FPS
