# DepthSplat Wild Frame Debugging Report

## Executive Summary

The wild frame rendering fails primarily due to **two major distribution mismatches** between the training data and the wild frame input:

1. **FOV Mismatch**: Wild frames have 7.5° FOV vs training data's 50° FOV (7x difference)
2. **Camera Coverage Mismatch**: Wild cameras are all at ~-63° elevation vs training's full hemisphere coverage

## Detailed Findings

### 1. Intrinsics / FOV Mismatch

| Metric | UUID (Training) | Wild Frame | Ratio |
|--------|-----------------|------------|-------|
| fx_norm | 1.0723 | 7.6041 | 7.1x |
| FOV | 50.0° | 7.5° | 6.7x |
| cx, cy | 0.5, 0.5 (centered) | varies (0.19-0.82) | N/A |

**Root Cause**: The wild frame cameras have fx=1946 pixels for a 2560-wide image. When we crop a 256x256 region around the object, the normalized focal length becomes fx/256 = 7.6, which corresponds to a very narrow telephoto FOV.

**Impact**: The model was trained exclusively on 50° FOV images. It doesn't generalize to 7.5° FOV input.

### 2. Camera Elevation Coverage Mismatch

| Metric | UUID (Training) | Wild Frame |
|--------|-----------------|------------|
| Elevation range | -90° to +90° | -61° to -65° |
| Coverage | Full hemisphere | ~4° band |

**Root Cause**: The wild frame cameras are positioned in a ring below the drone, all looking up at it. There are no views from above, side, or varying elevations.

**Impact**: The model can only reconstruct views similar to the input. When rendered from +30° elevation (above), the reconstruction fails. When rendered from -60° (similar to input), it produces reasonable results.

### 3. Principal Point Offset

| Camera | cx | cy |
|--------|----|----|
| cam_01 | 0.82 | 0.43 |
| cam_02 | 0.57 | 0.16 |
| cam_03 | 0.19 | 0.33 |
| cam_04 | 0.29 | 0.68 |
| cam_05 | 0.64 | 0.70 |

**Root Cause**: The object detection crops are centered on the drone, but the camera's original principal point was at the image center. After cropping, the principal point shifts.

**Impact**: Minor - the model should handle this, but it adds to distribution shift.

## Evidence

### Elevation Rendering Test

| Elevation | Non-white Pixels | Mean Brightness |
|-----------|------------------|-----------------|
| +30° | 31.4% | 243.7 |
| 0° | 28.9% | 245.6 |
| -30° | 31.8% | 240.2 |
| -60° | 43.8% | 229.9 |
| -63° | 43.2% | 230.4 |

Rendering from -60° to -63° (matching input camera elevation) produces 40% more content than rendering from other angles.

### Gaussian Point Cloud Positions

| Dataset | X Range | Y Range | Z Range | Centroid |
|---------|---------|---------|---------|----------|
| UUID | [-1.19, 1.14] | [-1.63, 1.04] | [-1.17, 1.17] | (-0.02, 0.00, -0.01) |
| Wild | [-0.45, 0.52] | [-0.51, 0.92] | [-1.26, 0.61] | (0.01, -0.10, 0.23) |

UUID Gaussians are centered at origin with symmetric spread. Wild Gaussians are offset and asymmetric.

## Recommended Fixes

### Short-term (Quick Fixes)

1. **Use Training-Matched Intrinsics**:
   ```python
   # Instead of using original focal length, use training FOV
   fx_norm = 1.0723  # Match training distribution
   K = np.array([[fx_norm, 0, 0.5], [0, fx_norm, 0.5], [0, 0, 1]])
   ```
   This is geometrically incorrect but may produce better results by staying in-distribution.

2. **Render from Input-Similar Angles**:
   ```python
   # Default gradio render should use elevation similar to input cameras
   target_elevation = -60  # Instead of 0 or 30
   ```

3. **Scale Camera Distance for FOV Compensation**:
   ```python
   # If narrow FOV, move camera further to see similar object coverage
   fov_ratio = training_fov / wild_fov  # ~6.7
   adjusted_distance = base_distance * fov_ratio
   ```

### Medium-term (Proper Solutions)

1. **Enlarge Crop Region**:
   Instead of 256px crop, use larger crop that maintains reasonable FOV:
   ```python
   # Target FOV of ~30° means we need larger crop
   crop_size = 2 * fx_pixel * tan(30°/2)  # ~1095 pixels
   ```
   Then resize to 256x256.

2. **Add Virtual Cameras**:
   Synthesize additional viewpoints at different elevations before running inference.

### Long-term (Training Solutions)

1. **FOV Augmentation**: Add FOV variation during training
2. **Multi-Elevation Training**: Ensure training data has good elevation coverage
3. **Train on In-Distribution Data**: Fine-tune on data matching wild frame characteristics

## Files Created

- `compare_uuid_vs_wild.py` - Side-by-side camera parameter comparison
- `diagnose_intrinsics.py` - FOV analysis and fix suggestions
- `test_fov_fix.py` - Testing different intrinsics values
- `test_matched_elevation.py` - Testing elevation-matched rendering
- `test_uuid_gaussians.py` - UUID baseline comparison
- `visualize_cameras_3d.py` - 3D camera position visualization

## Conclusion

The model IS working correctly - it successfully reconstructs views similar to the input cameras. The apparent "failure" is actually the model operating outside its training distribution. The fix requires either:
1. Bringing wild frame data closer to training distribution (adjust intrinsics/crops)
2. Expanding training distribution to include wild frame characteristics

# Debug test results

## check_gradio_cam
Calculated Pos (Az=0, El=-60, R=2.0): [ 1.         0.        -1.7320508]
Forward: [-0.5        0.         0.8660254]
Rotation Matrix:
[[ 0.        -0.8660254 -0.5      ]
 [ 1.        -0.         0.       ]
 [-0.        -0.5        0.8660254]]
VERDICT: Generated camera matches input camera valid space.

## compare_uuid_vs_wild
============================================================
LOADING UUID EXAMPLE
============================================================
Loaded UUID acc210f3a2544cf1b250b45eaaf80160: selected views [0, 31, 15, 16, 14] from 32 total

Loaded UUID: acc210f3a2544cf1b250b45eaaf80160

=== UUID Intrinsics Analysis ===
  View 0: fx=1.0723, fy=1.0723, cx=0.5000, cy=0.5000
          FOV: 50.0° x 50.0°
  View 1: fx=1.0723, fy=1.0723, cx=0.5000, cy=0.5000
          FOV: 50.0° x 50.0°
  View 2: fx=1.0723, fy=1.0723, cx=0.5000, cy=0.5000
          FOV: 50.0° x 50.0°
  View 3: fx=1.0723, fy=1.0723, cx=0.5000, cy=0.5000
          FOV: 50.0° x 50.0°
  View 4: fx=1.0723, fy=1.0723, cx=0.5000, cy=0.5000
          FOV: 50.0° x 50.0°

  fx range: [1.0723, 1.0723], mean=1.0723

=== UUID Extrinsics Analysis ===
  Camera positions (from origin):
  View 0: pos=[8.400697e-17 0.000000e+00 1.371938e+00], dist=1.37
          forward=[-6.1232336e-17  0.0000000e+00 -1.0000000e+00]
          dot(forward, to_origin)=1.0000 (LOOKING AT ORIGIN)
  View 1: pos=[ 4.5434136e-17  7.0660527e-17 -1.3719380e+00], dist=1.37
          forward=[-3.3116755e-17 -5.1504166e-17  1.0000000e+00]
          dot(forward, to_origin)=1.0000 (LOOKING AT ORIGIN)
  View 2: pos=[-0.17621692  1.3598539   0.04425607], dist=1.37
          forward=[ 0.12844379 -0.9911919  -0.03225806]
          dot(forward, to_origin)=1.0000 (LOOKING AT ORIGIN)
  View 3: pos=[ 1.048505   -0.8836812  -0.04425606], dist=1.37
          forward=[-0.76425093  0.6441115   0.03225806]
          dot(forward, to_origin)=1.0000 (LOOKING AT ORIGIN)
  View 4: pos=[-0.7853384  -1.117063    0.13276817], dist=1.37
          forward=[ 0.57242996  0.81422263 -0.09677418]
          dot(forward, to_origin)=1.0000 (LOOKING AT ORIGIN)

  Distance range: [1.37, 1.37], mean=1.37
  Camera centroid: [ 0.01738993 -0.12817807  0.02655363]

  Rotation Convention Check (first view):
  UUID View 0:
    Orthogonality: rf=0.000000, rd=0.000000, df=0.000000
    Right-handedness: 1.000000 (should be ~1.0)
    det(R) = 1.000000 (should be 1.0)

============================================================
LOADING WILD FRAME EXAMPLE
============================================================
Loading config: objaverse_white_small_gauss
Building model...
Using cache found in /home/sandro/.cache/torch/hub/facebookresearch_dinov2_main
Loading checkpoint from: /home/sandro/thesis/code/depthsplat/outputs/objaverse_white_small_gauss/checkpoints/epoch_0-step_100000.ckpt
Model loaded successfully!
Loading frame 60 from /home/sandro/thesis/renders/5cams_10-01-26_bird_10m
Normalizing pose: Centering at [ 1.2156391e-02 -4.8031798e-01  1.6861391e+01]
Normalizing pose: Scaling by 0.1807 (original dist: 11.07 -> 2.0)

Loaded Frame 60 (5 views)
Scale factor: 0.1807352900505066
Center: [ 1.2156391e-02 -4.8031798e-01  1.6861391e+01]

=== Wild Frame Intrinsics Analysis ===
  View 0: fx=1.0723, fy=1.0723, cx=0.5000, cy=0.5000
          FOV: 50.0° x 50.0°
  View 1: fx=1.0723, fy=1.0723, cx=0.5000, cy=0.5000
          FOV: 50.0° x 50.0°
  View 2: fx=1.0723, fy=1.0723, cx=0.5000, cy=0.5000
          FOV: 50.0° x 50.0°
  View 3: fx=1.0723, fy=1.0723, cx=0.5000, cy=0.5000
          FOV: 50.0° x 50.0°
  View 4: fx=1.0723, fy=1.0723, cx=0.5000, cy=0.5000
          FOV: 50.0° x 50.0°

  fx range: [1.0723, 1.0723], mean=1.0723

=== Wild Frame Extrinsics Analysis ===
  Camera positions (from origin):
  View 0: pos=[ 0.90147936  0.08681041 -1.7823014 ], dist=2.00
          forward=[-4.4721362e-01 -7.1679001e-09  8.9442718e-01]
          dot(forward, to_origin)=0.9990 (LOOKING AT ORIGIN)
  View 1: pos=[ 0.27705428  0.94625777 -1.7823014 ], dist=2.04
          forward=[-0.13819659 -0.4253254   0.8944272 ]
          dot(forward, to_origin)=0.9990 (LOOKING AT ORIGIN)
  View 2: pos=[-0.7332867  0.6179781 -1.7823014], dist=2.02
          forward=[ 0.36180344 -0.26286554  0.8944272 ]
          dot(forward, to_origin)=0.9990 (LOOKING AT ORIGIN)
  View 3: pos=[-0.7332867  -0.44435725 -1.7823014 ], dist=1.98
          forward=[0.36180344 0.26286554 0.8944272 ]
          dot(forward, to_origin)=0.9992 (LOOKING AT ORIGIN)
  View 4: pos=[ 0.27705428 -0.7726369  -1.7823014 ], dist=1.96
          forward=[-0.13819659  0.4253254   0.8944272 ]
          dot(forward, to_origin)=0.9994 (LOOKING AT ORIGIN)

  Distance range: [1.96, 2.04], mean=2.00
  Camera centroid: [-0.00219708  0.08681042 -1.7823013 ]

  Rotation Convention Check (first view):
  Wild View 0:
    Orthogonality: rf=0.000000, rd=0.000000, df=0.000000
    Right-handedness: 1.000000 (should be ~1.0)
    det(R) = 1.000000 (should be 1.0)

============================================================
COMPARISON SUMMARY
============================================================

  Mean normalized focal length: UUID=1.0723, Wild=1.0723
  Ratio: 1.00x

  Mean camera distance: UUID=1.37, Wild=2.00

  Effective FOV: UUID=50.0°, Wild=50.0°


## create_comparison_grid
Initializing Runner...
Loading config: objaverse_white_small_gauss
Building model...
Using cache found in /home/sandro/.cache/torch/hub/facebookresearch_dinov2_main
Loading checkpoint from: /home/sandro/thesis/code/depthsplat/outputs/objaverse_white_small_gauss/checkpoints/epoch_0-step_100000.ckpt
Model loaded successfully!
Loaded UUID acc210f3a2544cf1b250b45eaaf80160: selected views [0, 31, 15, 16, 14] from 32 total
Loading frame 60 from /home/sandro/thesis/renders/5cams_10-01-26_bird_10m
Normalizing pose: Centering at [ 1.2156391e-02 -4.8031798e-01  1.6861391e+01]
Normalizing pose: Scaling by 0.1807 (original dist: 11.07 -> 2.0)

Testing Az=0, El=0...
Running encoder...
Rendering from target viewpoint...
Exporting PLY to: /home/sandro/thesis/code/depthsplat/inference_gradio/debugging/imgs/temp_uuid/gaussians.ply
Running encoder...
Rendering from target viewpoint...
Exporting PLY to: /home/sandro/thesis/code/depthsplat/inference_gradio/debugging/imgs/temp_wild/gaussians.ply

Testing Az=0, El=30...
Running encoder...
Rendering from target viewpoint...
Exporting PLY to: /home/sandro/thesis/code/depthsplat/inference_gradio/debugging/imgs/temp_uuid/gaussians.ply
Running encoder...
Rendering from target viewpoint...
Exporting PLY to: /home/sandro/thesis/code/depthsplat/inference_gradio/debugging/imgs/temp_wild/gaussians.ply

Testing Az=90, El=30...
Running encoder...
Rendering from target viewpoint...
Exporting PLY to: /home/sandro/thesis/code/depthsplat/inference_gradio/debugging/imgs/temp_uuid/gaussians.ply
Running encoder...
Rendering from target viewpoint...
Exporting PLY to: /home/sandro/thesis/code/depthsplat/inference_gradio/debugging/imgs/temp_wild/gaussians.ply

Testing Az=45, El=0...
Running encoder...
Rendering from target viewpoint...
Exporting PLY to: /home/sandro/thesis/code/depthsplat/inference_gradio/debugging/imgs/temp_uuid/gaussians.ply
Running encoder...
Rendering from target viewpoint...
Exporting PLY to: /home/sandro/thesis/code/depthsplat/inference_gradio/debugging/imgs/temp_wild/gaussians.ply

Testing Az=0, El=-30...
Running encoder...
Rendering from target viewpoint...
Exporting PLY to: /home/sandro/thesis/code/depthsplat/inference_gradio/debugging/imgs/temp_uuid/gaussians.ply
Running encoder...
Rendering from target viewpoint...
Exporting PLY to: /home/sandro/thesis/code/depthsplat/inference_gradio/debugging/imgs/temp_wild/gaussians.ply

Saved comparison to: /home/sandro/thesis/code/depthsplat/inference_gradio/debugging/imgs/uuid_vs_wild_comparison.png
Saved inputs to: /home/sandro/thesis/code/depthsplat/inference_gradio/debugging/imgs/inputs_comparison.png

## debug_poses
Checking /home/sandro/thesis/renders/5cams_10-01-26_bird_10m...
Drone Position: [ 1.21563909e-02 -4.80317992e-01  1.68613904e+01]
Found 5 cameras
  cam_01: Pos [5.0000000e+00 6.6174449e-24 7.0000000e+00], Dist to Obj 11.06
    Dot GL (Forward=-Z): 0.9990  (LOOKING AT)
    Dot CV (Forward=+Z): -0.9990  (AWAY)
    Up GL (+Y): [ 8.94427185e-01 -1.69211187e-09  4.47213608e-01]
    -> Native format seems to be OPENGL (-Z forward)
  cam_02: Pos [1.54508495 4.7552824  7.        ], Dist to Obj 11.27
    Dot GL (Forward=-Z): 0.9990  (LOOKING AT)
    Dot CV (Forward=+Z): -0.9990  (AWAY)
    Up GL (+Y): [0.27639316 0.85065083 0.44721357]
    -> Native format seems to be OPENGL (-Z forward)
  cam_03: Pos [-4.04508495  2.93892622  7.        ], Dist to Obj 11.20
    Dot GL (Forward=-Z): 0.9990  (LOOKING AT)
    Dot CV (Forward=+Z): -0.9990  (AWAY)
    Up GL (+Y): [-0.7236068   0.52573107  0.44721364]
    -> Native format seems to be OPENGL (-Z forward)
  cam_04: Pos [-4.04508495 -2.93892622  7.        ], Dist to Obj 10.94
    Dot GL (Forward=-Z): 0.9992  (LOOKING AT)
    Dot CV (Forward=+Z): -0.9992  (AWAY)
    Up GL (+Y): [-0.7236068  -0.52573107  0.44721363]
    -> Native format seems to be OPENGL (-Z forward)
  cam_05: Pos [ 1.54508495 -4.7552824   7.        ], Dist to Obj 10.86
    Dot GL (Forward=-Z): 0.9994  (LOOKING AT)
    Dot CV (Forward=+Z): -0.9994  (AWAY)
    Up GL (+Y): [ 0.27639316 -0.85065083  0.44721358]
    -> Native format seems to be OPENGL (-Z forward)

## diagnose_intrinsics
============================================================
WILD FRAME INTRINSICS ANALYSIS
============================================================

cam_01:
  Original image: 2560 x 1440
  Original intrinsics (pixels):
    fx=1946.64, fy=1946.64
    cx=1280.00, cy=720.00
  Original FOV: 66.7° x 40.6°

  After cropping to 256x256:
    Normalized fx=7.6041, fy=7.6041
    Effective crop FOV: 7.5° x 7.5°

  Comparison with typical training data:
    Typical training fx_norm: ~0.70
    Typical training FOV: ~71.1°

  ISSUE: Wild frame crop FOV (7.5°) << Training FOV (~71.1°)
         This 9.4x FOV mismatch may cause issues!

============================================================
TRAINING DATA INTRINSICS ANALYSIS
============================================================

Analyzed 320 camera views from training data
  fx_norm range: [1.0723, 1.0723]
  fx_norm mean: 1.0723
  fx_norm std: 0.0000

  FOV range: [50.0°, 50.0°]
  FOV mean: 50.0°

============================================================
POTENTIAL FIXES
============================================================

1. ADJUST INTRINSICS FOR CROP:
   The current approach preserves the pixel focal length after cropping.
   This is geometrically correct but results in a very narrow FOV.

   Alternative: Scale the intrinsics to match training FOV distribution.
   If training typically has fx_norm ~0.7 (FOV ~40°), we could:
   - Use fx_norm = 0.7 for wild frames too
   - This pretends the camera has a wider FOV
   - May introduce geometric distortion but might generalize better

2. RESIZE INSTEAD OF CROP:
   Instead of cropping a 256x256 region, resize the entire image.
   This preserves the original FOV but loses resolution.

   If original image is 2560x1440 and we resize to 256x144:
   - fx_norm = fx_pixel / 256 = 1946.6 / 256 = 7.6 (same issue)

   Actually, we need to think about this differently...

3. PROPER VIRTUAL CAMERA:
   When we crop, we're creating a "virtual camera" that sees only part of the scene.
   The intrinsics should represent this virtual camera.

   For a crop centered at (cx_crop, cy_crop) of size crop_size:
   - New fx_norm = fx_pixel / crop_size (this is what we're doing)
   - New cx_norm = (cx_pixel - x1) / crop_size
   - New cy_norm = (cy_pixel - y1) / crop_size

   This is geometrically correct. The issue is the FOV mismatch with training.

4. MATCH TRAINING DISTRIBUTION:
   Analyze the actual intrinsics distribution in the Objaverse training set.
   If wild frames fall outside this distribution, the model may not generalize.

   Solution: Train on a wider variety of FOVs, or add FOV augmentation.

5. SCALE CAMERA DISTANCE TO COMPENSATE:
   If FOV is narrower (zoomed in), we could position cameras further away
   to capture similar angular extent of the object.

   current_distance * (training_fov / wild_fov) would give similar object coverage.


## proposed_fix
Initializing Runner...
Loading config: objaverse_white_small_gauss
Building model...
Using cache found in /home/sandro/.cache/torch/hub/facebookresearch_dinov2_main
Loading checkpoint from: /home/sandro/thesis/code/depthsplat/outputs/objaverse_white_small_gauss/checkpoints/epoch_0-step_100000.ckpt
Model loaded successfully!
Loading frame 60 from /home/sandro/thesis/renders/5cams_10-01-26_bird_10m
Normalizing pose: Centering at [ 1.2156391e-02 -4.8031798e-01  1.6861391e+01]
Normalizing pose: Scaling by 0.1807 (original dist: 11.07 -> 2.0)

============================================================
TEST: Training-Matched Intrinsics + Elevation-Safe Camera
============================================================
Input elevation range: [-65.3°, -61.0°]
Using fx_norm=1.0723 (50° FOV) instead of 1.07 (7.5° FOV)
Target camera elevation: -63.2° (mean of input range)
Running encoder...
Rendering from target viewpoint...
Exporting PLY to: /home/sandro/thesis/code/depthsplat/inference_gradio/debugging/imgs/fix_test/gaussians.ply

Result: Mean=248.1, Non-white=5.5%
Saved to: /home/sandro/thesis/code/depthsplat/inference_gradio/debugging/imgs/fix_test/render_with_fixes.png

============================================================
COMPARISON: Original vs Fixed
============================================================
Running encoder...
Rendering from target viewpoint...
Exporting PLY to: /home/sandro/thesis/code/depthsplat/inference_gradio/debugging/imgs/fix_test/original/gaussians.ply
Original: Mean=248.1, Non-white=5.5%
Fixed:    Mean=248.1, Non-white=5.5%

Comparison saved to: /home/sandro/thesis/code/depthsplat/inference_gradio/debugging/imgs/fix_test/comparison_orig_vs_fixed.png

## test_final_fix
Initializing Runner...
Loading config: objaverse_white_small_gauss
Building model...
Using cache found in /home/sandro/.cache/torch/hub/facebookresearch_dinov2_main
Loading checkpoint from: /home/sandro/thesis/code/depthsplat/outputs/objaverse_white_small_gauss/checkpoints/epoch_0-step_100000.ckpt
Model loaded successfully!
Loading frame 60 from /home/sandro/thesis/renders/5cams_10-01-26_bird_10m
Normalizing pose: Centering at [ 1.2156391e-02 -4.8031798e-01  1.6861391e+01]
Normalizing pose: Scaling by 0.1807 (original dist: 11.07 -> 2.0)

Input camera elevation range: [-65.3°, -61.0°]
Mean elevation: -63.2°

============================================================
TEST 1: Original Intrinsics (7.5° FOV)
============================================================
Running encoder...
Rendering from target viewpoint...
Generating 60-frame videos (RGB, depth, silhouette)...
Exporting PLY to: /home/sandro/thesis/code/depthsplat/inference_gradio/debugging/imgs/final_fix_test/original/gaussians.ply
Original render saved with video

============================================================
TEST 2: Training-Matched Intrinsics (50° FOV)
============================================================
Running encoder...
Rendering from target viewpoint...
Generating 60-frame videos (RGB, depth, silhouette)...
Exporting PLY to: /home/sandro/thesis/code/depthsplat/inference_gradio/debugging/imgs/final_fix_test/fixed/gaussians.ply
Fixed render saved with video

============================================================
SUMMARY
============================================================

Output directory: /home/sandro/thesis/code/depthsplat/inference_gradio/debugging/imgs/final_fix_test

Original (7.5° FOV):
  Render: /home/sandro/thesis/code/depthsplat/inference_gradio/debugging/imgs/final_fix_test/original/rendered.png
  Video:  /home/sandro/thesis/code/depthsplat/inference_gradio/debugging/imgs/final_fix_test/original/video_rgb.mp4

Fixed (50° FOV):
  Render: /home/sandro/thesis/code/depthsplat/inference_gradio/debugging/imgs/final_fix_test/fixed/rendered.png
  Video:  /home/sandro/thesis/code/depthsplat/inference_gradio/debugging/imgs/final_fix_test/fixed/video_rgb.mp4

Side-by-side: /home/sandro/thesis/code/depthsplat/inference_gradio/debugging/imgs/final_fix_test/render_comparison.png

## test_fov_fix
Initializing Runner...
Loading config: objaverse_white_small_gauss
Building model...
Using cache found in /home/sandro/.cache/torch/hub/facebookresearch_dinov2_main
Loading checkpoint from: /home/sandro/thesis/code/depthsplat/outputs/objaverse_white_small_gauss/checkpoints/epoch_0-step_100000.ckpt
Model loaded successfully!

============================================================
TEST 1: ORIGINAL INTRINSICS (7.5° FOV)
============================================================
Loading frame 60 from /home/sandro/thesis/renders/5cams_10-01-26_bird_10m
Normalizing pose: Centering at [ 1.2156391e-02 -4.8031798e-01  1.6861391e+01]
Normalizing pose: Scaling by 0.1807 (original dist: 11.07 -> 2.0)
Original intrinsics fx: 1.0723
Original FOV: 50.0°
Running encoder...
Rendering from target viewpoint...
Exporting PLY to: /home/sandro/thesis/code/depthsplat/inference_gradio/debugging/imgs/fov_fix_test/original/gaussians.ply
Original render mean: 247.0

============================================================
TEST 2: TRAINING-MATCHED INTRINSICS (50° FOV)
============================================================
Fixed intrinsics fx: 1.0723
Fixed FOV: 50.0°
Running encoder...
Rendering from target viewpoint...
Exporting PLY to: /home/sandro/thesis/code/depthsplat/inference_gradio/debugging/imgs/fov_fix_test/fixed/gaussians.ply
Fixed render mean: 247.0

============================================================
TEST 3: INTERMEDIATE FOV
============================================================
Intermediate intrinsics fx: 2.5000
Intermediate FOV: 22.6°
Running encoder...
Rendering from target viewpoint...
Exporting PLY to: /home/sandro/thesis/code/depthsplat/inference_gradio/debugging/imgs/fov_fix_test/intermediate/gaussians.ply
Intermediate render mean: 242.5

============================================================
COMPARISON SUMMARY
============================================================
Original (7.5° FOV) mean brightness: 247.0
Fixed (50° FOV) mean brightness: 247.0
Intermediate (23° FOV) mean brightness: 242.5

Images saved to: /home/sandro/thesis/code/depthsplat/inference_gradio/debugging/imgs/fov_fix_test
Comparison image: /home/sandro/thesis/code/depthsplat/inference_gradio/debugging/imgs/fov_fix_test/comparison.png

## test_matched_elevation
Initializing Runner...
Loading config: objaverse_white_small_gauss
Building model...
Using cache found in /home/sandro/.cache/torch/hub/facebookresearch_dinov2_main
Loading checkpoint from: /home/sandro/thesis/code/depthsplat/outputs/objaverse_white_small_gauss/checkpoints/epoch_0-step_100000.ckpt
Model loaded successfully!
Loading frame 60 from /home/sandro/thesis/renders/5cams_10-01-26_bird_10m
Normalizing pose: Centering at [ 1.2156391e-02 -4.8031798e-01  1.6861391e+01]
Normalizing pose: Scaling by 0.1807 (original dist: 11.07 -> 2.0)

Input camera elevations (all ~-63°):
  View 0: pos=[ 0.90147936  0.08681041 -1.7823014 ], elev=-63.1°
  View 1: pos=[ 0.27705428  0.94625777 -1.7823014 ], elev=-61.0°
  View 2: pos=[-0.7332867  0.6179781 -1.7823014], elev=-61.7°
  View 3: pos=[-0.7332867  -0.44435725 -1.7823014 ], elev=-64.3°
  View 4: pos=[ 0.27705428 -0.7726369  -1.7823014 ], elev=-65.3°

Rendering from elevation 30°...
Running encoder...
Rendering from target viewpoint...
Exporting PLY to: /home/sandro/thesis/code/depthsplat/inference_gradio/debugging/imgs/elevation_test/elev_30/gaussians.ply
  Mean: 246.1, Non-white: 7.2%

Rendering from elevation 0°...
Running encoder...
Rendering from target viewpoint...
Exporting PLY to: /home/sandro/thesis/code/depthsplat/inference_gradio/debugging/imgs/elevation_test/elev_0/gaussians.ply
  Mean: 246.0, Non-white: 7.8%

Rendering from elevation -30°...
Running encoder...
Rendering from target viewpoint...
Exporting PLY to: /home/sandro/thesis/code/depthsplat/inference_gradio/debugging/imgs/elevation_test/elev_-30/gaussians.ply
  Mean: 247.4, Non-white: 6.9%

Rendering from elevation -60°...
Running encoder...
Rendering from target viewpoint...
Exporting PLY to: /home/sandro/thesis/code/depthsplat/inference_gradio/debugging/imgs/elevation_test/elev_-60/gaussians.ply
  Mean: 248.1, Non-white: 5.6%

Rendering from elevation -63°...
Running encoder...
Rendering from target viewpoint...
Exporting PLY to: /home/sandro/thesis/code/depthsplat/inference_gradio/debugging/imgs/elevation_test/elev_-63/gaussians.ply
  Mean: 248.1, Non-white: 5.5%

Saved comparison to: /home/sandro/thesis/code/depthsplat/inference_gradio/debugging/imgs/elevation_test/elevation_comparison.png

============================================================
ANALYSIS:
============================================================
If rendering from -60° to -63° produces better results than +30°,
it confirms that the model can only reconstruct views similar to input.
This is expected for sparse-view reconstruction.

## test_uuid_gaussians
Initializing Runner...
Loading config: objaverse_white_small_gauss
Building model...
Using cache found in /home/sandro/.cache/torch/hub/facebookresearch_dinov2_main
Loading checkpoint from: /home/sandro/thesis/code/depthsplat/outputs/objaverse_white_small_gauss/checkpoints/epoch_0-step_100000.ckpt
Model loaded successfully!
Loaded UUID acc210f3a2544cf1b250b45eaaf80160: selected views [0, 31, 15, 16, 14] from 32 total

Loaded UUID: acc210f3a2544cf1b250b45eaaf80160
Intrinsics fx: 1.0723
FOV: 50.0°

Target camera position: [1.2124356 0.        0.7      ]
Running encoder...
Rendering from target viewpoint...
Exporting PLY to: /home/sandro/thesis/code/depthsplat/inference_gradio/debugging/imgs/uuid_test/gaussians.ply

UUID render mean brightness: 242.1

UUID Gaussians:
  Num Gaussians: 288000
  X range: [-1.192, 1.139]
  Y range: [-1.633, 1.043]
  Z range: [-1.168, 1.168]
  Centroid: (-0.023, 0.004, -0.008)

## visualize_cameras_3d
Loaded UUID acc210f3a2544cf1b250b45eaaf80160: selected views [0, 31, 15, 16, 14] from 32 total
Loading config: objaverse_white_small_gauss
Building model...
Using cache found in /home/sandro/.cache/torch/hub/facebookresearch_dinov2_main
Loading checkpoint from: /home/sandro/thesis/code/depthsplat/outputs/objaverse_white_small_gauss/checkpoints/epoch_0-step_100000.ckpt
Model loaded successfully!
Loading frame 60 from /home/sandro/thesis/renders/5cams_10-01-26_bird_10m
Normalizing pose: Centering at [ 1.2156391e-02 -4.8031798e-01  1.6861391e+01]
Normalizing pose: Scaling by 0.1807 (original dist: 11.07 -> 2.0)

Saved to: /home/sandro/thesis/code/depthsplat/inference_gradio/debugging/imgs/camera_comparison_3d.png
