#!/usr/bin/env python3
"""
Diagnose intrinsics issues between training and wild frames.

Key insight: When cropping from a large image (2560x1440) to a small crop (256x256),
the effective FOV becomes much narrower. This script analyzes whether this FOV
mismatch is causing the rendering issues.
"""

import sys
import json
import numpy as np
from pathlib import Path

sys.path.append("/home/sandro/thesis/code/depthsplat")
sys.path.append("/home/sandro/thesis/code/depthsplat/inference_gradio")


def compute_fov(fx_normalized: float) -> float:
    """Compute horizontal FOV from normalized focal length."""
    # For normalized intrinsics, image spans [0, 1]
    # FOV = 2 * arctan(0.5 / fx)
    return 2 * np.rad2deg(np.arctan(0.5 / fx_normalized))


def analyze_wild_frame_intrinsics(render_dir: str, crop_size: int = 256):
    """Analyze the intrinsics transformation for wild frames."""

    json_path = Path(render_dir) / "drone_camera_observations.json"
    with open(json_path, 'r') as f:
        data = json.load(f)

    print("=" * 60)
    print("WILD FRAME INTRINSICS ANALYSIS")
    print("=" * 60)

    for cam in data.get("cameras", [])[:1]:  # Just first camera
        name = cam["name"]
        intr = cam["intrinsics"]
        res = cam["resolution"]

        fx_pixel = intr["fx"]
        fy_pixel = intr["fy"]
        cx_pixel = intr["cx"]
        cy_pixel = intr["cy"]
        w, h = res["width"], res["height"]

        print(f"\n{name}:")
        print(f"  Original image: {w} x {h}")
        print(f"  Original intrinsics (pixels):")
        print(f"    fx={fx_pixel:.2f}, fy={fy_pixel:.2f}")
        print(f"    cx={cx_pixel:.2f}, cy={cy_pixel:.2f}")

        # Original FOV
        fov_h_orig = 2 * np.rad2deg(np.arctan(w / (2 * fx_pixel)))
        fov_v_orig = 2 * np.rad2deg(np.arctan(h / (2 * fy_pixel)))
        print(f"  Original FOV: {fov_h_orig:.1f}° x {fov_v_orig:.1f}°")

        # After cropping and normalizing
        print(f"\n  After cropping to {crop_size}x{crop_size}:")

        # The fx stays the same in pixels, we just shift principal point
        # Normalization: fx_norm = fx_pixel / crop_size
        fx_norm = fx_pixel / crop_size
        fy_norm = fy_pixel / crop_size

        print(f"    Normalized fx={fx_norm:.4f}, fy={fy_norm:.4f}")

        # Effective FOV of the crop
        fov_h_crop = 2 * np.rad2deg(np.arctan(crop_size / (2 * fx_pixel)))
        fov_v_crop = 2 * np.rad2deg(np.arctan(crop_size / (2 * fy_pixel)))
        print(f"    Effective crop FOV: {fov_h_crop:.1f}° x {fov_v_crop:.1f}°")

        # Compare with typical Objaverse training
        print(f"\n  Comparison with typical training data:")
        typical_fx_norm = 0.7  # Approximate value from Objaverse
        typical_fov = compute_fov(typical_fx_norm)
        print(f"    Typical training fx_norm: ~{typical_fx_norm:.2f}")
        print(f"    Typical training FOV: ~{typical_fov:.1f}°")

        print(f"\n  ISSUE: Wild frame crop FOV ({fov_h_crop:.1f}°) << Training FOV (~{typical_fov:.1f}°)")
        print(f"         This {typical_fov/fov_h_crop:.1f}x FOV mismatch may cause issues!")


def suggest_fixes():
    """Suggest potential fixes for the intrinsics mismatch."""

    print("\n" + "=" * 60)
    print("POTENTIAL FIXES")
    print("=" * 60)

    print("""
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
""")


def analyze_training_intrinsics():
    """Load and analyze intrinsics from training data."""

    import torch
    print("\n" + "=" * 60)
    print("TRAINING DATA INTRINSICS ANALYSIS")
    print("=" * 60)

    data_dir = Path("/mnt/raid0/objaverse/test")
    torch_files = sorted(data_dir.glob("*.torch"))

    if not torch_files:
        print("No training data found")
        return

    # Load first chunk
    chunk = torch.load(torch_files[0])

    fx_values = []
    for scene in chunk[:10]:  # First 10 scenes
        cameras = scene['cameras'].numpy()
        fx_values.extend(cameras[:, 0].tolist())  # fx is first element

    fx_values = np.array(fx_values)
    fov_values = [compute_fov(fx) for fx in fx_values]

    print(f"\nAnalyzed {len(fx_values)} camera views from training data")
    print(f"  fx_norm range: [{fx_values.min():.4f}, {fx_values.max():.4f}]")
    print(f"  fx_norm mean: {fx_values.mean():.4f}")
    print(f"  fx_norm std: {fx_values.std():.4f}")
    print(f"\n  FOV range: [{min(fov_values):.1f}°, {max(fov_values):.1f}°]")
    print(f"  FOV mean: {np.mean(fov_values):.1f}°")


if __name__ == "__main__":
    render_dir = "/home/sandro/thesis/renders/5cams_10-01-26_bird_10m"
    analyze_wild_frame_intrinsics(render_dir)
    analyze_training_intrinsics()
    suggest_fixes()
