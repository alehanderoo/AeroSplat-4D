#!/usr/bin/env python3
"""
Proposed fixes for wild frame loading in runner.py.

This module demonstrates the fixes needed to address:
1. FOV mismatch (7.5° vs 50°)
2. Camera elevation coverage

Apply these changes to runner.py to fix the wild frame rendering issues.
"""

import numpy as np


# ============================================================================
# FIX 1: Larger Crop with Padding
# ============================================================================

def compute_fov_preserving_crop(
    fx_pixel: float,
    target_fov_degrees: float = 40.0,
    max_crop_size: int = 1024,
    min_crop_size: int = 256,
) -> int:
    """
    Compute crop size that preserves a reasonable FOV.

    Args:
        fx_pixel: Original focal length in pixels
        target_fov_degrees: Desired FOV (default 40° is reasonable for objects)
        max_crop_size: Maximum allowed crop size
        min_crop_size: Minimum crop size

    Returns:
        Crop size in pixels
    """
    # FOV = 2 * arctan(crop_size / (2 * fx))
    # crop_size = 2 * fx * tan(FOV / 2)
    target_fov_rad = np.deg2rad(target_fov_degrees)
    ideal_crop_size = 2 * fx_pixel * np.tan(target_fov_rad / 2)

    # Clamp to valid range
    crop_size = int(np.clip(ideal_crop_size, min_crop_size, max_crop_size))

    return crop_size


# ============================================================================
# FIX 2: Training-Matched Intrinsics Override
# ============================================================================

TRAINING_FX_NORM = 1.0723  # From training data analysis
TRAINING_FOV_DEGREES = 50.0


def create_training_matched_intrinsics(num_views: int) -> np.ndarray:
    """
    Create intrinsics that match the training data distribution.

    This is a "hack" that ignores the true camera intrinsics and uses
    values that match what the model was trained on. This may produce
    geometrically incorrect results but can help with generalization.

    Args:
        num_views: Number of camera views

    Returns:
        [num_views, 3, 3] intrinsics matrices
    """
    K = np.array([
        [TRAINING_FX_NORM, 0, 0.5],
        [0, TRAINING_FX_NORM, 0.5],
        [0, 0, 1],
    ], dtype=np.float32)

    return np.stack([K] * num_views)


# ============================================================================
# FIX 3: Better Default Camera Generation
# ============================================================================

def get_input_camera_elevation_range(extrinsics: np.ndarray) -> tuple:
    """
    Analyze input camera elevations to determine valid rendering range.

    Args:
        extrinsics: [V, 4, 4] camera-to-world matrices

    Returns:
        (min_elevation, max_elevation) in degrees
    """
    positions = extrinsics[:, :3, 3]
    distances = np.linalg.norm(positions, axis=1)
    elevations = np.rad2deg(np.arcsin(positions[:, 2] / distances))

    return elevations.min(), elevations.max()


def generate_safe_target_camera(
    extrinsics: np.ndarray,
    azimuth: float = 0.0,
    elevation_override: float = None,
    distance: float = 1.0,
    base_radius: float = 2.0,
) -> np.ndarray:
    """
    Generate a target camera that stays within the valid rendering range.

    If no elevation_override is provided, uses the mean elevation of input cameras.

    Args:
        extrinsics: [V, 4, 4] input camera extrinsics
        azimuth: Azimuth angle in degrees
        elevation_override: If provided, use this elevation
        distance: Distance multiplier
        base_radius: Base radius from origin

    Returns:
        [4, 4] camera-to-world matrix
    """
    if elevation_override is None:
        min_el, max_el = get_input_camera_elevation_range(extrinsics)
        elevation = (min_el + max_el) / 2  # Use mean elevation
    else:
        elevation = elevation_override

    radius = base_radius * distance
    azimuth_rad = np.deg2rad(azimuth)
    elevation_rad = np.deg2rad(elevation)

    # Spherical to Cartesian
    x = radius * np.cos(elevation_rad) * np.cos(azimuth_rad)
    y = radius * np.cos(elevation_rad) * np.sin(azimuth_rad)
    z = radius * np.sin(elevation_rad)

    cam_pos = np.array([x, y, z], dtype=np.float32)

    # Look at origin
    forward = -cam_pos / np.linalg.norm(cam_pos)

    # Up vector
    world_up = np.array([0, 0, 1], dtype=np.float32)
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-6:
        world_up = np.array([0, 1, 0], dtype=np.float32)
        right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)

    up = np.cross(right, forward)
    down = -up

    # OpenCV convention: columns are [right, down, forward]
    R = np.stack([right, down, forward], axis=1)

    extrinsic = np.eye(4, dtype=np.float32)
    extrinsic[:3, :3] = R
    extrinsic[:3, 3] = cam_pos

    return extrinsic


# ============================================================================
# TEST: Verify fixes work
# ============================================================================

def test_fixes():
    """Test the proposed fixes."""
    import sys
    sys.path.append("/home/sandro/thesis/code/depthsplat")
    sys.path.append("/home/sandro/thesis/code/depthsplat/inference_gradio")

    from runner import DepthSplatRunner
    from PIL import Image
    import os

    output_dir = "/home/sandro/thesis/code/depthsplat/inference_gradio/debugging/imgs/fix_test"
    os.makedirs(output_dir, exist_ok=True)

    checkpoint = "/home/sandro/thesis/code/depthsplat/outputs/objaverse_white_small_gauss/checkpoints/epoch_0-step_100000.ckpt"
    config = "objaverse_white_small_gauss"
    render_dir = "/home/sandro/thesis/renders/5cams_10-01-26_bird_10m"

    print("Initializing Runner...")
    runner = DepthSplatRunner(checkpoint, config)

    # Load wild frame
    image_paths, _ = runner.load_wild_frame(60, render_dir)
    images = [np.array(Image.open(p).convert("RGB")) for p in image_paths]
    ctx = runner.current_example

    print("\n" + "=" * 60)
    print("TEST: Training-Matched Intrinsics + Elevation-Safe Camera")
    print("=" * 60)

    # Get safe camera based on input elevations
    min_el, max_el = get_input_camera_elevation_range(ctx['extrinsics'])
    print(f"Input elevation range: [{min_el:.1f}°, {max_el:.1f}°]")

    target_ext = generate_safe_target_camera(
        ctx['extrinsics'],
        azimuth=45,  # Rotate a bit from input cameras
        elevation_override=None,  # Auto-compute from input
        distance=1.0,
    )

    # Use training-matched intrinsics
    fixed_intrinsics = create_training_matched_intrinsics(len(images))

    print(f"Using fx_norm={TRAINING_FX_NORM} (50° FOV) instead of {ctx['intrinsics'][0,0,0]:.2f} (7.5° FOV)")
    print(f"Target camera elevation: {(min_el + max_el)/2:.1f}° (mean of input range)")

    result = runner.run_inference(
        images=images,
        extrinsics=ctx['extrinsics'],
        intrinsics=fixed_intrinsics,  # Use fixed intrinsics
        target_extrinsics=target_ext[np.newaxis],
        output_dir=output_dir,
        num_video_frames=0,
    )

    rendered = result['rendered_images'][0]
    non_white = np.sum(np.any(rendered < 250, axis=-1))
    total = rendered.shape[0] * rendered.shape[1]

    print(f"\nResult: Mean={rendered.mean():.1f}, Non-white={100*non_white/total:.1f}%")
    Image.fromarray(rendered).save(f"{output_dir}/render_with_fixes.png")
    print(f"Saved to: {output_dir}/render_with_fixes.png")

    # Compare with original
    print("\n" + "=" * 60)
    print("COMPARISON: Original vs Fixed")
    print("=" * 60)

    result_orig = runner.run_inference(
        images=images,
        extrinsics=ctx['extrinsics'],
        intrinsics=ctx['intrinsics'],  # Original intrinsics
        target_extrinsics=target_ext[np.newaxis],
        output_dir=f"{output_dir}/original",
        num_video_frames=0,
    )

    rendered_orig = result_orig['rendered_images'][0]
    non_white_orig = np.sum(np.any(rendered_orig < 250, axis=-1))

    print(f"Original: Mean={rendered_orig.mean():.1f}, Non-white={100*non_white_orig/total:.1f}%")
    print(f"Fixed:    Mean={rendered.mean():.1f}, Non-white={100*non_white/total:.1f}%")

    # Side by side
    comparison = np.hstack([rendered_orig, rendered])
    Image.fromarray(comparison).save(f"{output_dir}/comparison_orig_vs_fixed.png")
    print(f"\nComparison saved to: {output_dir}/comparison_orig_vs_fixed.png")


if __name__ == "__main__":
    test_fixes()
