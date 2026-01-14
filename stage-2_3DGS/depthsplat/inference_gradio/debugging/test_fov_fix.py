#!/usr/bin/env python3
"""
Test fix for FOV mismatch between wild frames and training data.

The issue: Wild frames have 7.5° FOV vs training data with 50° FOV.

This script tests using synthetic intrinsics that match the training distribution.
"""

import os
import sys
import numpy as np
from PIL import Image
from pathlib import Path

sys.path.append("/home/sandro/thesis/code/depthsplat")
sys.path.append("/home/sandro/thesis/code/depthsplat/inference_gradio")

from runner import DepthSplatRunner


def generate_target_camera(azimuth, elevation, distance, base_radius=2.0):
    """Generate a target camera looking at origin."""
    radius = base_radius * distance

    azimuth_rad = np.deg2rad(azimuth)
    elevation_rad = np.deg2rad(elevation)

    # Spherical coordinates
    x = radius * np.cos(elevation_rad) * np.cos(azimuth_rad)
    y = radius * np.cos(elevation_rad) * np.sin(azimuth_rad)
    z = radius * np.sin(elevation_rad)

    cam_pos = np.array([x, y, z], dtype=np.float32)

    target = np.array([0, 0, 0], dtype=np.float32)
    forward = target - cam_pos
    forward = forward / np.linalg.norm(forward)

    world_up = np.array([0, 0, 1], dtype=np.float32)

    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-6:
        world_up = np.array([0, 1, 0], dtype=np.float32)
        right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)

    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)

    down = -up
    R = np.stack([right, down, forward], axis=1)

    extrinsic = np.eye(4, dtype=np.float32)
    extrinsic[:3, :3] = R
    extrinsic[:3, 3] = cam_pos

    return extrinsic


def create_training_matched_intrinsics(num_views: int) -> np.ndarray:
    """
    Create intrinsics that match the training data distribution.

    Training data uses fx=1.0723 with 50° FOV and centered principal point.
    """
    fx = 1.0723  # From training data
    fy = 1.0723
    cx = 0.5
    cy = 0.5

    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1],
    ], dtype=np.float32)

    return np.stack([K] * num_views)


def main():
    checkpoint = "/home/sandro/thesis/code/depthsplat/outputs/objaverse_white_small_gauss/checkpoints/epoch_0-step_100000.ckpt"
    config = "objaverse_white_small_gauss"
    render_dir = "/home/sandro/thesis/renders/5cams_10-01-26_bird_10m"
    frame_id = 60
    output_dir = "/home/sandro/thesis/code/depthsplat/inference_gradio/debugging/imgs/fov_fix_test"
    os.makedirs(output_dir, exist_ok=True)

    print("Initializing Runner...")
    runner = DepthSplatRunner(checkpoint, config)

    print("\n" + "=" * 60)
    print("TEST 1: ORIGINAL INTRINSICS (7.5° FOV)")
    print("=" * 60)

    # Load wild frame with original intrinsics
    image_paths, _ = runner.load_wild_frame(frame_id, render_dir)
    images = [np.array(Image.open(p).convert("RGB")) for p in image_paths]

    ctx = runner.current_example
    extrinsics = ctx['extrinsics']
    intrinsics_original = ctx['intrinsics']

    print(f"Original intrinsics fx: {intrinsics_original[0, 0, 0]:.4f}")
    print(f"Original FOV: {2 * np.rad2deg(np.arctan(0.5 / intrinsics_original[0, 0, 0])):.1f}°")

    # Generate target camera
    target_ext = generate_target_camera(0, 30, 1.0)
    target_exts = target_ext[np.newaxis, ...]

    # Run inference with original intrinsics
    res_original = runner.run_inference(
        images=images,
        extrinsics=extrinsics,
        intrinsics=intrinsics_original,
        target_extrinsics=target_exts,
        output_dir=os.path.join(output_dir, "original"),
        num_video_frames=0
    )

    rendered_original = res_original['rendered_images'][0]
    Image.fromarray(rendered_original).save(os.path.join(output_dir, "render_original_fov.png"))
    print(f"Original render mean: {rendered_original.mean():.1f}")

    print("\n" + "=" * 60)
    print("TEST 2: TRAINING-MATCHED INTRINSICS (50° FOV)")
    print("=" * 60)

    # Create training-matched intrinsics
    intrinsics_fixed = create_training_matched_intrinsics(len(images))

    print(f"Fixed intrinsics fx: {intrinsics_fixed[0, 0, 0]:.4f}")
    print(f"Fixed FOV: {2 * np.rad2deg(np.arctan(0.5 / intrinsics_fixed[0, 0, 0])):.1f}°")

    # Run inference with fixed intrinsics
    res_fixed = runner.run_inference(
        images=images,
        extrinsics=extrinsics,
        intrinsics=intrinsics_fixed,
        target_extrinsics=target_exts,
        output_dir=os.path.join(output_dir, "fixed"),
        num_video_frames=0
    )

    rendered_fixed = res_fixed['rendered_images'][0]
    Image.fromarray(rendered_fixed).save(os.path.join(output_dir, "render_fixed_fov.png"))
    print(f"Fixed render mean: {rendered_fixed.mean():.1f}")

    print("\n" + "=" * 60)
    print("TEST 3: INTERMEDIATE FOV")
    print("=" * 60)

    # Try an intermediate value
    fx_intermediate = 2.5  # ~23° FOV - between 7.5° and 50°
    intrinsics_intermediate = np.array([[[fx_intermediate, 0, 0.5],
                                          [0, fx_intermediate, 0.5],
                                          [0, 0, 1]]], dtype=np.float32)
    intrinsics_intermediate = np.tile(intrinsics_intermediate, (len(images), 1, 1))

    print(f"Intermediate intrinsics fx: {fx_intermediate:.4f}")
    print(f"Intermediate FOV: {2 * np.rad2deg(np.arctan(0.5 / fx_intermediate)):.1f}°")

    res_intermediate = runner.run_inference(
        images=images,
        extrinsics=extrinsics,
        intrinsics=intrinsics_intermediate,
        target_extrinsics=target_exts,
        output_dir=os.path.join(output_dir, "intermediate"),
        num_video_frames=0
    )

    rendered_intermediate = res_intermediate['rendered_images'][0]
    Image.fromarray(rendered_intermediate).save(os.path.join(output_dir, "render_intermediate_fov.png"))
    print(f"Intermediate render mean: {rendered_intermediate.mean():.1f}")

    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"Original (7.5° FOV) mean brightness: {rendered_original.mean():.1f}")
    print(f"Fixed (50° FOV) mean brightness: {rendered_fixed.mean():.1f}")
    print(f"Intermediate (23° FOV) mean brightness: {rendered_intermediate.mean():.1f}")
    print(f"\nImages saved to: {output_dir}")

    # Create side-by-side comparison
    combined = np.hstack([rendered_original, rendered_fixed, rendered_intermediate])
    Image.fromarray(combined).save(os.path.join(output_dir, "comparison.png"))
    print(f"Comparison image: {os.path.join(output_dir, 'comparison.png')}")


if __name__ == "__main__":
    main()
