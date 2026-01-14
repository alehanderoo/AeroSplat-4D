#!/usr/bin/env python3
"""
Test rendering from elevation angles similar to the input cameras.

The wild frame cameras are all at elevation ~-63° (viewing from below).
This tests if rendering from similar angles produces better results.
"""

import os
import sys
import numpy as np
from PIL import Image

sys.path.append("/home/sandro/thesis/code/depthsplat")
sys.path.append("/home/sandro/thesis/code/depthsplat/inference_gradio")

from runner import DepthSplatRunner


def generate_target_camera(azimuth, elevation, distance, base_radius=2.0):
    """Generate a target camera looking at origin."""
    radius = base_radius * distance
    azimuth_rad = np.deg2rad(azimuth)
    elevation_rad = np.deg2rad(elevation)

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


def main():
    checkpoint = "/home/sandro/thesis/code/depthsplat/outputs/objaverse_white_small_gauss/checkpoints/epoch_0-step_100000.ckpt"
    config = "objaverse_white_small_gauss"
    render_dir = "/home/sandro/thesis/renders/5cams_10-01-26_bird_10m"
    output_dir = "/home/sandro/thesis/code/depthsplat/inference_gradio/debugging/imgs/elevation_test"
    os.makedirs(output_dir, exist_ok=True)

    print("Initializing Runner...")
    runner = DepthSplatRunner(checkpoint, config)

    # Load wild frame
    image_paths, _ = runner.load_wild_frame(60, render_dir)
    images = [np.array(Image.open(p).convert("RGB")) for p in image_paths]
    ctx = runner.current_example

    print("\nInput camera elevations (all ~-63°):")
    for i, ext in enumerate(ctx['extrinsics']):
        pos = ext[:3, 3]
        dist = np.linalg.norm(pos)
        elev = np.rad2deg(np.arcsin(pos[2] / dist))
        print(f"  View {i}: pos={pos}, elev={elev:.1f}°")

    # Test different rendering elevations
    test_elevations = [30, 0, -30, -60, -63]  # -63 matches input cameras

    renders = []
    for elev in test_elevations:
        print(f"\nRendering from elevation {elev}°...")
        target_ext = generate_target_camera(45, elev, 1.0)

        result = runner.run_inference(
            images=images,
            extrinsics=ctx['extrinsics'],
            intrinsics=ctx['intrinsics'],
            target_extrinsics=target_ext[np.newaxis],
            output_dir=f"{output_dir}/elev_{elev}",
            num_video_frames=0
        )

        rendered = result['rendered_images'][0]
        renders.append(rendered)

        # Calculate metrics
        non_white = np.sum(np.any(rendered < 250, axis=-1))
        total = rendered.shape[0] * rendered.shape[1]
        print(f"  Mean: {rendered.mean():.1f}, Non-white: {100*non_white/total:.1f}%")

        Image.fromarray(rendered).save(f"{output_dir}/render_elev_{elev}.png")

    # Create comparison strip
    combined = np.hstack(renders)
    Image.fromarray(combined).save(f"{output_dir}/elevation_comparison.png")
    print(f"\nSaved comparison to: {output_dir}/elevation_comparison.png")

    print("\n" + "=" * 60)
    print("ANALYSIS:")
    print("=" * 60)
    print("If rendering from -60° to -63° produces better results than +30°,")
    print("it confirms that the model can only reconstruct views similar to input.")
    print("This is expected for sparse-view reconstruction.")


if __name__ == "__main__":
    main()
