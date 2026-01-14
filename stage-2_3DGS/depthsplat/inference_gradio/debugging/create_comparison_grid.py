#!/usr/bin/env python3
"""
Create a visual comparison grid between UUID and wild frame renders.
"""

import os
import sys
import numpy as np
from PIL import Image

sys.path.append("/home/sandro/thesis/code/depthsplat")
sys.path.append("/home/sandro/thesis/code/depthsplat/inference_gradio")

from runner import DepthSplatRunner
from data_loader import get_example_by_uuid, HARDCODED_UUIDS


def generate_target_camera(azimuth, elevation, distance, base_radius):
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


def create_comparison():
    output_dir = "/home/sandro/thesis/code/depthsplat/inference_gradio/debugging/imgs"
    checkpoint = "/home/sandro/thesis/code/depthsplat/outputs/objaverse_white_small_gauss/checkpoints/epoch_0-step_100000.ckpt"
    config = "objaverse_white_small_gauss"

    print("Initializing Runner...")
    runner = DepthSplatRunner(checkpoint, config)

    # Load UUID
    uuid = HARDCODED_UUIDS[0]
    uuid_example = get_example_by_uuid("/mnt/raid0/objaverse/test", uuid, num_context_views=5)

    # Load wild frame
    render_dir = "/home/sandro/thesis/renders/5cams_10-01-26_bird_10m"
    wild_paths, _ = runner.load_wild_frame(60, render_dir)
    wild_images = [np.array(Image.open(p).convert("RGB")) for p in wild_paths]
    wild_ctx = runner.current_example

    # Test multiple view angles
    test_angles = [(0, 0), (0, 30), (90, 30), (45, 0), (0, -30)]

    all_uuid_renders = []
    all_wild_renders = []

    for az, el in test_angles:
        print(f"\nTesting Az={az}, El={el}...")

        # Generate target cameras with appropriate radius
        uuid_target = generate_target_camera(az, el, 1.0, base_radius=1.4)
        wild_target = generate_target_camera(az, el, 1.0, base_radius=2.0)

        # UUID render
        uuid_result = runner.run_inference(
            images=uuid_example['images'],
            extrinsics=uuid_example['extrinsics'],
            intrinsics=uuid_example['intrinsics'],
            target_extrinsics=uuid_target[np.newaxis],
            output_dir=f"{output_dir}/temp_uuid",
            num_video_frames=0
        )
        all_uuid_renders.append(uuid_result['rendered_images'][0])

        # Wild render
        wild_result = runner.run_inference(
            images=wild_images,
            extrinsics=wild_ctx['extrinsics'],
            intrinsics=wild_ctx['intrinsics'],
            target_extrinsics=wild_target[np.newaxis],
            output_dir=f"{output_dir}/temp_wild",
            num_video_frames=0
        )
        all_wild_renders.append(wild_result['rendered_images'][0])

    # Create grid
    rows = []

    # Header row: Input images (first 3 from each)
    uuid_inputs = [np.array(Image.fromarray(img).resize((128, 128))) for img in uuid_example['images'][:3]]
    wild_inputs = [np.array(Image.open(p).resize((128, 128))) for p in wild_paths[:3]]

    # Render rows
    h, w = 256, 256
    for i, (az, el) in enumerate(test_angles):
        uuid_render = all_uuid_renders[i]
        wild_render = all_wild_renders[i]

        # Add label
        uuid_labeled = Image.fromarray(uuid_render)
        wild_labeled = Image.fromarray(wild_render)

        rows.append(np.hstack([uuid_render, wild_render]))

    # Combine all rows
    combined = np.vstack(rows)

    # Save
    out_path = f"{output_dir}/uuid_vs_wild_comparison.png"
    Image.fromarray(combined).save(out_path)
    print(f"\nSaved comparison to: {out_path}")

    # Also save side by side of first input images
    uuid_input_row = np.hstack([np.array(Image.fromarray(img).resize((128, 128))) for img in uuid_example['images'][:5]])
    wild_input_row = np.hstack([np.array(Image.open(p).resize((128, 128))) for p in wild_paths[:5]])
    inputs_combined = np.vstack([uuid_input_row, wild_input_row])
    inputs_path = f"{output_dir}/inputs_comparison.png"
    Image.fromarray(inputs_combined).save(inputs_path)
    print(f"Saved inputs to: {inputs_path}")


if __name__ == "__main__":
    create_comparison()
