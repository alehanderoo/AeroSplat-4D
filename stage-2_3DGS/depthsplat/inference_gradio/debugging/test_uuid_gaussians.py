#!/usr/bin/env python3
"""
Test UUID render to compare Gaussian positions with wild frames.
"""

import os
import sys
import numpy as np
from PIL import Image

sys.path.append("/home/sandro/thesis/code/depthsplat")
sys.path.append("/home/sandro/thesis/code/depthsplat/inference_gradio")

from runner import DepthSplatRunner
from data_loader import get_example_by_uuid, HARDCODED_UUIDS


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
    output_dir = "/home/sandro/thesis/code/depthsplat/inference_gradio/debugging/imgs/uuid_test"
    os.makedirs(output_dir, exist_ok=True)

    print("Initializing Runner...")
    runner = DepthSplatRunner(checkpoint, config)

    # Load UUID example
    uuid = HARDCODED_UUIDS[0]
    data_dir = "/mnt/raid0/objaverse/test"
    example = get_example_by_uuid(data_dir, uuid, num_context_views=5)

    if not example:
        print(f"Failed to load UUID {uuid}")
        return

    print(f"\nLoaded UUID: {uuid}")
    print(f"Intrinsics fx: {example['intrinsics'][0, 0, 0]:.4f}")
    print(f"FOV: {2 * np.rad2deg(np.arctan(0.5 / example['intrinsics'][0, 0, 0])):.1f}°")

    # Generate same target camera as wild frame test
    target_ext = generate_target_camera(0, 30, 1.0, base_radius=1.4)  # Use radius matching UUID distance
    target_exts = target_ext[np.newaxis, ...]

    print(f"\nTarget camera position: {target_ext[:3, 3]}")

    # Run inference
    res = runner.run_inference(
        images=example['images'],
        extrinsics=example['extrinsics'],
        intrinsics=example['intrinsics'],
        target_extrinsics=target_exts,
        output_dir=output_dir,
        num_video_frames=0
    )

    rendered = res['rendered_images'][0]
    Image.fromarray(rendered).save(os.path.join(output_dir, "render_uuid.png"))
    print(f"\nUUID render mean brightness: {rendered.mean():.1f}")

    # Analyze Gaussian positions
    try:
        from plyfile import PlyData
        ply_path = os.path.join(output_dir, "gaussians.ply")
        if os.path.exists(ply_path):
            ply = PlyData.read(ply_path)
            vertex = ply['vertex']
            pos = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=1)

            print(f"\nUUID Gaussians:")
            print(f"  Num Gaussians: {len(pos)}")
            print(f"  X range: [{pos[:,0].min():.3f}, {pos[:,0].max():.3f}]")
            print(f"  Y range: [{pos[:,1].min():.3f}, {pos[:,1].max():.3f}]")
            print(f"  Z range: [{pos[:,2].min():.3f}, {pos[:,2].max():.3f}]")
            print(f"  Centroid: ({pos[:,0].mean():.3f}, {pos[:,1].mean():.3f}, {pos[:,2].mean():.3f})")
    except ImportError:
        print("\nplyfile not installed, skipping Gaussian analysis")


if __name__ == "__main__":
    main()
