#!/usr/bin/env python3
"""
Compare camera parameters between UUID (training data) and Wild frames.
This helps identify discrepancies in camera conventions and intrinsics.
"""

import sys
import numpy as np
import torch
from pathlib import Path

sys.path.append("/home/sandro/thesis/code/depthsplat")
sys.path.append("/home/sandro/thesis/code/depthsplat/inference_gradio")

from data_loader import get_example_by_uuid, HARDCODED_UUIDS
from runner import DepthSplatRunner


def analyze_intrinsics(intrinsics: np.ndarray, name: str):
    """Analyze intrinsics matrix and compute FOV."""
    print(f"\n=== {name} Intrinsics Analysis ===")

    for i, K in enumerate(intrinsics):
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        # Compute FOV (assuming normalized intrinsics)
        # FOV = 2 * arctan(0.5 / f) for normalized intrinsics where image spans [0,1]
        fov_h = 2 * np.rad2deg(np.arctan(0.5 / fx))
        fov_v = 2 * np.rad2deg(np.arctan(0.5 / fy))

        print(f"  View {i}: fx={fx:.4f}, fy={fy:.4f}, cx={cx:.4f}, cy={cy:.4f}")
        print(f"          FOV: {fov_h:.1f}° x {fov_v:.1f}°")

    # Statistics
    fx_vals = intrinsics[:, 0, 0]
    print(f"\n  fx range: [{fx_vals.min():.4f}, {fx_vals.max():.4f}], mean={fx_vals.mean():.4f}")


def analyze_extrinsics(extrinsics: np.ndarray, name: str):
    """Analyze extrinsics (c2w) matrices."""
    print(f"\n=== {name} Extrinsics Analysis ===")

    positions = extrinsics[:, :3, 3]
    distances = np.linalg.norm(positions, axis=1)

    print(f"  Camera positions (from origin):")
    for i, (pos, dist) in enumerate(zip(positions, distances)):
        forward = extrinsics[i, :3, 2]  # Third column = forward direction in OpenCV
        up = -extrinsics[i, :3, 1]  # -Second column = up direction in OpenCV

        # Check if camera looks at origin
        to_origin = -pos / np.linalg.norm(pos)
        dot_forward = np.dot(forward, to_origin)

        print(f"  View {i}: pos={pos}, dist={dist:.2f}")
        print(f"          forward={forward}")
        print(f"          dot(forward, to_origin)={dot_forward:.4f} {'(LOOKING AT ORIGIN)' if dot_forward > 0.9 else ''}")

    print(f"\n  Distance range: [{distances.min():.2f}, {distances.max():.2f}], mean={distances.mean():.2f}")

    # Check centroid
    centroid = positions.mean(axis=0)
    print(f"  Camera centroid: {centroid}")


def check_rotation_convention(R: np.ndarray, name: str):
    """Check if rotation matrix follows OpenCV convention."""
    # In OpenCV c2w:
    # Column 0 = Right direction in world
    # Column 1 = Down direction in world
    # Column 2 = Forward direction in world

    right = R[:, 0]
    down = R[:, 1]
    forward = R[:, 2]

    # Check orthogonality
    ortho_rf = abs(np.dot(right, forward))
    ortho_rd = abs(np.dot(right, down))
    ortho_df = abs(np.dot(down, forward))

    # Check right-handedness: right x down should equal forward
    cross = np.cross(right, down)
    handedness = np.dot(cross, forward)

    print(f"  {name}:")
    print(f"    Orthogonality: rf={ortho_rf:.6f}, rd={ortho_rd:.6f}, df={ortho_df:.6f}")
    print(f"    Right-handedness: {handedness:.6f} (should be ~1.0)")
    print(f"    det(R) = {np.linalg.det(R):.6f} (should be 1.0)")


def main():
    # Load UUID example
    print("=" * 60)
    print("LOADING UUID EXAMPLE")
    print("=" * 60)

    uuid = HARDCODED_UUIDS[0]
    data_dir = "/mnt/raid0/objaverse/test"
    uuid_example = get_example_by_uuid(data_dir, uuid, num_context_views=5)

    if uuid_example:
        print(f"\nLoaded UUID: {uuid}")
        analyze_intrinsics(uuid_example['intrinsics'], "UUID")
        analyze_extrinsics(uuid_example['extrinsics'], "UUID")

        print("\n  Rotation Convention Check (first view):")
        check_rotation_convention(uuid_example['extrinsics'][0, :3, :3], "UUID View 0")
    else:
        print(f"Failed to load UUID {uuid}")

    # Load Wild frame example
    print("\n" + "=" * 60)
    print("LOADING WILD FRAME EXAMPLE")
    print("=" * 60)

    checkpoint = "/home/sandro/thesis/code/depthsplat/outputs/objaverse_white_small_gauss/checkpoints/epoch_0-step_100000.ckpt"
    config = "objaverse_white_small_gauss"
    render_dir = "/home/sandro/thesis/renders/5cams_10-01-26_bird_10m"
    frame_id = 60

    runner = DepthSplatRunner(checkpoint, config)
    image_paths, status = runner.load_wild_frame(frame_id, render_dir)

    wild_ctx = runner.current_example
    print(f"\n{status}")
    print(f"Scale factor: {wild_ctx.get('scale_factor', 'N/A')}")
    print(f"Center: {wild_ctx.get('center', 'N/A')}")

    analyze_intrinsics(wild_ctx['intrinsics'], "Wild Frame")
    analyze_extrinsics(wild_ctx['extrinsics'], "Wild Frame")

    print("\n  Rotation Convention Check (first view):")
    check_rotation_convention(wild_ctx['extrinsics'][0, :3, :3], "Wild View 0")

    # Compare key statistics
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)

    if uuid_example:
        uuid_fx_mean = uuid_example['intrinsics'][:, 0, 0].mean()
        wild_fx_mean = wild_ctx['intrinsics'][:, 0, 0].mean()

        uuid_dist_mean = np.linalg.norm(uuid_example['extrinsics'][:, :3, 3], axis=1).mean()
        wild_dist_mean = np.linalg.norm(wild_ctx['extrinsics'][:, :3, 3], axis=1).mean()

        print(f"\n  Mean normalized focal length: UUID={uuid_fx_mean:.4f}, Wild={wild_fx_mean:.4f}")
        print(f"  Ratio: {wild_fx_mean/uuid_fx_mean:.2f}x")
        print(f"\n  Mean camera distance: UUID={uuid_dist_mean:.2f}, Wild={wild_dist_mean:.2f}")

        # FOV comparison
        uuid_fov = 2 * np.rad2deg(np.arctan(0.5 / uuid_fx_mean))
        wild_fov = 2 * np.rad2deg(np.arctan(0.5 / wild_fx_mean))
        print(f"\n  Effective FOV: UUID={uuid_fov:.1f}°, Wild={wild_fov:.1f}°")


if __name__ == "__main__":
    main()
