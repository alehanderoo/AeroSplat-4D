#!/usr/bin/env python3
"""
Diagnostic script to verify Objaverse data conversion.

Checks:
1. Camera pose validity (position, orientation)
2. Depth value ranges
3. Camera-depth consistency (do cameras see the object?)
"""

import argparse
import json
import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def load_chunk(chunk_path: Path):
    """Load a single chunk file."""
    return torch.load(chunk_path)


def analyze_cameras(cameras: torch.Tensor):
    """Analyze camera poses from the 18-dim tensor."""
    b = cameras.shape[0]

    # Extract intrinsics
    fx, fy, cx, cy = cameras[:, :4].T

    # Extract w2c and convert to c2w
    w2c = torch.eye(4).unsqueeze(0).repeat(b, 1, 1)
    w2c[:, :3] = cameras[:, 6:].reshape(b, 3, 4)
    c2w = torch.linalg.inv(w2c)

    # Camera positions (in world coordinates)
    positions = c2w[:, :3, 3].numpy()

    # Camera forward directions (third column of rotation, negated for OpenCV convention)
    # In OpenCV, camera looks along +Z
    forward = c2w[:, :3, 2].numpy()

    return {
        'positions': positions,
        'forward': forward,
        'fx': fx.numpy(),
        'fy': fy.numpy(),
        'cx': cx.numpy(),
        'cy': cy.numpy(),
        'c2w': c2w.numpy(),
        'w2c': w2c.numpy(),
    }


def analyze_depths(depths: torch.Tensor):
    """Analyze depth maps."""
    depths_np = depths.numpy()
    valid = depths_np[depths_np > 0]

    if len(valid) == 0:
        return {'min': float('nan'), 'max': float('nan'), 'mean': float('nan')}

    return {
        'min': float(valid.min()),
        'max': float(valid.max()),
        'mean': float(valid.mean()),
        'median': float(np.median(valid)),
        'valid_ratio': len(valid) / depths_np.size,
    }


def check_camera_depth_consistency(camera_info, depth_stats):
    """Check if camera distances are consistent with depth values."""
    positions = camera_info['positions']

    # Distance from cameras to origin (assuming object is at origin)
    camera_distances = np.linalg.norm(positions, axis=1)

    return {
        'camera_dist_min': float(camera_distances.min()),
        'camera_dist_max': float(camera_distances.max()),
        'camera_dist_mean': float(camera_distances.mean()),
        'depth_min': depth_stats.get('min', float('nan')),
        'depth_max': depth_stats.get('max', float('nan')),
    }


def visualize_cameras(camera_info, title="Camera Setup", save_path=None):
    """Visualize camera positions and orientations."""
    fig = plt.figure(figsize=(15, 5))

    positions = camera_info['positions']
    forward = camera_info['forward']

    # 3D view
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c='blue', s=50)

    # Draw camera forward directions
    scale = 0.3
    for i in range(len(positions)):
        ax1.quiver(positions[i, 0], positions[i, 1], positions[i, 2],
                   forward[i, 0], forward[i, 1], forward[i, 2],
                   length=scale, color='red', alpha=0.6)

    # Draw origin (object location)
    ax1.scatter([0], [0], [0], c='green', s=200, marker='*', label='Origin')

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('3D View (Cameras + Forward)')
    ax1.legend()

    # XY projection
    ax2 = fig.add_subplot(132)
    ax2.scatter(positions[:, 0], positions[:, 1], c='blue', s=50)
    ax2.quiver(positions[:, 0], positions[:, 1],
               forward[:, 0], forward[:, 1], scale=5, color='red', alpha=0.6)
    ax2.scatter([0], [0], c='green', s=200, marker='*')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('XY Projection')
    ax2.set_aspect('equal')
    ax2.grid(True)

    # XZ projection
    ax3 = fig.add_subplot(133)
    ax3.scatter(positions[:, 0], positions[:, 2], c='blue', s=50)
    ax3.quiver(positions[:, 0], positions[:, 2],
               forward[:, 0], forward[:, 2], scale=5, color='red', alpha=0.6)
    ax3.scatter([0], [0], c='green', s=200, marker='*')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Z')
    ax3.set_title('XZ Projection')
    ax3.set_aspect('equal')
    ax3.grid(True)

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved camera visualization to {save_path}")
    else:
        plt.show()

    plt.close()


def check_cameras_look_at_origin(camera_info):
    """Check if cameras are looking towards the origin."""
    positions = camera_info['positions']
    forward = camera_info['forward']

    # Direction from camera to origin
    to_origin = -positions / np.linalg.norm(positions, axis=1, keepdims=True)

    # Dot product between forward direction and direction to origin
    # Should be close to 1 if cameras look at origin
    dot_products = np.sum(forward * to_origin, axis=1)

    return {
        'dot_product_mean': float(dot_products.mean()),
        'dot_product_min': float(dot_products.min()),
        'dot_product_max': float(dot_products.max()),
        'cameras_looking_at_origin': bool(dot_products.mean() > 0.8),
    }


def main():
    parser = argparse.ArgumentParser(description="Diagnose Objaverse converted data")
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/objaverse"),
                        help="Path to converted dataset")
    parser.add_argument("--num-scenes", type=int, default=5,
                        help="Number of scenes to analyze")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate visualization plots")
    parser.add_argument("--save-dir", type=Path, default=None,
                        help="Directory to save visualizations")
    args = parser.parse_args()

    train_dir = args.data_dir / "train"
    if not train_dir.exists():
        print(f"Train directory not found: {train_dir}")
        return

    chunks = sorted(train_dir.glob("*.torch"))
    if not chunks:
        print("No chunk files found")
        return

    print(f"Found {len(chunks)} chunk files")
    print("=" * 60)

    # Analyze first chunk
    chunk = load_chunk(chunks[0])
    print(f"\nAnalyzing chunk: {chunks[0].name}")
    print(f"Number of scenes in chunk: {len(chunk)}")

    all_depth_stats = []
    all_camera_stats = []

    for i, scene in enumerate(chunk[:args.num_scenes]):
        print(f"\n--- Scene {i}: {scene['key']} ---")

        # Analyze cameras
        camera_info = analyze_cameras(scene['cameras'])
        print(f"Number of views: {len(camera_info['positions'])}")
        print(f"Camera positions range:")
        print(f"  X: [{camera_info['positions'][:, 0].min():.3f}, {camera_info['positions'][:, 0].max():.3f}]")
        print(f"  Y: [{camera_info['positions'][:, 1].min():.3f}, {camera_info['positions'][:, 1].max():.3f}]")
        print(f"  Z: [{camera_info['positions'][:, 2].min():.3f}, {camera_info['positions'][:, 2].max():.3f}]")

        # Check if cameras look at origin
        look_at_check = check_cameras_look_at_origin(camera_info)
        print(f"Cameras looking at origin: {look_at_check['cameras_looking_at_origin']}")
        print(f"  Forward·ToOrigin mean: {look_at_check['dot_product_mean']:.3f}")

        # Analyze depths
        if 'depths' in scene:
            depth_stats = analyze_depths(scene['depths'])
            print(f"Depth stats:")
            print(f"  Range: [{depth_stats['min']:.3f}, {depth_stats['max']:.3f}]")
            print(f"  Mean: {depth_stats['mean']:.3f}, Median: {depth_stats['median']:.3f}")
            print(f"  Valid ratio: {depth_stats['valid_ratio']:.2%}")

            # Check consistency
            consistency = check_camera_depth_consistency(camera_info, depth_stats)
            print(f"Camera-Depth consistency:")
            print(f"  Camera distance to origin: [{consistency['camera_dist_min']:.3f}, {consistency['camera_dist_max']:.3f}]")
            print(f"  Depth range: [{consistency['depth_min']:.3f}, {consistency['depth_max']:.3f}]")

            # Warning if mismatch
            if depth_stats['max'] > 0:
                if depth_stats['min'] < 1.0 or depth_stats['max'] > 3.5:
                    print(f"  ⚠️  WARNING: Depths outside config range [1.0, 3.5]!")

            all_depth_stats.append(depth_stats)
        else:
            print("  No depth data in scene")

        all_camera_stats.append({
            'camera_info': camera_info,
            'look_at': look_at_check,
        })

        # Visualize if requested
        if args.visualize:
            save_path = None
            if args.save_dir:
                args.save_dir.mkdir(parents=True, exist_ok=True)
                save_path = args.save_dir / f"cameras_{scene['key']}.png"
            visualize_cameras(camera_info, title=f"Scene: {scene['key']}", save_path=save_path)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if all_depth_stats:
        all_mins = [s['min'] for s in all_depth_stats if not np.isnan(s['min'])]
        all_maxs = [s['max'] for s in all_depth_stats if not np.isnan(s['max'])]
        if all_mins and all_maxs:
            print(f"Overall depth range: [{min(all_mins):.3f}, {max(all_maxs):.3f}]")
            print(f"Config expected range: [1.0, 3.5]")

            if min(all_mins) < 1.0:
                print("⚠️  Some depths are SMALLER than config 'near' (1.0)")
            if max(all_maxs) > 3.5:
                print("⚠️  Some depths are LARGER than config 'far' (3.5)")

    look_at_results = [s['look_at']['cameras_looking_at_origin'] for s in all_camera_stats]
    if all(look_at_results):
        print("✓ All cameras appear to look at origin")
    else:
        print("⚠️  Some cameras may not be looking at origin!")

    # Check intrinsics
    sample_intrinsics = all_camera_stats[0]['camera_info']
    print(f"\nSample intrinsics (normalized):")
    print(f"  fx: {sample_intrinsics['fx'][0]:.4f}")
    print(f"  fy: {sample_intrinsics['fy'][0]:.4f}")
    print(f"  cx: {sample_intrinsics['cx'][0]:.4f}")
    print(f"  cy: {sample_intrinsics['cy'][0]:.4f}")

    # Print sample w2c for verification
    print(f"\nSample w2c matrix (view 0):")
    print(all_camera_stats[0]['camera_info']['w2c'][0])

    print(f"\nSample c2w matrix (view 0):")
    print(all_camera_stats[0]['camera_info']['c2w'][0])


if __name__ == "__main__":
    main()
