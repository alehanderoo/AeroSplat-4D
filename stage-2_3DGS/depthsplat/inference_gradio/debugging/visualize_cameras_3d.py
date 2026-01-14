#!/usr/bin/env python3
"""
Visualize camera positions and orientations in 3D.
Creates a matplotlib 3D plot showing:
- Camera positions
- Camera forward directions
- Object center (origin)
- Optional: Gaussian point cloud from PLY
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

sys.path.append("/home/sandro/thesis/code/depthsplat")
sys.path.append("/home/sandro/thesis/code/depthsplat/inference_gradio")

from data_loader import get_example_by_uuid, HARDCODED_UUIDS
from runner import DepthSplatRunner


def read_ply_positions(ply_path: str) -> np.ndarray:
    """Read Gaussian positions from PLY file."""
    positions = []
    with open(ply_path, 'r') as f:
        # Skip header
        in_header = True
        vertex_count = 0
        for line in f:
            if in_header:
                if line.startswith('element vertex'):
                    vertex_count = int(line.split()[-1])
                if line.strip() == 'end_header':
                    in_header = False
                continue

            # Parse vertex data
            parts = line.strip().split()
            if len(parts) >= 3:
                positions.append([float(parts[0]), float(parts[1]), float(parts[2])])

            if len(positions) >= vertex_count:
                break

    return np.array(positions)


def plot_cameras(ax, extrinsics: np.ndarray, color: str, label: str, arrow_scale: float = 0.3):
    """Plot camera positions and forward directions."""
    positions = extrinsics[:, :3, 3]

    for i, ext in enumerate(extrinsics):
        pos = ext[:3, 3]
        forward = ext[:3, 2]  # Third column = forward in OpenCV

        # Plot camera position
        ax.scatter(*pos, c=color, s=100, marker='o', alpha=0.8)

        # Plot forward direction as arrow
        ax.quiver(pos[0], pos[1], pos[2],
                  forward[0], forward[1], forward[2],
                  length=arrow_scale, color=color, alpha=0.7, arrow_length_ratio=0.3)

        # Label
        ax.text(pos[0], pos[1], pos[2], f'{i}', fontsize=8)

    # For legend
    ax.scatter([], [], c=color, s=100, marker='o', label=f'{label} cameras')


def plot_gaussians(ax, ply_path: str, max_points: int = 5000):
    """Plot Gaussian positions from PLY file."""
    if not Path(ply_path).exists():
        print(f"PLY file not found: {ply_path}")
        return

    positions = read_ply_positions(ply_path)
    print(f"Loaded {len(positions)} Gaussians from {ply_path}")

    # Subsample if too many
    if len(positions) > max_points:
        indices = np.random.choice(len(positions), max_points, replace=False)
        positions = positions[indices]

    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
               c='green', s=1, alpha=0.3, label='Gaussians')

    # Print statistics
    print(f"  Position range X: [{positions[:, 0].min():.2f}, {positions[:, 0].max():.2f}]")
    print(f"  Position range Y: [{positions[:, 1].min():.2f}, {positions[:, 1].max():.2f}]")
    print(f"  Position range Z: [{positions[:, 2].min():.2f}, {positions[:, 2].max():.2f}]")
    print(f"  Centroid: {positions.mean(axis=0)}")


def main():
    fig = plt.figure(figsize=(16, 8))

    # Plot 1: UUID cameras
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title('UUID Example Cameras')

    uuid = HARDCODED_UUIDS[0]
    data_dir = "/mnt/raid0/objaverse/test"
    uuid_example = get_example_by_uuid(data_dir, uuid, num_context_views=5)

    if uuid_example:
        plot_cameras(ax1, uuid_example['extrinsics'], 'blue', 'UUID')

        # Plot origin
        ax1.scatter(0, 0, 0, c='red', s=200, marker='*', label='Origin')

        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.legend()

        # Set equal aspect ratio
        positions = uuid_example['extrinsics'][:, :3, 3]
        max_range = np.abs(positions).max() * 1.2
        ax1.set_xlim(-max_range, max_range)
        ax1.set_ylim(-max_range, max_range)
        ax1.set_zlim(-max_range, max_range)

    # Plot 2: Wild frame cameras
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title('Wild Frame Cameras')

    checkpoint = "/home/sandro/thesis/code/depthsplat/outputs/objaverse_white_small_gauss/checkpoints/epoch_0-step_100000.ckpt"
    config = "objaverse_white_small_gauss"
    render_dir = "/home/sandro/thesis/renders/5cams_10-01-26_bird_10m"
    frame_id = 60

    runner = DepthSplatRunner(checkpoint, config)
    image_paths, status = runner.load_wild_frame(frame_id, render_dir)

    wild_ctx = runner.current_example

    plot_cameras(ax2, wild_ctx['extrinsics'], 'orange', 'Wild')

    # Plot origin
    ax2.scatter(0, 0, 0, c='red', s=200, marker='*', label='Origin (object)')

    # Try to load and plot Gaussians
    ply_path = "/tmp/depthsplat_gradio_wild/gaussians.ply"
    if Path(ply_path).exists():
        plot_gaussians(ax2, ply_path)

    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.legend()

    # Set equal aspect ratio
    positions = wild_ctx['extrinsics'][:, :3, 3]
    max_range = np.abs(positions).max() * 1.2
    ax2.set_xlim(-max_range, max_range)
    ax2.set_ylim(-max_range, max_range)
    ax2.set_zlim(-max_range, max_range)

    plt.tight_layout()
    output_path = "/home/sandro/thesis/code/depthsplat/inference_gradio/debugging/imgs/camera_comparison_3d.png"
    plt.savefig(output_path, dpi=150)
    print(f"\nSaved to: {output_path}")
    plt.close()


if __name__ == "__main__":
    main()
