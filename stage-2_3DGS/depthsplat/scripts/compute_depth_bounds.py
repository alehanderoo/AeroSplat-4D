#!/usr/bin/env python3
"""
Compute optimal near/far depth bounds from Objaverse dataset.

This script analyzes all depth values in the dataset to recommend
appropriate near/far bounds for training. Tighter bounds improve
depth resolution in the cost volume.

Usage:
    python scripts/compute_depth_bounds.py --data-dir datasets/objaverse
    python scripts/compute_depth_bounds.py --data-dir datasets/objaverse --output-yaml
"""

import argparse
import glob
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm


def analyze_chunk(chunk_path: Path) -> dict:
    """Analyze depth statistics for a single chunk file."""
    chunk = torch.load(chunk_path)

    stats = {
        'depth_mins': [],
        'depth_maxs': [],
        'depth_means': [],
        'depth_p5': [],
        'depth_p95': [],
        'fg_ratios': [],
        'num_scenes': len(chunk),
    }

    for scene in chunk:
        if 'depths' not in scene:
            continue

        depths = scene['depths']
        valid_depths = depths[depths > 0].float()

        if len(valid_depths) == 0:
            continue

        stats['depth_mins'].append(valid_depths.min().item())
        stats['depth_maxs'].append(valid_depths.max().item())
        stats['depth_means'].append(valid_depths.mean().item())
        stats['depth_p5'].append(torch.quantile(valid_depths, 0.05).item())
        stats['depth_p95'].append(torch.quantile(valid_depths, 0.95).item())

        if 'masks' in scene:
            masks = scene['masks']
            fg_ratio = (masks > 0.5).float().mean().item()
            stats['fg_ratios'].append(fg_ratio)

    return stats


def compute_bounds(
    data_dir: Path,
    percentile_near: float = 5.0,
    percentile_far: float = 95.0,
    margin: float = 0.1,
) -> dict:
    """
    Compute recommended near/far bounds from dataset.

    Args:
        data_dir: Path to dataset directory containing train/test subdirs
        percentile_near: Percentile for near bound (lower = more conservative)
        percentile_far: Percentile for far bound (higher = more conservative)
        margin: Safety margin to add (fraction of range)

    Returns:
        Dictionary with computed statistics and recommendations
    """
    # Find all chunk files
    chunk_paths = list(Path(data_dir).glob('**/*.torch'))

    if not chunk_paths:
        raise ValueError(f"No .torch files found in {data_dir}")

    print(f"Found {len(chunk_paths)} chunk files")

    # Aggregate statistics
    all_stats = defaultdict(list)
    total_scenes = 0

    for chunk_path in tqdm(chunk_paths, desc="Analyzing chunks"):
        stats = analyze_chunk(chunk_path)
        total_scenes += stats['num_scenes']

        for key in ['depth_mins', 'depth_maxs', 'depth_means',
                    'depth_p5', 'depth_p95', 'fg_ratios']:
            all_stats[key].extend(stats[key])

    # Convert to numpy for percentile calculations
    results = {}

    for key, values in all_stats.items():
        if not values:
            continue
        arr = np.array(values)
        results[key] = {
            'min': float(arr.min()),
            'max': float(arr.max()),
            'mean': float(arr.mean()),
            'std': float(arr.std()),
            'p5': float(np.percentile(arr, 5)),
            'p25': float(np.percentile(arr, 25)),
            'p50': float(np.percentile(arr, 50)),
            'p75': float(np.percentile(arr, 75)),
            'p95': float(np.percentile(arr, 95)),
        }

    # Compute recommended bounds
    # Near: use percentile of minimum depths (conservative = lower percentile)
    near_candidates = np.array(all_stats['depth_mins'])
    suggested_near = np.percentile(near_candidates, percentile_near)

    # Far: use percentile of maximum depths (conservative = higher percentile)
    far_candidates = np.array(all_stats['depth_maxs'])
    suggested_far = np.percentile(far_candidates, percentile_far)

    # Add safety margin
    depth_range = suggested_far - suggested_near
    suggested_near = max(0.1, suggested_near - margin * depth_range)
    suggested_far = suggested_far + margin * depth_range

    # Round to nice values
    suggested_near = round(suggested_near, 2)
    suggested_far = round(suggested_far, 2)

    results['recommendations'] = {
        'near': suggested_near,
        'far': suggested_far,
        'percentile_near': percentile_near,
        'percentile_far': percentile_far,
        'margin': margin,
        'depth_range': round(suggested_far - suggested_near, 2),
    }

    results['summary'] = {
        'total_chunks': len(chunk_paths),
        'total_scenes': total_scenes,
        'scenes_with_depth': len(all_stats['depth_mins']),
    }

    return results


def print_report(results: dict):
    """Print human-readable analysis report."""
    print("\n" + "=" * 60)
    print("DEPTH BOUNDS ANALYSIS REPORT")
    print("=" * 60)

    summary = results['summary']
    print(f"\nDataset: {summary['total_chunks']} chunks, "
          f"{summary['total_scenes']} total scenes, "
          f"{summary['scenes_with_depth']} with valid depth")

    print("\n--- Depth Statistics (per-scene) ---")

    if 'depth_mins' in results:
        stats = results['depth_mins']
        print(f"Min depths:  {stats['min']:.3f}m - {stats['max']:.3f}m "
              f"(mean: {stats['mean']:.3f}m)")

    if 'depth_maxs' in results:
        stats = results['depth_maxs']
        print(f"Max depths:  {stats['min']:.3f}m - {stats['max']:.3f}m "
              f"(mean: {stats['mean']:.3f}m)")

    if 'depth_means' in results:
        stats = results['depth_means']
        print(f"Mean depths: {stats['min']:.3f}m - {stats['max']:.3f}m "
              f"(mean: {stats['mean']:.3f}m)")

    if 'fg_ratios' in results:
        stats = results['fg_ratios']
        print(f"\nForeground ratio: {100*stats['min']:.1f}% - {100*stats['max']:.1f}% "
              f"(mean: {100*stats['mean']:.1f}%)")

    rec = results['recommendations']
    print("\n--- Recommended Bounds ---")
    print(f"Near: {rec['near']}m")
    print(f"Far:  {rec['far']}m")
    print(f"Range: {rec['depth_range']}m")

    # Calculate depth resolution improvement
    current_near, current_far = 0.5, 5.0
    num_candidates = 128

    current_resolution = (current_far - current_near) / num_candidates
    new_resolution = rec['depth_range'] / num_candidates
    improvement = (current_resolution - new_resolution) / current_resolution * 100

    print(f"\nWith {num_candidates} depth candidates:")
    print(f"  Current (0.5-5.0m): {current_resolution*100:.2f}cm per candidate")
    print(f"  New ({rec['near']}-{rec['far']}m): {new_resolution*100:.2f}cm per candidate")
    print(f"  Resolution improvement: {improvement:.1f}%")

    print("\n--- Config Update ---")
    print("Add to config/experiment/objaverse.yaml:")
    print(f"""
dataset:
  near: {rec['near']}
  far: {rec['far']}

loss:
  depth:
    min_depth: {rec['near']}
    max_depth: {rec['far']}
""")


def output_yaml(results: dict, output_path: Path):
    """Output YAML fragment for config."""
    rec = results['recommendations']

    yaml_content = f"""# Auto-generated depth bounds from dataset analysis
# Generated by: python scripts/compute_depth_bounds.py

# Dataset statistics:
#   Scenes analyzed: {results['summary']['scenes_with_depth']}
#   Near percentile: {rec['percentile_near']}%
#   Far percentile: {rec['percentile_far']}%
#   Safety margin: {rec['margin']*100}%

dataset:
  near: {rec['near']}
  far: {rec['far']}

loss:
  depth:
    min_depth: {rec['near']}
    max_depth: {rec['far']}
"""

    output_path.write_text(yaml_content)
    print(f"\nYAML config written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute optimal depth bounds from Objaverse dataset"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("datasets/objaverse"),
        help="Path to dataset directory"
    )
    parser.add_argument(
        "--percentile-near",
        type=float,
        default=5.0,
        help="Percentile for near bound (default: 5)"
    )
    parser.add_argument(
        "--percentile-far",
        type=float,
        default=95.0,
        help="Percentile for far bound (default: 95)"
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.1,
        help="Safety margin as fraction of range (default: 0.1)"
    )
    parser.add_argument(
        "--output-yaml",
        action="store_true",
        help="Output YAML config fragment"
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("config/depth_bounds.yaml"),
        help="Path for YAML output"
    )

    args = parser.parse_args()

    results = compute_bounds(
        args.data_dir,
        percentile_near=args.percentile_near,
        percentile_far=args.percentile_far,
        margin=args.margin,
    )

    print_report(results)

    if args.output_yaml:
        output_yaml(results, args.output_path)


if __name__ == "__main__":
    main()
