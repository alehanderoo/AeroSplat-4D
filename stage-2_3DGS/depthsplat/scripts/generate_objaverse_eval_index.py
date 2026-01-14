#!/usr/bin/env python3
"""
Generate evaluation index for Objaverse dataset.

For object-centric evaluation with 32 views arranged in a circle around each object,
this script creates an evaluation index with:
- Evenly spaced context views (e.g., 4 views at 90-degree intervals)
- Target views between context views for novel view synthesis evaluation
"""

import argparse
import json
from pathlib import Path


def generate_objaverse_eval_index(
    index_path: Path,
    output_path: Path,
    num_context_views: int = 4,
    num_target_views: int = 2,
    total_views: int = 32,
):
    """
    Generate evaluation index for Objaverse.

    Args:
        index_path: Path to Objaverse test/index.json
        output_path: Path to write evaluation index
        num_context_views: Number of context views (evenly spaced around object)
        num_target_views: Number of target views to evaluate
        total_views: Total number of views per object (typically 32)
    """
    # Load scene keys from index
    with open(index_path, "r") as f:
        scene_index = json.load(f)

    # Calculate evenly spaced context view indices
    # For 4 context views with 32 total: indices 0, 8, 16, 24
    step = total_views // num_context_views
    context_indices = tuple(i * step for i in range(num_context_views))

    # Select target views - pick views between context views
    # For evaluation, we want views that are challenging (not too close to context)
    # Use views at quarter positions between context views
    target_indices = []
    for i in range(num_context_views):
        # Get midpoint between this context view and the next
        current = context_indices[i]
        next_ctx = context_indices[(i + 1) % num_context_views]
        if next_ctx < current:  # Wrap around
            next_ctx += total_views
        midpoint = (current + next_ctx) // 2
        midpoint = midpoint % total_views
        target_indices.append(midpoint)

    # Take only the requested number of target views
    target_indices = tuple(sorted(target_indices[:num_target_views]))

    print(f"Context views: {context_indices}")
    print(f"Target views: {target_indices}")
    print(f"Total scenes: {len(scene_index)}")

    # Create evaluation index - same views for all scenes
    # (Objaverse renders are standardized with same camera arrangement)
    eval_index = {}
    for scene_key in scene_index.keys():
        eval_index[scene_key] = {
            "context": context_indices,
            "target": target_indices,
        }

    # Write evaluation index
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(eval_index, f)

    print(f"Wrote evaluation index to {output_path}")
    return eval_index


def main():
    parser = argparse.ArgumentParser(
        description="Generate evaluation index for Objaverse dataset"
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/home/sandro/.objaverse/depthsplat/"),
        help="Root directory of Objaverse dataset",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("assets/evaluation_index_objaverse.json"),
        help="Output path for evaluation index",
    )
    parser.add_argument(
        "--num-context-views",
        type=int,
        default=5,
        help="Number of context views (evenly spaced)",
    )
    parser.add_argument(
        "--num-target-views",
        type=int,
        default=2,
        help="Number of target views for evaluation",
    )
    parser.add_argument(
        "--total-views",
        type=int,
        default=32,
        help="Total number of views per object",
    )

    args = parser.parse_args()

    index_path = args.dataset_root / "test" / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Index file not found: {index_path}")

    generate_objaverse_eval_index(
        index_path=index_path,
        output_path=args.output,
        num_context_views=args.num_context_views,
        num_target_views=args.num_target_views,
        total_views=args.total_views,
    )


if __name__ == "__main__":
    main()
