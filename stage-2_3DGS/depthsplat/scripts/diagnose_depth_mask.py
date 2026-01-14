#!/usr/bin/env python3
"""Quick diagnostic of depth vs mask correlation."""

import torch
import numpy as np
from pathlib import Path

chunk_path = Path("datasets/objaverse/train/000000.torch")
chunk = torch.load(chunk_path)

for i, scene in enumerate(chunk[:3]):
    print(f"\n=== Scene {i}: {scene['key']} ===")

    if 'depths' not in scene or 'masks' not in scene:
        print("Missing depth or mask data")
        continue

    depths = scene['depths'].numpy()
    masks = scene['masks'].numpy()

    # For first view
    depth = depths[0]
    mask = masks[0]

    print(f"Depth shape: {depth.shape}, Mask shape: {mask.shape}")
    print(f"Mask: min={mask.min():.3f}, max={mask.max():.3f}, mean={mask.mean():.3f}")

    # Foreground depth (where mask > 0.5)
    fg_mask = mask > 0.5
    fg_depth = depth[fg_mask]

    # Background depth (where mask <= 0.5)
    bg_mask = mask <= 0.5
    bg_depth = depth[bg_mask]

    print(f"\nForeground (mask > 0.5):")
    print(f"  Pixels: {fg_mask.sum()} ({fg_mask.mean()*100:.1f}%)")
    if len(fg_depth) > 0:
        print(f"  Depth: min={fg_depth.min():.3f}, max={fg_depth.max():.3f}, mean={fg_depth.mean():.3f}")

    print(f"\nBackground (mask <= 0.5):")
    print(f"  Pixels: {bg_mask.sum()} ({bg_mask.mean()*100:.1f}%)")
    if len(bg_depth) > 0:
        # Clip to avoid huge numbers
        bg_depth_clipped = np.clip(bg_depth, 0, 200)
        print(f"  Depth: min={bg_depth.min():.3f}, max={bg_depth.max():.3f}, mean={bg_depth_clipped.mean():.3f}")

    # Check if foreground depths are within expected range
    if len(fg_depth) > 0:
        in_range = (fg_depth >= 1.0) & (fg_depth <= 3.5)
        print(f"\nForeground depth in [1.0, 3.5] range: {in_range.mean()*100:.1f}%")

        # What is the actual range needed?
        print(f"Suggested depth range: [{fg_depth.min():.2f}, {fg_depth.max():.2f}]")
