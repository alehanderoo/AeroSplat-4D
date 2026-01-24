#!/usr/bin/env python3
"""
crop_depth.py - Crop depth maps to flying object bounding box using masks.

This script post-processes Isaac Sim renders to extract object-only depth maps,
significantly reducing storage requirements.

Storage comparison (per frame @ 2560x1440):
- Full scene depth (float32 .npy): ~14.7 MB
- Cropped object depth (compressed .npz): ~10-100 KB (depending on object size)
- Storage reduction: 99%+

Usage:
    # Process a single render directory
    python crop_depth.py --render-dir /path/to/renders/5cams_drone_10m

    # Process all renders in base directory
    python crop_depth.py --base-dir /path/to/renders --all

    # Keep original depth files (default: delete after cropping)
    python crop_depth.py --render-dir /path/to/renders/5cams_drone_10m --keep-originals

    # Adjust padding around object (default: 10 pixels)
    python crop_depth.py --render-dir /path/to/renders/5cams_drone_10m --padding 20
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image


def load_mask(mask_path: Path) -> np.ndarray:
    """Load a drone mask image as numpy array."""
    mask = np.array(Image.open(mask_path))
    return mask


def load_depth(depth_path: Path) -> np.ndarray:
    """Load depth data from .npy file."""
    return np.load(depth_path)


def get_mask_bbox(mask: np.ndarray, padding: int = 10) -> Optional[Tuple[int, int, int, int]]:
    """
    Get bounding box of non-zero mask pixels with padding.

    Returns:
        (x_min, y_min, x_max, y_max) or None if mask is empty
    """
    # Find non-zero pixels (mask > 0 includes both body=255 and props=128)
    ys, xs = np.where(mask > 0)

    if len(xs) == 0 or len(ys) == 0:
        return None

    x_min = max(0, int(np.min(xs)) - padding)
    y_min = max(0, int(np.min(ys)) - padding)
    x_max = min(mask.shape[1], int(np.max(xs)) + padding + 1)
    y_max = min(mask.shape[0], int(np.max(ys)) + padding + 1)

    return (x_min, y_min, x_max, y_max)


def crop_depth_to_object(
    depth: np.ndarray,
    mask: np.ndarray,
    bbox: Tuple[int, int, int, int],
    mask_background: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crop depth map to bounding box and optionally mask background.

    Args:
        depth: Full scene depth array (H, W) or (H, W, 1)
        mask: Object mask array (H, W)
        bbox: (x_min, y_min, x_max, y_max)
        mask_background: If True, set non-object pixels to inf

    Returns:
        Tuple of (cropped_depth, cropped_mask)
    """
    x_min, y_min, x_max, y_max = bbox

    # Handle depth shape (H, W) or (H, W, 1)
    if depth.ndim == 3 and depth.shape[2] == 1:
        depth = depth.squeeze(-1)

    # Crop to bounding box
    cropped_depth = depth[y_min:y_max, x_min:x_max].copy()
    cropped_mask = mask[y_min:y_max, x_min:x_max].copy()

    # Mask out background pixels (set to inf or NaN)
    if mask_background:
        background = cropped_mask == 0
        cropped_depth[background] = np.inf

    return cropped_depth, cropped_mask


def save_cropped_depth(
    cropped_depth: np.ndarray,
    cropped_mask: np.ndarray,
    bbox: Tuple[int, int, int, int],
    output_path: Path,
    original_shape: Tuple[int, int]
) -> dict:
    """
    Save cropped depth data in compressed format with metadata.

    The .npz file contains:
        - depth: cropped depth array (float32)
        - mask: cropped mask array (uint8)
        - bbox: bounding box [x_min, y_min, x_max, y_max]
        - original_shape: [height, width] of full image

    Returns:
        Metadata dict for JSON export
    """
    np.savez_compressed(
        output_path,
        depth=cropped_depth.astype(np.float32),
        mask=cropped_mask.astype(np.uint8),
        bbox=np.array(bbox, dtype=np.int32),
        original_shape=np.array(original_shape, dtype=np.int32)
    )

    return {
        "bbox": list(bbox),
        "original_shape": list(original_shape),
        "cropped_shape": list(cropped_depth.shape),
        "output_file": output_path.name
    }


def reconstruct_full_depth(npz_path: Path, fill_value: float = np.inf) -> np.ndarray:
    """
    Reconstruct full-resolution depth from cropped .npz file.

    Useful for visualization or if you need the full image back.
    """
    data = np.load(npz_path)
    depth = data['depth']
    bbox = data['bbox']
    original_shape = data['original_shape']

    # Create full image filled with fill_value
    full_depth = np.full(tuple(original_shape), fill_value, dtype=np.float32)

    # Place cropped region back
    x_min, y_min, x_max, y_max = bbox
    full_depth[y_min:y_max, x_min:x_max] = depth

    return full_depth


def process_camera(
    camera_dir: Path,
    padding: int = 10,
    keep_originals: bool = False,
    verbose: bool = True
) -> dict:
    """
    Process all frames for a single camera.

    Args:
        camera_dir: Path to camera directory (e.g., renders/5cams_drone_10m/cam_01)
        padding: Pixels of padding around object bbox
        keep_originals: If False, delete original .npy depth files
        verbose: Print progress

    Returns:
        Processing statistics dict
    """
    mask_dir = camera_dir / "mask"
    depth_dir = camera_dir / "depth"
    cropped_dir = camera_dir / "depth_cropped"

    if not mask_dir.exists():
        print(f"  Warning: No mask directory found at {mask_dir}")
        return {"error": "no_mask_dir", "frames_processed": 0}

    if not depth_dir.exists():
        print(f"  Warning: No depth directory found at {depth_dir}")
        return {"error": "no_depth_dir", "frames_processed": 0}

    # Create output directory
    cropped_dir.mkdir(exist_ok=True)

    # Find all mask files
    mask_files = sorted(mask_dir.glob("drone_mask_*.png"))

    if not mask_files:
        print(f"  Warning: No mask files found in {mask_dir}")
        return {"error": "no_masks", "frames_processed": 0}

    stats = {
        "frames_processed": 0,
        "frames_skipped": 0,
        "original_bytes": 0,
        "cropped_bytes": 0,
        "frames": []
    }

    for mask_path in mask_files:
        # Extract frame number from mask filename (drone_mask_0001.png)
        frame_str = mask_path.stem.split("_")[-1]
        frame_num = int(frame_str)

        # Find corresponding depth file
        depth_path = depth_dir / f"distance_to_image_plane_{frame_str}.npy"

        if not depth_path.exists():
            if verbose:
                print(f"    Frame {frame_num}: depth file not found, skipping")
            stats["frames_skipped"] += 1
            continue

        # Load data
        mask = load_mask(mask_path)
        depth = load_depth(depth_path)

        # Get bounding box
        bbox = get_mask_bbox(mask, padding=padding)

        if bbox is None:
            # No object in frame - save minimal placeholder
            if verbose:
                print(f"    Frame {frame_num}: no object visible, saving empty")

            output_path = cropped_dir / f"depth_cropped_{frame_str}.npz"
            np.savez_compressed(
                output_path,
                depth=np.array([], dtype=np.float32),
                mask=np.array([], dtype=np.uint8),
                bbox=np.array([0, 0, 0, 0], dtype=np.int32),
                original_shape=np.array(mask.shape, dtype=np.int32),
                empty=True
            )
            stats["frames_processed"] += 1
            stats["cropped_bytes"] += output_path.stat().st_size
            stats["original_bytes"] += depth_path.stat().st_size

            if not keep_originals:
                depth_path.unlink()
            continue

        # Crop depth to object
        original_shape = mask.shape[:2]
        cropped_depth, cropped_mask = crop_depth_to_object(depth, mask, bbox)

        # Save cropped data
        output_path = cropped_dir / f"depth_cropped_{frame_str}.npz"
        frame_meta = save_cropped_depth(
            cropped_depth, cropped_mask, bbox, output_path, original_shape
        )

        # Track sizes
        original_size = depth_path.stat().st_size
        cropped_size = output_path.stat().st_size
        stats["original_bytes"] += original_size
        stats["cropped_bytes"] += cropped_size
        stats["frames_processed"] += 1
        stats["frames"].append({
            "frame": frame_num,
            **frame_meta,
            "original_size": original_size,
            "cropped_size": cropped_size,
            "compression_ratio": original_size / cropped_size if cropped_size > 0 else 0
        })

        if verbose and frame_num % 30 == 0:
            ratio = original_size / cropped_size if cropped_size > 0 else 0
            print(f"    Frame {frame_num}: {original_size/1024:.1f}KB -> {cropped_size/1024:.1f}KB ({ratio:.0f}x)")

        # Delete original if requested
        if not keep_originals:
            depth_path.unlink()

    return stats


def process_render_directory(
    render_dir: Path,
    padding: int = 10,
    keep_originals: bool = False,
    verbose: bool = True
) -> dict:
    """
    Process all cameras in a render directory.

    Args:
        render_dir: Path to render (e.g., renders/5cams_drone_10m)
        padding: Pixels of padding around object bbox
        keep_originals: If False, delete original .npy depth files
        verbose: Print progress

    Returns:
        Processing statistics dict
    """
    render_dir = Path(render_dir)

    if not render_dir.exists():
        print(f"Error: Render directory not found: {render_dir}")
        return {"error": "not_found"}

    # Find camera directories
    camera_dirs = sorted([d for d in render_dir.iterdir() if d.is_dir() and d.name.startswith("cam_")])

    if not camera_dirs:
        print(f"Error: No camera directories found in {render_dir}")
        return {"error": "no_cameras"}

    print(f"Processing {render_dir.name} ({len(camera_dirs)} cameras)")

    all_stats = {
        "render_dir": str(render_dir),
        "cameras": {},
        "total_original_bytes": 0,
        "total_cropped_bytes": 0,
        "total_frames": 0
    }

    for camera_dir in camera_dirs:
        if verbose:
            print(f"  Camera: {camera_dir.name}")

        cam_stats = process_camera(
            camera_dir,
            padding=padding,
            keep_originals=keep_originals,
            verbose=verbose
        )

        all_stats["cameras"][camera_dir.name] = cam_stats
        all_stats["total_original_bytes"] += cam_stats.get("original_bytes", 0)
        all_stats["total_cropped_bytes"] += cam_stats.get("cropped_bytes", 0)
        all_stats["total_frames"] += cam_stats.get("frames_processed", 0)

    # Calculate overall compression
    if all_stats["total_cropped_bytes"] > 0:
        ratio = all_stats["total_original_bytes"] / all_stats["total_cropped_bytes"]
        saved_mb = (all_stats["total_original_bytes"] - all_stats["total_cropped_bytes"]) / (1024 * 1024)
        print(f"\nSummary for {render_dir.name}:")
        print(f"  Frames processed: {all_stats['total_frames']}")
        print(f"  Original size: {all_stats['total_original_bytes'] / (1024*1024):.1f} MB")
        print(f"  Cropped size: {all_stats['total_cropped_bytes'] / (1024*1024):.1f} MB")
        print(f"  Compression ratio: {ratio:.0f}x")
        print(f"  Storage saved: {saved_mb:.1f} MB")
        all_stats["compression_ratio"] = ratio
        all_stats["storage_saved_mb"] = saved_mb

    # Save metadata
    meta_path = render_dir / "depth_crop_metadata.json"
    with open(meta_path, 'w') as f:
        # Don't save per-frame details to JSON (too verbose)
        summary = {k: v for k, v in all_stats.items() if k != "cameras"}
        summary["cameras"] = {
            name: {k: v for k, v in stats.items() if k != "frames"}
            for name, stats in all_stats["cameras"].items()
        }
        json.dump(summary, f, indent=2)

    return all_stats


def main():
    parser = argparse.ArgumentParser(
        description="Crop depth maps to flying object bounding boxes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--render-dir",
        type=Path,
        help="Path to single render directory (e.g., renders/5cams_drone_10m)"
    )
    group.add_argument(
        "--base-dir",
        type=Path,
        help="Base renders directory (use with --all)"
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all render directories in base-dir"
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=10,
        help="Padding pixels around object bounding box (default: 10)"
    )
    parser.add_argument(
        "--keep-originals",
        action="store_true",
        help="Keep original .npy depth files (default: delete after cropping)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity"
    )

    args = parser.parse_args()
    verbose = not args.quiet

    if args.render_dir:
        process_render_directory(
            args.render_dir,
            padding=args.padding,
            keep_originals=args.keep_originals,
            verbose=verbose
        )
    elif args.base_dir and args.all:
        # Process all render directories
        base_dir = Path(args.base_dir)
        render_dirs = sorted([
            d for d in base_dir.iterdir()
            if d.is_dir() and d.name.startswith(("5cams_", "3cams_", "cam_"))
        ])

        if not render_dirs:
            print(f"No render directories found in {base_dir}")
            return

        print(f"Found {len(render_dirs)} render directories to process\n")

        total_saved = 0
        for render_dir in render_dirs:
            stats = process_render_directory(
                render_dir,
                padding=args.padding,
                keep_originals=args.keep_originals,
                verbose=verbose
            )
            total_saved += stats.get("storage_saved_mb", 0)
            print()

        print(f"\n{'='*50}")
        print(f"Total storage saved across all renders: {total_saved:.1f} MB")
    else:
        parser.error("--all requires --base-dir")


if __name__ == "__main__":
    main()
