#!/usr/bin/env python3
"""
Convert Objaverse Blender renders to DepthSplat training format.

This script converts the Blender-rendered Objaverse dataset (with RGB, depth, normals, masks)
into the .torch chunk format expected by DepthSplat for training.

Input format (per object directory):
    metadata.json       - Camera poses, intrinsics, file references
    000.png             - RGB image for view 0
    000_depth.exr       - Depth map for view 0
    000_mask.png        - Binary mask for view 0
    000_normal.exr      - Normal map for view 0 (optional)
    ...

Output format:
    train/
        000000.torch    - Chunk file containing multiple scenes
        index.json      - Maps scene_key -> chunk filename
    test/
        ...

Camera Convention Conversion:
    Blender camera local:  +X right, +Y up, -Z forward (looks along -Z axis)
    OpenCV camera local:   +X right, +Y down, +Z forward (looks along +Z axis)
    DepthSplat expects: normalized intrinsics and world-to-camera extrinsics (OpenCV convention)
"""

import argparse
import json
import os
import sys
import tempfile
import zipfile
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from functools import partial

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Optional OpenEXR support
try:
    import OpenEXR
    import Imath
    HAS_OPENEXR = True
except ImportError:
    HAS_OPENEXR = False
    print("Warning: OpenEXR not available. Using cv2 for EXR loading.")

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


@dataclass
class ConversionConfig:
    """Configuration for the conversion process."""
    input_dir: Path
    output_dir: Path
    train_split: float = 0.9
    scenes_per_chunk: int = 100
    image_size: tuple = (256, 256)
    include_depth: bool = True
    include_mask: bool = True
    include_normal: bool = False
    num_views: int = 32  # Expected views per object
    min_views: int = 4   # Minimum views required
    max_objects: int = 10000
    # Depth validation settings
    max_valid_depth: float = 50.0  # Depths larger than this are considered background/invalid
    min_valid_foreground_ratio: float = 0.5  # At least 50% of foreground pixels must have valid depth
    expected_depth_range: tuple = (0.5, 10.0)  # Expected depth range for valid scenes


# Blender to OpenCV CAMERA coordinate transformation
# Blender camera: +X right, +Y up, -Z forward (looks along -Z)
# OpenCV camera:  +X right, +Y down, +Z forward (looks along +Z)
# This transforms the camera's local axes, not world coordinates!
BLENDER_TO_OPENCV = np.array([
    [1,  0,  0,  0],
    [0, -1,  0,  0],  # Blender +Y (up) -> OpenCV -Y (down)
    [0,  0, -1,  0],  # Blender +Z (back) -> OpenCV -Z (forward is -Z_blender = +Z_opencv)
    [0,  0,  0,  1]
], dtype=np.float32)


def load_exr_depth(filepath: Path) -> Optional[np.ndarray]:
    """Load depth from EXR file."""
    if HAS_OPENEXR:
        try:
            exr_file = OpenEXR.InputFile(str(filepath))
            header = exr_file.header()
            dw = header['dataWindow']
            width = dw.max.x - dw.min.x + 1
            height = dw.max.y - dw.min.y + 1

            # Try different channel names
            channel_names = ['Z', 'Y', 'R', 'V']
            pt = Imath.PixelType(Imath.PixelType.FLOAT)

            for channel in channel_names:
                if channel in header['channels']:
                    depth_str = exr_file.channel(channel, pt)
                    depth = np.frombuffer(depth_str, dtype=np.float32)
                    depth = depth.reshape((height, width))
                    return depth

            # If no standard channel found, try first available
            channels = list(header['channels'].keys())
            if channels:
                depth_str = exr_file.channel(channels[0], pt)
                depth = np.frombuffer(depth_str, dtype=np.float32)
                depth = depth.reshape((height, width))
                return depth

        except Exception as e:
            print(f"OpenEXR failed for {filepath}: {e}")

    if HAS_CV2:
        try:
            depth = cv2.imread(str(filepath), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            if depth is not None:
                if len(depth.shape) == 3:
                    # Use first channel (usually contains depth)
                    depth = depth[:, :, 0]
                return depth.astype(np.float32)
        except Exception as e:
            print(f"cv2 failed for {filepath}: {e}")

    return None


def load_mask(filepath: Path) -> Optional[np.ndarray]:
    """Load binary mask from PNG file."""
    try:
        mask = Image.open(filepath)
        mask = np.array(mask)
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]  # Take first channel
        # Normalize to 0-1
        mask = (mask > 127).astype(np.float32)
        return mask
    except Exception as e:
        print(f"Failed to load mask {filepath}: {e}")
        return None


def convert_blender_camera_to_depthsplat(
    c2w_blender: np.ndarray,
    intrinsics_px: np.ndarray,
    image_width: int,
    image_height: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert Blender camera parameters to DepthSplat format.

    Args:
        c2w_blender: 4x4 camera-to-world matrix in Blender coordinates
        intrinsics_px: 3x3 intrinsics matrix in pixel units
        image_width: Image width in pixels
        image_height: Image height in pixels

    Returns:
        extrinsics: 4x4 camera-to-world matrix in OpenCV coordinates
        intrinsics_norm: 3x3 normalized intrinsics matrix
    """
    # Convert camera coordinate system: Blender -> OpenCV
    # Post-multiplying transforms the camera's local axes:
    # - Blender camera looks along -Z, OpenCV camera looks along +Z
    # - Blender +Y is up, OpenCV +Y is down
    c2w_opencv = c2w_blender @ BLENDER_TO_OPENCV

    # Normalize intrinsics by image dimensions
    # DepthSplat expects normalized intrinsics where coordinates are in [0, 1]
    intrinsics_norm = intrinsics_px.copy().astype(np.float32)
    intrinsics_norm[0, 0] /= image_width   # fx
    intrinsics_norm[1, 1] /= image_height  # fy
    intrinsics_norm[0, 2] /= image_width   # cx
    intrinsics_norm[1, 2] /= image_height  # cy

    return c2w_opencv.astype(np.float32), intrinsics_norm


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert PIL image to tensor for storage (as compressed bytes)."""
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    # Use np.frombuffer with copy to avoid non-writable tensor warning
    arr = np.frombuffer(buffer.getvalue(), dtype=np.uint8).copy()
    return torch.from_numpy(arr)


def validate_depth_with_mask(
    depth: np.ndarray,
    mask: np.ndarray,
    config: ConversionConfig
) -> tuple[bool, str, dict]:
    """
    Validate depth map using mask to identify foreground pixels.

    Returns:
        (is_valid, reason, stats)
    """
    stats = {}

    # Identify foreground pixels (mask > 0.5)
    fg_mask = mask > 0.5
    fg_count = fg_mask.sum()

    if fg_count == 0:
        return False, "No foreground pixels in mask", stats

    fg_depth = depth[fg_mask]

    # Check valid foreground depth (within expected range, not background value)
    valid_fg = (fg_depth > 0) & (fg_depth < config.max_valid_depth)
    valid_ratio = valid_fg.sum() / fg_count

    stats['fg_pixels'] = int(fg_count)
    stats['valid_fg_ratio'] = float(valid_ratio)
    stats['fg_depth_min'] = float(fg_depth[valid_fg].min()) if valid_fg.any() else float('nan')
    stats['fg_depth_max'] = float(fg_depth[valid_fg].max()) if valid_fg.any() else float('nan')
    stats['fg_depth_mean'] = float(fg_depth[valid_fg].mean()) if valid_fg.any() else float('nan')

    if valid_ratio < config.min_valid_foreground_ratio:
        return False, f"Only {valid_ratio:.1%} foreground pixels have valid depth", stats

    # Check if depths are in expected range
    min_depth, max_depth = config.expected_depth_range
    if valid_fg.any():
        actual_min = fg_depth[valid_fg].min()
        actual_max = fg_depth[valid_fg].max()
        if actual_min > max_depth or actual_max < min_depth:
            return False, f"Depth range [{actual_min:.2f}, {actual_max:.2f}] outside expected [{min_depth}, {max_depth}]", stats

    return True, "OK", stats


def clean_depth_with_mask(
    depth: np.ndarray,
    mask: np.ndarray,
    config: ConversionConfig
) -> np.ndarray:
    """
    Clean depth map by setting invalid/background values to a sentinel.

    Background pixels (mask <= 0.5) and invalid depths are set to -1.
    This allows the training code to identify and ignore these pixels.
    """
    cleaned = depth.copy()

    # Set background to sentinel
    bg_mask = mask <= 0.5
    cleaned[bg_mask] = -1.0

    # Set invalid foreground depths (too large) to sentinel
    invalid_fg = (mask > 0.5) & (depth >= config.max_valid_depth)
    cleaned[invalid_fg] = -1.0

    return cleaned


def process_object(
    object_dir: Path,
    config: ConversionConfig
) -> Optional[dict]:
    """
    Process a single Objaverse object directory.

    Returns:
        Scene dictionary compatible with DepthSplat format, or None if failed.
    """
    metadata_path = object_dir / "metadata.json"
    if not metadata_path.exists():
        return None

    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    except Exception as e:
        print(f"Failed to load metadata from {metadata_path}: {e}")
        return None

    views = metadata.get("views", [])
    if len(views) < config.min_views:
        print(f"Skipping {object_dir.name}: only {len(views)} views (min: {config.min_views})")
        return None

    # Get image dimensions from render settings
    render_settings = metadata.get("render_settings", {})
    resolution = render_settings.get("resolution", [256, 256])
    image_width, image_height = resolution[0], resolution[1]

    # Process each view
    images = []
    cameras = []
    masks = []
    depths = []

    for view in views:
        # Load RGB image
        image_path = object_dir / view["image"]
        if not image_path.exists():
            continue

        try:
            image = Image.open(image_path).convert('RGB')
            if image.size != tuple(config.image_size):
                image = image.resize(config.image_size, Image.LANCZOS)
        except Exception as e:
            print(f"Failed to load image {image_path}: {e}")
            continue

        # Load camera parameters
        c2w_blender = np.array(view["camera_to_world"], dtype=np.float32)
        intrinsics_px = np.array(view["intrinsics"], dtype=np.float32)

        # Convert to DepthSplat format
        c2w_opencv, intrinsics_norm = convert_blender_camera_to_depthsplat(
            c2w_blender, intrinsics_px, image_width, image_height
        )

        # Build camera tensor [18]: normalized intrinsics (4) + padding (2) + w2c (12)
        # Format: [fx, fy, cx, cy, 0, 0, w2c_row0, w2c_row1, w2c_row2]
        w2c = np.linalg.inv(c2w_opencv)
        camera = np.zeros(18, dtype=np.float32)
        camera[0] = intrinsics_norm[0, 0]  # fx normalized
        camera[1] = intrinsics_norm[1, 1]  # fy normalized
        camera[2] = intrinsics_norm[0, 2]  # cx normalized
        camera[3] = intrinsics_norm[1, 2]  # cy normalized
        # [4:6] are unused padding
        camera[6:18] = w2c[:3, :].flatten()  # w2c 3x4

        images.append(image_to_tensor(image))
        cameras.append(camera)

        # Load mask if requested
        if config.include_mask and "mask" in view:
            mask_path = object_dir / view["mask"]
            if mask_path.exists():
                mask = load_mask(mask_path)
                if mask is not None:
                    # Resize if needed
                    if mask.shape != tuple(config.image_size[::-1]):
                        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
                        mask_img = mask_img.resize(config.image_size, Image.NEAREST)
                        mask = np.array(mask_img).astype(np.float32) / 255.0
                    masks.append(torch.from_numpy(mask))

        # Load depth if requested
        if config.include_depth and "depth" in view:
            depth_path = object_dir / view["depth"]
            if depth_path.exists():
                depth = load_exr_depth(depth_path)
                if depth is not None:
                    # Resize if needed
                    if depth.shape != tuple(config.image_size[::-1]):
                        depth_img = Image.fromarray(depth)
                        depth_img = depth_img.resize(config.image_size, Image.BILINEAR)
                        depth = np.array(depth_img)

                    # Clean depth using mask if available
                    if masks and len(masks) == len(depths) + 1:
                        # Use the corresponding mask (just added)
                        mask_np = masks[-1].numpy() if isinstance(masks[-1], torch.Tensor) else masks[-1]
                        depth = clean_depth_with_mask(depth, mask_np, config)

                    # Make a copy to ensure writability
                    depths.append(torch.from_numpy(depth.copy()))

    if len(images) < config.min_views:
        print(f"Skipping {object_dir.name}: only {len(images)} valid views")
        return None

    # Validate depth data if both depth and mask are available
    if depths and masks and len(depths) == len(images) and len(masks) == len(images):
        # Check first view's depth validity as a proxy for the whole scene
        first_depth = depths[0].numpy()
        first_mask = masks[0].numpy()

        is_valid, reason, stats = validate_depth_with_mask(first_depth, first_mask, config)
        if not is_valid:
            print(f"Skipping {object_dir.name}: {reason}")
            return None

    # Build scene dictionary
    scene = {
        "key": object_dir.name,
        "cameras": torch.from_numpy(np.stack(cameras)),
        "images": images,
    }

    # Add optional data
    if masks and len(masks) == len(images):
        scene["masks"] = torch.stack(masks)

    if depths and len(depths) == len(images):
        scene["depths"] = torch.stack(depths)

    # Store normalization info for reference
    if "normalization" in metadata:
        scene["normalization"] = metadata["normalization"]

    return scene


def process_zip_file(zip_path: Path, config: ConversionConfig) -> list[dict]:
    """Process a single zip file and return list of scenes."""
    scenes = []
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(temp_dir)

            temp_path = Path(temp_dir)
            extracted_dirs = [d for d in temp_path.iterdir() if d.is_dir()]

            for obj_dir in extracted_dirs:
                if (obj_dir / "metadata.json").exists():
                    scene = process_object(obj_dir, config)
                    if scene is not None:
                        scenes.append(scene)

        except zipfile.BadZipFile:
            pass
        except Exception:
            pass
    return scenes


class ChunkWriter:
    """Incrementally write scenes to chunks without buffering all in memory."""

    def __init__(self, output_dir: Path, stage: str, scenes_per_chunk: int):
        self.stage_dir = output_dir / stage
        self.stage_dir.mkdir(parents=True, exist_ok=True)
        self.scenes_per_chunk = scenes_per_chunk
        self.current_chunk = []
        self.chunk_idx = 0
        self.index = {}
        self.total_scenes = 0

    def add_scene(self, scene: dict):
        """Add a scene, automatically flushing to disk when chunk is full."""
        self.current_chunk.append(scene)
        self.total_scenes += 1

        if len(self.current_chunk) >= self.scenes_per_chunk:
            self._flush_chunk()

    def _flush_chunk(self):
        """Write current chunk to disk and clear memory."""
        if not self.current_chunk:
            return

        chunk_filename = f"{self.chunk_idx:06d}.torch"
        chunk_path = self.stage_dir / chunk_filename

        torch.save(self.current_chunk, chunk_path)

        for scene in self.current_chunk:
            self.index[scene["key"]] = chunk_filename

        self.current_chunk = []
        self.chunk_idx += 1

    def finalize(self) -> dict[str, str]:
        """Flush any remaining scenes and save index."""
        self._flush_chunk()

        with open(self.stage_dir / "index.json", 'w') as f:
            json.dump(self.index, f, indent=2)

        return self.index


def create_chunks(
    scenes: list[dict],
    output_dir: Path,
    scenes_per_chunk: int,
    stage: str
) -> dict[str, str]:
    """
    Create .torch chunk files from scenes.

    Returns:
        Index mapping scene keys to chunk filenames.
    """
    stage_dir = output_dir / stage
    stage_dir.mkdir(parents=True, exist_ok=True)

    index = {}

    num_chunks = (len(scenes) + scenes_per_chunk - 1) // scenes_per_chunk

    for chunk_idx in tqdm(range(num_chunks), desc=f"Creating {stage} chunks"):
        start_idx = chunk_idx * scenes_per_chunk
        end_idx = min(start_idx + scenes_per_chunk, len(scenes))

        chunk_scenes = scenes[start_idx:end_idx]
        chunk_filename = f"{chunk_idx:06d}.torch"
        chunk_path = stage_dir / chunk_filename

        # Save chunk
        torch.save(chunk_scenes, chunk_path)

        # Update index
        for scene in chunk_scenes:
            index[scene["key"]] = chunk_filename

    # Save index
    with open(stage_dir / "index.json", 'w') as f:
        json.dump(index, f, indent=2)

    return index


def main():
    parser = argparse.ArgumentParser(
        description="Convert Objaverse Blender renders to DepthSplat format"
    )
    parser.add_argument(
        "--input-dir", "-i",
        type=Path,
        required=True,
        help="Input directory containing Objaverse object directories"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        required=True,
        help="Output directory for DepthSplat format data"
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.9,
        help="Fraction of data for training (default: 0.9)"
    )
    parser.add_argument(
        "--scenes-per-chunk",
        type=int,
        default=100,
        help="Number of scenes per .torch chunk file (default: 100)"
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=[256, 256],
        help="Target image size [width, height] (default: 256 256)"
    )
    parser.add_argument(
        "--include-depth",
        action="store_true",
        default=True,
        help="Include depth maps in output"
    )
    parser.add_argument(
        "--include-mask",
        action="store_true",
        default=True,
        help="Include segmentation masks in output"
    )
    parser.add_argument(
        "--num-views",
        type=int,
        default=32,
        help="Expected number of views per object (default: 32)"
    )
    parser.add_argument(
        "--min-views",
        type=int,
        default=4,
        help="Minimum number of views required (default: 4)"
    )
    parser.add_argument(
        "--max-objects",
        type=int,
        default=20000,
        help="Maximum number of objects to convert (default: 20000)"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=1, #os.cpu_count() or 4,
        help=f"Number of parallel workers (default: {os.cpu_count() or 4}, auto-detected CPUs)"
    )

    args = parser.parse_args()

    config = ConversionConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        train_split=args.train_split,
        scenes_per_chunk=args.scenes_per_chunk,
        image_size=tuple(args.image_size),
        include_depth=args.include_depth,
        include_mask=args.include_mask,
        num_views=args.num_views,
        min_views=args.min_views,
        max_objects=args.max_objects,
    )

    # Find all object directories or zip files
    print(f"Scanning input directory: {config.input_dir}")
    object_dirs = []
    zip_files = []
    for item in config.input_dir.iterdir():
        if item.is_dir() and (item / "metadata.json").exists():
            object_dirs.append(item)
        elif item.suffix == '.zip':
            zip_files.append(item)

    print(f"Found {len(object_dirs)} unzipped objects with metadata")
    print(f"Found {len(zip_files)} zip files")

    if len(object_dirs) == 0 and len(zip_files) == 0:
        print("No valid objects found. Exiting.")
        return

    # Combine all sources and shuffle for random train/test split
    all_sources = [(obj_dir, 'dir') for obj_dir in object_dirs] + \
                  [(zip_path, 'zip') for zip_path in zip_files]
    np.random.seed(42)
    np.random.shuffle(all_sources)

    if config.max_objects > 0 and len(all_sources) > config.max_objects:
        print(f"Limiting to {config.max_objects} objects (from {len(all_sources)} available)")
        all_sources = all_sources[:config.max_objects]

    split_idx = int(len(all_sources) * config.train_split)
    split_idx = int(len(all_sources) * config.train_split)
    train_sources = all_sources[:split_idx]
    test_sources = all_sources[split_idx:]

    print(f"Split: {len(train_sources)} sources for train, {len(test_sources)} for test")

    # Create output directory and chunk writers
    config.output_dir.mkdir(parents=True, exist_ok=True)
    train_writer = ChunkWriter(config.output_dir, "train", config.scenes_per_chunk)
    test_writer = ChunkWriter(config.output_dir, "test", config.scenes_per_chunk)

    num_workers = args.workers
    failed = 0

    def process_sources(sources: list, writer: ChunkWriter, desc: str):
        """Process sources and write scenes incrementally."""
        nonlocal failed

        dirs = [s[0] for s in sources if s[1] == 'dir']
        zips = [s[0] for s in sources if s[1] == 'zip']

        # Process directories
        if dirs:
            if num_workers > 1:
                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    futures = {executor.submit(process_object, d, config): d for d in dirs}
                    for future in tqdm(as_completed(futures), total=len(futures),
                                       desc=f"{desc} dirs"):
                        try:
                            scene = future.result()
                            if scene is not None:
                                writer.add_scene(scene)
                            else:
                                failed += 1
                        except Exception as e:
                            print(f"Worker error: {e}")
                            failed += 1
            else:
                for d in tqdm(dirs, desc=f"{desc} dirs"):
                    scene = process_object(d, config)
                    if scene is not None:
                        writer.add_scene(scene)
                    else:
                        failed += 1

        # Process zips
        if zips:
            if num_workers > 1:
                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    futures = {executor.submit(process_zip_file, z, config): z for z in zips}
                    for future in tqdm(as_completed(futures), total=len(futures),
                                       desc=f"{desc} zips"):
                        try:
                            for scene in future.result():
                                writer.add_scene(scene)
                        except Exception as e:
                            print(f"Worker error: {e}")
                            failed += 1
            else:
                for z in tqdm(zips, desc=f"{desc} zips"):
                    for scene in process_zip_file(z, config):
                        writer.add_scene(scene)

    # Process train and test sets with streaming writes
    print("Processing training data...")
    process_sources(train_sources, train_writer, "Train")
    train_index = train_writer.finalize()
    print(f"Created {len(train_index)} training scenes in {train_writer.chunk_idx} chunks")

    print("Processing test data...")
    process_sources(test_sources, test_writer, "Test")
    test_index = test_writer.finalize()
    print(f"Created {len(test_index)} test scenes in {test_writer.chunk_idx} chunks")

    print(f"Total failed: {failed}")

    # Save conversion config for reference
    config_dict = {
        "input_dir": str(config.input_dir),
        "output_dir": str(config.output_dir),
        "train_split": config.train_split,
        "scenes_per_chunk": config.scenes_per_chunk,
        "image_size": list(config.image_size),
        "include_depth": config.include_depth,
        "include_mask": config.include_mask,
        "num_views": config.num_views,
        "min_views": config.min_views,
        "total_scenes": len(train_index) + len(test_index),
        "train_scenes": len(train_index),
        "test_scenes": len(test_index),
    }
    with open(config.output_dir / "conversion_config.json", 'w') as f:
        json.dump(config_dict, f, indent=2)

    print(f"\nConversion complete! Output saved to: {config.output_dir}")
    print(f"  Train: {config.output_dir / 'train'}")
    print(f"  Test:  {config.output_dir / 'test'}")


if __name__ == "__main__":
    main()
