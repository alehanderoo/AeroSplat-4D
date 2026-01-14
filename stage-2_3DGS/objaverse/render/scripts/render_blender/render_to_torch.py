#!/usr/bin/env python3
"""
Integrated Objaverse rendering and DepthSplat .torch file creation.

This script renders Objaverse objects using Blender and directly converts them
to .torch chunk files for DepthSplat training, without intermediate zip storage.

Usage:
    uv run render_to_torch.py --gpu_devices=1 --render_depth True --render_mask True --num_workers=3
"""

import glob
import gzip
import json
import multiprocessing as mp
import os
import platform
import random
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import fire
import fsspec
import GPUtil
import numpy as np
import torch
from loguru import logger
from PIL import Image
from tqdm import tqdm

from objaverse.utils import get_file_hash

# Optional OpenEXR support
try:
    import OpenEXR
    import Imath
    HAS_OPENEXR = True
except ImportError:
    HAS_OPENEXR = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# ============================================================================
# Configuration
# ============================================================================

HIGH_QUALITY_UIDS_PATH = os.path.expanduser(
    # "~/thesis/assets/objaverse/high_quality_uids.txt"
    # "~/thesis/assets/objaverse/trainingsetB_part1.txt"
    "~/thesis/assets/objaverse/trainingsetB_part3.txt"
)
OBJAVERSE_V1_BASE = os.path.expanduser("/home/sandro/.objaverse/hf-objaverse-v1")
OBJECT_PATHS_FILE = os.path.join(OBJAVERSE_V1_BASE, "object-paths.json.gz")
MIN_FREE_BYTES = 50 * 1024**3  # 50 GiB

# Blender to OpenCV camera coordinate transformation
BLENDER_TO_OPENCV = np.array([
    [1,  0,  0,  0],
    [0, -1,  0,  0],
    [0,  0, -1,  0],
    [0,  0,  0,  1]
], dtype=np.float32)


@dataclass
class TorchConfig:
    """Configuration for torch file creation."""
    image_size: Tuple[int, int] = (256, 256)
    include_depth: bool = True
    include_mask: bool = True
    include_normal: bool = False
    min_views: int = 4
    max_valid_depth: float = 50.0
    min_valid_foreground_ratio: float = 0.5
    expected_depth_range: Tuple[float, float] = (0.5, 10.0)


# ============================================================================
# Utility Functions
# ============================================================================

def log_processed_object(csv_filename: str, *args) -> None:
    """Log when an object is done being used."""
    args_str = ",".join([str(arg) for arg in args])
    dirname = os.path.expanduser("~/.objaverse/logs/")
    os.makedirs(dirname, exist_ok=True)
    with open(os.path.join(dirname, csv_filename), "a", encoding="utf-8") as f:
        f.write(f"{time.time()},{args_str}\n")


def normalize_uid(uid: str) -> str:
    """Normalizes a UID string by removing hyphens and whitespace."""
    uid = uid.strip().lower()
    if uid.endswith(".glb"):
        uid = uid[:-4]
    return uid.replace("-", "")


def load_requested_uids(uids_path: str) -> List[str]:
    """Loads the requested Objaverse v1 UIDs."""
    if not os.path.isfile(uids_path):
        raise FileNotFoundError(f"Missing UID list at {uids_path}.")

    with open(uids_path, "r", encoding="utf-8") as f:
        uids = [normalize_uid(line) for line in f if line.strip()]

    if not uids:
        raise ValueError(f"No UIDs found in {uids_path}.")

    logger.info(f"Loaded {len(uids)} requested Objaverse v1 UIDs.")
    return uids


def load_object_paths(object_paths_file: str = OBJECT_PATHS_FILE) -> Dict[str, str]:
    """Loads UID -> relative object path mappings from Objaverse v1."""
    if not os.path.isfile(object_paths_file):
        raise FileNotFoundError(
            f"Missing object paths mapping at {object_paths_file}. "
            "Run objaverse.load_objects first to download it."
        )

    with gzip.open(object_paths_file, "rt", encoding="utf-8") as f:
        object_paths: Dict[str, str] = json.load(f)

    logger.info(f"Loaded {len(object_paths)} Objaverse v1 object paths.")
    return object_paths


def resolve_local_path(uid: str, object_paths: Dict[str, str]) -> Optional[str]:
    """Gets the absolute path to a local GLB for the provided UID."""
    relative_path = object_paths.get(uid)
    if not relative_path and uid.endswith(".glb"):
        relative_path = object_paths.get(uid[:-4])

    if not relative_path:
        return None

    abs_path = os.path.join(OBJAVERSE_V1_BASE, relative_path)
    if not os.path.isfile(abs_path):
        return None
    return abs_path


def ensure_free_space(
    min_free_bytes: int = MIN_FREE_BYTES, path: str = OBJAVERSE_V1_BASE
) -> None:
    """Ensures there is sufficient disk space; exits early if not."""
    if not os.path.exists(path):
        path = os.path.expanduser("~")
    _, _, free = shutil.disk_usage(path)
    if free <= min_free_bytes:
        logger.error(
            f"Low disk space detected ({free / 1024**3:.1f} GiB free). "
            "Stopping rendering to prevent running out of space."
        )
        raise SystemExit(1)


def choose_device(gpu_devices: Union[int, List[int]]) -> Tuple[bool, Optional[int]]:
    """Chooses a GPU/CPU device following the same convention as main.py."""
    using_gpu = True
    gpu_index: Optional[int] = 0
    if isinstance(gpu_devices, int) and gpu_devices > 0:
        gpu_index = random.randint(0, gpu_devices - 1)
    elif isinstance(gpu_devices, list) and gpu_devices:
        gpu_index = random.choice(gpu_devices)
    elif isinstance(gpu_devices, int) and gpu_devices == 0:
        using_gpu = False
        gpu_index = None
    else:
        raise ValueError(
            f"gpu_devices must be int > 0, 0, or list[int]. Got {gpu_devices}."
        )
    return using_gpu, gpu_index


# Global variable for multiprocessing job preparation
_object_paths_global: Dict[str, str] = {}


def _init_prepare_worker(object_paths: Dict[str, str]) -> None:
    """Initialize worker process with object paths dict."""
    global _object_paths_global
    _object_paths_global = object_paths


def _prepare_job_worker(uid: str) -> Optional[Tuple[str, str, str, Dict[str, str]]]:
    """Prepare a single job: resolve path and compute hash. Used in parallel."""
    local_path = resolve_local_path(uid, _object_paths_global)
    if not local_path:
        return None
    sha256 = get_file_hash(local_path)
    metadata = {
        "uid": uid,
        "source": "objaverse_v1",
        "relative_path": os.path.relpath(local_path, OBJAVERSE_V1_BASE),
    }
    return (uid, local_path, sha256, metadata)


# ============================================================================
# EXR/Image Loading Functions
# ============================================================================

def load_exr_depth(filepath: Path) -> Optional[np.ndarray]:
    """Load depth from EXR file."""
    if HAS_OPENEXR:
        try:
            exr_file = OpenEXR.InputFile(str(filepath))
            header = exr_file.header()
            dw = header['dataWindow']
            width = dw.max.x - dw.min.x + 1
            height = dw.max.y - dw.min.y + 1

            channel_names = ['Z', 'Y', 'R', 'V']
            pt = Imath.PixelType(Imath.PixelType.FLOAT)

            for channel in channel_names:
                if channel in header['channels']:
                    depth_str = exr_file.channel(channel, pt)
                    depth = np.frombuffer(depth_str, dtype=np.float32)
                    depth = depth.reshape((height, width))
                    return depth

            channels = list(header['channels'].keys())
            if channels:
                depth_str = exr_file.channel(channels[0], pt)
                depth = np.frombuffer(depth_str, dtype=np.float32)
                depth = depth.reshape((height, width))
                return depth

        except Exception as e:
            logger.warning(f"OpenEXR failed for {filepath}: {e}")

    if HAS_CV2:
        try:
            depth = cv2.imread(str(filepath), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            if depth is not None:
                if len(depth.shape) == 3:
                    depth = depth[:, :, 0]
                return depth.astype(np.float32)
        except Exception as e:
            logger.warning(f"cv2 failed for {filepath}: {e}")

    return None


def load_mask(filepath: Path) -> Optional[np.ndarray]:
    """Load binary mask from PNG file."""
    try:
        mask = Image.open(filepath)
        mask = np.array(mask)
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]
        mask = (mask > 127).astype(np.float32)
        return mask
    except Exception as e:
        logger.warning(f"Failed to load mask {filepath}: {e}")
        return None


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert PIL image to tensor for storage (as compressed bytes)."""
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    arr = np.frombuffer(buffer.getvalue(), dtype=np.uint8).copy()
    return torch.from_numpy(arr)


# ============================================================================
# Camera Conversion Functions
# ============================================================================

def convert_blender_camera_to_depthsplat(
    c2w_blender: np.ndarray,
    intrinsics_px: np.ndarray,
    image_width: int,
    image_height: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert Blender camera parameters to DepthSplat format.
    """
    c2w_opencv = c2w_blender @ BLENDER_TO_OPENCV

    intrinsics_norm = intrinsics_px.copy().astype(np.float32)
    intrinsics_norm[0, 0] /= image_width
    intrinsics_norm[1, 1] /= image_height
    intrinsics_norm[0, 2] /= image_width
    intrinsics_norm[1, 2] /= image_height

    return c2w_opencv.astype(np.float32), intrinsics_norm


def validate_depth_with_mask(
    depth: np.ndarray,
    mask: np.ndarray,
    config: TorchConfig
) -> Tuple[bool, str, dict]:
    """Validate depth map using mask to identify foreground pixels."""
    stats = {}

    fg_mask = mask > 0.5
    fg_count = fg_mask.sum()

    if fg_count == 0:
        return False, "No foreground pixels in mask", stats

    fg_depth = depth[fg_mask]

    valid_fg = (fg_depth > 0) & (fg_depth < config.max_valid_depth)
    valid_ratio = valid_fg.sum() / fg_count

    stats['fg_pixels'] = int(fg_count)
    stats['valid_fg_ratio'] = float(valid_ratio)
    stats['fg_depth_min'] = float(fg_depth[valid_fg].min()) if valid_fg.any() else float('nan')
    stats['fg_depth_max'] = float(fg_depth[valid_fg].max()) if valid_fg.any() else float('nan')
    stats['fg_depth_mean'] = float(fg_depth[valid_fg].mean()) if valid_fg.any() else float('nan')

    if valid_ratio < config.min_valid_foreground_ratio:
        return False, f"Only {valid_ratio:.1%} foreground pixels have valid depth", stats

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
    config: TorchConfig
) -> np.ndarray:
    """Clean depth map by setting invalid/background values to a sentinel."""
    cleaned = depth.copy()

    bg_mask = mask <= 0.5
    cleaned[bg_mask] = -1.0

    invalid_fg = (mask > 0.5) & (depth >= config.max_valid_depth)
    cleaned[invalid_fg] = -1.0

    return cleaned


# ============================================================================
# Scene Processing Functions
# ============================================================================

def process_rendered_object(
    object_dir: Path,
    config: TorchConfig
) -> Optional[dict]:
    """
    Process a rendered object directory into a scene dictionary.

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
        logger.warning(f"Failed to load metadata from {metadata_path}: {e}")
        return None

    views = metadata.get("views", [])
    if len(views) < config.min_views:
        logger.warning(f"Skipping {object_dir.name}: only {len(views)} views (min: {config.min_views})")
        return None

    render_settings = metadata.get("render_settings", {})
    resolution = render_settings.get("resolution", [256, 256])
    image_width, image_height = resolution[0], resolution[1]

    images = []
    cameras = []
    masks = []
    depths = []

    for view in views:
        image_path = object_dir / view["image"]
        if not image_path.exists():
            continue

        try:
            image = Image.open(image_path).convert('RGB')
            if image.size != tuple(config.image_size):
                image = image.resize(config.image_size, Image.LANCZOS)
        except Exception as e:
            logger.warning(f"Failed to load image {image_path}: {e}")
            continue

        c2w_blender = np.array(view["camera_to_world"], dtype=np.float32)
        intrinsics_px = np.array(view["intrinsics"], dtype=np.float32)

        c2w_opencv, intrinsics_norm = convert_blender_camera_to_depthsplat(
            c2w_blender, intrinsics_px, image_width, image_height
        )

        w2c = np.linalg.inv(c2w_opencv)
        camera = np.zeros(18, dtype=np.float32)
        camera[0] = intrinsics_norm[0, 0]
        camera[1] = intrinsics_norm[1, 1]
        camera[2] = intrinsics_norm[0, 2]
        camera[3] = intrinsics_norm[1, 2]
        camera[6:18] = w2c[:3, :].flatten()

        images.append(image_to_tensor(image))
        cameras.append(camera)

        if config.include_mask and "mask" in view:
            mask_path = object_dir / view["mask"]
            if mask_path.exists():
                mask = load_mask(mask_path)
                if mask is not None:
                    if mask.shape != tuple(config.image_size[::-1]):
                        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
                        mask_img = mask_img.resize(config.image_size, Image.NEAREST)
                        mask = np.array(mask_img).astype(np.float32) / 255.0
                    masks.append(torch.from_numpy(mask))

        if config.include_depth and "depth" in view:
            depth_path = object_dir / view["depth"]
            if depth_path.exists():
                depth = load_exr_depth(depth_path)
                if depth is not None:
                    if depth.shape != tuple(config.image_size[::-1]):
                        depth_img = Image.fromarray(depth)
                        depth_img = depth_img.resize(config.image_size, Image.BILINEAR)
                        depth = np.array(depth_img)

                    if masks and len(masks) == len(depths) + 1:
                        mask_np = masks[-1].numpy() if isinstance(masks[-1], torch.Tensor) else masks[-1]
                        depth = clean_depth_with_mask(depth, mask_np, config)

                    depths.append(torch.from_numpy(depth.copy()))

    if len(images) < config.min_views:
        logger.warning(f"Skipping {object_dir.name}: only {len(images)} valid views")
        return None

    if depths and masks and len(depths) == len(images) and len(masks) == len(images):
        first_depth = depths[0].numpy()
        first_mask = masks[0].numpy()

        is_valid, reason, stats = validate_depth_with_mask(first_depth, first_mask, config)
        if not is_valid:
            logger.warning(f"Skipping {object_dir.name}: {reason}")
            return None

    scene = {
        "key": object_dir.name,
        "cameras": torch.from_numpy(np.stack(cameras)),
        "images": images,
    }

    if masks and len(masks) == len(images):
        scene["masks"] = torch.stack(masks)

    if depths and len(depths) == len(images):
        scene["depths"] = torch.stack(depths)

    if "normalization" in metadata:
        scene["normalization"] = metadata["normalization"]

    return scene


# ============================================================================
# Chunk Writer
# ============================================================================

class ChunkWriter:
    """Incrementally write scenes to chunks without buffering all in memory."""

    def __init__(self, output_dir: Path, stage: str, scenes_per_chunk: int):
        self.stage_dir = output_dir / stage
        self.stage_dir.mkdir(parents=True, exist_ok=True)
        self.scenes_per_chunk = scenes_per_chunk
        self.current_chunk: List[dict] = []
        self.chunk_idx = 0
        self.index: Dict[str, str] = {}
        self.total_scenes = 0
        self._lock = mp.Lock()

    def add_scene(self, scene: dict) -> Optional[str]:
        """
        Add a scene, automatically flushing to disk when chunk is full.

        Returns:
            Path to the chunk file if a new chunk was written, None otherwise.
        """
        with self._lock:
            self.current_chunk.append(scene)
            self.total_scenes += 1

            if len(self.current_chunk) >= self.scenes_per_chunk:
                return self._flush_chunk()
            return None

    def _flush_chunk(self) -> Optional[str]:
        """Write current chunk to disk and clear memory."""
        if not self.current_chunk:
            return None

        chunk_filename = f"{self.chunk_idx:06d}.torch"
        chunk_path = self.stage_dir / chunk_filename

        torch.save(self.current_chunk, chunk_path)
        logger.info(f"Saved chunk {chunk_filename} with {len(self.current_chunk)} scenes")

        for scene in self.current_chunk:
            self.index[scene["key"]] = chunk_filename

        self.current_chunk = []
        self.chunk_idx += 1

        return str(chunk_path)

    def finalize(self) -> Dict[str, str]:
        """Flush any remaining scenes and save index."""
        with self._lock:
            self._flush_chunk()

            with open(self.stage_dir / "index.json", 'w') as f:
                json.dump(self.index, f, indent=2)

            return self.index


# ============================================================================
# Rendering Functions
# ============================================================================

def render_and_convert_object(
    uid: str,
    local_path: str,
    sha256: str,
    metadata: Dict[str, str],
    num_renders: int,
    only_northern_hemisphere: bool,
    gpu_devices: Union[int, List[int]],
    render_timeout: int,
    render_depth: bool,
    render_normals: bool,
    render_mask: bool,
    camera_radius_min: float,
    camera_radius_max: float,
    fov_min_degrees: float,
    fov_max_degrees: float,
    torch_config: TorchConfig,
) -> Optional[dict]:
    """
    Render a single object and convert it to a scene dictionary.

    Returns:
        Scene dictionary if successful, None otherwise.
    """
    save_uid = uid
    using_gpu, gpu_index = choose_device(gpu_devices)

    with tempfile.TemporaryDirectory() as temp_dir:
        target_directory = os.path.join(temp_dir, save_uid)
        os.makedirs(target_directory, exist_ok=True)

        args = [
            f"--object_path '{local_path}'",
            f"--num_renders {num_renders}",
            f"--output_dir {target_directory}",
        ]

        if platform.system() == "Linux" and using_gpu:
            args.append("--engine BLENDER_EEVEE")
        elif platform.system() in {"Darwin", "Linux"} and not using_gpu:
            args.append("--engine CYCLES")
        else:
            raise NotImplementedError(
                f"Platform {platform.system()} is not supported for rendering."
            )

        if only_northern_hemisphere:
            args.append("--only_northern_hemisphere")
        if render_depth:
            args.append("--render_depth")
        if render_normals:
            args.append("--render_normals")
        if render_mask:
            args.append("--render_mask")
        args.append(f"--camera_radius_min {camera_radius_min}")
        args.append(f"--camera_radius_max {camera_radius_max}")
        args.append(f"--fov_min_degrees {fov_min_degrees}")
        args.append(f"--fov_max_degrees {fov_max_degrees}")

        command = (
            "blender-3.2.2-linux-x64/blender "
            "--background --python blender_script.py -- "
            + " ".join(args)
        )
        display_override = os.environ.get("DISPLAY")
        if display_override:
            command = f"export DISPLAY={display_override} && {command}"
        elif using_gpu and gpu_index is not None:
            display_value = f":0.{gpu_index}"
            command = f"export DISPLAY={display_value} && {command}"

        # Create log directory in temp
        log_file = os.path.join(temp_dir, f"{uid}.log")

        try:
            with open(log_file, "w") as f:
                subprocess.run(
                    ["bash", "-c", command],
                    timeout=render_timeout,
                    check=False,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                )
        except subprocess.TimeoutExpired:
            logger.warning(f"Rendering timed out for UID {uid} after {render_timeout}s")
            log_processed_object("render-to-torch-failed.csv", uid, sha256, "timeout")
            return None

        # Rename files (same logic as original)
        def rename_files(pattern_glob, new_format_func):
            found_files = glob.glob(os.path.join(target_directory, pattern_glob))
            renamed_map = {}
            for fpath in found_files:
                fname = os.path.basename(fpath)
                parts = os.path.splitext(fname)[0].split("_")
                try:
                    frame_num = int(parts[-1])
                    new_name = new_format_func(frame_num)
                    new_path = os.path.join(target_directory, new_name)
                    os.rename(fpath, new_path)
                    renamed_map[fname] = new_name
                except (ValueError, IndexError):
                    logger.warning(f"Could not parse frame number from {fname}")
            return renamed_map

        renamed_files_map = {}
        renamed_files_map.update(rename_files("render_rgb_*.png", lambda i: f"{i:03d}.png"))

        if render_depth:
            renamed_files_map.update(rename_files("render_depth_*.exr", lambda i: f"{i:03d}_depth.exr"))

        if render_normals:
            renamed_files_map.update(rename_files("render_normal_*.exr", lambda i: f"{i:03d}_normal.exr"))

        if render_mask:
            renamed_files_map.update(rename_files("render_mask_*.png", lambda i: f"{i:03d}_mask.png"))

        # Update metadata with new filenames
        metadata_path = os.path.join(target_directory, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                meta = json.load(f)

            if "views" in meta:
                for view in meta["views"]:
                    if "image" in view and view["image"] in renamed_files_map:
                        view["image"] = renamed_files_map[view["image"]]
                    if "depth" in view and view["depth"] in renamed_files_map:
                        view["depth"] = renamed_files_map[view["depth"]]
                    if "normal" in view and view["normal"] in renamed_files_map:
                        view["normal"] = renamed_files_map[view["normal"]]
                    if "mask" in view and view["mask"] in renamed_files_map:
                        view["mask"] = renamed_files_map[view["mask"]]

            # Add additional metadata
            meta["sha256"] = sha256
            meta["file_identifier"] = uid
            meta["save_uid"] = save_uid
            meta["metadata"] = metadata

            with open(metadata_path, 'w') as f:
                json.dump(meta, f, indent=2, sort_keys=True)

        # Verify rendering outputs
        rgb_files = [f for f in glob.glob(os.path.join(target_directory, "*.png"))
                     if "mask" not in os.path.basename(f) and os.path.basename(f)[0].isdigit()]
        metadata_files = glob.glob(os.path.join(target_directory, "*.json"))
        depth_files = glob.glob(os.path.join(target_directory, "*_depth.exr"))
        mask_files = glob.glob(os.path.join(target_directory, "*_mask.png"))

        missing_pngs = len(rgb_files) != num_renders
        missing_depth = render_depth and len(depth_files) != num_renders
        missing_masks = render_mask and len(mask_files) != num_renders
        missing_metadata = len(metadata_files) != 1

        if missing_pngs or missing_depth or missing_masks or missing_metadata:
            logger.error(f"Rendering failed for UID {uid}")
            log_processed_object("render-to-torch-failed.csv", uid, sha256, "incomplete")
            return None

        # Convert to scene dictionary
        scene = process_rendered_object(Path(target_directory), torch_config)

        if scene is None:
            log_processed_object("render-to-torch-failed.csv", uid, sha256, "conversion")
            return None

        log_processed_object("render-to-torch-success.csv", uid, sha256)
        return scene


def _render_worker(args: Tuple) -> Optional[dict]:
    """Worker function for parallel rendering."""
    (
        uid,
        local_path,
        sha256,
        metadata,
        num_renders,
        only_northern_hemisphere,
        gpu_devices,
        render_timeout,
        render_depth,
        render_normals,
        render_mask,
        camera_radius_min,
        camera_radius_max,
        fov_min_degrees,
        fov_max_degrees,
        torch_config_dict,
    ) = args

    ensure_free_space()

    torch_config = TorchConfig(**torch_config_dict)

    return render_and_convert_object(
        uid=uid,
        local_path=local_path,
        sha256=sha256,
        metadata=metadata,
        num_renders=num_renders,
        only_northern_hemisphere=only_northern_hemisphere,
        gpu_devices=gpu_devices,
        render_timeout=render_timeout,
        render_depth=render_depth,
        render_normals=render_normals,
        render_mask=render_mask,
        camera_radius_min=camera_radius_min,
        camera_radius_max=camera_radius_max,
        fov_min_degrees=fov_min_degrees,
        fov_max_degrees=fov_max_degrees,
        torch_config=torch_config,
    )


# ============================================================================
# Main Entry Point
# ============================================================================

def render_objaverse_to_torch(
    output_dir: str = "~/.objaverse/depthsplat",
    num_renders: int = 32,
    only_northern_hemisphere: bool = False,
    render_timeout: int = 300,
    gpu_devices: Optional[Union[int, List[int]]] = None,
    render_depth: bool = True,
    render_normals: bool = False,
    render_mask: bool = True,
    uids_path: str = HIGH_QUALITY_UIDS_PATH,
    object_paths_file: str = OBJECT_PATHS_FILE,
    num_workers: int = 2,
    num_objects: int = 20000,
    scenes_per_chunk: int = 100,
    train_split: float = 0.9,
    camera_radius_min: float = 1.5,
    camera_radius_max: float = 2.8,
    fov_min_degrees: float = 45.0,
    fov_max_degrees: float = 90.0,
) -> None:
    """
    Render Objaverse objects and directly create DepthSplat .torch files.

    Args:
        output_dir: Output directory for .torch files
        num_renders: Number of views to render per object
        only_northern_hemisphere: Only render from northern hemisphere
        render_timeout: Timeout for each render in seconds
        gpu_devices: Number of GPUs or list of GPU indices
        render_depth: Render depth maps
        render_normals: Render normal maps
        render_mask: Render object masks
        uids_path: Path to file with UIDs to render
        object_paths_file: Path to object-paths.json.gz
        num_workers: Number of parallel rendering workers
        num_objects: Maximum number of objects to render
        scenes_per_chunk: Number of scenes per .torch file
        train_split: Fraction of data for training
        camera_radius_min: Minimum camera radius
        camera_radius_max: Maximum camera radius
        fov_min_degrees: Minimum field of view
        fov_max_degrees: Maximum field of view
    """
    if platform.system() not in {"Linux", "Darwin"}:
        raise NotImplementedError(
            f"Platform {platform.system()} is not supported. Use Linux or MacOS."
        )

    if gpu_devices is None:
        try:
            gpu_devices = len(GPUtil.getGPUs())
        except (ValueError, Exception) as e:
            logger.warning(f"Failed to detect GPUs via GPUtil: {e}")
            logger.warning("Defaulting to 1 GPU. Override with --gpu_devices=N if needed.")
            gpu_devices = 1
    logger.info(f"Using {gpu_devices} GPU devices for rendering.")

    enabled_modalities = ["rgb"]
    if render_depth:
        enabled_modalities.append("depth")
    if render_normals:
        enabled_modalities.append("normals")
    if render_mask:
        enabled_modalities.append("mask")
    logger.info(f"Rendering modalities: {', '.join(enabled_modalities)}")

    # Load UIDs and object paths
    requested_uids = load_requested_uids(uids_path)
    if num_objects > 0:
        logger.info(f"Limiting to first {num_objects} objects out of {len(requested_uids)} requested.")
        requested_uids = requested_uids[:num_objects]
    object_paths = load_object_paths(object_paths_file)
    ensure_free_space()

    camera_radius_min = max(0.1, camera_radius_min)
    camera_radius_max = max(camera_radius_min, camera_radius_max)
    fov_min_degrees = max(1.0, fov_min_degrees)
    fov_max_degrees = max(fov_max_degrees, fov_min_degrees + 1.0)

    # Check for already processed UIDs by looking at existing index files
    output_path = Path(os.path.expanduser(output_dir))
    existing_uids = set()

    for stage in ["train", "test"]:
        index_path = output_path / stage / "index.json"
        if index_path.exists():
            with open(index_path, 'r') as f:
                existing_index = json.load(f)
                existing_uids.update(existing_index.keys())

    if existing_uids:
        logger.info(f"Found {len(existing_uids)} already processed UIDs")

    # Filter UIDs that haven't been processed yet
    uids_to_process = [uid for uid in requested_uids if uid not in existing_uids]
    logger.info(f"Checking {len(uids_to_process)} UIDs for local availability (using {mp.cpu_count()} CPU cores)...")

    jobs: List[Tuple[str, str, str, Dict[str, str]]] = []
    missing_uids = []

    with mp.Pool(
        processes=mp.cpu_count(),
        initializer=_init_prepare_worker,
        initargs=(object_paths,)
    ) as pool:
        results = list(tqdm(
            pool.imap(_prepare_job_worker, uids_to_process, chunksize=100),
            total=len(uids_to_process),
            desc="Preparing jobs",
            unit="uid"
        ))

    for uid, result in zip(uids_to_process, results):
        if result is None:
            missing_uids.append(uid)
        else:
            jobs.append(result)

    if missing_uids:
        logger.warning(
            f"{len(missing_uids)} UID(s) missing locally and will be skipped."
        )

    if not jobs:
        logger.info("No new objects to render. Exiting.")
        return

    logger.info(f"Rendering {len(jobs)} new objects using {num_workers} parallel workers.")

    # Shuffle and split into train/test
    random.seed(42)
    random.shuffle(jobs)
    split_idx = int(len(jobs) * train_split)
    train_jobs = jobs[:split_idx]
    test_jobs = jobs[split_idx:]

    logger.info(f"Split: {len(train_jobs)} train, {len(test_jobs)} test")

    # Create output directory and chunk writers
    output_path.mkdir(parents=True, exist_ok=True)
    train_writer = ChunkWriter(output_path, "train", scenes_per_chunk)
    test_writer = ChunkWriter(output_path, "test", scenes_per_chunk)

    # Torch config as dict for multiprocessing
    torch_config_dict = {
        "image_size": (256, 256),
        "include_depth": render_depth,
        "include_mask": render_mask,
        "include_normal": render_normals,
        "min_views": 4,
    }

    def process_jobs(jobs_list: List, writer: ChunkWriter, desc: str):
        """Process a list of jobs and write to chunk writer."""
        success_count = 0
        failed_count = 0

        render_args = [
            (
                uid,
                local_path,
                sha256,
                metadata,
                num_renders,
                only_northern_hemisphere,
                gpu_devices,
                render_timeout,
                render_depth,
                render_normals,
                render_mask,
                camera_radius_min,
                camera_radius_max,
                fov_min_degrees,
                fov_max_degrees,
                torch_config_dict,
            )
            for uid, local_path, sha256, metadata in jobs_list
        ]

        if num_workers > 1:
            with mp.Pool(processes=num_workers) as pool:
                with tqdm(total=len(jobs_list), desc=desc, unit="object") as pbar:
                    for scene in pool.imap_unordered(_render_worker, render_args):
                        if scene is not None:
                            writer.add_scene(scene)
                            success_count += 1
                        else:
                            failed_count += 1

                        pbar.set_postfix({
                            "success": success_count,
                            "failed": failed_count,
                            "chunks": writer.chunk_idx
                        })
                        pbar.update(1)
        else:
            with tqdm(total=len(jobs_list), desc=desc, unit="object") as pbar:
                for args in render_args:
                    scene = _render_worker(args)
                    if scene is not None:
                        writer.add_scene(scene)
                        success_count += 1
                    else:
                        failed_count += 1

                    pbar.set_postfix({
                        "success": success_count,
                        "failed": failed_count,
                        "chunks": writer.chunk_idx
                    })
                    pbar.update(1)

        return success_count, failed_count

    # Process train and test sets
    logger.info("Processing training data...")
    train_success, train_failed = process_jobs(train_jobs, train_writer, "Train")
    train_index = train_writer.finalize()
    logger.info(f"Training: {train_success} succeeded, {train_failed} failed, {len(train_index)} total scenes in {train_writer.chunk_idx} chunks")

    logger.info("Processing test data...")
    test_success, test_failed = process_jobs(test_jobs, test_writer, "Test")
    test_index = test_writer.finalize()
    logger.info(f"Test: {test_success} succeeded, {test_failed} failed, {len(test_index)} total scenes in {test_writer.chunk_idx} chunks")

    # Save conversion config
    config_dict = {
        "output_dir": str(output_path),
        "train_split": train_split,
        "scenes_per_chunk": scenes_per_chunk,
        "image_size": [256, 256],
        "include_depth": render_depth,
        "include_mask": render_mask,
        "include_normal": render_normals,
        "num_views": num_renders,
        "total_scenes": len(train_index) + len(test_index),
        "train_scenes": len(train_index),
        "test_scenes": len(test_index),
        "train_failed": train_failed,
        "test_failed": test_failed,
    }
    with open(output_path / "conversion_config.json", 'w') as f:
        json.dump(config_dict, f, indent=2)

    logger.info(f"\nConversion complete! Output saved to: {output_path}")
    logger.info(f"  Train: {output_path / 'train'} ({len(train_index)} scenes)")
    logger.info(f"  Test:  {output_path / 'test'} ({len(test_index)} scenes)")


if __name__ == "__main__":
    fire.Fire(render_objaverse_to_torch)
