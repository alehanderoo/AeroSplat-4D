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
import zipfile
from typing import Dict, List, Optional, Tuple, Union

import fire
import fsspec
import GPUtil
from loguru import logger
from tqdm import tqdm

from objaverse.utils import get_file_hash

HIGH_QUALITY_UIDS_PATH = os.path.expanduser(
    # "~/thesis/assets/objaverse/high_quality_uids.txt"
    "/home/sandro/thesis/assets/objaverse/trainingsetB_part1.txt"
)
OBJAVERSE_V1_BASE = os.path.expanduser("~/.objaverse/hf-objaverse-v1")
# OBJAVERSE_V1_BASE = os.path.expanduser("/mnt/sda/thesis/.objaverse/hf-objaverse-v1")
OBJECT_PATHS_FILE = os.path.join(OBJAVERSE_V1_BASE, "object-paths.json.gz")
MIN_FREE_BYTES = 50 * 1024**3  # 50 GiB


def log_processed_object(csv_filename: str, *args) -> None:
    """Log when an object is done being used."""

    args = ",".join([str(arg) for arg in args])
    dirname = os.path.expanduser("~/.objaverse/logs/")
    os.makedirs(dirname, exist_ok=True)
    with open(os.path.join(dirname, csv_filename), "a", encoding="utf-8") as f:
        f.write(f"{time.time()},{args}\n")


def zipdir(path: str, ziph: zipfile.ZipFile) -> None:
    """Zip up a directory with an arcname structure."""

    for root, _, files in os.walk(path):
        for file in files:
            arcname = os.path.join(os.path.basename(root), file)
            ziph.write(os.path.join(root, file), arcname=arcname)


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


def _render_worker(args: Tuple) -> bool:
    """Worker function for parallel rendering."""
    (
        uid,
        local_path,
        sha256,
        metadata,
        num_renders,
        render_dir,
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
    ) = args
    
    ensure_free_space()
    
    return render_local_object(
        uid=uid,
        local_path=local_path,
        sha256=sha256,
        metadata=metadata,
        num_renders=num_renders,
        render_dir=render_dir,
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
    )


def render_local_object(
    uid: str,
    local_path: str,
    sha256: str,
    metadata: Dict[str, str],
    num_renders: int,
    render_dir: str,
    only_northern_hemisphere: bool,
    gpu_devices: Union[int, List[int]],
    render_timeout: int,
    render_depth: bool = False,
    render_normals: bool = False,
    render_mask: bool = False,
    success_log: str = "render-objaverse1-success.csv",
    failure_log: str = "render-objaverse1-failed.csv",
    camera_radius_min: float = 1.5,
    camera_radius_max: float = 2.8,
    fov_min_degrees: float = 45.0,
    fov_max_degrees: float = 90.0,
) -> bool:
    """Renders a locally available Objaverse v1 asset."""

    save_uid = uid
    using_gpu, gpu_index = choose_device(gpu_devices)
    # Don't check disk space here since _render_worker already does it

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
            # If DISPLAY is already set, use it (e.g., for Xvfb or virtual display)
            command = f"export DISPLAY={display_override} && {command}"
        elif using_gpu and gpu_index is not None:
            # Only use :0.{gpu_index} if DISPLAY is not set
            display_value = f":0.{gpu_index}"
            command = f"export DISPLAY={display_value} && {command}"

        # Create log file
        log_dir = os.path.join(render_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"{uid}.log")

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
            log_processed_object(failure_log, uid, sha256)
            return False

        # --- Renaming Logic ---
        # The blender script outputs files as render_TYPE_XXXX.ext
        # We want to rename them to:
        # RGB: XXX.png
        # Depth: XXX_depth.exr
        # Normal: XXX_normal.exr
        # Mask: XXX_mask.png

        # Helper to rename files
        def rename_files(pattern_glob, new_format_func):
            found_files = glob.glob(os.path.join(target_directory, pattern_glob))
            renamed_map = {} # old_name -> new_name (relative)
            for fpath in found_files:
                fname = os.path.basename(fpath)
                # Assume format render_TYPE_XXXX.ext
                # We extract the digits at the end of the stem (before ext)
                # render_rgb_0000.png -> 0000
                
                # A safer way might be to split by underscore
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
        
        # RGB
        renamed_files_map.update(rename_files("render_rgb_*.png", lambda i: f"{i:03d}.png"))

        # Depth
        if render_depth:
            renamed_files_map.update(rename_files("render_depth_*.exr", lambda i: f"{i:03d}_depth.exr"))

        # Normals
        if render_normals:
             renamed_files_map.update(rename_files("render_normal_*.exr", lambda i: f"{i:03d}_normal.exr"))

        # Mask
        if render_mask:
             renamed_files_map.update(rename_files("render_mask_*.png", lambda i: f"{i:03d}_mask.png"))

        # --- Update Metadata with new filenames ---
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

             with open(metadata_path, 'w') as f:
                 json.dump(meta, f, indent=2, sort_keys=True)

        # --- Verification ---
        png_files = glob.glob(os.path.join(target_directory, "*.png"))
        # We need to filter out mask PNGs from "png_files" if we count them separately,
        # otherwise checking len(png_files) might be ambiguous if masks are also pngs.
        # But wait, glob *.png includes masks.
        # RGB files are just digits e.g. "000.png".
        # Mask files are "000_mask.png".
        
        rgb_files = [f for f in png_files if "mask" not in os.path.basename(f) and os.path.basename(f)[0].isdigit()]
        
        metadata_files = glob.glob(os.path.join(target_directory, "*.json"))
        depth_files = glob.glob(os.path.join(target_directory, "*_depth.exr"))
        normal_files = glob.glob(os.path.join(target_directory, "*_normal.exr"))
        mask_files = glob.glob(os.path.join(target_directory, "*_mask.png"))

        missing_pngs = len(rgb_files) != num_renders
        missing_depth = render_depth and len(depth_files) != num_renders
        missing_normals = render_normals and len(normal_files) != num_renders
        missing_masks = render_mask and len(mask_files) != num_renders
        missing_metadata = len(metadata_files) != 1

        if (
            missing_pngs
            or missing_depth
            or missing_normals
            or missing_masks
            or missing_metadata
        ):
            logger.error(f"Rendering failed for UID {uid}")
            log_processed_object(failure_log, uid, sha256)
            return False

        metadata_path = os.path.join(target_directory, "metadata.json")
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata_file = json.load(f)

        metadata_file["sha256"] = sha256
        metadata_file["file_identifier"] = uid
        metadata_file["save_uid"] = save_uid
        metadata_file["metadata"] = metadata

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata_file, f, indent=2, sort_keys=True)

        with zipfile.ZipFile(
            f"{target_directory}.zip", "w", zipfile.ZIP_DEFLATED
        ) as ziph:
            zipdir(target_directory, ziph)

        fs, base_path = fsspec.core.url_to_fs(render_dir)
        fs.makedirs(os.path.join(base_path, "renders"), exist_ok=True)
        fs.put(
            os.path.join(f"{target_directory}.zip"),
            os.path.join(base_path, "renders", f"{save_uid}.zip"),
        )

        log_processed_object(success_log, uid, sha256)
        return True


def render_objaverse1_objects(
    render_dir: str = "~/.objaverse",
    num_renders: int = 32,
    only_northern_hemisphere: bool = False,
    render_timeout: int = 300,
    gpu_devices: Optional[Union[int, List[int]]] = None,
    render_depth: bool = False,
    render_normals: bool = False,
    render_mask: bool = False,
    uids_path: str = HIGH_QUALITY_UIDS_PATH,
    object_paths_file: str = OBJECT_PATHS_FILE,
    num_workers: int = 2,
    num_objects: int = 20000,
    camera_radius_min: float = 1.5,
    camera_radius_max: float = 2.8,
    fov_min_degrees: float = 45.0,
    fov_max_degrees: float = 90.0,
) -> None:
    """Renders locally downloaded Objaverse 1.0 objects using Blender."""

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

    fs, base_path = fsspec.core.url_to_fs(render_dir)
    try:
        zip_files = fs.glob(os.path.join(base_path, "renders", "*.zip"), refresh=True)
    except TypeError:
        zip_files = fs.glob(os.path.join(base_path, "renders", "*.zip"))
    existing = {os.path.splitext(os.path.basename(z))[0] for z in zip_files}

    jobs: List[Tuple[str, str, str, Dict[str, str]]] = []
    missing_uids = []
    for uid in requested_uids:
        if uid in existing:
            continue
        local_path = resolve_local_path(uid, object_paths)
        if not local_path:
            missing_uids.append(uid)
            continue
        sha256 = get_file_hash(local_path)
        metadata = {
            "uid": uid,
            "source": "objaverse_v1",
            "relative_path": os.path.relpath(local_path, OBJAVERSE_V1_BASE),
        }
        jobs.append((uid, local_path, sha256, metadata))

    if missing_uids:
        logger.warning(
            f"{len(missing_uids)} UID(s) missing locally and will be skipped."
        )

    logger.info(f"Rendering {len(jobs)} new objects using {num_workers} parallel workers.")

    # Prepare arguments for parallel processing
    render_args = [
        (
            uid,
            local_path,
            sha256,
            metadata,
            num_renders,
            render_dir,
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
        )
        for uid, local_path, sha256, metadata in jobs
    ]

    success_count = 0
    failed_count = 0
    
    # Use multiprocessing pool to render in parallel
    with mp.Pool(processes=num_workers) as pool:
        with tqdm(total=len(jobs), desc="Rendering progress", unit="object") as pbar:
            for success in pool.imap_unordered(_render_worker, render_args):
                if success:
                    success_count += 1
                else:
                    failed_count += 1
                
                pbar.set_postfix({"success": success_count, "failed": failed_count})
                pbar.update(1)
    
    logger.info(f"Rendering complete: {success_count} succeeded, {failed_count} failed.")


if __name__ == "__main__":
    fire.Fire(render_objaverse1_objects)
