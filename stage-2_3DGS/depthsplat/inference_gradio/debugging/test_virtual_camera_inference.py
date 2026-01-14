#!/usr/bin/env python3
"""
Test Virtual Camera Inference for In-the-Wild Frames.

Runs inference with both:
1. Current approach (original extrinsics, overridden intrinsics)
2. Virtual camera approach (adjusted extrinsics + consistent intrinsics)

Compares reconstruction quality.
"""

import sys
import os
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch

# Add parent directories to path
SCRIPT_DIR = Path(__file__).parent
INFERENCE_DIR = SCRIPT_DIR.parent
DEPTHSPLAT_ROOT = INFERENCE_DIR.parent
sys.path.insert(0, str(INFERENCE_DIR))
sys.path.insert(0, str(DEPTHSPLAT_ROOT))

from runner import DepthSplatRunner
from services.gt_detection_service import create_gt_detection_service
from camera_utils import normalize_intrinsics

# Constants
TRAINING_FX_NORM = 1.0723  # 50° FOV
TARGET_OBJECT_COVERAGE = 0.75


def compute_object_physical_size(bbox_pixels, fx_pixels, distance_meters):
    """Compute object physical size from projection."""
    return bbox_pixels * distance_meters / fx_pixels


def compute_virtual_distance(object_size_meters, target_coverage, fx_norm):
    """Compute virtual camera distance for target object coverage."""
    return fx_norm * object_size_meters / target_coverage


def create_virtual_extrinsics(
    original_c2w: np.ndarray,
    object_position: np.ndarray,
    virtual_distance: float
) -> np.ndarray:
    """Create virtual camera extrinsics at closer distance."""
    cam_pos_orig = original_c2w[:3, 3].copy()
    direction = cam_pos_orig - object_position
    direction = direction / np.linalg.norm(direction)
    cam_pos_virtual = object_position + direction * virtual_distance

    world_up = np.array([0, 0, 1], dtype=np.float32)
    forward = object_position - cam_pos_virtual
    forward = forward / np.linalg.norm(forward)

    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-6:
        world_up = np.array([0, 1, 0], dtype=np.float32)
        right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)

    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)

    down = -up
    R = np.stack([right, down, forward], axis=1).astype(np.float32)

    c2w_virtual = np.eye(4, dtype=np.float32)
    c2w_virtual[:3, :3] = R
    c2w_virtual[:3, 3] = cam_pos_virtual

    return c2w_virtual


def load_wild_frame_with_virtual_cameras(
    runner: DepthSplatRunner,
    frame_id: int,
    render_dir: str,
    cache_dir: str,
    use_virtual_cameras: bool = True
):
    """
    Load wild frame data with virtual camera support.

    Returns images, extrinsics, intrinsics ready for inference.
    """
    service = create_gt_detection_service(render_dir)
    detections = service.get_detections(frame_id)

    if not detections:
        raise ValueError(f"No detections for frame {frame_id}")

    object_position = np.array(detections.object_position_3d, dtype=np.float32)

    CROP_SIZE = 256
    image_paths = []
    images = []
    extrinsics_list = []
    intrinsics_list = []

    for cam_name in sorted(service.camera_names):
        img_path = Path(render_dir) / cam_name / "rgb" / f"rgb_{frame_id:04d}.png"
        if not img_path.exists():
            continue

        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        det = detections.get_detection(cam_name)
        if not det:
            continue

        # Get crop region
        crop = det.get_crop_region(crop_size=CROP_SIZE, image_width=w, image_height=h)
        x1, y1, x2, y2 = crop

        # Crop image
        img_crop = img.crop(crop)

        # Apply mask for white background
        mask_path = Path(render_dir) / cam_name / "mask" / f"drone_mask_{frame_id:04d}.png"
        if mask_path.exists():
            mask = Image.open(mask_path).convert("L")
            mask_crop = mask.crop(crop)
            img_np = np.array(img_crop)
            mask_np = np.array(mask_crop)
            white_bg = np.ones_like(img_np) * 255
            fg_mask = (mask_np > 127)[..., np.newaxis]
            img_np = np.where(fg_mask, img_np, white_bg)
            img_crop = Image.fromarray(img_np.astype(np.uint8))

        # Save and store image
        save_path = os.path.join(cache_dir, f"{cam_name}_frame{frame_id}.png")
        img_crop.save(save_path)
        image_paths.append(save_path)
        images.append(np.array(img_crop))

        # Get camera intrinsics
        intr_raw = service.get_camera_intrinsics(cam_name)
        if isinstance(intr_raw, dict):
            fx = intr_raw['fx']
            fy = intr_raw['fy']
        else:
            K_raw = np.array(intr_raw)
            fx, fy = K_raw[0, 0], K_raw[1, 1]

        # Get extrinsics
        ext_raw = service.get_camera_extrinsics(cam_name)
        ext_matrix = np.array(ext_raw, dtype=np.float32)

        # Apply OpenGL->OpenCV flip
        flip_mat = np.diag([1.0, -1.0, -1.0]).astype(np.float32)
        ext_matrix[:3, :3] = ext_matrix[:3, :3] @ flip_mat

        if use_virtual_cameras:
            # Compute virtual camera parameters
            bbox = det.bbox
            if bbox:
                bbox_size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
            else:
                bbox_size = 68  # Default

            depth = det.depth
            object_size = compute_object_physical_size(bbox_size, fx, depth)
            virtual_distance = compute_virtual_distance(
                object_size, TARGET_OBJECT_COVERAGE, TRAINING_FX_NORM
            )

            # Create virtual extrinsics
            ext_matrix = create_virtual_extrinsics(
                ext_matrix, object_position, virtual_distance
            )

        extrinsics_list.append(ext_matrix)

        # Use training-matched intrinsics
        training_intrinsics = np.array([
            [TRAINING_FX_NORM, 0, 0.5],
            [0, TRAINING_FX_NORM, 0.5],
            [0, 0, 1],
        ], dtype=np.float32)
        intrinsics_list.append(training_intrinsics)

    # Pose normalization
    extrinsics_array = np.stack(extrinsics_list)

    # Center on object
    for ext in extrinsics_array:
        ext[:3, 3] -= object_position

    # Scale to target radius
    distances = [np.linalg.norm(ext[:3, 3]) for ext in extrinsics_array]
    mean_dist = np.mean(distances)
    target_radius = 2.0
    scale_factor = target_radius / mean_dist

    print(f"  Mean distance before scaling: {mean_dist:.3f}m")
    print(f"  Scale factor: {scale_factor:.4f}")

    for ext in extrinsics_array:
        ext[:3, 3] *= scale_factor

    # Calculate mean elevation for render camera
    positions = extrinsics_array[:, :3, 3]
    distances = np.linalg.norm(positions, axis=1)
    elevations = np.rad2deg(np.arcsin(positions[:, 2] / distances))
    mean_elevation = float(np.mean(elevations))

    return {
        'images': images,
        'image_paths': image_paths,
        'extrinsics': extrinsics_array,
        'intrinsics': np.stack(intrinsics_list),
        'scale_factor': scale_factor,
        'object_position': object_position,
        'mean_elevation': mean_elevation,
    }


def create_render_camera(azimuth: float, elevation: float, distance: float = 2.0) -> np.ndarray:
    """Create a render camera at given spherical coordinates."""
    azimuth_rad = np.deg2rad(azimuth)
    elevation_rad = np.deg2rad(elevation)

    x = distance * np.cos(elevation_rad) * np.cos(azimuth_rad)
    y = distance * np.cos(elevation_rad) * np.sin(azimuth_rad)
    z = distance * np.sin(elevation_rad)

    cam_pos = np.array([x, y, z], dtype=np.float32)

    forward = -cam_pos / np.linalg.norm(cam_pos)
    world_up = np.array([0, 0, 1], dtype=np.float32)

    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-6:
        world_up = np.array([0, 1, 0], dtype=np.float32)
        right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)

    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)

    down = -up
    R = np.stack([right, down, forward], axis=1).astype(np.float32)

    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = R
    c2w[:3, 3] = cam_pos

    return c2w[np.newaxis, ...]  # [1, 4, 4]


def run_inference_comparison(
    runner: DepthSplatRunner,
    render_dir: str,
    frame_id: int,
    cache_dir: str,
    output_dir: str
):
    """Run inference with both approaches and compare."""

    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*60)
    print("LOADING CURRENT APPROACH (original extrinsics)")
    print("="*60)

    # Load with current approach (original extrinsics)
    data_current = load_wild_frame_with_virtual_cameras(
        runner, frame_id, render_dir, cache_dir,
        use_virtual_cameras=False
    )

    print("\n" + "="*60)
    print("LOADING VIRTUAL CAMERA APPROACH")
    print("="*60)

    # Load with virtual cameras
    data_virtual = load_wild_frame_with_virtual_cameras(
        runner, frame_id, render_dir, cache_dir,
        use_virtual_cameras=True
    )

    # Test rendering from multiple viewpoints
    test_viewpoints = [
        (0, 30, "Az=0, El=30"),
        (0, 0, "Az=0, El=0"),
        (0, -30, "Az=0, El=-30"),
        (0, -60, "Az=0, El=-60"),
        (90, 30, "Az=90, El=30"),
    ]

    print("\n" + "="*60)
    print("RUNNING INFERENCE COMPARISON")
    print("="*60)

    results = []

    for azimuth, elevation, name in test_viewpoints:
        print(f"\nRendering from {name}...")

        target_extrinsics = create_render_camera(azimuth, elevation)

        # Current approach
        result_current = runner.run_inference(
            images=data_current['images'],
            extrinsics=data_current['extrinsics'],
            intrinsics=data_current['intrinsics'],
            target_extrinsics=target_extrinsics,
            output_dir=os.path.join(output_dir, f"current_{name.replace('=', '').replace(', ', '_').replace(' ', '')}"),
            num_video_frames=0,
        )

        # Virtual camera approach
        result_virtual = runner.run_inference(
            images=data_virtual['images'],
            extrinsics=data_virtual['extrinsics'],
            intrinsics=data_virtual['intrinsics'],
            target_extrinsics=target_extrinsics,
            output_dir=os.path.join(output_dir, f"virtual_{name.replace('=', '').replace(', ', '_').replace(' ', '')}"),
            num_video_frames=0,
        )

        # Analyze results
        img_current = Image.open(result_current['result_image_path'])
        img_virtual = Image.open(result_virtual['result_image_path'])

        arr_current = np.array(img_current)
        arr_virtual = np.array(img_virtual)

        # Compute non-white pixels (object content)
        current_nonwhite = np.mean(arr_current.mean(axis=2) < 250) * 100
        virtual_nonwhite = np.mean(arr_virtual.mean(axis=2) < 250) * 100

        print(f"  Current: {current_nonwhite:.1f}% non-white")
        print(f"  Virtual: {virtual_nonwhite:.1f}% non-white")

        results.append({
            'name': name,
            'current_path': result_current['result_image_path'],
            'virtual_path': result_virtual['result_image_path'],
            'current_nonwhite': current_nonwhite,
            'virtual_nonwhite': virtual_nonwhite,
        })

    # Create comparison grid
    fig, axes = plt.subplots(len(test_viewpoints), 2, figsize=(10, 4*len(test_viewpoints)))

    for i, result in enumerate(results):
        img_current = Image.open(result['current_path'])
        img_virtual = Image.open(result['virtual_path'])

        axes[i, 0].imshow(img_current)
        axes[i, 0].set_title(f"Current: {result['name']}\n{result['current_nonwhite']:.1f}% content")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(img_virtual)
        axes[i, 1].set_title(f"Virtual Camera: {result['name']}\n{result['virtual_nonwhite']:.1f}% content")
        axes[i, 1].axis('off')

    plt.suptitle("Current Approach vs Virtual Camera Approach", fontsize=14)
    plt.tight_layout()
    comparison_path = os.path.join(output_dir, "comparison_grid.png")
    plt.savefig(comparison_path, dpi=150)
    plt.close()

    print(f"\nSaved comparison grid to: {comparison_path}")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\n{'Viewpoint':<20} {'Current':<15} {'Virtual':<15} {'Change':<15}")
    print("-"*65)
    for result in results:
        change = result['virtual_nonwhite'] - result['current_nonwhite']
        sign = "+" if change > 0 else ""
        print(f"{result['name']:<20} {result['current_nonwhite']:.1f}%{'':<10} "
              f"{result['virtual_nonwhite']:.1f}%{'':<10} {sign}{change:.1f}%")

    return results


def main():
    render_dir = "/home/sandro/thesis/renders/5cams_10-01-26_bird_10m"
    frame_id = 60
    cache_dir = "/tmp/depthsplat_virtual_test"
    output_dir = str(SCRIPT_DIR / "imgs" / "virtual_camera_inference")

    os.makedirs(cache_dir, exist_ok=True)

    print("="*60)
    print("VIRTUAL CAMERA INFERENCE TEST")
    print("="*60)

    # Initialize runner
    checkpoint_path = "/home/sandro/thesis/code/depthsplat/outputs/objaverse_white_small_gauss/checkpoints/epoch_0-step_100000.ckpt"

    print("Initializing Runner...")
    runner = DepthSplatRunner(
        checkpoint_path=checkpoint_path,
        config_name="objaverse_white_small_gauss",
        device="cuda",
    )

    # Run comparison
    results = run_inference_comparison(
        runner, render_dir, frame_id, cache_dir, output_dir
    )


if __name__ == "__main__":
    main()
