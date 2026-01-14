#!/usr/bin/env python3
"""
Test Virtual Camera Approach for In-the-Wild Frames.

This script explores the virtual camera approach to solve the FOV/distance mismatch
between wild frames (telephoto lens, far away) and training data (50° FOV, close).

The key insight:
- Wild cameras: far (11m), high fx (1946px), narrow FOV (~7.5°), object is ~68px in full frame
- Training: close (1.4m), fx=274 for 256x256, 50° FOV, object fills most of frame

Virtual camera solution:
1. Compute object physical size from original camera params
2. Calculate virtual distance where object fills 80% of frame with 50° FOV
3. Create new extrinsics at virtual distance (same viewing direction)
4. Crop tightly around object using mask
5. Use training-matched intrinsics (50° FOV)
"""

import sys
import os
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import json

# Add parent directories to path
SCRIPT_DIR = Path(__file__).parent
INFERENCE_DIR = SCRIPT_DIR.parent
DEPTHSPLAT_ROOT = INFERENCE_DIR.parent
sys.path.insert(0, str(INFERENCE_DIR))
sys.path.insert(0, str(DEPTHSPLAT_ROOT))

from services.gt_detection_service import create_gt_detection_service
from camera_utils import normalize_intrinsics, look_at

# Constants
TRAINING_FX_NORM = 1.0723  # 50° FOV normalized focal length
TRAINING_FOV_DEG = 50.0
TARGET_OBJECT_COVERAGE = 0.75  # Object should fill 75% of the frame


def compute_object_physical_size(
    bbox_pixels: float,
    fx_pixels: float,
    distance_meters: float
) -> float:
    """
    Compute object physical size from projection.

    For pinhole camera: proj_size = fx * (real_size / distance)
    Therefore: real_size = proj_size * distance / fx
    """
    return bbox_pixels * distance_meters / fx_pixels


def compute_virtual_distance(
    object_size_meters: float,
    target_coverage: float,
    crop_size: int,
    fx_norm: float
) -> float:
    """
    Compute virtual camera distance for target object coverage.

    Target: object fills target_coverage fraction of crop_size.
    target_proj_pixels = target_coverage * crop_size

    For pinhole: target_proj_pixels = fx_pixels * (object_size / virtual_distance)
    fx_pixels = fx_norm * crop_size

    Therefore: virtual_distance = fx_norm * crop_size * object_size / target_proj_pixels
                                = fx_norm * object_size / target_coverage
    """
    return fx_norm * object_size_meters / target_coverage


def create_virtual_extrinsics(
    original_c2w: np.ndarray,
    object_position: np.ndarray,
    virtual_distance: float
) -> np.ndarray:
    """
    Create virtual camera extrinsics at closer distance.

    Maintains the same viewing direction (line from camera to object)
    but positions camera at virtual_distance from object.
    """
    # Extract original camera position
    cam_pos_orig = original_c2w[:3, 3].copy()

    # Direction from object to camera (viewing direction is opposite)
    direction = cam_pos_orig - object_position
    direction = direction / np.linalg.norm(direction)

    # New camera position at virtual distance
    cam_pos_virtual = object_position + direction * virtual_distance

    # Create new extrinsics looking at the object
    # Use world up vector
    world_up = np.array([0, 0, 1], dtype=np.float32)

    # Forward direction (camera looks along +Z in OpenCV)
    forward = object_position - cam_pos_virtual
    forward = forward / np.linalg.norm(forward)

    # Right direction
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-6:
        # Camera looking straight up/down
        world_up = np.array([0, 1, 0], dtype=np.float32)
        right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)

    # Recompute up
    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)

    # Build rotation matrix - OpenCV convention (Y down)
    down = -up
    R = np.stack([right, down, forward], axis=1).astype(np.float32)

    # Build 4x4 matrix
    c2w_virtual = np.eye(4, dtype=np.float32)
    c2w_virtual[:3, :3] = R
    c2w_virtual[:3, 3] = cam_pos_virtual

    return c2w_virtual


def load_frame_data(render_dir: str, frame_id: int):
    """Load frame data including detections and camera info."""
    service = create_gt_detection_service(render_dir)
    detections = service.get_detections(frame_id)

    if not detections:
        raise ValueError(f"No detections for frame {frame_id}")

    frame_data = {
        'object_position_3d': np.array(detections.object_position_3d, dtype=np.float32),
        'cameras': {}
    }

    for cam_name in sorted(service.camera_names):
        det = detections.get_detection(cam_name)
        if det is None:
            continue

        intr_raw = service.get_camera_intrinsics(cam_name)
        ext_raw = service.get_camera_extrinsics(cam_name)
        resolution = service.get_camera_resolution(cam_name)

        if intr_raw is None or ext_raw is None:
            continue

        # Parse intrinsics
        if isinstance(intr_raw, dict):
            fx = intr_raw['fx']
            fy = intr_raw['fy']
            cx = intr_raw['cx']
            cy = intr_raw['cy']
        else:
            K_raw = np.array(intr_raw)
            fx, fy = K_raw[0, 0], K_raw[1, 1]
            cx, cy = K_raw[0, 2], K_raw[1, 2]

        # Get bbox
        bbox = det.bbox if det.bbox else None
        if bbox:
            bbox_width = bbox[2] - bbox[0]
            bbox_height = bbox[3] - bbox[1]
            bbox_size = max(bbox_width, bbox_height)
        else:
            bbox_size = 68  # Default

        frame_data['cameras'][cam_name] = {
            'center_2d': det.center_2d,
            'depth': det.depth,
            'bbox': bbox,
            'bbox_size': bbox_size,
            'fx': fx,
            'fy': fy,
            'cx': cx,
            'cy': cy,
            'resolution': resolution,
            'extrinsics': np.array(ext_raw, dtype=np.float32),
        }

    return frame_data, service


def analyze_virtual_camera_parameters(frame_data: dict):
    """Analyze and compute virtual camera parameters for each camera."""
    object_pos = frame_data['object_position_3d']
    print(f"\nObject 3D position: {object_pos}")

    results = {}

    for cam_name, cam_data in frame_data['cameras'].items():
        print(f"\n{'='*60}")
        print(f"Camera: {cam_name}")
        print(f"{'='*60}")

        # Original camera parameters
        fx = cam_data['fx']
        depth = cam_data['depth']
        bbox_size = cam_data['bbox_size']

        print(f"\nOriginal camera:")
        print(f"  Distance to object: {depth:.2f}m")
        print(f"  Focal length: {fx:.1f}px")
        print(f"  Object bbox size: {bbox_size:.1f}px")

        # Compute object physical size
        object_size = compute_object_physical_size(bbox_size, fx, depth)
        print(f"  Estimated object size: {object_size:.3f}m ({object_size*100:.1f}cm)")

        # Compute virtual distance for target coverage
        crop_size = 256
        virtual_distance = compute_virtual_distance(
            object_size, TARGET_OBJECT_COVERAGE, crop_size, TRAINING_FX_NORM
        )

        print(f"\nVirtual camera (for {TARGET_OBJECT_COVERAGE*100:.0f}% object coverage):")
        print(f"  Virtual distance: {virtual_distance:.3f}m")
        print(f"  Distance ratio: {depth / virtual_distance:.1f}x closer")

        # Compute virtual extrinsics
        ext_orig = cam_data['extrinsics'].copy()

        # Apply OpenGL->OpenCV flip (same as load_wild_frame)
        flip_mat = np.diag([1.0, -1.0, -1.0]).astype(np.float32)
        ext_orig[:3, :3] = ext_orig[:3, :3] @ flip_mat

        ext_virtual = create_virtual_extrinsics(ext_orig, object_pos, virtual_distance)

        # Verify distance
        cam_pos_orig = ext_orig[:3, 3]
        cam_pos_virtual = ext_virtual[:3, 3]
        dist_orig = np.linalg.norm(cam_pos_orig - object_pos)
        dist_virtual = np.linalg.norm(cam_pos_virtual - object_pos)

        print(f"\nExtrinsics verification:")
        print(f"  Original cam position: {cam_pos_orig}")
        print(f"  Virtual cam position: {cam_pos_virtual}")
        print(f"  Original distance to object: {dist_orig:.3f}m")
        print(f"  Virtual distance to object: {dist_virtual:.3f}m")

        # Check viewing direction
        forward_orig = ext_orig[:3, 2]  # Z column is forward direction
        forward_virtual = ext_virtual[:3, 2]
        direction_dot = np.dot(forward_orig, forward_virtual)
        print(f"  Forward direction alignment: {direction_dot:.4f} (1.0 = same direction)")

        results[cam_name] = {
            'object_size': object_size,
            'original_distance': depth,
            'virtual_distance': virtual_distance,
            'extrinsics_original': ext_orig,
            'extrinsics_virtual': ext_virtual,
        }

    return results


def visualize_camera_positions_2d(
    results: dict, 
    object_pos: np.ndarray, 
    output_path: str
):
    """
    Visualize original vs virtual camera positions in 2D top-view (X-Y plane).
    
    Shows:
    - Original camera positions (circles)
    - Virtual camera positions (triangles)
    - Target/object position (red cross)
    - Center lines from cameras to target
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot object/target as red cross
    ax.scatter([object_pos[0]], [object_pos[1]], 
               c='red', s=300, marker='x', linewidths=3, label='Target', zorder=10)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    
    for i, (cam_name, data) in enumerate(results.items()):
        ext_orig = data['extrinsics_original']
        ext_virtual = data['extrinsics_virtual']
        
        pos_orig = ext_orig[:3, 3]
        pos_virtual = ext_virtual[:3, 3]
        
        # Plot original camera
        ax.scatter([pos_orig[0]], [pos_orig[1]], 
                   c=[colors[i]], s=100, marker='o', alpha=0.5, edgecolors='black')
        
        # Plot virtual camera
        ax.scatter([pos_virtual[0]], [pos_virtual[1]], 
                   c=[colors[i]], s=100, marker='^', label=cam_name, edgecolors='black')
        
        # Draw line from original to virtual (dashed)
        ax.plot([pos_orig[0], pos_virtual[0]], 
                [pos_orig[1], pos_virtual[1]], 
                c=colors[i], linestyle='--', alpha=0.3, linewidth=1)
        
        # Draw center line from original camera to target (dotted, thin)
        ax.plot([pos_orig[0], object_pos[0]], 
                [pos_orig[1], object_pos[1]], 
                c=colors[i], linestyle=':', alpha=0.4, linewidth=1)
        
        # Draw center line from virtual camera to target (solid, thick)
        ax.plot([pos_virtual[0], object_pos[0]], 
                [pos_virtual[1], object_pos[1]], 
                c=colors[i], linestyle='-', alpha=0.8, linewidth=2)
        
        # Add camera facing direction arrow from virtual camera
        forward = ext_virtual[:3, 2]  # Z column is forward direction
        arrow_length = data['virtual_distance'] * 0.3
        ax.annotate('', 
                    xy=(pos_virtual[0] + forward[0]*arrow_length, 
                        pos_virtual[1] + forward[1]*arrow_length),
                    xytext=(pos_virtual[0], pos_virtual[1]),
                    arrowprops=dict(arrowstyle='->', color=colors[i], lw=2))
    
    ax.set_xlabel('X (meters)', fontsize=12)
    ax.set_ylabel('Y (meters)', fontsize=12)
    ax.set_title('Top View: Original (○) vs Virtual (△) Camera Positions\n'
                 'Solid lines: virtual camera center lines | Dotted: original camera center lines', 
                 fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved 2D visualization to: {output_path}")
    plt.close()


def draw_camera_frustum_3d(
    ax, 
    extrinsics: np.ndarray, 
    distance: float,
    fov_deg: float,
    color: str,
    label: str,
    alpha: float = 0.3,
    aspect_ratio: float = 1.0
):
    """
    Draw a camera frustum in 3D.
    
    Args:
        ax: matplotlib 3D axis
        extrinsics: [4, 4] camera-to-world matrix
        distance: Distance to draw the frustum near/far plane
        fov_deg: Field of view in degrees
        color: Color for the frustum
        label: Label for the camera
        alpha: Transparency
        aspect_ratio: Aspect ratio (width/height)
    """
    cam_pos = extrinsics[:3, 3]
    R = extrinsics[:3, :3]
    
    # Camera axes
    right = R[:, 0]    # X-axis
    down = R[:, 1]     # Y-axis (OpenCV down)
    forward = R[:, 2]  # Z-axis (forward)
    
    # Compute frustum corners at the given distance
    half_angle = np.radians(fov_deg / 2)
    half_height = distance * np.tan(half_angle)
    half_width = half_height * aspect_ratio
    
    # Frustum far plane corners
    center_far = cam_pos + forward * distance
    corners = [
        center_far + right * half_width + down * half_height,  # Top-left
        center_far - right * half_width + down * half_height,  # Top-right
        center_far - right * half_width - down * half_height,  # Bottom-right
        center_far + right * half_width - down * half_height,  # Bottom-left
    ]
    
    # Draw camera position
    ax.scatter([cam_pos[0]], [cam_pos[1]], [cam_pos[2]], 
               c=color, s=100, marker='o', label=label)
    
    # Draw frustum edges from camera to corners
    for corner in corners:
        ax.plot([cam_pos[0], corner[0]], 
                [cam_pos[1], corner[1]], 
                [cam_pos[2], corner[2]], 
                c=color, alpha=alpha, linewidth=1.5)
    
    # Draw far plane rectangle
    for i in range(4):
        next_i = (i + 1) % 4
        ax.plot([corners[i][0], corners[next_i][0]], 
                [corners[i][1], corners[next_i][1]], 
                [corners[i][2], corners[next_i][2]], 
                c=color, alpha=alpha, linewidth=1.5)
    
    # Draw camera forward axis
    axis_length = distance * 0.2
    ax.quiver(cam_pos[0], cam_pos[1], cam_pos[2],
              forward[0], forward[1], forward[2],
              length=axis_length, color=color, arrow_length_ratio=0.3, linewidth=2)


def visualize_single_camera_comparison(
    cam_name: str,
    cam_data: dict,
    result_data: dict,
    object_pos: np.ndarray,
    service,
    frame_id: int,
    output_dir: str
):
    """
    Create a comprehensive comparison visualization for a single camera.
    
    Shows:
    - Left: 3D plot with original and virtual camera frustrums + target
    - Right top: Original camera view (image from dataset)
    - Right bottom: Virtual camera view (cropped image)
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    # Create figure with GridSpec for layout
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1], hspace=0.3, wspace=0.2)
    
    # 3D frustum comparison (left side, spans both rows)
    ax_3d = fig.add_subplot(gs[:, 0], projection='3d')
    
    ext_orig = result_data['extrinsics_original']
    ext_virtual = result_data['extrinsics_virtual']
    orig_distance = result_data['original_distance']
    virtual_distance = result_data['virtual_distance']
    
    # Draw target as red cross
    ax_3d.scatter([object_pos[0]], [object_pos[1]], [object_pos[2]], 
                  c='red', s=300, marker='x', linewidths=3, label='Target', zorder=10)
    
    # Estimate original FOV from focal length
    resolution = cam_data.get('resolution', (1080, 1920))
    fx = cam_data['fx']
    fov_orig = 2 * np.degrees(np.arctan(resolution[1] / (2 * fx)))
    
    # Draw original camera frustum (blue)
    draw_camera_frustum_3d(
        ax_3d, ext_orig, 
        distance=min(orig_distance, 3.0),  # Limit frustum length for visibility
        fov_deg=fov_orig,
        color='blue',
        label=f'Original Camera\n(dist={orig_distance:.1f}m, FOV={fov_orig:.1f}°)',
        alpha=0.4
    )
    
    # Draw virtual camera frustum (green)
    draw_camera_frustum_3d(
        ax_3d, ext_virtual,
        distance=virtual_distance,
        fov_deg=TRAINING_FOV_DEG,
        color='green',
        label=f'Virtual Camera\n(dist={virtual_distance:.2f}m, FOV={TRAINING_FOV_DEG:.0f}°)',
        alpha=0.6
    )
    
    ax_3d.set_xlabel('X (m)')
    ax_3d.set_ylabel('Y (m)')
    ax_3d.set_zlabel('Z (m)')
    ax_3d.set_title(f'Camera Frustum Comparison: {cam_name}', fontsize=14)
    ax_3d.legend(loc='upper left', fontsize=9)
    
    # Set axis limits to include both original and virtual cameras
    orig_cam_pos = ext_orig[:3, 3]
    virtual_cam_pos = ext_virtual[:3, 3]
    
    # Compute bounds that include object, original cam, and virtual cam
    all_points = np.array([object_pos, orig_cam_pos, virtual_cam_pos])
    mins = all_points.min(axis=0)
    maxs = all_points.max(axis=0)
    center = (mins + maxs) / 2
    span = (maxs - mins).max() / 2 + 1.0  # Add margin
    
    ax_3d.set_xlim([center[0] - span, center[0] + span])
    ax_3d.set_ylim([center[1] - span, center[1] + span])
    ax_3d.set_zlim([center[2] - span, center[2] + span])
    
    # Load and display original camera image (top right)
    ax_orig = fig.add_subplot(gs[0, 1])
    
    try:
        # Get image path from service
        image_path = service.get_image_path(cam_name, frame_id)
        if image_path and Path(image_path).exists():
            img_orig = Image.open(image_path)
            ax_orig.imshow(img_orig)
            
            # Draw bbox if available
            bbox = cam_data.get('bbox')
            if bbox:
                x1, y1, x2, y2 = bbox
                rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                     fill=False, edgecolor='red', linewidth=2)
                ax_orig.add_patch(rect)
                
                # Draw crosshair at object center
                center = cam_data.get('center_2d')
                if center:
                    ax_orig.axhline(y=center[1], color='red', linestyle='--', alpha=0.5)
                    ax_orig.axvline(x=center[0], color='red', linestyle='--', alpha=0.5)
        else:
            ax_orig.text(0.5, 0.5, 'Image not found', ha='center', va='center')
    except Exception as e:
        ax_orig.text(0.5, 0.5, f'Error loading image:\n{e}', ha='center', va='center')
    
    ax_orig.set_title(f'Original Camera View\n(Full frame, {fov_orig:.1f}° FOV)', fontsize=12)
    ax_orig.axis('off')
    
    # Create cropped view for virtual camera (bottom right)
    ax_virtual = fig.add_subplot(gs[1, 1])
    
    try:
        if image_path and Path(image_path).exists():
            img = Image.open(image_path)
            bbox = cam_data.get('bbox')
            if bbox:
                # Expand bbox to square and add margin
                x1, y1, x2, y2 = bbox
                w, h = x2 - x1, y2 - y1
                size = max(w, h)
                cx, cy = cam_data['center_2d']
                
                # Add 30% margin
                margin_factor = 1.3
                size = int(size * margin_factor)
                
                # Calculate crop bounds
                crop_x1 = max(0, int(cx - size/2))
                crop_y1 = max(0, int(cy - size/2))
                crop_x2 = min(img.width, int(cx + size/2))
                crop_y2 = min(img.height, int(cy + size/2))
                
                # Crop and resize to 256x256
                img_cropped = img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
                img_cropped = img_cropped.resize((256, 256), Image.LANCZOS)
                ax_virtual.imshow(img_cropped)
            else:
                ax_virtual.text(0.5, 0.5, 'No bbox available', ha='center', va='center')
        else:
            ax_virtual.text(0.5, 0.5, 'Image not found', ha='center', va='center')
    except Exception as e:
        ax_virtual.text(0.5, 0.5, f'Error creating crop:\n{e}', ha='center', va='center')
    
    ax_virtual.set_title(f'Virtual Camera View\n(Cropped 256×256, {TRAINING_FOV_DEG:.0f}° FOV)', fontsize=12)
    ax_virtual.axis('off')
    
    # Add text with camera parameters
    param_text = (
        f"Object size: {result_data['object_size']*100:.1f} cm\n"
        f"Distance ratio: {orig_distance/virtual_distance:.1f}× closer\n"
        f"FOV change: {fov_orig:.1f}° → {TRAINING_FOV_DEG:.0f}°"
    )
    fig.text(0.75, 0.02, param_text, fontsize=10, ha='center', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(f'Virtual Camera Transformation: {cam_name}', fontsize=16, fontweight='bold')
    
    output_path = Path(output_dir) / f"camera_comparison_{cam_name}.png"
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    print(f"Saved single camera comparison to: {output_path}")
    plt.close()


def visualize_camera_positions(results: dict, object_pos: np.ndarray, output_path: str):
    """Visualize original vs virtual camera positions in 3D."""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot object
    ax.scatter([object_pos[0]], [object_pos[1]], [object_pos[2]],
               c='red', s=200, marker='*', label='Object')

    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    for i, (cam_name, data) in enumerate(results.items()):
        ext_orig = data['extrinsics_original']
        ext_virtual = data['extrinsics_virtual']

        pos_orig = ext_orig[:3, 3]
        pos_virtual = ext_virtual[:3, 3]

        # Plot original camera
        ax.scatter([pos_orig[0]], [pos_orig[1]], [pos_orig[2]],
                   c=[colors[i]], s=100, marker='o', alpha=0.5)

        # Plot virtual camera
        ax.scatter([pos_virtual[0]], [pos_virtual[1]], [pos_virtual[2]],
                   c=[colors[i]], s=100, marker='^', label=cam_name)

        # Draw line from original to virtual
        ax.plot([pos_orig[0], pos_virtual[0]],
                [pos_orig[1], pos_virtual[1]],
                [pos_orig[2], pos_virtual[2]],
                c=colors[i], linestyle='--', alpha=0.5)

        # Draw viewing direction from virtual camera
        forward = ext_virtual[:3, 2]
        line_length = data['virtual_distance']
        end_point = pos_virtual + forward * line_length
        ax.plot([pos_virtual[0], end_point[0]],
                [pos_virtual[1], end_point[1]],
                [pos_virtual[2], end_point[2]],
                c=colors[i], linestyle='-', linewidth=2)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Original (circles) vs Virtual (triangles) Camera Positions')
    ax.legend()

    # Equal aspect ratio
    max_range = max([
        max(abs(d['extrinsics_original'][:3, 3]).max(),
            abs(d['extrinsics_virtual'][:3, 3]).max())
        for d in results.values()
    ])
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([0, 2*max_range])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nSaved visualization to: {output_path}")
    plt.close()


def test_normalized_virtual_cameras(results: dict, object_pos: np.ndarray):
    """
    Test what happens after pose normalization (centering and scaling).

    The current pipeline:
    1. Centers object at origin (subtracts object_pos from camera positions)
    2. Scales so mean camera distance = 2.0

    Virtual cameras need to follow the same pipeline but maintain their
    relative positioning advantage.
    """
    print("\n" + "="*60)
    print("POSE NORMALIZATION ANALYSIS")
    print("="*60)

    # Step 1: Apply centering
    orig_positions = []
    virtual_positions = []

    for cam_name, data in results.items():
        ext_orig = data['extrinsics_original'].copy()
        ext_virtual = data['extrinsics_virtual'].copy()

        # Center on object
        ext_orig[:3, 3] -= object_pos
        ext_virtual[:3, 3] -= object_pos

        orig_positions.append(ext_orig[:3, 3])
        virtual_positions.append(ext_virtual[:3, 3])

    orig_positions = np.array(orig_positions)
    virtual_positions = np.array(virtual_positions)

    # Step 2: Calculate scaling
    orig_distances = np.linalg.norm(orig_positions, axis=1)
    virtual_distances = np.linalg.norm(virtual_positions, axis=1)

    target_radius = 2.0
    orig_scale = target_radius / np.mean(orig_distances)
    virtual_scale = target_radius / np.mean(virtual_distances)

    print(f"\nOriginal cameras:")
    print(f"  Mean distance (centered): {np.mean(orig_distances):.3f}m")
    print(f"  Scale factor to reach r=2: {orig_scale:.4f}")
    print(f"  After scaling, distances: {orig_distances * orig_scale}")

    print(f"\nVirtual cameras:")
    print(f"  Mean distance (centered): {np.mean(virtual_distances):.3f}m")
    print(f"  Scale factor to reach r=2: {virtual_scale:.4f}")
    print(f"  After scaling, distances: {virtual_distances * virtual_scale}")

    print("\n" + "="*60)
    print("KEY INSIGHT")
    print("="*60)
    print("""
With virtual cameras:
- Virtual cameras start closer to the object (0.5m vs 11m)
- After centering, they're still at 0.5m distance
- After scaling to r=2, the scale factor is much larger (~4x vs ~0.18x)
- This means the object's physical size is scaled UP proportionally

The key is that we crop a SMALLER region (just the object) and use
the correct intrinsics for that crop. The virtual camera positioning
helps ensure the object fills the crop appropriately.
    """)

    return orig_scale, virtual_scale


def main():
    render_dir = "/home/sandro/thesis/renders/5cams_bird_100m"
    frame_id = 60
    output_dir = SCRIPT_DIR / "imgs" / "virtual_camera_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("VIRTUAL CAMERA ANALYSIS FOR IN-THE-WILD FRAMES")
    print("="*60)
    print(f"Render directory: {render_dir}")
    print(f"Frame ID: {frame_id}")
    print(f"Training FOV: {TRAINING_FOV_DEG}° (fx_norm={TRAINING_FX_NORM})")
    print(f"Target object coverage: {TARGET_OBJECT_COVERAGE*100}%")
    print(f"Output directory: {output_dir}")

    # Load frame data
    frame_data, service = load_frame_data(render_dir, frame_id)

    # Analyze virtual camera parameters
    results = analyze_virtual_camera_parameters(frame_data)

    object_pos = frame_data['object_position_3d']

    # 1. Visualize camera positions in 3D (original function)
    print("\n--- Generating 3D camera positions plot ---")
    visualize_camera_positions(
        results,
        object_pos,
        str(output_dir / "camera_positions_3d.png")
    )

    # 2. NEW: Visualize camera positions in 2D top-view
    print("\n--- Generating 2D top-view camera positions plot ---")
    visualize_camera_positions_2d(
        results,
        object_pos,
        str(output_dir / "camera_positions_2d_topview.png")
    )

    # 3. NEW: Generate single camera comparison for each camera
    print("\n--- Generating single camera frustum comparisons ---")
    for cam_name, result_data in results.items():
        cam_data = frame_data['cameras'][cam_name]
        visualize_single_camera_comparison(
            cam_name=cam_name,
            cam_data=cam_data,
            result_data=result_data,
            object_pos=object_pos,
            service=service,
            frame_id=frame_id,
            output_dir=str(output_dir)
        )

    # Test normalization
    test_normalized_virtual_cameras(results, object_pos)

    print("\n" + "="*60)
    print("VISUALIZATION OUTPUTS")
    print("="*60)
    print(f"\nAll plots saved to: {output_dir}")
    print("Generated files:")
    print("  - camera_positions_3d.png: 3D view of all camera positions")
    print("  - camera_positions_2d_topview.png: Top-view (X-Y) of camera positions with center lines")
    for cam_name in results.keys():
        print(f"  - camera_comparison_{cam_name}.png: Frustum + image comparison")

    print("\n" + "="*60)
    print("RECOMMENDED SOLUTION")
    print("="*60)
    print("""
Based on this analysis, the virtual camera approach should:

1. CROP BASED ON MASK SIZE:
   - Instead of fixed 256x256 crop, calculate crop size to achieve target FOV
   - Use bbox/mask to determine object size in pixels
   - Crop tightly around object (with small margin)

2. COMPUTE VIRTUAL CAMERA POSITION:
   - Calculate physical object size from original camera params
   - Compute virtual distance for training-matched FOV (50°)
   - Update extrinsics to place camera at virtual distance

3. NORMALIZE POSES:
   - Center on object (already doing this)
   - Scale to target radius (already doing this)

4. USE TRAINING-MATCHED INTRINSICS:
   - Override intrinsics with fx_norm=1.0723 (already doing this)

The key improvement is #2 - instead of just overriding intrinsics,
we also position the virtual camera closer, which maintains geometric
consistency between the intrinsics and extrinsics.
    """)


if __name__ == "__main__":
    main()

