"""
Wild Frame Loader for the DepthSplat inference backend.

Handles loading in-the-wild frames from Isaac Sim renders with:
- Tight cropping around detected objects
- Virtual camera positioning for FOV matching
- Pose normalization (centering and scaling)
- Intrinsics adjustment for cropping
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
from PIL import Image

from .services import create_gt_detection_service
from .camera_utils import normalize_intrinsics, compute_mean_camera_elevation


# Target object coverage in the cropped frame (75% = object fills 75% of frame)
TARGET_OBJECT_COVERAGE = 0.75

# Training-matched intrinsics (50 degree FOV)
TRAINING_FX_NORM = 1.0723

# Output image size
OUTPUT_SIZE = 256

# Margin around object in tight crop
CROP_MARGIN = 0.15


class WildFrameLoader:
    """
    Loader for in-the-wild frames from Isaac Sim renders.

    Handles the complex preprocessing needed to make wild frames
    compatible with the DepthSplat model trained on Objaverse.
    """

    def __init__(
        self,
        render_dir: str,
        use_virtual_cameras: bool = True,
        target_radius: float = 2.0,
    ):
        """
        Initialize the wild frame loader.

        Args:
            render_dir: Path to render directory containing Isaac Sim outputs
            use_virtual_cameras: Whether to use virtual camera positioning
            target_radius: Target camera distance after normalization
        """
        self.render_dir = Path(render_dir)
        self.use_virtual_cameras = use_virtual_cameras
        self.target_radius = target_radius

        # Initialize detection service
        self.service = create_gt_detection_service(render_dir)

    def load_frame(
        self,
        frame_id: int,
        cache_dir: str = None,
    ) -> Dict:
        """
        Load a specific frame from the render directory.

        Args:
            frame_id: Frame index (0-119 typically)
            cache_dir: Directory for temporary files

        Returns:
            Dictionary containing:
            - image_paths: List of paths to processed images
            - extrinsics: [V, 4, 4] camera-to-world matrices
            - intrinsics: [V, 3, 3] normalized intrinsics
            - mean_elevation: Mean camera elevation in degrees
            - scale_factor: Applied scaling factor
            - center: Applied centering offset
            - status: Status message
        """
        if cache_dir is None:
            cache_dir = "/tmp/depthsplat_wild"
        os.makedirs(cache_dir, exist_ok=True)

        print(f"Loading frame {frame_id} from {self.render_dir}")
        if self.use_virtual_cameras:
            print(f"Virtual cameras enabled (target coverage: {TARGET_OBJECT_COVERAGE*100:.0f}%)")

        detections = self.service.get_detections(frame_id)
        if not detections:
            raise RuntimeError(f"No detections found for frame {frame_id}")

        image_paths = []
        extrinsics_list = []
        intrinsics_list = []
        camera_info_list = []

        # Iterate over cameras in order
        for cam_name in sorted(self.service.camera_names):
            result = self._process_camera(
                frame_id=frame_id,
                cam_name=cam_name,
                detections=detections,
                cache_dir=cache_dir,
            )
            if result is None:
                continue

            image_paths.append(result['image_path'])
            extrinsics_list.append(result['extrinsics'])
            intrinsics_list.append(result['intrinsics'])
            camera_info_list.append(result['camera_info'])

        if not image_paths:
            raise RuntimeError(f"Failed to load any valid views for frame {frame_id}")

        # Get object 3D position
        if detections.object_position_3d:
            object_position = np.array(detections.object_position_3d, dtype=np.float32)
        else:
            print("Warning: No drone_position_3d found, using mean camera target")
            object_position = np.mean([ext[:3, 3] for ext in extrinsics_list], axis=0)

        # Report crop info
        if self.use_virtual_cameras and camera_info_list:
            crop_sizes = [info['crop_size'] for info in camera_info_list]
            bbox_sizes = [info['bbox_size'] for info in camera_info_list]
            print(f"Tight cropping enabled:")
            print(f"  Bbox sizes: {[f'{s:.0f}' for s in bbox_sizes]} px")
            print(f"  Crop sizes: {crop_sizes} px (resized to 256x256)")
            print(f"  Object fills ~{TARGET_OBJECT_COVERAGE*100:.0f}% of output")

        # Pose normalization
        center = object_position
        print(f"Normalizing pose: Centering at {center}")

        for ext in extrinsics_list:
            ext[:3, 3] -= center

        # Scaling
        distances = [np.linalg.norm(ext[:3, 3]) for ext in extrinsics_list]
        mean_dist = np.mean(distances)

        scale_factor = self.target_radius / mean_dist
        print(f"Normalizing pose: Scaling by {scale_factor:.4f} (original dist: {mean_dist:.2f} -> {self.target_radius})")

        for ext in extrinsics_list:
            ext[:3, 3] *= scale_factor

        # Override intrinsics with training-matched values
        training_intrinsics = np.array([
            [TRAINING_FX_NORM, 0, 0.5],
            [0, TRAINING_FX_NORM, 0.5],
            [0, 0, 1],
        ], dtype=np.float32)
        intrinsics_list = [training_intrinsics.copy() for _ in intrinsics_list]

        # Calculate mean camera elevation
        extrinsics_array = np.stack(extrinsics_list)
        mean_elevation = compute_mean_camera_elevation(extrinsics_array)

        return {
            'image_paths': image_paths,
            'extrinsics': extrinsics_array,
            'intrinsics': np.stack(intrinsics_list),
            'mean_elevation': mean_elevation,
            'scale_factor': scale_factor,
            'center': center,
            'render_dir': str(self.render_dir),
            'status': f"Loaded Frame {frame_id} ({len(image_paths)} views)",
        }

    def _process_camera(
        self,
        frame_id: int,
        cam_name: str,
        detections,
        cache_dir: str,
    ) -> Optional[Dict]:
        """Process a single camera view."""
        # Load image
        img_path = self.render_dir / cam_name / "rgb" / f"rgb_{frame_id:04d}.png"
        if not img_path.exists():
            print(f"Warning: Image not found {img_path}")
            return None

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Failed to load {img_path}: {e}")
            return None

        w, h = img.size

        # Get detection for cropping
        det = detections.get_detection(cam_name)
        if not det:
            print(f"No detection for {cam_name} in frame {frame_id}")
            return None

        # Get bbox info
        bbox = det.bbox
        if bbox:
            bbox_width = bbox[2] - bbox[0]
            bbox_height = bbox[3] - bbox[1]
            bbox_size = max(bbox_width, bbox_height)
        else:
            bbox_size = 68.0  # Default fallback

        # Determine crop size
        if self.use_virtual_cameras:
            tight_crop_size = int(bbox_size / TARGET_OBJECT_COVERAGE * (1 + CROP_MARGIN))
            tight_crop_size = max(tight_crop_size, 64)
            crop_size = tight_crop_size
        else:
            crop_size = OUTPUT_SIZE

        # Calculate crop region
        crop = det.get_crop_region(crop_size=crop_size, image_width=w, image_height=h)
        x1, y1, x2, y2 = crop

        # Crop image
        img_crop = img.crop(crop)

        # Apply mask for white background
        mask_path = self.render_dir / cam_name / "mask" / f"drone_mask_{frame_id:04d}.png"
        if mask_path.exists():
            mask = Image.open(mask_path).convert("L")
            mask_crop = mask.crop(crop)

            img_np = np.array(img_crop)
            mask_np = np.array(mask_crop)

            white_bg = np.ones_like(img_np) * 255
            fg_mask = (mask_np > 127)[..., np.newaxis]
            img_np = np.where(fg_mask, img_np, white_bg)

            img_crop = Image.fromarray(img_np.astype(np.uint8))
        else:
            print(f"Warning: Mask not found at {mask_path}, using original image")

        # Resize to output size
        if img_crop.size[0] != OUTPUT_SIZE or img_crop.size[1] != OUTPUT_SIZE:
            img_crop = img_crop.resize((OUTPUT_SIZE, OUTPUT_SIZE), Image.LANCZOS)

        # Save processed image
        save_path = os.path.join(cache_dir, f"{cam_name}_frame{frame_id}.png")
        img_crop.save(save_path)

        # Process intrinsics
        intr_raw = self.service.get_camera_intrinsics(cam_name)
        if intr_raw is None:
            raise RuntimeError(f"No intrinsics for {cam_name}")

        if isinstance(intr_raw, dict):
            fx = intr_raw.get('fx')
            fy = intr_raw.get('fy')
            cx = intr_raw.get('cx')
            cy = intr_raw.get('cy')
        elif isinstance(intr_raw, list) and len(intr_raw) == 3:
            K_raw = np.array(intr_raw)
            fx, fy = K_raw[0, 0], K_raw[1, 1]
            cx, cy = K_raw[0, 2], K_raw[1, 2]
        else:
            raise RuntimeError(f"Unknown intrinsics format for {cam_name}")

        # Adjust principal point for crop
        new_cx = cx - x1
        new_cy = cy - y1

        K_new = np.array([
            [fx, 0, new_cx],
            [0, fy, new_cy],
            [0, 0, 1]
        ], dtype=np.float32)

        # Adjust for resize
        if crop_size != OUTPUT_SIZE:
            resize_scale = OUTPUT_SIZE / crop_size
            K_new[0, :] *= resize_scale
            K_new[1, :] *= resize_scale

        # Normalize intrinsics
        K_norm = normalize_intrinsics(K_new, (OUTPUT_SIZE, OUTPUT_SIZE))

        # Process extrinsics
        ext_raw = self.service.get_camera_extrinsics(cam_name)
        if ext_raw is None:
            raise RuntimeError(f"No extrinsics for {cam_name}")

        ext_matrix = np.array(ext_raw, dtype=np.float32)

        # Coordinate system fix: OpenGL -> OpenCV
        flip_mat = np.diag([1.0, -1.0, -1.0]).astype(np.float32)
        ext_matrix[:3, :3] = ext_matrix[:3, :3] @ flip_mat

        return {
            'image_path': save_path,
            'extrinsics': ext_matrix,
            'intrinsics': K_norm,
            'camera_info': {
                'fx': fx,
                'depth': det.depth if det.depth else 10.0,
                'bbox_size': bbox_size,
                'crop_size': crop_size,
            },
        }

    def load_images(self, image_paths: List[str]) -> List[np.ndarray]:
        """Load images from paths as numpy arrays."""
        images = []
        for path in image_paths:
            img = Image.open(path).convert("RGB")
            images.append(np.array(img))
        return images


def load_wild_frame(
    frame_id: int,
    render_dir: str,
    cache_dir: str = None,
    use_virtual_cameras: bool = True,
) -> Dict:
    """
    Convenience function to load a wild frame.

    Args:
        frame_id: Frame index
        render_dir: Path to render directory
        cache_dir: Temporary file directory
        use_virtual_cameras: Whether to use virtual camera positioning

    Returns:
        Dictionary with loaded frame data
    """
    loader = WildFrameLoader(
        render_dir=render_dir,
        use_virtual_cameras=use_virtual_cameras,
    )
    return loader.load_frame(frame_id, cache_dir)
