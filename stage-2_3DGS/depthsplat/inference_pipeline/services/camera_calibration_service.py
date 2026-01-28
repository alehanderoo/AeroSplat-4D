"""
Camera Calibration Service.

Loads camera intrinsics and extrinsics from Isaac Sim render metadata
(drone_camera_observations.json) and provides them in formats suitable
for the DepthSplat encoder.

Key features:
- Virtual camera transformation for crop regions: When cropping a sub-region
  of the original image, computes a "virtual camera" with adjusted extrinsics
  that maintains correct multi-view geometry.
- Proper intrinsics normalization for crops with principal point centering.
- OpenGL→OpenCV coordinate transformation for Isaac Sim cameras.
- Pose normalization (centering at object + scaling to target radius).
- Training-matched intrinsics override for better generalization.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)

# Training-matched intrinsics constant
# The model was trained on Objaverse with fx_norm=1.0723 (50° FOV)
# Wild frames have different FOV due to cropping from high-res cameras
# Using training-matched intrinsics improves generalization
TRAINING_FX_NORM = 1.0723

# Target radius for pose normalization (meters)
# The model expects cameras at ~2.0 meters from the object
TARGET_POSE_RADIUS = 2.0

# OpenGL to OpenCV coordinate transformation matrix
# OpenGL: +X right, +Y up, +Z toward viewer (out of screen)
# OpenCV: +X right, +Y down, +Z forward (into screen)
# Transformation: flip Y and Z axes
OPENGL_TO_OPENCV_FLIP = np.diag([1.0, -1.0, -1.0]).astype(np.float32)


def apply_coordinate_flip(extrinsics: np.ndarray) -> np.ndarray:
    """
    Apply OpenGL→OpenCV coordinate transformation to camera extrinsics.

    Isaac Sim (and many 3D tools) use OpenGL convention where:
    - +X is right
    - +Y is up
    - +Z is toward the viewer (out of screen)

    The model expects OpenCV convention where:
    - +X is right
    - +Y is down
    - +Z is forward (into screen)

    Args:
        extrinsics: [4, 4] or [N, 4, 4] camera-to-world matrix

    Returns:
        Transformed camera-to-world matrix with OpenCV convention
    """
    if extrinsics.ndim == 2:
        result = extrinsics.copy()
        result[:3, :3] = extrinsics[:3, :3] @ OPENGL_TO_OPENCV_FLIP
        return result
    else:
        result = extrinsics.copy()
        for i in range(len(result)):
            result[i, :3, :3] = extrinsics[i, :3, :3] @ OPENGL_TO_OPENCV_FLIP
        return result


def normalize_camera_poses(
    extrinsics_list: list,
    center: np.ndarray = None,
    target_radius: float = TARGET_POSE_RADIUS,
) -> tuple:
    """
    Normalize camera poses by centering and scaling.

    The model was trained with cameras centered around the object at a
    specific distance (~2.0m). This function transforms arbitrary camera
    poses to match the training distribution.

    Args:
        extrinsics_list: List of [4, 4] camera-to-world matrices
        center: Center point for normalization. If None, uses mean camera position.
        target_radius: Target distance from center (default: 2.0m)

    Returns:
        Tuple of (normalized_extrinsics_list, scale_factor, center)
    """
    if len(extrinsics_list) == 0:
        return [], 1.0, np.zeros(3)

    # Copy to avoid modifying originals
    result = [ext.copy() for ext in extrinsics_list]

    # Get camera positions
    positions = np.array([ext[:3, 3] for ext in result])

    # Determine center
    if center is None:
        # Use centroid of camera positions as default
        # This works when cameras are arranged around the object
        center = positions.mean(axis=0)
        logger.info(f"Pose normalization: Using camera centroid as center: {center}")
    else:
        logger.info(f"Pose normalization: Using provided center: {center}")

    # Center the cameras
    for ext in result:
        ext[:3, 3] -= center

    # Calculate current mean distance from center
    distances = np.array([np.linalg.norm(ext[:3, 3]) for ext in result])
    mean_dist = np.mean(distances)

    if mean_dist < 1e-6:
        logger.warning("Mean distance is near zero, skipping scaling")
        return result, 1.0, center

    # Scale to target radius
    scale_factor = target_radius / mean_dist
    logger.info(f"Pose normalization: Scaling by {scale_factor:.4f} (dist: {mean_dist:.2f} -> {target_radius:.1f})")

    for ext in result:
        ext[:3, 3] *= scale_factor

    return result, scale_factor, center


def compute_crop_rotation(
    cx: float, cy: float, fx: float, fy: float,
    crop_center_x: float, crop_center_y: float
) -> np.ndarray:
    """
    Compute the rotation matrix that aligns the crop center with the optical axis.

    When we crop a region not centered on the principal point, we're effectively
    looking at a different direction. This function computes the rotation needed
    to make that direction the new optical axis.

    Args:
        cx, cy: Original principal point in pixels
        fx, fy: Focal lengths in pixels
        crop_center_x, crop_center_y: Center of the crop region in pixels

    Returns:
        3x3 rotation matrix (camera frame rotation)
    """
    # Compute the offset from principal point to crop center
    dx = crop_center_x - cx
    dy = crop_center_y - cy

    # Convert pixel offset to angular offset
    # In camera coordinates: X right, Y down, Z forward (OpenCV convention)
    # A pixel at (u, v) corresponds to ray direction proportional to:
    # [(u - cx) / fx, (v - cy) / fy, 1]

    # The crop center ray direction (unnormalized)
    ray_x = dx / fx
    ray_y = dy / fy
    ray_z = 1.0

    # Normalize to get unit vector
    ray_norm = np.sqrt(ray_x**2 + ray_y**2 + ray_z**2)
    ray = np.array([ray_x / ray_norm, ray_y / ray_norm, ray_z / ray_norm])

    # We want to rotate this ray to align with [0, 0, 1] (optical axis)
    # Using Rodrigues' rotation formula: R = I + [v]_x + [v]_x^2 * (1-c)/s^2
    # where v = ray × [0,0,1], c = ray · [0,0,1], s = |v|

    target = np.array([0.0, 0.0, 1.0])

    # Cross product: ray × target
    v = np.cross(ray, target)
    s = np.linalg.norm(v)
    c = np.dot(ray, target)

    if s < 1e-8:
        # Rays are parallel, no rotation needed (or 180° flip, but that shouldn't happen)
        return np.eye(3, dtype=np.float32)

    # Skew-symmetric cross-product matrix of v
    vx = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

    # Rodrigues' formula
    R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))

    return R.astype(np.float32)


class CameraCalibrationService:
    """
    Service that loads and provides camera calibration data.

    Loads intrinsics and extrinsics from the drone_camera_observations.json
    file and provides normalized versions suitable for the DepthSplat encoder.

    The encoder expects:
    - Intrinsics: Normalized to [0,1] range (fx/width, cx/width, etc.)
    - Extrinsics: Camera-to-world 4x4 matrices

    Key transformations (matching the working Gradio demo):
    - OpenGL→OpenCV coordinate flip (Isaac Sim uses OpenGL convention)
    - Pose normalization (center at object, scale to target radius)
    - Optional training-matched intrinsics override
    """

    def __init__(
        self,
        json_path: Union[str, Path],
        camera_names: List[str] = None,
        apply_coordinate_flip: bool = True,
        apply_pose_normalization: bool = True,
        use_training_intrinsics: bool = True,
        target_radius: float = TARGET_POSE_RADIUS,
    ):
        """
        Initialize the camera calibration service.

        Args:
            json_path: Path to the drone_camera_observations.json file
            camera_names: List of camera names to use (auto-detected if None)
            apply_coordinate_flip: Apply OpenGL→OpenCV transformation (default True)
            apply_pose_normalization: Center and scale camera poses (default True)
            use_training_intrinsics: Override with training-matched intrinsics (default True)
            target_radius: Target distance from center for pose normalization (default 2.0)
        """
        self.json_path = Path(json_path)

        # Configuration flags
        self._apply_coordinate_flip = apply_coordinate_flip
        self._apply_pose_normalization = apply_pose_normalization
        self._use_training_intrinsics = use_training_intrinsics
        self._target_radius = target_radius

        # Normalization state (computed lazily)
        self._pose_center: Optional[np.ndarray] = None
        self._pose_scale: float = 1.0
        self._object_position: Optional[np.ndarray] = None
        self._normalized_extrinsics_cache: Optional[Dict[str, np.ndarray]] = None

        # Load and parse JSON
        self._data = self._load_json()
        self._cameras_data: List[Dict] = self._data.get("cameras", [])
        
        # Auto-detect camera names if not provided
        if camera_names is None:
            camera_names = self._data.get("metadata", {}).get(
                "cameras_recorded",
                [cam["name"] for cam in self._cameras_data]
            )
        
        self.camera_names = camera_names
        
        # Build lookup dict for camera data
        self._camera_lookup: Dict[str, Dict] = {}
        for cam_data in self._cameras_data:
            name = cam_data.get("name")
            if name:
                self._camera_lookup[name] = cam_data
        
        logger.info(
            f"CameraCalibrationService loaded: {len(self._camera_lookup)} cameras"
        )

    def _load_json(self) -> Dict:
        """Load and parse the JSON file."""
        if not self.json_path.exists():
            raise FileNotFoundError(f"Calibration file not found: {self.json_path}")

        logger.info(f"Loading camera calibration from: {self.json_path}")
        with open(self.json_path, 'r') as f:
            return json.load(f)

    def get_camera_data(self, camera_name: str) -> Optional[Dict]:
        """Get raw camera data for a specific camera."""
        return self._camera_lookup.get(camera_name)

    def get_resolution(self, camera_name: str) -> Optional[Tuple[int, int]]:
        """
        Get resolution for a specific camera.

        Returns:
            (width, height) tuple or None
        """
        cam_data = self._camera_lookup.get(camera_name)
        if cam_data is None:
            return None
        
        res = cam_data.get("resolution", {})
        return (res.get("width"), res.get("height"))

    def get_intrinsics(self, camera_name: str) -> Optional[np.ndarray]:
        """
        Get the 3x3 intrinsic matrix for a camera.

        Returns intrinsics in pixel units (not normalized).

        Returns:
            3x3 numpy array or None
        """
        cam_data = self._camera_lookup.get(camera_name)
        if cam_data is None:
            return None
        
        intr_data = cam_data.get("intrinsics", {})
        matrix = intr_data.get("matrix")
        
        if matrix is not None:
            return np.array(matrix, dtype=np.float32)
        
        # Fallback: build from fx, fy, cx, cy
        fx = intr_data.get("fx")
        fy = intr_data.get("fy")
        cx = intr_data.get("cx")
        cy = intr_data.get("cy")
        
        if all(v is not None for v in [fx, fy, cx, cy]):
            return np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ], dtype=np.float32)
        
        return None

    def get_extrinsics(self, camera_name: str) -> Optional[np.ndarray]:
        """
        Get the 4x4 camera-to-world extrinsic matrix for a camera.

        Returns:
            4x4 numpy array or None
        """
        cam_data = self._camera_lookup.get(camera_name)
        if cam_data is None:
            return None
        
        extr_data = cam_data.get("extrinsics", {})
        
        # Prefer camera-to-world matrix
        c2w = extr_data.get("camera_to_world_matrix")
        if c2w is not None:
            return np.array(c2w, dtype=np.float32)
        
        # Fallback: invert world-to-camera matrix
        w2c = extr_data.get("world_to_camera_matrix")
        if w2c is not None:
            return np.linalg.inv(np.array(w2c, dtype=np.float32))
        
        return None

    def get_normalized_intrinsics(
        self,
        camera_name: str,
        crop_region: Tuple[int, int, int, int] = None,
        output_size: Tuple[int, int] = (256, 256),
        use_virtual_camera: bool = True,
    ) -> Optional[np.ndarray]:
        """
        Get intrinsics normalized for the crop region and output size.

        The DepthSplat encoder expects intrinsics normalized to [0,1]:
        - fx, fy: focal length / image dimension
        - cx, cy: principal point / image dimension

        When use_virtual_camera=True (recommended), the principal point is
        centered at (0.5, 0.5) because we apply a virtual rotation to the
        camera extrinsics to align the crop center with the optical axis.

        Args:
            camera_name: Name of the camera
            crop_region: (x1, y1, x2, y2) crop region in original image coords
                        If None, uses full image
            output_size: (width, height) of the model input
            use_virtual_camera: If True, assume virtual camera rotation is applied,
                               so principal point is centered. Default True.

        Returns:
            3x3 numpy array with normalized intrinsics
        """
        intrinsics = self.get_intrinsics(camera_name)
        if intrinsics is None:
            return None

        resolution = self.get_resolution(camera_name)
        if resolution is None:
            return None

        orig_w, orig_h = resolution
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]

        if crop_region is not None:
            x1, y1, x2, y2 = crop_region
            crop_w = x2 - x1
            crop_h = y2 - y1

            # Normalize focal length by crop size
            # This preserves the FOV of the cropped region
            fx_norm = fx / crop_w
            fy_norm = fy / crop_h

            if use_virtual_camera:
                # With virtual camera rotation, the crop center becomes the
                # new optical axis, so principal point is at image center
                cx_norm = 0.5
                cy_norm = 0.5
            else:
                # Legacy behavior: shift principal point by crop offset
                cx_crop = cx - x1
                cy_crop = cy - y1
                cx_norm = cx_crop / crop_w
                cy_norm = cy_crop / crop_h
        else:
            # No crop: normalize by original image size
            fx_norm = fx / orig_w
            fy_norm = fy / orig_h
            cx_norm = cx / orig_w
            cy_norm = cy / orig_h

        return np.array([
            [fx_norm, 0, cx_norm],
            [0, fy_norm, cy_norm],
            [0, 0, 1]
        ], dtype=np.float32)

    def get_virtual_extrinsics(
        self,
        camera_name: str,
        crop_region: Tuple[int, int, int, int],
    ) -> Optional[np.ndarray]:
        """
        Get the virtual camera-to-world extrinsic matrix for a cropped region.

        When cropping a region not centered on the principal point, we create
        a "virtual camera" that has been rotated so that the crop center lies
        on the optical axis. This preserves correct multi-view geometry.

        The transformation is:
        1. Compute the rotation R that aligns crop center ray with optical axis
        2. Apply R to the camera orientation: C2W_virtual = C2W_original @ R^T
           (R^T because we're rotating the camera frame, not the world)

        Args:
            camera_name: Name of the camera
            crop_region: (x1, y1, x2, y2) crop region in original image coords

        Returns:
            4x4 camera-to-world matrix for the virtual camera, or None on error
        """
        extrinsics = self.get_extrinsics(camera_name)
        if extrinsics is None:
            return None

        intrinsics = self.get_intrinsics(camera_name)
        if intrinsics is None:
            return None

        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]

        # Compute crop center
        x1, y1, x2, y2 = crop_region
        crop_center_x = (x1 + x2) / 2.0
        crop_center_y = (y1 + y2) / 2.0

        # Compute the rotation that aligns crop center with optical axis
        R_crop = compute_crop_rotation(cx, cy, fx, fy, crop_center_x, crop_center_y)

        # Apply rotation to extrinsics
        # C2W = [R_c2w | t_c2w]
        #       [  0   |   1  ]
        # The virtual camera has its orientation rotated by R_crop
        # New orientation: R_virtual = R_c2w @ R_crop^T
        # (We use R_crop^T because R_crop rotates rays in camera space,
        #  but we need to rotate the camera itself in the opposite direction)

        R_c2w = extrinsics[:3, :3]
        t_c2w = extrinsics[:3, 3]

        R_virtual = R_c2w @ R_crop.T

        # Translation doesn't change (camera position stays the same)
        virtual_extrinsics = np.eye(4, dtype=np.float32)
        virtual_extrinsics[:3, :3] = R_virtual
        virtual_extrinsics[:3, 3] = t_c2w

        return virtual_extrinsics

    def get_all_extrinsics(self) -> Dict[str, np.ndarray]:
        """Get extrinsics for all cameras."""
        result = {}
        for name in self.camera_names:
            ext = self.get_extrinsics(name)
            if ext is not None:
                result[name] = ext
        return result

    def get_all_intrinsics(self) -> Dict[str, np.ndarray]:
        """Get intrinsics for all cameras."""
        result = {}
        for name in self.camera_names:
            intr = self.get_intrinsics(name)
            if intr is not None:
                result[name] = intr
        return result

    def get_intrinsics_tensor(
        self,
        device: "torch.device" = None,
        crop_regions: List[Tuple[int, int, int, int]] = None,
        use_virtual_camera: bool = True,
    ) -> "torch.Tensor":
        """
        Get intrinsics as a tensor for all cameras.

        Args:
            device: PyTorch device
            crop_regions: List of (x1, y1, x2, y2) crop regions per camera
                         If None, uses full image (no crop)
            use_virtual_camera: If True, use virtual camera intrinsics with
                               centered principal point. Should match the
                               use_virtual_camera flag in get_extrinsics_tensor.

        Returns:
            Tensor of shape [1, num_cameras, 3, 3] with normalized intrinsics
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available")

        # --- Use training-matched intrinsics if enabled ---
        # This is CRITICAL: The model was trained on 50° FOV images (fx_norm=1.0723)
        # Real cameras may have very different FOV due to cropping from telephoto lenses
        # Using training-matched intrinsics keeps input in-distribution
        if self._use_training_intrinsics:
            training_intrinsics = np.array([
                [TRAINING_FX_NORM, 0, 0.5],
                [0, TRAINING_FX_NORM, 0.5],
                [0, 0, 1],
            ], dtype=np.float32)

            logger.info(f"Using training-matched intrinsics: fx_norm={TRAINING_FX_NORM:.4f}, cx=cy=0.5")

            # Stack same intrinsics for all cameras
            intrinsics = np.stack([training_intrinsics.copy() for _ in self.camera_names], axis=0)

            # Convert to tensor and add batch dim: [1, V, 3, 3]
            tensor = torch.from_numpy(intrinsics).unsqueeze(0)

            if device is not None:
                tensor = tensor.to(device)

            return tensor

        # --- Otherwise compute actual intrinsics (legacy behavior) ---
        intrinsics_list = []

        for i, cam_name in enumerate(self.camera_names):
            crop_region = crop_regions[i] if crop_regions else None

            intr = self.get_normalized_intrinsics(
                cam_name,
                crop_region=crop_region,
                use_virtual_camera=use_virtual_camera,
            )

            if intr is None:
                # Fallback to default normalized intrinsics
                logger.warning(f"No intrinsics for {cam_name}, using defaults")
                intr = np.array([
                    [1.0, 0, 0.5],
                    [0, 1.0, 0.5],
                    [0, 0, 1]
                ], dtype=np.float32)
            else:
                # Log the intrinsics for debugging
                logger.debug(
                    f"{cam_name}: fx_norm={intr[0,0]:.3f}, fy_norm={intr[1,1]:.3f}, "
                    f"cx_norm={intr[0,2]:.3f}, cy_norm={intr[1,2]:.3f}"
                )

            intrinsics_list.append(intr)

        # Stack: [V, 3, 3]
        intrinsics = np.stack(intrinsics_list, axis=0)

        # Convert to tensor and add batch dim: [1, V, 3, 3]
        tensor = torch.from_numpy(intrinsics).unsqueeze(0)

        if device is not None:
            tensor = tensor.to(device)

        return tensor

    def get_extrinsics_tensor(
        self,
        device: "torch.device" = None,
        crop_regions: List[Tuple[int, int, int, int]] = None,
        use_virtual_camera: bool = True,
    ) -> "torch.Tensor":
        """
        Get extrinsics as a tensor for all cameras.

        When crop_regions is provided and use_virtual_camera=True, computes
        virtual camera extrinsics that account for the crop offset, preserving
        correct multi-view geometry.

        Args:
            device: PyTorch device
            crop_regions: List of (x1, y1, x2, y2) crop regions per camera.
                         Required when use_virtual_camera=True for crops.
            use_virtual_camera: If True and crop_regions provided, compute
                               virtual camera extrinsics for each crop.

        Returns:
            Tensor of shape [1, num_cameras, 4, 4] with camera-to-world transforms
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available")

        extrinsics_list = []

        for i, cam_name in enumerate(self.camera_names):
            crop_region = crop_regions[i] if crop_regions else None

            if crop_region is not None and use_virtual_camera:
                # Use virtual camera extrinsics for the crop
                ext = self.get_virtual_extrinsics(cam_name, crop_region)
                if ext is not None:
                    logger.debug(f"{cam_name}: Using virtual camera extrinsics for crop")
            else:
                # Use original extrinsics
                ext = self.get_extrinsics(cam_name)

            if ext is None:
                # Fallback to identity
                logger.warning(f"No extrinsics for {cam_name}, using identity")
                ext = np.eye(4, dtype=np.float32)

            extrinsics_list.append(ext)

        # --- Apply coordinate system transformation (OpenGL → OpenCV) ---
        # This is CRITICAL: Isaac Sim uses OpenGL convention, model expects OpenCV
        if self._apply_coordinate_flip:
            logger.debug("Applying OpenGL→OpenCV coordinate flip to extrinsics")
            extrinsics_list = [apply_coordinate_flip(ext) for ext in extrinsics_list]

        # --- Apply pose normalization (centering + scaling) ---
        # This is CRITICAL: Model was trained with cameras at ~2.0m from object
        if self._apply_pose_normalization:
            extrinsics_list, self._pose_scale, self._pose_center = normalize_camera_poses(
                extrinsics_list,
                center=self._object_position,
                target_radius=self._target_radius,
            )

        # Stack: [V, 4, 4]
        extrinsics = np.stack(extrinsics_list, axis=0)

        # Convert to tensor and add batch dim: [1, V, 4, 4]
        tensor = torch.from_numpy(extrinsics).unsqueeze(0)

        if device is not None:
            tensor = tensor.to(device)

        return tensor

    @property
    def num_cameras(self) -> int:
        """Number of cameras."""
        return len(self.camera_names)

    @property
    def metadata(self) -> Dict:
        """Raw metadata from the JSON file."""
        return self._data.get("metadata", {})

    def set_object_position(self, position: np.ndarray) -> None:
        """
        Set the 3D position of the object being tracked.

        This position is used as the center for pose normalization.
        Should be called with the object position from the detection service
        before requesting extrinsics tensors.

        Args:
            position: [3] 3D position of the object in world coordinates
        """
        if position is not None:
            self._object_position = np.array(position, dtype=np.float32)
            logger.info(f"Object position set to: {self._object_position}")
            # Clear cached normalized extrinsics since center changed
            self._normalized_extrinsics_cache = None
        else:
            self._object_position = None

    def get_object_position(self) -> Optional[np.ndarray]:
        """Get the currently set object position."""
        return self._object_position

    @property
    def pose_scale(self) -> float:
        """
        Get the scale factor applied during pose normalization.

        Returns 1.0 if pose normalization hasn't been applied yet.
        """
        return self._pose_scale

    @property
    def pose_center(self) -> Optional[np.ndarray]:
        """
        Get the center point used for pose normalization.

        Returns None if pose normalization hasn't been applied yet.
        """
        return self._pose_center

    def reset_normalization(self) -> None:
        """Reset the normalization state, clearing cached values."""
        self._pose_center = None
        self._pose_scale = 1.0
        self._object_position = None
        self._normalized_extrinsics_cache = None
        logger.debug("Normalization state reset")
