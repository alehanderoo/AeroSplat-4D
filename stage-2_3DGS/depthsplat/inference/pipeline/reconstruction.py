"""
Reconstruction Pipeline - Copied exactly from working Gradio demo.

This module provides the exact same reconstruction pipeline as the Gradio demo,
ensuring consistent results between the two interfaces.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# =============================================================================
# Constants from Gradio demo (MUST match exactly)
# =============================================================================

# Training-matched intrinsics constant
# The model was trained on Objaverse with fx_norm=1.0723 (50 deg FOV)
TRAINING_FX_NORM = 1.0723

# Target object coverage in the cropped frame (75% = object fills 75% of frame)
TARGET_OBJECT_COVERAGE = 0.75

# Crop margin around object
CROP_MARGIN = 0.15

# Output size (model expects 256x256)
OUTPUT_SIZE = 256

# Target radius for pose normalization
TARGET_RADIUS = 2.0


@dataclass
class ReconstructionResult:
    """Result from reconstruction pipeline."""
    gaussians: Any  # Gaussians dataclass from encoder
    rendered_image: np.ndarray  # [H, W, 3] RGB
    rendered_depth: Optional[np.ndarray] = None  # [H, W]
    inference_time_ms: float = 0.0
    render_time_ms: float = 0.0
    num_gaussians: int = 0


def normalize_intrinsics(K: np.ndarray, image_shape: Tuple[int, int]) -> np.ndarray:
    """
    Normalize intrinsics to [0, 1] range.

    Args:
        K: [3, 3] camera intrinsics matrix
        image_shape: (height, width) of the image

    Returns:
        [3, 3] normalized intrinsics
    """
    h, w = image_shape
    K_norm = K.copy()
    K_norm[0, :] /= w  # Normalize by width (fx, cx)
    K_norm[1, :] /= h  # Normalize by height (fy, cy)
    return K_norm


def create_orbit_extrinsics(
    num_views: int = 1,
    radius: float = 2.0,
    elevation: float = 0.3,
    azimuth_offset: float = 0.0,
) -> np.ndarray:
    """
    Create camera extrinsics for views orbiting around origin.

    Copied exactly from Gradio demo's create_orbit_extrinsics.

    Args:
        num_views: Number of camera views
        radius: Distance from origin
        elevation: Height offset (Z coordinate)
        azimuth_offset: Starting angle offset in radians

    Returns:
        [V, 4, 4] camera-to-world matrices
    """
    angles = np.linspace(0, 2 * np.pi, num_views, endpoint=False) + azimuth_offset
    extrinsics = []

    for angle in angles:
        # Camera position (Z-up coordinate system)
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = elevation

        # Look at origin
        forward = -np.array([x, y, z])
        forward = forward / np.linalg.norm(forward)

        # Up vector (world Z)
        up = np.array([0, 0, 1])

        # Right vector
        right = np.cross(forward, up)
        norm = np.linalg.norm(right)
        if norm < 1e-6:
            # Camera looking straight up/down
            up = np.array([0, 1, 0])
            right = np.cross(forward, up)
        right = right / np.linalg.norm(right)

        # Recompute up to be orthogonal
        up = np.cross(right, forward)

        # Build rotation matrix
        R = np.stack([right, up, -forward], axis=1)

        # Build 4x4 matrix
        extrinsic = np.eye(4, dtype=np.float32)
        extrinsic[:3, :3] = R
        extrinsic[:3, 3] = [x, y, z]

        extrinsics.append(extrinsic)

    return np.stack(extrinsics)


class GradioReconstructor:
    """
    Reconstruction pipeline copied exactly from working Gradio demo.

    This class encapsulates the exact same pipeline used in runner.py
    to ensure consistent reconstruction quality.
    """

    def __init__(
        self,
        encoder,
        decoder,
        data_shim,
        device: str = "cuda",
        near: float = 0.55,
        far: float = 2.54,
    ):
        """
        Initialize the reconstructor.

        Args:
            encoder: DepthSplat encoder model
            decoder: DepthSplat decoder model
            data_shim: Data shim from get_data_shim(encoder)
            device: Device for inference
            near: Near plane (from config)
            far: Far plane (from config)
        """
        self.encoder = encoder
        self.decoder = decoder
        self.data_shim = data_shim
        self.device = device
        self.near = near
        self.far = far

        # Cache for last reconstruction
        self._last_gaussians = None
        self._last_context = None

    def preprocess_images(
        self,
        images: List[np.ndarray],
        target_size: Tuple[int, int] = (256, 256),
    ) -> torch.Tensor:
        """
        Preprocess input images - copied from Gradio demo.

        Args:
            images: List of numpy arrays (H, W, 3) in [0, 255] uint8 or [0, 1] float
            target_size: Target size (H, W) for resizing

        Returns:
            Tensor of shape [1, V, 3, H, W] in [0, 1]
        """
        processed = []
        for img in images:
            # Convert to float [0, 1]
            if img.dtype == np.uint8:
                img = img.astype(np.float32) / 255.0

            # Convert to tensor [C, H, W]
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()

            # Resize if needed
            if img_tensor.shape[1:] != tuple(target_size):
                img_tensor = F.interpolate(
                    img_tensor.unsqueeze(0),
                    size=target_size,
                    mode='bilinear',
                    align_corners=False,
                ).squeeze(0)

            processed.append(img_tensor)

        # Stack and add batch dimension [1, V, 3, H, W]
        images_tensor = torch.stack(processed, dim=0).unsqueeze(0)
        return images_tensor.to(self.device)

    def build_batch(
        self,
        images: torch.Tensor,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
    ) -> dict:
        """
        Build batch dictionary - copied exactly from Gradio demo.

        Args:
            images: [1, V, 3, H, W] tensor
            extrinsics: [1, V, 4, 4] camera-to-world matrices
            intrinsics: [1, V, 3, 3] normalized intrinsics

        Returns:
            Batch dictionary with context and target
        """
        b, v, _, h, w = images.shape

        context = {
            'image': images,
            'extrinsics': extrinsics.to(self.device),
            'intrinsics': intrinsics.to(self.device),
            'near': torch.full((b, v), self.near, device=self.device),
            'far': torch.full((b, v), self.far, device=self.device),
            'index': torch.arange(v, device=self.device).unsqueeze(0),
        }

        # Create dummy target for data shim (uses first context view)
        target = {
            'image': images[:, :1],
            'extrinsics': extrinsics[:, :1].to(self.device),
            'intrinsics': intrinsics[:, :1].to(self.device),
            'near': torch.full((b, 1), self.near, device=self.device),
            'far': torch.full((b, 1), self.far, device=self.device),
            'index': torch.zeros(b, 1, dtype=torch.long, device=self.device),
        }

        batch = {
            'context': context,
            'target': target,
            'scene': ['inference_input'],
        }

        return batch

    def normalize_poses(
        self,
        extrinsics_list: List[np.ndarray],
        object_position: np.ndarray,
    ) -> Tuple[List[np.ndarray], float]:
        """
        Normalize camera poses - copied exactly from Gradio demo.

        1. Center at object position
        2. Scale to target radius (2.0)

        Args:
            extrinsics_list: List of [4, 4] camera-to-world matrices
            object_position: [3] 3D position of the object

        Returns:
            Tuple of (normalized extrinsics list, scale factor)
        """
        # 1. Center at object position
        center = object_position.copy()
        for ext in extrinsics_list:
            ext[:3, 3] -= center

        # 2. Scale to target radius
        distances = [np.linalg.norm(ext[:3, 3]) for ext in extrinsics_list]
        mean_dist = np.mean(distances)

        scale_factor = TARGET_RADIUS / mean_dist

        for ext in extrinsics_list:
            ext[:3, 3] *= scale_factor

        logger.debug(f"Pose normalization: center={center}, scale={scale_factor:.4f}")

        return extrinsics_list, scale_factor

    def apply_coordinate_flip(self, extrinsics: np.ndarray) -> np.ndarray:
        """
        Apply OpenGL -> OpenCV coordinate flip.

        Copied exactly from Gradio demo.

        Args:
            extrinsics: [4, 4] camera-to-world matrix

        Returns:
            [4, 4] flipped matrix
        """
        flip_mat = np.diag([1.0, -1.0, -1.0]).astype(np.float32)
        extrinsics[:3, :3] = extrinsics[:3, :3] @ flip_mat
        return extrinsics

    def get_training_intrinsics(self) -> np.ndarray:
        """
        Get training-matched intrinsics - copied from Gradio demo.

        Returns:
            [3, 3] normalized intrinsics with fx_norm = 1.0723
        """
        return np.array([
            [TRAINING_FX_NORM, 0, 0.5],
            [0, TRAINING_FX_NORM, 0.5],
            [0, 0, 1],
        ], dtype=np.float32)

    @torch.no_grad()
    def reconstruct(
        self,
        images: List[np.ndarray],
        extrinsics: np.ndarray,
        intrinsics: np.ndarray = None,
    ) -> Tuple[Any, torch.Tensor]:
        """
        Run reconstruction - copied exactly from Gradio demo's run_inference.

        Args:
            images: List of numpy arrays (H, W, 3) in [0, 255] or [0, 1]
            extrinsics: [V, 4, 4] camera-to-world matrices (already normalized)
            intrinsics: [V, 3, 3] normalized intrinsics (uses training-matched if None)

        Returns:
            Tuple of (gaussians, context)
        """
        import time
        start_time = time.perf_counter()

        # Preprocess images
        images_tensor = self.preprocess_images(images)
        b, v, c, h, w = images_tensor.shape

        # Use training-matched intrinsics if not provided
        if intrinsics is None:
            training_intr = self.get_training_intrinsics()
            intrinsics = np.stack([training_intr.copy() for _ in range(v)])

        # Convert to tensors
        extrinsics_tensor = torch.from_numpy(extrinsics).float().unsqueeze(0).to(self.device)
        intrinsics_tensor = torch.from_numpy(intrinsics).float().unsqueeze(0).to(self.device)

        # Build batch with context and dummy target
        batch = self.build_batch(images_tensor, extrinsics_tensor, intrinsics_tensor)

        # Apply data shim (critical for proper reconstruction!)
        batch = self.data_shim(batch)
        context = batch['context']

        # Update dimensions after data shim
        h, w = context['image'].shape[-2:]

        # Run encoder
        visualization_dump = {}
        gaussians = self.encoder(context, global_step=0, deterministic=True, visualization_dump=visualization_dump)

        if isinstance(gaussians, dict):
            gaussians = gaussians.get('gaussians', gaussians)

        # Cache results
        self._last_gaussians = gaussians
        self._last_context = context
        self._last_h = h
        self._last_w = w

        inference_time = (time.perf_counter() - start_time) * 1000
        logger.debug(f"Reconstruction took {inference_time:.1f}ms")

        return gaussians, context

    @torch.no_grad()
    def render(
        self,
        gaussians,
        extrinsics: np.ndarray,
        intrinsics: np.ndarray = None,
        image_shape: Tuple[int, int] = None,
    ) -> np.ndarray:
        """
        Render Gaussians from a viewpoint - copied from Gradio demo.

        Args:
            gaussians: Gaussians from encoder
            extrinsics: [4, 4] camera-to-world matrix
            intrinsics: [3, 3] normalized intrinsics (uses training-matched if None)
            image_shape: (H, W) output shape

        Returns:
            [H, W, 3] RGB image as uint8
        """
        import time
        start_time = time.perf_counter()

        if gaussians is None:
            return np.ones((256, 256, 3), dtype=np.uint8) * 255

        # Use cached dimensions or default
        if image_shape is None:
            h = getattr(self, '_last_h', 256)
            w = getattr(self, '_last_w', 256)
        else:
            h, w = image_shape

        # Use training-matched intrinsics if not provided
        if intrinsics is None:
            intrinsics = self.get_training_intrinsics()

        # Prepare tensors [1, 1, ...]
        extrinsics_tensor = torch.from_numpy(extrinsics).float().unsqueeze(0).unsqueeze(0).to(self.device)
        intrinsics_tensor = torch.from_numpy(intrinsics).float().unsqueeze(0).unsqueeze(0).to(self.device)

        near = torch.full((1, 1), self.near, device=self.device)
        far = torch.full((1, 1), self.far, device=self.device)

        # Call decoder - exactly like Gradio demo
        output = self.decoder.forward(
            gaussians,
            extrinsics_tensor,
            intrinsics_tensor,
            near,
            far,
            (h, w),
            depth_mode=None,  # Skip depth for speed
        )

        # Extract color [1, 1, 3, H, W] -> [H, W, 3]
        color = output.color[0, 0]  # [3, H, W]
        color = color.permute(1, 2, 0)  # [H, W, 3]
        color = color.clamp(0, 1)
        color = (color * 255).byte().cpu().numpy()

        render_time = (time.perf_counter() - start_time) * 1000
        logger.debug(f"Render took {render_time:.1f}ms")

        return color

    def get_orbit_camera(self, angle_deg: float, elevation: float = 0.3) -> np.ndarray:
        """
        Get orbit camera extrinsics at a specific angle.

        Args:
            angle_deg: Azimuth angle in degrees
            elevation: Height offset (Z coordinate)

        Returns:
            [4, 4] camera-to-world matrix
        """
        angle_rad = np.radians(angle_deg)
        return create_orbit_extrinsics(
            num_views=1,
            radius=TARGET_RADIUS,
            elevation=elevation,
            azimuth_offset=angle_rad,
        )[0]
