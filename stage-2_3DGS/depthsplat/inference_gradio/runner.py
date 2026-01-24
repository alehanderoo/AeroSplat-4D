"""
DepthSplat Runner for Gradio Demo.

Handles model loading and inference for the Gradio interface.
Includes depth analysis functionality for comparing:
- Standalone Depth Anything V2 (monocular)
- Coarse cost-volume depth (multi-view stereo)
- DPT residual (learned refinement)
- Final fused depth
"""

import os
import sys
import uuid
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from einops import rearrange, repeat
from omegaconf import OmegaConf, DictConfig
import hydra
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
import matplotlib.cm as cm

# Add the parent directory to path for imports
DEPTHSPLAT_ROOT = Path(__file__).parent.parent
if str(DEPTHSPLAT_ROOT) not in sys.path:
    sys.path.insert(0, str(DEPTHSPLAT_ROOT))

from src.config import load_typed_root_config
from src.global_cfg import set_cfg
from src.model.decoder import get_decoder
from src.model.encoder import get_encoder
from src.model.model_wrapper import ModelWrapper
from src.model.ply_export import export_ply
from src.dataset.data_module import get_data_shim
from src.misc.image_io import save_video as save_video_frames

from services.gt_detection_service import create_gt_detection_service
from camera_utils import normalize_intrinsics

# Training-matched intrinsics constant
# The model was trained on Objaverse with fx_norm=1.0723 (50° FOV)
# Wild frames have fx_norm~7.6 (7.5° FOV) due to cropping from high-res telephoto cameras
# Using training-matched intrinsics improves generalization
TRAINING_FX_NORM = 1.0723


class DepthAnythingV2Wrapper:
    """
    Wrapper for standalone Depth Anything V2 inference.

    This runs the pretrained monocular depth model independently of the
    DepthSplat pipeline to evaluate raw monocular depth quality.
    """

    def __init__(self, model_type: str = "vitb", device: str = "cuda"):
        """
        Initialize Depth Anything V2 model.

        Args:
            model_type: Model variant ('vits', 'vitb', 'vitl')
            device: Device to run inference on
        """
        self.device = device
        self.model_type = model_type
        self.model = None

        # Model configurations
        self.model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        }

    def _load_model(self):
        """Lazy load the model on first use."""
        if self.model is not None:
            return

        print(f"Loading Depth Anything V2 ({self.model_type})...")

        # Try to import from the Depth-Anything-V2 repo
        depth_anything_path = DEPTHSPLAT_ROOT.parent / "Depth-Anything-V2"
        if depth_anything_path.exists():
            if str(depth_anything_path) not in sys.path:
                sys.path.insert(0, str(depth_anything_path))

            from depth_anything_v2.dpt import DepthAnythingV2

            self.model = DepthAnythingV2(**self.model_configs[self.model_type])

            # Load pretrained weights
            weight_path = DEPTHSPLAT_ROOT / "pretrained" / f"depth_anything_v2_{self.model_type}.pth"
            if weight_path.exists():
                self.model.load_state_dict(torch.load(weight_path, map_location='cpu'))
                print(f"  Loaded weights from {weight_path}")
            else:
                print(f"  Warning: Weights not found at {weight_path}")

            self.model = self.model.to(self.device).eval()
        else:
            print(f"  Warning: Depth-Anything-V2 not found at {depth_anything_path}")
            self.model = None

    @torch.no_grad()
    def infer(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """
        Run depth estimation on a list of images.

        Args:
            images: List of RGB images as numpy arrays [H, W, 3] in uint8

        Returns:
            List of depth maps as numpy arrays [H, W]
        """
        self._load_model()

        if self.model is None:
            return [np.zeros((img.shape[0], img.shape[1])) for img in images]

        depths = []
        for img in images:
            # The model expects uint8 RGB images
            depth = self.model.infer_image(img)
            depths.append(depth)

        return depths


def compute_depth_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute depth estimation metrics.

    Args:
        pred: Predicted depth [H, W]
        gt: Ground truth depth [H, W]
        mask: Optional validity mask [H, W]

    Returns:
        Dictionary of metrics:
        - abs_rel: Absolute relative error
        - sq_rel: Squared relative error
        - rmse: Root mean squared error
        - rmse_log: RMSE in log space
        - delta1: % of pixels with max(pred/gt, gt/pred) < 1.25
        - delta2: % of pixels with threshold < 1.25^2
        - delta3: % of pixels with threshold < 1.25^3
    """
    if mask is None:
        mask = (gt > 0) & np.isfinite(gt) & np.isfinite(pred) & (pred > 0)
    else:
        mask = mask & (gt > 0) & np.isfinite(gt) & np.isfinite(pred) & (pred > 0)

    if mask.sum() == 0:
        return {
            'abs_rel': float('nan'),
            'sq_rel': float('nan'),
            'rmse': float('nan'),
            'rmse_log': float('nan'),
            'delta1': float('nan'),
            'delta2': float('nan'),
            'delta3': float('nan'),
        }

    pred_m = pred[mask]
    gt_m = gt[mask]

    # Absolute relative error
    abs_rel = np.mean(np.abs(pred_m - gt_m) / gt_m)

    # Squared relative error
    sq_rel = np.mean(((pred_m - gt_m) ** 2) / gt_m)

    # RMSE
    rmse = np.sqrt(np.mean((pred_m - gt_m) ** 2))

    # RMSE log
    rmse_log = np.sqrt(np.mean((np.log(pred_m) - np.log(gt_m)) ** 2))

    # Delta thresholds
    thresh = np.maximum(pred_m / gt_m, gt_m / pred_m)
    delta1 = np.mean(thresh < 1.25) * 100
    delta2 = np.mean(thresh < 1.25 ** 2) * 100
    delta3 = np.mean(thresh < 1.25 ** 3) * 100

    return {
        'abs_rel': float(abs_rel),
        'sq_rel': float(sq_rel),
        'rmse': float(rmse),
        'rmse_log': float(rmse_log),
        'delta1': float(delta1),
        'delta2': float(delta2),
        'delta3': float(delta3),
    }


def scale_and_shift_pred(
    pred: np.ndarray,
    gt: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Apply least-squares scale and shift to align predicted depth with GT.

    This is useful for monocular depth which has scale ambiguity.
    Solves: pred_aligned = scale * pred + shift to minimize ||pred_aligned - gt||^2

    Args:
        pred: Predicted depth [H, W]
        gt: Ground truth depth [H, W]
        mask: Optional validity mask

    Returns:
        Scale-and-shift aligned prediction
    """
    if mask is None:
        mask = (gt > 0) & np.isfinite(gt) & np.isfinite(pred)

    if mask.sum() < 2:
        return pred

    pred_m = pred[mask].flatten()
    gt_m = gt[mask].flatten()

    # Solve [pred, 1] @ [scale, shift]^T = gt via least squares
    A = np.stack([pred_m, np.ones_like(pred_m)], axis=1)
    x, _, _, _ = np.linalg.lstsq(A, gt_m, rcond=None)
    scale, shift = x

    return pred * scale + shift


def apply_colormap(depth: np.ndarray, cmap_name: str = 'plasma') -> np.ndarray:
    """
    Apply a matplotlib colormap to a depth map.

    Args:
        depth: Depth values [H, W]
        cmap_name: Colormap name

    Returns:
        RGB image [H, W, 3] as uint8
    """
    # Normalize using percentiles
    valid = np.isfinite(depth) & (depth > 0)
    if valid.any():
        d_min = np.percentile(depth[valid], 2)
        d_max = np.percentile(depth[valid], 98)
        if d_max > d_min:
            depth_norm = (depth - d_min) / (d_max - d_min)
        else:
            depth_norm = np.zeros_like(depth)
    else:
        depth_norm = np.zeros_like(depth)

    depth_norm = np.clip(depth_norm, 0, 1)

    cmap = cm.get_cmap(cmap_name)
    colored = cmap(depth_norm)
    rgb = (colored[:, :, :3] * 255).astype(np.uint8)

    return rgb

# Target object coverage in the cropped frame (75% = object fills 75% of frame)
TARGET_OBJECT_COVERAGE = 0.75


def compute_object_physical_size(
    bbox_pixels: float,
    fx_pixels: float,
    distance_meters: float
) -> float:
    """
    Compute object physical size from its projection.

    For a pinhole camera: projected_size = fx * (real_size / distance)
    Therefore: real_size = projected_size * distance / fx

    Args:
        bbox_pixels: Object size in pixels in the original image
        fx_pixels: Focal length in pixels
        distance_meters: Distance to object in meters

    Returns:
        Estimated physical size of object in meters
    """
    return bbox_pixels * distance_meters / fx_pixels


def compute_virtual_distance(
    object_size_meters: float,
    target_coverage: float,
    fx_norm: float
) -> float:
    """
    Compute virtual camera distance for target object coverage.

    We want the object to fill target_coverage fraction of the frame.
    With training-matched intrinsics (fx_norm), the projection is:
        proj_size = fx_norm * crop_size * (object_size / distance)

    For target coverage (e.g., 75%):
        target_pixels = target_coverage * crop_size = fx_norm * crop_size * (object_size / distance)
        distance = fx_norm * object_size / target_coverage

    Args:
        object_size_meters: Physical size of object in meters
        target_coverage: Target fraction of frame that object should fill
        fx_norm: Normalized focal length (training matched)

    Returns:
        Virtual camera distance in meters
    """
    return fx_norm * object_size_meters / target_coverage


def create_virtual_camera_extrinsics(
    original_c2w: np.ndarray,
    object_position: np.ndarray,
    virtual_distance: float
) -> np.ndarray:
    """
    Create virtual camera extrinsics at a closer distance.

    Maintains the same viewing direction (line from camera to object)
    but positions the camera at the virtual distance from the object.

    Args:
        original_c2w: [4, 4] original camera-to-world matrix
        object_position: [3] 3D position of the object
        virtual_distance: Desired distance from object in meters

    Returns:
        [4, 4] virtual camera-to-world matrix
    """
    # Extract original camera position
    cam_pos_orig = original_c2w[:3, 3].copy()

    # Direction from object to camera (viewing direction is opposite)
    direction = cam_pos_orig - object_position
    direction = direction / np.linalg.norm(direction)

    # New camera position at virtual distance
    cam_pos_virtual = object_position + direction * virtual_distance

    # Create new extrinsics looking at the object
    world_up = np.array([0, 0, 1], dtype=np.float32)

    # Forward direction (camera looks along +Z in OpenCV)
    forward = object_position - cam_pos_virtual
    forward = forward / np.linalg.norm(forward)

    # Right direction
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-6:
        # Camera looking straight up/down, use alternative up
        world_up = np.array([0, 1, 0], dtype=np.float32)
        right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)

    # Recompute up to be orthogonal
    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)

    # Build rotation matrix - OpenCV convention (Y points down)
    down = -up
    R = np.stack([right, down, forward], axis=1).astype(np.float32)

    # Build 4x4 matrix
    c2w_virtual = np.eye(4, dtype=np.float32)
    c2w_virtual[:3, :3] = R
    c2w_virtual[:3, 3] = cam_pos_virtual

    return c2w_virtual


class DepthSplatRunner:
    """Runner class for DepthSplat inference."""

    def __init__(
        self,
        checkpoint_path: str,
        config_name: str = "objaverse_white",
        device: str = "cuda",
    ):
        """
        Initialize the DepthSplat runner.

        Args:
            checkpoint_path: Path to the model checkpoint (.ckpt file)
            config_name: Name of experiment config (without .yaml extension)
            device: Device to run inference on
        """
        self.device = device
        self.checkpoint_path = checkpoint_path

        # Load config using Hydra compose
        print(f"Loading config: {config_name}")
        self.cfg_dict = self._load_config(config_name)
        self.cfg = load_typed_root_config(self.cfg_dict)
        set_cfg(self.cfg_dict)

        # Build model
        print("Building model...")
        self.encoder, self.encoder_visualizer = get_encoder(self.cfg.model.encoder)
        self.decoder = get_decoder(self.cfg.model.decoder, self.cfg.dataset)

        # Load checkpoint
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

        # Filter state dict for encoder and decoder
        encoder_state = {k.replace('encoder.', ''): v for k, v in state_dict.items() if k.startswith('encoder.')}
        decoder_state = {k.replace('decoder.', ''): v for k, v in state_dict.items() if k.startswith('decoder.')}

        self.encoder.load_state_dict(encoder_state, strict=False)
        self.decoder.load_state_dict(decoder_state, strict=False)

        self.encoder = self.encoder.to(device).eval()
        self.decoder = self.decoder.to(device).eval()

        # Get data shim for preprocessing
        self.data_shim = get_data_shim(self.encoder)

        # Store config values
        self.image_shape = self.cfg.dataset.image_shape
        self.near = self.cfg.dataset.near
        self.far = self.cfg.dataset.far
        self.background_color = self.cfg.dataset.background_color

        # Initialize standalone Depth Anything V2 for depth analysis
        # Uses same model type as the encoder's monocular branch
        self.depth_anything = DepthAnythingV2Wrapper(
            model_type=self.cfg.model.encoder.monodepth_vit_type,
            device=device,
        )

        print("Model loaded successfully!")

    def _load_config(self, config_name: str) -> DictConfig:
        """Load configuration using Hydra compose API."""
        config_dir = str(DEPTHSPLAT_ROOT / "config")

        # Clear any existing Hydra instance
        GlobalHydra.instance().clear()

        # Initialize Hydra with the config directory
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            # Compose the config with experiment overrides
            cfg = compose(
                config_name="main",
                overrides=[f"+experiment={config_name}", "mode=test"],
            )

        return cfg

    def preprocess_images(
        self,
        images: List[np.ndarray],
        target_size: Tuple[int, int] = None,
    ) -> torch.Tensor:
        """
        Preprocess input images.

        Args:
            images: List of numpy arrays (H, W, 3) in [0, 255]
            target_size: Target size (H, W) for resizing

        Returns:
            Tensor of shape [1, V, 3, H, W] in [0, 1]
        """
        if target_size is None:
            target_size = self.image_shape

        processed = []
        for img in images:
            # Convert to tensor
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

        # Stack and add batch dimension
        images_tensor = torch.stack(processed, dim=0).unsqueeze(0)  # [1, V, 3, H, W]
        return images_tensor.to(self.device)

    def build_batch(
        self,
        images: torch.Tensor,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        near: float = None,
        far: float = None,
    ) -> dict:
        """
        Build batch dictionary for the encoder (including context and dummy target for shim).

        Args:
            images: [1, V, 3, H, W] tensor
            extrinsics: [1, V, 4, 4] camera-to-world matrices
            intrinsics: [1, V, 3, 3] normalized intrinsics
            near: Near plane distance
            far: Far plane distance

        Returns:
            Batch dictionary with context and target
        """
        b, v, _, h, w = images.shape

        if near is None:
            near = self.near
        if far is None:
            far = self.far

        context = {
            'image': images,
            'extrinsics': extrinsics.to(self.device),
            'intrinsics': intrinsics.to(self.device),
            'near': torch.full((b, v), near, device=self.device),
            'far': torch.full((b, v), far, device=self.device),
            'index': torch.arange(v, device=self.device).unsqueeze(0),
        }

        # Create a dummy target for the data shim (uses first context view)
        target = {
            'image': images[:, :1],
            'extrinsics': extrinsics[:, :1].to(self.device),
            'intrinsics': intrinsics[:, :1].to(self.device),
            'near': torch.full((b, 1), near, device=self.device),
            'far': torch.full((b, 1), far, device=self.device),
            'index': torch.zeros(b, 1, dtype=torch.long, device=self.device),
        }

        batch = {
            'context': context,
            'target': target,
            'scene': ['gradio_input'],
        }

        return batch

    def load_wild_frame(
        self,
        frame_id: int,
        render_dir: str,
        cache_dir: str = None,
        use_virtual_cameras: bool = True,
    ) -> Tuple[List[str], str]:
        """
        Load a specific frame from "in-the-wild" renders.

        Supports virtual camera positioning to address the FOV/distance mismatch
        between wild frames (telephoto lens, far away) and training data (50° FOV, close).

        When use_virtual_cameras=True:
        - Computes object physical size from bbox and depth
        - Calculates virtual camera distance where object fills ~75% of frame
        - Updates extrinsics to position cameras at virtual distance
        - Results in geometrically consistent intrinsics + extrinsics

        Args:
            frame_id: Frame index (0-119)
            render_dir: Path to render directory
            cache_dir: Temp directory
            use_virtual_cameras: If True, compute virtual camera positions based on
                                 object size. This maintains geometric consistency
                                 between the training-matched intrinsics and extrinsics.

        Returns:
            Tuple of (list of image paths, status string)
        """
        if cache_dir is None:
            cache_dir = "/tmp/depthsplat_gradio_wild"
        os.makedirs(cache_dir, exist_ok=True)

        print(f"Loading frame {frame_id} from {render_dir}")
        if use_virtual_cameras:
            print(f"Virtual cameras enabled (target coverage: {TARGET_OBJECT_COVERAGE*100:.0f}%)")

        # Initialize service to get detections and camera info
        try:
            service = create_gt_detection_service(render_dir)
        except Exception as e:
            raise RuntimeError(f"Failed to init detection service: {str(e)}")

        detections = service.get_detections(frame_id)
        if not detections:
            raise RuntimeError(f"No detections found for frame {frame_id}")

        image_paths = []
        extrinsics_list = []
        intrinsics_list = []
        camera_info_list = []  # Store per-camera info for virtual camera computation

        # Output size (model expects 256x256)
        OUTPUT_SIZE = 256
        # Margin around object in tight crop (percentage)
        CROP_MARGIN = 0.15

        # Iterate over cameras in order
        for cam_name in sorted(service.camera_names):
            # Load image
            # Structure: {render_dir}/{cam_name}/rgb/rgb_{frame_id:04d}.png
            img_path = Path(render_dir) / cam_name / "rgb" / f"rgb_{frame_id:04d}.png"
            if not img_path.exists():
                print(f"Warning: Image not found {img_path}")
                continue

            try:
                img = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"Failed to load {img_path}: {e}")
                continue

            w, h = img.size

            # Get detection for cropping
            det = detections.get_detection(cam_name)
            if not det:
                print(f"No detection for {cam_name} in frame {frame_id}")
                continue

            # Get bbox info
            bbox = det.bbox
            if bbox:
                bbox_width = bbox[2] - bbox[0]
                bbox_height = bbox[3] - bbox[1]
                bbox_size = max(bbox_width, bbox_height)
            else:
                bbox_size = 68.0  # Default fallback

            # --- Determine crop size based on virtual camera mode ---
            if use_virtual_cameras:
                # TIGHT CROP: Crop just around the object with margin, then resize to 256x256
                # This makes the object appear larger in the final image
                # Target: object fills TARGET_OBJECT_COVERAGE of the output
                # crop_size = bbox_size / TARGET_OBJECT_COVERAGE
                # Add margin for safety
                tight_crop_size = int(bbox_size / TARGET_OBJECT_COVERAGE * (1 + CROP_MARGIN))
                # Ensure minimum size for quality
                tight_crop_size = max(tight_crop_size, 64)
                crop_size = tight_crop_size
            else:
                # Standard fixed-size crop
                crop_size = OUTPUT_SIZE

            # Calculate crop region centered on object
            crop = det.get_crop_region(crop_size=crop_size, image_width=w, image_height=h)
            x1, y1, x2, y2 = crop

            # Crop image
            img_crop = img.crop(crop)

            # Load and apply mask for white background (matching training data)
            mask_path = Path(render_dir) / cam_name / "mask" / f"drone_mask_{frame_id:04d}.png"
            if mask_path.exists():
                mask = Image.open(mask_path).convert("L")  # Grayscale
                mask_crop = mask.crop(crop)  # Same crop as RGB

                # Convert to numpy for compositing
                img_np = np.array(img_crop)
                mask_np = np.array(mask_crop)

                # Create white background
                white_bg = np.ones_like(img_np) * 255

                # Composite: foreground where mask > 0, white elsewhere
                fg_mask = (mask_np > 127)[..., np.newaxis]  # Threshold and expand dims
                img_np = np.where(fg_mask, img_np, white_bg)

                img_crop = Image.fromarray(img_np.astype(np.uint8))
            else:
                print(f"Warning: Mask not found at {mask_path}, using original image")

            # Resize to output size if needed (for virtual camera tight crops)
            if img_crop.size[0] != OUTPUT_SIZE or img_crop.size[1] != OUTPUT_SIZE:
                img_crop = img_crop.resize((OUTPUT_SIZE, OUTPUT_SIZE), Image.LANCZOS)

            # Save masked image for UI
            save_path = os.path.join(cache_dir, f"{cam_name}_frame{frame_id}.png")
            img_crop.save(save_path)
            image_paths.append(save_path)

            # --- Handle Intrinsics ---
            intr_raw = service.get_camera_intrinsics(cam_name)
            if intr_raw is None:
                raise RuntimeError(f"No intrinsics for {cam_name}")

            # Parse intrinsics (handle dict or list)
            if isinstance(intr_raw, dict):
                fx = intr_raw.get('fx')
                fy = intr_raw.get('fy')
                cx = intr_raw.get('cx')
                cy = intr_raw.get('cy')
            elif isinstance(intr_raw, list) and len(intr_raw) == 3:
                # Assume 3x3 matrix
                K_raw = np.array(intr_raw)
                fx, fy = K_raw[0, 0], K_raw[1, 1]
                cx, cy = K_raw[0, 2], K_raw[1, 2]
            else:
                raise RuntimeError(f"Unknown intrinsics format for {cam_name}")

            # Adjust principal point for crop
            new_cx = cx - x1
            new_cy = cy - y1

            # Build new intrinsics matrix (pixel coordinates relative to crop)
            K_new = np.array([
                [fx, 0, new_cx],
                [0, fy, new_cy],
                [0, 0, 1]
            ], dtype=np.float32)

            # If we resized the crop, adjust intrinsics accordingly
            if crop_size != OUTPUT_SIZE:
                resize_scale = OUTPUT_SIZE / crop_size
                K_new[0, :] *= resize_scale  # Scale fx, cx
                K_new[1, :] *= resize_scale  # Scale fy, cy

            # Normalize intrinsics relative to the OUTPUT size (256x256)
            K_norm = normalize_intrinsics(K_new, (OUTPUT_SIZE, OUTPUT_SIZE))
            intrinsics_list.append(K_norm)

            # --- Handle Extrinsics ---
            ext_raw = service.get_camera_extrinsics(cam_name)
            if ext_raw is None:
                raise RuntimeError(f"No extrinsics for {cam_name}")

            ext_matrix = np.array(ext_raw, dtype=np.float32)

            # --- Coordinate System Fix ---
            # Flip Y and Z axes (OpenGL -> OpenCV convention)
            flip_mat = np.diag([1.0, -1.0, -1.0]).astype(np.float32)
            ext_matrix[:3, :3] = ext_matrix[:3, :3] @ flip_mat

            extrinsics_list.append(ext_matrix)

            # Store camera info for virtual camera computation
            camera_info_list.append({
                'fx': fx,
                'depth': det.depth if det.depth else 10.0,  # Fallback depth
                'bbox_size': bbox_size,
                'crop_size': crop_size,
            })

        if not image_paths:
            raise RuntimeError(f"Failed to load any valid views for frame {frame_id}")

        # Get object 3D position (needed for virtual cameras and centering)
        if detections.object_position_3d:
            object_position = np.array(detections.object_position_3d, dtype=np.float32)
        else:
            # Fallback if no 3D position
            print("Warning: No drone_position_3d found, using mean camera target")
            # Estimate by averaging where cameras are looking
            object_position = np.mean([ext[:3, 3] for ext in extrinsics_list], axis=0)

        # --- Report crop info for debugging ---
        if use_virtual_cameras and camera_info_list:
            crop_sizes = [info['crop_size'] for info in camera_info_list]
            bbox_sizes = [info['bbox_size'] for info in camera_info_list]
            print(f"Tight cropping enabled:")
            print(f"  Bbox sizes: {[f'{s:.0f}' for s in bbox_sizes]} px")
            print(f"  Crop sizes: {crop_sizes} px (resized to 256x256)")
            print(f"  Object fills ~{TARGET_OBJECT_COVERAGE*100:.0f}% of output")
            # Note: Using original extrinsics (no virtual camera repositioning)
            # The tight crop makes the object appear larger, and training-matched
            # intrinsics are applied later to keep input in-distribution

        # --- Pose Normalization ---
        # 1. Center logic (use object_position computed above)
        center = object_position
        print(f"Normalizing pose: Centering at {center}")

        # Apply centering
        for ext in extrinsics_list:
             ext[:3, 3] -= center

        # 2. Scaling
        # Calculate mean distance from origin (which is now the object center)
        distances = [np.linalg.norm(ext[:3, 3]) for ext in extrinsics_list]
        mean_dist = np.mean(distances)
        
        target_radius = 2.0
        scale_factor = target_radius / mean_dist
        print(f"Normalizing pose: Scaling by {scale_factor:.4f} (original dist: {mean_dist:.2f} -> 2.0)")

        for ext in extrinsics_list:
            ext[:3, 3] *= scale_factor

        # --- Override intrinsics with training-matched values ---
        # The model was trained on 50° FOV images. Wild frames have ~7.5° FOV due to
        # cropping from high-res telephoto cameras. Using training-matched intrinsics
        # keeps the input in-distribution and produces better reconstructions.
        training_intrinsics = np.array([
            [TRAINING_FX_NORM, 0, 0.5],
            [0, TRAINING_FX_NORM, 0.5],
            [0, 0, 1],
        ], dtype=np.float32)
        intrinsics_list = [training_intrinsics.copy() for _ in intrinsics_list]

        # Calculate input camera elevation range for default render camera
        extrinsics_array = np.stack(extrinsics_list)
        positions = extrinsics_array[:, :3, 3]
        distances = np.linalg.norm(positions, axis=1)
        elevations = np.rad2deg(np.arcsin(positions[:, 2] / distances))
        mean_elevation = float(np.mean(elevations))

        # Store context
        self.current_example = {
            'key': f"wild_{frame_id}",
            'extrinsics': extrinsics_array,  # [V, 4, 4]
            'intrinsics': np.stack(intrinsics_list),  # [V, 3, 3]
            'render_dir': render_dir,
            'scale_factor': scale_factor,
            'center': center,
            'mean_elevation': mean_elevation,  # For default render camera
            'use_virtual_cameras': use_virtual_cameras,  # Track whether virtual cameras were used
        }

        return image_paths, f"Loaded Frame {frame_id} ({len(image_paths)} views)"

    @torch.no_grad()
    def run_inference(
        self,
        images: List[np.ndarray],
        extrinsics: np.ndarray,
        intrinsics: np.ndarray,
        target_extrinsics: np.ndarray,
        target_intrinsics: np.ndarray = None,
        output_dir: str = None,
        num_video_frames: int = 60,
    ) -> dict:
        """
        Run inference on input images.

        Args:
            images: List of numpy arrays (H, W, 3) in [0, 255]
            extrinsics: [V, 4, 4] camera-to-world matrices
            intrinsics: [V, 3, 3] normalized intrinsics
            target_extrinsics: [T, 4, 4] target camera-to-world matrices
            target_intrinsics: [T, 3, 3] target intrinsics (uses first context if None)
            output_dir: Directory to save outputs
            num_video_frames: Number of frames for video

        Returns:
            Dictionary with rendered images, video path, and PLY path
        """
        torch.cuda.empty_cache()

        # Create output directory
        if output_dir is None:
            output_dir = f"/tmp/depthsplat_output_{uuid.uuid4()}"
        os.makedirs(output_dir, exist_ok=True)

        # Preprocess images
        images_tensor = self.preprocess_images(images)
        b, v, c, h, w = images_tensor.shape

        # Convert camera parameters to tensors
        extrinsics_tensor = torch.from_numpy(extrinsics).float().unsqueeze(0).to(self.device)  # [1, V, 4, 4]
        intrinsics_tensor = torch.from_numpy(intrinsics).float().unsqueeze(0).to(self.device)  # [1, V, 3, 3]

        # Build batch with context and dummy target
        batch = self.build_batch(images_tensor, extrinsics_tensor, intrinsics_tensor)

        # Apply data shim (preprocessing expected by encoder - handles patch alignment)
        batch = self.data_shim(batch)
        context = batch['context']

        # Update h, w after potential cropping by data shim
        h, w = context['image'].shape[-2:]

        # Run encoder to get Gaussians
        print("Running encoder...")
        visualization_dump = {}  # Will be populated with scales/rotations for PLY export
        gaussians = self.encoder(context, global_step=0, deterministic=True, visualization_dump=visualization_dump)

        if isinstance(gaussians, dict):
            gaussians = gaussians['gaussians']

        # Store visualization dump for PLY export
        self._visualization_dump = visualization_dump
        self._context = context

        # Prepare target camera
        target_extrinsics_tensor = torch.from_numpy(target_extrinsics).float().unsqueeze(0).to(self.device)
        if target_intrinsics is None:
            target_intrinsics_tensor = intrinsics_tensor[:, :1].expand(-1, target_extrinsics_tensor.shape[1], -1, -1)
        else:
            target_intrinsics_tensor = torch.from_numpy(target_intrinsics).float().unsqueeze(0).to(self.device)

        t = target_extrinsics_tensor.shape[1]
        target_near = torch.full((1, t), self.near, device=self.device)
        target_far = torch.full((1, t), self.far, device=self.device)

        # Render from target viewpoint
        print("Rendering from target viewpoint...")
        output = self.decoder.forward(
            gaussians,
            target_extrinsics_tensor,
            target_intrinsics_tensor,
            target_near,
            target_far,
            (h, w),
            depth_mode="depth",  # Enable depth rendering
        )

        rendered_images = output.color[0]  # [T, 3, H, W]
        rendered_depth = output.depth[0] if output.depth is not None else None  # [T, H, W]
        rendered_alpha = output.alpha[0] if output.alpha is not None else None  # [T, H, W]

        # Convert to numpy
        rendered_np = (rendered_images.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)

        # Save single rendered image
        result_image_path = os.path.join(output_dir, "rendered.png")
        Image.fromarray(rendered_np[0]).save(result_image_path)

        # Save depth visualization
        depth_image_path = None
        if rendered_depth is not None:
            depth_np = rendered_depth[0].cpu().numpy()  # [H, W]
            # Normalize depth for visualization (ignore invalid/inf values)
            valid_mask = np.isfinite(depth_np) & (depth_np > 0)
            if valid_mask.any():
                d_min = depth_np[valid_mask].min()
                d_max = depth_np[valid_mask].max()
                if d_max > d_min:
                    depth_normalized = (depth_np - d_min) / (d_max - d_min)
                else:
                    depth_normalized = np.zeros_like(depth_np)
                depth_normalized = np.clip(depth_normalized, 0, 1)
                # Apply colormap (viridis-like: close=yellow, far=purple)
                depth_colored = self._apply_depth_colormap(depth_normalized)
                depth_image_path = os.path.join(output_dir, "depth.png")
                Image.fromarray(depth_colored).save(depth_image_path)

        # Save silhouette/alpha visualization
        silhouette_image_path = None
        if rendered_alpha is not None:
            alpha_np = rendered_alpha[0].cpu().numpy()  # [H, W]
            alpha_np = np.clip(alpha_np, 0, 1)
            # Convert to grayscale image
            silhouette_np = (alpha_np * 255).astype(np.uint8)
            silhouette_image_path = os.path.join(output_dir, "silhouette.png")
            Image.fromarray(silhouette_np, mode='L').save(silhouette_image_path)

        # Generate 360 videos if requested
        video_rgb_path = None
        video_depth_path = None
        video_silhouette_path = None
        if num_video_frames > 0:
            video_rgb_path, video_depth_path, video_silhouette_path = self._generate_video(
                gaussians, intrinsics_tensor[:, 0], output_dir, num_video_frames, h, w
            )

        # Export PLY (only if we have visualization dump with scales/rotations)
        ply_path = os.path.join(output_dir, "gaussians.ply")
        if self._visualization_dump and 'scales' in self._visualization_dump:
            self._export_ply_with_dump(gaussians, ply_path, v, h, w)
        else:
            print("Skipping PLY export (no visualization dump available)")
            ply_path = None

        # === DEPTH ANALYSIS: Extract and visualize all 4 depth types ===
        depth_analysis = self._compute_depth_analysis(
            images=images,
            visualization_dump=self._visualization_dump,
            output_dir=output_dir,
        )

        return {
            'rendered_images': rendered_np,
            'result_image_path': result_image_path,
            'depth_image_path': depth_image_path,
            'silhouette_image_path': silhouette_image_path,
            'video_rgb_path': video_rgb_path,
            'video_depth_path': video_depth_path,
            'video_silhouette_path': video_silhouette_path,
            'ply_path': ply_path,
            'output_dir': output_dir,
            'mono_depth_paths': depth_analysis.get('residual_paths', []),  # Backward compat
            'depth_analysis': depth_analysis,
        }

    def _compute_depth_analysis(
        self,
        images: List[np.ndarray],
        visualization_dump: dict,
        output_dir: str,
    ) -> dict:
        """
        Compute comprehensive depth analysis with all 4 depth types.

        Args:
            images: Input images [V, H, W, 3]
            visualization_dump: Dictionary from encoder containing depth tensors
            output_dir: Directory to save visualizations

        Returns:
            Dictionary containing:
            - standalone_da_paths: Paths to standalone Depth Anything V2 visualizations
            - coarse_mv_paths: Paths to coarse cost-volume depth visualizations
            - residual_paths: Paths to DPT residual visualizations
            - final_fused_paths: Paths to final fused depth visualizations
            - gt_paths: Paths to GT depth visualizations (if available)
            - metrics: Per-view metrics comparing each depth type to GT
        """
        depth_dir = os.path.join(output_dir, "depth_analysis")
        os.makedirs(depth_dir, exist_ok=True)

        num_views = len(images)
        result = {
            'standalone_da_paths': [],
            'coarse_mv_paths': [],
            'residual_paths': [],
            'final_fused_paths': [],
            'gt_paths': [],
            'metrics': {},
        }

        # 1. Run standalone Depth Anything V2 on input images
        print("Running standalone Depth Anything V2...")
        standalone_depths = self.depth_anything.infer(images)

        for i, depth in enumerate(standalone_depths):
            viz = apply_colormap(depth, 'plasma')
            path = os.path.join(depth_dir, f"standalone_da_view_{i}.png")
            Image.fromarray(viz).save(path)
            result['standalone_da_paths'].append(path)

        # 2. Extract coarse cost-volume depth (multi-view stereo result)
        coarse_mv_depths = []
        if visualization_dump and 'coarse_mv_depth' in visualization_dump:
            coarse_mv = visualization_dump['coarse_mv_depth'][0].cpu().numpy()  # [V, H, W]
            for i in range(coarse_mv.shape[0]):
                # Convert from inverse depth to depth
                depth = 1.0 / np.clip(coarse_mv[i], 1e-6, None)
                coarse_mv_depths.append(depth)
                viz = apply_colormap(depth, 'plasma')
                path = os.path.join(depth_dir, f"coarse_mv_view_{i}.png")
                Image.fromarray(viz).save(path)
                result['coarse_mv_paths'].append(path)

        # 3. Extract DPT residual (monocular refinement)
        residual_depths = []
        if visualization_dump and 'mono_depth' in visualization_dump:
            residual = visualization_dump['mono_depth'][0].cpu().numpy()  # [V, H, W]
            for i in range(residual.shape[0]):
                residual_depths.append(residual[i])
                # Residual can be positive/negative, use diverging colormap
                viz = apply_colormap(residual[i], 'coolwarm')
                path = os.path.join(depth_dir, f"residual_view_{i}.png")
                Image.fromarray(viz).save(path)
                result['residual_paths'].append(path)

        # 4. Compute final fused depth (coarse + residual)
        final_fused_depths = []
        if visualization_dump and 'depth' in visualization_dump:
            # depth in visualization_dump is [B, V, H, W, 1, 1] inverse depth
            fused = visualization_dump['depth'][0].squeeze(-1).squeeze(-1).cpu().numpy()  # [V, H, W]
            for i in range(fused.shape[0]):
                # Convert from inverse depth to depth
                depth = 1.0 / np.clip(fused[i], 1e-6, None)
                final_fused_depths.append(depth)
                viz = apply_colormap(depth, 'plasma')
                path = os.path.join(depth_dir, f"final_fused_view_{i}.png")
                Image.fromarray(viz).save(path)
                result['final_fused_paths'].append(path)

        # 5. Load GT depth if available (for wild frames from Isaac Sim)
        gt_depths = self._load_gt_depths(num_views)
        if gt_depths:
            for i, gt_depth in enumerate(gt_depths):
                viz = apply_colormap(gt_depth, 'plasma')
                path = os.path.join(depth_dir, f"gt_view_{i}.png")
                Image.fromarray(viz).save(path)
                result['gt_paths'].append(path)

            # 6. Compute metrics comparing each depth type to GT
            result['metrics'] = self._compute_all_metrics(
                standalone_depths=standalone_depths,
                coarse_mv_depths=coarse_mv_depths,
                final_fused_depths=final_fused_depths,
                gt_depths=gt_depths,
            )

        return result

    def _load_gt_depths(self, num_views: int) -> Optional[List[np.ndarray]]:
        """
        Load ground truth depth maps for wild frames.

        Returns:
            List of GT depth arrays, or None if not available
        """
        if not hasattr(self, 'current_example') or self.current_example is None:
            return None

        render_dir = self.current_example.get('render_dir')
        if render_dir is None:
            return None

        # Check if this is a wild frame with GT depth
        key = self.current_example.get('key', '')
        if not key.startswith('wild_'):
            return None

        frame_id = int(key.replace('wild_', ''))
        render_dir = Path(render_dir)

        gt_depths = []
        cam_names = sorted([d.name for d in render_dir.iterdir() if d.is_dir() and d.name.startswith('cam_')])

        for cam_name in cam_names[:num_views]:
            depth_path = render_dir / cam_name / "depth" / f"distance_to_image_plane_{frame_id:04d}.npy"
            if depth_path.exists():
                gt_depth = np.load(depth_path)
                # Resize to match model output (256x256)
                if gt_depth.shape != (256, 256):
                    from scipy.ndimage import zoom
                    scale = (256 / gt_depth.shape[0], 256 / gt_depth.shape[1])
                    gt_depth = zoom(gt_depth, scale, order=1)
                gt_depths.append(gt_depth)
            else:
                return None  # If any camera is missing, skip GT comparison

        return gt_depths if len(gt_depths) == num_views else None

    def _compute_all_metrics(
        self,
        standalone_depths: List[np.ndarray],
        coarse_mv_depths: List[np.ndarray],
        final_fused_depths: List[np.ndarray],
        gt_depths: List[np.ndarray],
    ) -> dict:
        """
        Compute depth metrics for all depth types vs GT.

        Returns:
            Dictionary with metrics per depth type, averaged across views
        """
        metrics = {
            'standalone_da': {'raw': [], 'aligned': []},
            'coarse_mv': {'raw': [], 'aligned': []},
            'final_fused': {'raw': [], 'aligned': []},
        }

        for i, gt in enumerate(gt_depths):
            # Standalone Depth Anything V2
            if i < len(standalone_depths):
                pred = standalone_depths[i]
                # Resize pred to match GT if needed
                if pred.shape != gt.shape:
                    from scipy.ndimage import zoom
                    scale = (gt.shape[0] / pred.shape[0], gt.shape[1] / pred.shape[1])
                    pred = zoom(pred, scale, order=1)
                metrics['standalone_da']['raw'].append(compute_depth_metrics(pred, gt))
                pred_aligned = scale_and_shift_pred(pred, gt)
                metrics['standalone_da']['aligned'].append(compute_depth_metrics(pred_aligned, gt))

            # Coarse MV depth
            if i < len(coarse_mv_depths):
                pred = coarse_mv_depths[i]
                if pred.shape != gt.shape:
                    from scipy.ndimage import zoom
                    scale = (gt.shape[0] / pred.shape[0], gt.shape[1] / pred.shape[1])
                    pred = zoom(pred, scale, order=1)
                metrics['coarse_mv']['raw'].append(compute_depth_metrics(pred, gt))
                pred_aligned = scale_and_shift_pred(pred, gt)
                metrics['coarse_mv']['aligned'].append(compute_depth_metrics(pred_aligned, gt))

            # Final fused depth
            if i < len(final_fused_depths):
                pred = final_fused_depths[i]
                if pred.shape != gt.shape:
                    from scipy.ndimage import zoom
                    scale = (gt.shape[0] / pred.shape[0], gt.shape[1] / pred.shape[1])
                    pred = zoom(pred, scale, order=1)
                metrics['final_fused']['raw'].append(compute_depth_metrics(pred, gt))
                pred_aligned = scale_and_shift_pred(pred, gt)
                metrics['final_fused']['aligned'].append(compute_depth_metrics(pred_aligned, gt))

        # Average metrics across views
        def avg_metrics(metric_list):
            if not metric_list:
                return {}
            keys = metric_list[0].keys()
            return {k: np.mean([m[k] for m in metric_list if not np.isnan(m[k])]) for k in keys}

        return {
            'standalone_da_raw': avg_metrics(metrics['standalone_da']['raw']),
            'standalone_da_aligned': avg_metrics(metrics['standalone_da']['aligned']),
            'coarse_mv_raw': avg_metrics(metrics['coarse_mv']['raw']),
            'coarse_mv_aligned': avg_metrics(metrics['coarse_mv']['aligned']),
            'final_fused_raw': avg_metrics(metrics['final_fused']['raw']),
            'final_fused_aligned': avg_metrics(metrics['final_fused']['aligned']),
        }

    def _apply_depth_colormap(self, depth_normalized: np.ndarray) -> np.ndarray:
        """
        Apply a colormap to normalized depth values.

        Args:
            depth_normalized: Normalized depth values in [0, 1] where 0=close, 1=far

        Returns:
            RGB image as uint8 array [H, W, 3]
        """
        # Use a turbo-like colormap (close=red/yellow, far=blue/purple)
        # This is a simplified version - could use matplotlib colormaps if available
        h, w = depth_normalized.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)

        # Invert so close is bright (yellow/red), far is dark (blue/purple)
        d = 1.0 - depth_normalized

        # Simple turbo-like colormap
        # Red channel: peaks at d=0.75
        rgb[..., 0] = np.clip(255 * (1.5 - np.abs(d - 0.75) * 4), 0, 255).astype(np.uint8)
        # Green channel: peaks at d=0.5
        rgb[..., 1] = np.clip(255 * (1.5 - np.abs(d - 0.5) * 4), 0, 255).astype(np.uint8)
        # Blue channel: peaks at d=0.25
        rgb[..., 2] = np.clip(255 * (1.5 - np.abs(d - 0.25) * 4), 0, 255).astype(np.uint8)

        return rgb

    def _apply_depth_colormap_mono(self, depth: np.ndarray) -> np.ndarray:
        """
        Apply a colormap to monocular depth values (residual depth from DPT upsampler).

        The mono_depth represents depth refinement from monocular features and can be
        positive or negative. We normalize based on percentiles to handle outliers.

        Args:
            depth: Raw depth values [H, W]

        Returns:
            RGB image as uint8 array [H, W, 3]
        """
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        # Normalize using percentiles to handle outliers
        valid_mask = np.isfinite(depth)
        if valid_mask.any():
            d_min = np.percentile(depth[valid_mask], 2)
            d_max = np.percentile(depth[valid_mask], 98)
            if d_max > d_min:
                depth_normalized = (depth - d_min) / (d_max - d_min)
            else:
                depth_normalized = np.zeros_like(depth)
        else:
            depth_normalized = np.zeros_like(depth)

        depth_normalized = np.clip(depth_normalized, 0, 1)

        # Use plasma colormap (similar to what's used in depth visualization)
        cmap = cm.get_cmap('plasma')
        colored = cmap(depth_normalized)
        rgb = (colored[:, :, :3] * 255).astype(np.uint8)

        return rgb

    def _generate_video(
        self,
        gaussians,
        base_intrinsics: torch.Tensor,
        output_dir: str,
        num_frames: int,
        h: int,
        w: int,
    ) -> Tuple[str, str, str]:
        """Generate 360-degree rotation videos for RGB, depth, and silhouette."""
        print(f"Generating {num_frames}-frame videos (RGB, depth, silhouette)...")

        # Generate circular camera trajectory
        # Use parameters similar to GT cameras from Objaverse dataset
        angles = np.linspace(0, 2 * np.pi, num_frames, endpoint=False)
        radius = 1.4  # Distance from object center (matches GT camera range)
        elevation_angle = np.deg2rad(30)  # 30 degrees above horizontal

        video_extrinsics = []
        for angle in angles:
            # Camera position on sphere using spherical coordinates
            # x = r * cos(el) * cos(az), y = r * cos(el) * sin(az), z = r * sin(el)
            x = radius * np.cos(elevation_angle) * np.cos(angle)
            y = radius * np.cos(elevation_angle) * np.sin(angle)
            z = radius * np.sin(elevation_angle)

            # Look at origin
            cam_pos = np.array([x, y, z])
            forward = -cam_pos  # Points from camera to origin
            forward = forward / np.linalg.norm(forward)

            # Up vector
            up = np.array([0, 0, 1])

            # Right vector
            right = np.cross(forward, up)
            right = right / np.linalg.norm(right)

            # Recompute up
            up = np.cross(right, forward)

            # Build rotation matrix (camera-to-world)
            # OpenCV convention: camera +X=right, +Y=down, +Z=forward (optical axis)
            down = -up
            R = np.stack([right, down, forward], axis=1)

            # Build 4x4 extrinsics (camera-to-world)
            extrinsic = np.eye(4, dtype=np.float32)
            extrinsic[:3, :3] = R
            extrinsic[:3, 3] = cam_pos

            video_extrinsics.append(extrinsic)

        video_extrinsics = torch.from_numpy(np.stack(video_extrinsics)).float().unsqueeze(0).to(self.device)
        video_intrinsics = base_intrinsics.unsqueeze(1).expand(-1, num_frames, -1, -1)
        video_near = torch.full((1, num_frames), self.near, device=self.device)
        video_far = torch.full((1, num_frames), self.far, device=self.device)

        # Render video frames with depth
        output = self.decoder.forward(
            gaussians,
            video_extrinsics,
            video_intrinsics,
            video_near,
            video_far,
            (h, w),
            depth_mode="depth",
        )

        video_frames_rgb = output.color[0]  # [T, 3, H, W]
        video_frames_depth = output.depth[0] if output.depth is not None else None  # [T, H, W]
        video_frames_alpha = output.alpha[0] if output.alpha is not None else None  # [T, H, W]

        # Save RGB video
        video_rgb_path = os.path.join(output_dir, "video_rgb.mp4")
        frames_list = [frame for frame in video_frames_rgb]
        save_video_frames(frames_list, video_rgb_path, fps=30)

        # Save depth video
        video_depth_path = None
        if video_frames_depth is not None:
            depth_frames = []
            for i in range(video_frames_depth.shape[0]):
                depth_np = video_frames_depth[i].cpu().numpy()
                valid_mask = np.isfinite(depth_np) & (depth_np > 0)
                if valid_mask.any():
                    d_min = depth_np[valid_mask].min()
                    d_max = depth_np[valid_mask].max()
                    if d_max > d_min:
                        depth_normalized = (depth_np - d_min) / (d_max - d_min)
                    else:
                        depth_normalized = np.zeros_like(depth_np)
                else:
                    depth_normalized = np.zeros_like(depth_np)
                depth_normalized = np.clip(depth_normalized, 0, 1)
                depth_colored = self._apply_depth_colormap(depth_normalized)
                # Convert to tensor [3, H, W]
                depth_tensor = torch.from_numpy(depth_colored).permute(2, 0, 1).float() / 255.0
                depth_frames.append(depth_tensor)
            video_depth_path = os.path.join(output_dir, "video_depth.mp4")
            save_video_frames(depth_frames, video_depth_path, fps=30)

        # Save silhouette video
        video_silhouette_path = None
        if video_frames_alpha is not None:
            silhouette_frames = []
            for i in range(video_frames_alpha.shape[0]):
                alpha_np = video_frames_alpha[i].cpu().numpy()
                alpha_np = np.clip(alpha_np, 0, 1)
                # Convert to 3-channel grayscale tensor [3, H, W]
                silhouette_tensor = torch.from_numpy(alpha_np).float().unsqueeze(0).expand(3, -1, -1)
                silhouette_frames.append(silhouette_tensor)
            video_silhouette_path = os.path.join(output_dir, "video_silhouette.mp4")
            save_video_frames(silhouette_frames, video_silhouette_path, fps=30)

        return video_rgb_path, video_depth_path, video_silhouette_path

    def _export_ply_with_dump(self, gaussians, ply_path: str, v: int, h: int, w: int):
        """Export Gaussians to PLY file using visualization dump."""
        from scipy.spatial.transform import Rotation as R
        print(f"Exporting PLY to: {ply_path}")

        # Get scales and rotations from visualization dump
        # These are in shape [B, V*H*W*spp, ...]
        scales = self._visualization_dump['scales']  # [B, G, 3]
        rotations = self._visualization_dump['rotations']  # [B, G, 4]

        # Reshape for trimming (throw away border Gaussians)
        means = rearrange(
            gaussians.means, "() (v h w spp) xyz -> h w spp v xyz", v=v, h=h, w=w
        )

        # Create mask to filter border Gaussians
        mask = torch.zeros_like(means[..., 0], dtype=torch.bool)
        GAUSSIAN_TRIM = 8
        mask[GAUSSIAN_TRIM:-GAUSSIAN_TRIM, GAUSSIAN_TRIM:-GAUSSIAN_TRIM, :, :] = 1

        def trim(element):
            element = rearrange(
                element, "() (v h w spp) ... -> h w spp v ...", v=v, h=h, w=w
            )
            return element[mask][None]

        # Convert rotations from camera space to world space
        cam_rotations = trim(rotations)[0]  # [G_trimmed, 4]
        c2w_mat = repeat(
            self._context["extrinsics"][0, :, :3, :3],
            "v a b -> h w spp v a b",
            h=h,
            w=w,
            spp=1,
        )
        c2w_mat = c2w_mat[mask]  # apply trim

        cam_rotations_np = R.from_quat(
            cam_rotations.detach().cpu().numpy()
        ).as_matrix()
        world_mat = c2w_mat.detach().cpu().numpy() @ cam_rotations_np
        world_rotations = R.from_matrix(world_mat).as_quat()
        world_rotations = torch.from_numpy(world_rotations).to(scales)

        # Export
        export_ply(
            self._context["extrinsics"][0, 0],
            trim(gaussians.means)[0],
            trim(scales)[0],
            world_rotations,
            trim(gaussians.harmonics)[0],
            trim(gaussians.opacities)[0],
            Path(ply_path),
        )

    @torch.no_grad()
    def generate_flight_tracking_video(
        self,
        render_dir: str,
        cache_dir: str = None,
        start_frame: int = 0,
        end_frame: int = 119,
        elevation: float = 30.0,
        distance: float = 1.0,
        progress_callback=None,
    ) -> dict:
        """
        Generate a 360° flight tracking video.

        For each frame (0-119), loads the in-the-wild frame data, builds a Gaussian
        model, and renders from a camera viewpoint that rotates 3° per frame to
        complete a full 360° orbit.

        Args:
            render_dir: Path to render directory containing in-the-wild frames
            cache_dir: Directory for temporary files and output
            start_frame: First frame to process (default: 0)
            end_frame: Last frame to process (default: 119)
            elevation: Camera elevation angle in degrees
            distance: Camera distance factor (0.6-1.4)
            progress_callback: Optional callback(frame_id, total) for progress updates

        Returns:
            Dictionary with 'flight_video_path' and 'num_frames'
        """
        torch.cuda.empty_cache()

        if cache_dir is None:
            cache_dir = "/tmp/depthsplat_flight_tracking"
        os.makedirs(cache_dir, exist_ok=True)

        num_frames = end_frame - start_frame + 1
        rendered_frames = []

        print(f"Starting 360° flight tracking: {num_frames} frames, elevation={elevation}°, distance={distance}")

        for frame_id in range(start_frame, end_frame + 1):
            if progress_callback:
                progress_callback(frame_id - start_frame, num_frames)

            print(f"\n[Flight Tracking] Processing frame {frame_id}/{end_frame}...")

            try:
                # Load the frame (this sets up extrinsics, intrinsics, centering, scaling)
                image_paths, status = self.load_wild_frame(frame_id, render_dir, cache_dir)

                # Load images
                images = []
                for img_path in image_paths:
                    img = Image.open(img_path).convert("RGB")
                    images.append(np.array(img))

                # Get context cameras from current_example (set by load_wild_frame)
                extrinsics = self.current_example['extrinsics']
                intrinsics = self.current_example['intrinsics']

                # Calculate azimuth for this frame (3° per frame for 360° total)
                azimuth = (frame_id - start_frame) * (360.0 / num_frames)

                # Create target camera from azimuth/elevation/distance
                base_radius = 2.0  # Match normalization target radius
                radius = base_radius * distance

                azimuth_rad = np.deg2rad(azimuth)
                elevation_rad = np.deg2rad(elevation)

                # Spherical coordinates
                x = radius * np.cos(elevation_rad) * np.cos(azimuth_rad)
                y = radius * np.cos(elevation_rad) * np.sin(azimuth_rad)
                z = radius * np.sin(elevation_rad)

                cam_pos = np.array([x, y, z], dtype=np.float32)

                # Look at origin
                target = np.array([0, 0, 0], dtype=np.float32)
                forward = target - cam_pos
                forward = forward / np.linalg.norm(forward)

                # World up vector (Z is up)
                world_up = np.array([0, 0, 1], dtype=np.float32)

                # Right vector
                right = np.cross(forward, world_up)
                if np.linalg.norm(right) < 1e-6:
                    world_up = np.array([0, 1, 0], dtype=np.float32)
                    right = np.cross(forward, world_up)
                right = right / np.linalg.norm(right)

                # Recompute up
                up = np.cross(right, forward)
                up = up / np.linalg.norm(up)

                # Build rotation matrix - camera-to-world (OpenCV convention)
                down = -up
                R = np.stack([right, down, forward], axis=1).astype(np.float32)

                target_extrinsic = np.eye(4, dtype=np.float32)
                target_extrinsic[:3, :3] = R
                target_extrinsic[:3, 3] = cam_pos
                target_extrinsics = target_extrinsic[np.newaxis, ...]  # [1, 4, 4]

                # Preprocess images
                images_tensor = self.preprocess_images(images)
                b, v, c, h, w = images_tensor.shape

                # Convert camera parameters to tensors
                extrinsics_tensor = torch.from_numpy(extrinsics).float().unsqueeze(0).to(self.device)
                intrinsics_tensor = torch.from_numpy(intrinsics).float().unsqueeze(0).to(self.device)

                # Build batch
                batch = self.build_batch(images_tensor, extrinsics_tensor, intrinsics_tensor)
                batch = self.data_shim(batch)
                context = batch['context']

                h, w = context['image'].shape[-2:]

                # Run encoder
                visualization_dump = {}
                gaussians = self.encoder(context, global_step=0, deterministic=True, visualization_dump=visualization_dump)

                if isinstance(gaussians, dict):
                    gaussians = gaussians['gaussians']

                # Prepare target camera
                target_extrinsics_tensor = torch.from_numpy(target_extrinsics).float().unsqueeze(0).to(self.device)
                target_intrinsics_tensor = intrinsics_tensor[:, :1]  # Use first context intrinsics

                target_near = torch.full((1, 1), self.near, device=self.device)
                target_far = torch.full((1, 1), self.far, device=self.device)

                # Render
                output = self.decoder.forward(
                    gaussians,
                    target_extrinsics_tensor,
                    target_intrinsics_tensor,
                    target_near,
                    target_far,
                    (h, w),
                    depth_mode=None,  # Skip depth for speed
                )

                rendered_image = output.color[0, 0]  # [3, H, W]
                rendered_frames.append(rendered_image.cpu())

                # Clear GPU memory
                del gaussians, output, batch
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error processing frame {frame_id}: {e}")
                # Add a black frame as placeholder
                rendered_frames.append(torch.zeros(3, 256, 256))

        # Save video
        video_path = os.path.join(cache_dir, "flight_tracking_360.mp4")
        print(f"\nSaving flight tracking video to {video_path}...")
        save_video_frames(rendered_frames, video_path, fps=30)

        return {
            'flight_video_path': video_path,
            'num_frames': len(rendered_frames),
        }


def create_default_intrinsics(
    fov_degrees: float = 50.0,
    image_size: Tuple[int, int] = (256, 256),
) -> np.ndarray:
    """
    Create normalized intrinsics matrix from FOV.

    Args:
        fov_degrees: Field of view in degrees
        image_size: (H, W) image dimensions

    Returns:
        [3, 3] normalized intrinsics matrix
    """
    h, w = image_size
    fov_rad = np.deg2rad(fov_degrees)
    focal = 0.5 / np.tan(fov_rad / 2)  # Normalized focal length

    K = np.array([
        [focal, 0, 0.5],
        [0, focal, 0.5],
        [0, 0, 1],
    ], dtype=np.float32)

    return K


def create_orbit_extrinsics(
    num_views: int,
    radius: float = 2.0,
    elevation: float = 0.3,
    azimuth_offset: float = 0.0,
) -> np.ndarray:
    """
    Create camera extrinsics for views orbiting around origin.

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
        # Camera position
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
