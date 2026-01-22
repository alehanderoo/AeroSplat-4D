"""
Visualization Pipeline Extension.

Extends the DepthSplatPipeline with:
- Server-side Gaussian rendering using the decoder
- WebSocket streaming to frontend clients
- Render camera management (orbit, preset views)
"""

import asyncio
import logging
import time
import threading
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Callable
from pathlib import Path

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from .deepstream_pipeline import DepthSplatPipeline, PipelineConfig, GaussianOutput

# Import data_shim from DepthSplat - CRITICAL for proper reconstruction
try:
    from src.dataset.data_module import get_data_shim
    DATA_SHIM_AVAILABLE = True
except ImportError:
    DATA_SHIM_AVAILABLE = False
    get_data_shim = None

# Import server module - handle both package and direct execution
try:
    from server import (
        VisualizationServer,
        VisualizationConfig,
        FramePacket,
        create_frame_packet,
        encode_image_to_base64,
    )
except ImportError:
    from ..server import (
        VisualizationServer,
        VisualizationConfig,
        FramePacket,
        create_frame_packet,
        encode_image_to_base64,
    )

logger = logging.getLogger(__name__)


@dataclass
class VisualizationPipelineConfig(PipelineConfig):
    """Extended configuration for visualization pipeline."""
    # Visualization settings
    visualization_enabled: bool = True
    websocket_host: str = "0.0.0.0"
    websocket_port: int = 8765
    render_width: int = 512
    render_height: int = 512
    default_view_mode: str = "orbit"
    orbit_speed_deg_per_sec: float = 15.0
    input_thumbnail_width: int = 192
    input_thumbnail_height: int = 108
    jpeg_quality: int = 85

    # Detection service settings
    detection_enabled: bool = True
    detection_gt_path: Optional[str] = None  # Path to ground truth JSON
    # crop_size is now dynamic (minimum size), not fixed
    min_crop_size: int = 64  # Minimum crop size in pixels
    
    # Tight cropping settings (matching Gradio demo)
    # The model was trained/demoed with objects filling ~75% of the frame
    target_object_coverage: float = 0.75  
    crop_margin: float = 0.15

    # Camera calibration settings
    calibration_json_path: Optional[str] = None  # Path to calibration JSON (defaults to detection_gt_path)
    
    # Ground truth depth visualization settings
    gt_depth_base_path: Optional[str] = None  # Base path to GT depth renders
    depth_near: float = 0.5  # Near plane for depth colormap
    depth_far: float = 100.0  # Far plane for depth colormap

    # File-based frame source settings (for development with pre-rendered frames)
    # When enabled, reads frames directly from disk instead of RTSP streams,
    # ensuring perfect synchronization with GT detection data
    use_file_source: bool = False
    file_source_dir: Optional[str] = None  # Base directory for rendered frames
    file_source_num_frames: int = 120  # Total number of frames
    file_source_loop: bool = True  # Whether to loop through frames

    def get_visualization_config(self) -> VisualizationConfig:
        """Convert to VisualizationConfig."""
        return VisualizationConfig(
            enabled=self.visualization_enabled,
            websocket_host=self.websocket_host,
            websocket_port=self.websocket_port,
            render_width=self.render_width,
            render_height=self.render_height,
            default_view_mode=self.default_view_mode,
            orbit_speed_deg_per_sec=self.orbit_speed_deg_per_sec,
            input_thumbnail_width=self.input_thumbnail_width,
            input_thumbnail_height=self.input_thumbnail_height,
            jpeg_quality=self.jpeg_quality,
        )


class VisualizationPipeline(DepthSplatPipeline):
    """
    Extended pipeline with visualization streaming.

    Adds:
    - Server-side Gaussian rendering
    - WebSocket broadcast to connected clients
    - Render camera animation (orbit mode)
    - Detection-based cropping
    """

    def __init__(self, config: VisualizationPipelineConfig):
        super().__init__(config)
        self.vis_config = config

        # Visualization server
        self.vis_server: Optional[VisualizationServer] = None
        self._vis_loop: Optional[asyncio.AbstractEventLoop] = None
        self._vis_thread: Optional[threading.Thread] = None

        # Decoder reference (set during model loading)
        self.decoder = None

        # Detection service
        self.detection_service = None
        self._init_detection_service()

        # Camera calibration service
        self.calibration_service = None
        self._init_calibration_service()
        
        # Ground truth depth service
        self.gt_depth_service = None
        self._init_gt_depth_service()

        # File-based frame source (for development with pre-rendered frames)
        self.file_frame_source = None
        self._init_file_frame_source()

        # Keep raw frames for visualization
        self._last_raw_frames: List[np.ndarray] = []
        self._last_gaussians = None
        self._last_crop_regions: List[tuple] = None  # Track crop regions for intrinsics adjustment
        
        # Depth visualization outputs from last inference
        self._last_mono_depth: List[np.ndarray] = []  # Monocular depth per camera
        self._last_predicted_depth: List[np.ndarray] = []  # MVS predicted depth per camera
        self._last_silhouette: List[np.ndarray] = []  # Silhouette/confidence per camera

        # Rendering stats
        self.render_times = []
        self.encode_times = []

    def _init_detection_service(self):
        """Initialize the detection service if configured."""
        if not self.vis_config.detection_enabled:
            logger.info("Detection service disabled")
            return

        if self.vis_config.detection_gt_path:
            try:
                from services import GroundTruthDetectionService
                self.detection_service = GroundTruthDetectionService(
                    json_path=self.vis_config.detection_gt_path,
                    camera_names=[f"cam_{i+1:02d}" for i in range(self.config.num_cameras)],
                    loop=True,
                )
                logger.info(f"Ground truth detection service initialized from: {self.vis_config.detection_gt_path}")
            except Exception as e:
                logger.warning(f"Failed to initialize detection service: {e}")
                self.detection_service = None
        else:
            logger.info("No detection ground truth path configured - using center crop")

    def _init_calibration_service(self):
        """Initialize the camera calibration service if configured."""
        # Use calibration_json_path if set, else fall back to detection_gt_path
        json_path = self.vis_config.calibration_json_path or self.vis_config.detection_gt_path

        if json_path:
            try:
                from services import CameraCalibrationService
                self.calibration_service = CameraCalibrationService(
                    json_path=json_path,
                    camera_names=[f"cam_{i+1:02d}" for i in range(self.config.num_cameras)],
                    # Enable all camera transformations matching the working Gradio demo
                    apply_coordinate_flip=True,      # OpenGL→OpenCV transformation
                    apply_pose_normalization=True,   # Center + scale cameras
                    use_training_intrinsics=True,    # Use fx=1.0723 (50° FOV)
                    target_radius=2.0,               # Scale to 2.0m from object
                )
                logger.info(f"Camera calibration service initialized from: {json_path}")
                logger.info("Camera transformations enabled: coordinate_flip=True, pose_norm=True, training_intrinsics=True")
            except Exception as e:
                logger.warning(f"Failed to initialize calibration service: {e}")
                import traceback
                traceback.print_exc()
                self.calibration_service = None
        else:
            logger.info("No calibration JSON path configured - using dummy intrinsics/extrinsics")
            
    def _init_gt_depth_service(self):
        """Initialize the ground truth depth service if configured."""
        if self.vis_config.gt_depth_base_path:
            try:
                from services import GroundTruthDepthService
                self.gt_depth_service = GroundTruthDepthService(
                    base_path=self.vis_config.gt_depth_base_path,
                    camera_names=[f"cam_{i+1:02d}" for i in range(self.config.num_cameras)],
                    loop=True,
                )
                logger.info(f"Ground truth depth service initialized from: {self.vis_config.gt_depth_base_path}")
            except Exception as e:
                logger.warning(f"Failed to initialize GT depth service: {e}")
                self.gt_depth_service = None
        else:
            logger.info("No GT depth base path configured - GT depth visualization disabled")

    def _init_file_frame_source(self):
        """Initialize file-based frame source if configured."""
        if not self.vis_config.use_file_source:
            logger.info("File frame source disabled - using RTSP streams")
            return

        if not self.vis_config.file_source_dir:
            logger.warning("use_file_source=True but file_source_dir not set")
            return

        try:
            from utils.file_frame_source import create_file_frame_source
            self.file_frame_source = create_file_frame_source(
                render_dir=self.vis_config.file_source_dir,
                camera_names=[f"cam_{i+1:02d}" for i in range(self.config.num_cameras)],
                num_frames=self.vis_config.file_source_num_frames,
                loop=self.vis_config.file_source_loop,
            )
            logger.info(
                f"File frame source initialized: {self.vis_config.file_source_dir}, "
                f"{self.vis_config.file_source_num_frames} frames"
            )
        except Exception as e:
            logger.warning(f"Failed to initialize file frame source: {e}")
            import traceback
            traceback.print_exc()
            self.file_frame_source = None

    def _load_model(self):
        """Load model with decoder preserved for rendering."""
        # Call parent's model loading
        super()._load_model()

        # Extract decoder reference
        if self.model is not None and hasattr(self.model, 'decoder'):
            self.decoder = self.model.decoder
            logger.info("Decoder extracted for visualization rendering")
        else:
            logger.warning("Decoder not found in model - rendering will be disabled")

        # Initialize data_shim - CRITICAL for proper reconstruction
        # The data_shim handles patch alignment and intrinsics adjustment
        # This is exactly what the Gradio demo does
        self.data_shim = None
        if DATA_SHIM_AVAILABLE and self.model is not None and hasattr(self.model, 'encoder'):
            try:
                self.data_shim = get_data_shim(self.model.encoder)
                logger.info("Data shim initialized from encoder")
            except Exception as e:
                logger.warning(f"Failed to initialize data_shim: {e}")

        # Initialize GradioReconstructor - uses the exact same pipeline as the working Gradio demo
        self.reconstructor = None
        if self.model is not None and self.decoder is not None and self.data_shim is not None:
            try:
                from .reconstruction import GradioReconstructor
                self.reconstructor = GradioReconstructor(
                    encoder=self.model.encoder,
                    decoder=self.decoder,
                    data_shim=self.data_shim,
                    device=self.device,
                    near=0.55,
                    far=2.54,
                )
                logger.info("GradioReconstructor initialized - using exact Gradio demo pipeline")
            except Exception as e:
                logger.warning(f"Failed to initialize GradioReconstructor: {e}")
                import traceback
                traceback.print_exc()
        else:
            logger.warning("Data shim not available - reconstruction quality may be degraded")

    def build_batch(
        self,
        images: "torch.Tensor",
        extrinsics: "torch.Tensor",
        intrinsics: "torch.Tensor",
        near: float = 0.55,
        far: float = 2.54,
    ) -> dict:
        """
        Build batch dictionary for the encoder (matching Gradio demo structure).

        The data_shim expects a batch with both 'context' and 'target' keys.
        This structure is required for proper patch alignment.

        Args:
            images: [1, V, 3, H, W] tensor
            extrinsics: [1, V, 4, 4] camera-to-world matrices
            intrinsics: [1, V, 3, 3] normalized intrinsics
            near: Near plane distance
            far: Far plane distance

        Returns:
            Batch dictionary with context and target (matching Gradio demo)
        """
        b, v, _, h, w = images.shape

        context = {
            'image': images,
            'extrinsics': extrinsics,
            'intrinsics': intrinsics,
            'near': torch.full((b, v), near, device=self.device),
            'far': torch.full((b, v), far, device=self.device),
            'index': torch.arange(v, device=self.device).unsqueeze(0),
        }

        # Create a dummy target for the data shim (uses first context view)
        # This is required by the data_shim - it expects both context and target
        target = {
            'image': images[:, :1],
            'extrinsics': extrinsics[:, :1],
            'intrinsics': intrinsics[:, :1],
            'near': torch.full((b, 1), near, device=self.device),
            'far': torch.full((b, 1), far, device=self.device),
            'index': torch.zeros(b, 1, dtype=torch.long, device=self.device),
        }

        batch = {
            'context': context,
            'target': target,
            'scene': ['inference_input'],
        }

        return batch

    def _run_visualization_server(self):
        """Run the WebSocket server in a separate thread."""
        # Create new event loop for this thread
        self._vis_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._vis_loop)

        # Create server
        vis_config = self.vis_config.get_visualization_config()
        self.vis_server = VisualizationServer(vis_config)

        # Run server
        try:
            self._vis_loop.run_until_complete(self.vis_server.start())
            self._vis_loop.run_forever()
        except Exception as e:
            logger.error(f"Visualization server error: {e}")
        finally:
            self._vis_loop.close()

    def _run_inference_with_gaussians(self, input_tensor, detections=None):
        """
        Run model inference and return both GaussianOutput and Gaussians dataclass.

        Uses GradioReconstructor when available (exact Gradio demo pipeline).

        Args:
            input_tensor: Preprocessed input tensor [B, V, C, H, W]
            detections: Optional FrameDetections from detection service.
                       If provided, uses the object_position_3d for pose normalization.
        """
        import torch
        import torch.nn.functional as F
        from .deepstream_pipeline import GaussianOutput
        from services import depth_to_colormap
        from .reconstruction import TRAINING_FX_NORM, TARGET_RADIUS

        with torch.no_grad():
            start_time = time.perf_counter()

            try:
                b, v, c, h, w = input_tensor.shape

                # --- Update object position for pose normalization ---
                # This is CRITICAL: The calibration service needs the object position
                # to center the cameras correctly for pose normalization
                object_position = None
                if detections is not None:
                    object_position = detections.object_position_3d
                    if object_position is not None:
                        object_position = np.array(object_position, dtype=np.float32)
                        # CRITICAL FIX: Apply coordinate flip to object position
                        # The object position from Isaac Sim is in OpenGL coordinates (Y-up, Z-back)
                        # But the model expects OpenCV coordinates (Y-down, Z-forward)
                        # We need to flip Y and Z axes to match the camera extrinsics flip
                        object_position_opencv = object_position.copy()
                        object_position_opencv[1] *= -1.0  # Flip Y
                        object_position_opencv[2] *= -1.0  # Flip Z
                        logger.debug(f"Object position OpenGL: {object_position} -> OpenCV: {object_position_opencv}")

                        if self.calibration_service is not None:
                            self.calibration_service.set_object_position(object_position_opencv)

                # --- Use GradioReconstructor if available (preferred) ---
                if self.reconstructor is not None:
                    # Convert input tensor to list of numpy images for GradioReconstructor
                    # input_tensor is [B, V, C, H, W] in [0, 1]
                    images_np = []
                    for cam_idx in range(v):
                        img = input_tensor[0, cam_idx].permute(1, 2, 0).cpu().numpy()  # [H, W, C]
                        images_np.append(img)

                    # Get extrinsics from calibration service (already normalized)
                    if self.calibration_service is not None:
                        # CRITICAL FIX: Don't pass crop_regions to extrinsics
                        # The Gradio demo uses original (uncropped) extrinsics
                        # Only the intrinsics are affected by crops (and then overridden)
                        ext_tensor = self.calibration_service.get_extrinsics_tensor(
                            device='cpu',
                            crop_regions=None,  # FIX: Use original extrinsics like Gradio
                            use_virtual_camera=False,
                        )
                        extrinsics_np = ext_tensor[0].numpy()  # [V, 4, 4]

                        # Debug logging to verify camera parameters
                        logger.debug(f"Extrinsics shape: {extrinsics_np.shape}")
                        logger.debug(f"Camera positions after normalization:")
                        for i in range(min(v, 3)):  # Log first 3 cameras
                            pos = extrinsics_np[i, :3, 3]
                            dist = np.linalg.norm(pos)
                            logger.debug(f"  cam_{i+1}: pos={pos}, dist={dist:.3f}")
                    else:
                        # Fallback: create orbit cameras
                        from .reconstruction import create_orbit_extrinsics
                        extrinsics_np = create_orbit_extrinsics(
                            num_views=v,
                            radius=TARGET_RADIUS,
                            elevation=0.3,
                        )

                    # Run reconstruction using GradioReconstructor (uses training-matched intrinsics)
                    gaussians, context = self.reconstructor.reconstruct(
                        images=images_np,
                        extrinsics=extrinsics_np,
                        intrinsics=None,  # Use training-matched intrinsics
                    )

                    h, w = context['image'].shape[-2:]
                    inference_time = (time.perf_counter() - start_time) * 1000

                    # Extract visualizations - simplified since we don't have visualization_dump
                    self._last_predicted_depth = [None] * v
                    self._last_silhouette = [None] * v
                    self._last_mono_depth = [None] * v

                    # Extract numpy arrays for GaussianOutput
                    if gaussians is not None and hasattr(gaussians, "means"):
                        positions = gaussians.means.cpu().numpy()
                        covariances = gaussians.covariances.cpu().numpy()
                        colors = gaussians.harmonics.cpu().numpy()
                        opacities = gaussians.opacities.cpu().numpy()

                        if positions.ndim > 2:
                            positions = positions.reshape(-1, 3)
                        if covariances.ndim > 3:
                            covariances = covariances.reshape(-1, covariances.shape[-2], covariances.shape[-1])
                        if colors.ndim > 3:
                            colors = colors.reshape(-1, colors.shape[-2], colors.shape[-1])
                        if opacities.ndim > 1:
                            opacities = opacities.reshape(-1, 1)

                        # Extract silhouettes from opacities
                        try:
                            opacity_data = gaussians.opacities
                            n_per_pixel = opacity_data.shape[1] // (v * h * w)
                            if n_per_pixel > 0:
                                opacity_per_pixel = opacity_data.reshape(1, v, h, w, -1).mean(dim=-1)
                                for cam_idx in range(v):
                                    opacity_cam = opacity_per_pixel[0, cam_idx].cpu().numpy()
                                    opacity_vis = (np.clip(opacity_cam, 0, 1) * 255).astype(np.uint8)
                                    opacity_rgb = np.stack([opacity_vis]*3, axis=-1)
                                    self._last_silhouette[cam_idx] = opacity_rgb
                        except:
                            pass
                    else:
                        positions = np.zeros((0, 3), dtype=np.float32)
                        covariances = np.zeros((0, 3, 3), dtype=np.float32)
                        colors = np.zeros((0, 3), dtype=np.float32)
                        opacities = np.zeros((0, 1), dtype=np.float32)

                # --- Fallback to original code if GradioReconstructor not available ---
                else:
                    # Get camera parameters from calibration service or use defaults
                    if self.calibration_service is not None:
                        intrinsics = self.calibration_service.get_intrinsics_tensor(
                            device=self.device,
                            crop_regions=self._last_crop_regions,
                            use_virtual_camera=False,
                        )
                        # CRITICAL FIX: Don't pass crop_regions to extrinsics (same as above)
                        extrinsics = self.calibration_service.get_extrinsics_tensor(
                            device=self.device,
                            crop_regions=None,  # FIX: Use original extrinsics
                            use_virtual_camera=False,
                        )
                    else:
                        # Fallback to training-matched intrinsics
                        intrinsics = torch.zeros(b, v, 3, 3, device=self.device)
                        intrinsics[:, :, 0, 0] = TRAINING_FX_NORM
                        intrinsics[:, :, 1, 1] = TRAINING_FX_NORM
                        intrinsics[:, :, 0, 2] = 0.5
                        intrinsics[:, :, 1, 2] = 0.5
                        intrinsics[:, :, 2, 2] = 1.0

                        extrinsics = torch.eye(4, device=self.device).unsqueeze(0).unsqueeze(0).expand(b, v, 4, 4).clone()
                        for i in range(v):
                            angle = 2 * np.pi * i / v
                            radius = TARGET_RADIUS
                            extrinsics[:, i, 0, 3] = radius * np.cos(angle)
                            extrinsics[:, i, 1, 3] = radius * np.sin(angle)
                            extrinsics[:, i, 2, 3] = 0.0

                    # Build batch and apply data_shim
                    batch = self.build_batch(
                        images=input_tensor,
                        extrinsics=extrinsics,
                        intrinsics=intrinsics,
                        near=0.55,
                        far=2.54,
                    )

                    if self.data_shim is not None:
                        batch = self.data_shim(batch)

                    context = batch['context']
                    h, w = context['image'].shape[-2:]

                    visualization_dump = {}
                    result = self.model.encoder(
                        context,
                        global_step=0,
                        deterministic=True,
                        visualization_dump=visualization_dump,
                    )

                    inference_time = (time.perf_counter() - start_time) * 1000

                    # Handle both dict output (when return_depth=True) and direct Gaussians output
                    if isinstance(result, dict):
                        gaussians = result.get("gaussians")
                        predicted_depths = result.get("depths")
                    else:
                        gaussians = result
                        predicted_depths = None

                    # Try to get depth from visualization_dump
                    if "depth" in visualization_dump and predicted_depths is None:
                        predicted_depths = visualization_dump["depth"]
                        if predicted_depths.dim() > 4:
                            predicted_depths = predicted_depths.squeeze(-1).squeeze(-1)

                    # Extract and visualize depth per camera
                    self._last_predicted_depth = []
                    self._last_silhouette = []
                    self._last_mono_depth = []

                    if predicted_depths is not None:
                        for cam_idx in range(v):
                            depth_cam = predicted_depths[0, cam_idx].cpu().numpy()
                            depth_vis = depth_to_colormap(
                                depth_cam,
                                near=self.vis_config.depth_near,
                                far=self.vis_config.depth_far
                            )
                            self._last_predicted_depth.append(depth_vis)
                    else:
                        self._last_predicted_depth = [None] * v

                    # Extract silhouettes from opacities
                    if gaussians is not None and hasattr(gaussians, "opacities"):
                        try:
                            opacities = gaussians.opacities
                            n_gaussians_per_pixel = opacities.shape[1] // (v * h * w)
                            if n_gaussians_per_pixel > 0:
                                opacity_per_pixel = opacities.reshape(b, v, h, w, -1).mean(dim=-1)
                                for cam_idx in range(v):
                                    opacity_cam = opacity_per_pixel[0, cam_idx].cpu().numpy()
                                    opacity_vis = (np.clip(opacity_cam, 0, 1) * 255).astype(np.uint8)
                                    opacity_rgb = np.stack([opacity_vis]*3, axis=-1)
                                    self._last_silhouette.append(opacity_rgb)
                            else:
                                self._last_silhouette = [None] * v
                        except Exception as e:
                            logger.debug(f"Failed to extract silhouette: {e}")
                            self._last_silhouette = [None] * v
                    else:
                        self._last_silhouette = [None] * v

                    # Extract monocular depth from visualization_dump
                    mono_depths = visualization_dump.get("mono_depth")
                    if mono_depths is not None:
                        for cam_idx in range(v):
                            mono_cam = mono_depths[0, cam_idx].cpu().numpy()
                            mono_vis = depth_to_colormap(
                                mono_cam,
                                near=float(mono_cam.min()) if np.isfinite(mono_cam.min()) else -1.0,
                                far=float(mono_cam.max()) if np.isfinite(mono_cam.max()) else 1.0,
                            )
                            self._last_mono_depth.append(mono_vis)
                    else:
                        self._last_mono_depth = [None] * v

                    # Extract numpy arrays for GaussianOutput
                    if gaussians is not None and hasattr(gaussians, "means"):
                        positions = gaussians.means.cpu().numpy()
                        covariances = gaussians.covariances.cpu().numpy()
                        colors = gaussians.harmonics.cpu().numpy()
                        opacities = gaussians.opacities.cpu().numpy()

                        if positions.ndim > 2:
                            positions = positions.reshape(-1, 3)
                        if covariances.ndim > 3:
                            covariances = covariances.reshape(-1, covariances.shape[-2], covariances.shape[-1])
                        if colors.ndim > 3:
                            colors = colors.reshape(-1, colors.shape[-2], colors.shape[-1])
                        if opacities.ndim > 1:
                            opacities = opacities.reshape(-1, 1)
                    else:
                        positions = np.zeros((0, 3), dtype=np.float32)
                        covariances = np.zeros((0, 3, 3), dtype=np.float32)
                        colors = np.zeros((0, 3), dtype=np.float32)
                        opacities = np.zeros((0, 1), dtype=np.float32)

            except Exception as e:
                logger.error(f"Inference error: {e}")
                import traceback
                traceback.print_exc()
                inference_time = (time.perf_counter() - start_time) * 1000
                gaussians = None
                positions = np.zeros((0, 3), dtype=np.float32)
                covariances = np.zeros((0, 3, 3), dtype=np.float32)
                colors = np.zeros((0, 3), dtype=np.float32)
                opacities = np.zeros((0, 1), dtype=np.float32)
                # Clear visualizations on error
                self._last_predicted_depth = []
                self._last_silhouette = []
                self._last_mono_depth = []

        output = GaussianOutput(
            frame_id=self.frame_count,
            timestamp=time.time(),
            positions=positions,
            covariances=covariances,
            colors=colors,
            opacities=opacities,
            inference_time_ms=inference_time,
        )

        return output, gaussians

    def _render_gaussians(self, gaussians, render_camera) -> Optional[np.ndarray]:
        """
        Render Gaussians using the GradioReconstructor (exact Gradio demo pipeline).

        Args:
            gaussians: Gaussian parameters from encoder
            render_camera: RenderCamera with extrinsics/intrinsics

        Returns:
            HWC RGB numpy array of rendered image, or None on error
        """
        if gaussians is None:
            return None

        try:
            render_start = time.perf_counter()

            # Use GradioReconstructor if available (preferred - exact Gradio demo pipeline)
            if self.reconstructor is not None:
                # Get render resolution
                h, w = self.vis_config.render_height, self.vis_config.render_width

                # Use the orbit camera from GradioReconstructor (exact Gradio demo camera)
                color = self.reconstructor.render(
                    gaussians=gaussians,
                    extrinsics=render_camera.extrinsics,
                    intrinsics=render_camera.intrinsics,
                    image_shape=(h, w),
                )

                render_time = (time.perf_counter() - render_start) * 1000
                self.render_times.append(render_time)
                if len(self.render_times) > 100:
                    self.render_times.pop(0)

                return color

            # Fallback to direct decoder call if GradioReconstructor not available
            elif self.decoder is not None:
                import torch

                with torch.no_grad():
                    h, w = self.vis_config.render_height, self.vis_config.render_width
                    device = self.device

                    extrinsics = torch.from_numpy(
                        render_camera.extrinsics
                    ).float().to(device).unsqueeze(0).unsqueeze(0)

                    intrinsics = torch.from_numpy(
                        render_camera.intrinsics
                    ).float().to(device).unsqueeze(0).unsqueeze(0)

                    near = torch.tensor([[render_camera.near]], device=device)
                    far = torch.tensor([[render_camera.far]], device=device)

                    output = self.decoder(
                        gaussians,
                        extrinsics,
                        intrinsics,
                        near,
                        far,
                        image_shape=(h, w),
                    )

                    color = output.color[0, 0]
                    color = color.permute(1, 2, 0)
                    color = color.clamp(0, 1)
                    color = (color * 255).byte().cpu().numpy()

                render_time = (time.perf_counter() - render_start) * 1000
                self.render_times.append(render_time)
                if len(self.render_times) > 100:
                    self.render_times.pop(0)

                return color

            return None

        except Exception as e:
            logger.error(f"Rendering error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _create_cropped_views(
        self,
        frames: List[np.ndarray],
        detections=None
    ) -> List[Optional[np.ndarray]]:
        """
        Create cropped views centered on detected objects for all cameras.

        Uses detection service coordinates if available, otherwise falls back
        to center cropping.

        Args:
            frames: List of raw frames from cameras
            detections: Optional FrameDetections from detection service

        Returns:
            List of cropped images (one per camera), same order as input frames.
        """
        cropped_views = []
        # No longer using fixed crop_size from config
        # crop_size = self.vis_config.crop_size

        for i, frame in enumerate(frames):
            if frame is None:
                cropped_views.append(None)
                continue

            h, w = frame.shape[:2]
            camera_name = f"cam_{i+1:02d}"

            # Try to get detection coordinates
            detection = None
            if detections is not None:
                detection = detections.get_detection(camera_name)

            if detection is not None and detection.visible:
                # Calculate dynamic crop size based on bbox size
                bbox = detection.bbox
                if bbox:
                    bbox_width = bbox[2] - bbox[0]
                    bbox_height = bbox[3] - bbox[1]
                    bbox_size = max(bbox_width, bbox_height)
                else:
                    bbox_size = 68.0  # Default fallback
                
                # Tight crop calculation:
                # crop_size = bbox_size / TARGET_OBJECT_COVERAGE
                # This ensures the object fills approx target_coverage of the final frame
                tight_crop_size = int(bbox_size / self.vis_config.target_object_coverage * (1 + self.vis_config.crop_margin))
                
                # Ensure minimum size for quality
                crop_size = max(tight_crop_size, self.vis_config.min_crop_size)
                
                # Use detection coordinates for cropping with dynamic size
                x1, y1, x2, y2 = detection.get_crop_region(
                    crop_size=crop_size,
                    image_width=w,
                    image_height=h,
                    use_bbox=False  # We already used bbox for size calculation
                )
            else:
                # Fall back to center crop with reasonable default
                crop_size = max(w // 4, self.vis_config.min_crop_size)
                size = min(h, w, crop_size)
                x1 = (w - size) // 2
                y1 = (h - size) // 2
                x2 = x1 + size
                y2 = y1 + size

            # Extract crop
            cropped = frame[y1:y2, x1:x2]

            # Apply mask if available (for visualization consistency)
            if detection and detection.mask_path:
                try:
                    mask = cv2.imread(detection.mask_path, cv2.IMREAD_GRAYSCALE)
                    if mask is not None:
                        if mask.shape[:2] != (h, w):
                            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                        mask_cropped = mask[y1:y2, x1:x2]
                        _, fg_mask = cv2.threshold(mask_cropped, 127, 255, cv2.THRESH_BINARY)
                        white_bg = np.ones_like(cropped) * 255
                        fg_mask_bool = fg_mask > 0
                        cropped = np.where(fg_mask_bool[..., None], cropped, white_bg).astype(np.uint8)
                except Exception as e:
                    pass

            # Resize to standard output size (for frontend display)
            if cropped.size > 0:
                cropped = cv2.resize(cropped, (256, 256))
                cropped_views.append(cropped)
            else:
                cropped_views.append(None)

        return cropped_views

    def _preprocess_frames_with_detections(
        self,
        frames: List[np.ndarray],
        detections=None
    ) -> "torch.Tensor":
        """
        Preprocess frames with detection-based cropping for model input.
        
        Instead of center cropping, crops around detected object positions
        to ensure the flying object is centered in each view.
        
        Args:
            frames: List of raw camera frames
            detections: FrameDetections from detection service
            
        Returns:
            Preprocessed tensor [1, num_cameras, 3, H, W]
        """
        import cv2
        import torch
        
        processed = []
        crop_regions = []  # Track crop regions for intrinsics adjustment
        h, w = self.config.model_input_height, self.config.model_input_width
        
        for i, frame in enumerate(frames):
            if frame is None:
                # Create blank frame if capture failed
                frame_cropped = np.zeros((h, w, 3), dtype=np.uint8)
                # Use center crop region as fallback
                crop_regions.append((0, 0, self.vis_config.min_crop_size, self.vis_config.min_crop_size))
            else:
                frame_h, frame_w = frame.shape[:2]
                camera_name = f"cam_{i+1:02d}"
                
                # Try to get detection coordinates for cropping
                detection = None
                if detections is not None:
                    detection = detections.get_detection(camera_name)
                
                if detection is not None and detection.visible:
                    # Calculate dynamic crop size based on bbox size
                    bbox = detection.bbox
                    if bbox:
                        bbox_width = bbox[2] - bbox[0]
                        bbox_height = bbox[3] - bbox[1]
                        bbox_size = max(bbox_width, bbox_height)
                    else:
                        bbox_size = 68.0  # Default fallback
                    
                    # Tight crop calculation:
                    # crop_size = bbox_size / TARGET_OBJECT_COVERAGE
                    tight_crop_size = int(bbox_size / self.vis_config.target_object_coverage * (1 + self.vis_config.crop_margin))
                    
                    # Ensure minimum size for quality
                    crop_size = max(tight_crop_size, self.vis_config.min_crop_size)
                    
                    # Use detection coordinates for cropping
                    x1, y1, x2, y2 = detection.get_crop_region(
                        crop_size=crop_size,
                        image_width=frame_w,
                        image_height=frame_h,
                        use_bbox=False
                    )
                else:
                    # Fall back to center crop
                    crop_size = max(frame_w // 4, self.vis_config.min_crop_size)
                    size = min(frame_h, frame_w, crop_size)
                    x1 = (frame_w - size) // 2
                    y1 = (frame_h - size) // 2
                    x2 = x1 + size
                    y2 = y1 + size
                
                # Track crop region for intrinsics adjustment
                crop_regions.append((x1, y1, x2, y2))
                
                # Extract crop
                frame_cropped = frame[y1:y2, x1:x2]
                
                # Apply mask if available (Crucial for 3DGS reconstruction!)
                if detection and detection.mask_path:
                    try:
                        mask = cv2.imread(detection.mask_path, cv2.IMREAD_GRAYSCALE)
                        if mask is not None:
                            # Resize mask to match full frame if needed
                            if mask.shape[:2] != (frame_h, frame_w):
                                mask = cv2.resize(mask, (frame_w, frame_h), interpolation=cv2.INTER_NEAREST)
                            
                            # Crop mask
                            mask_cropped = mask[y1:y2, x1:x2]
                            
                            # Apply white background where mask is 0
                            # Threshold > 127 is foreground
                            _, fg_mask = cv2.threshold(mask_cropped, 127, 255, cv2.THRESH_BINARY)
                            
                            # Create white background
                            white_bg = np.ones_like(frame_cropped) * 255
                            
                            # Combine (broadcast mask to 3 channels)
                            fg_mask_bool = fg_mask > 0
                            frame_cropped = np.where(fg_mask_bool[..., None], frame_cropped, white_bg).astype(np.uint8)
                    except Exception as e:
                        logger.debug(f"Failed to load/apply mask: {e}")
                
                # Resize to model input size
                if frame_cropped.size > 0:
                    frame_cropped = cv2.resize(frame_cropped, (w, h))
                else:
                    frame_cropped = np.zeros((h, w, 3), dtype=np.uint8)
            
            # BGR to RGB
            frame_cropped = cv2.cvtColor(frame_cropped, cv2.COLOR_BGR2RGB)
            
            # Normalize to [0, 1] - NO ImageNet normalization!
            # The DepthSplat model expects [0, 1] range (same as torchvision.ToTensor())
            # ImageNet mean/std normalization would corrupt the input distribution
            frame_cropped = frame_cropped.astype(np.float32) / 255.0
            
            # HWC to CHW
            frame_cropped = np.transpose(frame_cropped, (2, 0, 1))
            processed.append(frame_cropped)
        
        # Store crop regions for calibration service
        self._last_crop_regions = crop_regions
        
        # Stack into batch: [num_cameras, 3, H, W]
        batch = np.stack(processed, axis=0)
        
        # Convert to tensor and add batch dimension: [1, num_cameras, 3, H, W]
        tensor = torch.from_numpy(batch).unsqueeze(0).to(self.device)
        
        return tensor

    def _processing_loop(self):
        """Extended processing loop with visualization."""
        frame_interval = self.config.frame_interval_us / 1e6

        # Determine frame source mode
        use_file_source = self.file_frame_source is not None
        if use_file_source:
            logger.info(
                f"Visualization pipeline using FILE SOURCE at {self.config.target_fps} FPS target"
            )
        else:
            logger.info(
                f"Visualization pipeline using RTSP streams at {self.config.target_fps} FPS target"
            )

        while self.running:
            loop_start = time.perf_counter()

            # Get the frame ID FIRST (for synchronization)
            if use_file_source:
                # File source: advance frame and get synchronized frame ID
                frame_id = self.file_frame_source.advance_frame()
                frames = self.file_frame_source.get_current_frames()
                logger.debug(f"File source frame {frame_id}")
            else:
                # RTSP: capture frames (no frame ID available)
                frames = self._capture_frames()
                frame_id = None

            self._last_raw_frames = frames

            # Skip if no valid frames
            valid_frames = [f for f in frames if f is not None]
            if not valid_frames:
                time.sleep(0.01)
                continue

            # Get detections for current frame from detection service
            # CRITICAL: Use the same frame_id for perfect synchronization
            detections = None
            if self.detection_service is not None:
                if use_file_source and frame_id is not None:
                    # File source: get detections for the EXACT frame ID
                    detections = self.detection_service.get_detections(frame_id)
                    if detections is not None:
                        logger.debug(
                            f"Frame {frame_id}: Got synchronized detections for "
                            f"{len(detections.detections)} cameras"
                        )
                else:
                    # RTSP: advance detection service independently (may drift!)
                    detections = self.detection_service.advance_frame()
                    if detections is not None:
                        logger.debug(
                            f"Frame {self.frame_count}: Got detections for "
                            f"{len(detections.detections)} cameras (may be out of sync)"
                        )

            # Preprocess for model with detection-based cropping
            input_tensor = self._preprocess_frames_with_detections(frames, detections)

            # Run inference and get gaussians for rendering
            # Pass detections to provide object position for pose normalization
            output, self._last_gaussians = self._run_inference_with_gaussians(input_tensor, detections=detections)

            # Update stats
            self.frame_count += 1
            self.total_inference_time += output.inference_time_ms

            # Callback
            if self._callback:
                self._callback(output)

            # Queue output
            try:
                self.output_queue.put_nowait(output)
            except:
                pass

            # Visualization streaming with detection-based cropping
            if self.vis_server and self._last_gaussians is not None:
                self._stream_visualization(
                    frames,
                    output,
                    detections=detections,
                )

            # Maintain target frame rate
            elapsed = time.perf_counter() - loop_start
            sleep_time = max(0, frame_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _get_gaussians_object(self, input_tensor):
        """
        Get the Gaussians object from encoder for rendering.

        This returns the Gaussians dataclass, not numpy arrays.
        """
        if self.model is None:
            return None

        try:
            import torch
            with torch.no_grad():
                b, v, c, h, w = input_tensor.shape

                # Get camera parameters from calibration service or use defaults
                # CRITICAL FIX: Don't pass crop_regions to match Gradio demo behavior
                if self.calibration_service is not None:
                    intrinsics = self.calibration_service.get_intrinsics_tensor(
                        device=self.device,
                        crop_regions=self._last_crop_regions,
                        use_virtual_camera=False,
                    )
                    extrinsics = self.calibration_service.get_extrinsics_tensor(
                        device=self.device,
                        crop_regions=None,  # FIX: Use original extrinsics
                        use_virtual_camera=False,
                    )
                else:
                    # Fallback to dummy parameters
                    intrinsics = torch.eye(3, device=self.device).unsqueeze(0).unsqueeze(0).expand(b, v, 3, 3).clone()
                    intrinsics[:, :, 0, 0] = 1.0
                    intrinsics[:, :, 1, 1] = 1.0
                    intrinsics[:, :, 0, 2] = 0.5
                    intrinsics[:, :, 1, 2] = 0.5

                    extrinsics = torch.eye(4, device=self.device).unsqueeze(0).unsqueeze(0).expand(b, v, 4, 4).clone()
                    for i in range(v):
                        angle = 2 * np.pi * i / v
                        radius = 1.5
                        extrinsics[:, i, 0, 3] = radius * np.cos(angle)
                        extrinsics[:, i, 1, 3] = 0.0
                        extrinsics[:, i, 2, 3] = radius * np.sin(angle)

                context = {
                    "image": input_tensor,
                    "intrinsics": intrinsics,
                    "extrinsics": extrinsics,
                    "near": torch.tensor([[0.55]], device=self.device).expand(b, v),
                    "far": torch.tensor([[2.54]], device=self.device).expand(b, v),
                }

                result = self.model.encoder(
                    context,
                    global_step=0,
                    deterministic=True,
                )

                # Handle both dict output (when return_depth=True) and direct Gaussians output
                if isinstance(result, dict):
                    gaussians = result.get("gaussians")
                else:
                    gaussians = result

                return gaussians

        except Exception as e:
            logger.error(f"Error getting Gaussians object: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _stream_visualization(
        self,
        raw_frames: List[np.ndarray],
        output: GaussianOutput,
        detections=None,
    ):
        """Stream visualization data to connected clients."""
        if self.vis_server is None:
            return

        try:
            total_start = time.perf_counter()
            column_latency = {}

            # Get render camera from server
            render_camera = self.vis_server.get_render_camera()

            # Render Gaussians
            gaussian_render = self._render_gaussians(
                self._last_gaussians,
                render_camera
            )

            # Create cropped views for all cameras using detection coordinates
            t0 = time.perf_counter()
            cropped_views = self._create_cropped_views(raw_frames, detections)
            column_latency["cropped_ms"] = (time.perf_counter() - t0) * 1000

            # Get ground truth depth for visualization (only if enabled)
            gt_depth_views = []
            gt_depth_enabled = self.vis_server.gt_depth_enabled if self.vis_server else False

            # Always advance GT depth service frame counter to stay in sync,
            # but only do the expensive depth reading/encoding when enabled
            if self.gt_depth_service is not None:
                # Advance frame counter regardless of enabled state
                self.gt_depth_service.advance_frame()

                if gt_depth_enabled and self._last_crop_regions:
                    from services import depth_to_colormap
                    t0 = time.perf_counter()
                    try:
                        # Get cropped depth per camera
                        for i, camera_name in enumerate([f"cam_{j+1:02d}" for j in range(self.config.num_cameras)]):
                            if i < len(self._last_crop_regions) and self._last_crop_regions[i]:
                                depth = self.gt_depth_service.get_cropped_depth(
                                    camera_name,
                                    self._last_crop_regions[i],
                                    output_size=(256, 256)
                                )
                                if depth is not None:
                                    # Use actual depth range in cropped region for better visualization
                                    # Exclude invalid depths (0, NaN, inf)
                                    valid_depth = depth[(depth > 0) & np.isfinite(depth)]
                                    if valid_depth.size > 0:
                                        crop_near = float(np.nanmin(valid_depth))
                                        crop_far = float(np.nanmax(valid_depth))
                                        # Sanity check
                                        if not np.isfinite(crop_near) or not np.isfinite(crop_far):
                                            crop_near = self.vis_config.depth_near
                                            crop_far = self.vis_config.depth_far
                                        else:
                                            # Add small margin to avoid clipping
                                            margin = (crop_far - crop_near) * 0.05
                                            crop_near = max(0.1, crop_near - margin)
                                            crop_far = crop_far + margin
                                    else:
                                        crop_near = self.vis_config.depth_near
                                        crop_far = self.vis_config.depth_far

                                    depth_vis = depth_to_colormap(
                                        depth,
                                        near=crop_near,
                                        far=crop_far
                                    )
                                    gt_depth_views.append(depth_vis)
                                else:
                                    gt_depth_views.append(None)
                            else:
                                gt_depth_views.append(None)
                    except Exception as e:
                        logger.debug(f"Failed to get GT depth: {e}")
                        gt_depth_views = [None] * self.config.num_cameras
                    column_latency["gt_depth_ms"] = (time.perf_counter() - t0) * 1000
                else:
                    gt_depth_views = [None] * self.config.num_cameras
                    column_latency["gt_depth_ms"] = 0.0
            else:
                gt_depth_views = [None] * self.config.num_cameras
                column_latency["gt_depth_ms"] = 0.0

            # Track input frame encoding time (estimated from total encode time)
            column_latency["input_ms"] = 0.0  # Will be updated below

            # Track other column latencies (these are produced during inference)
            column_latency["mono_depth_ms"] = 0.0  # Already computed in inference
            column_latency["predicted_depth_ms"] = 0.0  # Already computed in inference
            column_latency["silhouette_ms"] = 0.0  # Already computed in inference

            # Compute stats
            avg_render_time = np.mean(self.render_times) if self.render_times else 0
            avg_encode_time = np.mean(self.encode_times) if self.encode_times else 0

            stats = {
                "num_gaussians": int(output.positions.shape[0]) if output.positions.size > 0 else 0,
                "encoder_ms": output.inference_time_ms,
                "decoder_ms": avg_render_time,
                "total_latency_ms": output.inference_time_ms + avg_render_time + avg_encode_time,
                "fps": self.frame_count / max(1, time.time() - self.start_time) if self.start_time else 0,
                "column_latency": column_latency,
            }

            # Create frame packet with all visualizations
            vis_config = self.vis_config.get_visualization_config()
            packet = create_frame_packet(
                frame_id=output.frame_id,
                input_frames=raw_frames,
                cropped_frames=cropped_views,
                gaussian_render=gaussian_render,
                stats=stats,
                config=vis_config,
                gt_depth_frames=gt_depth_views if gt_depth_enabled else None,
                mono_depth_frames=self._last_mono_depth,
                predicted_depth_frames=self._last_predicted_depth,
                silhouette_frames=self._last_silhouette,
            )

            encode_time = (time.perf_counter() - total_start) * 1000
            self.encode_times.append(encode_time)
            if len(self.encode_times) > 100:
                self.encode_times.pop(0)

            # Queue for broadcast
            self.vis_server.queue_frame(packet)

        except Exception as e:
            logger.error(f"Visualization streaming error: {e}")

    def start(self, callback: Optional[Callable[[GaussianOutput], None]] = None):
        """Start the visualization pipeline."""
        # Start detection service if configured
        if self.detection_service is not None:
            self.detection_service.start()
            logger.info("Detection service started")
            
        # Start GT depth service if configured
        if self.gt_depth_service is not None:
            self.gt_depth_service.start()
            logger.info("GT depth service started")

        # Start visualization server in separate thread
        if self.vis_config.visualization_enabled:
            self._vis_thread = threading.Thread(
                target=self._run_visualization_server,
                daemon=True
            )
            self._vis_thread.start()
            # Give server time to start
            time.sleep(0.5)

        # Start the main pipeline
        super().start(callback)

    def stop(self):
        """Stop the visualization pipeline."""
        # Stop main pipeline first
        super().stop()

        # Stop detection service
        if self.detection_service is not None:
            self.detection_service.stop()
            logger.info("Detection service stopped")
            
        # Stop GT depth service
        if self.gt_depth_service is not None:
            self.gt_depth_service.stop()
            logger.info("GT depth service stopped")

        # Stop visualization server
        if self._vis_loop and self.vis_server:
            async def shutdown():
                await self.vis_server.stop()
                self._vis_loop.stop()

            asyncio.run_coroutine_threadsafe(shutdown(), self._vis_loop)

            if self._vis_thread:
                self._vis_thread.join(timeout=2.0)

        logger.info("Visualization pipeline stopped")

    def get_stats(self) -> Dict[str, Any]:
        """Get extended pipeline statistics."""
        base_stats = super().get_stats()

        vis_stats = {
            "render_time_ms": np.mean(self.render_times) if self.render_times else 0,
            "encode_time_ms": np.mean(self.encode_times) if self.encode_times else 0,
        }

        if self.vis_server:
            vis_stats.update(self.vis_server.get_stats())

        return {**base_stats, **vis_stats}


def create_visualization_pipeline(
    stream_urls: List[str],
    checkpoint_path: str,
    experiment_config: str = None,
    **kwargs
) -> VisualizationPipeline:
    """
    Factory function to create a visualization pipeline.

    Args:
        stream_urls: List of RTSP stream URLs
        checkpoint_path: Path to model checkpoint
        experiment_config: Hydra experiment config name
        **kwargs: Additional config options

    Returns:
        Configured VisualizationPipeline instance
    """
    config = VisualizationPipelineConfig(
        stream_urls=stream_urls,
        checkpoint_path=checkpoint_path,
        experiment_config=experiment_config,
        **kwargs
    )

    return VisualizationPipeline(config)
