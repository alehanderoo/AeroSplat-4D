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
    crop_size: int = 1024  # Size of cropped region (before resize to model input)

    # Camera calibration settings
    calibration_json_path: Optional[str] = None  # Path to calibration JSON (defaults to detection_gt_path)
    
    # Ground truth depth visualization settings
    gt_depth_base_path: Optional[str] = None  # Base path to GT depth renders
    depth_near: float = 0.5  # Near plane for depth colormap
    depth_far: float = 100.0  # Far plane for depth colormap

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
                )
                logger.info(f"Camera calibration service initialized from: {json_path}")
            except Exception as e:
                logger.warning(f"Failed to initialize calibration service: {e}")
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

    def _run_inference_with_gaussians(self, input_tensor):
        """
        Run model inference and return both GaussianOutput and Gaussians dataclass.

        This avoids running the encoder twice per frame.
        Also extracts depth and silhouette visualizations.
        """
        import torch
        import torch.nn.functional as F
        from .deepstream_pipeline import GaussianOutput
        from services import depth_to_colormap

        with torch.no_grad():
            start_time = time.perf_counter()

            try:
                b, v, c, h, w = input_tensor.shape

                # Get camera parameters from calibration service or use defaults
                # With virtual camera transformation, both intrinsics and extrinsics
                # are adjusted to account for the crop offset, preserving multi-view geometry
                if self.calibration_service is not None:
                    intrinsics = self.calibration_service.get_intrinsics_tensor(
                        device=self.device,
                        crop_regions=self._last_crop_regions,
                        use_virtual_camera=True,
                    )
                    extrinsics = self.calibration_service.get_extrinsics_tensor(
                        device=self.device,
                        crop_regions=self._last_crop_regions,
                        use_virtual_camera=True,
                    )

                    # Log intrinsics for debugging (first camera only)
                    if self._last_crop_regions:
                        logger.debug(
                            f"Virtual camera intrinsics[0]: fx={intrinsics[0,0,0,0]:.3f}, "
                            f"cx={intrinsics[0,0,0,2]:.3f}, cy={intrinsics[0,0,1,2]:.3f}"
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

                # Use visualization_dump to capture intermediate outputs
                visualization_dump = {}
                
                # Run encoder once
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
                    # Get depths if available from dict
                    predicted_depths = result.get("depths")  # [B, V, H, W] if return_depth=True
                else:
                    gaussians = result
                    predicted_depths = None
                    
                # Try to get depth from visualization_dump
                if "depth" in visualization_dump and predicted_depths is None:
                    predicted_depths = visualization_dump["depth"]  # [B, V, H, W, srf, s]
                    if predicted_depths.dim() > 4:
                        predicted_depths = predicted_depths.squeeze(-1).squeeze(-1)  # [B, V, H, W]
                
                # Extract and visualize depth per camera
                self._last_predicted_depth = []
                self._last_silhouette = []
                self._last_mono_depth = []  # Placeholder for now
                
                if predicted_depths is not None:
                    # Predicted depth: [B, V, H, W]
                    for cam_idx in range(v):
                        depth_cam = predicted_depths[0, cam_idx].cpu().numpy()  # [H, W]
                        # Apply colormap
                        depth_vis = depth_to_colormap(
                            depth_cam, 
                            near=self.vis_config.depth_near,
                            far=self.vis_config.depth_far
                        )
                        self._last_predicted_depth.append(depth_vis)
                else:
                    self._last_predicted_depth = [None] * v
                
                # For silhouette, we use the opacities from Gaussians (reshaped back to image)
                if gaussians is not None and hasattr(gaussians, "opacities"):
                    # Opacities are [B, V*H*W*srf*spp]
                    # We need to reshape back to [V, H, W]
                    try:
                        opacities = gaussians.opacities  # [B, N]
                        # Compute per-pixel mean opacity as silhouette
                        # N = V * H * W * srf * spp, we need to sum over srf*spp
                        n_gaussians_per_pixel = opacities.shape[1] // (v * h * w)
                        if n_gaussians_per_pixel > 0:
                            opacity_per_pixel = opacities.reshape(b, v, h, w, -1).mean(dim=-1)  # [B, V, H, W]
                            for cam_idx in range(v):
                                opacity_cam = opacity_per_pixel[0, cam_idx].cpu().numpy()  # [H, W]
                                # Convert to grayscale visualization
                                opacity_vis = (np.clip(opacity_cam, 0, 1) * 255).astype(np.uint8)
                                opacity_rgb = np.stack([opacity_vis, opacity_vis, opacity_vis], axis=-1)
                                self._last_silhouette.append(opacity_rgb)
                        else:
                            self._last_silhouette = [None] * v
                    except Exception as e:
                        logger.debug(f"Failed to extract silhouette: {e}")
                        self._last_silhouette = [None] * v
                else:
                    self._last_silhouette = [None] * v
                    
                # Extract monocular depth from visualization_dump
                mono_depths = visualization_dump.get("mono_depth")  # [B, V, H, W]
                if mono_depths is not None:
                    for cam_idx in range(v):
                        mono_cam = mono_depths[0, cam_idx].cpu().numpy()  # [H, W]
                        # Mono depth is residual (can be negative), visualize with different range
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

                    # Flatten batch dimensions if needed
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
        Render Gaussians using the decoder.

        Args:
            gaussians: Gaussian parameters from encoder
            render_camera: RenderCamera with extrinsics/intrinsics

        Returns:
            HWC RGB numpy array of rendered image, or None on error
        """
        if self.decoder is None or gaussians is None:
            return None

        try:
            import torch
            render_start = time.perf_counter()

            with torch.no_grad():
                # Get render resolution
                h, w = self.vis_config.render_height, self.vis_config.render_width

                # Prepare camera parameters
                device = self.device

                # Extrinsics: [1, 1, 4, 4]
                extrinsics = torch.from_numpy(
                    render_camera.extrinsics
                ).float().to(device).unsqueeze(0).unsqueeze(0)

                # Intrinsics: [1, 1, 3, 3]
                intrinsics = torch.from_numpy(
                    render_camera.intrinsics
                ).float().to(device).unsqueeze(0).unsqueeze(0)

                # Near/far: [1, 1]
                near = torch.tensor([[render_camera.near]], device=device)
                far = torch.tensor([[render_camera.far]], device=device)

                # Call decoder
                output = self.decoder(
                    gaussians,
                    extrinsics,
                    intrinsics,
                    near,
                    far,
                    image_shape=(h, w),
                )

                # Extract color output: [1, 1, 3, H, W] -> [H, W, 3]
                color = output.color[0, 0]  # [3, H, W]
                color = color.permute(1, 2, 0)  # [H, W, 3]
                color = color.clamp(0, 1)
                color = (color * 255).byte().cpu().numpy()

            render_time = (time.perf_counter() - render_start) * 1000
            self.render_times.append(render_time)
            if len(self.render_times) > 100:
                self.render_times.pop(0)

            return color

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
        crop_size = self.vis_config.crop_size

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
                # Use detection coordinates for cropping
                x1, y1, x2, y2 = detection.get_crop_region(
                    crop_size=crop_size,
                    image_width=w,
                    image_height=h,
                    use_bbox=True  # Use bbox if available for better sizing
                )
            else:
                # Fall back to center crop
                size = min(h, w, crop_size)
                x1 = (w - size) // 2
                y1 = (h - size) // 2
                x2 = x1 + size
                y2 = y1 + size

            # Extract crop
            cropped = frame[y1:y2, x1:x2]

            # Resize to standard output size
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
        crop_size = self.vis_config.crop_size
        
        for i, frame in enumerate(frames):
            if frame is None:
                # Create blank frame if capture failed
                frame_cropped = np.zeros((h, w, 3), dtype=np.uint8)
                # Use center crop region as fallback
                crop_regions.append((0, 0, crop_size, crop_size))
            else:
                frame_h, frame_w = frame.shape[:2]
                camera_name = f"cam_{i+1:02d}"
                
                # Try to get detection coordinates for cropping
                detection = None
                if detections is not None:
                    detection = detections.get_detection(camera_name)
                
                if detection is not None and detection.visible:
                    # Use detection coordinates for cropping
                    x1, y1, x2, y2 = detection.get_crop_region(
                        crop_size=crop_size,
                        image_width=frame_w,
                        image_height=frame_h,
                        use_bbox=True
                    )
                else:
                    # Fall back to center crop
                    size = min(frame_h, frame_w, crop_size)
                    x1 = (frame_w - size) // 2
                    y1 = (frame_h - size) // 2
                    x2 = x1 + size
                    y2 = y1 + size
                
                # Track crop region for intrinsics adjustment
                crop_regions.append((x1, y1, x2, y2))
                
                # Extract crop
                frame_cropped = frame[y1:y2, x1:x2]
                
                # Resize to model input size
                if frame_cropped.size > 0:
                    frame_cropped = cv2.resize(frame_cropped, (w, h))
                else:
                    frame_cropped = np.zeros((h, w, 3), dtype=np.uint8)
            
            # BGR to RGB
            frame_cropped = cv2.cvtColor(frame_cropped, cv2.COLOR_BGR2RGB)
            
            # Normalize to [0, 1]
            frame_cropped = frame_cropped.astype(np.float32) / 255.0
            
            # ImageNet normalization
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            frame_cropped = (frame_cropped - mean) / std
            
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

        logger.info(f"Visualization pipeline processing at {self.config.target_fps} FPS target")

        while self.running:
            loop_start = time.perf_counter()

            # Capture frames from all streams
            frames = self._capture_frames()
            self._last_raw_frames = frames

            # Skip if no valid frames
            valid_frames = [f for f in frames if f is not None]
            if not valid_frames:
                time.sleep(0.01)
                continue

            # Get detections for current frame from detection service
            detections = None
            if self.detection_service is not None:
                detections = self.detection_service.advance_frame()
                if detections is not None:
                    logger.debug(
                        f"Frame {self.frame_count}: Got detections for "
                        f"{len(detections.detections)} cameras"
                    )

            # Preprocess for model with detection-based cropping
            input_tensor = self._preprocess_frames_with_detections(frames, detections)

            # Run inference and get gaussians for rendering
            output, self._last_gaussians = self._run_inference_with_gaussians(input_tensor)

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
                # With virtual camera transformation for crop regions
                if self.calibration_service is not None:
                    intrinsics = self.calibration_service.get_intrinsics_tensor(
                        device=self.device,
                        crop_regions=self._last_crop_regions,
                        use_virtual_camera=True,
                    )
                    extrinsics = self.calibration_service.get_extrinsics_tensor(
                        device=self.device,
                        crop_regions=self._last_crop_regions,
                        use_virtual_camera=True,
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
            encode_start = time.perf_counter()

            # Get render camera from server
            render_camera = self.vis_server.get_render_camera()

            # Render Gaussians
            gaussian_render = self._render_gaussians(
                self._last_gaussians,
                render_camera
            )

            # Create cropped views for all cameras using detection coordinates
            cropped_views = self._create_cropped_views(raw_frames, detections)
            
            # Get ground truth depth for visualization
            gt_depth_views = []
            if self.gt_depth_service is not None and self._last_crop_regions:
                from services import depth_to_colormap
                try:
                    # Advance GT depth service frame to stay in sync
                    self.gt_depth_service.advance_frame()
                    
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
            else:
                gt_depth_views = [None] * self.config.num_cameras

            # Compute stats
            avg_render_time = np.mean(self.render_times) if self.render_times else 0
            avg_encode_time = np.mean(self.encode_times) if self.encode_times else 0

            stats = {
                "num_gaussians": int(output.positions.shape[0]) if output.positions.size > 0 else 0,
                "encoder_ms": output.inference_time_ms,
                "decoder_ms": avg_render_time,
                "total_latency_ms": output.inference_time_ms + avg_render_time + avg_encode_time,
                "fps": self.frame_count / max(1, time.time() - self.start_time) if self.start_time else 0,
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
                gt_depth_frames=gt_depth_views,
                mono_depth_frames=self._last_mono_depth,
                predicted_depth_frames=self._last_predicted_depth,
                silhouette_frames=self._last_silhouette,
            )

            encode_time = (time.perf_counter() - encode_start) * 1000
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
