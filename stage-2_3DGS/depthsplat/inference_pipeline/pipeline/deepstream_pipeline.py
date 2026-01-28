"""
DepthSplat Inference Pipeline.

This module implements the inference pipeline that:
1. Receives synchronized camera streams from RTSP
2. Preprocesses frames
3. Runs PyTorch inference
4. Outputs Gaussian parameters
"""

import sys
import logging
import time
import threading
from pathlib import Path
from dataclasses import dataclass
from typing import List, Callable, Optional, Dict, Any
from queue import Queue

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the inference pipeline."""

    # Stream sources (RTSP URLs)
    stream_urls: List[str]

    # Model paths
    checkpoint_path: str  # PyTorch checkpoint (.ckpt)
    experiment_config: Optional[str] = None  # Experiment config name (e.g., "objaverse_white_small_gauss")

    # Input dimensions (after preprocessing)
    model_input_width: int = 256
    model_input_height: int = 256

    # Frame timing
    target_fps: int = 30

    # Performance settings
    gpu_id: int = 0

    @property
    def num_cameras(self) -> int:
        return len(self.stream_urls)

    @property
    def frame_interval_us(self) -> int:
        """Frame interval in microseconds."""
        return int(1e6 / self.target_fps)


@dataclass
class GaussianOutput:
    """Container for 3D Gaussian output from inference."""
    frame_id: int
    timestamp: float
    positions: np.ndarray      # [N, 3] XYZ positions
    covariances: np.ndarray    # [N, 6] covariance parameters
    colors: np.ndarray         # [N, C] RGB or SH coefficients
    opacities: np.ndarray      # [N, 1] opacity values
    inference_time_ms: float


class DepthSplatPipeline:
    """
    PyTorch-based pipeline for DepthSplat inference.

    Captures RTSP streams with OpenCV and runs the model directly.

    Usage:
        config = PipelineConfig(
            stream_urls=["rtsp://..."],
            checkpoint_path="/path/to/checkpoint.ckpt",
        )
        pipeline = DepthSplatPipeline(config)
        pipeline.start(callback=my_handler)
        # ... run ...
        pipeline.stop()
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.running = False
        self.output_queue: Queue[GaussianOutput] = Queue(maxsize=30)
        self.frame_count = 0
        self.total_inference_time = 0.0
        self.start_time = None

        self.thread = None
        self._callback = None
        self.model = None
        self.captures = []
        self.device = None

        # Import torch
        try:
            import torch
            self.torch = torch
            self.device = torch.device(f"cuda:{config.gpu_id}" if torch.cuda.is_available() else "cpu")
            logger.info(f"PyTorch device: {self.device}")
        except ImportError:
            raise ImportError("PyTorch not available. Install with: pip install torch")

    def _load_model(self):
        """Load the DepthSplat model from checkpoint using Hydra config."""
        import sys
        from pathlib import Path
        import torch

        # Add the parent directory to path to import DepthSplat modules
        depthsplat_root = Path(__file__).parent.parent.parent
        if str(depthsplat_root) not in sys.path:
            sys.path.insert(0, str(depthsplat_root))

        checkpoint_path = self.config.checkpoint_path
        if not checkpoint_path:
            raise ValueError("checkpoint_path is required")

        experiment_config = self.config.experiment_config
        if not experiment_config:
            # Try to infer from checkpoint path
            ckpt_path = Path(checkpoint_path)
            output_name = ckpt_path.parent.parent.name  # e.g., "objaverse_white_small_gauss"
            experiment_config = output_name
            logger.info(f"Inferred experiment config: {experiment_config}")

        logger.info(f"Loading model from: {checkpoint_path}")
        logger.info(f"Using experiment config: {experiment_config}")

        try:
            # Use Hydra to load the config
            from hydra import compose, initialize_config_dir
            from hydra.core.global_hydra import GlobalHydra

            config_dir = str(depthsplat_root / "config")

            # Clear any existing Hydra instance
            GlobalHydra.instance().clear()

            # Initialize Hydra with the config directory
            with initialize_config_dir(version_base=None, config_dir=config_dir):
                cfg = compose(config_name="main", overrides=[f"+experiment={experiment_config}"])

            # Import model components
            from src.model.encoder import get_encoder
            from src.model.decoder import get_decoder
            from src.model.model_wrapper import ModelWrapper
            from src.misc.step_tracker import StepTracker
            from src.config import load_typed_root_config

            # Convert to typed config
            typed_cfg = load_typed_root_config(cfg)

            # Build encoder
            logger.info("Building encoder...")
            encoder, encoder_visualizer = get_encoder(typed_cfg.model.encoder)

            # Build decoder
            logger.info("Building decoder...")
            decoder = get_decoder(typed_cfg.model.decoder, typed_cfg.dataset)

            # Create step tracker
            step_tracker = StepTracker()

            # Create model wrapper
            logger.info("Creating model wrapper...")
            self.model = ModelWrapper(
                optimizer_cfg=typed_cfg.optimizer,
                test_cfg=typed_cfg.test,
                train_cfg=typed_cfg.train,
                encoder=encoder,
                encoder_visualizer=encoder_visualizer,
                decoder=decoder,
                losses=[],  # Empty for inference
                step_tracker=step_tracker,
            )

            # Load checkpoint
            logger.info("Loading checkpoint...")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            state_dict = checkpoint.get("state_dict", checkpoint)

            # Load state dict
            logger.info("Loading state dict...")
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            if missing:
                logger.warning(f"Missing keys: {len(missing)}")
            if unexpected:
                logger.warning(f"Unexpected keys: {len(unexpected)}")

            self.model.to(self.device)
            self.model.eval()

            logger.info("Model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _init_captures(self):
        """Initialize OpenCV video captures for RTSP streams."""
        import cv2

        self.captures = []
        for i, url in enumerate(self.config.stream_urls):
            logger.info(f"Connecting to stream {i}: {url}")

            # Use GStreamer backend for better RTSP handling
            cap = cv2.VideoCapture(url, cv2.CAP_GSTREAMER)

            if not cap.isOpened():
                # Fallback to FFmpeg
                cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

            if not cap.isOpened():
                # Last resort - default backend
                cap = cv2.VideoCapture(url)

            if cap.isOpened():
                logger.info(f"  Connected to stream {i}")
                self.captures.append(cap)
            else:
                logger.warning(f"  Failed to connect to stream {i}: {url}")
                self.captures.append(None)

    def _preprocess_frames(self, frames: List[np.ndarray]) -> "torch.Tensor":
        """Preprocess frames for model input."""
        import cv2
        import torch

        processed = []
        h, w = self.config.model_input_height, self.config.model_input_width

        for frame in frames:
            if frame is None:
                # Create blank frame if capture failed
                frame = np.zeros((h, w, 3), dtype=np.uint8)

            # Resize
            frame = cv2.resize(frame, (w, h))

            # BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Normalize to [0, 1] - NO ImageNet normalization!
            # The DepthSplat model expects [0, 1] range (same as torchvision.ToTensor())
            # ImageNet mean/std normalization would corrupt the input distribution
            frame = frame.astype(np.float32) / 255.0

            # HWC to CHW
            frame = np.transpose(frame, (2, 0, 1))
            processed.append(frame)

        # Stack into batch: [num_cameras, 3, H, W]
        batch = np.stack(processed, axis=0)

        # Convert to tensor and add batch dimension: [1, num_cameras, 3, H, W]
        tensor = torch.from_numpy(batch).unsqueeze(0).to(self.device)

        return tensor

    def _capture_frames(self) -> List[np.ndarray]:
        """Capture a frame from each RTSP stream."""
        frames = []
        for cap in self.captures:
            if cap is not None and cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
                else:
                    frames.append(None)
            else:
                frames.append(None)
        return frames

    def _run_inference(self, input_tensor: "torch.Tensor") -> GaussianOutput:
        """Run model inference and extract Gaussian parameters."""
        import torch

        with torch.no_grad():
            start_time = time.perf_counter()

            try:
                b, v, c, h, w = input_tensor.shape

                # Create dummy camera parameters for testing
                # In production, these should come from camera calibration
                # Intrinsics: normalized [fx, 0, cx; 0, fy, cy; 0, 0, 1]
                intrinsics = torch.eye(3, device=self.device).unsqueeze(0).unsqueeze(0).expand(b, v, 3, 3)
                fx = fy = 1.0
                cx = cy = 0.5
                intrinsics = intrinsics.clone()
                intrinsics[:, :, 0, 0] = fx
                intrinsics[:, :, 1, 1] = fy
                intrinsics[:, :, 0, 2] = cx
                intrinsics[:, :, 1, 2] = cy

                # Extrinsics: camera-to-world transforms [4, 4]
                # Place cameras in a circle around the origin
                extrinsics = torch.eye(4, device=self.device).unsqueeze(0).unsqueeze(0).expand(b, v, 4, 4).clone()
                for i in range(v):
                    angle = 2 * np.pi * i / v
                    radius = 1.5
                    extrinsics[:, i, 0, 3] = radius * np.cos(angle)
                    extrinsics[:, i, 1, 3] = 0.0
                    extrinsics[:, i, 2, 3] = radius * np.sin(angle)

                # Create context dict for encoder
                context = {
                    "image": input_tensor,
                    "intrinsics": intrinsics,
                    "extrinsics": extrinsics,
                    "near": torch.tensor([[0.55]], device=self.device).expand(b, v),
                    "far": torch.tensor([[2.54]], device=self.device).expand(b, v),
                }

                # Run encoder directly
                gaussians = self.model.encoder(
                    context,
                    global_step=0,
                    deterministic=True,
                )

                inference_time = (time.perf_counter() - start_time) * 1000

                # Extract Gaussian parameters
                if hasattr(gaussians, "means"):
                    positions = gaussians.means.cpu().numpy()
                else:
                    positions = np.zeros((100, 3), dtype=np.float32)

                if hasattr(gaussians, "covariances"):
                    covariances = gaussians.covariances.cpu().numpy()
                else:
                    covariances = np.zeros((positions.shape[0], 6), dtype=np.float32)

                if hasattr(gaussians, "harmonics"):
                    colors = gaussians.harmonics.cpu().numpy()
                elif hasattr(gaussians, "sh"):
                    colors = gaussians.sh.cpu().numpy()
                else:
                    colors = np.zeros((positions.shape[0], 3), dtype=np.float32)

                if hasattr(gaussians, "opacities"):
                    opacities = gaussians.opacities.cpu().numpy()
                else:
                    opacities = np.ones((positions.shape[0], 1), dtype=np.float32)

                # Flatten batch dimensions if needed
                if positions.ndim > 2:
                    positions = positions.reshape(-1, 3)
                if covariances.ndim > 2:
                    covariances = covariances.reshape(-1, covariances.shape[-1])
                if colors.ndim > 2:
                    colors = colors.reshape(-1, colors.shape[-1])
                if opacities.ndim > 2:
                    opacities = opacities.reshape(-1, 1)

            except Exception as e:
                logger.error(f"Inference error: {e}")
                import traceback
                traceback.print_exc()
                inference_time = (time.perf_counter() - start_time) * 1000
                # Return empty output on error
                positions = np.zeros((0, 3), dtype=np.float32)
                covariances = np.zeros((0, 6), dtype=np.float32)
                colors = np.zeros((0, 3), dtype=np.float32)
                opacities = np.zeros((0, 1), dtype=np.float32)

        return GaussianOutput(
            frame_id=self.frame_count,
            timestamp=time.time(),
            positions=positions,
            covariances=covariances,
            colors=colors,
            opacities=opacities,
            inference_time_ms=inference_time,
        )

    def _processing_loop(self):
        """Main processing loop."""
        frame_interval = self.config.frame_interval_us / 1e6  # Convert to seconds

        logger.info(f"Pipeline processing at {self.config.target_fps} FPS target")

        while self.running:
            loop_start = time.perf_counter()

            # Capture frames from all streams
            frames = self._capture_frames()

            # Skip if no valid frames
            valid_frames = [f for f in frames if f is not None]
            if not valid_frames:
                time.sleep(0.01)
                continue

            # Preprocess
            input_tensor = self._preprocess_frames(frames)

            # Run inference
            output = self._run_inference(input_tensor)

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
                pass  # Queue full, drop frame

            # Maintain target frame rate
            elapsed = time.perf_counter() - loop_start
            sleep_time = max(0, frame_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def start(self, callback: Optional[Callable[[GaussianOutput], None]] = None):
        """Start the pipeline."""
        self._callback = callback

        # Load model
        self._load_model()

        # Initialize stream captures
        self._init_captures()

        connected = sum(1 for c in self.captures if c is not None and c.isOpened())
        logger.info(f"Connected to {connected}/{len(self.config.stream_urls)} streams")

        if connected == 0:
            logger.warning("No streams connected! Check RTSP URLs.")

        self.running = True
        self.start_time = time.time()

        self.thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.thread.start()

        logger.info("Pipeline started")

    def stop(self):
        """Stop the pipeline."""
        self.running = False

        if self.thread:
            self.thread.join(timeout=2.0)

        # Release captures
        for cap in self.captures:
            if cap is not None:
                cap.release()
        self.captures = []

        # Free model
        if self.model is not None:
            del self.model
            self.model = None

            # Clear CUDA cache
            if self.torch.cuda.is_available():
                self.torch.cuda.empty_cache()

        logger.info("Pipeline stopped")

    def wait(self):
        """Wait for the pipeline to finish."""
        if self.thread:
            self.thread.join()

    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        elapsed = time.time() - self.start_time if self.start_time else 0
        avg_inference = (
            self.total_inference_time / self.frame_count
            if self.frame_count > 0 else 0
        )
        return {
            "frame_count": self.frame_count,
            "elapsed_time": elapsed,
            "fps": self.frame_count / elapsed if elapsed > 0 else 0,
            "avg_inference_ms": avg_inference,
        }
