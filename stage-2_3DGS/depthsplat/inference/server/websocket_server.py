"""
WebSocket server for real-time visualization streaming.

Broadcasts frame packets containing:
- Input camera thumbnails
- Cropped object region
- Rendered 3DGS output
- Performance statistics
"""

import asyncio
import json
import base64
import logging
import time
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Optional, Set, Callable, Dict, Any
from io import BytesIO

import numpy as np

try:
    import websockets
    from websockets.server import WebSocketServerProtocol
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    WebSocketServerProtocol = None  # Type hint placeholder

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

logger = logging.getLogger(__name__)


class ViewMode(str, Enum):
    """Render view modes for the 3DGS output."""
    ORBIT = "orbit"
    FRONT = "front"
    TOP = "top"
    SIDE = "side"
    INPUT_MATCH = "input_match"


@dataclass
class RenderCamera:
    """Camera parameters for rendering."""
    extrinsics: np.ndarray  # [4, 4] camera-to-world
    intrinsics: np.ndarray  # [3, 3] normalized
    near: float = 0.5
    far: float = 5.0


@dataclass
class FramePacket:
    """Data packet sent to clients each frame."""
    type: str = "frame"
    timestamp: float = 0.0
    frame_id: int = 0
    inputs: list = None  # Base64 JPEG thumbnails
    cropped: list = None  # Base64 JPEG cropped objects (one per camera)
    gt_depth: list = None  # Base64 JPEG ground truth depth (one per camera)
    mono_depth: list = None  # Base64 JPEG monocular depth (one per camera)
    predicted_depth: list = None  # Base64 JPEG MVS predicted depth (one per camera)
    silhouette: list = None  # Base64 JPEG silhouette/confidence (one per camera)
    gaussian_render: str = None  # Base64 JPEG rendered output
    stats: dict = None

    def __post_init__(self):
        if self.inputs is None:
            self.inputs = []
        if self.cropped is None:
            self.cropped = []
        if self.gt_depth is None:
            self.gt_depth = []
        if self.mono_depth is None:
            self.mono_depth = []
        if self.predicted_depth is None:
            self.predicted_depth = []
        if self.silhouette is None:
            self.silhouette = []
        if self.stats is None:
            self.stats = {}

    def to_json(self) -> str:
        return json.dumps(asdict(self))


@dataclass
class VisualizationConfig:
    """Configuration for the visualization server."""
    enabled: bool = True
    websocket_host: str = "0.0.0.0"
    websocket_port: int = 8765
    render_width: int = 512
    render_height: int = 512
    default_view_mode: str = "orbit"
    orbit_speed_deg_per_sec: float = 15.0
    input_thumbnail_width: int = 192
    input_thumbnail_height: int = 108
    jpeg_quality: int = 85  # Quality for main render
    thumbnail_quality: int = 70  # Lower quality for thumbnails/depth (faster)
    max_fps: int = 30


class VisualizationServer:
    """
    WebSocket server for streaming visualization data.

    Handles:
    - Client connections and disconnections
    - Frame packet broadcasting
    - View mode control messages
    - Render camera animation (orbit mode)
    """

    def __init__(self, config: VisualizationConfig):
        if not WEBSOCKETS_AVAILABLE:
            raise ImportError("websockets library required. Install with: pip install websockets")
        if not PIL_AVAILABLE:
            raise ImportError("PIL library required. Install with: pip install Pillow")

        self.config = config
        self.clients: Set[WebSocketServerProtocol] = set()
        self.server = None
        self.server_task = None
        self.broadcast_task = None
        self.running = False

        # View mode state
        self.view_mode = ViewMode(config.default_view_mode)
        self.orbit_angle = 0.0
        self.last_orbit_update = time.time()

        # GT depth enabled state (disabled by default for performance)
        self.gt_depth_enabled = False

        # Frame queue for broadcasting
        self._frame_queue: asyncio.Queue = None
        self._loop: asyncio.AbstractEventLoop = None

        # Statistics
        self.total_frames_sent = 0
        self.connected_clients = 0

    async def _handle_client(self, websocket: WebSocketServerProtocol):
        """Handle a single WebSocket client connection."""
        self.clients.add(websocket)
        self.connected_clients = len(self.clients)
        client_addr = websocket.remote_address
        logger.info(f"Client connected: {client_addr} (total: {self.connected_clients})")

        try:
            # Send initial state
            await websocket.send(json.dumps({
                "type": "init",
                "view_mode": self.view_mode.value,
                "gt_depth_enabled": self.gt_depth_enabled,
                "config": {
                    "render_width": self.config.render_width,
                    "render_height": self.config.render_height,
                    "input_thumbnail_width": self.config.input_thumbnail_width,
                    "input_thumbnail_height": self.config.input_thumbnail_height,
                }
            }))

            # Listen for messages from client
            async for message in websocket:
                await self._handle_message(websocket, message)

        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception as e:
            logger.error(f"Client error: {e}")
        finally:
            self.clients.discard(websocket)
            self.connected_clients = len(self.clients)
            logger.info(f"Client disconnected: {client_addr} (total: {self.connected_clients})")

    async def _handle_message(self, websocket: WebSocketServerProtocol, message: str):
        """Handle incoming message from client."""
        try:
            data = json.loads(message)
            msg_type = data.get("type")

            if msg_type == "set_view":
                mode = data.get("mode", "orbit")
                try:
                    self.view_mode = ViewMode(mode)
                    logger.info(f"View mode changed to: {mode}")
                    # Broadcast mode change to all clients
                    await self._broadcast(json.dumps({
                        "type": "view_mode_changed",
                        "mode": mode
                    }))
                except ValueError:
                    logger.warning(f"Invalid view mode: {mode}")

            elif msg_type == "set_gt_depth":
                enabled = data.get("enabled", False)
                self.gt_depth_enabled = enabled
                logger.info(f"GT depth enabled changed to: {enabled}")
                # Broadcast state change to all clients
                await self._broadcast(json.dumps({
                    "type": "gt_depth_changed",
                    "enabled": enabled
                }))

            elif msg_type == "ping":
                await websocket.send(json.dumps({
                    "type": "pong",
                    "timestamp": time.time() * 1000
                }))

        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON message: {message}")

    async def _broadcast(self, message: str):
        """Broadcast message to all connected clients."""
        if not self.clients:
            return

        # Create tasks for all clients
        tasks = [
            asyncio.create_task(client.send(message))
            for client in self.clients
        ]

        if tasks:
            # Wait with timeout, ignore failures
            done, pending = await asyncio.wait(
                tasks,
                timeout=0.1,
                return_when=asyncio.ALL_COMPLETED
            )
            for task in pending:
                task.cancel()

    async def _broadcast_loop(self):
        """Main loop for broadcasting frame packets."""
        while self.running:
            try:
                # Get frame packet from queue (with timeout)
                packet = await asyncio.wait_for(
                    self._frame_queue.get(),
                    timeout=0.1
                )

                if packet and self.clients:
                    await self._broadcast(packet.to_json())
                    self.total_frames_sent += 1

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Broadcast error: {e}")

    def queue_frame(self, packet: FramePacket):
        """Queue a frame packet for broadcasting (thread-safe)."""
        if self._loop and self._frame_queue and self.running:
            try:
                # Schedule the put in the event loop
                self._loop.call_soon_threadsafe(
                    self._frame_queue.put_nowait,
                    packet
                )
            except Exception:
                pass  # Queue full or loop closed

    def get_render_camera(self, base_intrinsics: np.ndarray = None) -> RenderCamera:
        """
        Get current render camera parameters based on view mode.

        Returns a RenderCamera with extrinsics and intrinsics for rendering.
        """
        # Default intrinsics - MUST use training-matched fx_norm = 1.0723 (50° FOV)
        # This matches what the Gradio demo uses and what the model was trained with
        TRAINING_FX_NORM = 1.0723
        if base_intrinsics is None:
            base_intrinsics = np.array([
                [TRAINING_FX_NORM, 0.0, 0.5],
                [0.0, TRAINING_FX_NORM, 0.5],
                [0.0, 0.0, 1.0]
            ], dtype=np.float32)

        # Update orbit angle if in orbit mode
        if self.view_mode == ViewMode.ORBIT:
            now = time.time()
            dt = now - self.last_orbit_update
            self.orbit_angle += self.config.orbit_speed_deg_per_sec * dt
            self.orbit_angle %= 360.0
            self.last_orbit_update = now

        # Compute extrinsics based on view mode
        extrinsics = self._compute_extrinsics()

        return RenderCamera(
            extrinsics=extrinsics,
            intrinsics=base_intrinsics,
            near=0.55,  # Match Gradio demo
            far=2.54,   # Match Gradio demo
        )

    def _compute_extrinsics(self) -> np.ndarray:
        """
        Compute camera extrinsics based on current view mode.

        Uses Z-up coordinate system matching the Gradio demo and DepthSplat training.
        Camera positions orbit in the XY plane with optional Z elevation.
        """
        # Camera distance - must match pose normalization target_radius (2.0)
        radius = 2.0
        elevation = 0.3  # Small Z offset for better viewing angle

        if self.view_mode == ViewMode.ORBIT:
            angle_rad = np.radians(self.orbit_angle)
            # XY plane orbit with Z elevation (matching Gradio demo)
            x = radius * np.cos(angle_rad)
            y = radius * np.sin(angle_rad)
            z = elevation
        elif self.view_mode == ViewMode.FRONT:
            x, y, z = 0.0, -radius, elevation
        elif self.view_mode == ViewMode.TOP:
            x, y, z = 0.0, 0.01, radius  # Looking down from +Z
        elif self.view_mode == ViewMode.SIDE:
            x, y, z = radius, 0.0, elevation
        else:  # INPUT_MATCH - use first camera position
            x, y, z = 0.0, -radius, elevation

        # Build camera-to-world matrix matching Gradio demo's create_orbit_extrinsics
        # Z-up coordinate system, camera looks at origin

        # Forward vector: from camera position toward origin
        forward = -np.array([x, y, z], dtype=np.float32)
        forward = forward / np.linalg.norm(forward)

        # Up vector: world Z axis
        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        # Right vector: cross(forward, up)
        right = np.cross(forward, up)
        norm = np.linalg.norm(right)
        if norm < 1e-6:
            # Handle singularity when looking straight up/down
            right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            right = right / norm

        # Recompute up to be orthogonal
        up = np.cross(right, forward)

        # Build rotation matrix: [right, up, -forward] as columns
        R = np.stack([right, up, -forward], axis=1)

        # Build 4x4 camera-to-world matrix
        extrinsics = np.eye(4, dtype=np.float32)
        extrinsics[:3, :3] = R
        extrinsics[:3, 3] = [x, y, z]

        return extrinsics

    async def start(self):
        """Start the WebSocket server."""
        if not self.config.enabled:
            logger.info("Visualization server disabled")
            return

        self._loop = asyncio.get_event_loop()
        self._frame_queue = asyncio.Queue(maxsize=5)
        self.running = True

        # Start WebSocket server
        self.server = await websockets.serve(
            self._handle_client,
            self.config.websocket_host,
            self.config.websocket_port
        )

        # Start broadcast loop
        self.broadcast_task = asyncio.create_task(self._broadcast_loop())

        logger.info(
            f"Visualization server started on "
            f"ws://{self.config.websocket_host}:{self.config.websocket_port}"
        )

    async def stop(self):
        """Stop the WebSocket server."""
        self.running = False

        if self.broadcast_task:
            self.broadcast_task.cancel()
            try:
                await self.broadcast_task
            except asyncio.CancelledError:
                pass

        if self.server:
            self.server.close()
            await self.server.wait_closed()

        # Close all client connections
        for client in list(self.clients):
            await client.close()
        self.clients.clear()

        logger.info("Visualization server stopped")

    def get_stats(self) -> Dict[str, Any]:
        """Get server statistics."""
        return {
            "connected_clients": self.connected_clients,
            "total_frames_sent": self.total_frames_sent,
            "view_mode": self.view_mode.value,
            "orbit_angle": self.orbit_angle if self.view_mode == ViewMode.ORBIT else None,
        }


def encode_image_to_base64(
    image: np.ndarray,
    quality: int = 85,
    resize: tuple = None
) -> str:
    """
    Encode a numpy image array to base64 JPEG.

    Uses TurboJPEG if available (10-20x faster than PIL), falls back to PIL.

    Args:
        image: HWC numpy array (RGB or BGR)
        quality: JPEG quality (1-100)
        resize: Optional (width, height) to resize

    Returns:
        Base64 encoded JPEG string
    """
    try:
        import cv2
        CV2_AVAILABLE = True
    except ImportError:
        CV2_AVAILABLE = False

    # Convert to uint8 if needed
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    # Resize if requested (use cv2 for speed if available)
    if resize:
        if CV2_AVAILABLE:
            image = cv2.resize(image, resize, interpolation=cv2.INTER_AREA)
        elif PIL_AVAILABLE:
            pil_image = Image.fromarray(image)
            pil_image = pil_image.resize(resize, Image.Resampling.LANCZOS)
            image = np.array(pil_image)

    # Try TurboJPEG first (much faster)
    try:
        from turbojpeg import TurboJPEG
        jpeg = TurboJPEG()
        # TurboJPEG expects BGR for encoding
        if image.shape[2] == 3:
            # Assume input is RGB, convert to BGR for TurboJPEG
            bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if CV2_AVAILABLE else image[:, :, ::-1]
        else:
            bgr = image
        jpeg_bytes = jpeg.encode(bgr, quality=quality)
        return base64.b64encode(jpeg_bytes).decode("utf-8")
    except ImportError:
        pass  # Fall back to PIL
    except Exception as e:
        logger.debug(f"TurboJPEG encoding failed, falling back to PIL: {e}")

    # Fallback to PIL
    if not PIL_AVAILABLE:
        raise ImportError("PIL required for image encoding")

    pil_image = Image.fromarray(image)
    buffer = BytesIO()
    pil_image.save(buffer, format="JPEG", quality=quality, optimize=False)  # optimize=False is faster
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def create_frame_packet(
    frame_id: int,
    input_frames: list,
    cropped_frames: list,
    gaussian_render: np.ndarray,
    stats: dict,
    config: VisualizationConfig,
    gt_depth_frames: list = None,
    mono_depth_frames: list = None,
    predicted_depth_frames: list = None,
    silhouette_frames: list = None,
) -> FramePacket:
    """
    Create a frame packet from raw data.

    Args:
        frame_id: Sequential frame number
        input_frames: List of HWC numpy arrays (BGR format from OpenCV)
        cropped_frames: List of HWC numpy arrays of cropped objects (one per camera)
        gaussian_render: HWC numpy array of rendered 3DGS
        stats: Dictionary of performance statistics
        config: Visualization configuration
        gt_depth_frames: List of HWC numpy arrays of GT depth visualizations (one per camera)
        mono_depth_frames: List of HWC numpy arrays of monocular depth (one per camera)
        predicted_depth_frames: List of HWC numpy arrays of predicted depth (one per camera)
        silhouette_frames: List of HWC numpy arrays of silhouettes (one per camera)

    Returns:
        FramePacket ready for transmission
    """
    import cv2

    thumbnail_size = (config.input_thumbnail_width, config.input_thumbnail_height)

    # Track per-column encoding latency
    column_latency = stats.get("column_latency", {})

    def encode_frame_list(frames, convert_bgr=True, resize=None, quality=None):
        """Helper to encode a list of frames to base64."""
        if not frames:
            return []
        if quality is None:
            quality = config.thumbnail_quality
        result = []
        for frame in frames:
            if frame is not None:
                if convert_bgr and len(frame.shape) == 3 and frame.shape[2] == 3:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else:
                    rgb = frame
                b64 = encode_image_to_base64(rgb, quality, resize=resize)
                result.append(b64)
            else:
                result.append(None)
        return result

    def timed_encode(frames, convert_bgr=True, resize=None, quality=None):
        """Helper to encode frames and return (result, time_ms)."""
        if not frames or all(f is None for f in frames):
            return [], 0.0
        t0 = time.perf_counter()
        result = encode_frame_list(frames, convert_bgr, resize=resize, quality=quality)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return result, elapsed_ms

    # Use thumbnail_quality (70) for thumbnails - faster encoding
    thumbnail_q = config.thumbnail_quality

    # Encode input thumbnails with lower quality for speed
    inputs, input_encode_ms = timed_encode(input_frames, convert_bgr=True, resize=thumbnail_size, quality=thumbnail_q)
    column_latency["input_ms"] = input_encode_ms

    # Encode cropped frames with thumbnail quality - already 256x256
    cropped_b64_list, cropped_encode_ms = timed_encode(cropped_frames, convert_bgr=True, resize=None, quality=thumbnail_q)
    column_latency["cropped_ms"] = column_latency.get("cropped_ms", 0) + cropped_encode_ms

    # Encode depth and silhouette visualizations with thumbnail quality
    # Skip encoding if lists are empty/None
    gt_depth_b64, gt_encode_ms = timed_encode(gt_depth_frames, convert_bgr=False, resize=None, quality=thumbnail_q)
    column_latency["gt_depth_ms"] = column_latency.get("gt_depth_ms", 0) + gt_encode_ms

    mono_depth_b64, mono_encode_ms = timed_encode(mono_depth_frames, convert_bgr=False, resize=None, quality=thumbnail_q)
    column_latency["mono_depth_ms"] = column_latency.get("mono_depth_ms", 0) + mono_encode_ms

    predicted_depth_b64, pred_encode_ms = timed_encode(predicted_depth_frames, convert_bgr=False, resize=None, quality=thumbnail_q)
    column_latency["predicted_depth_ms"] = column_latency.get("predicted_depth_ms", 0) + pred_encode_ms

    silhouette_b64, sil_encode_ms = timed_encode(silhouette_frames, convert_bgr=False, resize=None, quality=thumbnail_q)
    column_latency["silhouette_ms"] = column_latency.get("silhouette_ms", 0) + sil_encode_ms

    # Encode gaussian render with high quality (this is the main output)
    render_b64 = None
    if gaussian_render is not None:
        # Assume already RGB, use higher quality for main render
        render_b64 = encode_image_to_base64(gaussian_render, config.jpeg_quality)

    # Update stats with column latency
    stats["column_latency"] = column_latency

    return FramePacket(
        type="frame",
        timestamp=time.time() * 1000,  # Unix ms
        frame_id=frame_id,
        inputs=inputs,
        cropped=cropped_b64_list,
        gt_depth=gt_depth_b64,
        mono_depth=mono_depth_b64,
        predicted_depth=predicted_depth_b64,
        silhouette=silhouette_b64,
        gaussian_render=render_b64,
        stats=stats
    )

