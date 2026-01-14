"""
Frame Publisher for accurate frame-by-frame playback control.

This module provides precise timing control for frame publishing,
useful for synchronized multi-camera playback and testing.
"""

import time
import threading
import logging
from pathlib import Path
from typing import Optional, Callable, Any, List
from dataclasses import dataclass
from queue import Queue, Empty
import numpy as np

logger = logging.getLogger(__name__)

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    logger.warning("OpenCV not available, frame loading will be limited")


@dataclass
class FrameInfo:
    """Information about a published frame."""
    frame_id: int
    timestamp: float
    camera_id: str
    path: Path
    data: Optional[np.ndarray] = None


class FramePublisher:
    """
    Publishes frames from a sequence at precise intervals.

    Features:
    - Accurate timing using high-resolution timer
    - Optional frame caching for low-latency access
    - Callback support for frame events
    - Synchronized playback control

    Usage:
        from stream_simulator.rtsp_server import CameraStreamConfig

        config = CameraStreamConfig(
            camera_id="cam_01",
            rgb_dir=Path("/path/to/rgb"),
            fps=30
        )

        def on_frame(frame_info):
            print(f"Frame {frame_info.frame_id} at {frame_info.timestamp}")

        publisher = FramePublisher(config, on_frame_callback=on_frame)
        publisher.start()

        # Get current frame
        frame = publisher.get_current_frame()

        # Stop playback
        publisher.stop()
    """

    def __init__(
        self,
        config: Any,  # CameraStreamConfig
        loop: bool = True,
        preload_frames: int = 0,  # Number of frames to preload (0 = load on demand)
        on_frame_callback: Optional[Callable[[FrameInfo], None]] = None,
    ):
        self.config = config
        self.loop = loop
        self.preload_frames = preload_frames
        self.on_frame_callback = on_frame_callback

        # State
        self.running = False
        self.paused = False
        self.current_frame_id = 0
        self.current_frame: Optional[np.ndarray] = None
        self.current_frame_info: Optional[FrameInfo] = None

        # Threading
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        self._pause_event = threading.Event()
        self._pause_event.set()  # Start unpaused

        # Frame cache
        self._frame_cache: dict = {}
        self._frame_paths: List[Path] = []

        # Timing
        self._frame_interval = 1.0 / config.fps
        self._start_time: Optional[float] = None
        self._frame_count = 0

        # Initialize frame paths
        self._discover_frames()

        # Preload if requested
        if preload_frames > 0:
            self._preload_frames(preload_frames)

    def _discover_frames(self):
        """Discover all frame files."""
        self._frame_paths = sorted(self.config.rgb_dir.glob("rgb_*.png"))
        self._frame_count = len(self._frame_paths)

        if self._frame_count == 0:
            raise ValueError(f"No frames found in {self.config.rgb_dir}")

        logger.debug(f"Discovered {self._frame_count} frames for {self.config.camera_id}")

    def _preload_frames(self, count: int):
        """Preload frames into cache."""
        if not HAS_OPENCV:
            logger.warning("OpenCV not available, skipping preload")
            return

        count = min(count, self._frame_count)
        logger.info(f"Preloading {count} frames for {self.config.camera_id}")

        for i in range(count):
            path = self._frame_paths[i]
            frame = cv2.imread(str(path))
            if frame is not None:
                self._frame_cache[i] = frame

    def _load_frame(self, frame_id: int) -> Optional[np.ndarray]:
        """Load a frame by ID."""
        # Check cache first
        if frame_id in self._frame_cache:
            return self._frame_cache[frame_id]

        if not HAS_OPENCV:
            return None

        # Load from disk
        if 0 <= frame_id < self._frame_count:
            path = self._frame_paths[frame_id]
            frame = cv2.imread(str(path))
            return frame

        return None

    def _publish_loop(self):
        """Main publishing loop."""
        self._start_time = time.perf_counter()
        next_frame_time = self._start_time

        while self.running:
            # Wait if paused
            self._pause_event.wait()

            if not self.running:
                break

            # Calculate target frame based on elapsed time
            current_time = time.perf_counter()

            if current_time >= next_frame_time:
                # Load and publish frame
                frame = self._load_frame(self.current_frame_id)

                with self._lock:
                    self.current_frame = frame
                    self.current_frame_info = FrameInfo(
                        frame_id=self.current_frame_id,
                        timestamp=current_time - self._start_time,
                        camera_id=self.config.camera_id,
                        path=self._frame_paths[self.current_frame_id],
                        data=frame,
                    )

                # Callback
                if self.on_frame_callback and self.current_frame_info:
                    try:
                        self.on_frame_callback(self.current_frame_info)
                    except Exception as e:
                        logger.error(f"Frame callback error: {e}")

                # Advance to next frame
                self.current_frame_id += 1

                if self.current_frame_id >= self._frame_count:
                    if self.loop:
                        self.current_frame_id = 0
                        self._start_time = time.perf_counter()
                        next_frame_time = self._start_time
                    else:
                        break

                # Schedule next frame
                next_frame_time += self._frame_interval
            else:
                # Sleep until next frame (with some margin for wake-up latency)
                sleep_time = max(0, next_frame_time - current_time - 0.001)
                if sleep_time > 0:
                    time.sleep(sleep_time)

    def start(self):
        """Start frame publishing."""
        if self.running:
            logger.warning("Publisher already running")
            return

        self.running = True
        self.current_frame_id = 0
        self._thread = threading.Thread(target=self._publish_loop, daemon=True)
        self._thread.start()

        logger.info(f"Started frame publisher for {self.config.camera_id}")

    def stop(self):
        """Stop frame publishing."""
        self.running = False
        self._pause_event.set()  # Unblock if paused

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

        logger.info(f"Stopped frame publisher for {self.config.camera_id}")

    def pause(self):
        """Pause frame publishing."""
        self.paused = True
        self._pause_event.clear()
        logger.debug(f"Paused {self.config.camera_id}")

    def resume(self):
        """Resume frame publishing."""
        self.paused = False
        self._pause_event.set()
        logger.debug(f"Resumed {self.config.camera_id}")

    def seek(self, frame_id: int):
        """Seek to a specific frame."""
        with self._lock:
            self.current_frame_id = frame_id % self._frame_count
            # Reset timing
            self._start_time = time.perf_counter() - (self.current_frame_id * self._frame_interval)
        logger.debug(f"Seeked {self.config.camera_id} to frame {frame_id}")

    def get_current_frame(self) -> Optional[np.ndarray]:
        """Get the current frame data."""
        with self._lock:
            return self.current_frame.copy() if self.current_frame is not None else None

    def get_current_frame_info(self) -> Optional[FrameInfo]:
        """Get current frame information."""
        with self._lock:
            return self.current_frame_info

    @property
    def frame_count(self) -> int:
        """Total number of frames."""
        return self._frame_count

    @property
    def fps(self) -> int:
        """Frames per second."""
        return self.config.fps

    @property
    def duration(self) -> float:
        """Total duration in seconds."""
        return self._frame_count / self.config.fps


class MultiCameraPublisher:
    """
    Coordinates synchronized publishing across multiple cameras.

    Usage:
        from stream_simulator.rtsp_server import CameraStreamConfig

        configs = [
            CameraStreamConfig("cam_01", Path("/path/cam_01/rgb"), fps=30),
            CameraStreamConfig("cam_02", Path("/path/cam_02/rgb"), fps=30),
        ]

        multi_pub = MultiCameraPublisher(configs)
        multi_pub.start()

        # Get synchronized frames from all cameras
        frames = multi_pub.get_current_frames()

        multi_pub.stop()
    """

    def __init__(
        self,
        configs: List[Any],  # List[CameraStreamConfig]
        loop: bool = True,
        preload_frames: int = 0,
    ):
        self.configs = configs
        self.publishers: dict = {}

        # Create publishers for each camera
        for config in configs:
            self.publishers[config.camera_id] = FramePublisher(
                config,
                loop=loop,
                preload_frames=preload_frames,
            )

    def start(self):
        """Start all publishers synchronously."""
        # Start all publishers at approximately the same time
        start_time = time.perf_counter() + 0.1  # 100ms from now

        for pub in self.publishers.values():
            pub._start_time = start_time
            pub.running = True
            pub._thread = threading.Thread(target=pub._publish_loop, daemon=True)

        # Start all threads together
        for pub in self.publishers.values():
            pub._thread.start()

        logger.info(f"Started {len(self.publishers)} synchronized publishers")

    def stop(self):
        """Stop all publishers."""
        for pub in self.publishers.values():
            pub.stop()
        logger.info("Stopped all publishers")

    def pause(self):
        """Pause all publishers."""
        for pub in self.publishers.values():
            pub.pause()

    def resume(self):
        """Resume all publishers."""
        for pub in self.publishers.values():
            pub.resume()

    def seek(self, frame_id: int):
        """Seek all publishers to same frame."""
        for pub in self.publishers.values():
            pub.seek(frame_id)

    def get_current_frames(self) -> dict:
        """Get current frames from all cameras."""
        return {
            cam_id: pub.get_current_frame()
            for cam_id, pub in self.publishers.items()
        }

    def get_current_frame_infos(self) -> dict:
        """Get current frame info from all cameras."""
        return {
            cam_id: pub.get_current_frame_info()
            for cam_id, pub in self.publishers.items()
        }

    def get_publisher(self, camera_id: str) -> Optional[FramePublisher]:
        """Get publisher for a specific camera."""
        return self.publishers.get(camera_id)
