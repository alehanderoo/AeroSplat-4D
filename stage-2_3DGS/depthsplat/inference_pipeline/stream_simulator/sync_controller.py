"""
Synchronization Controller for Multi-Camera Frame Coordination.

This module provides mechanisms for synchronized frame capture across
multiple camera streams, supporting both internal timing and external triggers.
"""

import time
import threading
import logging
from typing import Optional, Callable, Dict, List, Any
from dataclasses import dataclass, field
from enum import Enum
from queue import Queue, Empty
import socket

logger = logging.getLogger(__name__)


class SyncMode(Enum):
    """Synchronization modes."""
    FREERUN = "freerun"           # Each camera runs independently
    INTERNAL_CLOCK = "internal"   # Synchronized to internal clock
    EXTERNAL_TRIGGER = "external" # Synchronized to external trigger
    NETWORK_TRIGGER = "network"   # Synchronized via network messages


@dataclass
class SyncEvent:
    """A synchronization event."""
    frame_id: int
    timestamp: float
    source: str  # "internal", "external", "network"


@dataclass
class SyncConfig:
    """Configuration for synchronization."""
    mode: SyncMode = SyncMode.INTERNAL_CLOCK
    fps: int = 30

    # Network trigger settings
    network_port: int = 9999
    network_host: str = "0.0.0.0"

    # Timing tolerances
    max_inter_camera_drift_ms: float = 5.0  # Max allowed drift between cameras
    sync_timeout_ms: float = 100.0  # Timeout waiting for all cameras


class SyncController:
    """
    Controls synchronization across multiple cameras.

    Usage:
        # Create controller
        controller = SyncController(SyncConfig(mode=SyncMode.INTERNAL_CLOCK, fps=30))

        # Register cameras
        controller.register_camera("cam_01", callback=on_sync_cam01)
        controller.register_camera("cam_02", callback=on_sync_cam02)

        # Start synchronization
        controller.start()

        # Wait for sync events
        event = controller.wait_for_sync()
        print(f"Sync event: frame {event.frame_id}")

        # Stop
        controller.stop()
    """

    def __init__(self, config: Optional[SyncConfig] = None):
        self.config = config or SyncConfig()

        # Registered cameras and callbacks
        self._cameras: Dict[str, Callable] = {}
        self._camera_ready: Dict[str, threading.Event] = {}
        self._camera_frames: Dict[str, int] = {}

        # State
        self.running = False
        self.current_frame_id = 0

        # Threading
        self._sync_thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()

        # Sync event queue
        self._sync_queue: Queue[SyncEvent] = Queue(maxsize=100)

        # Frame interval
        self._frame_interval = 1.0 / self.config.fps

        # Network socket for network trigger mode
        self._socket: Optional[socket.socket] = None

    def register_camera(
        self,
        camera_id: str,
        callback: Optional[Callable[[SyncEvent], None]] = None
    ):
        """
        Register a camera with the sync controller.

        Args:
            camera_id: Unique camera identifier
            callback: Optional callback called on sync events
        """
        with self._lock:
            self._cameras[camera_id] = callback
            self._camera_ready[camera_id] = threading.Event()
            self._camera_frames[camera_id] = 0

        logger.debug(f"Registered camera: {camera_id}")

    def unregister_camera(self, camera_id: str):
        """Unregister a camera."""
        with self._lock:
            self._cameras.pop(camera_id, None)
            self._camera_ready.pop(camera_id, None)
            self._camera_frames.pop(camera_id, None)

        logger.debug(f"Unregistered camera: {camera_id}")

    def _internal_sync_loop(self):
        """Internal clock synchronization loop."""
        start_time = time.perf_counter()
        next_sync_time = start_time

        while self.running:
            current_time = time.perf_counter()

            if current_time >= next_sync_time:
                # Generate sync event
                event = SyncEvent(
                    frame_id=self.current_frame_id,
                    timestamp=current_time - start_time,
                    source="internal"
                )

                # Notify all cameras
                self._dispatch_sync_event(event)

                # Advance frame
                self.current_frame_id += 1
                next_sync_time += self._frame_interval
            else:
                # Sleep until next sync
                sleep_time = max(0, next_sync_time - current_time - 0.0005)
                if sleep_time > 0:
                    time.sleep(sleep_time)

    def _network_sync_loop(self):
        """Network trigger synchronization loop."""
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind((self.config.network_host, self.config.network_port))
        self._socket.settimeout(1.0)

        logger.info(f"Listening for sync triggers on port {self.config.network_port}")
        start_time = time.perf_counter()

        while self.running:
            try:
                data, addr = self._socket.recvfrom(1024)
                current_time = time.perf_counter()

                # Parse trigger message
                # Expected format: "SYNC:<frame_id>" or just "SYNC"
                message = data.decode().strip()

                if message.startswith("SYNC"):
                    parts = message.split(":")
                    if len(parts) > 1:
                        try:
                            frame_id = int(parts[1])
                        except ValueError:
                            frame_id = self.current_frame_id
                    else:
                        frame_id = self.current_frame_id

                    event = SyncEvent(
                        frame_id=frame_id,
                        timestamp=current_time - start_time,
                        source="network"
                    )

                    self._dispatch_sync_event(event)
                    self.current_frame_id = frame_id + 1

            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    logger.error(f"Network sync error: {e}")

        if self._socket:
            self._socket.close()
            self._socket = None

    def _dispatch_sync_event(self, event: SyncEvent):
        """Dispatch sync event to all cameras and queue."""
        # Add to queue
        try:
            self._sync_queue.put_nowait(event)
        except:
            # Queue full, drop oldest
            try:
                self._sync_queue.get_nowait()
                self._sync_queue.put_nowait(event)
            except:
                pass

        # Call camera callbacks
        for camera_id, callback in self._cameras.items():
            if callback:
                try:
                    callback(event)
                except Exception as e:
                    logger.error(f"Sync callback error for {camera_id}: {e}")

    def start(self):
        """Start synchronization."""
        if self.running:
            logger.warning("Sync controller already running")
            return

        self.running = True
        self.current_frame_id = 0

        if self.config.mode == SyncMode.INTERNAL_CLOCK:
            self._sync_thread = threading.Thread(
                target=self._internal_sync_loop,
                daemon=True
            )
        elif self.config.mode == SyncMode.NETWORK_TRIGGER:
            self._sync_thread = threading.Thread(
                target=self._network_sync_loop,
                daemon=True
            )
        elif self.config.mode == SyncMode.FREERUN:
            # No sync thread needed
            logger.info("Sync mode: freerun (no synchronization)")
            return
        else:
            logger.warning(f"Sync mode {self.config.mode} not fully implemented")
            return

        self._sync_thread.start()
        logger.info(f"Sync controller started in {self.config.mode.value} mode")

    def stop(self):
        """Stop synchronization."""
        self.running = False

        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=2.0)

        if self._socket:
            self._socket.close()
            self._socket = None

        logger.info("Sync controller stopped")

    def wait_for_sync(self, timeout: Optional[float] = None) -> Optional[SyncEvent]:
        """
        Wait for the next sync event.

        Args:
            timeout: Maximum time to wait in seconds (None = blocking)

        Returns:
            SyncEvent or None if timeout
        """
        try:
            return self._sync_queue.get(timeout=timeout)
        except Empty:
            return None

    def get_latest_sync(self) -> Optional[SyncEvent]:
        """Get the latest sync event without blocking."""
        latest = None
        while True:
            try:
                latest = self._sync_queue.get_nowait()
            except Empty:
                break
        return latest

    def send_trigger(self, frame_id: Optional[int] = None, target: str = "localhost"):
        """
        Send a network sync trigger (for testing or external triggering).

        Args:
            frame_id: Frame ID to include in trigger
            target: Target host to send trigger to
        """
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            message = f"SYNC:{frame_id}" if frame_id is not None else "SYNC"
            sock.sendto(message.encode(), (target, self.config.network_port))
        finally:
            sock.close()

    def report_camera_ready(self, camera_id: str, frame_id: int):
        """
        Report that a camera is ready with a frame.

        Used for barrier synchronization to ensure all cameras
        are ready before proceeding.
        """
        with self._lock:
            if camera_id in self._camera_ready:
                self._camera_frames[camera_id] = frame_id
                self._camera_ready[camera_id].set()

    def wait_all_cameras_ready(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for all cameras to report ready.

        Returns:
            True if all cameras ready, False if timeout
        """
        timeout_per_camera = timeout / len(self._cameras) if timeout else None

        for camera_id, event in self._camera_ready.items():
            if not event.wait(timeout=timeout_per_camera):
                logger.warning(f"Timeout waiting for camera {camera_id}")
                return False

        # Reset for next sync
        for event in self._camera_ready.values():
            event.clear()

        return True

    def check_sync_quality(self) -> Dict[str, Any]:
        """
        Check the quality of synchronization across cameras.

        Returns:
            Dictionary with sync quality metrics
        """
        with self._lock:
            frame_ids = list(self._camera_frames.values())

        if not frame_ids:
            return {"status": "no_data"}

        max_drift = max(frame_ids) - min(frame_ids)
        in_sync = max_drift <= 1  # Allow 1 frame drift

        return {
            "status": "in_sync" if in_sync else "drift_detected",
            "max_frame_drift": max_drift,
            "camera_frames": dict(self._camera_frames),
            "tolerance_frames": 1,
        }


class SyncClient:
    """
    Client for receiving sync triggers from a remote controller.

    Usage:
        client = SyncClient(port=9999)
        client.start()

        while True:
            event = client.wait_for_trigger(timeout=1.0)
            if event:
                # Capture frame
                pass
    """

    def __init__(self, port: int = 9999, host: str = "0.0.0.0"):
        self.port = port
        self.host = host
        self._socket: Optional[socket.socket] = None
        self._trigger_queue: Queue[SyncEvent] = Queue(maxsize=100)
        self._thread: Optional[threading.Thread] = None
        self.running = False

    def _receive_loop(self):
        """Receive trigger messages."""
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind((self.host, self.port))
        self._socket.settimeout(1.0)

        start_time = time.perf_counter()

        while self.running:
            try:
                data, addr = self._socket.recvfrom(1024)
                current_time = time.perf_counter()

                message = data.decode().strip()
                if message.startswith("SYNC"):
                    parts = message.split(":")
                    frame_id = int(parts[1]) if len(parts) > 1 else 0

                    event = SyncEvent(
                        frame_id=frame_id,
                        timestamp=current_time - start_time,
                        source=f"network:{addr[0]}"
                    )

                    try:
                        self._trigger_queue.put_nowait(event)
                    except:
                        pass

            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    logger.error(f"Receive error: {e}")

        if self._socket:
            self._socket.close()

    def start(self):
        """Start receiving triggers."""
        self.running = True
        self._thread = threading.Thread(target=self._receive_loop, daemon=True)
        self._thread.start()
        logger.info(f"Sync client listening on port {self.port}")

    def stop(self):
        """Stop receiving triggers."""
        self.running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def wait_for_trigger(self, timeout: Optional[float] = None) -> Optional[SyncEvent]:
        """Wait for the next trigger."""
        try:
            return self._trigger_queue.get(timeout=timeout)
        except Empty:
            return None
