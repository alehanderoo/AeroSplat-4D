"""
Unit tests for the RTSP Stream Simulator.

Run with:
    pytest tests/test_stream_simulator.py -v
    pytest tests/test_stream_simulator.py -v -k "test_camera_config"
"""

import pytest
import time
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np

# Test fixtures path
TEST_RENDER_DIR = "/home/sandro/thesis/renders/5cams_08-01-26_drone_50m"


def has_test_data():
    """Check if test render data is available."""
    path = Path(TEST_RENDER_DIR)
    return path.exists() and (path / "cam_01" / "rgb").exists()


def has_opencv():
    """Check if OpenCV is available."""
    try:
        import cv2
        return True
    except ImportError:
        return False


def has_gstreamer():
    """Check if GStreamer RTSP Server is available."""
    try:
        import gi
        gi.require_version('GstRtspServer', '1.0')
        from gi.repository import GstRtspServer
        return True
    except (ImportError, ValueError):
        return False


# =============================================================================
# CameraStreamConfig Tests
# =============================================================================

class TestCameraStreamConfig:
    """Tests for CameraStreamConfig."""

    @pytest.mark.skipif(not has_test_data(), reason="Test data not available")
    def test_frame_count(self):
        """Test that frame count is correct."""
        from stream_simulator.rtsp_server import CameraStreamConfig

        config = CameraStreamConfig(
            camera_id="cam_01",
            rgb_dir=Path(TEST_RENDER_DIR) / "cam_01" / "rgb",
        )

        assert config.count_frames() == 120, "Expected 120 frames"

    @pytest.mark.skipif(not has_test_data(), reason="Test data not available")
    def test_frame_path(self):
        """Test frame path generation."""
        from stream_simulator.rtsp_server import CameraStreamConfig

        config = CameraStreamConfig(
            camera_id="cam_01",
            rgb_dir=Path(TEST_RENDER_DIR) / "cam_01" / "rgb",
        )

        path = config.get_frame_path(42)
        assert path.name == "rgb_0042.png"

    def test_frame_glob_pattern(self):
        """Test glob pattern generation."""
        from stream_simulator.rtsp_server import CameraStreamConfig

        config = CameraStreamConfig(
            camera_id="cam_01",
            rgb_dir=Path("/test/path/rgb"),
        )

        assert "rgb_*.png" in config.frame_glob


# =============================================================================
# ServerConfig Tests
# =============================================================================

class TestServerConfig:
    """Tests for ServerConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        from stream_simulator.rtsp_server import ServerConfig

        config = ServerConfig(render_dir="/test/path")

        assert config.host == "0.0.0.0"
        assert config.port == 8554
        assert config.fps == 30
        assert config.loop is True
        assert len(config.camera_ids) == 5

    def test_path_conversion(self):
        """Test that render_dir is converted to Path."""
        from stream_simulator.rtsp_server import ServerConfig

        config = ServerConfig(render_dir="/test/path")

        assert isinstance(config.render_dir, Path)


# =============================================================================
# FramePublisher Tests
# =============================================================================

class TestFramePublisher:
    """Tests for FramePublisher."""

    @pytest.fixture
    def temp_render_dir(self):
        """Create temporary directory with fake frames."""
        temp_dir = tempfile.mkdtemp()
        rgb_dir = Path(temp_dir) / "rgb"
        rgb_dir.mkdir()

        # Create fake PNG files (minimal valid PNG)
        # PNG signature + minimal IHDR
        png_header = bytes([
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,  # PNG signature
            0x00, 0x00, 0x00, 0x0D,  # IHDR length
            0x49, 0x48, 0x44, 0x52,  # IHDR type
            0x00, 0x00, 0x00, 0x01,  # width
            0x00, 0x00, 0x00, 0x01,  # height
            0x08, 0x02,              # bit depth, color type
            0x00, 0x00, 0x00,        # compression, filter, interlace
            0x90, 0x77, 0x53, 0xDE,  # CRC
        ])

        for i in range(10):
            (rgb_dir / f"rgb_{i:04d}.png").write_bytes(png_header)

        yield temp_dir

        # Cleanup
        shutil.rmtree(temp_dir)

    def test_discover_frames(self, temp_render_dir):
        """Test frame discovery."""
        from stream_simulator.rtsp_server import CameraStreamConfig
        from stream_simulator.frame_publisher import FramePublisher

        config = CameraStreamConfig(
            camera_id="test",
            rgb_dir=Path(temp_render_dir) / "rgb",
            fps=30,
        )

        publisher = FramePublisher(config, loop=False)
        assert publisher.frame_count == 10

    def test_frame_timing(self, temp_render_dir):
        """Test frame timing accuracy."""
        from stream_simulator.rtsp_server import CameraStreamConfig
        from stream_simulator.frame_publisher import FramePublisher

        config = CameraStreamConfig(
            camera_id="test",
            rgb_dir=Path(temp_render_dir) / "rgb",
            fps=10,  # 100ms per frame
        )

        frame_times = []

        def on_frame(info):
            frame_times.append(time.perf_counter())

        publisher = FramePublisher(
            config,
            loop=False,
            on_frame_callback=on_frame
        )

        publisher.start()
        time.sleep(0.5)  # Let it run for 500ms
        publisher.stop()

        # Check timing (should have ~5 frames in 500ms at 10fps)
        assert len(frame_times) >= 4
        assert len(frame_times) <= 6

        # Check intervals
        if len(frame_times) > 1:
            intervals = [
                frame_times[i+1] - frame_times[i]
                for i in range(len(frame_times) - 1)
            ]
            avg_interval = sum(intervals) / len(intervals)
            # Allow 20% tolerance
            assert 0.08 < avg_interval < 0.12

    def test_pause_resume(self, temp_render_dir):
        """Test pause and resume functionality."""
        from stream_simulator.rtsp_server import CameraStreamConfig
        from stream_simulator.frame_publisher import FramePublisher

        config = CameraStreamConfig(
            camera_id="test",
            rgb_dir=Path(temp_render_dir) / "rgb",
            fps=30,
        )

        publisher = FramePublisher(config, loop=True)
        publisher.start()

        time.sleep(0.1)
        frame_before_pause = publisher.current_frame_id

        publisher.pause()
        time.sleep(0.1)
        frame_after_pause = publisher.current_frame_id

        # Should not have advanced while paused
        assert frame_after_pause == frame_before_pause or \
               frame_after_pause == frame_before_pause + 1

        publisher.resume()
        time.sleep(0.1)
        frame_after_resume = publisher.current_frame_id

        # Should have advanced after resume
        # Note: frame may have looped around if we're near the end
        assert frame_after_resume != frame_after_pause or frame_after_pause >= 8

        publisher.stop()


# =============================================================================
# SyncController Tests
# =============================================================================

class TestSyncController:
    """Tests for SyncController."""

    def test_internal_sync_timing(self):
        """Test internal clock synchronization timing."""
        from stream_simulator.sync_controller import SyncController, SyncConfig, SyncMode

        config = SyncConfig(mode=SyncMode.INTERNAL_CLOCK, fps=10)
        controller = SyncController(config)

        events = []

        def on_sync(event):
            events.append((event.frame_id, time.perf_counter()))

        controller.register_camera("test", callback=on_sync)
        controller.start()

        time.sleep(0.5)  # 500ms
        controller.stop()

        # Should have ~5 events at 10fps in 500ms
        assert len(events) >= 4
        assert len(events) <= 6

        # Check frame IDs are sequential
        frame_ids = [e[0] for e in events]
        for i in range(len(frame_ids) - 1):
            assert frame_ids[i+1] == frame_ids[i] + 1

    def test_network_trigger(self):
        """Test network trigger synchronization."""
        from stream_simulator.sync_controller import SyncController, SyncConfig, SyncMode

        config = SyncConfig(mode=SyncMode.NETWORK_TRIGGER, network_port=19999)
        controller = SyncController(config)

        events = []

        def on_sync(event):
            events.append(event)

        controller.register_camera("test", callback=on_sync)
        controller.start()

        time.sleep(0.1)  # Let server start

        # Send triggers
        controller.send_trigger(frame_id=1, target="localhost")
        controller.send_trigger(frame_id=2, target="localhost")
        controller.send_trigger(frame_id=3, target="localhost")

        time.sleep(0.2)  # Let triggers be received
        controller.stop()

        # Should have received the triggers
        assert len(events) >= 2  # May not receive all due to timing

    def test_camera_registration(self):
        """Test camera registration and unregistration."""
        from stream_simulator.sync_controller import SyncController

        controller = SyncController()

        controller.register_camera("cam_01")
        controller.register_camera("cam_02")

        assert "cam_01" in controller._cameras
        assert "cam_02" in controller._cameras

        controller.unregister_camera("cam_01")
        assert "cam_01" not in controller._cameras
        assert "cam_02" in controller._cameras


# =============================================================================
# GStreamer Server Tests
# =============================================================================

@pytest.mark.skipif(not has_test_data(), reason="Test data not available")
@pytest.mark.skipif(not has_gstreamer(), reason="GStreamer not available")
class TestGStreamerServer:
    """Tests for GStreamerRTSPServer."""

    def test_server_creation(self):
        """Test that server can be created with valid config."""
        from stream_simulator.rtsp_server import GStreamerRTSPServer, ServerConfig

        config = ServerConfig(render_dir=TEST_RENDER_DIR)
        server = GStreamerRTSPServer(config)

        assert len(server.camera_configs) == 5
        for cam_id in ["cam_01", "cam_02", "cam_03", "cam_04", "cam_05"]:
            assert cam_id in server.camera_configs

    def test_stream_urls(self):
        """Test that stream URLs are generated correctly."""
        from stream_simulator.rtsp_server import GStreamerRTSPServer, ServerConfig

        config = ServerConfig(render_dir=TEST_RENDER_DIR, port=9999)
        server = GStreamerRTSPServer(config)

        urls = server.get_stream_urls()
        assert len(urls) == 5
        assert urls["cam_01"] == "rtsp://0.0.0.0:9999/cam_01"


# =============================================================================
# Integration Tests
# =============================================================================

@pytest.mark.skipif(not has_test_data(), reason="Test data not available")
class TestIntegration:
    """Integration tests with actual render data."""

    def test_multi_camera_publisher(self):
        """Test synchronized multi-camera publishing."""
        from stream_simulator.rtsp_server import CameraStreamConfig
        from stream_simulator.frame_publisher import MultiCameraPublisher

        configs = []
        for i in range(1, 6):
            cam_id = f"cam_{i:02d}"
            configs.append(CameraStreamConfig(
                camera_id=cam_id,
                rgb_dir=Path(TEST_RENDER_DIR) / cam_id / "rgb",
                fps=30,
            ))

        multi_pub = MultiCameraPublisher(configs, loop=True)
        multi_pub.start()

        time.sleep(0.2)

        # Get frames from all cameras
        frames = multi_pub.get_current_frame_infos()
        assert len(frames) == 5

        # Check frame IDs are roughly synchronized
        frame_ids = [
            info.frame_id for info in frames.values()
            if info is not None
        ]

        if len(frame_ids) > 1:
            max_drift = max(frame_ids) - min(frame_ids)
            assert max_drift <= 2, f"Frame drift too high: {max_drift}"

        multi_pub.stop()


# =============================================================================
# Error Handling Tests
# =============================================================================

@pytest.mark.skipif(not has_gstreamer(), reason="GStreamer not available")
class TestErrorHandling:
    """Tests for error handling."""

    def test_invalid_render_dir(self):
        """Test error on invalid render directory."""
        from stream_simulator.rtsp_server import ServerConfig, GStreamerRTSPServer

        config = ServerConfig(render_dir="/nonexistent/path")

        with pytest.raises(ValueError, match="not found"):
            GStreamerRTSPServer(config)

    def test_empty_rgb_dir(self):
        """Test error on empty RGB directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create structure without frames
            for i in range(1, 6):
                (Path(temp_dir) / f"cam_{i:02d}" / "rgb").mkdir(parents=True)

            from stream_simulator.rtsp_server import ServerConfig, GStreamerRTSPServer

            config = ServerConfig(render_dir=temp_dir)

            with pytest.raises(ValueError, match="No frames found"):
                GStreamerRTSPServer(config)


# =============================================================================
# Performance Tests
# =============================================================================

@pytest.mark.skipif(not has_test_data(), reason="Test data not available")
@pytest.mark.skipif(not has_opencv(), reason="OpenCV not available")
class TestPerformance:
    """Performance tests."""

    def test_frame_load_performance(self):
        """Test frame loading performance."""
        import cv2
        from stream_simulator.rtsp_server import CameraStreamConfig

        config = CameraStreamConfig(
            camera_id="cam_01",
            rgb_dir=Path(TEST_RENDER_DIR) / "cam_01" / "rgb",
        )

        # Load 10 frames and measure time
        start = time.perf_counter()
        for i in range(10):
            path = config.get_frame_path(i)
            frame = cv2.imread(str(path))
            assert frame is not None
        elapsed = time.perf_counter() - start

        avg_load_time_ms = (elapsed / 10) * 1000
        print(f"\nAverage frame load time: {avg_load_time_ms:.2f} ms")

        # Should be faster than real-time at 30fps (33ms)
        assert avg_load_time_ms < 33, f"Frame loading too slow: {avg_load_time_ms:.2f}ms"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
