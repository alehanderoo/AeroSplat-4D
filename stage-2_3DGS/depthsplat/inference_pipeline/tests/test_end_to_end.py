"""
End-to-End Integration Tests for DepthSplat Inference Pipeline.

These tests verify the complete pipeline integration including:
- Configuration loading
- Component initialization
- Metrics collection

Run with:
    pytest tests/test_end_to_end.py -v
    pytest tests/test_end_to_end.py -v -k "test_config"
"""

import pytest
import time
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch


# Test paths
INFERENCE_DIR = Path(__file__).parent.parent
CONFIG_PATH = INFERENCE_DIR / "config" / "pipeline_config.yaml"


def has_config_file():
    """Check if configuration file exists."""
    return CONFIG_PATH.exists()


# =============================================================================
# Configuration Tests
# =============================================================================

class TestConfiguration:
    """Tests for configuration loading."""

    def test_load_config_dataclasses(self):
        """Test that config loads into proper dataclasses."""
        from config import PipelineSettings, load_config
        
        if not has_config_file():
            pytest.skip("Config file not found")
        
        config = load_config(CONFIG_PATH)
        
        assert isinstance(config, PipelineSettings)
        assert config.pipeline.name == "depthsplat-inference"
        assert config.pipeline.version == "1.0.0"

    def test_config_sources(self):
        """Test source configuration loading."""
        from config import load_config
        
        if not has_config_file():
            pytest.skip("Config file not found")
        
        config = load_config(CONFIG_PATH)
        
        assert config.sources.mode in ["rtsp", "file", "simulator"]
        assert len(config.sources.rtsp.urls) == 5
        assert config.sources.simulator.fps == 30

    def test_config_model(self):
        """Test model configuration loading."""
        from config import load_config
        
        if not has_config_file():
            pytest.skip("Config file not found")
        
        config = load_config(CONFIG_PATH)
        
        assert config.model.input.num_cameras == 5
        assert config.model.input.channels == 3
        assert config.model.preprocessing.normalize.mean == [0.485, 0.456, 0.406]

    def test_config_deepstream(self):
        """Test DeepStream configuration loading."""
        from config import load_config
        
        if not has_config_file():
            pytest.skip("Config file not found")
        
        config = load_config(CONFIG_PATH)
        
        assert config.deepstream.muxer.batch_size == 5
        assert config.deepstream.muxer.sync_inputs is True

    def test_config_defaults(self):
        """Test that defaults are applied for missing values."""
        from config import PipelineSettings
        
        config = PipelineSettings()
        
        assert config.pipeline.name == "depthsplat-inference"
        assert config.sources.mode == "rtsp"
        assert config.monitoring.metrics.enabled is True

    def test_env_var_expansion(self):
        """Test environment variable expansion in config."""
        import os
        from config import _expand_env_vars
        
        os.environ["TEST_VAR"] = "test_value"
        
        # Test simple expansion
        result = _expand_env_vars("${TEST_VAR}")
        assert result == "test_value"
        
        # Test with default
        result = _expand_env_vars("${NONEXISTENT:default}")
        assert result == "default"
        
        # Cleanup
        del os.environ["TEST_VAR"]

    def test_config_file_not_found(self):
        """Test error handling for missing config file."""
        from config import load_config
        
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.yaml")


# =============================================================================
# Metrics Tests
# =============================================================================

class TestMetrics:
    """Tests for metrics collection."""

    def test_metrics_server(self):
        """Test metrics server."""
        from utils.metrics import MetricsServer
        
        server = MetricsServer(port=9999)
        server.start()
        
        assert server.running
        
        # Record some metrics
        server.record_frame_processed("cam_01")
        server.record_inference_latency(25.5)
        server.record_gaussians_count(1000)
        server.set_stream_status("cam_01", True)
        
        server.stop()
        assert not server.running

    def test_metrics_disabled(self):
        """Test that disabled metrics don't cause errors."""
        from utils.metrics import MetricsServer
        
        server = MetricsServer(enabled=False)
        server.start()
        
        # Should not raise
        server.record_frame_processed()
        server.record_inference_latency(25.5)
        
        server.stop()

    def test_global_metrics_server(self):
        """Test global metrics server singleton."""
        from utils.metrics import get_metrics_server
        
        server1 = get_metrics_server(port=9998)
        server2 = get_metrics_server(port=9997)  # Different port, but same instance
        
        assert server1 is server2


# =============================================================================
# Logging Tests
# =============================================================================

class TestLogging:
    """Tests for logging utilities."""

    def test_setup_logging(self):
        """Test logging setup."""
        from utils.logging_utils import setup_logging, get_logger
        
        setup_logging(level="DEBUG", use_colors=False)
        logger = get_logger(__name__)
        
        # Should not raise
        logger.debug("Test debug")
        logger.info("Test info")
        logger.warning("Test warning")

    def test_logging_to_file(self):
        """Test file logging."""
        import tempfile
        from utils.logging_utils import setup_logging, get_logger
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            log_file = f.name
        
        try:
            setup_logging(level="INFO", file=log_file, use_colors=False)
            logger = get_logger("test_file_logging")
            logger.info("Test message to file")
            
            # Check file was written
            with open(log_file, 'r') as f:
                content = f.read()
            assert "Test message to file" in content
        finally:
            Path(log_file).unlink(missing_ok=True)


# =============================================================================
# Camera Utils Tests
# =============================================================================

class TestCameraUtils:
    """Tests for camera utilities."""

    def test_camera_intrinsics(self):
        """Test camera intrinsics."""
        from utils.camera_utils import CameraIntrinsics
        import numpy as np
        
        intrinsics = CameraIntrinsics(fx=500, fy=500, cx=320, cy=240)
        
        # Test matrix property
        K = intrinsics.matrix
        assert K.shape == (3, 3)
        assert K[0, 0] == 500  # fx
        assert K[1, 1] == 500  # fy
        
        # Test projection
        point_3d = np.array([1.0, 0.5, 2.0])
        pixel = intrinsics.project(point_3d)
        
        assert len(pixel) == 2
        
        # Test unprojection
        point_back = intrinsics.unproject(pixel, depth=2.0)
        np.testing.assert_allclose(point_back, point_3d, rtol=1e-5)

    def test_camera_extrinsics(self):
        """Test camera extrinsics."""
        from utils.camera_utils import CameraExtrinsics
        import numpy as np
        
        # Test look-at
        extrinsics = CameraExtrinsics.look_at(
            camera_pos=np.array([0, 0, 5]),
            target=np.array([0, 0, 0]),
        )
        
        # Camera should be at (0, 0, 5)
        np.testing.assert_allclose(
            extrinsics.camera_position,
            [0, 0, 5],
            atol=1e-5
        )

    def test_camera_metadata(self):
        """Test camera metadata loading."""
        from utils.camera_utils import CameraMetadata, CameraIntrinsics, CameraExtrinsics
        
        metadata = CameraMetadata(
            camera_id="cam_01",
            intrinsics=CameraIntrinsics(fx=500, fy=500, cx=320, cy=240),
            extrinsics=CameraExtrinsics(),
            near=0.1,
            far=100.0,
        )
        
        assert metadata.camera_id == "cam_01"
        assert metadata.near == 0.1


# =============================================================================
# Gaussian Utils Tests
# =============================================================================

class TestGaussianUtils:
    """Tests for Gaussian utilities."""

    def test_gaussian_buffer(self):
        """Test Gaussian buffer."""
        from utils.gaussian_utils import GaussianBuffer
        import numpy as np
        
        buffer = GaussianBuffer(max_frames=5)
        
        # Add frames
        for i in range(10):
            buffer.add(
                positions=np.random.randn(100, 3),
                covariances=np.random.randn(100, 6),
                colors=np.random.rand(100, 3),
                opacities=np.random.rand(100, 1),
                frame_id=i,
            )
        
        # Should only keep last 5
        assert buffer.size == 5
        assert buffer.frame_ids == [5, 6, 7, 8, 9]
        
        # Get latest
        latest = buffer.get_latest()
        assert latest.frame_id == 9

    def test_gaussian_export_npz(self):
        """Test Gaussian export to NPZ."""
        from utils.gaussian_utils import export_gaussians
        import numpy as np
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            positions = np.random.randn(50, 3).astype(np.float32)
            covariances = np.random.randn(50, 6).astype(np.float32)
            colors = np.random.rand(50, 3).astype(np.float32)
            opacities = np.random.rand(50, 1).astype(np.float32)
            
            path = export_gaussians(
                f"{tmpdir}/test.npz",
                positions, covariances, colors, opacities,
                format="npz"
            )
            
            assert Path(path).exists()
            
            # Verify contents
            data = np.load(path)
            np.testing.assert_array_equal(data["positions"], positions)

    def test_gaussian_export_ply(self):
        """Test Gaussian export to PLY."""
        from utils.gaussian_utils import export_gaussians
        import numpy as np
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            positions = np.random.randn(50, 3).astype(np.float32)
            covariances = np.abs(np.random.randn(50, 6).astype(np.float32)) * 0.1
            colors = np.random.rand(50, 3).astype(np.float32)
            opacities = np.random.rand(50, 1).astype(np.float32)
            
            path = export_gaussians(
                f"{tmpdir}/test.ply",
                positions, covariances, colors, opacities,
                format="ply"
            )
            
            assert Path(path).exists()
            assert Path(path).stat().st_size > 0


# =============================================================================
# Pipeline Integration Tests
# =============================================================================

class TestPipelineIntegration:
    """Integration tests for the pipeline."""

    def test_pipeline_creation(self):
        """Test pipeline can be created."""
        from pipeline import DepthSplatPipeline, PipelineConfig
        
        config = PipelineConfig(
            stream_urls=["rtsp://localhost:8554/cam_01"],
            checkpoint_path="/path/to/model.ckpt",
        )
        
        pipeline = DepthSplatPipeline(config)
        assert pipeline.running is False


# =============================================================================
# Main Entry Point Tests
# =============================================================================

class TestMainEntryPoint:
    """Tests for main.py entry point."""

    def test_import_main(self):
        """Test that main.py can be imported."""
        import sys
        sys.path.insert(0, str(INFERENCE_DIR))
        
        # Should not raise
        import main
        
        assert hasattr(main, 'main')


# =============================================================================
# Performance Tests
# =============================================================================

class TestPerformance:
    """Performance tests."""

    def test_config_load_time(self):
        """Test configuration loading is fast."""
        from config import load_config
        
        if not has_config_file():
            pytest.skip("Config file not found")
        
        start = time.perf_counter()
        for _ in range(10):
            config = load_config(CONFIG_PATH)
        elapsed = time.perf_counter() - start
        
        avg_ms = (elapsed / 10) * 1000
        assert avg_ms < 100, f"Config loading too slow: {avg_ms:.2f}ms"

    def test_gaussian_buffer_performance(self):
        """Test Gaussian buffer performance."""
        from utils.gaussian_utils import GaussianBuffer
        import numpy as np
        
        buffer = GaussianBuffer(max_frames=120)
        
        # Simulate 1000 Gaussians per frame
        positions = np.random.randn(1000, 3).astype(np.float32)
        covariances = np.random.randn(1000, 6).astype(np.float32)
        colors = np.random.rand(1000, 3).astype(np.float32)
        opacities = np.random.rand(1000, 1).astype(np.float32)
        
        start = time.perf_counter()
        for i in range(120):
            buffer.add(positions, covariances, colors, opacities, frame_id=i)
        elapsed = time.perf_counter() - start
        
        avg_ms = (elapsed / 120) * 1000
        assert avg_ms < 5, f"Buffer add too slow: {avg_ms:.2f}ms per frame"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
