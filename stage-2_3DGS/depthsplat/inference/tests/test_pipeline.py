"""
Unit tests for the DepthSplat Inference Pipeline.

Run with:
    pytest tests/test_pipeline.py -v
    pytest tests/test_pipeline.py -v -k "test_config"
"""

import pytest
import time
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np

# Test fixtures
TEST_RENDER_DIR = "/home/sandro/thesis/renders/5cams_08-01-26_drone_50m"


def has_test_data():
    """Check if test render data is available."""
    path = Path(TEST_RENDER_DIR)
    return path.exists() and (path / "cam_01" / "rgb").exists()


# =============================================================================
# PipelineConfig Tests
# =============================================================================

class TestPipelineConfig:
    """Tests for PipelineConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        from pipeline.deepstream_pipeline import PipelineConfig

        config = PipelineConfig(
            stream_urls=[
                "rtsp://localhost:8554/cam_01",
                "rtsp://localhost:8554/cam_02",
            ],
            checkpoint_path="/path/to/model.ckpt",
        )

        assert config.num_cameras == 2
        assert config.model_input_width == 256
        assert config.model_input_height == 256
        assert config.gpu_id == 0

    def test_num_cameras_property(self):
        """Test num_cameras property."""
        from pipeline.deepstream_pipeline import PipelineConfig

        config = PipelineConfig(
            stream_urls=[f"rtsp://localhost:8554/cam_{i:02d}" for i in range(1, 6)],
            checkpoint_path="/path/to/model.ckpt",
        )

        assert config.num_cameras == 5


# =============================================================================
# GaussianOutput Tests
# =============================================================================

class TestGaussianOutput:
    """Tests for GaussianOutput."""

    def test_output_creation(self):
        """Test GaussianOutput creation."""
        from pipeline.deepstream_pipeline import GaussianOutput

        output = GaussianOutput(
            frame_id=42,
            timestamp=123.456,
            positions=np.zeros((100, 3)),
            covariances=np.zeros((100, 6)),
            colors=np.zeros((100, 3)),
            opacities=np.zeros((100, 1)),
            inference_time_ms=10.5,
        )

        assert output.frame_id == 42
        assert output.positions.shape == (100, 3)
        assert output.inference_time_ms == 10.5


# =============================================================================
# DepthSplatPipeline Tests
# =============================================================================

class TestDepthSplatPipeline:
    """Tests for DepthSplatPipeline."""

    def test_pipeline_creation(self):
        """Test that DepthSplatPipeline can be created."""
        from pipeline.deepstream_pipeline import (
            PipelineConfig,
            DepthSplatPipeline,
        )

        config = PipelineConfig(
            stream_urls=["rtsp://localhost:8554/cam_01"],
            checkpoint_path="/path/to/model.ckpt",
        )

        pipeline = DepthSplatPipeline(config)
        assert pipeline.running is False

    def test_pipeline_stats_initial(self):
        """Test initial stats are correct."""
        from pipeline.deepstream_pipeline import (
            PipelineConfig,
            DepthSplatPipeline,
        )

        config = PipelineConfig(
            stream_urls=["rtsp://localhost:8554/cam_01"],
            checkpoint_path="/path/to/model.ckpt",
        )

        pipeline = DepthSplatPipeline(config)
        stats = pipeline.get_stats()

        assert stats["frame_count"] == 0


# =============================================================================
# Integration Tests
# =============================================================================

@pytest.mark.skipif(not has_test_data(), reason="Test data not available")
class TestIntegration:
    """Integration tests with stream_simulator."""

    def test_pipeline_config_with_test_data(self):
        """Test pipeline config with test data path."""
        from pipeline.deepstream_pipeline import PipelineConfig

        config = PipelineConfig(
            stream_urls=[f"rtsp://localhost:8554/cam_{i:02d}" for i in range(1, 6)],
            checkpoint_path="/path/to/model.ckpt",
        )

        assert config.num_cameras == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
