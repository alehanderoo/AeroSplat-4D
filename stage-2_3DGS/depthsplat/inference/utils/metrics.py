"""
Prometheus Metrics for DepthSplat Inference Pipeline.

This module provides monitoring metrics collection using Prometheus.

Usage:
    from utils.metrics import MetricsServer, get_metrics_server
    
    server = get_metrics_server(host="0.0.0.0", port=9090)
    server.start()
    
    # Record metrics
    server.record_frame_processed()
    server.record_inference_latency(25.5)
    
    server.stop()
"""

import time
import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class MetricsConfig:
    """Configuration for metrics server."""
    host: str = "0.0.0.0"
    port: int = 9090
    enabled: bool = True


class MetricsBase(ABC):
    """Abstract base class for metrics implementations."""
    
    def __init__(self, config: MetricsConfig):
        self.config = config
        self.running = False
    
    @abstractmethod
    def start(self):
        """Start the metrics server."""
        pass
    
    @abstractmethod
    def stop(self):
        """Stop the metrics server."""
        pass
    
    @abstractmethod
    def record_frame_processed(self, camera_id: Optional[str] = None):
        """Record that a frame was processed."""
        pass
    
    @abstractmethod
    def record_inference_latency(self, latency_ms: float):
        """Record inference latency in milliseconds."""
        pass
    
    @abstractmethod
    def record_gaussians_count(self, count: int):
        """Record number of Gaussians output."""
        pass
    
    @abstractmethod
    def set_stream_status(self, camera_id: str, connected: bool):
        """Set stream connection status for a camera."""
        pass
    
    @abstractmethod
    def record_memory_usage(self, gpu_mb: float, cpu_mb: float):
        """Record memory usage."""
        pass


class PrometheusMetrics(MetricsBase):
    """Prometheus metrics implementation."""
    
    def __init__(self, config: MetricsConfig):
        super().__init__(config)
        
        try:
            from prometheus_client import (
                Counter, Histogram, Gauge, 
                start_http_server, REGISTRY
            )
            self._prometheus = True
            self._start_http_server = start_http_server
            
            # Create metrics
            self.frames_processed = Counter(
                'depthsplat_frames_total',
                'Total frames processed',
                ['camera_id']
            )
            
            self.inference_latency = Histogram(
                'depthsplat_inference_seconds',
                'Inference latency in seconds',
                buckets=[0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.05, 0.1]
            )
            
            self.gaussians_count = Gauge(
                'depthsplat_gaussians_count',
                'Number of Gaussians per frame'
            )
            
            self.stream_status = Gauge(
                'depthsplat_stream_status',
                'Stream connection status (1=connected, 0=disconnected)',
                ['camera_id']
            )
            
            self.gpu_memory = Gauge(
                'depthsplat_gpu_memory_mb',
                'GPU memory usage in MB'
            )
            
            self.cpu_memory = Gauge(
                'depthsplat_cpu_memory_mb',
                'CPU memory usage in MB'
            )
            
            self.fps = Gauge(
                'depthsplat_fps',
                'Current frames per second'
            )
            
        except ImportError:
            raise ImportError(
                "prometheus_client not available. "
                "Install with: pip install prometheus-client"
            )
        
        self._server_thread = None
        self._frame_count = 0
        self._last_fps_update = time.time()
    
    def start(self):
        """Start the Prometheus HTTP server."""
        if self.running:
            return
        
        try:
            self._start_http_server(self.config.port, addr=self.config.host)
            self.running = True
            logger.info(
                f"Prometheus metrics server started at "
                f"http://{self.config.host}:{self.config.port}/metrics"
            )
        except Exception as e:
            logger.error(f"Failed to start Prometheus server: {e}")
            raise
    
    def stop(self):
        """Stop the Prometheus server."""
        self.running = False
        logger.info("Prometheus metrics server stopped")
    
    def record_frame_processed(self, camera_id: Optional[str] = None):
        """Record a processed frame."""
        camera = camera_id or "all"
        self.frames_processed.labels(camera_id=camera).inc()
        
        # Update FPS calculation
        self._frame_count += 1
        now = time.time()
        elapsed = now - self._last_fps_update
        
        if elapsed >= 1.0:
            current_fps = self._frame_count / elapsed
            self.fps.set(current_fps)
            self._frame_count = 0
            self._last_fps_update = now
    
    def record_inference_latency(self, latency_ms: float):
        """Record inference latency."""
        # Convert ms to seconds for Prometheus conventions
        self.inference_latency.observe(latency_ms / 1000.0)
    
    def record_gaussians_count(self, count: int):
        """Record Gaussians count."""
        self.gaussians_count.set(count)
    
    def set_stream_status(self, camera_id: str, connected: bool):
        """Set stream status."""
        self.stream_status.labels(camera_id=camera_id).set(1 if connected else 0)
    
    def record_memory_usage(self, gpu_mb: float, cpu_mb: float):
        """Record memory usage."""
        self.gpu_memory.set(gpu_mb)
        self.cpu_memory.set(cpu_mb)


class SimpleMetrics(MetricsBase):
    """
    Simple in-memory metrics implementation.
    
    Used when prometheus_client is not available.
    Logs metrics instead of exposing them via HTTP.
    """
    
    def __init__(self, config: MetricsConfig):
        super().__init__(config)
        self._metrics: Dict[str, Any] = {
            "frames_processed": 0,
            "latencies": [],
            "gaussians_count": 0,
            "stream_status": {},
        }
        self._frame_count = 0
        self._last_log_time = time.time()
        self._log_interval = 5.0  # Log every 5 seconds
    
    def start(self):
        """Start simple metrics collection."""
        self.running = True
        logger.info("Simple metrics collection started (no HTTP server)")
    
    def stop(self):
        """Stop simple metrics collection."""
        self.running = False
        self._log_summary()
        logger.info("Simple metrics collection stopped")
    
    def _log_summary(self):
        """Log a summary of collected metrics."""
        latencies = self._metrics["latencies"]
        if latencies:
            avg_latency = sum(latencies) / len(latencies)
            max_latency = max(latencies)
            min_latency = min(latencies)
        else:
            avg_latency = max_latency = min_latency = 0.0
        
        logger.info(
            f"Metrics Summary: "
            f"frames={self._metrics['frames_processed']}, "
            f"latency(avg/min/max)={avg_latency:.1f}/{min_latency:.1f}/{max_latency:.1f}ms, "
            f"gaussians={self._metrics['gaussians_count']}"
        )
    
    def record_frame_processed(self, camera_id: Optional[str] = None):
        """Record a processed frame."""
        self._metrics["frames_processed"] += 1
        self._frame_count += 1
        
        # Periodic logging
        now = time.time()
        if now - self._last_log_time >= self._log_interval:
            fps = self._frame_count / (now - self._last_log_time)
            logger.debug(f"Processing at {fps:.1f} FPS")
            self._frame_count = 0
            self._last_log_time = now
    
    def record_inference_latency(self, latency_ms: float):
        """Record inference latency."""
        self._metrics["latencies"].append(latency_ms)
        # Keep only last 1000 samples
        if len(self._metrics["latencies"]) > 1000:
            self._metrics["latencies"] = self._metrics["latencies"][-1000:]
    
    def record_gaussians_count(self, count: int):
        """Record Gaussians count."""
        self._metrics["gaussians_count"] = count
    
    def set_stream_status(self, camera_id: str, connected: bool):
        """Set stream status."""
        self._metrics["stream_status"][camera_id] = connected
    
    def record_memory_usage(self, gpu_mb: float, cpu_mb: float):
        """Record memory usage."""
        logger.debug(f"Memory: GPU={gpu_mb:.1f}MB, CPU={cpu_mb:.1f}MB")


class MetricsServer:
    """
    High-level metrics server that auto-selects the best backend.
    
    Usage:
        server = MetricsServer(host="0.0.0.0", port=9090)
        server.start()
        
        server.record_frame_processed()
        server.record_inference_latency(25.5)
        
        server.stop()
    """
    
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 9090,
        enabled: bool = True,
    ):
        self.config = MetricsConfig(
            host=host,
            port=port,
            enabled=enabled,
        )
        
        if not enabled:
            self._metrics = None
        else:
            self._metrics = self._create_metrics()
    
    def _create_metrics(self) -> MetricsBase:
        """Create the appropriate metrics backend."""
        try:
            return PrometheusMetrics(self.config)
        except ImportError:
            logger.warning(
                "prometheus_client not available, using simple metrics. "
                "Install with: pip install prometheus-client"
            )
            return SimpleMetrics(self.config)
    
    def start(self):
        """Start the metrics server."""
        if self._metrics:
            self._metrics.start()
    
    def stop(self):
        """Stop the metrics server."""
        if self._metrics:
            self._metrics.stop()
    
    def record_frame_processed(self, camera_id: Optional[str] = None):
        """Record a processed frame."""
        if self._metrics:
            self._metrics.record_frame_processed(camera_id)
    
    def record_inference_latency(self, latency_ms: float):
        """Record inference latency."""
        if self._metrics:
            self._metrics.record_inference_latency(latency_ms)
    
    def record_gaussians_count(self, count: int):
        """Record Gaussians count."""
        if self._metrics:
            self._metrics.record_gaussians_count(count)
    
    def set_stream_status(self, camera_id: str, connected: bool):
        """Set stream status."""
        if self._metrics:
            self._metrics.set_stream_status(camera_id, connected)
    
    def record_memory_usage(self, gpu_mb: float, cpu_mb: float):
        """Record memory usage."""
        if self._metrics:
            self._metrics.record_memory_usage(gpu_mb, cpu_mb)
    
    @property
    def running(self) -> bool:
        """Check if server is running."""
        return self._metrics.running if self._metrics else False


# Global metrics instance
_global_metrics: Optional[MetricsServer] = None


def get_metrics_server(
    host: str = "0.0.0.0",
    port: int = 9090,
    **kwargs
) -> MetricsServer:
    """
    Get or create the global metrics server.
    
    This ensures only one metrics server is running.
    """
    global _global_metrics
    
    if _global_metrics is None:
        _global_metrics = MetricsServer(host=host, port=port, **kwargs)
    
    return _global_metrics


if __name__ == "__main__":
    # Test the metrics server
    server = MetricsServer(port=9091)  # Different port for testing
    server.start()
    
    print("Metrics server running. Press Ctrl+C to stop.")
    
    try:
        for i in range(100):
            server.record_frame_processed(f"cam_0{(i % 5) + 1}")
            server.record_inference_latency(20 + (i % 10))
            server.record_gaussians_count(1000 + i * 10)
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
