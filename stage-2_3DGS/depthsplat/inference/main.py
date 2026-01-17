#!/usr/bin/env python3
"""
DepthSplat Real-Time Visualization Server

Main entry point for the inference and visualization system that includes:
- Inference pipeline with server-side Gaussian rendering
- WebSocket server for real-time streaming to browser
- Optional HTTP server for serving the frontend

Usage:
    # Start visualization server with frontend
    python main_visualization.py --config config/pipeline_config.yaml

    # Without frontend (WebSocket only)
    python main_visualization.py --no-frontend

    # Custom ports
    python main_visualization.py --ws-port 8765 --http-port 8080

    # Quick test mode (runs for 2 seconds)
    python main_visualization.py --test

Environment Variables:
    DEPTHSPLAT_CONFIG: Default config file path
    DEPTHSPLAT_MODE: Default mode (dev/prod)
    DEPTHSPLAT_LOG_LEVEL: Logging level
"""

import argparse
import logging
import signal
import sys
import time
import threading
from pathlib import Path
from typing import Optional
from http.server import HTTPServer, SimpleHTTPRequestHandler
import functools
import os

# Add inference directory to path
INFERENCE_DIR = Path(__file__).parent
sys.path.insert(0, str(INFERENCE_DIR))


class CORSHTTPRequestHandler(SimpleHTTPRequestHandler):
    """HTTP handler with CORS support for development."""

    def __init__(self, *args, directory=None, **kwargs):
        self.directory = directory
        super().__init__(*args, directory=directory, **kwargs)

    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def log_message(self, format, *args):
        # Reduce noise - only log errors
        if args[1] != '200':
            super().log_message(format, *args)


def start_http_server(port: int, directory: str) -> HTTPServer:
    """Start HTTP server for frontend files."""
    handler = functools.partial(CORSHTTPRequestHandler, directory=directory)
    server = HTTPServer(('0.0.0.0', port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="DepthSplat Real-Time Visualization Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Start with defaults (WebSocket: 8765, HTTP: 8080)
    python main_visualization.py --config config/pipeline_config.yaml

    # WebSocket only (no frontend server)
    python main_visualization.py --no-frontend

    # Custom WebSocket port
    python main_visualization.py --ws-port 9000

    # Development mode with RTSP simulator
    python main_visualization.py --mode dev

    # Quick test (runs for 2 seconds)
    python main_visualization.py --test

    # Enable metrics on specific port
    python main_visualization.py --metrics-port 9090
        """
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config/pipeline_config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["dev", "prod"],
        default="dev",
        help="Pipeline mode: dev (with simulator) or prod (real cameras)",
    )
    parser.add_argument(
        "--ws-port",
        type=int,
        default=None,
        help="WebSocket server port (default: 8765 or from config)",
    )
    parser.add_argument(
        "--http-port",
        type=int,
        default=8080,
        help="HTTP server port for frontend (default: 8080)",
    )
    parser.add_argument(
        "--no-frontend",
        action="store_true",
        help="Disable frontend HTTP server",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=0,
        help="Run duration in seconds (0 for infinite)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run quick test (2 seconds)",
    )
    parser.add_argument(
        "--metrics-port",
        type=int,
        default=None,
        help="Prometheus metrics port (0 to disable, overrides config)",
    )

    args = parser.parse_args()

    # Import modules
    from config import load_config, PipelineSettings
    from utils.logging_utils import setup_logging, get_logger
    from utils.metrics import MetricsServer, get_metrics_server

    # Load configuration
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = INFERENCE_DIR / config_path

    # Load config or use defaults for test mode
    if args.test:
        config = PipelineSettings()
        config.monitoring.logging.level = "INFO"
    elif config_path.exists():
        config = load_config(config_path)
    else:
        print(f"Warning: Config file not found: {config_path}")
        print("Using default configuration")
        config = PipelineSettings()

    # Setup logging
    log_level = args.log_level or config.monitoring.logging.level
    setup_logging(
        level=log_level,
        format=config.monitoring.logging.format,
        file=config.monitoring.logging.file,
        use_colors=config.monitoring.logging.use_colors,
    )
    logger = get_logger(__name__)

    logger.info("=" * 60)
    logger.info(f"DepthSplat Visualization Server v{config.pipeline.version}")
    logger.info(f"Mode: {args.mode}")
    if args.test:
        logger.info("Running in TEST mode (2 second duration)")
    logger.info("=" * 60)

    # Determine WebSocket port
    ws_port = args.ws_port
    if ws_port is None:
        # Try to get from config visualization section
        try:
            ws_port = config._raw_dict.get('visualization', {}).get('websocket', {}).get('port', 8765)
        except:
            ws_port = 8765

    # Start HTTP server for frontend
    http_server = None
    if not args.no_frontend:
        frontend_dir = INFERENCE_DIR / "frontend"
        if frontend_dir.exists():
            try:
                http_server = start_http_server(args.http_port, str(frontend_dir))
                logger.info(f"Frontend server started at http://localhost:{args.http_port}")
            except Exception as e:
                logger.warning(f"Failed to start HTTP server: {e}")
        else:
            logger.warning(f"Frontend directory not found: {frontend_dir}")

    # Start metrics server
    metrics_server: Optional[MetricsServer] = None
    metrics_port = args.metrics_port if args.metrics_port is not None else config.monitoring.metrics.port
    
    if config.monitoring.metrics.enabled and metrics_port != 0:
        try:
            metrics_server = get_metrics_server(
                host=config.monitoring.metrics.host,
                port=metrics_port,
            )
            metrics_server.start()
            logger.info(f"Metrics server started at http://localhost:{metrics_port}")
        except Exception as e:
            logger.warning(f"Failed to start metrics server: {e}")

    # Create visualization pipeline
    pipeline = None

    try:
        from pipeline import VisualizationPipeline, VisualizationPipelineConfig

        # Get visualization config values
        vis_config = config._raw_dict.get('visualization', {})
        render_config = vis_config.get('render', {})
        thumb_config = vis_config.get('thumbnails', {})
        view_config = vis_config.get('view', {})
        detection_config = vis_config.get('detection', {})
        gt_depth_config = vis_config.get('gt_depth', {})
        file_source_config = vis_config.get('file_source', {})

        # Get render directory (single source of truth for all render-related paths)
        render_dir = config._raw_dict.get('render_dir')
        if render_dir:
            logger.info(f"Using render directory: {render_dir}")
            # Construct paths from render_dir
            gt_path = os.path.join(render_dir, "drone_camera_observations.json")
            gt_depth_base_path = render_dir if gt_depth_config.get('enabled', True) else None
            file_source_dir = render_dir
        else:
            # Fallback to individual paths (legacy config support)
            logger.warning("No render_dir configured, using individual paths")
            gt_path = detection_config.get('gt_path')
            gt_depth_base_path = gt_depth_config.get('base_path') if gt_depth_config.get('enabled', True) else None
            file_source_dir = file_source_config.get('directory')

        # Create pipeline config
        pipeline_config = VisualizationPipelineConfig(
            stream_urls=config.sources.rtsp.urls,
            checkpoint_path=config.model.checkpoint_path or "",
            experiment_config=None,
            model_input_width=config.model.input.width,
            model_input_height=config.model.input.height,
            gpu_id=config.deepstream.gpu_id,
            # Visualization settings
            visualization_enabled=vis_config.get('enabled', True),
            websocket_host="0.0.0.0",
            websocket_port=ws_port,
            render_width=render_config.get('width', 512),
            render_height=render_config.get('height', 512),
            default_view_mode=view_config.get('default_mode', 'orbit'),
            orbit_speed_deg_per_sec=view_config.get('orbit_speed_deg_per_sec', 15.0),
            input_thumbnail_width=thumb_config.get('width', 192),
            input_thumbnail_height=thumb_config.get('height', 108),
            jpeg_quality=render_config.get('jpeg_quality', 85),
            # Detection service settings (paths derived from render_dir)
            detection_enabled=detection_config.get('enabled', True),
            detection_gt_path=gt_path,
            min_crop_size=detection_config.get('min_crop_size', 64),
            target_object_coverage=detection_config.get('target_object_coverage', 0.75),
            crop_margin=detection_config.get('crop_margin', 0.15),
            # GT depth visualization settings (paths derived from render_dir)
            gt_depth_base_path=gt_depth_base_path,
            depth_near=gt_depth_config.get('depth_near', 0.5),
            depth_far=gt_depth_config.get('depth_far', 100.0),
            # File-based frame source settings (paths derived from render_dir)
            use_file_source=file_source_config.get('enabled', False),
            file_source_dir=file_source_dir,
            file_source_num_frames=file_source_config.get('num_frames', 120),
            file_source_loop=file_source_config.get('loop', True),
        )

        pipeline = VisualizationPipeline(pipeline_config)

        # Frame counter and logging
        frame_count = 0
        log_interval = config.monitoring.performance.log_interval
        last_log_time = time.time()

        def on_gaussians(output):
            nonlocal frame_count, last_log_time

            frame_count += 1

            # Record metrics
            if metrics_server:
                metrics_server.record_frame_processed()
                metrics_server.record_inference_latency(output.inference_time_ms)
                metrics_server.record_gaussians_count(len(output.positions))

            # Periodic logging
            if frame_count % log_interval == 0:
                now = time.time()
                elapsed = now - last_log_time
                fps = log_interval / elapsed if elapsed > 0 else 0

                logger.info(
                    f"Frame {frame_count} | "
                    f"FPS: {fps:.1f} | "
                    f"Latency: {output.inference_time_ms:.1f}ms | "
                    f"Gaussians: {len(output.positions)}"
                )

                last_log_time = now

        # Signal handlers
        shutdown_event = False

        def signal_handler(signum, frame):
            nonlocal shutdown_event
            logger.info("Shutdown signal received")
            shutdown_event = True

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Start pipeline
        logger.info("Starting visualization pipeline...")
        logger.info(f"WebSocket server: ws://localhost:{ws_port}")
        if not args.no_frontend:
            logger.info(f"Open browser: http://localhost:{args.http_port}")

        pipeline.start(callback=on_gaussians)

        logger.info("Pipeline running. Press Ctrl+C to stop.")

        # Main loop
        start_time = time.time()

        while not shutdown_event:
            time.sleep(0.1)

            # Check duration limit
            if args.duration > 0:
                if time.time() - start_time >= args.duration:
                    logger.info(f"Duration limit reached ({args.duration}s)")
                    break

            # Quick test mode: run for 2 seconds
            if args.test and time.time() - start_time >= 2.0:
                logger.info("Test complete")
                break

        # Print final stats
        stats = pipeline.get_stats()
        logger.info("=" * 60)
        logger.info("Pipeline Statistics:")
        logger.info(f"  Total frames: {stats['frame_count']}")
        logger.info(f"  Elapsed time: {stats['elapsed_time']:.1f}s")
        logger.info(f"  Average FPS: {stats['fps']:.1f}")
        logger.info(f"  Avg inference: {stats['avg_inference_ms']:.1f}ms")
        if 'render_time_ms' in stats:
            logger.info(f"  Avg render: {stats['render_time_ms']:.1f}ms")
        if 'connected_clients' in stats:
            logger.info(f"  Connected clients: {stats['connected_clients']}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Cleanup
        logger.info("Shutting down...")

        if pipeline:
            pipeline.stop()

        if http_server:
            http_server.shutdown()

        if metrics_server:
            metrics_server.stop()

        logger.info("Shutdown complete")


if __name__ == "__main__":
    main()
