"""
RTSP Stream Simulator for IsaacSim Rendered Frames.

This module creates RTSP streams from rendered PNG sequences,
simulating real IP cameras for pipeline development.

Usage:
    python rtsp_server.py --render-dir /path/to/renders --fps 30

Streams will be available at:
    rtsp://localhost:8554/cam_01
    rtsp://localhost:8554/cam_02
    ...
    rtsp://localhost:8554/cam_05

Requirements:
    - GStreamer with RTSP Server (gir1.2-gst-rtsp-server-1.0)
    - Environment variables for conda compatibility:
        export GI_TYPELIB_PATH=/usr/lib/x86_64-linux-gnu/girepository-1.0
        export GST_PLUGIN_SYSTEM_PATH=/usr/lib/x86_64-linux-gnu/gstreamer-1.0
        export GST_PLUGIN_PATH=""
"""

import os
import sys
import glob
import signal
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class CameraStreamConfig:
    """Configuration for a single camera RTSP stream."""
    camera_id: str
    rgb_dir: Path
    frame_pattern: str = "rgb_%04d.png"  # Matches IsaacSim output
    fps: int = 30
    width: int = 2560  # From actual renders
    height: int = 1440
    bitrate: int = 8000  # kbps, higher for 2.5K resolution

    @property
    def frame_glob(self) -> str:
        """Glob pattern for finding frames."""
        return str(self.rgb_dir / "rgb_*.png")

    def count_frames(self) -> int:
        """Count available frames."""
        return len(glob.glob(self.frame_glob))

    def get_frame_path(self, index: int) -> Path:
        """Get path to a specific frame."""
        return self.rgb_dir / f"rgb_{index:04d}.png"


@dataclass
class ServerConfig:
    """Configuration for the RTSP server."""
    render_dir: Path
    host: str = "0.0.0.0"
    port: int = 8554
    fps: int = 30
    loop: bool = True  # Loop frames for continuous streaming
    camera_ids: List[str] = field(default_factory=lambda: [
        "cam_01", "cam_02", "cam_03", "cam_04", "cam_05"
    ])

    def __post_init__(self):
        self.render_dir = Path(self.render_dir)


class GStreamerRTSPServer:
    """RTSP server using GStreamer RTSP Server library."""

    def __init__(self, config: ServerConfig):
        self.config = config
        self.camera_configs: Dict[str, CameraStreamConfig] = {}
        self.running = False
        self.server = None
        self.loop = None

        # Import GStreamer
        try:
            import gi
            gi.require_version('Gst', '1.0')
            gi.require_version('GstRtspServer', '1.0')
            from gi.repository import Gst, GstRtspServer, GLib

            self.Gst = Gst
            self.GstRtspServer = GstRtspServer
            self.GLib = GLib

            Gst.init(None)
        except (ImportError, ValueError) as e:
            raise ImportError(
                f"GStreamer RTSP Server not available: {e}\n"
                "Install with: sudo apt install gir1.2-gst-rtsp-server-1.0 libgstrtspserver-1.0-dev\n"
                "For conda, set: export GI_TYPELIB_PATH=/usr/lib/x86_64-linux-gnu/girepository-1.0"
            )

        self._validate_render_dir()
        self._setup_camera_configs()

    def _validate_render_dir(self):
        """Validate that render directory has expected structure."""
        if not self.config.render_dir.exists():
            raise ValueError(f"Render directory not found: {self.config.render_dir}")

        for cam_id in self.config.camera_ids:
            rgb_dir = self.config.render_dir / cam_id / "rgb"
            if not rgb_dir.exists():
                raise ValueError(f"Missing RGB directory: {rgb_dir}")

            frames = list(rgb_dir.glob("rgb_*.png"))
            if len(frames) == 0:
                raise ValueError(f"No frames found in: {rgb_dir}")

        logger.info(f"Validated render directory: {self.config.render_dir}")

    def _setup_camera_configs(self):
        """Set up configuration for each camera."""
        for cam_id in self.config.camera_ids:
            rgb_dir = self.config.render_dir / cam_id / "rgb"

            # Detect frame dimensions from first frame
            first_frame = sorted(rgb_dir.glob("rgb_*.png"))[0]
            width, height = self._get_image_dimensions(first_frame)

            self.camera_configs[cam_id] = CameraStreamConfig(
                camera_id=cam_id,
                rgb_dir=rgb_dir,
                fps=self.config.fps,
                width=width,
                height=height,
            )

            frame_count = self.camera_configs[cam_id].count_frames()
            logger.info(f"Camera {cam_id}: {frame_count} frames @ {width}x{height}")

    def _get_image_dimensions(self, image_path: Path) -> tuple:
        """Get image dimensions without loading full image."""
        try:
            import cv2
            img = cv2.imread(str(image_path))
            if img is not None:
                return img.shape[1], img.shape[0]
        except ImportError:
            pass

        # Fallback: use file command
        try:
            import subprocess
            result = subprocess.run(
                ["file", str(image_path)],
                capture_output=True, text=True
            )
            # Parse: "PNG image data, 2560 x 1440, ..."
            parts = result.stdout.split(",")
            for part in parts:
                if "x" in part and any(c.isdigit() for c in part):
                    dims = part.strip().split("x")
                    if len(dims) == 2:
                        return int(dims[0].strip()), int(dims[1].strip())
        except Exception:
            pass

        # Default fallback
        return 2560, 1440

    def _create_pipeline_string(self, cam_config: CameraStreamConfig) -> str:
        """Create GStreamer pipeline string for a camera."""
        loop_str = "loop=true" if self.config.loop else ""

        # RTSPMediaFactory requires pipeline wrapped in parentheses
        pipeline = (
            f'( multifilesrc location="{cam_config.rgb_dir}/rgb_%04d.png" '
            f'start-index=0 {loop_str} '
            f'caps="image/png,framerate={cam_config.fps}/1" ! '
            f'pngdec ! '
            f'videoconvert ! '
            f'video/x-raw,format=I420 ! '
            f'x264enc tune=zerolatency bitrate={cam_config.bitrate} '
            f'speed-preset=ultrafast key-int-max={cam_config.fps} ! '
            f'rtph264pay config-interval=1 name=pay0 pt=96 )'
        )
        return pipeline

    def start(self):
        """Start the GStreamer RTSP server."""
        self.server = self.GstRtspServer.RTSPServer()
        self.server.set_address(self.config.host)
        self.server.set_service(str(self.config.port))

        mounts = self.server.get_mount_points()

        for cam_id, cam_config in self.camera_configs.items():
            factory = self.GstRtspServer.RTSPMediaFactory()
            pipeline = self._create_pipeline_string(cam_config)
            factory.set_launch(pipeline)
            factory.set_shared(True)

            mount_point = f"/{cam_id}"
            mounts.add_factory(mount_point, factory)
            logger.info(f"Mounted {cam_id} at rtsp://{self.config.host}:{self.config.port}{mount_point}")

        self.server.attach(None)
        self.running = True

        self.loop = self.GLib.MainLoop()
        logger.info(f"GStreamer RTSP Server started at rtsp://{self.config.host}:{self.config.port}")

    def stop(self):
        """Stop the GStreamer RTSP server."""
        self.running = False
        if self.loop and self.loop.is_running():
            self.loop.quit()
        logger.info("GStreamer RTSP Server stopped")

    def wait(self):
        """Wait for server to finish (blocking)."""
        if self.loop:
            self.loop.run()

    def get_stream_urls(self) -> Dict[str, str]:
        """Get RTSP URLs for all cameras."""
        return {
            cam_id: f"rtsp://{self.config.host}:{self.config.port}/{cam_id}"
            for cam_id in self.config.camera_ids
        }


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="IsaacSim RTSP Stream Simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Start with default settings
    python rtsp_server.py --render-dir /path/to/renders

    # Custom FPS and port
    python rtsp_server.py --render-dir /path/to/renders --fps 60 --port 8555

Environment variables (required for conda):
    export GI_TYPELIB_PATH=/usr/lib/x86_64-linux-gnu/girepository-1.0
    export GST_PLUGIN_SYSTEM_PATH=/usr/lib/x86_64-linux-gnu/gstreamer-1.0
    export GST_PLUGIN_PATH=""
        """
    )
    parser.add_argument(
        "--render-dir",
        type=str,
        default="/home/sandro/thesis/renders/5cams_08-01-26_drone_50m",
        help="Path to IsaacSim renders directory",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8554)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--no-loop", action="store_true", help="Don't loop frames")

    args = parser.parse_args()

    config = ServerConfig(
        render_dir=Path(args.render_dir),
        host=args.host,
        port=args.port,
        fps=args.fps,
        loop=not args.no_loop,
    )

    server = GStreamerRTSPServer(config)

    # Handle Ctrl+C gracefully - use GLib's signal handling
    def signal_handler(sig, frame):
        print("\nShutting down...")
        server.stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        server.start()

        print("\n" + "=" * 60)
        print("RTSP Stream Simulator Running")
        print("=" * 60)
        print("\nAvailable streams:")
        for cam_id, url in server.get_stream_urls().items():
            print(f"  {cam_id}: {url}")
        print("\nPress Ctrl+C to stop")
        print("=" * 60 + "\n")

        server.wait()
    except Exception as e:
        logger.error(f"Server error: {e}")
        server.stop()
        sys.exit(1)


if __name__ == "__main__":
    main()
