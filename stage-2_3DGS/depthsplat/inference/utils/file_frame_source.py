"""
File-based Frame Source.

Reads frames directly from disk by frame ID, ensuring perfect synchronization
with ground truth detection data. This mirrors how the working Gradio demo
loads frames.

Unlike RTSP streaming, file-based loading guarantees that frame N from the
images matches frame N from the detection service.
"""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class FileFrameSource:
    """
    File-based frame source that reads frames directly from disk.

    This provides perfect synchronization with GT detection data because
    frames are loaded by explicit frame ID rather than streamed.

    Directory structure expected:
        {base_dir}/{camera_name}/rgb/rgb_{frame_id:04d}.png
        {base_dir}/{camera_name}/mask/drone_mask_{frame_id:04d}.png
    """

    def __init__(
        self,
        base_dir: str,
        camera_names: List[str],
        num_frames: int = 120,
        loop: bool = True,
    ):
        """
        Initialize the file frame source.

        Args:
            base_dir: Base directory containing camera subdirectories
            camera_names: List of camera names (e.g., ["cam_01", "cam_02", ...])
            num_frames: Total number of frames available
            loop: Whether to loop back to frame 0 after reaching the end
        """
        self.base_dir = Path(base_dir)
        self.camera_names = camera_names
        self.num_frames = num_frames
        self.loop = loop

        self._current_frame_id = -1
        self._verify_paths()

        logger.info(
            f"FileFrameSource initialized: {self.base_dir}, "
            f"{len(camera_names)} cameras, {num_frames} frames"
        )

    def _verify_paths(self):
        """Verify that expected directories exist."""
        if not self.base_dir.exists():
            raise FileNotFoundError(f"Base directory not found: {self.base_dir}")

        for cam_name in self.camera_names:
            rgb_dir = self.base_dir / cam_name / "rgb"
            if not rgb_dir.exists():
                logger.warning(f"RGB directory not found: {rgb_dir}")

    def get_rgb_path(self, camera_name: str, frame_id: int) -> Optional[Path]:
        """Get the RGB image path for a camera and frame."""
        if self.loop and self.num_frames > 0:
            frame_id = frame_id % self.num_frames

        # Try common naming patterns
        patterns = [
            self.base_dir / camera_name / "rgb" / f"rgb_{frame_id:04d}.png",
            self.base_dir / camera_name / "rgb" / f"rgb_{frame_id:05d}.png",
            self.base_dir / camera_name / "rgb" / f"frame_{frame_id:04d}.png",
            self.base_dir / camera_name / f"rgb_{frame_id:04d}.png",
        ]

        for path in patterns:
            if path.exists():
                return path

        return None

    def get_mask_path(self, camera_name: str, frame_id: int) -> Optional[Path]:
        """Get the mask image path for a camera and frame."""
        if self.loop and self.num_frames > 0:
            frame_id = frame_id % self.num_frames

        patterns = [
            self.base_dir / camera_name / "mask" / f"drone_mask_{frame_id:04d}.png",
            self.base_dir / camera_name / "mask" / f"mask_{frame_id:04d}.png",
        ]

        for path in patterns:
            if path.exists():
                return path

        return None

    def read_frame(self, camera_name: str, frame_id: int) -> Optional[np.ndarray]:
        """
        Read a single frame for a camera.

        Args:
            camera_name: Name of the camera
            frame_id: Frame index

        Returns:
            BGR numpy array or None if not found
        """
        rgb_path = self.get_rgb_path(camera_name, frame_id)
        if rgb_path is None:
            logger.warning(f"Frame not found: {camera_name} frame {frame_id}")
            return None

        frame = cv2.imread(str(rgb_path))
        if frame is None:
            logger.warning(f"Failed to read: {rgb_path}")
            return None

        return frame

    def read_all_cameras(self, frame_id: int) -> List[Optional[np.ndarray]]:
        """
        Read frames from all cameras for a given frame ID.

        Args:
            frame_id: Frame index

        Returns:
            List of BGR numpy arrays (one per camera), None for missing frames
        """
        frames = []
        for cam_name in self.camera_names:
            frame = self.read_frame(cam_name, frame_id)
            frames.append(frame)
        return frames

    def advance_frame(self) -> int:
        """
        Advance to the next frame and return its ID.

        Returns:
            The new frame ID
        """
        self._current_frame_id += 1
        if self.loop and self.num_frames > 0:
            self._current_frame_id = self._current_frame_id % self.num_frames
        return self._current_frame_id

    def get_current_frames(self) -> List[Optional[np.ndarray]]:
        """
        Get frames for the current frame ID.

        Returns:
            List of BGR numpy arrays (one per camera)
        """
        return self.read_all_cameras(self._current_frame_id)

    def reset(self):
        """Reset to the beginning."""
        self._current_frame_id = -1

    @property
    def current_frame_id(self) -> int:
        """Current frame index."""
        return self._current_frame_id


def create_file_frame_source(
    render_dir: str,
    camera_names: List[str] = None,
    num_frames: int = 120,
    loop: bool = True,
) -> FileFrameSource:
    """
    Factory function to create a FileFrameSource.

    Args:
        render_dir: Directory containing Isaac Sim renders
        camera_names: List of camera names (auto-detects if None)
        num_frames: Total number of frames
        loop: Whether to loop frames

    Returns:
        Configured FileFrameSource
    """
    render_dir = Path(render_dir)

    # Auto-detect camera names if not provided
    if camera_names is None:
        camera_names = []
        for d in sorted(render_dir.iterdir()):
            if d.is_dir() and d.name.startswith("cam_"):
                camera_names.append(d.name)

        if not camera_names:
            # Fallback to default names
            camera_names = [f"cam_{i+1:02d}" for i in range(5)]

    # Try to detect num_frames from first camera's RGB directory
    if num_frames <= 0:
        first_cam_rgb = render_dir / camera_names[0] / "rgb"
        if first_cam_rgb.exists():
            png_files = list(first_cam_rgb.glob("*.png"))
            num_frames = len(png_files)
            logger.info(f"Auto-detected {num_frames} frames")

    return FileFrameSource(
        base_dir=str(render_dir),
        camera_names=camera_names,
        num_frames=num_frames,
        loop=loop,
    )
