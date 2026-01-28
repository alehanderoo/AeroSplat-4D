"""
Ground Truth Detection Service.

Reads pre-computed object detections from Isaac Sim render metadata.
This is used during development; in production, this will be replaced
by a track-before-detect service processing live RGB streams.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

from .detection_service import DetectionService, Detection, FrameDetections

logger = logging.getLogger(__name__)


class GroundTruthDetectionService(DetectionService):
    """
    Detection service that reads ground-truth data from Isaac Sim exports.

    The JSON file contains per-frame, per-camera 2D projections of objects
    including center coordinates, visibility, depth, and bounding boxes.

    Expected JSON structure:
    {
        "metadata": { "num_frames": N, ... },
        "cameras": [...],
        "frames": [
            {
                "frame_index": 0,
                "time_seconds": 0.0,
                "drone_position_3d": [x, y, z],
                "cameras": [
                    {
                        "name": "cam_01",
                        "drone_center_2d": [x, y],
                        "visible": true,
                        "depth": 30.5,
                        "bbox_2d": {"x_min": ..., "y_min": ..., "x_max": ..., "y_max": ...}
                    },
                    ...
                ]
            },
            ...
        ]
    }
    """

    def __init__(
        self,
        json_path: Union[str, Path],
        camera_names: List[str] = None,
        loop: bool = True,
    ):
        """
        Initialize the ground truth detection service.

        Args:
            json_path: Path to the drone_camera_observations.json file
            camera_names: List of camera names to use (auto-detected if None)
            loop: Whether to loop back to frame 0 after reaching the end
        """
        self.json_path = Path(json_path)
        self.loop = loop

        # Load and parse JSON
        self._data = self._load_json()
        self._frames: List[Dict] = self._data.get("frames", [])
        self._num_frames = len(self._frames)

        # Auto-detect camera names if not provided
        if camera_names is None:
            camera_names = self._data.get("metadata", {}).get(
                "cameras_recorded",
                ["cam_01", "cam_02", "cam_03", "cam_04", "cam_05"]
            )

        super().__init__(camera_names)

        # Pre-parse all frames for faster access
        self._parsed_frames: Dict[int, FrameDetections] = {}
        self._parse_all_frames()

        logger.info(
            f"GroundTruthDetectionService loaded: {self._num_frames} frames, "
            f"{len(self.camera_names)} cameras"
        )

    def _load_json(self) -> Dict:
        """Load and parse the JSON file."""
        if not self.json_path.exists():
            raise FileNotFoundError(f"Ground truth file not found: {self.json_path}")

        logger.info(f"Loading ground truth from: {self.json_path}")
        with open(self.json_path, 'r') as f:
            return json.load(f)

    def _parse_all_frames(self):
        """Pre-parse all frames into FrameDetections objects."""
        for frame_data in self._frames:
            frame_id = frame_data.get("frame_index", 0)
            timestamp = frame_data.get("time_seconds", 0.0)

            # Parse 3D position if available
            pos_3d = frame_data.get("drone_position_3d")
            object_position_3d = tuple(pos_3d) if pos_3d else None

            # Parse per-camera detections
            detections: Dict[str, List[Detection]] = {}

            for cam_data in frame_data.get("cameras", []):
                cam_name = cam_data.get("name")
                if cam_name not in self.camera_names:
                    continue

                center_2d = cam_data.get("drone_center_2d")
                if center_2d is None:
                    continue

                visible = cam_data.get("visible", True)
                depth = cam_data.get("depth")

                # Parse bbox if available
                bbox = None
                bbox_data = cam_data.get("bbox_2d")
                if bbox_data:
                    bbox = (
                        bbox_data.get("x_min", 0),
                        bbox_data.get("y_min", 0),
                        bbox_data.get("x_max", 0),
                        bbox_data.get("y_max", 0),
                    )

                # Construct mask path
                # {render_dir}/{cam_name}/mask/drone_mask_{frame_id:04d}.png
                mask_path = self.json_path.parent / cam_name / "mask" / f"drone_mask_{frame_id:04d}.png"
                
                detection = Detection(
                    center_2d=tuple(center_2d),
                    visible=visible,
                    confidence=1.0,  # Ground truth is always confident
                    bbox=bbox,
                    depth=depth,
                    object_id="drone",
                    mask_path=str(mask_path) if mask_path.exists() else None,
                )

                detections[cam_name] = [detection]

            self._parsed_frames[frame_id] = FrameDetections(
                frame_id=frame_id,
                timestamp=timestamp,
                detections=detections,
                object_position_3d=object_position_3d,
            )

    def start(self):
        """Start the detection service (no-op for GT service)."""
        self.reset()
        logger.info("GroundTruthDetectionService started")

    def stop(self):
        """Stop the detection service (no-op for GT service)."""
        logger.info("GroundTruthDetectionService stopped")

    def get_detections(self, frame_id: int) -> Optional[FrameDetections]:
        """
        Get detections for a specific frame.

        Args:
            frame_id: Frame index

        Returns:
            FrameDetections for the requested frame
        """
        if self._num_frames == 0:
            return None

        # Handle looping
        if self.loop:
            frame_id = frame_id % self._num_frames
        elif frame_id < 0 or frame_id >= self._num_frames:
            return None

        return self._parsed_frames.get(frame_id)

    @property
    def num_frames(self) -> int:
        """Total number of frames in the ground truth data."""
        return self._num_frames

    @property
    def metadata(self) -> Dict:
        """Raw metadata from the JSON file."""
        return self._data.get("metadata", {})

    @property
    def camera_info(self) -> List[Dict]:
        """Camera configuration from the JSON file."""
        return self._data.get("cameras", [])

    def get_camera_intrinsics(self, camera_name: str) -> Optional[Dict]:
        """
        Get intrinsics for a specific camera.

        Args:
            camera_name: Name of the camera

        Returns:
            Dict with intrinsics data or None
        """
        for cam in self.camera_info:
            if cam.get("name") == camera_name:
                return cam.get("intrinsics")
        return None

    def get_camera_resolution(self, camera_name: str) -> Optional[tuple]:
        """
        Get resolution for a specific camera.

        Args:
            camera_name: Name of the camera

        Returns:
            (width, height) tuple or None
        """
        for cam in self.camera_info:
            if cam.get("name") == camera_name:
                res = cam.get("resolution", {})
                return (res.get("width"), res.get("height"))
        return None


def create_gt_detection_service(
    render_dir: Union[str, Path],
    camera_names: List[str] = None,
    loop: bool = True,
) -> GroundTruthDetectionService:
    """
    Factory function to create a GroundTruthDetectionService.

    Args:
        render_dir: Directory containing Isaac Sim renders
        camera_names: List of camera names (auto-detected if None)
        loop: Whether to loop frames

    Returns:
        Configured GroundTruthDetectionService
    """
    render_dir = Path(render_dir)
    json_path = render_dir / "drone_camera_observations.json"

    return GroundTruthDetectionService(
        json_path=json_path,
        camera_names=camera_names,
        loop=loop,
    )
