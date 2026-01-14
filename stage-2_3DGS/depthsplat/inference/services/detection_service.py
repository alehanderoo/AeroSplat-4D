"""
Abstract Detection Service Interface.

This module defines the interface for detection services that provide
2D object locations per camera view. Implementations can be:
- Ground-truth based (from simulation data)
- Track-before-detect based (from RGB input processing)
- Neural network based (learned detectors)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class Detection:
    """
    A single object detection in a camera view.

    Attributes:
        center_2d: (x, y) pixel coordinates of object center
        visible: Whether the object is visible in this view
        confidence: Detection confidence score (0-1)
        bbox: Optional bounding box (x_min, y_min, x_max, y_max)
        depth: Optional depth in meters
        object_id: Optional object identifier for tracking
    """
    center_2d: Tuple[float, float]
    visible: bool = True
    confidence: float = 1.0
    bbox: Optional[Tuple[float, float, float, float]] = None
    depth: Optional[float] = None
    object_id: Optional[str] = None

    @property
    def x(self) -> float:
        """X coordinate of center."""
        return self.center_2d[0]

    @property
    def y(self) -> float:
        """Y coordinate of center."""
        return self.center_2d[1]

    def get_crop_region(
        self,
        crop_size: int,
        image_width: int,
        image_height: int,
        use_bbox: bool = False
    ) -> Tuple[int, int, int, int]:
        """
        Calculate crop region centered on the detection.

        Args:
            crop_size: Size of the square crop (before any resize). This is the
                      MINIMUM crop size - ensures consistent normalized focal length.
            image_width: Width of the source image
            image_height: Height of the source image
            use_bbox: If True and bbox is available, expand crop if object is larger

        Returns:
            (x1, y1, x2, y2) crop coordinates, clamped to image bounds
        """
        cx, cy = self.center_2d

        # Start with the configured crop size (ensures consistent focal length)
        half_size = crop_size / 2

        # If bbox available, expand crop if object is larger than crop_size
        if use_bbox and self.bbox is not None:
            x_min, y_min, x_max, y_max = self.bbox
            bbox_width = x_max - x_min
            bbox_height = y_max - y_min
            # Use max dimension with margin, but at least crop_size
            bbox_half_size = max(bbox_width, bbox_height) * 1.5 / 2
            half_size = max(half_size, bbox_half_size)

        # Calculate crop bounds
        x1 = int(cx - half_size)
        y1 = int(cy - half_size)
        x2 = int(cx + half_size)
        y2 = int(cy + half_size)

        # Clamp to image bounds while maintaining size
        if x1 < 0:
            x2 -= x1
            x1 = 0
        if y1 < 0:
            y2 -= y1
            y1 = 0
        if x2 > image_width:
            x1 -= (x2 - image_width)
            x2 = image_width
        if y2 > image_height:
            y1 -= (y2 - image_height)
            y2 = image_height

        # Final clamp
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image_width, x2)
        y2 = min(image_height, y2)

        return (x1, y1, x2, y2)


@dataclass
class FrameDetections:
    """
    Detections for all cameras in a single frame.

    Attributes:
        frame_id: Frame identifier/index
        timestamp: Frame timestamp in seconds
        detections: Dict mapping camera name to list of detections
        object_position_3d: Optional 3D world position if available
    """
    frame_id: int
    timestamp: float = 0.0
    detections: Dict[str, List[Detection]] = field(default_factory=dict)
    object_position_3d: Optional[Tuple[float, float, float]] = None

    def get_detection(self, camera_name: str, object_idx: int = 0) -> Optional[Detection]:
        """
        Get a specific detection for a camera.

        Args:
            camera_name: Name of the camera (e.g., "cam_01")
            object_idx: Index of the object (for multi-object tracking)

        Returns:
            Detection if found, None otherwise
        """
        if camera_name not in self.detections:
            return None
        detections = self.detections[camera_name]
        if object_idx >= len(detections):
            return None
        return detections[object_idx]

    def get_all_centers(self) -> Dict[str, Optional[Tuple[float, float]]]:
        """
        Get 2D centers for all cameras.

        Returns:
            Dict mapping camera name to (x, y) center or None if not visible
        """
        result = {}
        for camera_name, detections in self.detections.items():
            if detections and detections[0].visible:
                result[camera_name] = detections[0].center_2d
            else:
                result[camera_name] = None
        return result


class DetectionService(ABC):
    """
    Abstract base class for detection services.

    A detection service provides 2D object locations for each camera view.
    Implementations should handle:
    - Frame synchronization with the video streams
    - Multi-camera coordination
    - Object tracking across frames (if applicable)
    """

    def __init__(self, camera_names: List[str]):
        """
        Initialize the detection service.

        Args:
            camera_names: List of camera names to track
        """
        self.camera_names = camera_names
        self._current_frame_id = -1

    @abstractmethod
    def start(self):
        """Start the detection service."""
        pass

    @abstractmethod
    def stop(self):
        """Stop the detection service."""
        pass

    @abstractmethod
    def get_detections(self, frame_id: int) -> Optional[FrameDetections]:
        """
        Get detections for a specific frame.

        Args:
            frame_id: Frame index/identifier

        Returns:
            FrameDetections for the requested frame, or None if not available
        """
        pass

    def get_current_detections(self) -> Optional[FrameDetections]:
        """
        Get detections for the current frame.

        Returns:
            FrameDetections for current frame, or None if not available
        """
        return self.get_detections(self._current_frame_id)

    def advance_frame(self) -> Optional[FrameDetections]:
        """
        Advance to the next frame and return its detections.

        Returns:
            FrameDetections for the next frame
        """
        self._current_frame_id += 1
        return self.get_detections(self._current_frame_id)

    def reset(self):
        """Reset the service to the beginning."""
        self._current_frame_id = -1

    @property
    def current_frame_id(self) -> int:
        """Current frame index."""
        return self._current_frame_id
