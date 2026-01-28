"""
Detection services for the DepthSplat inference backend.

These services provide object detection and camera calibration data
for processing in-the-wild frames (e.g., from Isaac Sim renders).
"""

from .detection_service import Detection, FrameDetections, DetectionService
from .gt_detection_service import GroundTruthDetectionService, create_gt_detection_service

__all__ = [
    "Detection",
    "FrameDetections",
    "DetectionService",
    "GroundTruthDetectionService",
    "create_gt_detection_service",
]
