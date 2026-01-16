"""Detection, tracking, and calibration services for the inference pipeline."""

from .detection_service import (
    DetectionService,
    Detection,
    FrameDetections,
)
from .gt_detection_service import GroundTruthDetectionService
from .camera_calibration_service import (
    CameraCalibrationService,
    # Camera transformation constants and utilities
    TRAINING_FX_NORM,
    TARGET_POSE_RADIUS,
    OPENGL_TO_OPENCV_FLIP,
    apply_coordinate_flip,
    normalize_camera_poses,
)
from .ground_truth_depth_service import GroundTruthDepthService, depth_to_colormap

__all__ = [
    "DetectionService",
    "Detection",
    "FrameDetections",
    "GroundTruthDetectionService",
    "CameraCalibrationService",
    "GroundTruthDepthService",
    "depth_to_colormap",
    # Camera transformation utilities
    "TRAINING_FX_NORM",
    "TARGET_POSE_RADIUS",
    "OPENGL_TO_OPENCV_FLIP",
    "apply_coordinate_flip",
    "normalize_camera_poses",
]

