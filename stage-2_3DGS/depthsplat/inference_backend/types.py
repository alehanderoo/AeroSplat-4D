"""
Type definitions for the DepthSplat inference backend.

This module defines data classes for inputs and outputs to ensure
a clean, well-typed API for the inference service.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np


@dataclass
class CameraParameters:
    """
    Camera parameters for a single view.

    Attributes:
        extrinsics: [4, 4] camera-to-world matrix (OpenCV convention)
        intrinsics: [3, 3] normalized intrinsics matrix (fx, fy, cx, cy in [0, 1])
    """
    extrinsics: np.ndarray  # [4, 4]
    intrinsics: np.ndarray  # [3, 3]

    def __post_init__(self):
        """Validate shapes."""
        assert self.extrinsics.shape == (4, 4), f"Extrinsics must be [4, 4], got {self.extrinsics.shape}"
        assert self.intrinsics.shape == (3, 3), f"Intrinsics must be [3, 3], got {self.intrinsics.shape}"


@dataclass
class InputContext:
    """
    Input context for 3D reconstruction.

    Contains the multi-view images and camera parameters needed
    to reconstruct a 3D Gaussian Splat model.

    Attributes:
        images: List of RGB images as numpy arrays [H, W, 3] in uint8
        extrinsics: [V, 4, 4] camera-to-world matrices (OpenCV convention)
        intrinsics: [V, 3, 3] normalized intrinsics matrices
        scene_id: Optional identifier for the scene
    """
    images: List[np.ndarray]
    extrinsics: np.ndarray  # [V, 4, 4]
    intrinsics: np.ndarray  # [V, 3, 3]
    scene_id: Optional[str] = None

    @property
    def num_views(self) -> int:
        """Number of input views."""
        return len(self.images)

    def validate(self):
        """Validate that all inputs are consistent."""
        n_views = len(self.images)
        assert self.extrinsics.shape[0] == n_views, \
            f"Extrinsics has {self.extrinsics.shape[0]} views, but {n_views} images provided"
        assert self.intrinsics.shape[0] == n_views, \
            f"Intrinsics has {self.intrinsics.shape[0]} views, but {n_views} images provided"

        for i, img in enumerate(self.images):
            assert img.ndim == 3 and img.shape[2] == 3, \
                f"Image {i} must be [H, W, 3], got {img.shape}"


@dataclass
class RenderSettings:
    """
    Settings for rendering novel views.

    Attributes:
        azimuth: Target view azimuth in degrees (0-360, 0=front)
        elevation: Target view elevation in degrees (-90 to 90)
        distance: Distance factor from object center (0.6-1.4, 1.0=normal)
    """
    azimuth: float = 0.0
    elevation: float = 30.0
    distance: float = 1.0

    def validate(self):
        """Validate settings are within valid ranges."""
        assert 0 <= self.azimuth <= 360, f"Azimuth must be in [0, 360], got {self.azimuth}"
        assert -90 <= self.elevation <= 90, f"Elevation must be in [-90, 90], got {self.elevation}"
        assert 0.3 <= self.distance <= 2.0, f"Distance must be in [0.3, 2.0], got {self.distance}"


@dataclass
class VideoSettings:
    """
    Settings for 360-degree video generation.

    Attributes:
        enabled: Whether to generate videos
        num_frames: Number of frames for the video (30-120)
        fps: Frames per second for output video
        elevation: Camera elevation for video orbit
        distance: Camera distance for video orbit
        include_depth: Whether to generate depth video
        include_silhouette: Whether to generate silhouette video
    """
    enabled: bool = True
    num_frames: int = 60
    fps: int = 30
    elevation: float = 30.0
    distance: float = 1.0
    include_depth: bool = True
    include_silhouette: bool = True


@dataclass
class InferenceResult:
    """
    Result of 3D Gaussian Splat inference.

    Contains paths to all generated outputs and metadata.

    Attributes:
        ply_path: Path to the exported PLY file with 3D Gaussians
        rendered_image_path: Path to the rendered novel view image
        depth_image_path: Path to the depth visualization (optional)
        silhouette_image_path: Path to the silhouette/alpha image (optional)
        video_rgb_path: Path to RGB 360 video (optional)
        video_depth_path: Path to depth 360 video (optional)
        video_silhouette_path: Path to silhouette 360 video (optional)
        output_dir: Directory containing all outputs
        depth_analysis: Dictionary with depth analysis results (optional)
        metadata: Additional metadata about the reconstruction
    """
    ply_path: Optional[str] = None
    rendered_image_path: Optional[str] = None
    depth_image_path: Optional[str] = None
    silhouette_image_path: Optional[str] = None
    video_rgb_path: Optional[str] = None
    video_depth_path: Optional[str] = None
    video_silhouette_path: Optional[str] = None
    output_dir: Optional[str] = None
    depth_analysis: Optional[Dict] = None
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'ply_path': self.ply_path,
            'rendered_image_path': self.rendered_image_path,
            'depth_image_path': self.depth_image_path,
            'silhouette_image_path': self.silhouette_image_path,
            'video_rgb_path': self.video_rgb_path,
            'video_depth_path': self.video_depth_path,
            'video_silhouette_path': self.video_silhouette_path,
            'output_dir': self.output_dir,
            'depth_analysis': self.depth_analysis,
            'metadata': self.metadata,
        }


@dataclass
class DepthAnalysisResult:
    """
    Result of depth analysis comparing different depth estimation methods.

    Attributes:
        standalone_da_paths: Paths to standalone Depth Anything V2 visualizations
        coarse_mv_paths: Paths to coarse multi-view depth visualizations
        residual_paths: Paths to DPT residual visualizations
        final_fused_paths: Paths to final fused depth visualizations
        gt_paths: Paths to ground truth depth (if available)
        metrics: Depth metrics comparing predictions to ground truth
    """
    standalone_da_paths: List[str] = field(default_factory=list)
    coarse_mv_paths: List[str] = field(default_factory=list)
    residual_paths: List[str] = field(default_factory=list)
    final_fused_paths: List[str] = field(default_factory=list)
    gt_paths: List[str] = field(default_factory=list)
    metrics: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'standalone_da_paths': self.standalone_da_paths,
            'coarse_mv_paths': self.coarse_mv_paths,
            'residual_paths': self.residual_paths,
            'final_fused_paths': self.final_fused_paths,
            'gt_paths': self.gt_paths,
            'metrics': self.metrics,
        }
