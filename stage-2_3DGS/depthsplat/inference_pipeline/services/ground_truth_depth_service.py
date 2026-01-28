"""
Ground Truth Depth Service.

Loads ground truth depth images from rendered .npy files and provides
synchronized access for visualization comparison.
"""

import logging
from pathlib import Path
from typing import List, Optional, Dict
import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

logger = logging.getLogger(__name__)


class GroundTruthDepthService:
    """
    Service for loading and providing ground truth depth images.
    
    Loads depth maps from .npy files and synchronizes with frame index.
    Supports cropping to match detection-based cropped views.
    """
    
    def __init__(
        self,
        base_path: str,
        camera_names: List[str],
        depth_subdir: str = "depth",
        file_pattern: str = "distance_to_image_plane_{:04d}.npy",
        loop: bool = True,
    ):
        """
        Initialize the ground truth depth service.
        
        Args:
            base_path: Base path containing camera folders (e.g., ~/thesis/renders/5cams_08-01-26_drone_50m)
            camera_names: List of camera folder names (e.g., ["cam_01", "cam_02", ...])
            depth_subdir: Subdirectory within each camera folder containing depth files
            file_pattern: Format string for depth filenames with frame index placeholder
            loop: Whether to loop back to start when reaching end of frames
        """
        self.base_path = Path(base_path).expanduser()
        self.camera_names = camera_names
        self.depth_subdir = depth_subdir
        self.file_pattern = file_pattern
        self.loop = loop
        
        self.current_frame_idx = 0
        self.num_frames = 0
        self.running = False
        
        # Cache loaded depth for current frame per camera
        self._current_depths: Dict[str, Optional[np.ndarray]] = {}
        
        # Discover available frames
        self._discover_frames()
        
    def _discover_frames(self):
        """Discover available depth frames."""
        # Check first camera to determine number of frames
        if not self.camera_names:
            logger.warning("No camera names provided")
            return
            
        first_cam = self.camera_names[0]
        depth_dir = self.base_path / first_cam / self.depth_subdir
        
        if not depth_dir.exists():
            logger.warning(f"Depth directory not found: {depth_dir}")
            return
            
        # Count .npy files
        npy_files = sorted(depth_dir.glob("*.npy"))
        self.num_frames = len(npy_files)
        
        logger.info(f"Ground truth depth service: found {self.num_frames} frames in {depth_dir}")
        
    def start(self):
        """Start the service (resets frame counter)."""
        self.current_frame_idx = 0
        self.running = True
        logger.info("Ground truth depth service started")
        
    def stop(self):
        """Stop the service."""
        self.running = False
        self._current_depths.clear()
        logger.info("Ground truth depth service stopped")
        
    def advance_frame(self) -> int:
        """
        Advance to the next frame.
        
        Returns:
            Current frame index after advancing
        """
        if not self.running or self.num_frames == 0:
            return self.current_frame_idx
            
        # Clear cached depths
        self._current_depths.clear()
        
        # Advance frame
        self.current_frame_idx += 1
        
        if self.current_frame_idx >= self.num_frames:
            if self.loop:
                self.current_frame_idx = 0
            else:
                self.current_frame_idx = self.num_frames - 1
                
        return self.current_frame_idx
        
    def get_depth(self, camera_name: str) -> Optional[np.ndarray]:
        """
        Get depth map for a camera at current frame.
        
        Args:
            camera_name: Camera folder name (e.g., "cam_01")
            
        Returns:
            Depth array [H, W] in meters, or None if not found
        """
        # Check cache first
        if camera_name in self._current_depths:
            return self._current_depths.get(camera_name)
            
        # Load from file
        depth_path = (
            self.base_path / camera_name / self.depth_subdir / 
            self.file_pattern.format(self.current_frame_idx)
        )
        
        if not depth_path.exists():
            logger.debug(f"Depth file not found: {depth_path}")
            self._current_depths[camera_name] = None
            return None
            
        try:
            depth = np.load(depth_path)
            self._current_depths[camera_name] = depth
            return depth
        except Exception as e:
            logger.error(f"Failed to load depth from {depth_path}: {e}")
            self._current_depths[camera_name] = None
            return None
            
    def get_cropped_depth(
        self,
        camera_name: str,
        crop_region: tuple,
        output_size: tuple = (256, 256),
    ) -> Optional[np.ndarray]:
        """
        Get cropped depth map for a camera.
        
        Args:
            camera_name: Camera folder name
            crop_region: (x1, y1, x2, y2) crop box in original image coords
            output_size: (width, height) to resize cropped depth to
            
        Returns:
            Cropped and resized depth array [H, W], or None if not found
        """
        depth = self.get_depth(camera_name)
        if depth is None:
            return None
            
        x1, y1, x2, y2 = crop_region
        
        # Clamp to valid range
        h, w = depth.shape[:2]
        x1 = max(0, min(x1, w - 1))
        x2 = max(x1 + 1, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(y1 + 1, min(y2, h))
        
        # Crop
        cropped = depth[y1:y2, x1:x2]
        
        # Resize if needed
        if CV2_AVAILABLE and cropped.size > 0:
            cropped = cv2.resize(cropped, output_size, interpolation=cv2.INTER_LINEAR)
            
        return cropped
        
    def get_all_cropped_depths(
        self,
        crop_regions: List[tuple],
        output_size: tuple = (256, 256),
    ) -> List[Optional[np.ndarray]]:
        """
        Get cropped depth maps for all cameras.
        
        Args:
            crop_regions: List of (x1, y1, x2, y2) crop boxes, one per camera
            output_size: Output size for cropped depths
            
        Returns:
            List of cropped depth arrays (one per camera)
        """
        depths = []
        for i, camera_name in enumerate(self.camera_names):
            if i < len(crop_regions) and crop_regions[i] is not None:
                depth = self.get_cropped_depth(camera_name, crop_regions[i], output_size)
            else:
                depth = self.get_depth(camera_name)
            depths.append(depth)
        return depths
        

def depth_to_colormap(
    depth: np.ndarray,
    near: float = 0.5,
    far: float = 100.0,
    colormap: int = None,
) -> np.ndarray:
    """
    Convert depth array to RGB colormap visualization.
    
    Args:
        depth: Depth values [H, W] in meters
        near: Near plane distance
        far: Far plane distance  
        colormap: OpenCV colormap constant (default: COLORMAP_TURBO)
        
    Returns:
        RGB image [H, W, 3] as uint8
    """
    # Handle NaN/inf in near/far
    if not np.isfinite(near) or not np.isfinite(far):
        near = 0.5
        far = 100.0
    
    # Ensure minimum range to avoid divide by zero
    if far <= near:
        far = near + 0.01
    
    # Handle invalid values in depth array FIRST
    depth = np.nan_to_num(depth, nan=far, posinf=far, neginf=near)
    
    if not CV2_AVAILABLE:
        # Fallback to grayscale if no OpenCV
        depth_normalized = (depth - near) / (far - near)
        depth_normalized = np.clip(depth_normalized, 0, 1)
        gray = (depth_normalized * 255).astype(np.uint8)
        return np.stack([gray, gray, gray], axis=-1)
        
    if colormap is None:
        colormap = cv2.COLORMAP_TURBO
    
    # Normalize to [0, 1]
    depth_normalized = (depth - near) / (far - near)
    depth_normalized = np.clip(depth_normalized, 0, 1)
    
    # Convert to uint8
    depth_uint8 = (depth_normalized * 255).astype(np.uint8)
    
    # Apply colormap (returns BGR)
    colored = cv2.applyColorMap(depth_uint8, colormap)
    
    # Convert BGR to RGB
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    
    return colored
