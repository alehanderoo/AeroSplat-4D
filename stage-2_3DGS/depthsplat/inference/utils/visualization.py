"""
Visualization utilities for DepthSplat Inference Pipeline.

This module provides debug visualization and rendering utilities.

Usage:
    from utils.visualization import visualize_gaussians, create_debug_overlay
    
    # Visualize Gaussians
    visualize_gaussians(positions, colors)
    
    # Create debug overlay on frame
    overlay = create_debug_overlay(frame, metrics)
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def visualize_gaussians(
    positions: np.ndarray,
    colors: Optional[np.ndarray] = None,
    opacities: Optional[np.ndarray] = None,
    output_path: Optional[str] = None,
    title: str = "3D Gaussians",
    point_size: float = 5.0,
    show: bool = True,
) -> Optional[Any]:
    """
    Visualize 3D Gaussian positions.
    
    Args:
        positions: [N, 3] XYZ positions
        colors: Optional [N, 3] RGB colors (0-1)
        opacities: Optional [N, 1] opacities
        output_path: Optional path to save figure
        title: Plot title
        point_size: Size of points
        show: Whether to display the plot
        
    Returns:
        Figure object if matplotlib available
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        logger.warning("matplotlib not available for visualization")
        return None
    
    positions = np.asarray(positions)
    n = positions.shape[0]
    
    # Default colors
    if colors is None:
        colors = np.ones((n, 3)) * 0.5  # Gray
    else:
        colors = np.asarray(colors)
        if colors.max() > 1:
            colors = colors / 255.0
    
    # Apply opacities to colors (as alpha)
    if opacities is not None:
        opacities = np.asarray(opacities).ravel()
        alpha = opacities
    else:
        alpha = np.ones(n)
    
    # Create RGBA colors
    rgba = np.column_stack([colors[:, :3], alpha])
    
    # Create figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Scatter plot
    ax.scatter(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        c=rgba,
        s=point_size,
        alpha=0.7,
    )
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f"{title} (N={n:,})")
    
    # Equal aspect ratio
    max_range = np.abs(positions).max()
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])
    
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved visualization to {output_path}")
    
    if show:
        plt.show()
    
    plt.close(fig)
    return fig


def create_debug_overlay(
    frame: np.ndarray,
    metrics: Optional[Dict[str, Any]] = None,
    camera_id: Optional[str] = None,
    frame_id: Optional[int] = None,
    font_scale: float = 0.6,
    color: Tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """
    Create a debug overlay on a frame with metrics information.
    
    Args:
        frame: Input frame [H, W, 3] or [H, W]
        metrics: Dictionary of metrics to display
        camera_id: Camera identifier
        frame_id: Frame number
        font_scale: Font scale for text
        color: Text color in RGB
        
    Returns:
        Frame with overlay
    """
    try:
        import cv2
    except ImportError:
        logger.warning("OpenCV not available for overlay")
        return frame
    
    # Ensure frame is writable
    frame = np.asarray(frame).copy()
    
    # Convert grayscale to RGB if needed
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    
    # Build text lines
    lines = []
    
    if camera_id:
        lines.append(f"Camera: {camera_id}")
    
    if frame_id is not None:
        lines.append(f"Frame: {frame_id}")
    
    if metrics:
        for key, value in metrics.items():
            if isinstance(value, float):
                lines.append(f"{key}: {value:.2f}")
            else:
                lines.append(f"{key}: {value}")
    
    # Draw text
    y = 30
    for line in lines:
        cv2.putText(
            frame,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            2,
        )
        y += int(30 * font_scale + 10)
    
    return frame


def create_multi_camera_grid(
    frames: Dict[str, np.ndarray],
    grid_size: Optional[Tuple[int, int]] = None,
    frame_size: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """
    Create a grid view of multiple camera frames.
    
    Args:
        frames: Dictionary mapping camera_id to frame
        grid_size: (rows, cols) for grid layout
        frame_size: (width, height) to resize frames to
        
    Returns:
        Combined grid image
    """
    try:
        import cv2
    except ImportError:
        logger.warning("OpenCV not available for grid creation")
        # Return first frame as fallback
        if frames:
            return next(iter(frames.values()))
        return np.zeros((480, 640, 3), dtype=np.uint8)
    
    if not frames:
        return np.zeros((480, 640, 3), dtype=np.uint8)
    
    n_cameras = len(frames)
    
    # Determine grid size
    if grid_size is None:
        cols = int(np.ceil(np.sqrt(n_cameras)))
        rows = int(np.ceil(n_cameras / cols))
        grid_size = (rows, cols)
    
    rows, cols = grid_size
    
    # Get frame size
    first_frame = next(iter(frames.values()))
    if frame_size is None:
        h, w = first_frame.shape[:2]
        # Scale down for grid
        scale = 0.5 if max(h, w) > 800 else 1.0
        frame_size = (int(w * scale), int(h * scale))
    
    w, h = frame_size
    
    # Create grid
    grid = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    
    for idx, (cam_id, frame) in enumerate(sorted(frames.items())):
        row = idx // cols
        col = idx % cols
        
        # Resize frame
        if frame.shape[:2] != (h, w):
            frame = cv2.resize(frame, frame_size)
        
        # Convert grayscale if needed
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        
        # Add camera label
        cv2.putText(
            frame,
            cam_id,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        
        # Place in grid
        y1, y2 = row * h, (row + 1) * h
        x1, x2 = col * w, (col + 1) * w
        grid[y1:y2, x1:x2] = frame
    
    return grid


def draw_gaussians_on_frame(
    frame: np.ndarray,
    positions_2d: np.ndarray,
    colors: Optional[np.ndarray] = None,
    radii: Optional[np.ndarray] = None,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Draw projected Gaussians on a frame.
    
    Args:
        frame: Input frame [H, W, 3]
        positions_2d: [N, 2] projected pixel coordinates
        colors: Optional [N, 3] RGB colors (0-1)
        radii: Optional [N] radii in pixels
        alpha: Blending alpha
        
    Returns:
        Frame with Gaussians drawn
    """
    try:
        import cv2
    except ImportError:
        logger.warning("OpenCV not available")
        return frame
    
    frame = np.asarray(frame).copy()
    positions_2d = np.asarray(positions_2d)
    n = positions_2d.shape[0]
    
    # Default colors and radii
    if colors is None:
        colors = np.ones((n, 3)) * 0.5
    if radii is None:
        radii = np.ones(n) * 3
    
    # Convert colors to BGR uint8
    if colors.max() <= 1.0:
        colors = (colors * 255).astype(np.uint8)
    
    # Draw circles
    overlay = frame.copy()
    
    for i in range(n):
        center = tuple(positions_2d[i].astype(int))
        radius = int(radii[i])
        color = tuple(int(c) for c in colors[i][::-1])  # RGB to BGR
        
        cv2.circle(overlay, center, radius, color, -1)
    
    # Blend
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    
    return frame


def save_video(
    frames: List[np.ndarray],
    output_path: str,
    fps: int = 30,
    codec: str = "mp4v",
) -> str:
    """
    Save a list of frames as a video.
    
    Args:
        frames: List of frames [H, W, 3]
        output_path: Path to save video
        fps: Frames per second
        codec: FourCC codec code
        
    Returns:
        Path to saved video
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("OpenCV required for video export")
    
    if not frames:
        raise ValueError("No frames to save")
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get frame dimensions
    h, w = frames[0].shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    
    for frame in frames:
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        writer.write(frame)
    
    writer.release()
    
    logger.info(f"Saved video ({len(frames)} frames) to {output_path}")
    return str(output_path)


if __name__ == "__main__":
    # Test visualization
    n = 100
    positions = np.random.randn(n, 3).astype(np.float32)
    colors = np.random.rand(n, 3).astype(np.float32)
    
    visualize_gaussians(positions, colors, show=False, output_path="/tmp/test_viz.png")
    print("Saved visualization to /tmp/test_viz.png")
    
    # Test overlay
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    metrics = {"FPS": 30.5, "Latency": 25.3, "Gaussians": 1000}
    result = create_debug_overlay(frame, metrics, camera_id="cam_01", frame_id=42)
    print(f"Overlay shape: {result.shape}")
