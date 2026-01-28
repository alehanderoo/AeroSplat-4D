"""
Gaussian utilities for DepthSplat Inference Pipeline.

This module provides utilities for manipulating and exporting
3D Gaussian representations.

Usage:
    from utils.gaussian_utils import GaussianBuffer, export_gaussians
    
    # Create buffer for 4DGS
    buffer = GaussianBuffer(max_frames=120)
    buffer.add(positions, covariances, colors, opacities, frame_id=0)
    
    # Export to PLY
    export_gaussians("output.ply", positions, covariances, colors, opacities)
"""

import logging
import struct
import threading
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Union, Tuple
from collections import OrderedDict

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class GaussianData:
    """Container for Gaussian parameters for a single frame."""
    frame_id: int
    timestamp: float
    positions: np.ndarray      # [N, 3]
    covariances: np.ndarray    # [N, 6] or [N, 3, 3]
    colors: np.ndarray         # [N, 3] or [N, D] for SH
    opacities: np.ndarray      # [N, 1]
    
    @property
    def num_gaussians(self) -> int:
        return self.positions.shape[0]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "positions": self.positions,
            "covariances": self.covariances,
            "colors": self.colors,
            "opacities": self.opacities,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GaussianData":
        """Create from dictionary."""
        return cls(
            frame_id=data["frame_id"],
            timestamp=data["timestamp"],
            positions=np.asarray(data["positions"]),
            covariances=np.asarray(data["covariances"]),
            colors=np.asarray(data["colors"]),
            opacities=np.asarray(data["opacities"]),
        )


class GaussianBuffer:
    """
    Temporal buffer for 4D Gaussian Splatting sequences.
    
    Maintains a fixed-size rolling buffer of Gaussian frames.
    Thread-safe for producer-consumer patterns.
    
    Usage:
        buffer = GaussianBuffer(max_frames=120)
        
        # Add frames
        buffer.add(positions, covariances, colors, opacities, frame_id=0)
        
        # Get frames
        latest = buffer.get_latest()
        sequence = buffer.get_sequence(start_frame=0, end_frame=30)
    """
    
    def __init__(self, max_frames: int = 120):
        """
        Initialize the buffer.
        
        Args:
            max_frames: Maximum number of frames to store
        """
        self.max_frames = max_frames
        self._frames: OrderedDict[int, GaussianData] = OrderedDict()
        self._lock = threading.Lock()
    
    def add(
        self,
        positions: np.ndarray,
        covariances: np.ndarray,
        colors: np.ndarray,
        opacities: np.ndarray,
        frame_id: int,
        timestamp: Optional[float] = None,
    ):
        """
        Add a frame to the buffer.
        
        Args:
            positions: [N, 3] Gaussian positions
            covariances: [N, 6] or [N, 3, 3] covariances
            colors: [N, 3] or [N, D] colors/SH coefficients
            opacities: [N, 1] opacities
            frame_id: Frame identifier
            timestamp: Optional timestamp (uses current time if not provided)
        """
        import time
        
        if timestamp is None:
            timestamp = time.time()
        
        data = GaussianData(
            frame_id=frame_id,
            timestamp=timestamp,
            positions=np.asarray(positions, dtype=np.float32),
            covariances=np.asarray(covariances, dtype=np.float32),
            colors=np.asarray(colors, dtype=np.float32),
            opacities=np.asarray(opacities, dtype=np.float32),
        )
        
        with self._lock:
            self._frames[frame_id] = data
            
            # Remove old frames if over capacity
            while len(self._frames) > self.max_frames:
                self._frames.popitem(last=False)
    
    def get(self, frame_id: int) -> Optional[GaussianData]:
        """Get a specific frame by ID."""
        with self._lock:
            return self._frames.get(frame_id)
    
    def get_latest(self) -> Optional[GaussianData]:
        """Get the most recent frame."""
        with self._lock:
            if not self._frames:
                return None
            return next(reversed(self._frames.values()))
    
    def get_sequence(
        self,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
    ) -> List[GaussianData]:
        """
        Get a sequence of frames.
        
        Args:
            start_frame: Start frame ID (inclusive, None for earliest)
            end_frame: End frame ID (inclusive, None for latest)
            
        Returns:
            List of GaussianData in order
        """
        with self._lock:
            frames = list(self._frames.values())
        
        if start_frame is not None:
            frames = [f for f in frames if f.frame_id >= start_frame]
        
        if end_frame is not None:
            frames = [f for f in frames if f.frame_id <= end_frame]
        
        return sorted(frames, key=lambda f: f.frame_id)
    
    def clear(self):
        """Clear all frames from buffer."""
        with self._lock:
            self._frames.clear()
    
    @property
    def size(self) -> int:
        """Current number of frames in buffer."""
        return len(self._frames)
    
    @property
    def frame_ids(self) -> List[int]:
        """List of frame IDs in buffer."""
        with self._lock:
            return list(self._frames.keys())


def export_gaussians(
    path: Union[str, Path],
    positions: np.ndarray,
    covariances: np.ndarray,
    colors: np.ndarray,
    opacities: np.ndarray,
    format: str = "ply",
) -> str:
    """
    Export Gaussian parameters to file.
    
    Args:
        path: Output file path
        positions: [N, 3] XYZ positions
        covariances: [N, 6] or [N, 3, 3] covariance matrices
        colors: [N, 3] RGB colors (0-1) or [N, D] SH coefficients
        opacities: [N, 1] opacity values
        format: Output format ("ply", "splat", "npz")
        
    Returns:
        Path to exported file
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Ensure correct suffix
    if format == "ply" and path.suffix != ".ply":
        path = path.with_suffix(".ply")
    elif format == "splat" and path.suffix != ".splat":
        path = path.with_suffix(".splat")
    elif format == "npz" and path.suffix != ".npz":
        path = path.with_suffix(".npz")
    
    # Validate inputs
    positions = np.asarray(positions, dtype=np.float32)
    covariances = np.asarray(covariances, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.float32)
    opacities = np.asarray(opacities, dtype=np.float32)
    
    n = positions.shape[0]
    assert positions.shape == (n, 3), f"Expected positions [N, 3], got {positions.shape}"
    assert opacities.shape[0] == n, f"Opacities count mismatch: {opacities.shape[0]} vs {n}"
    
    if format == "ply":
        _export_ply(path, positions, covariances, colors, opacities)
    elif format == "splat":
        _export_splat(path, positions, covariances, colors, opacities)
    elif format == "npz":
        _export_npz(path, positions, covariances, colors, opacities)
    else:
        raise ValueError(f"Unknown format: {format}")
    
    logger.info(f"Exported {n} Gaussians to {path}")
    return str(path)


def _export_ply(
    path: Path,
    positions: np.ndarray,
    covariances: np.ndarray,
    colors: np.ndarray,
    opacities: np.ndarray,
):
    """Export to PLY format compatible with Gaussian Splatting viewers."""
    n = positions.shape[0]
    
    # Flatten covariances if needed
    if covariances.ndim == 3:
        # [N, 3, 3] -> [N, 6] (upper triangular)
        cov_flat = np.zeros((n, 6), dtype=np.float32)
        cov_flat[:, 0] = covariances[:, 0, 0]
        cov_flat[:, 1] = covariances[:, 0, 1]
        cov_flat[:, 2] = covariances[:, 0, 2]
        cov_flat[:, 3] = covariances[:, 1, 1]
        cov_flat[:, 4] = covariances[:, 1, 2]
        cov_flat[:, 5] = covariances[:, 2, 2]
        covariances = cov_flat
    
    # Convert colors to uint8 if in [0, 1] range
    if colors.max() <= 1.0:
        colors_u8 = (colors * 255).clip(0, 255).astype(np.uint8)
    else:
        colors_u8 = colors.clip(0, 255).astype(np.uint8)
    
    # Flatten opacities
    if opacities.ndim == 2:
        opacities = opacities.ravel()
    
    # Write PLY file
    with open(path, 'wb') as f:
        # Header
        header = f"""ply
format binary_little_endian 1.0
element vertex {n}
property float x
property float y
property float z
property float cov_0
property float cov_1
property float cov_2
property float cov_3
property float cov_4
property float cov_5
property uchar red
property uchar green
property uchar blue
property float opacity
end_header
"""
        f.write(header.encode('ascii'))
        
        # Write data
        for i in range(n):
            # Position
            f.write(struct.pack('<fff', *positions[i]))
            # Covariance
            f.write(struct.pack('<ffffff', *covariances[i]))
            # Color
            f.write(struct.pack('<BBB', *colors_u8[i, :3]))
            # Opacity
            f.write(struct.pack('<f', opacities[i]))


def _export_splat(
    path: Path,
    positions: np.ndarray,
    covariances: np.ndarray,
    colors: np.ndarray,
    opacities: np.ndarray,
):
    """
    Export to .splat format (web-friendly binary format).
    
    Format per Gaussian (32 bytes):
        - position: 3 x float32 (12 bytes)
        - scale: 3 x float32 (12 bytes)  -- derived from covariance
        - color: 4 x uint8 (4 bytes, RGBA)
        - quaternion: 4 x uint8 (4 bytes, normalized)
    """
    n = positions.shape[0]
    
    # Extract scale and rotation from covariance
    scales, rotations = _decompose_covariance(covariances)
    
    # Convert colors to uint8 RGBA
    if colors.max() <= 1.0:
        colors_u8 = (colors * 255).clip(0, 255).astype(np.uint8)
    else:
        colors_u8 = colors.clip(0, 255).astype(np.uint8)
    
    # Add alpha channel from opacities
    if opacities.ndim == 2:
        opacities = opacities.ravel()
    alpha = (opacities * 255).clip(0, 255).astype(np.uint8)
    
    # Pad colors to 3 channels if needed
    if colors_u8.shape[1] < 3:
        colors_u8 = np.pad(colors_u8, ((0, 0), (0, 3 - colors_u8.shape[1])))
    
    # Normalize quaternions to uint8 (-1 to 1 -> 0 to 255)
    quat_u8 = ((rotations + 1) * 127.5).clip(0, 255).astype(np.uint8)
    
    # Write binary data
    with open(path, 'wb') as f:
        for i in range(n):
            # Position (12 bytes)
            f.write(struct.pack('<fff', *positions[i]))
            # Scale (12 bytes)
            f.write(struct.pack('<fff', *scales[i]))
            # RGBA (4 bytes)
            f.write(struct.pack('<BBBB', *colors_u8[i, :3], alpha[i]))
            # Quaternion (4 bytes)
            f.write(struct.pack('<BBBB', *quat_u8[i]))


def _export_npz(
    path: Path,
    positions: np.ndarray,
    covariances: np.ndarray,
    colors: np.ndarray,
    opacities: np.ndarray,
):
    """Export to NumPy compressed archive."""
    np.savez_compressed(
        path,
        positions=positions,
        covariances=covariances,
        colors=colors,
        opacities=opacities,
    )


def _decompose_covariance(covariances: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decompose covariance matrices into scale and rotation.
    
    Args:
        covariances: [N, 6] or [N, 3, 3]
        
    Returns:
        scales: [N, 3] scale factors
        rotations: [N, 4] quaternions
    """
    n = covariances.shape[0]
    
    # Convert [N, 6] to [N, 3, 3] if needed
    if covariances.ndim == 2 and covariances.shape[1] == 6:
        cov_mat = np.zeros((n, 3, 3), dtype=np.float32)
        cov_mat[:, 0, 0] = covariances[:, 0]
        cov_mat[:, 0, 1] = covariances[:, 1]
        cov_mat[:, 0, 2] = covariances[:, 2]
        cov_mat[:, 1, 0] = covariances[:, 1]
        cov_mat[:, 1, 1] = covariances[:, 3]
        cov_mat[:, 1, 2] = covariances[:, 4]
        cov_mat[:, 2, 0] = covariances[:, 2]
        cov_mat[:, 2, 1] = covariances[:, 4]
        cov_mat[:, 2, 2] = covariances[:, 5]
    else:
        cov_mat = covariances
    
    scales = np.zeros((n, 3), dtype=np.float32)
    rotations = np.zeros((n, 4), dtype=np.float32)
    
    for i in range(n):
        # Eigendecomposition: Σ = R @ S^2 @ R^T
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(cov_mat[i])
            
            # Scale is sqrt of eigenvalues
            scales[i] = np.sqrt(np.maximum(eigenvalues, 1e-10))
            
            # Rotation matrix to quaternion
            R = eigenvectors
            rotations[i] = _rotation_matrix_to_quaternion(R)
        except Exception:
            # Fallback to identity
            scales[i] = [1, 1, 1]
            rotations[i] = [1, 0, 0, 0]
    
    return scales, rotations


def _rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion [w, x, y, z]."""
    # Ensure proper rotation matrix (det = 1)
    if np.linalg.det(R) < 0:
        R = -R
    
    trace = np.trace(R)
    
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    
    q = np.array([w, x, y, z], dtype=np.float32)
    return q / np.linalg.norm(q)


def load_gaussians(path: Union[str, Path]) -> GaussianData:
    """
    Load Gaussian parameters from file.
    
    Args:
        path: Path to Gaussian file (.npz, .ply)
        
    Returns:
        GaussianData instance
    """
    path = Path(path)
    
    if path.suffix == ".npz":
        data = np.load(path)
        return GaussianData(
            frame_id=int(data.get("frame_id", 0)),
            timestamp=float(data.get("timestamp", 0)),
            positions=data["positions"],
            covariances=data["covariances"],
            colors=data["colors"],
            opacities=data["opacities"],
        )
    elif path.suffix == ".ply":
        return _load_ply(path)
    else:
        raise ValueError(f"Unknown format: {path.suffix}")


def _load_ply(path: Path) -> GaussianData:
    """Load Gaussians from PLY file."""
    # This is a simplified loader - real implementation would use plyfile
    try:
        from plyfile import PlyData
        
        plydata = PlyData.read(path)
        vertex = plydata['vertex']
        
        positions = np.stack([
            vertex['x'],
            vertex['y'],
            vertex['z'],
        ], axis=1)
        
        # Try to load covariances
        cov_names = ['cov_0', 'cov_1', 'cov_2', 'cov_3', 'cov_4', 'cov_5']
        if all(name in vertex.data.dtype.names for name in cov_names):
            covariances = np.stack([vertex[name] for name in cov_names], axis=1)
        else:
            covariances = np.eye(3).reshape(1, 3, 3).repeat(len(positions), axis=0)
        
        # Colors
        if 'red' in vertex.data.dtype.names:
            colors = np.stack([
                vertex['red'] / 255.0,
                vertex['green'] / 255.0,
                vertex['blue'] / 255.0,
            ], axis=1)
        else:
            colors = np.ones((len(positions), 3))
        
        # Opacity
        if 'opacity' in vertex.data.dtype.names:
            opacities = vertex['opacity'].reshape(-1, 1)
        else:
            opacities = np.ones((len(positions), 1))
        
        return GaussianData(
            frame_id=0,
            timestamp=0,
            positions=positions.astype(np.float32),
            covariances=covariances.astype(np.float32),
            colors=colors.astype(np.float32),
            opacities=opacities.astype(np.float32),
        )
        
    except ImportError:
        raise ImportError("plyfile required for PLY loading. Install with: pip install plyfile")


def merge_gaussians(*gaussian_data: GaussianData) -> GaussianData:
    """
    Merge multiple GaussianData into one.
    
    Useful for combining Gaussians from multiple frames or sources.
    """
    if not gaussian_data:
        raise ValueError("No data to merge")
    
    if len(gaussian_data) == 1:
        return gaussian_data[0]
    
    return GaussianData(
        frame_id=gaussian_data[0].frame_id,
        timestamp=gaussian_data[0].timestamp,
        positions=np.concatenate([g.positions for g in gaussian_data], axis=0),
        covariances=np.concatenate([g.covariances for g in gaussian_data], axis=0),
        colors=np.concatenate([g.colors for g in gaussian_data], axis=0),
        opacities=np.concatenate([g.opacities for g in gaussian_data], axis=0),
    )


if __name__ == "__main__":
    import tempfile
    
    # Test GaussianBuffer
    buffer = GaussianBuffer(max_frames=5)
    
    for i in range(10):
        n = 100
        buffer.add(
            positions=np.random.randn(n, 3).astype(np.float32),
            covariances=np.random.randn(n, 6).astype(np.float32),
            colors=np.random.rand(n, 3).astype(np.float32),
            opacities=np.random.rand(n, 1).astype(np.float32),
            frame_id=i,
        )
    
    print(f"Buffer size: {buffer.size}")
    print(f"Frame IDs: {buffer.frame_ids}")
    
    # Test export
    with tempfile.TemporaryDirectory() as tmpdir:
        positions = np.random.randn(50, 3).astype(np.float32)
        covariances = np.abs(np.random.randn(50, 6).astype(np.float32)) * 0.1
        colors = np.random.rand(50, 3).astype(np.float32)
        opacities = np.random.rand(50, 1).astype(np.float32)
        
        # Export to different formats
        ply_path = export_gaussians(
            f"{tmpdir}/test.ply",
            positions, covariances, colors, opacities,
            format="ply"
        )
        print(f"Exported to: {ply_path}")
        
        npz_path = export_gaussians(
            f"{tmpdir}/test.npz",
            positions, covariances, colors, opacities,
            format="npz"
        )
        print(f"Exported to: {npz_path}")
