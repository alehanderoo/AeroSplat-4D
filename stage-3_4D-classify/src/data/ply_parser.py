"""
PLY file parser for 3D Gaussian Splatting reconstructions.

Expected PLY format from feed-forward reconstruction:
- vertex properties: x, y, z (position)
- scale_0, scale_1, scale_2 (log-scale)
- rot_0, rot_1, rot_2, rot_3 (quaternion wxyz or xyzw - verify!)
- opacity (logit form)
- f_dc_0, f_dc_1, f_dc_2 (DC spherical harmonics)
- f_rest_0 ... f_rest_44 (higher-order SH coefficients)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import numpy as np
import torch

try:
    from plyfile import PlyData
    PLYFILE_AVAILABLE = True
except ImportError:
    PLYFILE_AVAILABLE = False


@dataclass
class GaussianCloud:
    """Container for parsed Gaussian attributes."""
    positions: torch.Tensor      # (N, 3) - xyz coordinates
    scales: torch.Tensor         # (N, 3) - scale in each axis (exp applied)
    rotations: torch.Tensor      # (N, 4) - quaternion (normalized)
    opacities: torch.Tensor      # (N, 1) - opacity [0,1] (sigmoid applied)
    sh_coeffs: torch.Tensor      # (N, 48) - spherical harmonics

    # Derived features (computed lazily)
    _covariance: Optional[torch.Tensor] = field(default=None, repr=False)
    _eigenvalues: Optional[torch.Tensor] = field(default=None, repr=False)
    _eigenvectors: Optional[torch.Tensor] = field(default=None, repr=False)
    _sh_band_energies: Optional[torch.Tensor] = field(default=None, repr=False)

    @property
    def num_gaussians(self) -> int:
        return self.positions.shape[0]

    @property
    def device(self) -> torch.device:
        return self.positions.device

    def to(self, device: torch.device) -> 'GaussianCloud':
        """Move all tensors to specified device."""
        return GaussianCloud(
            positions=self.positions.to(device),
            scales=self.scales.to(device),
            rotations=self.rotations.to(device),
            opacities=self.opacities.to(device),
            sh_coeffs=self.sh_coeffs.to(device),
        )

    @property
    def covariance(self) -> torch.Tensor:
        """Compute 3x3 covariance matrices from scale and rotation."""
        if self._covariance is None:
            self._covariance = self._compute_covariance()
        return self._covariance

    @property
    def eigenvalues(self) -> torch.Tensor:
        """Eigenvalues of covariance (rotation-invariant shape descriptor)."""
        if self._eigenvalues is None:
            self._compute_eigen_decomposition()
        return self._eigenvalues

    @property
    def eigenvectors(self) -> torch.Tensor:
        """Principal axes (rotation-equivariant)."""
        if self._eigenvectors is None:
            self._compute_eigen_decomposition()
        return self._eigenvectors

    @property
    def sh_band_energies(self) -> torch.Tensor:
        """SH band energies (rotation-invariant)."""
        if self._sh_band_energies is None:
            self._sh_band_energies = self._compute_sh_band_energies()
        return self._sh_band_energies

    def _compute_covariance(self) -> torch.Tensor:
        """Σ = R @ S @ S^T @ R^T where R is rotation matrix, S is diagonal scale."""
        # Convert quaternion to rotation matrix
        R = quaternion_to_rotation_matrix(self.rotations)  # (N, 3, 3)

        # Scale matrix (diagonal)
        S = torch.diag_embed(self.scales)  # (N, 3, 3)

        # Covariance
        RS = R @ S
        cov = RS @ RS.transpose(-1, -2)
        return cov  # (N, 3, 3)

    def _compute_eigen_decomposition(self):
        """Compute eigenvalues and eigenvectors of covariance matrices."""
        # torch.linalg.eigh for symmetric matrices (covariance is symmetric)
        eigenvalues, eigenvectors = torch.linalg.eigh(self.covariance)

        # Sort in descending order
        idx = torch.argsort(eigenvalues, dim=-1, descending=True)
        self._eigenvalues = torch.gather(eigenvalues, -1, idx)
        self._eigenvectors = torch.gather(
            eigenvectors, -1,
            idx.unsqueeze(-2).expand(-1, 3, -1)
        )

    def _compute_sh_band_energies(self) -> torch.Tensor:
        """
        Compute rotation-invariant SH band energies.
        E_l = Σ_{m=-l}^{l} |c_l^m|² for each degree l ∈ {0,1,2,3}

        SH coefficients layout (per channel):
        - l=0: 1 coefficient (index 0)
        - l=1: 3 coefficients (indices 1-3)
        - l=2: 5 coefficients (indices 4-8)
        - l=3: 7 coefficients (indices 9-15)
        Total: 16 per channel × 3 channels = 48
        """
        # Reshape to (N, 3, 16) for per-channel processing
        sh = self.sh_coeffs.view(-1, 3, 16)

        # Band indices
        l0 = sh[:, :, 0:1]           # (N, 3, 1)
        l1 = sh[:, :, 1:4]           # (N, 3, 3)
        l2 = sh[:, :, 4:9]           # (N, 3, 5)
        l3 = sh[:, :, 9:16]          # (N, 3, 7)

        # Compute band energies (sum of squared coefficients)
        e0 = (l0 ** 2).sum(dim=-1)   # (N, 3)
        e1 = (l1 ** 2).sum(dim=-1)   # (N, 3)
        e2 = (l2 ** 2).sum(dim=-1)   # (N, 3)
        e3 = (l3 ** 2).sum(dim=-1)   # (N, 3)

        # Concatenate: (N, 12) - 4 bands × 3 channels
        band_energies = torch.cat([e0, e1, e2, e3], dim=-1)
        return band_energies


def parse_ply(filepath: Path, device: str = 'cpu') -> GaussianCloud:
    """
    Parse a PLY file containing 3D Gaussian Splatting data.

    Args:
        filepath: Path to .ply file
        device: Target device for tensors

    Returns:
        GaussianCloud with all attributes
    """
    if not PLYFILE_AVAILABLE:
        raise ImportError("plyfile package required. Install with: pip install plyfile")

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"PLY file not found: {filepath}")

    plydata = PlyData.read(str(filepath))
    vertex = plydata['vertex']

    # Extract positions
    positions = np.stack([
        vertex['x'], vertex['y'], vertex['z']
    ], axis=-1).astype(np.float32)

    # Extract scales (stored as log-scale, need exp)
    try:
        scales_log = np.stack([
            vertex['scale_0'], vertex['scale_1'], vertex['scale_2']
        ], axis=-1).astype(np.float32)
        scales = np.exp(scales_log)
    except ValueError:
        # Try alternative naming
        try:
            scales_log = np.stack([
                vertex['scaling_0'], vertex['scaling_1'], vertex['scaling_2']
            ], axis=-1).astype(np.float32)
            scales = np.exp(scales_log)
        except ValueError:
            # Default to uniform scale
            scales = np.ones((positions.shape[0], 3), dtype=np.float32)

    # Extract rotations (quaternion - verify order from your reconstruction!)
    try:
        rotations = np.stack([
            vertex['rot_0'], vertex['rot_1'], vertex['rot_2'], vertex['rot_3']
        ], axis=-1).astype(np.float32)
    except ValueError:
        try:
            rotations = np.stack([
                vertex['rotation_0'], vertex['rotation_1'],
                vertex['rotation_2'], vertex['rotation_3']
            ], axis=-1).astype(np.float32)
        except ValueError:
            # Default to identity rotation
            rotations = np.zeros((positions.shape[0], 4), dtype=np.float32)
            rotations[:, 0] = 1.0  # w=1 for identity

    # Normalize quaternions
    rotations = rotations / (np.linalg.norm(rotations, axis=-1, keepdims=True) + 1e-8)

    # Extract opacity (stored as logit, need sigmoid)
    try:
        opacity_logit = vertex['opacity'].astype(np.float32)
        opacities = 1.0 / (1.0 + np.exp(-opacity_logit))
    except ValueError:
        opacities = np.ones(positions.shape[0], dtype=np.float32)
    opacities = opacities[:, np.newaxis]

    # Extract spherical harmonics
    try:
        sh_dc = np.stack([
            vertex['f_dc_0'], vertex['f_dc_1'], vertex['f_dc_2']
        ], axis=-1).astype(np.float32)

        # Higher order SH (indices 0-44 for f_rest)
        sh_rest_names = [f'f_rest_{i}' for i in range(45)]
        sh_rest_list = []
        for name in sh_rest_names:
            if name in vertex.dtype.names:
                sh_rest_list.append(vertex[name])

        if len(sh_rest_list) == 45:
            sh_rest = np.stack(sh_rest_list, axis=-1).astype(np.float32)
            # Full SH up to degree 3
            sh_rest = sh_rest.reshape(-1, 15, 3).transpose(0, 2, 1)  # (N, 3, 15)
            sh_dc = sh_dc.reshape(-1, 3, 1)  # (N, 3, 1)
            sh_full = np.concatenate([sh_dc, sh_rest], axis=-1)  # (N, 3, 16)
            sh_coeffs = sh_full.reshape(-1, 48)  # (N, 48)
        else:
            # Handle partial SH (pad with zeros)
            sh_coeffs = np.zeros((positions.shape[0], 48), dtype=np.float32)
            sh_coeffs[:, :3] = sh_dc  # DC term only
    except ValueError:
        # No SH data
        sh_coeffs = np.zeros((positions.shape[0], 48), dtype=np.float32)

    # Convert to tensors
    return GaussianCloud(
        positions=torch.from_numpy(positions).to(device),
        scales=torch.from_numpy(scales).to(device),
        rotations=torch.from_numpy(rotations).to(device),
        opacities=torch.from_numpy(opacities).to(device),
        sh_coeffs=torch.from_numpy(sh_coeffs).to(device),
    )


def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion to rotation matrix.

    Args:
        q: Quaternion tensor (N, 4) in wxyz order

    Returns:
        Rotation matrices (N, 3, 3)
    """
    # Normalize
    q = q / (q.norm(dim=-1, keepdim=True) + 1e-8)

    w, x, y, z = q.unbind(-1)

    # Rotation matrix from quaternion
    R = torch.stack([
        1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y,
        2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x,
        2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y
    ], dim=-1).view(-1, 3, 3)

    return R


def create_synthetic_gaussian_cloud(
    num_gaussians: int = 1000,
    device: str = 'cpu',
    seed: int = None,
) -> GaussianCloud:
    """
    Create a synthetic Gaussian cloud for testing.

    Args:
        num_gaussians: Number of Gaussians to generate
        device: Target device
        seed: Random seed for reproducibility

    Returns:
        GaussianCloud with random attributes
    """
    if seed is not None:
        torch.manual_seed(seed)

    positions = torch.randn(num_gaussians, 3, device=device)
    scales = torch.rand(num_gaussians, 3, device=device) * 0.5 + 0.1

    # Random quaternions (normalized)
    rotations = torch.randn(num_gaussians, 4, device=device)
    rotations = rotations / rotations.norm(dim=-1, keepdim=True)

    opacities = torch.rand(num_gaussians, 1, device=device)
    sh_coeffs = torch.randn(num_gaussians, 48, device=device) * 0.1

    return GaussianCloud(
        positions=positions,
        scales=scales,
        rotations=rotations,
        opacities=opacities,
        sh_coeffs=sh_coeffs,
    )
