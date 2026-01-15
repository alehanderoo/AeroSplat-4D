"""
Extract and prepare features from GaussianCloud for the network.

Separates equivariant (Type-1) and invariant (scalar) features.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import torch

from .ply_parser import GaussianCloud


@dataclass
class GaussianFeatures:
    """Prepared features for VN-Transformer."""
    # Type-1 (Equivariant) features
    positions: torch.Tensor        # (M, 3) - Gaussian positions
    eigenvectors: torch.Tensor     # (M, 3, 3) - Principal axes of covariance

    # Scalar (Invariant) features
    eigenvalues: torch.Tensor      # (M, 3) - Shape descriptors
    opacities: torch.Tensor        # (M, 1) - Transparency
    sh_band_energies: torch.Tensor # (M, 12) - Color invariants

    @property
    def num_gaussians(self) -> int:
        return self.positions.shape[0]

    @property
    def scalar_dim(self) -> int:
        """Total dimension of scalar features."""
        return 3 + 1 + 12  # eigenvalues + opacity + sh_band_energies = 16


class GaussianFeatureExtractor:
    """
    Extract VN-compatible features from GaussianCloud.
    """

    def __init__(
        self,
        center_positions: bool = True,
        normalize_positions: bool = True,
        scale_eigenvalues: bool = True,
    ):
        """
        Args:
            center_positions: Subtract centroid from positions
            normalize_positions: Scale to unit sphere
            scale_eigenvalues: Log-scale eigenvalues for numerical stability
        """
        self.center_positions = center_positions
        self.normalize_positions = normalize_positions
        self.scale_eigenvalues = scale_eigenvalues

    def extract(self, cloud: GaussianCloud) -> GaussianFeatures:
        """Extract features from a single GaussianCloud."""
        positions = cloud.positions.clone()

        # Center positions (translation invariance)
        if self.center_positions:
            centroid = positions.mean(dim=0, keepdim=True)
            positions = positions - centroid

        # Normalize to unit sphere
        if self.normalize_positions:
            max_dist = positions.norm(dim=-1).max() + 1e-8
            positions = positions / max_dist

        # Get eigendecomposition (lazy computed in GaussianCloud)
        eigenvalues = cloud.eigenvalues  # (M, 3)
        eigenvectors = cloud.eigenvectors  # (M, 3, 3)

        # Log-scale eigenvalues for better numerical properties
        if self.scale_eigenvalues:
            eigenvalues = torch.log(eigenvalues.clamp(min=1e-8))

        # Get other invariants
        opacities = cloud.opacities  # (M, 1)
        sh_band_energies = cloud.sh_band_energies  # (M, 12)

        return GaussianFeatures(
            positions=positions,
            eigenvectors=eigenvectors,
            eigenvalues=eigenvalues,
            opacities=opacities,
            sh_band_energies=sh_band_energies,
        )

    def extract_batch(
        self,
        clouds: List[GaussianCloud],
        max_gaussians: Optional[int] = None,
        subsample_strategy: str = 'random',
    ) -> Dict[str, torch.Tensor]:
        """
        Extract features from a sequence of clouds.

        Args:
            clouds: List of T GaussianClouds
            max_gaussians: Maximum Gaussians per frame
            subsample_strategy: How to subsample if needed

        Returns:
            Dictionary with batched tensors (T, M, ...)
        """
        features_list = []

        for cloud in clouds:
            # Subsample if needed
            if max_gaussians and cloud.num_gaussians > max_gaussians:
                cloud = self._subsample(cloud, max_gaussians, subsample_strategy)

            features = self.extract(cloud)
            features_list.append(features)

        # Stack into sequence tensors
        # Need to pad to same M across frames
        max_M = max(f.num_gaussians for f in features_list)
        T = len(features_list)

        # Get device from first cloud
        device = features_list[0].positions.device

        # Initialize tensors
        positions = torch.zeros(T, max_M, 3, device=device)
        eigenvectors = torch.zeros(T, max_M, 3, 3, device=device)
        scalars = torch.zeros(T, max_M, features_list[0].scalar_dim, device=device)
        mask = torch.zeros(T, max_M, dtype=torch.bool, device=device)

        for t, f in enumerate(features_list):
            M = f.num_gaussians
            positions[t, :M] = f.positions
            eigenvectors[t, :M] = f.eigenvectors
            scalars[t, :M] = torch.cat([
                f.eigenvalues,
                f.opacities,
                f.sh_band_energies
            ], dim=-1)
            mask[t, :M] = True

        return {
            'positions': positions,
            'eigenvectors': eigenvectors,
            'scalars': scalars,
            'mask': mask,
        }

    def _subsample(
        self,
        cloud: GaussianCloud,
        n: int,
        strategy: str
    ) -> GaussianCloud:
        """Subsample cloud to n Gaussians."""
        M = cloud.num_gaussians

        if strategy == 'random':
            indices = torch.randperm(M, device=cloud.device)[:n]
        elif strategy == 'fps':
            # Farthest Point Sampling
            indices = self._fps_subsample(cloud.positions, n)
        elif strategy == 'grid':
            # Voxel grid subsampling
            indices = self._grid_subsample(cloud.positions, n)
        elif strategy == 'importance':
            # Sample based on opacity (keep high-opacity Gaussians)
            indices = self._importance_subsample(cloud, n)
        else:
            raise ValueError(f"Unknown subsample strategy: {strategy}")

        return GaussianCloud(
            positions=cloud.positions[indices],
            scales=cloud.scales[indices],
            rotations=cloud.rotations[indices],
            opacities=cloud.opacities[indices],
            sh_coeffs=cloud.sh_coeffs[indices],
        )

    def _fps_subsample(self, positions: torch.Tensor, n: int) -> torch.Tensor:
        """Farthest Point Sampling."""
        device = positions.device
        N = positions.shape[0]

        # Start with random point
        indices = [torch.randint(0, N, (1,), device=device)]
        distances = torch.full((N,), float('inf'), device=device)

        for _ in range(n - 1):
            # Update distances
            last_idx = indices[-1]
            dist_to_last = ((positions - positions[last_idx]) ** 2).sum(dim=-1)
            distances = torch.min(distances, dist_to_last)

            # Select farthest point
            farthest = distances.argmax()
            indices.append(farthest.unsqueeze(0))

        return torch.cat(indices)

    def _grid_subsample(self, positions: torch.Tensor, n: int) -> torch.Tensor:
        """Voxel grid subsampling - keep one point per voxel."""
        device = positions.device
        M = positions.shape[0]

        pmin = positions.min(dim=0).values
        pmax = positions.max(dim=0).values
        extent = (pmax - pmin).max()

        # Estimate voxel size to get approximately n points
        voxel_size = extent / (n ** (1/3))

        # Quantize positions
        grid_coords = ((positions - pmin) / (voxel_size + 1e-8)).long()

        # Hash voxels
        hash_vals = (grid_coords[:, 0] * 73856093) ^ (grid_coords[:, 1] * 19349663) ^ (grid_coords[:, 2] * 83492791)

        # Get unique voxels
        unique_hashes, inverse_indices = torch.unique(hash_vals, return_inverse=True)

        # Keep first point in each voxel (up to n)
        indices = []
        for h in unique_hashes[:n]:
            idx = (hash_vals == h).nonzero(as_tuple=True)[0][0]
            indices.append(idx)

        if len(indices) < n:
            # If we got fewer points, pad with random
            remaining = torch.randperm(M, device=device)
            for idx in remaining:
                if idx not in indices:
                    indices.append(idx)
                if len(indices) >= n:
                    break

        return torch.stack(indices[:n]) if indices else torch.arange(n, device=device)

    def _importance_subsample(self, cloud: GaussianCloud, n: int) -> torch.Tensor:
        """Sample based on opacity (importance sampling)."""
        # Higher opacity = more important
        weights = cloud.opacities.squeeze(-1)
        weights = weights / weights.sum()

        indices = torch.multinomial(weights, n, replacement=False)
        return indices


class FastFeatureExtractor:
    """
    Faster feature extractor that skips eigenvector computation.

    Use when eigenvectors are not needed or pre-computed.
    """

    def __init__(
        self,
        center_positions: bool = True,
        normalize_positions: bool = True,
    ):
        self.center_positions = center_positions
        self.normalize_positions = normalize_positions

    def extract(self, cloud: GaussianCloud) -> Dict[str, torch.Tensor]:
        """Extract minimal features."""
        positions = cloud.positions.clone()

        if self.center_positions:
            centroid = positions.mean(dim=0, keepdim=True)
            positions = positions - centroid

        if self.normalize_positions:
            max_dist = positions.norm(dim=-1).max() + 1e-8
            positions = positions / max_dist

        # Scalar features without eigendecomposition
        # Use scale directly as shape descriptor (approximate eigenvalues)
        scales_log = torch.log(cloud.scales.clamp(min=1e-8))

        scalars = torch.cat([
            scales_log,              # (M, 3)
            cloud.opacities,         # (M, 1)
            cloud.sh_band_energies,  # (M, 12)
        ], dim=-1)  # (M, 16)

        return {
            'positions': positions,
            'scalars': scalars,
        }

    def extract_batch(
        self,
        clouds: List[GaussianCloud],
        max_gaussians: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Extract features from sequence."""
        features_list = [self.extract(c) for c in clouds]

        max_M = max(f['positions'].shape[0] for f in features_list)
        T = len(features_list)
        device = features_list[0]['positions'].device

        positions = torch.zeros(T, max_M, 3, device=device)
        scalars = torch.zeros(T, max_M, 16, device=device)
        mask = torch.zeros(T, max_M, dtype=torch.bool, device=device)

        for t, f in enumerate(features_list):
            M = f['positions'].shape[0]
            if max_gaussians and M > max_gaussians:
                indices = torch.randperm(M, device=device)[:max_gaussians]
                positions[t, :max_gaussians] = f['positions'][indices]
                scalars[t, :max_gaussians] = f['scalars'][indices]
                mask[t, :max_gaussians] = True
            else:
                positions[t, :M] = f['positions']
                scalars[t, :M] = f['scalars']
                mask[t, :M] = True

        return {
            'positions': positions,
            'scalars': scalars,
            'mask': mask,
        }
