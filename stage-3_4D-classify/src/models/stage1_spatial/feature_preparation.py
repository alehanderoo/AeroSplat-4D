"""
Prepare Gaussian features for VN-Transformer input.

Handles the separation of features into:
- Type-1 (equivariant): positions, principal axes
- Scalars (invariant): opacity, eigenvalues, SH band energies
"""

import torch
import torch.nn as nn
from typing import Tuple


class GaussianFeaturePreparation(nn.Module):
    """
    Prepare Gaussian attributes for VN-Transformer.

    Separates features into:
    - Type-1 (equivariant): positions, principal axes
    - Scalars (invariant): opacity, eigenvalues, SH band energies
    """

    def __init__(
        self,
        use_eigenvectors: bool = True,
        use_positions: bool = True,
        scalar_dim: int = 16,  # eigenvals(3) + opacity(1) + sh_bands(12)
    ):
        super().__init__()

        self.use_eigenvectors = use_eigenvectors
        self.use_positions = use_positions

        # Number of 3D vector features per Gaussian
        # positions (1) + eigenvectors (3) = 4 vectors max
        self.num_vectors = int(use_positions) + 3 * int(use_eigenvectors)

        # Project multiple vectors to single coordinate representation
        if self.num_vectors > 1:
            self.vector_compress = nn.Linear(self.num_vectors, 1)

    def forward(
        self,
        positions: torch.Tensor,
        eigenvectors: torch.Tensor,
        scalars: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            positions: (B, M, 3) Gaussian positions
            eigenvectors: (B, M, 3, 3) Principal axes (columns are eigenvectors)
            scalars: (B, M, D_scalar) Scalar features

        Returns:
            coors: (B, M, 3) Combined equivariant coordinates
            feats: (B, M, D_scalar) Scalar features (unchanged)
        """
        vectors = []

        if self.use_positions:
            vectors.append(positions)  # (B, M, 3)

        if self.use_eigenvectors:
            # Extract the 3 principal axes
            for i in range(3):
                vectors.append(eigenvectors[..., i])  # (B, M, 3)

        if len(vectors) == 1:
            coors = vectors[0]
        else:
            # Stack and compress
            stacked = torch.stack(vectors, dim=-1)  # (B, M, 3, num_vectors)
            coors = self.vector_compress(stacked).squeeze(-1)  # (B, M, 3)

        return coors, scalars


class GaussianFeaturePreparationV2(nn.Module):
    """
    Alternative feature preparation using attention-based fusion.

    Uses attention to combine position and eigenvector information.
    """

    def __init__(
        self,
        use_eigenvectors: bool = True,
        use_positions: bool = True,
        hidden_dim: int = 64,
    ):
        super().__init__()

        self.use_eigenvectors = use_eigenvectors
        self.use_positions = use_positions

        # Number of input vectors
        self.num_vectors = int(use_positions) + 3 * int(use_eigenvectors)

        if self.num_vectors > 1:
            # Attention-based fusion
            self.query = nn.Parameter(torch.randn(1, 1, 3))
            self.key_proj = nn.Linear(3, hidden_dim)
            self.value_proj = nn.Linear(3, 3)
            self.scale = hidden_dim ** -0.5

    def forward(
        self,
        positions: torch.Tensor,
        eigenvectors: torch.Tensor,
        scalars: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            positions: (B, M, 3) Gaussian positions
            eigenvectors: (B, M, 3, 3) Principal axes
            scalars: (B, M, D_scalar) Scalar features

        Returns:
            coors: (B, M, 3) Combined equivariant coordinates
            feats: (B, M, D_scalar) Scalar features
        """
        vectors = []

        if self.use_positions:
            vectors.append(positions)

        if self.use_eigenvectors:
            for i in range(3):
                vectors.append(eigenvectors[..., i])

        if len(vectors) == 1:
            return vectors[0], scalars

        # Stack vectors: (B, M, num_vectors, 3)
        stacked = torch.stack(vectors, dim=2)
        B, M, N, _ = stacked.shape

        # Compute attention scores
        # Query: (1, 1, 3) -> (B, M, 1, hidden_dim) via broadcasting
        query = self.query.expand(B, M, -1)  # (B, M, 3)
        keys = self.key_proj(stacked)  # (B, M, N, hidden_dim)

        # Attention: query @ keys.T
        query_expanded = query.unsqueeze(2)  # (B, M, 1, 3)
        # Use simple dot product for equivariance
        attn_scores = (query_expanded * stacked).sum(dim=-1)  # (B, M, N)
        attn_weights = torch.softmax(attn_scores * self.scale, dim=-1)  # (B, M, N)

        # Weighted sum of vectors
        coors = torch.einsum('bmn,bmnd->bmd', attn_weights, stacked)  # (B, M, 3)

        return coors, scalars
