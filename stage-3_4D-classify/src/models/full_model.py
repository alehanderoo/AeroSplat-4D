"""
Complete 2-stage 4D Gaussian classification model.

Combines:
- Stage 1: VN-Transformer for rotation-invariant spatial encoding
- Stage 2: Mamba for temporal dynamics modeling
- Classification head: MLP for final prediction
"""

import torch
import torch.nn as nn
from typing import Dict, Optional

from .stage1_spatial.vn_transformer import GaussianSpatialEncoder
from .stage1_spatial.feature_preparation import GaussianFeaturePreparation
from .stage2_temporal.mamba_temporal import TemporalMambaEncoder
from .classification_head import ClassificationHead


class Gaussian4DClassifier(nn.Module):
    """
    Full 2-stage architecture for 4D Gaussian classification.

    Stage 1: VN-Transformer for rotation-invariant spatial encoding
    Stage 2: Mamba for temporal dynamics modeling
    Head: MLP for classification

    Architecture Overview:
    ---------------------
    Input: (B, T, M, features) - Gaussian sequences
           ↓
    Stage 1: Per-frame VN-Transformer processing
           → (B, T, spatial_output_dim) frame embeddings
           ↓
    Stage 2: Temporal Mamba encoder
           → (B, temporal_hidden_dim) sequence embedding
           ↓
    Classification Head: MLP
           → (B, num_classes) logits
    """

    def __init__(
        self,
        # Stage 1 config
        spatial_dim: int = 128,
        spatial_depth: int = 4,
        spatial_heads: int = 8,
        spatial_dim_feat: int = 64,
        spatial_output_dim: int = 256,
        spatial_bias_epsilon: float = 1e-6,

        # Stage 2 config
        temporal_hidden_dim: int = 256,
        temporal_num_layers: int = 4,
        temporal_d_state: int = 16,
        temporal_bidirectional: bool = False,
        temporal_aggregation: str = 'mean',

        # Classification config
        num_classes: int = 1,
        dropout: float = 0.1,

        # Feature config
        use_eigenvectors: bool = True,
    ):
        super().__init__()

        self.spatial_output_dim = spatial_output_dim
        self.temporal_hidden_dim = temporal_hidden_dim

        # Feature preparation
        self.feature_prep = GaussianFeaturePreparation(
            use_eigenvectors=use_eigenvectors,
            use_positions=True,
        )

        # Stage 1: Spatial encoder
        self.spatial_encoder = GaussianSpatialEncoder(
            dim=spatial_dim,
            depth=spatial_depth,
            heads=spatial_heads,
            dim_feat=spatial_dim_feat,
            dropout=dropout,
            output_dim=spatial_output_dim,
            bias_epsilon=spatial_bias_epsilon,
        )

        # Stage 2: Temporal encoder
        self.temporal_encoder = TemporalMambaEncoder(
            input_dim=spatial_output_dim,
            hidden_dim=temporal_hidden_dim,
            output_dim=temporal_hidden_dim,
            num_layers=temporal_num_layers,
            d_state=temporal_d_state,
            dropout=dropout,
            bidirectional=temporal_bidirectional,
            aggregation=temporal_aggregation,
        )

        # Classification head
        self.classifier = ClassificationHead(
            input_dim=temporal_hidden_dim,
            hidden_dim=temporal_hidden_dim // 2,
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(
        self,
        positions: torch.Tensor,
        eigenvectors: torch.Tensor,
        scalars: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through full model.

        Args:
            positions: (B, T, M, 3) Gaussian positions per frame
            eigenvectors: (B, T, M, 3, 3) Principal axes per frame
            scalars: (B, T, M, D_scalar) Scalar features per frame
            mask: (B, T, M) Boolean mask for valid Gaussians

        Returns:
            Dictionary containing:
                - 'logits': (B, num_classes) classification logits
                - 'probabilities': (B, num_classes) sigmoid probabilities
                - 'frame_embeddings': (B, T, D) intermediate frame embeddings
                - 'temporal_embedding': (B, D) final temporal embedding
        """
        B, T, M, _ = positions.shape

        # Stage 1: Process each frame through spatial encoder
        frame_embeddings = []
        for t in range(T):
            # Prepare features for this frame
            coors_t, scalars_t = self.feature_prep(
                positions[:, t],
                eigenvectors[:, t],
                scalars[:, t],
            )

            mask_t = mask[:, t] if mask is not None else None

            # Encode frame
            emb_t = self.spatial_encoder(coors_t, scalars_t, mask_t)
            frame_embeddings.append(emb_t)

        # Stack frame embeddings: (B, T, D)
        frame_embeddings = torch.stack(frame_embeddings, dim=1)

        # Stage 2: Temporal encoding
        # Create frame-level mask if needed
        frame_mask = None
        if mask is not None:
            frame_mask = mask.any(dim=-1)  # (B, T)

        temporal_embedding = self.temporal_encoder(frame_embeddings, frame_mask)

        # Classification
        logits = self.classifier(temporal_embedding)
        probabilities = torch.sigmoid(logits)

        return {
            'logits': logits,
            'probabilities': probabilities,
            'frame_embeddings': frame_embeddings,
            'temporal_embedding': temporal_embedding,
        }

    def predict(self, *args, **kwargs) -> torch.Tensor:
        """Convenience method returning only class probabilities."""
        return self.forward(*args, **kwargs)['probabilities']

    def forward_efficient(
        self,
        positions: torch.Tensor,
        scalars: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Memory-efficient forward pass without eigenvectors.

        Uses only positions as spatial features, avoiding eigenvector computation.

        Args:
            positions: (B, T, M, 3) Gaussian positions per frame
            scalars: (B, T, M, D_scalar) Scalar features per frame
            mask: (B, T, M) Boolean mask for valid Gaussians

        Returns:
            Same as forward()
        """
        B, T, M, _ = positions.shape

        # Stage 1: Process each frame
        frame_embeddings = []
        for t in range(T):
            pos_t = positions[:, t]
            scalar_t = scalars[:, t]
            mask_t = mask[:, t] if mask is not None else None

            # Use positions directly (no feature preparation needed)
            emb_t = self.spatial_encoder(pos_t, scalar_t, mask_t)
            frame_embeddings.append(emb_t)

        frame_embeddings = torch.stack(frame_embeddings, dim=1)

        # Stage 2: Temporal encoding
        frame_mask = mask.any(dim=-1) if mask is not None else None
        temporal_embedding = self.temporal_encoder(frame_embeddings, frame_mask)

        # Classification
        logits = self.classifier(temporal_embedding)
        probabilities = torch.sigmoid(logits)

        return {
            'logits': logits,
            'probabilities': probabilities,
            'frame_embeddings': frame_embeddings,
            'temporal_embedding': temporal_embedding,
        }


class Gaussian4DClassifierLite(nn.Module):
    """
    Lightweight version of Gaussian4DClassifier.

    Uses simpler architecture for faster inference with reduced memory.
    Suitable for deployment or when full VN-Transformer is too heavy.
    """

    def __init__(
        self,
        # Spatial config
        spatial_dim: int = 64,
        spatial_depth: int = 2,
        spatial_heads: int = 4,
        spatial_output_dim: int = 128,

        # Temporal config
        temporal_hidden_dim: int = 128,
        temporal_num_layers: int = 2,

        # General
        num_classes: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Simpler spatial encoder (still uses VN-Transformer but smaller)
        self.spatial_encoder = GaussianSpatialEncoder(
            dim=spatial_dim,
            depth=spatial_depth,
            heads=spatial_heads,
            dim_feat=32,
            dropout=dropout,
            output_dim=spatial_output_dim,
        )

        # Temporal encoder
        self.temporal_encoder = TemporalMambaEncoder(
            input_dim=spatial_output_dim,
            hidden_dim=temporal_hidden_dim,
            output_dim=temporal_hidden_dim,
            num_layers=temporal_num_layers,
            dropout=dropout,
            use_mixer_model=False,  # Use simpler stacked blocks
        )

        # Classification head
        self.classifier = ClassificationHead(
            input_dim=temporal_hidden_dim,
            hidden_dim=temporal_hidden_dim // 2,
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(
        self,
        positions: torch.Tensor,
        scalars: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            positions: (B, T, M, 3) Gaussian positions
            scalars: (B, T, M, D_scalar) Scalar features
            mask: (B, T, M) Validity mask

        Returns:
            Dictionary with logits, probabilities, embeddings
        """
        B, T, M, _ = positions.shape

        # Process frames
        frame_embeddings = []
        for t in range(T):
            mask_t = mask[:, t] if mask is not None else None
            emb_t = self.spatial_encoder(positions[:, t], scalars[:, t], mask_t)
            frame_embeddings.append(emb_t)

        frame_embeddings = torch.stack(frame_embeddings, dim=1)

        # Temporal encoding
        frame_mask = mask.any(dim=-1) if mask is not None else None
        temporal_embedding = self.temporal_encoder(frame_embeddings, frame_mask)

        # Classification
        logits = self.classifier(temporal_embedding)
        probabilities = torch.sigmoid(logits)

        return {
            'logits': logits,
            'probabilities': probabilities,
            'frame_embeddings': frame_embeddings,
            'temporal_embedding': temporal_embedding,
        }
