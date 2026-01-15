"""
MLP classification head for binary drone/bird classification.
"""

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """
    MLP classification head.

    Architecture: Linear -> ReLU -> Dropout -> Linear -> ReLU -> Dropout -> Linear
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        num_classes: int = 1,  # Binary classification
        dropout: float = 0.3,
    ):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_dim) temporal embedding

        Returns:
            logits: (B, num_classes) raw logits (apply sigmoid for probabilities)
        """
        return self.mlp(x)


class ClassificationHeadWithAuxiliary(nn.Module):
    """
    Classification head with auxiliary outputs for intermediate supervision.

    Can optionally output frame-level predictions in addition to sequence-level.
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        num_classes: int = 1,
        dropout: float = 0.3,
        frame_dim: int = None,  # If provided, enables frame-level prediction
    ):
        super().__init__()

        self.frame_dim = frame_dim

        # Main classification head
        self.main_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        # Optional frame-level head
        if frame_dim is not None:
            self.frame_head = nn.Sequential(
                nn.Linear(frame_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )

    def forward(
        self,
        temporal_embedding: torch.Tensor,
        frame_embeddings: torch.Tensor = None
    ) -> dict:
        """
        Args:
            temporal_embedding: (B, input_dim) sequence-level embedding
            frame_embeddings: (B, T, frame_dim) optional frame-level embeddings

        Returns:
            Dictionary with:
                - 'logits': (B, num_classes) sequence-level logits
                - 'frame_logits': (B, T, num_classes) frame-level logits (if frame_embeddings provided)
        """
        outputs = {
            'logits': self.main_head(temporal_embedding)
        }

        if frame_embeddings is not None and self.frame_dim is not None:
            B, T, D = frame_embeddings.shape
            frame_flat = frame_embeddings.view(B * T, D)
            frame_logits = self.frame_head(frame_flat)
            outputs['frame_logits'] = frame_logits.view(B, T, -1)

        return outputs
