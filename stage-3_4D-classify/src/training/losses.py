"""
Loss functions for training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryFocalLoss(nn.Module):
    """
    Focal loss for binary classification.
    Handles class imbalance by down-weighting easy examples.

    FL(p) = -α(1-p)^γ log(p)  for positive class
    FL(p) = -(1-α)p^γ log(1-p)  for negative class
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        """
        Args:
            alpha: Weighting factor for positive class (use > 0.5 if positive rare)
            gamma: Focusing parameter (higher = more focus on hard examples)
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, 1) raw logits
            targets: (B,) binary labels {0, 1}
        """
        probs = torch.sigmoid(logits).squeeze(-1)

        # Focal weights
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma

        # Alpha weighting
        alpha_weight = torch.where(targets == 1, self.alpha, 1 - self.alpha)

        # BCE loss
        bce = F.binary_cross_entropy_with_logits(
            logits.squeeze(-1), targets.float(), reduction='none'
        )

        loss = alpha_weight * focal_weight * bce
        return loss.mean()


class ClassificationLoss(nn.Module):
    """
    Combined loss for classification with various options.
    """

    def __init__(
        self,
        loss_type: str = 'bce',  # 'bce', 'focal'
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.0,
        pos_weight: float = None,
    ):
        """
        Args:
            loss_type: Type of loss ('bce' or 'focal')
            focal_alpha: Alpha parameter for focal loss
            focal_gamma: Gamma parameter for focal loss
            label_smoothing: Label smoothing factor [0, 1]
            pos_weight: Weight for positive class in BCE
        """
        super().__init__()

        self.loss_type = loss_type
        self.label_smoothing = label_smoothing

        if loss_type == 'bce':
            if pos_weight is not None:
                self.criterion = nn.BCEWithLogitsLoss(
                    pos_weight=torch.tensor([pos_weight])
                )
            else:
                self.criterion = nn.BCEWithLogitsLoss()
        elif loss_type == 'focal':
            self.criterion = BinaryFocalLoss(focal_alpha, focal_gamma)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            logits: (B, 1) or (B,) logits
            targets: (B,) labels

        Returns:
            Scalar loss
        """
        if logits.dim() == 2:
            logits = logits.squeeze(-1)

        targets = targets.float()

        # Apply label smoothing
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        if self.loss_type == 'bce':
            return self.criterion(logits, targets)
        else:
            return self.criterion(logits.unsqueeze(-1), targets.long())


class AuxiliaryLoss(nn.Module):
    """
    Loss with auxiliary frame-level supervision.
    """

    def __init__(
        self,
        main_loss_type: str = 'focal',
        aux_weight: float = 0.3,
        **kwargs
    ):
        super().__init__()
        self.main_loss = ClassificationLoss(loss_type=main_loss_type, **kwargs)
        self.aux_loss = ClassificationLoss(loss_type='bce', **kwargs)
        self.aux_weight = aux_weight

    def forward(
        self,
        main_logits: torch.Tensor,
        frame_logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> dict:
        """
        Args:
            main_logits: (B, 1) sequence-level logits
            frame_logits: (B, T, 1) frame-level logits
            targets: (B,) labels

        Returns:
            Dictionary with loss components
        """
        main = self.main_loss(main_logits, targets)

        # Frame-level: all frames should have same label
        B, T, _ = frame_logits.shape
        frame_targets = targets.unsqueeze(1).expand(B, T).reshape(-1)
        frame_logits_flat = frame_logits.reshape(-1)
        aux = self.aux_loss(frame_logits_flat, frame_targets)

        total = main + self.aux_weight * aux

        return {
            'total': total,
            'main': main,
            'aux': aux,
        }


class ConsistencyLoss(nn.Module):
    """
    Consistency loss for temporal coherence.

    Encourages frame embeddings to be consistent across time.
    """

    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin

    def forward(self, frame_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frame_embeddings: (B, T, D) frame-level embeddings

        Returns:
            Consistency loss
        """
        B, T, D = frame_embeddings.shape

        # Compute pairwise differences between consecutive frames
        diffs = frame_embeddings[:, 1:] - frame_embeddings[:, :-1]  # (B, T-1, D)

        # L2 distance
        distances = diffs.norm(dim=-1)  # (B, T-1)

        # Hinge loss: penalize distances above margin
        loss = F.relu(distances - self.margin).mean()

        return loss
