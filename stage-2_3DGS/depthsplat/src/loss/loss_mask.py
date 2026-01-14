"""
Direct mask supervision loss for object-centric 3D Gaussian Splatting.

Supervises the rendered alpha/opacity directly against the ground truth mask.
This is the literature-standard approach (LGM, GRM, TriplaneGaussian, etc.)
for object-centric reconstruction.

Key insight: Decoupling alpha supervision from RGB loss prevents the model
from trading off alpha accuracy for RGB accuracy.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss


@dataclass
class LossMaskCfg:
    weight: float = 0.1
    # Use BCE loss instead of MSE (MSE is the standard approach)
    use_bce: bool = False


@dataclass
class LossMaskCfgWrapper:
    mask: LossMaskCfg


class LossMask(Loss[LossMaskCfg, LossMaskCfgWrapper]):
    """
    Direct mask supervision loss for object-centric reconstruction.

    This loss supervises the rendered alpha channel against the ground truth
    segmentation mask, providing explicit foreground/background separation
    independent of RGB color.
    """

    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        global_step: int,
        **kwargs,
    ) -> Float[Tensor, ""]:
        """
        Compute direct mask supervision loss.

        Args:
            prediction: Decoder output containing rendered alpha
            batch: Batch containing target masks
            gaussians: Predicted Gaussians (not used)
            global_step: Current training step

        Returns:
            Scalar loss tensor
        """
        # Check if alpha is available from decoder
        if prediction.alpha is None:
            return torch.tensor(0.0, device=prediction.color.device)

        # Check if masks are available in batch
        if "mask" not in batch["target"]:
            return torch.tensor(0.0, device=prediction.color.device)

        pred_alpha = prediction.alpha  # [B, V, H, W]
        gt_mask = batch["target"]["mask"]  # [B, V, H, W] or [B, V, 1, H, W]

        # Handle different mask shapes
        if gt_mask.dim() == 5:
            gt_mask = gt_mask.squeeze(2)  # [B, V, H, W]

        # Ensure mask is on same device as prediction
        gt_mask = gt_mask.to(pred_alpha.device, dtype=pred_alpha.dtype)

        # Resize if shapes don't match
        if pred_alpha.shape[-2:] != gt_mask.shape[-2:]:
            gt_mask = F.interpolate(
                gt_mask.unsqueeze(2),  # [B, V, 1, H, W]
                size=pred_alpha.shape[-2:],
                mode='nearest'
            ).squeeze(2)

        # Compute loss
        if self.cfg.use_bce:
            # Binary cross entropy loss
            pred_alpha_clamped = pred_alpha.clamp(1e-7, 1 - 1e-7)
            gt_mask_clamped = gt_mask.clamp(0, 1)
            loss = F.binary_cross_entropy(pred_alpha_clamped, gt_mask_clamped)
        else:
            # MSE loss (standard approach from LGM, GRM, etc.)
            loss = F.mse_loss(pred_alpha, gt_mask)

        return self.cfg.weight * loss
