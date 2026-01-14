"""
Depth supervision loss for object-centric 3D Gaussian Splatting.

Supervises the predicted depth with ground truth depth from Blender renders.
Only computes loss on valid foreground pixels (where GT depth is valid).
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
class LossDepthCfg:
    weight: float = 0.1
    # Minimum valid depth (ignore depths below this)
    min_depth: float = 0.1
    # Maximum valid depth (ignore depths above this)
    max_depth: float = 10.0
    # Weight for background depth regularization (push background to far depth)
    background_weight: float = 0.1


@dataclass
class LossDepthCfgWrapper:
    depth: LossDepthCfg


class LossDepth(Loss[LossDepthCfg, LossDepthCfgWrapper]):
    """
    Depth supervision loss using ground truth depth from Blender.

    This loss helps the model learn correct depth estimation, which is critical
    for proper Gaussian placement in 3D space.
    """

    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        global_step: int,
        pred_depths: Tensor | None = None,
        valid_depth_mask: Tensor | None = None,
        **kwargs,
    ) -> Float[Tensor, ""]:
        """
        Compute depth supervision loss.

        Args:
            prediction: Decoder output (not used directly)
            batch: Batch containing context depth and mask
            gaussians: Predicted Gaussians (not used)
            global_step: Current training step
            pred_depths: Predicted depth maps from encoder [B, V, H, W]

        Returns:
            Scalar loss tensor
        """
        # Check if we have predicted depths and GT depths
        if pred_depths is None:
            return torch.tensor(0.0, device=prediction.color.device)

        if "depth" not in batch["context"]:
            return torch.tensor(0.0, device=prediction.color.device)

        gt_depth = batch["context"]["depth"]  # [B, V, H, W]

        # Get mask if available
        mask = None
        if "mask" in batch["context"]:
            mask = batch["context"]["mask"]  # [B, V, H, W]
            if mask.dim() == 5:
                mask = mask.squeeze(2)

        # Resize pred_depths to match GT if needed
        if pred_depths.shape[-2:] != gt_depth.shape[-2:]:
            pred_depths = F.interpolate(
                pred_depths.unsqueeze(1) if pred_depths.dim() == 3 else pred_depths,
                size=gt_depth.shape[-2:],
                mode='bilinear',
                align_corners=True
            )
            if pred_depths.dim() == 5:
                pred_depths = pred_depths.squeeze(1)

        # Create validity mask
        valid = (gt_depth > self.cfg.min_depth) & (gt_depth < self.cfg.max_depth)

        # Also use foreground mask if available
        if mask is not None:
            valid = valid & (mask > 0.5)

        if valid.sum() == 0:
            return torch.tensor(0.0, device=prediction.color.device)

        # Compute L1 loss on valid pixels
        depth_error = (pred_depths - gt_depth).abs()
        loss = depth_error[valid].mean()

        # Background depth regularization: push background to far depth
        # This prevents grid artifacts and arbitrary depth predictions in background
        if mask is not None and self.cfg.background_weight > 0:
            bg_mask = mask < 0.5
            if bg_mask.sum() > 0:
                far_depth = self.cfg.max_depth
                bg_loss = (pred_depths[bg_mask] - far_depth).abs().mean()
                loss = loss + self.cfg.background_weight * bg_loss

        return self.cfg.weight * loss
