"""
Gradient matching loss for depth edge sharpness.

Supervises depth gradients to match ground truth depth gradients, with optional
mask boundary weighting to emphasize edge sharpness at object boundaries.

Based on research from Depth Anything V2 and EG-MVSNet demonstrating that
gradient matching loss at mask boundaries yields sharper depth reconstructions.
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
class LossGradientCfg:
    weight: float = 0.1
    # Weight multiplier at mask boundaries (1.0 = no extra weighting)
    boundary_weight: float = 2.0
    # Dilation radius for mask boundary detection
    boundary_dilation: int = 3
    # Use multi-scale gradient matching
    multi_scale: bool = True
    # Minimum valid depth for gradient computation
    min_depth: float = 0.1
    # Maximum valid depth for gradient computation
    max_depth: float = 10.0


@dataclass
class LossGradientCfgWrapper:
    gradient: LossGradientCfg


def sobel_gradient(x: Tensor) -> tuple[Tensor, Tensor]:
    """
    Compute Sobel gradients in x and y directions.

    Args:
        x: Input tensor [B, C, H, W] or [B, H, W]

    Returns:
        Tuple of (grad_x, grad_y) tensors
    """
    if x.dim() == 3:
        x = x.unsqueeze(1)

    # Sobel kernels
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                           dtype=x.dtype, device=x.device).view(1, 1, 3, 3)

    # Apply to each channel
    b, c, h, w = x.shape
    x_flat = x.view(b * c, 1, h, w)
    grad_x = F.conv2d(x_flat, sobel_x, padding=1).view(b, c, h, w)
    grad_y = F.conv2d(x_flat, sobel_y, padding=1).view(b, c, h, w)

    return grad_x.squeeze(1), grad_y.squeeze(1)


def compute_mask_boundary(mask: Tensor, dilation: int = 3) -> Tensor:
    """
    Compute mask boundary using morphological operations.

    Args:
        mask: Binary mask tensor [B, V, H, W] or [B, H, W]
        dilation: Dilation radius for boundary detection

    Returns:
        Boundary mask tensor (same shape as input)
    """
    if mask.dim() == 4:
        b, v, h, w = mask.shape
        mask_flat = mask.view(b * v, 1, h, w)
    else:
        mask_flat = mask.unsqueeze(1)

    # Create dilation kernel
    kernel_size = 2 * dilation + 1
    kernel = torch.ones(1, 1, kernel_size, kernel_size,
                        dtype=mask.dtype, device=mask.device)

    # Dilate and erode
    dilated = F.conv2d(mask_flat, kernel, padding=dilation)
    dilated = (dilated > 0).float()
    eroded = F.conv2d(mask_flat, kernel, padding=dilation)
    eroded = (eroded >= kernel.numel()).float()

    # Boundary = dilated XOR eroded
    boundary = dilated - eroded
    boundary = boundary.abs()

    if mask.dim() == 4:
        boundary = boundary.view(b, v, h, w)
    else:
        boundary = boundary.squeeze(1)

    return boundary


class LossGradient(Loss[LossGradientCfg, LossGradientCfgWrapper]):
    """
    Gradient matching loss for sharper depth edges.

    This loss enforces that predicted depth gradients match GT depth gradients,
    with optional extra weighting at mask boundaries where sharp edges are
    most important for Gaussian reconstruction quality.
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
        Compute gradient matching loss.

        Args:
            prediction: Decoder output
            batch: Batch containing context depth and mask
            gaussians: Predicted Gaussians
            global_step: Current training step
            pred_depths: Predicted depth maps [B, V, H, W]

        Returns:
            Scalar loss tensor
        """
        if pred_depths is None:
            return torch.tensor(0.0, device=prediction.color.device)

        if "depth" not in batch["context"]:
            return torch.tensor(0.0, device=prediction.color.device)

        gt_depth = batch["context"]["depth"]  # [B, V, H, W]

        # Get mask if available for boundary weighting
        mask = None
        if "mask" in batch["context"]:
            mask = batch["context"]["mask"]  # [B, V, H, W]
            if mask.dim() == 5:
                mask = mask.squeeze(2)

        # Resize pred_depths to match GT if needed
        if pred_depths.shape[-2:] != gt_depth.shape[-2:]:
            pred_depths = F.interpolate(
                pred_depths.view(-1, 1, *pred_depths.shape[-2:]),
                size=gt_depth.shape[-2:],
                mode='bilinear',
                align_corners=True
            ).view(*gt_depth.shape)

        # Create validity mask
        valid = (gt_depth > self.cfg.min_depth) & (gt_depth < self.cfg.max_depth)
        if mask is not None:
            valid = valid & (mask > 0.5)

        # Flatten batch and view dimensions for gradient computation
        b, v = pred_depths.shape[:2]
        pred_flat = pred_depths.view(b * v, *pred_depths.shape[2:])
        gt_flat = gt_depth.view(b * v, *gt_depth.shape[2:])
        valid_flat = valid.view(b * v, *valid.shape[2:])

        # Compute boundary weight map
        if mask is not None and self.cfg.boundary_weight > 1.0:
            mask_flat = mask.view(b * v, *mask.shape[2:])
            boundary = compute_mask_boundary(mask_flat, self.cfg.boundary_dilation)
            weight_map = 1.0 + (self.cfg.boundary_weight - 1.0) * boundary
        else:
            weight_map = torch.ones_like(pred_flat)

        total_loss = torch.tensor(0.0, device=prediction.color.device)
        num_scales = 0

        # Multi-scale gradient matching
        scales = [1, 2, 4] if self.cfg.multi_scale else [1]

        for scale in scales:
            if scale > 1:
                curr_pred = F.avg_pool2d(pred_flat.unsqueeze(1), scale).squeeze(1)
                curr_gt = F.avg_pool2d(gt_flat.unsqueeze(1), scale).squeeze(1)
                curr_valid = F.avg_pool2d(valid_flat.float().unsqueeze(1), scale).squeeze(1) > 0.5
                curr_weight = F.avg_pool2d(weight_map.unsqueeze(1), scale).squeeze(1)
            else:
                curr_pred = pred_flat
                curr_gt = gt_flat
                curr_valid = valid_flat
                curr_weight = weight_map

            # Compute gradients
            pred_grad_x, pred_grad_y = sobel_gradient(curr_pred)
            gt_grad_x, gt_grad_y = sobel_gradient(curr_gt)

            # Gradient matching loss
            grad_x_diff = (pred_grad_x - gt_grad_x).abs()
            grad_y_diff = (pred_grad_y - gt_grad_y).abs()

            grad_loss = grad_x_diff + grad_y_diff

            # Apply validity mask and boundary weighting
            if curr_valid.sum() > 0:
                weighted_loss = (grad_loss * curr_weight)[curr_valid]
                total_loss = total_loss + weighted_loss.mean()
                num_scales += 1

        if num_scales > 0:
            total_loss = total_loss / num_scales

        return self.cfg.weight * total_loss
