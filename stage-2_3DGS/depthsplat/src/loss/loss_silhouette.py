"""
Silhouette consistency loss for object-centric 3D Gaussian Splatting.

This loss encourages the rendered silhouette to match the ground truth
segmentation mask, providing geometric supervision independent of texture.

As described in the thesis (05_method.tex:174-176):
    L_silhouette = BCE(sum_i(alpha_i * G_i(p)), M_gt(p))

where:
    - alpha_i is Gaussian opacity
    - G_i(p) is the Gaussian contribution at pixel p
    - M_gt is the ground truth foreground mask
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
class LossSilhouetteCfg:
    weight: float = 0.1
    # Method to compute silhouette from rendered image
    # "alpha_from_color": Estimate alpha from difference to background color
    # "alpha_from_depth": Use depth map validity as alpha
    method: str = "alpha_from_color"
    # Background color (normalized 0-1) for alpha estimation
    background_color: list[float] | tuple[float, ...] = (0.0, 0.0, 0.0)
    # Threshold for alpha estimation
    alpha_threshold: float = 0.01


@dataclass
class LossSilhouetteCfgWrapper:
    silhouette: LossSilhouetteCfg


class LossSilhouette(Loss[LossSilhouetteCfg, LossSilhouetteCfgWrapper]):
    """
    Silhouette consistency loss for object-centric reconstruction.

    This loss provides geometric supervision that works even in textureless
    regions (like sky backgrounds) where photometric losses fail.
    """

    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        global_step: int,
        valid_depth_mask: Tensor | None = None,
        **kwargs,
    ) -> Float[Tensor, ""]:
        """
        Compute silhouette consistency loss.

        Args:
            prediction: Decoder output containing rendered color and depth
            batch: Batch containing target masks
            gaussians: Predicted Gaussians (not directly used)
            global_step: Current training step

        Returns:
            Scalar loss tensor
        """
        # Check if masks are available
        if "mask" not in batch["target"]:
            # No masks available, return zero loss
            return torch.tensor(0.0, device=prediction.color.device)

        target_mask = batch["target"]["mask"]  # [B, V, H, W] or [B, V, 1, H, W]

        # Handle different mask shapes
        if len(target_mask.shape) == 5:
            target_mask = target_mask.squeeze(2)  # [B, V, H, W]

        # Ensure mask is on same device as prediction
        target_mask = target_mask.to(prediction.color.device)

        # Compute predicted alpha/silhouette
        if self.cfg.method == "alpha_from_depth" and prediction.depth is not None:
            # Use depth validity as alpha
            # Assume invalid depth (0 or inf) indicates background
            pred_alpha = self._alpha_from_depth(prediction.depth)
        else:
            # Estimate alpha from color difference to background
            pred_alpha = self._alpha_from_color(prediction.color)

        # Resize if shapes don't match
        if pred_alpha.shape[-2:] != target_mask.shape[-2:]:
            target_mask = F.interpolate(
                target_mask.unsqueeze(1),
                size=pred_alpha.shape[-2:],
                mode='nearest'
            ).squeeze(1)

        # Binary cross entropy loss
        # Clamp predictions to avoid log(0)
        pred_alpha = torch.clamp(pred_alpha, 1e-7, 1 - 1e-7)
        target_mask = torch.clamp(target_mask, 0, 1)

        loss = F.binary_cross_entropy(pred_alpha, target_mask)

        return self.cfg.weight * loss

    def _alpha_from_color(
        self,
        color: Float[Tensor, "batch view 3 height width"],
    ) -> Float[Tensor, "batch view height width"]:
        """
        Estimate alpha from rendered color by comparing to background.

        For a black background, alpha is estimated as how far the pixel
        is from black in RGB space.
        For a white background, alpha is estimated as how far the pixel
        is from white in RGB space.
        """
        # Background color tensor
        bg_color = torch.tensor(
            self.cfg.background_color,
            device=color.device,
            dtype=color.dtype
        ).view(1, 1, 3, 1, 1)

        # Detect if white or black background
        bg_mean = sum(self.cfg.background_color) / 3.0
        is_white_bg = bg_mean > 0.5

        if is_white_bg:
            # White background: alpha = 1 - min distance to white
            # Pixels close to white (background) have low alpha
            diff = (1.0 - color).abs()  # Distance from white
            alpha = diff.max(dim=2)[0]  # [B, V, H, W]
        else:
            # Black background: alpha = max color channel
            # Pixels close to black (background) have low alpha
            diff = (color - bg_color).abs()
            alpha = diff.max(dim=2)[0]  # [B, V, H, W]

        # Apply threshold to reduce noise
        alpha = torch.where(
            alpha > self.cfg.alpha_threshold,
            alpha,
            torch.zeros_like(alpha)
        )

        # Clamp to [0, 1]
        alpha = torch.clamp(alpha, 0, 1)

        return alpha

    def _alpha_from_depth(
        self,
        depth: Float[Tensor, "batch view height width"],
    ) -> Float[Tensor, "batch view height width"]:
        """
        Estimate alpha from depth map.

        Valid (finite, non-zero) depth indicates foreground.
        """
        # Foreground has valid depth
        valid = (depth > 0) & torch.isfinite(depth)
        return valid.float()
