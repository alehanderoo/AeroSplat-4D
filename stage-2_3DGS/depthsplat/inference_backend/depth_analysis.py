"""
Depth analysis utilities for the DepthSplat inference backend.

Provides functions for:
- Computing depth estimation metrics
- Applying colormaps to depth maps
- Comparing different depth estimation methods
"""

from typing import Dict, List, Optional
import numpy as np
import matplotlib.cm as cm
from PIL import Image


def compute_depth_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute depth estimation metrics.

    Args:
        pred: Predicted depth [H, W]
        gt: Ground truth depth [H, W]
        mask: Optional validity mask [H, W]

    Returns:
        Dictionary of metrics:
        - abs_rel: Absolute relative error
        - sq_rel: Squared relative error
        - rmse: Root mean squared error
        - rmse_log: RMSE in log space
        - delta1: % of pixels with max(pred/gt, gt/pred) < 1.25
        - delta2: % of pixels with threshold < 1.25^2
        - delta3: % of pixels with threshold < 1.25^3
    """
    if mask is None:
        mask = (gt > 0) & np.isfinite(gt) & np.isfinite(pred) & (pred > 0)
    else:
        mask = mask & (gt > 0) & np.isfinite(gt) & np.isfinite(pred) & (pred > 0)

    if mask.sum() == 0:
        return {
            'abs_rel': float('nan'),
            'sq_rel': float('nan'),
            'rmse': float('nan'),
            'rmse_log': float('nan'),
            'delta1': float('nan'),
            'delta2': float('nan'),
            'delta3': float('nan'),
        }

    pred_m = pred[mask]
    gt_m = gt[mask]

    # Absolute relative error
    abs_rel = np.mean(np.abs(pred_m - gt_m) / gt_m)

    # Squared relative error
    sq_rel = np.mean(((pred_m - gt_m) ** 2) / gt_m)

    # RMSE
    rmse = np.sqrt(np.mean((pred_m - gt_m) ** 2))

    # RMSE log
    rmse_log = np.sqrt(np.mean((np.log(pred_m) - np.log(gt_m)) ** 2))

    # Delta thresholds
    thresh = np.maximum(pred_m / gt_m, gt_m / pred_m)
    delta1 = np.mean(thresh < 1.25) * 100
    delta2 = np.mean(thresh < 1.25 ** 2) * 100
    delta3 = np.mean(thresh < 1.25 ** 3) * 100

    return {
        'abs_rel': float(abs_rel),
        'sq_rel': float(sq_rel),
        'rmse': float(rmse),
        'rmse_log': float(rmse_log),
        'delta1': float(delta1),
        'delta2': float(delta2),
        'delta3': float(delta3),
    }


def scale_and_shift_pred(
    pred: np.ndarray,
    gt: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Apply least-squares scale and shift to align predicted depth with GT.

    This is useful for monocular depth which has scale ambiguity.
    Solves: pred_aligned = scale * pred + shift to minimize ||pred_aligned - gt||^2

    Args:
        pred: Predicted depth [H, W]
        gt: Ground truth depth [H, W]
        mask: Optional validity mask

    Returns:
        Scale-and-shift aligned prediction
    """
    if mask is None:
        mask = (gt > 0) & np.isfinite(gt) & np.isfinite(pred)

    if mask.sum() < 2:
        return pred

    pred_m = pred[mask].flatten()
    gt_m = gt[mask].flatten()

    # Solve [pred, 1] @ [scale, shift]^T = gt via least squares
    A = np.stack([pred_m, np.ones_like(pred_m)], axis=1)
    x, _, _, _ = np.linalg.lstsq(A, gt_m, rcond=None)
    scale, shift = x

    return pred * scale + shift


def apply_colormap(
    depth: np.ndarray,
    cmap_name: str = 'plasma',
    percentile_clip: bool = True,
) -> np.ndarray:
    """
    Apply a matplotlib colormap to a depth map.

    Args:
        depth: Depth values [H, W]
        cmap_name: Colormap name
        percentile_clip: If True, clip to 2nd/98th percentiles

    Returns:
        RGB image [H, W, 3] as uint8
    """
    valid = np.isfinite(depth) & (depth > 0)
    if valid.any():
        if percentile_clip:
            d_min = np.percentile(depth[valid], 2)
            d_max = np.percentile(depth[valid], 98)
        else:
            d_min = depth[valid].min()
            d_max = depth[valid].max()

        if d_max > d_min:
            depth_norm = (depth - d_min) / (d_max - d_min)
        else:
            depth_norm = np.zeros_like(depth)
    else:
        depth_norm = np.zeros_like(depth)

    depth_norm = np.clip(depth_norm, 0, 1)

    cmap = cm.get_cmap(cmap_name)
    colored = cmap(depth_norm)
    rgb = (colored[:, :, :3] * 255).astype(np.uint8)

    return rgb


def apply_turbo_colormap(depth_normalized: np.ndarray) -> np.ndarray:
    """
    Apply a turbo-like colormap to normalized depth values.

    Args:
        depth_normalized: Normalized depth values in [0, 1] where 0=close, 1=far

    Returns:
        RGB image as uint8 array [H, W, 3]
    """
    h, w = depth_normalized.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    # Invert so close is bright (yellow/red), far is dark (blue/purple)
    d = 1.0 - depth_normalized

    # Simple turbo-like colormap
    rgb[..., 0] = np.clip(255 * (1.5 - np.abs(d - 0.75) * 4), 0, 255).astype(np.uint8)
    rgb[..., 1] = np.clip(255 * (1.5 - np.abs(d - 0.5) * 4), 0, 255).astype(np.uint8)
    rgb[..., 2] = np.clip(255 * (1.5 - np.abs(d - 0.25) * 4), 0, 255).astype(np.uint8)

    return rgb


def normalize_depth_for_display(
    depth: np.ndarray,
    near: Optional[float] = None,
    far: Optional[float] = None,
) -> np.ndarray:
    """
    Normalize depth values for display.

    Args:
        depth: Raw depth values [H, W]
        near: Optional near plane (if None, uses min valid value)
        far: Optional far plane (if None, uses max valid value)

    Returns:
        Normalized depth in [0, 1]
    """
    valid_mask = np.isfinite(depth) & (depth > 0)
    if not valid_mask.any():
        return np.zeros_like(depth)

    if near is None:
        near = depth[valid_mask].min()
    if far is None:
        far = depth[valid_mask].max()

    if far <= near:
        return np.zeros_like(depth)

    depth_normalized = (depth - near) / (far - near)
    depth_normalized = np.clip(depth_normalized, 0, 1)
    depth_normalized[~valid_mask] = 0

    return depth_normalized


def format_metrics_for_display(metrics: Dict) -> Dict:
    """
    Format depth metrics for user-friendly display.

    Args:
        metrics: Raw metrics dictionary from compute_all_metrics

    Returns:
        Formatted dictionary suitable for JSON display
    """
    formatted = {}

    metric_names = {
        'standalone_da_aligned': 'Standalone DA',
        'coarse_mv_aligned': 'Coarse MV',
        'final_fused_aligned': 'Final Fused',
    }

    for key, display_name in metric_names.items():
        if key in metrics and metrics[key]:
            m = metrics[key]
            formatted[display_name] = {
                'AbsRel': f"{m.get('abs_rel', 0):.4f}",
                'RMSE': f"{m.get('rmse', 0):.3f}",
                'delta<1.25': f"{m.get('delta1', 0):.1f}%",
                'delta<1.25^2': f"{m.get('delta2', 0):.1f}%",
            }

    if formatted:
        formatted['Note'] = 'Metrics computed after scale-shift alignment to GT'

    return formatted
