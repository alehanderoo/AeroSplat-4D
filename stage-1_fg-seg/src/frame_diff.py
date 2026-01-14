from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np


class FrameDifferencer:
    """
    Compute frame-to-frame differences to detect moving objects (drone).
    """

    def __init__(
        self,
        threshold: float = 30.0,
        blur_kernel_size: int = 5,
        min_change_area: int = 10,
    ) -> None:
        """
        Args:
            threshold: Pixel intensity difference threshold (0-255).
            blur_kernel_size: Gaussian blur kernel size for noise reduction.
            min_change_area: Minimum number of connected changed pixels to keep.
        """
        self.threshold = threshold
        self.blur_kernel_size = blur_kernel_size
        self.min_change_area = min_change_area

    def compute_difference(
        self,
        frame_current: np.ndarray,
        frame_previous: np.ndarray,
        save_path: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute pixel-wise difference between two frames.

        Args:
            frame_current: Current frame RGB (H, W, 3) uint8.
            frame_previous: Previous frame RGB (H, W, 3) uint8.
            save_path: Optional path to save debug images (without extension).

        Returns:
            diff_map: Grayscale difference magnitude (H, W) float32.
            changed_mask: Binary mask of changed pixels (H, W) bool.
        """
        if frame_current.shape != frame_previous.shape:
            raise ValueError("Frame dimensions must match")

        # Convert to grayscale if RGB
        if frame_current.ndim == 3 and frame_current.shape[2] == 3:
            gray_current = self._rgb_to_gray(frame_current)
            gray_previous = self._rgb_to_gray(frame_previous)
        else:
            gray_current = frame_current.astype(np.float32)
            gray_previous = frame_previous.astype(np.float32)

        # Compute absolute difference
        diff_map = np.abs(gray_current - gray_previous)

        # Apply Gaussian blur to reduce noise
        if self.blur_kernel_size > 1:
            diff_map = self._gaussian_blur(diff_map, self.blur_kernel_size)

        # Threshold to binary mask
        changed_mask = diff_map > self.threshold

        # Optional: filter small regions
        if self.min_change_area > 1:
            changed_mask = self._filter_small_regions(changed_mask, self.min_change_area)

        # Save debug images if requested
        if save_path is not None:
            self._save_debug_images(diff_map, changed_mask, save_path)

        return diff_map, changed_mask

    def get_changed_pixel_coords(self, changed_mask: np.ndarray) -> np.ndarray:
        """
        Extract (u, v) pixel coordinates of all changed pixels.

        Args:
            changed_mask: Binary mask (H, W) bool.

        Returns:
            coords: (N, 2) array of [u, v] pixel coordinates (int).
        """
        # Ensure mask is 2D
        if changed_mask.ndim > 2:
            # Take first channel or squeeze
            changed_mask = changed_mask.squeeze()
            if changed_mask.ndim > 2:
                changed_mask = changed_mask[:, :, 0]
        
        v_coords, u_coords = np.where(changed_mask)
        return np.stack([u_coords, v_coords], axis=1)

    @staticmethod
    def _rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
        """Convert RGB to grayscale using standard weights."""
        if rgb.dtype == np.uint8:
            rgb = rgb.astype(np.float32)
        return 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]

    @staticmethod
    def _gaussian_blur(image: np.ndarray, kernel_size: int) -> np.ndarray:
        """Apply Gaussian blur using scipy or fallback to simple averaging."""
        try:
            from scipy.ndimage import gaussian_filter

            sigma = kernel_size / 6.0
            return gaussian_filter(image, sigma=sigma)
        except ImportError:
            # Simple box blur fallback
            from numpy.lib.stride_tricks import sliding_window_view

            k = kernel_size // 2
            padded = np.pad(image, k, mode="edge")
            windows = sliding_window_view(padded, (kernel_size, kernel_size))
            return windows.mean(axis=(2, 3))

    @staticmethod
    def _filter_small_regions(mask: np.ndarray, min_size: int) -> np.ndarray:
        """Remove small connected components from binary mask."""
        try:
            from scipy.ndimage import label

            labeled, num_features = label(mask)
            for i in range(1, num_features + 1):
                component = labeled == i
                if component.sum() < min_size:
                    mask = mask & ~component
            return mask
        except ImportError:
            # Skip filtering if scipy not available
            return mask

    @staticmethod
    def _save_debug_images(diff_map: np.ndarray, changed_mask: np.ndarray, save_path: str) -> None:
        """Save diff_map and changed_mask as PNG images."""
        try:
            from PIL import Image
            
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Normalize diff_map to 0-255
            diff_normalized = (diff_map / diff_map.max() * 255.0 if diff_map.max() > 0 else diff_map).astype(np.uint8)
            Image.fromarray(diff_normalized).save(f"{save_path}_diff.png")
            
            # Save mask as binary image
            mask_img = (changed_mask.astype(np.uint8) * 255)
            Image.fromarray(mask_img).save(f"{save_path}_mask.png")
        except ImportError:
            pass  # PIL not available, skip saving

