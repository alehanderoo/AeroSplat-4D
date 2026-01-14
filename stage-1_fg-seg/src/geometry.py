from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def quat_wxyz_to_rotation_matrix(quat_wxyz: Iterable[float]) -> np.ndarray:
    """
    Convert a quaternion expressed as (w, x, y, z) into a 3x3 rotation matrix.

    The incoming data from the Omniverse export already uses a right-handed,
    Z-up world. We keep that convention here.
    """
    w, x, y, z = quat_wxyz
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm == 0.0:
        raise ValueError("Zero-norm quaternion is invalid")

    w /= norm
    x /= norm
    y /= norm
    z /= norm

    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def homogenise(points: np.ndarray) -> np.ndarray:
    """
    Append a homogeneous coordinate of 1.0 to a (N, 3) array of points.
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("homogenise expects shape (N, 3)")
    ones = np.ones((points.shape[0], 1), dtype=points.dtype)
    return np.concatenate([points, ones], axis=1)

