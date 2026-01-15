"""
Quaternion and rotation utilities.
"""

import torch
from torch import Tensor
import math


def quaternion_to_rotation_matrix(q: Tensor) -> Tensor:
    """
    Convert quaternion to rotation matrix.

    Args:
        q: Quaternion tensor (N, 4) in wxyz order

    Returns:
        Rotation matrices (N, 3, 3)
    """
    # Normalize
    q = q / (q.norm(dim=-1, keepdim=True) + 1e-8)

    w, x, y, z = q.unbind(-1)

    # Rotation matrix from quaternion
    R = torch.stack([
        1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y,
        2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x,
        2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y
    ], dim=-1).view(-1, 3, 3)

    return R


def rotation_matrix_to_quaternion(R: Tensor) -> Tensor:
    """
    Convert rotation matrix to quaternion.

    Args:
        R: Rotation matrices (N, 3, 3)

    Returns:
        Quaternions (N, 4) in wxyz order
    """
    batch_dim = R.shape[:-2]
    R = R.view(-1, 3, 3)
    N = R.shape[0]

    # Compute quaternion components
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

    q = torch.zeros(N, 4, device=R.device, dtype=R.dtype)

    # Case 1: trace > 0
    mask1 = trace > 0
    s1 = torch.sqrt(trace[mask1] + 1.0) * 2  # s = 4 * w
    q[mask1, 0] = 0.25 * s1
    q[mask1, 1] = (R[mask1, 2, 1] - R[mask1, 1, 2]) / s1
    q[mask1, 2] = (R[mask1, 0, 2] - R[mask1, 2, 0]) / s1
    q[mask1, 3] = (R[mask1, 1, 0] - R[mask1, 0, 1]) / s1

    # Case 2: R[0,0] > R[1,1] and R[0,0] > R[2,2]
    mask2 = ~mask1 & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
    s2 = torch.sqrt(1.0 + R[mask2, 0, 0] - R[mask2, 1, 1] - R[mask2, 2, 2]) * 2
    q[mask2, 0] = (R[mask2, 2, 1] - R[mask2, 1, 2]) / s2
    q[mask2, 1] = 0.25 * s2
    q[mask2, 2] = (R[mask2, 0, 1] + R[mask2, 1, 0]) / s2
    q[mask2, 3] = (R[mask2, 0, 2] + R[mask2, 2, 0]) / s2

    # Case 3: R[1,1] > R[2,2]
    mask3 = ~mask1 & ~mask2 & (R[:, 1, 1] > R[:, 2, 2])
    s3 = torch.sqrt(1.0 + R[mask3, 1, 1] - R[mask3, 0, 0] - R[mask3, 2, 2]) * 2
    q[mask3, 0] = (R[mask3, 0, 2] - R[mask3, 2, 0]) / s3
    q[mask3, 1] = (R[mask3, 0, 1] + R[mask3, 1, 0]) / s3
    q[mask3, 2] = 0.25 * s3
    q[mask3, 3] = (R[mask3, 1, 2] + R[mask3, 2, 1]) / s3

    # Case 4: else
    mask4 = ~mask1 & ~mask2 & ~mask3
    s4 = torch.sqrt(1.0 + R[mask4, 2, 2] - R[mask4, 0, 0] - R[mask4, 1, 1]) * 2
    q[mask4, 0] = (R[mask4, 1, 0] - R[mask4, 0, 1]) / s4
    q[mask4, 1] = (R[mask4, 0, 2] + R[mask4, 2, 0]) / s4
    q[mask4, 2] = (R[mask4, 1, 2] + R[mask4, 2, 1]) / s4
    q[mask4, 3] = 0.25 * s4

    # Normalize
    q = q / (q.norm(dim=-1, keepdim=True) + 1e-8)

    return q.view(*batch_dim, 4)


def random_rotation_matrix(device: torch.device = None) -> Tensor:
    """
    Generate a random 3D rotation matrix.

    Uses axis-angle representation for uniform sampling.

    Args:
        device: Target device

    Returns:
        (3, 3) rotation matrix
    """
    if device is None:
        device = torch.device('cpu')

    # Random axis (uniform on sphere)
    axis = torch.randn(3, device=device)
    axis = axis / axis.norm()

    # Random angle [0, 2π)
    angle = torch.rand(1, device=device) * 2 * math.pi

    # Rodrigues formula: R = I + sin(θ)K + (1-cos(θ))K²
    K = torch.tensor([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ], device=device, dtype=torch.float32)

    R = torch.eye(3, device=device) + \
        torch.sin(angle) * K + \
        (1 - torch.cos(angle)) * (K @ K)

    return R


def apply_rotation(
    points: Tensor,
    R: Tensor
) -> Tensor:
    """
    Apply rotation matrix to points.

    Args:
        points: (..., 3) points
        R: (3, 3) rotation matrix

    Returns:
        (..., 3) rotated points
    """
    return torch.einsum('ij,...j->...i', R, points)


def rotation_error(R1: Tensor, R2: Tensor) -> Tensor:
    """
    Compute rotation error between two rotation matrices.

    Error is the angle of the rotation needed to align R1 with R2.

    Args:
        R1: (N, 3, 3) rotation matrices
        R2: (N, 3, 3) rotation matrices

    Returns:
        (N,) rotation errors in radians
    """
    # R_diff = R1^T @ R2
    R_diff = torch.bmm(R1.transpose(-1, -2), R2)

    # Angle from trace: tr(R) = 1 + 2cos(θ)
    trace = R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2]
    cos_angle = (trace - 1) / 2
    cos_angle = cos_angle.clamp(-1, 1)
    angle = torch.acos(cos_angle)

    return angle


def euler_to_rotation_matrix(
    roll: float,
    pitch: float,
    yaw: float,
    device: torch.device = None
) -> Tensor:
    """
    Convert Euler angles to rotation matrix.

    Uses ZYX convention (yaw-pitch-roll).

    Args:
        roll: Rotation around x-axis (radians)
        pitch: Rotation around y-axis (radians)
        yaw: Rotation around z-axis (radians)
        device: Target device

    Returns:
        (3, 3) rotation matrix
    """
    if device is None:
        device = torch.device('cpu')

    # Rotation around x-axis (roll)
    c_r, s_r = math.cos(roll), math.sin(roll)
    Rx = torch.tensor([
        [1, 0, 0],
        [0, c_r, -s_r],
        [0, s_r, c_r]
    ], device=device, dtype=torch.float32)

    # Rotation around y-axis (pitch)
    c_p, s_p = math.cos(pitch), math.sin(pitch)
    Ry = torch.tensor([
        [c_p, 0, s_p],
        [0, 1, 0],
        [-s_p, 0, c_p]
    ], device=device, dtype=torch.float32)

    # Rotation around z-axis (yaw)
    c_y, s_y = math.cos(yaw), math.sin(yaw)
    Rz = torch.tensor([
        [c_y, -s_y, 0],
        [s_y, c_y, 0],
        [0, 0, 1]
    ], device=device, dtype=torch.float32)

    # Combined rotation: R = Rz @ Ry @ Rx
    return Rz @ Ry @ Rx
