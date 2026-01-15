"""
Data augmentation for Gaussian sequences.

Includes temporal, spatial, and feature-level augmentations.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
import random


class TemporalAugmentation(nn.Module):
    """
    Temporal augmentations for sequence data.

    Includes:
    - Random temporal cropping
    - Temporal reversal
    - Frame dropping
    - Speed variation (stride change)
    """

    def __init__(
        self,
        reverse_prob: float = 0.5,
        drop_prob: float = 0.1,
        max_drop_frames: int = 5,
    ):
        super().__init__()
        self.reverse_prob = reverse_prob
        self.drop_prob = drop_prob
        self.max_drop_frames = max_drop_frames

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply temporal augmentations."""
        # Temporal reversal
        if random.random() < self.reverse_prob:
            batch = self._reverse_sequence(batch)

        # Frame dropping
        if random.random() < self.drop_prob:
            batch = self._drop_frames(batch)

        return batch

    def _reverse_sequence(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Reverse temporal order."""
        for key in ['positions', 'eigenvectors', 'scalars', 'mask']:
            if key in batch:
                batch[key] = torch.flip(batch[key], dims=[0])
        return batch

    def _drop_frames(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Randomly drop frames (replace with neighbors)."""
        T = batch['positions'].shape[0]
        num_drop = random.randint(1, min(self.max_drop_frames, T // 4))

        drop_indices = random.sample(range(1, T - 1), min(num_drop, T - 2))

        for idx in drop_indices:
            # Interpolate from neighbors
            for key in ['positions', 'eigenvectors', 'scalars']:
                if key in batch:
                    batch[key][idx] = (batch[key][idx - 1] + batch[key][idx + 1]) / 2

        return batch


class SpatialAugmentation(nn.Module):
    """
    Spatial augmentations for point cloud data.

    Includes:
    - Random rotation (while maintaining equivariance testing)
    - Random scaling
    - Random translation
    - Jittering
    """

    def __init__(
        self,
        rotation_prob: float = 0.5,
        scale_prob: float = 0.3,
        scale_range: tuple = (0.8, 1.2),
        jitter_prob: float = 0.3,
        jitter_std: float = 0.01,
        translate_prob: float = 0.3,
        translate_range: float = 0.1,
    ):
        super().__init__()
        self.rotation_prob = rotation_prob
        self.scale_prob = scale_prob
        self.scale_range = scale_range
        self.jitter_prob = jitter_prob
        self.jitter_std = jitter_std
        self.translate_prob = translate_prob
        self.translate_range = translate_range

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply spatial augmentations."""
        # Random rotation
        if random.random() < self.rotation_prob:
            batch = self._random_rotation(batch)

        # Random scaling
        if random.random() < self.scale_prob:
            batch = self._random_scale(batch)

        # Random translation
        if random.random() < self.translate_prob:
            batch = self._random_translate(batch)

        # Jittering
        if random.random() < self.jitter_prob:
            batch = self._jitter(batch)

        return batch

    def _random_rotation(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply random 3D rotation."""
        # Generate random rotation matrix
        R = self._random_rotation_matrix(batch['positions'].device)

        # Apply to positions
        batch['positions'] = torch.einsum('ij,...j->...i', R, batch['positions'])

        # Apply to eigenvectors
        if 'eigenvectors' in batch:
            # Rotate each eigenvector
            batch['eigenvectors'] = torch.einsum(
                'ij,...jk->...ik', R, batch['eigenvectors']
            )

        return batch

    def _random_rotation_matrix(self, device: torch.device) -> torch.Tensor:
        """Generate random rotation matrix using axis-angle."""
        # Random axis
        axis = torch.randn(3, device=device)
        axis = axis / axis.norm()

        # Random angle
        angle = torch.rand(1, device=device) * 2 * 3.14159

        # Rodrigues formula
        K = torch.tensor([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ], device=device)

        R = torch.eye(3, device=device) + \
            torch.sin(angle) * K + \
            (1 - torch.cos(angle)) * (K @ K)

        return R

    def _random_scale(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply random scaling."""
        scale = random.uniform(*self.scale_range)
        batch['positions'] = batch['positions'] * scale
        return batch

    def _random_translate(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply random translation."""
        device = batch['positions'].device
        offset = (torch.rand(3, device=device) - 0.5) * 2 * self.translate_range
        batch['positions'] = batch['positions'] + offset
        return batch

    def _jitter(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Add random noise to positions."""
        noise = torch.randn_like(batch['positions']) * self.jitter_std
        batch['positions'] = batch['positions'] + noise
        return batch


class GaussianAugmentation(nn.Module):
    """
    Gaussian-specific augmentations.

    Includes:
    - Random Gaussian dropout
    - Feature noise
    - Opacity perturbation
    """

    def __init__(
        self,
        dropout_prob: float = 0.1,
        max_dropout_ratio: float = 0.2,
        feature_noise_prob: float = 0.3,
        feature_noise_std: float = 0.05,
    ):
        super().__init__()
        self.dropout_prob = dropout_prob
        self.max_dropout_ratio = max_dropout_ratio
        self.feature_noise_prob = feature_noise_prob
        self.feature_noise_std = feature_noise_std

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply Gaussian-specific augmentations."""
        # Random Gaussian dropout
        if random.random() < self.dropout_prob:
            batch = self._random_dropout(batch)

        # Feature noise
        if random.random() < self.feature_noise_prob:
            batch = self._add_feature_noise(batch)

        return batch

    def _random_dropout(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Randomly mask out some Gaussians."""
        mask = batch['mask']
        T, M = mask.shape

        # Dropout ratio
        ratio = random.uniform(0, self.max_dropout_ratio)
        num_drop = int(M * ratio)

        for t in range(T):
            valid_indices = mask[t].nonzero(as_tuple=True)[0]
            if len(valid_indices) > num_drop:
                drop_indices = valid_indices[
                    torch.randperm(len(valid_indices))[:num_drop]
                ]
                mask[t, drop_indices] = False

        batch['mask'] = mask
        return batch

    def _add_feature_noise(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Add noise to scalar features."""
        if 'scalars' in batch:
            noise = torch.randn_like(batch['scalars']) * self.feature_noise_std
            batch['scalars'] = batch['scalars'] + noise
        return batch


class ComposedAugmentation(nn.Module):
    """Compose multiple augmentations."""

    def __init__(
        self,
        augmentations: list,
        p: float = 1.0,  # Probability of applying any augmentation
    ):
        super().__init__()
        self.augmentations = nn.ModuleList(augmentations)
        self.p = p

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if random.random() > self.p:
            return batch

        for aug in self.augmentations:
            batch = aug(batch)
        return batch


def get_train_augmentation(
    temporal: bool = True,
    spatial: bool = True,
    gaussian: bool = True,
) -> Optional[ComposedAugmentation]:
    """Get default training augmentation pipeline."""
    augs = []

    if temporal:
        augs.append(TemporalAugmentation(
            reverse_prob=0.5,
            drop_prob=0.1,
        ))

    if spatial:
        augs.append(SpatialAugmentation(
            rotation_prob=0.5,
            scale_prob=0.3,
            jitter_prob=0.3,
        ))

    if gaussian:
        augs.append(GaussianAugmentation(
            dropout_prob=0.1,
            feature_noise_prob=0.3,
        ))

    if augs:
        return ComposedAugmentation(augs, p=0.8)
    return None
