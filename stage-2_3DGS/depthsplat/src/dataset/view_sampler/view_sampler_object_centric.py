"""
View sampler for object-centric datasets (e.g., Objaverse).

Unlike video-based view samplers that assume sequential frames,
this sampler is designed for 360-degree object captures where:
- Views are distributed around the object
- View selection should maximize coverage
- No temporal ordering is assumed
"""

from dataclasses import dataclass
from typing import Literal

import torch
from jaxtyping import Float, Int64
from torch import Tensor

from ...misc.step_tracker import StepTracker
from ..types import Stage
from .view_sampler import ViewSampler


@dataclass
class ViewSamplerObjectCentricCfg:
    name: Literal["object_centric"]
    num_context_views: int
    num_target_views: int
    min_context_views: int = 2  # Minimum context views required
    sampling_strategy: Literal["farthest_point", "random", "uniform"] = "farthest_point"
    target_from_context: bool = False  # If True, sample targets from different views than context


class ViewSamplerObjectCentric(ViewSampler[ViewSamplerObjectCentricCfg]):
    """
    View sampler for object-centric datasets.

    Supports multiple sampling strategies:
    - farthest_point: Greedily select views that maximize angular coverage
    - random: Randomly sample views
    - uniform: Sample views at uniform angular intervals (requires sorted views)
    """

    def sample(
        self,
        scene: str,
        extrinsics: Float[Tensor, "view 4 4"],
        intrinsics: Float[Tensor, "view 3 3"],
        device: torch.device = torch.device("cpu"),
        **kwargs,
    ) -> tuple[
        Int64[Tensor, " context_view"],  # indices for context views
        Int64[Tensor, " target_view"],   # indices for target views
    ]:
        num_views = extrinsics.shape[0]

        # Ensure we have enough views
        min_required = self.cfg.num_context_views + self.cfg.num_target_views
        if self.cfg.target_from_context:
            min_required = max(self.cfg.num_context_views, self.cfg.num_target_views)

        if num_views < min_required:
            raise ValueError(
                f"Not enough views: {num_views} < {min_required} required"
            )

        # Extract camera positions
        camera_positions = extrinsics[:, :3, 3]  # [V, 3]

        if self.cfg.sampling_strategy == "farthest_point":
            context_indices = self._farthest_point_sampling(
                camera_positions,
                self.cfg.num_context_views
            )
        elif self.cfg.sampling_strategy == "uniform":
            context_indices = self._uniform_sampling(
                num_views,
                self.cfg.num_context_views
            )
        else:  # random
            context_indices = self._random_sampling(
                num_views,
                self.cfg.num_context_views
            )

        # Sample target views
        if self.cfg.target_from_context:
            # Sample targets from context views (for reconstruction validation)
            target_pool = context_indices.tolist()
        else:
            # Sample targets from non-context views
            all_indices = set(range(num_views))
            context_set = set(context_indices.tolist())
            target_pool = list(all_indices - context_set)

        if len(target_pool) < self.cfg.num_target_views:
            # Fall back to including context views
            target_pool = list(range(num_views))

        # Sample targets
        if self.stage == "test":
            # For testing, use deterministic sampling
            target_indices = torch.tensor(
                target_pool[:self.cfg.num_target_views],
                dtype=torch.int64
            )
        else:
            # For training, sample randomly
            perm = torch.randperm(len(target_pool))
            target_indices = torch.tensor(
                [target_pool[i] for i in perm[:self.cfg.num_target_views]],
                dtype=torch.int64
            )

        return context_indices, target_indices

    def _farthest_point_sampling(
        self,
        positions: Float[Tensor, "view 3"],
        num_samples: int,
    ) -> Int64[Tensor, " samples"]:
        """
        Greedily select views that maximize spatial coverage.

        Starts with a random view, then iteratively adds the view
        that is farthest from all currently selected views.
        """
        num_views = positions.shape[0]
        num_samples = min(num_samples, num_views)

        # Compute pairwise distances
        dist_matrix = torch.cdist(positions, positions)  # [V, V]

        # Start with a random view
        selected = []
        first_idx = torch.randint(0, num_views, (1,)).item()
        selected.append(first_idx)

        # Iteratively select farthest point
        min_distances = dist_matrix[first_idx].clone()

        for _ in range(num_samples - 1):
            # Mask already selected
            for idx in selected:
                min_distances[idx] = -float('inf')

            # Select farthest point
            farthest_idx = min_distances.argmax().item()
            selected.append(farthest_idx)

            # Update minimum distances
            min_distances = torch.minimum(min_distances, dist_matrix[farthest_idx])

        return torch.tensor(selected, dtype=torch.int64)

    def _uniform_sampling(
        self,
        num_views: int,
        num_samples: int,
    ) -> Int64[Tensor, " samples"]:
        """
        Sample views at uniform intervals.

        Assumes views are ordered (e.g., by azimuth angle).
        """
        num_samples = min(num_samples, num_views)
        indices = torch.linspace(0, num_views - 1, num_samples)
        return indices.long()

    def _random_sampling(
        self,
        num_views: int,
        num_samples: int,
    ) -> Int64[Tensor, " samples"]:
        """Randomly sample views."""
        num_samples = min(num_samples, num_views)
        perm = torch.randperm(num_views)
        return perm[:num_samples]

    @property
    def num_context_views(self) -> int:
        return self.cfg.num_context_views

    @property
    def num_target_views(self) -> int:
        return self.cfg.num_target_views
