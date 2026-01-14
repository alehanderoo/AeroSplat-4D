"""
Objaverse dataset loader for object-centric 3D Gaussian Splatting.

This dataset is designed for training on Objaverse objects with:
- Multiple views per object (typically 32 views around the object)
- Optional segmentation masks for silhouette supervision
- Optional ground truth depth maps for depth supervision
- Object-centric camera configurations (cameras looking at object center)
"""

import json
from dataclasses import dataclass
from functools import cached_property
from io import BytesIO
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
import torchvision.transforms as tf
from einops import rearrange, repeat
from jaxtyping import Float, UInt8
from PIL import Image
from torch import Tensor
from torch.utils.data import IterableDataset

from ..geometry.projection import get_fov
from .dataset import DatasetCfgCommon
from .shims.augmentation_shim import apply_augmentation_shim
from .shims.crop_shim import apply_crop_shim
from .types import Stage
from .view_sampler import ViewSampler


@dataclass
class DatasetObjaverseCfg(DatasetCfgCommon):
    name: Literal["objaverse"]
    roots: list[Path]
    baseline_epsilon: float
    max_fov: float
    augment: bool
    test_len: int
    test_chunk_interval: int
    skip_bad_shape: bool = True

    # Object-centric depth bounds (tighter than scene-level)
    near: float = 0.5
    far: float = 5.0

    # Data options
    use_masks: bool = True
    use_depth: bool = True
    shuffle_val: bool = True
    train_times_per_scene: int = 1


class DatasetObjaverse(IterableDataset):
    """
    Objaverse dataset for object-centric 3D Gaussian Splatting.

    Key differences from RE10K/DL3DV:
    - Object-centric: cameras arranged around a central object
    - Includes segmentation masks for silhouette loss
    - Includes ground truth depth for depth supervision
    - Tighter depth bounds suitable for objects (vs room-scale scenes)
    """

    cfg: DatasetObjaverseCfg
    stage: Stage
    view_sampler: ViewSampler

    to_tensor: tf.ToTensor
    chunks: list[Path]
    near: float = 0.5
    far: float = 5.0

    def __init__(
        self,
        cfg: DatasetObjaverseCfg,
        stage: Stage,
        view_sampler: ViewSampler,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()

        # Set depth bounds
        if cfg.near != -1:
            self.near = cfg.near
        if cfg.far != -1:
            self.far = cfg.far

        # Collect chunks
        self.chunks = []
        for root in cfg.roots:
            root = Path(root) / self.data_stage
            if not root.exists():
                print(f"Warning: {root} does not exist")
                continue

            root_chunks = sorted(
                [path for path in root.iterdir() if path.suffix == ".torch"]
            )
            self.chunks.extend(root_chunks)

        if self.cfg.overfit_to_scene is not None:
            chunk_path = self.index[self.cfg.overfit_to_scene]
            self.chunks = [chunk_path] * len(self.chunks)

        if self.stage == "test":
            # Testing on a subset for faster evaluation
            self.chunks = self.chunks[::cfg.test_chunk_interval]

        print(f"DatasetObjaverse [{stage}]: {len(self.chunks)} chunks, "
              f"near={self.near}, far={self.far}")

    def shuffle(self, lst: list) -> list:
        indices = torch.randperm(len(lst))
        return [lst[x] for x in indices]

    def __iter__(self):
        # Shuffle chunks for training
        if self.stage in (("train", "val") if self.cfg.shuffle_val else ("train")):
            self.chunks = self.shuffle(self.chunks)

        # Handle multi-worker data loading
        worker_info = torch.utils.data.get_worker_info()
        if self.stage == "test" and worker_info is not None:
            self.chunks = [
                chunk
                for chunk_index, chunk in enumerate(self.chunks)
                if chunk_index % worker_info.num_workers == worker_info.id
            ]

        for chunk_path in self.chunks:
            # Load the chunk
            chunk = torch.load(chunk_path)

            if self.cfg.overfit_to_scene is not None:
                item = [x for x in chunk if x["key"] == self.cfg.overfit_to_scene]
                assert len(item) == 1
                chunk = item * len(chunk)

            if self.stage in (("train", "val") if self.cfg.shuffle_val else ("train")):
                chunk = self.shuffle(chunk)

            times_per_scene = (
                1 if self.stage == "test" else self.cfg.train_times_per_scene
            )

            for run_idx in range(int(times_per_scene * len(chunk))):
                example = chunk[run_idx // times_per_scene]
                extrinsics, intrinsics = self.convert_poses(example["cameras"])
                scene = example["key"]

                try:
                    context_indices, target_indices = self.view_sampler.sample(
                        scene,
                        extrinsics,
                        intrinsics,
                    )
                except ValueError:
                    # Skip if not enough frames
                    continue

                # Skip if field of view is too wide
                if (get_fov(intrinsics).rad2deg() > self.cfg.max_fov).any():
                    continue

                # Load images
                context_images = [
                    example["images"][index.item()] for index in context_indices
                ]
                context_images = self.convert_images(context_images)
                target_images = [
                    example["images"][index.item()] for index in target_indices
                ]
                target_images = self.convert_images(target_images)

                # Skip if images have wrong shape
                expected_h, expected_w = self.cfg.image_shape
                context_shape = context_images.shape[1:]
                target_shape = target_images.shape[1:]

                if self.cfg.skip_bad_shape:
                    if context_shape[1:] != (expected_h, expected_w):
                        print(f"Skipping {scene}: context shape {context_shape}")
                        continue
                    if target_shape[1:] != (expected_h, expected_w):
                        print(f"Skipping {scene}: target shape {target_shape}")
                        continue

                # Build example dict
                result = {
                    "context": {
                        "extrinsics": extrinsics[context_indices],
                        "intrinsics": intrinsics[context_indices],
                        "image": context_images,
                        "near": self.get_bound("near", len(context_indices)),
                        "far": self.get_bound("far", len(context_indices)),
                        "index": context_indices,
                    },
                    "target": {
                        "extrinsics": extrinsics[target_indices],
                        "intrinsics": intrinsics[target_indices],
                        "image": target_images,
                        "near": self.get_bound("near", len(target_indices)),
                        "far": self.get_bound("far", len(target_indices)),
                        "index": target_indices,
                    },
                    "scene": scene,
                }

                # Add masks if available
                if self.cfg.use_masks and "masks" in example:
                    context_masks = example["masks"][context_indices]
                    target_masks = example["masks"][target_indices]
                    result["context"]["mask"] = context_masks
                    result["target"]["mask"] = target_masks

                # Add depth if available
                if self.cfg.use_depth and "depths" in example:
                    context_depths = example["depths"][context_indices]
                    target_depths = example["depths"][target_indices]
                    result["context"]["depth"] = context_depths
                    result["target"]["depth"] = target_depths

                if self.stage == "train" and self.cfg.augment:
                    result = apply_augmentation_shim(result)

                yield apply_crop_shim(result, tuple(self.cfg.image_shape))

    def convert_poses(
        self,
        poses: Float[Tensor, "batch 18"],
    ) -> tuple[
        Float[Tensor, "batch 4 4"],  # extrinsics (C2W)
        Float[Tensor, "batch 3 3"],  # intrinsics (normalized)
    ]:
        """Convert stored camera tensors to extrinsics and intrinsics."""
        b, _ = poses.shape

        # Extract normalized intrinsics
        intrinsics = torch.eye(3, dtype=torch.float32)
        intrinsics = repeat(intrinsics, "h w -> b h w", b=b).clone()
        fx, fy, cx, cy = poses[:, :4].T
        intrinsics[:, 0, 0] = fx
        intrinsics[:, 1, 1] = fy
        intrinsics[:, 0, 2] = cx
        intrinsics[:, 1, 2] = cy

        # Extract W2C matrix and convert to C2W
        w2c = repeat(torch.eye(4, dtype=torch.float32), "h w -> b h w", b=b).clone()
        w2c[:, :3] = rearrange(poses[:, 6:], "b (h w) -> b h w", h=3, w=4)

        # Return C2W (camera-to-world) extrinsics
        c2w = w2c.inverse()
        return c2w, intrinsics

    def convert_images(
        self,
        images: list[UInt8[Tensor, "..."]],
    ) -> Float[Tensor, "batch 3 height width"]:
        """Convert stored image tensors to float tensors."""
        torch_images = []
        for image in images:
            image = Image.open(BytesIO(image.numpy().tobytes()))
            torch_images.append(self.to_tensor(image))
        return torch.stack(torch_images)

    def get_bound(
        self,
        bound: Literal["near", "far"],
        num_views: int,
    ) -> Float[Tensor, " view"]:
        value = torch.tensor(getattr(self, bound), dtype=torch.float32)
        return repeat(value, "-> v", v=num_views)

    @property
    def data_stage(self) -> Stage:
        if self.cfg.overfit_to_scene is not None:
            return "test"
        if self.stage == "val":
            return "test"
        return self.stage

    @cached_property
    def index(self) -> dict[str, Path]:
        """Build index mapping scene keys to chunk paths."""
        merged_index = {}
        data_stages = [self.data_stage]
        if self.cfg.overfit_to_scene is not None:
            data_stages = ("test", "train")

        for data_stage in data_stages:
            for root in self.cfg.roots:
                root = Path(root)
                index_path = root / data_stage / "index.json"
                if not index_path.exists():
                    continue

                with index_path.open("r") as f:
                    index = json.load(f)
                index = {k: Path(root / data_stage / v) for k, v in index.items()}

                # Merge indices (should have unique keys)
                merged_index = {**merged_index, **index}

        return merged_index

    def __len__(self) -> int:
        return (
            min(len(self.index.keys()), self.cfg.test_len)
            if self.stage == "test" and self.cfg.test_len > 0
            else len(self.index.keys()) * self.cfg.train_times_per_scene
        )
