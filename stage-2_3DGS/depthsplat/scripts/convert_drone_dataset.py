#!/usr/bin/env python3
"""Convert drone captures into the chunked format expected by DatasetRE10k.

This script scans a folder for files named ``cam_##_frame_####.png`` plus the
associated ``drone_camera_observations.json`` metadata, normalizes the camera
parameters, and packs them into the chunk/index layout that the RE10K data
loader consumes."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import torch

TARGET_BYTES_PER_CHUNK = int(1e8)
IMAGE_PATTERN = re.compile(r"cam_(\d+)_frame_(\d+)\.(png|jpg|jpeg)$", re.IGNORECASE)
USD_TO_OPENCV = np.array(
    [
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)


@dataclass
class CameraRecord:
    name: str
    width: int
    height: int
    intrinsics: list[list[float]]
    world_to_camera: list[list[float]]

    @classmethod
    def from_json(cls, payload: dict) -> "CameraRecord":
        return cls(
            name=payload["name"],
            width=int(payload["resolution"]["width"]),
            height=int(payload["resolution"]["height"]),
            intrinsics=payload["intrinsics"]["matrix"],
            world_to_camera=payload["extrinsics"]["world_to_camera_matrix"],
        )

    def to_pose_vector(self) -> torch.Tensor:
        fx = self.intrinsics[0][0] / self.width
        fy = self.intrinsics[1][1] / self.height
        cx = self.intrinsics[0][2] / self.width
        cy = self.intrinsics[1][2] / self.height

        w2c = torch.tensor(self.world_to_camera, dtype=torch.float32)
        # Replicator/Omniverse cameras use +Z forward, +Y up (USD/OpenGL style).
        # MVSplat expects OpenCV-style extrinsics (+Z forward, +X right, -Y up).
        # Yawing the USD matrix by 180° around the Y axis converts between both.
        w2c = torch.from_numpy(USD_TO_OPENCV) @ w2c

        pose = torch.zeros(18, dtype=torch.float32)
        pose[:4] = torch.tensor([fx, fy, cx, cy], dtype=torch.float32)
        pose[6:] = w2c[:3].reshape(-1)
        return pose


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Folder that contains the drone_* images and drone_camera_observations.json.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Destination root (e.g. datasets/drone_5cams_sky).",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="test",
        help="Sub-folder name under output_dir (defaults to 'test').",
    )
    parser.add_argument(
        "--scene_prefix",
        type=str,
        default=None,
        help="Optional prefix for scene keys; defaults to the input directory name.",
    )
    return parser.parse_args()


def load_image_bytes(path: Path) -> torch.Tensor:
    data = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    return torch.from_numpy(data.copy())


def collect_frames(image_dir: Path) -> Dict[int, Dict[str, Path]]:
    frames: Dict[int, Dict[str, Path]] = {}
    for path in image_dir.glob("*"):
        match = IMAGE_PATTERN.match(path.name)
        if not match:
            continue
        cam_idx = int(match.group(1))
        frame_idx = int(match.group(2))
        cam_name = f"cam_{cam_idx:02d}"
        frames.setdefault(frame_idx, {})[cam_name] = path
    return frames


def build_examples(
    frames: Dict[int, Dict[str, Path]],
    cameras: Dict[str, CameraRecord],
    scene_prefix: str,
) -> list[dict]:
    required_cams = sorted(cameras.keys())
    examples = []
    for frame_idx in sorted(frames.keys()):
        frame_images = frames[frame_idx]
        if not all(name in frame_images for name in required_cams):
            continue
        images = [load_image_bytes(frame_images[name]) for name in required_cams]
        poses = torch.stack([cameras[name].to_pose_vector() for name in required_cams])
        timestamps = torch.arange(len(required_cams), dtype=torch.int64)
        examples.append(
            {
                "key": f"{scene_prefix}_frame_{frame_idx:04d}",
                "url": "",
                "timestamps": timestamps,
                "cameras": poses,
                "images": images,
            }
        )
    return examples


def chunk_and_save(
    examples: Iterable[dict],
    output_stage_dir: Path,
) -> None:
    output_stage_dir.mkdir(parents=True, exist_ok=True)
    index: Dict[str, str] = {}
    chunk: list[dict] = []
    chunk_size = 0
    chunk_idx = 0

    def flush() -> None:
        nonlocal chunk, chunk_size, chunk_idx
        if not chunk:
            return
        chunk_name = f"{chunk_idx:06d}.torch"
        torch.save(chunk, output_stage_dir / chunk_name)
        for example in chunk:
            index[example["key"]] = chunk_name
        chunk_idx += 1
        chunk.clear()
        chunk_size = 0

    for example in examples:
        example_bytes = sum(int(image.numel()) for image in example["images"])
        chunk.append(example)
        chunk_size += example_bytes
        if chunk_size >= TARGET_BYTES_PER_CHUNK:
            flush()

    flush()

    if not index:
        raise RuntimeError("No examples were written. Did you provide any complete frames?")

    with (output_stage_dir / "index.json").open("w") as f:
        json.dump(index, f, indent=2)

    print(f"Wrote {len(index)} scenes to {output_stage_dir}.")


def main() -> None:
    args = parse_args()
    input_dir: Path = args.input_dir
    meta_path = input_dir / "drone_camera_observations.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Could not find {meta_path}")

    metadata = json.loads(meta_path.read_text())
    cameras = {
        cam["name"]: CameraRecord.from_json(cam)
        for cam in metadata["cameras"]
    }

    frames = collect_frames(input_dir)
    if not frames:
        raise RuntimeError(f"No frames found in {input_dir}.")

    scene_prefix = args.scene_prefix or input_dir.name
    examples = build_examples(frames, cameras, scene_prefix)
    if not examples:
        raise RuntimeError(
            "No valid scenes were built. Ensure each frame has images for all cameras."
        )

    output_stage_dir = args.output_dir / args.stage
    chunk_and_save(examples, output_stage_dir)


if __name__ == "__main__":
    main()
