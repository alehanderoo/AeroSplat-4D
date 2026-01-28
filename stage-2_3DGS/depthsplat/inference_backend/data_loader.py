"""
Data loader for the DepthSplat inference backend.

Handles loading examples from:
- Objaverse .torch files (test set)
- Custom image directories with camera parameters
"""

import json
import os
import random
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image


# 10 Hardcoded UUIDs for consistent testing
HARDCODED_UUIDS = [
    "acc210f3a2544cf1b250b45eaaf80160",
    "fcfc68f4adfa410a8b5855225a905c85",
    "f60358d2b71646598b1b961c35261f89",
    "00e32088111346ac84b9318b97e52c8e",
    "cd785cd85e5d495e97185587871ff975",
    "c6769a0400fb4496823eb5ef666adaa0",
    "3442d066a0e249138f1e407e7e6ec30c",
    "d9973e55e5e94e2c962d3494d848ad95",
    "8fea3562f05345f7ba7ba7f80dbff2f5",
    "dc0e741a15124119ae12efc17e50c593",
]


def farthest_point_sampling(
    positions: np.ndarray,
    num_samples: int,
    start_idx: int = 0,
) -> List[int]:
    """
    Greedily select views that maximize spatial coverage.

    This matches the training behavior in ViewSamplerObjectCentric.

    Args:
        positions: [V, 3] array of camera positions
        num_samples: Number of views to sample
        start_idx: Starting view index

    Returns:
        List of selected view indices
    """
    num_views = positions.shape[0]
    num_samples = min(num_samples, num_views)

    diff = positions[:, None, :] - positions[None, :, :]
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))

    selected = [start_idx]
    min_distances = dist_matrix[start_idx].copy()

    for _ in range(num_samples - 1):
        for idx in selected:
            min_distances[idx] = -float('inf')

        farthest_idx = np.argmax(min_distances)
        selected.append(int(farthest_idx))

        min_distances = np.minimum(min_distances, dist_matrix[farthest_idx])

    return selected


def decode_image_tensor(img_tensor: torch.Tensor) -> np.ndarray:
    """Decode a PNG-encoded tensor back to numpy array."""
    png_bytes = img_tensor.numpy().tobytes()
    image = Image.open(BytesIO(png_bytes))
    return np.array(image.convert('RGB'))


def parse_camera_tensor(camera: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse camera tensor from .torch format.

    Camera format (18 floats):
    - [0-3]: fx, fy, cx, cy (normalized intrinsics)
    - [4-5]: unused
    - [6-17]: w2c[:3, :] flattened (world-to-camera 3x4 matrix)

    Returns:
        Tuple of (extrinsics [4,4], intrinsics [3,3])
    """
    fx, fy, cx, cy = camera[0], camera[1], camera[2], camera[3]

    intrinsics = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1],
    ], dtype=np.float32)

    w2c_3x4 = camera[6:18].reshape(3, 4)
    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, :] = w2c_3x4

    c2w = np.linalg.inv(w2c)

    return c2w, intrinsics


def load_scene_from_torch(scene: Dict) -> Dict:
    """Load and parse a scene from the .torch format."""
    key = scene['key']

    images = [decode_image_tensor(img) for img in scene['images']]

    cameras = scene['cameras'].numpy()
    extrinsics = []
    intrinsics = []

    for cam in cameras:
        c2w, K = parse_camera_tensor(cam)
        extrinsics.append(c2w)
        intrinsics.append(K)

    extrinsics = np.stack(extrinsics)
    intrinsics = np.stack(intrinsics)

    result = {
        'key': key,
        'images': images,
        'extrinsics': extrinsics,
        'intrinsics': intrinsics,
    }

    if 'masks' in scene:
        result['masks'] = scene['masks'].numpy()

    if 'depths' in scene:
        result['depths'] = scene['depths'].numpy()

    return result


def load_torch_file(torch_path: str) -> List[Dict]:
    """Load all scenes from a .torch file."""
    data = torch.load(torch_path, map_location='cpu')
    return [load_scene_from_torch(scene) for scene in data]


def get_random_example(
    data_dir: str = "/mnt/raid0/objaverse/test",
    num_context_views: int = 5,
    seed: Optional[int] = None,
) -> Optional[Dict]:
    """Get a random example from the test dataset."""
    if seed is not None:
        random.seed(seed)

    torch_files = sorted(Path(data_dir).glob("*.torch"))
    if not torch_files:
        print(f"No .torch files found in {data_dir}")
        return None

    torch_file = random.choice(torch_files)

    scenes = load_torch_file(str(torch_file))
    if not scenes:
        print(f"No scenes in {torch_file}")
        return None

    scene = random.choice(scenes)

    num_views = len(scene['images'])
    if num_views <= num_context_views:
        selected_indices = list(range(num_views))
    else:
        selected_indices = np.linspace(0, num_views - 1, num_context_views, dtype=int).tolist()

    return {
        'key': scene['key'],
        'images': [scene['images'][i] for i in selected_indices],
        'extrinsics': scene['extrinsics'][selected_indices],
        'intrinsics': scene['intrinsics'][selected_indices],
        'all_extrinsics': scene['extrinsics'],
        'all_intrinsics': scene['intrinsics'],
        'source_file': str(torch_file),
    }


def get_example_by_uuid(
    data_dir: str,
    uuid: str,
    num_context_views: int = 5,
) -> Optional[Dict]:
    """
    Get a specific example by its UUID using farthest_point sampling.

    This function uses the same farthest_point sampling strategy as training.

    Args:
        data_dir: Directory containing .torch files
        uuid: Scene UUID to find
        num_context_views: Number of context views to sample

    Returns:
        Example dictionary with farthest_point sampled views
    """
    index_path = Path(data_dir) / "index.json"
    if not index_path.exists():
        print(f"No index.json found in {data_dir}")
        return None

    with open(index_path) as f:
        index = json.load(f)

    if uuid not in index:
        print(f"UUID {uuid} not found in index")
        return None

    torch_file = Path(data_dir) / index[uuid]
    scenes = load_torch_file(str(torch_file))

    for scene in scenes:
        if scene['key'] == uuid:
            num_views = len(scene['images'])

            if num_views <= num_context_views:
                selected_indices = list(range(num_views))
            else:
                camera_positions = scene['extrinsics'][:, :3, 3]
                selected_indices = farthest_point_sampling(
                    camera_positions,
                    num_context_views,
                    start_idx=0,
                )

            print(f"Loaded UUID {uuid}: selected views {selected_indices} from {num_views} total")

            return {
                'key': scene['key'],
                'images': [scene['images'][i] for i in selected_indices],
                'extrinsics': scene['extrinsics'][selected_indices],
                'intrinsics': scene['intrinsics'][selected_indices],
                'all_extrinsics': scene['extrinsics'],
                'all_intrinsics': scene['intrinsics'],
                'all_images': scene['images'],
                'selected_indices': selected_indices,
                'source_file': str(torch_file),
            }

    return None


def list_available_examples(
    data_dir: str = "/mnt/raid0/objaverse/test",
    max_examples: int = 10,
) -> List[str]:
    """List available example keys from the dataset."""
    keys = []

    index_path = Path(data_dir) / "index.json"
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        keys = list(index.keys())[:max_examples]
    else:
        for torch_file in sorted(Path(data_dir).glob("*.torch"))[:3]:
            scenes = load_torch_file(str(torch_file))
            for scene in scenes:
                keys.append(scene['key'])
                if len(keys) >= max_examples:
                    break
            if len(keys) >= max_examples:
                break

    return keys


def save_example_images(
    example: Dict,
    output_dir: str,
) -> List[str]:
    """Save example images to disk."""
    os.makedirs(output_dir, exist_ok=True)

    paths = []
    for i, img in enumerate(example['images']):
        path = os.path.join(output_dir, f"view_{i:02d}.png")
        Image.fromarray(img).save(path)
        paths.append(path)

    return paths
