"""
PyTorch Dataset for temporal sequences of Gaussian reconstructions.
"""

from pathlib import Path
from typing import Dict, List, Optional, Callable
import torch
from torch.utils.data import Dataset
import json
import logging

from .ply_parser import parse_ply, GaussianCloud
from .feature_extractor import GaussianFeatureExtractor

logger = logging.getLogger(__name__)


class GaussianSequenceDataset(Dataset):
    """
    Dataset for loading temporal sequences of 3D Gaussian reconstructions.

    Directory structure expected:
        data_root/
            sequence_001/
                frame_0001.ply
                frame_0002.ply
                ...
                metadata.json  # optional: contains label
            sequence_002/
                ...

    Or with separate label file:
        data_root/
            sequences/
                ...
            labels.json  # {"sequence_001": 0, "sequence_002": 1, ...}
    """

    def __init__(
        self,
        data_root: Path,
        sequence_length: int = 30,
        stride: int = 1,
        max_gaussians: Optional[int] = None,
        subsample_strategy: str = 'random',  # 'random', 'fps', 'grid', 'importance'
        labels_file: Optional[Path] = None,
        transform: Optional[Callable] = None,
        feature_extractor: Optional[GaussianFeatureExtractor] = None,
        cache_sequences: bool = False,
        file_pattern: str = "*.ply",
    ):
        """
        Args:
            data_root: Root directory containing sequences
            sequence_length: Number of frames per sample (T)
            stride: Frame stride for sampling
            max_gaussians: Maximum Gaussians per frame (subsample if exceeded)
            subsample_strategy: How to reduce Gaussians
            labels_file: Path to labels JSON (if not in per-sequence metadata)
            transform: Optional transform to apply to features
            feature_extractor: Feature extraction module
            cache_sequences: Whether to cache parsed sequences in memory
            file_pattern: Glob pattern for PLY files
        """
        self.data_root = Path(data_root)
        self.sequence_length = sequence_length
        self.stride = stride
        self.max_gaussians = max_gaussians
        self.subsample_strategy = subsample_strategy
        self.transform = transform
        self.feature_extractor = feature_extractor or GaussianFeatureExtractor()
        self.cache_sequences = cache_sequences
        self.file_pattern = file_pattern

        # Discover sequences
        self.sequences = self._discover_sequences()

        if len(self.sequences) == 0:
            logger.warning(f"No sequences found in {data_root}")

        # Load labels
        self.labels = self._load_labels(labels_file)

        # Cache storage
        self._cache: Dict[str, List[GaussianCloud]] = {}

    def _discover_sequences(self) -> List[Path]:
        """Find all sequence directories."""
        sequences = []
        if not self.data_root.exists():
            return sequences

        for d in sorted(self.data_root.iterdir()):
            if d.is_dir():
                ply_files = list(d.glob(self.file_pattern))
                if len(ply_files) >= self.sequence_length:
                    sequences.append(d)

        logger.info(f"Found {len(sequences)} sequences in {self.data_root}")
        return sequences

    def _load_labels(self, labels_file: Optional[Path]) -> Dict[str, int]:
        """Load labels from file or per-sequence metadata."""
        labels = {}

        if labels_file and Path(labels_file).exists():
            with open(labels_file) as f:
                labels = json.load(f)
        else:
            # Try per-sequence metadata
            for seq_dir in self.sequences:
                meta_file = seq_dir / "metadata.json"
                if meta_file.exists():
                    try:
                        with open(meta_file) as f:
                            meta = json.load(f)
                            if 'label' in meta:
                                labels[seq_dir.name] = meta['label']
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse metadata: {meta_file}")

        return labels

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            Dictionary with:
                - 'positions': (T, M, 3) tensor of positions (equivariant)
                - 'eigenvectors': (T, M, 3, 3) principal axes (equivariant)
                - 'scalars': (T, M, S) invariant scalar features
                - 'mask': (T, M) boolean mask for valid Gaussians
                - 'label': int (0=bird, 1=drone)
                - 'sequence_name': str
        """
        seq_dir = self.sequences[idx]
        seq_name = seq_dir.name

        # Load or retrieve from cache
        if self.cache_sequences and seq_name in self._cache:
            frames = self._cache[seq_name]
        else:
            frames = self._load_sequence(seq_dir)
            if self.cache_sequences:
                self._cache[seq_name] = frames

        # Sample frames according to stride
        total_frames = len(frames)
        max_start = max(0, total_frames - self.sequence_length * self.stride)
        start_idx = torch.randint(0, max_start + 1, (1,)).item() if max_start > 0 else 0

        frame_indices = list(range(start_idx, total_frames, self.stride))[:self.sequence_length]
        sampled_frames = [frames[i] for i in frame_indices]

        # Pad if needed
        while len(sampled_frames) < self.sequence_length:
            sampled_frames.append(sampled_frames[-1])

        # Extract features for each frame
        batch = self.feature_extractor.extract_batch(
            sampled_frames,
            max_gaussians=self.max_gaussians,
            subsample_strategy=self.subsample_strategy
        )

        # Apply transforms if any
        if self.transform:
            batch = self.transform(batch)

        # Add label
        label = self.labels.get(seq_name, -1)
        batch['label'] = torch.tensor(label, dtype=torch.long)
        batch['sequence_name'] = seq_name

        return batch

    def _load_sequence(self, seq_dir: Path) -> List[GaussianCloud]:
        """Load all frames in a sequence."""
        ply_files = sorted(seq_dir.glob(self.file_pattern))
        frames = []
        for f in ply_files:
            try:
                cloud = parse_ply(f)
                frames.append(cloud)
            except Exception as e:
                logger.warning(f"Failed to load {f}: {e}")

        return frames


class GaussianCollator:
    """
    Custom collator for variable-length Gaussian sequences.
    Pads sequences to maximum length in batch.
    """

    def __init__(self, pad_value: float = 0.0):
        self.pad_value = pad_value

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate batch with padding."""
        # Filter out items without valid data
        batch = [b for b in batch if b['positions'].shape[0] > 0]

        if len(batch) == 0:
            raise ValueError("Empty batch after filtering")

        # Get max dimensions
        max_T = max(b['positions'].shape[0] for b in batch)
        max_M = max(b['positions'].shape[1] for b in batch)

        B = len(batch)

        # Initialize padded tensors
        positions = torch.zeros(B, max_T, max_M, 3)
        scalars = torch.zeros(B, max_T, max_M, batch[0]['scalars'].shape[-1])
        eigenvectors = torch.zeros(B, max_T, max_M, 3, 3)
        mask = torch.zeros(B, max_T, max_M, dtype=torch.bool)
        labels = torch.zeros(B, dtype=torch.long)

        # Fill tensors
        for i, b in enumerate(batch):
            T, M = b['positions'].shape[:2]
            positions[i, :T, :M] = b['positions']
            scalars[i, :T, :M] = b['scalars']
            if 'eigenvectors' in b:
                eigenvectors[i, :T, :M] = b['eigenvectors']
            mask[i, :T, :M] = b['mask']
            labels[i] = b['label']

        result = {
            'positions': positions,
            'scalars': scalars,
            'eigenvectors': eigenvectors,
            'mask': mask,
            'labels': labels,
        }

        # Collect sequence names
        result['sequence_names'] = [b['sequence_name'] for b in batch]

        return result


class SyntheticGaussianDataset(Dataset):
    """
    Synthetic dataset for testing and debugging.

    Generates random Gaussian sequences with controllable properties.
    """

    def __init__(
        self,
        num_samples: int = 100,
        sequence_length: int = 30,
        num_gaussians: int = 1000,
        feature_dim: int = 16,
        num_classes: int = 2,
    ):
        self.num_samples = num_samples
        self.sequence_length = sequence_length
        self.num_gaussians = num_gaussians
        self.feature_dim = feature_dim
        self.num_classes = num_classes

        # Pre-generate labels
        self.labels = torch.randint(0, num_classes, (num_samples,))

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        T, M = self.sequence_length, self.num_gaussians

        # Generate random data
        positions = torch.randn(T, M, 3)
        eigenvectors = torch.randn(T, M, 3, 3)
        scalars = torch.randn(T, M, self.feature_dim)
        mask = torch.ones(T, M, dtype=torch.bool)

        return {
            'positions': positions,
            'eigenvectors': eigenvectors,
            'scalars': scalars,
            'mask': mask,
            'label': self.labels[idx],
            'sequence_name': f'synthetic_{idx}',
        }
