from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class VoxelizerConfig:
    """
    Runtime configuration for the voxel overlap builder.

    Attributes:
        dataset_root: Root directory containing cam_0X folders and metadata JSON.
        metadata_json: Optional explicit path to drone_camera_observations.json.
        frame_stride: Subsampling factor for temporal frames when aggregating stats.
        resolution: Target number of voxels along the longest axis of the grid.
        points_per_side: Overrides resolution when provided (legacy compatibility).
        margin_meters: Extra padding around the observed bounding box.
        min_cameras: Minimum number of cameras that must see a voxel to keep it.
        use_cuda: Force-enable / disable CUDA. If None, auto-detect.
        chunk_size: Number of voxels to evaluate per CUDA batch to manage memory.
        color_sample_frame: Index of frame to use when sampling colours (optional).
    """

    dataset_root: Path
    metadata_json: Optional[Path] = None
    frame_stride: int = 4
    resolution: int = 160
    points_per_side: Optional[int] = None
    margin_meters: float = 0.5
    min_cameras: int = 3
    use_cuda: Optional[bool] = None
    chunk_size: int = 2_000_000
    color_sample_frame: Optional[int] = 0

    def __post_init__(self) -> None:
        self.dataset_root = Path(self.dataset_root).expanduser().resolve()
        if self.metadata_json is None:
            self.metadata_json = self.dataset_root / "drone_camera_observations.json"
        else:
            self.metadata_json = Path(self.metadata_json).expanduser().resolve()

        if not self.dataset_root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.dataset_root}")
        if not self.metadata_json.exists():
            raise FileNotFoundError(f"Metadata JSON not found: {self.metadata_json}")

        if self.frame_stride < 1:
            raise ValueError("frame_stride must be >= 1")

        if self.points_per_side is not None:
            self.resolution = self.points_per_side

