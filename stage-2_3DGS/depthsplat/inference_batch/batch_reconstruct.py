#!/usr/bin/env python3
"""
Batch 3D Gaussian Reconstruction from Isaac Sim Renders.

Processes all simulation sequences from renders/batch/ and generates
PLY files for each frame to be used by stage-3 classification.

Usage:
    python batch_reconstruct.py --input /path/to/renders/batch --output /path/to/output
    python batch_reconstruct.py --sequences rivermark_10m_15h_ip_cam_2k_DJI_Inspire_3
    python batch_reconstruct.py --frames 0-30  # Process only first 30 frames
"""

import os
import sys
import json
import shutil
import argparse
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import torch

# Add parent directories for imports
SCRIPT_DIR = Path(__file__).parent
DEPTHSPLAT_ROOT = SCRIPT_DIR.parent
if str(DEPTHSPLAT_ROOT) not in sys.path:
    sys.path.insert(0, str(DEPTHSPLAT_ROOT))

from inference_backend import (
    InferenceService,
    RenderSettings,
    VideoSettings,
    WildFrameLoader,
)


# Default paths
DEFAULT_INPUT_DIR = "/home/sandro/aeroSplat-4D/renders/batch"
DEFAULT_OUTPUT_DIR = "/home/sandro/aeroSplat-4D/stage-2_3DGS/output"
DEFAULT_CHECKPOINT = "/home/sandro/aeroSplat-4D/stage-2_3DGS/depthsplat/outputs/objaverse_white_small_gauss/checkpoints/epoch_0-step_100000.ckpt"
DEFAULT_CONFIG = "objaverse_white_small_gauss"


@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    input_dir: Path
    output_dir: Path
    checkpoint_path: Path
    config_name: str = DEFAULT_CONFIG
    sequences: Optional[List[str]] = None  # None = all sequences
    frame_range: Optional[tuple] = None  # (start, end) or None for all frames
    skip_existing: bool = True
    generate_videos: bool = False  # Disable videos for speed
    device: str = "cuda"
    cache_dir: Path = field(default_factory=lambda: Path("/tmp/depthsplat_batch"))


@dataclass
class ProcessingStats:
    """Track processing statistics."""
    total_sequences: int = 0
    total_frames: int = 0
    processed_frames: int = 0
    skipped_frames: int = 0
    failed_frames: int = 0
    start_time: float = field(default_factory=time.time)

    def elapsed_time(self) -> float:
        return time.time() - self.start_time

    def avg_time_per_frame(self) -> float:
        if self.processed_frames == 0:
            return 0.0
        return self.elapsed_time() / self.processed_frames

    def eta_seconds(self) -> float:
        remaining = self.total_frames - self.processed_frames - self.skipped_frames - self.failed_frames
        return remaining * self.avg_time_per_frame()


class BatchReconstructor:
    """Batch processor for 3D Gaussian reconstruction."""

    def __init__(self, config: BatchConfig):
        self.config = config
        self.service: Optional[InferenceService] = None
        self.stats = ProcessingStats()

        # Create output directory
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)

    def initialize_service(self) -> None:
        """Initialize the inference service."""
        print(f"Initializing DepthSplat inference service...")
        print(f"  Checkpoint: {self.config.checkpoint_path}")
        print(f"  Config: {self.config.config_name}")
        print(f"  Device: {self.config.device}")

        torch.set_float32_matmul_precision("medium")

        self.service = InferenceService.from_checkpoint(
            checkpoint_path=str(self.config.checkpoint_path),
            config_name=self.config.config_name,
            device=self.config.device,
        )
        print("Service initialized successfully.\n")

    def discover_sequences(self) -> List[Path]:
        """Discover all valid sequences in the input directory."""
        sequences = []

        for entry in sorted(self.config.input_dir.iterdir()):
            if not entry.is_dir():
                continue

            # Check for required files
            json_file = entry / "drone_camera_observations.json"
            render_complete = entry / ".render_complete"

            if json_file.exists() and render_complete.exists():
                # Filter by sequence names if specified
                if self.config.sequences is None or entry.name in self.config.sequences:
                    sequences.append(entry)
            else:
                if json_file.exists() and not render_complete.exists():
                    print(f"  Skipping {entry.name}: render not complete")

        return sequences

    def get_num_frames(self, sequence_dir: Path) -> int:
        """Get the number of frames in a sequence."""
        json_file = sequence_dir / "drone_camera_observations.json"
        with open(json_file, 'r') as f:
            data = json.load(f)
        return data.get('metadata', {}).get('num_frames', 120)

    def get_object_label(self, sequence_name: str) -> Dict[str, Any]:
        """
        Determine object label (drone/bird) from sequence name.

        Returns:
            Dictionary with 'name' (str) and 'id' (int: 0=bird, 1=drone)
        """
        # Bird assets typically have bird-related names
        bird_keywords = ['pigeon', 'grus', 'eagle', 'owl', 'seagull', 'bird']
        drone_keywords = ['vtol', 'uav', 'dji', 'quadcopter', 'quadcoptor', 'tarot', 'drone', 'matrice', 'inspire']

        name_lower = sequence_name.lower()

        for keyword in bird_keywords:
            if keyword in name_lower:
                return {'name': 'bird', 'id': 0}

        for keyword in drone_keywords:
            if keyword in name_lower:
                return {'name': 'drone', 'id': 1}

        return {'name': 'unknown', 'id': -1}

    def process_frame(
        self,
        sequence_dir: Path,
        frame_id: int,
        output_dir: Path,
        loader: WildFrameLoader,
    ) -> bool:
        """
        Process a single frame and save the PLY file.

        Returns:
            True if successful, False otherwise
        """
        ply_output_path = output_dir / f"frame_{frame_id:04d}.ply"

        # Skip if already exists
        if self.config.skip_existing and ply_output_path.exists():
            return True  # Count as success (skipped)

        try:
            # Load frame data
            frame_data = loader.load_frame(
                frame_id=frame_id,
                cache_dir=str(self.config.cache_dir / sequence_dir.name),
            )

            # Load images as numpy arrays
            images = loader.load_images(frame_data['image_paths'])

            # Create minimal settings (no video generation for speed)
            render_settings = RenderSettings(
                azimuth=0,
                elevation=frame_data['mean_elevation'],
                distance=1.0,
            )
            video_settings = VideoSettings(enabled=self.config.generate_videos)

            # Run reconstruction
            result = self.service.reconstruct(
                images=images,
                extrinsics=frame_data['extrinsics'],
                intrinsics=frame_data['intrinsics'],
                render_settings=render_settings,
                video_settings=video_settings,
                output_dir=str(self.config.cache_dir / sequence_dir.name / f"frame_{frame_id:04d}"),
            )

            # Copy PLY to output directory
            if result.ply_path and Path(result.ply_path).exists():
                shutil.copy2(result.ply_path, ply_output_path)
                return True
            else:
                print(f"    Warning: No PLY generated for frame {frame_id}")
                return False

        except Exception as e:
            print(f"    Error processing frame {frame_id}: {e}")
            return False

    def process_sequence(self, sequence_dir: Path) -> Dict[str, Any]:
        """
        Process all frames in a sequence.

        Returns:
            Dictionary with processing results
        """
        sequence_name = sequence_dir.name
        label_info = self.get_object_label(sequence_name)
        num_frames = self.get_num_frames(sequence_dir)

        print(f"\nProcessing sequence: {sequence_name}")
        print(f"  Label: {label_info['name']} (id={label_info['id']})")
        print(f"  Total frames: {num_frames}")

        # Determine frame range
        if self.config.frame_range:
            start_frame, end_frame = self.config.frame_range
            end_frame = min(end_frame, num_frames)
        else:
            start_frame, end_frame = 0, num_frames

        frames_to_process = list(range(start_frame, end_frame))
        print(f"  Processing frames {start_frame}-{end_frame-1} ({len(frames_to_process)} frames)")

        # Create output directory for this sequence
        output_dir = self.config.output_dir / sequence_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize loader for this sequence
        loader = WildFrameLoader(
            render_dir=str(sequence_dir),
            use_virtual_cameras=True,
        )

        # Process frames
        sequence_stats = {
            'name': sequence_name,
            'label': label_info['id'],
            'label_name': label_info['name'],
            'total_frames': len(frames_to_process),
            'processed': 0,
            'skipped': 0,
            'failed': 0,
            'frame_results': {},
        }

        for i, frame_id in enumerate(frames_to_process):
            ply_path = output_dir / f"frame_{frame_id:04d}.ply"

            # Check if already exists
            if self.config.skip_existing and ply_path.exists():
                sequence_stats['skipped'] += 1
                self.stats.skipped_frames += 1
                continue

            # Process frame
            print(f"  Frame {frame_id:04d} ({i+1}/{len(frames_to_process)})...", end=" ", flush=True)
            frame_start = time.time()

            success = self.process_frame(sequence_dir, frame_id, output_dir, loader)

            frame_time = time.time() - frame_start

            if success:
                sequence_stats['processed'] += 1
                self.stats.processed_frames += 1
                print(f"done ({frame_time:.1f}s)")
            else:
                sequence_stats['failed'] += 1
                self.stats.failed_frames += 1
                print(f"FAILED ({frame_time:.1f}s)")

            sequence_stats['frame_results'][frame_id] = success

            # Clear CUDA cache periodically
            if (i + 1) % 10 == 0:
                torch.cuda.empty_cache()

            # Progress update
            if (i + 1) % 20 == 0:
                eta = self.stats.eta_seconds()
                print(f"    Progress: {self.stats.processed_frames}/{self.stats.total_frames} frames, "
                      f"ETA: {eta/60:.1f} min")

        # Save sequence metadata
        # Note: 'label' must be integer for stage-3 compatibility (0=bird, 1=drone)
        metadata = {
            'sequence_name': sequence_name,
            'label': label_info['id'],  # Integer label for stage-3
            'label_name': label_info['name'],  # Human-readable label
            'num_frames': len(frames_to_process),
            'frame_range': [start_frame, end_frame],
            'timestamp': datetime.now().isoformat(),
            'stats': {
                'processed': sequence_stats['processed'],
                'skipped': sequence_stats['skipped'],
                'failed': sequence_stats['failed'],
            }
        }

        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        return sequence_stats

    def run(self) -> None:
        """Run the full batch processing pipeline."""
        print("=" * 70)
        print("DepthSplat Batch Reconstruction")
        print("=" * 70)
        print(f"Input directory:  {self.config.input_dir}")
        print(f"Output directory: {self.config.output_dir}")
        print(f"Skip existing:    {self.config.skip_existing}")
        print()

        # Discover sequences
        print("Discovering sequences...")
        sequences = self.discover_sequences()

        if not sequences:
            print("No valid sequences found!")
            return

        print(f"Found {len(sequences)} sequences:")
        for seq in sequences:
            label_info = self.get_object_label(seq.name)
            num_frames = self.get_num_frames(seq)
            print(f"  - {seq.name} [{label_info['name']}] ({num_frames} frames)")

        # Calculate total frames
        self.stats.total_sequences = len(sequences)
        for seq in sequences:
            num_frames = self.get_num_frames(seq)
            if self.config.frame_range:
                start, end = self.config.frame_range
                num_frames = min(end, num_frames) - start
            self.stats.total_frames += num_frames

        print(f"\nTotal frames to process: {self.stats.total_frames}")
        print()

        # Initialize service
        self.initialize_service()

        # Process each sequence
        results = []
        for seq in sequences:
            result = self.process_sequence(seq)
            results.append(result)

        # Print summary
        print("\n" + "=" * 70)
        print("BATCH PROCESSING COMPLETE")
        print("=" * 70)
        print(f"Total time: {self.stats.elapsed_time()/60:.1f} minutes")
        print(f"Sequences processed: {self.stats.total_sequences}")
        print(f"Frames processed: {self.stats.processed_frames}")
        print(f"Frames skipped: {self.stats.skipped_frames}")
        print(f"Frames failed: {self.stats.failed_frames}")
        if self.stats.processed_frames > 0:
            print(f"Average time per frame: {self.stats.avg_time_per_frame():.1f}s")

        # Save batch summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'input_dir': str(self.config.input_dir),
                'output_dir': str(self.config.output_dir),
                'checkpoint': str(self.config.checkpoint_path),
                'frame_range': list(self.config.frame_range) if self.config.frame_range else None,
                'skip_existing': self.config.skip_existing,
            },
            'stats': {
                'total_sequences': self.stats.total_sequences,
                'total_frames': self.stats.total_frames,
                'processed_frames': self.stats.processed_frames,
                'skipped_frames': self.stats.skipped_frames,
                'failed_frames': self.stats.failed_frames,
                'elapsed_seconds': self.stats.elapsed_time(),
            },
            'sequences': results,
        }

        summary_path = self.config.output_dir / "batch_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary saved to: {summary_path}")

        # Generate labels.json for stage-3 compatibility
        labels = {}
        for seq in sequences:
            label_info = self.get_object_label(seq.name)
            labels[seq.name] = label_info['id']

        labels_path = self.config.output_dir / "labels.json"
        with open(labels_path, 'w') as f:
            json.dump(labels, f, indent=2)
        print(f"Labels saved to: {labels_path}")

        # Print output structure for stage-3
        print("\nOutput structure for stage-3:")
        print(f"  {self.config.output_dir}/")
        for seq in sequences:
            label_info = self.get_object_label(seq.name)
            print(f"    {seq.name}/ [{label_info['name']}]")
            print(f"      frame_0000.ply")
            print(f"      frame_0001.ply")
            print(f"      ...")
            print(f"      metadata.json")


def parse_frame_range(range_str: str) -> tuple:
    """Parse frame range string like '0-30' into tuple (0, 30)."""
    if '-' in range_str:
        start, end = range_str.split('-')
        return (int(start), int(end))
    else:
        return (0, int(range_str))


def main():
    parser = argparse.ArgumentParser(
        description="Batch 3D Gaussian reconstruction from Isaac Sim renders",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input", "-i",
        type=str,
        default=DEFAULT_INPUT_DIR,
        help="Input directory containing simulation renders",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for PLY files",
    )
    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        default=DEFAULT_CHECKPOINT,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG,
        help="Model config name",
    )
    parser.add_argument(
        "--sequences", "-s",
        type=str,
        nargs="+",
        default=None,
        help="Specific sequences to process (default: all)",
    )
    parser.add_argument(
        "--frames", "-f",
        type=str,
        default=None,
        help="Frame range to process, e.g., '0-30' or '120' (default: all)",
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Reprocess existing PLY files",
    )
    parser.add_argument(
        "--videos",
        action="store_true",
        help="Generate 360° videos for each frame (slow)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run inference on",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="/tmp/depthsplat_batch",
        help="Cache directory for temporary files",
    )

    args = parser.parse_args()

    # Validate inputs
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input directory does not exist: {input_path}")
        sys.exit(1)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    # Parse frame range
    frame_range = None
    if args.frames:
        frame_range = parse_frame_range(args.frames)
        print(f"Frame range: {frame_range[0]} to {frame_range[1]}")

    # Create config
    config = BatchConfig(
        input_dir=input_path,
        output_dir=Path(args.output),
        checkpoint_path=checkpoint_path,
        config_name=args.config,
        sequences=args.sequences,
        frame_range=frame_range,
        skip_existing=not args.no_skip,
        generate_videos=args.videos,
        device=args.device,
        cache_dir=Path(args.cache_dir),
    )

    # Run batch processing
    processor = BatchReconstructor(config)
    processor.run()


if __name__ == "__main__":
    main()
