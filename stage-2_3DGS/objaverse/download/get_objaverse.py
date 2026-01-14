#!/usr/bin/env python3
"""
Download the entire Objaverse dataset.

This script downloads all 3D objects from the Objaverse dataset using multiprocessing.
Objects are saved as .glb files locally.
"""

import multiprocessing
import contextlib
import re
from pathlib import Path

import objaverse
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================
# Set to True to download only high-quality filtered objects from high_quality_uids.txt
# Set to False to download all objects from the Objaverse dataset
USE_HIGH_QUALITY_FILTER = True
# HIGH_QUALITY_UIDS_FILE = "high_quality_uids.txt"
# HIGH_QUALITY_UIDS_FILE = "trainingsetB_part1.txt"
# HIGH_QUALITY_UIDS_FILE = "trainingsetB_part2.txt"
HIGH_QUALITY_UIDS_FILE = "trainingsetB_part3.txt"

GLB_DIR_CANDIDATES = [
    Path(__file__).resolve().parent / ".objaverse" / "hf-objaverse-v1" / "glbs",
    Path.home() / ".objaverse" / "hf-objaverse-v1" / "glbs",
]
# ============================================================================

def count_existing_downloads():
    """Return (count, path) for existing downloaded .glb files."""
    for candidate in GLB_DIR_CANDIDATES:
        if candidate.exists():
            count = sum(1 for _ in candidate.rglob("*.glb"))
            return count, candidate
    return 0, GLB_DIR_CANDIDATES[0]

class TqdmCapture:
    """Capture stdout and update tqdm based on objaverse progress messages."""
    def __init__(self, pbar):
        self.pbar = pbar
        self.last_count = 0
        self.pattern = re.compile(r'Downloaded (\d+) / \d+ objects')
        self.buffer = ""

    def write(self, text):
        if not text:
            return

        self.buffer += text.replace("\r", "\n")
        self._process_buffer()

    def flush(self):
        self._process_buffer(final=True)

    def _process_buffer(self, final=False):
        split_token = "\n"
        while split_token in self.buffer:
            line, self.buffer = self.buffer.split(split_token, 1)
            self._handle_line(line)

        if final and self.buffer:
            self._handle_line(self.buffer)
            self.buffer = ""

    def _handle_line(self, line):
        match = self.pattern.search(line)
        if match:
            current = int(match.group(1))
            if current > self.last_count:
                self.pbar.update(current - self.last_count)
                self.last_count = current

def main():
    print("Objaverse Dataset Downloader")
    print("=" * 50)
    
    # Load UIDs based on filter setting
    if USE_HIGH_QUALITY_FILTER:
        print(f"Loading high-quality filtered UIDs from '{HIGH_QUALITY_UIDS_FILE}'...")
        uids_file = Path(HIGH_QUALITY_UIDS_FILE)
        if not uids_file.exists():
            print(f"ERROR: {HIGH_QUALITY_UIDS_FILE} not found!")
            print(f"Please run the analyze_parquet.ipynb notebook to generate this file,")
            print(f"or set USE_HIGH_QUALITY_FILTER = False to download all objects.")
            return
        
        with open(uids_file, 'r') as f:
            uids = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(uids)} high-quality filtered UIDs")
    else:
        print("Loading all object UIDs...")
        uids = objaverse.load_uids()
        print(f"Total objects to download: {len(uids)}")
    
    existing_count, download_dir = count_existing_downloads()
    if existing_count:
        print(f"Found {existing_count} previously downloaded objects in '{download_dir}'")
    else:
        print("No previously downloaded objects found. Starting fresh download.")

    # Get number of processes (CPU count)
    num_processes = multiprocessing.cpu_count()
    print(f"Using {num_processes} processes for download")
    
    # Download all objects
    print("\nStarting download... This may take a significant amount of time and disk space!")
    
    initial_progress = min(existing_count, len(uids))

    with tqdm(total=len(uids), desc="Downloading objects", unit="object", initial=initial_progress) as pbar:
        capture = TqdmCapture(pbar)
        with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
            objects = objaverse.load_objects(
                uids=uids,
                download_processes=num_processes
            )

        capture.flush()
        remaining = len(uids) - pbar.n
        if remaining > 0:
            pbar.update(remaining)
    
    print(f"Downloaded {len(objects)} objects successfully")
    
    # Load annotations
    print("\nLoading annotations...")
    annotations = objaverse.load_annotations()
    print(f"Loaded annotations for {len(annotations)} objects")
    
    # Load LVIS annotations
    print("Loading LVIS annotations...")
    lvis_annotations = objaverse.load_lvis_annotations()
    print(f"LVIS categories: {len(lvis_annotations)}")
    for category, count in list(lvis_annotations.items())[:5]:
        print(f"  - {category}: {len(count)} objects")
    print("  ...")
    
    print("\n" + "=" * 50)
    print("Download complete!")

if __name__ == "__main__":
    main()
