#!/usr/bin/env python3

"""
batch_render.py - Batch Rendering Orchestrator

This script orchestrates batch rendering by:
1. Reading batch_config.yaml to get parameter sweeps
2. Generating all parameter combinations
3. Launching headless Isaac Sim for each combination

Usage:
    # Run all combinations:
    python batch_render.py

    # Dry run (print what would be rendered):
    python batch_render.py --dry-run

    # Filter by asset type:
    python batch_render.py --asset-type bird
    python batch_render.py --asset-type drone

    # Filter by specific assets:
    python batch_render.py --assets "bald-eagle-med-poly.usdc,DJI Inspire 3.usdc"

    # Specify Isaac Sim path:
    python batch_render.py --isaac-sim-path ~/.local/share/ov/pkg/isaac-sim-4.2.0
"""

import argparse
import itertools
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).parent.resolve()


def load_batch_config(config_path=None):
    """Load batch configuration from YAML."""
    if config_path is None:
        config_path = SCRIPT_DIR / "batch_config.yaml"

    config_path = Path(config_path)
    if not config_path.exists():
        print(f"[BATCH] ERROR: Batch config not found: {config_path}")
        return None

    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_asset_config():
    """Load asset configuration to get USD paths and types."""
    asset_config_path = SCRIPT_DIR / "asset_config.yaml"
    if not asset_config_path.exists():
        print(f"[BATCH] ERROR: Asset config not found: {asset_config_path}")
        return None

    with open(asset_config_path, 'r') as f:
        return yaml.safe_load(f)


def get_available_assets(asset_config):
    """Get list of all available assets with their types."""
    assets = []
    for name, data in asset_config.items():
        assets.append({
            "name": name,
            "type": data.get("type", "unknown"),
            "usd_path": data.get("usd_path", ""),
        })
    return assets


def filter_assets(assets, batch_cfg, args):
    """Filter assets based on batch config and command line args."""
    # Get assets from batch config
    config_assets = batch_cfg.get("assets", [])

    # If "all" specified, use all assets
    if config_assets == "all" or (len(config_assets) == 1 and config_assets[0] == "all"):
        filtered = [a["name"] for a in assets]
    else:
        # Use only specified assets
        available_names = {a["name"] for a in assets}
        filtered = []
        for asset_name in config_assets:
            if asset_name in available_names:
                filtered.append(asset_name)
            else:
                print(f"[BATCH] WARNING: Asset '{asset_name}' not found in asset_config.yaml")

    # Apply command line filters
    if args.assets:
        requested = [a.strip() for a in args.assets.split(",")]
        filtered = [a for a in filtered if a in requested]

    if args.asset_type:
        asset_types = {a["name"]: a["type"] for a in assets}
        filtered = [a for a in filtered if asset_types.get(a) == args.asset_type]

    return filtered


def generate_combinations(batch_cfg, assets):
    """Generate all parameter combinations for batch rendering."""
    scenes = batch_cfg.get("scenes", [])
    side_meters = batch_cfg.get("side_meters", [10.0])
    cam_heights = batch_cfg.get("cam_heights", [7.0])
    flight_heights = batch_cfg.get("flight_height_offsets", [15.0])
    rotation_offsets = batch_cfg.get("rotation_offsets", [0.0])
    camera_types = batch_cfg.get("camera_types", ["ip_cam_2k"])

    # Generate all combinations
    combinations = []
    for asset, scene, side, cam_h, flight_h, rot, cam_type in itertools.product(
        assets, scenes, side_meters, cam_heights, flight_heights, rotation_offsets, camera_types
    ):
        combinations.append({
            "asset": asset,
            "scene_url": scene.get("url", scene) if isinstance(scene, dict) else scene,
            "scene_name": scene.get("name", "scene") if isinstance(scene, dict) else "scene",
            "side_meters": side,
            "cam_height": cam_h,
            "flight_height": flight_h,
            "rotation_offset": rot,
            "camera_type": cam_type,
        })

    return combinations


def get_output_dir(combo, batch_cfg):
    """Generate output directory path for a combination."""
    output_cfg = batch_cfg.get("output", {})
    base_path = output_cfg.get("base_path", "/home/sandro/aeroSplat-4D/renders")
    pattern = output_cfg.get("folder_pattern", "{asset}_{scene}_{side_m}m_{height_m}h_{cam_type}")

    # Clean asset name for filesystem
    asset_clean = Path(combo["asset"]).stem
    asset_clean = re.sub(r'[^\w\-]', '_', asset_clean)

    # Format the pattern
    folder_name = pattern.format(
        asset=asset_clean,
        scene=combo["scene_name"],
        side_m=int(combo["side_meters"]),
        height_m=int(combo["flight_height"]),
        cam_type=combo["camera_type"],
    )

    return os.path.join(base_path, folder_name)


def is_already_rendered(output_dir, batch_cfg):
    """Check if a render has already been completed."""
    if not batch_cfg.get("execution", {}).get("resume", True):
        return False

    output_path = Path(output_dir)
    if not output_path.exists():
        return False

    # Check for completion marker or minimum files
    completion_marker = output_path / ".render_complete"
    if completion_marker.exists():
        return True

    # Check if we have rendered frames (at least camera_0 folder with images)
    cam0_dir = output_path / "camera_0"
    if cam0_dir.exists():
        rgb_files = list(cam0_dir.glob("rgb_*.png"))
        expected_frames = batch_cfg.get("render", {}).get("num_frames", 120)
        if len(rgb_files) >= expected_frames:
            return True

    return False


def find_isaac_sim_python(isaac_sim_path=None):
    """Find Isaac Sim's Python interpreter."""
    # Common Isaac Sim locations (check ~/isaacsim first as per user's setup)
    search_paths = [
        isaac_sim_path,
        os.environ.get("ISAAC_SIM_PATH"),
        os.path.expanduser("~/isaacsim"),
        os.path.expanduser("~/.local/share/ov/pkg/isaac-sim-4.5.0"),
        os.path.expanduser("~/.local/share/ov/pkg/isaac-sim-4.2.0"),
        os.path.expanduser("~/.local/share/ov/pkg/isaac-sim-4.1.0"),
        os.path.expanduser("~/.local/share/ov/pkg/isaac-sim-4.0.0"),
        "/isaac-sim",
    ]

    for path in search_paths:
        if path is None:
            continue
        python_sh = Path(path) / "python.sh"
        if python_sh.exists():
            return str(python_sh)

    return None


def run_depth_crop(output_dir, keep_originals=False):
    """Run depth cropping post-processing on a render directory."""
    try:
        from crop_depth import process_render_directory
        stats = process_render_directory(
            Path(output_dir),
            padding=10,
            keep_originals=keep_originals,
            verbose=False
        )
        if stats.get("storage_saved_mb"):
            print(f"[BATCH]   Depth cropping saved {stats['storage_saved_mb']:.1f} MB")
        return True
    except ImportError:
        print("[BATCH]   WARNING: crop_depth.py not found, skipping depth cropping")
        return False
    except Exception as e:
        print(f"[BATCH]   WARNING: Depth cropping failed: {e}")
        return False


def run_headless_render(combo, batch_cfg, isaac_python, dry_run=False, crop_depth=False, keep_full_depth=False):
    """Launch a single headless render."""
    output_dir = get_output_dir(combo, batch_cfg)
    render_cfg = batch_cfg.get("render", {})

    # Build command
    headless_script = SCRIPT_DIR / "headless_runner.py"
    cmd = [
        isaac_python,
        str(headless_script),
        "--asset", combo["asset"],
        "--scene", combo["scene_url"],
        "--side-meters", str(combo["side_meters"]),
        "--cam-height", str(combo["cam_height"]),
        "--flight-height", str(combo["flight_height"]),
        "--rotation-offset", str(combo["rotation_offset"]),
        "--camera-type", combo["camera_type"],
        "--num-frames", str(render_cfg.get("num_frames", 120)),
        "--num-cameras", str(render_cfg.get("num_cameras", 5)),
        "--output-dir", output_dir,
    ]

    if batch_cfg.get("execution", {}).get("verbose", True):
        cmd.append("--verbose")

    print(f"\n[BATCH] Rendering: {combo['asset']}")
    print(f"[BATCH]   Scene: {combo['scene_name']}")
    print(f"[BATCH]   Side: {combo['side_meters']}m, Height: {combo['flight_height']}m")
    print(f"[BATCH]   Camera: {combo['camera_type']}")
    print(f"[BATCH]   Output: {output_dir}")

    if dry_run:
        print(f"[BATCH]   Command: {' '.join(cmd)}")
        return True

    # Run the command
    try:
        result = subprocess.run(
            cmd,
            check=False,
            capture_output=False,
        )

        if result.returncode == 0:
            # Mark as complete
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            (Path(output_dir) / ".render_complete").touch()
            print(f"[BATCH] SUCCESS: {combo['asset']}")

            # Post-processing: crop depth maps if enabled
            if crop_depth:
                print(f"[BATCH]   Running depth cropping post-process...")
                run_depth_crop(output_dir, keep_originals=keep_full_depth)

            return True
        else:
            print(f"[BATCH] FAILED: {combo['asset']} (exit code {result.returncode})")
            return False

    except Exception as e:
        print(f"[BATCH] ERROR: {combo['asset']} - {e}")
        return False


def save_batch_manifest(combinations, batch_cfg, output_path):
    """Save a manifest of all render combinations."""
    manifest = {
        "generated_at": datetime.now().isoformat(),
        "total_combinations": len(combinations),
        "render_settings": batch_cfg.get("render", {}),
        "combinations": []
    }

    for combo in combinations:
        output_dir = get_output_dir(combo, batch_cfg)
        manifest["combinations"].append({
            **combo,
            "output_dir": output_dir,
            "already_rendered": is_already_rendered(output_dir, batch_cfg),
        })

    with open(output_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Batch rendering orchestrator for Isaac Sim",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to batch_config.yaml"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be rendered without actually rendering"
    )
    parser.add_argument(
        "--assets",
        type=str,
        default=None,
        help="Comma-separated list of specific assets to render"
    )
    parser.add_argument(
        "--asset-type",
        type=str,
        choices=["bird", "drone"],
        help="Filter assets by type"
    )
    parser.add_argument(
        "--isaac-sim-path",
        type=str,
        default=None,
        help="Path to Isaac Sim installation"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Don't skip already rendered combinations"
    )
    parser.add_argument(
        "--list-assets",
        action="store_true",
        help="List available assets and exit"
    )
    parser.add_argument(
        "--save-manifest",
        type=str,
        default=None,
        help="Save batch manifest to JSON file"
    )

    # Depth cropping options
    parser.add_argument(
        "--crop-depth",
        action="store_true",
        help="Crop depth maps to object bounding box after each render (saves ~99%% storage)"
    )
    parser.add_argument(
        "--keep-full-depth",
        action="store_true",
        help="Keep original full-scene depth files after cropping"
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("Isaac Sim Batch Rendering Orchestrator")
    print("=" * 70)

    # Load configs
    batch_cfg = load_batch_config(args.config)
    if batch_cfg is None:
        sys.exit(1)

    asset_config = load_asset_config()
    if asset_config is None:
        sys.exit(1)

    # Get available assets
    all_assets = get_available_assets(asset_config)

    if args.list_assets:
        print("\nAvailable assets:")
        for asset in sorted(all_assets, key=lambda x: (x["type"], x["name"])):
            print(f"  [{asset['type']:5}] {asset['name']}")
        sys.exit(0)

    # Override resume setting
    if args.no_resume:
        batch_cfg.setdefault("execution", {})["resume"] = False

    # Apply dry-run from args if specified
    if args.dry_run:
        batch_cfg.setdefault("execution", {})["dry_run"] = True

    # Filter assets
    assets_to_render = filter_assets(all_assets, batch_cfg, args)
    if not assets_to_render:
        print("[BATCH] ERROR: No assets to render after filtering")
        sys.exit(1)

    print(f"\n[BATCH] Assets to render: {len(assets_to_render)}")
    for asset in assets_to_render:
        print(f"  - {asset}")

    # Generate combinations
    combinations = generate_combinations(batch_cfg, assets_to_render)
    print(f"\n[BATCH] Total combinations: {len(combinations)}")

    # Save manifest if requested
    if args.save_manifest:
        manifest = save_batch_manifest(combinations, batch_cfg, args.save_manifest)
        print(f"[BATCH] Manifest saved to: {args.save_manifest}")
        already_done = sum(1 for c in manifest["combinations"] if c["already_rendered"])
        print(f"[BATCH] Already rendered: {already_done}/{len(combinations)}")

    # Check dry run
    dry_run = batch_cfg.get("execution", {}).get("dry_run", False)
    if dry_run:
        print("\n[BATCH] DRY RUN MODE - No actual rendering will occur")

    # Find Isaac Sim Python
    isaac_python = find_isaac_sim_python(args.isaac_sim_path)
    if isaac_python is None and not dry_run:
        print("[BATCH] ERROR: Could not find Isaac Sim Python interpreter")
        print("[BATCH] Set ISAAC_SIM_PATH environment variable or use --isaac-sim-path")
        sys.exit(1)

    if isaac_python:
        print(f"[BATCH] Using Isaac Sim Python: {isaac_python}")

    # Depth cropping settings (command line overrides config)
    crop_depth_cfg = batch_cfg.get("render", {}).get("crop_depth", {})
    crop_depth = args.crop_depth or crop_depth_cfg.get("enabled", False)
    keep_full_depth = args.keep_full_depth or crop_depth_cfg.get("keep_originals", False)
    if crop_depth:
        print(f"[BATCH] Depth cropping: ENABLED (keep originals: {keep_full_depth})")

    # Run renders
    print("\n" + "-" * 70)
    print("Starting batch rendering...")
    print("-" * 70)

    success_count = 0
    skip_count = 0
    fail_count = 0

    for i, combo in enumerate(combinations):
        output_dir = get_output_dir(combo, batch_cfg)

        # Check if already rendered
        if is_already_rendered(output_dir, batch_cfg):
            print(f"\n[BATCH] [{i+1}/{len(combinations)}] SKIPPING (already rendered): {combo['asset']}")
            skip_count += 1
            continue

        print(f"\n[BATCH] [{i+1}/{len(combinations)}] Processing...")

        success = run_headless_render(
            combo, batch_cfg, isaac_python,
            dry_run=dry_run,
            crop_depth=crop_depth,
            keep_full_depth=keep_full_depth
        )

        if success:
            success_count += 1
        else:
            fail_count += 1

    # Summary
    print("\n" + "=" * 70)
    print("Batch Rendering Complete")
    print("=" * 70)
    print(f"  Total combinations: {len(combinations)}")
    print(f"  Successful: {success_count}")
    print(f"  Skipped (already done): {skip_count}")
    print(f"  Failed: {fail_count}")
    print("=" * 70 + "\n")

    sys.exit(0 if fail_count == 0 else 1)


if __name__ == "__main__":
    main()
