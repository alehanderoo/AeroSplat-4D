#!/usr/bin/env python3

"""
headless_runner.py - Isaac Sim Headless Rendering Runner

This script runs the Isaac Sim rendering workflow in headless mode.
It must be executed using Isaac Sim's Python interpreter (python.sh).

Usage:
    # Run with default config:
    ~/isaacsim/python.sh headless_runner.py

    # Run with specific asset:
    ~/isaacsim/python.sh headless_runner.py --asset "bald-eagle-med-poly.usdc"

    # Run with full overrides:
    ~/isaacsim/python.sh headless_runner.py \
        --asset "bald-eagle-med-poly.usdc" \
        --scene "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.2/Isaac/Environments/Simple_Warehouse/warehouse.usd" \
        --side-meters 10.0 \
        --flight-height 15.0 \
        --output-dir /path/to/output

Based on Isaac Sim documentation:
https://docs.isaacsim.omniverse.nvidia.com/5.0.0/python_scripting/manual_standalone_python.html
"""

import argparse
import sys
import os
from pathlib import Path

# Force unbuffered output so we can see print statements in logs
os.environ['PYTHONUNBUFFERED'] = '1'

# Parse arguments BEFORE initializing SimulationApp
# This allows --help to work without starting Isaac Sim
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Isaac Sim Headless Rendering Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with a config file:
    ~/isaacsim/python.sh headless_runner.py --config /path/to/config.yaml

    # Run with inline overrides:
    ~/isaacsim/python.sh headless_runner.py --asset "bald-eagle-med-poly.usdc" --side-meters 10.0
        """
    )

    # Config file
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yaml file (default: uses isaacsim/config.yaml)"
    )

    # Asset overrides
    parser.add_argument(
        "--asset",
        type=str,
        default=None,
        help="Asset name from asset_config.yaml (overrides config)"
    )

    # Scene overrides
    parser.add_argument(
        "--scene",
        type=str,
        default=None,
        help="Scene USD path or URL (overrides config)"
    )

    # Camera rig overrides
    parser.add_argument(
        "--side-meters",
        type=float,
        default=None,
        help="Camera rig diameter in meters"
    )
    parser.add_argument(
        "--cam-height",
        type=float,
        default=None,
        help="Camera height above ground"
    )
    parser.add_argument(
        "--flight-height",
        type=float,
        default=None,
        help="Flight height offset from waypoint"
    )
    parser.add_argument(
        "--rotation-offset",
        type=float,
        default=None,
        help="Camera rotation offset in radians"
    )
    parser.add_argument(
        "--camera-type",
        type=str,
        default=None,
        help="Camera type from cam_intrinsics.yaml"
    )

    # Render settings
    parser.add_argument(
        "--num-frames",
        type=int,
        default=None,
        help="Number of frames to render"
    )
    parser.add_argument(
        "--num-cameras",
        type=int,
        default=None,
        help="Number of cameras"
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for renders"
    )
    parser.add_argument(
        "--base-path",
        type=str,
        default=None,
        help="Base path for renders"
    )

    # Execution
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable verbose logging"
    )

    return parser.parse_args()


def apply_overrides(cfg, args):
    """Apply command line overrides to config."""

    # Asset override
    if args.asset:
        if "drone" not in cfg:
            cfg["drone"] = {}
        cfg["drone"]["asset_name"] = args.asset

    # Scene override
    if args.scene:
        if "scene" not in cfg:
            cfg["scene"] = {}
        cfg["scene"]["path"] = args.scene

    # Camera rig overrides
    if args.side_meters is not None:
        if "cameras" not in cfg:
            cfg["cameras"] = {}
        cfg["cameras"]["side_meters"] = args.side_meters

    if args.cam_height is not None:
        if "cameras" not in cfg:
            cfg["cameras"] = {}
        cfg["cameras"]["cam_height"] = args.cam_height

    if args.flight_height is not None:
        if "drone" not in cfg:
            cfg["drone"] = {}
        cfg["drone"]["flight_height_offset"] = args.flight_height

    if args.rotation_offset is not None:
        if "cameras" not in cfg:
            cfg["cameras"] = {}
        cfg["cameras"]["rotation_offset"] = args.rotation_offset

    if args.camera_type is not None:
        if "cameras" not in cfg:
            cfg["cameras"] = {}
        cfg["cameras"]["camera_type"] = args.camera_type

    # Render settings
    if args.num_frames is not None:
        if "render" not in cfg:
            cfg["render"] = {}
        cfg["render"]["num_frames"] = args.num_frames
        # Also update drone timeline
        if "drone" not in cfg:
            cfg["drone"] = {}
        if "timeline" not in cfg["drone"]:
            cfg["drone"]["timeline"] = {}
        cfg["drone"]["timeline"]["end_frame"] = float(args.num_frames)
        cfg["drone"]["timeline"]["middle_frame"] = float(args.num_frames) / 2.0

    if args.num_cameras is not None:
        if "render" not in cfg:
            cfg["render"] = {}
        cfg["render"]["num_cameras"] = args.num_cameras
        if "cameras" not in cfg:
            cfg["cameras"] = {}
        cfg["cameras"]["num_cameras"] = args.num_cameras

    # Output overrides
    if args.output_dir:
        if "render" not in cfg:
            cfg["render"] = {}
        cfg["render"]["output_dir"] = args.output_dir

    if args.base_path:
        if "render" not in cfg:
            cfg["render"] = {}
        cfg["render"]["base_path"] = args.base_path

    # Execution overrides
    if "execution" not in cfg:
        cfg["execution"] = {}

    if args.quiet:
        cfg["execution"]["verbose"] = False
    elif args.verbose:
        cfg["execution"]["verbose"] = True

    # Force render step to run in headless mode
    if "steps" not in cfg["execution"]:
        cfg["execution"]["steps"] = {}
    cfg["execution"]["steps"]["render"] = True

    return cfg


def main():
    """Main entry point for headless rendering."""
    args = parse_args()

    print("\n" + "=" * 70, flush=True)
    print("Isaac Sim Headless Rendering Runner", flush=True)
    print("=" * 70, flush=True)

    # =========================================================================
    # CRITICAL: Initialize SimulationApp BEFORE any other omni imports
    # See: https://docs.isaacsim.omniverse.nvidia.com/5.0.0/python_scripting/manual_standalone_python.html
    # =========================================================================
    print("[HEADLESS] Initializing Isaac Sim in headless mode...", flush=True)

    from isaacsim import SimulationApp

    # Configure headless rendering with RTX
    simulation_config = {
        "headless": True,
        "renderer": "RayTracedLighting",
        "anti_aliasing": "FXAA",
        "width": 2560,
        "height": 1440,
    }

    simulation_app = SimulationApp(simulation_config)
    print("[HEADLESS] Isaac Sim initialized successfully", flush=True)

    # =========================================================================
    # NOW we can import omni modules (after SimulationApp is created)
    # =========================================================================

    # Add isaacsim directory to path for our custom modules
    SCRIPT_DIR = Path(__file__).parent.resolve()
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))

    import yaml
    from runner import load_config, _refresh_workflow_modules, _run_workflow_async

    # Refresh modules to ensure they're loaded
    if not _refresh_workflow_modules(show_help=True):
        print("[HEADLESS] ERROR: Failed to load workflow modules", flush=True)
        simulation_app.close()
        sys.exit(1)

    # Load configuration
    config_path = args.config
    if config_path is None:
        config_path = SCRIPT_DIR / "config.yaml"

    cfg = load_config(config_path)
    if cfg is None:
        print(f"[HEADLESS] ERROR: Failed to load config from {config_path}", flush=True)
        simulation_app.close()
        sys.exit(1)

    # Apply command line overrides
    cfg = apply_overrides(cfg, args)

    print(f"[HEADLESS] Configuration loaded", flush=True)
    print(f"[HEADLESS] Asset: {cfg.get('drone', {}).get('asset_name', 'N/A')}", flush=True)
    print(f"[HEADLESS] Scene: {cfg.get('scene', {}).get('path', 'N/A')}", flush=True)
    print(f"[HEADLESS] Output: {cfg.get('render', {}).get('output_dir', 'auto')}", flush=True)
    print("=" * 70 + "\n", flush=True)

    # =========================================================================
    # Execute workflow with proper simulation stepping
    # =========================================================================
    success = False
    try:
        import asyncio

        async def run_headless_workflow():
            """Run the workflow and wait for completion."""
            verbose = cfg.get("execution", {}).get("verbose", True)
            steps_to_run = cfg.get("execution", {}).get("steps", {})
            result = await _run_workflow_async(cfg, verbose, steps_to_run)
            return result

        # Create the workflow coroutine
        workflow_coro = run_headless_workflow()

        # Get or create event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Schedule the workflow
        workflow_task = loop.create_task(workflow_coro)

        # Step the simulation until workflow completes
        # This is the proper way to drive async tasks in headless mode
        print("[HEADLESS] Starting render loop...", flush=True)

        max_iterations = 1000000  # Safety limit
        iteration = 0

        while not workflow_task.done() and iteration < max_iterations:
            # Step the simulation (this drives rendering and async tasks)
            simulation_app.update()

            # Give the event loop a chance to process
            loop.run_until_complete(asyncio.sleep(0))

            iteration += 1

            # Progress indicator every 1000 iterations
            if iteration % 1000 == 0:
                print(f"[HEADLESS] Simulation step {iteration}...", flush=True)

        # Get the result
        if workflow_task.done():
            try:
                success = workflow_task.result()
            except Exception as e:
                print(f"[HEADLESS] Workflow raised exception: {e}", flush=True)
                import traceback
                traceback.print_exc()
                success = False
        else:
            print(f"[HEADLESS] WARNING: Workflow did not complete within {max_iterations} iterations", flush=True)
            success = False

        if success:
            print("\n[HEADLESS] Workflow completed successfully!", flush=True)
        else:
            print("\n[HEADLESS] Workflow failed!", flush=True)

    except Exception as e:
        print(f"\n[HEADLESS] ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()
        success = False

    finally:
        # Clean up
        print("[HEADLESS] Shutting down Isaac Sim...", flush=True)
        simulation_app.close()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
