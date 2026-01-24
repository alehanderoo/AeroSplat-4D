#!/usr/bin/env python3

"""
runner.py - Isaac Sim Multi-Camera Object Rendering Orchestrator

This script orchestrates the full workflow:
1. Load USD scene
2. Position cameras in circular rig
3. Animate object flight path (supports drone, bird, and other assets)
4. Render frames from all cameras

Configuration is centralized in config.yaml for easy modification.

Usage in Script Editor:
    # Paste once to bootstrap (sets up paths):
    exec(open("/home/sandro/thesis/code/isaacSim_snippets/runner.py").read())
    
    # Then run:
    run_workflow()
"""

from pathlib import Path
import importlib
import sys
import yaml

LoadScene = None
LoadDrone = None
PositionCameras = None
Animate = None
Render = None
resolve_fps_from_config = None

_MODULE_CACHE = {}
MODULES_LOADED = False

# =====================================================================
# Setup Python path for GUI Script Editor compatibility
# =====================================================================

# Determine the root project folder
# When run from Script Editor, __file__ may be a temp path, so we need to be clever
def _get_project_root():
    """Get the project root folder, hardcoded location."""
    return Path("/home/sandro/aeroSplat-4D/isaacsim")

ROOT = _get_project_root()
print(f"[RUNNER] Project root: {ROOT}")

# Add to sys.path FIRST (highest priority)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
    print(f"[RUNNER] Added {ROOT} to sys.path")

# Optional: Add folder to Kit's scriptFolders for future sessions
def setup_kit_script_folders():
    """
    Register this folder in Kit's scriptFolders setting to avoid manual sys.path edits.
    Run once in the GUI Script Editor to persist settings across sessions.
    """
    try:
        import carb
        import pathlib
        settings = carb.settings.get_settings()
        root = str(ROOT)
        folders = list(settings.get("/app/python/scriptFolders") or [])
        if root not in folders:
            folders.append(root)
            settings.set("/app/python/scriptFolders", folders)
            print(f"[RUNNER] Added {root} to Kit scriptFolders")
        else:
            print(f"[RUNNER] {root} already in Kit scriptFolders")
    except Exception as e:
        print(f"[RUNNER] Warning: Could not set Kit scriptFolders: {e}")
        print("[RUNNER] You may need to manually add this folder to Kit settings")


def bootstrap_for_script_editor():
    """
    Bootstrap function for Isaac Sim Script Editor.
    Call this once at the start of a Script Editor session to ensure imports work.
    
    Usage in Script Editor:
        from runner import bootstrap_for_script_editor
        bootstrap_for_script_editor()
        run_workflow()
    """
    print(f"\n[RUNNER] Bootstrap: Ensuring {ROOT} is in Python path...")
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
        print(f"[RUNNER] ✓ Added {ROOT} to sys.path")
    else:
        print(f"[RUNNER] ✓ {ROOT} already in sys.path")
    
    # Try to verify modules are importable
    try:
        import load_scene
        print("[RUNNER] ✓ load_scene module is importable")
    except ImportError as e:
        print(f"[RUNNER] ✗ load_scene not importable: {e}")
    
    try:
        import position_cameras
        print("[RUNNER] ✓ position_cameras module is importable")
    except ImportError as e:
        print(f"[RUNNER] ✗ position_cameras not importable: {e}")
    
    try:
        import animate
        print("[RUNNER] ✓ animate module is importable")
    except ImportError as e:
        print(f"[RUNNER] ✗ animate not importable: {e}")
    
    try:
        import render
        print("[RUNNER] ✓ render module is importable")
    except ImportError as e:
        print(f"[RUNNER] ✗ render not importable: {e}")
    
    _refresh_workflow_modules(show_help=False)
    print("[RUNNER] Bootstrap complete!\n")


# =====================================================================
# Import workflow modules
# =====================================================================
def _load_workflow_modules():
    """Import or hot-reload workflow modules when running inside Script Editor."""
    modules = {}
    importlib.invalidate_caches()
    for name in [
        "load_scene",
        "load_drone",
        "position_cameras",
        "animate",
        "render",
        "config_utils",
    ]:
        try:
            module = sys.modules.get(name)
            if module is None:
                module = importlib.import_module(name)
            else:
                module = importlib.reload(module)
            modules[name] = module
        except Exception as exc:
            print(f"[RUNNER] ERROR: Failed to import {name}: {exc}")
            return None
    return modules

def _apply_module_bindings(modules):
    """Bind refreshed module classes/functions to globals."""
    global LoadScene, LoadDrone, PositionCameras, Animate, Render, resolve_fps_from_config
    LoadScene = modules["load_scene"].LoadScene
    LoadDrone = modules["load_drone"].LoadDrone
    PositionCameras = modules["position_cameras"].PositionCameras
    Animate = modules["animate"].Animate
    Render = modules["render"].Render
    resolve_fps_from_config = modules["config_utils"].resolve_fps_from_config


def _print_module_load_help():
    print(f"[RUNNER] Project root: {ROOT}")
    print(f"[RUNNER] sys.path[0]: {sys.path[0] if sys.path else 'empty'}")
    print(f"[RUNNER] Attempting to list files in {ROOT}:")
    try:
        files = list(ROOT.glob("*.py"))
        for f in files:
            print(f"  - {f.name}")
    except Exception:
        pass
    print(f"\n[RUNNER] SOLUTION: If running from Script Editor, paste this first:")
    print(f"    from runner import bootstrap_for_script_editor")
    print(f"    bootstrap_for_script_editor()")
    print(f"\n[RUNNER] Or manually call: run_workflow() after bootstrap succeeds.")


def _refresh_workflow_modules(show_help=True):
    """Reload workflow modules and update global bindings."""
    global _MODULE_CACHE, MODULES_LOADED
    modules = _load_workflow_modules()
    if modules is None:
        MODULES_LOADED = False
        if show_help:
            _print_module_load_help()
        return False

    _MODULE_CACHE = modules
    _apply_module_bindings(modules)
    MODULES_LOADED = True
    return True


if not _refresh_workflow_modules():
    # Keep MODULES_LOADED False and instructions already printed
    pass


# =====================================================================
# Load configuration
# =====================================================================
def load_config(config_path=None):
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = ROOT / "config.yaml"
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        print(f"[RUNNER] ERROR: Config file not found: {config_path}")
        print(f"[RUNNER] Please create config.yaml in {ROOT}")
        return None
    
    try:
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        print(f"[RUNNER] Loaded configuration from {config_path}")

        # Load asset configuration
        asset_config_path = ROOT / "asset_config.yaml"
        if asset_config_path.exists():
            try:
                with open(asset_config_path, 'r') as f:
                    asset_configs = yaml.safe_load(f)
                
                # Merge into main config
                drone_cfg = cfg.get("drone", {})
                asset_name = drone_cfg.get("asset_name")
                
                if asset_name and asset_configs and asset_name in asset_configs:
                    print(f"[RUNNER] Applying asset config for: {asset_name}")
                    asset_data = asset_configs[asset_name]
                    # Merge asset_data into drone_cfg
                    drone_cfg.update(asset_data)
                    cfg["drone"] = drone_cfg
                elif asset_name:
                    print(f"[RUNNER] WARNING: Asset '{asset_name}' not found in asset_config.yaml")

            except Exception as e:
                print(f"[RUNNER] ERROR: Failed to parse asset_config.yaml: {e}")
        else:
             print(f"[RUNNER] Warning: asset_config.yaml not found at {asset_config_path}")

        # Calculate bird-specific parameters (wingspan -> scale, flap_frequency -> animation_speed_factor)
        cfg = _apply_bird_parameters(cfg)

        return cfg
    except Exception as e:
        print(f"[RUNNER] ERROR: Failed to parse config.yaml: {e}")
        return None


def _apply_bird_parameters(cfg):
    """Apply wingspan and flap_frequency for birds.

    For birds (type="bird"), this maps intuitive parameters directly:
    - wingspan (meters) -> uniform scale (assets normalized to 1m wingspan at scale=1)
    - flap_frequency (Hz) -> animation_speed_factor (assets normalized to 1 flap per loop)

    If wingspan or flap_frequency is 0 or null, the raw scale/animation_speed_factor
    from asset_config.yaml is used instead.
    """
    drone_cfg = cfg.get("drone", {})
    asset_type = drone_cfg.get("type")

    # Only apply to birds
    if asset_type != "bird":
        return cfg

    bird_params = drone_cfg.get("bird_parameters", {})
    target_wingspan = bird_params.get("wingspan")
    target_flap_freq = bird_params.get("flap_frequency")

    # Apply wingspan as direct scale (assets normalized to 1m wingspan at scale=1)
    if target_wingspan is not None and target_wingspan > 0:
        drone_cfg["scale"] = [target_wingspan, target_wingspan, target_wingspan]
        print(f"[RUNNER] Bird wingspan: {target_wingspan}m -> scale=[{target_wingspan}, {target_wingspan}, {target_wingspan}]")
    else:
        # Use scale from asset_config.yaml
        scale = drone_cfg.get("scale", [1.0, 1.0, 1.0])
        print(f"[RUNNER] Bird using asset_config scale: {scale}")

    # Apply flap_frequency as direct animation_speed_factor (assets normalized to 1 flap per loop)
    if target_flap_freq is not None and target_flap_freq > 0:
        drone_cfg["animation_speed_factor"] = target_flap_freq
        print(f"[RUNNER] Bird flap frequency: {target_flap_freq}Hz -> animation_speed_factor={target_flap_freq}")
    else:
        # Use animation_speed_factor from asset_config.yaml
        speed = drone_cfg.get("animation_speed_factor", 1.0)
        print(f"[RUNNER] Bird using asset_config animation_speed_factor: {speed}")

    cfg["drone"] = drone_cfg
    return cfg


# =====================================================================
# Workflow orchestration
# =====================================================================
def run_workflow(cfg=None):
    """Execute the complete rendering workflow."""
    
    if not _refresh_workflow_modules(show_help=False):
        print("[RUNNER] ERROR: Workflow modules not loaded.")
        print("[RUNNER] Call bootstrap_for_script_editor() first if running from Script Editor:")
        print("    from runner import bootstrap_for_script_editor")
        print("    bootstrap_for_script_editor()")
        return False

    # Check if modules are loaded
    if not MODULES_LOADED:
        print("[RUNNER] ERROR: Workflow modules not loaded.")
        print("[RUNNER] Call bootstrap_for_script_editor() first if running from Script Editor:")
        print("    from runner import bootstrap_for_script_editor")
        print("    bootstrap_for_script_editor()")
        return False
    
    if cfg is None:
        cfg = load_config()
    
    if cfg is None:
        print("[RUNNER] ERROR: No valid configuration. Aborting.")
        return False
    
    # Extract execution settings
    execution_cfg = cfg.get("execution", {})
    verbose = execution_cfg.get("verbose", True)
    steps_to_run = execution_cfg.get("steps", {})
    
    if verbose:
        print("\n" + "="*70)
        print("Isaac Sim Multi-Camera Drone Rendering Workflow")
        print("="*70)
        print(f"Configuration: {cfg}")
        print("="*70 + "\n")
    
    # Schedule the async workflow
    import asyncio
    task = asyncio.ensure_future(_run_workflow_async(cfg, verbose, steps_to_run))
    
    if verbose:
        print("[RUNNER] Workflow task scheduled")
    
    return True


async def _run_workflow_async(cfg, verbose, steps_to_run):
    """Execute the complete workflow asynchronously with proper loading waits."""
    import omni.kit.app
    
    kit_app = omni.kit.app.get_app()
    
    try:
        # Step 1: Load scene
        if steps_to_run.get("load_scene", True):
            if verbose:
                print("\n[STEP 1/5] Loading scene...")
            loader = LoadScene(cfg)
            scene_prim = loader.run()
            if scene_prim is None:
                print("[RUNNER] ERROR: Failed to load scene. Aborting workflow.")
                return False
            
            # Wait for scene to fully load
            if verbose:
                print("[STEP 1/5] Waiting for scene to fully load...")
            for _ in range(10):
                await kit_app.next_update_async()
            
            if verbose:
                print("[STEP 1/5] ✓ Scene loaded successfully")
        
        # Step 2: Load drone
        if steps_to_run.get("load_drone", True):
            if verbose:
                print("\n[STEP 2/5] Loading drone...")
            drone_loader = LoadDrone(cfg)
            drone_prim = drone_loader.run()
            if drone_prim is None:
                print("[RUNNER] ERROR: Failed to load drone. Aborting workflow.")
                return False
            
            # Wait for drone to fully load
            if verbose:
                print("[STEP 2/5] Waiting for drone to fully load...")
            for _ in range(10):
                await kit_app.next_update_async()
            
            prim_path = drone_loader.drone_cfg.get("prim_path", "/World/Drone")
            semantics_applied = getattr(drone_loader, "last_semantics_applied", None)
            if verbose:
                if semantics_applied:
                    print(f"[STEP 2/5] ✓ Semantic labels confirmed for drone at {prim_path}")
                    try:
                        drone_loader.print_class_ids(prim_path)
                    except Exception:
                        print("[STEP 2/5] Warning: Unable to print drone semantic mapping")
                elif semantics_applied is False:
                    print(f"[STEP 2/5] WARNING: Semantic labels were not applied at {prim_path}")

            if verbose:
                print("[STEP 2/5] ✓ Drone loaded successfully")
        
        # Step 3: Position cameras
        if steps_to_run.get("position_cameras", True):
            if verbose:
                print("\n[STEP 3/5] Positioning cameras...")
            camera_positioner = PositionCameras(cfg)
            cameras, aim_point, scene_params = camera_positioner.run()
            if not cameras:
                print("[RUNNER] ERROR: Failed to position cameras. Aborting workflow.")
                return False
            
            # Wait for cameras to be positioned
            if verbose:
                print("[STEP 3/5] Waiting for cameras to be ready...")
            for _ in range(10):
                await kit_app.next_update_async()
            
            if verbose:
                print(f"[STEP 3/5] ✓ Positioned {len(cameras)} cameras")
        
        # Step 4: Animate object (drone, bird, etc.)
        if steps_to_run.get("animate_drone", True):
            if verbose:
                print("\n[STEP 4/5] Animating object...")
            animator = Animate(cfg)
            obj_prim = animator.run()
            if obj_prim is None:
                print("[RUNNER] ERROR: Failed to animate object. Aborting workflow.")
                return False

            # Wait for animation to be set up
            if verbose:
                print("[STEP 4/5] Waiting for animation setup to complete...")
            for _ in range(10):
                await kit_app.next_update_async()

            if verbose:
                asset_type = animator.asset_type or "unknown"
                print(f"[STEP 4/5] ✓ Animation setup complete (asset type: {asset_type})")
        
        # Final wait to ensure everything is fully loaded and stable
        if verbose:
            print("\n[STEP 4.5/5] Final stabilization - ensuring all assets are fully loaded...")
        for _ in range(20):
            await kit_app.next_update_async()
        if verbose:
            print("[STEP 4.5/5] ✓ All assets loaded and scene stabilized")
        
        # Step 5: Render frames
        if steps_to_run.get("render", True):
            if verbose:
                print("\n[STEP 5/5] Starting rendering...")
            
            # Call the async render function directly
            import omni.usd
            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print("[RUNNER] ERROR: No active stage found for rendering")
                return False
            
            # Extract render configuration
            render_cfg = cfg.get("render", {})
            num_cameras = int(render_cfg.get("num_cameras", 5))
            num_frames = int(render_cfg.get("num_frames", 120))
            output_dir = render_cfg.get("output_dir")
            base_path = render_cfg.get("base_path", "/home/sandro/thesis/renders")
            stage_fps = resolve_fps_from_config(cfg)
            warmup_frames = int(cfg.get("execution", {}).get("warmup_frames", 0))
            
            from datetime import datetime
            if output_dir is None:
                date_str = datetime.now().strftime("%d-%m-%y")
                output_dir = f"{base_path}/{num_cameras}cams_{date_str}"
            
            # Import the async render function
            from render import render_multi_camera_async
            
            if verbose:
                print(f"[STEP 5/5] Output directory: {output_dir}")
                print(f"[STEP 5/5] Starting render of {num_frames} frames from {num_cameras} cameras")
            
            # Extract additional render config
            drone_prim_path = cfg.get("drone", {}).get("prim_path", "/World/Drone")
            crop_depth_config = render_cfg.get("crop_depth", {})

            # Execute rendering and wait for completion
            await render_multi_camera_async(
                num_cameras=num_cameras,
                num_frames=num_frames,
                output_dir=output_dir,
                base_path=base_path,
                stage_fps=stage_fps,
                warmup_frames=warmup_frames,
                drone_prim_path=drone_prim_path,
                crop_depth_config=crop_depth_config
            )
            
            if verbose:
                print(f"[STEP 5/5] ✓ Rendering complete!")
                print(f"[STEP 5/5]    Output saved to: {output_dir}")
        
        if verbose:
            print("\n" + "="*70)
            print("✓ Workflow completed successfully!")
            print("="*70 + "\n")
        
        return True
        
    except Exception as e:
        print(f"\n[RUNNER] ERROR: Workflow failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


# =====================================================================
# Main entry point
# =====================================================================
def main():
    """Main entry point for the script."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Isaac Sim Multi-Camera Drone Rendering Orchestrator"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=f"Path to config.yaml (default: {ROOT}/config.yaml)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--setup-kit",
        action="store_true",
        help="Register this folder in Kit's scriptFolders for future sessions"
    )
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="Bootstrap for Script Editor (verify all imports)"
    )
    
    args = parser.parse_args()
    
    # Bootstrap if requested
    if args.bootstrap:
        bootstrap_for_script_editor()
        return MODULES_LOADED
    
    # Setup Kit scriptFolders if requested
    if args.setup_kit:
        setup_kit_script_folders()
    
    # Load configuration
    cfg = load_config(args.config)
    
    if cfg is None:
        return False
    
    # Override verbose setting if specified
    if args.verbose:
        if "execution" not in cfg:
            cfg["execution"] = {}
        cfg["execution"]["verbose"] = True
    
    # Run the workflow
    return run_workflow(cfg)


if __name__ == "__main__":
    import sys
    # bootstrap_for_script_editor()
    success = main()
    # Don't exit when running from Script Editor - let async tasks complete
    # sys.exit(0 if success else 1)
