#!/usr/bin/env python3
"""
verify_assets.py - Headless Asset Verification for Isaac Sim

This script renders all bird and drone assets from asset_config.yaml
in Isaac Sim headless mode and creates a grid visualization.

Usage:
    ~/isaacsim/python.sh verify_assets.py
    ~/isaacsim/python.sh verify_assets.py --category birds
    ~/isaacsim/python.sh verify_assets.py --category drones
    ~/isaacsim/python.sh verify_assets.py --output birds_grid.png
"""

import argparse
import asyncio
import sys
import os
import math
import tempfile
import shutil
from pathlib import Path

# Force unbuffered output
os.environ['PYTHONUNBUFFERED'] = '1'

SCRIPT_DIR = Path(__file__).parent.resolve()
ISAACSIM_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = SCRIPT_DIR

# Grid room scene
GRID_SCENE_URL = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.2/Isaac/Environments/Grid/default_environment.usd"

TARGET = [0.0, 0.0, 1.0]
CAMERA = [1.5, 1.5, 2.0]

def parse_args():
    parser = argparse.ArgumentParser(description="Verify assets by rendering in Isaac Sim")
    parser.add_argument(
        "--category",
        type=str,
        choices=["birds", "drones", "all"],
        default="all",
        help="Category of assets to render (default: all)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output filename for grid image (default: {category}_grid.png)"
    )
    parser.add_argument(
        "--render-size",
        type=int,
        default=512,
        help="Size of each rendered thumbnail (default: 512)"
    )
    parser.add_argument(
        "--grid-cols",
        type=int,
        default=6,
        help="Number of columns in grid (default: 6)"
    )
    return parser.parse_args()


def load_asset_config():
    """Load asset configuration from asset_config.yaml."""
    import yaml
    config_path = ISAACSIM_DIR / "asset_config.yaml"
    if not config_path.exists():
        print(f"[ERROR] asset_config.yaml not found at {config_path}")
        return {}

    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_assets_by_category(asset_config, category):
    """Get assets filtered by category."""
    assets = []
    for name, cfg in asset_config.items():
        if not isinstance(cfg, dict):
            continue
        asset_type = cfg.get("type", "unknown")
        if category == "all" or asset_type == category.rstrip('s'):  # birds -> bird
            assets.append((name, cfg))
    return assets


def compute_lookat_matrix(position, target):
    """Compute LookAt matrix for camera at position looking at target."""
    from pxr import Gf, UsdGeom
    
    # Gf.Matrix4d.SetLookAt(eye, center, up)
    # Isaac Sim (and USD) uses Y-up by default for world, but cameras look down -Z.
    # SetLookAt creates a transform where -Z points from eye to center.
    
    matrix = Gf.Matrix4d()
    matrix.SetLookAt(Gf.Vec3d(*position), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1))
    
    # However, Gf.Matrix4d.SetLookAt produces a VIEW matrix (inverse of camera transform).
    # We need the camera's transform matrix (local-to-parent).
    return matrix.GetInverse()


def setup_scene(stage):
    """Load the grid room scene."""
    from pxr import UsdGeom, Sdf

    print(f"[SCENE] Loading grid room: {GRID_SCENE_URL}")

    scene_prim = stage.DefinePrim("/World/Scene", "Xform")
    scene_prim.GetReferences().AddReference(GRID_SCENE_URL)

    return scene_prim


def setup_camera(stage, render_size):
    """Create camera at (2, 2, 1) looking at origin."""
    from pxr import UsdGeom, Gf

    cam_path = "/World/VerifyCamera"
    camera = UsdGeom.Camera.Define(stage, cam_path)

    # Position at (2, 2, 1) for better view angle
    position = tuple(CAMERA)
    target = tuple(TARGET)

    xform = UsdGeom.Xformable(camera.GetPrim())
    xform.ClearXformOpOrder()

    # Set transform using LookAt
    xform_op = xform.AddTransformOp()
    matrix = compute_lookat_matrix(position, target)
    xform_op.Set(matrix)

    # Camera properties
    camera.GetFocalLengthAttr().Set(35.0)
    camera.GetFocusDistanceAttr().Set(3.0)
    camera.GetFStopAttr().Set(0.0)  # No DOF blur
    camera.GetClippingRangeAttr().Set(Gf.Vec2f(0.1, 100.0))

    print(f"[CAMERA] Created at {position} looking at {target}")

    return cam_path


def load_asset(stage, asset_name, asset_cfg):
    """Load a single asset at the origin."""
    from pxr import UsdGeom, Gf

    usd_path = asset_cfg.get("usd_path")
    if not usd_path or not Path(usd_path).exists():
        print(f"[ASSET] WARNING: USD file not found: {usd_path}")
        return None

    prim_path = "/World/Asset"

    # Remove existing asset if present
    existing = stage.GetPrimAtPath(prim_path)
    if existing and existing.IsValid():
        stage.RemovePrim(prim_path)

    # Create and load asset
    asset_prim = stage.DefinePrim(prim_path, "Xform")
    asset_prim.GetReferences().AddReference(usd_path)

    # Apply transforms from config
    xform = UsdGeom.Xformable(asset_prim)
    xform.ClearXformOpOrder()

    # Translation (Force to TARGET)
    # trans = asset_cfg.get("translation", TARGET) # Ignore config for verification
    trans = TARGET
    translate_op = xform.AddTranslateOp()
    translate_op.Set(Gf.Vec3d(*trans))

    # Scale
    scale = asset_cfg.get("scale", [1.0, 1.0, 1.0])
    scale_op = xform.AddScaleOp()
    scale_op.Set(Gf.Vec3f(*scale))

    return asset_prim


async def render_asset_async(simulation_app, render_product, writer, temp_dir, asset_name, render_size):
    """Render the current scene and return the image using orchestrator."""
    import numpy as np
    import omni.kit.app
    import omni.replicator.core as rep
    from PIL import Image

    kit_app = omni.kit.app.get_app()

    # Settle the scene with multiple updates
    for _ in range(20):
        simulation_app.update()
        await kit_app.next_update_async()

    # Wait 2 seconds for full rendering
    await asyncio.sleep(1.0)

    # Clear any previous renders
    for f in Path(temp_dir).glob("rgb_*.png"):
        f.unlink()

    # Capture using orchestrator (this is the proper way in headless mode)
    rep.orchestrator.set_capture_on_play(False)
    await rep.orchestrator.step_async(rt_subframes=32, delta_time=0.0, pause_timeline=True)
    await rep.orchestrator.wait_until_complete_async()

    # Give time for file write
    for _ in range(5):
        await kit_app.next_update_async()

    # Find the rendered file
    rgb_files = sorted(Path(temp_dir).glob("rgb_*.png"))
    if not rgb_files:
        print(f"[RENDER] WARNING: No RGB file found for {asset_name}")
        return None

    # Load the most recent render
    try:
        img = Image.open(rgb_files[-1])
        img_data = np.array(img)

        # Handle RGBA -> RGB
        if len(img_data.shape) == 3 and img_data.shape[2] == 4:
            img_data = img_data[:, :, :3]

        return img_data
    except Exception as e:
        print(f"[RENDER] WARNING: Failed to load image for {asset_name}: {e}")
        return None


def create_grid_image(renders, asset_names, title, cols, output_path):
    """Create and save a grid image from renders."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    if not renders:
        print("[GRID] No renders to create grid")
        return

    num_assets = len(renders)
    rows = math.ceil(num_assets / cols)

    # Get render size from first image
    h, w = renders[0].shape[:2]

    # Add space for labels
    label_height = 30
    cell_height = h + label_height

    # Create grid image
    grid_w = cols * w
    grid_h = rows * cell_height + 50  # Extra for title

    grid_img = Image.new('RGB', (grid_w, grid_h), color=(240, 240, 240))
    draw = ImageDraw.Draw(grid_img)

    # Try to load a font, fall back to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except:
        font = ImageFont.load_default()
        title_font = font

    # Draw title
    draw.text((grid_w // 2, 15), title, fill=(0, 0, 0), font=title_font, anchor="mt")

    # Place renders in grid
    for idx, (render, name) in enumerate(zip(renders, asset_names)):
        row = idx // cols
        col = idx % cols

        x = col * w
        y = 50 + row * cell_height

        # Convert numpy to PIL and paste
        img = Image.fromarray(render)
        grid_img.paste(img, (x, y))

        # Add label
        label = name[:25] + "..." if len(name) > 25 else name
        text_x = x + w // 2
        text_y = y + h + 5
        draw.text((text_x, text_y), label, fill=(0, 0, 0), font=font, anchor="mt")

    # Save
    grid_img.save(output_path)
    print(f"[GRID] Saved grid image to {output_path}")

    return grid_img


async def run_verification_async(args, simulation_app):
    """Main async verification workflow."""
    import omni.usd
    import omni.kit.app
    import omni.replicator.core as rep
    import numpy as np

    kit_app = omni.kit.app.get_app()

    # Load asset config
    asset_config = load_asset_config()
    if not asset_config:
        return False

    # Filter assets by category
    if args.category == "all":
        birds = get_assets_by_category(asset_config, "birds")
        drones = get_assets_by_category(asset_config, "drones")
        categories_to_render = [("birds", birds), ("drones", drones)]
    else:
        assets = get_assets_by_category(asset_config, args.category)
        categories_to_render = [(args.category, assets)]

    # Get stage
    stage = omni.usd.get_context().get_stage()

    # Setup scene and camera
    setup_scene(stage)
    cam_path = setup_camera(stage, args.render_size)

    # Wait for scene to load
    print("[INIT] Waiting for scene to load...")
    for _ in range(50):
        simulation_app.update()
        await kit_app.next_update_async()

    # Create temp directory for renders
    temp_dir = tempfile.mkdtemp(prefix="verify_assets_")
    print(f"[INIT] Temp directory: {temp_dir}")

    # Create render product
    rp = rep.create.render_product(cam_path, (args.render_size, args.render_size))

    # Create writer (BasicWriter handles file output properly)
    writer = rep.writers.get("BasicWriter")
    writer.initialize(output_dir=temp_dir, rgb=True)
    writer.attach(rp)

    # Warm up with proper async stepping
    print("[INIT] Warming up renderer...")
    for _ in range(30):
        simulation_app.update()
        await kit_app.next_update_async()

    # Render each category
    for category_name, assets in categories_to_render:
        if not assets:
            print(f"[{category_name.upper()}] No assets found")
            continue

        print(f"\n[{category_name.upper()}] Rendering {len(assets)} assets...")

        renders = []
        names = []

        for idx, (asset_name, asset_cfg) in enumerate(assets):
            print(f"  [{idx+1}/{len(assets)}] {asset_name}...", end=" ", flush=True)

            # Load asset
            asset_prim = load_asset(stage, asset_name, asset_cfg)
            if asset_prim is None:
                print("SKIP (not found)")
                continue

            # Render using async orchestrator
            img = await render_asset_async(
                simulation_app, rp, writer, temp_dir, asset_name, args.render_size
            )
            if img is not None:
                renders.append(img)
                names.append(asset_name)
                print("OK")
            else:
                print("FAILED")

            # Remove asset for next iteration
            stage.RemovePrim("/World/Asset")
            await kit_app.next_update_async()
            
            # Wait 2 seconds for removal
            await asyncio.sleep(1.0)

        # Create grid image
        if renders:
            output_name = args.output if args.output else f"{category_name}_grid.png"
            output_path = OUTPUT_DIR / output_name

            # If rendering all, append category to filename
            if args.category == "all" and not args.output:
                output_path = OUTPUT_DIR / f"{category_name}_grid.png"

            create_grid_image(
                renders,
                names,
                f"{category_name.capitalize()} ({len(renders)} assets)",
                args.grid_cols,
                output_path
            )

    # Cleanup
    writer.detach()
    rp.destroy()
    shutil.rmtree(temp_dir, ignore_errors=True)

    return True


def main():
    args = parse_args()

    print("\n" + "=" * 70)
    print("Isaac Sim Asset Verification")
    print("=" * 70)

    # Initialize SimulationApp BEFORE any omni imports
    print("[INIT] Initializing Isaac Sim in headless mode...")

    from isaacsim import SimulationApp

    simulation_config = {
        "headless": True,
        "renderer": "RayTracedLighting",
        "anti_aliasing": "FXAA",
        "width": args.render_size,
        "height": args.render_size,
    }

    simulation_app = SimulationApp(simulation_config)
    print("[INIT] Isaac Sim initialized successfully")

    # Run the async workflow
    async def run_workflow():
        return await run_verification_async(args, simulation_app)

    # Get or create event loop
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    # Schedule the workflow
    workflow_task = loop.create_task(run_workflow())

    # Step the simulation until workflow completes
    print("[MAIN] Starting render loop...")
    max_iterations = 1000000
    iteration = 0

    while not workflow_task.done() and iteration < max_iterations:
        simulation_app.update()
        loop.run_until_complete(asyncio.sleep(0))
        iteration += 1

    # Get result
    try:
        success = workflow_task.result()
    except Exception as e:
        print(f"[ERROR] Workflow failed: {e}")
        import traceback
        traceback.print_exc()
        success = False

    print("\n" + "=" * 70)
    if success:
        print("Asset verification complete!")
    else:
        print("Asset verification failed!")
    print("=" * 70)

    simulation_app.close()


if __name__ == "__main__":
    main()
