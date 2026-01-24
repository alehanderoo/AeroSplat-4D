#!/usr/bin/env python3

"""
Render: Isaac Sim Multi-Camera Rendering
Renders from multiple cameras and saves RGB, depth, and instance segmentation masks.
"""

import asyncio
import ast
from datetime import datetime
import json
import os
import shutil
from pathlib import Path

import carb.settings

import numpy as np
import omni.kit.app
import omni.replicator.core as rep
import omni.timeline
import omni.usd
from omni.syntheticdata import SyntheticData
from PIL import Image
from pxr import Usd

from config_utils import resolve_fps_from_config
from render_utils import (
    reshape_to_matrix as _reshape_to_matrix,
    matrix_to_list as _matrix_to_list,
    extract_camera_pose as _extract_camera_pose,
    compute_camera_intrinsics as _compute_camera_intrinsics,
    build_camera_metadata as _build_camera_metadata,
    extract_instance_drone_ids as _extract_instance_drone_ids,
    compute_world_bbox_3d as _compute_world_bbox_3d,
    project_point_to_screen as _project_point_to_screen,
    compute_mask_2d_stats as _compute_mask_2d_stats,
    find_drone_prim as _find_drone_prim,
    safe_timeline_call as _safe_timeline_call,
    create_camera_api_helper as _create_camera_api_helper,
)


def _parse_color_tuple(color_str):
    """Convert BasicWriter color string to an integer tuple."""
    try:
        value = ast.literal_eval(color_str)
    except (SyntaxError, ValueError):
        return None

    if isinstance(value, (list, tuple)):
        try:
            return tuple(int(c) for c in value)
        except (TypeError, ValueError):
            return None
    return None


class Render:
    """Render synthetic data from multiple cameras in Isaac Sim."""
    
    def __init__(self, cfg):
        """Initialize with configuration dict."""
        self.cfg = cfg
        self.render_cfg = cfg.get("render", {})
        self.verbose = cfg.get("execution", {}).get("verbose", False)
        self.stage_fps = resolve_fps_from_config(cfg)
    
    def run(self):
        """Execute the rendering workflow (schedules async task).
        
        NOTE: This method schedules an async task. For workflows that need to wait
        for rendering to complete, call render_multi_camera_async() directly with await.
        """
        if self.verbose:
            print("[RENDER] Initializing render workflow")
        
        # Extract configuration
        num_cameras = int(self.render_cfg.get("num_cameras", 5))
        num_frames = int(self.render_cfg.get("num_frames", 120))
        output_dir = self.render_cfg.get("output_dir")
        base_path = self.render_cfg.get("base_path", "/home/sandro/thesis/renders")
        stage_fps = self.stage_fps
        warmup_frames = int(self.cfg.get("execution", {}).get("warmup_frames", 0))
        
        # Auto-generate output directory if not specified
        if output_dir is None:
            date_str = datetime.now().strftime("%d-%m-%y")
            output_dir = os.path.join(base_path, f"{num_cameras}cams_{date_str}")
        
        if self.verbose:
            print(f"[RENDER] Output directory: {output_dir}")
            print(f"[RENDER] Target FPS: {stage_fps:.2f}")
            print(f"[RENDER] Warmup frames: {warmup_frames}")
            print(f"[RENDER] Scheduling rendering task...")
            print(f"[RENDER] NOTE: This schedules async rendering without waiting for completion")
            print(f"[RENDER]       For synchronized workflows, call render_multi_camera_async() with await")
        
        # Schedule the async rendering task
        # This uses the omni.kit async context which is already running in Isaac Sim
        drone_prim_path = self.cfg.get("drone", {}).get("prim_path", "/World/Drone")
        crop_depth_config = self.render_cfg.get("crop_depth", {})

        import asyncio
        task = asyncio.ensure_future(
            render_multi_camera_async(
                num_cameras, num_frames, output_dir, base_path, stage_fps, warmup_frames,
                drone_prim_path=drone_prim_path,
                crop_depth_config=crop_depth_config
            )
        )
        
        if self.verbose:
            print(f"[RENDER] Task object created: {task}")
            print(f"[RENDER] Task done? {task.done()}")
            print(f"[RENDER] Rendering should begin automatically...")
        
        # Return the output directory path for reference
        return output_dir


async def render_multi_camera_async(
    num_cameras=5,
    num_frames=120,
    output_dir=None,
    base_path="/home/sandro/thesis/renders",
    stage_fps=30.0,
    warmup_frames=0,
    drone_prim_path=None,
    crop_depth_config=None,
):
    """
    Render from multiple cameras and save RGB, depth, and instance segmentation.
    
    Args:
        num_cameras: Number of cameras to render from (default 5)
        num_frames: Number of frames to render (default 120)
        output_dir: Output directory for rendered images
        base_path: Base directory for rendered images (default /home/sandro/thesis/renders)
        stage_fps: Frames per second to drive stage, timeline, and capture cadence
        warmup_frames: Number of warmup frames to run before capturing (default 10)
    """
    
    print(f"[RENDER] *** ASYNC TASK STARTED *** (This confirms the task is executing)")
    
    # Setup output directory
    if output_dir is None:
        # Create folder name: {num_cameras}cams_{DD-MM-YY}
        date_str = datetime.now().strftime("%d-%m-%y")
        folder_name = f"{num_cameras}cams_{date_str}"
        output_dir = os.path.join(base_path, folder_name)

    os.makedirs(output_dir, exist_ok=True)
    
    print(f"[RENDER] Starting multi-camera render")
    print(f"[RENDER] Number of cameras: {num_cameras}")
    print(f"[RENDER] Number of frames: {num_frames}")
    print(f"[RENDER] Output directory: {output_dir}")
    
    # Get the current stage
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        print("[RENDER] Error: No active stage found")
        return

    try:
        original_stage_timecodes = float(stage.GetTimeCodesPerSecond())
    except Exception:
        original_stage_timecodes = None
    try:
        original_stage_frames = float(stage.GetFramesPerSecond())
    except Exception:
        original_stage_frames = None

    try:
        stage_fps = float(stage_fps)
    except (TypeError, ValueError):
        stage_fps = 30.0
    if stage_fps <= 0.0:
        stage_fps = 30.0

    try:
        stage.SetTimeCodesPerSecond(stage_fps)
        stage.SetFramesPerSecond(stage_fps)
    except Exception as exc:
        print(f"[RENDER] Warning: Failed to apply stage FPS {stage_fps}: {exc}")

    try:
        applied_stage_fps = float(stage.GetTimeCodesPerSecond() or stage_fps)
    except Exception:
        applied_stage_fps = stage_fps
    if applied_stage_fps <= 0.0:
        applied_stage_fps = stage_fps if stage_fps > 0.0 else 30.0
    if abs(applied_stage_fps - stage_fps) > 1e-3:
        print(
            f"[RENDER] Warning: Stage reports {applied_stage_fps:.2f} FPS after applying {stage_fps:.2f}. Using reported value."
        )
    stage_fps = applied_stage_fps
    frame_dt = 1.0 / stage_fps

    synthetic_data = SyntheticData.Get()
    previous_semantic_filter = None
    try:
        previous_semantic_filter = synthetic_data.get_instance_mapping_semantic_filter()
    except Exception:
        pass

    try:
        synthetic_data.set_instance_mapping_semantic_filter("class:*")
    except Exception:
        pass

    # Initialize tracking variables
    mask_output_dirs = []
    camera_output_dirs = []
    rgb_dirs = []
    depth_dirs = []
    camera_names = []
    camera_prims = []
    camera_param_annots = []
    camera_helpers = []
    camera_metadata_cache = {}
    frame_records = []

    # Find the drone/object prim for bounding box computation
    drone_prim = _find_drone_prim(stage, preferred_path=drone_prim_path)
    drone_prim_path = str(drone_prim.GetPath()) if drone_prim and drone_prim.IsValid() else drone_prim_path

    # Also get the mover prim path (parent of drone) for mask filtering
    mover_prim_path = None
    if drone_prim and drone_prim.IsValid():
        parent = drone_prim.GetParent()
        if parent and parent.IsValid() and "mover" in str(parent.GetPath()).lower():
            mover_prim_path = str(parent.GetPath())

    # Fallback: check common mover paths
    if not mover_prim_path:
        for path in ["/World/DroneMover", "/World/ObjectMover"]:
            mover_candidate = stage.GetPrimAtPath(path)
            if mover_candidate and mover_candidate.IsValid():
                mover_prim_path = path
                break

    # Build list of valid prim path prefixes for mask filtering
    valid_object_paths = []
    if mover_prim_path:
        valid_object_paths.append(mover_prim_path)
    if drone_prim_path:
        valid_object_paths.append(drone_prim_path)

    if drone_prim_path:
        print(f"[RENDER] Found flying object prim at: {drone_prim_path}")
    if mover_prim_path:
        print(f"[RENDER] Found mover prim at: {mover_prim_path}")
    if not valid_object_paths:
        print("[RENDER] Warning: Could not find drone/object prim. 3D bbox tracking and mask creation will be disabled.")

    timeline = omni.timeline.get_timeline_interface()
    kit_settings = carb.settings.get_settings()
    previous_use_fixed_timestep = None
    previous_stage_timecodes = None
    if kit_settings:
        try:
            previous_use_fixed_timestep = kit_settings.get("/app/player/useFixedTimeStepping")
        except Exception:
            previous_use_fixed_timestep = None
        try:
            previous_stage_timecodes = kit_settings.get("/app/stage/timeCodesPerSecond")
        except Exception:
            previous_stage_timecodes = None

    previous_timeline_fps = _safe_timeline_call(timeline, "get_time_codes_per_seconds")
    previous_time = _safe_timeline_call(timeline, "get_current_time") or 0.0
    previous_end_time = _safe_timeline_call(timeline, "get_end_time")
    was_playing = bool(_safe_timeline_call(timeline, "is_playing"))

    # Force stop timeline before changing FPS to ensure settings apply
    _safe_timeline_call(timeline, "stop")

    if kit_settings:
        try:
            kit_settings.set("/app/player/useFixedTimeStepping", True)
        except Exception as exc:
            print(f"[RENDER] Warning: Failed to enable fixed time stepping: {exc}")
        try:
            kit_settings.set("/app/stage/timeCodesPerSecond", stage_fps)
        except Exception:
            pass

    # Set timeline FPS with multiple attempts to ensure it takes effect
    timeline_fps = stage_fps
    if timeline:
        try:
            timeline.set_time_codes_per_second(stage_fps)
        except Exception as exc:
            print(f"[RENDER] Warning: Failed to set timeline FPS {stage_fps}: {exc}")
        
        # Force ticks per frame to 1 before fetching to ensure proper timing
        _safe_timeline_call(timeline, "set_ticks_per_frame", 1)
        
        # Fetch the actual applied value
        fetched = _safe_timeline_call(timeline, "get_time_codes_per_seconds")
        if fetched:
            try:
                fetched_fps = float(fetched)
                # If timeline doesn't match, try setting again
                if abs(fetched_fps - stage_fps) > 1e-3:
                    print(f"[RENDER] Timeline FPS mismatch detected ({fetched_fps:.2f} vs {stage_fps:.2f}), retrying...")
                    _safe_timeline_call(timeline, "set_time_codes_per_second", stage_fps)
                    fetched = _safe_timeline_call(timeline, "get_time_codes_per_seconds")
                    if fetched:
                        fetched_fps = float(fetched)
                
                # Use the actual timeline FPS for calculations
                timeline_fps = fetched_fps
                if abs(timeline_fps - stage_fps) > 1e-3:
                    print(
                        f"[RENDER] Warning: Timeline FPS {timeline_fps:.2f} differs from stage FPS {stage_fps:.2f}"
                    )
                    print(f"[RENDER] Using stage FPS {stage_fps:.2f} for frame timing regardless")
                    # Force timeline_fps to match stage_fps for consistency
                    timeline_fps = stage_fps
            except (TypeError, ValueError):
                timeline_fps = stage_fps

    print(f"[RENDER] Stage FPS: {stage_fps:.2f} (Δt={frame_dt:.4f}s)")

    # Disable capture on play to ensure manual frame control and prevent extra captures
    # This is CRITICAL for RGB/mask synchronization:
    # - With capture-on-play disabled, only explicit step_async() calls trigger captures
    # - Each step_async() captures exactly once for all attached render products and annotators
    # - This keeps per-iteration outputs (RGB, depth, masks) perfectly aligned
    rep.orchestrator.set_capture_on_play(False)

    # Warm up the stage before attaching render products to avoid stale frames
    kit_app = omni.kit.app.get_app()
    if kit_app and hasattr(kit_app, "next_update_async"):
        print(f"[RENDER] Warming up simulation for {warmup_frames} update cycles...")
        _safe_timeline_call(timeline, "stop")
        _safe_timeline_call(timeline, "set_current_time", 0.0)
        for _ in range(warmup_frames):
            await kit_app.next_update_async()
        print("[RENDER] Warmup complete. Scene stabilized at t=0")

    _safe_timeline_call(timeline, "stop")
    _safe_timeline_call(timeline, "set_current_time", 0.0)

    try:
        # Create render products and writers for each camera
        render_products = []
        writers = []

        for i in range(num_cameras):
            camera_name = f"cam_{i+1:02d}"
            camera_path = f"/World/{camera_name}"

            camera_prim = stage.GetPrimAtPath(camera_path)
            if not camera_prim.IsValid():
                print(f"[RENDER] Warning: Camera {camera_path} not found, skipping")
                continue

            print(f"[RENDER] Setting up render product for {camera_name} at {camera_path}")

            # Create render product with QHD resolution
            rp = rep.create.render_product(camera_path, (2560, 1440))
            render_products.append(rp)

            writer = rep.writers.get("BasicWriter")
            camera_output_dir = os.path.join(output_dir, camera_name)
            os.makedirs(camera_output_dir, exist_ok=True)
            rgb_dir = os.path.join(camera_output_dir, "rgb")
            depth_dir = os.path.join(camera_output_dir, "depth")
            mask_dir = os.path.join(camera_output_dir, "mask")
            for folder in (rgb_dir, depth_dir, mask_dir):
                os.makedirs(folder, exist_ok=True)

            writer.initialize(
                output_dir=camera_output_dir,
                rgb=True,
                instance_segmentation=True,  # Enable instance segmentation in BasicWriter
                distance_to_image_plane=True,
            )

            writer.attach(rp)
            writers.append(writer)
            
            cam_params_annot = rep.AnnotatorRegistry.get_annotator("camera_params")
            cam_params_annot.attach(rp)

            mask_output_dirs.append(mask_dir)
            camera_output_dirs.append(camera_output_dir)
            rgb_dirs.append(rgb_dir)
            depth_dirs.append(depth_dir)
            camera_names.append(camera_name)
            camera_prims.append(camera_prim)
            camera_param_annots.append(cam_params_annot)

            rp_path_attr = getattr(rp, "path", None)
            helper = _create_camera_api_helper(
                camera_path,
                camera_name,
                str(rp_path_attr) if rp_path_attr is not None else None,
            )
            camera_helpers.append(helper)

            print(f"[RENDER] {camera_name} output: {camera_output_dir}")
            print(f"[RENDER] {camera_name} drone masks: {mask_dir}")

        if not render_products:
            print("[RENDER] Error: No valid cameras found")
            return

        print(f"[RENDER] Successfully set up {len(render_products)} camera render products")

        # Now setup timeline for actual capture (keep stopped - step_async will advance it)
        _safe_timeline_call(timeline, "stop")
        _safe_timeline_call(timeline, "set_current_time", 0.0)
        
        # Set timeline end time to accommodate all frames
        desired_end_time = max(previous_end_time or 0.0, num_frames * frame_dt)
        if desired_end_time <= 0.0:
            desired_end_time = max(num_frames * frame_dt, 1.0)
        _safe_timeline_call(timeline, "set_end_time", desired_end_time)
        _safe_timeline_call(timeline, "commit")
        # DO NOT call play() - let step_async handle timeline advancement
        # This prevents an initial capture at t=0 before the loop starts
        
        # CRITICAL FIX: Reset BasicWriter frame counters to prevent initial capture misalignment
        # BasicWriter creates an internal frame counter when attached. To ensure frame 0 is
        # our first controlled capture, we detach and re-attach all writers to reset counters.
        print("[RENDER] Resetting writer frame counters to ensure proper synchronization...")
        for writer in writers:
            writer.detach()
        
        # Small delay to ensure detach completes
        if kit_app and hasattr(kit_app, "next_update_async"):
            await kit_app.next_update_async()
        
        # Re-attach writers to render products (this resets their frame counters to 0)
        for idx, (writer, rp) in enumerate(zip(writers, render_products)):
            writer.attach(rp)
        
        print("[RENDER] Writers reset and re-attached, frame counters start at 0...")

        print(f"[RENDER] Starting frame capture (frames 0000-{num_frames-1:04d}, total {num_frames} frames)...")
        frames_captured = 0

        cam_params_annot = rep.AnnotatorRegistry.get_annotator("camera_params")
        for rp in render_products:
            cam_params_annot.attach(rp)

        # ============================================================================
        # Frame Capture Loop - Synchronization Strategy
        # ============================================================================
        # Each iteration captures exactly one frame for all cameras:
        # 1. Set timeline to exact position (frame * frame_dt)
        # 2. Update stage to reflect timeline change
        # 3. Compute drone 3D bbox at this exact timeline position
        # 4. Call step_async(delta_time=0.0) to capture WITHOUT advancing time
        # 5. BasicWriter writes RGB, depth, AND instance segmentation in lockstep
        #
        # Mask post-processing happens after capture completes (see below)
        # ============================================================================
        try:
            for frame in range(0, num_frames):
                # Set timeline to exact frame position
                current_time = frame * frame_dt
                _safe_timeline_call(timeline, "set_current_time", current_time)
                
                # Update stage to reflect timeline change
                if kit_app and hasattr(kit_app, "next_update_async"):
                    await kit_app.next_update_async()
                
                # Compute drone's 3D bounding box in world space at this timeline position
                drone_bbox_3d = None
                if drone_prim and drone_prim.IsValid():
                    drone_bbox_3d = _compute_world_bbox_3d(drone_prim, seconds=current_time)
                
                # Capture at this exact timeline position without advancing time
                # delta_time=0.0 ensures no implicit timeline advance
                # BasicWriter automatically writes RGB, depth, and instance segmentation
                await rep.orchestrator.step_async(rt_subframes=32, delta_time=0.0, pause_timeline=True)
                frames_captured += 1

                # Build frame record for drone position tracking
                frame_record = {
                    "frame_index": frame,
                    "time_seconds": float(current_time),
                    "drone_position_3d": drone_bbox_3d["center"] if drone_bbox_3d else None,
                    "cameras": [],
                }

                for idx in range(len(camera_names)):
                    width = height = None
                    camera_entry = {
                        "name": camera_names[idx],
                        "drone_center_2d": None,
                        "visible": False,
                        "depth": None,
                        "bbox_2d": None,
                    }

                    camera_params = None
                    if idx < len(camera_param_annots):
                        try:
                            camera_params = camera_param_annots[idx].get_data()
                        except Exception:
                            pass

                    if camera_params:
                        try:
                            resolution = camera_params["renderProductResolution"]
                            width = int(resolution[0])
                            height = int(resolution[1])

                            if camera_names[idx] not in camera_metadata_cache:
                                metadata = _build_camera_metadata(
                                    camera_names[idx], camera_prims[idx], render_products[idx], camera_params
                                )
                                if metadata:
                                    camera_metadata_cache[camera_names[idx]] = metadata
                            
                            # Compute depth from projected center
                            if drone_bbox_3d and drone_bbox_3d.get("center") is not None:
                                try:
                                    view_matrix = _reshape_to_matrix(camera_params["cameraViewTransform"])
                                    projection_matrix = _reshape_to_matrix(camera_params["cameraProjection"])
                                    
                                    center_proj = _project_point_to_screen(
                                        drone_bbox_3d["center"],
                                        view_matrix,
                                        projection_matrix,
                                        width,
                                        height,
                                    )
                                    if center_proj and center_proj["in_front"]:
                                        camera_entry["depth"] = float(center_proj["depth"])
                                except Exception as exc:
                                    print(f"[RENDER] Warning: Failed to compute depth for {camera_names[idx]} frame {frame}: {exc}")
                        except Exception:
                            pass

                    frame_record["cameras"].append(camera_entry)

                    # _write_camera_frame_record(
                    #     output_dir,
                    #     camera_names[idx],
                    #     frame,
                    #     camera_entry,
                    # )

                frame_records.append(frame_record)

                if frame % 10 == 0 or frame == 0:
                    print(f"[RENDER] Captured frame {frame}/{num_frames-1}")
                    if drone_bbox_3d:
                        print(f"[RENDER]   Drone 3D bbox center: {drone_bbox_3d['center']}")
        finally:
            _safe_timeline_call(timeline, "stop")
            if previous_end_time is not None:
                _safe_timeline_call(timeline, "set_end_time", previous_end_time)
            _safe_timeline_call(timeline, "set_current_time", previous_time)
            _safe_timeline_call(timeline, "commit")
            if timeline and previous_timeline_fps is not None:
                try:
                    timeline.set_time_codes_per_second(previous_timeline_fps)
                except Exception:
                    pass
            if was_playing:
                _safe_timeline_call(timeline, "play")
            if original_stage_timecodes is not None:
                try:
                    stage.SetTimeCodesPerSecond(original_stage_timecodes)
                except Exception:
                    pass
            if original_stage_frames is not None:
                try:
                    stage.SetFramesPerSecond(original_stage_frames)
                except Exception:
                    pass
            if kit_settings and previous_stage_timecodes is not None:
                try:
                    kit_settings.set("/app/stage/timeCodesPerSecond", previous_stage_timecodes)
                except Exception:
                    pass
            if kit_settings and previous_use_fixed_timestep is not None:
                try:
                    kit_settings.set("/app/player/useFixedTimeStepping", previous_use_fixed_timestep)
                except Exception:
                    pass

        print(f"[RENDER] Frame capture complete ({frames_captured} frames)")
        
        # Stop timeline immediately to prevent any further captures
        _safe_timeline_call(timeline, "stop")

        print("[RENDER] Cleaning up render products...")
        for writer in writers:
            writer.detach()

        for annot in camera_param_annots:
            try:
                annot.detach()
            except Exception:
                pass

        for helper in camera_helpers:
            if helper is None:
                continue
            try:
                helper.destroy()
            except Exception:
                pass

        for rp in render_products:
            rp.destroy()

        print("[RENDER] Waiting for file writes to complete...")
        await rep.orchestrator.wait_until_complete_async()

        # ============================================================================
        # Post-process instance segmentation to create drone masks
        # ============================================================================
        print("[RENDER] Post-processing instance segmentation to extract drone masks...")
        for idx, camera_name in enumerate(camera_names):
            camera_output_dir = camera_output_dirs[idx]
            mask_dir = mask_output_dirs[idx]
            
            inst_png_paths = sorted(Path(camera_output_dir).glob("instance_segmentation_*.png"))
            if not inst_png_paths:
                print(f"[RENDER] Warning: No instance segmentation files found for {camera_name}; skipping mask extraction")
                continue

            first_frame_idx = int(inst_png_paths[0].stem.split("_")[-1]) if inst_png_paths else 0

            # Process each available frame on disk (handles extra captures)
            for inst_png_path in inst_png_paths:
                try:
                    frame = int(inst_png_path.stem.split("_")[-1])
                except (IndexError, ValueError):
                    print(f"[RENDER] Warning: Could not parse frame index from {inst_png_path.name}; skipping")
                    continue

                inst_map_json_path = inst_png_path.with_name(f"instance_segmentation_mapping_{frame:04d}.json")
                
                if not inst_map_json_path.exists():
                    print(f"[RENDER] Warning: Missing instance segmentation files for {camera_name} frame {frame}")
                    continue

                try:
                    # Load instance segmentation image (RGBA format from BasicWriter)
                    inst_img = Image.open(inst_png_path)
                    img = np.array(inst_img)
                    
                    # Load instance ID to prim path mapping
                    with open(inst_map_json_path, 'r') as f:
                        id2label = json.load(f)
                    
                    inst_semantics_json_path = inst_png_path.with_name(
                        f"instance_segmentation_semantics_mapping_{frame:04d}.json"
                    )

                    semantics_map = {}
                    if inst_semantics_json_path.exists():
                        try:
                            with open(inst_semantics_json_path, "r") as f:
                                semantics_map = json.load(f)
                        except Exception as exc:
                            print(f"[RENDER] Warning: Failed to load semantics mapping for {camera_name} frame {frame}: {exc}")

                    body_colors = set()
                    prop_colors = set()
                    other_object_colors = set()
                    unlabelled_colors = set()

                    def _is_flying_object_prim(prim_path_str):
                        """Check if prim path belongs to the flying object (drone/bird)."""
                        if not prim_path_str:
                            return False
                        for valid_prefix in valid_object_paths:
                            if prim_path_str.startswith(valid_prefix):
                                return True
                        return False

                    # First pass: check semantics map for labeled objects
                    if semantics_map:
                        for color_str, payload in semantics_map.items():
                            classes = []
                            if isinstance(payload, dict):
                                cls_value = payload.get("class")
                                if isinstance(cls_value, str):
                                    classes = [part.strip().lower() for part in cls_value.split(",") if part.strip()]
                                elif isinstance(cls_value, (list, tuple)):
                                    classes = [str(part).strip().lower() for part in cls_value]
                            else:
                                classes = [str(payload).strip().lower()]

                            color_tuple = _parse_color_tuple(color_str)
                            if not color_tuple:
                                continue

                            # Categorize by semantic class (supports both drone and bird labels)
                            # Body parts: drone_body, bird_body
                            if any("_body" in cls for cls in classes):
                                body_colors.add(color_tuple)
                                if frame == first_frame_idx:
                                    print(f"[RENDER] {camera_name}: Body color {color_tuple} for classes {classes}")
                            # Propeller/wing parts: drone_prop, bird_wing
                            elif any("_prop" in cls or "_wing" in cls for cls in classes):
                                prop_colors.add(color_tuple)
                                if frame == first_frame_idx:
                                    print(f"[RENDER] {camera_name}: Prop/wing color {color_tuple} for classes {classes}")
                            # Other flying object parts: drone, bird, bird_head, bird_tail, bird_leg
                            elif any(cls in ["drone", "bird"] or cls.startswith("bird_") or cls.startswith("drone_") for cls in classes):
                                other_object_colors.add(color_tuple)
                                if frame == first_frame_idx:
                                    print(f"[RENDER] {camera_name}: Flying object color {color_tuple} for classes {classes}")
                
                    # Update: Use more precise keyword matching to avoid matching 'props' folder
                    def is_part(label, part_keywords):
                        label_parts = label.replace("/", "_").split("_")
                        return any(kw in label_parts for kw in part_keywords)

                    # Second pass: check id2label mapping using PRIM PATH filtering
                    # This ensures we only include objects under the mover/drone path
                    for color_str, prim_info in id2label.items():
                        if isinstance(prim_info, dict):
                            label = prim_info.get("class", "")
                            # Some formats store prim path directly
                            prim_path_str = str(prim_info) if not label else ""
                        else:
                            label = ""
                            prim_path_str = str(prim_info)

                        color_tuple = _parse_color_tuple(color_str)
                        if color_tuple is None:
                            continue

                        # CRITICAL: Only include if prim path is under the flying object hierarchy
                        path_to_check = prim_path_str or label
                        if not _is_flying_object_prim(path_to_check):
                            if label.strip().upper() == "UNLABELLED":
                                unlabelled_colors.add(color_tuple)
                            continue

                        # This prim is part of the flying object - add to mask
                        label_lower = label.lower() if label else prim_path_str.lower()
                        
                        # Refined keyword matching
                        body_kws = ["body", "hull", "torso", "head", "beak", "tail", "leg", "foot", "claw"]
                        prop_kws = ["prop", "propeller", "wing"]
                        
                        if is_part(label_lower, body_kws):
                            body_colors.add(color_tuple)
                            if frame == first_frame_idx:
                                print(f"[RENDER] {camera_name}: Found body color {color_tuple} -> {path_to_check}")
                        elif is_part(label_lower, prop_kws):
                            prop_colors.add(color_tuple)
                            if frame == first_frame_idx:
                                print(f"[RENDER] {camera_name}: Found prop/wing color {color_tuple} -> {path_to_check}")
                        else:
                            # Any other part of the flying object
                            other_object_colors.add(color_tuple)
                            if frame == first_frame_idx:
                                print(f"[RENDER] {camera_name}: Found flying object color {color_tuple} -> {path_to_check}")

                    # If no parts found but unlabelled exists under flying object path, assume it's all body
                    if not (body_colors or prop_colors or other_object_colors) and unlabelled_colors:
                        # Only use unlabelled if we couldn't find anything else
                        body_colors.update(unlabelled_colors)
                        if frame == first_frame_idx:
                            print(f"[RENDER] {camera_name}: No explicit parts found, using UNLABELLED colors as body")

                    # Create multi-level mask: 255 for body (white), 128 for props/wings (gray), 0 for background
                    if not (body_colors or prop_colors or other_object_colors):
                        mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
                        if frame == first_frame_idx:
                            print(f"[RENDER] {camera_name}: No flying object colors found (checked {len(id2label)} entries, valid paths: {valid_object_paths})")
                    else:
                        if frame == first_frame_idx:
                            print(f"[RENDER] {camera_name}: Creating multi-level mask - Body: {len(body_colors)}, Props/Wings: {len(prop_colors)}, Other: {len(other_object_colors)} colors")

                        mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
                        channel_count = img.shape[2]

                        # Apply body pixels (white = 255)
                        for color_tuple in body_colors:
                            compare_channels = min(len(color_tuple), channel_count)
                            color_arr = np.array(color_tuple[:compare_channels], dtype=img.dtype)
                            match = np.all(img[..., :compare_channels] == color_arr, axis=-1)
                            mask[match] = 255

                        # Apply propeller/wing pixels (gray = 128)
                        for color_tuple in prop_colors:
                            compare_channels = min(len(color_tuple), channel_count)
                            color_arr = np.array(color_tuple[:compare_channels], dtype=img.dtype)
                            match = np.all(img[..., :compare_channels] == color_arr, axis=-1)
                            mask[match] = 128

                        # Apply other flying object pixels (white = 255, treat as body)
                        for color_tuple in other_object_colors:
                            compare_channels = min(len(color_tuple), channel_count)
                            color_arr = np.array(color_tuple[:compare_channels], dtype=img.dtype)
                            match = np.all(img[..., :compare_channels] == color_arr, axis=-1)
                            mask[match] = 255

                    
                    # Save drone-only mask
                    mask_path = Path(mask_dir) / f"drone_mask_{frame:04d}.png"
                    Image.fromarray(mask).save(mask_path)
                    
                    # Update frame record with mask statistics if drone is visible
                    if frame < len(frame_records) and idx < len(frame_records[frame]["cameras"]):
                        if mask.any():
                            # Compute mask statistics
                            ys, xs = np.where(mask > 0)
                            if len(xs) > 0 and len(ys) > 0:
                                center_x = float(np.mean(xs))
                                center_y = float(np.mean(ys))
                                bbox = {
                                    "x_min": float(np.min(xs)),
                                    "y_min": float(np.min(ys)),
                                    "x_max": float(np.max(xs)),
                                    "y_max": float(np.max(ys)),
                                }
                                
                                frame_records[frame]["cameras"][idx]["drone_center_2d"] = [center_x, center_y]
                                frame_records[frame]["cameras"][idx]["visible"] = True
                                frame_records[frame]["cameras"][idx]["bbox_2d"] = bbox
                    elif frame >= len(frame_records):
                        print(f"[RENDER] Note: Mask frame {frame:04d} for {camera_name} has no matching frame record entry; skipping metadata update")
                except Exception as exc:
                    print(f"[RENDER] Error processing mask for {camera_name} frame {frame}: {exc}")
            
            # Clean up instance segmentation files after mask extraction
            print(f"[RENDER] Cleaning up instance segmentation files for {camera_name}...")
            for file_path in Path(camera_output_dir).glob("instance_*"):
                try:
                    file_path.unlink()
                except Exception as exc:
                    print(f"[RENDER] Warning: Failed to remove {file_path.name}: {exc}")
        
        print("[RENDER] Mask post-processing complete")

        # Fill in missing camera metadata
        for idx, name in enumerate(camera_names):
            if name not in camera_metadata_cache:
                rp_path = None
                if idx < len(render_products):
                    rp_candidate = getattr(render_products[idx], "path", None)
                    rp_path = str(rp_candidate) if rp_candidate is not None else None
                prim_path = None
                if idx < len(camera_prims):
                    prim = camera_prims[idx]
                    prim_path = str(prim.GetPath()) if prim and prim.IsValid() else None

                camera_metadata_cache[name] = {
                    "name": name,
                    "camera_prim_path": prim_path,
                    "render_product_path": rp_path,
                    "resolution": None,
                    "intrinsics": None,
                    "extrinsics": None,
                    "projection_matrix": None,
                    "meters_per_scene_unit": None,
                    "note": "Camera parameters not available; check camera configuration.",
                }

        observation_payload = {
            "description": "Drone world pose and per-camera projections exported from render.py",
            "metadata": {
                "num_frames": len(frame_records),
                "stage_fps": stage_fps,
                "timeline_fps": timeline_fps,
                "frame_dt": frame_dt,
                "drone_prim_path": drone_prim_path,
                "cameras_recorded": camera_names,
            },
            "cameras": list(camera_metadata_cache.values()),
            "frames": frame_records,
        }

        if frame_records:
            observation_payload["metadata"]["time_span_seconds"] = {
                "start": frame_records[0]["time_seconds"],
                "end": frame_records[-1]["time_seconds"],
            }
        else:
            observation_payload["metadata"]["time_span_seconds"] = {"start": 0.0, "end": 0.0}

        observation_path = os.path.join(output_dir, "drone_camera_observations.json")
        try:
            with open(observation_path, "w", encoding="utf-8") as json_file:
                json.dump(observation_payload, json_file, indent=2)
            print(f"[RENDER] Drone/camera observation JSON saved: {observation_path}")
        except Exception as exc:
            print(f"[RENDER] Warning: Failed to write observation JSON: {exc}")

        # Organize output files
        for cam_dir, rgb_dir, depth_dir in zip(camera_output_dirs, rgb_dirs, depth_dirs):
            try:
                for entry in os.listdir(cam_dir):
                    src_path = os.path.join(cam_dir, entry)
                    if not os.path.isfile(src_path):
                        continue
                    if entry.startswith("rgb_"):
                        shutil.move(src_path, os.path.join(rgb_dir, entry))
                    elif entry.startswith("distance_to_image_plane_") or entry.startswith("distance_to_camera_"):
                        shutil.move(src_path, os.path.join(depth_dir, entry))
            except Exception:
                pass

        print(f"[RENDER] Rendering complete! Output saved to: {output_dir}")

        # Optional: Crop depth maps to object bounding boxes to save storage
        # This is a post-processing step controlled by config
        # Full scene depth: ~14.7MB per frame
        # Cropped object depth: ~10-100KB per frame (99%+ reduction)
        if crop_depth_config is None:
            crop_depth_config = {}

        crop_depth_enabled = crop_depth_config.get("enabled", False)
        crop_depth_padding = crop_depth_config.get("padding", 10)
        crop_depth_keep_originals = crop_depth_config.get("keep_originals", False)

        # Also check environment variable override
        if os.environ.get("ISAACSIM_CROP_DEPTH", "").lower() in ("1", "true", "yes"):
            crop_depth_enabled = True
            if os.environ.get("ISAACSIM_CROP_DEPTH_PADDING"):
                crop_depth_padding = int(os.environ.get("ISAACSIM_CROP_DEPTH_PADDING", "10"))
            if os.environ.get("ISAACSIM_CROP_DEPTH_KEEP"):
                crop_depth_keep_originals = os.environ.get("ISAACSIM_CROP_DEPTH_KEEP", "").lower() in ("1", "true", "yes")

        if crop_depth_enabled:
            print("[RENDER] Cropping depth maps to object bounding boxes...")
            try:
                from crop_depth import process_render_directory
                crop_stats = process_render_directory(
                    Path(output_dir),
                    padding=crop_depth_padding,
                    keep_originals=crop_depth_keep_originals,
                    verbose=True
                )
                if crop_stats.get("storage_saved_mb"):
                    print(f"[RENDER] Depth cropping saved {crop_stats['storage_saved_mb']:.1f} MB")
            except ImportError:
                print("[RENDER] Warning: crop_depth.py not found, skipping depth cropping")
            except Exception as exc:
                print(f"[RENDER] Warning: Depth cropping failed: {exc}")
    finally:
        try:
            restore_filter = previous_semantic_filter if previous_semantic_filter else "class:*"
            synthetic_data.set_instance_mapping_semantic_filter(restore_filter)
        except Exception:
            pass


def render_multi_camera(num_cameras=5, num_frames=120, output_dir=None):
    """Legacy function - wraps Render class for backward compatibility."""
    cfg = {
        "render": {
            "num_cameras": num_cameras,
            "num_frames": num_frames,
            "output_dir": output_dir,
            "fps": 30.0,
        },
        "execution": {"verbose": False}
    }
    renderer = Render(cfg)
    return renderer.run()


if __name__ == "__main__":
    # Example: load from config.yaml
    import yaml
    from pathlib import Path
    
    config_path = Path(__file__).parent / "config.yaml"
    if config_path.exists():
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        renderer = Render(cfg)
        renderer.run()
    else:
        # Default configuration
        NUM_CAMERAS = 5
        NUM_FRAMES = 120
        OUTPUT_DIR = None
        
        render_multi_camera(NUM_CAMERAS, NUM_FRAMES, OUTPUT_DIR)
