#!/usr/bin/env python3

"""
PositionCameras: Isaac Sim Camera Rig Setup
Creates N cameras in a circle looking at a central aim point.
"""

import argparse
import numpy as np
from math import pi, cos, sin
from pathlib import Path
import json
import yaml

import config_utils

from config_utils import resolve_fps_from_config

# IsaacSim imports
from isaacsim.core.api import World
from isaacsim.sensors.camera import Camera
import isaacsim.core.utils.numpy.rotations as rot_utils
import omni.usd
import omni.kit.commands
import omni.client
from pxr import Usd, UsdGeom
import carb
from isaacsim.core.utils.stage import get_stage_units


def _collect_search_roots():
    roots = []
    script_dir = Path(__file__).resolve().parent
    cwd = Path.cwd()
    roots.extend([script_dir, cwd])

    try:
        cfg_module_dir = Path(config_utils.__file__).resolve().parent
        roots.append(cfg_module_dir)
    except Exception:
        cfg_module_dir = None

    deduped = []
    seen = set()
    for root in filter(None, roots):
        lineage = [root] + list(root.parents)
        for candidate in lineage:
            key = str(candidate)
            if key not in seen:
                deduped.append(candidate)
                seen.add(key)
    return deduped


_SEARCH_ROOTS = _collect_search_roots()


def _resolve_existing_path(path_arg):
    path_candidate = Path(path_arg)

    if path_candidate.is_absolute():
        return path_candidate if path_candidate.exists() else None

    seen = set()
    for base_dir in _SEARCH_ROOTS:
        if base_dir is None:
            continue
        for candidate in (base_dir / path_candidate, base_dir / "isaacSim_snippets" / path_candidate):
            try:
                resolved = candidate.resolve()
            except Exception:
                resolved = candidate
            key = str(resolved)
            if key in seen:
                continue
            seen.add(key)
            if resolved.exists():
                return resolved

    return None


def load_camera_intrinsics_from_yaml(yaml_path="cam_intrinsics.yaml"):
    """Load camera intrinsics database from YAML file.
    
    Args:
        yaml_path: Path to the camera intrinsics YAML file
        
    Returns:
        dict: Dictionary of camera intrinsics configurations
    """
    resolved_yaml = _resolve_existing_path(yaml_path)
    if resolved_yaml is None:
        raise FileNotFoundError(f"Camera intrinsics file not found: {Path(yaml_path)}")

    with open(resolved_yaml, 'r') as f:
        intrinsics_db = yaml.safe_load(f)
    
    return intrinsics_db


def get_camera_intrinsics(camera_type, intrinsics_yaml="cam_intrinsics.yaml"):
    """Get camera intrinsics for a specific camera type.
    
    Args:
        camera_type: Camera type string (e.g., "advanced_pinhole", "fisheye")
        intrinsics_yaml: Path to camera intrinsics YAML file
        
    Returns:
        dict: Camera intrinsics configuration
    """
    intrinsics_db = load_camera_intrinsics_from_yaml(intrinsics_yaml)
    
    if camera_type not in intrinsics_db:
        available = list(intrinsics_db.keys())
        raise ValueError(
            f"Camera type '{camera_type}' not found in {intrinsics_yaml}. "
            f"Available types: {', '.join(available)}"
        )
    
    return intrinsics_db[camera_type]


def get_scene_parameters():
    """Get scene parameters from the currently loaded Isaac Sim stage."""
    stage = omni.usd.get_context().get_stage()
    
    if stage is None:
        print("ERROR: No stage is currently loaded!")
        return None
    
    up_axis_token = UsdGeom.GetStageUpAxis(stage)
    up_axis = str(up_axis_token)
    
    meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage)
    stage_units = get_stage_units()
    
    if meters_per_unit == 1.0:
        units_description = "meters (m)"
        units_short = "m"
    elif meters_per_unit == 0.01:
        units_description = "centimeters (cm)"
        units_short = "cm"
    elif meters_per_unit == 0.001:
        units_description = "millimeters (mm)"
        units_short = "mm"
    elif meters_per_unit == 1000.0:
        units_description = "kilometers (km)"
        units_short = "km"
    else:
        units_description = f"custom ({meters_per_unit} meters per unit)"
        units_short = "custom"
    
    scene_params = {
        "up_axis": up_axis,
        "meters_per_unit": meters_per_unit,
        "stage_units": stage_units,
        "units_description": units_description,
        "units_short": units_short
    }
    
    return scene_params


def scale_K_to_resolution(intr_cfg, target_width, target_height):
    """Return a copy of intrinsics with camera matrix scaled to a new resolution."""
    if not intr_cfg:
        return intr_cfg

    scaled = intr_cfg.copy()
    K = np.array(intr_cfg.get("camera_matrix", np.eye(3)), dtype=float)
    original_resolution = intr_cfg.get("original_resolution") or [target_width, target_height]
    orig_w = float(original_resolution[0]) if original_resolution[0] else float(target_width)
    orig_h = float(original_resolution[1]) if original_resolution[1] else float(target_height)

    sx = float(target_width) / orig_w if orig_w else 1.0
    sy = float(target_height) / orig_h if orig_h else 1.0

    K[0, 0] *= sx
    K[1, 1] *= sy
    K[0, 2] *= sx
    K[1, 2] *= sy

    scaled["camera_matrix"] = K.tolist()
    scaled["original_resolution"] = [target_width, target_height]
    return scaled


class PositionCameras:
    """Create and position cameras in a circular rig."""
    
    def __init__(self, cfg):
        """Initialize with configuration dict."""
        self.cfg = cfg
        self.cameras_cfg = cfg.get("cameras", {})
        self.verbose = cfg.get("execution", {}).get("verbose", False)
        self.cameras = []
        self.aim_point = None
        self.scene_params = None
    
    def run(self):
        """Create and position cameras based on config."""
        if self.verbose:
            print("[CAMS] Starting camera setup...")
        
        # Detect scene parameters first
        auto_detect = self.cameras_cfg.get("auto_detect_scene", True)
        if auto_detect:
            if self.verbose:
                print("[CAMS] Detecting scene parameters...")
            self.scene_params = get_scene_parameters()
        
        # Extract configuration parameters
        cfg = self.cameras_cfg
        
        NUM_CAMERAS = max(1, min(8, int(cfg.get("num_cameras", 5))))
        SIDE_METERS = float(cfg.get("side_meters", 5.0))
        CAM_HEIGHT = float(cfg.get("cam_height", 2.0))
        AIM_POINT_HEIGHT_OFFSET = float(cfg.get("aim_point_height_offset", -5.0))
        ROTATION_OFFSET = float(cfg.get("rotation_offset", -pi/2))
        POSITION_OFFSET_X = float(cfg.get("position_offset_x", 1800.0))
        POSITION_OFFSET_Y = float(cfg.get("position_offset_y", 8000.0))
        
        RESOLUTION = tuple(cfg.get("resolution", [512, 512]))
        if cfg.get("frequency") is not None:
            FREQUENCY = int(cfg.get("frequency"))
        else:
            FREQUENCY = max(1, int(round(resolve_fps_from_config(self.cfg))))
        
        # Load camera intrinsics from YAML based on camera_type
        camera_type = cfg.get("camera_type", "advanced_pinhole")
        intrinsics_yaml = cfg.get("intrinsics_yaml", "cam_intrinsics.yaml")
        
        try:
            intrinsics_cfg = get_camera_intrinsics(camera_type, intrinsics_yaml)
            if self.verbose:
                print(f"[CAMS] Loaded '{camera_type}' intrinsics from {intrinsics_yaml}")
                print(f"[CAMS]   Description: {intrinsics_cfg.get('description', 'N/A')}")
        except (FileNotFoundError, ValueError) as e:
            print(f"[CAMS] WARNING: {e}")
            print(f"[CAMS] Using default intrinsics configuration")
            intrinsics_cfg = {}
        
        # Optional focus distance override from config
        focus_distance_override = cfg.get("focus_distance", None)
        if focus_distance_override is not None:
            intrinsics_cfg = intrinsics_cfg.copy()
            intrinsics_cfg["focus_distance"] = focus_distance_override
        
        # Get or create world using detected stage units
        stage_units = self.scene_params['stage_units'] if self.scene_params else 1.0
        try:
            world = World.instance()
            if world is None:
                world = World(stage_units_in_meters=stage_units)
        except:
            world = World(stage_units_in_meters=stage_units)
        
        if self.verbose:
            print(f"[CAMS] Using stage units: {stage_units} meters per unit")
        
        # Calculate camera rig geometry
        radius = SIDE_METERS / 2.0
        cam_height = CAM_HEIGHT
        aim_height = cam_height + AIM_POINT_HEIGHT_OFFSET
        
        if self.scene_params and self.scene_params['up_axis'] == 'Y':
            # Y-up coordinate system
            self.aim_point = np.array([
                POSITION_OFFSET_X,
                aim_height,
                POSITION_OFFSET_Y
            ])
            if self.verbose:
                print("[CAMS] Using Y-up coordinate system")
        else:
            # Z-up coordinate system (default)
            self.aim_point = np.array([
                POSITION_OFFSET_X,
                POSITION_OFFSET_Y,
                aim_height
            ])
            if self.verbose:
                print(f"[CAMS] Using Z-up coordinate system")
        
        # Apply unit scaling if detected
        if self.scene_params and self.scene_params['meters_per_unit'] != 1.0:
            scale_factor = 1.0 / self.scene_params['meters_per_unit']
            radius *= scale_factor
            cam_height *= scale_factor
            self.aim_point *= scale_factor
            POSITION_OFFSET_X *= scale_factor
            POSITION_OFFSET_Y *= scale_factor
        
        # Create cameras in circular arrangement
        for i in range(NUM_CAMERAS):
            ang = 2.0 * pi * i / NUM_CAMERAS + ROTATION_OFFSET
            
            # Position cameras based on up axis
            if self.scene_params and self.scene_params['up_axis'] == 'Y':
                cam_pos = np.array([
                    radius * cos(ang) + POSITION_OFFSET_X,
                    cam_height,
                    radius * sin(ang) + POSITION_OFFSET_Y,
                ])
            else:
                cam_pos = np.array([
                    radius * cos(ang) + POSITION_OFFSET_X,
                    radius * sin(ang) + POSITION_OFFSET_Y,
                    cam_height,
                ])
            
            camera_name = f"cam_{i + 1:02d}"
            
            # Calculate look direction and orientation
            dx = self.aim_point[0] - cam_pos[0]
            dy = self.aim_point[1] - cam_pos[1]
            dz = self.aim_point[2] - cam_pos[2]
            
            horizontal_dist = np.sqrt(dx*dx + dy*dy)
            yaw = np.arctan2(dy, dx)
            pitch = -np.arctan2(dz, horizontal_dist)
            roll = 0.0
            
            # Convert to quaternion (ZYX Euler order)
            cy = np.cos(yaw * 0.5)
            sy = np.sin(yaw * 0.5)
            cp = np.cos(pitch * 0.5)
            sp = np.sin(pitch * 0.5)
            cr = np.cos(roll * 0.5)
            sr = np.sin(roll * 0.5)
            
            w = cr * cp * cy + sr * sp * sy
            x = sr * cp * cy - cr * sp * sy
            y = cr * sp * cy + sr * cp * sy
            z = cr * cp * sy - sr * sp * cy
            
            quaternion_wxyz = np.array([w, x, y, z])
            
            # Create camera
            camera = Camera(
                prim_path=f"/World/{camera_name}",
                position=cam_pos,
                frequency=FREQUENCY,
                resolution=RESOLUTION,
                orientation=quaternion_wxyz,
            )
            
            self.cameras.append(camera)
            
            if self.verbose:
                distance_to_aim = np.linalg.norm(self.aim_point - cam_pos)
                yaw_deg = np.degrees(yaw)
                pitch_deg = np.degrees(pitch)
                print(f"[CAMS] {camera_name}: pos={cam_pos}, distance_to_aim={distance_to_aim:.2f}")
        
        # Initialize cameras first
        initialize_cameras(self.cameras, self.scene_params, intrinsics_cfg, self.aim_point)
        
        # Configure intrinsics after initialization
        if intrinsics_cfg:
            self._configure_intrinsics(intrinsics_cfg)
        
        if self.verbose:
            print(f"[CAMS] Created {len(self.cameras)} cameras")
        
        return self.cameras, self.aim_point, self.scene_params
    
    def _configure_intrinsics(self, intrinsics_cfg):
        """Configure camera intrinsics for all cameras."""
        for i, camera in enumerate(self.cameras):
            if self.aim_point is not None:
                camera_distance = np.linalg.norm(self.aim_point - np.array(camera.get_world_pose()[0]))
                configure_camera_intrinsics(camera, intrinsics_cfg, camera_distance)


def configure_camera_intrinsics(camera, intrinsics_config, camera_distance):
    """Configure camera intrinsics."""
    if not intrinsics_config:
        return
    
    width, height = camera.get_resolution()
    pixel_size = float(intrinsics_config.get("pixel_size", 3.0))
    f_stop = float(intrinsics_config.get("f_stop", 1.8))
    focus_distance = float(intrinsics_config.get("focus_distance", camera_distance))
    camera_matrix = intrinsics_config.get("camera_matrix", None)
    distortion_model = intrinsics_config.get("distortion_model", "pinhole")
    distortion_coeffs = intrinsics_config.get("distortion_coefficients", None)
    focal_length_override = intrinsics_config.get("focal_length", None)
    horizontal_aperture_override = intrinsics_config.get("horizontal_aperture", None)
    vertical_aperture_override = intrinsics_config.get("vertical_aperture", None)

    # Calculate apertures and focal length from camera matrix or use defaults
    if camera_matrix is not None:
        ((fx, _, cx), (_, fy, cy), (_, _, _)) = camera_matrix

        horizontal_aperture = pixel_size * width * 1e-6
        vertical_aperture = pixel_size * height * 1e-6
        
        focal_length_x = pixel_size * fx * 1e-6
        focal_length_y = pixel_size * fy * 1e-6
        focal_length = (focal_length_x + focal_length_y) / 2.0
        
    else:
        horizontal_aperture = horizontal_aperture_override or (pixel_size * width * 1e-6)
        vertical_aperture = vertical_aperture_override or (pixel_size * height * 1e-6)
        focal_length = focal_length_override or 0.024
        
        fx = focal_length / (pixel_size * 1e-6)
        fy = fx
        cx = width / 2.0
        cy = height / 2.0

    # Set basic camera parameters
    camera.set_focal_length(focal_length)
    camera.set_focus_distance(focus_distance)
    camera.set_lens_aperture(f_stop)
    camera.set_horizontal_aperture(horizontal_aperture)
    camera.set_vertical_aperture(vertical_aperture)

    # Configure distortion model if specified
    if distortion_coeffs is not None and camera_matrix is not None:
        ((fx, _, cx), (_, fy, cy), (_, _, _)) = camera_matrix
        
        if distortion_model == "pinhole":
            camera.set_opencv_pinhole_properties(cx=cx, cy=cy, fx=fx, fy=fy, pinhole=distortion_coeffs)
        elif distortion_model == "fisheye":
            camera.set_opencv_fisheye_properties(cx=cx, cy=cy, fx=fx, fy=fy, fisheye=distortion_coeffs)
        elif distortion_model == "kannala_brandt":
            nominal_width = float(width)
            nominal_height = float(height)
            optical_centre_x = cx
            optical_centre_y = cy
            camera.set_kannala_brandt_properties(
                nominal_width=nominal_width,
                nominal_height=nominal_height,
                optical_centre_x=optical_centre_x,
                optical_centre_y=optical_centre_y,
                max_fov=None,
                distortion_model=distortion_coeffs
            )
        elif distortion_model == "rational_polynomial":
            nominal_width = float(width)
            nominal_height = float(height)
            optical_centre_x = cx
            optical_centre_y = cy
            camera.set_rational_polynomial_properties(
                nominal_width=nominal_width,
                nominal_height=nominal_height,
                optical_centre_x=optical_centre_x,
                optical_centre_y=optical_centre_y,
                max_fov=None,
                distortion_model=distortion_coeffs
            )


def initialize_cameras(cameras, scene_params=None, intrinsics_config=None, aim_point=None):
    """Initialize cameras without running simulation."""
    if not cameras:
        print("[CAMS] No cameras to initialize")
        return
    
    stage_units = scene_params['stage_units'] if scene_params else 1.0
    try:
        world = World.instance()
        if world is None:
            world = World(stage_units_in_meters=stage_units)
    except:
        world = World(stage_units_in_meters=stage_units)
    
    world.reset()
    
    for camera in cameras:
        camera.initialize()
        pos, rot = camera.get_world_pose()
        print(f"[CAMS] {camera.name} initialized at position: {pos}")


if __name__ == "__main__":
    print("PositionCameras - Isaac Sim Camera Setup")
    print("=========================================")

    parser = argparse.ArgumentParser(description="Create and position cameras in Isaac Sim")
    parser.add_argument(
        "--config",
        "-c",
        default="config.yaml",
        help="Path to the configuration YAML file (default: config.yaml)",
    )
    args = parser.parse_args()

    resolved_cfg_path = _resolve_existing_path(args.config)
    if resolved_cfg_path is None:
        raise FileNotFoundError(f"Configuration file not found: {Path(args.config)}")

    cfg_path = resolved_cfg_path

    with cfg_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}

    if not isinstance(cfg, dict):
        raise ValueError(f"Configuration must be a mapping, got: {type(cfg).__name__}")

    steps_cfg = cfg.get("execution", {}).get("steps", {})
    if steps_cfg and not steps_cfg.get("position_cameras", True):
        print("[CAMS] position_cameras step disabled in configuration; nothing to do.")
    else:
        positioner = PositionCameras(cfg)
        cameras, aim_point, scene_params = positioner.run()
        print(f"Created {len(cameras)} cameras")
