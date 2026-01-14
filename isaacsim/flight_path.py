"""
FlightPath: Isaac Sim Flight Path Animation
Handles flight path calculation and translation keyframe animation for any prim.
"""

import omni.usd
import omni.kit.commands
from pxr import UsdGeom, Gf, Sdf
from math import sqrt


class FlightPath:
    """Calculate and animate flight paths for prims."""

    def __init__(self, cfg, verbose=False):
        """Initialize with configuration dict.

        Args:
            cfg: Configuration dict containing 'drone' section with flight parameters
            verbose: Enable verbose logging
        """
        self.cfg = cfg
        self.drone_cfg = cfg.get("drone", {})
        self.verbose = verbose

    def calculate_path(self, waypoint=None, flight_distance=None,
                       flight_height_offset=None, flight_direction=None):
        """Calculate start and end points for flight path passing through waypoint.

        Args:
            waypoint: Center point of flight path (x, y, z). Uses config if None.
            flight_distance: Total distance of flight. Uses config if None.
            flight_height_offset: Height offset from waypoint. Uses config if None.
            flight_direction: Direction unit vector (dx, dy, dz). Uses config if None.

        Returns:
            dict with 'start', 'waypoint', 'end', and 'total_distance' keys
        """
        # Use config values as defaults
        if waypoint is None:
            waypoint = tuple(self.drone_cfg.get("waypoint", [0.0, 0.0, 0.0]))
        if flight_distance is None:
            flight_distance = float(self.drone_cfg.get("flight_distance", 10.0))
        if flight_height_offset is None:
            flight_height_offset = float(self.drone_cfg.get("flight_height_offset", 10.0))
        if flight_direction is None:
            flight_direction = tuple(self.drone_cfg.get("flight_direction", [1.0, 0.0, 0.0]))

        wx, wy, wz = waypoint
        wz += flight_height_offset

        # Normalize direction vector
        dx, dy, dz = flight_direction
        length = sqrt(dx*dx + dy*dy + dz*dz)
        if length > 0:
            dx, dy, dz = dx/length, dy/length, dz/length
        else:
            dx, dy, dz = 1.0, 0.0, 0.0

        # Calculate start and end points
        half_distance = flight_distance / 2.0

        start_point = (
            wx - dx * half_distance,
            wy - dy * half_distance,
            wz - dz * half_distance
        )

        end_point = (
            wx + dx * half_distance,
            wy + dy * half_distance,
            wz + dz * half_distance
        )

        return {
            "start": start_point,
            "waypoint": (wx, wy, wz),
            "end": end_point,
            "total_distance": flight_distance
        }

    def animate_prim(self, prim, flight_path=None, start_frame=None,
                     middle_frame=None, end_frame=None):
        """Apply translation keyframes to animate a prim along a flight path.

        Args:
            prim: USD prim to animate
            flight_path: Path dict from calculate_path(). Calculates if None.
            start_frame: Start frame for animation. Uses config if None.
            middle_frame: Middle frame for animation. Uses config if None.
            end_frame: End frame for animation. Uses config if None.

        Returns:
            True if animation was applied successfully, False otherwise
        """
        try:
            if not prim or not prim.IsValid():
                print("[FLIGHT_PATH] animate_prim received invalid prim; aborting.")
                return False

            # Calculate flight path if not provided
            if flight_path is None:
                flight_path = self.calculate_path()

            # Get timeline config
            timeline_cfg = self.drone_cfg.get("timeline", {})
            if start_frame is None:
                start_frame = float(timeline_cfg.get("start_frame", 1.0))
            if middle_frame is None:
                middle_frame = float(timeline_cfg.get("middle_frame", 60.0))
            if end_frame is None:
                end_frame = float(timeline_cfg.get("end_frame", 120.0))

            # Ensure prim is Xformable
            prim = self._ensure_xformable(prim)
            if prim is None:
                return False

            xformable = UsdGeom.Xformable(prim)
            if not xformable:
                print("[FLIGHT_PATH] Could not obtain Xformable for prim; abort.")
                return False

            # Get or create translate operation
            translate_op = self._get_or_create_translate_op(xformable)
            if translate_op is None:
                print("[FLIGHT_PATH] Failed to acquire/create translate op; abort.")
                return False

            # Apply keyframes (use frame numbers as time codes directly)
            translate_op.Set(Gf.Vec3d(*flight_path["start"]), start_frame)
            translate_op.Set(Gf.Vec3d(*flight_path["waypoint"]), middle_frame)
            translate_op.Set(Gf.Vec3d(*flight_path["end"]), end_frame)

            if self.verbose:
                print(f"[FLIGHT_PATH] Set keyframes: start={start_frame}, mid={middle_frame}, end={end_frame}")
                print(f"[FLIGHT_PATH] Path: {flight_path['start']} -> {flight_path['waypoint']} -> {flight_path['end']}")

            return True
        except Exception as e:
            print(f"[FLIGHT_PATH] Failed to create animation: {e}")
            return False

    def _ensure_xformable(self, prim):
        """Ensure the prim is Xformable, creating a wrapper if needed."""
        if UsdGeom.Xformable(prim):
            return prim

        if prim.GetTypeName() not in ("Xform", "Scope", "Mesh", "Cube", "Cylinder", "Sphere"):
            if self.verbose:
                print("[FLIGHT_PATH] Prim not naturally Xformable; creating temporary parent Xform.")

            stage = prim.GetStage()
            temp_parent_path = prim.GetPath().GetParentPath().AppendChild(
                prim.GetPath().name + "_AnimXform"
            )

            if not stage.GetPrimAtPath(temp_parent_path).IsValid():
                stage.DefinePrim(temp_parent_path, "Xform")
                try:
                    omni.kit.commands.execute(
                        "ParentPrims",
                        parent_path=str(temp_parent_path),
                        child_paths=[str(prim.GetPath())],
                        keep_world_transform=True,
                    )
                    return stage.GetPrimAtPath(temp_parent_path)
                except Exception as e:
                    print(f"[FLIGHT_PATH] Failed to create temp animation parent: {e}")
                    return None
            else:
                return stage.GetPrimAtPath(temp_parent_path)

        return prim

    def _get_or_create_translate_op(self, xformable):
        """Get existing or create new translate xform op."""
        translate_ops = xformable.GetOrderedXformOps()
        for op in translate_ops:
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                return op

        try:
            return xformable.AddTranslateOp()
        except Exception:
            return xformable.GetTranslateOp()


def resolve_root_under_world(prim):
    """Resolve the top-level prim root (direct child of /World).

    Args:
        prim: Any USD prim

    Returns:
        The ancestor prim that is a direct child of /World
    """
    p = prim
    while p:
        parent = p.GetParent()
        if not parent:
            return p
        if parent.GetPath() == Sdf.Path("/World") or parent.GetPath() == Sdf.Path.absoluteRootPath:
            return p
        p = parent
    return prim


def ensure_mover_wrapper(stage, prim_root, mover_path, verbose=False):
    """Ensure prim is wrapped with a mover parent prim.

    Args:
        stage: USD stage
        prim_root: Prim to wrap
        mover_path: Path for the mover wrapper prim
        verbose: Enable verbose logging

    Returns:
        The mover prim (or original prim if wrapping failed)
    """
    mover_prim = stage.GetPrimAtPath(mover_path)
    if not mover_prim.IsValid():
        mover_prim = stage.DefinePrim(mover_path, "Xform")

    # Already wrapped
    if prim_root.GetParent() == mover_prim:
        return mover_prim

    # Parent the prim root under the mover while keeping world transform
    try:
        omni.kit.commands.execute(
            "ParentPrims",
            parent_path=str(mover_prim.GetPath()),
            child_paths=[str(prim_root.GetPath())],
            keep_world_transform=True,
        )
        new_child_path = mover_prim.GetPath().AppendChild(prim_root.GetPath().name)
        new_child = stage.GetPrimAtPath(new_child_path)
        if not new_child.IsValid():
            if verbose:
                print("[FLIGHT_PATH] Warning: parenting reported success but child prim not found under mover; falling back to root animation")
            return prim_root
    except Exception as e:
        if verbose:
            print(f"[FLIGHT_PATH] Failed to wrap prim with mover: {e}. Falling back to root prim animation.")
        return prim_root

    return mover_prim
