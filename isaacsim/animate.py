"""
Animate: Isaac Sim Object Animation
Unified animation module that handles different asset types (drone, bird, etc.).
Automatically detects asset type and applies appropriate animations.
"""

import omni.usd
import omni.timeline
import carb.settings
from pxr import UsdSkel, Sdf, Usd
from config_utils import resolve_fps_from_config
from flight_path import FlightPath, resolve_root_under_world, ensure_mover_wrapper


class Animate:
    """Animate objects along flight paths with optional skeletal animation."""

    # Asset type constants
    TYPE_DRONE = "drone"
    TYPE_BIRD = "bird"
    TYPE_UNKNOWN = "unknown"

    # Keywords used to detect asset types from file paths
    BIRD_KEYWORDS = ["bird", "eagle", "hawk", "crow", "raven", "sparrow", "pigeon", "owl", "falcon"]
    DRONE_KEYWORDS = ["drone", "quadcopter", "uav", "copter", "multirotor"]

    def __init__(self, cfg):
        """Initialize with configuration dict."""
        self.cfg = cfg
        self.drone_cfg = cfg.get("drone", {})
        self.verbose = cfg.get("execution", {}).get("verbose", False)
        self.stage_fps = resolve_fps_from_config(cfg)
        self.asset_type = None
        self.skel_animation_range = None

    def run(self):
        """Set up and animate the object based on its type."""
        if self.verbose:
            print("[ANIMATE] Starting animation setup...")

        # Extract configuration
        prim_path = self.drone_cfg.get("prim_path", "/World/Drone")
        if not prim_path:
            print("[ANIMATE] ERROR: No prim_path specified in config")
            return None

        usd_path = self.drone_cfg.get("usd_path", "")
        wrap_with_mover = bool(self.drone_cfg.get("wrap_with_mover", True))
        mover_prim_path = self.drone_cfg.get("mover_prim_path", "/World/DroneMover")

        # Get current stage
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            print("[ANIMATE] Error: No active stage found")
            return None

        # Set stage and timeline FPS
        self._set_stage_fps(stage)

        # Get existing prim
        obj_prim = stage.GetPrimAtPath(prim_path)
        if not obj_prim.IsValid():
            print(f"[ANIMATE] Error: No prim found at {prim_path}")
            return None

        # Detect asset type
        self.asset_type = self._detect_asset_type(stage, obj_prim, usd_path)
        if self.verbose:
            print(f"[ANIMATE] Detected asset type: {self.asset_type}")

        # Resolve the top-level root (direct child of /World)
        obj_root = resolve_root_under_world(obj_prim)

        # Optionally wrap with mover parent for flight path animation
        prim_to_animate = obj_root
        if wrap_with_mover:
            if obj_root.IsInstanceable() or obj_root.IsInstanceProxy():
                if self.verbose:
                    print("[ANIMATE] Root prim is instanceable/proxy; skipping wrapper and animating root directly.")
            else:
                prim_to_animate = ensure_mover_wrapper(stage, obj_root, mover_prim_path, self.verbose)

        # Apply animations based on asset type
        success = False
        if self.asset_type == self.TYPE_BIRD:
            success = self._animate_bird(stage, prim_to_animate)
        else:
            # Default to drone-style animation (flight path only)
            success = self._animate_drone(prim_to_animate)

        if success and self.verbose:
            print("[ANIMATE] Animation setup complete")

        return obj_prim

    def _detect_asset_type(self, stage, prim, usd_path):
        """Detect the asset type based on file path and prim content.

        Args:
            stage: USD stage
            prim: The loaded prim
            usd_path: Path to the USD/FBX file

        Returns:
            Asset type constant (TYPE_DRONE, TYPE_BIRD, etc.)
        """
        usd_path_lower = usd_path.lower()

        # Check file path for keywords
        for keyword in self.BIRD_KEYWORDS:
            if keyword in usd_path_lower:
                return self.TYPE_BIRD

        for keyword in self.DRONE_KEYWORDS:
            if keyword in usd_path_lower:
                return self.TYPE_DRONE

        # Check if the prim has skeletal animation (indicative of animated creatures)
        has_skel_animation = self._find_skel_animation(stage, prim)
        if has_skel_animation:
            return self.TYPE_BIRD

        # Default to drone
        return self.TYPE_DRONE

    def _find_skel_animation(self, stage, prim):
        """Check if prim or its descendants have SkelAnimation.

        Args:
            stage: USD stage
            prim: Prim to search from

        Returns:
            True if SkelAnimation found, False otherwise
        """
        # Search entire stage for SkelAnimation prims
        for p in stage.Traverse():
            if p.IsA(UsdSkel.Animation):
                return True
        return False

    def _get_skel_animation_range(self, stage):
        """Find the animation time range from SkelAnimation prims.

        Args:
            stage: USD stage

        Returns:
            Tuple of (min_time, max_time) or (None, None) if not found
        """
        min_time = float('inf')
        max_time = float('-inf')
        found_animation = False

        for prim in stage.Traverse():
            if prim.IsA(UsdSkel.Animation):
                if self.verbose:
                    print(f"[ANIMATE] Found SkelAnimation: {prim.GetPath()}")

                # Get attributes to find time samples
                for attr_name in ['translations', 'rotations', 'scales', 'blendShapeWeights']:
                    attr = prim.GetAttribute(attr_name)
                    if attr and attr.HasValue():
                        time_samples = attr.GetTimeSamples()
                        if time_samples:
                            found_animation = True
                            min_time = min(min_time, min(time_samples))
                            max_time = max(max_time, max(time_samples))
                            if self.verbose:
                                print(f"[ANIMATE]   - {attr_name}: {len(time_samples)} keyframes, "
                                      f"range [{min(time_samples):.1f} - {max(time_samples):.1f}]")

        if found_animation:
            return min_time, max_time
        return None, None

    def _animate_bird(self, stage, prim_to_animate):
        """Animate a bird with skeletal animation and flight path.

        The skeletal animation (wing flapping) loops independently while the
        flight path progresses linearly. This is achieved by:
        1. Using the config timeline for the flight path (no looping)
        2. Setting up value clips to make the skeletal animation loop

        Args:
            stage: USD stage
            prim_to_animate: Prim to apply flight path to

        Returns:
            True if successful, False otherwise
        """
        if self.verbose:
            print("[ANIMATE] Setting up bird animation...")

        # Get skeletal animation range (for the wing flapping cycle)
        skel_anim_start, skel_anim_end = self._get_skel_animation_range(stage)

        if skel_anim_start is not None and skel_anim_end is not None:
            self.skel_animation_range = (skel_anim_start, skel_anim_end)
            if self.verbose:
                print(f"[ANIMATE] Detected skeletal animation range: {skel_anim_start} - {skel_anim_end} frames")
        else:
            self.skel_animation_range = None
            if self.verbose:
                print("[ANIMATE] No skeletal animation found")

        # Get config timeline settings for the flight path
        timeline_cfg = self.drone_cfg.get("timeline", {})
        start_frame = float(timeline_cfg.get("start_frame", 1.0))
        middle_frame = float(timeline_cfg.get("middle_frame", 60.0))
        end_frame = float(timeline_cfg.get("end_frame", 120.0))

        # Configure timeline for the full flight duration (NOT looping)
        self._setup_timeline_for_flight(start_frame, end_frame)

        # Set up looping for skeletal animation if present
        if self.skel_animation_range:
            self._setup_skel_animation_loop(stage, skel_anim_start, skel_anim_end, start_frame, end_frame)

        # Apply flight path animation
        flight_path_helper = FlightPath(self.cfg, verbose=self.verbose)
        flight_path = flight_path_helper.calculate_path()

        success = flight_path_helper.animate_prim(
            prim_to_animate,
            flight_path=flight_path,
            start_frame=start_frame,
            middle_frame=middle_frame,
            end_frame=end_frame
        )

        if success and self.verbose:
            print("[ANIMATE] Bird flight path animation applied")

        return success

    def _setup_timeline_for_flight(self, start_frame, end_frame):
        """Configure timeline for the full flight duration without looping.

        Args:
            start_frame: Start frame of flight
            end_frame: End frame of flight
        """
        try:
            timeline = omni.timeline.get_timeline_interface()
            if not timeline:
                print("[ANIMATE] Warning: Could not get timeline interface")
                return

            # Convert frames to seconds
            start_seconds = start_frame / self.stage_fps
            end_seconds = end_frame / self.stage_fps

            # Configure timeline for full flight duration
            timeline.set_start_time(start_seconds)
            timeline.set_end_time(end_seconds)
            timeline.set_current_time(start_seconds)

            # Do NOT loop - flight path should progress linearly
            timeline.set_looping(False)

            if self.verbose:
                duration = end_seconds - start_seconds
                print(f"[ANIMATE] Timeline configured for flight path:")
                print(f"[ANIMATE]   - Range: {start_frame} - {end_frame} frames")
                print(f"[ANIMATE]   - Duration: {duration:.2f}s")
                print(f"[ANIMATE]   - Looping: Disabled (flight path is linear)")

        except Exception as e:
            print(f"[ANIMATE] Warning: Failed to configure timeline: {e}")

    def _setup_skel_animation_loop(self, stage, skel_start, skel_end, flight_start, flight_end):
        """Set up the skeletal animation to loop independently of the flight path.

        This creates looping by re-timing the skeletal animation keyframes to repeat
        throughout the flight duration.

        Args:
            stage: USD stage
            skel_start: Start frame of skeletal animation cycle
            skel_end: End frame of skeletal animation cycle
            flight_start: Start frame of flight
            flight_end: End frame of flight
        """
        try:
            skel_duration = skel_end - skel_start
            if skel_duration <= 0:
                return

            flight_duration = flight_end - flight_start
            num_loops = int(flight_duration / skel_duration) + 1

            if self.verbose:
                print(f"[ANIMATE] Setting up skeletal animation loop:")
                print(f"[ANIMATE]   - Skeletal cycle: {skel_duration} frames")
                print(f"[ANIMATE]   - Flight duration: {flight_duration} frames")
                print(f"[ANIMATE]   - Loops needed: {num_loops}")

            # Find all SkelAnimation prims and extend their keyframes
            for prim in stage.Traverse():
                if prim.IsA(UsdSkel.Animation):
                    self._extend_skel_animation_keyframes(
                        prim, skel_start, skel_end, flight_start, flight_end, num_loops
                    )

        except Exception as e:
            print(f"[ANIMATE] Warning: Failed to set up skeletal animation loop: {e}")

    def _extend_skel_animation_keyframes(self, skel_anim_prim, skel_start, skel_end,
                                          flight_start, flight_end, num_loops):
        """Extend skeletal animation keyframes to loop throughout the flight.

        Args:
            skel_anim_prim: The SkelAnimation prim
            skel_start: Original animation start frame
            skel_end: Original animation end frame
            flight_start: Flight start frame
            flight_end: Flight end frame
            num_loops: Number of times to repeat the animation
        """
        skel_duration = skel_end - skel_start

        # Process each animatable attribute
        for attr_name in ['translations', 'rotations', 'scales', 'blendShapeWeights']:
            attr = skel_anim_prim.GetAttribute(attr_name)
            if not attr or not attr.HasValue():
                continue

            time_samples = attr.GetTimeSamples()
            if not time_samples:
                continue

            # Get original keyframe values
            original_keyframes = {}
            for t in time_samples:
                if skel_start <= t <= skel_end:
                    original_keyframes[t - skel_start] = attr.Get(t)

            if not original_keyframes:
                continue

            # Clear existing keyframes and create looped ones
            # Note: We offset to start at flight_start instead of skel_start
            offset = flight_start - skel_start

            for loop_idx in range(num_loops):
                loop_offset = loop_idx * skel_duration + offset
                for rel_time, value in original_keyframes.items():
                    new_time = rel_time + skel_start + loop_offset
                    if new_time <= flight_end:
                        attr.Set(value, new_time)

        if self.verbose:
            print(f"[ANIMATE]   Extended keyframes for {skel_anim_prim.GetPath()}")

    def _animate_drone(self, prim_to_animate):
        """Animate a drone with flight path only.

        Args:
            prim_to_animate: Prim to apply flight path to

        Returns:
            True if successful, False otherwise
        """
        if self.verbose:
            print("[ANIMATE] Setting up drone animation...")

        flight_path_helper = FlightPath(self.cfg, verbose=self.verbose)
        flight_path = flight_path_helper.calculate_path()

        if self.verbose:
            print(f"[ANIMATE] Flight path: {flight_path}")

        return flight_path_helper.animate_prim(prim_to_animate, flight_path=flight_path)

    def _set_stage_fps(self, stage):
        """Set stage and timeline FPS to match config."""
        try:
            # Set stage FPS
            stage.SetTimeCodesPerSecond(self.stage_fps)
            stage.SetFramesPerSecond(self.stage_fps)

            # Set timeline FPS
            timeline = omni.timeline.get_timeline_interface()
            if timeline:
                timeline.set_time_codes_per_second(self.stage_fps)
                timeline.set_ticks_per_frame(1)

            # Set kit settings
            kit_settings = carb.settings.get_settings()
            if kit_settings:
                kit_settings.set("/app/player/useFixedTimeStepping", True)
                kit_settings.set("/app/stage/timeCodesPerSecond", self.stage_fps)

            if self.verbose:
                print(f"[ANIMATE] Set stage and timeline FPS to {self.stage_fps}")
        except Exception as e:
            print(f"[ANIMATE] Warning: Failed to set FPS: {e}")


# Legacy class name for backwards compatibility
AnimateDrone = Animate


def setup_object_animation(config: dict | None = None):
    """Helper function to animate an object based on config.

    Args:
        config: Configuration dict

    Returns:
        The animated prim or None on failure
    """
    animator = Animate(config or {})
    return animator.run()


# Legacy function name for backwards compatibility
def setup_drone_flight(config: dict | None = None):
    """Legacy function - wraps Animate class for backward compatibility."""
    return setup_object_animation(config)


if __name__ == "__main__":
    # Example configuration for testing
    example_cfg = {
        "drone": {
            "prim_path": "/World/Drone",
            "usd_path": "/home/sandro/thesis/assets/birds/white-eagle-animation-fast-fly/source/Eagle Fly/EAGLE FLY.fbx",
            "waypoint": [0.0, 0.0, 2.0],
            "flight_distance": 12.0,
            "flight_height_offset": 15.0,
            "flight_direction": [1.0, 0.0, 0.0],
            "wrap_with_mover": True,
            "mover_prim_path": "/World/DroneMover",
            "timeline": {
                "start_frame": 1.0,
                "middle_frame": 60.0,
                "end_frame": 120.0
            }
        },
        "fps": 30.0,
        "execution": {"verbose": True}
    }

    animator = Animate(example_cfg)
    animator.run()
