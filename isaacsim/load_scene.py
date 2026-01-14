"""
LoadScene: Isaac Sim USD Scene Loader
Loads a USD scene and configures lighting.
"""

import omni.usd
import omni.timeline
import carb.settings
from pxr import UsdGeom, Sdf

from config_utils import resolve_fps_from_config


class LoadScene:
    """Load and configure a USD scene in Isaac Sim."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.scene_cfg = cfg.get("scene", {})
        self.lighting_cfg = self.scene_cfg.get("lighting", {})
        self.verbose = cfg.get("execution", {}).get("verbose", False)
        self.stage_fps = resolve_fps_from_config(cfg)
 
    def run(self):
        """Load the configured scene unless the same scene is already active."""
        scene_path = self.scene_cfg.get("path")
        if not scene_path:
            print("[LOAD] ERROR: No scene path specified in config")
            return None

        scene_path = str(scene_path)
        scene_prim_path = self.scene_cfg.get("prim_path", "/World/CityScene")

        if self.verbose:
            print("[LOAD] Starting scene loading")
            print(f"[LOAD] Scene path: {scene_path}")
            print(f"[LOAD] Prim path: {scene_prim_path}")

        usd_context = omni.usd.get_context()
        stage = usd_context.get_stage()

        if self._is_scene_already_loaded(stage, scene_prim_path, scene_path):
            if self.verbose:
                print(f"[LOAD] Scene already loaded at {scene_prim_path}, skipping reload")
            self._set_stage_fps(stage)
            self._setup_lighting(stage)
            return stage.GetPrimAtPath(scene_prim_path)

        usd_context.new_stage()
        stage = usd_context.get_stage()
        if stage is None:
            print("[LOAD] ERROR: Failed to create new stage")
            return None

        print(f"[LOAD] Loading scene from: {scene_path}")

        scene_prim = stage.DefinePrim(scene_prim_path, "Xform")
        scene_prim.GetReferences().ClearReferences()
        scene_prim.GetReferences().AddReference(scene_path)
        self._mark_scene_loaded(scene_prim, scene_path)

        self._set_stage_fps(stage)
        self._setup_lighting(stage)

        print("[LOAD] Scene loaded successfully!")
        print(f"[LOAD] Scene loaded at prim path: {scene_prim_path}")
        return scene_prim

    def _is_scene_already_loaded(self, stage, prim_path, scene_path):
        if stage is None:
            return False

        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            return False

        source_path = prim.GetCustomData().get("source_scene_path")
        return source_path == str(scene_path)

    def _mark_scene_loaded(self, prim, scene_path):
        if prim is None:
            return
        try:
            prim.SetCustomDataByKey("source_scene_path", str(scene_path))
        except Exception:
            pass

    def _set_stage_fps(self, stage):
        if stage is None:
            return
        try:
            stage.SetTimeCodesPerSecond(self.stage_fps)
            stage.SetFramesPerSecond(self.stage_fps)

            timeline = omni.timeline.get_timeline_interface()
            if timeline:
                timeline.set_time_codes_per_second(self.stage_fps)
                timeline.set_ticks_per_frame(1)

            kit_settings = carb.settings.get_settings()
            if kit_settings:
                kit_settings.set("/app/player/useFixedTimeStepping", True)
                kit_settings.set("/app/stage/timeCodesPerSecond", self.stage_fps)

            if self.verbose:
                print(f"[LOAD] Set stage and timeline FPS to {self.stage_fps}")
        except Exception as exc:
            print(f"[LOAD] Warning: Failed to set FPS: {exc}")

    def _setup_lighting(self, stage):
        if stage is None:
            return

        lighting_cfg = self.lighting_cfg
        if not lighting_cfg:
            if self.verbose:
                print("[LOAD] No lighting configuration provided, skipping lighting setup")
            return

        dome_cfg = lighting_cfg.get("dome_light", {})
        if dome_cfg.get("enabled", False):
            dome_light_path = dome_cfg.get("prim_path", "/World/DomeLight")
            if not stage.GetPrimAtPath(dome_light_path):
                dome_light = stage.DefinePrim(dome_light_path, "DomeLight")
                dome_light.CreateAttribute("inputs:intensity", Sdf.ValueTypeNames.Float).Set(
                    float(dome_cfg.get("intensity", 1000.0))
                )
                dome_light.CreateAttribute("inputs:exposure", Sdf.ValueTypeNames.Float).Set(
                    float(dome_cfg.get("exposure", 0.0))
                )
                if self.verbose:
                    print(f"[LOAD] Created dome light at: {dome_light_path}")

        distant_cfg = lighting_cfg.get("distant_light", {})
        if distant_cfg.get("enabled", False):
            distant_light_path = distant_cfg.get("prim_path", "/World/DistantLight")
            if not stage.GetPrimAtPath(distant_light_path):
                distant_light = stage.DefinePrim(distant_light_path, "DistantLight")
                distant_light.CreateAttribute("inputs:intensity", Sdf.ValueTypeNames.Float).Set(
                    float(distant_cfg.get("intensity", 3000.0))
                )
                distant_light.CreateAttribute("inputs:angle", Sdf.ValueTypeNames.Float).Set(
                    float(distant_cfg.get("angle", 1.0))
                )

                rotation = distant_cfg.get("rotation", [-45.0, 30.0, 0.0])
                UsdGeom.Xformable(distant_light).AddRotateXYZOp().Set(tuple(rotation))

                if self.verbose:
                    print(f"[LOAD] Created distant light at: {distant_light_path}")


if __name__ == "__main__":
    example_cfg = {
        "scene": {
            "path": "/home/sandro/thesis/assets/AECO_CityTowerDemoPack_NVD@10011/Demos/AEC/TowerDemo/CityTowerDemopack/World_CityTowerDemopack.usd",
            "prim_path": "/World/CityScene",
        },
        "execution": {"verbose": True},
    }

    loader = LoadScene(example_cfg)
    loader.run()
