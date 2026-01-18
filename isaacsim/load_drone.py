"""
LoadDrone: Isaac Sim Object Loader
Loads a USD/FBX asset (drone, bird, etc.) into the scene and applies semantic labels for instance segmentation.
Automatically detects asset type and applies appropriate semantic labels.
"""

import omni.usd
from isaacsim.core.utils.semantics import (  # type: ignore
    add_labels,
    upgrade_prim_semantics_to_labels,
)
from pxr import Usd, UsdGeom, Gf


# Asset type constants
TYPE_DRONE = "drone"
TYPE_BIRD = "bird"
TYPE_UNKNOWN = "unknown"

# Keywords used to detect asset types from file paths
BIRD_KEYWORDS = ["bird", "eagle", "hawk", "crow", "raven", "sparrow", "pigeon", "owl", "falcon"]
DRONE_KEYWORDS = ["drone", "quadcopter", "uav", "copter", "multirotor"]


def _set_semantic_label(prim, label):
    """Apply a semantic class label to a prim via the Labels API."""
    if prim is None or not prim.IsValid():
        return False

    try:
        add_labels(prim, labels=[label], instance_name="class")
        return True
    except Exception:
        return False


def _detect_asset_type(usd_path):
    """Detect the asset type based on file path.

    Args:
        usd_path: Path to the USD/FBX file

    Returns:
        Asset type constant (TYPE_DRONE, TYPE_BIRD, etc.)
    """
    usd_path_lower = usd_path.lower()

    for keyword in BIRD_KEYWORDS:
        if keyword in usd_path_lower:
            return TYPE_BIRD

    for keyword in DRONE_KEYWORDS:
        if keyword in usd_path_lower:
            return TYPE_DRONE

    return TYPE_UNKNOWN


class LoadDrone:
    """Load a USD/FBX asset (drone, bird, etc.) into the Isaac Sim stage."""

    def __init__(self, cfg):
        """Initialize with configuration dict."""
        self.cfg = cfg
        self.drone_cfg = cfg.get("drone", {})
        self.verbose = cfg.get("execution", {}).get("verbose", False)
        self.last_semantics_applied = False
        self.asset_type = None
    
    def run(self):
        """Load the asset (drone, bird, etc.) into the stage."""
        usd_path = self.drone_cfg.get("usd_path")
        if not usd_path:
            print("[LOAD_OBJECT] ERROR: No usd_path specified in config")
            return None

        prim_path = self.drone_cfg.get("prim_path", "/World/Drone")

        # Detect asset type
        self.asset_type = _detect_asset_type(usd_path)

        if self.verbose:
            print(f"[LOAD_OBJECT] Loading asset from: {usd_path}")
            print(f"[LOAD_OBJECT] Target prim path: {prim_path}")
            print(f"[LOAD_OBJECT] Detected asset type: {self.asset_type}")

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            print("[LOAD_OBJECT] ERROR: No active stage found")
            return None

        # Create the prim and add USD/FBX reference
        obj_prim = stage.DefinePrim(prim_path, "Xform")
        obj_prim.GetReferences().AddReference(usd_path)

        # Apply config-defined transforms (translation, orientation, scale)
        self._apply_config_transforms(obj_prim)
        
        # Apply unit resolution transforms for specific asset types (legacy method)
        # self._apply_units_resolve(obj_prim)

        if self.verbose:
            print(f"[LOAD_OBJECT] ✓ Asset loaded at: {prim_path}")

        try:
            self.last_semantics_applied = bool(self.apply_semantic_labels(prim_path))
            if self.verbose:
                if self.last_semantics_applied:
                    print(f"[LOAD_OBJECT] ✓ Applied semantic labels at {prim_path}")
                else:
                    print(f"[LOAD_OBJECT] WARNING: Failed to apply semantic labels at {prim_path}")
        except Exception as exc:
            self.last_semantics_applied = False
            print(f"[LOAD_OBJECT] WARNING: Exception while applying semantics: {exc}")

        return obj_prim

    def _apply_config_transforms(self, prim):
        """Apply translation, orientation, and scale from config."""
        xformable = UsdGeom.Xformable(prim)
        if not xformable:
            return

        # Translation
        translation = self.drone_cfg.get("translation")
        if translation:
            try:
                # Add or update translation op
                # Check for existing op or add new one
                xform_ops = xformable.GetOrderedXformOps()
                translate_op = None
                for op in xform_ops:
                    if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                        translate_op = op
                        break
                
                if not translate_op:
                    translate_op = xformable.AddTranslateOp()
                
                translate_op.Set(Gf.Vec3d(translation))
                if self.verbose:
                    print(f"[LOAD_OBJECT] Applied translation: {translation}")
            except Exception as e:
                print(f"[LOAD_OBJECT] WARNING: Failed to apply translation: {e}")

        # Orientation (Euler degrees)
        orientation = self.drone_cfg.get("orientation")
        if orientation:
            try:
                # Use RotateXYZ op for Euler angles
                # Check for existing op
                xform_ops = xformable.GetOrderedXformOps()
                rotate_op = None
                for op in xform_ops:
                    if op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ:
                        rotate_op = op
                        break
                
                if not rotate_op:
                    rotate_op = xformable.AddRotateXYZOp()
                
                rotate_op.Set(Gf.Vec3d(orientation))
                if self.verbose:
                    print(f"[LOAD_OBJECT] Applied orientation: {orientation}")
            except Exception as e:
                print(f"[LOAD_OBJECT] WARNING: Failed to apply orientation: {e}")

        # Scale
        scale = self.drone_cfg.get("scale")
        if scale:
            try:
                xform_ops = xformable.GetOrderedXformOps()
                scale_op = None
                for op in xform_ops:
                    if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                        scale_op = op
                        break
                
                if not scale_op:
                    scale_op = xformable.AddScaleOp()
                
                scale_op.Set(Gf.Vec3f(scale))
                if self.verbose:
                    print(f"[LOAD_OBJECT] Applied scale: {scale}")
            except Exception as e:
                print(f"[LOAD_OBJECT] WARNING: Failed to apply scale: {e}")

    def _apply_units_resolve(self, prim):
        """Apply unit resolution transforms for specific asset types.
        
        DEPRECATED: Prefer using explicit transforms in asset_config.yaml
        """
        # ... legacy implementation kept if needed ...
        pass
        
        # Original implementation below for reference if revert needed
        """
        if self.asset_type != TYPE_BIRD:
            # Only apply for birds - drones and other USD assets don't need this
            return
        
        xformable = UsdGeom.Xformable(prim)
        # ... (rest of original method)
        """

    def apply_semantic_labels(self, prim_path):
        """Apply semantic labels based on asset type for instance segmentation."""
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            print("[LOAD_OBJECT] ERROR: No active stage found")
            return False

        label_success = False

        root_prim = stage.GetPrimAtPath(prim_path)
        if not root_prim or not root_prim.IsValid():
            print(f"[LOAD_OBJECT] WARNING: Prim not found at {prim_path}")
            return False

        try:
            upgrade_prim_semantics_to_labels(root_prim, include_descendants=True)
        except Exception as exc:
            if self.verbose:
                print(f"[LOAD_OBJECT] WARNING: Failed to upgrade semantics at {prim_path}: {exc}")

        labeled_prims = 0
        verbose_print_limit = 20

        # Determine base label from asset type
        base_label = self.asset_type if self.asset_type else TYPE_UNKNOWN

        if _set_semantic_label(root_prim, base_label):
            label_success = True
            labeled_prims += 1
            if self.verbose and labeled_prims <= verbose_print_limit:
                print(f"[LOAD_OBJECT] ✓ Labeled {prim_path} as '{base_label}'")

        # Traverse descendants and tag any mesh-like prims with appropriate semantics
        for prim in Usd.PrimRange(root_prim):
            if prim == root_prim:
                continue
            if not prim.IsValid():
                continue

            if not prim.IsA(UsdGeom.Imageable):
                continue

            path_str = str(prim.GetPath()).lower()
            label = self._determine_part_label(path_str, base_label)

            if _set_semantic_label(prim, label):
                label_success = True
                labeled_prims += 1
                if self.verbose and labeled_prims <= verbose_print_limit:
                    print(f"[LOAD_OBJECT] ✓ Labeled {prim.GetPath()} as '{label}'")

        if self.verbose:
            if label_success:
                print(f"[LOAD_OBJECT] ✓ Applied semantics to {labeled_prims} prims under {prim_path}")
            else:
                print(f"[LOAD_OBJECT] WARNING: No semantics applied under {prim_path}")

        return label_success

    def _determine_part_label(self, path_str, base_label):
        """Determine semantic label for a prim based on its path and asset type.

        Args:
            path_str: Lowercase path string of the prim
            base_label: Base label for the asset type

        Returns:
            Appropriate semantic label string
        """
        if base_label == TYPE_DRONE:
            # Drone-specific part labels
            if "prop" in path_str:
                return "drone_prop"
            elif "body" in path_str or "hull" in path_str:
                return "drone_body"
            else:
                return "drone"
        elif base_label == TYPE_BIRD:
            # Bird-specific part labels
            if "wing" in path_str:
                return "bird_wing"
            elif "head" in path_str or "beak" in path_str:
                return "bird_head"
            elif "tail" in path_str:
                return "bird_tail"
            elif "body" in path_str or "torso" in path_str:
                return "bird_body"
            elif "leg" in path_str or "foot" in path_str or "claw" in path_str:
                return "bird_leg"
            else:
                return "bird"
        else:
            # Unknown asset type - use generic label
            return base_label

    def print_class_ids(self, prim_path):
        """Print semantic class labels and instance IDs for verification."""
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            print("[LOAD_OBJECT] ERROR: No active stage found")
            return

        print(f"\n[LOAD_OBJECT] Instance Segmentation Class Mapping for {self.asset_type}:")
        print("-" * 60)

        root_prim = stage.GetPrimAtPath(prim_path)
        if not root_prim.IsValid():
            print(f"[LOAD_OBJECT] ERROR: Prim not found at {prim_path}")
            return

        # Print labels for all descendants with Imageable type
        printed_count = 0
        max_print = 30

        for prim in Usd.PrimRange(root_prim):
            if not prim.IsValid():
                continue
            if not prim.IsA(UsdGeom.Imageable):
                continue

            label_attr = prim.GetAttribute("labels:class")
            label = "none"
            if label_attr and label_attr.IsValid():
                value = label_attr.Get()
                if isinstance(value, (list, tuple)):
                    label = value[0] if value else "none"
                elif value:
                    label = value

            instance_id = prim.GetInstanceId() if hasattr(prim, 'GetInstanceId') else "N/A"
            print(f"Prim: {prim.GetPath()}, Class: {label}, Instance ID: {instance_id}")

            printed_count += 1
            if printed_count >= max_print:
                print(f"... (truncated, {printed_count}+ prims)")
                break

        print("-" * 60)



if __name__ == "__main__":
    example_cfg = {
        "drone": {
            "usd_path": "/home/sandro/thesis/assets/drones/ofm-seeker-drone_c8fed575-ee29-4297-91f5-9a0f97baa042/ofm-seeker-drone_2K_c55c3fe7-e3f2-4dcd-9fff-175121f13263.usdc",
            "prim_path": "/World/Drone"
        },
        "execution": {"verbose": True}
    }
    
    loader = LoadDrone(example_cfg)
    drone_prim = loader.run()
    
    if drone_prim:
        # Apply semantic labels for instance segmentation
        loader.apply_semantic_labels(example_cfg["drone"]["prim_path"])
        
        # Verify labels and print class-to-ID mapping
        loader.print_class_ids(example_cfg["drone"]["prim_path"])