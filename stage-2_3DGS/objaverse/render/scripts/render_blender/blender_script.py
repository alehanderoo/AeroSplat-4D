"""Blender script to render images of 3D models."""

import argparse
import json
import math
import os
import random
import sys
from typing import Any, Callable, Dict, Generator, List, Literal, Optional, Set, Tuple, Union

import bpy
import numpy as np
from mathutils import Matrix, Vector

GS_LRM_RESOLUTION: int = 256
CAMERA_RADIUS_RANGE: Tuple[float, float] = (1.5, 2.0)
ELEVATION_RANGE_DEGREES = (-45.0, 45.0)
FOV_DEGREES_RANGE: Tuple[float, float] = (45.0, 90.0)
CAMERA_RADIUS_MARGIN = 0.05
FOV_SAFETY_MARGIN_DEGREES = 2.0


IMPORT_FUNCTIONS: Dict[str, Callable] = {
    "obj": bpy.ops.import_scene.obj,
    "glb": bpy.ops.import_scene.gltf,
    "gltf": bpy.ops.import_scene.gltf,
    "usd": bpy.ops.import_scene.usd,
    "fbx": bpy.ops.import_scene.fbx,
    "stl": bpy.ops.import_mesh.stl,
    "usda": bpy.ops.import_scene.usda,
    "dae": bpy.ops.wm.collada_import,
    "ply": bpy.ops.import_mesh.ply,
    "abc": bpy.ops.wm.alembic_import,
    "blend": bpy.ops.wm.append,
}


def reset_cameras() -> None:
    """Resets the cameras in the scene to a single default camera."""
    # Delete all existing cameras
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_by_type(type="CAMERA")
    bpy.ops.object.delete()

    # Create a new camera with default properties
    bpy.ops.object.camera_add()

    # Rename the new camera to 'NewDefaultCamera'
    new_camera = bpy.context.active_object
    new_camera.name = "Camera"

    # Set the new camera as the active camera for the scene
    scene.camera = new_camera


def sample_point_on_sphere(radius: float) -> Tuple[float, float, float]:
    """Samples a point on a sphere with the given radius.

    Args:
        radius (float): Radius of the sphere.

    Returns:
        Tuple[float, float, float]: A point on the sphere.
    """
    theta = random.random() * 2 * math.pi
    phi = math.acos(2 * random.random() - 1)
    return (
        radius * math.sin(phi) * math.cos(theta),
        radius * math.sin(phi) * math.sin(theta),
        radius * math.cos(phi),
    )


def _sample_spherical(
    radius_min: float = 1.5,
    radius_max: float = 2.0,
    maxz: float = 1.6,
    minz: float = -0.75,
) -> np.ndarray:
    """Sample a random point in a spherical shell.

    Args:
        radius_min (float): Minimum radius of the spherical shell.
        radius_max (float): Maximum radius of the spherical shell.
        maxz (float): Maximum z value of the spherical shell.
        minz (float): Minimum z value of the spherical shell.

    Returns:
        np.ndarray: A random (x, y, z) point in the spherical shell.
    """
    correct = False
    vec = np.array([0, 0, 0])
    while not correct:
        vec = np.random.uniform(-1, 1, 3)
        #         vec[2] = np.abs(vec[2])
        radius = np.random.uniform(radius_min, radius_max, 1)
        vec = vec / np.linalg.norm(vec, axis=0) * radius[0]
        if maxz > vec[2] > minz:
            correct = True
    return vec


def randomize_camera(
    only_northern_hemisphere: bool = False,
    radius_range: Tuple[float, float] = CAMERA_RADIUS_RANGE,
    object_radius: float = 1.0,
    radius_margin: float = CAMERA_RADIUS_MARGIN,
) -> Tuple[bpy.types.Object, Dict[str, float]]:
    """Randomizes the camera location according to the GS-LRM sampling spec.

    Args:
        only_northern_hemisphere (bool, optional): Whether to restrict samples to the
            northern hemisphere. Defaults to False.
        radius_range (Tuple[float, float], optional): Allowed camera radius range.
        object_radius (float, optional): Bounding sphere radius of the object.
        radius_margin (float, optional): Safety padding added to the radius.

    Returns:
        Tuple[bpy.types.Object, Dict[str, float]]: The camera object and pose info.
    """

    radius_min, radius_max = radius_range
    if radius_max < radius_min:
        radius_max = radius_min
    radius = random.uniform(radius_min, radius_max)
    radius = max(radius, object_radius + radius_margin)
    azimuth = random.uniform(0.0, 2.0 * math.pi)
    elevation = math.radians(random.uniform(*ELEVATION_RANGE_DEGREES))

    if only_northern_hemisphere:
        elevation = abs(elevation)

    x = radius * math.cos(elevation) * math.cos(azimuth)
    y = radius * math.cos(elevation) * math.sin(azimuth)
    z = radius * math.sin(elevation)

    camera = bpy.data.objects["Camera"]

    camera.location = Vector((x, y, z))

    direction = -camera.location
    rot_quat = direction.to_track_quat("-Z", "Y")
    camera.rotation_euler = rot_quat.to_euler()

    pose_info = {
        "radius": radius,
        "azimuth_degrees": math.degrees(azimuth),
        "elevation_degrees": math.degrees(elevation),
    }

    return camera, pose_info


def fibonacci_sphere_points(num_points: int, only_northern_hemisphere: bool = False) -> List[Tuple[float, float]]:
    """Generate evenly distributed points on a sphere using Fibonacci spiral.

    Args:
        num_points: Number of points to generate.
        only_northern_hemisphere: If True, only generate points in the upper hemisphere.

    Returns:
        List of (azimuth, elevation) tuples in radians.
    """
    points = []

    if only_northern_hemisphere:
        # Generate more points and take the northern ones
        effective_points = num_points
        phi = math.pi * (math.sqrt(5.0) - 1.0)  # golden angle in radians

        for i in range(effective_points):
            # y goes from 0 to 1 for northern hemisphere
            y = i / (effective_points - 1) if effective_points > 1 else 0.5
            theta = phi * i

            # Convert to spherical coordinates
            elevation = math.asin(y)  # 0 to pi/2 for northern hemisphere
            azimuth = theta % (2 * math.pi)

            points.append((azimuth, elevation))
    else:
        # Full sphere distribution
        phi = math.pi * (math.sqrt(5.0) - 1.0)  # golden angle in radians

        for i in range(num_points):
            # y goes from -1 to 1
            y = 1 - (i / (num_points - 1)) * 2 if num_points > 1 else 0
            theta = phi * i

            # Convert to spherical coordinates
            elevation = math.asin(y)  # -pi/2 to pi/2
            azimuth = theta % (2 * math.pi)

            points.append((azimuth, elevation))

    return points


def compute_optimal_camera_distance(
    object_radius: float,
    fov_degrees: float,
    fill_ratio: float = 0.9,
) -> float:
    """Compute the camera distance so the object fills the frame.

    Args:
        object_radius: Bounding sphere radius of the object.
        fov_degrees: Vertical field of view in degrees.
        fill_ratio: How much of the frame the object should fill (0.0-1.0).
                   1.0 means the object exactly fits, 0.9 means 90% of frame.

    Returns:
        The optimal camera distance from the origin.
    """
    # The object should fill fill_ratio of the frame
    # tan(fov/2) = (object_radius / fill_ratio) / distance
    # distance = object_radius / (fill_ratio * tan(fov/2))
    half_fov_rad = math.radians(fov_degrees) / 2.0
    tan_half_fov = math.tan(half_fov_rad)

    if tan_half_fov <= 1e-6:
        return object_radius * 10  # fallback for very small FOV

    distance = object_radius / (fill_ratio * tan_half_fov)
    return distance


def set_camera_at_position(
    azimuth: float,
    elevation: float,
    radius: float,
) -> Tuple[bpy.types.Object, Dict[str, float]]:
    """Set camera at a specific spherical position looking at origin.

    Args:
        azimuth: Azimuth angle in radians (0 to 2*pi).
        elevation: Elevation angle in radians (-pi/2 to pi/2).
        radius: Distance from origin.

    Returns:
        Tuple of camera object and pose info dictionary.
    """
    x = radius * math.cos(elevation) * math.cos(azimuth)
    y = radius * math.cos(elevation) * math.sin(azimuth)
    z = radius * math.sin(elevation)

    camera = bpy.data.objects["Camera"]
    camera.location = Vector((x, y, z))

    # Look at origin
    direction = -camera.location
    rot_quat = direction.to_track_quat("-Z", "Y")
    camera.rotation_euler = rot_quat.to_euler()

    pose_info = {
        "radius": radius,
        "azimuth_degrees": math.degrees(azimuth),
        "elevation_degrees": math.degrees(elevation),
    }

    return camera, pose_info


def _set_camera_at_size(i: int, scale: float = 1.5) -> bpy.types.Object:
    """Debugging function to set the camera on the 6 faces of a cube.

    Args:
        i (int): Index of the face of the cube.
        scale (float, optional): Scale of the cube. Defaults to 1.5.

    Returns:
        bpy.types.Object: The camera object.
    """
    if i == 0:
        x, y, z = scale, 0, 0
    elif i == 1:
        x, y, z = -scale, 0, 0
    elif i == 2:
        x, y, z = 0, scale, 0
    elif i == 3:
        x, y, z = 0, -scale, 0
    elif i == 4:
        x, y, z = 0, 0, scale
    elif i == 5:
        x, y, z = 0, 0, -scale
    else:
        raise ValueError(f"Invalid index: i={i}, must be int in range [0, 5].")
    camera = bpy.data.objects["Camera"]
    camera.location = Vector(np.array([x, y, z]))
    direction = -camera.location
    rot_quat = direction.to_track_quat("-Z", "Y")
    camera.rotation_euler = rot_quat.to_euler()
    return camera


def _create_light(
    name: str,
    light_type: Literal["POINT", "SUN", "SPOT", "AREA"],
    location: Tuple[float, float, float],
    rotation: Tuple[float, float, float],
    energy: float,
    use_shadow: bool = False,
    specular_factor: float = 1.0,
):
    """Creates a light object.

    Args:
        name (str): Name of the light object.
        light_type (Literal["POINT", "SUN", "SPOT", "AREA"]): Type of the light.
        location (Tuple[float, float, float]): Location of the light.
        rotation (Tuple[float, float, float]): Rotation of the light.
        energy (float): Energy of the light.
        use_shadow (bool, optional): Whether to use shadows. Defaults to False.
        specular_factor (float, optional): Specular factor of the light. Defaults to 1.0.

    Returns:
        bpy.types.Object: The light object.
    """

    light_data = bpy.data.lights.new(name=name, type=light_type)
    light_object = bpy.data.objects.new(name, light_data)
    bpy.context.collection.objects.link(light_object)
    light_object.location = location
    light_object.rotation_euler = rotation
    light_data.use_shadow = use_shadow
    light_data.specular_factor = specular_factor
    light_data.energy = energy
    return light_object


def setup_compositor_nodes(
    output_dir: str,
    base_name: str,
    render_depth: bool = False,
    render_normals: bool = False,
    render_mask: bool = False,
) -> None:
    """Sets up the compositor nodes for multi-pass rendering.

    Args:
        output_dir (str): Directory to save the rendered files.
        base_name (str): Base name for the output files.
        render_depth (bool): Whether to render depth pass.
        render_normals (bool): Whether to render normal pass.
        render_mask (bool): Whether to render mask pass.
    """
    scene = bpy.context.scene
    scene.use_nodes = True
    tree = scene.node_tree

    # Clear existing nodes
    for node in tree.nodes:
        tree.nodes.remove(node)

    # Create compositor nodes
    render_layers = tree.nodes.new("CompositorNodeRLayers")
    render_layers.location = (0, 0)
    
    # We don't need a Composite node if we are using FileOutput nodes for everything,
    # but it's good practice to have one for the "Render Result" image.
    # However, the user script didn't include one for the main image in the same way,
    # it used a FileOutput node for "RGB Output" which is clearer.
    # Let's fully adopt the user's script approach of File Output nodes.

    # RGB Output
    rgb_output = tree.nodes.new("CompositorNodeOutputFile")
    rgb_output.label = "RGB Output"
    rgb_output.base_path = output_dir
    rgb_output.file_slots[0].path = f"{base_name}_rgb_"
    rgb_output.format.file_format = "PNG"
    rgb_output.format.color_mode = "RGBA"
    rgb_output.location = (400, 300)

    # Create Alpha Over node for white background
    alpha_over = tree.nodes.new("CompositorNodeAlphaOver")
    alpha_over.location = (200, 300)
    alpha_over.use_premultiply = True
    alpha_over.inputs[1].default_value = (1.0, 1.0, 1.0, 1.0)  # White background
    
    # Connect Render Layer Image to Alpha Over (Foreground)
    tree.links.new(render_layers.outputs["Image"], alpha_over.inputs[2])

    # Connect Alpha Over to RGB Output
    # Note: We still use RGBA output format, but the alpha will be 1.0 from AlphaOver
    tree.links.new(alpha_over.outputs["Image"], rgb_output.inputs[0])

    if render_depth:
        depth_output = tree.nodes.new("CompositorNodeOutputFile")
        depth_output.label = "Depth Output"
        depth_output.base_path = output_dir
        depth_output.file_slots[0].path = f"{base_name}_depth_"
        depth_output.format.file_format = "OPEN_EXR"
        depth_output.format.color_depth = "32"
        depth_output.location = (400, 100)
        tree.links.new(render_layers.outputs["Depth"], depth_output.inputs[0])

    if render_normals:
        normal_output = tree.nodes.new("CompositorNodeOutputFile")
        normal_output.label = "Normal Output"
        normal_output.base_path = output_dir
        normal_output.file_slots[0].path = f"{base_name}_normal_"
        normal_output.format.file_format = "OPEN_EXR"
        normal_output.format.color_depth = "32"
        normal_output.location = (400, -100)
        tree.links.new(render_layers.outputs["Normal"], normal_output.inputs[0])

    if render_mask:
        mask_output = tree.nodes.new("CompositorNodeOutputFile")
        mask_output.label = "Mask Output"
        mask_output.base_path = output_dir
        mask_output.file_slots[0].path = f"{base_name}_mask_"
        mask_output.format.file_format = "PNG"
        mask_output.format.color_mode = "BW"
        mask_output.location = (400, -300)
        # Verify if Alpha or separate mask pass is better. User script used 'Alpha'.
        # But 'IndexOB' is what we might want for specific object masking if scene has background.
        # User script: links.new(render_layers.outputs['Alpha'], mask_output.inputs[0])
        # The user script sets film_transparent = True, so Alpha is the mask.
        tree.links.new(render_layers.outputs["Alpha"], mask_output.inputs[0])



def randomize_lighting() -> Dict[str, bpy.types.Object]:
    """Randomizes the lighting in the scene.

    Returns:
        Dict[str, bpy.types.Object]: Dictionary of the lights in the scene. The keys are
            "key_light", "fill_light", "rim_light", and "bottom_light".
    """

    # Clear existing lights
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_by_type(type="LIGHT")
    bpy.ops.object.delete()

    # Create key light
    key_light = _create_light(
        name="Key_Light",
        light_type="SUN",
        location=(0, 0, 0),
        rotation=(0.785398, 0, -0.785398),
        energy=random.choice([3, 4, 5]),
    )

    # Create fill light
    fill_light = _create_light(
        name="Fill_Light",
        light_type="SUN",
        location=(0, 0, 0),
        rotation=(0.785398, 0, 2.35619),
        energy=random.choice([2, 3, 4]),
    )

    # Create rim light
    rim_light = _create_light(
        name="Rim_Light",
        light_type="SUN",
        location=(0, 0, 0),
        rotation=(-0.785398, 0, -3.92699),
        energy=random.choice([3, 4, 5]),
    )

    # Create bottom light
    bottom_light = _create_light(
        name="Bottom_Light",
        light_type="SUN",
        location=(0, 0, 0),
        rotation=(3.14159, 0, 0),
        energy=random.choice([1, 2, 3]),
    )

    return dict(
        key_light=key_light,
        fill_light=fill_light,
        rim_light=rim_light,
        bottom_light=bottom_light,
    )


def setup_lighting_enhancements() -> None:
    """Sets a white background and adds an additional uniform light."""
    # White Background
    scene = bpy.context.scene
    world = scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        scene.world = world

    world.use_nodes = True
    world.node_tree.links.clear()
    for node in world.node_tree.nodes:
        world.node_tree.nodes.remove(node)

    bg_node = world.node_tree.nodes.new(type="ShaderNodeBackground")
    bg_node.inputs[0].default_value = (1.0, 1.0, 1.0, 1.0)
    bg_node.inputs[1].default_value = 1.0

    world_output = world.node_tree.nodes.new(type="ShaderNodeOutputWorld")
    world.node_tree.links.new(bg_node.outputs["Background"], world_output.inputs["Surface"])

    # # Additional Uniform Light
    # bpy.ops.object.light_add(type="SUN", location=(0, 0, 10))
    # light = bpy.context.active_object
    # light.name = "Additional_Uniform_Light"
    # light.data.energy = 2.0
    # light.data.use_shadow = False
    # light.data.angle = 3.14  # Softest possible (hemisphere) if applicable, or just generic fill


def reset_scene() -> None:
    """Resets the scene to a clean state.

    Returns:
        None
    """
    # delete everything that isn't part of a camera or a light
    for obj in bpy.data.objects:
        if obj.type not in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)

    # delete all the materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)

    # delete all the textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)

    # delete all the images
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)


def load_object(object_path: str) -> None:
    """Loads a model with a supported file extension into the scene.

    Args:
        object_path (str): Path to the model file.

    Raises:
        ValueError: If the file extension is not supported.

    Returns:
        None
    """
    file_extension = object_path.split(".")[-1].lower()
    if file_extension is None:
        raise ValueError(f"Unsupported file type: {object_path}")

    if file_extension == "usdz":
        # install usdz io package
        dirname = os.path.dirname(os.path.realpath(__file__))
        usdz_package = os.path.join(dirname, "io_scene_usdz.zip")
        bpy.ops.preferences.addon_install(filepath=usdz_package)
        # enable it
        addon_name = "io_scene_usdz"
        bpy.ops.preferences.addon_enable(module=addon_name)
        # import the usdz
        from io_scene_usdz.import_usdz import import_usdz

        import_usdz(context, filepath=object_path, materials=True, animations=True)
        return None

    # load from existing import functions
    import_function = IMPORT_FUNCTIONS[file_extension]

    if file_extension == "blend":
        import_function(directory=object_path, link=False)
    elif file_extension in {"glb", "gltf"}:
        import_function(filepath=object_path, merge_vertices=True)
    else:
        import_function(filepath=object_path)


def scene_bbox(
    single_obj: Optional[bpy.types.Object] = None, ignore_matrix: bool = False
) -> Tuple[Vector, Vector]:
    """Returns the bounding box of the scene.

    Taken from Shap-E rendering script
    (https://github.com/openai/shap-e/blob/main/shap_e/rendering/blender/blender_script.py#L68-L82)

    Args:
        single_obj (Optional[bpy.types.Object], optional): If not None, only computes
            the bounding box for the given object. Defaults to None.
        ignore_matrix (bool, optional): Whether to ignore the object's matrix. Defaults
            to False.

    Raises:
        RuntimeError: If there are no objects in the scene.

    Returns:
        Tuple[Vector, Vector]: The minimum and maximum coordinates of the bounding box.
    """
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in get_scene_meshes() if single_obj is None else [single_obj]:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            if not ignore_matrix:
                coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))

    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")

    return Vector(bbox_min), Vector(bbox_max)


def bounding_box_corners(bbox_min: Vector, bbox_max: Vector) -> List[Vector]:
    """Returns the 8 corners for the provided bounding box."""
    return [
        Vector((x, y, z))
        for x in (bbox_min.x, bbox_max.x)
        for y in (bbox_min.y, bbox_max.y)
        for z in (bbox_min.z, bbox_max.z)
    ]


def compute_bounding_sphere_radius(bbox_min: Vector, bbox_max: Vector) -> float:
    """Computes the radius of the bounding sphere that encloses the bbox."""
    corners = bounding_box_corners(bbox_min, bbox_max)
    if not corners:
        return 0.0
    return max(corner.length for corner in corners)


def minimum_fov_for_radius(object_radius: float, camera_radius: float) -> float:
    """Returns the minimum vertical FOV (in degrees) to see the object."""
    if camera_radius <= 1e-6 or object_radius <= 0.0:
        return 0.0
    ratio = max(0.0, min(object_radius / camera_radius, 0.999999))
    return math.degrees(2.0 * math.asin(ratio))


def compute_safe_camera_radius_range(
    requested_range: Tuple[float, float],
    object_radius: float,
    max_fov_degrees: float,
) -> Tuple[float, float]:
    """Raises the minimum camera radius when needed to keep the object in frame."""
    min_radius, max_radius = requested_range
    safe_min = max(min_radius, object_radius + CAMERA_RADIUS_MARGIN)
    half_angle_rad = math.radians(max_fov_degrees) / 2.0
    sin_half = math.sin(half_angle_rad)
    if sin_half > 1e-6:
        safe_min = max(safe_min, (object_radius / sin_half) + CAMERA_RADIUS_MARGIN)
    safe_min = min(safe_min, max_radius)
    return (safe_min, max_radius)


def get_scene_root_objects() -> Generator[bpy.types.Object, None, None]:
    """Returns all root objects in the scene.

    Yields:
        Generator[bpy.types.Object, None, None]: Generator of all root objects in the
            scene.
    """
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj


def get_scene_meshes() -> Generator[bpy.types.Object, None, None]:
    """Returns all meshes in the scene.

    Yields:
        Generator[bpy.types.Object, None, None]: Generator of all meshes in the scene.
    """
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj


def assign_object_pass_indices(index: int = 1) -> None:
    """Assigns the same object pass index to all meshes in the scene."""
    for obj in get_scene_meshes():
        obj.pass_index = index


def get_3x4_RT_matrix_from_blender(cam: bpy.types.Object) -> Matrix:
    """Returns the 3x4 RT matrix from the given camera.

    Taken from Zero123, which in turn was taken from
    https://github.com/panmari/stanford-shapenet-renderer/blob/master/render_blender.py

    Args:
        cam (bpy.types.Object): The camera object.

    Returns:
        Matrix: The 3x4 RT matrix from the given camera.
    """
    # Use matrix_world instead to account for all constraints
    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()

    # Use location from matrix_world to account for constraints:
    T_world2bcam = -1 * R_world2bcam @ location

    # put into 3x4 matrix
    RT = Matrix(
        (
            R_world2bcam[0][:] + (T_world2bcam[0],),
            R_world2bcam[1][:] + (T_world2bcam[1],),
            R_world2bcam[2][:] + (T_world2bcam[2],),
        )
    )
    return RT


def matrix4x4_to_list(mat: Matrix) -> List[List[float]]:
    """Converts a Blender Matrix to a nested Python list."""
    return [[float(mat[row][col]) for col in range(4)] for row in range(4)]


def compute_intrinsics(fov_radians: float, width: int, height: int) -> List[List[float]]:
    """Computes a 3x3 intrinsics matrix for a square sensor."""
    focal = width / (2.0 * math.tan(fov_radians / 2.0))
    cx = width / 2.0
    cy = height / 2.0
    return [
        [focal, 0.0, cx],
        [0.0, focal, cy],
        [0.0, 0.0, 1.0],
    ]




def delete_invisible_objects() -> None:
    """Deletes all invisible objects in the scene.

    Returns:
        None
    """
    bpy.ops.object.select_all(action="DESELECT")
    for obj in scene.objects:
        if obj.hide_viewport or obj.hide_render:
            obj.hide_viewport = False
            obj.hide_render = False
            obj.hide_select = False
            obj.select_set(True)
    bpy.ops.object.delete()

    # Delete invisible collections
    invisible_collections = [col for col in bpy.data.collections if col.hide_viewport]
    for col in invisible_collections:
        bpy.data.collections.remove(col)


def normalize_scene(
    target_extent: float = 1.0,
) -> Tuple[Dict[str, List[float]], Vector, Vector]:
    """Normalizes the scene by scaling and translating it to fit a cube centered at
    the origin.

    Mostly taken from the Point-E / Shap-E rendering script
    (https://github.com/openai/point-e/blob/main/point_e/evals/scripts/blender_script.py#L97-L112),
    but fix for multiple root objects: (see bug report here:
    https://github.com/openai/shap-e/pull/60).

    Args:
        target_extent (float): Desired edge length of the canonical cube. Defaults to 1.

    Returns:
        Tuple[Dict[str, List[float]], Vector, Vector]: Normalization metadata and the
        final bounding box min/max vectors in canonical space.
    """
    if len(list(get_scene_root_objects())) > 1:
        parent_empty = bpy.data.objects.new("ParentEmpty", None)
        bpy.context.scene.collection.objects.link(parent_empty)
        for obj in get_scene_root_objects():
            if obj != parent_empty:
                obj.parent = parent_empty

    bbox_min, bbox_max = scene_bbox()
    max_extent = max(bbox_max - bbox_min)
    scale = target_extent / max_extent if max_extent > 0 else 1.0
    for obj in get_scene_root_objects():
        obj.scale = obj.scale * scale

    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    for obj in get_scene_root_objects():
        obj.matrix_world.translation += offset

    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    normalization = {
        "scale": scale,
        "translation": list(offset),
        "bbox_min": list(bbox_min),
        "bbox_max": list(bbox_max),
    }

    bpy.ops.object.select_all(action="DESELECT")
    bpy.data.objects["Camera"].parent = None
    return normalization, bbox_min, bbox_max


def delete_missing_textures() -> Dict[str, Any]:
    """Deletes all missing textures in the scene.

    Returns:
        Dict[str, Any]: Dictionary with keys "count", "files", and "file_path_to_color".
            "count" is the number of missing textures, "files" is a list of the missing
            texture file paths, and "file_path_to_color" is a dictionary mapping the
            missing texture file paths to a random color.
    """
    missing_file_count = 0
    out_files = []
    file_path_to_color = {}

    # Check all materials in the scene
    for material in bpy.data.materials:
        if material.use_nodes:
            for node in material.node_tree.nodes:
                if node.type == "TEX_IMAGE":
                    image = node.image
                    if image is not None:
                        file_path = bpy.path.abspath(image.filepath)
                        if file_path == "":
                            # means it's embedded
                            continue

                        if not os.path.exists(file_path):
                            # Find the connected Principled BSDF node
                            connected_node = node.outputs[0].links[0].to_node

                            if connected_node.type == "BSDF_PRINCIPLED":
                                if file_path not in file_path_to_color:
                                    # Set a random color for the unique missing file path
                                    random_color = [random.random() for _ in range(3)]
                                    file_path_to_color[file_path] = random_color + [1]

                                connected_node.inputs[
                                    "Base Color"
                                ].default_value = file_path_to_color[file_path]

                            # Delete the TEX_IMAGE node
                            material.node_tree.nodes.remove(node)
                            missing_file_count += 1
                            out_files.append(image.filepath)
    return {
        "count": missing_file_count,
        "files": out_files,
        "file_path_to_color": file_path_to_color,
    }


def _get_random_color() -> Tuple[float, float, float, float]:
    """Generates a random RGB-A color.

    The alpha value is always 1.

    Returns:
        Tuple[float, float, float, float]: A random RGB-A color. Each value is in the
        range [0, 1].
    """
    return (random.random(), random.random(), random.random(), 1)


def _apply_color_to_object(
    obj: bpy.types.Object, color: Tuple[float, float, float, float]
) -> None:
    """Applies the given color to the object.

    Args:
        obj (bpy.types.Object): The object to apply the color to.
        color (Tuple[float, float, float, float]): The color to apply to the object.

    Returns:
        None
    """
    mat = bpy.data.materials.new(name=f"RandomMaterial_{obj.name}")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    principled_bsdf = nodes.get("Principled BSDF")
    if principled_bsdf:
        principled_bsdf.inputs["Base Color"].default_value = color
    obj.data.materials.append(mat)


def apply_single_random_color_to_all_objects() -> Tuple[float, float, float, float]:
    """Applies a single random color to all objects in the scene.

    Returns:
        Tuple[float, float, float, float]: The random color that was applied to all
        objects.
    """
    rand_color = _get_random_color()
    for obj in bpy.context.scene.objects:
        if obj.type == "MESH":
            _apply_color_to_object(obj, rand_color)
    return rand_color


class MetadataExtractor:
    """Class to extract metadata from a Blender scene."""

    def __init__(
        self, object_path: str, scene: bpy.types.Scene, bdata: bpy.types.BlendData
    ) -> None:
        """Initializes the MetadataExtractor.

        Args:
            object_path (str): Path to the object file.
            scene (bpy.types.Scene): The current scene object from `bpy.context.scene`.
            bdata (bpy.types.BlendData): The current blender data from `bpy.data`.

        Returns:
            None
        """
        self.object_path = object_path
        self.scene = scene
        self.bdata = bdata

    def get_poly_count(self) -> int:
        """Returns the total number of polygons in the scene."""
        total_poly_count = 0
        for obj in self.scene.objects:
            if obj.type == "MESH":
                total_poly_count += len(obj.data.polygons)
        return total_poly_count

    def get_vertex_count(self) -> int:
        """Returns the total number of vertices in the scene."""
        total_vertex_count = 0
        for obj in self.scene.objects:
            if obj.type == "MESH":
                total_vertex_count += len(obj.data.vertices)
        return total_vertex_count

    def get_edge_count(self) -> int:
        """Returns the total number of edges in the scene."""
        total_edge_count = 0
        for obj in self.scene.objects:
            if obj.type == "MESH":
                total_edge_count += len(obj.data.edges)
        return total_edge_count

    def get_lamp_count(self) -> int:
        """Returns the number of lamps in the scene."""
        return sum(1 for obj in self.scene.objects if obj.type == "LIGHT")

    def get_mesh_count(self) -> int:
        """Returns the number of meshes in the scene."""
        return sum(1 for obj in self.scene.objects if obj.type == "MESH")

    def get_material_count(self) -> int:
        """Returns the number of materials in the scene."""
        return len(self.bdata.materials)

    def get_object_count(self) -> int:
        """Returns the number of objects in the scene."""
        return len(self.bdata.objects)

    def get_animation_count(self) -> int:
        """Returns the number of animations in the scene."""
        return len(self.bdata.actions)

    def get_linked_files(self) -> List[str]:
        """Returns the filepaths of all linked files."""
        image_filepaths = self._get_image_filepaths()
        material_filepaths = self._get_material_filepaths()
        linked_libraries_filepaths = self._get_linked_libraries_filepaths()

        all_filepaths = (
            image_filepaths | material_filepaths | linked_libraries_filepaths
        )
        if "" in all_filepaths:
            all_filepaths.remove("")
        return list(all_filepaths)

    def _get_image_filepaths(self) -> Set[str]:
        """Returns the filepaths of all images used in the scene."""
        filepaths = set()
        for image in self.bdata.images:
            if image.source == "FILE":
                filepaths.add(bpy.path.abspath(image.filepath))
        return filepaths

    def _get_material_filepaths(self) -> Set[str]:
        """Returns the filepaths of all images used in materials."""
        filepaths = set()
        for material in self.bdata.materials:
            if material.use_nodes:
                for node in material.node_tree.nodes:
                    if node.type == "TEX_IMAGE":
                        image = node.image
                        if image is not None:
                            filepaths.add(bpy.path.abspath(image.filepath))
        return filepaths

    def _get_linked_libraries_filepaths(self) -> Set[str]:
        """Returns the filepaths of all linked libraries."""
        filepaths = set()
        for library in self.bdata.libraries:
            filepaths.add(bpy.path.abspath(library.filepath))
        return filepaths

    def get_scene_size(self) -> Dict[str, list]:
        """Returns the size of the scene bounds in meters."""
        bbox_min, bbox_max = scene_bbox()
        return {"bbox_max": list(bbox_max), "bbox_min": list(bbox_min)}

    def get_shape_key_count(self) -> int:
        """Returns the number of shape keys in the scene."""
        total_shape_key_count = 0
        for obj in self.scene.objects:
            if obj.type == "MESH":
                shape_keys = obj.data.shape_keys
                if shape_keys is not None:
                    total_shape_key_count += (
                        len(shape_keys.key_blocks) - 1
                    )  # Subtract 1 to exclude the Basis shape key
        return total_shape_key_count

    def get_armature_count(self) -> int:
        """Returns the number of armatures in the scene."""
        total_armature_count = 0
        for obj in self.scene.objects:
            if obj.type == "ARMATURE":
                total_armature_count += 1
        return total_armature_count

    def read_file_size(self) -> int:
        """Returns the size of the file in bytes."""
        return os.path.getsize(self.object_path)

    def get_metadata(self) -> Dict[str, Any]:
        """Returns the metadata of the scene.

        Returns:
            Dict[str, Any]: Dictionary of the metadata with keys for "file_size",
            "poly_count", "vert_count", "edge_count", "material_count", "object_count",
            "lamp_count", "mesh_count", "animation_count", "linked_files", "scene_size",
            "shape_key_count", and "armature_count".
        """
        return {
            "file_size": self.read_file_size(),
            "poly_count": self.get_poly_count(),
            "vert_count": self.get_vertex_count(),
            "edge_count": self.get_edge_count(),
            "material_count": self.get_material_count(),
            "object_count": self.get_object_count(),
            "lamp_count": self.get_lamp_count(),
            "mesh_count": self.get_mesh_count(),
            "animation_count": self.get_animation_count(),
            "linked_files": self.get_linked_files(),
            "scene_size": self.get_scene_size(),
            "shape_key_count": self.get_shape_key_count(),
            "armature_count": self.get_armature_count(),
        }


def render_object(
    object_file: str,
    num_renders: int,
    only_northern_hemisphere: bool,
    output_dir: str,
    render_depth: bool = False,
    render_normals: bool = False,
    render_mask: bool = False,
    camera_radius_range: Tuple[float, float] = CAMERA_RADIUS_RANGE,
    fov_degrees_range: Tuple[float, float] = FOV_DEGREES_RANGE,
) -> None:
    """Saves rendered images with its camera matrix and metadata of the object.

    Args:
        object_file (str): Path to the object file.
        num_renders (int): Number of renders to save of the object.
        only_northern_hemisphere (bool): Whether to only render sides of the object that
            are in the northern hemisphere. This is useful for rendering objects that
            are photogrammetrically scanned, as the bottom of the object often has
            holes.
        output_dir (str): Path to the directory where the rendered images and metadata
            will be saved.
        render_depth (bool): Whether to render linear depth maps.
        render_normals (bool): Whether to render surface normal maps.
        render_mask (bool): Whether to render object index masks.
        camera_radius_range (Tuple[float, float]): Camera radius sampling range.
        fov_degrees_range (Tuple[float, float]): Allowed field-of-view range.

    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)

    # load the object
    if object_file.endswith(".blend"):
        bpy.ops.object.mode_set(mode="OBJECT")
        reset_cameras()
        delete_invisible_objects()
    else:
        reset_scene()
        load_object(object_file)

    # Set up cameras
    cam = scene.objects["Camera"]
    cam.data.lens = 35
    cam.data.sensor_width = 32

    # Set up camera constraints
    cam_constraint = cam.constraints.new(type="TRACK_TO")
    cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
    cam_constraint.up_axis = "UP_Y"
    empty = bpy.data.objects.new("Empty", None)
    scene.collection.objects.link(empty)
    cam_constraint.target = empty

    # Extract the metadata. This must be done before normalizing the scene to get
    # accurate bounding box information.
    metadata_extractor = MetadataExtractor(
        object_path=object_file, scene=scene, bdata=bpy.data
    )
    metadata = metadata_extractor.get_metadata()

    # delete all objects that are not meshes
    if object_file.lower().endswith(".usdz"):
        # don't delete missing textures on usdz files, lots of them are embedded
        missing_textures = None
    else:
        missing_textures = delete_missing_textures()
    metadata["missing_textures"] = missing_textures

    # possibly apply a random color to all objects
    if object_file.endswith(".stl") or object_file.endswith(".ply"):
        assert len(bpy.context.selected_objects) == 1
        rand_color = apply_single_random_color_to_all_objects()
        metadata["random_color"] = rand_color
    else:
        metadata["random_color"] = None

    # normalize the scene
    normalization, bbox_min, bbox_max = normalize_scene()
    bounding_sphere_radius = compute_bounding_sphere_radius(bbox_min, bbox_max)
    normalization["bounding_sphere_radius"] = bounding_sphere_radius
    normalization["bbox_min"] = list(bbox_min)
    normalization["bbox_max"] = list(bbox_max)

    # ensure sampled radii keep the normalized object inside the frame
    camera_radius_range = compute_safe_camera_radius_range(
        requested_range=camera_radius_range,
        object_radius=bounding_sphere_radius,
        max_fov_degrees=fov_degrees_range[1],
    )
    if empty.parent is not None:
        empty.parent = None
    empty.location = Vector((0.0, 0.0, 0.0))

    # randomize the lighting
    randomize_lighting()
    
    # enhance lighting (white background + uniform fill)
    setup_lighting_enhancements()
    if render_mask:
        assign_object_pass_indices()

    enabled_modalities = ["rgb"]
    if render_depth:
        enabled_modalities.append("depth")
    if render_normals:
        enabled_modalities.append("normals")
    if render_mask:
        enabled_modalities.append("mask")
    
    # Base name for output files
    base_name = "render"

    # Setup compositor nodes
    setup_compositor_nodes(
        output_dir=output_dir,
        base_name=base_name,
        render_depth=render_depth,
        render_normals=render_normals,
        render_mask=render_mask,
    )

    # Generate evenly spaced camera positions using Fibonacci sphere
    camera_positions = fibonacci_sphere_points(num_renders, only_northern_hemisphere)

    # Use fixed FOV and compute optimal distance so object fills frame
    # Using 50 degrees FOV as a good balance between perspective distortion and view size
    fixed_fov_deg = 50.0
    # Compute distance so object fills 95% of frame (leaving some margin)
    optimal_distance = compute_optimal_camera_distance(
        object_radius=bounding_sphere_radius,
        fov_degrees=fixed_fov_deg,
        fill_ratio=0.95,
    )

    # render the images
    views: List[Dict[str, Any]] = []
    for i in range(num_renders):
        # set camera pose using evenly distributed position
        azimuth, elevation = camera_positions[i]
        camera, pose_info = set_camera_at_position(
            azimuth=azimuth,
            elevation=elevation,
            radius=optimal_distance,
        )

        # Use fixed FOV for consistent framing
        fov_deg = fixed_fov_deg
        pose_info["fov_degrees"] = fov_deg
        fov_rad = math.radians(fov_deg)
        camera.data.angle = fov_rad
        intrinsics = compute_intrinsics(
            fov_radians=fov_rad,
            width=GS_LRM_RESOLUTION,
            height=GS_LRM_RESOLUTION,
        )


        # render the image
        # Scene frame drives the naming of the compositor output (e.g. _0001)
        scene.frame_set(i) # output filenames will be suffixed with frame number
        
        # We don"t need to set filepath here as compositor handles it, 
        # but we can set it to a dummy or keep it to avoid errors if any.
        scene.render.filepath = os.path.join(output_dir, f"temp_render_{i:03d}")
        
        bpy.ops.render.render(write_still=False) # write_still=False bc compositor writes files

        # Expected filenames based on Compositor output
        # Format: {base_name}_{type}_{frame:04d}.{ext}
        # e.g. render_rgb_0000.png (if i=0)
        # Note: Blender frame numbering usually starts at 1 by default or follows current frame.
        # We set scene.frame_set(i). If i=0, we get 0000.
        
        # We will reconstruct the filenames in the metadata
        frame_str = f"{i:04d}"
        rgb_filename = f"{base_name}_rgb_{frame_str}.png"
        
        depth_filename: Optional[str] = None
        if render_depth:
            depth_filename = f"{base_name}_depth_{frame_str}.exr"

        normal_filename: Optional[str] = None
        if render_normals:
            normal_filename = f"{base_name}_normal_{frame_str}.exr"

        mask_filename: Optional[str] = None
        if render_mask:
            mask_filename = f"{base_name}_mask_{frame_str}.png"

            # Post-process RGB using mask: set background (black) to white
            try:
                rgb_path = os.path.join(output_dir, rgb_filename)
                mask_path = os.path.join(output_dir, mask_filename)

                if os.path.exists(rgb_path) and os.path.exists(mask_path):
                    # Load images
                    img_rgb = bpy.data.images.load(rgb_path)
                    img_mask = bpy.data.images.load(mask_path)

                    # Prepare numpy arrays
                    w, h = img_rgb.size
                    rgb_pixels = np.empty(w * h * 4, dtype=np.float32)
                    mask_pixels = np.empty(w * h * 4, dtype=np.float32)

                    img_rgb.pixels.foreach_get(rgb_pixels)
                    img_mask.pixels.foreach_get(mask_pixels)

                    # Reshape
                    rgb_pixels = rgb_pixels.reshape((h, w, 4))
                    mask_pixels = mask_pixels.reshape((h, w, 4))

                    # Mask logic: where mask is 0 (black), set RGB to 1 (white)
                    # Use red channel of mask (BW mask loaded as RGBA)
                    is_bg = mask_pixels[:, :, 0] < 0.5
                    rgb_pixels[is_bg, 0] = 1.0  # R
                    rgb_pixels[is_bg, 1] = 1.0  # G
                    rgb_pixels[is_bg, 2] = 1.0  # B
                    rgb_pixels[is_bg, 3] = 1.0  # Alpha (ensure opaque)

                    # Write back
                    img_rgb.pixels.foreach_set(rgb_pixels.ravel())
                    img_rgb.save()

                    # Cleanup
                    bpy.data.images.remove(img_rgb)
                    bpy.data.images.remove(img_mask)
            except Exception as e:
                print(f"Error in post-processing frame {i}: {e}")


        view_entry: Dict[str, Any] = {
            "image": rgb_filename,
            "fov_degrees": fov_deg,
            "intrinsics": intrinsics,
            "focal_length_px": intrinsics[0][0],
            "camera_to_world": matrix4x4_to_list(camera.matrix_world),
            "pose": pose_info,
        }
        if depth_filename is not None:
            view_entry["depth"] = depth_filename
        if normal_filename is not None:
            view_entry["normal"] = normal_filename
        if mask_filename is not None:
            view_entry["mask"] = mask_filename
        views.append(view_entry)

    render_settings: Dict[str, Any] = {
        "resolution": [GS_LRM_RESOLUTION, GS_LRM_RESOLUTION],
        "num_views": num_renders,
        "camera_distance": optimal_distance,
        "fov_degrees": fixed_fov_deg,
        "camera_distribution": "fibonacci_sphere",
        "only_northern_hemisphere": only_northern_hemisphere,
        "fill_ratio": 0.95,
        "engine": scene.render.engine,
        "modalities": enabled_modalities,
    }
    if render_depth:
        render_settings["depth_pass"] = "Z"
    if render_normals:
        render_settings["normal_pass"] = "Normal"
    if render_mask:
        render_settings["mask_pass"] = "IndexOB"

    render_summary = {
        "normalization": normalization,
        "render_settings": render_settings,
        "views": views,
    }

    metadata.update(render_summary)
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, sort_keys=True, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--object_path",
        type=str,
        required=True,
        help="Path to the object file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the directory where the rendered images and metadata will be saved.",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="BLENDER_EEVEE",
        choices=["CYCLES", "BLENDER_EEVEE"],
    )
    parser.add_argument(
        "--only_northern_hemisphere",
        action="store_true",
        help="Only render the northern hemisphere of the object.",
        default=False,
    )
    parser.add_argument(
        "--num_renders",
        type=int,
        default=32,
        help="Number of renders to save of the object.",
    )
    parser.add_argument(
        "--render_depth",
        action="store_true",
        help="Render per-view depth maps.",
        default=False,
    )
    parser.add_argument(
        "--render_normals",
        action="store_true",
        help="Render per-view surface normal maps.",
        default=False,
    )
    parser.add_argument(
        "--render_mask",
        action="store_true",
        help="Render per-view binary object masks.",
        default=False,
    )
    parser.add_argument(
        "--camera_radius_min",
        type=float,
        default=CAMERA_RADIUS_RANGE[0],
        help="Minimum radius for sampled camera positions.",
    )
    parser.add_argument(
        "--camera_radius_max",
        type=float,
        default=CAMERA_RADIUS_RANGE[1],
        help="Maximum radius for sampled camera positions.",
    )
    parser.add_argument(
        "--fov_min_degrees",
        type=float,
        default=FOV_DEGREES_RANGE[0],
        help="Minimum sampled camera FOV in degrees.",
    )
    parser.add_argument(
        "--fov_max_degrees",
        type=float,
        default=FOV_DEGREES_RANGE[1],
        help="Maximum sampled camera FOV in degrees.",
    )
    argv = sys.argv[sys.argv.index("--") + 1 :]
    args = parser.parse_args(argv)

    context = bpy.context
    scene = context.scene
    render = scene.render

    # Set render settings
    render.engine = args.engine
    render.image_settings.file_format = "PNG"
    render.image_settings.color_mode = "RGB"
    render.resolution_x = GS_LRM_RESOLUTION
    render.resolution_y = GS_LRM_RESOLUTION
    render.resolution_percentage = 100
    # scene.view_layers["ViewLayer"].use_pass_z = args.render_depth
    # scene.view_layers["ViewLayer"].use_pass_normal = args.render_normals
    # scene.view_layers["ViewLayer"].use_pass_object_index = args.render_mask
    
    # The user script sets these directly on view_layer
    view_layer = context.view_layer
    view_layer.use_pass_z = args.render_depth
    view_layer.use_pass_normal = args.render_normals
    # Alpha is used for mask in the user script approach with film_transparent
    scene.render.film_transparent = True


    # Set cycles settings
    scene.cycles.device = "GPU"
    scene.cycles.samples = 128
    scene.cycles.diffuse_bounces = 1
    scene.cycles.glossy_bounces = 1
    scene.cycles.transparent_max_bounces = 3
    scene.cycles.transmission_bounces = 3
    scene.cycles.filter_width = 0.01
    scene.cycles.use_denoising = True
    scene.cycles.use_denoising = True
    # scene.render.film_transparent = False  <-- REMOVED
    bpy.context.preferences.addons["cycles"].preferences.get_devices()
    bpy.context.preferences.addons["cycles"].preferences.get_devices()
    bpy.context.preferences.addons[
        "cycles"
    ].preferences.compute_device_type = "CUDA"  # or "OPENCL"

    # Render the images
    camera_radius_min = max(0.1, args.camera_radius_min)
    camera_radius_max = max(camera_radius_min, args.camera_radius_max)
    fov_min = max(1.0, args.fov_min_degrees)
    fov_max = max(fov_min + 1e-3, args.fov_max_degrees)

    render_object(
        object_file=args.object_path,
        num_renders=args.num_renders,
        only_northern_hemisphere=args.only_northern_hemisphere,
        output_dir=args.output_dir,
        render_depth=args.render_depth,
        render_normals=args.render_normals,
        render_mask=args.render_mask,
        camera_radius_range=(camera_radius_min, camera_radius_max),
        fov_degrees_range=(fov_min, fov_max),
    )
