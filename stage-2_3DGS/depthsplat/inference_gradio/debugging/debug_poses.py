
import json
import numpy as np
import os
import sys

def check_render(render_dir, frame_id=60):
    json_path = os.path.join(render_dir, "drone_camera_observations.json")
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    # Find frame
    frame = None
    for f_data in data.get("frames", []):
        if f_data["frame_index"] == frame_id:
            frame = f_data
            break
    
    if not frame:
        print(f"Error: Frame {frame_id} not found")
        return

    drone_pos = np.array(frame["drone_position_3d"])
    print(f"Drone Position: {drone_pos}")

    cameras = frame.get("cameras", [])
    print(f"Found {len(cameras)} cameras")

    for cam in cameras:
        name = cam["name"]
        # Try to find extrinsics in metadata
        extrinsics = None
        for c_info in data["metadata"]["cameras_recorded"]: # Or "cameras" in newer format
             # Checking structure from code... 
             pass
        
        # Actually structure is likely: data['cameras'] list in root
        # Check root cameras
        for c_info in data.get("cameras", []):
            if c_info["name"] == name:
                extrinsics = np.array(c_info["extrinsics"]["camera_to_world_matrix"])
                break
        
        if extrinsics is None:
            print(f"  {name}: No extrinsics found")
            continue

        pos = extrinsics[:3, 3]
        
        # Vector from Camera to Object
        cam_to_obj = drone_pos - pos
        dist = np.linalg.norm(cam_to_obj)
        cam_to_obj_norm = cam_to_obj / dist

        print(f"  {name}: Pos {pos}, Dist to Obj {dist:.2f}")

        # Check conventions
        # 1. GL: Forward is -Z (col 2)
        forward_gl = -extrinsics[:3, 2]
        dot_gl = np.dot(forward_gl, cam_to_obj_norm)
        
        # 2. CV: Forward is +Z (col 2)
        forward_cv = extrinsics[:3, 2]
        dot_cv = np.dot(forward_cv, cam_to_obj_norm)

        print(f"    Dot GL (Forward=-Z): {dot_gl:.4f}  ({'LOOKING AT' if dot_gl > 0.9 else 'AWAY'})")
        print(f"    Dot CV (Forward=+Z): {dot_cv:.4f}  ({'LOOKING AT' if dot_cv > 0.9 else 'AWAY'})")
        
        # Check Up vector
        # GL: Up is +Y (col 1)
        up_gl = extrinsics[:3, 1]
        # CV: Up is -Y (col 1)
        up_cv = -extrinsics[:3, 1]
        
        print(f"    Up GL (+Y): {up_gl}")

        # Check what transformation makes it CV
        # If GL is correct (dot_gl > 0.9), and we want CV (dot_cv > 0.9)...
        # We need to transform the matrix so that New +Z points in direction of GL -Z.
        # New Z = -Old Z.
        # This requires flipping Z.
        # To maintain right-handedness, we usually flip Y too.
        
        if dot_gl > 0.9:
            print("    -> Native format seems to be OPENGL (-Z forward)")
        elif dot_cv > 0.9:
            print("    -> Native format seems to be OPENCV (+Z forward)")
        else:
            print("    -> Native format seems weird / not pointing at object")

if __name__ == "__main__":
    dirs = [
        "/home/sandro/thesis/renders/5cams_10-01-26_bird_10m",
    ]
    for d in dirs:
        print(f"Checking {d}...")
        check_render(d)
