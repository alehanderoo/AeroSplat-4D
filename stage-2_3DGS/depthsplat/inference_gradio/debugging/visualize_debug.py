
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Add path to runner
sys.path.append("/home/sandro/thesis/code/depthsplat")
sys.path.append("/home/sandro/thesis/code/depthsplat/inference_gradio")

from runner import DepthSplatRunner

def generate_target_camera(azimuth, elevation, distance):
    # Standard logic from app.py
    base_radius = 2.0  # MATCHING UPDATED APP.PY
    radius = base_radius * distance
    
    azimuth_rad = np.deg2rad(azimuth)
    elevation_rad = np.deg2rad(elevation)
    
    # Spherical coordinates
    x = radius * np.cos(elevation_rad) * np.cos(azimuth_rad)
    y = radius * np.cos(elevation_rad) * np.sin(azimuth_rad)
    z = radius * np.sin(elevation_rad)
    
    cam_pos = np.array([x, y, z], dtype=np.float32)
    
    target = np.array([0, 0, 0], dtype=np.float32)
    forward = target - cam_pos
    forward = forward / np.linalg.norm(forward)
    
    world_up = np.array([0, 0, 1], dtype=np.float32)
    
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-6:
        world_up = np.array([0, 1, 0], dtype=np.float32)
        right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)
    
    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)

    down = -up
    R = np.stack([right, down, forward], axis=1) # OpenCV cols: Right, Down, Forward
    
    extrinsic = np.eye(4, dtype=np.float32)
    extrinsic[:3, :3] = R
    extrinsic[:3, 3] = cam_pos
    
    return extrinsic

def main():
    checkpoint = "/home/sandro/thesis/code/depthsplat/outputs/objaverse_white_small_gauss/checkpoints/epoch_0-step_100000.ckpt"
    config = "objaverse_white_small_gauss"
    render_dir = "/home/sandro/thesis/renders/5cams_10-01-26"
    frame_id = 60
    output_dir = "/home/sandro/thesis/code/depthsplat/inference_gradio/debugging/imgs"
    
    print("Initializing Runner...")
    runner = DepthSplatRunner(checkpoint, config)
    
    print("Loading Wild Frame...")
    image_paths, _ = runner.load_wild_frame(frame_id, render_dir)
    images = [np.array(Image.open(p).convert("RGB")) for p in image_paths]
    
    ctx = runner.current_example
    extrinsics = ctx['extrinsics']
    intrinsics = ctx['intrinsics']
    
    # Generate Synthetic Camera (Az=0, El=-60, Dist=1.0)
    print("Generating Synthetic Target Cam (Az=0, El=-60, Dist=1.0)...")
    target_ext_synth = generate_target_camera(0, -60, 1.0)
    target_exts_synth = target_ext_synth[np.newaxis, ...]
    
    # Save comparison of positions
    print("Input Cam 0 Pos:", extrinsics[0][:3, 3])
    print("Synth Cam Pos:", target_ext_synth[:3, 3])
    
    res = runner.run_inference(
        images=images,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        target_extrinsics=target_exts_synth,
        output_dir=output_dir,
        num_video_frames=0
    )
    
    rendered = res['rendered_images'][0]
    save_path = os.path.join(output_dir, "render_synth_el_minus60.png")
    Image.fromarray(rendered).save(save_path)
    print(f"Saved synthetic render to {save_path}")
    print(f"Mean pixel val: {rendered.mean()}")


if __name__ == "__main__":
    main()
