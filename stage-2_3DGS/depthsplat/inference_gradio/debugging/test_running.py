
import os
import sys
import torch
import numpy as np
from PIL import Image

# Add parent dir to path
sys.path.append("/home/sandro/thesis/code/depthsplat")
sys.path.append("/home/sandro/thesis/code/depthsplat/inference_gradio")

from runner import DepthSplatRunner

def test_inference():
    checkpoint = "/home/sandro/thesis/code/depthsplat/outputs/objaverse_white_small_gauss/checkpoints/epoch_0-step_100000.ckpt"
    config = "objaverse_white_small_gauss"
    render_dir = "/home/sandro/thesis/renders/5cams_10-01-26"
    frame_id = 60
    
    print(f"Initializing Runner...")
    runner = DepthSplatRunner(checkpoint, config)
    
    print(f"Loading Wild Frame {frame_id}...")
    image_paths, status = runner.load_wild_frame(frame_id, render_dir)
    print(status)
    
    # Get loaded context
    ctx = runner.current_example
    extrinsics = ctx['extrinsics']
    intrinsics = ctx['intrinsics']
    scales = ctx['scale_factor']
    center = ctx['center']
    
    print(f"Scale Factor: {scales}")
    print(f"Center: {center}")
    print(f"Near: {runner.near}, Far: {runner.far}")
    
    # Check 1st camera
    ext0 = extrinsics[0]
    pos0 = ext0[:3, 3]
    rot0 = ext0[:3, :3]
    forward0 = rot0[:, 2] # OpenCV +Z
    
    print(f"Cam 0 Scaled Pos: {pos0}")
    print(f"Cam 0 Forward (World): {forward0}")
    
    # Check if pointing at origin (0,0,0) (since we centered)
    # Vector from Cam to Origin: -Pos
    to_origin = -pos0
    to_origin_n = to_origin / np.linalg.norm(to_origin)
    dot = np.dot(forward0, to_origin_n)
    print(f"Dot with To-Origin: {dot:.4f}")
    
    # Load images
    images = []
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        images.append(np.array(img))
        
    # Run inference with a dummy target (same as cam 0)
    print("Running Inference...")
    target_ext = extrinsics[0:1] # [1, 4, 4]
    
    res = runner.run_inference(
        images=images,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        target_extrinsics=target_ext,
        num_video_frames=0,
        output_dir="/tmp/debug_depthsplat"
    )
    
    # Check output
    rendered = res['rendered_images'][0] # [H, W, 3]
    print(f"Rendered shape: {rendered.shape}")
    print(f"Rendered mean: {rendered.mean()}")
    print(f"Rendered min/max: {rendered.min()}, {rendered.max()}")
    
    if rendered.mean() > 250:
        print("FAIL: Image is WHITE")
    else:
        print("SUCCESS: Image has content")

if __name__ == "__main__":
    test_inference()
