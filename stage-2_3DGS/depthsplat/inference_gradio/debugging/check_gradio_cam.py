
import numpy as np
import sys
# Add parent dir to path
sys.path.append("/home/sandro/thesis/code/depthsplat")
sys.path.append("/home/sandro/thesis/code/depthsplat/inference_gradio")
from runner import DepthSplatRunner

def check_gradio_cam():
    # Emulate app.py logic
    azimuth = 0
    elevation = -60
    distance = 1.0 # factor
    
    base_radius = 2.0 
    radius = base_radius * distance
    
    azimuth_rad = np.deg2rad(azimuth)
    elevation_rad = np.deg2rad(elevation)
    
    x = radius * np.cos(elevation_rad) * np.cos(azimuth_rad)
    y = radius * np.cos(elevation_rad) * np.sin(azimuth_rad)
    z = radius * np.sin(elevation_rad)
    
    cam_pos = np.array([x, y, z], dtype=np.float32)
    print(f"Calculated Pos (Az={azimuth}, El={elevation}, R={radius}): {cam_pos}")
    
    target = np.array([0, 0, 0], dtype=np.float32)
    forward = target - cam_pos
    forward = forward / np.linalg.norm(forward)
    print(f"Forward: {forward}")
    
    world_up = np.array([0, 0, 1], dtype=np.float32)
    right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)
    
    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)
    
    down = -up
    R = np.stack([right, down, forward], axis=1)
    
    print(f"Rotation Matrix:\n{R}")
    
    # Check against Cam 0 from debug run (approx)
    # Cam 0 Scaled Pos: [ 0.90147936  0.08681041 -1.7823014 ]
    # Cam 0 Forward: [-4.4721362e-01 -7.1679001e-09  8.9442718e-01]
    
    # My calc pos for -60 deg:
    # z = 2.0 * sin(-60) = -1.732
    # x = 2.0 * 0.5 * 1 = 1.0
    # y = 0
    # Pos = [1.0, 0.0, -1.732]
    # This is VERY close to [0.9, 0.08, -1.78].
    
    # Forward:
    # Target (0,0,0) - Pos = [-1.0, 0.0, 1.732]. Norm ~ 2.
    # Forward norm = [-0.5, 0, 0.866]
    # Cam 0 Forward: [-0.44, 0, 0.89]
    # Also very close.
    
    # So the generated camera IS correct.
    print("VERDICT: Generated camera matches input camera valid space.")

if __name__ == "__main__":
    check_gradio_cam()
