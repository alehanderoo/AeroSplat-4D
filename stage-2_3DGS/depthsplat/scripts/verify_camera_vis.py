
import torch
import sys
from pathlib import Path

# Add src to python path
sys.path.append(str(Path(__file__).parent.parent))

from src.visualization.validation_in_3d import render_cameras
from src.misc.image_io import save_image
from src.visualization.layout import hcat

def test_render_cameras():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create dummy batch
    batch_size = 1
    num_context = 2
    num_target = 1
    
    extrinsics = torch.eye(4, device=device).unsqueeze(0).unsqueeze(0) # [B, V, 4, 4]
    
    # Offset cameras slightly
    context_extrinsics = extrinsics.repeat(1, num_context, 1, 1)
    context_extrinsics[0, 0, 2, 3] = -2.0
    context_extrinsics[0, 1, 2, 3] = -1.0
    
    target_extrinsics = extrinsics.repeat(1, num_target, 1, 1)
    target_extrinsics[0, 0, 2, 3] = 0.0
    
    intrinsics = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0)
    context_intrinsics = intrinsics.repeat(1, num_context, 1, 1)
    target_intrinsics = intrinsics.repeat(1, num_target, 1, 1)
    
    batch = {
        "context": {
            "extrinsics": context_extrinsics,
            "intrinsics": context_intrinsics,
            "near": torch.tensor([0.1], device=device).repeat(1, num_context),
            "far": torch.tensor([5.0], device=device).repeat(1, num_context),
        },
        "target": {
            "extrinsics": target_extrinsics,
            "intrinsics": target_intrinsics,
            "near": torch.tensor([0.1], device=device).repeat(1, num_target),
            "far": torch.tensor([5.0], device=device).repeat(1, num_target),
        }
    }
    
    print("Rendering cameras with default scale...")
    # Default scale
    cameras_default = render_cameras(batch, 256, frustum_scale=0.05)
    img_default = hcat(*cameras_default)
    save_image(img_default, "cameras_default.png")
    
    print("Rendering cameras with reduced scale...")
    # Reduced scale
    cameras_small = render_cameras(batch, 256, frustum_scale=0.01)
    img_small = hcat(*cameras_small)
    save_image(img_small, "cameras_small.png")
    
    print("Saved 'cameras_default.png' and 'cameras_small.png'")

if __name__ == "__main__":
    test_render_cameras()
