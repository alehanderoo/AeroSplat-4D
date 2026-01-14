
import zipfile
import numpy as np
import io
import os
import tempfile
import glob
import sys
import shutil

# Try importing cv2
try:
    os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
    import cv2
except ImportError:
    print("Error: opencv-python is not installed. Please install it to verify EXR content.")
    sys.exit(1)

def verify_zip(zip_path):
    print(f"Inspecting {zip_path}...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(temp_dir)
            
        # Files expected: {base_name}_rgb_{frame}.png etc or canonical 000.png
        # We know we renamed them to 000.png etc in render_objaverse1.py
        
        # Files expected: 000.png, 000_depth.exr, 000_normal.exr, 000_mask.png
        # Find the folder containing 000.png
        base_dir = temp_dir
        found_base = False
        for root, dirs, files in os.walk(temp_dir):
            if "000.png" in files:
                base_dir = root
                found_base = True
                break
        
        if not found_base:
             print("Could not find 000.png in extracted zip.")
             # Fallback to temp_dir, loop will fail on existence checks
        
        files = {
            "RGB": os.path.join(base_dir, "000.png"),
            "Depth": os.path.join(base_dir, "000_depth.exr"),
            "Normals": os.path.join(base_dir, "000_normal.exr"),
            "Mask": os.path.join(base_dir, "000_mask.png")
        }

        images = {}
        mask_img = None

        for name, path in files.items():
            if not os.path.exists(path):
                print(f"File not found: {os.path.basename(path)}")
                continue
                
            if path.endswith('.exr'):
                # Load EXR files 'unchanged' to preserve float data
                img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            else:
                img = cv2.imread(path)
                if name == "RGB":
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                elif name == "Mask":
                    # Ensure mask is grayscale
                    if len(img.shape) == 3:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    mask_img = img # Save for depth check
            
            images[name] = img
            if img is not None:
                 print(f"{name}: shape={img.shape}, dtype={img.dtype}, min={img.min()}, max={img.max()}, mean={img.mean()}")

        # Create subplots
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        ax_flat = axes.flatten()

        mask_img = images.get("Mask")

        for i, (name, img) in enumerate(images.items()):
            if img is None:
                continue
                
            ax = ax_flat[i]
            ax.set_title(name)
            
            if name == "Normals":
                vis_img = img.copy()
                
                # Handle 2-channel normals (common in some EXR outputs)
                if vis_img.ndim == 3 and vis_img.shape[2] == 2:
                    # Reconstruct Z assuming unit length normals: z = sqrt(1 - x^2 - y^2)
                    x = vis_img[:,:,0]
                    y = vis_img[:,:,1]
                    z_sq = 1.0 - x**2 - y**2
                    # Clip to 0 to avoid NaNs from floating point errors
                    z = np.sqrt(np.maximum(z_sq, 0))
                    vis_img = np.dstack((x, y, z))
                
                # Normalize for display if needed
                # Assuming normals are in [-1, 1], map to [0, 1] for visualization
                if vis_img.min() < 0:
                    vis_img = (vis_img + 1) / 2
                
                vis_img = np.clip(vis_img, 0, 1)
                ax.imshow(vis_img)

            elif name == "Depth":
                # Force extraction of the first channel for depth
                if img.ndim == 3:
                     img_view = img[:,:,0]
                else:
                     img_view = img

                img_valid = img_view.copy()
                
                # Apply mask if available to filter out background and set scale
                vmin, vmax = None, None
                if mask_img is not None:
                    # Use a stricter threshold for the mask to avoid anti-aliasing artifacts at edges
                    # which might include background depth values
                    valid_mask = mask_img > 127
                    img_valid = np.where(valid_mask, img_valid, np.nan)
                    
                    # Compute vmin/vmax from valid pixels only
                    # Use percentiles to be robust against any remaining outliers
                    valid_pixels = img_valid[~np.isnan(img_valid)]
                    if valid_pixels.size > 0:
                        vmin = np.nanpercentile(valid_pixels, 1)
                        vmax = np.nanpercentile(valid_pixels, 99)
                        print(f"  Valid Depth Stats (masked): min={valid_pixels.min()}, max={valid_pixels.max()}, mean={valid_pixels.mean()}")
                else:
                    # Fallback to infinity check if no mask
                    is_inf = np.isinf(img_view)
                    if is_inf.any():
                         img_valid[is_inf] = np.nan
                         valid_pixels = img_valid[~np.isnan(img_valid)]
                         if valid_pixels.size > 0:
                             vmin = np.nanpercentile(valid_pixels, 1)
                             vmax = np.nanpercentile(valid_pixels, 99)

                im = ax.imshow(img_valid, cmap='inferno', vmin=vmin, vmax=vmax)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            elif name == "Mask":
                ax.imshow(img, cmap='gray')
            else:
                ax.imshow(img)
            
            ax.axis('off')

        plt.tight_layout()
        output_plot = "verification_plot.png"
        plt.savefig(output_plot)
        print(f"Verification plot saved to {output_plot} in current directory.")


if __name__ == "__main__":
    # Find the latest zip in the debug output directory
    debug_renders_dir = "scripts/render_blender/debug/renders"
    if len(sys.argv) > 1:
        zip_path = sys.argv[1]
    else:
        # Default to finding one
        zips = glob.glob(os.path.join(debug_renders_dir, "*.zip"))
        if not zips:
            print(f"No zip files found in {debug_renders_dir}")
            sys.exit(1)
        # sort by mod time
        zips.sort(key=os.path.getmtime, reverse=True)
        zip_path = zips[0]

    verify_zip(zip_path)
