"""
DepthSplat Gradio Demo.

A Gradio interface for the DepthSplat object-centric 3D Gaussian Splatting model.
Input 5 images of an object with camera poses, and render novel views.
"""

import os
import sys
from pathlib import Path
from functools import partial
from typing import Optional, Tuple, List

# Set environment variables before imports
if 'OMP_NUM_THREADS' not in os.environ:
    os.environ['OMP_NUM_THREADS'] = '16'

import numpy as np
import torch
import gradio as gr
from PIL import Image

# Add parent to path
DEPTHSPLAT_ROOT = Path(__file__).parent.parent
if str(DEPTHSPLAT_ROOT) not in sys.path:
    sys.path.insert(0, str(DEPTHSPLAT_ROOT))

from runner import DepthSplatRunner, create_default_intrinsics
from data_loader import (
    get_random_example,
    save_example_images,
    list_available_examples,
    get_example_by_uuid,
    HARDCODED_UUIDS,
)
from services.gt_detection_service import create_gt_detection_service
from camera_utils import normalize_intrinsics


# Default dataset path
DEFAULT_DATA_DIR = "/mnt/raid0/objaverse/test"

_HEADER_ = '''
# DepthSplat Object-Centric 3D Gaussian Splatting Demo
'''




class GradioRunner:
    """Wrapper for running DepthSplat inference in Gradio."""

    def __init__(self, checkpoint_path: str, config_name: str = "objaverse_white"):
        """Initialize the runner with a checkpoint."""
        self.runner = DepthSplatRunner(
            checkpoint_path=checkpoint_path,
            config_name=config_name,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        self.image_shape = self.runner.image_shape
        self.current_example = None  # Store current loaded example for GT cameras

    def run_with_custom_camera(
        self,
        image_files: List[str],
        azimuth: float,
        elevation: float,
        distance: float,
        enable_video: bool,
        num_video_frames: int,
        cache_dir: str = None,
    ) -> Tuple[str, str, str, str, str, str, str, List[str]]:
        """
        Run inference using custom target camera.

        Args:
            image_files: List of image file paths
            azimuth: Target view azimuth in degrees (0-360)
            elevation: Target view elevation in degrees (-30 to 60)
            distance: Distance factor (0.6-1.4)
            enable_video: Whether to generate 360 videos
            num_video_frames: Number of frames for 360 video
            cache_dir: Directory for outputs

        Returns:
            Tuple of (rendered_image_path, depth_image_path, silhouette_image_path,
                      video_rgb_path, video_depth_path, video_silhouette_path, ply_path,
                      mono_depth_paths)
        """
        torch.cuda.empty_cache()

        if cache_dir is None:
            cache_dir = "/tmp/depthsplat_gradio"

        if image_files is None or len(image_files) == 0:
            raise gr.Error("Please load an example first")

        # Load images
        images = []
        for img_path in image_files:
            if isinstance(img_path, tuple):
                img_path = img_path[0]
            img = Image.open(img_path).convert("RGB")
            img_np = np.array(img)
            images.append(img_np)

        num_views = len(images)
        print(f"Loaded {num_views} images")
        print(f"Target camera: azimuth={azimuth}°, elevation={elevation}°, distance={distance}")

        # Use GT cameras for context if available
        if self.current_example is not None:
            extrinsics = self.current_example['extrinsics']
            intrinsics = self.current_example['intrinsics']
            print(f"Using GT context cameras from example: {self.current_example['key']}")
        else:
            raise gr.Error("No example loaded. Please load an example first.")

        # Create custom target camera from azimuth/elevation/distance
        # Objaverse cameras are on a hemisphere ABOVE the object, looking DOWN at origin
        # Elevation: 60° = near top (looking down), 0° = equator (horizontal), -30° = below equator
        # Azimuth: 0° = +X direction, 90° = +Y direction
        
        # Use radius similar to GT cameras (they are at distance ~1.2-1.5 from origin)
        base_radius = 2.0  # Match normalization target radius (runner.py)
        radius = base_radius * distance
        
        azimuth_rad = np.deg2rad(azimuth)
        elevation_rad = np.deg2rad(elevation)
        
        # Spherical coordinates: elevation is measured from horizontal plane
        # x = r * cos(el) * cos(az)
        # y = r * cos(el) * sin(az)  
        # z = r * sin(el)
        x = radius * np.cos(elevation_rad) * np.cos(azimuth_rad)
        y = radius * np.cos(elevation_rad) * np.sin(azimuth_rad)
        z = radius * np.sin(elevation_rad)
        
        # Camera position
        cam_pos = np.array([x, y, z], dtype=np.float32)
        
        # Look at origin
        target = np.array([0, 0, 0], dtype=np.float32)
        forward = target - cam_pos
        forward = forward / np.linalg.norm(forward)
        
        # World up vector (Z is up)
        world_up = np.array([0, 0, 1], dtype=np.float32)
        
        # Right vector = forward x up
        right = np.cross(forward, world_up)
        if np.linalg.norm(right) < 1e-6:
            # Handle case when looking straight up/down - use Y as temporary up
            world_up = np.array([0, 1, 0], dtype=np.float32)
            right = np.cross(forward, world_up)
        right = right / np.linalg.norm(right)
        
        # Recompute up to be orthogonal: up = right x forward
        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)

        # Build rotation matrix - camera-to-world
        # OpenCV convention: camera +X=right, +Y=down, +Z=forward (optical axis)
        # The columns of R are where camera axes point in world coordinates
        down = -up  # Camera Y axis points down in world
        R = np.stack([right, down, forward], axis=1).astype(np.float32)
        
        # Build 4x4 camera-to-world extrinsics matrix
        target_extrinsic = np.eye(4, dtype=np.float32)
        target_extrinsic[:3, :3] = R
        target_extrinsic[:3, 3] = cam_pos
        target_extrinsics = target_extrinsic[np.newaxis, ...]  # [1, 4, 4]
        
        print(f"Target camera position: x={x:.3f}, y={y:.3f}, z={z:.3f}, radius={radius:.3f}")

        # Run inference (only generate videos if enabled)
        actual_video_frames = num_video_frames if enable_video else 0
        result = self.runner.run_inference(
            images=images,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            target_extrinsics=target_extrinsics,
            target_intrinsics=None,
            output_dir=cache_dir,
            num_video_frames=actual_video_frames,
        )

        return (
            result['result_image_path'],
            result['depth_image_path'],
            result['silhouette_image_path'],
            result['video_rgb_path'],
            result['video_depth_path'],
            result['video_silhouette_path'],
            result['ply_path'],
            result.get('mono_depth_paths', []),
        )

    def load_random_example(self, cache_dir: str = None) -> Tuple[List[str], str]:
        """Load a random example from the dataset."""
        if cache_dir is None:
            cache_dir = "/tmp/depthsplat_examples"

        example = get_random_example(DEFAULT_DATA_DIR, num_context_views=5)
        if example is None:
            raise gr.Error(f"Failed to load example from {DEFAULT_DATA_DIR}")

        self.current_example = example

        # Save images to disk
        example_dir = os.path.join(cache_dir, example['key'])
        image_paths = save_example_images(example, example_dir)

        return image_paths, f"Loaded: {example['key']}"

    def load_selected_uuid(self, uuid: str, cache_dir: str = None) -> Tuple[List[str], str]:
        """Load a specific UUID from the dataset using farthest_point sampling."""
        if cache_dir is None:
            cache_dir = "/tmp/depthsplat_examples"

        if not uuid or uuid == "":
            raise gr.Error("Please select a UUID from the dropdown")

        example = get_example_by_uuid(DEFAULT_DATA_DIR, uuid, num_context_views=5)
        if example is None:
            raise gr.Error(f"Failed to load UUID {uuid} from {DEFAULT_DATA_DIR}")

        self.current_example = example

        # Save images to disk
        example_dir = os.path.join(cache_dir, example['key'])
        image_paths = save_example_images(example, example_dir)

        return image_paths, f"Loaded: {example['key']} (views: {example['selected_indices']})"

    def load_wild_frame(self, frame_id: int, render_dir: str, cache_dir: str = None) -> Tuple[List[str], str, float]:
        """
        Load a specific frame from "in-the-wild" renders.

        Delegates to DepthSplatRunner.load_wild_frame which has the correct
        coordinate system transformations, centering, and scaling.

        Args:
            frame_id: Frame index (0-119)
            render_dir: Path to render directory
            cache_dir: Temp directory

        Returns:
            Tuple of (list of image paths, status string, recommended elevation)
        """
        # Delegate to the fully-functional runner implementation
        image_paths, status = self.runner.load_wild_frame(frame_id, render_dir, cache_dir)

        # Copy the context from runner to GradioRunner
        self.current_example = self.runner.current_example

        # Get the recommended elevation (mean of input cameras)
        mean_elevation = self.current_example.get('mean_elevation', -60.0)

        return image_paths, status, mean_elevation

    def run_flight_tracking(
        self,
        render_dir: str,
        elevation: float,
        distance: float,
        cache_dir: str = None,
    ) -> str:
        """
        Run 360° flight tracking video generation.

        Args:
            render_dir: Path to render directory
            elevation: Camera elevation angle in degrees
            distance: Camera distance factor
            cache_dir: Directory for outputs

        Returns:
            Path to the generated video, or None on failure
        """
        if cache_dir is None:
            cache_dir = "/tmp/depthsplat_flight_tracking"

        try:
            result = self.runner.generate_flight_tracking_video(
                render_dir=render_dir,
                cache_dir=cache_dir,
                start_frame=0,
                end_frame=119,
                elevation=elevation,
                distance=distance,
            )
            return result.get('flight_video_path')
        except Exception as e:
            print(f"Flight tracking error: {e}")
            return None


def create_demo(runner: GradioRunner) -> gr.Blocks:
    """Create the Gradio demo interface."""
    

    with gr.Blocks(
        analytics_enabled=False,
        title='DepthSplat Demo',
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(_HEADER_)

        with gr.Row():
            with gr.Column(scale=1):
                # Example loading section
                gr.Markdown("### Load Example from Dataset")
                
                # UUID dropdown selection
                uuid_dropdown = gr.Dropdown(
                    choices=HARDCODED_UUIDS,
                    label="Select UUID",
                    info="Choose a specific object for consistent testing",
                    value=None,
                )
                load_uuid_btn = gr.Button('Load Selected UUID', variant='primary')
                
                with gr.Row():
                    load_example_btn = gr.Button('Load Random Example', variant='secondary')
                    example_status = gr.Textbox(
                        label="Example Status",
                        value="No example loaded",
                        interactive=False,
                        max_lines=1,
                    )
                
                # In-the-Wild Frame loading
                gr.Markdown("### Load In-the-Wild Frame (Frames 0000-0119)")
                with gr.Row():
                    wild_frame_slider = gr.Slider(
                        minimum=0, 
                        maximum=119, 
                        step=1, 
                        label="Frame ID", 
                        value=0
                    )
                    load_wild_btn = gr.Button('Load Wild Frame', variant='primary')
                
                wild_status = gr.Textbox(
                    label="Wild Status",
                    value="No frame loaded",
                    interactive=False,
                    max_lines=1,
                )

                # Input images
                image_gallery = gr.Gallery(
                    label='Input Images (farthest_point sampled views)',
                    type="filepath",
                    file_types=['image'],
                    show_label=True,
                    columns=5,
                    rows=1,
                    object_fit='contain',
                    height=200,
                )

                # Monocular depth estimations per input view
                gr.Markdown("### Monocular Depth Refinement (Depth Anything)")
                mono_depth_gallery = gr.Gallery(
                    label='Monocular Depth Refinement per Input View (DPT upsampler output)',
                    type="filepath",
                    file_types=['image'],
                    show_label=True,
                    columns=5,
                    rows=1,
                    object_fit='contain',
                    height=200,
                )

                # Config section
                gr.Markdown("### Configuration")
                with gr.Row():
                    enable_video = gr.Checkbox(
                        label="Generate 360° rotation videos",
                        value=True,
                        info="Generate RGB, depth, and silhouette videos (slower)",
                    )
                    num_video_frames = gr.Slider(
                        minimum=30,
                        maximum=120,
                        value=60,
                        step=10,
                        label="Video Frames",
                        visible=True,
                    )

                enable_flight_tracking = gr.Checkbox(
                    label="Enable 360° Flight Tracking",
                    value=False,
                    info="Generate a video of the object moving across all 120 frames with a rotating 360° viewpoint (3°/frame). Takes several minutes.",
                )

                # Camera Control Sliders
                gr.Markdown("### 📷 Novel View Camera Settings")
                gr.Markdown("*Adjust azimuth, elevation, and distance to set the target camera position*")
                
                azimuth_slider = gr.Slider(
                    minimum=0,
                    maximum=360,
                    step=1,
                    value=0,
                    label="Azimuth (degrees)",
                    info="Horizontal rotation around the object (0° = front, 90° = right, 180° = back)",
                )
                elevation_slider = gr.Slider(
                    minimum=-90,
                    maximum=90,
                    step=1,
                    value=30,
                    label="Elevation (degrees)",
                    info="Vertical angle (-90° = below, 0° = horizontal, 90° = above). Wild frames auto-set to input camera elevation.",
                )
                distance_slider = gr.Slider(
                    minimum=0.6,
                    maximum=1.4,
                    step=0.05,
                    value=1.0,
                    label="Distance (factor)",
                    info="Distance from object center (0.6 = close, 1.0 = normal, 1.4 = far)",
                )


                # Run button
                run_btn = gr.Button('🚀 Reconstruct', variant='primary', size='lg')

            with gr.Column(scale=1):
                # Outputs
                gr.Markdown("### Rendered Novel View")
                rendered_image = gr.Image(
                    label='Rendered View',
                    type="filepath",
                    height=250,
                )

                gr.Markdown("### Predicted Depth & Silhouette")
                with gr.Row():
                    depth_image = gr.Image(
                        label='Depth Map',
                        type="filepath",
                        height=200,
                    )
                    silhouette_image = gr.Image(
                        label='Silhouette',
                        type="filepath",
                        height=200,
                    )

                gr.Markdown("### 360 Rotation Videos")
                with gr.Row():
                    video_rgb = gr.Video(
                        label='RGB',
                        autoplay=True,
                        height=180,
                    )
                    video_depth = gr.Video(
                        label='Depth',
                        autoplay=False,
                        height=180,
                    )
                    video_silhouette = gr.Video(
                        label='Silhouette',
                        autoplay=False,
                        height=180,
                    )

                gr.Markdown("### 360° Flight Tracking Video")
                flight_tracking_video = gr.Video(
                    label='Flight Tracking (120 frames, rotating viewpoint)',
                    autoplay=True,
                    height=250,
                )

                gr.Markdown("### 3D Gaussian Splat (PLY)")
                ply_download = gr.File(
                    label='Download PLY file',
                )

        # Initialize demo
        demo.load(fn=None, inputs=None, outputs=None)

        # Connect the load UUID button
        load_uuid_btn.click(
            fn=partial(runner.load_selected_uuid, cache_dir="/tmp/depthsplat_examples"),
            inputs=[uuid_dropdown],
            outputs=[image_gallery, example_status],
        )

        # Connect the load random example button
        load_example_btn.click(
            fn=partial(runner.load_random_example, cache_dir="/tmp/depthsplat_examples"),
            inputs=[],
            outputs=[image_gallery, example_status],
        )

        # Connect the load wild frame button
        # RENDER_DIR = "/home/sandro/thesis/renders/5cams_bird_10m"
        # RENDER_DIR = "/home/sandro/thesis/renders/5cams_drone_10m"
        # RENDER_DIR = "/home/sandro/thesis/renders/5cams_bird_50m"
        RENDER_DIR = "/home/sandro/aeroSplat-4D/renders/5cams_drone_50m"
        # RENDER_DIR = "/home/sandro/thesis/renders/5cams_bird_100m"
        # RENDER_DIR = "/home/sandro/thesis/renders/5cams_drone_100m"

        load_wild_btn.click(
            fn=partial(runner.load_wild_frame, render_dir=RENDER_DIR, cache_dir="/tmp/depthsplat_gradio_wild"),
            inputs=[wild_frame_slider],
            outputs=[image_gallery, wild_status, elevation_slider],  # Also update elevation slider
        )

        # Inference using custom camera from 3D control
        def run_inference_wrapper(image_files, azimuth, elevation, distance, video_enabled, n_frames, flight_enabled):
            cache_dir = "/tmp/depthsplat_gradio"
            print(f"[Gradio] Camera state: azimuth={azimuth}, elevation={elevation}, distance={distance}")

            # Run standard inference
            result = runner.run_with_custom_camera(
                image_files, azimuth, elevation, distance, video_enabled, n_frames, cache_dir
            )

            # Unpack result (8 elements: 7 original + mono_depth_paths)
            (rendered_path, depth_path, silhouette_path, video_rgb, video_depth,
             video_silhouette, ply_path, mono_depth_paths) = result

            # Run flight tracking if enabled
            flight_video = None
            if flight_enabled:
                print("[Gradio] Starting 360° flight tracking video generation...")
                flight_video = runner.run_flight_tracking(
                    render_dir=RENDER_DIR,
                    elevation=elevation,
                    distance=distance,
                    cache_dir="/tmp/depthsplat_flight_tracking",
                )

            return (
                rendered_path,
                depth_path,
                silhouette_path,
                video_rgb,
                video_depth,
                video_silhouette,
                ply_path,
                mono_depth_paths,  # List of paths for gallery
                flight_video,
            )

        # Connect the run button
        run_btn.click(
            fn=run_inference_wrapper,
            inputs=[
                image_gallery,
                azimuth_slider,
                elevation_slider,
                distance_slider,
                enable_video,
                num_video_frames,
                enable_flight_tracking,
            ],
            outputs=[
                rendered_image,
                depth_image,
                silhouette_image,
                video_rgb,
                video_depth,
                video_silhouette,
                ply_download,
                mono_depth_gallery,
                flight_tracking_video,
            ],
            concurrency_id='default_group',
            api_name='run',
        )


    return demo


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="DepthSplat Gradio Demo")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEPTHSPLAT_ROOT / "outputs/objaverse_white/checkpoints/epoch_0-step_65000.ckpt"),
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="objaverse_white",
        help="Name of experiment config (without .yaml extension)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run the server on",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public link",
    )

    args = parser.parse_args()

    # Check checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        print("Please provide a valid checkpoint path with --checkpoint")
        sys.exit(1)

    # Initialize runner
    print("Initializing DepthSplat model...")
    torch.set_float32_matmul_precision("medium")
    runner = GradioRunner(args.checkpoint, args.config)

    # Create and launch demo
    demo = create_demo(runner)
    demo.queue().launch(
        share=args.share,
        server_name="0.0.0.0",
        server_port=args.port,
        ssl_verify=False,
    )


if __name__ == "__main__":
    main()
