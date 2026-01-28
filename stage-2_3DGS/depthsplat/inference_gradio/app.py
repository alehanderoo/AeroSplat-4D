"""
DepthSplat Gradio Demo.

A Gradio interface for the DepthSplat object-centric 3D Gaussian Splatting model.
This is the frontend that uses the inference_backend service for all processing.
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

# Add parent to path for inference_backend import
DEPTHSPLAT_ROOT = Path(__file__).parent.parent
if str(DEPTHSPLAT_ROOT) not in sys.path:
    sys.path.insert(0, str(DEPTHSPLAT_ROOT))

# Import from inference_backend
from inference_backend import (
    InferenceService,
    RenderSettings,
    VideoSettings,
    WildFrameLoader,
    get_random_example,
    get_example_by_uuid,
    save_example_images,
    HARDCODED_UUIDS,
    format_metrics_for_display,
)


# Default paths
DEFAULT_DATA_DIR = "/mnt/raid0/objaverse/test"
DEFAULT_RENDER_DIR = "/home/sandro/aeroSplat-4D/renders/5cams_drone_50m"

_HEADER_ = '''
# DepthSplat Object-Centric 3D Gaussian Splatting Demo
'''


class GradioRunner:
    """Wrapper for running DepthSplat inference in Gradio using the backend service."""

    def __init__(self, checkpoint_path: str, config_name: str = "objaverse_white"):
        """Initialize the runner with a checkpoint."""
        self.service = InferenceService.from_checkpoint(
            checkpoint_path=checkpoint_path,
            config_name=config_name,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        self.image_shape = self.service.image_shape
        self.current_example = None

    def run_with_custom_camera(
        self,
        image_files: List[str],
        azimuth: float,
        elevation: float,
        distance: float,
        enable_video: bool,
        num_video_frames: int,
        cache_dir: str = None,
    ) -> Tuple:
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
            Tuple of output paths and depth analysis data
        """
        torch.cuda.empty_cache()

        if cache_dir is None:
            cache_dir = "/tmp/depthsplat_gradio"

        if image_files is None or len(image_files) == 0:
            raise gr.Error("Please load an example first")

        if self.current_example is None:
            raise gr.Error("No example loaded. Please load an example first.")

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

        # Get camera parameters from loaded example
        extrinsics = self.current_example['extrinsics']
        intrinsics = self.current_example['intrinsics']
        print(f"Using context cameras from example: {self.current_example.get('key', 'unknown')}")

        # Create settings
        render_settings = RenderSettings(
            azimuth=azimuth,
            elevation=elevation,
            distance=distance,
        )
        video_settings = VideoSettings(
            enabled=enable_video,
            num_frames=num_video_frames if enable_video else 0,
            elevation=elevation,
            distance=distance,
        )

        # Run inference using the backend service
        result = self.service.reconstruct(
            images=images,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            render_settings=render_settings,
            video_settings=video_settings,
            output_dir=cache_dir,
        )

        # Store result for access
        self._last_result = result

        # Extract depth analysis paths
        depth_analysis = result.depth_analysis or {}

        return (
            result.rendered_image_path,
            result.depth_image_path,
            result.silhouette_image_path,
            result.video_rgb_path,
            result.video_depth_path,
            result.video_silhouette_path,
            result.ply_path,
            depth_analysis.get('residual_paths', []),
            depth_analysis,
        )

    def load_random_example(self, cache_dir: str = None) -> Tuple[List[str], str]:
        """Load a random example from the dataset."""
        if cache_dir is None:
            cache_dir = "/tmp/depthsplat_examples"

        example = get_random_example(DEFAULT_DATA_DIR, num_context_views=5)
        if example is None:
            raise gr.Error(f"Failed to load example from {DEFAULT_DATA_DIR}")

        self.current_example = example

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

        example_dir = os.path.join(cache_dir, example['key'])
        image_paths = save_example_images(example, example_dir)

        return image_paths, f"Loaded: {example['key']} (views: {example.get('selected_indices', [])})"

    def load_wild_frame(
        self,
        frame_id: int,
        render_dir: str = None,
        cache_dir: str = None,
    ) -> Tuple[List[str], str, float]:
        """Load a specific frame from "in-the-wild" renders."""
        if render_dir is None:
            render_dir = DEFAULT_RENDER_DIR
        if cache_dir is None:
            cache_dir = "/tmp/depthsplat_gradio_wild"

        # Use the WildFrameLoader from backend
        loader = WildFrameLoader(
            render_dir=render_dir,
            use_virtual_cameras=True,
        )

        frame_data = loader.load_frame(frame_id, cache_dir)

        # Store as current example
        self.current_example = {
            'key': f"wild_{frame_id}",
            'extrinsics': frame_data['extrinsics'],
            'intrinsics': frame_data['intrinsics'],
            'render_dir': render_dir,
            'scale_factor': frame_data['scale_factor'],
            'center': frame_data['center'],
            'mean_elevation': frame_data['mean_elevation'],
        }

        return (
            frame_data['image_paths'],
            frame_data['status'],
            frame_data['mean_elevation'],
        )

    def run_flight_tracking(
        self,
        render_dir: str,
        elevation: float,
        distance: float,
        cache_dir: str = None,
    ) -> Optional[str]:
        """
        Run 360° flight tracking video generation.

        This processes all 120 frames, running inference on each.
        """
        if cache_dir is None:
            cache_dir = "/tmp/depthsplat_flight_tracking"

        os.makedirs(cache_dir, exist_ok=True)

        loader = WildFrameLoader(render_dir=render_dir, use_virtual_cameras=True)
        rendered_frames = []

        try:
            for frame_id in range(120):
                print(f"\n[Flight Tracking] Processing frame {frame_id}/119...")

                # Load frame
                frame_data = loader.load_frame(frame_id, cache_dir)
                images = loader.load_images(frame_data['image_paths'])

                # Calculate azimuth for this frame (3° per frame)
                azimuth = frame_id * (360.0 / 120.0)

                # Create settings for this frame
                render_settings = RenderSettings(
                    azimuth=azimuth,
                    elevation=elevation,
                    distance=distance,
                )
                video_settings = VideoSettings(enabled=False)

                # Run reconstruction
                result = self.service.reconstruct(
                    images=images,
                    extrinsics=frame_data['extrinsics'],
                    intrinsics=frame_data['intrinsics'],
                    render_settings=render_settings,
                    video_settings=video_settings,
                    output_dir=os.path.join(cache_dir, f"frame_{frame_id:04d}"),
                )

                # Load rendered frame
                if result.rendered_image_path:
                    frame_img = Image.open(result.rendered_image_path)
                    frame_tensor = torch.from_numpy(np.array(frame_img)).permute(2, 0, 1).float() / 255.0
                    rendered_frames.append(frame_tensor)
                else:
                    rendered_frames.append(torch.zeros(3, 256, 256))

                torch.cuda.empty_cache()

            # Save video
            from src.misc.image_io import save_video as save_video_frames
            video_path = os.path.join(cache_dir, "flight_tracking_360.mp4")
            save_video_frames(rendered_frames, video_path, fps=30)

            return video_path

        except Exception as e:
            print(f"Flight tracking error: {e}")
            import traceback
            traceback.print_exc()
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

                # Monocular depth per view
                gr.Markdown("### Monocular Depth Refinement (Depth Anything)")
                mono_depth_gallery = gr.Gallery(
                    label='Monocular Depth Refinement per Input View',
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
                        info="Generate RGB, depth, and silhouette videos",
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
                    info="Generate rotating viewpoint video across all 120 frames",
                )

                # Camera Control
                gr.Markdown("### Novel View Camera Settings")

                azimuth_slider = gr.Slider(
                    minimum=0,
                    maximum=360,
                    step=1,
                    value=0,
                    label="Azimuth (degrees)",
                    info="Horizontal rotation (0° = front, 90° = right)",
                )
                elevation_slider = gr.Slider(
                    minimum=-90,
                    maximum=90,
                    step=1,
                    value=30,
                    label="Elevation (degrees)",
                    info="Vertical angle (-90° = below, 0° = horizontal, 90° = above)",
                )
                distance_slider = gr.Slider(
                    minimum=0.6,
                    maximum=1.4,
                    step=0.05,
                    value=1.0,
                    label="Distance (factor)",
                    info="Distance from object center",
                )

                run_btn = gr.Button('Reconstruct', variant='primary', size='lg')

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

        # Depth Analysis Section
        gr.Markdown("---")
        gr.Markdown("## Depth Analysis: Monocular vs Multi-View")
        gr.Markdown("""
        Compare depth estimation methods:
        - **Standalone DA V2**: Pure monocular depth from Depth Anything V2
        - **Coarse MV Depth**: Multi-view stereo from cost volume matching
        - **DPT Residual**: Learned refinement from monocular features
        - **Final Fused**: Combined depth used for 3D reconstruction
        - **Ground Truth**: Available for Isaac Sim wild frames
        """)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Standalone Depth Anything V2")
                standalone_da_gallery = gr.Gallery(
                    label='Monocular depth',
                    type="filepath",
                    columns=5,
                    rows=1,
                    object_fit='contain',
                    height=180,
                )

            with gr.Column(scale=1):
                gr.Markdown("### Coarse MV Depth (Cost Volume)")
                coarse_mv_gallery = gr.Gallery(
                    label='Multi-view stereo result',
                    type="filepath",
                    columns=5,
                    rows=1,
                    object_fit='contain',
                    height=180,
                )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### DPT Residual (Monocular Refinement)")
                residual_gallery = gr.Gallery(
                    label='Learned depth correction',
                    type="filepath",
                    columns=5,
                    rows=1,
                    object_fit='contain',
                    height=180,
                )

            with gr.Column(scale=1):
                gr.Markdown("### Final Fused Depth")
                final_fused_gallery = gr.Gallery(
                    label='Coarse + Residual (used for 3D)',
                    type="filepath",
                    columns=5,
                    rows=1,
                    object_fit='contain',
                    height=180,
                )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Ground Truth Depth (Isaac Sim)")
                gt_depth_gallery = gr.Gallery(
                    label='GT depth (wild frames only)',
                    type="filepath",
                    columns=5,
                    rows=1,
                    object_fit='contain',
                    height=180,
                )

            with gr.Column(scale=1):
                gr.Markdown("### Depth Quality Metrics")
                metrics_display = gr.JSON(
                    label='Metrics vs GT',
                    value={},
                )

        # Event handlers
        demo.load(fn=None, inputs=None, outputs=None)

        load_uuid_btn.click(
            fn=partial(runner.load_selected_uuid, cache_dir="/tmp/depthsplat_examples"),
            inputs=[uuid_dropdown],
            outputs=[image_gallery, example_status],
        )

        load_example_btn.click(
            fn=partial(runner.load_random_example, cache_dir="/tmp/depthsplat_examples"),
            inputs=[],
            outputs=[image_gallery, example_status],
        )

        load_wild_btn.click(
            fn=partial(runner.load_wild_frame, render_dir=DEFAULT_RENDER_DIR, cache_dir="/tmp/depthsplat_gradio_wild"),
            inputs=[wild_frame_slider],
            outputs=[image_gallery, wild_status, elevation_slider],
        )

        def run_inference_wrapper(image_files, azimuth, elevation, distance, video_enabled, n_frames, flight_enabled):
            cache_dir = "/tmp/depthsplat_gradio"
            print(f"[Gradio] Camera: azimuth={azimuth}, elevation={elevation}, distance={distance}")

            result = runner.run_with_custom_camera(
                image_files, azimuth, elevation, distance, video_enabled, n_frames, cache_dir
            )

            (rendered_path, depth_path, silhouette_path, video_rgb, video_depth,
             video_silhouette, ply_path, mono_depth_paths, depth_analysis) = result

            # Run flight tracking if enabled
            flight_video = None
            if flight_enabled:
                print("[Gradio] Starting 360° flight tracking...")
                flight_video = runner.run_flight_tracking(
                    render_dir=DEFAULT_RENDER_DIR,
                    elevation=elevation,
                    distance=distance,
                    cache_dir="/tmp/depthsplat_flight_tracking",
                )

            # Format metrics
            raw_metrics = depth_analysis.get('metrics', {}) if depth_analysis else {}
            formatted_metrics = format_metrics_for_display(raw_metrics) if raw_metrics else {}

            return (
                rendered_path,
                depth_path,
                silhouette_path,
                video_rgb,
                video_depth,
                video_silhouette,
                ply_path,
                mono_depth_paths,
                flight_video,
                depth_analysis.get('standalone_da_paths', []) if depth_analysis else [],
                depth_analysis.get('coarse_mv_paths', []) if depth_analysis else [],
                depth_analysis.get('residual_paths', []) if depth_analysis else [],
                depth_analysis.get('final_fused_paths', []) if depth_analysis else [],
                depth_analysis.get('gt_paths', []) if depth_analysis else [],
                formatted_metrics,
            )

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
                standalone_da_gallery,
                coarse_mv_gallery,
                residual_gallery,
                final_fused_gallery,
                gt_depth_gallery,
                metrics_display,
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

    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        print("Please provide a valid checkpoint path with --checkpoint")
        sys.exit(1)

    print("Initializing DepthSplat model...")
    torch.set_float32_matmul_precision("medium")
    runner = GradioRunner(args.checkpoint, args.config)

    demo = create_demo(runner)
    demo.queue().launch(
        share=args.share,
        server_name="0.0.0.0",
        server_port=args.port,
        ssl_verify=False,
    )


if __name__ == "__main__":
    main()
