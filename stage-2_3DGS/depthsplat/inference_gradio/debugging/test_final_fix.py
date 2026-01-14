#!/usr/bin/env python3
"""
Final test of the proposed fixes with 360 video generation.
"""

import os
import sys
import numpy as np
from PIL import Image

sys.path.append("/home/sandro/thesis/code/depthsplat")
sys.path.append("/home/sandro/thesis/code/depthsplat/inference_gradio")

from runner import DepthSplatRunner
from proposed_fix import (
    create_training_matched_intrinsics,
    get_input_camera_elevation_range,
    generate_safe_target_camera,
    TRAINING_FX_NORM,
)


def main():
    checkpoint = "/home/sandro/thesis/code/depthsplat/outputs/objaverse_white_small_gauss/checkpoints/epoch_0-step_100000.ckpt"
    config = "objaverse_white_small_gauss"
    render_dir = "/home/sandro/thesis/renders/5cams_10-01-26_bird_10m"
    output_dir = "/home/sandro/thesis/code/depthsplat/inference_gradio/debugging/imgs/final_fix_test"
    os.makedirs(output_dir, exist_ok=True)

    print("Initializing Runner...")
    runner = DepthSplatRunner(checkpoint, config)

    # Load wild frame
    image_paths, _ = runner.load_wild_frame(60, render_dir)
    images = [np.array(Image.open(p).convert("RGB")) for p in image_paths]
    ctx = runner.current_example

    min_el, max_el = get_input_camera_elevation_range(ctx['extrinsics'])
    mean_el = (min_el + max_el) / 2

    print(f"\nInput camera elevation range: [{min_el:.1f}°, {max_el:.1f}°]")
    print(f"Mean elevation: {mean_el:.1f}°")

    # Test 1: Original intrinsics with video
    print("\n" + "=" * 60)
    print("TEST 1: Original Intrinsics (7.5° FOV)")
    print("=" * 60)

    target_ext = generate_safe_target_camera(ctx['extrinsics'], azimuth=45)

    result_orig = runner.run_inference(
        images=images,
        extrinsics=ctx['extrinsics'],
        intrinsics=ctx['intrinsics'],
        target_extrinsics=target_ext[np.newaxis],
        output_dir=f"{output_dir}/original",
        num_video_frames=60,
    )
    print(f"Original render saved with video")

    # Test 2: Fixed intrinsics with video
    print("\n" + "=" * 60)
    print("TEST 2: Training-Matched Intrinsics (50° FOV)")
    print("=" * 60)

    fixed_intrinsics = create_training_matched_intrinsics(len(images))

    result_fixed = runner.run_inference(
        images=images,
        extrinsics=ctx['extrinsics'],
        intrinsics=fixed_intrinsics,
        target_extrinsics=target_ext[np.newaxis],
        output_dir=f"{output_dir}/fixed",
        num_video_frames=60,
    )
    print(f"Fixed render saved with video")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nOutput directory: {output_dir}")
    print(f"\nOriginal (7.5° FOV):")
    print(f"  Render: {output_dir}/original/rendered.png")
    print(f"  Video:  {output_dir}/original/video_rgb.mp4")
    print(f"\nFixed (50° FOV):")
    print(f"  Render: {output_dir}/fixed/rendered.png")
    print(f"  Video:  {output_dir}/fixed/video_rgb.mp4")

    # Quick comparison
    orig_img = np.array(Image.open(f"{output_dir}/original/rendered.png"))
    fixed_img = np.array(Image.open(f"{output_dir}/fixed/rendered.png"))

    comparison = np.hstack([orig_img, fixed_img])
    Image.fromarray(comparison).save(f"{output_dir}/render_comparison.png")

    print(f"\nSide-by-side: {output_dir}/render_comparison.png")


if __name__ == "__main__":
    main()
