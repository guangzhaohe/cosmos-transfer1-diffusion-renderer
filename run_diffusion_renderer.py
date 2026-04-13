#!/usr/bin/env python3
"""
Run Diffusion Renderer on all prepared benchmark data.

This script:
1. Extracts frames from input videos
2. Runs inverse rendering once per video to get G-buffers
3. Runs forward rendering (relighting) 3 times per video:
   - 1 time with rotation HDR (using --rotate_light flag)
   - 2 times with static HDRs

Usage:
    # Activate conda environment first
    conda activate cosmos-predict1

    # Run the script on all videos
    python run_diffusion_renderer.py --input_dir diffusion_renderer_input --output_dir diffusion_renderer_output

    # Run the script on specific videos (e.g., 001, 002, 003)
    python run_diffusion_renderer.py --input_dir diffusion_renderer_input --output_dir diffusion_renderer_output --video_ids 001 002 003

Prerequisites:
    - Activate cosmos-predict1 conda environment
    - Download model checkpoints to cosmos-transfer1-diffusion-renderer/checkpoints/
"""

import os
import sys
import json
import argparse
import subprocess
import shutil
from glob import glob
from pathlib import Path

# Path to diffusion renderer
DIFFUSION_RENDERER_DIR = "/home/gh466/projects/olat/cosmos-transfer1-diffusion-renderer"

# Default checkpoint directory
DEFAULT_CHECKPOINT_DIR = os.path.join(DIFFUSION_RENDERER_DIR, "checkpoints")


def run_command(cmd, cwd=None, env=None, capture_output=False):
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    if capture_output:
        result = subprocess.run(cmd, cwd=cwd, env=merged_env, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"STDERR: {result.stderr}")
        return result.returncode == 0
    else:
        result = subprocess.run(cmd, cwd=cwd, env=merged_env)
        return result.returncode == 0


def extract_frames_ffmpeg(video_path, output_dir, video_id, num_frames=57):
    """Extract frames from video using ffmpeg."""
    frames_subdir = os.path.join(output_dir, video_id)
    os.makedirs(frames_subdir, exist_ok=True)

    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vframes", str(num_frames),
        "-q:v", "2",
        os.path.join(frames_subdir, "%06d.jpg")
    ]

    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0


def verify_dataset(frames_dir, num_frames=57):
    """Verify that the dataset can be loaded correctly."""
    import sys
    sys.path.insert(0, DIFFUSION_RENDERER_DIR)

    from cosmos_predict1.diffusion.inference.diffusion_renderer_utils.inference_utils import (
        find_images_recursive, group_images_into_videos, split_list_with_overlap
    )

    # Find images
    all_images = find_images_recursive(frames_dir)
    print(f"  Found {len(all_images)} images in {frames_dir}")
    if all_images:
        print(f"  Sample paths: {all_images[:3]}")

    # Group by folder
    video_groups = group_images_into_videos(all_images, image_group_mode="folder")
    print(f"  Grouped into {len(video_groups)} videos")

    if not video_groups:
        return False

    # Check chunking
    for i, group in enumerate(video_groups):
        chunks = split_list_with_overlap(group, num_frames, 0, chunk_mode="first")
        print(f"  Video {i}: {len(group)} frames -> {len(chunks)} chunks")

    return len(video_groups) > 0


def run_inverse_renderer(frames_dir, output_dir, checkpoint_dir, num_frames=57,
                         offload=False):
    """Run inverse rendering to get G-buffers."""
    # First verify the dataset can be loaded
    print(f"  Verifying dataset at {frames_dir}...")
    if not verify_dataset(frames_dir, num_frames):
        print(f"  ERROR: Dataset verification failed - no images found!")
        return False

    env = {
        "CUDA_HOME": os.environ.get("CONDA_PREFIX", ""),
        "PYTHONPATH": DIFFUSION_RENDERER_DIR,
    }

    cmd = [
        "python", "cosmos_predict1/diffusion/inference/inference_inverse_renderer.py",
        "--checkpoint_dir", checkpoint_dir,
        "--diffusion_transformer_dir", "Diffusion_Renderer_Inverse_Cosmos_7B",
        "--dataset_path", frames_dir,
        "--num_video_frames", str(num_frames),
        "--group_mode", "folder",
        "--video_save_folder", output_dir,
    ]

    if offload:
        cmd.extend(["--offload_diffusion_transformer", "--offload_tokenizer"])

    return run_command(cmd, cwd=DIFFUSION_RENDERER_DIR, env=env)


def run_forward_renderer_direct(gbuffer_dir, output_dir, hdr_path, checkpoint_dir,
                                num_frames=57, rotate_light=False, offload=False):
    """Run forward rendering (relighting) with a custom HDR by patching ENV_LIGHT_PATH_LIST."""
    env = {
        "CUDA_HOME": os.environ.get("CONDA_PREFIX", ""),
        "PYTHONPATH": DIFFUSION_RENDERER_DIR,
    }

    # Escape backslashes in paths for Windows compatibility
    hdr_path_escaped = hdr_path.replace("\\", "\\\\")
    gbuffer_dir_escaped = gbuffer_dir.replace("\\", "\\\\")
    output_dir_escaped = output_dir.replace("\\", "\\\\")
    checkpoint_dir_escaped = checkpoint_dir.replace("\\", "\\\\")

    # Create a temporary Python script that patches ENV_LIGHT_PATH_LIST and runs forward renderer
    patch_script = f'''#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, "{DIFFUSION_RENDERER_DIR}")
os.chdir("{DIFFUSION_RENDERER_DIR}")

# Patch the ENV_LIGHT_PATH_LIST before running
import cosmos_predict1.diffusion.inference.inference_forward_renderer as fwd_module
fwd_module.ENV_LIGHT_PATH_LIST = ["{hdr_path_escaped}"]

# Build command line args
sys.argv = [
    "inference_forward_renderer.py",
    "--checkpoint_dir", "{checkpoint_dir_escaped}",
    "--diffusion_transformer_dir", "Diffusion_Renderer_Forward_Cosmos_7B",
    "--dataset_path", "{gbuffer_dir_escaped}",
    "--num_video_frames", "{num_frames}",
    "--envlight_ind", "0",
    "--use_custom_envmap", "True",
    "--video_save_folder", "{output_dir_escaped}",
'''

    if rotate_light:
        patch_script += '''    "--rotate_light", "True",
'''

    if offload:
        patch_script += '''    "--offload_diffusion_transformer",
    "--offload_tokenizer",
'''

    patch_script += ''']

# Parse arguments and run
args = fwd_module.parse_arguments()
fwd_module.demo(args)
'''

    # Write to temp file
    os.makedirs(output_dir, exist_ok=True)
    tmp_script = os.path.join(output_dir, "_tmp_forward_renderer.py")
    with open(tmp_script, 'w') as f:
        f.write(patch_script)

    cmd = ["python", tmp_script]
    success = run_command(cmd, cwd=DIFFUSION_RENDERER_DIR, env=env)

    # Cleanup
    if os.path.exists(tmp_script):
        os.remove(tmp_script)

    return success


def process_video(video_id, input_dir, output_dir, checkpoint_dir, offload=False,
                  skip_inverse=False, skip_forward=False):
    """Process a single video through inverse and forward rendering."""
    video_data_dir = os.path.join(input_dir, video_id)
    annotation_path = os.path.join(video_data_dir, "annotation.json")

    if not os.path.exists(annotation_path):
        print(f"Annotation not found for {video_id}, skipping...")
        return False

    with open(annotation_path, 'r') as f:
        annotation = json.load(f)

    num_frames = annotation["num_frames"]
    hdrs = annotation.get("hdrs", [])

    if not hdrs:
        print(f"No HDRs found for {video_id}, skipping...")
        return False

    # Create output directory for this video
    video_output_dir = os.path.join(output_dir, video_id)
    os.makedirs(video_output_dir, exist_ok=True)

    # Step 1: Extract frames from input video
    input_video_path = os.path.join(video_data_dir, "input.mp4")
    frames_dir = os.path.join(video_output_dir, "input_frames")

    if not os.path.exists(input_video_path):
        print(f"Input video not found for {video_id}")
        return False

    # Check if frames already extracted
    frames_subdir = os.path.join(frames_dir, video_id)
    if not os.path.exists(frames_subdir) or len(glob(os.path.join(frames_subdir, "*.jpg"))) < num_frames:
        print(f"\n[{video_id}] Extracting frames from input video...")
        if not extract_frames_ffmpeg(input_video_path, frames_dir, video_id, num_frames):
            print(f"Failed to extract frames for {video_id}")
            return False
    else:
        print(f"\n[{video_id}] Frames already extracted, skipping extraction...")

    # Step 2: Run inverse rendering (once per video)
    inverse_output_dir = os.path.join(video_output_dir, "inverse_output")
    # G-buffer frames are saved under gbuffer_frames/{clip_name}/ by the inverse renderer
    gbuffer_dir = os.path.join(inverse_output_dir, "gbuffer_frames", video_id)

    if skip_inverse and os.path.exists(gbuffer_dir):
        print(f"\n[{video_id}] Skipping inverse rendering (already exists)...")
    else:
        print(f"\n[{video_id}] Running inverse rendering...")
        success = run_inverse_renderer(
            frames_dir=frames_dir,
            output_dir=inverse_output_dir,
            checkpoint_dir=checkpoint_dir,
            num_frames=num_frames,
            offload=offload
        )

        if not success:
            print(f"Inverse rendering failed for {video_id}")
            return False

    # Verify G-buffer output exists
    if not os.path.exists(gbuffer_dir):
        print(f"G-buffer output not found at {gbuffer_dir}")
        return False

    if skip_forward:
        print(f"\n[{video_id}] Skipping forward rendering...")
        return True

    # Step 3: Run forward rendering for each HDR (3 times total)
    print(f"\n[{video_id}] Running forward rendering for {len(hdrs)} HDRs...")

    all_success = True
    for hdr_info in hdrs:
        hdr_filename = hdr_info["filename"]
        hdr_mode = hdr_info["mode"]
        hdr_path = os.path.join(video_data_dir, "hdrs", hdr_filename)

        if not os.path.exists(hdr_path):
            print(f"  HDR not found: {hdr_path}, skipping...")
            continue

        # Determine if we need rotating light
        rotate_light = (hdr_mode == "rotation")

        hdr_base = os.path.splitext(hdr_filename)[0]
        relight_output_dir = os.path.join(video_output_dir, "relight_output", hdr_base)

        # Check if already processed
        existing_outputs = glob(os.path.join(relight_output_dir, "*.mp4"))
        if existing_outputs:
            print(f"\n[{video_id}] Skipping {hdr_filename} (already processed)...")
            continue

        print(f"\n[{video_id}] Relighting with {hdr_filename} (rotate_light={rotate_light})...")

        success = run_forward_renderer_direct(
            gbuffer_dir=gbuffer_dir,
            output_dir=relight_output_dir,
            hdr_path=os.path.abspath(hdr_path),
            checkpoint_dir=checkpoint_dir,
            num_frames=num_frames,
            rotate_light=rotate_light,
            offload=offload
        )

        if not success:
            print(f"  Forward rendering failed for {hdr_filename}")
            all_success = False

    print(f"\n[{video_id}] Completed!")
    return all_success


def main():
    parser = argparse.ArgumentParser(description="Run Diffusion Renderer on benchmark data")
    parser.add_argument("--input_dir", type=str,
                        default="/home/gh466/projects/olat/data/diffusion_renderer_input",
                        help="Input directory from make_diffusion_renderer_data.py")
    parser.add_argument("--output_dir", type=str,
                        default="/home/gh466/projects/olat/data/diffusion_renderer_output",
                        help="Output directory for inference results")
    parser.add_argument("--checkpoint_dir", type=str,
                        default=DEFAULT_CHECKPOINT_DIR,
                        help="Path to model checkpoints")
    parser.add_argument("--video_ids", type=str, nargs="+", default=None,
                        help="Specific video IDs to process (e.g., 001 002 003). If not specified, process all videos.")
    parser.add_argument("--offload", action="store_true",
                        help="Offload models to CPU to reduce GPU memory (slower but works on smaller GPUs)")
    parser.add_argument("--skip_inverse", action="store_true",
                        help="Skip inverse rendering if G-buffers already exist")
    parser.add_argument("--skip_forward", action="store_true",
                        help="Only run inverse rendering, skip forward rendering")
    parser.add_argument("--only_forward", action="store_true",
                        help="Only run forward rendering (assumes G-buffers exist)")

    args = parser.parse_args()

    # Validate paths
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory not found: {args.input_dir}")
        sys.exit(1)

    if not os.path.exists(args.checkpoint_dir):
        print(f"Error: Checkpoint directory not found: {args.checkpoint_dir}")
        print("Please download the model weights first.")
        sys.exit(1)

    # Get video IDs to process
    if args.video_ids:
        # Validate that specified video IDs exist
        all_available_videos = [d for d in os.listdir(args.input_dir)
                               if os.path.isdir(os.path.join(args.input_dir, d))]
        invalid_ids = [vid for vid in args.video_ids if vid not in all_available_videos]
        if invalid_ids:
            print(f"Error: The following video IDs were not found in {args.input_dir}:")
            for vid in invalid_ids:
                print(f"  - {vid}")
            print(f"\nAvailable video IDs: {sorted(all_available_videos)}")
            sys.exit(1)
        video_ids = args.video_ids
        print(f"Processing {len(video_ids)} specified video(s): {', '.join(video_ids)}")
    else:
        video_ids = sorted([d for d in os.listdir(args.input_dir)
                           if os.path.isdir(os.path.join(args.input_dir, d))])
        print(f"Found {len(video_ids)} videos to process (processing all)")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Checkpoint directory: {args.checkpoint_dir}")
    print(f"Offload mode: {args.offload}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Process each video
    successful = 0
    failed = 0

    for i, video_id in enumerate(video_ids):
        print(f"\n{'#'*60}")
        print(f"Processing video {i+1}/{len(video_ids)}: {video_id}")
        print(f"{'#'*60}")

        try:
            success = process_video(
                video_id=video_id,
                input_dir=args.input_dir,
                output_dir=args.output_dir,
                checkpoint_dir=args.checkpoint_dir,
                offload=args.offload,
                skip_inverse=args.skip_inverse or args.only_forward,
                skip_forward=args.skip_forward
            )

            if success:
                successful += 1
            else:
                failed += 1
        except Exception as e:
            print(f"Error processing {video_id}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total videos: {len(video_ids)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"\nOutput saved to: {args.output_dir}")
    print(f"\nOutput structure:")
    print(f"  {{video_id}}/")
    print(f"    ├── input_frames/       (extracted video frames)")
    print(f"    ├── inverse_output/     (G-buffers from inverse rendering)")
    print(f"    │   └── gbuffer_frames/")
    print(f"    └── relight_output/     (relit videos)")
    print(f"        ├── {{hdr_name}}_rotation/")
    print(f"        ├── {{hdr_name}}_static/")
    print(f"        └── {{hdr_name}}_static/")


if __name__ == "__main__":
    main()
