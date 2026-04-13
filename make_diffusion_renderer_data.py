#!/usr/bin/env python3
"""
Data generation pipeline for Diffusion Renderer benchmarking.

This script prepares data from the BMD dataset for Diffusion Renderer inference:
- Crops videos using global bbox
- Pads to 704x1280 aspect ratio (height x width) and resizes to 704x1280
- Only saves the first 57 frames
- Saves 3 HDR environment maps per video (1 rotation + 2 static)
  - HDR/EXR files are copied as-is
  - Non-HDR files (e.g., PNG) are converted to EXR (sRGB to linear)
- Skips HDR files that already exist in the output directory
- Saves annotation info for post-processing back to original video dimensions

Usage:
    python make_diffusion_renderer_data.py --input_dir /path/to/bmd_data --output_dir /path/to/output
"""

import os
# Enable OpenEXR support in OpenCV
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import json
import shutil
import argparse
import numpy as np
import cv2
from PIL import Image, ImageOps
import imageio.v2 as imageio
from glob import glob
from tqdm import tqdm

# Constants for Diffusion Renderer
TARGET_HEIGHT = 704
TARGET_WIDTH = 1280
NUM_FRAMES = 57

# Gamma for sRGB conversion (from utils.py)
GAMMA = 2.4


def srgb2linear(srgb):
    """
    Conversion from gamma-corrected sRGB to linear RGB.
    From utils.py.
    """
    srgb = np.clip(srgb, 0, 1)
    linear = np.where(
        srgb <= 0.04045,
        srgb / 12.92,
        ((srgb + 0.055) / 1.055) ** GAMMA
    )
    return linear


def is_hdr_file(filepath):
    """Check if file is HDR/EXR format."""
    ext = os.path.splitext(filepath)[1].lower()
    return ext in [".exr", ".hdr"]


def flip_and_rotate_envmap(img):
    """
    Flip equirectangular environment map along the width axis (horizontal flip),
    then rotate 180 degrees around vertical axis.

    For equirectangular maps, 180° rotation is done by shifting horizontally by half the width.
    """
    # First flip horizontally
    img = np.flip(img, axis=1)
    # Then rotate 180 degrees
    width = img.shape[1]
    img = np.roll(img, shift=width // 2, axis=1)
    return img


def convert_to_exr(input_path, output_path):
    """
    Convert non-HDR image (e.g., PNG) to EXR format.

    Converts from sRGB to linear color space, flips horizontally, and rotates 180 degrees.
    """
    # Load image
    img_bgr = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not load image: {input_path}")

    # Handle grayscale
    if img_bgr.ndim == 2:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)

    # Take only RGB channels (ignore alpha if present)
    img_bgr = img_bgr[..., :3]

    # Convert to float [0, 1]
    img_float = img_bgr.astype(np.float32) / 255.0

    # Convert from sRGB to linear
    img_linear = srgb2linear(img_float)

    # Flip horizontally and rotate 180 degrees
    img_linear = flip_and_rotate_envmap(img_linear)

    # Save as EXR (OpenCV expects BGR order)
    cv2.imwrite(output_path, img_linear.astype(np.float32))

    return output_path


def pad_and_resize(img, target_height, target_width, bg_color=(0, 0, 0)):
    """
    Pad image to match target aspect ratio, then resize to target dimensions.

    Args:
        img: PIL Image or numpy array
        target_height: target height (704)
        target_width: target width (1280)
        bg_color: background color for padding

    Returns:
        PIL Image of size (target_width, target_height)
    """
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)

    # Use ImageOps.pad which pads to maintain aspect ratio then resizes
    out = ImageOps.pad(img, (target_width, target_height), color=bg_color, centering=(0.5, 0.5))
    return out


def load_video_frames(video_path, bbox=None, start_idx=0, end_idx=None):
    """Load video frames with optional bbox cropping."""
    reader = imageio.get_reader(video_path)
    num_frames = int(reader.count_frames())

    if end_idx is None:
        end_idx = num_frames

    frames = []
    for frame_id in range(num_frames):
        if frame_id < start_idx:
            continue
        elif frame_id >= end_idx:
            break
        frame = reader.get_data(frame_id)
        if bbox is not None:
            xmin, ymin, xmax, ymax = bbox
            frame = frame[ymin:ymax, xmin:xmax]
        frames.append(frame)

    reader.close()
    return frames


def save_video_mp4(frames, output_path, fps=30):
    """Save list of frames as mp4 video."""
    if len(frames) == 0:
        return

    # Convert to numpy if needed
    if isinstance(frames[0], Image.Image):
        frames = [np.array(f) for f in frames]

    writer = imageio.get_writer(output_path, fps=fps, codec='libx264',
                                 pixelformat='yuv420p', quality=8)
    for frame in frames:
        writer.append_data(frame)
    writer.close()


def load_masks(mask_dir, bbox, start_idx=0, num_frames=57):
    """Load mask images from directory with bbox cropping."""
    mask_files = sorted(glob(os.path.join(mask_dir, "*.png")))

    masks = []
    for i, mask_file in enumerate(mask_files[start_idx:start_idx + num_frames]):
        mask = np.array(Image.open(mask_file))
        if bbox is not None:
            xmin, ymin, xmax, ymax = bbox
            mask = mask[ymin:ymax, xmin:xmax]
        masks.append(mask)

    return masks


def process_video(video_id, input_dir, output_dir, env_maps_dir,
                  rotation_config, static_config):
    """
    Process a single video for Diffusion Renderer inference.

    Creates one folder per video containing:
    - input.mp4 (cropped & resized video)
    - masks/ (per-frame masks)
    - hdrs/ (3 HDR files: 1 rotation + 2 static)
    - annotation.json

    Args:
        video_id: video identifier (e.g., "001")
        input_dir: path to bmd_data directory
        output_dir: output directory for processed data
        env_maps_dir: path to Comp_env_maps directory
        rotation_config: dict from bmd_multiview_rotation.json
        static_config: dict from bmd_multiview_static.json
    """
    video_path = os.path.join(input_dir, "bmd_out", f"{video_id}.mp4")
    json_path = os.path.join(input_dir, "bmd_out", f"{video_id}.json")
    mask_dir = os.path.join(input_dir, "bmd_msk", video_id)

    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        return

    # Load metadata
    with open(json_path, 'r') as f:
        meta = json.load(f)

    bbox = tuple(meta["global_bbox"])

    # Load video frames (first 57 frames only)
    frames = load_video_frames(video_path, bbox, start_idx=0, end_idx=NUM_FRAMES)
    num_frames = len(frames)

    if num_frames == 0:
        print(f"No frames loaded for {video_id}")
        return

    # Calculate original dimensions after bbox crop
    orig_h, orig_w = frames[0].shape[:2]

    # Pad and resize to 704x1280
    resized_frames = [pad_and_resize(f, TARGET_HEIGHT, TARGET_WIDTH) for f in frames]
    resized_frames = [np.array(f) for f in resized_frames]

    # Load and resize masks
    masks = load_masks(mask_dir, bbox, num_frames=num_frames)
    resized_masks = [np.array(pad_and_resize(Image.fromarray(m), TARGET_HEIGHT, TARGET_WIDTH, bg_color=0))
                     for m in masks]

    # Collect environment maps (1 rotation + 2 static)
    env_maps_info = []

    # Get rotation env map (1)
    if video_id in rotation_config and len(rotation_config[video_id]) > 0:
        for env_map_name in rotation_config[video_id]:
            env_maps_info.append({"name": env_map_name, "mode": "rotation"})

    # Get static env maps (2)
    if video_id in static_config and len(static_config[video_id]) > 0:
        for env_map_name in static_config[video_id]:
            env_maps_info.append({"name": env_map_name, "mode": "static"})

    if not env_maps_info:
        print(f"No environment maps configured for {video_id}")
        return

    # Create output directory for this video
    video_out_dir = os.path.join(output_dir, video_id)
    os.makedirs(video_out_dir, exist_ok=True)

    # Save input video
    input_video_path = os.path.join(video_out_dir, "input.mp4")
    save_video_mp4(resized_frames, input_video_path, fps=30)

    # Save masks
    mask_out_dir = os.path.join(video_out_dir, "masks")
    os.makedirs(mask_out_dir, exist_ok=True)
    for i, mask in enumerate(resized_masks):
        mask_path = os.path.join(mask_out_dir, f"{i:06d}.png")
        Image.fromarray(mask).save(mask_path)

    # Save HDR environment maps (convert non-HDR to EXR if needed)
    hdr_out_dir = os.path.join(video_out_dir, "hdrs")
    os.makedirs(hdr_out_dir, exist_ok=True)

    saved_hdrs = []
    for env_info in env_maps_info:
        env_map_name = env_info["name"]
        mode = env_info["mode"]
        env_map_path = os.path.join(env_maps_dir, env_map_name)

        if not os.path.exists(env_map_path):
            print(f"  Warning: Environment map not found: {env_map_path}")
            continue

        # Create output filename with mode suffix
        env_name_base = os.path.splitext(env_map_name)[0]
        env_ext = os.path.splitext(env_map_name)[1]

        if is_hdr_file(env_map_path):
            # Already HDR/EXR - keep original extension
            out_filename = f"{env_name_base}_{mode}{env_ext}"
            out_path = os.path.join(hdr_out_dir, out_filename)

            # Skip if already exists
            if os.path.exists(out_path):
                saved_hdrs.append({"filename": out_filename, "original": env_map_name, "mode": mode})
                continue

            # Load, flip horizontally and rotate 180 degrees, then save the HDR file
            hdr_img = cv2.imread(env_map_path, cv2.IMREAD_UNCHANGED)
            if hdr_img is not None:
                hdr_img = flip_and_rotate_envmap(hdr_img)
                cv2.imwrite(out_path, hdr_img)
            else:
                print(f"  Warning: Could not load HDR file: {env_map_path}")
                continue
        else:
            # Non-HDR (e.g., PNG) - convert to EXR
            out_filename = f"{env_name_base}_{mode}.exr"
            out_path = os.path.join(hdr_out_dir, out_filename)

            # Skip if already exists
            if os.path.exists(out_path):
                saved_hdrs.append({"filename": out_filename, "original": env_map_name, "mode": mode})
                continue

            # Convert to EXR
            try:
                convert_to_exr(env_map_path, out_path)
            except Exception as e:
                print(f"  Warning: Failed to convert {env_map_name} to EXR: {e}")
                continue

        saved_hdrs.append({"filename": out_filename, "original": env_map_name, "mode": mode})

    # Create annotation info for post-processing
    annotation = {
        "video_id": video_id,
        "original_video_path": video_path,
        "original_bbox": list(bbox),
        "original_size_after_crop": [orig_h, orig_w],
        "target_size": [TARGET_HEIGHT, TARGET_WIDTH],
        "num_frames": num_frames,
        "hdrs": saved_hdrs
    }

    # Save annotation
    annotation_path = os.path.join(video_out_dir, "annotation.json")
    with open(annotation_path, 'w') as f:
        json.dump(annotation, f, indent=2)

    print(f"Processed {video_id}: {num_frames} frames, {len(saved_hdrs)} HDRs")


def main():
    parser = argparse.ArgumentParser(description="Generate Diffusion Renderer data from BMD dataset")
    parser.add_argument("--input_dir", type=str, default="/home/gh466/projects/olat/data/bmd_data",
                        help="Path to bmd_data directory")
    parser.add_argument("--output_dir", type=str, default="/home/gh466/projects/olat/data/diffusion_renderer_input",
                        help="Output directory for processed data")
    parser.add_argument("--env_maps_dir", type=str, default="/home/gh466/projects/olat/data/Comp_env_maps",
                        help="Path to Comp_env_maps directory")
    parser.add_argument("--rotation_json", type=str, default="/home/gh466/projects/olat/data/bmd_multiview_rotation.json",
                        help="Path to bmd_multiview_rotation.json")
    parser.add_argument("--static_json", type=str, default="/home/gh466/projects/olat/data/bmd_multiview_static.json",
                        help="Path to bmd_multiview_static.json")
    parser.add_argument("--video_ids", type=str, nargs="+", default=None,
                        help="Specific video IDs to process (e.g., 001 002). If not specified, process all.")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load configs
    with open(args.rotation_json, 'r') as f:
        rotation_config = json.load(f)

    with open(args.static_json, 'r') as f:
        static_config = json.load(f)

    # Get video IDs to process
    if args.video_ids:
        video_ids = args.video_ids
    else:
        # Get all video IDs from bmd_out
        video_files = glob(os.path.join(args.input_dir, "bmd_out", "*.mp4"))
        video_ids = sorted([os.path.splitext(os.path.basename(f))[0] for f in video_files])

    print(f"Processing {len(video_ids)} videos for Diffusion Renderer...")
    print(f"Target resolution: {TARGET_HEIGHT}x{TARGET_WIDTH}")
    print(f"Number of frames: {NUM_FRAMES}")

    for video_id in tqdm(video_ids, desc="Processing videos"):
        process_video(
            video_id=video_id,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            env_maps_dir=args.env_maps_dir,
            rotation_config=rotation_config,
            static_config=static_config
        )

    print(f"Done! Output saved to: {args.output_dir}")
    print(f"\nOutput structure for each video:")
    print(f"  {{video_id}}/")
    print(f"    ├── input.mp4           (cropped & resized input video, 704x1280)")
    print(f"    ├── masks/              (per-frame masks, 704x1280)")
    print(f"    │   └── {{frame:06d}}.png")
    print(f"    ├── hdrs/               (3 HDR environment maps, non-HDR converted to EXR)")
    print(f"    │   ├── {{env_name}}_rotation.hdr/.exr")
    print(f"    │   ├── {{env_name}}_static.hdr/.exr")
    print(f"    │   └── {{env_name}}_static.hdr/.exr")
    print(f"    └── annotation.json     (metadata for post-processing)")


if __name__ == "__main__":
    main()
