#!/usr/bin/env python3
"""
Post-processing pipeline for Diffusion Renderer inference output.

This script takes Diffusion Renderer output (704x1280 mp4 videos with background)
and converts them back to the original video format by:
- Cropping out the object using GT masks (Diffusion Renderer output contains background)
- Reversing the padding/resizing to match original dimensions
- Placing the result back into the original video frame at the bbox location

Input structure (from make_diffusion_renderer_data.py):
  {video_id}/
    ├── input.mp4
    ├── masks/
    ├── hdrs/
    └── annotation.json

Inference output structure (expected):
  {video_id}/
    ├── input_frames/{video_id}/*.jpg
    ├── inverse_output/
    │   ├── {video_id}.basecolor.mp4
    │   ├── {video_id}.depth.mp4
    │   ├── {video_id}.metallic.mp4
    │   ├── {video_id}.normal.mp4
    │   └── {video_id}.roughness.mp4
    └── relight_output/
        └── {hdr_basename}/  (e.g., circus_arena_8k_rotation)
            └── 0000.relit_0000.mp4

Output:
  {video_id}_{hdr_basename}.mp4  (uncropped, placed back in original frame)

Usage:
    python proc_diffusion_renderer_data.py \\
        --inference_dir diffusion_renderer_output \\
        --data_dir diffusion_renderer_input \\
        --output_dir diffusion_renderer_proc_output \\
        --original_videos_dir bmd_data/bmd_out
"""

import os
import json
import argparse
import numpy as np
import cv2
from PIL import Image
import imageio.v2 as imageio
from glob import glob
from tqdm import tqdm


def load_video_frames(video_path, start_idx=0, end_idx=None):
    """Load video frames as numpy arrays."""
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
        frames.append(frame)

    reader.close()
    return frames


def load_masks(mask_dir, num_frames=None):
    """Load mask images from directory."""
    mask_files = sorted(glob(os.path.join(mask_dir, "*.png")))
    if num_frames is not None:
        mask_files = mask_files[:num_frames]

    masks = []
    for mask_file in mask_files:
        mask = np.array(Image.open(mask_file))
        # Normalize to 0-1
        if mask.max() > 1:
            mask = mask.astype(np.float32) / 255.0
        masks.append(mask)

    return masks


def apply_mask_to_frame(frame, mask, bg_frame=None):
    """
    Apply mask to extract foreground from relit frame.

    Args:
        frame: relit frame (H, W, 3) uint8
        mask: binary mask (H, W) float32 in [0, 1]
        bg_frame: optional background frame to composite with

    Returns:
        Masked frame with background if provided, else black background
    """
    if mask.ndim == 2:
        mask = mask[:, :, np.newaxis]

    # Ensure mask is same size as frame
    if mask.shape[:2] != frame.shape[:2]:
        mask = cv2.resize(mask.squeeze(), (frame.shape[1], frame.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
        mask = mask[:, :, np.newaxis]

    mask = mask.astype(np.float32)
    frame = frame.astype(np.float32)

    if bg_frame is not None:
        bg_frame = bg_frame.astype(np.float32)
        result = frame * mask + bg_frame * (1 - mask)
    else:
        # Just apply mask - background becomes black
        result = frame * mask

    return result.astype(np.uint8)


def unpad_frame(frame, original_size, current_size):
    """
    Reverse the padding operation to get back to original aspect ratio.

    Args:
        frame: padded frame (current_h, current_w, 3)
        original_size: (orig_h, orig_w) tuple of original dimensions after bbox crop
        current_size: (current_h, current_w) tuple of current dimensions

    Returns:
        Unpadded and resized frame matching original_size
    """
    orig_h, orig_w = original_size
    current_h, current_w = current_size

    # Calculate the aspect ratios
    orig_aspect = orig_w / orig_h
    current_aspect = current_w / current_h

    if orig_aspect > current_aspect:
        # Original was wider - height was padded
        scaled_w = current_w
        scaled_h = int(current_w / orig_aspect)

        pad_total = current_h - scaled_h
        pad_top = pad_total // 2
        pad_bottom = pad_total - pad_top

        if pad_top > 0 or pad_bottom > 0:
            end_row = current_h - pad_bottom if pad_bottom > 0 else current_h
            frame = frame[pad_top:end_row, :]
    else:
        # Original was taller or square - width was padded
        scaled_h = current_h
        scaled_w = int(current_h * orig_aspect)

        pad_total = current_w - scaled_w
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left

        if pad_left > 0 or pad_right > 0:
            end_col = current_w - pad_right if pad_right > 0 else current_w
            frame = frame[:, pad_left:end_col]

    # Resize to original dimensions
    if frame.shape[:2] != (orig_h, orig_w):
        frame = cv2.resize(frame, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)

    return frame


def images_to_video(imgs, out_path, fps=30):
    """Save list of frames as mp4 video with good quality settings."""
    if len(imgs) == 0:
        return

    writer = imageio.get_writer(
        out_path,
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        ffmpeg_params=[
            '-crf', str(18),
            '-preset', 'medium',
        ],
    )
    try:
        for img in imgs:
            if not isinstance(img, np.ndarray):
                img = np.array(img)
            writer.append_data(img)
    finally:
        writer.close()


def uncrop_frames(relit_frames, original_video_path, bbox):
    """
    Place cropped relit frames back into original frame at bbox location.

    Args:
        relit_frames: list of cropped relit frames
        original_video_path: path to original video
        bbox: [xmin, ymin, xmax, ymax] global bounding box

    Returns:
        List of uncropped frames
    """
    orig_frames = load_video_frames(original_video_path)

    xmin, ymin, xmax, ymax = bbox
    bbox_w = xmax - xmin
    bbox_h = ymax - ymin

    output_frames = []
    num_frames = min(len(relit_frames), len(orig_frames))

    for i in range(num_frames):
        # Start with original frame as background
        out_frame = orig_frames[i].copy()

        # Get relit frame and resize to bbox dimensions if needed
        relit_frame = relit_frames[i]
        relit_h, relit_w = relit_frame.shape[:2]

        if (relit_h, relit_w) != (bbox_h, bbox_w):
            relit_frame = cv2.resize(relit_frame, (bbox_w, bbox_h),
                                     interpolation=cv2.INTER_LANCZOS4)

        # Place relit content into bbox region
        out_frame[ymin:ymax, xmin:xmax] = relit_frame

        output_frames.append(out_frame)

    return output_frames


def process_inference_video(inference_video_path, mask_dir, annotation):
    """
    Process a single inference output video and return processed frames.

    Args:
        inference_video_path: path to inference output video
        mask_dir: path to masks directory
        annotation: annotation dict from annotation.json

    Returns:
        List of processed frames, or None if failed
    """
    if not os.path.exists(inference_video_path):
        print(f"Inference video not found: {inference_video_path}")
        return None

    original_size = tuple(annotation["original_size_after_crop"])
    target_size = tuple(annotation["target_size"])
    num_frames = annotation["num_frames"]

    # Load inference output frames
    relit_frames = load_video_frames(inference_video_path)

    if len(relit_frames) == 0:
        print(f"No frames loaded from {inference_video_path}")
        return None

    # Load masks
    masks = load_masks(mask_dir, num_frames=len(relit_frames))

    if len(masks) != len(relit_frames):
        print(f"Warning: mask count ({len(masks)}) != frame count ({len(relit_frames)})")
        while len(masks) < len(relit_frames):
            masks.append(masks[-1] if masks else np.ones(target_size[::-1]))

    # Process each frame
    processed_frames = []
    for i in range(min(num_frames, len(relit_frames))):
        relit_frame = relit_frames[i]
        mask = masks[i] if i < len(masks) else masks[-1]

        # Apply mask to extract foreground
        masked_frame = apply_mask_to_frame(relit_frame, mask)

        # Unpad to original dimensions
        unpadded_frame = unpad_frame(masked_frame, original_size, target_size)

        processed_frames.append(unpadded_frame)

    return processed_frames


def process_video_folder(video_id, data_dir, inference_dir, output_dir, original_videos_dir):
    """
    Process all inference outputs for a single video folder.

    Args:
        video_id: video identifier
        data_dir: path to prepared data directory (contains annotation.json)
        inference_dir: path to inference output directory
        output_dir: output directory for processed results
        original_videos_dir: path to original videos (bmd_data/bmd_out)
    """
    video_data_dir = os.path.join(data_dir, video_id)
    annotation_path = os.path.join(video_data_dir, "annotation.json")

    if not os.path.exists(annotation_path):
        print(f"Annotation not found: {annotation_path}")
        return

    with open(annotation_path, 'r') as f:
        annotation = json.load(f)

    mask_dir = os.path.join(video_data_dir, "masks")
    hdrs = annotation.get("hdrs", [])
    bbox = tuple(annotation["original_bbox"])

    # Get original video path
    orig_video = os.path.join(original_videos_dir, f"{video_id}.mp4")
    if not os.path.exists(orig_video):
        print(f"Original video not found: {orig_video}")
        return

    # Look for inference outputs
    video_inference_dir = os.path.join(inference_dir, video_id)

    if not os.path.exists(video_inference_dir):
        print(f"Inference directory not found: {video_inference_dir}")
        return

    # Process each HDR's inference output
    for hdr_info in hdrs:
        hdr_filename = hdr_info["filename"]
        hdr_base = os.path.splitext(hdr_filename)[0]

        # New structure: relight_output/{hdr_base}/0000.relit_0000.mp4
        relight_output_dir = os.path.join(video_inference_dir, "relight_output", hdr_base)

        # Try to find inference output for this HDR
        possible_paths = [
            # New structure
            os.path.join(relight_output_dir, "0000.relit_0000.mp4"),
            # Fallback to old structure patterns
            os.path.join(video_inference_dir, f"{hdr_base}.mp4"),
            os.path.join(video_inference_dir, f"{hdr_base}_relit.mp4"),
            os.path.join(video_inference_dir, f"{video_id}_{hdr_base}.mp4"),
            os.path.join(video_inference_dir, f"{video_id}_{hdr_base}_relit.mp4"),
        ]

        inference_video = None
        for path in possible_paths:
            if os.path.exists(path):
                inference_video = path
                break

        if inference_video is None:
            print(f"  Warning: No inference output found for {video_id}/{hdr_filename}")
            print(f"    Searched: {relight_output_dir}")
            continue

        # Process inference video to get masked and unpadded frames
        processed_frames = process_inference_video(
            inference_video_path=inference_video,
            mask_dir=mask_dir,
            annotation=annotation
        )

        if processed_frames is None:
            continue

        # Uncrop: place back into original frame at bbox location
        uncropped_frames = uncrop_frames(processed_frames, orig_video, bbox)

        # Generate output name and save
        output_name = f"{video_id}_{hdr_base}"
        output_path = os.path.join(output_dir, f"{output_name}.mp4")
        images_to_video(uncropped_frames, output_path, fps=30)

        print(f"Processed {output_name}: {len(uncropped_frames)} frames -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Post-process Diffusion Renderer inference output")
    parser.add_argument("--inference_dir", type=str, required=True,
                        help="Directory containing Diffusion Renderer inference outputs")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing the prepared input data (from make_diffusion_renderer_data.py)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for processed results")
    parser.add_argument("--original_videos_dir", type=str, required=True,
                        help="Path to bmd_data/bmd_out containing original videos")
    parser.add_argument("--video_ids", type=str, nargs="+", default=None,
                        help="Specific video IDs to process. If not specified, process all.")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Get video IDs to process
    if args.video_ids:
        video_ids = args.video_ids
    else:
        # Find all video folders in data_dir
        video_ids = sorted([d for d in os.listdir(args.data_dir)
                           if os.path.isdir(os.path.join(args.data_dir, d))])

    print(f"Processing {len(video_ids)} videos...")

    for video_id in tqdm(video_ids, desc="Processing videos"):
        process_video_folder(
            video_id=video_id,
            data_dir=args.data_dir,
            inference_dir=args.inference_dir,
            output_dir=args.output_dir,
            original_videos_dir=args.original_videos_dir
        )

    print(f"Done! Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
