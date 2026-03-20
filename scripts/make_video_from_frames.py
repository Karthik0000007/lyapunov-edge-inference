"""
scripts/make_video_from_frames.py
================================
Create an MP4 from a directory of image frames (PNG/JPG). Useful when
you want to generate a demo input video from dataset frames.

Usage (PowerShell):
    python scripts\make_video_from_frames.py --input dataset/KolektorSDD2/images --output demo/demo_input.mp4 --fps 30

The script sorts filenames lexicographically (so name frames with zero-padding).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import cv2


def sorted_images(path: Path) -> List[Path]:
    exts = (".png", ".jpg", ".jpeg", ".bmp")
    files = [p for p in path.iterdir() if p.suffix.lower() in exts]
    return sorted(files)


def make_video(frames_dir: Path, output: Path, fps: int = 30) -> None:
    imgs = sorted_images(frames_dir)
    if not imgs:
        raise FileNotFoundError(f"No image frames found in {frames_dir}")

    first = cv2.imread(str(imgs[0]))
    if first is None:
        raise RuntimeError(f"Failed to read first image {imgs[0]}")
    h, w = first.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output), fourcc, float(fps), (w, h))
    if not out.isOpened():
        raise RuntimeError(f"Cannot open video writer for {output}")

    for i, p in enumerate(imgs):
        img = cv2.imread(str(p))
        if img is None:
            print(f"Warning: failed to read {p} — skipping", file=sys.stderr)
            continue
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h))
        out.write(img)
        if (i + 1) % 200 == 0:
            print(f"Wrote {i+1}/{len(imgs)} frames")

    out.release()
    print(f"Video written to: {output} ({len(imgs)} frames)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Make MP4 from frames directory")
    parser.add_argument("--input", "-i", required=True, help="Frames directory")
    parser.add_argument("--output", "-o", required=True, help="Output MP4 path")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    args = parser.parse_args()

    frames_dir = Path(args.input)
    if not frames_dir.exists() or not frames_dir.is_dir():
        print(f"Frames dir not found: {frames_dir}", file=sys.stderr)
        sys.exit(2)

    make_video(frames_dir, Path(args.output), fps=args.fps)


if __name__ == "__main__":
    main()
