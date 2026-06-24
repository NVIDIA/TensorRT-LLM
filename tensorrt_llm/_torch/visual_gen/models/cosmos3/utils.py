# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared Cosmos3 media helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

import PIL.Image

IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp"})
VIDEO_EXTENSIONS = frozenset({".mp4", ".avi"})


def pil_to_rgb(value: Any) -> PIL.Image.Image:
    if isinstance(value, str):
        return PIL.Image.open(value).convert("RGB")
    if isinstance(value, PIL.Image.Image):
        return value.convert("RGB")
    raise TypeError(
        f"Cosmos3 preprocessing expected PIL image or image path, got {type(value)!r}."
    )


def decode_video_file(path: Path, max_frames: Optional[int] = None) -> List[PIL.Image.Image]:
    import torchvision.io as io

    frames, _, _ = io.read_video(str(path), pts_unit="sec")
    if frames.numel() == 0:
        raise ValueError(f"Cosmos3 video file contains no frames: {path}")
    if max_frames is not None:
        frames = frames[:max_frames]
    return [PIL.Image.fromarray(frames[i].numpy()) for i in range(frames.shape[0])]


def normalize_video_input_path(path: Path, max_frames: Optional[int] = None) -> List[Any]:
    if not path.exists():
        raise ValueError(f"Cosmos3 video path does not exist: {path}")
    if path.is_dir():
        frames = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        if not frames:
            raise ValueError(f"No image frames found in Cosmos3 video directory: {path}")
        frame_paths = [str(p) for p in frames]
        if max_frames is not None:
            frame_paths = frame_paths[:max_frames]
        return frame_paths

    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return [str(path)]
    if suffix in VIDEO_EXTENSIONS:
        return decode_video_file(path, max_frames=max_frames)
    raise ValueError(
        "Cosmos3 video path must be a frame directory, an image file "
        f"{sorted(IMAGE_EXTENSIONS)}, or a video file "
        f"{sorted(VIDEO_EXTENSIONS)}; got {path}"
    )


def normalize_video_input(video: Any, max_frames: Optional[int] = None) -> List[Any]:
    """Normalize video input to a frame list.

    Accepts a list of PIL images / paths, a single image or video file path,
    or a directory of frame images (sorted lexicographically).
    """
    if video is None:
        return []
    if isinstance(video, list):
        if not video:
            raise ValueError("Cosmos3 video input must contain at least one frame.")
        if max_frames is not None:
            return video[:max_frames]
        return video
    if isinstance(video, (str, Path)):
        return normalize_video_input_path(Path(video), max_frames=max_frames)
    return [video]
