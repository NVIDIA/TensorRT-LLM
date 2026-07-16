# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared Cosmos3 media helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

import PIL.Image

IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp"})


def pil_to_rgb(value: Any) -> PIL.Image.Image:
    if isinstance(value, str):
        return PIL.Image.open(value).convert("RGB")
    if isinstance(value, PIL.Image.Image):
        return value.convert("RGB")
    raise TypeError(f"Cosmos3 preprocessing expected PIL image or image path, got {type(value)!r}.")


def decode_video_file(path: Path, max_frames: Optional[int] = None) -> List[PIL.Image.Image]:
    from tensorrt_llm.inputs.media_io import decode_video_frames

    return decode_video_frames(path, max_frames=max_frames)


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

    # Classify a single file by content, not by suffix: a decodable still is
    # one conditioning frame; a decodable video is expanded to its frames.
    # Extensions are unreliable — the serve path stores references with no
    # type-suffix at all — so the container decides, not the name.
    from tensorrt_llm.inputs.media_io import is_image_file, is_video_file

    if is_image_file(path):
        return [str(path)]
    if is_video_file(path):
        return decode_video_file(path, max_frames=max_frames)
    raise ValueError(
        f"Cosmos3 reference must be a frame directory, a decodable image, "
        f"or a decodable video; got {path}"
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


def load_reference_video(src: Any):
    """Decode a reference into the framework's ``VideoData`` (all frames).

    Offline entry point: decodes any supported reference into ``VideoData`` for
    ``multi_modal_data["video"]``. It does **not** crop and carries no temporal
    metadata — the Cosmos3 worker crops the first/last conditioning window, and
    the temporal placement is set by ``condition_video_latent_indexes`` +
    ``frame_rate`` (request params), not the reference. The worker reads
    ``.frames`` and VAE-encodes them; it never decodes media itself.
    """
    from tensorrt_llm.inputs.multimodal_data import VideoData

    frames = normalize_video_input(src, max_frames=None)
    return VideoData(frames=[pil_to_rgb(frame) for frame in frames], metadata={})
