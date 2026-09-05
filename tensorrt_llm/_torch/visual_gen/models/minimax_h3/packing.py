# Copyright 2026 The MiniMax and HuggingFace Teams. All rights reserved.

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Packed-sequence geometry and conditioning helpers for MiniMax-H3.

The FL2VA checkpoint runs one full-attention sequence laid out as
``[text | keyframe conditions | target audio | target video]``.  Video and
audio coordinates share a 40-unit-per-second rotary clock, so the float64
coordinate construction here is part of the checkpoint contract.

This module is adapted from the Apache-2.0 MiniMax-H3 Diffusers implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

try:
    from PIL import Image
except ModuleNotFoundError:  # pragma: no cover - depends on deployment image
    Image = None  # type: ignore[assignment]

MINIMAX_H3_VIDEO_TAG = 0
MINIMAX_H3_TEXT_TAG = 1
MINIMAX_H3_AUDIO_TAG = 2

MINIMAX_H3_FPS = 24
MINIMAX_H3_SHORT_EDGE = 768
MINIMAX_H3_MAX_PIXELS = 768 * 1344
MINIMAX_H3_CANVAS_MULTIPLE = 32
MINIMAX_H3_MIN_ASPECT_RATIO = 1 / 4
MINIMAX_H3_MAX_ASPECT_RATIO = 4
MINIMAX_H3_MIN_DURATION = 5.0
MINIMAX_H3_MAX_DURATION = 15.0

MINIMAX_H3_FRAMES_PER_CHUNK = 17
MINIMAX_H3_LATENTS_PER_CHUNK = 5

MINIMAX_H3_PIXEL_MEAN = (0.485, 0.456, 0.406)
MINIMAX_H3_PIXEL_STD = (0.229, 0.224, 0.225)
MINIMAX_H3_TEXT_ENCODER_LAYER = 50
MINIMAX_H3_AUDIO_LATENTS_PER_SECOND = 40
MINIMAX_H3_AUDIO_CHANNELS = 2
MINIMAX_H3_KEYFRAME_NOISE_AUG = 0.999
MINIMAX_H3_KEYFRAME_ENCODE_SEED = 42

_ROPE_FRAME_RESCALE = 5.0 / 3.0
_ROPE_FRAMES_PER_LATENT = (1, 4, 4, 4, 4)
_ROPE_SPATIAL_SCALE = 32


@dataclass
class MiniMaxH3PackedSequence:
    """Structural description of one packed FL2VA transformer sequence."""

    sequence_length: int
    position_ids: torch.Tensor
    token_tags: torch.Tensor
    video_indices: torch.Tensor
    audio_indices: torch.Tensor
    text_indices: torch.Tensor
    num_condition_video_rows: int
    num_condition_audio_rows: int


def resolve_canvas_size(aspect_width: float, aspect_height: float) -> tuple[int, int]:
    """Resolve an aspect ratio to the H3 canvas, returned as ``(height, width)``."""
    if aspect_width <= 0 or aspect_height <= 0:
        raise ValueError(f"The aspect ratio must be positive, got {aspect_width}:{aspect_height}.")

    ratio = aspect_width / aspect_height
    if not MINIMAX_H3_MIN_ASPECT_RATIO <= ratio <= MINIMAX_H3_MAX_ASPECT_RATIO:
        raise ValueError(
            "MiniMax-H3 supports aspect ratios from 1:4 to 4:1, got "
            f"{aspect_width}:{aspect_height} ({ratio:g})."
        )

    if ratio >= 1.0:
        width = MINIMAX_H3_SHORT_EDGE * ratio
        height = float(MINIMAX_H3_SHORT_EDGE)
    else:
        width = float(MINIMAX_H3_SHORT_EDGE)
        height = MINIMAX_H3_SHORT_EDGE / ratio

    area = width * height
    if area > MINIMAX_H3_MAX_PIXELS:
        scale = (MINIMAX_H3_MAX_PIXELS / area) ** 0.5
        width, height = width * scale, height * scale

    multiple = MINIMAX_H3_CANVAS_MULTIPLE
    return (
        max(multiple, round(height / multiple) * multiple),
        max(multiple, round(width / multiple) * multiple),
    )


def align_num_frames(num_frames: int) -> int:
    """Snap upward to the next frame count of the form ``17 * n + 5``."""
    if num_frames < 1:
        raise ValueError(f"`num_frames` must be positive, got {num_frames}.")
    while num_frames % MINIMAX_H3_FRAMES_PER_CHUNK != MINIMAX_H3_LATENTS_PER_CHUNK:
        num_frames += 1
    return num_frames


def video_latent_num_frames(num_frames: int) -> int:
    """Return the number of video latent frames for an aligned pixel frame count."""
    if num_frames % MINIMAX_H3_FRAMES_PER_CHUNK != MINIMAX_H3_LATENTS_PER_CHUNK:
        raise ValueError(f"`num_frames` must be of the form 17 * n + 5, got {num_frames}.")
    return (
        num_frames - MINIMAX_H3_LATENTS_PER_CHUNK
    ) // MINIMAX_H3_FRAMES_PER_CHUNK * MINIMAX_H3_LATENTS_PER_CHUNK + 2


def audio_latent_num_frames(num_frames: int) -> int:
    """Return the number of 40 Hz audio latents covering ``num_frames`` at 24 fps."""
    return int(round(num_frames / MINIMAX_H3_FPS * MINIMAX_H3_AUDIO_LATENTS_PER_SECOND))


def prepare_keyframe_image(
    image: Image.Image,
    height: int,
    width: int,
    stretch: bool,
) -> Image.Image:
    """Stretch or cover-crop a keyframe onto the target canvas."""
    if Image is None:
        raise ModuleNotFoundError(
            "Pillow is required to prepare MiniMax-H3 keyframe images. "
            "Install Pillow or run text-only/synthetic paths that do not load keyframes."
        )
    if image.size == (width, height):
        return image
    if stretch:
        return image.resize((width, height), Image.Resampling.LANCZOS)

    scale = max(width / image.size[0], height / image.size[1])
    resized_size = (
        max(width, round(image.size[0] * scale)),
        max(height, round(image.size[1] * scale)),
    )
    left = max(0, (resized_size[0] - width) // 2)
    top = max(0, (resized_size[1] - height) // 2)
    resized = image.resize(resized_size, Image.Resampling.LANCZOS)
    return resized.crop((left, top, left + width, top + height))


def patchify_video_latents(
    latents: torch.Tensor,
    patch_size: tuple[int, int, int],
) -> torch.Tensor:
    """Pack ``[B, C, T, H, W]`` video latents into frame-major rows."""
    patch_t, patch_h, patch_w = patch_size
    batch_size, channels, num_frames, height, width = latents.shape
    if num_frames % patch_t or height % patch_h or width % patch_w:
        raise ValueError(
            f"Latents of shape {tuple(latents.shape)} are not divisible by patch {patch_size}."
        )

    latents = latents.reshape(
        batch_size,
        channels,
        num_frames // patch_t,
        patch_t,
        height // patch_h,
        patch_h,
        width // patch_w,
        patch_w,
    )
    latents = latents.permute(0, 2, 4, 6, 1, 3, 5, 7)
    return latents.reshape(-1, channels * patch_t * patch_h * patch_w).contiguous()


def unpatchify_video_tokens(
    rows: torch.Tensor,
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    channels: int,
    patch_size: tuple[int, int, int],
) -> torch.Tensor:
    """Invert :func:`patchify_video_latents`."""
    patch_t, patch_h, patch_w = patch_size
    if num_latent_frames % patch_t or latent_height % patch_h or latent_width % patch_w:
        raise ValueError("The target latent shape is not divisible by the requested patch size.")
    rows = rows.reshape(
        -1,
        num_latent_frames // patch_t,
        latent_height // patch_h,
        latent_width // patch_w,
        channels,
        patch_t,
        patch_h,
        patch_w,
    )
    rows = rows.permute(0, 4, 1, 5, 2, 6, 3, 7)
    return rows.reshape(
        -1,
        channels,
        num_latent_frames,
        latent_height,
        latent_width,
    ).contiguous()


def unpack_audio_tokens(rows: torch.Tensor, num_audio_latents: int) -> torch.Tensor:
    """Unpack channel-major audio rows to ``[2, channels, audio_frames]``."""
    expected_rows = MINIMAX_H3_AUDIO_CHANNELS * num_audio_latents
    if rows.shape[0] != expected_rows:
        raise ValueError(f"Expected {expected_rows} audio rows, got {rows.shape[0]}.")
    rows = rows.reshape(MINIMAX_H3_AUDIO_CHANNELS, num_audio_latents, rows.shape[-1])
    return rows.permute(0, 2, 1).contiguous()


def _spatial_position_grid(dim: int, patch: int, sqrt_area: float) -> torch.Tensor:
    ratio = dim / sqrt_area
    left = (1.0 - ratio) / 2.0
    grid = np.linspace(left, left + ratio, dim // patch, endpoint=False)
    return torch.from_numpy(grid * _ROPE_SPATIAL_SCALE).to(torch.float64)


def _temporal_position_grid(num_latent_frames: int, origin: float) -> torch.Tensor:
    spans = torch.tensor(
        [
            _ROPE_FRAME_RESCALE * _ROPE_FRAMES_PER_LATENT[index % len(_ROPE_FRAMES_PER_LATENT)]
            for index in range(num_latent_frames)
        ],
        dtype=torch.float64,
    )
    return origin + torch.cat([torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0)])


def _temporal_position_span(num_latent_frames: int) -> float:
    spans = np.ones(num_latent_frames, dtype=np.float64) * _ROPE_FRAME_RESCALE
    for index, frame_count in enumerate(_ROPE_FRAMES_PER_LATENT):
        spans[index :: len(_ROPE_FRAMES_PER_LATENT)] *= frame_count
    return float(spans.sum())


def build_packed_sequence(
    text_token_tags: torch.Tensor,
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    num_audio_latents: int,
    patch_size: tuple[int, int, int],
    keyframe_anchors: tuple[str, ...] = (),
) -> MiniMaxH3PackedSequence:
    """Build the packed FL2VA layout, indices, modality tags, and 3-D positions."""
    patch_t, patch_h, patch_w = patch_size
    if patch_t != 1:
        raise ValueError(f"MiniMax-H3 requires a temporal patch size of 1, got {patch_t}.")
    if latent_height % patch_h or latent_width % patch_w:
        raise ValueError("The latent canvas must be divisible by the spatial patch size.")
    if text_token_tags.ndim != 1:
        raise ValueError("`text_token_tags` must be one-dimensional.")

    rows_per_frame = (latent_height // patch_h) * (latent_width // patch_w)
    num_text_tokens = text_token_tags.shape[0]
    num_condition_rows = len(keyframe_anchors) * rows_per_frame
    num_audio_rows = num_audio_latents * MINIMAX_H3_AUDIO_CHANNELS
    num_video_rows = num_latent_frames * rows_per_frame
    sequence_length = num_text_tokens + num_condition_rows + num_audio_rows + num_video_rows

    condition_start = num_text_tokens
    audio_start = condition_start + num_condition_rows
    video_start = audio_start + num_audio_rows

    position_ids = torch.zeros(sequence_length, 3, dtype=torch.float64)
    position_ids[:num_text_tokens, 0] = torch.arange(
        num_text_tokens,
        dtype=torch.float64,
    )

    sqrt_area = np.sqrt(latent_height * latent_width)
    height_grid = _spatial_position_grid(latent_height, patch_h, sqrt_area)
    width_grid = _spatial_position_grid(latent_width, patch_w, sqrt_area)
    frame_grid = torch.stack(
        [grid.reshape(-1) for grid in torch.meshgrid(height_grid, width_grid, indexing="ij")],
        dim=-1,
    )

    for index, anchor in enumerate(keyframe_anchors):
        if anchor == "first":
            anchor_time = float(num_text_tokens)
        elif anchor == "last":
            anchor_time = (
                float(num_text_tokens)
                + _temporal_position_span(num_latent_frames)
                - _ROPE_FRAME_RESCALE
            )
        else:
            raise ValueError(f"A keyframe anchor must be 'first' or 'last', got {anchor!r}.")
        rows = slice(
            condition_start + index * rows_per_frame,
            condition_start + (index + 1) * rows_per_frame,
        )
        position_ids[rows, 0] = anchor_time
        position_ids[rows, 1:] = frame_grid

    audio_time = float(num_text_tokens) + torch.arange(
        num_audio_latents,
        dtype=torch.float64,
    )
    position_ids[audio_start:video_start, 0] = audio_time.repeat(MINIMAX_H3_AUDIO_CHANNELS)
    position_ids[audio_start:video_start, 2] = torch.cat(
        [
            torch.full(
                (num_audio_latents,),
                float(width_grid[0]),
                dtype=torch.float64,
            ),
            torch.full(
                (num_audio_rows - num_audio_latents,),
                float(width_grid[-1]),
                dtype=torch.float64,
            ),
        ]
    )

    video_position_ids = torch.empty(
        num_latent_frames,
        rows_per_frame,
        3,
        dtype=torch.float64,
    )
    video_position_ids[:, :, 0] = _temporal_position_grid(
        num_latent_frames,
        float(num_text_tokens),
    )[:, None]
    video_position_ids[:, :, 1:] = frame_grid[None]
    position_ids[video_start:] = video_position_ids.reshape(-1, 3)

    video_indices = torch.cat(
        [
            torch.arange(condition_start, audio_start),
            torch.arange(video_start, sequence_length),
        ]
    )
    audio_indices = torch.arange(audio_start, video_start)
    text_indices = torch.arange(num_text_tokens)

    token_tags = torch.empty(sequence_length, dtype=torch.long)
    token_tags[text_indices] = text_token_tags.to(torch.long)
    token_tags[audio_indices] = MINIMAX_H3_AUDIO_TAG
    token_tags[video_indices] = MINIMAX_H3_VIDEO_TAG

    return MiniMaxH3PackedSequence(
        sequence_length=sequence_length,
        position_ids=position_ids,
        token_tags=token_tags,
        video_indices=video_indices,
        audio_indices=audio_indices,
        text_indices=text_indices,
        num_condition_video_rows=num_condition_rows,
        num_condition_audio_rows=0,
    )


def build_row_timesteps(
    layout: MiniMaxH3PackedSequence,
    video_timestep: float,
    audio_timestep: float,
    condition_video_timestep: float,
    condition_audio_timestep: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assign per-row times and return distinct values plus row-to-value indices."""
    row_timesteps = torch.full(
        (layout.sequence_length,),
        video_timestep,
        dtype=torch.float32,
    )
    row_timesteps[layout.video_indices[: layout.num_condition_video_rows]] = (
        condition_video_timestep
    )
    row_timesteps[layout.audio_indices[layout.num_condition_audio_rows :]] = audio_timestep
    row_timesteps[layout.audio_indices[: layout.num_condition_audio_rows]] = (
        condition_audio_timestep
    )
    return torch.unique(row_timesteps, sorted=True, return_inverse=True)


def _randn_tensor(
    shape: tuple[int, ...],
    generator: torch.Generator | list[torch.Generator] | None,
    device: torch.device | None,
    dtype: torch.dtype,
) -> torch.Tensor:
    target_device = device or torch.device("cpu")
    generator_device = None
    if generator is not None:
        first_generator = generator[0] if isinstance(generator, list) else generator
        generator_device = first_generator.device
    random_device = target_device
    if generator_device is not None and generator_device.type != target_device.type:
        if generator_device.type != "cpu":
            raise ValueError(
                f"Cannot generate a {target_device} tensor from a "
                f"{generator_device.type} generator."
            )
        random_device = torch.device("cpu")

    if isinstance(generator, list):
        if len(generator) != shape[0]:
            raise ValueError(
                "A generator list must have one entry per batch element; "
                f"got {len(generator)} for batch size {shape[0]}."
            )
        samples = [
            torch.randn(
                (1, *shape[1:]),
                generator=batch_generator,
                device=random_device,
                dtype=dtype,
            )
            for batch_generator in generator
        ]
        return torch.cat(samples).to(target_device)
    return torch.randn(
        shape,
        generator=generator,
        device=random_device,
        dtype=dtype,
    ).to(target_device)


def keyframe_condition_noise(
    condition_latent_shapes: tuple[tuple[int, int, int], ...],
    patch_size: tuple[int, int, int],
    latent_channels: int,
    generator: torch.Generator | list[torch.Generator] | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Draw and patchify keyframe conditioning noise in packed order."""
    rows = []
    for num_latent_frames, latent_height, latent_width in condition_latent_shapes:
        noise = _randn_tensor(
            (1, latent_channels, num_latent_frames, latent_height, latent_width),
            generator=generator,
            device=device,
            dtype=dtype,
        )
        rows.append(patchify_video_latents(noise, patch_size))
    if not rows:
        patch_volume = int(np.prod(patch_size))
        return torch.empty(
            (0, latent_channels * patch_volume),
            device=device,
            dtype=dtype,
        )
    return torch.cat(rows)
