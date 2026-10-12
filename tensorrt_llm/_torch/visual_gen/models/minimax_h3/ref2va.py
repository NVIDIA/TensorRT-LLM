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

"""H3 reference preparation using the shared TRTLLM encoding and packing path."""

import math
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING

import numpy as np
import torch
from diffusers.modular_pipelines.minimax_h3.references import (
    MiniMaxH3AudioReference,
    MiniMaxH3ImageReference,
    MiniMaxH3Reference,
    MiniMaxH3VideoReference,
)
from PIL import Image, ImageOps

from .packing import (
    MINIMAX_H3_CANVAS_MULTIPLE,
    MINIMAX_H3_FPS,
    MINIMAX_H3_FRAMES_PER_CHUNK,
    MINIMAX_H3_LATENTS_PER_CHUNK,
    MiniMaxH3PackedSequence,
    audio_latent_num_frames,
    build_reference_sequence,
    resolve_canvas_size,
    video_latent_num_frames,
)

if TYPE_CHECKING:
    from transformers import Qwen2TokenizerFast, Qwen3VLProcessor

    from tensorrt_llm.visual_gen.params import VisualGenParams

    from .pipeline_minimax_h3 import MiniMaxH3Pipeline


def validate_reference_order(value: object) -> list[str] | None:
    """Validate the optional cross-modality order, e.g. ['video:0', 'image:0']."""
    if value is None:
        return None
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError("reference_order must be a list of 'image:N', 'video:N', or 'audio:N'.")
    for item in value:
        kind, separator, index = item.partition(":")
        if kind not in ("image", "video", "audio") or not separator or not index.isdecimal():
            raise ValueError(f"Invalid reference_order entry: {item!r}.")
        if str(int(index)) != index:
            raise ValueError(f"Reference indices must be canonical nonnegative integers: {item!r}.")
    if len(set(value)) != len(value):
        raise ValueError("reference_order must not repeat a reference.")
    return value


def _decode_reference(kind: str, content: object) -> MiniMaxH3Reference:
    if kind == "image":
        if isinstance(content, Image.Image):
            return MiniMaxH3ImageReference(ImageOps.exif_transpose(content).convert("RGB"))
        source = (
            BytesIO(bytes(content))
            if isinstance(content, (bytes, bytearray, memoryview))
            else content
        )
        with Image.open(source) as image:
            return MiniMaxH3ImageReference(ImageOps.exif_transpose(image).convert("RGB"))
    reference_cls = MiniMaxH3VideoReference if kind == "video" else MiniMaxH3AudioReference
    if isinstance(content, (str, Path)):
        return reference_cls.from_file(content)
    if not isinstance(content, (bytes, bytearray, memoryview)):
        raise ValueError(f"MiniMax-H3 {kind} references require encoded media bytes or a path.")
    # Workers receive resolved bytes. The upstream decoder accepts filenames;
    # keep the file alive only for decoding, preserving its FPS and sample rate.
    with NamedTemporaryFile() as media:
        media.write(bytes(content))
        media.flush()
        return reference_cls.from_file(media.name)


def load_references(params: "VisualGenParams") -> list[MiniMaxH3Reference]:
    """Decode existing VisualGen media slots in an explicitly reproducible order."""
    slots = {}
    for kind in ("image", "video", "audio"):
        refs = getattr(params, f"{kind}_reference", None) or []
        if not isinstance(refs, list):
            refs = [refs]
        for index, ref in enumerate(refs):
            if ref.role not in (None, "reference"):
                raise ValueError("MiniMax-H3 ref2va only accepts role 'reference'.")
            slots[f"{kind}:{index}"] = (kind, ref.content)
    order = validate_reference_order((params.extra_params or {}).get("reference_order"))
    if order is None:
        order = list(slots)
    if set(order) != set(slots):
        raise ValueError("reference_order must include every supplied reference exactly once.")
    return [_decode_reference(*slots[key]) for key in order]


def _normalize_video_condition(
    frames: np.ndarray, fps: float, num_frames: int, target_fps: float
) -> np.ndarray:
    if isinstance(frames, list):
        frames = np.stack([np.asarray(frame.convert("RGB")) for frame in frames])
    if isinstance(frames, torch.Tensor):
        frames = frames.movedim(-3, -1).cpu().numpy()
    frames = np.asarray(frames)
    if frames.dtype != np.uint8:
        frames = (frames * 255.0).round().clip(0, 255).astype(np.uint8)
    if frames.ndim != 4 or frames.shape[3] != 3:
        raise ValueError(
            f"A reference video must be `(num_frames, height, width, 3)` RGB frames, got {tuple(frames.shape)}."
        )
    if fps <= 0:
        raise ValueError(f"A reference video must have a positive frame rate, got {fps}.")
    if fps != target_fps:
        scale = target_fps / fps
        slots = np.floor(np.arange(frames.shape[0]) * scale + 0.5).astype(np.int64)
        frames = np.repeat(
            frames, np.diff(slots, append=math.floor(frames.shape[0] * scale + 0.5)), axis=0
        )
    frames = frames[:num_frames]
    height, width = resolve_canvas_size(frames.shape[2], frames.shape[1])
    if frames.shape[1:3] == (height, width):
        return frames
    return np.stack(
        [
            np.asarray(Image.fromarray(frame).resize((width, height), Image.Resampling.LANCZOS))
            for frame in frames
        ]
    )


def _normalize_audio_condition(
    waveform: torch.Tensor, sample_rate: int, target_sample_rate: int, max_duration: float
) -> torch.Tensor:
    waveform = torch.as_tensor(waveform)
    if waveform.ndim != 2 or waveform.shape[0] not in (1, 2):
        raise ValueError(
            f"Expected mono/stereo audio [channels, samples], got {tuple(waveform.shape)}."
        )
    waveform = waveform.to(torch.float32)[:, : int(max_duration * sample_rate)]
    if waveform.shape[0] != 2:
        waveform = waveform.expand(2, -1).contiguous()
    if sample_rate == target_sample_rate:
        return waveform
    try:
        import torchaudio
    except ImportError as error:
        raise ImportError(
            f"Resampling {sample_rate} Hz to {target_sample_rate} Hz requires torchaudio."
        ) from error
    return torchaudio.transforms.Resample(sample_rate, target_sample_rate)(waveform)


def _sample_video_condition_frames(
    frames: np.ndarray, fps: float, sample_fps: float, temporal_patch: int
) -> tuple[list[np.ndarray], list[float]]:
    stride = fps / sample_fps
    indices, cursor = ([], 0.0)
    while round(cursor) < frames.shape[0]:
        if not indices or round(cursor) > indices[-1]:
            indices.append(round(cursor))
        cursor += stride
    if len(indices) < temporal_patch:
        minimum = round((temporal_patch - 1) * stride) + 1
        raise ValueError(
            f"Reference video needs at least {minimum} frames at {fps:g} fps; got {frames.shape[0]}."
        )
    timestamps = [index / sample_fps for index in range(len(indices))]
    timestamps += [timestamps[-1]] * (-len(timestamps) % temporal_patch)
    block_timestamps = [
        (timestamps[index] + timestamps[index + temporal_patch - 1]) / 2
        for index in range(0, len(timestamps), temporal_patch)
    ]
    return ([frames[index] for index in indices], block_timestamps)


def _gather_vision_features(
    processor: "Qwen3VLProcessor", references: list[MiniMaxH3Reference], fps: float
) -> tuple[dict, list[int], list[int], list[list[float]]]:
    merge_size = processor.image_processor.merge_size**2
    vision_inputs = {}
    image_token_counts = []
    images = [reference.image for reference in references if reference.kind == "image"]
    if images:
        image_features = processor.image_processor(images=images, return_tensors="pt")
        vision_inputs["pixel_values"] = image_features["pixel_values"]
        vision_inputs["image_grid_thw"] = image_features["image_grid_thw"]
        image_token_counts = [
            int(grid.prod()) // merge_size for grid in image_features["image_grid_thw"]
        ]
    video_block_token_counts, video_block_timestamps = ([], [])
    videos = [reference for reference in references if reference.kind == "video"]
    if videos:
        temporal_patch = processor.video_processor.temporal_patch_size
        sampled = [
            _sample_video_condition_frames(reference.frames, fps, 2.0, temporal_patch)
            for reference in videos
        ]
        video_block_timestamps = [timestamps for _, timestamps in sampled]
        video_features = processor.video_processor(
            videos=[np.stack(frames) for frames, _ in sampled],
            do_sample_frames=False,
            return_tensors="pt",
        )
        vision_inputs["pixel_values_videos"] = video_features["pixel_values_videos"]
        vision_inputs["video_grid_thw"] = video_features["video_grid_thw"]
        video_block_token_counts = [
            int(grid[1]) * int(grid[2]) // merge_size for grid in video_features["video_grid_thw"]
        ]
        for timestamps, grid in zip(video_block_timestamps, video_features["video_grid_thw"]):
            if int(grid[0]) != len(timestamps):
                raise ValueError(
                    f"Processor emitted {int(grid[0])} video blocks but H3 expects {len(timestamps)}."
                )
    return (vision_inputs, image_token_counts, video_block_token_counts, video_block_timestamps)


def _build_presentation(
    tokenizer: "Qwen2TokenizerFast",
    prompt: str,
    references: list[MiniMaxH3Reference],
    image_token_counts: list[int],
    video_block_token_counts: list[int],
    video_block_timestamps: list[list[float]],
    text_tag: int = 1,
    video_tag: int = 0,
) -> tuple[list[int], list[int]]:
    def text(value: str) -> tuple[list[int], list[int]]:
        token_ids = tokenizer(value, add_special_tokens=False)["input_ids"]
        return (token_ids, [text_tag] * len(token_ids))

    def vision(pad_token: str, num_tokens: int) -> tuple[list[int], list[int]]:
        token_ids = (
            [tokenizer.convert_tokens_to_ids("<|vision_start|>")]
            + [tokenizer.convert_tokens_to_ids(pad_token)] * num_tokens
            + [tokenizer.convert_tokens_to_ids("<|vision_end|>")]
        )
        return (token_ids, [video_tag] * len(token_ids))

    token_ids, token_tags = ([], [])

    def emit(segment: tuple[list[int], list[int]]) -> None:
        token_ids.extend(segment[0])
        token_tags.extend(segment[1])

    counts = {"image": 0, "video": 0, "audio": 0}
    for reference in references:
        if reference.has_audio:
            counts["audio"] += 1
            emit(text(f"<Audio {counts['audio']}>: "))
        if reference.kind == "image":
            counts["image"] += 1
            emit(text(f"<Picture {counts['image']}>: "))
            emit(vision("<|image_pad|>", image_token_counts[counts["image"] - 1]))
        elif reference.kind == "video":
            counts["video"] += 1
            emit(text(f"<Video {counts['video']}>: "))
            for timestamp in video_block_timestamps[counts["video"] - 1]:
                emit(text(f"<{timestamp:.1f} seconds>"))
                emit(vision("<|video_pad|>", video_block_token_counts[counts["video"] - 1]))
    emit(text(prompt))
    return (token_ids, token_tags)


def normalize_references(
    references: list[MiniMaxH3Reference], num_frames: int, sample_rate: int
) -> list[MiniMaxH3Reference]:
    """Validate and normalize decoded media while preserving reference order."""
    kinds = [entry.kind for entry in references]
    if not kinds or set(kinds) == {"audio"}:
        raise ValueError("Ref2VA requires at least one image or video reference.")
    for kind, limit in (("image", 9), ("video", 3), ("audio", 3)):
        if kinds.count(kind) > limit:
            raise ValueError(f"MiniMax-H3 accepts at most {limit} {kind} references.")
    if len(kinds) > 12:
        raise ValueError("MiniMax-H3 accepts at most 12 references in total.")
    result = []
    for entry in references:
        audio = None
        if entry.has_audio:
            audio = _normalize_audio_condition(
                entry.audio,
                entry.sample_rate or sample_rate,
                sample_rate,
                num_frames / MINIMAX_H3_FPS,
            )
        if entry.kind == "image":
            image = entry.image
            width, height = image.size
            if min(width, height) <= 0 or max(width, height) > 4 * min(width, height):
                raise ValueError("Reference image aspect ratio must be within 1:4 and 4:1.")
            scale = 2048 / min(width, height)
            multiple = MINIMAX_H3_CANVAS_MULTIPLE
            size = tuple(
                max(multiple, round(axis * scale / multiple) * multiple) for axis in (width, height)
            )
            if image.size != size:
                image = image.resize(size, Image.Resampling.LANCZOS)
            result.append(MiniMaxH3ImageReference(image))
        elif entry.kind == "video":
            frames = _normalize_video_condition(
                entry.frames, float(entry.fps), num_frames, MINIMAX_H3_FPS
            )
            result.append(
                MiniMaxH3VideoReference(
                    frames=frames,
                    fps=MINIMAX_H3_FPS,
                    audio=audio,
                    sample_rate=sample_rate if audio is not None else None,
                )
            )
        elif entry.kind == "audio":
            result.append(MiniMaxH3AudioReference(audio=audio, sample_rate=sample_rate))
        else:
            raise ValueError(f"Unknown H3 reference kind: {entry.kind!r}.")
    return result


def prepare_references(
    pipeline: "MiniMaxH3Pipeline",
    references: list[MiniMaxH3Reference],
    prompt: str,
    height: int,
    width: int,
    num_frames: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, MiniMaxH3PackedSequence, torch.Tensor, torch.Tensor | None]:
    """Prepare reference conditioning through TRTLLM's shared H3 helpers."""
    references = normalize_references(
        references, num_frames, pipeline.audio_vae.config.sampling_rate
    )
    vision, image_counts, video_counts, timestamps = _gather_vision_features(
        pipeline.processor, references, MINIMAX_H3_FPS
    )
    ids, tags = _build_presentation(
        pipeline.tokenizer, prompt, references, image_counts, video_counts, timestamps
    )
    prompt_embeds = pipeline._encode_text_tokens(ids, vision)
    visual_latents, audio_latents = [], []
    audio_mean = torch.tensor(pipeline.audio_vae.config.latents_mean).view(1, 1, -1)
    audio_std = torch.tensor(pipeline.audio_vae.config.latents_std).view(1, 1, -1)
    for reference in references:
        if reference.kind == "image":
            pixels = (
                torch.from_numpy(np.array(reference.image))
                .to(pipeline.device)
                .permute(2, 0, 1)[None, :, None]
            )
            visual_latents.append(pipeline._encode_visual_condition(pixels))
        elif reference.kind == "video":
            chunks = max(
                1,
                (reference.frames.shape[0] - MINIMAX_H3_LATENTS_PER_CHUNK)
                // MINIMAX_H3_FRAMES_PER_CHUNK,
            )
            frames = chunks * MINIMAX_H3_FRAMES_PER_CHUNK + MINIMAX_H3_LATENTS_PER_CHUNK
            pixels = (
                torch.from_numpy(reference.frames[:frames].copy())
                .to(pipeline.device)
                .permute(3, 0, 1, 2)[None]
            )
            visual_latents.append(pipeline._encode_visual_condition(pixels))
        if reference.has_audio:
            posterior = pipeline.audio_vae.encode(
                reference.audio.to(pipeline.device)[:, None], return_dict=False
            )[0]
            latents = posterior.mode().float().cpu().transpose(1, 2)
            audio_latents.append(
                ((latents - audio_mean) / audio_std).reshape(
                    -1, pipeline.audio_vae.config.latent_channels
                )
            )
    ratio = pipeline.vae.spatial_compression_ratio
    layout = build_reference_sequence(
        torch.tensor(tags, dtype=torch.long),
        [(ref.kind, ref.has_audio) for ref in references],
        [tuple(value.shape[2:]) for value in visual_latents],
        [value.shape[0] for value in audio_latents],
        video_latent_num_frames(num_frames),
        height // ratio,
        width // ratio,
        audio_latent_num_frames(num_frames),
        pipeline.transformer.config.patch_size,
    )
    condition_rows = pipeline._prepare_condition_rows(visual_latents, generator)
    audio_rows = torch.cat(audio_latents).to(pipeline.device) if audio_latents else None
    return prompt_embeds, layout, condition_rows, audio_rows
