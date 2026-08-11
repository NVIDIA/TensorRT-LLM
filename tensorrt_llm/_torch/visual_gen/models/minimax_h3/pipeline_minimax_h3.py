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
"""MiniMax-H3 text-to-video+audio pipeline for VisualGen.

Port of the Diffusers ``MiniMaxH3ModularPipeline`` (``t2va`` / ``fl2va``
workflows) onto the TRT-LLM VisualGen stack. MiniMax-H3 denoises one packed
sequence that holds the text conditioning, the keyframe conditioning latents,
the audio latents and the video latents at once, with full self-attention over
that single sequence. The checkpoint is guidance-distilled, so there is no
guider, no ``negative_prompt`` and no ``guidance_scale``, and every step runs
exactly one forward pass. Two schedulers (video ``shift=12.0``, audio
``shift=3.0``) are stepped inside that single transformer call.

The text conditioner (Qwen3-VL) and the two VAEs (video ``AutoencoderKLMiniMaxH3``,
audio ``AutoencoderKLMiniMaxH3Audio``) are loaded from Diffusers / Transformers
as-is; only the transformer runs on the TRT-LLM VisualGen stack.
"""

from contextlib import contextmanager
from typing import List, Optional, Tuple

import numpy as np
import torch
from diffusers.models import AutoencoderKLMiniMaxH3, AutoencoderKLMiniMaxH3Audio
from diffusers.schedulers import MiniMaxH3Scheduler
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image
from transformers import Qwen2TokenizerFast, Qwen3VLForConditionalGeneration, Qwen3VLProcessor

from tensorrt_llm._torch.visual_gen.models.minimax_h3.transformer_minimax_h3 import (
    MiniMaxH3Transformer3DModel,
)
from tensorrt_llm._torch.visual_gen.output import PipelineOutput
from tensorrt_llm._torch.visual_gen.pipeline import BasePipeline, ExtraParamSchema
from tensorrt_llm._torch.visual_gen.pipeline_registry import PipelineComponent, register_pipeline
from tensorrt_llm._utils import nvtx_range
from tensorrt_llm.inputs.utils import load_image
from tensorrt_llm.logger import logger

# Per-row modality tags. They index the transformer's AdaLN table, so the
# values are a checkpoint contract.
MINIMAX_H3_VIDEO_TAG = 0
MINIMAX_H3_TEXT_TAG = 1
MINIMAX_H3_AUDIO_TAG = 2

# MiniMax-H3 generates at a fixed 24 fps.
MINIMAX_H3_FPS = 24
MINIMAX_H3_MIN_ASPECT_RATIO = 1 / 4
MINIMAX_H3_MAX_ASPECT_RATIO = 4

# Request defaults. 124 frames is ~5.2 s at 24 fps, the shortest clip the
# checkpoint was tuned for, and the guidance-distilled model needs the full
# 50-point sigma grid.
MINIMAX_H3_DEFAULT_NUM_FRAMES = 124
MINIMAX_H3_DEFAULT_NUM_INFERENCE_STEPS = 50

# The audio VAE hops 800 samples at 32 kHz, i.e. 40 latents per second. Stereo
# is carried as two channel-major blocks of audio rows (and as two batch items
# at the audio VAE boundary, which is mono).
MINIMAX_H3_AUDIO_LATENTS_PER_SECOND = 40
MINIMAX_H3_AUDIO_CHANNELS = 2

# Rotary-time constants. One latent frame spans `5/3 * frames_per_latent`
# rotary units, where the pattern `(1, 4, 4, 4, 4)` mirrors the VAE's
# 17-pixel-frames-to-5-latent-frames grouping; the spatial axes are normalized
# by the square root of the latent area and scaled by 32.
_ROPE_FRAME_RESCALE = 5.0 / 3.0
_ROPE_FRAMES_PER_LATENT = (1, 4, 4, 4, 4)
_ROPE_SPATIAL_SCALE = 32

# The `t` a visual conditioning anchor is held at: 0.999, just short of clean.
_KEYFRAME_NOISE_AUG = 0.999

# bfloat16 footprint of the released Qwen3-VL conditioner, used by the
# `conditioner_offload="auto"` decision.
_CONDITIONER_BYTES = 62 * 1024**3

# Video-VAE decode activation peak on the reference canvas (fp16 autocast,
# fp32 VAE weights), the largest activation spike of a request. The
# conditioner only fits alongside the transformer when the decode spike also
# fits; otherwise it is swapped in per request.
_VAE_DECODE_BYTES = 17 * 1024**3


def resolve_canvas_size(
    aspect_width: float,
    aspect_height: float,
    canvas_multiple: int,
    short_edge: int,
    max_pixels: int,
    min_aspect_ratio: float = MINIMAX_H3_MIN_ASPECT_RATIO,
    max_aspect_ratio: float = MINIMAX_H3_MAX_ASPECT_RATIO,
) -> Tuple[int, int]:
    """Resolve a display aspect ratio into a MiniMax-H3 canvas.

    The short edge starts at `short_edge`, the area is capped at `max_pixels`
    and both axes are then rounded to the nearest `canvas_multiple`.
    """
    if aspect_width <= 0 or aspect_height <= 0:
        raise ValueError(f"The aspect ratio must be positive, got {aspect_width}:{aspect_height}.")

    ratio = aspect_width / aspect_height
    if not min_aspect_ratio <= ratio <= max_aspect_ratio:
        raise ValueError(
            f"MiniMax-H3 supports aspect ratios from 1:{1 / min_aspect_ratio:g} to {max_aspect_ratio:g}:1, got "
            f"{aspect_width}:{aspect_height} ({ratio:g})."
        )

    if ratio >= 1.0:
        width, height = short_edge * ratio, float(short_edge)
    else:
        width, height = float(short_edge), short_edge / ratio

    area = width * height
    if area > max_pixels:
        scale = (max_pixels / area) ** 0.5
        width, height = width * scale, height * scale

    multiple = canvas_multiple
    return max(multiple, round(height / multiple) * multiple), max(
        multiple, round(width / multiple) * multiple
    )


def fit_keyframe_to_canvas(
    keyframe: Image.Image, width: int, height: int, stretch: bool
) -> Image.Image:
    """Fit a `fl2va` keyframe onto the canvas.

    The anchor that opens the video is stretched onto the canvas, a follower is
    cover-cropped: scaled by the larger of the two axis ratios, then cropped
    about the center. This is the released model's arithmetic, which differs
    from `VaeImageProcessor`'s by a pixel on some aspect ratios.
    """
    if keyframe.size == (width, height):
        return keyframe
    if stretch:
        return keyframe.resize((width, height), Image.Resampling.LANCZOS)

    scale = max(width / keyframe.size[0], height / keyframe.size[1])
    resized_size = (
        max(width, round(keyframe.size[0] * scale)),
        max(height, round(keyframe.size[1] * scale)),
    )
    left = max(0, (resized_size[0] - width) // 2)
    top = max(0, (resized_size[1] - height) // 2)
    resized = keyframe.resize(resized_size, Image.Resampling.LANCZOS)
    return resized.crop((left, top, left + width, top + height))


def align_num_frames(num_frames: int, frames_per_chunk: int, latents_per_chunk: int) -> int:
    """Snap a frame count up to the next `frames_per_chunk * n + latents_per_chunk`."""
    if num_frames < 1:
        raise ValueError(f"`num_frames` must be positive, got {num_frames}.")
    while num_frames % frames_per_chunk != latents_per_chunk:
        num_frames += 1
    return num_frames


def video_latent_num_frames(num_frames: int, frames_per_chunk: int, latents_per_chunk: int) -> int:
    """The number of latent frames the video VAE produces for a `17 * n + 5` frame count."""
    if num_frames % frames_per_chunk != latents_per_chunk:
        raise ValueError(
            f"`num_frames` must be of the form {frames_per_chunk} * n + {latents_per_chunk}, got {num_frames}."
        )
    return (num_frames - latents_per_chunk) // frames_per_chunk * latents_per_chunk + 2


def audio_latent_num_frames(num_frames: int) -> int:
    """The number of audio latents that covers a video of `num_frames` frames."""
    return int(round(num_frames / MINIMAX_H3_FPS * MINIMAX_H3_AUDIO_LATENTS_PER_SECOND))


def patchify_video_latents(latents: torch.Tensor, patch_size: Tuple[int, int, int]) -> torch.Tensor:
    """Pack video latents into transformer rows, frame-major then row-major."""
    patch_t, patch_h, patch_w = patch_size
    batch_size, channels, num_frames, height, width = latents.shape
    if num_frames % patch_t or height % patch_h or width % patch_w:
        raise ValueError(
            f"Latents of shape {tuple(latents.shape)} are not divisible by the patch {patch_size}."
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


def unpatchify_video_latents(
    rows: torch.Tensor,
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    channels: int,
    patch_size: Tuple[int, int, int],
) -> torch.Tensor:
    """The inverse of `patchify_video_latents`."""
    patch_t, patch_h, patch_w = patch_size
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
    return rows.reshape(-1, channels, num_latent_frames, latent_height, latent_width).contiguous()


def _spatial_position_grid(dim: int, patch: int, sqrt_area: float) -> torch.Tensor:
    """One aspect-normalized spatial rotary axis, in float64."""
    ratio = dim / sqrt_area
    left = (1.0 - ratio) / 2.0
    # `np.linspace(..., endpoint=False)` is `start + arange(num) * (stop - start) / num`, which is not what
    # `torch.linspace` computes; the float64 grid has to be reproduced exactly.
    grid = np.linspace(left, left + ratio, dim // patch, endpoint=False) * _ROPE_SPATIAL_SCALE
    return torch.from_numpy(grid).to(torch.float64)


def _temporal_position_grid(num_latent_frames: int, origin: float) -> torch.Tensor:
    """The rotary time of every latent frame, starting at `origin`."""
    spans = torch.tensor(
        [
            _ROPE_FRAME_RESCALE * _ROPE_FRAMES_PER_LATENT[index % len(_ROPE_FRAMES_PER_LATENT)]
            for index in range(num_latent_frames)
        ],
        dtype=torch.float64,
    )
    return origin + torch.cat([torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0)])


def _frame_position_grid(
    latent_height: int, latent_width: int, patch_h: int, patch_w: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The `(h, w)` rotary coordinates of one latent frame, and the width axis."""
    sqrt_area = np.sqrt(latent_height * latent_width)
    height_grid = _spatial_position_grid(latent_height, patch_h, sqrt_area)
    width_grid = _spatial_position_grid(latent_width, patch_w, sqrt_area)
    grids = torch.meshgrid(height_grid, width_grid, indexing="ij")
    return torch.stack([grid.reshape(-1) for grid in grids], dim=-1), width_grid


def build_packed_sequence(
    text_token_tags: torch.Tensor,
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    num_audio_latents: int,
    patch_size: Tuple[int, int, int],
    keyframe_anchors: Tuple[str, ...] = (),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """Build the `[text | keyframe conditions | target audio | target video]` layout.

    Returns `position_ids`, `token_tags`, `video_indices`, `audio_indices`,
    `text_indices`, and the number of leading video rows that are
    conditioning rows (audio conditioning is not produced here).
    """
    _, patch_h, patch_w = patch_size
    rows_per_frame = (latent_height // patch_h) * (latent_width // patch_w)
    num_text_tokens = text_token_tags.shape[0]
    num_condition_rows = len(keyframe_anchors) * rows_per_frame
    num_audio_rows = num_audio_latents * MINIMAX_H3_AUDIO_CHANNELS
    num_video_rows = num_latent_frames * rows_per_frame
    sequence_length = num_text_tokens + num_condition_rows + num_audio_rows + num_video_rows

    condition_start = num_text_tokens
    audio_start = condition_start + num_condition_rows
    video_start = audio_start + num_audio_rows

    # 1. The (t, h, w) grid. Text rows sit on the time axis at their row
    # index, and the media rows continue the time axis from there.
    position_ids = torch.zeros(sequence_length, 3, dtype=torch.float64)
    position_ids[:num_text_tokens, 0] = torch.arange(num_text_tokens, dtype=torch.float64)

    frame_grid, width_grid = _frame_position_grid(latent_height, latent_width, patch_h, patch_w)

    for index, anchor in enumerate(keyframe_anchors):
        if anchor == "first":
            anchor_time = float(num_text_tokens)
        elif anchor == "last":
            # The rotary time the generated frames span, summed by numpy's
            # pairwise summation because that is how the reference computes
            # this anchor.
            spans = np.ones(num_latent_frames, dtype=np.float64) * _ROPE_FRAME_RESCALE
            for offset in range(len(_ROPE_FRAMES_PER_LATENT)):
                spans[offset :: len(_ROPE_FRAMES_PER_LATENT)] *= _ROPE_FRAMES_PER_LATENT[offset]
            anchor_time = float(num_text_tokens) + float(spans.sum()) - _ROPE_FRAME_RESCALE
        else:
            raise ValueError(f"A keyframe anchor must be 'first' or 'last', got {anchor!r}.")
        rows = slice(
            condition_start + index * rows_per_frame, condition_start + (index + 1) * rows_per_frame
        )
        position_ids[rows, 0] = anchor_time
        position_ids[rows, 1:] = frame_grid

    # Audio rows are channel-major and share the video's rotary clock: one
    # unit per latent at 40 latents/s equals 24 fps * 5/3. They carry no
    # height coordinate and are pinned to the two extremes of the width grid.
    audio_time = float(num_text_tokens) + torch.arange(num_audio_latents, dtype=torch.float64)
    position_ids[audio_start:video_start, 0] = audio_time.repeat(MINIMAX_H3_AUDIO_CHANNELS)
    position_ids[audio_start:video_start, 2] = torch.cat(
        [
            torch.full((num_audio_latents,), float(width_grid[0]), dtype=torch.float64),
            torch.full(
                (num_audio_rows - num_audio_latents,), float(width_grid[-1]), dtype=torch.float64
            ),
        ]
    )

    video_position_ids = torch.empty(num_latent_frames, rows_per_frame, 3, dtype=torch.float64)
    video_position_ids[:, :, 0] = _temporal_position_grid(
        num_latent_frames, float(num_text_tokens)
    )[:, None]
    video_position_ids[:, :, 1:] = frame_grid[None]
    position_ids[video_start:] = video_position_ids.reshape(-1, 3)

    # 2. Row indices and modality tags.
    video_indices = torch.cat(
        [torch.arange(condition_start, audio_start), torch.arange(video_start, sequence_length)]
    )
    audio_indices = torch.arange(audio_start, video_start)
    text_indices = torch.arange(num_text_tokens)

    token_tags = torch.empty(sequence_length, dtype=torch.long)
    token_tags[text_indices] = text_token_tags.to(torch.long)
    token_tags[audio_indices] = MINIMAX_H3_AUDIO_TAG
    token_tags[video_indices] = MINIMAX_H3_VIDEO_TAG

    return position_ids, token_tags, video_indices, audio_indices, text_indices, num_condition_rows


def build_row_timesteps(
    video_indices: torch.Tensor,
    audio_indices: torch.Tensor,
    num_condition_video_rows: int,
    num_condition_audio_rows: int,
    num_text_tokens: int,
    video_timestep: float,
    audio_timestep: float,
    condition_video_timestep: float,
    condition_audio_timestep: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Assign a timestep to every row and reduce to `(unique, inverse)`.

    One forward serves rows at different noise levels: the generated video and
    audio rows step down their own schedules while conditioning rows stay
    pinned at their noise-augmentation level. Text rows never reach an output
    head and inherit the video timestep.
    """
    sequence_length = int(video_indices.numel() + audio_indices.numel() + num_text_tokens)
    video_indices = video_indices.cpu()
    audio_indices = audio_indices.cpu()
    row_timesteps = torch.full((sequence_length,), video_timestep, dtype=torch.float32)
    row_timesteps[video_indices[:num_condition_video_rows]] = condition_video_timestep
    row_timesteps[audio_indices[num_condition_audio_rows:]] = audio_timestep
    row_timesteps[audio_indices[:num_condition_audio_rows]] = condition_audio_timestep
    return torch.unique(row_timesteps, sorted=True, return_inverse=True)


@register_pipeline(
    "MiniMaxH3Pipeline",
    hf_ids=["MiniMaxAI/MiniMax-H3"],
    defaults={"conditioner_offload": "auto"},
    doc="MiniMax-H3 joint video + audio generation (t2va / fl2va).",
)
class MiniMaxH3Pipeline(BasePipeline):
    """Text-to-video+audio (and first/last keyframe) with MiniMax-H3."""

    fps = MINIMAX_H3_FPS
    min_duration = 5.0
    max_duration = 15.0
    audio_channels = MINIMAX_H3_AUDIO_CHANNELS
    canvas_short_edge = 768
    canvas_max_pixels = 768 * 1344

    def __init__(self, pipeline_config):
        super().__init__(pipeline_config)

        self.vae = None
        self.audio_vae = None
        self.text_encoder = None
        self.tokenizer = None
        self.processor = None
        self.scheduler = None
        self.audio_scheduler = None

        # The transformer and the Qwen3-VL conditioner are ~62 GB each, so
        # they only fit together on a >=128 GB accelerator. ``auto`` keeps the
        # conditioner on the host and swaps it with the transformer around the
        # one encode call per request when they do not both fit.
        knob = (pipeline_config.extra_attrs or {}).get("conditioner_offload", "auto")
        if knob not in ("auto", "always", "never"):
            raise ValueError(
                f"`conditioner_offload` must be 'auto', 'always' or 'never', got {knob!r}."
            )
        self._conditioner_offload_mode = knob
        self._conditioner_offloaded = False

    # ------------------------------------------------------------------
    # Component setup
    # ------------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        return next(self.transformer.parameters()).device

    @property
    def canvas_multiple(self) -> int:
        # A canvas has to survive the VAE's spatial compression and still be
        # a whole number of patch rows wide, so the multiple is the product.
        return 16 * self.transformer.patch_size[2]

    @property
    def vae_spatial_compression_ratio(self) -> int:
        return 16

    @property
    def vae_latent_channels(self) -> int:
        return 24

    @property
    def vae_frames_per_chunk(self) -> int:
        return 17

    @property
    def vae_latents_per_chunk(self) -> int:
        return 5

    @property
    def audio_latent_channels(self) -> int:
        return 32

    @property
    def audio_sampling_rate(self) -> int:
        return 32000

    @property
    def text_encoder_layer(self) -> int:
        # MiniMax-H3 reads `hidden_states[50]`, not the final one: the last
        # layer is post-norm and is not the conditioning the released weights
        # were trained against.
        return 50

    @property
    def pixel_mean(self) -> Tuple[float, ...]:
        return (0.485, 0.456, 0.406)

    @property
    def pixel_std(self) -> Tuple[float, ...]:
        return (0.229, 0.224, 0.225)

    @property
    def default_generation_params(self) -> dict:
        """Fields the executor merges into every request.

        Key membership is also what marks a field supported during request
        validation, so every knob the pipeline honours has to appear here or
        a request that sets it is rejected before reaching ``infer()``.
        ``height``/``width`` stay ``None``: the canvas is resolved from the
        first keyframe's aspect ratio, or 16:9 for text-only requests.
        """
        return {
            "height": None,
            "width": None,
            "num_frames": MINIMAX_H3_DEFAULT_NUM_FRAMES,
            "num_inference_steps": MINIMAX_H3_DEFAULT_NUM_INFERENCE_STEPS,
            "frame_rate": self.fps,
        }

    @property
    def extra_param_specs(self) -> dict:
        """``last_image`` rides in ``extra_params``.

        ``VisualGenParams`` forbids extra fields and declares only ``image``,
        so the ``fl2va`` end keyframe has to be an explicitly declared extra
        rather than an attribute read off the params model.
        """
        return {
            "last_image": ExtraParamSchema(
                type="str",
                default=None,
                description="Last keyframe for fl2va end-frame conditioning (MiniMax-H3).",
            )
        }

    def _init_transformer(self) -> None:
        logger.info("Initializing MiniMaxH3Transformer3DModel")
        model_config = self.pipeline_config.model_configs["transformer"]
        self.transformer = MiniMaxH3Transformer3DModel(model_config)

    def load_standard_components(
        self,
        checkpoint_dir: str,
        device: torch.device,
        skip_components: Optional[list] = None,
    ) -> None:
        skip_components = skip_components or []

        if PipelineComponent.VAE not in skip_components:
            logger.info("Loading MiniMax-H3 video VAE...")
            self.vae = AutoencoderKLMiniMaxH3.from_pretrained(
                checkpoint_dir, subfolder=PipelineComponent.VAE, torch_dtype=torch.float32
            ).to(device)

        if "audio_vae" not in skip_components:
            logger.info("Loading MiniMax-H3 audio VAE...")
            self.audio_vae = AutoencoderKLMiniMaxH3Audio.from_pretrained(
                checkpoint_dir, subfolder="audio_vae", torch_dtype=torch.float32
            ).to(device)

        if PipelineComponent.TEXT_ENCODER not in skip_components:
            offload = self._resolve_conditioner_offload(device)
            where = "host (swapped in per request)" if offload else "accelerator"
            logger.info(f"Loading Qwen3-VL text encoder onto the {where}...")
            self.text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
                checkpoint_dir,
                subfolder=PipelineComponent.TEXT_ENCODER,
                torch_dtype=torch.bfloat16,
            )
            self.text_encoder.eval()
            self._conditioner_offloaded = offload
            if not offload:
                self.text_encoder = self.text_encoder.to(device)

        if PipelineComponent.TOKENIZER not in skip_components:
            logger.info("Loading Qwen2 tokenizer...")
            self.tokenizer = Qwen2TokenizerFast.from_pretrained(
                checkpoint_dir, subfolder=PipelineComponent.TOKENIZER
            )
            self.processor = Qwen3VLProcessor.from_pretrained(checkpoint_dir, subfolder="processor")

        if PipelineComponent.SCHEDULER not in skip_components:
            logger.info("Loading MiniMax-H3 schedulers...")
            self.scheduler = MiniMaxH3Scheduler.from_pretrained(
                checkpoint_dir, subfolder=PipelineComponent.SCHEDULER
            )
            self.audio_scheduler = MiniMaxH3Scheduler.from_pretrained(
                checkpoint_dir, subfolder="audio_scheduler"
            )

    def _resolve_conditioner_offload(self, device: torch.device) -> bool:
        """Decide whether the conditioner has to be swapped in per request.

        ``auto`` compares the accelerator's total memory against what the
        transformer already holds, the conditioner's own footprint and the
        video-VAE decode spike, the largest activation peak of a request. The
        released checkpoint needs ~124 GB for the transformer and conditioner
        together, so anything smaller offloads; an FP4-quantized transformer
        frees enough that the decode peak -- not the conditioner -- becomes
        the binding constraint.
        """
        if self._conditioner_offload_mode == "always":
            return True
        if self._conditioner_offload_mode == "never":
            return False
        if not torch.cuda.is_available() or torch.device(device).type != "cuda":
            return False
        total = torch.cuda.get_device_properties(device).total_memory
        resident = torch.cuda.memory_allocated(device)
        # bfloat16 Qwen3-VL conditioner plus headroom for its activations.
        needed = int(_CONDITIONER_BYTES * 1.05)
        # Video-VAE decode peak on the reference canvas (~16 GB, fp16 autocast).
        decode_headroom = _VAE_DECODE_BYTES
        fits = (total - resident) > needed + decode_headroom
        logger.info(
            f"conditioner_offload=auto: total={total / 1e9:.0f} GB resident={resident / 1e9:.0f} GB "
            f"conditioner~{needed / 1e9:.0f} GB decode+{decode_headroom / 1e9:.0f} GB "
            f"-> {'resident' if fits else 'offloaded'}"
        )
        return not fits

    @contextmanager
    def _conditioner_on_device(self, device: torch.device):
        """Swap the transformer out and the conditioner in for one encode call.

        Only one of the two ~62 GB models can be resident on a <128 GB
        accelerator, so the transformer moves to the host for the duration of
        the encode and back afterwards.
        """
        if not self._conditioner_offloaded:
            yield
            return

        transformer_was_on_device = (
            self.transformer is not None
            and next(self.transformer.parameters()).device.type == "cuda"
        )
        try:
            if transformer_was_on_device:
                self.transformer.to("cpu")
                torch.cuda.empty_cache()
            self.text_encoder.to(device)
            yield
        finally:
            self.text_encoder.to("cpu")
            torch.cuda.empty_cache()
            if transformer_was_on_device:
                self.transformer.to(device)

    def load_weights(self, weights: dict) -> None:
        if self.transformer is not None and hasattr(self.transformer, "load_weights"):
            transformer_weights = weights.get("transformer", weights)
            self.transformer.load_weights(transformer_weights)
            self.transformer.eval()

    # ------------------------------------------------------------------
    # Text conditioning
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _encode_prompt(
        self, prompt: str, keyframes: Optional[List[Image.Image]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize the prompt and encode it with the Qwen3-VL conditioner.

        For a `fl2va` request every keyframe prepends a ``"<Picture i>: "`` label
        and a vision block to the prompt presentation, and the keyframe pixels
        are passed to the conditioner as image inputs. The vision-block rows
        are tagged as video, matching the transformer's per-row AdaLN modality.

        Returns `(prompt_embeds, text_token_tags)` where `prompt_embeds` is
        `(1, num_text_tokens, text_dim)` read after decoder layer
        `text_encoder_layer`.
        """
        if not isinstance(prompt, str):
            raise ValueError(
                "MiniMax-H3 packs one request into one sequence, so `prompt` must be a single string, got "
                f"{type(prompt)}."
            )

        vision_inputs = {}
        if keyframes:
            vision = self.processor.image_processor(images=keyframes, return_tensors="pt")
            vision_inputs = {
                "pixel_values": vision["pixel_values"],
                "image_grid_thw": vision["image_grid_thw"],
            }

        # The presentation, tokenized: a `"<Picture i>: "` label and a vision
        # block per keyframe, then the prompt verbatim, with no chat template
        # and no special tokens.
        token_ids, token_tags = [], []
        if keyframes:
            merge_size = self.processor.image_processor.merge_size**2
            for index in range(len(keyframes)):
                num_image_tokens = int(vision["image_grid_thw"][index].prod()) // merge_size
                label_ids = self.tokenizer(f"<Picture {index + 1}>: ", add_special_tokens=False)[
                    "input_ids"
                ]
                vision_ids = (
                    [self.tokenizer.convert_tokens_to_ids("<|vision_start|>")]
                    + [self.tokenizer.convert_tokens_to_ids("<|image_pad|>")] * num_image_tokens
                    + [self.tokenizer.convert_tokens_to_ids("<|vision_end|>")]
                )
                token_ids += label_ids + vision_ids
                token_tags += [MINIMAX_H3_TEXT_TAG] * len(label_ids) + [MINIMAX_H3_VIDEO_TAG] * len(
                    vision_ids
                )
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        token_ids += prompt_ids
        token_tags += [MINIMAX_H3_TEXT_TAG] * len(prompt_ids)

        num_layers = self.text_encoder.config.text_config.num_hidden_layers
        if num_layers <= self.text_encoder_layer:
            raise ValueError(
                f"MiniMax-H3 conditions on `hidden_states[{self.text_encoder_layer}]` of its Qwen3-VL "
                f"conditioner, which needs more than {self.text_encoder_layer} decoder layers, but "
                f"`text_encoder` has {num_layers}."
            )

        # Resolve the accelerator before the swap: inside the context the
        # transformer sits on the host, so `self.device` would report the host.
        device = self.device
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
        mm_token_type_ids = torch.tensor(
            self.processor.create_mm_token_type_ids([token_ids]), dtype=torch.long, device=device
        )
        vision_kwargs = {}
        for name, value in vision_inputs.items():
            vision_kwargs[name] = (
                value.to(device, self.text_encoder.dtype)
                if name.startswith("pixel_")
                else value.to(device)
            )
        with self._conditioner_on_device(device):
            outputs = self.text_encoder.model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                mm_token_type_ids=mm_token_type_ids,
                use_cache=False,
                output_hidden_states=True,
                **vision_kwargs,
            )
            prompt_embeds = outputs.hidden_states[self.text_encoder_layer].to(
                device=device, dtype=self.text_encoder.dtype
            )
            del outputs
        text_token_tags = torch.tensor(token_tags, dtype=torch.long)
        return prompt_embeds, text_token_tags

    def _encode_keyframes(
        self, keyframes: List[Image.Image], encode_seed: int = 42
    ) -> List[torch.Tensor]:
        """Encode the `fl2va` keyframes into normalized conditioning latents.

        The released model's recipe, reproduced exactly: pixels are
        ImageNet-normalized, the posterior is sampled under a fresh generator
        seeded independently of the request, and the sampled latent is rounded
        to float16 before being normalized by the VAE's `latents_mean` /
        `latents_std`.
        """
        device = self.vae.device
        latents_mean = torch.tensor(self.vae.config.latents_mean, device=device).view(
            1, -1, 1, 1, 1
        )
        latents_std = torch.tensor(self.vae.config.latents_std, device=device).view(1, -1, 1, 1, 1)
        pixel_mean = torch.tensor((0.485, 0.456, 0.406), device=device).view(1, -1, 1, 1, 1)
        pixel_std = torch.tensor((0.229, 0.224, 0.225), device=device).view(1, -1, 1, 1, 1)

        conditions = []
        for keyframe in keyframes:
            # (H, W, 3) -> (1, 3, 1, H, W): batch, channels, single frame, H, W.
            pixels = torch.from_numpy(np.asarray(keyframe)).permute(2, 0, 1)[None, :, None]
            pixels = pixels.to(device).to(torch.float32).div(255.0)
            pixels = (pixels - pixel_mean) / pixel_std
            posterior = self.vae.encode(pixels, return_dict=False)[0]
            latents = posterior.sample(generator=torch.Generator().manual_seed(encode_seed))
            latents = latents.to(torch.float16).float().cpu()
            conditions.append((latents - latents_mean.cpu()) / latents_std.cpu())
        return conditions

    @staticmethod
    def _load_keyframe(keyframe, anchor: str) -> Image.Image:
        """Decode one request keyframe to RGB PIL.

        Requests carry keyframes as a path, URL, data URI, or a one-element
        list of those; a already-decoded ``PIL.Image`` is passed through so
        in-process callers can hand one over directly. An undecodable
        reference is the client's fault, so PIL's ``OSError`` becomes a
        ``ValueError`` rather than surfacing as a server fault.
        """
        if isinstance(keyframe, (list, tuple)):
            if len(keyframe) != 1:
                raise ValueError(
                    f"MiniMax-H3 takes a single {anchor} keyframe, got {len(keyframe)}."
                )
            keyframe = keyframe[0]
        try:
            return load_image(keyframe, format="pil")
        except OSError as exc:
            raise ValueError(
                f"The {anchor} keyframe could not be decoded; it may be truncated, "
                f"corrupt, or in an unsupported format: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    @nvtx_range("MiniMaxH3Pipeline.infer")
    def infer(self, req) -> PipelineOutput:
        params = req.params
        prompt = req.prompt

        seed = params.seed if params.seed is not None else 0
        generator = torch.Generator(device=self.device).manual_seed(seed)

        # `fl2va` keyframes: `image` anchors the video start, `last_image` the
        # end. The first keyframe's aspect ratio resolves the canvas when the
        # request does not. `image` arrives as a path/URL/bytes and `last_image`
        # rides in `extra_params`, so both are decoded to PIL here.
        extra = params.extra_params or {}
        supplied_keyframes = tuple(
            (anchor, kf)
            for anchor, kf in (("first", params.image), ("last", extra.get("last_image")))
            if kf is not None
        )
        keyframe_anchors = tuple(anchor for anchor, _ in supplied_keyframes)
        keyframe_images = [self._load_keyframe(kf, anchor) for anchor, kf in supplied_keyframes]

        height = params.height
        width = params.width
        if (height is None) != (width is None):
            raise ValueError("`height` and `width` have to be passed together, or neither of them.")
        if height is None:
            if keyframe_images:
                height, width = resolve_canvas_size(
                    *keyframe_images[0].size,
                    self.canvas_multiple,
                    self.canvas_short_edge,
                    self.canvas_max_pixels,
                )
            else:
                height, width = resolve_canvas_size(
                    16, 9, self.canvas_multiple, self.canvas_short_edge, self.canvas_max_pixels
                )
        if height % self.canvas_multiple or width % self.canvas_multiple:
            raise ValueError(
                f"`height` and `width` must be multiples of {self.canvas_multiple}, got {height}x{width}."
            )

        keyframes = [
            fit_keyframe_to_canvas(keyframe, width, height, stretch=index == 0)
            for index, keyframe in enumerate(keyframe_images)
        ]

        num_frames = params.num_frames if params.num_frames is not None else 124
        num_inference_steps = (
            params.num_inference_steps if params.num_inference_steps is not None else 50
        )

        frames_per_chunk = self.vae_frames_per_chunk
        latents_per_chunk = self.vae_latents_per_chunk
        aligned_num_frames = align_num_frames(num_frames, frames_per_chunk, latents_per_chunk)
        duration = aligned_num_frames / self.fps
        if not self.min_duration <= duration <= self.max_duration:
            raise ValueError(
                f"MiniMax-H3 generates between {self.min_duration} and {self.max_duration} seconds at "
                f"{self.fps} fps, got {num_frames} frames (rounded up to {aligned_num_frames})."
            )
        if aligned_num_frames != num_frames:
            logger.info(
                f"Rounding `num_frames` from {num_frames} up to {aligned_num_frames} for the video VAE."
            )
            num_frames = aligned_num_frames

        ratio = self.vae_spatial_compression_ratio
        num_latent_frames = video_latent_num_frames(num_frames, frames_per_chunk, latents_per_chunk)
        latent_height = height // ratio
        latent_width = width // ratio
        num_audio_latents = audio_latent_num_frames(num_frames)

        # 1. Text conditioning. For `fl2va` the keyframes are presented as
        # vision blocks in the prompt and additionally encoded into latent
        # conditioning rows.
        prompt_embeds, text_token_tags = self._encode_prompt(prompt, keyframes or None)
        condition_latents = self._encode_keyframes(keyframes) if keyframes else []

        # 2. Packed layout: [text | keyframe conditions | target audio | target video].
        (
            position_ids,
            token_tags,
            video_indices,
            audio_indices,
            text_indices,
            num_condition_video_rows,
        ) = build_packed_sequence(
            text_token_tags,
            num_latent_frames,
            latent_height,
            latent_width,
            num_audio_latents,
            self.transformer.patch_size,
            keyframe_anchors=keyframe_anchors,
        )
        device = self.device
        position_ids = position_ids.to(device)
        token_tags = token_tags.to(device)
        video_indices = video_indices.to(device)
        audio_indices = audio_indices.to(device)
        text_indices = text_indices.to(device)

        # 3. Schedules: the video noise as a latent tensor first, then the
        # audio noise directly in row layout, both off the request's
        # generator, in that order.
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        self.audio_scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        audio_timesteps = self.audio_scheduler.timesteps

        # 4. Conditioning rows: one noise draw per keyframe, in packed order,
        # before the generated rows' noise. The anchors are not fully clean —
        # the released model noises them to `t = _KEYFRAME_NOISE_AUG` and holds
        # them there for every step.
        condition_rows = []
        for condition in condition_latents:
            noise = randn_tensor(
                condition.shape, generator=generator, device=device, dtype=torch.float32
            )
            noised = self.scheduler.scale_noise(condition.to(device), _KEYFRAME_NOISE_AUG, noise)
            condition_rows.append(patchify_video_latents(noised, self.transformer.patch_size))

        latents = randn_tensor(
            (1, self.vae_latent_channels, num_latent_frames, latent_height, latent_width),
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        video_rows = patchify_video_latents(latents, self.transformer.patch_size)
        if condition_rows:
            video_rows = torch.cat(condition_rows + [video_rows])
        audio_rows = randn_tensor(
            (num_audio_latents * self.audio_channels, self.audio_latent_channels),
            generator=generator,
            device=device,
            dtype=torch.float32,
        )

        # 4. Denoise loop: one forward pass per step over the packed
        # sequence; the video and audio rows step down their own schedules.
        for i, (timestep, audio_timestep) in enumerate(zip(timesteps, audio_timesteps)):
            unique_timesteps, timestep_indices = build_row_timesteps(
                video_indices,
                audio_indices,
                num_condition_video_rows,
                0,
                text_indices.numel(),
                float(timestep),
                float(audio_timestep),
                max(float(timestep), _KEYFRAME_NOISE_AUG),
                1.0,
            )
            # The row plan is built on the CPU (the sequence layout is fp64
            # bookkeeping); the transformer consumes it on the accelerator.
            unique_timesteps = unique_timesteps.to(device)
            timestep_indices = timestep_indices.to(device)
            video_velocity, audio_velocity = self.transformer(
                hidden_states=video_rows[None],
                audio_hidden_states=audio_rows[None],
                encoder_hidden_states=prompt_embeds,
                timestep=unique_timesteps,
                timestep_indices=timestep_indices,
                token_tags=token_tags,
                position_ids=position_ids,
                video_indices=video_indices,
                audio_indices=audio_indices,
                text_indices=text_indices,
            )
            # In-place: only the generated rows are written, so the
            # conditioning anchors survive the whole loop untouched.
            video_rows[num_condition_video_rows:] = self.scheduler.step(
                video_velocity[0, num_condition_video_rows:].float(),
                timestep,
                video_rows[num_condition_video_rows:],
                return_dict=False,
            )[0]
            audio_rows = self.audio_scheduler.step(
                audio_velocity[0].float(),
                audio_timestep,
                audio_rows,
                return_dict=False,
            )[0]

        # 5. Unpack the denoised rows back into latents. The conditioning rows
        # rode through the loop untouched (only the generated rows were ever
        # written) and are dropped here — only the generated rows are decoded.
        video_latents = unpatchify_video_latents(
            video_rows[num_condition_video_rows:],
            num_latent_frames,
            latent_height,
            latent_width,
            self.vae_latent_channels,
            self.transformer.patch_size,
        )
        audio_latents = (
            audio_rows.reshape(self.audio_channels, num_audio_latents, audio_rows.shape[-1])
            .permute(0, 2, 1)
            .contiguous()
        )

        # 6. Decode. The video VAE runs under float16 autocast even though its
        # weights are float32, and it produces ImageNet-normalized RGB that is
        # reverted here. The audio VAE is mono and takes the two stereo
        # channels as two batch items.
        latents_mean = torch.tensor(self.vae.config.latents_mean, device=device).view(
            1, -1, 1, 1, 1
        )
        latents_std = torch.tensor(self.vae.config.latents_std, device=device).view(1, -1, 1, 1, 1)
        video_latents = video_latents * latents_std + latents_mean

        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
        ):
            video = self.vae.decode(video_latents, return_dict=False)[0]
        pixel_mean = torch.tensor(self.pixel_mean, device=device).view(1, -1, 1, 1, 1)
        pixel_std = torch.tensor(self.pixel_std, device=device).view(1, -1, 1, 1, 1)
        video = (video.float() * pixel_std + pixel_mean).clamp(0, 1)

        audio_latents_mean = torch.tensor(self.audio_vae.config.latents_mean, device=device).view(
            1, -1, 1
        )
        audio_latents_std = torch.tensor(self.audio_vae.config.latents_std, device=device).view(
            1, -1, 1
        )
        audio_latents = audio_latents * audio_latents_std + audio_latents_mean
        audio = self.audio_vae.decode(audio_latents, return_dict=False)[0]
        audio = audio.float().permute(1, 0, 2)

        # 7. Video frames to uint8, [B, T, H, W, C].
        video = (video.permute(0, 2, 3, 4, 1) * 255.0).to(torch.uint8)

        return PipelineOutput(
            video=video,
            frame_rate=self.fps,
            audio=audio,
            audio_sample_rate=self.audio_sampling_rate,
        )
