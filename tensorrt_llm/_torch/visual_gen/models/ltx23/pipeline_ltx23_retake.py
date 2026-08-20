# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native LTX-2.3 retake workflow for VisualGen.

This pipeline keeps a resident retake model for ``trtllm-serve`` and extends the
native LTX-2.3 pipeline with source-media encoders and masked denoising.
It either loads Gemma for a live prompt or consumes precomputed post-connector
conditioning. Retake generalizes image-to-video masking to condition both sides
of the regenerated window.

Source video/audio decode, stream metadata, audio/video VAE encode, diffusion,
and VAE decode all use TensorRT-LLM native components.
"""

from __future__ import annotations

import copy
import math
from typing import TYPE_CHECKING, Any

import safetensors.torch
import torch

from tensorrt_llm._torch.visual_gen.output import CudaPhaseTimer, PipelineOutput
from tensorrt_llm._torch.visual_gen.pipeline import ExtraParamSchema
from tensorrt_llm._torch.visual_gen.pipeline_registry import PipelineComponent, register_pipeline
from tensorrt_llm._torch.visual_gen.utils import postprocess_video_tensor
from tensorrt_llm.inputs.multimodal_data import AudioData
from tensorrt_llm.logger import logger

from ..ltx2.ltx2_core.patchifier import get_pixel_coords
from ..ltx2.ltx2_core.scheduler_adapter import NativeSchedulerAdapter
from ..ltx2.ltx2_core.types import (
    VIDEO_SCALE_FACTORS,
    AudioLatentShape,
    VideoLatentShape,
    VideoPixelShape,
)
from ..ltx2.ltx2_core.video_vae import SpatialTilingConfig, TemporalTilingConfig, TilingConfig
from ..ltx2.pipeline_ltx2 import _load_component_weights
from .ltx23_core.audio_vae import AudioEncoderConfigurator, encode_audio
from .ltx23_core.modality import LTX23Modality
from .ltx23_core.video_vae_ltx23 import LTX23VideoEncoderConfigurator
from .media_io import (
    decode_audio_from_file,
    decode_video_by_frame,
    get_videostream_metadata,
    pad_audio_to_video_duration,
)
from .pipeline_ltx23 import LTX23Pipeline
from .text_conditioning_ltx23 import LTX23TextConditioning

if TYPE_CHECKING:
    from tensorrt_llm._torch.visual_gen.config import DiffusionPipelineConfig
    from tensorrt_llm._torch.visual_gen.executor import DiffusionRequest

# Distilled retake noise schedule (8 Euler steps).
_RETAKE_DISTILLED_SIGMA_VALUES = [
    1.0,
    0.99375,
    0.9875,
    0.98125,
    0.975,
    0.909375,
    0.725,
    0.421875,
    0.0,
]
_RETAKE_NUM_INFERENCE_STEPS = len(_RETAKE_DISTILLED_SIGMA_VALUES) - 1

_DEFAULT_MAX_SEQUENCE_LENGTH = 1024
_PROMPT_CONDITIONING_KEYS = ("video_embeds", "audio_embeds", "connector_mask")

# Retake uses the same tiling geometry for source encode and output decode. Keep
# it separate from the generation pipeline's default because tile boundaries are
# part of the retake numerical contract.
_RETAKE_TILING_CONFIG = TilingConfig(
    spatial_config=SpatialTilingConfig(tile_size_in_pixels=768, tile_overlap_in_pixels=64),
    temporal_config=TemporalTilingConfig(tile_size_in_frames=80, tile_overlap_in_frames=24),
)


def _load_prompt_conditioning(path: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load precomputed post-connector prompt tensors from safetensors."""
    tensors = safetensors.torch.load_file(path, device="cpu")
    keys = set(tensors)
    required = set(_PROMPT_CONDITIONING_KEYS)
    if keys != required:
        raise ValueError(
            "Prompt-conditioning tensor keys do not match the expected schema: "
            f"missing={sorted(required - keys)}, unexpected={sorted(keys - required)}"
        )

    video_embeds, audio_embeds, connector_mask = (tensors[key] for key in _PROMPT_CONDITIONING_KEYS)
    if video_embeds.ndim != 3 or audio_embeds.ndim != 3:
        raise ValueError(
            "Prompt embeddings must be rank 3 [batch, sequence, channels]; "
            f"got video={tuple(video_embeds.shape)}, audio={tuple(audio_embeds.shape)}"
        )
    if connector_mask.ndim != 4:
        raise ValueError(
            "Connector mask must be rank 4 [batch, 1, 1, sequence]; "
            f"got {tuple(connector_mask.shape)}"
        )
    if video_embeds.shape[0] != 1:
        raise ValueError(
            "LTX-2.3 retake prompt conditioning must contain one prompt; "
            f"got batch size {video_embeds.shape[0]}."
        )
    if video_embeds.shape[:2] != audio_embeds.shape[:2] or (
        connector_mask.shape[0] != video_embeds.shape[0]
        or connector_mask.shape[-1] != video_embeds.shape[1]
    ):
        raise ValueError(
            "Prompt-conditioning batch and sequence dimensions do not align: "
            f"video={tuple(video_embeds.shape)}, audio={tuple(audio_embeds.shape)}, "
            f"mask={tuple(connector_mask.shape)}"
        )
    return video_embeds, audio_embeds, connector_mask


def _retake_pixel_window(
    start_time: float, end_time: float, fps: float, num_frames: int
) -> tuple[int, int]:
    """Half-open source pixel-frame window ``[start, end)`` for a retake window.

    Frames are indexed by ``round(time * fps)`` and clamped to ``[0, num_frames]``
    with ``start <= end`` so out-of-range or inverted times are safe (an inverted
    or degenerate window yields an empty ``[start, start)`` span).
    """
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    if num_frames < 0:
        raise ValueError(f"num_frames must be non-negative, got {num_frames}")
    start = max(0, min(int(round(start_time * fps)), num_frames))
    end = max(start, min(int(round(end_time * fps)), num_frames))
    return start, end


def _resolve_retake_time_window(
    start_time: float | None,
    end_time: float | None,
    fps: float,
    num_frames: int,
) -> tuple[float, float]:
    """Resolve an optional retake window against the source video duration."""
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    if num_frames < 0:
        raise ValueError(f"num_frames must be non-negative, got {num_frames}")

    resolved_start = 0.0 if start_time is None else float(start_time)
    resolved_end = num_frames / fps if end_time is None else float(end_time)
    if not math.isfinite(resolved_start) or not math.isfinite(resolved_end):
        raise ValueError("retake_start_time and retake_end_time must be finite.")
    if resolved_start >= resolved_end:
        raise ValueError(
            f"retake_start_time ({resolved_start}) must be less than "
            f"retake_end_time ({resolved_end})"
        )
    return resolved_start, resolved_end


def _pixel_frame_to_latent_index(pixel_frame: int, temporal_ratio: int) -> int:
    """Latent-frame index of a source pixel frame under the causal LTX-2.3 VAE.

    Pixel frame 0 maps to latent frame 0; pixel frames ``[1+(i-1)*r, 1+i*r)`` map
    to latent frame ``i`` (``r = temporal_ratio``), i.e. ``(f - 1)//r + 1``.
    """
    if pixel_frame <= 0:
        return 0
    return (pixel_frame - 1) // temporal_ratio + 1


def _latent_frame_count(num_frames: int, temporal_ratio: int) -> int:
    """Latent frame count for ``num_frames`` pixel frames: ``(T-1)//r + 1``."""
    return (num_frames - 1) // temporal_ratio + 1


def _retake_conditioned_latent_ranges(
    pixel_start: int, pixel_end: int, num_frames: int, temporal_ratio: int
) -> list[tuple[int, int]]:
    """Return conditioned latent ranges outside a pixel retake window.

    Any latent frame touched by ``[pixel_start, pixel_end)`` is regenerated.
    Leading and trailing latent ranges are returned for conditioning.

    A full-frame window returns ``[]``; an empty window returns ``[(0, L)]``.
    """
    if temporal_ratio <= 0:
        raise ValueError(f"temporal_ratio must be positive, got {temporal_ratio}")
    if num_frames < 0:
        raise ValueError(f"num_frames must be non-negative, got {num_frames}")
    total_latent = _latent_frame_count(num_frames, temporal_ratio)
    if pixel_end <= pixel_start:
        return [(0, total_latent)]
    lat_start = max(0, min(_pixel_frame_to_latent_index(pixel_start, temporal_ratio), total_latent))
    lat_end = _pixel_frame_to_latent_index(pixel_end - 1, temporal_ratio) + 1
    lat_end = max(lat_start, min(lat_end, total_latent))
    cond_ranges = []
    if lat_start > 0:
        cond_ranges.append((0, lat_start))
    if lat_end < total_latent:
        cond_ranges.append((lat_end, total_latent))
    return cond_ranges


def _init_retake_patchified_latents(
    noise_latents: torch.Tensor,
    source_latents: torch.Tensor,
    denoise_mask: torch.Tensor,
) -> torch.Tensor:
    """Initialize retake latents in the denoiser's patchified token layout.

    Sampling in 5D and then patchifying produces a different seeded random field
    because the memory order changes, so noise is sampled directly in token
    layout.
    """
    if noise_latents.shape != source_latents.shape:
        raise ValueError(
            f"noise/source latent shape mismatch: {tuple(noise_latents.shape)} vs "
            f"{tuple(source_latents.shape)}"
        )
    if denoise_mask.dim() == 2:
        mask = denoise_mask.unsqueeze(-1)
    elif denoise_mask.dim() == 3:
        mask = denoise_mask
    else:
        raise ValueError(f"expected denoise mask rank 2 or 3, got {tuple(denoise_mask.shape)}")
    if mask.shape[:2] != source_latents.shape[:2] or mask.shape[-1] != 1:
        raise ValueError(
            f"denoise mask shape {tuple(denoise_mask.shape)} is not compatible with "
            f"patchified latents {tuple(source_latents.shape)}"
        )
    return torch.lerp(source_latents.float(), noise_latents.float(), mask.float()).to(
        source_latents.dtype
    )


def _conform_latent_length(latent: torch.Tensor, expected_frames_count: int) -> torch.Tensor:
    """Crop or zero-pad *latent* along dim 2 so it has exactly the expected frames.

    Encoders emit a frame count driven by the decoded stream length, which need
    not match the count the target shape implies; the transformer needs the exact
    count. Missing audio latent frames carry no conditioning, so they are padded
    with zeros.
    """
    actual_frames = latent.shape[2]
    if actual_frames > expected_frames_count:
        return latent[:, :, :expected_frames_count]
    if actual_frames < expected_frames_count:
        pad_shape = list(latent.shape)
        pad_shape[2] = expected_frames_count - actual_frames
        pad = torch.zeros(pad_shape, device=latent.device, dtype=latent.dtype)
        return torch.cat([latent, pad], dim=2)
    return latent


# Retake shares checkpoints with LTX23Pipeline and is selected by ``workflow``.
@register_pipeline("LTX23RetakePipeline", doc="Native LTX-2.3 video retake pipeline.")
class LTX23RetakePipeline(LTX23Pipeline):
    """Persistent VisualGen pipeline for native LTX-2.3 retake requests."""

    def __init__(self, pipeline_config: "DiffusionPipelineConfig") -> None:
        if pipeline_config.cache_backend is not None:
            raise NotImplementedError("Cache acceleration is not supported for LTX-2.3 retake.")
        if pipeline_config.parallel.cfg_size != 1:
            raise NotImplementedError("LTX-2.3 retake requires parallel.cfg_size=1.")
        if pipeline_config.parallel.parallel_vae_size != 1:
            raise NotImplementedError("Parallel VAE is not supported for LTX-2.3 retake.")
        if pipeline_config.parallel.seq_parallel_size != 1 or pipeline_config.parallel.tp_size != 1:
            raise NotImplementedError("LTX-2.3 retake currently supports a single GPU only.")
        if pipeline_config.cuda_graph.enable:
            raise NotImplementedError("LTX-2.3 retake currently requires cuda_graph.enable=false.")
        super().__init__(pipeline_config)

    @property
    def default_generation_params(self) -> dict[str, int]:
        return {
            "num_inference_steps": _RETAKE_NUM_INFERENCE_STEPS,
            "seed": 42,
            "max_sequence_length": _DEFAULT_MAX_SEQUENCE_LENGTH,
        }

    @property
    def extra_param_specs(self) -> dict[str, ExtraParamSchema]:
        return {
            "retake_video_path": ExtraParamSchema(
                type="str",
                description="Path to the source video file for retake.",
            ),
            "retake_prompt_conditioning_path": ExtraParamSchema(
                type="str",
                default=None,
                description="Path to precomputed post-connector prompt conditioning.",
            ),
            "retake_start_time": ExtraParamSchema(
                type="float",
                default=0.0,
                description=(
                    "Start time in seconds for the regenerated window. "
                    "Defaults to the beginning of the video."
                ),
            ),
            "retake_end_time": ExtraParamSchema(
                type="float",
                default=None,
                description=(
                    "End time in seconds for the regenerated window. "
                    "Defaults to the end of the video."
                ),
            ),
        }

    def _load_native_components(
        self,
        config: dict[str, Any],
        safetensors_paths: list[str],
        device: torch.device,
        dtype: torch.dtype,
        skip_components: list | None = None,
    ) -> None:
        """Extend the LTX-2.3 components with source-media encoders."""
        skip_components = skip_components or []
        super()._load_native_components(
            config,
            safetensors_paths,
            device,
            dtype,
            skip_components,
        )

        if "video_encoder" not in skip_components:
            encoder_blocks = config.get("vae", {}).get("encoder_blocks", [])
            if not encoder_blocks:
                raise ValueError("LTX-2.3 checkpoint config has no video VAE encoder blocks.")
            self.video_encoder = LTX23VideoEncoderConfigurator.from_config(config)
            _load_component_weights(safetensors_paths, self.video_encoder, "vae.encoder.")
            _load_component_weights(
                safetensors_paths,
                self.video_encoder.per_channel_statistics,
                "vae.per_channel_statistics.",
            )
            self.video_encoder = self.video_encoder.to(device=device, dtype=dtype)
        else:
            self.video_encoder = None

        self._audio_encoder = AudioEncoderConfigurator.from_config(config)
        _load_component_weights(safetensors_paths, self._audio_encoder, "audio_vae.encoder.")
        _load_component_weights(
            safetensors_paths,
            self._audio_encoder.per_channel_statistics,
            "audio_vae.per_channel_statistics.",
        )
        self._audio_encoder = self._audio_encoder.to(device=device, dtype=dtype)
        self.audio_patchifier = self._audio_encoder.patchifier

    def _build_denoise_mask(
        self,
        video_shape: VideoLatentShape,
        *,
        cond_latent_frame_ranges: list[tuple[int, int]],
    ) -> torch.Tensor:
        patch_t = self.video_patchifier.patch_size[0]
        grid_frames = video_shape.frames // patch_t
        total_tokens = self.video_patchifier.get_token_count(video_shape)
        tokens_per_frame = total_tokens // grid_frames
        mask = torch.ones(
            1,
            total_tokens,
            device=self.device,
            dtype=torch.float32,
        )
        for start, stop in cond_latent_frame_ranges:
            start = max(0, start)
            stop = min(grid_frames, stop)
            if stop > start:
                mask[:, start * tokens_per_frame : stop * tokens_per_frame] = 0.0
        return mask

    def _init_transformer(self) -> None:
        model_config = self.pipeline_config.model_configs["transformer"]
        quant_config = getattr(model_config, "quant_config", None)
        if quant_config is not None and getattr(quant_config, "quant_algo", None) is not None:
            raise NotImplementedError("LTX-2.3 retake currently supports only BF16.")
        super()._init_transformer()

    def _masked_transformer_step(
        self,
        video_latents: torch.Tensor,
        audio_latents: torch.Tensor | None,
        step_index: int,
        timestep: torch.Tensor,
        *,
        video_positions: torch.Tensor,
        audio_positions: torch.Tensor | None,
        denoise_mask: torch.Tensor,
        clean_latent: torch.Tensor,
        num_steps: int,
        text_cache: LTX23TextConditioning,
    ) -> torch.Tensor:
        """Run one LTX-2.3 denoise step with two-sided video conditioning."""
        video_float = video_latents.float()
        video_model = video_latents.to(self.dtype)
        audio_model = audio_latents.to(self.dtype) if audio_latents is not None else None
        video_timestep = (denoise_mask * timestep.unsqueeze(-1)).to(self.dtype)
        sigma = timestep.to(self.dtype)
        video = LTX23Modality(
            latent=video_model,
            timesteps=video_timestep,
            positions=video_positions,
            context=text_cache.video_context,
            sigma=sigma,
            cross_modality_sigma=sigma if audio_model is not None else None,
        )
        audio = None
        if audio_model is not None:
            if audio_positions is None:
                raise ValueError("Audio positions are required when audio latents are present.")
            audio = LTX23Modality(
                latent=audio_model,
                timesteps=torch.zeros_like(timestep),
                positions=audio_positions,
                context=text_cache.audio_context,
                sigma=sigma,
                cross_modality_sigma=sigma,
            )

        velocity_video, _ = self.transformer(
            video=video,
            audio=audio,
            text_cache=text_cache,
            timestep=timestep.new_tensor(float(step_index) / num_steps),
            step_index=step_index,
        )
        if velocity_video is None:
            raise RuntimeError("LTX-2.3 retake transformer returned no video prediction.")
        expanded_sigma = sigma.float()
        while expanded_sigma.ndim < velocity_video.ndim:
            expanded_sigma = expanded_sigma.unsqueeze(-1)
        denoised_video = (video_float - velocity_video.float() * expanded_sigma).to(self.dtype)
        blend = denoise_mask.unsqueeze(-1).float()
        return (denoised_video.float() * blend + clean_latent.float() * (1.0 - blend)).to(
            self.dtype
        )

    def load_standard_components(
        self,
        checkpoint_dir: str,
        device: torch.device,
        skip_components: list | None = None,
        *,
        text_encoder_path: str = "",
        **kwargs: Any,
    ) -> None:
        native_skip_components = list(skip_components or [])
        native_skip_components += [PipelineComponent.SCHEDULER, "audio_decoder", "vocoder"]
        if not text_encoder_path:
            native_skip_components.extend(
                [
                    PipelineComponent.TOKENIZER,
                    PipelineComponent.TEXT_ENCODER,
                    "connectors",
                ]
            )
        super().load_standard_components(
            checkpoint_dir,
            device,
            text_encoder_path=text_encoder_path,
            skip_components=list(dict.fromkeys(native_skip_components)),
        )

    def warmup(self) -> None:
        logger.info("Skipping LTX-2.3 retake warmup; retake requires a source video request.")

    @torch.inference_mode()
    def infer(self, req: "DiffusionRequest") -> PipelineOutput:
        extra = req.params.extra_params or {}
        video_path = self._require_extra(extra, "retake_video_path")
        start_time = extra.get("retake_start_time")
        end_time = extra.get("retake_end_time")

        prompt = self._single_prompt(req.prompt)
        timer = CudaPhaseTimer()
        timer.mark_pre_start()
        video, audio, output_shape = self._run_retake(
            req, video_path, start_time, end_time, prompt, timer=timer
        )
        audio_tensor = audio.samples if audio is not None else None
        sample_rate = audio.sample_rate if audio is not None else None
        if audio_tensor is not None and sample_rate is not None:
            audio_tensor = pad_audio_to_video_duration(
                audio_tensor,
                num_frames=int(video.shape[1]),
                frame_rate=float(output_shape.fps),
                sample_rate=sample_rate,
            )
        return timer.fill(
            PipelineOutput(
                video=video,
                audio=audio_tensor,
                frame_rate=float(output_shape.fps),
                audio_sample_rate=sample_rate,
            )
        )

    def _prepare_prompt_conditioning(
        self,
        prompt: str,
        max_sequence_length: int,
        conditioning_path: str | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return supplied prompt conditioning or encode the prompt with Gemma."""
        if conditioning_path is not None:
            if prompt:
                raise ValueError(
                    "LTX-2.3 retake accepts either a prompt or precomputed prompt "
                    "conditioning, not both."
                )
            video_embeds, audio_embeds, connector_mask = _load_prompt_conditioning(
                conditioning_path
            )
            logger.info(f"Using LTX-2.3 retake prompt conditioning from {conditioning_path}.")
            return (
                video_embeds.to(device=self.device, dtype=self.dtype),
                audio_embeds.to(device=self.device, dtype=self.dtype),
                connector_mask.to(device=self.device, dtype=self.dtype),
            )

        text_encoder = getattr(self, "text_encoder", None)
        if text_encoder is None:
            raise RuntimeError("Gemma is not loaded for LTX-2.3 retake prompt encoding.")
        prompt_embeds, prompt_attention_mask = self._encode_prompt(
            prompt,
            max_sequence_length=max_sequence_length,
        )
        return self._process_connectors(prompt_embeds, prompt_attention_mask)

    def _run_retake(
        self,
        req: "DiffusionRequest",
        video_path: str,
        start_time: float | None,
        end_time: float | None,
        prompt: str,
        timer: CudaPhaseTimer,
    ) -> tuple[torch.Tensor, AudioData | None, VideoPixelShape]:
        """Run native source encode, masked denoise, and full-clip decode.

        Uses the native video VAE, patchifier, scheduler, prompt conditioning,
        and masked-transformer step with
        retake-specific inputs: initial latents seeded from the encoded source,
        a two-sided ``denoise_mask`` conditioning the leading + trailing context
        while regenerating the middle window, and a two-sided ``clean_latent``
        from the source. The entire decoded clip is returned, and the source
        audio is passed through unchanged (video-only regeneration).

        Returns ``(video_uint8, source_audio, output_shape)`` where
        ``video_uint8`` is ``(1, T, H, W, C)`` uint8.
        """
        if req.params.num_inference_steps != _RETAKE_NUM_INFERENCE_STEPS:
            raise ValueError(
                "LTX-2.3 native retake uses the fixed distilled "
                f"{_RETAKE_NUM_INFERENCE_STEPS}-step schedule; "
                f"got num_inference_steps={req.params.num_inference_steps}."
            )

        device = self.device
        dtype = self.dtype
        seed = req.params.seed
        generator = torch.Generator(device=device).manual_seed(seed)

        # ---- 1. Source read + validation --------------------------------
        pixel_shape = get_videostream_metadata(video_path)
        num_frames = int(pixel_shape.frames)
        height = int(pixel_shape.height)
        width = int(pixel_shape.width)
        fps = float(pixel_shape.fps)
        self._validate_retake_source(num_frames, height, width)

        start_time, end_time = _resolve_retake_time_window(start_time, end_time, fps, num_frames)

        source_norm_5d = self._read_source_video(
            video_path, num_frames, height, width, device, dtype
        )

        # ---- 2. Retake windows ------------------------------------------
        temporal_ratio = VIDEO_SCALE_FACTORS.time
        pixel_start, pixel_end = _retake_pixel_window(start_time, end_time, fps, num_frames)
        if pixel_start == pixel_end:
            raise ValueError(
                "The retake time window does not contain a source frame after "
                f"conversion at {fps:g} FPS: [{start_time}, {end_time})."
            )
        conditioned_latent_ranges = _retake_conditioned_latent_ranges(
            pixel_start, pixel_end, num_frames, temporal_ratio
        )

        # ---- 3. Native VAE encode + seed initial latents ----------------
        video_shape = VideoLatentShape.from_pixel_shape(
            pixel_shape, latent_channels=self.transformer_in_channels
        )
        # Source latents and sampled noise start in the model dtype, while the
        # blend/math path widens internally.
        if self.video_encoder is None:
            raise RuntimeError("LTX-2.3 retake requires native video VAE encoder weights.")
        source_window_latents = self.video_encoder.tiled_encode(
            source_norm_5d, _RETAKE_TILING_CONFIG
        )
        expected_latent_shape = tuple(video_shape.to_torch_shape())
        if tuple(source_window_latents.shape) != expected_latent_shape:
            raise ValueError(
                "LTX-2.3 native retake: encoded source latent shape "
                f"{tuple(source_window_latents.shape)} != expected "
                f"{expected_latent_shape}; check source resolution/frame count."
            )

        denoise_mask = self._build_denoise_mask(
            video_shape, cond_latent_frame_ranges=conditioned_latent_ranges
        )
        source_patch_latents = self.video_patchifier.patchify(source_window_latents)
        noise_latents = torch.randn(
            source_patch_latents.shape,
            generator=generator,
            device=device,
            dtype=dtype,
        )
        # Denoising carries the latent in the model dtype and casts it back after
        # every Euler step; retaining the float32 blend would change the trajectory.
        latents = _init_retake_patchified_latents(
            noise_latents, source_patch_latents, denoise_mask
        ).to(dtype)

        # Keep the full encoded source in patchified layout. Tokens with
        # denoise_mask=1 ignore it; conditioned tokens use it after every
        # transformer prediction.
        clean_latent = source_patch_latents

        # ---- 4. Native prompt conditioning + text cache -----------------
        max_sequence_length = (
            getattr(req.params, "max_sequence_length", None) or _DEFAULT_MAX_SEQUENCE_LENGTH
        )
        conditioning_path = (req.params.extra_params or {}).get("retake_prompt_conditioning_path")
        video_embeds, audio_embeds, connector_mask = self._prepare_prompt_conditioning(
            prompt, max_sequence_length, conditioning_path
        )

        video_positions = self.video_patchifier.get_patch_grid_bounds(video_shape, device=device)
        video_positions = get_pixel_coords(
            video_positions.float(), VIDEO_SCALE_FACTORS, causal_fix=True
        )
        video_positions[:, 0, ...] = video_positions[:, 0, ...] / fps
        # RoPE positions remain float32 because position error is amplified by
        # the distilled schedule's large final Euler steps.

        # ---- Frozen audio conditioning ----------------------------------
        # Encode the source audio as a frozen conditioning modality. Video-only
        # retake never regenerates the audio.
        source_audio = decode_audio_from_file(video_path)
        audio_latent = self._encode_source_audio_latent(
            source_audio,
            pixel_shape,
        )
        # The frozen audio follows a noise-to-clean trajectory on the same sigma
        # schedule as the video rather than remaining constant.
        if audio_latent is not None:
            audio_shape = AudioLatentShape.from_torch_shape(audio_latent.shape)
            audio_clean_latents = self.audio_patchifier.patchify(audio_latent.float())
            audio_positions = self.audio_patchifier.get_patch_grid_bounds(
                audio_shape, device=device
            )
        else:
            audio_clean_latents = None
            audio_embeds = None
            audio_positions = None

        text_cache = self.transformer.prepare_text_cache(
            video_context=video_embeds,
            video_context_mask=connector_mask,
            video_positions=video_positions,
            audio_context=audio_embeds,
            audio_context_mask=connector_mask if audio_embeds is not None else None,
            audio_positions=audio_positions,
            dtype=dtype,
        )

        # ---- 5. Native masked video denoise (distilled, non-guided) --------
        scheduler = NativeSchedulerAdapter()
        scheduler.sigmas = torch.tensor(
            _RETAKE_DISTILLED_SIGMA_VALUES, dtype=torch.float32, device=device
        )
        timesteps = scheduler.timesteps
        num_steps = len(timesteps)

        # Denoise frozen audio toward its clean x0 target on the video sigma
        # schedule. This keeps cross-attention noise-matched without predicting
        # an audio velocity with the transformer.
        extra_streams = None
        clean_audio = None
        if audio_clean_latents is not None:
            clean_audio = audio_clean_latents.to(dtype)
            audio_noise = torch.randn(
                clean_audio.shape, generator=generator, device=device, dtype=dtype
            )
            sigma = scheduler.sigmas[0]
            audio_state = (audio_noise * sigma + clean_audio.float() * (1.0 - sigma)).to(dtype)
            audio_scheduler = copy.deepcopy(scheduler)
            extra_streams = {
                "audio": (audio_state, audio_scheduler),
            }

        def retake_forward_fn(
            video_latents: torch.Tensor,
            extra_stream_latents: dict[str, torch.Tensor] | None,
            step_index: int,
            timestep: torch.Tensor,
            _encoder_hidden_states: torch.Tensor,
            _extra_tensors: dict[str, Any],
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
            audio_latents = (
                extra_stream_latents.get("audio") if extra_stream_latents is not None else None
            )
            denoised_video = self._masked_transformer_step(
                video_latents,
                audio_latents,
                step_index,
                timestep,
                video_positions=video_positions,
                audio_positions=audio_positions,
                denoise_mask=denoise_mask,
                clean_latent=clean_latent,
                num_steps=num_steps,
                text_cache=text_cache,
            )
            audio_prediction = {"audio": clean_audio} if clean_audio is not None else {}
            return denoised_video, audio_prediction

        timer.mark_denoise_start()
        denoise_result = self.denoise(
            latents=latents,
            scheduler=scheduler,
            prompt_embeds=video_embeds,
            guidance_scale=1.0,
            forward_fn=retake_forward_fn,
            timesteps=timesteps,
            extra_streams=extra_streams,
        )
        denoised_latents = denoise_result[0] if extra_streams is not None else denoise_result

        # ---- 6. Native decode -------------------------------------------
        timer.mark_post_start()
        video_latents_5d = self.video_patchifier.unpatchify(denoised_latents, video_shape).to(dtype)
        # Encode and decode must share the retake tiling geometry.
        chunks = list(
            self.video_decoder.tiled_decode(
                video_latents_5d, _RETAKE_TILING_CONFIG, generator=generator
            )
        )
        decoded = torch.cat(chunks, dim=2)  # (B, C, T, H, W)
        decoded = postprocess_video_tensor(decoded)  # (B, T, H, W, C) uint8
        output_video = decoded.contiguous()
        timer.mark_end()

        return output_video, source_audio, pixel_shape

    @staticmethod
    def _validate_retake_source(num_frames: int, height: int, width: int) -> None:
        """Fail fast on source video shapes the native VAE cannot round-trip.

        The LTX-2.3 causal video VAE requires ``8k + 1`` pixel frames and spatial
        dimensions divisible by the VAE spatial scale factors so encode/decode
        preserves shape.
        """
        ratio = VIDEO_SCALE_FACTORS.time
        if num_frames <= 0:
            raise ValueError(f"retake source must have frames; got {num_frames}")
        if (num_frames - 1) % ratio != 0:
            snapped = ((num_frames - 1) // ratio) * ratio + 1
            raise ValueError(
                f"retake source frame count must satisfy {ratio}k+1 (e.g. 97, 193); "
                f"got {num_frames}. Use a source with {snapped} frames."
            )
        height_scale = VIDEO_SCALE_FACTORS.height
        width_scale = VIDEO_SCALE_FACTORS.width
        if height % height_scale != 0 or width % width_scale != 0:
            raise ValueError(
                "retake source resolution must be divisible by the VAE spatial "
                f"scale factors {height_scale}x{width_scale}; got {height}x{width}."
            )

    def _read_source_video(
        self,
        video_path: str,
        num_frames: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Read and validate the source video as a normalized VAE input.

        Returns ``(1, 3, T, H, W)`` in ``[-1, 1]``. Pixel decoding is
        deterministic and uses sequential frame indices.
        """
        frames = list(decode_video_by_frame(video_path))
        if not frames:
            raise ValueError(f"retake source video decoded no frames: {video_path}")
        source_uint8 = torch.cat(frames, dim=0).unsqueeze(0)  # (1, T, H, W, C)
        decoded_frames = source_uint8.shape[1]
        if decoded_frames != num_frames:
            raise ValueError(
                f"retake source decoded {decoded_frames} frames but metadata reported {num_frames}."
            )
        if source_uint8.shape[2] != height or source_uint8.shape[3] != width:
            raise ValueError(
                f"retake source frame size {tuple(source_uint8.shape[2:4])} does not "
                f"match metadata {(height, width)}."
            )
        # uint8 [0, 255] -> [-1, 1], laid out (1, C, T, H, W) for the VAE encoder.
        normalized = source_uint8[0].to(torch.float32) / 127.5 - 1.0  # (T, H, W, C)
        source_norm_5d = normalized.permute(3, 0, 1, 2).unsqueeze(0).to(device=device, dtype=dtype)
        return source_norm_5d

    def _encode_source_audio_latent(
        self,
        audio: AudioData | None,
        pixel_shape: VideoPixelShape,
    ) -> torch.Tensor | None:
        """Encode the source audio into the frozen conditioning latent, natively.

        Runs the native audio VAE encoder over the decoded mel spectrogram and
        conforms the latent length to what the video pixel shape implies. Returns
        ``None`` when the source carries no audio stream.

        The duration is derived from the *video* shape (``frames / fps``) rather
        than the audio stream's own length, because the audio latent has to line
        up token-for-token with the video latent the transformer cross-attends
        to.
        """
        if audio is None:
            return None
        max_samples = round(pixel_shape.frames * audio.sample_rate / pixel_shape.fps)
        conditioning_audio = AudioData(
            samples=audio.samples[..., :max_samples],
            sample_rate=audio.sample_rate,
        )
        latents = encode_audio(conditioning_audio, self._audio_encoder).to(self.device, self.dtype)
        required_latent_frames = AudioLatentShape.from_video_pixel_shape(
            pixel_shape,
            sample_rate=self._audio_encoder.sample_rate,
            hop_length=self._audio_encoder.mel_hop_length,
            audio_latent_downsample_factor=(
                self._audio_encoder.patchifier.audio_latent_downsample_factor
            ),
        ).frames
        return _conform_latent_length(latents, required_latent_frames)

    @staticmethod
    def _single_prompt(prompt: str | list[str]) -> str:
        if isinstance(prompt, str):
            return prompt
        if len(prompt) == 1:
            return prompt[0]
        raise ValueError("LTX-2.3 retake workflow supports one prompt per request.")

    @staticmethod
    def _require_extra(extra: dict[str, Any], key: str) -> Any:
        value = extra.get(key)
        if value is None:
            raise ValueError(f"extra_params['{key}'] is required for LTX-2.3 retake.")
        return value
