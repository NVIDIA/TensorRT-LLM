# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native LTX-2.3 retake workflow for VisualGen.

This pipeline keeps a resident retake model for ``trtllm-serve`` and extends the
native LTX-2.3 pipeline with source-media encoders and masked denoising.
It encodes prompts with Gemma and consumes a standard video reference.
Retake generalizes image-to-video masking to condition both sides of the
regenerated window.

Source video/audio decode, stream metadata, audio/video VAE encode, diffusion,
and VAE decode all use TensorRT-LLM native components.
"""

from __future__ import annotations

import copy
import math
from functools import partial
from typing import TYPE_CHECKING, Any

import torch

from tensorrt_llm._torch.visual_gen.output import CudaPhaseTimer, PipelineOutput
from tensorrt_llm._torch.visual_gen.pipeline import ExtraParamSchema, RefSlotSpec, RoleSpec
from tensorrt_llm._torch.visual_gen.pipeline_registry import PipelineComponent, register_pipeline
from tensorrt_llm.inputs.multimodal_data import AudioData
from tensorrt_llm.logger import logger

from ..ltx2.ltx2_core.audio_vae import decode_audio
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
from ..ltx2.text_cache import TextCache
from .ltx23_core.audio_vae import AudioEncoderConfigurator, encode_audio
from .ltx23_core.modality import LTX23Modality
from .ltx23_core.video_vae import LTX23VideoEncoderConfigurator
from .media_io import decode_audio_from_file, decode_video_by_frame, get_videostream_metadata
from .pipeline_ltx23 import LTX23Pipeline

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

# Retake uses the same tiling geometry for source encode and output decode. Keep
# it separate from the generation pipeline's default because tile boundaries are
# part of the retake numerical contract.
_RETAKE_TILING_CONFIG = TilingConfig(
    spatial_config=SpatialTilingConfig(tile_size_in_pixels=768, tile_overlap_in_pixels=64),
    temporal_config=TemporalTilingConfig(tile_size_in_frames=80, tile_overlap_in_frames=24),
)


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
    """Crop or zero-pad audio latent frames along dimension 2."""
    missing_frames = expected_frames_count - latent.shape[2]
    if missing_frames < 0:
        return latent[:, :, :expected_frames_count]
    if missing_frames > 0:
        padding = [0] * (2 * (latent.ndim - 2))
        padding[-1] = missing_frames
        return torch.nn.functional.pad(latent, padding)
    return latent


def _retake_temporal_denoise_mask(
    positions: torch.Tensor, start_time: float, end_time: float
) -> torch.Tensor:
    """Select tokens overlapping a window on a modality's own timeline in seconds."""
    if positions.ndim != 4 or positions.shape[1] < 1 or positions.shape[-1] != 2:
        raise ValueError("Positions must have shape (batch, axes, tokens, 2).")
    start, end = positions[:, 0].unbind(dim=-1)
    return ((end > start_time) & (start < end_time)).to(torch.float32)


def _init_retake_audio_state(
    clean_audio: torch.Tensor,
    generator: torch.Generator,
    denoise_mask: torch.Tensor,
) -> torch.Tensor:
    """Sample audio after video noise, then apply source conditioning in token layout."""
    noise = torch.randn(
        clean_audio.shape, generator=generator, device=clean_audio.device, dtype=clean_audio.dtype
    )
    return _init_retake_patchified_latents(noise, clean_audio, denoise_mask)


# Retake shares checkpoints with LTX23Pipeline and is selected by ``workflow``.
@register_pipeline("LTX23RetakePipeline", doc="Native LTX-2.3 audio/video retake pipeline.")
class LTX23RetakePipeline(LTX23Pipeline):
    """Persistent VisualGen pipeline for native LTX-2.3 retake requests."""

    def __init__(self, pipeline_config: "DiffusionPipelineConfig") -> None:
        if pipeline_config.cache_backend is not None:
            raise NotImplementedError("Cache acceleration is not supported for LTX-2.3 retake.")
        if pipeline_config.parallel.cfg_size != 1:
            raise NotImplementedError("LTX-2.3 retake requires parallel.cfg_size=1.")
        if pipeline_config.parallel.tp_size != 1:
            raise NotImplementedError("Tensor parallelism is not supported for LTX-2.3 retake.")
        if pipeline_config.parallel.seq_parallel_size != pipeline_config.parallel.ulysses_size:
            raise NotImplementedError("LTX-2.3 retake supports Ulysses sequence parallelism only.")
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
            "retake_regenerate_audio": ExtraParamSchema(
                type="bool",
                default=True,
                description="Regenerate source audio in the window; missing audio is generated from scratch.",
            ),
            "retake_regenerate_video": ExtraParamSchema(
                type="bool",
                default=True,
                description="Regenerate video inside the retake window.",
            ),
        }

    @property
    def ref_slot_specs(self) -> dict[str, RefSlotSpec]:
        return {
            "video_reference": RefSlotSpec(
                modality="video",
                roles=[RoleSpec(role="reference", min=1, max=1)],
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
        text_cache: TextCache,
        audio_denoise_mask: torch.Tensor | None = None,
        clean_audio_latent: torch.Tensor | None = None,
        regenerate_video: bool = True,
        regenerate_audio: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Predict masked video x0 and optionally masked audio x0."""
        video_float = video_latents.float()
        video_model = video_latents.to(self.dtype)
        audio_model = audio_latents.to(self.dtype) if audio_latents is not None else None
        # Upstream preserves float32 sigma/mask products through x0 conversion.
        # The transformer handles casting inside its native embedding modules.
        sigma = timestep.float()
        video_sigma = sigma if regenerate_video else torch.zeros_like(sigma)
        audio_sigma = sigma if regenerate_audio else torch.zeros_like(sigma)
        video_timestep = denoise_mask * video_sigma.unsqueeze(-1)
        audio_timestep = (
            audio_denoise_mask * audio_sigma.unsqueeze(-1)
            if audio_denoise_mask is not None
            else torch.zeros_like(timestep)
        )
        video = LTX23Modality(
            latent=video_model,
            timesteps=video_timestep,
            positions=video_positions,
            context=text_cache.video_context,
            sigma=video_sigma,
            cross_modality_sigma=audio_sigma if audio_model is not None else None,
        )
        audio = None
        if audio_model is not None:
            if audio_positions is None:
                raise ValueError("Audio positions are required when audio latents are present.")
            audio = LTX23Modality(
                latent=audio_model,
                timesteps=audio_timestep,
                positions=audio_positions,
                context=text_cache.audio_context,
                sigma=audio_sigma,
                cross_modality_sigma=video_sigma,
            )

        velocity_video, velocity_audio = self.transformer(
            video=video,
            audio=audio,
            text_cache=text_cache,
            timestep=timestep.new_tensor(float(step_index) / num_steps),
            step_index=step_index,
        )
        if velocity_video is None:
            raise RuntimeError("LTX-2.3 retake transformer returned no video prediction.")
        denoised_video = (video_float - velocity_video.float() * video_timestep.unsqueeze(-1)).to(
            self.dtype
        )
        blend = denoise_mask.unsqueeze(-1).float()
        denoised_video = (denoised_video.float() * blend + clean_latent.float() * (1.0 - blend)).to(
            self.dtype
        )
        denoised_audio = None
        if audio_denoise_mask is not None:
            if audio_latents is None or clean_audio_latent is None or velocity_audio is None:
                raise RuntimeError(
                    "Audio retake requires source latents and an audio velocity prediction."
                )
            audio_x0 = (
                audio_latents.float() - velocity_audio.float() * audio_timestep.unsqueeze(-1)
            ).to(self.dtype)
            audio_blend = audio_denoise_mask.unsqueeze(-1).float()
            denoised_audio = (
                audio_x0.float() * audio_blend + clean_audio_latent.float() * (1.0 - audio_blend)
            ).to(self.dtype)
        return denoised_video, denoised_audio

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
        native_skip_components += [PipelineComponent.SCHEDULER]
        super().load_standard_components(
            checkpoint_dir,
            device,
            text_encoder_path=text_encoder_path,
            skip_components=list(dict.fromkeys(native_skip_components)),
        )

    def warmup(self) -> None:
        logger.info("LTX-2.3 retake compiles and captures from the first real source request.")

    @torch.inference_mode()
    def infer(self, req: "DiffusionRequest") -> PipelineOutput:
        extra = req.params.extra_params or {}
        refs = req.params.video_reference or []
        if len(refs) != 1:
            raise ValueError("LTX-2.3 retake requires exactly one video_reference.")
        if refs[0].format != "bytes" or not isinstance(refs[0].content, bytes):
            raise ValueError("LTX-2.3 retake video_reference must be resolved to bytes.")
        video_bytes = refs[0].content
        start_time = extra.get("retake_start_time")
        end_time = extra.get("retake_end_time")
        regenerate_video = extra.get("retake_regenerate_video", True)
        regenerate_audio = extra.get("retake_regenerate_audio", True)
        for name, value in (("video", regenerate_video), ("audio", regenerate_audio)):
            if not isinstance(value, bool):
                raise ValueError(f"retake_regenerate_{name} must be a boolean.")

        prompt = self._single_prompt(req.prompt)
        timer = CudaPhaseTimer()
        timer.mark_pre_start()
        video, audio, output_shape = self._run_retake(
            req,
            video_bytes,
            start_time,
            end_time,
            prompt,
            timer=timer,
            regenerate_video=regenerate_video,
            regenerate_audio=regenerate_audio,
        )
        if video is None:
            return timer.fill(PipelineOutput(frame_rate=float(output_shape.fps)))
        audio_tensor = audio.samples
        sample_rate = audio.sample_rate
        if sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {sample_rate}")
        return timer.fill(
            PipelineOutput(
                video=video,
                audio=audio_tensor,
                frame_rate=float(output_shape.fps),
                audio_sample_rate=sample_rate,
            )
        )

    def _run_retake(
        self,
        req: "DiffusionRequest",
        video_bytes: bytes,
        start_time: float | None,
        end_time: float | None,
        prompt: str,
        timer: CudaPhaseTimer,
        regenerate_video: bool = True,
        regenerate_audio: bool = True,
    ) -> tuple[torch.Tensor | None, AudioData | None, VideoPixelShape]:
        """Encode source media, denoise the masked window, and decode the full clip.

        Returns uint8 ``(1, T, H, W, C)`` video, output audio, and video shape.
        Non-decoding ranks return None for both media streams.
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
        pixel_shape = get_videostream_metadata(video_bytes)
        num_frames = int(pixel_shape.frames)
        height = int(pixel_shape.height)
        width = int(pixel_shape.width)
        fps = float(pixel_shape.fps)
        self._validate_retake_source(num_frames, height, width)

        start_time, end_time = _resolve_retake_time_window(start_time, end_time, fps, num_frames)

        source_norm_5d = self._read_source_video(
            video_bytes, num_frames, height, width, device, dtype
        )

        # ---- 2. Native VAE encode + seed initial latents ----------------
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

        video_positions = self.video_patchifier.get_patch_grid_bounds(video_shape, device=device)
        video_positions = get_pixel_coords(
            video_positions.float(), VIDEO_SCALE_FACTORS, causal_fix=True
        )
        video_positions[:, 0, ...] = video_positions[:, 0, ...] / fps
        # TemporalRegionMask uses overlap in seconds on each modality's own
        # causal token bounds, not rounded pixel-frame or shared audio indices.
        denoise_mask = (
            _retake_temporal_denoise_mask(video_positions, start_time, end_time)
            if regenerate_video
            else torch.zeros_like(video_positions[:, 0, :, 0])
        )
        source_patch_latents = self.video_patchifier.patchify(source_window_latents)
        # Sample both streams even when frozen to preserve the shared RNG order.
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
        if getattr(self, "text_encoder", None) is None:
            raise RuntimeError("Gemma is not loaded for LTX-2.3 retake prompt encoding.")
        prompt_embeds, prompt_attention_mask = self._encode_prompt(
            prompt, max_sequence_length=max_sequence_length
        )
        video_embeds, audio_embeds, connector_mask = self._process_connectors(
            prompt_embeds, prompt_attention_mask
        )

        # RoPE positions remain float32 because position error is amplified by
        # the distilled schedule's large final Euler steps.

        # ---- Source audio conditioning ----------------------------------
        source_audio = decode_audio_from_file(video_bytes)
        if getattr(self, "audio_decoder", None) is None or getattr(self, "vocoder", None) is None:
            raise RuntimeError(
                "Audio retake requires the native audio decoder and vocoder weights."
            )
        audio_latent = self._encode_source_audio_latent(
            source_audio,
            pixel_shape,
        )
        scratch_audio = audio_latent is None
        regenerate_audio = scratch_audio or regenerate_audio
        if scratch_audio:
            audio_shape = AudioLatentShape.from_video_pixel_shape(pixel_shape)
            audio_latent = torch.zeros(audio_shape.to_torch_shape(), device=device, dtype=dtype)
        audio_shape = AudioLatentShape.from_torch_shape(audio_latent.shape)
        self.transformer.configure_audio_ulysses(audio_shape.frames)
        audio_clean_latents = self.audio_patchifier.patchify(audio_latent.float())
        audio_positions = self.audio_patchifier.get_patch_grid_bounds(audio_shape, device=device)
        audio_denoise_mask = (
            _retake_temporal_denoise_mask(audio_positions, start_time, end_time)
            if regenerate_audio and not scratch_audio
            else torch.ones_like(audio_positions[:, 0, :, 0])
        )
        if not regenerate_audio:
            audio_denoise_mask = torch.zeros_like(audio_denoise_mask)

        text_cache = self.transformer.prepare_text_cache(
            video_context=video_embeds,
            video_context_mask=connector_mask,
            video_positions=video_positions,
            audio_context=audio_embeds,
            audio_context_mask=connector_mask if audio_embeds is not None else None,
            audio_positions=audio_positions,
            dtype=dtype,
        )

        # ---- 5. Native masked denoise (distilled, non-guided) ---------------
        scheduler = NativeSchedulerAdapter()
        scheduler.sigmas = torch.tensor(
            _RETAKE_DISTILLED_SIGMA_VALUES, dtype=torch.float32, device=device
        )
        timesteps = scheduler.timesteps
        num_steps = len(timesteps)

        # Frozen audio follows the clean x0 target with conditioning sigma zero;
        # regenerated audio instead uses the transformer's masked x0 prediction.
        clean_audio = audio_clean_latents.to(dtype)
        audio_state = _init_retake_audio_state(clean_audio, generator, audio_denoise_mask)
        audio_scheduler = copy.deepcopy(scheduler)
        extra_streams = {"audio": (audio_state, audio_scheduler)}

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
            denoised_video, denoised_audio = self._masked_transformer_step(
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
                audio_denoise_mask=audio_denoise_mask,
                clean_audio_latent=clean_audio,
                regenerate_video=regenerate_video,
                regenerate_audio=regenerate_audio,
            )
            audio_prediction = {"audio": denoised_audio}
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
        denoised_latents = denoise_result[0]

        # ---- 6. Native decode -------------------------------------------
        timer.mark_post_start()
        output_video, output_audio = self.decode_latents(
            latents=denoised_latents,
            decode_fn=partial(
                self._decode_video,
                video_shape=video_shape,
                tiling_config=_RETAKE_TILING_CONFIG,
                generator=generator,
            ),
            extra_latents={
                "audio": (
                    denoise_result[1]["audio"],
                    partial(self._decode_retake_audio, audio_shape=audio_shape),
                )
            },
        )
        if output_video is not None:
            output_video = output_video.contiguous()
        timer.mark_end()

        return output_video, output_audio, pixel_shape

    def _decode_retake_audio(
        self, latents: torch.Tensor, audio_shape: AudioLatentShape
    ) -> AudioData:
        """Decode the updated audio stream through the existing native audio stack."""
        if getattr(self, "audio_decoder", None) is None or getattr(self, "vocoder", None) is None:
            raise RuntimeError(
                "Audio retake requires the native audio decoder and vocoder weights."
            )
        unpatchified = self.audio_patchifier.unpatchify(latents, audio_shape).to(self.dtype)
        samples = decode_audio(unpatchified, self.audio_decoder, self.vocoder)
        if samples.ndim != 3 or samples.shape[0] != 1 or samples.shape[-1] == 0:
            raise RuntimeError(
                f"Audio retake decoder returned invalid waveform shape {tuple(samples.shape)}."
            )
        return AudioData(samples=samples[0], sample_rate=int(self.vocoder.output_sampling_rate))

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
        video_bytes: bytes,
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
        frames = list(decode_video_by_frame(video_bytes))
        if not frames:
            raise ValueError("retake source video decoded no frames.")
        source_uint8 = torch.cat(frames, dim=0)  # (T, H, W, C)
        decoded_frames = source_uint8.shape[0]
        if decoded_frames != num_frames:
            raise ValueError(
                f"retake source decoded {decoded_frames} frames but metadata reported {num_frames}."
            )
        if source_uint8.shape[1] != height or source_uint8.shape[2] != width:
            raise ValueError(
                f"retake source frame size {tuple(source_uint8.shape[1:3])} does not "
                f"match metadata {(height, width)}."
            )
        # uint8 [0, 255] -> [-1, 1], laid out (1, C, T, H, W) for the VAE encoder.
        normalized = source_uint8.to(torch.float32) / 127.5 - 1.0
        return normalized.permute(3, 0, 1, 2).unsqueeze(0).to(device=device, dtype=dtype)

    def _encode_source_audio_latent(
        self,
        audio: AudioData | None,
        pixel_shape: VideoPixelShape,
    ) -> torch.Tensor | None:
        """Encode source audio and align its latent length to the video timeline."""
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
