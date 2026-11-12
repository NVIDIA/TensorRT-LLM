# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-LTX-2

"""Full-model LTX-2.3 generation: half-resolution CFG, learned x2, LoRA refinement."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from tensorrt_llm._torch.visual_gen.output import CudaPhaseTimer, PipelineOutput
from tensorrt_llm._torch.visual_gen.pipeline_registry import register_pipeline
from tensorrt_llm.logger import logger

from ..ltx2.ltx2_core.diffusion_steps import EulerDiffusionStep
from ..ltx2.ltx2_core.patchifier import get_pixel_coords
from ..ltx2.ltx2_core.types import VIDEO_SCALE_FACTORS, AudioLatentShape, VideoLatentShape
from ..ltx2.ltx2_core.upsampler import upsample_video
from ..ltx2.pipeline_ltx2 import _assert_resolution
from ..ltx2.pipeline_ltx2_two_stages import (
    _load_lora_deltas,
    _load_spatial_upsampler,
    _PersistentLoRAWeightCache,
)
from .ltx23_core.modality import LTX23Modality
from .pipeline_ltx23 import LTX23Pipeline

if TYPE_CHECKING:
    from tensorrt_llm._torch.visual_gen.config import DiffusionPipelineConfig


_STAGE_2_SIGMAS = (0.909375, 0.725, 0.421875, 0.0)


@dataclass
class _LTX23LatentOutput:
    """State carried through upsampling, LoRA refinement and final decoding."""

    video: torch.Tensor
    audio: torch.Tensor
    video_shape: VideoLatentShape
    audio_shape: AudioLatentShape
    video_context: torch.Tensor
    audio_context: torch.Tensor
    context_mask: torch.Tensor | None
    generator: torch.Generator
    timer: CudaPhaseTimer


@register_pipeline("LTX23TwoStagesPipeline")
class LTX23TwoStagesPipeline(LTX23Pipeline):
    """Opt-in full-dev-model two-stage generation; the baseline is BF16 only."""

    def __init__(self, pipeline_config: DiffusionPipelineConfig) -> None:
        if pipeline_config.extra_attrs.get("workflow", "generation") != "generation":
            raise ValueError("LTX-2.3 two-stage supports generation only.")
        if pipeline_config.quant_config.quant_algo is not None:
            raise NotImplementedError("LTX-2.3 two-stage baseline requires BF16 weights.")
        if pipeline_config.torch_dtype != torch.bfloat16:
            raise ValueError("LTX-2.3 two-stage baseline requires torch_dtype=bfloat16.")
        if pipeline_config.cache is not None:
            raise NotImplementedError(
                "LTX-2.3 two-stage baseline does not support cache acceleration."
            )
        if getattr(pipeline_config, "runtime_lora", None) is not None:
            raise NotImplementedError(
                "LTX-2.3 two-stage baseline supports only its fixed distilled LoRA, "
                "not additional runtime LoRA adapters."
            )
        offload = getattr(pipeline_config, "cpu_offload_config", None)
        if offload is not None and offload.enable:
            raise NotImplementedError(
                "LTX-2.3 two-stage baseline requires resident LoRA weight states; "
                "CPU offloading is not supported."
            )
        self._lora_state = "original"
        super().__init__(pipeline_config)

    def _setup_cuda_graphs(self) -> None:
        if not self.pipeline_config.cuda_graph.enable:
            return
        super()._setup_cuda_graphs()
        self._cuda_graph_runners["transformer"].register_extra_key_fn(
            "ltx23_two_stage_lora_state",
            lambda *args, **kwargs: self._lora_state,
        )

    def load_standard_components(
        self,
        checkpoint_dir: str,
        device: torch.device,
        skip_components: list | None = None,
        **kwargs,
    ) -> None:
        extra = self.pipeline_config.extra_attrs
        upsampler_path = extra.get("spatial_upsampler_path")
        lora_path = extra.get("distilled_lora_path")
        for path in (upsampler_path, lora_path):
            if not path or not Path(path).is_file():
                raise FileNotFoundError(f"LTX-2.3 two-stage component not found: {path}")
        # CPU matrix products overlap Gemma/VAE loads; no collectives in this worker.
        with ThreadPoolExecutor(max_workers=1) as executor:
            lora = executor.submit(
                _load_lora_deltas,
                str(lora_path),
                self.transformer,
                self._TRANSFORMER_PREFIX,
                strict=True,
            )
            super().load_standard_components(
                checkpoint_dir, device, skip_components=skip_components, **kwargs
            )
            self.spatial_upsampler = _load_spatial_upsampler(
                str(upsampler_path), device, self.dtype, strict=True
            ).eval()
            deltas = lora.result()
            if not deltas:
                raise ValueError("Distilled LoRA did not match any LTX-2.3 transformer weights.")
            # Keep both weight storages resident for CUDA graph replay.
            self._lora_weights = _PersistentLoRAWeightCache.build(self.transformer, deltas)
            if self._lora_weights.applied_count != len(deltas):
                raise ValueError("Not every LTX-2.3 distilled LoRA delta could be applied.")
            self._lora_weights.bind_original()
            logger.info(f"LTX-2.3 two-stage: {len(deltas)} resident LoRA weight pairs loaded.")

    @torch.inference_mode()
    def forward(
        self,
        prompt: str | list[str],
        seed: int,
        negative_prompt: str | list[str] | None = None,
        height: int = 512,
        width: int = 768,
        num_frames: int = 121,
        frame_rate: float = 24.0,
        num_inference_steps: int = 40,
        guidance_scale: float = 4.0,
        guidance_rescale: float = 0.0,
        output_type: str = "pt",
        max_sequence_length: int = 1024,
    ) -> PipelineOutput:
        if height <= 0 or width <= 0:
            raise ValueError("LTX-2.3 two-stage height and width must be positive.")
        _assert_resolution(height, width, is_two_stage=True)
        if not isinstance(prompt, str) and len(prompt) != 1:
            raise ValueError("LTX-2.3 two-stage baseline supports one prompt per request.")
        if output_type not in ("pt", "latent"):
            raise ValueError("output_type must be 'pt' or 'latent'.")
        return super().forward(
            prompt=prompt,
            seed=seed,
            negative_prompt=negative_prompt,
            height=height // 2,
            width=width // 2,
            num_frames=num_frames,
            frame_rate=frame_rate,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            guidance_rescale=guidance_rescale,
            output_type=output_type,
            max_sequence_length=max_sequence_length,
        )

    def _decode_denoise_output(
        self,
        *,
        latents: torch.Tensor,
        audio_latents: torch.Tensor,
        video_shape: VideoLatentShape,
        audio_shape: AudioLatentShape,
        video_context: torch.Tensor,
        audio_context: torch.Tensor,
        context_mask: torch.Tensor | None,
        generator: torch.Generator,
        timer: CudaPhaseTimer,
        frame_rate: float,
        output_type: str,
    ) -> PipelineOutput:
        stage_1 = _LTX23LatentOutput(
            video=latents,
            audio=audio_latents,
            video_shape=video_shape,
            audio_shape=audio_shape,
            video_context=video_context,
            audio_context=audio_context,
            context_mask=context_mask,
            generator=generator,
            timer=timer,
        )
        video = self.video_patchifier.unpatchify(latents, video_shape)
        video = upsample_video(
            video, self.video_decoder.per_channel_statistics, self.spatial_upsampler
        )
        try:
            self._lora_weights.bind_merged()
            self._lora_state = "merged"
            refined = self._refine_video(stage_1, video, frame_rate)
        finally:
            self._lora_weights.bind_original()
            self._lora_state = "original"
        # End denoising after both stages, before final video/audio decode.
        refined.timer.mark_post_start()
        return super()._decode_denoise_output(
            latents=refined.video,
            audio_latents=refined.audio,
            video_shape=refined.video_shape,
            audio_shape=refined.audio_shape,
            video_context=refined.video_context,
            audio_context=refined.audio_context,
            context_mask=refined.context_mask,
            generator=refined.generator,
            timer=refined.timer,
            frame_rate=frame_rate,
            output_type=output_type,
        )

    def _refine_video(
        self,
        stage_1: _LTX23LatentOutput,
        video: torch.Tensor,
        frame_rate: float,
    ) -> _LTX23LatentOutput:
        shape = VideoLatentShape.from_torch_shape(video.shape)
        positions = self.video_patchifier.get_patch_grid_bounds(shape, device=self.device)
        positions = get_pixel_coords(positions.float(), VIDEO_SCALE_FACTORS, causal_fix=True)
        positions[:, 0, ...] /= frame_rate
        audio_positions = self.audio_patchifier.get_patch_grid_bounds(
            stage_1.audio_shape, device=self.device
        )
        sigmas = torch.tensor(_STAGE_2_SIGMAS, device=self.device, dtype=torch.float32)
        working = self.video_patchifier.patchify(video)
        noise = torch.randn(
            working.shape, generator=stage_1.generator, device=working.device, dtype=working.dtype
        )
        working = torch.lerp(working.float(), noise.float(), sigmas[0]).to(working.dtype)
        # Upstream noises a frozen audio spec with scale=0. Consume that draw too
        # so the shared generator reaches VAE decode at the same RNG state.
        torch.randn(
            stage_1.audio.shape,
            device=self.device,
            dtype=stage_1.audio.dtype,
            generator=stage_1.generator,
        )
        text_cache = self.transformer.prepare_text_cache(
            video_context=stage_1.video_context,
            video_context_mask=stage_1.context_mask,
            video_positions=positions,
            audio_context=stage_1.audio_context,
            audio_context_mask=stage_1.context_mask,
            audio_positions=audio_positions,
            dtype=self.dtype,
        )
        step = EulerDiffusionStep()
        frozen_sigma = torch.zeros(working.shape[0], device=self.device, dtype=sigmas.dtype)
        # Frozen audio still conditions video, with its own timestep and sigma at zero.
        audio_mod = LTX23Modality(
            latent=stage_1.audio,
            timesteps=frozen_sigma,
            sigma=frozen_sigma,
            positions=audio_positions,
            context=stage_1.audio_context,
            context_mask=stage_1.context_mask,
        )
        for index, sigma in self._profile_denoise_steps(sigmas[:-1]):
            timestep = sigma.expand(working.shape[0])
            video_mod = LTX23Modality(
                latent=working,
                timesteps=timestep,
                sigma=timestep,
                cross_modality_sigma=frozen_sigma,
                positions=positions,
                context=stage_1.video_context,
                context_mask=stage_1.context_mask,
            )
            velocity, _ = self.transformer(
                video=video_mod,
                # Cross-modality gates use the other stream's sigma.
                audio=replace(audio_mod, cross_modality_sigma=timestep),
                text_cache=text_cache,
                timestep=sigma.new_tensor(0.0),
            )
            # Match upstream's BF16 X0Model and to_velocity rounding boundaries.
            denoised = (working.float() - velocity.float() * sigma).to(working.dtype)
            working = step.step(working, denoised, sigmas, index)
        return replace(stage_1, video=working, video_shape=shape)
