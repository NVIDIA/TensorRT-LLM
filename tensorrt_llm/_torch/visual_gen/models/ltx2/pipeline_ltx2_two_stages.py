# Copyright 2025 Lightricks. All rights reserved.
#
# Two-stage LTX-2 pipeline: stage 1 at half spatial resolution,
# 2x latent upsample, optional refinement denoising, then decode.
# Reference: https://github.com/Lightricks/LTX-2
# WIP

import copy
import time
from typing import List, Optional, Union

import torch

from tensorrt_llm._torch.visual_gen.output import MediaOutput
from tensorrt_llm._torch.visual_gen.pipeline_registry import register_pipeline
from tensorrt_llm._torch.visual_gen.utils import postprocess_video_tensor
from tensorrt_llm.logger import logger

from .ltx2_core.audio_vae import decode_audio
from .ltx2_core.modality import Modality
from .ltx2_core.patchifier import get_pixel_coords
from .ltx2_core.types import VIDEO_SCALE_FACTORS, VideoLatentShape, VideoPixelShape
from .pipeline_ltx2 import LTX2Pipeline

STAGE_2_DISTILLED_SIGMA_VALUES = [0.909375, 0.725, 0.421875, 0.0]


@register_pipeline("LTX2TwoStagesPipeline")
class LTX2TwoStagesPipeline(LTX2Pipeline):
    """Two-stage text-to-video with audio.

    Stage 1: denoise at half spatial resolution with full guidance.
    Stage 2: 2x spatial upsample, optional refinement denoising
             (with distilled sigma schedule, no guidance), then decode.
    """

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 512,
        width: int = 768,
        num_frames: int = 121,
        frame_rate: float = 24.0,
        num_inference_steps: int = 40,
        num_refinement_steps: int = 0,
        guidance_scale: float = 3.0,
        guidance_rescale: float = 0.0,
        seed: int = 42,
        output_type: str = "pt",
        max_sequence_length: int = 1024,
        image: Optional[Union[str, torch.Tensor]] = None,
        image_cond_strength: float = 1.0,
        use_learned_upsampler: bool = False,
        stg_scale: float = 0.0,
        stg_blocks: Optional[List[int]] = None,
        modality_scale: float = 1.0,
        rescale_scale: float = 0.0,
        guidance_skip_step: int = 0,
        enhance_prompt: bool = False,
    ):
        """Generate video and audio via two stages, optionally conditioned on an image.

        Stage 1: Denoise at half spatial resolution (height//2, width//2).
        Stage 2: Upsample video latents 2x, optionally run refinement
                 denoising steps with the distilled sigma schedule,
                 then decode.
        """
        pipeline_start = time.time()
        height_s1 = height // 2
        width_s1 = width // 2
        logger.info(
            f"LTX2 two-stage: stage1 at {height_s1}x{width_s1}, final {height}x{width}"
        )

        # Stage 1 (i2v image is passed to the base pipeline at half resolution)
        out = super().__call__(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height_s1,
            width=width_s1,
            num_frames=num_frames,
            frame_rate=frame_rate,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            guidance_rescale=guidance_rescale,
            seed=seed,
            output_type="latent",
            max_sequence_length=max_sequence_length,
            image=image,
            image_cond_strength=image_cond_strength,
            stg_scale=stg_scale,
            stg_blocks=stg_blocks,
            modality_scale=modality_scale,
            rescale_scale=rescale_scale,
            guidance_skip_step=guidance_skip_step,
            enhance_prompt=enhance_prompt,
        )

        video_latents = out.video
        audio_out = out.audio

        latent_num_frames = (num_frames - 1) // VIDEO_SCALE_FACTORS.time + 1
        latent_height_s2 = height // VIDEO_SCALE_FACTORS.height
        latent_width_s2 = width // VIDEO_SCALE_FACTORS.width

        # 2x spatial upsample
        if use_learned_upsampler and getattr(self, "spatial_upsampler", None) is not None:
            video_latents = self.spatial_upsampler(video_latents)
        else:
            video_latents = torch.nn.functional.interpolate(
                video_latents,
                size=(latent_num_frames, latent_height_s2, latent_width_s2),
                mode="trilinear",
                align_corners=False,
            )

        # Stage 2 refinement denoising
        if num_refinement_steps > 0:
            video_latents = self._refinement_denoise(
                video_latents=video_latents,
                prompt=prompt,
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=frame_rate,
                seed=seed,
                max_sequence_length=max_sequence_length,
            )

        if output_type == "latent":
            if self.rank == 0:
                logger.info(f"Two-stage total time: {time.time() - pipeline_start:.2f}s")
            return MediaOutput(video=video_latents, audio=audio_out)

        # Decode video
        logger.info("Decoding upsampled video...")
        video_latents = video_latents.to(self.dtype)
        video = self.video_decoder(video_latents, generator=None)
        video = postprocess_video_tensor(video, remove_batch_dim=True)

        # Decode audio
        if audio_out is not None:
            audio_out = audio_out.to(self.dtype)
            audio_out = decode_audio(audio_out, self.audio_decoder, self.vocoder)

        if self.rank == 0:
            logger.info(f"Two-stage total time: {time.time() - pipeline_start:.2f}s")

        return MediaOutput(video=video, audio=audio_out)

    def _refinement_denoise(
        self,
        video_latents: torch.Tensor,
        prompt: Union[str, List[str]],
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        seed: int,
        max_sequence_length: int,
    ) -> torch.Tensor:
        """Run a short refinement denoising pass on upsampled latents.

        Uses the distilled sigma schedule (3 steps, no guidance) matching
        the reference two-stage pipeline.
        """
        logger.info("Stage 2: refinement denoising...")
        generator = torch.Generator(device=self.device).manual_seed(seed)

        # Re-encode prompt (no negative prompt for refinement)
        prompt_embeds, prompt_attention_mask = self._encode_prompt(
            prompt, num_videos_per_prompt=1, max_sequence_length=max_sequence_length,
        )
        video_embeds, audio_embeds, connector_mask = self._process_connectors(
            prompt_embeds, prompt_attention_mask,
        )

        # Prepare shapes at full resolution
        pixel_shape = VideoPixelShape(
            batch=1, frames=num_frames, height=height, width=width, fps=frame_rate,
        )
        video_shape = VideoLatentShape.from_pixel_shape(
            pixel_shape, latent_channels=self.transformer_in_channels,
        )

        # Patchify the upsampled latents
        latents_patchified = self.video_patchifier.patchify(video_latents)

        # Positions
        video_positions = self.video_patchifier.get_patch_grid_bounds(
            video_shape, device=self.device,
        )
        video_positions = get_pixel_coords(
            video_positions.float(), VIDEO_SCALE_FACTORS, causal_fix=True,
        )
        video_positions[:, 0, ...] = video_positions[:, 0, ...] / frame_rate
        video_positions = video_positions.to(self.dtype)

        # Distilled sigma schedule
        sigmas = torch.tensor(
            STAGE_2_DISTILLED_SIGMA_VALUES, device=self.device, dtype=torch.float32,
        )

        # Add noise to the upsampled latents at the first sigma level
        noise = torch.randn_like(latents_patchified, generator=generator)
        noisy_latents = latents_patchified + sigmas[0] * noise

        # Simple Euler denoising (no guidance, no audio refinement)
        latents_working = noisy_latents
        for i in range(len(sigmas) - 1):
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]
            timestep = sigma.unsqueeze(0).expand(latents_working.shape[0])

            video_mod = Modality(
                latent=latents_working.to(self.dtype),
                timesteps=timestep,
                positions=video_positions,
                context=video_embeds,
                context_mask=connector_mask,
            )
            vel_video, _ = self.transformer(video=video_mod, audio=None)

            sigma_broad = sigma.float()
            while sigma_broad.dim() < vel_video.dim():
                sigma_broad = sigma_broad.unsqueeze(-1)

            denoised = latents_working.float() - vel_video.float() * sigma_broad

            dt = sigma_next - sigma
            velocity = (latents_working.float() - denoised) / sigma_broad
            latents_working = (latents_working.float() + velocity * dt).to(
                latents_working.dtype
            )

        refined = self.video_patchifier.unpatchify(latents_working, video_shape)
        logger.info("Stage 2 refinement complete")
        return refined
