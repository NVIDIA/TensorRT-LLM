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
"""FastWan: distilled (DMD) Wan 2.2 TI2V-5B for 3-step text-to-video.

FastWan shares Wan 2.2 TI2V-5B's architecture exactly — only the weights
(distilled) and the sampling recipe differ. So this pipeline subclasses
``WanPipeline`` and overrides only the denoising loop.
"""

import time
from typing import List, Optional, Union

import PIL.Image
import torch
from diffusers.utils.torch_utils import randn_tensor

from tensorrt_llm._torch.visual_gen.output import CudaPhaseTimer, PipelineOutput
from tensorrt_llm._torch.visual_gen.pipeline_registry import register_pipeline
from tensorrt_llm._utils import nvtx_range
from tensorrt_llm.logger import logger

from .defaults import get_fastwan_default_params
from .pipeline_wan import WanPipeline


@register_pipeline(
    "WanDMDPipeline",
    hf_ids=[
        "FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers",
    ],
    doc="FastWan 2.2 distilled (DMD) — 3-step Wan 2.2 TI2V-5B text-to-video.",
)
class WanDMDPipeline(WanPipeline):
    """Wan 2.2 TI2V-5B with the DMD 3-step sampling loop.

    DMD inference (Distribution Matching Distillation): at each step the model
    predicts the clean video ``x0``, then we re-noise it with FRESH noise to the
    next (lower) noise level. The last step keeps ``x0``. CFG-free — one
    transformer forward per step. Faithful to FastVideo's ``DmdDenoisingStage``.
    """

    # Fixed DMD schedule for FastWan2.2-TI2V-5B (see model card / FastVideo config).
    DMD_TIMESTEPS = (1000, 757, 522)
    NUM_TRAIN_TIMESTEPS = 1000

    @property
    def default_generation_params(self):
        return get_fastwan_default_params()

    @nvtx_range("WanDMDPipeline.forward")
    @torch.no_grad()
    def forward(
        self,
        prompt: Union[str, List[str]],
        seed: int,
        negative_prompt: Optional[str] = None,
        height: int = 704,
        width: int = 1280,
        num_frames: int = 121,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        guidance_scale_2: Optional[float] = None,
        boundary_ratio: Optional[float] = None,
        max_sequence_length: int = 512,
        image: Optional[Union[PIL.Image.Image, torch.Tensor, str]] = None,
    ):
        # FastWan is text-to-video only for now; CFG-free, so guidance_scale and
        # the Wan-2.2 two-stage knobs (guidance_scale_2/boundary_ratio) are
        # accepted for infer()/base-class compatibility but unused.
        if image is not None:
            raise NotImplementedError(
                "FastWan currently supports text-to-video only (no image conditioning)."
            )

        pipeline_start = time.time()
        timer = CudaPhaseTimer()
        timer.mark_pre_start()

        if isinstance(prompt, str):
            prompt = [prompt]
        batch_size = len(prompt)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        self.validate_resolution(height, width, num_frames)

        if negative_prompt:
            raise ValueError("FastWan is CFG-free and does not support negative prompts.")
        prompt_embeds, _ = self._encode_prompt(prompt, "", max_sequence_length)

        if self._fixed_latent is not None:
            latents = self._fixed_latent.to(device=self.device, dtype=self.dtype)
        else:
            latents = self._prepare_latents(batch_size, height, width, num_frames, generator)

        timer.mark_denoise_start()
        latents = self._denoise(latents, prompt_embeds, generator)
        timer.mark_post_start()

        logger.info("Decoding video...")
        decode_start = time.time()
        video = self.decode_latents(latents, self._decode_latents)
        if self.rank == 0:
            logger.info(f"Video decoded in {time.time() - decode_start:.2f}s")
            logger.info(f"Total pipeline time: {time.time() - pipeline_start:.2f}s")

        timer.mark_end()
        frame_rate = self.default_generation_params.get("frame_rate", 24.0)
        return timer.fill(PipelineOutput(video=video, frame_rate=frame_rate))

    @nvtx_range("WanDMDPipeline._denoise", color="blue")
    def _denoise(self, latents, prompt_embeds, generator):
        """3-step DMD loop: predict x0, then re-noise with fresh noise.

        The per-step sigma reduces to ``sigma = t / NUM_TRAIN_TIMESTEPS`` because
        FastVideo's scheduler keeps ``timesteps = sigmas * num_train_timesteps``
        (co-scaled), so the lookup returns ``matched_timestep / 1000`` and the
        flow-shift cancels (verified numerically for shift in {1, 5, 8, 12}).
        """
        timesteps = self.DMD_TIMESTEPS
        num_steps = len(timesteps)
        _, ph, pw = self.transformer.config.patch_size

        # The DMD ops are linear (mul/sub/add), so they run in the transformer's
        # native (bf16) dtype to match the FastVideo reference; no fp32 needed.
        latents = latents.to(self.dtype)
        # Patch grid dimensions are fixed for the entire denoising loop.
        nf = latents.shape[2]
        nh = latents.shape[3] // ph
        nw = latents.shape[4] // pw
        start = time.time()
        for i, t in enumerate(timesteps):
            t_tensor = torch.full(
                (latents.shape[0], nf * nh * nw), float(t), device=latents.device
            )

            pred_noise = self.transformer(
                hidden_states=latents,
                timestep=t_tensor / self.NUM_TRAIN_TIMESTEPS,
                encoder_hidden_states=prompt_embeds,
            )

            sigma = t / self.NUM_TRAIN_TIMESTEPS
            pred_video = latents - sigma * pred_noise  # clean-video estimate (x0)

            if i < num_steps - 1:
                sigma_next = timesteps[i + 1] / self.NUM_TRAIN_TIMESTEPS
                noise = randn_tensor(
                    latents.shape, generator=generator,
                    device=latents.device, dtype=self.dtype,
                )
                latents = (1.0 - sigma_next) * pred_video + sigma_next * noise
            else:
                latents = pred_video

            if self.rank == 0:
                logger.info(f"DMD step {i + 1}/{num_steps} (t={t}, sigma={sigma:.3f})")

        if self.rank == 0:
            logger.info(f"DMD denoising done in {time.time() - start:.2f}s")
        return latents
