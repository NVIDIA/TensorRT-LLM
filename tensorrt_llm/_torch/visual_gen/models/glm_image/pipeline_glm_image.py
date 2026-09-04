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

import inspect
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import PIL
import torch
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, SchedulerMixin
from diffusers.utils.torch_utils import randn_tensor
from transformers import (
    ByT5Tokenizer,
    GlmImageForConditionalGeneration,
    GlmImageProcessor,
    T5EncoderModel,
)

from tensorrt_llm import logger
from tensorrt_llm._torch.visual_gen.config import DiffusionPipelineConfig
from tensorrt_llm._torch.visual_gen.output import CudaPhaseTimer, PipelineOutput
from tensorrt_llm._torch.visual_gen.pipeline import BasePipeline
from tensorrt_llm._torch.visual_gen.pipeline_registry import PipelineComponent, register_pipeline
from tensorrt_llm._torch.visual_gen.utils import make_noise_generator

from .transformer_glm_image import GlmImageTransformer2DModel


# ------------------------------------------------------------------
# HF Port
# ------------------------------------------------------------------
# Copied from diffusers.pipelines.cogview4.pipeline_cogview4.retrieve_timesteps
def retrieve_timesteps(
    scheduler: SchedulerMixin,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs: Any,
) -> Tuple[torch.Tensor, int]:
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`list[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`list[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    accepts_timesteps = "timesteps" in set(
        inspect.signature(scheduler.set_timesteps).parameters.keys()
    )
    accepts_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())

    if timesteps is not None and sigmas is not None:
        if not accepts_timesteps and not accepts_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep or sigma schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif timesteps is not None and sigmas is None:
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif timesteps is None and sigmas is not None:
        if not accepts_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


def calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    base_shift: float = 0.25,
    max_shift: float = 0.75,
) -> float:
    m = (image_seq_len / base_seq_len) ** 0.5
    mu = m * max_shift + base_shift
    return mu


@register_pipeline(
    "GlmImagePipeline",
    hf_ids=["zai-org/GLM-Image"],
    doc="GlmImage family (text-to-image).",
)
class GlmImagePipeline(BasePipeline):
    # ------------------------------------------------------------------
    # HF Port
    # ------------------------------------------------------------------
    def generate_prior_tokens(
        self,
        prompt: Union[str, List[str]],
        height: int,
        width: int,
        device: Optional[torch.device] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Generate prior tokens for the DiT model using the AR model.

        Args:
            prompt: Single prompt or list of prompts
            height: Target image height
            width: Target image width
            device: Target device
            generator: Random generator for reproducibility

        Returns:
            prior_token_ids: Tensor of shape (batch_size, num_tokens) with upsampled prior tokens
        """
        device = device or self._execution_device

        # Normalize prompt to list format
        prompt_list = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt_list)

        # Build messages for each sample in the batch
        all_messages = [
            [{"role": "user", "content": [{"type": "text", "text": p}]}] for p in prompt_list
        ]
        # Process with the processor (supports batch with left padding)
        inputs = self.processor.apply_chat_template(
            all_messages,
            tokenize=True,
            padding=True if batch_size > 1 else False,
            target_h=height,
            target_w=width,
            return_dict=True,
            return_tensors="pt",
        ).to(device)

        image_grid_thw = inputs.get("image_grid_thw")
        images_per_sample = inputs.get("images_per_sample")

        if images_per_sample is not None:
            num_grids_per_sample = images_per_sample[0].item()
        else:
            # Fallback for batch_size=1: total grids is for single sample
            num_grids_per_sample = image_grid_thw.shape[0]

        # Compute generation params (same for all samples in homogeneous batch)
        first_sample_grids = image_grid_thw[:num_grids_per_sample]
        max_new_tokens, large_image_offset, token_h, token_w = self._compute_generation_params(
            image_grid_thw=first_sample_grids
        )

        # Generate with AR model
        # Set torch random seed from generator for reproducibility
        # (transformers generate() doesn't accept generator parameter)
        # fork_rng keeps the reseed out of the process-global RNG
        fork_devices = [device] if device is not None and device.type == "cuda" else []
        with torch.random.fork_rng(devices=fork_devices, enabled=generator is not None):
            if generator is not None:
                seed = generator.initial_seed()
                torch.manual_seed(seed)
                if device is not None and device.type == "cuda":
                    torch.cuda.manual_seed(seed)
            outputs = self.vision_language_encoder.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
            )

        # Extract and upsample prior tokens for each sample
        # For left-padded inputs, generated tokens start after the padded input sequence
        all_prior_token_ids = []
        max_input_length = inputs["input_ids"].shape[-1]
        for idx in range(batch_size):
            # For left-padded sequences, generated tokens start at max_input_length
            # (padding is on the left, so all sequences end at the same position)
            prior_token_ids_d32 = self._extract_large_image_tokens(
                outputs[idx : idx + 1], max_input_length, large_image_offset, token_h * token_w
            )
            prior_token_ids = self._upsample_token_ids(prior_token_ids_d32, token_h, token_w)
            all_prior_token_ids.append(prior_token_ids)
        return torch.cat(all_prior_token_ids, dim=0)

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        do_classifier_free_guidance: bool = True,
        num_images_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        max_sequence_length: int = 2048,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `list[str]`, *optional*):
                prompt to be encoded
            do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
                Whether to use classifier free guidance or not.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                Number of images that should be generated per prompt. torch device to place the resulting embeddings on
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            device: (`torch.device`, *optional*):
                torch device
            dtype: (`torch.dtype`, *optional*):
                torch dtype
            max_sequence_length (`int`, defaults to `2048`):
                Maximum sequence length in encoded prompt. Can be set to other values but may lead to poorer results.
        """
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds = self._get_glyph_embeds(prompt, max_sequence_length, device, dtype)

        # Repeat embeddings for num_images_per_prompt
        if num_images_per_prompt > 1:
            prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)

        # For GLM-Image, negative_prompt must be "" instead of None
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = ""
            negative_prompt = (
                batch_size * [negative_prompt]
                if isinstance(negative_prompt, str)
                else negative_prompt
            )
            negative_prompt_embeds = self._get_glyph_embeds(
                negative_prompt, max_sequence_length, device, dtype
            )

            if num_images_per_prompt > 1:
                negative_prompt_embeds = negative_prompt_embeds.repeat_interleave(
                    num_images_per_prompt, dim=0
                )

        return prompt_embeds, negative_prompt_embeds

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]],
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if latents is not None:
            return latents.to(device)

        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return latents

    @staticmethod
    def _compute_generation_params(image_grid_thw: torch.Tensor) -> Tuple[int, int, int, int]:
        grid_sizes = []
        grid_hw = []

        for i in range(image_grid_thw.shape[0]):
            t, h, w = image_grid_thw[i].tolist()
            grid_sizes.append(int(h * w))
            grid_hw.append((int(h), int(w)))

        max_new_tokens = sum(grid_sizes) + 1
        large_image_start_offset = sum(grid_sizes[1:])
        target_grid_h, target_grid_w = grid_hw[0]
        return max_new_tokens, large_image_start_offset, target_grid_h, target_grid_w

    @staticmethod
    def _extract_large_image_tokens(
        outputs: torch.Tensor,
        input_length: int,
        large_image_start_offset: int,
        large_image_tokens: int,
    ) -> torch.Tensor:
        generated_tokens = outputs[0][input_length:]
        large_image_start = large_image_start_offset
        large_image_end = large_image_start + large_image_tokens
        return generated_tokens[large_image_start:large_image_end]

    @staticmethod
    def _upsample_token_ids(token_ids: torch.Tensor, token_h: int, token_w: int) -> torch.Tensor:
        token_ids = token_ids.view(1, 1, token_h, token_w)
        token_ids = torch.nn.functional.interpolate(
            token_ids.float(), scale_factor=2, mode="nearest"
        ).to(dtype=torch.long)
        token_ids = token_ids.view(1, -1)
        return token_ids

    @property
    def do_classifier_free_guidance(self) -> bool:
        return self._guidance_scale > 1

    def _get_glyph_embeds(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        max_sequence_length: int = 2048,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Get glyph embeddings for each prompt in the batch."""
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        # get_glyph_texts now returns a list of lists (one per prompt)
        all_glyph_texts = self.get_glyph_texts(prompt)

        all_glyph_embeds = []
        for glyph_texts in all_glyph_texts:
            if len(glyph_texts) == 0:
                glyph_texts = [""]
            input_ids = self.tokenizer(
                glyph_texts,
                max_length=max_sequence_length,
                truncation=True,
            ).input_ids
            input_ids = [
                [self.tokenizer.pad_token_id] * ((len(input_ids) + 1) % 2) + input_ids_
                for input_ids_ in input_ids
            ]
            max_length = max(len(input_ids_) for input_ids_ in input_ids)
            attention_mask = torch.tensor(
                [
                    [1] * len(input_ids_) + [0] * (max_length - len(input_ids_))
                    for input_ids_ in input_ids
                ],
                device=device,
            )
            input_ids = torch.tensor(
                [
                    input_ids_ + [self.tokenizer.pad_token_id] * (max_length - len(input_ids_))
                    for input_ids_ in input_ids
                ],
                device=device,
            )
            outputs = self.text_encoder(input_ids, attention_mask=attention_mask)
            glyph_embeds = outputs.last_hidden_state[attention_mask.bool()].unsqueeze(0)
            all_glyph_embeds.append(glyph_embeds)

        # Pad to same sequence length and stack (use left padding to match transformers)
        max_seq_len = max(emb.size(1) for emb in all_glyph_embeds)
        padded_embeds = []
        for emb in all_glyph_embeds:
            if emb.size(1) < max_seq_len:
                pad = torch.zeros(
                    emb.size(0),
                    max_seq_len - emb.size(1),
                    emb.size(2),
                    device=device,
                    dtype=emb.dtype,
                )
                emb = torch.cat([pad, emb], dim=1)  # left padding
            padded_embeds.append(emb)

        glyph_embeds = torch.cat(padded_embeds, dim=0)
        return glyph_embeds.to(device=device, dtype=dtype)

    def get_glyph_texts(self, prompt: Union[str, List[str]]) -> List[List[str]]:
        """Extract glyph texts from prompt(s). Returns a list of lists for batch processing."""
        if isinstance(prompt, str):
            prompt = [prompt]
        all_ocr_texts = []
        for p in prompt:
            ocr_texts = (
                re.findall(r"'([^']*)'", p)
                + re.findall(r"\u201c([^\u201c\u201d]*)\u201d", p)
                + re.findall(r'"([^"]*)"', p)
                + re.findall(r"「([^「」]*)」", p)
            )
            all_ocr_texts.append(ocr_texts)
        return all_ocr_texts

    @property
    def guidance_scale(self) -> float:
        return self._guidance_scale

    # ------------------------------------------------------------------
    # TRT-LLM
    # ------------------------------------------------------------------
    def __init__(self, pipeline_config: DiffusionPipelineConfig) -> None:
        super().__init__(pipeline_config)

    def load_standard_components(
        self,
        checkpoint_dir: str,
        device: torch.device,
        skip_components: Optional[list] = None,
        **kwargs: Any,
    ) -> None:
        skip_components = skip_components or []

        # Tokenizer (ByT5Tokenizer)
        if PipelineComponent.TOKENIZER not in skip_components:
            logger.info("Loading tokenizer (ByT5Tokenizer)...")
            tokenizer_path = os.path.join(checkpoint_dir, PipelineComponent.TOKENIZER)
            self.tokenizer = ByT5Tokenizer.from_pretrained(
                tokenizer_path, torch_dtype=self.pipeline_config.torch_dtype
            )

        # Text Encoder (T5EncoderModel)
        if PipelineComponent.TEXT_ENCODER not in skip_components:
            logger.info("Loading text encoder (T5EncoderModel)...")
            text_encoder_path = os.path.join(checkpoint_dir, PipelineComponent.TEXT_ENCODER)
            self.text_encoder = T5EncoderModel.from_pretrained(
                text_encoder_path, torch_dtype=self.pipeline_config.torch_dtype
            ).to(device)

        # VAE (AutoencoderKL)
        if PipelineComponent.VAE not in skip_components:
            logger.info("Loading VAE (AutoencoderKL)...")
            vae_path = os.path.join(checkpoint_dir, PipelineComponent.VAE)
            self.vae = AutoencoderKL.from_pretrained(
                vae_path, torch_dtype=self.pipeline_config.torch_dtype
            ).to(device)

            self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        # Scheduler (FlowMatchEulerDiscreteScheduler)
        if PipelineComponent.SCHEDULER not in skip_components:
            logger.info("Loading Scheduler (FlowMatchEulerDiscreteScheduler)...")
            scheduler_path = os.path.join(checkpoint_dir, PipelineComponent.SCHEDULER)
            self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                scheduler_path, torch_dtype=self.pipeline_config.torch_dtype
            )

        # Vision Language Encoder (GlmImageForConditionalGeneration)
        if PipelineComponent.VISION_LANGUAGE_ENCODER not in skip_components:
            logger.info("Loading processor (GlmImageProcessor)...")
            processor_path = os.path.join(checkpoint_dir, "processor")
            self.processor = GlmImageProcessor.from_pretrained(processor_path)

            logger.info("Loading vision language encoder (GlmImageForConditionalGeneration)...")
            vision_language_encoder_path = os.path.join(
                checkpoint_dir, PipelineComponent.VISION_LANGUAGE_ENCODER
            )
            self.vision_language_encoder = GlmImageForConditionalGeneration.from_pretrained(
                vision_language_encoder_path, torch_dtype=self.pipeline_config.torch_dtype
            ).to(device)

    @property
    def device(self) -> torch.device:
        if self.transformer is not None:
            return next(self.transformer.parameters()).device
        return torch.device("cuda:0")

    @property
    def dtype(self) -> torch.dtype:
        return self.pipeline_config.torch_dtype

    @property
    def default_generation_params(self) -> dict:
        """Model-specific defaults for None fields in VisualGenParams."""
        return {
            "height": 1024,
            "width": 1024,
            "num_inference_steps": 50,
            "guidance_scale": 1.5,
            "max_sequence_length": 2048,
        }

    @property
    def default_warmup_resolutions(self) -> List[Tuple[int, int]]:
        return [(1024, 1024)]

    @property
    def default_warmup_num_frames(self) -> List[int]:
        # Image model: a single "frame" per sample.
        return [1]

    @property
    def resolution_multiple_of(self) -> Tuple[int, int]:
        # vae_scale_factor * patch_size is the 16px latent pitch; the AR prior grid is
        # generated at d32 and upsampled 2x, so the resolution must clear 32.
        patch_size = self.transformer.config.patch_size if self.transformer is not None else 2
        multiple = getattr(self, "vae_scale_factor", 8) * patch_size * 2
        return (multiple, multiple)

    def infer(self, req: Any) -> PipelineOutput:
        """Run inference from DiffusionRequest."""
        params = req.params
        if getattr(params, "image", None) is not None:
            raise NotImplementedError(
                "The TensorRT-LLM GlmImage pipeline is text-to-image only; "
                "image-to-image conditioning is not available."
            )
        generator = None
        if params.seed is not None:
            generator = make_noise_generator(params.seed, self.device)
        return self.forward(
            prompt=req.prompt,
            height=params.height,
            width=params.width,
            num_inference_steps=params.num_inference_steps,
            guidance_scale=params.guidance_scale,
            generator=generator,
            num_images_per_prompt=params.num_images_per_prompt,
            max_sequence_length=params.max_sequence_length,
        )

    def _run_warmup(self, height: int, width: int, num_frames: int, steps: int) -> None:
        with torch.no_grad():
            self.forward(
                prompt="warmup",
                height=height,
                width=width,
                num_inference_steps=steps,
                generator=make_noise_generator(42, self.device),
            )

    @torch.inference_mode()
    def forward(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        image: Optional[
            Union[
                torch.Tensor,
                PIL.Image.Image,
                np.ndarray,
                List[torch.Tensor],
                List[PIL.Image.Image],
                List[np.ndarray],
            ]
        ] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: Optional[List[int]] = None,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 1.5,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        prior_token_ids: Optional[torch.Tensor] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        attention_kwargs: Optional[Dict[str, Any]] = None,
        max_sequence_length: int = 2048,
    ) -> PipelineOutput:
        """
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `list[str]`, *optional*):
                The prompt or prompts to guide the image generation.
            image: Must be `None`; this pipeline is text-to-image.
            height (`int`, *optional*):
                The height in pixels, a multiple of 32. Defaults to 1024.
            width (`int`, *optional*):
                The width in pixels, a multiple of 32. Defaults to 1024.
            num_inference_steps (`int`, *optional*, defaults to `50`):
                The number of denoising steps for DiT.
            guidance_scale (`float`, *optional*, defaults to `1.5`):
                Guidance scale for classifier-free guidance.
            num_images_per_prompt (`int`, *optional*, defaults to `1`):
                The number of images to generate per prompt.
            generator (`torch.Generator`, *optional*):
                Random generator for reproducibility.

        Returns:
            PipelineOutput with image tensor ``(B, H, W, C)`` dtype uint8.
        """
        if image is not None:
            raise NotImplementedError(
                "The TensorRT-LLM GlmImage pipeline is text-to-image only; "
                "image-to-image conditioning is not available."
            )

        pipeline_start = time.time()
        timer = CudaPhaseTimer()
        timer.mark_pre_start()

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None

        # forward() can be called directly, bypassing infer()'s VisualGenParams defaults
        if height is None:
            height = self.default_generation_params["height"]
        if width is None:
            width = self.default_generation_params["width"]
        self.validate_resolution(height, width, 1)

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self.device

        # 1. Generate prior tokens (batch mode)
        # Get a single generator for AR model (use first if list provided)
        logger.info("Generating prior tokens...")
        ar_generator = generator[0] if isinstance(generator, list) else generator
        if prior_token_ids is None:
            prior_token_ids = self.generate_prior_tokens(
                prompt=prompt,
                height=height,
                width=width,
                device=device,
                generator=ar_generator,
            )

        # 2. Encode input prompt
        logger.info("Encoding prompt...")
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            self.do_classifier_free_guidance,
            num_images_per_prompt=num_images_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=self.dtype,
        )

        # 3. Prepare latents
        latent_channels = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size=batch_size * num_images_per_prompt,
            num_channels_latents=latent_channels,
            height=height,
            width=width,
            dtype=prompt_embeds.dtype,
            device=device,
            generator=generator,
            latents=latents,
        )

        # 4. Prepare additional timestep conditions
        target_size = (height, width)
        target_size = torch.tensor([target_size], dtype=prompt_embeds.dtype, device=device)
        crops_coords_top_left = torch.tensor(
            [crops_coords_top_left], dtype=prompt_embeds.dtype, device=device
        )

        target_size = target_size.repeat(batch_size * num_images_per_prompt, 1)
        crops_coords_top_left = crops_coords_top_left.repeat(batch_size * num_images_per_prompt, 1)

        # Prepare timesteps
        image_seq_len = ((height // self.vae_scale_factor) * (width // self.vae_scale_factor)) // (
            self.transformer.config.patch_size**2
        )
        timesteps = (
            np.linspace(self.scheduler.config.num_train_timesteps, 1.0, num_inference_steps + 1)[
                :-1
            ]
            if timesteps is None
            else np.array(timesteps)
        )
        timesteps = timesteps.astype(np.int64).astype(np.float32)
        sigmas = timesteps / self.scheduler.config.num_train_timesteps if sigmas is None else sigmas
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("base_shift", 0.25),
            self.scheduler.config.get("max_shift", 0.75),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas, mu=mu
        )
        self._num_timesteps = len(timesteps)

        # 5. Denoising loop
        transformer_dtype = self.dtype

        # Repeat prior_token_ids for num_images_per_prompt
        if num_images_per_prompt > 1:
            prior_token_ids = prior_token_ids.repeat_interleave(num_images_per_prompt, dim=0)
        prior_token_drop_cond = torch.full_like(prior_token_ids, False, dtype=torch.bool)
        prior_token_drop_uncond = torch.full_like(prior_token_ids, True, dtype=torch.bool)

        # Batched CFG concatenates the positive/negative streams, so align their
        # lengths and carry a text attention mask (None when no padding is needed).
        text_attention_mask = neg_text_attention_mask = None
        if self.do_classifier_free_guidance:
            (
                prompt_embeds,
                negative_prompt_embeds,
                text_attention_mask,
                neg_text_attention_mask,
            ) = self._align_cfg_embeds(prompt_embeds, negative_prompt_embeds)

        def forward_fn(
            latents: torch.Tensor,
            extra_stream_latents: Optional[torch.Tensor],
            step_index: int,
            timestep: torch.Tensor,
            encoder_hidden_states: torch.Tensor,
            extra_tensors: Dict[str, Any],
        ) -> torch.Tensor:
            """Forward function for GlmImage transformer."""
            return self.transformer(
                hidden_states=latents.to(transformer_dtype),
                encoder_hidden_states=encoder_hidden_states,
                prior_token_id=extra_tensors["prior_token_id"],
                prior_token_drop=extra_tensors["prior_token_drop"],
                timestep=(timestep - 1) / self.scheduler.config.num_train_timesteps,
                target_size=extra_tensors["target_size"],
                crop_coords=extra_tensors["crop_coords"],
                attention_mask=extra_tensors.get("attention_mask"),
                attention_kwargs=attention_kwargs,
                return_dict=False,
            )[0].float()

        extra_cfg_tensors = {
            "prior_token_id": (prior_token_ids, prior_token_ids),
            "prior_token_drop": (prior_token_drop_cond, prior_token_drop_uncond),
            "target_size": (target_size, target_size),
            "crop_coords": (crops_coords_top_left, crops_coords_top_left),
        }
        if text_attention_mask is not None:
            extra_cfg_tensors["attention_mask"] = (text_attention_mask, neg_text_attention_mask)

        timer.mark_denoise_start()
        latents = self.denoise(
            latents=latents,
            scheduler=self.scheduler,
            prompt_embeds=prompt_embeds,
            guidance_scale=self.guidance_scale,
            forward_fn=forward_fn,
            timesteps=timesteps,
            neg_prompt_embeds=(
                negative_prompt_embeds if self.do_classifier_free_guidance else None
            ),
            extra_cfg_tensors=extra_cfg_tensors,
        )
        timer.mark_post_start()

        # Decode
        logger.info("Decoding...")
        image = self.decode_latents(latents, lambda lat: self._decode_latents(lat, generator))

        if self.rank == 0:
            logger.info("Pipeline total: %.2fs", time.time() - pipeline_start)

        timer.mark_end()
        return timer.fill(PipelineOutput(image=image))

    def _decode_latents(
        self, latents: torch.Tensor, generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        latents = latents.to(self.vae.dtype)
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.latent_channels, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = (
            torch.tensor(self.vae.config.latents_std)
            .view(1, self.vae.config.latent_channels, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents = latents * latents_std + latents_mean
        image = self.vae.decode(latents, return_dict=False, generator=generator)[0]

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
        image = (image * 255).round().to(torch.uint8)
        return image

    @staticmethod
    def _align_cfg_embeds(
        prompt_embeds: torch.Tensor, negative_prompt_embeds: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Left-pad positive/negative encoder streams to a common length for batched CFG.

        Returns the (padded) embeds and text attention masks, or ``None`` masks when both
        streams already match and no padding is needed.
        """
        pos_len = prompt_embeds.shape[1]
        neg_len = negative_prompt_embeds.shape[1]
        if pos_len == neg_len:
            return prompt_embeds, negative_prompt_embeds, None, None

        max_len = max(pos_len, neg_len)

        def _left_pad(embeds: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            batch, seq_len, dim = embeds.shape
            mask = torch.ones(batch, seq_len, device=embeds.device, dtype=torch.long)
            if seq_len < max_len:
                pad_len = max_len - seq_len
                embeds = torch.cat([embeds.new_zeros(batch, pad_len, dim), embeds], dim=1)
                mask = torch.cat([mask.new_zeros(batch, pad_len), mask], dim=1)
            return embeds, mask

        prompt_embeds, pos_mask = _left_pad(prompt_embeds)
        negative_prompt_embeds, neg_mask = _left_pad(negative_prompt_embeds)
        return prompt_embeds, negative_prompt_embeds, pos_mask, neg_mask

    def load_weights(self, weights: dict) -> None:
        """Load transformer weights."""
        if self.transformer is not None and hasattr(self.transformer, "load_weights"):
            logger.info("Loading transformer weights...")
            transformer_weights = weights.get("transformer", weights)
            self.transformer.load_weights(transformer_weights)
            logger.info("Transformer weights loaded successfully.")

        self._target_dtype = self.pipeline_config.torch_dtype

        if self.transformer is not None:
            self.transformer.eval()

    def _init_transformer(self) -> None:
        """Initialize GlmImage transformer with quantization support."""
        logger.info("Creating GlmImage transformer with quantization support...")
        self.transformer = GlmImageTransformer2DModel(
            model_config=self.pipeline_config.model_configs["transformer"]
        )
