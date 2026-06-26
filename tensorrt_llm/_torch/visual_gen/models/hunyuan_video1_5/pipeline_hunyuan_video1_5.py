# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""HunyuanVideo 1.5 text-to-video pipeline."""

import inspect
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from diffusers.guiders import ClassifierFreeGuidance
from diffusers.models import AutoencoderKLHunyuanVideo15
from diffusers.pipelines.hunyuan_video1_5.image_processor import HunyuanVideo15ImageProcessor
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor
from transformers import ByT5Tokenizer, Qwen2_5_VLTextModel, Qwen2Tokenizer, T5EncoderModel

from tensorrt_llm import logger
from tensorrt_llm._torch.visual_gen.config import DiffusionPipelineConfig
from tensorrt_llm._torch.visual_gen.output import CudaPhaseTimer, PipelineOutput
from tensorrt_llm._torch.visual_gen.pipeline import BasePipeline
from tensorrt_llm._torch.visual_gen.pipeline_registry import PipelineComponent, register_pipeline
from tensorrt_llm._torch.visual_gen.utils import postprocess_video_tensor

from .transformer_hunyuan_video1_5 import HunyuanVideo15Transformer3DModel


# Copied from HF
def format_text_input(prompt: List[str], system_message: str) -> List[Dict[str, Any]]:
    """
    Apply text to template.

    Args:
        prompt (list[str]): Input text.
        system_message (str): System message.

    Returns:
        list[dict[str, Any]]: List of chat conversation.
    """

    template = [
        [
            {"role": "system", "content": system_message},
            {"role": "user", "content": p if p else " "},
        ]
        for p in prompt
    ]

    return template


# Copied from HF
def extract_glyph_texts(prompt: str) -> List[str]:
    """
    Extract glyph texts from prompt using regex pattern.

    Args:
        prompt: Input prompt string

    Returns:
        List of extracted glyph texts
    """
    pattern = r"\"(.*?)\"|“(.*?)”"
    matches = re.findall(pattern, prompt)
    result = [match[0] or match[1] for match in matches]
    result = list(dict.fromkeys(result)) if len(result) > 1 else result

    if result:
        formatted_result = ". ".join([f'Text "{text}"' for text in result]) + ". "
    else:
        formatted_result = None

    return formatted_result


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used,
            `timesteps` must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
                Custom timesteps used to support arbitrary spacing between timesteps. If `None`, then the default
                timestep spacing strategy of the scheduler is used. If `timesteps` is passed, `num_inference_steps`
                must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


@register_pipeline(
    "HunyuanVideo15Pipeline",
    hf_ids=[
        "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v",
        "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v",
        "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v_distilled",
    ],
    doc="Tencent HunyuanVideo 1.5 family (text-to-video).",
)
class HunyuanVideo15Pipeline(BasePipeline):
    def __init__(self, pipeline_config: DiffusionPipelineConfig):
        super().__init__(pipeline_config)

    # Copied from HF
    @staticmethod
    def _get_mllm_prompt_embeds(
        text_encoder: Qwen2_5_VLTextModel,
        tokenizer: Qwen2Tokenizer,
        prompt: Union[str, List[str]],
        device: torch.device,
        tokenizer_max_length: int = 1000,
        num_hidden_layers_to_skip: int = 2,
        # fmt: off
        system_message: str = "You are a helpful assistant. Describe the video by detailing the following aspects: \
        1. The main content and theme of the video. \
        2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects. \
        3. Actions, events, behaviors temporal relationships, physical movement changes of the objects. \
        4. background environment, light, style and atmosphere. \
        5. camera angles, movements, and transitions used in the video.",
        # fmt: on
        crop_start: int = 108,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        prompt = [prompt] if isinstance(prompt, str) else prompt

        prompt = format_text_input(prompt, system_message)

        text_inputs = tokenizer.apply_chat_template(
            prompt,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            padding="max_length",
            max_length=tokenizer_max_length + crop_start,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids.to(device=device)
        prompt_attention_mask = text_inputs.attention_mask.to(device=device)

        prompt_embeds = text_encoder(
            input_ids=text_input_ids,
            attention_mask=prompt_attention_mask,
            output_hidden_states=True,
        ).hidden_states[-(num_hidden_layers_to_skip + 1)]

        if crop_start is not None and crop_start > 0:
            prompt_embeds = prompt_embeds[:, crop_start:]
            prompt_attention_mask = prompt_attention_mask[:, crop_start:]

        return prompt_embeds, prompt_attention_mask

    # Copied from HF
    @staticmethod
    def _get_byt5_prompt_embeds(
        tokenizer: ByT5Tokenizer,
        text_encoder: T5EncoderModel,
        prompt: Union[str, List[str]],
        device: torch.device,
        tokenizer_max_length: int = 256,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt

        glyph_texts = [extract_glyph_texts(p) for p in prompt]

        prompt_embeds_list = []
        prompt_embeds_mask_list = []

        for glyph_text in glyph_texts:
            if glyph_text is None:
                glyph_text_embeds = torch.zeros(
                    (1, tokenizer_max_length, text_encoder.config.d_model),
                    device=device,
                    dtype=text_encoder.dtype,
                )
                glyph_text_embeds_mask = torch.zeros(
                    (1, tokenizer_max_length), device=device, dtype=torch.int64
                )
            else:
                txt_tokens = tokenizer(
                    glyph_text,
                    padding="max_length",
                    max_length=tokenizer_max_length,
                    truncation=True,
                    add_special_tokens=True,
                    return_tensors="pt",
                ).to(device)

                glyph_text_embeds = text_encoder(
                    input_ids=txt_tokens.input_ids,
                    attention_mask=txt_tokens.attention_mask.float(),
                )[0]
                glyph_text_embeds = glyph_text_embeds.to(device=device)
                glyph_text_embeds_mask = txt_tokens.attention_mask.to(device=device)

            prompt_embeds_list.append(glyph_text_embeds)
            prompt_embeds_mask_list.append(glyph_text_embeds_mask)

        prompt_embeds = torch.cat(prompt_embeds_list, dim=0)
        prompt_embeds_mask = torch.cat(prompt_embeds_mask_list, dim=0)

        return prompt_embeds, prompt_embeds_mask

    # Copied from HF
    def _encode_prompt(
        self,
        prompt: Optional[Union[str, List[str]]],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        batch_size: int = 1,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_embeds_mask: Optional[torch.Tensor] = None,
        prompt_embeds_2: Optional[torch.Tensor] = None,
        prompt_embeds_mask_2: Optional[torch.Tensor] = None,
    ):
        r"""

        Args:
            prompt (`str` or `list[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            batch_size (`int`):
                batch size of prompts, defaults to 1
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. If not provided, text embeddings will be generated from `prompt` input
                argument.
            prompt_embeds_mask (`torch.Tensor`, *optional*):
                Pre-generated text mask. If not provided, text mask will be generated from `prompt` input argument.
            prompt_embeds_2 (`torch.Tensor`, *optional*):
                Pre-generated glyph text embeddings from ByT5. If not provided, will be generated from `prompt` input
                argument using self.tokenizer_2 and self.text_encoder_2.
            prompt_embeds_mask_2 (`torch.Tensor`, *optional*):
                Pre-generated glyph text mask from ByT5. If not provided, will be generated from `prompt` input
                argument using self.tokenizer_2 and self.text_encoder_2.
        """
        dtype = dtype or self.text_encoder.dtype

        if prompt is None:
            prompt = [""] * batch_size
        # Broadcast a single prompt string to batch_size so prompt and negative prompt
        # can have different batch sizes.
        elif isinstance(prompt, str):
            prompt = [prompt] * batch_size

        if prompt_embeds is None:
            prompt_embeds, prompt_embeds_mask = self._get_mllm_prompt_embeds(
                tokenizer=self.tokenizer,
                text_encoder=self.text_encoder,
                prompt=prompt,
                device=device,
                tokenizer_max_length=self.tokenizer_max_length,
                system_message=self.system_message,
                crop_start=self.prompt_template_encode_start_idx,
            )

        if prompt_embeds_2 is None:
            prompt_embeds_2, prompt_embeds_mask_2 = self._get_byt5_prompt_embeds(
                tokenizer=self.tokenizer_2,
                text_encoder=self.text_encoder_2,
                prompt=prompt,
                device=device,
                tokenizer_max_length=self.tokenizer_2_max_length,
            )

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)
        prompt_embeds_mask = prompt_embeds_mask.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds_mask = prompt_embeds_mask.view(batch_size * num_videos_per_prompt, seq_len)

        _, seq_len_2, _ = prompt_embeds_2.shape
        prompt_embeds_2 = prompt_embeds_2.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds_2 = prompt_embeds_2.view(batch_size * num_videos_per_prompt, seq_len_2, -1)
        prompt_embeds_mask_2 = prompt_embeds_mask_2.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds_mask_2 = prompt_embeds_mask_2.view(
            batch_size * num_videos_per_prompt, seq_len_2
        )

        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds_mask = prompt_embeds_mask.to(dtype=dtype, device=device)
        prompt_embeds_2 = prompt_embeds_2.to(dtype=dtype, device=device)
        prompt_embeds_mask_2 = prompt_embeds_mask_2.to(dtype=dtype, device=device)

        return prompt_embeds, prompt_embeds_mask, prompt_embeds_2, prompt_embeds_mask_2

    # Copied from HF
    def prepare_cond_latents_and_mask(
        self, latents, dtype: Optional[torch.dtype], device: Optional[torch.device]
    ):
        batch, channels, frames, height, width = latents.shape
        cond_latents_concat = torch.zeros(
            batch, channels, frames, height, width, dtype=dtype, device=device
        )
        mask_concat = torch.zeros(batch, 1, frames, height, width, dtype=dtype, device=device)
        return cond_latents_concat, mask_concat

    def _prepare_mask(self, latents, dtype: Optional[torch.dtype], device: Optional[torch.device]):
        return torch.zeros(*latents.shape, dtype=dtype, device=device)

    def _init_transformer(self) -> None:
        """Initialize HunyuanVideo1.5 transformer with quantization support."""
        logger.info("Creating HunyuanVideo1.5  transformer with quantization support...")
        self.transformer = HunyuanVideo15Transformer3DModel(
            model_config=self.pipeline_config.model_configs["transformer"]
        )

    def load_standard_components(
        self,
        checkpoint_dir: str,
        device: torch.device,
        skip_components: Optional[List] = None,
        **kwargs,
    ) -> None:
        skip_components = skip_components or []

        # Tokenizer (Qwen2Tokenizer)
        if PipelineComponent.TOKENIZER not in skip_components:
            logger.info("Loading tokenizer (Qwen2Tokenizer)...")
            tokenizer_path = os.path.join(checkpoint_dir, PipelineComponent.TOKENIZER)
            self.tokenizer = Qwen2Tokenizer.from_pretrained(
                tokenizer_path, torch_dtype=self.pipeline_config.torch_dtype
            )

        # Tokenizer 2 (ByT5Tokenizer)
        if PipelineComponent.TOKENIZER_2 not in skip_components:
            logger.info("Loading tokenizer 2 (ByT5Tokenizer)...")
            tokenizer_2_path = os.path.join(checkpoint_dir, PipelineComponent.TOKENIZER_2)
            self.tokenizer_2 = ByT5Tokenizer.from_pretrained(
                tokenizer_2_path, torch_dtype=self.pipeline_config.torch_dtype
            )

        # Text Encoder (Qwen2.5-VL-7B-Instruct)
        if PipelineComponent.TEXT_ENCODER not in skip_components:
            logger.info("Loading text encoder (Qwen2.5-VL-7B-Instruct)...")
            text_encoder_path = os.path.join(checkpoint_dir, PipelineComponent.TEXT_ENCODER)
            self.text_encoder = Qwen2_5_VLTextModel.from_pretrained(
                text_encoder_path, torch_dtype=self.pipeline_config.torch_dtype
            ).to(device)

        # Text Encoder 2 (T5EncoderModel)
        if PipelineComponent.TEXT_ENCODER_2 not in skip_components:
            logger.info("Loading text encoder 2 (T5EncoderModel)...")
            text_encoder_2_path = os.path.join(checkpoint_dir, PipelineComponent.TEXT_ENCODER_2)
            self.text_encoder_2 = T5EncoderModel.from_pretrained(
                text_encoder_2_path, torch_dtype=self.pipeline_config.torch_dtype
            ).to(device)

        # VAE (AutoencoderKLHunyuanVideo15)
        if PipelineComponent.VAE not in skip_components:
            logger.info("Loading VAE (AutoencoderKLHunyuanVideo15)...")
            vae_path = os.path.join(checkpoint_dir, PipelineComponent.VAE)
            self.vae = AutoencoderKLHunyuanVideo15.from_pretrained(
                vae_path, torch_dtype=self.pipeline_config.torch_dtype
            ).to(device)

        # Scheduler (FlowMatchEulerDiscreteScheduler)
        if PipelineComponent.SCHEDULER not in skip_components:
            logger.info("Loading Scheduler (FlowMatchEulerDiscreteScheduler)...")
            scheduler_path = os.path.join(checkpoint_dir, PipelineComponent.SCHEDULER)
            self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                scheduler_path, torch_dtype=self.pipeline_config.torch_dtype
            )

        if PipelineComponent.GUIDER not in skip_components:
            logger.info("Loading Guider (ClassifierFreeGuidance)...")
            guider_path = os.path.join(checkpoint_dir, PipelineComponent.GUIDER)
            self.guider = ClassifierFreeGuidance.from_pretrained(
                guider_path, torch_dtype=self.pipeline_config.torch_dtype
            )

        # HunyuanVideo-1.5 Config

        # Default from HF
        # fmt: off
        self.system_message = "You are a helpful assistant. Describe the video by detailing the following aspects: \
        1. The main content and theme of the video. \
        2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects. \
        3. Actions, events, behaviors temporal relationships, physical movement changes of the objects. \
        4. background environment, light, style and atmosphere. \
        5. camera angles, movements, and transitions used in the video."
        # fmt: on

        self.vae_scale_factor_temporal = (
            self.vae.temporal_compression_ratio if getattr(self, "vae", None) else 4
        )
        self.vae_scale_factor_spatial = (
            self.vae.spatial_compression_ratio if getattr(self, "vae", None) else 16
        )
        self.video_processor = HunyuanVideo15ImageProcessor(
            vae_scale_factor=self.vae_scale_factor_spatial
        )
        self.target_size = (
            self.transformer.config.target_size if getattr(self, "transformer", None) else 640
        )
        self.vision_states_dim = (
            self.transformer.config.image_embed_dim if getattr(self, "transformer", None) else 1152
        )

        self.num_channels_latents = (
            self.vae.config.latent_channels if getattr(self, "vae", None) else 32
        )
        self.prompt_template_encode_start_idx = 108
        self.tokenizer_max_length = 1000
        self.tokenizer_2_max_length = 256
        self.vision_num_semantic_tokens = 729
        self.default_aspect_ratio = (16, 9)

    # Copied from HF
    def _prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int = 32,
        height: int = 720,
        width: int = 1280,
        num_frames: int = 129,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        shape = (
            batch_size,
            num_channels_latents,
            (num_frames - 1) // self.vae_scale_factor_temporal + 1,
            int(height) // self.vae_scale_factor_spatial,
            int(width) // self.vae_scale_factor_spatial,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return latents

    def _decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents to a ``(B, T, H, W, C)`` uint8 video tensor."""

        latents = latents.to(self.vae.dtype) / self.vae.config.scaling_factor
        video = self.vae.decode(latents, return_dict=False)[0]
        return postprocess_video_tensor(video)

    @property
    def device(self):
        if self.transformer is not None:
            return next(self.transformer.parameters()).device
        return torch.device("cuda:0")

    @property
    def dtype(self):
        return self.pipeline_config.torch_dtype

    @property
    def default_generation_params(self) -> dict:
        """Model-specific defaults for None fields in VisualGenParams."""
        return {
            "height": 480,
            "width": 832,
            "num_inference_steps": 50,
            "num_frames": 121,
            "frame_rate": 24.0,
        }

    @property
    def default_warmup_resolutions(self) -> List[Tuple[int, int]]:
        # Only the 480p checkpoint is supported; 720p (1280) is not a multiple of
        # resolution_multiple_of (32), so it would be skipped during warmup anyway.
        return [(480, 832)]

    @property
    def default_warmup_num_frames(self) -> List[int]:
        return [121]

    @property
    def resolution_multiple_of(self) -> Tuple[int, int]:
        patch_size = self.transformer.config.patch_size if self.transformer is not None else 2
        return (
            self.vae_scale_factor_spatial * patch_size,
            self.vae_scale_factor_spatial * patch_size,
        )

    def _run_warmup(self, height: int, width: int, num_frames: int, steps: int) -> None:
        with torch.no_grad():
            self.forward(
                prompt="warmup",
                seed=42,
                height=height,
                width=width,
                negative_prompt="",
                num_frames=num_frames,
                num_inference_steps=steps,
            )

    def infer(self, req):
        """Run inference from a DiffusionRequest (serve / high-level API path)."""
        if req.params.image is not None:
            raise ValueError(
                "HunyuanVideo 1.5 currently supports text-to-video only; "
                "image conditioning (I2V) is not supported."
            )
        return self.forward(
            prompt=req.prompt,
            seed=req.params.seed,
            height=req.params.height,
            width=req.params.width,
            negative_prompt=req.params.negative_prompt,
            num_frames=req.params.num_frames,
            num_inference_steps=req.params.num_inference_steps,
            num_videos_per_prompt=req.params.num_images_per_prompt,
        )

    @torch.inference_mode()
    def forward(
        self,
        prompt: Union[str, List[str]],
        seed: int,
        height: int,
        width: int,
        negative_prompt: Union[str, List[str]] = None,
        num_frames: int = 121,
        num_inference_steps: int = 50,
        num_videos_per_prompt: int = 1,
        sigmas: Optional[List[float]] = None,
        output_type: Optional[str] = "np",
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_embeds_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds_mask: Optional[torch.Tensor] = None,
        prompt_embeds_2: Optional[torch.Tensor] = None,
        prompt_embeds_mask_2: Optional[torch.Tensor] = None,
        negative_prompt_embeds_2: Optional[torch.Tensor] = None,
        negative_prompt_embeds_mask_2: Optional[torch.Tensor] = None,
        latents: Optional[torch.Tensor] = None,
    ):
        pipeline_start = time.time()
        timer = CudaPhaseTimer()
        timer.mark_pre_start()

        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if num_videos_per_prompt < 1:
            raise ValueError(f"num_videos_per_prompt must be >= 1, got {num_videos_per_prompt}")

        # Batch generation is unsupported upstream: the diffusers HunyuanVideo1.5 VAE builds a
        # causal attention mask without a head dimension, so scaled_dot_product_attention fails
        # for an effective batch > 1.
        effective_batch_size = batch_size * num_videos_per_prompt
        if effective_batch_size > 1:
            raise ValueError(
                "HunyuanVideo1.5 currently supports an effective batch size of 1 only "
                f"(batch_size={batch_size}, num_videos_per_prompt={num_videos_per_prompt})."
            )

        generator = torch.Generator(device=self.device).manual_seed(seed)

        device = self.device

        # 1. Encode
        prompt_embeds, prompt_embeds_mask, prompt_embeds_2, prompt_embeds_mask_2 = (
            self._encode_prompt(
                prompt=prompt,
                device=device,
                dtype=self.dtype,
                batch_size=batch_size,
                num_videos_per_prompt=num_videos_per_prompt,
                prompt_embeds=prompt_embeds,
                prompt_embeds_mask=prompt_embeds_mask,
                prompt_embeds_2=prompt_embeds_2,
                prompt_embeds_mask_2=prompt_embeds_mask_2,
            )
        )

        # 2. Guidance
        do_cfg = self.guider._enabled and self.guider.num_conditions > 1
        if do_cfg:
            (
                negative_prompt_embeds,
                negative_prompt_embeds_mask,
                negative_prompt_embeds_2,
                negative_prompt_embeds_mask_2,
            ) = self._encode_prompt(
                prompt=negative_prompt,
                device=device,
                dtype=self.dtype,
                batch_size=batch_size,
                num_videos_per_prompt=num_videos_per_prompt,
                prompt_embeds=negative_prompt_embeds,
                prompt_embeds_mask=negative_prompt_embeds_mask,
                prompt_embeds_2=negative_prompt_embeds_2,
                prompt_embeds_mask_2=negative_prompt_embeds_mask_2,
            )

        # 3. Timesteps
        sigmas = np.linspace(1.0, 0.0, num_inference_steps + 1)[:-1] if sigmas is None else sigmas
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, sigmas=sigmas
        )

        # 4. Latents
        latents = self._prepare_latents(
            batch_size * num_videos_per_prompt,
            self.num_channels_latents,
            height,
            width,
            num_frames,
            self.dtype,
            device,
            generator,
            latents,
        )

        cond_latents_concat, mask_concat = self.prepare_cond_latents_and_mask(
            latents, self.dtype, device
        )
        image_embeds = torch.zeros(
            batch_size,
            self.vision_num_semantic_tokens,
            self.vision_states_dim,
            dtype=self.dtype,
            device=device,
        )

        # 5. Run Transformer
        self._num_timesteps = len(timesteps)

        def forward_fn(
            latents, extra_stream_latents, timestep, encoder_hidden_states, extra_tensors
        ):
            # Concatenate the conditioning latents and binary mask onto the channel
            # dim to build the transformer input (in_channels = latent channels +
            # cond latent channels + mask channel). denoise() has already selected
            # the correct cond/uncond branch for every entry in extra_tensors.
            latent_model_input = torch.cat(
                [latents, extra_tensors["cond_latents_concat"], extra_tensors["mask_concat"]],
                dim=1,
            )

            noise_pred = self.transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=extra_tensors["encoder_attention_mask"],
                encoder_hidden_states_2=extra_tensors["encoder_hidden_states_2"],
                encoder_attention_mask_2=extra_tensors["encoder_attention_mask_2"],
                image_embeds=extra_tensors["image_embeds"],
                return_dict=False,
            )[0]
            return noise_pred

        timer.mark_denoise_start()
        # Route every per-batch conditioning input through ``extra_cfg_tensors``
        # so it flows via the same (conditional, unconditional) channel that
        # classifier-free guidance will use. With guidance disabled the negative
        # entry is unused; tensors that are shared across cond/uncond pass the
        # same tensor for both so they duplicate correctly once CFG is enabled.
        latents = self.denoise(
            latents,
            self.scheduler,
            prompt_embeds=prompt_embeds,
            neg_prompt_embeds=negative_prompt_embeds if do_cfg else None,
            guidance_scale=self.guider.guidance_scale if do_cfg else 1.0,
            guidance_rescale=self.guider.guidance_rescale if do_cfg else 0.0,
            forward_fn=forward_fn,
            extra_cfg_tensors={
                "encoder_attention_mask": (prompt_embeds_mask, negative_prompt_embeds_mask),
                "encoder_hidden_states_2": (prompt_embeds_2, negative_prompt_embeds_2),
                "encoder_attention_mask_2": (prompt_embeds_mask_2, negative_prompt_embeds_mask_2),
                "image_embeds": (image_embeds, image_embeds),
                "cond_latents_concat": (cond_latents_concat, cond_latents_concat),
                "mask_concat": (mask_concat, mask_concat),
            },
        )
        timer.mark_post_start()

        # 6. Decode
        logger.info("Decoding image...")
        decode_start = time.time()

        if output_type != "latent":
            video = self.decode_latents(latents, self._decode_latents)
        else:
            video = latents

        if self.rank == 0:
            logger.info(f"Image decoded in {time.time() - decode_start:.2f}s")
            logger.info(f"Total pipeline time: {time.time() - pipeline_start:.2f}s")

        timer.mark_end()
        return timer.fill(PipelineOutput(video=video, frame_rate=24.0))

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
