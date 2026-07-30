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
"""Qwen-Image-Layered image decomposition pipeline."""

import io
import math
import time
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from tensorrt_llm._torch.visual_gen.output import CudaPhaseTimer, PipelineOutput
from tensorrt_llm._torch.visual_gen.pipeline import BasePipeline, ExtraParamSchema
from tensorrt_llm._torch.visual_gen.pipeline_registry import PipelineComponent, register_pipeline
from tensorrt_llm.logger import logger

from .transformer_qwen_image_layered import QwenImageLayeredTransformer2DModel

_PROMPT_TEMPLATE = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, "
    "size, texture, quantity, text, spatial relationships of the objects and "
    "background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n"
    "<|im_start|>assistant\n"
)
_PROMPT_TEMPLATE_START_IDX = 34
_PROCESSOR_SUBFOLDER = "processor"

_LAYERED_DEFAULT_GENERATION_PARAMS = {
    "height": None,
    "width": None,
    "num_inference_steps": 50,
    "guidance_scale": 4.0,
    "max_sequence_length": 512,
}

_LAYERED_CAPTION_PROMPT = """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
# 图像标注器
你是一个专业的图像标注器。请基于输入图像，撰写图注:
1.
使用自然、描述性的语言撰写图注，不要使用结构化形式或富文本形式。
2. 通过加入以下内容，丰富图注细节：
 - 对象的属性：如数量、颜色、形状、大小、位置、材质、状态、动作等
 -
对象间的视觉关系：如空间关系、功能关系、动作关系、从属关系、比较关系、因果关系等
 - 环境细节：例如天气、光照、颜色、纹理、气氛等
 - 文字内容：识别图像中清晰可见的文字，不做翻译和解释，用引号在图注中强调
3.
保持真实性与准确性：
 - 不要使用笼统的描述
 -
描述图像中所有可见的信息，但不要加入没有在图像中出现的内容
<|vision_start|><|image_pad|><|vision_end|><|im_end|>
<|im_start|>assistant
"""

_LAYERED_CAPTION_PROMPT_EN = """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
# Image Annotator
You are a professional
image annotator. Please write an image caption based on the input image:
1. Write the caption using natural,
descriptive language without structured formats or rich text.
2. Enrich caption details by including:\x20
 - Object
attributes, such as quantity, color, shape, size, material, state, position, actions, and so on
 - Vision Relations
between objects, such as spatial relations, functional relations, possessive relations, attachment relations, action
relations, comparative relations, causal relations, and so on
 - Environmental details, such as weather, lighting,
colors, textures, atmosphere, and so on
 - Identify the text clearly visible in the image, without translation or
explanation, and highlight it in the caption with quotation marks
3. Maintain authenticity and accuracy:
 - Avoid
generalizations
 - Describe all visible information in the image, while do not add information not explicitly shown in
the image
<|vision_start|><|image_pad|><|vision_end|><|im_end|>
<|im_start|>assistant
"""


def _calculate_dimensions(target_area: int, ratio: float) -> Tuple[int, int]:
    width = math.sqrt(target_area * ratio)
    height = width / ratio
    width = round(width / 32) * 32
    height = round(height / 32) * 32
    return int(width), int(height)


def _retrieve_latents(
    encoder_output: torch.Tensor,
    generator: Optional[torch.Generator] = None,
    sample_mode: str = "sample",
) -> torch.Tensor:
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    if hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    if hasattr(encoder_output, "latents"):
        return encoder_output.latents
    raise AttributeError("Could not access latents of provided encoder_output")


@register_pipeline(
    "QwenImageLayeredPipeline",
    hf_ids=["Qwen/Qwen-Image-Layered"],
    doc=(
        "Qwen-Image-Layered image-conditioned layer decomposition. Loads Diffusers-format "
        "checkpoints with tokenizer, text encoder, transformer, VAE, scheduler, image "
        "processor, and Qwen2VL processor components. Requests take an input image and "
        "optional prompt, then return generated RGBA layers as a saveable image grid. "
        "Use VisualGen.default_params or VisualGen.extra_param_specs to set request knobs "
        "such as extra_params['layers'] and extra_params['resolution']."
    ),
)
class QwenImageLayeredPipeline(BasePipeline):
    """Qwen-Image-Layered image decomposition pipeline."""

    DEFAULT_GENERATION_PARAMS = _LAYERED_DEFAULT_GENERATION_PARAMS

    def __init__(self, pipeline_config):
        super().__init__(pipeline_config)
        self.vae_scale_factor = 8
        self.tokenizer_max_length = 1024
        self.latent_channels = 16

    def _init_transformer(self) -> None:
        logger.info("Creating Qwen-Image-Layered transformer")
        model_config = self.pipeline_config.model_configs["transformer"]
        pretrained = getattr(model_config, "pretrained_config", None)

        def _cfg(name: str, default):
            if pretrained is None:
                return default
            if isinstance(pretrained, dict):
                return pretrained.get(name, default)
            return getattr(pretrained, name, default)

        self.guidance_embeds = _cfg("guidance_embeds", False)
        self.zero_cond_t = _cfg("zero_cond_t", False)
        if self.guidance_embeds:
            raise NotImplementedError("Guidance-distilled Qwen-Image-Layered is not supported.")
        if self.zero_cond_t:
            raise NotImplementedError("Qwen-Image-Layered zero_cond_t is not supported yet.")

        self.transformer = QwenImageLayeredTransformer2DModel(
            model_config=model_config,
            patch_size=_cfg("patch_size", 2),
            in_channels=_cfg("in_channels", 64),
            out_channels=_cfg("out_channels", 16),
            num_layers=_cfg("num_layers", 60),
            attention_head_dim=_cfg("attention_head_dim", 128),
            num_attention_heads=_cfg("num_attention_heads", 24),
            joint_attention_dim=_cfg("joint_attention_dim", 3584),
            axes_dims_rope=tuple(_cfg("axes_dims_rope", (16, 56, 56))),
            use_additional_t_cond=_cfg("use_additional_t_cond", False),
            use_layer3d_rope=_cfg("use_layer3d_rope", False),
        )

    @property
    def default_warmup_resolutions(self) -> List[Tuple[int, int]]:
        return [(640, 640)]

    @property
    def default_warmup_num_frames(self) -> List[int]:
        return [1]

    @property
    def default_generation_params(self) -> dict:
        return dict(_LAYERED_DEFAULT_GENERATION_PARAMS)

    @property
    def resolution_multiple_of(self) -> Tuple[int, int]:
        return (self.vae_scale_factor * 2, self.vae_scale_factor * 2)

    def warmup_cache_key(self, height: Optional[int], width: Optional[int], **kwargs) -> tuple:
        if height is None or width is None:
            height, width = self.default_warmup_resolutions[0]
        return (height, width)

    def _run_warmup(self, height: int, width: int, num_frames: int, steps: int) -> None:
        from PIL import Image

        del num_frames
        resolution = 1024 if max(height, width) > 640 else 640
        dummy_image = Image.new("RGBA", (width, height))
        with torch.no_grad():
            self.forward(
                image=dummy_image,
                prompt="warmup",
                true_cfg_scale=1.0,
                layers=4,
                num_inference_steps=max(steps, 2),
                seed=42,
                max_sequence_length=64,
                resolution=resolution,
            )

    @property
    def extra_param_specs(self) -> dict:
        return {
            "layers": ExtraParamSchema(
                type="int",
                default=4,
                description="Number of latent output layers to generate.",
                range=(1, 16),
            ),
            "resolution": ExtraParamSchema(
                type="int",
                default=640,
                description="Layered model resolution bucket. Supported values: 640 or 1024.",
            ),
            "cfg_normalize": ExtraParamSchema(
                type="bool",
                default=False,
                description="Normalize classifier-free guidance prediction by conditional norm.",
            ),
            "use_en_prompt": ExtraParamSchema(
                type="bool",
                default=False,
                description="Use English auto-caption prompt when prompt is empty.",
            ),
        }

    def load_standard_components(
        self,
        checkpoint_dir: str,
        device: torch.device,
        skip_components: Optional[list] = None,
    ) -> None:
        skip_components = skip_components or []

        if PipelineComponent.TOKENIZER not in skip_components:
            try:
                from transformers import Qwen2Tokenizer
            except ImportError as e:  # pragma: no cover
                raise ImportError(
                    "Qwen-Image-Layered requires transformers with Qwen2Tokenizer."
                ) from e
            logger.info("Loading Qwen2 tokenizer...")
            self.tokenizer = Qwen2Tokenizer.from_pretrained(
                checkpoint_dir, subfolder=PipelineComponent.TOKENIZER
            )

        if PipelineComponent.TEXT_ENCODER not in skip_components:
            try:
                from transformers import Qwen2_5_VLForConditionalGeneration
            except ImportError as e:  # pragma: no cover
                raise ImportError(
                    "Qwen-Image-Layered requires transformers with Qwen2_5_VL."
                ) from e
            logger.info("Loading Qwen2.5-VL text encoder...")
            self.text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                checkpoint_dir,
                subfolder=PipelineComponent.TEXT_ENCODER,
                torch_dtype=self.pipeline_config.torch_dtype,
            ).to(device)

        if PipelineComponent.VAE not in skip_components:
            try:
                from diffusers import AutoencoderKLQwenImage
            except ImportError as e:  # pragma: no cover
                raise ImportError(
                    "Qwen-Image-Layered requires diffusers with AutoencoderKLQwenImage."
                ) from e
            logger.info("Loading Qwen-Image VAE...")
            self.vae = AutoencoderKLQwenImage.from_pretrained(
                checkpoint_dir,
                subfolder=PipelineComponent.VAE,
                torch_dtype=torch.bfloat16,
            ).to(device)
            temperal_downsample = getattr(self.vae, "temperal_downsample", [1, 1, 1])
            self.vae_scale_factor = 2 ** len(temperal_downsample)

        if PipelineComponent.SCHEDULER not in skip_components:
            try:
                from diffusers import FlowMatchEulerDiscreteScheduler
            except ImportError as e:  # pragma: no cover
                raise ImportError(
                    "Qwen-Image-Layered requires diffusers with FlowMatchEulerDiscreteScheduler."
                ) from e
            logger.info("Loading Qwen-Image scheduler...")
            self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                checkpoint_dir, subfolder=PipelineComponent.SCHEDULER
            )

        if PipelineComponent.IMAGE_PROCESSOR not in skip_components:
            try:
                from diffusers.image_processor import VaeImageProcessor
            except ImportError as e:  # pragma: no cover
                raise ImportError(
                    "Qwen-Image-Layered requires diffusers with VaeImageProcessor."
                ) from e
            self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)

        if _PROCESSOR_SUBFOLDER not in skip_components:
            try:
                from transformers import Qwen2VLProcessor
            except ImportError as e:  # pragma: no cover
                raise ImportError(
                    "Qwen-Image-Layered requires transformers with Qwen2VLProcessor."
                ) from e
            logger.info("Loading Qwen2-VL processor...")
            self.vl_processor = Qwen2VLProcessor.from_pretrained(
                checkpoint_dir, subfolder=_PROCESSOR_SUBFOLDER
            )
            self.processor = self.vl_processor

        self.default_height = 640
        self.default_width = 640
        self.max_sequence_length = 512
        if getattr(self, "vae", None) is not None:
            self.latent_channels = self.vae.config.z_dim

    def load_weights(self, weights: dict) -> None:
        if self.transformer is not None:
            transformer_weights = weights.get("transformer", weights)
            self.transformer.load_weights(transformer_weights)
            self.transformer.to_inference_dtype().eval()
        self._target_dtype = self.pipeline_config.torch_dtype

    @staticmethod
    def _load_image_input(image):
        from PIL import Image

        if isinstance(image, list):
            return [QwenImageLayeredPipeline._load_image_input(item) for item in image]
        if isinstance(image, str):
            return Image.open(image).convert("RGBA")
        if isinstance(image, bytes):
            return Image.open(io.BytesIO(image)).convert("RGBA")
        if hasattr(image, "convert") and getattr(image, "mode", None) != "RGBA":
            return image.convert("RGBA")
        return image

    @staticmethod
    def _image_size(image) -> Tuple[int, int]:
        first = image[0] if isinstance(image, list) else image
        if isinstance(first, torch.Tensor):
            if first.ndim == 5:
                return int(first.shape[-1]), int(first.shape[-2])
            if first.ndim == 4:
                return int(first.shape[-1]), int(first.shape[-2])
            if first.ndim == 3:
                return int(first.shape[-1]), int(first.shape[-2])
            raise ValueError(f"Unsupported tensor image shape: {tuple(first.shape)}")
        if hasattr(first, "size"):
            size = first.size
            if isinstance(size, tuple):
                return int(size[0]), int(size[1])
        raise ValueError(f"Unsupported image input type: {type(first).__name__}")

    def _is_layered_latent_image(self, image) -> bool:
        return (
            isinstance(image, torch.Tensor)
            and image.ndim == 5
            and image.shape[1] == self.latent_channels
        )

    @staticmethod
    def _validate_single_conditioning_frame(image: torch.Tensor) -> None:
        if image.shape[2] != 1:
            raise ValueError(
                "Layered latent image inputs must have exactly one conditioning frame "
                f"(F=1), got F={image.shape[2]}."
            )

    @staticmethod
    def _is_empty_prompt(prompt) -> bool:
        return prompt is None or (isinstance(prompt, str) and prompt.strip() == "")

    @staticmethod
    def _image_batch_size(image) -> int:
        if isinstance(image, list):
            return len(image)
        if isinstance(image, torch.Tensor) and image.ndim >= 4:
            return int(image.shape[0])
        return 1

    @staticmethod
    def _expand_values_to_batch(values: List[str], batch_size: int, name: str) -> List[str]:
        if len(values) == batch_size:
            return values
        if len(values) == 1:
            return values * batch_size
        if batch_size % len(values) == 0:
            repeat = batch_size // len(values)
            return [value for value in values for _ in range(repeat)]
        raise ValueError(
            f"Cannot align {name} batch {len(values)} to effective batch {batch_size}."
        )

    @staticmethod
    def _align_prompts_to_image_batch(prompts: List[str], image_batch: int) -> List[str]:
        prompt_batch = len(prompts)
        if prompt_batch == image_batch or image_batch == 1 or prompt_batch % image_batch == 0:
            return prompts
        if prompt_batch == 1:
            return prompts * image_batch
        if image_batch % prompt_batch == 0:
            repeat = image_batch // prompt_batch
            return [prompt for prompt in prompts for _ in range(repeat)]
        raise ValueError(
            f"Prompt batch size {prompt_batch} must match image batch size {image_batch}, "
            "or one must divide the other exactly."
        )

    @staticmethod
    def _repeat_conditioning_batch(
        conditioning: torch.Tensor,
        batch_size: int,
        name: str,
    ) -> torch.Tensor:
        conditioning_batch = conditioning.shape[0]
        if conditioning_batch == batch_size:
            return conditioning
        if batch_size > conditioning_batch and batch_size % conditioning_batch == 0:
            return conditioning.repeat_interleave(batch_size // conditioning_batch, dim=0)
        if batch_size > conditioning_batch:
            raise ValueError(f"Cannot duplicate {name} batch {conditioning_batch} to {batch_size}.")
        raise ValueError(
            f"{name} batch size {conditioning_batch} must match prompt batch size {batch_size}, "
            "or divide it exactly."
        )

    @staticmethod
    def _layer_stack_to_image_grid(layer_stack: torch.Tensor) -> torch.Tensor:
        if layer_stack.ndim != 5:
            raise ValueError(
                "Qwen-Image-Layered output must have shape (B, layers, H, W, C), "
                f"got {tuple(layer_stack.shape)}."
            )
        batch_size, layers, height, width, channels = layer_stack.shape
        grid_cols = math.ceil(math.sqrt(layers))
        grid_rows = math.ceil(layers / grid_cols)
        pad_layers = grid_rows * grid_cols - layers
        if pad_layers:
            padding = layer_stack.new_zeros(batch_size, pad_layers, height, width, channels)
            layer_stack = torch.cat([layer_stack, padding], dim=1)
        grid = layer_stack.reshape(batch_size, grid_rows, grid_cols, height, width, channels)
        grid = grid.permute(0, 1, 3, 2, 4, 5)
        return grid.reshape(batch_size, grid_rows * height, grid_cols * width, channels)

    @staticmethod
    def _extract_masked_hidden(
        hidden_states: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        return torch.split(selected, valid_lengths.tolist(), dim=0)

    def _encode_prompt(
        self,
        prompt: List[str],
        device: torch.device,
        max_sequence_length: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        drop_idx = _PROMPT_TEMPLATE_START_IDX
        txt = [_PROMPT_TEMPLATE.format(e) for e in prompt]
        tok = self.tokenizer(
            txt,
            max_length=self.tokenizer_max_length + drop_idx,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        encoder_outputs = self.text_encoder(
            input_ids=tok.input_ids,
            attention_mask=tok.attention_mask,
            output_hidden_states=True,
        )
        hidden_states = encoder_outputs.hidden_states[-1]

        split_hidden = self._extract_masked_hidden(hidden_states, tok.attention_mask)
        split_hidden = [h[drop_idx:] for h in split_hidden]
        attn_masks = [
            torch.ones(h.size(0), dtype=torch.long, device=h.device) for h in split_hidden
        ]
        max_len = max(h.size(0) for h in split_hidden)
        prompt_embeds = torch.stack(
            [torch.cat([h, h.new_zeros(max_len - h.size(0), h.size(1))]) for h in split_hidden]
        )
        prompt_embeds_mask = torch.stack(
            [torch.cat([m, m.new_zeros(max_len - m.size(0))]) for m in attn_masks]
        )

        prompt_embeds = prompt_embeds[:, :max_sequence_length]
        prompt_embeds_mask = prompt_embeds_mask[:, :max_sequence_length]
        prompt_embeds = prompt_embeds.to(dtype=self.dtype, device=device)
        return prompt_embeds, prompt_embeds_mask

    @staticmethod
    def _pack_layered_latents(
        latents: torch.Tensor,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        layers: int,
    ) -> torch.Tensor:
        latents = latents.view(
            batch_size,
            layers,
            num_channels_latents,
            height // 2,
            2,
            width // 2,
            2,
        )
        latents = latents.permute(0, 1, 3, 5, 2, 4, 6)
        return latents.reshape(
            batch_size,
            layers * (height // 2) * (width // 2),
            num_channels_latents * 4,
        )

    @staticmethod
    def _unpack_layered_latents(
        latents: torch.Tensor,
        height: int,
        width: int,
        layers: int,
        vae_scale_factor: int,
    ) -> torch.Tensor:
        batch_size, _, channels = latents.shape
        h = 2 * (int(height) // (vae_scale_factor * 2))
        w = 2 * (int(width) // (vae_scale_factor * 2))
        latents = latents.view(batch_size, layers + 1, h // 2, w // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 1, 4, 2, 5, 3, 6)
        latents = latents.reshape(batch_size, layers + 1, channels // 4, h, w)
        return latents.permute(0, 2, 1, 3, 4)

    def _encode_vae_image(
        self,
        image: torch.Tensor,
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        if isinstance(generator, list):
            image_latents = [
                _retrieve_latents(
                    self.vae.encode(image[i : i + 1]),
                    generator=generator[i],
                    sample_mode="argmax",
                )
                for i in range(image.shape[0])
            ]
            image_latents = torch.cat(image_latents, dim=0)
        else:
            image_latents = _retrieve_latents(
                self.vae.encode(image),
                generator=generator,
                sample_mode="argmax",
            )
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.latent_channels, 1, 1, 1)
            .to(image_latents.device, image_latents.dtype)
        )
        latents_std = (
            torch.tensor(self.vae.config.latents_std)
            .view(1, self.latent_channels, 1, 1, 1)
            .to(image_latents.device, image_latents.dtype)
        )
        return (image_latents - latents_mean) / latents_std

    def _prepare_layered_latents(
        self,
        image,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        layers: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[torch.Generator],
        latents: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        from diffusers.utils.torch_utils import randn_tensor

        h = 2 * (int(height) // (self.vae_scale_factor * 2))
        w = 2 * (int(width) // (self.vae_scale_factor * 2))
        latent_shape = (batch_size, layers + 1, num_channels_latents, h, w)

        image = image.to(device=device, dtype=dtype)
        if image.shape[1] != self.latent_channels:
            image_latents = self._encode_vae_image(image=image, generator=generator)
        else:
            image_latents = image
        self._validate_single_conditioning_frame(image_latents)
        image_latents = self._repeat_conditioning_batch(image_latents, batch_size, "image")

        image_latent_height, image_latent_width = image_latents.shape[3:]
        image_latents = image_latents.permute(0, 2, 1, 3, 4)
        image_latents = self._pack_layered_latents(
            image_latents,
            batch_size,
            num_channels_latents,
            image_latent_height,
            image_latent_width,
            1,
        )

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"Received {len(generator)} generators for effective batch size {batch_size}."
            )
        if latents is None:
            latents = randn_tensor(latent_shape, generator=generator, device=device, dtype=dtype)
            latents = self._pack_layered_latents(
                latents,
                batch_size,
                num_channels_latents,
                h,
                w,
                layers + 1,
            )
        else:
            latents = latents.to(device=device, dtype=dtype)
        return latents, image_latents

    def _decode_layered_latents(
        self,
        latents: torch.Tensor,
        height: int,
        width: int,
        layers: int,
    ) -> torch.Tensor:
        latents = self._unpack_layered_latents(
            latents,
            height,
            width,
            layers,
            self.vae_scale_factor,
        )
        latents = latents.to(self.vae.dtype)

        z_dim = self.vae.config.z_dim
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents = latents / latents_std + latents_mean

        batch_size, channels, frames, latent_h, latent_w = latents.shape
        latents = latents[:, :, 1:]
        latents = latents.permute(0, 2, 1, 3, 4).reshape(-1, channels, 1, latent_h, latent_w)
        image = self.vae.decode(latents, return_dict=False)[0].squeeze(2)
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.permute(0, 2, 3, 1)
        image = (image * 255).round().to(torch.uint8)
        return image.reshape(batch_size, frames - 1, image.shape[1], image.shape[2], image.shape[3])

    def get_image_caption(
        self,
        prompt_image,
        use_en_prompt: bool = False,
        device=None,
    ) -> List[str]:
        if not hasattr(self, "vl_processor"):
            raise ValueError("Automatic image captioning requires Qwen2VLProcessor.")
        prompt = _LAYERED_CAPTION_PROMPT_EN if use_en_prompt else _LAYERED_CAPTION_PROMPT
        image_batch = self._image_batch_size(prompt_image)
        model_inputs = self.vl_processor(
            text=[prompt] * image_batch,
            images=prompt_image,
            padding=True,
            return_tensors="pt",
        ).to(device)
        generated_ids = self.text_encoder.generate(**model_inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids, strict=True)
        ]
        captions = self.vl_processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if len(captions) != image_batch:
            raise ValueError(
                f"Caption batch size {len(captions)} does not match image batch {image_batch}."
            )
        return [caption.strip() for caption in captions]

    def infer(self, req):
        extra = req.params.extra_params or {}
        num_per = getattr(req.params, "num_images_per_prompt", 1) or 1
        base_prompts = req.prompt if isinstance(req.prompt, list) else [req.prompt]
        prompts = [p for p in base_prompts for _ in range(num_per)]

        negative = req.params.negative_prompt
        if negative is not None:
            negatives = negative if isinstance(negative, list) else [negative]
            if len(negatives) == 1:
                negatives = negatives * len(base_prompts)
            elif len(negatives) != len(base_prompts):
                raise ValueError(
                    "negative_prompt must be a string, a singleton list, "
                    "or a list with the same length as prompt"
                )
            negative = [n for n in negatives for _ in range(num_per)]

        return self.forward(
            image=req.params.image,
            prompt=prompts,
            negative_prompt=negative,
            height=req.params.height,
            width=req.params.width,
            true_cfg_scale=req.params.guidance_scale,
            layers=extra.get("layers", 4),
            num_inference_steps=req.params.num_inference_steps,
            seed=req.params.seed,
            max_sequence_length=req.params.max_sequence_length,
            resolution=extra.get("resolution", 640),
            cfg_normalize=extra.get("cfg_normalize", False),
            use_en_prompt=extra.get("use_en_prompt", False),
        )

    @torch.inference_mode()
    def forward(
        self,
        image,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        true_cfg_scale: float = 4.0,
        layers: int = 4,
        num_inference_steps: int = 50,
        seed: int = 42,
        max_sequence_length: int = 512,
        resolution: int = 640,
        cfg_normalize: bool = False,
        use_en_prompt: bool = False,
        sigmas: Optional[list] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> PipelineOutput:
        if image is None:
            raise ValueError("QwenImageLayeredPipeline requires an input image.")
        if resolution not in (640, 1024):
            raise ValueError(f"resolution must be 640 or 1024, got {resolution}")
        if layers < 1:
            raise ValueError(f"layers must be >= 1, got {layers}")
        if (height is None) != (width is None):
            raise ValueError("height and width must be set together for QwenImageLayeredPipeline.")

        pipeline_start = time.time()
        timer = CudaPhaseTimer()
        timer.mark_pre_start()
        image = self._load_image_input(image)

        device = self.device
        generator = torch.Generator(device=device).manual_seed(seed)
        is_latent_image = self._is_layered_latent_image(image)
        if (
            isinstance(image, torch.Tensor)
            and image.ndim >= 2
            and image.shape[1] == self.latent_channels
            and not is_latent_image
        ):
            raise ValueError(
                "Layered latent image inputs must have shape (B, C, F, H, W), "
                f"got {tuple(image.shape)}."
            )
        multiple_of = self.vae_scale_factor * 2

        if is_latent_image:
            self._validate_single_conditioning_frame(image)
            if image.shape[-2] % 2 != 0 or image.shape[-1] % 2 != 0:
                raise ValueError(
                    "Layered latent image spatial dimensions must be even for 2x2 packing, "
                    f"got H={image.shape[-2]}, W={image.shape[-1]}."
                )
            image = image.to(dtype=self.dtype, device=device)
            calculated_width = int(image.shape[-1]) * self.vae_scale_factor
            calculated_height = int(image.shape[-2]) * self.vae_scale_factor
            prompt_image = None
        else:
            if height is None or width is None:
                image_width, image_height = self._image_size(image)
                calculated_width, calculated_height = _calculate_dimensions(
                    resolution * resolution,
                    image_width / image_height,
                )
            else:
                calculated_width = width
                calculated_height = height
            if not hasattr(self, "image_processor"):
                from diffusers.image_processor import VaeImageProcessor

                self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)
            image = self.image_processor.resize(image, calculated_height, calculated_width)
            prompt_image = image
            image = self.image_processor.preprocess(image, calculated_height, calculated_width)
            image = image.unsqueeze(2).to(dtype=self.dtype, device=device)

        width = calculated_width // multiple_of * multiple_of
        height = calculated_height // multiple_of * multiple_of

        image_batch = int(image.shape[0])
        prompt = [prompt] if isinstance(prompt, str) else list(prompt)
        prompt = self._align_prompts_to_image_batch(prompt, image_batch)
        needs_caption = any(self._is_empty_prompt(item) for item in prompt)
        if needs_caption:
            if prompt_image is None:
                raise ValueError(
                    "Automatic image captioning requires a non-latent image input; provide prompt "
                    "when image is already a latent tensor."
                )
            captions = self.get_image_caption(
                prompt_image,
                use_en_prompt=use_en_prompt,
                device=device,
            )
            captions = self._expand_values_to_batch(captions, len(prompt), "caption")
            prompt = [
                caption if self._is_empty_prompt(item) else item
                for item, caption in zip(prompt, captions, strict=True)
            ]
        batch_size = len(prompt)

        has_neg = negative_prompt is not None
        do_true_cfg = true_cfg_scale > 1.0 and has_neg
        logger.info("Encoding layered prompt...")
        prompt_embeds, prompt_embeds_mask = self._encode_prompt(prompt, device, max_sequence_length)
        neg_prompt_embeds = neg_prompt_embeds_mask = None
        if do_true_cfg:
            if isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt] * batch_size
            elif len(negative_prompt) == 1:
                negative_prompt = negative_prompt * batch_size
            elif len(negative_prompt) != batch_size:
                raise ValueError(
                    "negative_prompt must be a string, a singleton list, "
                    "or a list with the same effective batch size as prompt"
                )
            neg_prompt_embeds, neg_prompt_embeds_mask = self._encode_prompt(
                negative_prompt,
                device,
                max_sequence_length,
            )

        num_channels_latents = self.transformer.in_channels // 4
        latents, image_latents = self._prepare_layered_latents(
            image,
            batch_size,
            num_channels_latents,
            height,
            width,
            layers,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        img_shapes = [
            [
                *[
                    (
                        1,
                        height // self.vae_scale_factor // 2,
                        width // self.vae_scale_factor // 2,
                    )
                    for _ in range(layers + 1)
                ],
                (
                    1,
                    calculated_height // self.vae_scale_factor // 2,
                    calculated_width // self.vae_scale_factor // 2,
                ),
            ]
        ] * batch_size

        sigmas_np = (
            sigmas if sigmas is not None else np.linspace(1.0, 0, num_inference_steps + 1)[:-1]
        )
        base_seqlen = 256 * 256 / 16 / 16
        mu = (image_latents.shape[1] / base_seqlen) ** 0.5
        self.scheduler.set_timesteps(sigmas=sigmas_np, device=device, mu=mu)
        timesteps = self.scheduler.timesteps
        self.scheduler.set_begin_index(0)

        additional_t_cond = torch.zeros(batch_size, device=device, dtype=torch.long)
        timer.mark_denoise_start()
        logger.info("Denoising layered output (%d steps)...", len(timesteps))
        for t in timesteps:
            latent_model_input = torch.cat([latents, image_latents], dim=1)
            timestep = t.expand(latents.shape[0]).to(latents.dtype)
            noise_pred = self.transformer(
                hidden_states=latent_model_input,
                timestep=timestep / 1000,
                encoder_hidden_states_mask=prompt_embeds_mask,
                encoder_hidden_states=prompt_embeds,
                img_shapes=img_shapes,
                additional_t_cond=additional_t_cond,
                return_dict=False,
            )[0]
            noise_pred = noise_pred[:, : latents.size(1)]

            if do_true_cfg:
                neg_noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep / 1000,
                    encoder_hidden_states_mask=neg_prompt_embeds_mask,
                    encoder_hidden_states=neg_prompt_embeds,
                    img_shapes=img_shapes,
                    additional_t_cond=additional_t_cond,
                    return_dict=False,
                )[0]
                neg_noise_pred = neg_noise_pred[:, : latents.size(1)]
                comb = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)
                if cfg_normalize:
                    cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
                    noise_norm = torch.norm(comb, dim=-1, keepdim=True)
                    noise_pred = comb * (cond_norm / noise_norm)
                else:
                    noise_pred = comb

            latents_dtype = latents.dtype
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            if latents.dtype != latents_dtype:
                latents = latents.to(latents_dtype)

        timer.mark_post_start()
        logger.info("Decoding layered output...")
        layer_stack = self._decode_layered_latents(latents, height, width, layers)
        if getattr(self, "rank", 0) == 0:
            logger.info("Layered pipeline total: %.2fs", time.time() - pipeline_start)

        timer.mark_end()
        image_grid = self._layer_stack_to_image_grid(layer_stack)
        return timer.fill(PipelineOutput(image=image_grid))
