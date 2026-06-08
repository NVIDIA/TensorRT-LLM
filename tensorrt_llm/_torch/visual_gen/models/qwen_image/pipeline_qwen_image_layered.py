# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen-Image-Layered image decomposition pipeline."""

import io
import math
import time
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from tensorrt_llm._torch.visual_gen.output import CudaPhaseTimer, PipelineOutput
from tensorrt_llm._torch.visual_gen.pipeline import ExtraParamSchema
from tensorrt_llm._torch.visual_gen.pipeline_registry import PipelineComponent, register_pipeline
from tensorrt_llm.logger import logger

from .pipeline_qwen_image import QwenImagePipeline
from .transformer_qwen_image import QwenImageTransformer2DModel

_LAYERED_DEFAULT_GENERATION_PARAMS = {
    "height": None,
    "width": None,
    "num_inference_steps": 50,
    "guidance_scale": 4.0,
    "max_sequence_length": 512,
}

_LAYERED_CAPTION_PROMPT = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\nDescribe the input image faithfully and in detail. "
    "Mention visible objects, attributes, spatial relationships, background, "
    "lighting, texture, and any readable text."
    "<|vision_start|><|image_pad|><|vision_end|><|im_end|>\n"
    "<|im_start|>assistant\n"
)

_LAYERED_CAPTION_PROMPT_EN = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\nWrite a detailed English caption for the input image. "
    "Use natural language, keep the description grounded in visible content, "
    "and do not add details that are not shown."
    "<|vision_start|><|image_pad|><|vision_end|><|im_end|>\n"
    "<|im_start|>assistant\n"
)


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


@register_pipeline("QwenImageLayeredPipeline")
class QwenImageLayeredPipeline(QwenImagePipeline):
    """Qwen-Image-Layered image decomposition pipeline."""

    DEFAULT_GENERATION_PARAMS = _LAYERED_DEFAULT_GENERATION_PARAMS

    def __init__(self, model_config):
        super().__init__(model_config)
        self.latent_channels = 16

    def _init_transformer(self) -> None:
        logger.info("Creating Qwen-Image-Layered transformer")
        pretrained = getattr(self.model_config, "pretrained_config", None)

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

        self.transformer = QwenImageTransformer2DModel(
            model_config=self.model_config,
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
    def default_generation_params(self) -> dict:
        return dict(_LAYERED_DEFAULT_GENERATION_PARAMS)

    def warmup_cache_key(self, height: Optional[int], width: Optional[int], **kwargs) -> tuple:
        if height is None or width is None:
            height, width = self.default_warmup_resolutions[0]
        return super().warmup_cache_key(height, width, **kwargs)

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
        super().load_standard_components(checkpoint_dir, device, skip_components)

        skip_components = skip_components or []
        if PipelineComponent.IMAGE_PROCESSOR not in skip_components:
            try:
                from diffusers.image_processor import VaeImageProcessor
            except ImportError as e:  # pragma: no cover
                raise ImportError(
                    "Qwen-Image-Layered requires diffusers with VaeImageProcessor."
                ) from e
            self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)

        if PipelineComponent.PROCESSOR not in skip_components:
            try:
                from transformers import Qwen2VLProcessor
            except ImportError as e:  # pragma: no cover
                raise ImportError(
                    "Qwen-Image-Layered requires transformers with Qwen2VLProcessor."
                ) from e
            logger.info("Loading Qwen2-VL processor...")
            self.vl_processor = Qwen2VLProcessor.from_pretrained(
                checkpoint_dir, subfolder=PipelineComponent.PROCESSOR
            )
            self.processor = self.vl_processor

        self.default_height = 640
        self.default_width = 640
        self.max_sequence_length = 512
        if getattr(self, "vae", None) is not None:
            self.latent_channels = self.vae.config.z_dim

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
        if batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] == 0:
            image_latents = torch.cat(
                [image_latents] * (batch_size // image_latents.shape[0]),
                dim=0,
            )
        elif batch_size > image_latents.shape[0]:
            raise ValueError(
                f"Cannot duplicate image batch {image_latents.shape[0]} to {batch_size} prompts."
            )
        elif batch_size != image_latents.shape[0]:
            raise ValueError(
                f"Image batch size {image_latents.shape[0]} must match prompt batch size "
                f"{batch_size}, or divide it exactly."
            )

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

    def get_image_caption(self, prompt_image, use_en_prompt: bool = False, device=None) -> str:
        if not hasattr(self, "vl_processor"):
            raise ValueError("Automatic image captioning requires Qwen2VLProcessor.")
        prompt = _LAYERED_CAPTION_PROMPT_EN if use_en_prompt else _LAYERED_CAPTION_PROMPT
        model_inputs = self.vl_processor(
            text=prompt,
            images=prompt_image,
            padding=True,
            return_tensors="pt",
        ).to(device)
        generated_ids = self.text_encoder.generate(**model_inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids, strict=True)
        ]
        return self.vl_processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

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

        if isinstance(prompt, list):
            needs_caption = any(self._is_empty_prompt(item) for item in prompt)
        else:
            needs_caption = self._is_empty_prompt(prompt)
        if needs_caption:
            if prompt_image is None:
                raise ValueError(
                    "Automatic image captioning requires a non-latent image input; provide prompt "
                    "when image is already a latent tensor."
                )
            caption = self.get_image_caption(
                prompt_image,
                use_en_prompt=use_en_prompt,
                device=device,
            )
            if isinstance(prompt, list):
                prompt = [caption if self._is_empty_prompt(item) else item for item in prompt]
            else:
                prompt = caption
        if isinstance(prompt, str):
            prompt = [prompt]
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
        # VisualGenOutput has no layer-list field. The layer stack is exposed
        # through the video slot as (B, layers, H, W, C).
        return timer.fill(PipelineOutput(video=layer_stack, frame_rate=1.0))
