# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen-Image-Edit pipeline.

The edit checkpoint uses the existing Qwen-Image transformer architecture
with extra image-conditioning paths from Diffusers
``QwenImageEditPlusPipeline``.
"""

import math
import time
from io import BytesIO
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist

from tensorrt_llm._torch.visual_gen.output import CudaPhaseTimer, PipelineOutput
from tensorrt_llm._torch.visual_gen.pipeline_registry import register_pipeline
from tensorrt_llm.inputs.utils import load_image
from tensorrt_llm.logger import logger

from .pipeline_qwen_image import QwenImagePipeline, _calculate_shift

_EDIT_PROMPT_TEMPLATE = (
    "<|im_start|>system\nDescribe the key features of the input image "
    "(color, shape, size, texture, objects, background), then explain how "
    "the user's text instruction should alter or modify the image. Generate "
    "a new image that meets the user's requirements while maintaining "
    "consistency with the original input where appropriate.<|im_end|>\n"
    "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
)
_EDIT_PROMPT_TEMPLATE_START_IDX = 64
_CONDITION_IMAGE_SIZE = 384 * 384
_VAE_IMAGE_SIZE = 1024 * 1024


def _calculate_dimensions(target_area: int, ratio: float) -> Tuple[int, int]:
    """Match Diffusers Qwen-Image-Edit area-preserving 32px rounding."""
    width = math.sqrt(target_area * ratio)
    height = width / ratio
    width = round(width / 32) * 32
    height = round(height / 32) * 32
    return width, height


def _retrieve_latents(
    encoder_output: torch.Tensor,
    generator: Optional[torch.Generator] = None,
    sample_mode: str = "sample",
) -> torch.Tensor:
    """Return latent tensor from a Diffusers VAE encoder output."""
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    if hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    if hasattr(encoder_output, "latents"):
        return encoder_output.latents
    raise AttributeError("Could not access latents of provided encoder_output")


_EDIT_DEFAULT_GENERATION_PARAMS = {
    "height": None,
    "width": None,
    "num_inference_steps": 50,
    "guidance_scale": 4.0,
    "max_sequence_length": 512,
    "negative_prompt": " ",
}


@register_pipeline(
    "QwenImageEditPlusPipeline",
    hf_ids=[
        "Qwen/Qwen-Image-Edit-2509",
        "Qwen/Qwen-Image-Edit-2511",
    ],
    doc="Qwen-Image-Edit image editing pipeline.",
)
class QwenImageEditPlusPipeline(QwenImagePipeline):
    """Qwen-Image-Edit pipeline using the existing Qwen-Image transformer.

    Diffusers implements this model by feeding the input image through two
    paths: Qwen2-VL sees resized condition images during text encoding, while
    the Qwen-Image VAE encodes appearance latents that are appended after the
    generated latent tokens. The transformer predicts every token in the
    concatenated sequence, and the scheduler only steps the generated prefix.
    """

    DEFAULT_GENERATION_PARAMS = _EDIT_DEFAULT_GENERATION_PARAMS

    def load_standard_components(
        self,
        checkpoint_dir: str,
        device: torch.device,
        skip_components: Optional[list] = None,
    ) -> None:
        super().load_standard_components(checkpoint_dir, device, skip_components)
        skip_components = skip_components or []

        if "processor" not in skip_components:
            try:
                from diffusers.image_processor import VaeImageProcessor
                from transformers import Qwen2VLProcessor
            except ImportError as e:  # pragma: no cover
                raise ImportError(
                    "Qwen-Image-Edit requires diffusers.VaeImageProcessor "
                    "and transformers.Qwen2VLProcessor."
                ) from e

            logger.info("Loading Qwen2-VL processor...")
            try:
                self.processor = Qwen2VLProcessor.from_pretrained(
                    checkpoint_dir,
                    subfolder="processor",
                )
            except (OSError, ValueError):
                self.processor = Qwen2VLProcessor.from_pretrained(checkpoint_dir)
            self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)

        self.latent_channels = self.vae.config.z_dim
        self.default_height = None
        self.default_width = None
        self.max_sequence_length = 512

    @property
    def default_generation_params(self) -> dict:
        return dict(_EDIT_DEFAULT_GENERATION_PARAMS)

    @property
    def default_warmup_resolutions(self) -> List[Tuple[int, int]]:
        return [(1024, 1024)]

    def warmup_cache_key(self, height: Optional[int], width: Optional[int], **kwargs) -> tuple:
        return (height, width)

    def torch_compile(self) -> None:
        vgm = self.pipeline_config.visual_gen_mapping
        if vgm and vgm.cfg_size > 1:
            logger.warning(
                "Qwen-Image-Edit disables torch.compile when CFG parallelism is enabled; "
                "compiled per-rank conditional/unconditional transformer execution does not "
                "currently preserve CFG=1 parity."
            )
            return
        super().torch_compile()

    def _run_warmup(self, height: int, width: int, num_frames: int, steps: int) -> None:
        from PIL import Image

        with torch.no_grad():
            self.forward(
                image=Image.new("RGB", (width, height), color=(127, 127, 127)),
                prompt="warmup",
                height=height,
                width=width,
                num_inference_steps=max(steps, 2),
                true_cfg_scale=1.0,
                seed=42,
                max_sequence_length=64,
            )

    @staticmethod
    def _load_edit_images(image: Union[str, bytes, Any, List[Union[str, bytes, Any]]]) -> List[Any]:
        if image is None:
            raise ValueError("Qwen-Image-Edit requires params.image.")
        images = image if isinstance(image, list) else [image]
        pil_images = []
        for item in images:
            if isinstance(item, bytes):
                from PIL import Image

                pil_images.append(Image.open(BytesIO(item)).convert("RGB"))
            else:
                pil_images.append(load_image(item, format="pil"))
        return pil_images

    def _preprocess_edit_images(
        self,
        pil_images: List[Any],
    ) -> Tuple[List[Any], List[torch.Tensor], List[Tuple[int, int]]]:
        condition_images = []
        vae_images = []
        vae_image_sizes = []
        for img in pil_images:
            image_width, image_height = img.size
            ratio = image_width / image_height
            condition_width, condition_height = _calculate_dimensions(_CONDITION_IMAGE_SIZE, ratio)
            vae_width, vae_height = _calculate_dimensions(_VAE_IMAGE_SIZE, ratio)
            condition_images.append(
                self.image_processor.resize(img, condition_height, condition_width)
            )
            vae_images.append(
                self.image_processor.preprocess(img, vae_height, vae_width).unsqueeze(2)
            )
            vae_image_sizes.append((vae_width, vae_height))
        return condition_images, vae_images, vae_image_sizes

    def _get_qwen_edit_prompt_embeds(
        self,
        prompt: List[str],
        image: Optional[List[Any]],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        img_prompt_template = "Picture {}: <|vision_start|><|image_pad|><|vision_end|>"
        if isinstance(image, list):
            base_img_prompt = "".join(
                img_prompt_template.format(i + 1) for i, _ in enumerate(image)
            )
        elif image is not None:
            base_img_prompt = img_prompt_template.format(1)
        else:
            base_img_prompt = ""

        txt = [_EDIT_PROMPT_TEMPLATE.format(base_img_prompt + p) for p in prompt]
        model_inputs = self.processor(
            text=txt,
            images=image,
            padding=True,
            return_tensors="pt",
        ).to(device)

        encoder_outputs = self.text_encoder(
            input_ids=model_inputs["input_ids"],
            attention_mask=model_inputs["attention_mask"],
            pixel_values=model_inputs.get("pixel_values"),
            image_grid_thw=model_inputs.get("image_grid_thw"),
            output_hidden_states=True,
        )
        hidden_states = encoder_outputs.hidden_states[-1]

        split_hidden = self._extract_masked_hidden(hidden_states, model_inputs["attention_mask"])
        split_hidden = [h[_EDIT_PROMPT_TEMPLATE_START_IDX:] for h in split_hidden]
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
        return prompt_embeds.to(dtype=dtype, device=device), prompt_embeds_mask

    def _encode_edit_prompt(
        self,
        prompt: List[str],
        image: List[Any],
        device: torch.device,
        max_sequence_length: int,
        num_images_per_prompt: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = len(prompt)
        prompt_embeds, prompt_embeds_mask = self._get_qwen_edit_prompt_embeds(
            prompt,
            image,
            device,
            self.dtype,
        )
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
        prompt_embeds_mask = prompt_embeds_mask.repeat(1, num_images_per_prompt, 1)
        prompt_embeds_mask = prompt_embeds_mask.view(batch_size * num_images_per_prompt, seq_len)

        prompt_embeds = prompt_embeds[:, :max_sequence_length]
        prompt_embeds_mask = prompt_embeds_mask[:, :max_sequence_length]
        if bool(prompt_embeds_mask.all()):
            prompt_embeds_mask = None
        return prompt_embeds, prompt_embeds_mask

    def _encode_vae_image(
        self,
        image: torch.Tensor,
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
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

    def _prepare_edit_latents(
        self,
        images: List[torch.Tensor],
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[torch.Generator],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        latents = self._prepare_latents(
            batch_size,
            num_channels_latents,
            height,
            width,
            dtype,
            device,
            generator,
        )

        all_image_latents = []
        for image in images:
            image = image.to(device=device, dtype=dtype)
            if image.shape[1] != self.latent_channels:
                image_latents = self._encode_vae_image(image=image, generator=generator)
            else:
                image_latents = image

            if batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] == 0:
                repeat = batch_size // image_latents.shape[0]
                image_latents = torch.cat([image_latents] * repeat, dim=0)
            elif batch_size > image_latents.shape[0]:
                raise ValueError(
                    "Cannot duplicate image batch size "
                    f"{image_latents.shape[0]} to {batch_size} prompts."
                )
            else:
                image_latents = torch.cat([image_latents], dim=0)

            image_latent_height, image_latent_width = image_latents.shape[3:]
            image_latents = self._pack_latents(
                image_latents,
                batch_size,
                num_channels_latents,
                image_latent_height,
                image_latent_width,
            )
            all_image_latents.append(image_latents)
        return latents, torch.cat(all_image_latents, dim=1)

    def infer(self, req):
        params = req.params
        prompts = req.prompt if isinstance(req.prompt, list) else [req.prompt]
        num_per = params.num_images_per_prompt or 1
        prompts = [p for p in prompts for _ in range(num_per)]
        pil_images = self._load_edit_images(params.image)
        height = params.height
        width = params.width
        if height is None or width is None:
            source_width, source_height = pil_images[-1].size
            inferred_width, inferred_height = _calculate_dimensions(
                _VAE_IMAGE_SIZE,
                source_width / source_height,
            )
            height = height or inferred_height
            width = width or inferred_width

        return self.forward(
            image=pil_images,
            prompt=prompts,
            negative_prompt=params.negative_prompt,
            height=height,
            width=width,
            num_inference_steps=params.num_inference_steps,
            true_cfg_scale=params.guidance_scale,
            seed=params.seed,
            max_sequence_length=params.max_sequence_length,
        )

    @torch.inference_mode()
    def forward(
        self,
        image: Union[str, bytes, Any, List[Union[str, bytes, Any]]],
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = " ",
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        true_cfg_scale: float = 4.0,
        seed: int = 42,
        max_sequence_length: int = 512,
        sigmas: Optional[list] = None,
    ) -> PipelineOutput:
        """Image editing generation matching Diffusers QwenImageEditPlusPipeline."""
        pipeline_start = time.time()
        timer = CudaPhaseTimer()
        timer.mark_pre_start()

        pil_images = self._load_edit_images(image)
        source_size = pil_images[-1].size
        calculated_width, calculated_height = _calculate_dimensions(
            _VAE_IMAGE_SIZE,
            source_size[0] / source_size[1],
        )
        height = height or calculated_height
        width = width or calculated_width
        multiple_of = self.vae_scale_factor * 2
        width = width // multiple_of * multiple_of
        height = height // multiple_of * multiple_of

        if isinstance(prompt, str):
            prompt = [prompt]
        batch_size = len(prompt)
        if batch_size > 1:
            raise ValueError(
                "QwenImageEditPlusPipeline currently supports one prompt at a time."
            )

        condition_images, vae_images, vae_image_sizes = self._preprocess_edit_images(pil_images)
        device = self.device
        generator = torch.Generator(device=device).manual_seed(seed)

        has_neg = negative_prompt is not None
        do_true_cfg = true_cfg_scale > 1.0 and has_neg
        vgm = self.pipeline_config.visual_gen_mapping
        cfg_size = vgm.cfg_size if vgm else 1
        cfg_rank = vgm.cfg_rank if vgm else 0
        cfg_pg = vgm.cfg_group if vgm else None
        do_cfg_parallel = do_true_cfg and cfg_size > 1
        if do_cfg_parallel:
            if cfg_size != 2:
                raise ValueError(
                    f"Qwen-Image-Edit CFG parallel requires cfg_size=2, got {cfg_size}"
                )
            if not dist.is_available() or not dist.is_initialized():
                raise RuntimeError(
                    "Qwen-Image-Edit CFG parallel requires initialized torch.distributed"
                )
            if getattr(self, "rank", 0) == 0:
                logger.info("Qwen-Image-Edit CFG parallel enabled: cfg_size=2")

        logger.info("Encoding edit prompt...")
        prompt_embeds, prompt_embeds_mask = self._encode_edit_prompt(
            prompt,
            condition_images,
            device,
            max_sequence_length,
            num_images_per_prompt=1,
        )
        neg_prompt_embeds = neg_prompt_embeds_mask = None
        if do_true_cfg:
            if isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt] * batch_size
            neg_prompt_embeds, neg_prompt_embeds_mask = self._encode_edit_prompt(
                negative_prompt,
                condition_images,
                device,
                max_sequence_length,
                num_images_per_prompt=1,
            )

        num_channels_latents = self.transformer.in_channels // 4
        latents, image_latents = self._prepare_edit_latents(
            vae_images,
            batch_size,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
        )
        img_shapes = [
            [
                (1, height // self.vae_scale_factor // 2, width // self.vae_scale_factor // 2),
                *[
                    (
                        1,
                        vae_height // self.vae_scale_factor // 2,
                        vae_width // self.vae_scale_factor // 2,
                    )
                    for vae_width, vae_height in vae_image_sizes
                ],
            ]
        ] * batch_size

        sigmas_np = (
            sigmas
            if sigmas is not None
            else np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        )
        image_seq_len = latents.shape[1]
        mu = _calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        self.scheduler.set_timesteps(sigmas=sigmas_np, device=device, mu=mu)
        timesteps = self.scheduler.timesteps
        self.scheduler.set_begin_index(0)

        timer.mark_denoise_start()
        logger.info("Denoising edit (%d steps)...", len(timesteps))
        for t in timesteps:
            latent_model_input = torch.cat([latents, image_latents], dim=1)
            timestep = t.expand(latents.shape[0]).to(latents.dtype)

            if do_cfg_parallel:
                if cfg_rank == 0:
                    local_noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep / 1000,
                        encoder_hidden_states_mask=prompt_embeds_mask,
                        encoder_hidden_states=prompt_embeds,
                        img_shapes=img_shapes,
                        return_dict=False,
                    )[0]
                else:
                    local_noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep / 1000,
                        encoder_hidden_states_mask=neg_prompt_embeds_mask,
                        encoder_hidden_states=neg_prompt_embeds,
                        img_shapes=img_shapes,
                        return_dict=False,
                    )[0]
                local_noise_pred = local_noise_pred[:, : latents.size(1)].contiguous()
                gathered_noise_pred = [torch.empty_like(local_noise_pred) for _ in range(cfg_size)]
                dist.all_gather(gathered_noise_pred, local_noise_pred, group=cfg_pg)
                noise_pred = gathered_noise_pred[0]
                neg_noise_pred = gathered_noise_pred[1]
                comb = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)
                cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
                noise_norm = torch.norm(comb, dim=-1, keepdim=True)
                noise_pred = comb * (cond_norm / noise_norm)
            else:
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep / 1000,
                    encoder_hidden_states_mask=prompt_embeds_mask,
                    encoder_hidden_states=prompt_embeds,
                    img_shapes=img_shapes,
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
                        return_dict=False,
                    )[0]
                    neg_noise_pred = neg_noise_pred[:, : latents.size(1)]
                    comb = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)
                    cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
                    noise_norm = torch.norm(comb, dim=-1, keepdim=True)
                    noise_pred = comb * (cond_norm / noise_norm)

            latents_dtype = latents.dtype
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            if latents.dtype != latents_dtype:
                latents = latents.to(latents_dtype)

        timer.mark_post_start()
        logger.info("Decoding edit...")
        output_image = self._decode_latents(latents, height, width)

        if getattr(self, "rank", 0) == 0:
            logger.info("Edit pipeline total: %.2fs", time.time() - pipeline_start)

        timer.mark_end()
        return timer.fill(PipelineOutput(image=output_image))
