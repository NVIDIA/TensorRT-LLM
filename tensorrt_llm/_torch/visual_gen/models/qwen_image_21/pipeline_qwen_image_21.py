# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the TensorRT-LLM project
"""Qwen-Image 2.1 TRTLLM VisualGen pipeline.

The runtime entrypoint is TRTLLM-owned and does not import the upstream
Diffusers pipeline, transformer, UNet/DiT or attention components.  The image
VAE may use the exact Diffusers ``AutoencoderKLQwenImage21`` as a declared
fallback until a native TRTLLM VAE port is available; the fallback is scoped to
VAE encode/decode only and is recorded in ``design/external_component_reuse``.
"""

from __future__ import annotations

import time
from io import BytesIO
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch

from tensorrt_llm._torch.visual_gen.output import CudaPhaseTimer, PipelineOutput
from tensorrt_llm._torch.visual_gen.pipeline import (
    ExtraParamSchema,
    RefSlotSpec,
    RoleSpec,
)
from tensorrt_llm._torch.visual_gen.pipeline_registry import PipelineComponent, register_pipeline
from tensorrt_llm._torch.visual_gen.utils import make_noise_generator
from tensorrt_llm.logger import logger

from ..qwen_image.pipeline_qwen_image import QwenImagePipeline
from .ops import (
    QwenImage21FlowMatchEulerScheduler,
    append_target_slots,
    calculate_dimensions,
    calculate_shift,
    pack_latents,
    unpack_latents,
)
from .transformer_qwen_image_21 import QwenImage21KVCache, QwenImage21Transformer2DModel

_QWEN_IMAGE_21_DEFAULT_GENERATION_PARAMS = {
    "height": 1024,
    "width": 1024,
    "num_inference_steps": 40,
    "guidance_scale": 1.0,
    "max_sequence_length": 4096,
}


def _resolve_qwen_image_21_vae_fallback():
    """Resolve the declared Diffusers Qwen-Image-2.1 image-VAE fallback.

    Qwen-Image 2.1 checkpoints declare ``AutoencoderKLQwenImage21`` in
    ``vae/config.json`` and store 64-channel 2.1-specific residual VAE weights.
    Older Diffusers releases expose ``AutoencoderKLQwenImage`` for Qwen-Image
    1.x, but that class has a different default architecture (for example a
    16-channel latent VAE) and cannot load the 2.1 checkpoint safely.  Do not
    silently fall back to that alias: using the wrong VAE architecture fails at
    load time or produces invalid decoded media.  The run manifest declares only
    the exact 2.1 image-VAE fallback, so runtime must require that symbol.
    """

    try:
        from diffusers import AutoencoderKLQwenImage21
    except ImportError as exc:  # pragma: no cover - depends on installed Diffusers version.
        raise ImportError(
            "Qwen-Image 2.1 VAE fallback requires a Diffusers build exposing "
            "AutoencoderKLQwenImage21. The older AutoencoderKLQwenImage alias "
            "is not architecture-compatible with Qwen-Image-2.1 checkpoints."
        ) from exc
    return AutoencoderKLQwenImage21, "diffusers.AutoencoderKLQwenImage21"


@register_pipeline(
    "QwenImage21Pipeline",
    hf_ids=["Qwen/Qwen-Image-2.1"],
    defaults={},
    download_patterns=[
        "model_index.json",
        "processor/*",
        "scheduler/*",
        "text_encoder/*",
        "transformer/*",
        "vae/*",
    ],
    doc="Qwen-Image 2.1 text-to-image / image-conditioned generation.",
)
class QwenImage21Pipeline(QwenImagePipeline):
    """TRTLLM-owned Qwen-Image 2.1 VisualGen pipeline entrypoint."""

    DEFAULT_GENERATION_PARAMS = _QWEN_IMAGE_21_DEFAULT_GENERATION_PARAMS
    transformer_class = QwenImage21Transformer2DModel
    kv_cache_class = QwenImage21KVCache
    scheduler_class = QwenImage21FlowMatchEulerScheduler
    default_num_inference_steps = 40
    vae_scale_factor = 16
    latent_channels = 64
    supports_image_edit = True
    # When an image-edit request omits size/width/height, the HTTP adapter sets
    # height/width to None and this flag prevents executor default merging from
    # overwriting the request-dependent dimensions derived from the reference.
    derive_output_size_from_reference = True
    sys_prompt = "Comprehend and analyze the provided prompt."
    prompt_template_t2i = (
        f"<|im_start|>system\n{sys_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{{}}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    prompt_template_ti2i = (
        f"<|im_start|>system\n{sys_prompt}<|im_end|>\n"
        f"<|im_start|>user\n<image1><|vision_start|><|image_pad|><|vision_end|>{{}}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    def __init__(self, pipeline_config):
        super().__init__(pipeline_config)
        self.vae_scale_factor = 16
        self.latent_channels = 64
        self.tokenizer_max_length = 4096
        self.default_height = 1024
        self.default_width = 1024
        self.max_sequence_length = 4096
        self._drop_idx = None
        self._img_token_id = None

    @property
    def default_warmup_resolutions(self) -> List[Tuple[int, int]]:
        return [(1024, 1024)]

    @property
    def default_warmup_num_frames(self) -> List[int]:
        return [1]

    @property
    def resolution_multiple_of(self) -> Tuple[int, int]:
        return (self.vae_scale_factor * 2, self.vae_scale_factor * 2)

    @property
    def default_generation_params(self) -> dict:
        return dict(_QWEN_IMAGE_21_DEFAULT_GENERATION_PARAMS)

    @property
    def extra_param_specs(self) -> dict[str, ExtraParamSchema]:
        """Model-specific request knobs accepted by Qwen-Image 2.1.

        ``VisualGen`` validates ``extra_params`` before enqueueing a request.
        The two knobs below intentionally live in ``extra_params`` because they
        are Qwen-Image-2.1-specific and are not universal ``VisualGenParams``
        fields.  Keeping them out of ``default_generation_params`` also makes
        ``VisualGen.default_params`` constructible through the public API.
        """

        return {
            "output_resolution": ExtraParamSchema(
                type="int",
                default=1024,
                range=(32, 4096),
                description=(
                    "Reference-conditioned output-resolution hint used when height/width are "
                    "omitted and an image reference determines the aspect ratio."
                ),
            ),
            "use_kv_cache": ExtraParamSchema(
                type="bool",
                default=True,
                description="Enable the Qwen-Image 2.1 prefix KV cache for static conditioning tokens.",
            ),
        }

    @property
    def ref_slot_specs(self) -> dict[str, RefSlotSpec]:
        """Qwen-Image 2.1 accepts zero or more reference images.

        Text-to-image requests omit the slot.  Image-conditioned generation and
        the OpenAI-compatible ``/v1/images/edits`` route provide references as
        ``image_reference`` with the single unambiguous ``reference`` role.
        """

        return {
            "image_reference": RefSlotSpec(
                modality="image", roles=[RoleSpec(role="reference", min=0, max=None)]
            )
        }

    @staticmethod
    def calculate_shift(*args: Any, **kwargs: Any) -> float:
        return calculate_shift(*args, **kwargs)

    @staticmethod
    def calculate_dimensions(*args: Any, **kwargs: Any) -> tuple[int, int, None]:
        return calculate_dimensions(*args, **kwargs)

    @staticmethod
    def _pack_latents(
        latents: torch.Tensor,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
    ) -> torch.Tensor:
        return pack_latents(latents, batch_size, num_channels_latents, height, width)

    @staticmethod
    def _unpack_latents(latents: torch.Tensor, height: int, width: int, vae_scale_factor: int) -> torch.Tensor:
        return unpack_latents(latents, height, width, vae_scale_factor)

    def _init_transformer(self) -> None:
        logger.info("Creating Qwen-Image 2.1 transformer")
        model_config = self.pipeline_config.model_configs["transformer"]
        pretrained = getattr(model_config, "pretrained_config", None)

        def _cfg(name: str, default):
            if pretrained is None:
                return default
            if isinstance(pretrained, dict):
                return pretrained.get(name, default)
            return getattr(pretrained, name, default)

        self.transformer = QwenImage21Transformer2DModel(
            model_config=model_config,
            patch_size=_cfg("patch_size", 1),
            in_channels=_cfg("in_channels", 64),
            out_channels=_cfg("out_channels", 64),
            num_layers=_cfg("num_layers", 32),
            attention_head_dim=_cfg("attention_head_dim", 128),
            num_attention_heads=_cfg("num_attention_heads", 32),
            context_in_dim=_cfg("context_in_dim", 4096),
            mlp_ratio=_cfg("mlp_ratio", 3),
            axes_dims_rope=tuple(_cfg("axes_dims_rope", (16, 56, 56))),
            eps=_cfg("eps", 1e-6),
            causal_condition=_cfg("causal_condition", True),
        )

    def load_standard_components(
        self,
        checkpoint_dir: str,
        device: torch.device,
        skip_components: Optional[list] = None,
    ) -> None:
        skip_components = skip_components or []
        if PipelineComponent.PROCESSOR not in skip_components:
            from transformers import Qwen3VLProcessor

            logger.info("Loading Qwen3-VL processor for Qwen-Image 2.1...")
            self.processor = Qwen3VLProcessor.from_pretrained(checkpoint_dir, subfolder="processor")
            self.tokenizer = self.processor.tokenizer
            sys_message = [{"role": "system", "content": [{"type": "text", "text": self.sys_prompt}]}]
            sys_tokens = self.processor.apply_chat_template(sys_message, tokenize=True, return_dict=False)
            self._drop_idx = len(sys_tokens[0])
            self._img_token_id = self.processor.tokenizer.encode("<|image_pad|>")[0]

        if PipelineComponent.TEXT_ENCODER not in skip_components:
            from transformers import Qwen3VLForConditionalGeneration

            logger.info("Loading Qwen3-VL text encoder for Qwen-Image 2.1...")
            self.text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
                checkpoint_dir,
                subfolder="text_encoder",
                torch_dtype=self.pipeline_config.torch_dtype,
            ).to(self.device)

        if PipelineComponent.VAE not in skip_components:
            vae_cls, vae_name = _resolve_qwen_image_21_vae_fallback()
            logger.info("Loading declared Diffusers %s image-VAE fallback...", vae_name)
            self.vae = vae_cls.from_pretrained(
                checkpoint_dir,
                subfolder="vae",
                torch_dtype=self.pipeline_config.torch_dtype,
            ).to(self.device)
            self.vae_scale_factor = int(getattr(self.vae.config, "scale_factor_spatial", 16))
            self.latent_channels = int(getattr(self.vae.config, "z_dim", 64))
            if hasattr(self.vae, "enable_tiling"):
                self.vae.enable_tiling()

        if PipelineComponent.SCHEDULER not in skip_components:
            logger.info("Loading native Qwen-Image 2.1 FlowMatch scheduler...")
            self.scheduler = self.scheduler_class.from_pretrained(checkpoint_dir, subfolder="scheduler")

    def load_weights(self, weights: dict) -> None:
        if self.transformer is not None:
            transformer_weights = weights.get("transformer", weights)
            self.transformer.load_weights(transformer_weights)
            self.transformer.to_inference_dtype().eval()
        self._target_dtype = self.pipeline_config.torch_dtype

    def offload_pipeline_components(self) -> dict[str, torch.nn.Module]:
        components = super().offload_pipeline_components()
        if self.transformer is not None:
            components[PipelineComponent.TRANSFORMER.value] = self.transformer
        return components

    def default_offload_stages(self):
        from tensorrt_llm._torch.visual_gen.offloading import OffloadPipelineStage

        return (
            OffloadPipelineStage((PipelineComponent.TEXT_ENCODER.value,)),
            OffloadPipelineStage((PipelineComponent.TRANSFORMER.value,)),
            OffloadPipelineStage((PipelineComponent.VAE.value,)),
        )

    def _run_warmup(self, height: int, width: int, num_frames: int, steps: int) -> None:
        del num_frames
        with torch.no_grad():
            self.forward(
                prompt="warmup",
                height=height,
                width=width,
                num_inference_steps=max(steps, 2),
                seed=42,
                max_sequence_length=64,
                use_kv_cache=False,
            )

    def _qwen_image_21_runtime_config(self) -> dict[str, Any]:
        return {
            "model_name": "qwen-image-2.1",
            "pipeline_class_name": "QwenImage21Pipeline",
            "default_num_inference_steps": self.default_num_inference_steps,
            "vae_scale_factor": self.vae_scale_factor,
            "latent_channels": self.latent_channels,
            "uses_qwen3_vl_conditioning": True,
            "latent_packing": "unpatched_spatial_flatten",
            "target_mask_stride": 4,
            "scheduler": "QwenImage21FlowMatchEulerScheduler",
            "image_vae": "diffusers.AutoencoderKLQwenImage21 fallback",
        }

    def _extract_masked_hidden(self, hidden_states: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        return torch.split(selected, valid_lengths.tolist(), dim=0)

    def _get_qwen_prompt_embeds(
        self,
        prompt: str | list[str] | None = None,
        image: list | None = None,
        device: torch.device | None = None,
    ):
        from PIL import Image as PILImage

        device = device or self.device
        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [" " if not p else p for p in prompt]
        is_t2i = image is None

        if is_t2i:
            prompts = [self.prompt_template_t2i.format(t) for t in prompt]
        else:
            prompts = []
            condition_pil_list = []
            for t in prompt:
                n_imgs = len(image)
                replace = "<image1><|vision_start|><|image_pad|><|vision_end|>"
                for i in range(2, n_imgs + 1):
                    replace += f" <image{i}><|vision_start|><|image_pad|><|vision_end|>"
                template = self.prompt_template_ti2i.replace(
                    "<image1><|vision_start|><|image_pad|><|vision_end|>", replace
                )
                prompts.append(template.format(t))
            for _ in prompt:
                for img in image:
                    if not isinstance(img, PILImage.Image):
                        img = PILImage.fromarray(img)
                    if img.mode == "RGBA":
                        white = PILImage.new("RGB", img.size, (255, 255, 255))
                        white.paste(img, mask=img.getchannel("A"))
                        img = white
                    condition_pil_list.append(img)

        processor_kwargs = {"text": prompts, "padding": True, "padding_side": "left", "return_tensors": "pt"}
        if not is_t2i:
            processor_kwargs["images"] = condition_pil_list
        model_inputs = self.processor(**processor_kwargs).to(device)

        forward_kwargs = {
            "input_ids": model_inputs.input_ids,
            "attention_mask": model_inputs.attention_mask,
            "output_hidden_states": True,
        }
        if not is_t2i and hasattr(model_inputs, "pixel_values"):
            forward_kwargs.update(pixel_values=model_inputs.pixel_values, image_grid_thw=model_inputs.image_grid_thw)
        if hasattr(model_inputs, "mm_token_type_ids"):
            forward_kwargs["mm_token_type_ids"] = model_inputs.mm_token_type_ids

        text_model = getattr(self.text_encoder.model, "language_model", self.text_encoder.model)
        handle = text_model.norm.register_forward_hook(lambda module, args, output: args[0])
        try:
            outputs = self.text_encoder(**forward_kwargs)
        finally:
            handle.remove()
        hidden_states = outputs.hidden_states[-1]

        split_hidden_states = list(self._extract_masked_hidden(hidden_states, model_inputs.attention_mask))
        split_hidden_states = [e[self._drop_idx :] for e in split_hidden_states]
        image_pad_mask = [
            (sample_ids[sample_mask.bool()] == self._img_token_id)
            for sample_ids, sample_mask in zip(model_inputs.input_ids, model_inputs.attention_mask)
        ]
        image_pad_mask = [e[self._drop_idx :] for e in image_pad_mask]

        attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
        max_seq_len = max(e.size(0) for e in split_hidden_states)
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states]
        )
        encoder_attention_mask = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list]
        )
        image_pad_mask = torch.stack([torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in image_pad_mask])
        return prompt_embeds, encoder_attention_mask, image_pad_mask

    def _encode_prompt(
        self,
        prompt: List[str],
        device: torch.device,
        max_sequence_length: int,
        image: list | None = None,
        num_images_per_prompt: int = 1,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        prompt_embeds, prompt_embeds_mask, image_pad_mask = self._get_qwen_prompt_embeds(prompt, image=image, device=device)
        batch_size, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1).view(batch_size * num_images_per_prompt, seq_len, -1)
        prompt_embeds_mask = prompt_embeds_mask.repeat(1, num_images_per_prompt).view(batch_size * num_images_per_prompt, seq_len)
        image_pad_mask = image_pad_mask.repeat(1, num_images_per_prompt).view(batch_size * num_images_per_prompt, seq_len)
        if prompt_embeds_mask is not None and prompt_embeds_mask.all():
            prompt_embeds_mask = None
        prompt_embeds = prompt_embeds[:, :max_sequence_length].to(dtype=self.dtype, device=device)
        if prompt_embeds_mask is not None:
            prompt_embeds_mask = prompt_embeds_mask[:, :max_sequence_length]
        image_pad_mask = image_pad_mask[:, :max_sequence_length].bool()
        return prompt_embeds, prompt_embeds_mask, image_pad_mask

    def _prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[torch.Generator],
        image_latents: Optional[torch.Tensor] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        h = 2 * (int(height) // (self.vae_scale_factor * 2))
        w = 2 * (int(width) // (self.vae_scale_factor * 2))
        if latents is None:
            shape = (batch_size, 1, num_channels_latents, h, w)
            latents = torch.randn(shape, generator=generator, device=device, dtype=dtype)
            latents = self._pack_latents(latents, batch_size, num_channels_latents, h, w)
        else:
            latents = latents.to(device=device, dtype=dtype)
        return latents, image_latents

    def _decode_latents(self, latents: torch.Tensor, height: int, width: int) -> torch.Tensor:
        latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
        latents = latents.to(self.vae.dtype)
        z_dim = self.vae.config.z_dim
        latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, z_dim, 1, 1, 1).to(latents.device, latents.dtype)
        latents_std = torch.tensor(self.vae.config.latents_std).view(1, z_dim, 1, 1, 1).to(latents.device, latents.dtype)
        latents = latents * latents_std + latents_mean
        image = self.vae.decode(latents, return_dict=False)[0]
        if image.shape[1] >= 3:
            image = image[:, :3]
        image = image[:, :, 0]
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.permute(0, 2, 3, 1)
        image = (image * 255).round().to(torch.uint8)
        return image

    def infer(self, req):
        params = req.params
        num_per = params.num_images_per_prompt or 1
        prompts = req.prompt if isinstance(req.prompt, list) else [req.prompt]
        prompts = [p for p in prompts for _ in range(num_per)]
        extra_params = params.extra_params or {}
        refs = params.image_reference or []
        return self.forward(
            prompt=prompts,
            image=[r.content for r in refs] if refs else None,
            height=params.height,
            width=params.width,
            num_inference_steps=params.num_inference_steps,
            negative_prompt_cfg_scale=params.guidance_scale,
            seed=params.seed,
            max_sequence_length=params.max_sequence_length,
            output_resolution=extra_params.get("output_resolution", 1024),
            use_kv_cache=extra_params.get("use_kv_cache", True),
        )

    @torch.inference_mode()
    def forward(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        image: Any | None = None,
        height: int | None = 1024,
        width: int | None = 1024,
        num_inference_steps: int = 40,
        negative_prompt_cfg_scale: float = 1.0,
        true_cfg_scale: Optional[float] = None,
        seed: int = 42,
        max_sequence_length: int = 4096,
        sigmas: Optional[list] = None,
        output_resolution: int = 1024,
        use_kv_cache: bool = True,
        **kwargs: Any,
    ) -> PipelineOutput:
        """Qwen-Image 2.1 text-to-image / image-conditioned forward path."""
        del negative_prompt, true_cfg_scale, kwargs
        from PIL import Image as PILImage

        pipeline_start = time.time()
        timer = CudaPhaseTimer()
        timer.mark_pre_start()
        if isinstance(prompt, str):
            prompt = [prompt]
        batch_size = len(prompt)
        device = self.device
        generator = make_noise_generator(seed, device)

        condition_images = None
        if image is not None:
            image = image if isinstance(image, list) else [image]
            condition_images = []
            for img in image:
                if isinstance(img, bytes):
                    pil_img = PILImage.open(BytesIO(img)).convert("RGBA")
                elif isinstance(img, PILImage.Image):
                    pil_img = img
                elif isinstance(img, np.ndarray):
                    pil_img = PILImage.fromarray(img)
                else:
                    raise ValueError(f"image accepts bytes, PIL image, numpy array, or list thereof, got {type(img).__name__}")
                condition_images.append(pil_img)
            calculated_width, calculated_height, _ = calculate_dimensions(
                output_resolution * output_resolution,
                condition_images[-1].size[0] / condition_images[-1].size[1],
            )
            height = height or calculated_height
            width = width or calculated_width

        if height is None or width is None:
            raise ValueError("height and width must be set for text-to-image requests without references.")
        multiple_of = self.vae_scale_factor * 2
        width = int(width) // multiple_of * multiple_of
        height = int(height) // multiple_of * multiple_of

        logger.info("Encoding Qwen-Image 2.1 prompt...")
        prompt_embeds, prompt_embeds_mask, image_pad_mask = self._encode_prompt(
            prompt, device, max_sequence_length, image=condition_images
        )
        num_channels_latents = int(getattr(self.transformer.config, "in_channels", self.latent_channels))
        latents, input_images_latents = self._prepare_latents(
            batch_size,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
        )
        if input_images_latents is not None:
            latent_model_token_count = input_images_latents.shape[1] + latents.shape[1]
        else:
            latent_model_token_count = latents.shape[1]
        del latent_model_token_count
        img_shapes = [[(1, height // self.vae_scale_factor, width // self.vae_scale_factor)]] * batch_size
        img_mask = append_target_slots(image_pad_mask.bool(), latents.shape[1])

        sigmas_np = sigmas if sigmas is not None else np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        mu = calculate_shift(
            latents.shape[1],
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 8192),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 0.9),
        )
        self.scheduler.set_timesteps(sigmas=sigmas_np, device=device, mu=mu)
        timesteps = self.scheduler.timesteps
        self.scheduler.set_begin_index(0)

        timer.mark_denoise_start()
        kv_cache = None
        cache_enabled = bool(use_kv_cache) and hasattr(self.transformer, "transformer_blocks") and getattr(
            self.transformer.config, "causal_condition", True
        )
        if cache_enabled:
            kv_cache = QwenImage21KVCache(len(self.transformer.transformer_blocks))
        logger.info("Denoising Qwen-Image 2.1 (%d steps, kv_cache=%s)...", len(timesteps), cache_enabled)
        for i, t in self._profile_denoise_steps(timesteps):
            timestep = t.expand(latents.shape[0]).to(latents.dtype)
            kv_cache_mode = "extract" if cache_enabled and i == 0 else ("cached" if cache_enabled else None)
            latent_model_input = latents if input_images_latents is None else torch.cat([input_images_latents, latents], dim=1)
            noise_pred = self.transformer(
                hidden_states=latent_model_input,
                timestep=timestep / 1000,
                encoder_hidden_states=prompt_embeds,
                encoder_hidden_states_mask=prompt_embeds_mask,
                img_shapes=img_shapes,
                img_mask=img_mask,
                kv_cache=kv_cache,
                kv_cache_mode=kv_cache_mode,
                return_dict=False,
            )[0]
            noise_pred = noise_pred[:, -latents.size(1) :]
            latents_dtype = latents.dtype
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            if latents.dtype != latents_dtype:
                latents = latents.to(latents_dtype)

        timer.mark_post_start()
        logger.info("Decoding Qwen-Image 2.1 latents...")
        image = self._decode_latents(latents, height, width)
        logger.info("Qwen-Image 2.1 pipeline total: %.2fs", time.time() - pipeline_start)
        timer.mark_end()
        return timer.fill(PipelineOutput(image=image))


__all__ = ["QwenImage21Pipeline"]
