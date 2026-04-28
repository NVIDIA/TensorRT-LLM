# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen-Image text-to-image pipeline.

Ports the denoise loop, Qwen2.5-VL text encoding, FlowMatchEuler
sampling, and VAE decode from ``diffusers.QwenImagePipeline`` onto the
TensorRT-LLM VisualGen executor. The transformer backbone is our port
from ``transformer_qwen_image.py``.

Non-transformer components (VAE, text encoder, tokenizer, scheduler)
are loaded directly from the HF checkpoint using diffusers/transformers.
"""

import time
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch

from tensorrt_llm._torch.visual_gen.config import PipelineComponent
from tensorrt_llm._torch.visual_gen.output import MediaOutput
from tensorrt_llm._torch.visual_gen.pipeline import BasePipeline
from tensorrt_llm._torch.visual_gen.pipeline_registry import register_pipeline
from tensorrt_llm.logger import logger

from .transformer_qwen_image import QwenImageTransformer2DModel


# ``self.prompt_template_encode`` from diffusers.QwenImagePipeline.
_PROMPT_TEMPLATE = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, "
    "size, texture, quantity, text, spatial relationships of the objects and "
    "background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n"
    "<|im_start|>assistant\n"
)
_PROMPT_TEMPLATE_START_IDX = 34  # tokens to drop before the user prompt


def _calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    """Dynamic mu shift for FlowMatchEulerDiscreteScheduler.

    Identical formula to diffusers' ``pipeline_qwenimage.calculate_shift``.
    """
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


_DEFAULT_GENERATION_PARAMS = {
    "height": 1328,
    "width": 1328,
    "num_inference_steps": 50,
    "guidance_scale": 4.0,
    "max_sequence_length": 1024,
}


@register_pipeline("QwenImagePipeline")
class QwenImagePipeline(BasePipeline):
    """Qwen-Image text-to-image pipeline.

    References:
        - Model card: https://huggingface.co/Qwen/Qwen-Image
        - Tech report: https://arxiv.org/abs/2508.02324
        - diffusers: ``diffusers.pipelines.qwenimage.pipeline_qwenimage``
    """

    # Older releases of TensorRT-LLM read a class-level
    # ``DEFAULT_GENERATION_PARAMS`` dict; newer releases read a
    # ``default_generation_params`` property. Expose both so we work on
    # either version.
    DEFAULT_GENERATION_PARAMS = _DEFAULT_GENERATION_PARAMS

    def __init__(self, model_config):
        super().__init__(model_config)
        # Qwen-Image uses 8x VAE downsample + 2x2 patch packing. Both
        # scheduler and image-prep assume a latent grid divisible by
        # (vae_scale_factor * 2 == 16). vae_scale_factor is updated by
        # load_standard_components() once the VAE is loaded.
        self.vae_scale_factor = 8
        self.tokenizer_max_length = 1024

    @property
    def dtype(self):
        return self.model_config.torch_dtype

    @property
    def device(self):
        if self.transformer is not None:
            return next(self.transformer.parameters()).device
        return torch.device("cuda:0")

    # ------------------------------------------------------------------
    # Warmup / resolution constraints.
    # ------------------------------------------------------------------
    @property
    def default_warmup_resolutions(self) -> List[Tuple[int, int]]:
        return [(1328, 1328)]

    @property
    def default_warmup_num_frames(self) -> List[int]:
        return [1]

    def warmup_cache_key(self, height: int, width: int, **kwargs) -> tuple:
        return (height, width)

    @property
    def resolution_multiple_of(self) -> Tuple[int, int]:
        # VAE scale factor * 2 for the 2x2 patch packing. The default
        # vae_scale_factor (8) gives 16. If the VAE is loaded and reports
        # a different compression ratio, ``load_standard_components``
        # updates ``self.vae_scale_factor``.
        return (self.vae_scale_factor * 2, self.vae_scale_factor * 2)

    # ------------------------------------------------------------------
    # Component initialisation.
    # ------------------------------------------------------------------
    def _init_transformer(self) -> None:
        logger.info("Creating Qwen-Image transformer")
        # ``pretrained_config`` on the DiffusionModelConfig is populated
        # from ``<ckpt>/transformer/config.json`` as a SimpleNamespace by
        # ``DiffusionModelConfig.from_pretrained``. Read the fields we
        # care about with sensible defaults (matching the Qwen-Image 20B
        # reference model).
        pretrained = getattr(self.model_config, "pretrained_config", None)

        def _cfg(name: str, default):
            if pretrained is None:
                return default
            if isinstance(pretrained, dict):
                return pretrained.get(name, default)
            return getattr(pretrained, name, default)

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
        )

    def _run_warmup(self, height: int, width: int, num_frames: int, steps: int) -> None:
        """Run a single warmup forward with dummy inputs.

        Following FLUX's pattern: short denoise through the real
        ``forward`` path so CUDA graphs / torch.compile / VAE kernels
        all get triggered with the runtime shape.
        """
        with torch.no_grad():
            self.forward(
                prompt="warmup",
                height=height,
                width=width,
                num_inference_steps=max(steps, 2),
                true_cfg_scale=1.0,
                seed=42,
                max_sequence_length=64,
            )

    def load_standard_components(
        self,
        checkpoint_dir: str,
        device: torch.device,
        skip_components: Optional[list] = None,
    ) -> None:
        skip_components = skip_components or []

        try:
            from diffusers import (
                AutoencoderKLQwenImage,
                FlowMatchEulerDiscreteScheduler,
            )
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "Qwen-Image requires diffusers with AutoencoderKLQwenImage "
                "(`pip install -U diffusers`)."
            ) from e

        try:
            from transformers import (
                Qwen2_5_VLForConditionalGeneration,
                Qwen2Tokenizer,
            )
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "Qwen-Image requires transformers with Qwen2_5_VL* "
                "(`pip install -U transformers`)."
            ) from e

        if PipelineComponent.TOKENIZER not in skip_components:
            logger.info("Loading Qwen2 tokenizer...")
            self.tokenizer = Qwen2Tokenizer.from_pretrained(
                checkpoint_dir, subfolder=PipelineComponent.TOKENIZER
            )

        if PipelineComponent.TEXT_ENCODER not in skip_components:
            logger.info("Loading Qwen2.5-VL text encoder...")
            self.text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                checkpoint_dir,
                subfolder=PipelineComponent.TEXT_ENCODER,
                torch_dtype=self.model_config.torch_dtype,
            ).to(device)

        if PipelineComponent.VAE not in skip_components:
            logger.info("Loading Qwen-Image VAE...")
            self.vae = AutoencoderKLQwenImage.from_pretrained(
                checkpoint_dir,
                subfolder=PipelineComponent.VAE,
                torch_dtype=torch.bfloat16,
            ).to(device)
            # Qwen-Image VAE has a ``temperal_downsample`` list (sic --
            # typo in diffusers source); vae_scale_factor = 2**len(it).
            temperal_downsample = getattr(
                self.vae, "temperal_downsample", [1, 1, 1]
            )
            self.vae_scale_factor = 2 ** len(temperal_downsample)

        if PipelineComponent.SCHEDULER not in skip_components:
            logger.info("Loading Qwen-Image scheduler...")
            self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                checkpoint_dir, subfolder=PipelineComponent.SCHEDULER
            )

        self.default_height = 1328
        self.default_width = 1328
        self.max_sequence_length = 1024

    def load_weights(self, weights: dict) -> None:
        if self.transformer is not None:
            transformer_weights = weights.get("transformer", weights)
            self.transformer.load_weights(transformer_weights)
            # PipelineLoader materialises MetaInit tensors as fp32 by
            # default; cast the transformer to the configured inference
            # dtype (normally bf16 for Qwen-Image) so forward doesn't
            # hit Float vs BFloat16 mat1/mat2 mismatches.
            self.transformer.to(self.model_config.torch_dtype).eval()
        self._target_dtype = self.model_config.torch_dtype

    # ------------------------------------------------------------------
    # Prompt encoding (Qwen2.5-VL chat template).
    # ------------------------------------------------------------------
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
        """Encode a list of prompts via Qwen2.5-VL + chat template.

        Returns:
            prompt_embeds: ``(B, S, 3584)`` in transformer dtype.
            prompt_embeds_mask: ``(B, S)`` bool mask.
        """
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
        attn_masks = [torch.ones(h.size(0), dtype=torch.long, device=h.device) for h in split_hidden]
        max_len = max(h.size(0) for h in split_hidden)
        prompt_embeds = torch.stack(
            [torch.cat([h, h.new_zeros(max_len - h.size(0), h.size(1))]) for h in split_hidden]
        )
        prompt_embeds_mask = torch.stack(
            [torch.cat([m, m.new_zeros(max_len - m.size(0))]) for m in attn_masks]
        )

        # Clamp to user's max.
        prompt_embeds = prompt_embeds[:, :max_sequence_length]
        prompt_embeds_mask = prompt_embeds_mask[:, :max_sequence_length]
        prompt_embeds = prompt_embeds.to(dtype=self.dtype, device=device)
        return prompt_embeds, prompt_embeds_mask

    # ------------------------------------------------------------------
    # Latent prep / packing (identical shape to FLUX).
    # ------------------------------------------------------------------
    @staticmethod
    def _pack_latents(
        latents: torch.Tensor,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
    ) -> torch.Tensor:
        latents = latents.view(
            batch_size, num_channels_latents, height // 2, 2, width // 2, 2
        )
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        return latents.reshape(
            batch_size, (height // 2) * (width // 2), num_channels_latents * 4
        )

    @staticmethod
    def _unpack_latents(
        latents: torch.Tensor, height: int, width: int, vae_scale_factor: int
    ) -> torch.Tensor:
        batch_size, _, channels = latents.shape
        h = 2 * (int(height) // (vae_scale_factor * 2))
        w = 2 * (int(width) // (vae_scale_factor * 2))

        latents = latents.view(batch_size, h // 2, w // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        # Qwen-Image VAE decode expects a frame dim (3D VAE), so emit
        # (B, C, 1, H, W).
        return latents.reshape(batch_size, channels // 4, 1, h, w)

    def _prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        from diffusers.utils.torch_utils import randn_tensor

        h = 2 * (int(height) // (self.vae_scale_factor * 2))
        w = 2 * (int(width) // (self.vae_scale_factor * 2))
        shape = (batch_size, 1, num_channels_latents, h, w)
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return self._pack_latents(latents, batch_size, num_channels_latents, h, w)

    def _decode_latents(
        self, latents: torch.Tensor, height: int, width: int
    ) -> torch.Tensor:
        latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
        latents = latents.to(self.vae.dtype)

        z_dim = self.vae.config.z_dim
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = (
            1.0
            / torch.tensor(self.vae.config.latents_std)
            .view(1, z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents = latents / latents_std + latents_mean
        image = self.vae.decode(latents, return_dict=False)[0][:, :, 0]

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
        image = (image * 255).round().to(torch.uint8)
        return image

    # ------------------------------------------------------------------
    # Inference entry points.
    # ------------------------------------------------------------------
    @property
    def default_generation_params(self) -> dict:
        return dict(_DEFAULT_GENERATION_PARAMS)

    def infer(self, req):
        # Fan out by num_images_per_prompt so ``n > 1`` produces multiple
        # images in a single batched forward. Qwen-Image supports this
        # naturally via the batch dim of its transformer.
        num_per = getattr(req, "num_images_per_prompt", 1) or 1
        base_prompts = req.prompt if isinstance(req.prompt, list) else [req.prompt]
        prompts = [p for p in base_prompts for _ in range(num_per)]
        negative = getattr(req, "negative_prompt", None)
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
            prompt=prompts,
            negative_prompt=negative,
            height=req.height,
            width=req.width,
            num_inference_steps=req.num_inference_steps,
            true_cfg_scale=getattr(req, "guidance_scale", 4.0),
            seed=req.seed,
            max_sequence_length=req.max_sequence_length,
        )

    @torch.inference_mode()
    def forward(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 1328,
        width: int = 1328,
        num_inference_steps: int = 50,
        true_cfg_scale: float = 4.0,
        seed: int = 42,
        max_sequence_length: int = 1024,
        sigmas: Optional[list] = None,
    ) -> MediaOutput:
        """Text-to-image generation.

        Implementation mirrors ``diffusers.QwenImagePipeline.__call__``
        with the FlowMatchEuler sampler and real CFG (``true_cfg_scale``).
        """
        pipeline_start = time.time()

        if isinstance(prompt, str):
            prompt = [prompt]
        batch_size = len(prompt)

        has_neg = negative_prompt is not None
        do_true_cfg = true_cfg_scale > 1.0 and has_neg

        device = self.device
        generator = torch.Generator(device=device).manual_seed(seed)

        # Text encoding.
        logger.info("Encoding prompt...")
        prompt_embeds, prompt_embeds_mask = self._encode_prompt(
            prompt, device, max_sequence_length
        )
        neg_prompt_embeds = neg_prompt_embeds_mask = None
        if do_true_cfg:
            if isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt] * batch_size
            neg_prompt_embeds, neg_prompt_embeds_mask = self._encode_prompt(
                negative_prompt, device, max_sequence_length
            )

        # Latents.
        num_channels_latents = self.transformer.in_channels // 4  # 16
        latents = self._prepare_latents(
            batch_size,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
        )
        # img_shapes: list-of-list-of-(frame, h_patch, w_patch), one per batch item.
        img_shapes = [
            [
                (
                    1,
                    height // self.vae_scale_factor // 2,
                    width // self.vae_scale_factor // 2,
                )
            ]
        ] * batch_size

        # Timesteps with dynamic shift.
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
        self.scheduler.set_timesteps(
            sigmas=sigmas_np, device=device, mu=mu
        )
        timesteps = self.scheduler.timesteps
        self.scheduler.set_begin_index(0)

        # Denoise loop.
        logger.info("Denoising (%d steps)...", len(timesteps))
        for i, t in enumerate(timesteps):
            timestep = t.expand(latents.shape[0]).to(latents.dtype)
            noise_pred = self.transformer(
                hidden_states=latents,
                timestep=timestep / 1000,
                encoder_hidden_states_mask=prompt_embeds_mask,
                encoder_hidden_states=prompt_embeds,
                img_shapes=img_shapes,
                return_dict=False,
            )[0]

            if do_true_cfg:
                neg_noise_pred = self.transformer(
                    hidden_states=latents,
                    timestep=timestep / 1000,
                    encoder_hidden_states_mask=neg_prompt_embeds_mask,
                    encoder_hidden_states=neg_prompt_embeds,
                    img_shapes=img_shapes,
                    return_dict=False,
                )[0]
                comb = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)
                cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
                noise_norm = torch.norm(comb, dim=-1, keepdim=True)
                noise_pred = comb * (cond_norm / noise_norm)

            latents_dtype = latents.dtype
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            if latents.dtype != latents_dtype:
                latents = latents.to(latents_dtype)

        logger.info("Decoding...")
        image = self._decode_latents(latents, height, width)

        if getattr(self, "rank", 0) == 0:
            logger.info("Pipeline total: %.2fs", time.time() - pipeline_start)

        return MediaOutput(image=image)
