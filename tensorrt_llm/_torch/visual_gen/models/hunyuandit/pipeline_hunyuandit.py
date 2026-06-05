# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HunyuanDiT text-to-image pipeline.

Ports the denoise loop, bilingual text encoding (BertModel + MT5EncoderModel),
DDPM sampling, and VAE decode from ``diffusers.HunyuanDiTPipeline`` onto the
TensorRT-LLM VisualGen executor.

The transformer backbone is loaded via :class:`HunyuanDiT2DModelWrapper`
which wraps the diffusers ``HunyuanDiT2DModel``.  All other components (VAE,
text encoders, tokenizers, scheduler) are loaded directly from the HuggingFace
checkpoint using diffusers / transformers.

References:
    - Model card:  https://huggingface.co/Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers
    - Tencent repo: https://github.com/Tencent-Hunyuan/HunyuanDiT
    - diffusers:    ``diffusers.pipelines.hunyuan_dit.pipeline_hunyuan_dit``
"""

import math
import time
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from tensorrt_llm._torch.visual_gen.output import CudaPhaseTimer, PipelineOutput
from tensorrt_llm._torch.visual_gen.pipeline import BasePipeline
from tensorrt_llm._torch.visual_gen.pipeline_registry import PipelineComponent, register_pipeline
from tensorrt_llm.logger import logger

from .defaults import get_hunyuandit_default_params, get_hunyuandit_extra_param_specs
from .transformer_hunyuandit import HunyuanDiT2DModelWrapper

# ---------------------------------------------------------------------------
# Resolution binning
# ---------------------------------------------------------------------------

# HunyuanDiT was trained on these aspect-ratio buckets (height × width).
# We snap user-requested resolutions to the closest bucket by default.
_SUPPORTED_SHAPE_BINNING = (
    (1024, 1024),
    (1280, 1280),
    (1024, 768),
    (768, 1024),
    (1280, 960),
    (960, 1280),
    (1280, 768),
    (768, 1280),
)


def _map_to_standard_shapes(
    target_height: int, target_width: int
) -> Tuple[int, int]:
    """Return the closest supported (height, width) pair.

    Matching strategy: minimise the Euclidean distance in log-space between
    the user's aspect ratio and the training buckets, then prefer the bucket
    whose total pixel count is closest to the user's.  This matches the
    reference diffusers implementation.
    """
    target_ratio = target_height / target_width
    best = None
    best_dist = float("inf")
    for h, w in _SUPPORTED_SHAPE_BINNING:
        dist = abs(math.log(h / w) - math.log(target_ratio))
        if dist < best_dist:
            best_dist = dist
            best = (h, w)
    return best  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

_DEFAULT_GENERATION_PARAMS = get_hunyuandit_default_params()


@register_pipeline(
    "HunyuanDiTPipeline",
    hf_ids=[
        "Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers",
        "Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers",
        "Tencent-Hunyuan/HunyuanDiT-v1.0-Diffusers",
    ],
    doc="Tencent HunyuanDiT bilingual (Chinese/English) text-to-image pipeline.",
)
class HunyuanDiTPipeline(BasePipeline):
    """HunyuanDiT Text-to-Image Pipeline.

    Supports HunyuanDiT-v1.0, v1.1, and v1.2 diffusers checkpoints.

    Text conditioning uses a dual-encoder architecture:
      * A bilingual CLIP-like ``BertModel`` for short (≤ 77 token) sequences.
      * An ``MT5EncoderModel`` for long (≤ 256 token) sequences.
    Both encodings are passed jointly to the transformer.
    """

    DEFAULT_GENERATION_PARAMS = _DEFAULT_GENERATION_PARAMS

    def __init__(self, model_config):
        super().__init__(model_config)
        self.vae_scale_factor = 8  # SD-style VAE, 8× spatial compression

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def dtype(self) -> torch.dtype:
        return self.model_config.torch_dtype

    @property
    def device(self) -> torch.device:
        if self.transformer is not None:
            return next(self.transformer.parameters()).device
        return torch.device("cuda:0")

    @property
    def default_warmup_resolutions(self) -> List[Tuple[int, int]]:
        return [(512, 512), (1024, 1024)]

    @property
    def default_warmup_num_frames(self) -> List[int]:
        return [1]

    @property
    def resolution_multiple_of(self) -> Tuple[int, int]:
        return (self.vae_scale_factor, self.vae_scale_factor)

    @property
    def default_generation_params(self) -> dict:
        return dict(_DEFAULT_GENERATION_PARAMS)

    @property
    def extra_param_specs(self) -> dict:
        return get_hunyuandit_extra_param_specs()

    # ------------------------------------------------------------------
    # Component initialisation
    # ------------------------------------------------------------------

    def _init_transformer(self) -> None:
        logger.info("Creating HunyuanDiT2D transformer")
        self.transformer = HunyuanDiT2DModelWrapper(model_config=self.model_config)

    def _run_warmup(self, height: int, width: int, num_frames: int, steps: int) -> None:
        with torch.no_grad():
            self.forward(
                prompt="warmup",
                height=height,
                width=width,
                num_inference_steps=max(steps, 2),
                guidance_scale=7.5,
                seed=42,
                use_resolution_binning=False,
            )

    def load_standard_components(
        self,
        checkpoint_dir: str,
        device: torch.device,
        skip_components: Optional[list] = None,
    ) -> None:
        skip_components = skip_components or []

        try:
            from diffusers import AutoencoderKL, DDPMScheduler
        except ImportError as exc:
            raise ImportError(
                "HunyuanDiT requires diffusers >= 0.26 (`pip install -U diffusers`)."
            ) from exc

        try:
            from transformers import AutoTokenizer, BertModel, MT5EncoderModel, T5Tokenizer
        except ImportError as exc:
            raise ImportError(
                "HunyuanDiT requires transformers (`pip install -U transformers`)."
            ) from exc

        if PipelineComponent.TOKENIZER not in skip_components:
            logger.info("Loading HunyuanDiT CLIP tokenizer (BertTokenizer)...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                checkpoint_dir, subfolder=PipelineComponent.TOKENIZER
            )

        if PipelineComponent.TOKENIZER_2 not in skip_components:
            logger.info("Loading HunyuanDiT T5 tokenizer (MT5Tokenizer)...")
            self.tokenizer_2 = T5Tokenizer.from_pretrained(
                checkpoint_dir, subfolder=PipelineComponent.TOKENIZER_2
            )

        if PipelineComponent.TEXT_ENCODER not in skip_components:
            logger.info("Loading HunyuanDiT CLIP text encoder (BertModel)...")
            self.text_encoder = BertModel.from_pretrained(
                checkpoint_dir,
                subfolder=PipelineComponent.TEXT_ENCODER,
                torch_dtype=self.model_config.torch_dtype,
            ).to(device)
            self.text_encoder.eval()

        if PipelineComponent.TEXT_ENCODER_2 not in skip_components:
            logger.info("Loading HunyuanDiT T5 text encoder (MT5EncoderModel)...")
            self.text_encoder_2 = MT5EncoderModel.from_pretrained(
                checkpoint_dir,
                subfolder=PipelineComponent.TEXT_ENCODER_2,
                torch_dtype=self.model_config.torch_dtype,
            ).to(device)
            self.text_encoder_2.eval()

        if PipelineComponent.VAE not in skip_components:
            logger.info("Loading HunyuanDiT VAE (AutoencoderKL)...")
            self.vae = AutoencoderKL.from_pretrained(
                checkpoint_dir,
                subfolder=PipelineComponent.VAE,
                torch_dtype=torch.float32,  # VAE decode in fp32 for numerical stability
            ).to(device)
            self.vae.eval()
            self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        if PipelineComponent.SCHEDULER not in skip_components:
            logger.info("Loading HunyuanDiT DDPM scheduler...")
            self.scheduler = DDPMScheduler.from_pretrained(
                checkpoint_dir, subfolder=PipelineComponent.SCHEDULER
            )

    def load_weights(self, weights: dict) -> None:
        if self.transformer is not None:
            transformer_weights = weights.get("transformer", weights)
            self.transformer.load_weights(transformer_weights)
            self.transformer.to_inference_dtype().eval()
        self._target_dtype = self.model_config.torch_dtype

    # ------------------------------------------------------------------
    # Text encoding (bilingual: BertModel + MT5EncoderModel)
    # ------------------------------------------------------------------

    def _encode_prompt_clip(
        self,
        prompt: List[str],
        device: torch.device,
        max_sequence_length: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode via BertModel (short CLIP-like encoder, max 77 tokens)."""
        max_len = min(max_sequence_length, 77)
        tok_out = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_len,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            out = self.text_encoder(
                input_ids=tok_out.input_ids,
                attention_mask=tok_out.attention_mask,
            )
        return out.last_hidden_state.to(self.dtype), tok_out.attention_mask.to(device)

    def _encode_prompt_t5(
        self,
        prompt: List[str],
        device: torch.device,
        max_sequence_length: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode via MT5EncoderModel (long sequence encoder, max 256 tokens)."""
        max_len = min(max_sequence_length, 256)
        tok_out = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=max_len,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            out = self.text_encoder_2(
                input_ids=tok_out.input_ids,
                attention_mask=tok_out.attention_mask,
            )
        return out.last_hidden_state.to(self.dtype), tok_out.attention_mask.to(device)

    def _encode_prompt(
        self,
        prompt: List[str],
        device: torch.device,
        max_sequence_length: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (clip_embeds, clip_mask, t5_embeds, t5_mask)."""
        clip_embeds, clip_mask = self._encode_prompt_clip(prompt, device, max_sequence_length)
        t5_embeds, t5_mask = self._encode_prompt_t5(prompt, device, max_sequence_length)
        return clip_embeds, clip_mask, t5_embeds, t5_mask

    # ------------------------------------------------------------------
    # Latent utilities
    # ------------------------------------------------------------------

    def _prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: torch.Generator,
    ) -> torch.Tensor:
        from diffusers.utils.torch_utils import randn_tensor

        shape = (
            batch_size,
            num_channels_latents,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        return randn_tensor(shape, generator=generator, device=device, dtype=dtype)

    def _decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """VAE decode → uint8 (B, H, W, 3) tensor."""
        # Scale latents per the SD VAE convention
        latents = latents / self.vae.config.scaling_factor
        with torch.no_grad():
            image = self.vae.decode(latents.to(torch.float32), return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.permute(0, 2, 3, 1)  # (B, C, H, W) → (B, H, W, C)
        return (image * 255).round().to(torch.uint8)

    # ------------------------------------------------------------------
    # RoPE image embedding helper
    # ------------------------------------------------------------------

    @staticmethod
    def _get_image_rotary_emb(
        patch_size: int,
        vae_scale_factor: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Compute 2D RoPE embeddings for the latent grid.

        Follows diffusers' ``get_2d_rotary_pos_embed``.
        """
        try:
            from diffusers.models.embeddings import get_2d_rotary_pos_embed
        except ImportError:
            return None  # type: ignore[return-value]

        grid_height = height // (vae_scale_factor * patch_size)
        grid_width = width // (vae_scale_factor * patch_size)
        base_size = 512 // (vae_scale_factor * patch_size)
        grid_crops_coords = (
            (0, 0),
            (grid_height, grid_width),
        )
        freqs_cos, freqs_sin = get_2d_rotary_pos_embed(
            embed_dim=88,
            crops_coords=grid_crops_coords,
            grid_size=(grid_height, grid_width),
            use_real=True,
            base_size=base_size,
        )
        return (
            freqs_cos.to(device=device, dtype=dtype),
            freqs_sin.to(device=device, dtype=dtype),
        )

    # ------------------------------------------------------------------
    # Inference entry points
    # ------------------------------------------------------------------

    def infer(self, req):
        params = req.params
        num_per = params.num_images_per_prompt or 1
        base_prompts = req.prompt if isinstance(req.prompt, list) else [req.prompt]
        prompts = [p for p in base_prompts for _ in range(num_per)]

        negative = params.negative_prompt
        if negative is not None:
            negatives = negative if isinstance(negative, list) else [negative]
            if len(negatives) == 1:
                negatives = negatives * len(base_prompts)
            negative = [n for n in negatives for _ in range(num_per)]

        extra = getattr(params, "extra_params", {}) or {}
        return self.forward(
            prompt=prompts,
            negative_prompt=negative,
            height=params.height,
            width=params.width,
            num_inference_steps=params.num_inference_steps,
            guidance_scale=params.guidance_scale,
            seed=params.seed,
            max_sequence_length=params.max_sequence_length,
            use_resolution_binning=extra.get("use_resolution_binning", True),
        )

    @torch.inference_mode()
    def forward(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        seed: int = 42,
        max_sequence_length: int = 77,
        use_resolution_binning: bool = True,
        **kwargs,
    ) -> PipelineOutput:
        """Text-to-image generation with HunyuanDiT.

        Mirrors ``diffusers.HunyuanDiTPipeline.__call__`` with classifier-free
        guidance (separate negative prompt path).
        """
        pipeline_start = time.time()
        timer = CudaPhaseTimer()
        timer.mark_pre_start()

        if isinstance(prompt, str):
            prompt = [prompt]
        batch_size = len(prompt)

        do_cfg = guidance_scale > 1.0

        device = self.device
        generator = torch.Generator(device=device).manual_seed(seed)

        # Optionally snap to a training bucket
        if use_resolution_binning:
            height, width = _map_to_standard_shapes(height, width)
            logger.info("HunyuanDiT: using binned resolution %d×%d", height, width)

        # Text encoding (bilingual)
        logger.info("Encoding prompt...")
        clip_embeds, clip_mask, t5_embeds, t5_mask = self._encode_prompt(
            prompt, device, max_sequence_length
        )

        if do_cfg:
            neg = negative_prompt
            if neg is None:
                neg = [""] * batch_size
            elif isinstance(neg, str):
                neg = [neg] * batch_size
            neg_clip, neg_clip_mask, neg_t5, neg_t5_mask = self._encode_prompt(
                neg, device, max_sequence_length
            )
            # Concatenate along batch dim for a single forward pass
            clip_embeds = torch.cat([neg_clip, clip_embeds])
            clip_mask = torch.cat([neg_clip_mask, clip_mask])
            t5_embeds = torch.cat([neg_t5, t5_embeds])
            t5_mask = torch.cat([neg_t5_mask, t5_mask])

        # Latents
        num_channels_latents = self.transformer.in_channels
        latents = self._prepare_latents(
            batch_size,
            num_channels_latents,
            height,
            width,
            self.dtype,
            device,
            generator,
        )

        # Image meta (target/source sizes for HunyuanDiT style conditioning)
        image_meta_size = torch.tensor(
            [height, width, height, width, 0, 0] * batch_size,
            dtype=torch.float32,
            device=device,
        ).view(batch_size, 6)
        if do_cfg:
            image_meta_size = image_meta_size.repeat(2, 1)

        # Style embedding (0 = natural photo, per HunyuanDiT convention)
        style = torch.zeros(batch_size, dtype=torch.int64, device=device)
        if do_cfg:
            style = style.repeat(2)

        # RoPE embeddings for the latent grid
        image_rotary_emb = self._get_image_rotary_emb(
            patch_size=2,
            vae_scale_factor=self.vae_scale_factor,
            height=height,
            width=width,
            device=device,
            dtype=self.dtype,
        )

        # Scheduler timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # Denoise loop
        timer.mark_denoise_start()
        logger.info("Denoising (%d steps)...", len(timesteps))

        for _, t in enumerate(timesteps):
            lat_in = torch.cat([latents] * 2) if do_cfg else latents

            noise_pred = self.transformer(
                hidden_states=lat_in,
                timestep=t.expand(lat_in.shape[0]),
                encoder_hidden_states=clip_embeds,
                text_embedding_mask=clip_mask,
                encoder_hidden_states_t5=t5_embeds,
                text_embedding_mask_t5=t5_mask,
                image_meta_size=image_meta_size,
                style=style,
                image_rotary_emb=image_rotary_emb,
                return_dict=False,
            )[0]

            if do_cfg:
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_cond - noise_pred_uncond
                )

            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        timer.mark_post_start()
        logger.info("Decoding...")
        image = self._decode_latents(latents)

        if getattr(self, "rank", 0) == 0:
            logger.info("Pipeline total: %.2fs", time.time() - pipeline_start)

        timer.mark_end()
        return timer.fill(PipelineOutput(image=image))
