# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FLUX.1 Pipeline implementation following WAN pattern."""

import time
from typing import Optional, Tuple

import numpy as np
import torch
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

from tensorrt_llm._torch.visual_gen.config import PipelineComponent
from tensorrt_llm._torch.visual_gen.output import MediaOutput
from tensorrt_llm._torch.visual_gen.pipeline import BasePipeline
from tensorrt_llm._torch.visual_gen.pipeline_registry import register_pipeline
from tensorrt_llm._torch.visual_gen.teacache import ExtractorConfig, register_extractor_from_config
from tensorrt_llm.logger import logger

from .transformer_flux import FluxTransformer2DModel

# TeaCache coefficients for FLUX.1 variants
FLUX_TEACACHE_COEFFICIENTS = {
    "dev": {
        "ret_steps": [2.57151496e05, -3.54229917e04, 1.40286849e03, -1.35890334e01, 1.32517977e-01],
        "standard": [2.57151496e05, -3.54229917e04, 1.40286849e03, -1.35890334e01, 1.32517977e-01],
    },
    "schnell": {
        "ret_steps": [1.0, 0.0],  # Schnell is already fast, minimal caching
        "standard": [1.0, 0.0],
    },
}


@register_pipeline("FluxPipeline")
class FluxPipeline(BasePipeline):
    """FLUX.1 Text-to-Image Pipeline.

    Supports FLUX.1-dev (50 steps, guidance) and FLUX.1-schnell (4 steps, no guidance).
    """

    def __init__(self, model_config):
        if model_config.parallel.dit_cfg_size != 1:
            raise ValueError(
                "FluxPipeline does not support CFG parallelism. Please set dit_cfg_size to 1."
            )

        super().__init__(model_config)

    @staticmethod
    def _compute_flux_timestep_embedding(module, timestep, guidance=None):
        """Compute timestep embedding for FLUX transformer.

        FLUX combines timestep and guidance embeddings.

        Args:
            module: FluxTransformer2DModel instance
            timestep: Timestep tensor [B]
            guidance: Guidance scale tensor [B] (optional)

        Returns:
            Combined timestep embedding for TeaCache distance calculation
        """
        # Cast to embedder's dtype (avoid int8 quantized layers)
        te_dtype = next(iter(module.time_text_embed.parameters())).dtype
        if timestep.dtype != te_dtype and te_dtype != torch.int8:
            timestep = timestep.to(te_dtype)

        temb = module.time_text_embed(timestep)

        if module.guidance_embeds and guidance is not None:
            if guidance.dtype != te_dtype and te_dtype != torch.int8:
                guidance = guidance.to(te_dtype)
            temb = temb + module.guidance_embed(guidance)

        return temb

    @property
    def dtype(self):
        return self.model_config.torch_dtype

    @property
    def device(self):
        if self.transformer is not None:
            return next(self.transformer.parameters()).device
        return torch.device("cuda:0")

    @property
    def common_warmup_shapes(self) -> list:
        """Return list of common warmup shapes (height, width, num_frames)."""
        return [(1024, 1024, 1)]

    def _init_transformer(self) -> None:
        """Initialize FLUX transformer with quantization support."""
        logger.info("Creating FLUX transformer with quantization support...")
        self.transformer = FluxTransformer2DModel(model_config=self.model_config)

    def _run_warmup(self, warmup_steps: int) -> None:
        """Run warmup inference to trigger torch.compile and CUDA init."""
        for height, width, _ in self.common_warmup_shapes:
            logger.info(f"Warmup: FLUX.1 {height}x{width}, {warmup_steps} steps")
            with torch.no_grad():
                self.forward(
                    prompt="warmup",
                    height=height,
                    width=width,
                    num_inference_steps=warmup_steps,
                    guidance_scale=3.5,
                    seed=0,
                    max_sequence_length=512,
                )

    def load_standard_components(
        self,
        checkpoint_dir: str,
        device: torch.device,
        skip_components: Optional[list] = None,
    ) -> None:
        """Load VAE, text encoders, tokenizers, and scheduler from checkpoint."""
        skip_components = skip_components or []

        # CLIP tokenizer and text encoder (for pooled embeddings)
        if PipelineComponent.TOKENIZER not in skip_components:
            logger.info("Loading CLIP tokenizer...")
            self.tokenizer = CLIPTokenizer.from_pretrained(
                checkpoint_dir, subfolder=PipelineComponent.TOKENIZER
            )

        if PipelineComponent.TEXT_ENCODER not in skip_components:
            logger.info("Loading CLIP text encoder...")
            self.text_encoder = CLIPTextModel.from_pretrained(
                checkpoint_dir,
                subfolder=PipelineComponent.TEXT_ENCODER,
                torch_dtype=self.model_config.torch_dtype,
            ).to(device)

        # T5 tokenizer and text encoder (for sequence embeddings)
        if PipelineComponent.TOKENIZER_2 not in skip_components:
            logger.info("Loading T5 tokenizer...")
            self.tokenizer_2 = T5TokenizerFast.from_pretrained(
                checkpoint_dir, subfolder=PipelineComponent.TOKENIZER_2
            )

        if PipelineComponent.TEXT_ENCODER_2 not in skip_components:
            logger.info("Loading T5 text encoder...")
            self.text_encoder_2 = T5EncoderModel.from_pretrained(
                checkpoint_dir,
                subfolder=PipelineComponent.TEXT_ENCODER_2,
                torch_dtype=self.model_config.torch_dtype,
            ).to(device)

        # VAE
        if PipelineComponent.VAE not in skip_components:
            logger.info("Loading VAE...")
            self.vae = AutoencoderKL.from_pretrained(
                checkpoint_dir, subfolder=PipelineComponent.VAE, torch_dtype=torch.bfloat16
            ).to(device)

            self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        # Scheduler
        if PipelineComponent.SCHEDULER not in skip_components:
            logger.info("Loading scheduler...")
            self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                checkpoint_dir, subfolder=PipelineComponent.SCHEDULER
            )

        # Default config values
        self.max_sequence_length = 512
        self.default_height = 1024
        self.default_width = 1024

    def load_weights(self, weights: dict) -> None:
        """Load transformer weights."""
        if self.transformer is not None and hasattr(self.transformer, "load_weights"):
            logger.info("Loading transformer weights...")
            transformer_weights = weights.get("transformer", weights)
            self.transformer.load_weights(transformer_weights)
            logger.info("Transformer weights loaded successfully.")

        self._target_dtype = self.model_config.torch_dtype

        if self.transformer is not None:
            self.transformer.eval()

    def post_load_weights(self) -> None:
        """Post-load setup: TeaCache registration."""
        super().post_load_weights()
        if self.transformer is not None:
            # Register TeaCache extractor for FLUX (must be after device placement)
            register_extractor_from_config(
                ExtractorConfig(
                    model_class_name="FluxTransformer2DModel",
                    timestep_embed_fn=self._compute_flux_timestep_embedding,
                    guidance_param_name="guidance",
                    forward_params=[
                        "hidden_states",
                        "encoder_hidden_states",
                        "pooled_projections",
                        "timestep",
                        "img_ids",
                        "txt_ids",
                        "guidance",
                        "return_dict",
                    ],
                    return_dict_default=False,
                )
            )

            # Enable TeaCache with FLUX-specific coefficients
            self._setup_teacache(self.transformer, coefficients=FLUX_TEACACHE_COEFFICIENTS)

    def infer(self, req):
        """Run inference from DiffusionRequest."""
        return self.forward(
            prompt=req.prompt,
            height=req.height,
            width=req.width,
            num_inference_steps=req.num_inference_steps,
            guidance_scale=req.guidance_scale,
            seed=req.seed,
            max_sequence_length=req.max_sequence_length,
        )

    @torch.inference_mode()
    def forward(
        self,
        prompt: str,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 50,
        guidance_scale: float = 3.5,
        seed: int = 42,
        max_sequence_length: int = 512,
    ):
        """Generate image from text prompt.

        Args:
            prompt: Text prompt for image generation
            height: Output image height (default: 1024)
            width: Output image width (default: 1024)
            num_inference_steps: Number of denoising steps (50 for dev, 4 for schnell)
            guidance_scale: Embedded guidance scale (3.5 for dev)
            seed: Random seed for reproducibility
            max_sequence_length: Maximum text sequence length

        Returns:
            MediaOutput with image tensor
        """
        pipeline_start = time.time()
        generator = torch.Generator(device=self.device).manual_seed(seed)

        # Encode prompt
        logger.info("Encoding prompt...")
        encode_start = time.time()
        prompt_embeds, pooled_prompt_embeds, text_ids = self._encode_prompt(
            prompt, max_sequence_length
        )
        logger.info(f"Prompt encoding completed in {time.time() - encode_start:.2f}s")

        # Prepare latents
        latents, latent_ids = self._prepare_latents(height, width, generator)
        logger.info(f"Latents shape: {latents.shape}")

        # Prepare timesteps with dynamic shifting (FLUX uses mu parameter)
        image_seq_len = latents.shape[1]
        mu = self._compute_mu(image_seq_len)

        # Match HF: create sigmas for non-flow mode, or None for flow mode
        use_flow_sigmas = getattr(self.scheduler.config, "use_flow_sigmas", None)
        if use_flow_sigmas:
            # Flow mode: let scheduler compute sigmas internally
            self.scheduler.set_timesteps(num_inference_steps, device=self.device, mu=mu)
        else:
            # Non-flow mode: provide linear sigmas
            sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
            self.scheduler.set_timesteps(sigmas=sigmas, device=self.device, mu=mu)

        timesteps = self.scheduler.timesteps

        # Prepare guidance (embedded guidance for FLUX)
        guidance = None
        if self.transformer.guidance_embeds:
            guidance = torch.full(
                [latents.shape[0]], guidance_scale, device=self.device, dtype=torch.float32
            )

        # Denoising loop
        def forward_fn(
            latents, extra_stream_latents, timestep, encoder_hidden_states, extra_tensors
        ):
            """Forward function for FLUX transformer."""
            return self.transformer(
                hidden_states=latents,
                encoder_hidden_states=encoder_hidden_states,
                pooled_projections=pooled_prompt_embeds,
                timestep=timestep / 1000,  # FLUX expects normalized timesteps
                img_ids=latent_ids,
                txt_ids=text_ids,
                guidance=guidance,
                return_dict=False,
            )[0]

        latents = self.denoise(
            latents=latents,
            scheduler=self.scheduler,
            prompt_embeds=prompt_embeds,
            guidance_scale=1.0,  # No CFG: guidance is embedded
            forward_fn=forward_fn,
            timesteps=timesteps,
        )

        # Decode
        logger.info("Decoding image...")
        decode_start = time.time()
        image = self.decode_latents(latents, lambda lat: self._decode_latents(lat, height, width))

        if self.rank == 0:
            logger.info(f"Image decoded in {time.time() - decode_start:.2f}s")
            logger.info(f"Total pipeline time: {time.time() - pipeline_start:.2f}s")

        return MediaOutput(image=image)

    def _encode_prompt(
        self,
        prompt: str,
        max_sequence_length: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode prompt using CLIP and T5.

        Args:
            prompt: Text prompt
            max_sequence_length: Maximum T5 sequence length

        Returns:
            Tuple of (T5 embeddings, CLIP pooled embeddings, text position IDs)
        """
        prompt = [prompt] if isinstance(prompt, str) else prompt

        # CLIP encoding (pooled embeddings)
        clip_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        clip_input_ids = clip_inputs.input_ids.to(self.device)

        clip_outputs = self.text_encoder(clip_input_ids, output_hidden_states=False)
        pooled_prompt_embeds = clip_outputs.pooler_output.to(self.dtype)

        # T5 encoding (sequence embeddings)
        t5_inputs = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        t5_input_ids = t5_inputs.input_ids.to(self.device)

        # NOTE: HF diffusers does NOT pass attention_mask to T5 for FLUX.1.
        # T5 treats padding tokens as real tokens, and the transformer sees
        # their non-zero embeddings. Matching HF behavior for PSNR parity.
        t5_outputs = self.text_encoder_2(t5_input_ids, output_hidden_states=False)
        prompt_embeds = t5_outputs.last_hidden_state.to(self.dtype)

        # Prepare text position IDs
        text_ids = self._prepare_text_ids(prompt_embeds)
        text_ids = text_ids.to(self.device)

        return prompt_embeds, pooled_prompt_embeds, text_ids

    def _compute_mu(self, image_seq_len: int) -> float:
        """Compute mu parameter for FLUX's dynamic timestep shifting.

        FLUX uses flow matching with dynamic shifting based on image resolution.
        Formula matches HuggingFace diffusers implementation.

        Args:
            image_seq_len: Number of latent patches (packed format)

        Returns:
            mu value for scheduler
        """
        # HuggingFace formula: mu = m * image_seq_len + b
        m = 0.000169271
        b = 0.456666667
        return float(m * image_seq_len + b)

    def _prepare_text_ids(self, text_embeds: torch.Tensor) -> torch.Tensor:
        """Prepare 3D position IDs for text tokens.

        Returns 2D tensor [seq_len, 3] (HF expects unbatched).
        """
        batch_size, seq_len, _ = text_embeds.shape
        text_ids = torch.zeros(seq_len, 3, device=text_embeds.device)
        return text_ids

    def _prepare_latent_ids(self, height: int, width: int) -> torch.Tensor:
        """Prepare 3D position IDs for packed latent patches.

        FLUX uses 2x2 spatial packing, so IDs are for (H/2, W/2) grid.
        HF convention: index 0 unused, index 1 = row, index 2 = col.

        Returns 2D tensor [seq_len, 3] (HF expects unbatched).
        """
        latent_height = height // self.vae_scale_factor
        latent_width = width // self.vae_scale_factor

        # Packed dimensions (2x2 packing)
        packed_h = latent_height // 2
        packed_w = latent_width // 2

        # Create grid with HF's index convention
        latent_ids = torch.zeros(packed_h, packed_w, 3)
        latent_ids[..., 1] = torch.arange(packed_h)[:, None]  # Row index
        latent_ids[..., 2] = torch.arange(packed_w)[None, :]  # Col index

        latent_ids = latent_ids.reshape(-1, 3)
        return latent_ids  # [seq_len, 3]

    def _pack_latents(
        self,
        latents: torch.Tensor,
        batch_size: int,
        num_channels: int,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """Pack latents from VAE spatial format to FLUX sequence format.

        FLUX uses 2x2 spatial packing:
        VAE format: [B, 16, H, W] -> FLUX format: [B, (H/2)*(W/2), 64]
        """
        # [B, C, H, W] -> [B, C, H/2, 2, W/2, 2]
        latents = latents.view(batch_size, num_channels, height // 2, 2, width // 2, 2)
        # -> [B, H/2, W/2, C, 2, 2]
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        # -> [B, (H/2)*(W/2), C*4]
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels * 4)
        return latents

    def _unpack_latents(self, latents: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """Unpack latents from FLUX sequence format to VAE spatial format.

        FLUX format: [B, (H/2)*(W/2), 64] -> VAE format: [B, 16, H, W]
        """
        batch_size, num_patches, channels = latents.shape
        latent_height = height // self.vae_scale_factor
        latent_width = width // self.vae_scale_factor

        # [B, seq_len, 64] -> [B, H/2, W/2, 16, 2, 2]
        latents = latents.view(
            batch_size, latent_height // 2, latent_width // 2, channels // 4, 2, 2
        )
        # -> [B, 16, H/2, 2, W/2, 2]
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        # -> [B, 16, H, W]
        latents = latents.reshape(batch_size, channels // 4, latent_height, latent_width)
        return latents

    def _prepare_latents(
        self,
        height: int,
        width: int,
        generator: torch.Generator,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare random latents in FLUX packed format and position IDs."""
        latent_height = height // self.vae_scale_factor
        latent_width = width // self.vae_scale_factor

        # Use VAE channels (16), not transformer channels (64)
        # The packing will convert 16 -> 64
        vae_channels = self.vae.config.latent_channels  # 16

        # Create random latents in VAE spatial format [B, 16, H, W]
        shape = (1, vae_channels, latent_height, latent_width)
        latents = randn_tensor(shape, generator=generator, device=self.device, dtype=self.dtype)

        # Prepare position IDs for packed format
        latent_ids = self._prepare_latent_ids(height, width)
        latent_ids = latent_ids.to(self.device)

        # Pack latents to FLUX sequence format [B, seq_len, 64]
        latents = self._pack_latents(latents, 1, vae_channels, latent_height, latent_width)

        return latents, latent_ids

    def _decode_latents(self, latents: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """Decode latents to image tensor."""
        # Unpack latents: (batch, seq_len, channels) -> (batch, channels, h, w)
        latents = self._unpack_latents(latents, height, width)

        # Scale latents (must include shift_factor, matching HF diffusers)
        latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor

        # VAE decode
        latents = latents.to(self.vae.dtype)
        image = self.vae.decode(latents, return_dict=False)[0]

        # Post-process to tensor (H, W, C) uint8
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
        image = (image * 255).round().to(torch.uint8)

        return image[0]  # Remove batch dimension
