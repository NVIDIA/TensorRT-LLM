# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FLUX.2 Pipeline implementation following WAN pattern.

This pipeline uses the TRT-LLM FLUX.2 transformer implementation.
FLUX.2 uses Mistral3-based text encoder with multi-layer extraction (not CLIP + T5 like FLUX.1).

Key differences from FLUX.1:
- Text encoder: Mistral3 (decoder-only, used as encoder via hidden state extraction)
- Multi-layer fusion: Layers 10, 20, 30 are stacked -> 3 x 5120 = 15360 dim
- No pooled embeddings: Guidance is handled via timestep embedding only
- 4-axis RoPE: (32, 32, 32, 32) instead of 3-axis
"""

import os
import time
from typing import List, Optional, Tuple

import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.models.autoencoders.autoencoder_kl_flux2 import AutoencoderKLFlux2
from diffusers.utils.torch_utils import randn_tensor
from transformers import AutoProcessor, Mistral3ForConditionalGeneration

from tensorrt_llm._torch.visual_gen.config import PipelineComponent
from tensorrt_llm._torch.visual_gen.output import MediaOutput
from tensorrt_llm._torch.visual_gen.pipeline import BasePipeline
from tensorrt_llm._torch.visual_gen.pipeline_registry import register_pipeline
from tensorrt_llm._torch.visual_gen.teacache import ExtractorConfig, register_extractor_from_config
from tensorrt_llm.logger import logger

from .transformer_flux2 import Flux2Transformer2DModel

# TeaCache coefficients for FLUX.2
FLUX2_TEACACHE_COEFFICIENTS = {
    "dev": {
        "ret_steps": [2.57151496e05, -3.54229917e04, 1.40286849e03, -1.35890334e01, 1.32517977e-01],
        "standard": [2.57151496e05, -3.54229917e04, 1.40286849e03, -1.35890334e01, 1.32517977e-01],
    },
}

# System message for Mistral3 chat template (matches HF diffusers exactly)
SYSTEM_MESSAGE = (
    "You are an AI that reasons about image descriptions. You give structured "
    "responses focusing on object relationships, object\nattribution and actions "
    "without speculation."
)


def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    """Compute empirical mu for scheduler shift (matches HF diffusers exactly)."""
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666

    if image_seq_len > 4300:
        return float(a2 * image_seq_len + b2)

    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1

    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    return float(a * num_steps + b)


def format_input(prompts: List[str], system_message: str) -> List[List[dict]]:
    """Format prompts for Mistral3 chat template (PixtralProcessor format)."""
    return [
        [
            {"role": "system", "content": [{"type": "text", "text": system_message}]},
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ]
        for prompt in prompts
    ]


@register_pipeline("Flux2Pipeline")
class Flux2Pipeline(BasePipeline):
    """FLUX.2 Text-to-Image Pipeline (35B params).

    Uses Mistral3 for text encoding and native Flux2Transformer2DModel.
    Follows WAN pipeline pattern for DiffusionModelLoader integration.
    """

    # Layers to extract from Mistral3 for multi-layer fusion
    HIDDEN_STATE_LAYERS: Tuple[int, ...] = (10, 20, 30)

    @staticmethod
    def _compute_flux2_timestep_embedding(module, timestep, guidance=None):
        """Compute timestep embedding for FLUX.2 transformer.

        FLUX.2 uses time_guidance_embed (timestep + guidance combined).

        Args:
            module: Flux2Transformer2DModel instance
            timestep: Timestep tensor [B]
            guidance: Guidance scale tensor [B]

        Returns:
            Timestep + guidance embedding for TeaCache distance calculation
        """
        # Cast to embedder's dtype (avoid int8 quantized layers)
        te_dtype = next(module.time_guidance_embed.timestep_embedder.linear_1.parameters()).dtype
        if te_dtype != torch.int8:
            t = timestep.to(te_dtype)
            g = guidance.to(te_dtype) if guidance is not None else None
        else:
            t = timestep
            g = guidance
        return module.time_guidance_embed(t, g)

    @property
    def dtype(self):
        return self.model_config.torch_dtype

    @property
    def device(self):
        if self.transformer is not None:
            return next(self.transformer.parameters()).device
        return torch.device("cuda:0")

    def _init_transformer(self) -> None:
        """Initialize FLUX.2 transformer with quantization support."""
        logger.info("Creating FLUX.2 transformer with quantization support...")
        self.transformer = Flux2Transformer2DModel(model_config=self.model_config)

    def load_standard_components(
        self,
        checkpoint_dir: str,
        device: torch.device,
        skip_components: Optional[list] = None,
    ) -> None:
        """Load VAE, text encoder (Mistral3), tokenizer, and scheduler from checkpoint."""
        skip_components = skip_components or []

        # Tokenizer (PixtralProcessor for Mistral3)
        # Use full path instead of subfolder to avoid AutoConfig trying to read root config.json
        if PipelineComponent.TOKENIZER not in skip_components:
            logger.info("Loading tokenizer (PixtralProcessor)...")
            tokenizer_path = os.path.join(checkpoint_dir, PipelineComponent.TOKENIZER)
            self.tokenizer = AutoProcessor.from_pretrained(tokenizer_path)

        # Text encoder (Mistral3)
        # Use full path to avoid AutoConfig issues
        if PipelineComponent.TEXT_ENCODER not in skip_components:
            logger.info("Loading text encoder (Mistral3)...")
            text_encoder_path = os.path.join(checkpoint_dir, PipelineComponent.TEXT_ENCODER)
            self.text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
                text_encoder_path,
                torch_dtype=self.model_config.torch_dtype,
            ).to(device)

        # VAE (FLUX.2-specific VAE with BatchNorm)
        # Use full path to avoid AutoConfig issues
        if PipelineComponent.VAE not in skip_components:
            logger.info("Loading VAE...")
            vae_path = os.path.join(checkpoint_dir, PipelineComponent.VAE)
            self.vae = AutoencoderKLFlux2.from_pretrained(vae_path, torch_dtype=torch.bfloat16).to(
                device
            )

            self.vae_scale_factor = 8  # FLUX.2 uses scale_factor=8

        # Scheduler
        if PipelineComponent.SCHEDULER not in skip_components:
            logger.info("Loading scheduler...")
            scheduler_path = os.path.join(checkpoint_dir, PipelineComponent.SCHEDULER)
            self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(scheduler_path)

        # FLUX.2 config
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
            # Register TeaCache extractor for FLUX.2 (must be after device placement)
            register_extractor_from_config(
                ExtractorConfig(
                    model_class_name="Flux2Transformer2DModel",
                    timestep_embed_fn=self._compute_flux2_timestep_embedding,
                    guidance_param_name="guidance",
                    forward_params=[
                        "hidden_states",
                        "encoder_hidden_states",
                        "timestep",
                        "img_ids",
                        "txt_ids",
                        "guidance",
                        "return_dict",
                    ],
                    return_dict_default=False,
                )
            )

            # Enable TeaCache with FLUX.2-specific coefficients
            self._setup_teacache(self.transformer, coefficients=FLUX2_TEACACHE_COEFFICIENTS)

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

    @torch.no_grad()
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
            num_inference_steps: Number of denoising steps
            guidance_scale: Embedded guidance scale
            seed: Random seed for reproducibility
            max_sequence_length: Maximum text sequence length

        Returns:
            Dict with "image" key containing PIL.Image
        """
        pipeline_start = time.time()
        generator = torch.Generator(device=self.device).manual_seed(seed)

        # Encode prompt using Mistral3 multi-layer extraction
        logger.info("Encoding prompt...")
        encode_start = time.time()
        prompt_embeds, text_ids = self._encode_prompt(prompt, max_sequence_length)
        logger.info(f"Prompt encoding completed in {time.time() - encode_start:.2f}s")

        # Prepare latents
        latents, latent_ids = self._prepare_latents(height, width, generator)
        logger.info(f"Latents shape: {latents.shape}")

        # Prepare timesteps with dynamic shifting
        image_seq_len = latents.shape[1]
        mu = compute_empirical_mu(image_seq_len, num_inference_steps)

        # Check if scheduler uses dynamic shifting
        if (
            hasattr(self.scheduler.config, "use_dynamic_shifting")
            and self.scheduler.config.use_dynamic_shifting
        ):
            self.scheduler.set_timesteps(num_inference_steps, device=self.device, mu=mu)
        else:
            self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps

        # Prepare guidance (embedded for FLUX.2)
        guidance = torch.full(
            [latents.shape[0]], guidance_scale, device=self.device, dtype=torch.float32
        )

        # Denoising loop using forward_fn callback (WAN pattern)
        def forward_fn(
            latents, extra_stream_latents, timestep, encoder_hidden_states, extra_tensors
        ):
            """Forward function for FLUX.2 transformer."""
            return self.transformer(
                hidden_states=latents,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep / 1000,  # FLUX.2 expects normalized timesteps
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
        image = self.decode_latents(latents, lambda lat: self._decode_latents(lat, latent_ids))

        if self.rank == 0:
            logger.info(f"Image decoded in {time.time() - decode_start:.2f}s")
            logger.info(f"Total pipeline time: {time.time() - pipeline_start:.2f}s")

        return MediaOutput(image=image)

    def __call__(self, *args, **kwargs):
        """Backward compatibility wrapper."""
        return self.forward(*args, **kwargs)

    def _encode_prompt(
        self,
        prompt: str,
        max_sequence_length: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode prompt using Mistral3 multi-layer extraction.

        FLUX.2 uses:
        1. Format prompts with system message using chat template
        2. Run through Mistral3 with output_hidden_states=True
        3. Extract hidden states from layers 10, 20, 30
        4. Stack and reshape: [B, 3, seq, 5120] -> [B, seq, 15360]

        Returns:
            Tuple of (prompt_embeds, text_ids)
        """
        prompt = [prompt] if isinstance(prompt, str) else prompt

        # Format prompts with system message (PixtralProcessor format)
        messages_batch = format_input(prompt, SYSTEM_MESSAGE)

        # Tokenize using chat template
        inputs = self.tokenizer.apply_chat_template(
            messages_batch,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_sequence_length,
        )

        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        # Forward pass - extract hidden states
        outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

        # Multi-layer extraction: stack layers 10, 20, 30
        stacked = torch.stack(
            [outputs.hidden_states[k] for k in self.HIDDEN_STATE_LAYERS], dim=1
        )  # [B, 3, seq_len, 5120]

        stacked = stacked.to(dtype=self.dtype, device=self.device)

        # Reshape: [B, 3, seq, 5120] -> [B, seq, 15360]
        batch_size, num_layers, seq_len, hidden_dim = stacked.shape
        prompt_embeds = stacked.permute(0, 2, 1, 3).reshape(
            batch_size, seq_len, num_layers * hidden_dim
        )

        # NOTE: Do NOT zero out padding tokens for Mistral3 (decoder-only).
        # Unlike encoder models (T5/CLIP in WAN), Mistral3's causal attention
        # means hidden states at padding positions carry meaningful context
        # from earlier tokens. Zeroing them destroys the embedding information.
        # (HF diffusers does not zero out either.)

        # Prepare 4-axis text IDs for FLUX.2 RoPE
        text_ids = self._prepare_text_ids(prompt_embeds)
        text_ids = text_ids.to(self.device)

        return prompt_embeds, text_ids

    def _prepare_text_ids(self, text_embeds: torch.Tensor) -> torch.Tensor:
        """Prepare 4-axis position IDs for text tokens.

        FLUX.2 uses 4-axis: (t, h, w, l) where text has t=h=w=0, l=position.
        Returns 2D tensor [seq_len, 4] (unbatched, like FLUX.1).
        """
        batch_size, seq_len, _ = text_embeds.shape

        l_ids = torch.arange(seq_len, device=text_embeds.device)
        text_ids = torch.stack(
            [
                torch.zeros(seq_len, device=text_embeds.device),  # t = 0
                torch.zeros(seq_len, device=text_embeds.device),  # h = 0
                torch.zeros(seq_len, device=text_embeds.device),  # w = 0
                l_ids.float(),  # l = position
            ],
            dim=-1,
        )

        return text_ids  # [seq_len, 4]

    def _prepare_latent_ids(self, height: int, width: int) -> torch.Tensor:
        """Prepare 4-axis position IDs for latent patches.

        FLUX.2 uses 4-axis: (t, h, w, l) where image has t=0, l=0, h=row, w=col.
        Returns 2D tensor [seq_len, 4] (unbatched, like FLUX.1).
        """
        latent_height = height // self.vae_scale_factor // 2  # Account for packing
        latent_width = width // self.vae_scale_factor // 2

        t_dim = torch.arange(1, device=self.device)  # [0]
        h_dim = torch.arange(latent_height, device=self.device)
        w_dim = torch.arange(latent_width, device=self.device)
        l_dim = torch.arange(1, device=self.device)  # [0]

        latent_ids = torch.cartesian_prod(t_dim, h_dim, w_dim, l_dim).float()

        return latent_ids  # [seq_len, 4]

    def _prepare_latents(
        self,
        height: int,
        width: int,
        generator: torch.Generator,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare random latents in FLUX.2 packed format and position IDs."""
        # FLUX.2: in_channels=128, VAE scale=8, 2x2 packing
        latent_height = 2 * (height // (self.vae_scale_factor * 2))
        latent_width = 2 * (width // (self.vae_scale_factor * 2))

        in_channels = self.transformer.config.in_channels  # 128

        # Create 4D latents then pack (matches HF for seed reproducibility)
        latent_shape = (1, in_channels, latent_height // 2, latent_width // 2)
        latents_4d = randn_tensor(
            latent_shape, generator=generator, device=self.device, dtype=self.dtype
        )

        # Pack latents: [B, C, H, W] -> [B, H*W, C]
        latents = self._pack_latents(latents_4d)

        # Prepare position IDs
        latent_ids = self._prepare_latent_ids(height, width)

        return latents, latent_ids

    def _pack_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Pack latents: [B, C, H, W] -> [B, H*W, C]"""
        batch_size, num_channels, height, width = latents.shape
        latents = latents.reshape(batch_size, num_channels, height * width)
        latents = latents.permute(0, 2, 1)  # [B, H*W, C]
        return latents

    def _unpack_latents_with_ids(
        self,
        latents: torch.Tensor,
        latent_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Unpack latents using position IDs to scatter tokens into spatial positions."""
        _, ch = latents[0].shape

        h_ids = latent_ids[:, 1].to(torch.int64)
        w_ids = latent_ids[:, 2].to(torch.int64)

        h = torch.max(h_ids) + 1
        w = torch.max(w_ids) + 1

        flat_ids = h_ids * w + w_ids

        x_list = []
        for data in latents:
            out = torch.zeros((h * w, ch), device=data.device, dtype=data.dtype)
            out.scatter_(0, flat_ids.unsqueeze(1).expand(-1, ch), data)
            out = out.view(h, w, ch).permute(2, 0, 1)
            x_list.append(out)

        return torch.stack(x_list, dim=0)

    def _unpatchify_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Unpatchify latents: [B, 128, H, W] -> [B, 32, H*2, W*2]"""
        batch_size, num_channels, height, width = latents.shape

        # 128 channels = 32 * 2 * 2 (2x2 patches)
        latents = latents.reshape(batch_size, num_channels // 4, 2, 2, height, width)
        latents = latents.permute(0, 1, 4, 2, 5, 3)  # [B, 32, H, 2, W, 2]
        latents = latents.reshape(batch_size, num_channels // 4, height * 2, width * 2)

        return latents

    def _decode_latents(self, latents: torch.Tensor, latent_ids: torch.Tensor) -> torch.Tensor:
        """Decode latents to image tensor."""
        # Unpack latents using position IDs
        latents = self._unpack_latents_with_ids(latents, latent_ids)

        # BatchNorm denormalization (critical for FLUX.2 VAE!)
        if hasattr(self.vae, "bn") and hasattr(self.vae.bn, "running_mean"):
            bn_eps = getattr(self.vae.config, "batch_norm_eps", 1e-5)
            latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents_bn_std = torch.sqrt(self.vae.bn.running_var.view(1, -1, 1, 1) + bn_eps).to(
                latents.device, latents.dtype
            )
            latents = latents * latents_bn_std + latents_bn_mean

        # Unpatchify
        latents = self._unpatchify_latents(latents)

        # VAE decode
        latents = latents.to(self.vae.dtype)
        image = self.vae.decode(latents, return_dict=False)[0]

        # Post-process to tensor (H, W, C) uint8
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
        image = (image * 255).round().to(torch.uint8)

        return image[0]  # Remove batch dimension
