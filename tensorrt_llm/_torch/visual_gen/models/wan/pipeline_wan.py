import time
from typing import Optional

import torch
from diffusers import AutoencoderKLWan, FlowMatchEulerDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from transformers import AutoTokenizer, UMT5EncoderModel

from tensorrt_llm._torch.visual_gen.config import PipelineComponent
from tensorrt_llm._torch.visual_gen.output import MediaOutput
from tensorrt_llm._torch.visual_gen.pipeline import BasePipeline
from tensorrt_llm._torch.visual_gen.pipeline_registry import register_pipeline
from tensorrt_llm._torch.visual_gen.teacache import ExtractorConfig, register_extractor_from_config
from tensorrt_llm._torch.visual_gen.utils import postprocess_video_tensor
from tensorrt_llm.logger import logger

from .transformer_wan import WanTransformer3DModel

# Supported Wan T2V models:
# - Wan2.1-T2V-14B: Single-stage text-to-video (14B parameters)
# - Wan2.1-T2V-1.3B: Single-stage text-to-video (1.3B parameters)
# - Wan2.2-T2V-A14B: Two-stage text-to-video (14B, boundary_ratio for high/low-noise stages; supports 480P & 720P)

WAN_TEACACHE_COEFFICIENTS = {
    "1.3B": {
        "ret_steps": [
            -5.21862437e04,
            9.23041404e03,
            -5.28275948e02,
            1.36987616e01,
            -4.99875664e-02,
        ],
        "standard": [2.39676752e03, -1.31110545e03, 2.01331979e02, -8.29855975e00, 1.37887774e-01],
    },
    "14B": {
        "ret_steps": [
            -3.03318725e05,
            4.90537029e04,
            -2.65530556e03,
            5.87365115e01,
            -3.15583525e-01,
        ],
        "standard": [-5784.54975374, 5449.50911966, -1811.16591783, 256.27178429, -13.02252404],
    },
}


# Default negative prompt for Wan T2V models
WAN_DEFAULT_NEGATIVE_PROMPT = (
    "Vibrant colors, overexposed, static, blurry details, subtitles, style, artwork, painting, image, "
    "still image, overall grayish tone, worst quality, low quality, JPEG compression artifacts, ugly, "
    "incomplete, extra fingers, poorly drawn hands, poorly drawn face, deformed, disfigured, malformed limbs, "
    "fused fingers, motionless image, cluttered background, three legs, many people in the background, walking backward"
)


@register_pipeline("WanPipeline")
class WanPipeline(BasePipeline):
    def __init__(self, model_config):
        # Wan2.2 two-stage denoising parameters
        self.transformer_2 = None
        self.boundary_ratio = getattr(model_config.pretrained_config, "boundary_ratio", None)
        self.is_wan22 = self.boundary_ratio is not None

        super().__init__(model_config)

    @staticmethod
    def _compute_wan_timestep_embedding(module, timestep, guidance=None):
        """Compute timestep embedding for WAN transformer.

        WAN uses a condition_embedder with timesteps_proj and time_embedder layers.
        Handles dtype casting to match the embedder's dtype.

        Args:
            module: WanTransformer3DModel instance
            timestep: Timestep tensor (shape: [batch_size])
            guidance: Unused for WAN (no guidance embedding)

        Returns:
            Timestep embedding tensor used by TeaCache for distance calculation
        """
        ce = module.condition_embedder
        t_freq = ce.timesteps_proj(timestep)

        # Cast to embedder's dtype (avoid int8 quantized layers)
        te_dtype = next(iter(ce.time_embedder.parameters())).dtype
        if t_freq.dtype != te_dtype and te_dtype != torch.int8:
            t_freq = t_freq.to(te_dtype)

        return ce.time_embedder(t_freq)

    @property
    def dtype(self):
        return self.model_config.torch_dtype

    @property
    def device(self):
        return self.transformer.device

    @property
    def transformer_components(self) -> list:
        """Return list of transformer components this pipeline needs."""
        if self.transformer_2 is not None:
            return ["transformer", "transformer_2"]
        return ["transformer"]

    def _init_transformer(self) -> None:
        logger.info("Creating WAN transformer with quantization support...")
        self.transformer = WanTransformer3DModel(model_config=self.model_config)

        # Wan2.2: create second transformer for two-stage denoising
        if self.boundary_ratio is not None:
            logger.info("Creating second transformer for Wan2.2 two-stage denoising...")
            self.transformer_2 = WanTransformer3DModel(model_config=self.model_config)

    def load_standard_components(
        self,
        checkpoint_dir: str,
        device: torch.device,
        skip_components: Optional[list] = None,
    ) -> None:
        """Load VAE, text encoder, tokenizer, and scheduler from checkpoint."""
        skip_components = skip_components or []

        if self.transformer_2 is not None and self.boundary_ratio is None:
            raise RuntimeError(
                "transformer_2 exists but boundary_ratio is not set. "
                "This indicates an inconsistent pipeline configuration."
            )

        # Detect model version
        if self.is_wan22:
            logger.info("Detected Wan 2.2 T2V (two-stage denoising)")
        else:
            logger.info("Detected Wan 2.1 T2V (single-stage denoising)")

        # Set default VAE scale factors (will be overridden if VAE is loaded)
        self.vae_scale_factor_temporal = 4
        self.vae_scale_factor_spatial = 8

        if PipelineComponent.TOKENIZER not in skip_components:
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                checkpoint_dir,
                subfolder=PipelineComponent.TOKENIZER,
            )

        if PipelineComponent.TEXT_ENCODER not in skip_components:
            logger.info("Loading text encoder...")
            self.text_encoder = UMT5EncoderModel.from_pretrained(
                checkpoint_dir,
                subfolder=PipelineComponent.TEXT_ENCODER,
                torch_dtype=self.model_config.torch_dtype,
            ).to(device)

        if PipelineComponent.VAE not in skip_components:
            logger.info("Loading VAE...")
            self.vae = AutoencoderKLWan.from_pretrained(
                checkpoint_dir,
                subfolder=PipelineComponent.VAE,
                torch_dtype=torch.bfloat16,  # load VAE in BF16 for memory saving
            ).to(device)

            self.vae_scale_factor_temporal = getattr(self.vae.config, "scale_factor_temporal", 4)
            self.vae_scale_factor_spatial = getattr(self.vae.config, "scale_factor_spatial", 8)

        if PipelineComponent.SCHEDULER not in skip_components:
            logger.info("Loading scheduler...")
            self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                checkpoint_dir,
                subfolder=PipelineComponent.SCHEDULER,
            )
            if not hasattr(self.scheduler.config, "shift") or self.scheduler.config.shift == 1.0:
                self.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
                    self.scheduler.config,
                    shift=5.0,
                )

        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

    def load_weights(self, weights: dict) -> None:
        # Store weights for later use (in case transformer_2 is created after this call)
        self._weights_dict = weights

        has_separate_weights = "transformer" in weights and "transformer_2" in weights

        if self.transformer is not None and hasattr(self.transformer, "load_weights"):
            logger.info("Loading transformer weights...")
            transformer_weights = weights.get("transformer", weights)
            self.transformer.load_weights(transformer_weights)
            logger.info("Transformer weights loaded successfully.")

        # Wan2.2: Load weights for second transformer if it exists
        if self.transformer_2 is not None and hasattr(self.transformer_2, "load_weights"):
            logger.info("Loading transformer_2 weights for Wan2.2...")
            if not has_separate_weights:
                raise ValueError(
                    "Wan2.2 model requires separate 'transformer' and 'transformer_2' weights in checkpoint, "
                    f"but only found: {list(weights.keys())}. "
                    "Two-stage denoising requires distinct weights for high-noise and low-noise transformers."
                )
            transformer_2_weights = weights["transformer_2"]
            self.transformer_2.load_weights(transformer_2_weights)
            logger.info("Transformer_2 weights loaded successfully.")

        # Cache the target dtype from model config (default: bfloat16)
        self._target_dtype = self.model_config.torch_dtype

        # Set model to eval mode
        if self.transformer is not None:
            self.transformer.eval()
        if self.transformer_2 is not None:
            self.transformer_2.eval()

    def post_load_weights(self) -> None:
        super().post_load_weights()  # Calls transformer.post_load_weights() for FP8 scale transformations
        if self.transformer is not None:
            # Register TeaCache extractor for this model type
            # Tells TeaCache how to compute timestep embeddings for Wan
            register_extractor_from_config(
                ExtractorConfig(
                    model_class_name="WanTransformer3DModel",
                    timestep_embed_fn=self._compute_wan_timestep_embedding,
                    return_dict_default=False,  # Wan returns raw tensors, not wrapped outputs
                )
            )

            # Enable TeaCache optimization with WAN-specific coefficients
            self._setup_teacache(self.transformer, coefficients=WAN_TEACACHE_COEFFICIENTS)
            # Save transformer backend before it gets overwritten
            self.transformer_cache_backend = self.cache_backend

        # Wan2.2: Setup TeaCache for second transformer (low-noise stage)
        if self.transformer_2 is not None:
            if hasattr(self.transformer_2, "post_load_weights"):
                self.transformer_2.post_load_weights()

            # Enable TeaCache for low-noise stage with same coefficients
            self._setup_teacache(self.transformer_2, coefficients=WAN_TEACACHE_COEFFICIENTS)
            # Save transformer_2 backend
            self.transformer_2_cache_backend = self.cache_backend

    def infer(self, req):
        """Run inference with request parameters."""
        return self.forward(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            height=req.height,
            width=req.width,
            num_frames=req.num_frames,
            num_inference_steps=req.num_inference_steps,
            guidance_scale=req.guidance_scale,
            guidance_scale_2=req.guidance_scale_2,
            boundary_ratio=req.boundary_ratio,
            seed=req.seed,
            max_sequence_length=req.max_sequence_length,
        )

    @torch.no_grad()
    def forward(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        height: int = 720,
        width: int = 1280,
        num_frames: int = 81,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        guidance_scale_2: Optional[float] = None,
        boundary_ratio: Optional[float] = None,
        seed: int = 42,
        max_sequence_length: int = 226,
    ):
        pipeline_start = time.time()
        generator = torch.Generator(device=self.device).manual_seed(seed)

        # Use user-provided boundary_ratio if given, otherwise fall back to model config
        boundary_ratio = boundary_ratio if boundary_ratio is not None else self.boundary_ratio

        # Validate that Wan 2.2 models have boundary_ratio set
        if self.transformer_2 is not None and boundary_ratio is None:
            raise ValueError(
                "Wan 2.2 models require boundary_ratio to be set. "
                "boundary_ratio was not found in model config. "
                "Please pass boundary_ratio as a parameter."
            )

        # Set default negative prompt if not provided
        if negative_prompt is None:
            negative_prompt = WAN_DEFAULT_NEGATIVE_PROMPT

        # Set model-specific defaults based on Wan version
        logger.info(
            f"Running {'Wan 2.2' if self.is_wan22 else 'Wan 2.1'} T2V inference"
            f"(boundary_ratio={boundary_ratio}, has_transformer_2={self.transformer_2 is not None})"
        )

        if num_inference_steps is None:
            num_inference_steps = 40 if self.is_wan22 else 50

        if guidance_scale is None:
            guidance_scale = 4.0 if self.is_wan22 else 5.0

        if self.is_wan22 and guidance_scale_2 is None:
            guidance_scale_2 = 3.0

        # Validate two-stage denoising configuration
        if guidance_scale_2 is not None and boundary_ratio is None:
            logger.warning(
                "guidance_scale_2 is specified but boundary_ratio is None. "
                "guidance_scale_2 will be ignored."
                "Set boundary_ratio in config or pass as parameter to enable two-stage denoising."
            )
            guidance_scale_2 = None

        # Encode Prompt
        logger.info("Encoding prompts...")
        encode_start = time.time()
        prompt_embeds, neg_prompt_embeds = self._encode_prompt(
            prompt, negative_prompt, max_sequence_length
        )
        logger.info(f"Prompt encoding completed in {time.time() - encode_start:.2f}s")

        # Prepare Latents
        latents = self._prepare_latents(height, width, num_frames, generator)
        logger.info(f"Latents shape: {latents.shape}")

        self.scheduler.set_timesteps(num_inference_steps, device=self.device)

        # Wan2.2: Calculate boundary timestep for two-stage denoising
        boundary_timestep = None
        if boundary_ratio is not None and self.transformer_2 is not None:
            boundary_timestep = boundary_ratio * self.scheduler.config.num_train_timesteps
            logger.info(
                f"Wan2.2 two-stage denoising: boundary_timestep={boundary_timestep:.1f}, "
                f"guidance_scale={guidance_scale}, guidance_scale_2={guidance_scale_2}"
            )

        # Denoising with two-stage support
        # Track which model was used in last step (for logging model transitions)
        last_model_used = [None]

        def forward_fn(
            latents, extra_stream_latents, timestep, encoder_hidden_states, extra_tensors
        ):
            """Forward function for Wan transformer with two-stage support.

            extra_stream_latents and extra_tensors are unused for WAN (single stream, no additional embeddings).
            """
            # Select model based on timestep (if two-stage denoising is enabled)
            if boundary_timestep is not None and self.transformer_2 is not None:
                # Extract scalar timestep for comparison
                current_t = timestep if timestep.dim() == 0 else timestep[0]
                if current_t >= boundary_timestep:
                    current_model = self.transformer
                    model_name = "transformer (high-noise)"
                else:
                    current_model = self.transformer_2
                    model_name = "transformer_2 (low-noise)"

                # Log when switching models
                if last_model_used[0] != model_name:
                    if self.rank == 0:
                        logger.info(f"Switched to {model_name} at timestep {current_t:.1f}")
                    last_model_used[0] = model_name
            else:
                current_model = self.transformer

            return current_model(
                hidden_states=latents,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
            )

        # Two-stage denoising: model switching in forward_fn, guidance scale switching in denoise()
        latents = self.denoise(
            latents=latents,
            scheduler=self.scheduler,
            prompt_embeds=prompt_embeds,
            neg_prompt_embeds=neg_prompt_embeds,
            guidance_scale=guidance_scale,
            forward_fn=forward_fn,
            guidance_scale_2=guidance_scale_2,
            boundary_timestep=boundary_timestep,
        )

        # Log TeaCache statistics - show stats for each transformer separately
        if self.rank == 0 and self.model_config.teacache.enable_teacache:
            logger.info("=" * 80)
            logger.info("TeaCache Statistics:")

            # Stats for transformer (high-noise)
            if hasattr(self, "transformer_cache_backend") and self.transformer_cache_backend:
                stats = self.transformer_cache_backend.get_stats()
                total_steps = stats.get("total_steps", 0)
                cache_hits = stats.get("cached_steps", 0)
                cache_misses = stats.get("compute_steps", 0)
                hit_rate = (cache_hits / total_steps * 100) if total_steps > 0 else 0.0

                logger.info("  Transformer (High-Noise):")
                logger.info(f"    Total steps: {total_steps}")
                logger.info(f"    Cache hits: {cache_hits}")
                logger.info(f"    Cache misses: {cache_misses}")
                logger.info(f"    Hit rate: {hit_rate:.1f}%")

            # Stats for transformer_2 (low-noise)
            if hasattr(self, "transformer_2_cache_backend") and self.transformer_2_cache_backend:
                stats = self.transformer_2_cache_backend.get_stats()
                total_steps = stats.get("total_steps", 0)
                cache_hits = stats.get("cached_steps", 0)
                cache_misses = stats.get("compute_steps", 0)
                hit_rate = (cache_hits / total_steps * 100) if total_steps > 0 else 0.0

                logger.info("  Transformer_2 (Low-Noise):")
                logger.info(f"    Total steps: {total_steps}")
                logger.info(f"    Cache hits: {cache_hits}")
                logger.info(f"    Cache misses: {cache_misses}")
                logger.info(f"    Hit rate: {hit_rate:.1f}%")

            logger.info("=" * 80)

        # Decode
        logger.info("Decoding video...")
        decode_start = time.time()
        video = self.decode_latents(latents, self._decode_latents)

        if self.rank == 0:
            logger.info(f"Video decoded in {time.time() - decode_start:.2f}s")
            logger.info(f"Total pipeline time: {time.time() - pipeline_start:.2f}s")

        return MediaOutput(video=video)

    def _encode_prompt(self, prompt, negative_prompt, max_sequence_length):
        prompt = [prompt] if isinstance(prompt, str) else prompt

        def get_embeds(texts):
            text_inputs = self.tokenizer(
                texts,
                padding="max_length",
                max_length=max_sequence_length,
                truncation=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            input_ids = text_inputs.input_ids.to(self.device)
            attention_mask = text_inputs.attention_mask.to(self.device)

            embeds = self.text_encoder(input_ids, attention_mask=attention_mask).last_hidden_state
            embeds = embeds.to(self.dtype)

            # Zero-out padded tokens based on mask
            seq_lens = attention_mask.gt(0).sum(dim=1).long()
            cleaned_embeds = []
            for u, v in zip(embeds, seq_lens):
                real_content = u[:v]
                pad_len = max_sequence_length - real_content.size(0)
                if pad_len > 0:
                    padded = torch.cat(
                        [real_content, real_content.new_zeros(pad_len, real_content.size(1))]
                    )
                else:
                    padded = real_content
                cleaned_embeds.append(padded)

            return torch.stack(cleaned_embeds, dim=0)

        prompt_embeds = get_embeds(prompt)

        if negative_prompt is None:
            negative_prompt = ""

        neg_prompt = [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
        if len(neg_prompt) == 1 and len(prompt) > 1:
            neg_prompt = neg_prompt * len(prompt)

        neg_embeds = get_embeds(neg_prompt)

        return prompt_embeds, neg_embeds

    def _prepare_latents(self, height, width, num_frames, generator):
        num_channels_latents = 16
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1

        shape = (
            1,
            num_channels_latents,
            num_latent_frames,
            height // self.vae_scale_factor_spatial,
            width // self.vae_scale_factor_spatial,
        )
        return randn_tensor(shape, generator=generator, device=self.device, dtype=self.dtype)

    def _decode_latents(self, latents):
        latents = latents.to(self.vae.dtype)

        # Denormalization
        if hasattr(self.vae.config, "latents_mean") and hasattr(self.vae.config, "latents_std"):
            if not hasattr(self, "_latents_mean"):
                self._latents_mean = (
                    torch.tensor(self.vae.config.latents_mean)
                    .view(1, -1, 1, 1, 1)
                    .to(self.device, self.vae.dtype)
                )
                self._latents_std = (
                    torch.tensor(self.vae.config.latents_std)
                    .view(1, -1, 1, 1, 1)
                    .to(self.device, self.vae.dtype)
                )
            latents = (latents * self._latents_std) + self._latents_mean
        else:
            scaling_factor = self.vae.config.get("scaling_factor", 1.0)
            latents = latents / scaling_factor

        # VAE decode: returns (B, C, T, H, W)
        video = self.vae.decode(latents, return_dict=False)[0]

        # Post-process video tensor: (B, C, T, H, W) -> (T, H, W, C) uint8
        video = postprocess_video_tensor(video, remove_batch_dim=True)

        return video
