import json
import os
import time
from typing import List, Optional, Tuple, Union

import diffusers
import PIL.Image
import torch
from diffusers import AutoencoderKLWan, FlowMatchEulerDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from transformers import AutoTokenizer, CLIPImageProcessor, CLIPVisionModel, UMT5EncoderModel

from tensorrt_llm._torch.visual_gen.cache.teacache import (
    ExtractorConfig,
    register_extractor_from_config,
)
from tensorrt_llm._torch.visual_gen.config import PipelineComponent
from tensorrt_llm._torch.visual_gen.models.wan.defaults import (
    get_wan_default_params,
    get_wan_extra_param_specs,
)
from tensorrt_llm._torch.visual_gen.models.wan.pipeline_wan_utils import retrieve_latents
from tensorrt_llm._torch.visual_gen.output import CudaPhaseTimer, PipelineOutput
from tensorrt_llm._torch.visual_gen.pipeline import BasePipeline, ExtraParamSchema
from tensorrt_llm._torch.visual_gen.pipeline_registry import register_pipeline
from tensorrt_llm._torch.visual_gen.utils import postprocess_video_tensor
from tensorrt_llm.logger import logger

# Supported Wan I2V 14B models:
# - Wan2.1-I2V-14B-480P: Single-stage image-to-video
# - Wan2.1-I2V-14B-720P: Single-stage image-to-video
# - Wan2.2-I2V-14B: Two-stage image-to-video (no CLIP, boundary_ratio for two-stage denoising)
from .transformer_wan import WanTransformer3DModel

WAN_I2V_TEACACHE_COEFFICIENTS = {
    # Wan 2.1 I2V 14B 480P
    "480P": {
        "ret_steps": [
            2.57151496e05,
            -3.54229917e04,
            1.40286849e03,
            -1.35890334e01,
            1.32517977e-01,
        ],
        "standard": [
            -3.02331670e02,
            2.23948934e02,
            -5.25463970e01,
            5.87348440e00,
            -2.01973289e-01,
        ],
    },
    # Wan 2.1 I2V 14B 720P
    "720P": {
        "ret_steps": [
            8.10705460e03,
            2.13393892e03,
            -3.72934672e02,
            1.66203073e01,
            -4.17769401e-02,
        ],
        "standard": [
            -114.36346466,
            65.26524496,
            -18.82220707,
            4.91518089,
            -0.23412683,
        ],
    },
}

# Default negative prompt for Wan I2V models
WAN_DEFAULT_NEGATIVE_PROMPT = (
    "Vibrant colors, overexposed, static, blurry details, subtitles, style, artwork, painting, image, "
    "still image, overall grayish tone, worst quality, low quality, JPEG compression artifacts, ugly, "
    "incomplete, extra fingers, poorly drawn hands, poorly drawn face, deformed, disfigured, malformed limbs, "
    "fused fingers, motionless image, cluttered background, three legs, many people in the background, walking backward"
)


@register_pipeline("WanImageToVideoPipeline")
class WanImageToVideoPipeline(BasePipeline):
    def __init__(self, model_config):
        # Wan2.2 14B two-stage denoising parameters
        self.transformer_2 = None
        self.boundary_ratio = getattr(model_config.pretrained_config, "boundary_ratio", None)
        self.is_wan22_14b = self.boundary_ratio is not None

        # Validate TeaCache compatibility before allocating GPU memory
        if self.is_wan22_14b and model_config.cache_backend == "teacache":
            raise ValueError(
                "TeaCache is not supported for Wan 2.2 models. "
                "Use cache_backend='none' or 'cache_dit' (not 'teacache')."
            )

        super().__init__(model_config)

    def _compute_wan_timestep_embedding(self, module, timestep=None, **kwargs):
        """Compute timestep embedding for Wan I2V transformer.

        Returns timestep_proj when use_ret_steps=True (matches ret_steps coefficient
        calibration), or temb when use_ret_steps=False (standard mode).
        """
        ce = module.condition_embedder
        t_freq = ce.timesteps_proj(timestep)

        # Cast to embedder's dtype (avoid int8 quantized layers)
        te_dtype = next(iter(ce.time_embedder.parameters())).dtype
        if t_freq.dtype != te_dtype and te_dtype != torch.int8:
            t_freq = t_freq.to(te_dtype)

        t_emb = ce.time_embedder(t_freq)

        teacache = self.model_config.teacache
        if teacache is not None and teacache.use_ret_steps:
            # ret_steps mode: use timestep_proj — what the ret_steps coefficients were calibrated for
            return ce.time_proj(ce.act_fn(t_emb)).to(torch.float32)
        else:
            return t_emb.to(torch.float32)

    @property
    def dtype(self):
        return self.model_config.torch_dtype

    @property
    def device(self):
        return self.transformer.device

    @property
    def transformer_components(self) -> list:
        if self.transformer_2 is not None:
            return ["transformer", "transformer_2"]
        return ["transformer"]

    @property
    def default_warmup_resolutions(self):
        return [(480, 832), (720, 1280)]

    @property
    def default_warmup_num_frames(self):
        return [33, 81]

    @property
    def default_warmup_steps(self):
        return 4 if self.is_wan22_14b else 2

    @property
    def resolution_multiple_of(self):
        patch_size = (
            self.transformer.config.patch_size
            if self.transformer is not None
            else self.transformer_2.config.patch_size
        )
        return (
            self.vae_scale_factor_spatial * patch_size[1],
            self.vae_scale_factor_spatial * patch_size[2],
        )

    def _init_transformer(self) -> None:
        logger.info("Creating WAN I2V transformer with quantization support...")
        self.transformer = WanTransformer3DModel(model_config=self.model_config)

        # Wan2.2: Optionally create second transformer for two-stage denoising
        if self.boundary_ratio is not None:
            logger.info("Creating second transformer for Wan2.2 I2V two-stage denoising...")
            self.transformer_2 = WanTransformer3DModel(model_config=self.model_config)

    def load_standard_components(
        self,
        checkpoint_dir: str,
        device: torch.device,
        skip_components: Optional[list] = None,
    ) -> None:
        """Load VAE, text encoder, tokenizer, scheduler, and I2V-specific components from checkpoint."""
        skip_components = skip_components or []

        # Load boundary_ratio and transformer_2 info from model_index.json (pipeline-level config)
        # Wan 2.2 has both transformer_2 and boundary_ratio, Wan 2.1 doesn't
        model_index_path = os.path.join(checkpoint_dir, "model_index.json")
        has_transformer_2 = False
        if os.path.exists(model_index_path):
            with open(model_index_path) as f:
                model_index = json.load(f)
                # Check for boundary_ratio in model_index
                if "boundary_ratio" in model_index:
                    self.boundary_ratio = model_index["boundary_ratio"]
                    logger.info(f"Found boundary_ratio in model_index.json: {self.boundary_ratio}")
                else:
                    logger.info("No boundary_ratio found in model_index.json")
                # Check for transformer_2 component
                transformer_2_spec = model_index.get("transformer_2", None)
                has_transformer_2 = (
                    transformer_2_spec is not None and transformer_2_spec[0] is not None
                )
                logger.info(f"transformer_2 in model_index.json: {has_transformer_2}")

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
            sched_cfg = FlowMatchEulerDiscreteScheduler.load_config(
                checkpoint_dir, subfolder=PipelineComponent.SCHEDULER
            )
            scheduler_class_name = sched_cfg.get("_class_name", "FlowMatchEulerDiscreteScheduler")
            if not hasattr(diffusers, scheduler_class_name):
                raise ValueError(
                    f"Scheduler class '{scheduler_class_name}' not found in diffusers "
                    f"(checkpoint: {checkpoint_dir}). Check the scheduler config."
                )
            SchedulerClass = getattr(diffusers, scheduler_class_name)
            if issubclass(SchedulerClass, FlowMatchEulerDiscreteScheduler):
                if sched_cfg.get("shift", 1.0) == 1.0:
                    sched_cfg["shift"] = sched_cfg.get("flow_shift") or 5.0
            self.scheduler = SchedulerClass.from_config(sched_cfg)

        if self.transformer_2 is not None and self.boundary_ratio is None:
            raise RuntimeError(
                "transformer_2 exists but boundary_ratio is not set. "
                "This indicates an inconsistent pipeline configuration."
            )

        # Load image encoder and processor (only for Wan 2.1)
        # Wan 2.2: Has both transformer_2 and boundary_ratio (two-stage denoising)
        if self.is_wan22_14b:
            logger.info("Detected Wan 2.2 I2V (two-stage, no CLIP)")
        else:
            logger.info("Detected Wan 2.1 I2V (single-stage, uses CLIP)")

        if PipelineComponent.IMAGE_ENCODER not in skip_components and not self.is_wan22_14b:
            logger.info("Loading CLIP image encoder for I2V conditioning (Wan 2.1 only)...")
            self.image_encoder = CLIPVisionModel.from_pretrained(
                checkpoint_dir,
                subfolder=PipelineComponent.IMAGE_ENCODER,
                torch_dtype=torch.float32,  # Keep CLIP in FP32 for stability
            ).to(device)

        if PipelineComponent.IMAGE_PROCESSOR not in skip_components and not self.is_wan22_14b:
            logger.info("Loading CLIP image processor...")
            self.image_processor = CLIPImageProcessor.from_pretrained(
                checkpoint_dir,
                subfolder=PipelineComponent.IMAGE_PROCESSOR,
            )

        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

    def load_weights(self, weights: dict) -> None:
        # Store weights for later use
        self._weights_dict = weights

        # Check if weights dict has separate transformer/transformer_2 keys (Wan2.2)
        has_separate_weights = "transformer" in weights and "transformer_2" in weights

        if self.transformer is not None and hasattr(self.transformer, "load_weights"):
            logger.info("Loading transformer weights...")
            transformer_weights = weights.get("transformer", weights)
            self.transformer.load_weights(transformer_weights)
            logger.info("Transformer weights loaded successfully.")

        # Wan2.2: Load weights for second transformer if it exists
        if self.transformer_2 is not None and hasattr(self.transformer_2, "load_weights"):
            logger.info("Loading transformer_2 weights for Wan2.2 I2V...")
            if has_separate_weights:
                transformer_2_weights = weights["transformer_2"]
                logger.info("Using separate transformer_2 weights from checkpoint")
            else:
                # For Wan 2.2, transformer_2 weights must exist
                raise ValueError(
                    "Wan2.2 model requires separate 'transformer' and 'transformer_2' weights in checkpoint, "
                    f"but only found: {list(weights.keys())}"
                )
            self.transformer_2.load_weights(transformer_2_weights)
            logger.info("Transformer_2 weights loaded successfully.")

        # Cache the target dtype from model config (default: bfloat16)
        self._target_dtype = self.model_config.torch_dtype

        # Set model to eval mode
        if self.transformer is not None:
            self.transformer.eval()
        if self.transformer_2 is not None:
            self.transformer_2.eval()
        if hasattr(self, "image_encoder") and self.image_encoder is not None:
            self.image_encoder.eval()

    def post_load_weights(self) -> None:
        super().post_load_weights()  # Calls transformer.post_load_weights() for FP8 scale transformations
        if self.transformer is not None:
            if self.model_config.cache_backend == "teacache":
                register_extractor_from_config(
                    ExtractorConfig(
                        model_class_name="WanTransformer3DModel",
                        timestep_embed_fn=self._compute_wan_timestep_embedding,
                        return_dict_default=False,  # Wan returns raw tensors, not wrapped outputs
                    )
                )

            if not self.is_wan22_14b:
                self._setup_cache_acceleration(
                    self.transformer, coefficients=WAN_I2V_TEACACHE_COEFFICIENTS
                )
                self.transformer_cache_backend = self.cache_accelerator
            else:
                if self.model_config.cache_backend == "cache_dit":
                    self._setup_cache_acceleration(self.transformer, coefficients=None)
                self.transformer_cache_backend = self.cache_accelerator

        if self.transformer_2 is not None:
            if hasattr(self.transformer_2, "post_load_weights"):
                self.transformer_2.post_load_weights()

    def _run_warmup(self, height: int, width: int, num_frames: int, steps: int) -> None:
        dummy_image = PIL.Image.new("RGB", (width, height))
        with torch.no_grad():
            self.forward(
                image=dummy_image,
                prompt="warmup",
                negative_prompt="",
                height=height,
                width=width,
                num_frames=num_frames,
                num_inference_steps=steps,
                guidance_scale=5.0,
                seed=42,
                max_sequence_length=512,
            )

    @property
    def default_generation_params(self):
        return get_wan_default_params(
            is_wan22_14b=self.is_wan22_14b,
            name_or_path=getattr(self.config, "_name_or_path", ""),
            num_heads=getattr(self.config, "num_attention_heads", 40),
            include_i2v=True,
        )

    @property
    def extra_param_specs(self):
        specs = get_wan_extra_param_specs(self.is_wan22_14b)
        specs["last_image"] = ExtraParamSchema(
            type="str",
            default=None,
            description="Last frame path for video interpolation (Wan I2V).",
        )
        return specs

    def infer(self, req):
        """Run inference with request parameters."""
        # Extract image from request (can be path, PIL Image, or torch.Tensor)
        if req.params.image is None:
            raise ValueError("I2V pipeline requires 'image' parameter")

        image = req.params.image[0] if isinstance(req.params.image, list) else req.params.image
        extra = req.params.extra_params or {}
        last_image = extra.get("last_image")

        if last_image is not None and isinstance(last_image, list):
            last_image = last_image[0] if last_image else None

        return self.forward(
            image=image,
            prompt=req.prompt,
            negative_prompt=req.params.negative_prompt,
            height=req.params.height,
            width=req.params.width,
            num_frames=req.params.num_frames,
            num_inference_steps=req.params.num_inference_steps,
            guidance_scale=req.params.guidance_scale,
            guidance_scale_2=extra.get("guidance_scale_2"),
            boundary_ratio=extra.get("boundary_ratio"),
            seed=req.params.seed,
            max_sequence_length=req.params.max_sequence_length,
            last_image=last_image,
        )

    @torch.no_grad()
    def forward(
        self,
        image: Union[PIL.Image.Image, torch.Tensor, str],
        prompt: Union[str, List[str]],
        negative_prompt: Optional[str] = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        guidance_scale_2: Optional[float] = None,
        boundary_ratio: Optional[float] = None,
        seed: int = 42,
        max_sequence_length: int = 512,
        last_image: Optional[Union[PIL.Image.Image, torch.Tensor, str]] = None,
    ):
        pipeline_start = time.time()
        timer = CudaPhaseTimer()
        timer.mark_pre_start()

        # Validate image input — only single image is supported for batch generation
        if not isinstance(image, (PIL.Image.Image, torch.Tensor, str)):
            raise ValueError(
                f"`image` must be a PIL.Image, torch.Tensor, or file path string, "
                f"got {type(image)}. Batch of different images is not supported; "
                f"use a single image with multiple prompts instead."
            )

        # Determine batch size
        if isinstance(prompt, str):
            prompt = [prompt]
        batch_size = len(prompt)
        if batch_size > 1:
            logger.info(
                f"Batch generation: {batch_size} prompts with a single input image. "
                "All videos will be conditioned on the same image."
            )

        generator = torch.Generator(device=self.device).manual_seed(seed)

        # Use user-provided boundary_ratio if given, otherwise fall back to model config
        boundary_ratio = boundary_ratio if boundary_ratio is not None else self.boundary_ratio

        if self.transformer_2 is not None and boundary_ratio is None:
            raise ValueError(
                "Wan 2.2 models require boundary_ratio to be set. "
                "boundary_ratio was not found in model config. "
                "Please pass boundary_ratio as a parameter."
            )

        # Set default negative prompt if not provided
        if negative_prompt is None:
            negative_prompt = WAN_DEFAULT_NEGATIVE_PROMPT

        # Set model-specific defaults if not provided
        defaults = self.default_generation_params
        if num_inference_steps is None:
            num_inference_steps = defaults["num_inference_steps"]
        if guidance_scale is None:
            guidance_scale = defaults["guidance_scale"]

        if self.is_wan22_14b and guidance_scale_2 is None:
            guidance_scale_2 = guidance_scale  # Match HF: default to guidance_scale when unset

        # Validate two-stage denoising configuration
        if guidance_scale_2 is not None and boundary_ratio is None:
            logger.warning(
                "guidance_scale_2 is specified but boundary_ratio is None. "
                "guidance_scale_2 will be ignored."
                "Set boundary_ratio in config or pass as parameter to enable two-stage denoising."
            )
            guidance_scale_2 = None

        self.validate_resolution(height, width, num_frames)

        if num_frames % self.vae_scale_factor_temporal != 1:
            logger.warning(
                f"`num_frames - 1` must be divisible by {self.vae_scale_factor_temporal}. "
                f"Rounding {num_frames} to nearest valid value."
            )
            num_frames = (
                num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
            )
        num_frames = max(num_frames, 1)

        # Encode Prompt
        logger.info("Encoding prompts...")
        encode_start = time.time()
        prompt_embeds, neg_prompt_embeds = self._encode_prompt(
            prompt, negative_prompt, max_sequence_length
        )
        logger.info(f"Prompt encoding completed in {time.time() - encode_start:.2f}s")

        # Encode Image (I2V-specific)
        logger.info("Encoding input image...")
        image_encode_start = time.time()

        # Determine model version
        model_version = "Wan 2.2" if self.is_wan22_14b else "Wan 2.1"
        logger.info(
            f"Running {model_version} I2V inference "
            f"(boundary_ratio={boundary_ratio}, has_transformer_2={self.transformer_2 is not None})"
        )

        if not self.is_wan22_14b:
            # Wan 2.1 I2V: Compute CLIP image embeddings
            image_embeds = self._encode_image(image, last_image)
            image_embeds = image_embeds.to(self.dtype)
            # Repeat for batch: single image, multiple prompts
            if batch_size > 1:
                image_embeds = image_embeds.repeat(batch_size, 1, 1)
        else:
            # Wan 2.2 I2V: No image embeddings needed
            image_embeds = None

        logger.info(f"Image encoding completed in {time.time() - image_encode_start:.2f}s")

        # Prepare Latents with image conditioning (I2V-specific)
        latents, condition_data = self._prepare_latents(
            batch_size, image, height, width, num_frames, generator, last_image
        )

        self.scheduler.set_timesteps(num_inference_steps, device=self.device)

        # Wan2.2: Calculate boundary timestep for two-stage denoising
        boundary_timestep = None
        if boundary_ratio is not None and self.transformer_2 is not None:
            boundary_timestep = boundary_ratio * self.scheduler.config.num_train_timesteps
            logger.info(
                f"Wan2.2 I2V two-stage denoising: boundary_timestep={boundary_timestep:.1f}, "
                f"guidance_scale={guidance_scale}, guidance_scale_2={guidance_scale_2}"
            )

        # Denoising with two-stage support
        # Track which model was used in last step (for logging model transitions)
        last_model_used = [None]

        def forward_fn(
            latents_input, extra_stream_latents, timestep, encoder_hidden_states, extra_tensors
        ):
            """Forward function for WAN I2V transformer with two-stage support.

            Both Wan 2.1 and Wan 2.2 14B use concatenation approach: [latents, condition].
            Difference: Wan 2.1 passes image_embeds, Wan 2.2 passes None.
            """
            # Select model based on timestep (if two-stage denoising is enabled)
            if boundary_timestep is not None and self.transformer_2 is not None:
                # Extract scalar timestep for comparison
                current_t = timestep if timestep.dim() == 0 else timestep[0]
                if current_t >= boundary_timestep:
                    current_model = self.transformer
                    model_name = "transformer"
                else:
                    current_model = self.transformer_2
                    model_name = "transformer_2"

                # Log when switching models
                if last_model_used[0] != model_name:
                    if self.rank == 0:
                        logger.info(
                            f"[TRTLLM] Switched to {model_name} at timestep {current_t:.1f}"
                        )
                    last_model_used[0] = model_name
            else:
                current_model = self.transformer

            # Wan 2.1 & Wan 2.2 14B: concatenate latents and condition
            # Handle CFG: duplicate condition if batch dimension doubled
            if latents_input.shape[0] != condition_data.shape[0]:
                condition_to_use = torch.cat([condition_data] * 2)
            else:
                condition_to_use = condition_data

            latent_model_input = torch.cat([latents_input, condition_to_use], dim=1).to(self.dtype)
            timestep_input = timestep.expand(latents_input.shape[0])

            # Forward pass with I2V conditioning
            # Wan 2.1: image_embeds is not None (CLIP embeddings)
            # Wan 2.2 14B: image_embeds is None (no CLIP)
            # Handle CFG: duplicate image_embeds if batch dimension doubled
            image_embeds_to_use = image_embeds
            if (
                image_embeds_to_use is not None
                and latents_input.shape[0] != image_embeds_to_use.shape[0]
            ):
                repeat_factor = latents_input.shape[0] // image_embeds_to_use.shape[0]
                image_embeds_to_use = image_embeds_to_use.repeat(repeat_factor, 1, 1)

            return current_model(
                hidden_states=latent_model_input,
                timestep=timestep_input,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_image=image_embeds_to_use,
            )

        # Two-stage denoising: model switching in forward_fn, guidance scale switching in denoise()
        timer.mark_denoise_start()
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
        timer.mark_post_start()

        # Decode
        logger.info("Decoding video...")
        decode_start = time.time()
        video = self.decode_latents(latents, self._decode_latents)

        if self.rank == 0:
            logger.info(f"Video decoded in {time.time() - decode_start:.2f}s")
            logger.info(f"Total pipeline time: {time.time() - pipeline_start:.2f}s")

        timer.mark_end()
        return timer.fill(PipelineOutput(video=video, frame_rate=16.0))

    def _encode_prompt(self, prompt: List[str], negative_prompt, max_sequence_length):
        """Encode text prompts to embeddings (same as T2V)."""

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

    def _encode_image(
        self,
        image: Union[PIL.Image.Image, torch.Tensor, str],
        last_image: Optional[Union[PIL.Image.Image, torch.Tensor, str]] = None,
    ) -> torch.Tensor:
        """Encode image(s) using CLIP image encoder (Wan 2.1 I2V only)."""
        if isinstance(image, str):
            image = PIL.Image.open(image).convert("RGB")
        if isinstance(last_image, str):
            last_image = PIL.Image.open(last_image).convert("RGB")

        images_to_encode = [image] if last_image is None else [image, last_image]

        image_inputs = self.image_processor(images=images_to_encode, return_tensors="pt").to(
            self.device
        )
        image_embeds = self.image_encoder(**image_inputs, output_hidden_states=True)

        return image_embeds.hidden_states[-2]

    def _prepare_latents(
        self,
        batch_size: int,
        image: Union[PIL.Image.Image, torch.Tensor, str],
        height: int,
        width: int,
        num_frames: int,
        generator: torch.Generator,
        last_image: Optional[Union[PIL.Image.Image, torch.Tensor, str]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare latents with image conditioning for I2V generation."""
        num_channels_latents = 16
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        latent_height = height // self.vae_scale_factor_spatial
        latent_width = width // self.vae_scale_factor_spatial

        shape = (batch_size, num_channels_latents, num_latent_frames, latent_height, latent_width)
        latents = randn_tensor(shape, generator=generator, device=self.device, dtype=self.dtype)

        # Load and preprocess image(s)
        if isinstance(image, str):
            image = PIL.Image.open(image).convert("RGB")
        image = self.video_processor.preprocess(image, height=height, width=width).to(
            self.device, dtype=torch.float32
        )

        if last_image is not None:
            if isinstance(last_image, str):
                last_image = PIL.Image.open(last_image).convert("RGB")
            last_image = self.video_processor.preprocess(last_image, height=height, width=width).to(
                self.device, dtype=torch.float32
            )

        image = image.unsqueeze(2)

        # Create video conditioning tensor (same for both Wan 2.1 and Wan 2.2 14B)
        if last_image is None:
            # First frame + zeros
            video_condition = torch.cat(
                [
                    image,
                    image.new_zeros(image.shape[0], image.shape[1], num_frames - 1, height, width),
                ],
                dim=2,
            )
        else:
            # First frame + zeros + last frame (interpolation)
            last_image = last_image.unsqueeze(2)
            video_condition = torch.cat(
                [
                    image,
                    image.new_zeros(image.shape[0], image.shape[1], num_frames - 2, height, width),
                    last_image,
                ],
                dim=2,
            )

        # Encode video condition through VAE
        video_condition = video_condition.to(device=self.device, dtype=self.vae.dtype)
        latent_condition = retrieve_latents(self.vae.encode(video_condition), sample_mode="argmax")
        latent_condition = latent_condition.to(self.dtype)

        # Normalize latents to match diffusion model's latent space
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
            1, self.vae.config.z_dim, 1, 1, 1
        ).to(latents.device, latents.dtype)
        latent_condition = (latent_condition - latents_mean) * latents_std

        # Create mask in video frame space
        # Reshaping is required to match the transformer's expected input format
        mask_lat_size = torch.ones(1, 1, num_frames, latent_height, latent_width)

        if last_image is None:
            mask_lat_size[:, :, list(range(1, num_frames))] = 0
        else:
            mask_lat_size[:, :, list(range(1, num_frames - 1))] = 0

        first_frame_mask = mask_lat_size[:, :, 0:1]
        first_frame_mask = torch.repeat_interleave(
            first_frame_mask, dim=2, repeats=self.vae_scale_factor_temporal
        )

        mask_lat_size = torch.cat([first_frame_mask, mask_lat_size[:, :, 1:, :]], dim=2)

        mask_lat_size = mask_lat_size.view(
            1, -1, self.vae_scale_factor_temporal, latent_height, latent_width
        )

        mask_lat_size = mask_lat_size.transpose(1, 2)

        mask_lat_size = mask_lat_size.to(self.device, dtype=self.dtype)

        # Concatenate mask and condition along channel dimension
        condition = torch.cat([mask_lat_size, latent_condition], dim=1)

        # Repeat condition for batch (single image, multiple prompts)
        if batch_size > 1:
            condition = condition.repeat(batch_size, 1, 1, 1, 1)

        return latents, condition

    def _decode_latents(self, latents):
        """Decode latents to video."""
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

        # Post-process video tensor: (B, C, T, H, W) -> (B, T, H, W, C)
        video = postprocess_video_tensor(video)

        return video
