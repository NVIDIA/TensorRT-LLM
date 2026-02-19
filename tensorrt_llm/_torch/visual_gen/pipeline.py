import time
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn

from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

from .teacache import TeaCacheBackend

if TYPE_CHECKING:
    from .config import DiffusionModelConfig


class BasePipeline(nn.Module):
    """
    Base class for diffusion pipelines.
    """

    def __init__(self, model_config: "DiffusionModelConfig"):
        super().__init__()
        self.model_config = model_config
        self.config = model_config.pretrained_config
        self.mapping: Mapping = getattr(model_config, "mapping", None) or Mapping()

        # Components
        self.transformer: Optional[nn.Module] = None
        self.vae: Optional[nn.Module] = None
        self.text_encoder: Optional[nn.Module] = None
        self.tokenizer: Optional[Any] = None
        self.scheduler: Optional[Any] = None

        # Initialize transformer
        self._init_transformer()

    @property
    def rank(self):
        return dist.get_rank() if dist.is_initialized() else 0

    @property
    def world_size(self):
        return dist.get_world_size() if dist.is_initialized() else 1

    @property
    def dtype(self):
        if hasattr(self, "transformer"):
            return next(self.transformer.parameters()).dtype
        return torch.float32

    @property
    def device(self):
        return self.transformer.device

    def infer(self, req: Any):
        raise NotImplementedError

    def _init_transformer(self) -> None:
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def load_standard_components(
        self,
        checkpoint_dir: str,
        device: torch.device,
        skip_components: Optional[list] = None,
    ) -> None:
        raise NotImplementedError

    def load_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        if self.transformer is not None and hasattr(self.transformer, "load_weights"):
            self.transformer.load_weights(weights)

    def post_load_weights(self) -> None:
        if self.transformer is not None and hasattr(self.transformer, "post_load_weights"):
            self.transformer.post_load_weights()

    def _setup_teacache(self, model, coefficients: Optional[Dict] = None):
        """Setup TeaCache optimization for the transformer model.

        TeaCache caches transformer block outputs when timestep embeddings change slowly,
        reducing computation during the denoising loop.

        Args:
            model: The transformer model to optimize
            coefficients: Optional dict of model-specific polynomial coefficients for cache decisions
                         Format: {model_size: {"ret_steps": [...], "standard": [...]}}
        """
        self.cache_backend = None

        # Get teacache config from model_config (always present now)
        teacache_cfg = self.model_config.teacache
        if not teacache_cfg.enable_teacache:
            return

        # Apply model-specific polynomial coefficients
        # Coefficients are used to rescale embedding distances for cache decisions
        if coefficients:
            checkpoint_path = (
                getattr(self.model_config.pretrained_config, "_name_or_path", "") or ""
            )
            for model_size, coeff_data in coefficients.items():
                # Match model size in path (case-insensitive, e.g., "1.3B", "14B", "dev")
                if model_size.lower() in checkpoint_path.lower():
                    if isinstance(coeff_data, dict):
                        # Select coefficient set based on warmup mode
                        mode = "ret_steps" if teacache_cfg.use_ret_steps else "standard"
                        if mode in coeff_data:
                            teacache_cfg.coefficients = coeff_data[mode]
                            logger.info(f"TeaCache: Using {model_size} coefficients ({mode} mode)")
                    else:
                        # Single coefficient list (no mode distinction)
                        teacache_cfg.coefficients = coeff_data
                        logger.info(f"TeaCache: Using {model_size} coefficients")
                    break

        # Initialize and enable TeaCache backend
        logger.info("TeaCache: Initializing...")
        self.cache_backend = TeaCacheBackend(teacache_cfg)
        self.cache_backend.enable(model)

    def decode_latents(
        self,
        latents: torch.Tensor,
        decode_fn: Callable[[torch.Tensor], Any],
        extra_latents: Optional[Dict[str, Tuple[torch.Tensor, Callable]]] = None,
    ):
        """Execute VAE decoding. Only rank 0 performs decoding.

        Args:
            latents: Primary latents to decode (e.g., video)
            decode_fn: Decoder function for primary latents
            extra_latents: Optional dict of additional latents to decode.
                          Format: {name: (latents_tensor, decode_fn)}
                          Example: {"audio": (audio_latents, audio_decode_fn)}

        Returns:
            Single result if no extra_latents, tuple of results if extra_latents provided.
            Non-rank-0 processes return None placeholders.
        """
        if self.rank == 0:
            primary_result = decode_fn(latents)

            if extra_latents:
                extra_results = []
                for name, (extra_latent, extra_decode_fn) in extra_latents.items():
                    extra_results.append(extra_decode_fn(extra_latent))
                return (primary_result,) + tuple(extra_results)

            return primary_result

        # Return None placeholders for non-rank-0 processes
        n_results = 1 + (len(extra_latents) if extra_latents else 0)
        return (None,) * n_results if n_results > 1 else None

    @staticmethod
    def _rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
        """Rescale noise to fix overexposure (https://huggingface.co/papers/2305.08891)."""
        std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
        std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
        noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
        return guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg

    def _setup_cfg_config(
        self, guidance_scale, prompt_embeds, neg_prompt_embeds, extra_cfg_tensors=None
    ):
        """Setup CFG parallel configuration.

        Args:
            guidance_scale: CFG guidance scale
            prompt_embeds: Positive prompt embeddings
            neg_prompt_embeds: Negative prompt embeddings (None if already concatenated)
            extra_cfg_tensors: Optional dict of additional tensors to split for CFG parallel.
                              Format: {name: (positive_tensor, negative_tensor)}
                              Example: {"audio_embeds": (pos_audio, neg_audio),
                                       "attention_mask": (pos_mask, neg_mask)}

        Returns:
            Dict with CFG configuration including split tensors
        """
        # Access parallel config directly (always present now)
        cfg_size = self.model_config.parallel.dit_cfg_size
        ulysses_size = self.model_config.parallel.dit_ulysses_size

        cfg_group = self.rank // ulysses_size
        is_split_embeds = neg_prompt_embeds is not None
        do_cfg_parallel = cfg_size >= 2 and guidance_scale > 1.0

        local_extras = {}

        if do_cfg_parallel:
            if self.rank == 0:
                logger.info(f"CFG Parallel: cfg_size={cfg_size}, ulysses_size={ulysses_size}")

            # Split main embeddings
            if is_split_embeds:
                pos_embeds, neg_embeds = prompt_embeds, neg_prompt_embeds
            else:
                neg_embeds, pos_embeds = prompt_embeds.chunk(2)

            local_embeds = pos_embeds if cfg_group == 0 else neg_embeds

            # Split extra tensors if provided
            if extra_cfg_tensors:
                for name, (pos_tensor, neg_tensor) in extra_cfg_tensors.items():
                    if pos_tensor is not None and neg_tensor is not None:
                        local_extras[name] = pos_tensor if cfg_group == 0 else neg_tensor
                    elif pos_tensor is not None:
                        # Only positive provided, use it for both
                        local_extras[name] = pos_tensor
        else:
            local_embeds = None
            if is_split_embeds and guidance_scale > 1.0:
                prompt_embeds = torch.cat([neg_prompt_embeds, prompt_embeds])

            # For standard CFG, concatenate extra tensors
            if extra_cfg_tensors:
                for name, (pos_tensor, neg_tensor) in extra_cfg_tensors.items():
                    if pos_tensor is not None and neg_tensor is not None and guidance_scale > 1.0:
                        local_extras[name] = torch.cat([neg_tensor, pos_tensor], dim=0)
                    elif pos_tensor is not None:
                        local_extras[name] = pos_tensor

        return {
            "enabled": do_cfg_parallel,
            "cfg_size": cfg_size,
            "ulysses_size": ulysses_size,
            "cfg_group": cfg_group,
            "local_embeds": local_embeds,
            "prompt_embeds": prompt_embeds,
            "local_extras": local_extras,
        }

    def _denoise_step_cfg_parallel(
        self,
        latents,
        extra_stream_latents,
        timestep,
        local_embeds,
        forward_fn,
        guidance_scale,
        guidance_rescale,
        ulysses_size,
        local_extras,
    ):
        """Execute single denoising step with CFG parallel."""
        t_start = time.time()
        result = forward_fn(latents, extra_stream_latents, timestep, local_embeds, local_extras)

        # Handle return format: (primary_noise, extra_noises_dict) or just primary_noise
        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
            noise_pred_local, extra_noise_locals = result
        else:
            noise_pred_local = result
            extra_noise_locals = {}

        t_transformer = time.time() - t_start

        c_start = time.time()

        # All-gather primary noise
        gather_list = [torch.empty_like(noise_pred_local) for _ in range(self.world_size)]
        dist.all_gather(gather_list, noise_pred_local)
        noise_cond = gather_list[0]
        noise_uncond = gather_list[ulysses_size]
        noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

        # All-gather extra stream noises
        extra_noise_preds = {}
        for name, noise_local in extra_noise_locals.items():
            gather_list_extra = [torch.empty_like(noise_local) for _ in range(self.world_size)]
            dist.all_gather(gather_list_extra, noise_local)
            noise_cond_extra = gather_list_extra[0]
            noise_uncond_extra = gather_list_extra[ulysses_size]
            extra_noise_preds[name] = noise_uncond_extra + guidance_scale * (
                noise_cond_extra - noise_uncond_extra
            )

            if guidance_rescale > 0.0:
                extra_noise_preds[name] = self._rescale_noise_cfg(
                    extra_noise_preds[name], noise_cond_extra, guidance_rescale
                )

        if guidance_rescale > 0.0:
            noise_pred = self._rescale_noise_cfg(noise_pred, noise_cond, guidance_rescale)

        t_cfg = time.time() - c_start
        return noise_pred, extra_noise_preds, t_transformer, t_cfg

    def _denoise_step_standard(
        self,
        latents,
        extra_stream_latents,
        timestep,
        prompt_embeds,
        forward_fn,
        guidance_scale,
        guidance_rescale,
        local_extras,
    ):
        """Execute single denoising step without CFG parallel."""
        if guidance_scale > 1.0:
            latent_input = torch.cat([latents] * 2)
            # Duplicate extra stream latents for CFG
            extra_stream_input = {
                name: torch.cat([stream_latents] * 2)
                for name, stream_latents in extra_stream_latents.items()
            }
        else:
            latent_input = latents
            extra_stream_input = extra_stream_latents

        timestep_expanded = timestep.expand(latent_input.shape[0])

        t_start = time.time()
        result = forward_fn(
            latent_input, extra_stream_input, timestep_expanded, prompt_embeds, local_extras
        )

        # Handle return format: (primary_noise, extra_noises_dict) or just primary_noise
        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
            noise_pred, extra_noise_preds = result
        else:
            noise_pred = result
            extra_noise_preds = {}

        t_transformer = time.time() - t_start

        c_start = time.time()
        if guidance_scale > 1.0:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # Apply CFG to extra streams
            for name, noise_extra in extra_noise_preds.items():
                noise_uncond_extra, noise_text_extra = noise_extra.chunk(2)
                extra_noise_preds[name] = noise_uncond_extra + guidance_scale * (
                    noise_text_extra - noise_uncond_extra
                )

                if guidance_rescale > 0.0:
                    extra_noise_preds[name] = self._rescale_noise_cfg(
                        extra_noise_preds[name], noise_text_extra, guidance_rescale
                    )

            if guidance_rescale > 0.0:
                noise_pred = self._rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale)

            t_cfg = time.time() - c_start
        else:
            t_cfg = 0.0

        return noise_pred, extra_noise_preds, t_transformer, t_cfg

    def _scheduler_step(
        self,
        latents,
        extra_stream_latents,
        noise_pred,
        extra_noise_preds,
        timestep,
        scheduler,
        extra_stream_schedulers,
    ):
        """Execute scheduler step for all streams."""
        t_start = time.time()
        latents = scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]

        # Step schedulers for extra streams
        for name, noise_extra in extra_noise_preds.items():
            if name in extra_stream_schedulers:
                extra_stream_latents[name] = extra_stream_schedulers[name].step(
                    noise_extra, timestep, extra_stream_latents[name], return_dict=False
                )[0]

        t_sched = time.time() - t_start
        return latents, extra_stream_latents, t_sched

    def denoise(
        self,
        latents: torch.Tensor,
        scheduler: Any,
        prompt_embeds: torch.Tensor,
        guidance_scale: float,
        forward_fn: Callable,
        timesteps: Optional[torch.Tensor] = None,
        neg_prompt_embeds: Optional[torch.Tensor] = None,
        guidance_rescale: float = 0.0,
        extra_cfg_tensors: Optional[Dict[str, Tuple[torch.Tensor, Optional[torch.Tensor]]]] = None,
        extra_streams: Optional[Dict[str, Tuple[torch.Tensor, Any]]] = None,
        guidance_scale_2: Optional[float] = None,
        boundary_timestep: Optional[float] = None,
    ):
        """Execute denoising loop with optional CFG parallel and TeaCache support.

        Args:
            latents: Initial noise latents (primary stream, e.g., video)
            scheduler: Diffusion scheduler for primary stream
            prompt_embeds: Text embeddings (positive)
            guidance_scale: CFG strength (1.0 = no guidance)
            forward_fn: Transformer forward function
                       Signature: forward_fn(latents, extra_stream_latents, timestep,
                                            encoder_hidden_states, extra_tensors_dict)
                       Returns: (primary_noise, extra_stream_noises_dict) or just primary_noise
            timesteps: Optional custom timesteps (defaults to scheduler.timesteps)
            neg_prompt_embeds: Optional negative text embeddings for CFG
            guidance_rescale: CFG rescale factor to prevent overexposure
            extra_cfg_tensors: Optional dict of additional tensors to split for CFG parallel
                              Format: {name: (positive_tensor, negative_tensor)}
                              Example: {"audio_embeds": (pos_audio, neg_audio)}
            extra_streams: Optional dict of additional streams to denoise in parallel
                          Format: {name: (stream_latents, stream_scheduler)}
                          Example: {"audio": (audio_latents, audio_scheduler)}
            guidance_scale_2: Optional guidance scale for two-stage denoising.
                             When provided with boundary_timestep, switches from guidance_scale
                             to guidance_scale_2 when timestep < boundary_timestep.
            boundary_timestep: Optional timestep boundary for two-stage denoising.
                              Switches guidance scale when crossing this threshold.

        Returns:
            Single latents if no extra_streams
            Tuple (primary_latents, extra_streams_dict) if extra_streams provided
        """
        if timesteps is None:
            timesteps = scheduler.timesteps

        total_steps = len(timesteps)
        has_extra_streams = extra_streams is not None and len(extra_streams) > 0

        # Reset TeaCache state for new generation
        # Sets warmup/cutoff steps based on total_steps
        if (
            hasattr(self, "cache_backend")
            and self.cache_backend
            and self.cache_backend.is_enabled()
        ):
            self.cache_backend.refresh(total_steps)

        if self.rank == 0:
            if has_extra_streams:
                stream_names = ", ".join(["primary"] + list(extra_streams.keys()))
                logger.info(
                    f"Denoising [{stream_names}]: {total_steps} steps, guidance={guidance_scale}"
                )
            else:
                logger.info(f"Denoising: {total_steps} steps, guidance={guidance_scale}")

        cfg_config = self._setup_cfg_config(
            guidance_scale, prompt_embeds, neg_prompt_embeds, extra_cfg_tensors
        )
        do_cfg_parallel = cfg_config["enabled"]
        prompt_embeds = cfg_config["prompt_embeds"]
        local_extras = cfg_config["local_extras"]

        # Extract extra stream latents and schedulers
        extra_stream_latents = {}
        extra_stream_schedulers = {}
        if extra_streams:
            for name, (stream_latents, stream_scheduler) in extra_streams.items():
                extra_stream_latents[name] = stream_latents
                extra_stream_schedulers[name] = stream_scheduler

        start_time = time.time()

        for i, t in enumerate(timesteps):
            step_start = time.time()

            # Two-stage denoising: switch guidance scale at boundary
            current_guidance_scale = guidance_scale
            if guidance_scale_2 is not None and boundary_timestep is not None:
                t_scalar = t.item() if t.dim() == 0 else t[0].item()
                if t_scalar < boundary_timestep:
                    current_guidance_scale = guidance_scale_2

            # Denoise
            if do_cfg_parallel:
                timestep = t.expand(latents.shape[0])
                noise_pred, extra_noise_preds, t_trans, t_cfg = self._denoise_step_cfg_parallel(
                    latents,
                    extra_stream_latents,
                    timestep,
                    cfg_config["local_embeds"],
                    forward_fn,
                    current_guidance_scale,
                    guidance_rescale,
                    cfg_config["ulysses_size"],
                    local_extras,
                )
            else:
                noise_pred, extra_noise_preds, t_trans, t_cfg = self._denoise_step_standard(
                    latents,
                    extra_stream_latents,
                    t,
                    prompt_embeds,
                    forward_fn,
                    current_guidance_scale,
                    guidance_rescale,
                    local_extras,
                )

            # Scheduler step for all streams
            latents, extra_stream_latents, t_sched = self._scheduler_step(
                latents,
                extra_stream_latents,
                noise_pred,
                extra_noise_preds,
                t,
                scheduler,
                extra_stream_schedulers,
            )

            # Logging
            if self.rank == 0:
                step_time = time.time() - step_start
                avg_time = (time.time() - start_time) / (i + 1)
                eta = avg_time * (total_steps - i - 1)
                logger.info(
                    f"Step {i + 1}/{total_steps} | {step_time:.2f}s "
                    f"(trans={t_trans:.2f}s cfg={t_cfg:.3f}s sched={t_sched:.3f}s) | "
                    f"Avg={avg_time:.2f}s/step ETA={eta:.1f}s"
                )

        if self.rank == 0:
            total_time = time.time() - start_time
            logger.info("=" * 80)
            logger.info(f"Denoising done: {total_time:.2f}s ({total_time / total_steps:.2f}s/step)")

            # Log TeaCache performance statistics
            # Shows how many transformer steps were skipped (cache hits) vs computed
            if (
                hasattr(self, "cache_backend")
                and self.cache_backend
                and self.cache_backend.is_enabled()
            ):
                stats = self.cache_backend.get_stats()
                if stats:
                    logger.info(
                        f"TeaCache: {stats['hit_rate']:.1%} hit rate ({stats['cached']}/{stats['total']} steps)"
                    )

        return (latents, extra_stream_latents) if has_extra_streams else latents
