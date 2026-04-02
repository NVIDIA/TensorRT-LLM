import itertools
import os
import time
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple, Type

import torch
import torch.distributed as dist
import torch.nn as nn

from tensorrt_llm._utils import nvtx_range
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

from .cache import CacheDiTAccelerator, TeaCacheAccelerator
from .checkpoints import WeightLoader
from .config import PipelineComponent
from .cuda_graph_runner import CUDAGraphRunner, CUDAGraphRunnerConfig, SharedGraphPool
from .modules.vae.parallel_vae_interface import ParallelVAEFactory

if TYPE_CHECKING:
    from .cache import CacheAccelerator
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
        self._cuda_graph_runners: Dict[str, CUDAGraphRunner] = {}
        self._parallel_vae_enabled: bool = False
        self._warmed_up_shapes: Set[tuple] = set()

        # Unified cache acceleration (TeaCache, Cache-DiT); see _setup_cache_acceleration
        self.cache_accelerator: Optional["CacheAccelerator"] = None

        # Components
        self.transformer: Optional[nn.Module] = None
        self.vae: Optional[nn.Module] = None
        self.text_encoder: Optional[nn.Module] = None
        self.tokenizer: Optional[Any] = None
        self.scheduler: Optional[Any] = None

        # Initialize transformer
        self._init_transformer()

        # CUDA graph runner - wrap transformer.forward
        # Order matters: TeaCache will wrap on top of it and still call the
        # graphed transformer.forward if should_compute == True.
        self._setup_cuda_graphs()

    def _setup_cuda_graphs(self):
        """Wrap all transformer components with CUDA graph capture/replay."""
        if not self.model_config.cuda_graph.enable_cuda_graph:
            return

        if self.model_config.torch_compile.enable_torch_compile:
            logger.warning(
                "CUDA graphs with torch.compile not yet supported. Using torch.compile only."
            )
            return

        if len(self.transformer_components) > 1:
            logger.info(
                "CUDA graph runner: multiple transformer components, using shared graph pool"
            )
            shared_pool = SharedGraphPool()
        else:
            shared_pool = None

        for name in self.transformer_components:
            model = getattr(self, name, None)
            if model is None:
                continue

            runner = CUDAGraphRunner(CUDAGraphRunnerConfig(use_cuda_graph=True), shared_pool)
            logger.info(f"CUDA graph runner: wrapping {name}.forward")
            model.forward = runner.wrap(model.forward)
            self._cuda_graph_runners[name] = runner

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

    @property
    def transformer_components(self) -> list:
        """Return list of transformer components this pipeline needs."""
        return [PipelineComponent.TRANSFORMER] if self.transformer is not None else []

    def warmup_cache_key(self, height: int, width: int, num_frames: int) -> tuple:
        """Return the cache key for a given warmup shape.

        Image models (FLUX) override to return (height, width), ignoring
        num_frames.  Video models use the default (height, width, num_frames).
        The executor uses this to check whether a request shape was warmed up.
        """
        return (height, width, num_frames)

    @property
    def default_warmup_resolutions(self) -> List[Tuple[int, int]]:
        """Model-specific default warmup resolutions (height, width).

        Subclasses should override. Combined with default_warmup_num_frames
        via Cartesian product to produce warmup shapes.
        """
        return []

    @property
    def default_warmup_num_frames(self) -> List[int]:
        """Model-specific default warmup frame counts.

        Subclasses should override. Combined with default_warmup_resolutions
        via Cartesian product to produce warmup shapes.
        """
        return []

    @property
    def default_warmup_steps(self) -> int:
        """Model-specific default denoising steps for warmup. Subclass override."""
        return 2

    @property
    def resolution_multiple_of(self) -> Tuple[int, int]:
        """(h_multiple, w_multiple) resolution constraint. Subclass override."""
        return (1, 1)

    def validate_resolution(self, height: int, width: int, num_frames: int) -> None:
        """Validate resolution against model constraints. Raises ValueError.

        Only checks resolution constraints (must be positive and divisible by
        model-specific multiples). Frame count is NOT validated here — following
        HuggingFace diffusers convention, invalid frame counts are silently
        rounded in forward() instead of rejected.
        """
        if height <= 0 or width <= 0 or num_frames <= 0:
            raise ValueError(
                f"Dimensions must be positive: height={height}, width={width}, "
                f"num_frames={num_frames} for {self.__class__.__name__}."
            )
        h_mul, w_mul = self.resolution_multiple_of
        if h_mul > 1 or w_mul > 1:
            if height % h_mul != 0 or width % w_mul != 0:
                raise ValueError(
                    f"Resolution ({height}x{width}) must be multiples of "
                    f"({h_mul}x{w_mul}) for {self.__class__.__name__}."
                )

    def resolve_warmup_plan(self) -> Tuple[List[Tuple[int, int, int]], int]:
        """Resolve warmup shapes and steps from config or model defaults.

        Shapes are the Cartesian product of resolutions x num_frames.

        Priority:
            1. User-specified: model_config.compilation.resolutions / num_frames
            2. Model defaults: default_warmup_resolutions / default_warmup_num_frames
            3. Empty: skip warmup

        Steps: always from model subclass (default_warmup_steps).

        Returns:
            (shapes, steps) tuple where shapes = list of (h, w, f)
        """
        warmup_cfg = self.model_config.compilation

        if warmup_cfg.resolutions is not None or warmup_cfg.num_frames is not None:
            resolutions = (
                warmup_cfg.resolutions
                if warmup_cfg.resolutions is not None
                else self.default_warmup_resolutions
            )
            num_frames_list = (
                warmup_cfg.num_frames
                if warmup_cfg.num_frames is not None
                else self.default_warmup_num_frames
            )
        else:
            resolutions = self.default_warmup_resolutions
            num_frames_list = self.default_warmup_num_frames

        all_shapes = [(h, w, f) for (h, w), f in itertools.product(resolutions, num_frames_list)]

        valid_shapes = []
        for h, w, f in all_shapes:
            try:
                self.validate_resolution(h, w, f)
                valid_shapes.append((h, w, f))
            except ValueError as e:
                logger.warning(f"Skipping invalid warmup shape ({h}x{w}, {f} frames): {e}")

        return valid_shapes, self.default_warmup_steps

    @property
    def vae_adapter_class(self) -> Type[ParallelVAEFactory] | None:
        """Return the VAE adapter class for the pipeline."""
        return None

    def infer(self, req: Any):
        raise NotImplementedError

    def _init_transformer(self) -> None:
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def load_transformer_weights(self, checkpoint_dir: str) -> Dict[str, torch.Tensor]:
        """Load transformer weights from checkpoint.

        Default implementation reads from a ``transformer/`` sub-directory
        using :class:`WeightLoader`.  Override for custom checkpoint formats
        (e.g. LTX-2 single-safetensor with embedded prefix).
        """
        if self.transformer is None:
            raise ValueError("Pipeline has no transformer component")

        transformer_components = self.transformer_components
        logger.info(f"Transformer components: {transformer_components}")

        transformer_path = os.path.join(checkpoint_dir, PipelineComponent.TRANSFORMER)
        if not os.path.exists(transformer_path):
            raise FileNotFoundError(
                f"Transformer path does not exist: {transformer_path}. "
                f"Checkpoint directory must contain a 'transformer' subdirectory."
            )

        weight_loader = WeightLoader(components=transformer_components)
        return weight_loader.load_weights(checkpoint_dir, self.mapping)

    def load_standard_components(
        self,
        checkpoint_dir: str,
        device: torch.device,
        skip_components: Optional[list] = None,
        **kwargs,
    ) -> None:
        raise NotImplementedError

    def load_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        if self.transformer is not None and hasattr(self.transformer, "load_weights"):
            self.transformer.load_weights(weights)

    def post_load_weights(self) -> None:
        if self.transformer is not None and hasattr(self.transformer, "post_load_weights"):
            self.transformer.post_load_weights()

    def _apply_teacache_coefficients(self, coefficients: Optional[Dict]) -> None:
        """Pick TeaCache coefficients from checkpoint path; updates model_config.teacache in place."""
        if not coefficients:
            return
        teacache_cfg = self.model_config.teacache
        checkpoint_path = (
            getattr(getattr(self.model_config, "pretrained_config", None), "_name_or_path", "")
            or ""
        )
        matched = False
        for model_size, coeff_data in coefficients.items():
            if model_size.lower() in checkpoint_path.lower():
                matched = True
                if isinstance(coeff_data, dict):
                    mode = "ret_steps" if teacache_cfg.use_ret_steps else "standard"
                    if mode in coeff_data:
                        teacache_cfg.coefficients = coeff_data[mode]
                        logger.info(f"TeaCache: Using {model_size} coefficients ({mode} mode)")
                    default_thresh = coeff_data.get("default_thresh")
                    if (
                        default_thresh is not None
                        and "teacache_thresh" not in teacache_cfg.model_fields_set
                    ):
                        teacache_cfg.teacache_thresh = default_thresh
                        logger.info(
                            f"TeaCache: Using {model_size} default threshold {default_thresh}"
                        )
                else:
                    teacache_cfg.coefficients = coeff_data
                    logger.info(f"TeaCache: Using {model_size} coefficients")
                break
        if not matched:
            raise ValueError(
                f"TeaCache: No coefficients found for checkpoint '{checkpoint_path}'. "
                f"Available variants: {list(coefficients.keys())}. "
                f"TeaCache is not supported for this model variant."
            )

    def _setup_cache_acceleration(
        self,
        model: Optional[nn.Module] = None,
        coefficients: Optional[Dict] = None,
    ) -> None:
        """Enable TeaCache or Cache-DiT from model_config.cache_backend."""

        if getattr(self, "cache_accelerator", None) is not None:
            self.cache_accelerator.unwrap()
            self.cache_accelerator = None

        cfg = self.model_config

        if cfg.cache_backend == "cache_dit":
            acc = CacheDiTAccelerator.try_create(self, cfg.cache_dit)
            if acc is None:
                return
            try:
                acc.wrap()
            except Exception as exc:
                logger.error("Cache-DiT: failed to enable: %s", exc)
                return
            self.cache_accelerator = acc
            return

        use_teacache = cfg.cache_backend == "teacache"
        if not use_teacache:
            return

        if coefficients is not None:
            BasePipeline._apply_teacache_coefficients(self, coefficients)

        if model is None:
            return

        acc = TeaCacheAccelerator(cfg.teacache)
        acc.wrap(model=model)
        if acc.is_enabled():
            self.cache_accelerator = acc

    def setup_parallel_vae(self):
        if not self.model_config.parallel.enable_parallel_vae:
            return
        if not dist.is_initialized() or dist.get_world_size() <= 1:
            return
        if self.vae is None:
            return

        # Uses all ranks today; replace with a subset to dedicate specific ranks to VAE.
        pg = dist.new_group(list(range(dist.get_world_size())))
        try:
            self.vae = ParallelVAEFactory.from_vae(
                self.vae,
                split_dim=self.model_config.parallel.parallel_vae_split_dim,
                pg=pg,
            )
        except ValueError:
            logger.warning(
                f"Parallel VAE not supported for {self.__class__.__name__} "
                f"(VAE type: {type(self.vae).__name__}). "
                "Add an entry to ParallelVAEFactory._LAZY_REGISTRY to enable "
                "parallel VAE for this VAE type."
            )
            return

        self._parallel_vae_enabled = True
        logger.info(
            f"Parallel VAE enabled: {type(self.vae).__name__}, "
            f"split_dim={self.model_config.parallel.parallel_vae_split_dim}, "
            f"world_size={dist.get_world_size(pg)}"
        )

    def torch_compile(self) -> None:
        """Apply torch.compile to pipeline components based on TorchCompileConfig.

        For transformer models, compiles each block in the ModuleList individually.
        This enables future block-wise offloading and keeps compilation efficient
        (all blocks share the same structure, so compile cost is paid once).

        For non-transformer components, compiles the entire module.
        """
        tc_config = self.model_config.torch_compile

        # Using default as max-autotune mode takes more initialization time and
        # does not improve performance a lot.
        compile_mode = "default"

        # Compiling transformer blocks provides max performance value.
        targets = self.transformer_components

        for name in targets:
            model = getattr(self, name, None)
            if model is None:
                logger.warning(f"torch.compile: component '{name}' not found, skipping")
                continue

            blocks_attr = self._find_transformer_blocks(model)
            if blocks_attr:
                for block_name in blocks_attr:
                    blocks = getattr(model, block_name)
                    logger.info(
                        f"torch.compile: {name}.{block_name} "
                        f"({len(blocks)} blocks, mode={compile_mode})"
                    )
                    compiled_blocks = []
                    for block in blocks:
                        compiled_blocks.append(
                            torch.compile(
                                block,
                                mode=compile_mode,
                                dynamic=None,
                                fullgraph=tc_config.enable_fullgraph,
                            )
                        )
                    setattr(model, block_name, nn.ModuleList(compiled_blocks))
            else:
                logger.info(f"torch.compile: {name} (whole module, mode={compile_mode})")
                compiled = torch.compile(
                    model,
                    mode=compile_mode,
                    dynamic=None,
                    fullgraph=tc_config.enable_fullgraph,
                )
                setattr(self, name, compiled)

    @staticmethod
    def _find_transformer_blocks(model: nn.Module) -> list:
        """Find ModuleList children that look like transformer blocks.

        Returns list of attribute names containing nn.ModuleList with >1 elements.
        """
        block_names = []
        for name, child in model.named_children():
            if isinstance(child, nn.ModuleList) and len(child) > 1:
                block_names.append(name)
        return block_names

    def warmup(self) -> None:
        """Run warmup inference to trigger torch.compile and CUDA initialization.

        Resolves warmup shapes from user config or model defaults, then runs
        a short denoising loop with dummy inputs for each shape. This:
        1. Triggers torch.compile's lazy compilation (first forward trace + codegen)
        2. Pre-captures CUDA graphs (if enabled)
        3. Warms up CUDA kernels and allocators
        4. Populates any lazy caches (e.g., RoPE frequencies)

        Called automatically by PipelineLoader after model loading and torch.compile.
        OOM is not caught — if a warmup shape OOMs, the server fails fast at startup.
        """
        shapes, steps = self.resolve_warmup_plan()
        if not shapes:
            logger.info("Warmup disabled (no warmup shapes)")
            return

        logger.info(
            f"Running warmup for {self.__class__.__name__} "
            f"with {len(shapes)} shapes and {steps} steps..."
        )
        warmup_start = time.time()

        for height, width, num_frames in shapes:
            logger.info(f"Warmup: {height}x{width}, {num_frames} frames, {steps} steps")
            self._run_warmup(height, width, num_frames, steps)
            torch.cuda.synchronize()

        self._warmed_up_shapes = set(
            self.warmup_cache_key(h, w, num_frames=f) for h, w, f in shapes
        )
        elapsed = time.time() - warmup_start
        logger.info(f"Warmup completed in {elapsed:.2f}s")

    def _run_warmup(self, height: int, width: int, num_frames: int, steps: int) -> None:
        """Run warmup for a single shape. Subclasses must override.

        Args:
            height: Video/image height in pixels
            width: Video/image width in pixels
            num_frames: Number of frames (1 for image models)
            steps: Number of denoising steps
        """
        logger.warning(
            f"{self.__class__.__name__} does not implement _run_warmup(); "
            "skipping warmup for this shape."
        )

    def decode_latents(
        self,
        latents: torch.Tensor,
        decode_fn: Callable[[torch.Tensor], Any],
        extra_latents: Optional[Dict[str, Tuple[torch.Tensor, Callable]]] = None,
    ):
        """Execute VAE decoding. Only rank 0 performs decoding.
        If parallel VAE is enabled, all processes perform decoding.

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

        if self._parallel_vae_enabled:
            primary_result = decode_fn(latents)
            if extra_latents:
                extra_results = [efn(elat) for _, (elat, efn) in extra_latents.items()]
                return (primary_result,) + tuple(extra_results)
            return primary_result

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

        # All-gather primary noise (must be contiguous for NCCL)
        noise_pred_local = noise_pred_local.contiguous()
        gather_list = [torch.empty_like(noise_pred_local) for _ in range(self.world_size)]
        dist.all_gather(gather_list, noise_pred_local)
        noise_cond = gather_list[0]
        noise_uncond = gather_list[ulysses_size]
        noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

        # All-gather extra stream noises
        extra_noise_preds = {}
        for name, noise_local in extra_noise_locals.items():
            noise_local = noise_local.contiguous()
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

    @nvtx_range("_scheduler_step", color="blue")
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

    @nvtx_range("denoise_loop", color="blue")
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

        # Reset cache acceleration state for new generation (TeaCache / Cache-DiT)
        if getattr(self, "cache_accelerator", None) and self.cache_accelerator.is_enabled():
            self.cache_accelerator.refresh(total_steps)

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
            with nvtx_range(f"denoise_step {i}"):
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

            # Single logging site for TeaCache and Cache-DiT.
            if getattr(self, "cache_accelerator", None) and self.cache_accelerator.is_enabled():
                stats = self.cache_accelerator.get_stats()
                if stats:
                    if self.model_config.cache_backend == "cache_dit":
                        logger.info("Cache-DiT stats: %s", stats)
                    elif "hit_rate" in stats:
                        logger.info(
                            f"TeaCache: {stats['hit_rate']:.1%} hit rate "
                            f"({stats['cached']}/{stats['total']} steps)"
                        )
                    else:
                        logger.info("Cache acceleration stats: %s", stats)

        return (latents, extra_stream_latents) if has_extra_streams else latents

    def cleanup(self):
        """Call before dist.destroy_process_group()."""
        for name, runner in self._cuda_graph_runners.items():
            logger.info(f"Releasing CUDA graphs for {name}")
            runner.clear()
