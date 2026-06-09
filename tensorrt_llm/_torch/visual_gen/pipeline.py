import itertools
import os
import time
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple, Type

import torch
import torch.distributed as dist
import torch.nn as nn
from pydantic import Field

from tensorrt_llm._torch.visual_gen.pipeline_registry import PipelineComponent
from tensorrt_llm._utils import nvtx_range
from tensorrt_llm.llmapi.utils import StrictBaseModel
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

from .cache import CacheDiTAccelerator, TeaCacheAccelerator
from .checkpoints import WeightLoader
from .cuda_graph_runner import CUDAGraphRunner, CUDAGraphRunnerConfig, SharedGraphPool
from .modules.vae.parallel_vae_interface import ParallelVAEFactory


class ExtraParamSchema(StrictBaseModel):
    """Schema for a model-specific extra parameter.

    Returned by ``VisualGen.extra_param_specs`` so callers can
    discover which ``extra_params`` keys are valid for the loaded pipeline.
    """

    type: str = Field(description="Python type name (e.g. 'float', 'int', 'bool').")
    default: Any = Field(default=None, description="Default value used when omitted.")
    description: str = Field(default="", description="Human-readable description.")
    range: Optional[tuple] = Field(
        default=None, description="Optional (min, max) range for numeric params."
    )


def _parse_profile_range():
    """Parse ``TLLM_PROFILE_VISUAL_GEN_START_STOP`` for CUDA profiler scoping.

    Visual-gen-specific env var (separate from the LLM path's
    ``TLLM_PROFILE_START_STOP``). Use with ``nsys profile -c cudaProfilerApi ...``.

    Supported formats:

    * ``A-B``            – profile denoise steps A through B
    * ``A-B,C-D,...``    – multiple ranges; profiler toggles on/off per range
    * ``A,B,...``        – individual steps treated as single-step ranges
    * ``predenoise``     – profile the per-request pre-loop work inside
                           ``denoise()`` (CFG config setup, scheduler refresh,
                           TeaCache reset) up to the first denoise step.
                           Single-shot.
    * ``postdenoise``    – profile from the end of the last denoise step to
                           pipeline cleanup, covering VAE decode. Single-shot.
    * ``all``            – profile the full generation forward (denoise + VAE), skip warmup
    * (unset)            – no profiler API calls; plain ``nsys profile`` captures everything

    Returns ``None`` when unset, one of ``"all"`` / ``"predenoise"`` /
    ``"postdenoise"`` for keyword modes, or ``(frozenset(starts), frozenset(stops))``
    for numeric ranges.

    .. note::
       Step indices are **per-request**: each ``denoise()`` call resets the
       loop counter to 0, so e.g. ``0-4`` profiles steps 0-4 of *every*
       request. This differs from the LLM path's ``TLLM_PROFILE_START_STOP``
       which indexes a global executor iteration counter (one forward pass
       services all in-flight requests, so there is no "per request" index).

       ``predenoise`` and ``postdenoise`` are **single-shot per process**:
       they fire once around the first user request after warmup and do not
       re-arm on subsequent requests. Pair ``predenoise`` with
       ``nsys --capture-range-end=stop`` (keeps the app running cleanly after
       collection ends). ``postdenoise`` ends collection at process exit, so
       either ``stop`` or ``stop-shutdown`` works. For multi-request capture,
       use a numeric range with ``--capture-range-end=repeat:N``.
    """
    val = os.environ.get("TLLM_PROFILE_VISUAL_GEN_START_STOP")
    if not val:
        return None
    val = val.strip()
    if val.lower() in ("all", "predenoise", "postdenoise"):
        return val.lower()
    # Parse comma-separated ranges: "A-B,C-D,..." or single steps "A,B,..."
    # Same format as the LLM path (PyExecutor._load_iteration_indexes).
    starts, stops = [], []
    for span in val.split(","):
        span = span.strip()
        if "-" in span:
            start, stop = span.split("-", 1)
            starts.append(int(start))
            stops.append(int(stop))
        else:
            v = int(span)
            starts.append(v)
            stops.append(v)
    return frozenset(starts), frozenset(stops)


if TYPE_CHECKING:
    from .cache import CacheAccelerator
    from .config import DiffusionPipelineConfig


class BasePipeline(nn.Module):
    """
    Base class for diffusion pipelines.
    """

    @classmethod
    def resolve_variant(cls, config: "DiffusionPipelineConfig") -> Type["BasePipeline"]:
        """Return *cls* or a more specialized subclass based on *config*.

        Override in subclasses to select a variant pipeline at creation
        time (e.g. upgrading a base pipeline to a two-stage variant when
        extra checkpoint paths are provided).  The default returns *cls*
        unchanged.
        """
        return cls

    def __init__(self, pipeline_config: "DiffusionPipelineConfig"):
        super().__init__()
        self.pipeline_config = pipeline_config
        self.config = pipeline_config.primary_pretrained_config
        self.mapping: Mapping = getattr(pipeline_config, "mapping", None) or Mapping()
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
        self._is_warmup: bool = False

        # CUDA profiler scoping (TLLM_PROFILE_VISUAL_GEN_START_STOP env var)
        self._profile_range = _parse_profile_range()
        self._profiling_active: bool = False
        # Single-shot guards for predenoise/postdenoise modes — fire once
        # around the first non-warmup denoise() invocation, then disarm.
        self._predenoise_pending: bool = self._profile_range == "predenoise"
        self._postdenoise_pending: bool = self._profile_range == "postdenoise"

        # Initialize transformer
        self._init_transformer()

        # CUDA graph runner - wrap transformer.forward
        # Order matters: TeaCache will wrap on top of it and still call the
        # graphed transformer.forward if should_compute == True.
        self._setup_cuda_graphs()

    def _cuda_profiler_start(self):
        """Start CUDA profiler if configured and not already active."""
        if self._profile_range is not None and not self._profiling_active:
            torch.cuda.cudart().cudaProfilerStart()
            self._profiling_active = True
            if self.rank == 0:
                logger.info("CUDA profiler started")

    def _cuda_profiler_stop(self):
        """Stop CUDA profiler if currently active."""
        if self._profiling_active:
            torch.cuda.cudart().cudaProfilerStop()
            self._profiling_active = False
            if self.rank == 0:
                logger.info("CUDA profiler stopped")

    def _setup_cuda_graphs(self):
        """Wrap all transformer components with CUDA graph capture/replay.

        Composes with torch.compile: the runner wraps the (outer) transformer
        ``forward`` while torch.compile compiles the inner transformer blocks
        (see ``torch_compile``). Graph capture happens during warmup, by which
        point the runner's own ``WARMUP_STEPS`` eager iterations have already
        triggered torch.compile's lazy compilation, so the captured graph
        contains the optimized compiled kernels. (The ``LTX2Pipeline`` override
        relies on the same ordering.)
        """
        if not self.pipeline_config.cuda_graph.enable:
            return

        if len(self.transformer_components) > 1:
            logger.info(
                "CUDA graph runner: multiple transformer components, using shared graph pool"
            )
            shared_pool = SharedGraphPool()
        else:
            shared_pool = None

        compile_note = " (with torch.compile)" if self.pipeline_config.torch_compile.enable else ""
        for name in self.transformer_components:
            model = getattr(self, name, None)
            if model is None:
                continue

            runner = CUDAGraphRunner(
                CUDAGraphRunnerConfig(use_cuda_graph=True),
                shared_pool,
            )
            model.register_cuda_graph_extra_key_fns(runner)
            logger.info(f"CUDA graph runner: wrapping {name}.forward{compile_note}")
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
        return self.model_config.torch_dtype

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
        warmup_cfg = self.pipeline_config.compilation

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

    @property
    def extra_param_specs(self) -> Dict[str, ExtraParamSchema]:
        """Model-specific extra parameter specs.

        Subclasses override to declare which ``extra_params`` keys they
        accept and their metadata.  Maps parameter names to
        ``ExtraParamSchema`` instances.
        """
        return {}

    @property
    def default_generation_params(self) -> dict:
        """Model-specific defaults for ``None`` fields in ``VisualGenParams``.

        Keys should match ``VisualGenParams`` field names.  The executor
        merges these into ``request.params`` before calling ``infer()``.
        """
        return {}

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

    def _apply_teacache_coefficients(self, coefficients: Optional[Dict] = None) -> None:
        """Resolve TeaCache polynomial coefficients into pipeline_config.cache (TeaCacheConfig).

        Precedence:

        1. User-specified TeaCacheConfig.coefficients — any non-None list skips built-in
           variant matching.

        2. Pipeline table — if step 1 does not apply and coefficients is a non-empty dict
           (model-specific tables from the pipeline subclass), match
           pretrained_config._name_or_path against keys and set coefficients (and optional
           default_thresh).

        3. If coefficients are still unresolved after step 2, _setup_cache_acceleration
           raises: TeaCache must not run without resolved coefficients.

        Args:
            coefficients: Optional mapping from variant key to coefficient list or nested
                dict (ret_steps / standard), from the pipeline subclass.
        """
        teacache_cfg = self.pipeline_config.teacache
        if teacache_cfg is None:
            return
        if teacache_cfg.is_explicit_user_override():
            logger.info(
                "TeaCache: Using user-configured coefficients "
                "(skipping built-in checkpoint variant matching)"
            )
            return

        teacache_explicit = teacache_cfg.model_dump(exclude_unset=True)

        if not coefficients:
            if teacache_cfg.coefficients is None:
                raise ValueError(
                    "TeaCache is enabled but no polynomial coefficients were resolved. "
                    "Set teacache.coefficients in VisualGenArgs, or use a pipeline and "
                    "checkpoint whose path matches a built-in coefficient table."
                )
            return

        checkpoint_path = (
            getattr(self.pipeline_config.primary_pretrained_config, "_name_or_path", "") or ""
        )

        for model_size, coeff_data in coefficients.items():
            # Match model size in path (case-insensitive, e.g., "1.3B", "14B", "dev")
            path_l = checkpoint_path.lower()
            key_l = model_size.lower()
            if key_l not in path_l:
                continue

            if isinstance(coeff_data, dict):
                # Select coefficient set based on warmup mode
                mode = "ret_steps" if teacache_cfg.use_ret_steps else "standard"
                if mode not in coeff_data:
                    logger.warning(
                        "TeaCache: matched variant %r but table has no %r entry "
                        "(available keys: %s). Trying other variants.",
                        model_size,
                        mode,
                        list(coeff_data.keys()),
                    )
                    continue
                teacache_cfg.coefficients = coeff_data[mode]
                logger.info(f"TeaCache: Using {model_size} coefficients ({mode} mode)")
                # Apply model-specific default threshold if user didn't explicitly set one
                default_thresh = coeff_data.get("default_thresh")
                if default_thresh is not None and "teacache_thresh" not in teacache_explicit:
                    teacache_cfg.teacache_thresh = default_thresh
                    logger.info(f"TeaCache: Using {model_size} default threshold {default_thresh}")
                break
            else:
                # Single coefficient list (no mode distinction)
                teacache_cfg.coefficients = coeff_data
                logger.info(f"TeaCache: Using {model_size} coefficients")
                break
        else:
            raise ValueError(
                f"TeaCache: No coefficients found for checkpoint '{checkpoint_path}'. "
                f"Available variants: {list(coefficients.keys())}. "
                f"Set teacache.coefficients explicitly in VisualGenArgs to use TeaCache anyway, "
                f"or use a checkpoint path that contains one of the variant keys."
            )

    def _setup_cache_acceleration(self) -> None:
        """Enable TeaCache or Cache-DiT from model_config.cache_backend."""

        if getattr(self, "cache_accelerator", None) is not None:
            self.cache_accelerator.unwrap()
            self.cache_accelerator = None

        cfg = self.pipeline_config

        if cfg.cache_backend == "cache_dit":
            acc = CacheDiTAccelerator(self, cfg.cache_dit)
            acc.wrap()
            self.cache_accelerator = acc
            return

        use_teacache = cfg.cache_backend == "teacache"
        if not use_teacache:
            return

        acc = TeaCacheAccelerator(self, cfg.teacache)
        acc.wrap()
        self.cache_accelerator = acc

    def setup_parallel_vae(self):
        """Enable parallel-VAE decode mode and wrap the VAE on participating ranks.

        ``self._parallel_vae_enabled`` is a *global* mode flag: it is computed
        from inputs that are identical on every rank (config + mapping +
        deterministic capability check), so every rank agrees on whether
        parallel-VAE decode ownership applies. The actual ``ParallelVAEFactory``
        wrap is a local side effect that only runs on ranks in ``vae_ranks``.
        """
        parallel_cfg = self.pipeline_config.parallel
        vgm = self.pipeline_config.visual_gen_mapping

        # Global preconditions — evaluate identically on every rank.
        self._parallel_vae_enabled = (
            parallel_cfg.parallel_vae_size > 1
            and dist.is_initialized()
            and dist.get_world_size() > 1
            and self.vae is not None
            and vgm is not None
            and ParallelVAEFactory.supports(type(self.vae))
        )
        if not self._parallel_vae_enabled:
            # Quick check to see if it wasn't enabled due to missing support.
            if (
                parallel_cfg.parallel_vae_size > 1
                and self.vae is not None
                and not ParallelVAEFactory.supports(type(self.vae))
            ):
                logger.warning(
                    f"Parallel VAE not supported for {self.__class__.__name__} "
                    f"(VAE type: {type(self.vae).__name__}). "
                    "Add an entry to ParallelVAEFactory._LAZY_REGISTRY to enable "
                    "parallel VAE for this VAE type."
                )
            return

        # Local side effect: only ranks in the VAE group wrap the VAE module.
        if self.rank not in vgm.vae_ranks or vgm.vae_group is None:
            return

        self.vae = ParallelVAEFactory.from_vae(
            self.vae,
            split_dim=parallel_cfg.parallel_vae_split_dim,
            pg=vgm.vae_group,
            adj_groups=vgm.vae_adj_groups,
        )
        logger.info(
            f"Parallel VAE enabled: {type(self.vae).__name__}, "
            f"split_dim={parallel_cfg.parallel_vae_split_dim}, "
            f"world_size={dist.get_world_size(vgm.vae_group)}"
        )

    def torch_compile(self) -> None:
        """Apply torch.compile to pipeline components based on TorchCompileConfig.

        For transformer models, compiles each block in the ModuleList individually.
        This enables future block-wise offloading and keeps compilation efficient
        (all blocks share the same structure, so compile cost is paid once).

        For non-transformer components, compiles the entire module.
        """
        tc_config = self.pipeline_config.torch_compile

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

        self._is_warmup = True
        for height, width, num_frames in shapes:
            logger.info(f"Warmup: {height}x{width}, {num_frames} frames, {steps} steps")
            self._run_warmup(height, width, num_frames, steps)
            torch.cuda.synchronize()
        self._is_warmup = False

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
        """Execute VAE decoding.

        Decode ownership is decided from the global ``_parallel_vae_enabled``
        flag set by ``setup_parallel_vae``:

        - parallel-VAE mode on: ranks in ``vgm.vae_ranks`` decode collectively.
        - parallel-VAE mode off: only rank 0 decodes.

        Non-decoding ranks return ``None`` placeholders.

        Args:
            latents: Primary latents to decode (e.g., video).
            decode_fn: Decoder function for primary latents.
            extra_latents: Optional dict of additional latents to decode.
                Format: ``{name: (latents_tensor, decode_fn)}``.
                Example: ``{"audio": (audio_latents, audio_decode_fn)}``.

        Returns:
            Single result if no ``extra_latents``, else a tuple of results.
            Non-decoding ranks return ``None`` (or a tuple of ``None``).
        """
        if self._parallel_vae_enabled:
            vgm = self.pipeline_config.visual_gen_mapping
            decode_ranks = set(vgm.vae_ranks)
        else:
            decode_ranks = {0}

        if self.rank in decode_ranks:
            primary_result = decode_fn(latents)
            if extra_latents:
                extra_results = [efn(elat) for _, (elat, efn) in extra_latents.items()]
                return (primary_result,) + tuple(extra_results)
            return primary_result

        n_results = 1 + (len(extra_latents) if extra_latents else 0)
        return (None,) * n_results if n_results > 1 else None

    @staticmethod
    def _rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
        """Rescale noise to fix overexposure (https://huggingface.co/papers/2305.08891)."""
        std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
        std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
        noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
        return guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg

    @staticmethod
    def _resolve_step_guidance_scale(
        t: torch.Tensor,
        guidance_scale: float,
        guidance_interval: Optional[Tuple[float, float]] = None,
        guidance_scale_2: Optional[float] = None,
        boundary_timestep: Optional[float] = None,
    ) -> float:
        """Per-step CFG scale, including two-stage and guidance-interval gating."""
        current = guidance_scale
        t_scalar = t.item() if t.dim() == 0 else t[0].item()
        if guidance_scale_2 is not None and boundary_timestep is not None:
            if t_scalar < boundary_timestep:
                current = guidance_scale_2
        if guidance_interval is not None:
            interval_lo, interval_hi = guidance_interval
            if not (interval_lo <= t_scalar <= interval_hi):
                current = 1.0
        return current

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
        vgm = self.pipeline_config.visual_gen_mapping
        cfg_size = vgm.cfg_size if vgm else 1
        ulysses_size = vgm.ulysses_size if vgm else 1
        attn2d_row_size = vgm.attn2d_row_size if vgm else 1
        attn2d_col_size = vgm.attn2d_col_size if vgm else 1
        seq_parallel_size = vgm.seq_size if vgm is not None else 1

        is_conditional = vgm.is_cfg_conditional if vgm else True
        is_split_embeds = neg_prompt_embeds is not None
        do_cfg_parallel = cfg_size >= 2 and guidance_scale > 1.0

        local_extras = {}

        if do_cfg_parallel:
            if self.rank == 0:
                if attn2d_row_size * attn2d_col_size > 1:
                    logger.info(
                        f"CFG Parallel: cfg_size={cfg_size}, "
                        f"attn2d_row_size={attn2d_row_size}, attn2d_col_size={attn2d_col_size}, "
                        f"ulysses_size={ulysses_size}"
                    )

            # Split main embeddings
            if is_split_embeds:
                pos_embeds, neg_embeds = prompt_embeds, neg_prompt_embeds
            else:
                neg_embeds, pos_embeds = prompt_embeds.chunk(2)

            local_embeds = pos_embeds if is_conditional else neg_embeds

            # Split extra tensors if provided
            if extra_cfg_tensors:
                for name, (pos_tensor, neg_tensor) in extra_cfg_tensors.items():
                    if pos_tensor is not None and neg_tensor is not None:
                        local_extras[name] = pos_tensor if is_conditional else neg_tensor
                    elif pos_tensor is not None:
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
            "seq_parallel_size": seq_parallel_size,
            "local_embeds": local_embeds,
            "prompt_embeds": prompt_embeds,
            "local_extras": local_extras,
        }

    def _denoise_step_cfg_parallel(
        self,
        latents,
        extra_stream_latents,
        step_index,
        timestep,
        local_embeds,
        forward_fn,
        guidance_scale,
        guidance_rescale,
        seq_parallel_size,
        local_extras,
    ):
        """Execute single denoising step with CFG parallel."""
        vgm = self.pipeline_config.visual_gen_mapping
        cfg_pg = vgm.cfg_group if vgm else None
        cfg_size = vgm.cfg_size if vgm else 1

        t_start = time.time()
        result = forward_fn(
            latents,
            extra_stream_latents,
            step_index,
            timestep,
            local_embeds,
            local_extras,
        )

        # Handle return format: (primary_noise, extra_noises_dict) or just primary_noise
        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
            noise_pred_local, extra_noise_locals = result
        else:
            noise_pred_local = result
            extra_noise_locals = {}

        t_transformer = time.time() - t_start

        c_start = time.time()

        # All-gather primary noise over the CFG group.
        # Each entry in gather_list corresponds to one CFG rank
        # (index 0 = conditional, index 1 = unconditional).
        noise_pred_local = noise_pred_local.contiguous()
        gather_list = [torch.empty_like(noise_pred_local) for _ in range(cfg_size)]
        dist.all_gather(gather_list, noise_pred_local, group=cfg_pg)
        noise_cond = gather_list[0]
        noise_uncond = gather_list[1]
        noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

        # All-gather extra stream noises
        extra_noise_preds = {}
        for name, noise_local in extra_noise_locals.items():
            noise_local = noise_local.contiguous()
            gather_list_extra = [torch.empty_like(noise_local) for _ in range(cfg_size)]
            dist.all_gather(gather_list_extra, noise_local, group=cfg_pg)
            noise_cond_extra = gather_list_extra[0]
            noise_uncond_extra = gather_list_extra[1]
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
        step_index,
        timestep,
        prompt_embeds,
        forward_fn,
        guidance_scale,
        guidance_rescale,
        local_extras,
        do_cfg: bool = False,
    ):
        """Execute single denoising step without CFG parallel."""
        if do_cfg:
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
            latent_input,
            extra_stream_input,
            step_index,
            timestep_expanded,
            prompt_embeds,
            local_extras,
        )

        # Handle return format: (primary_noise, extra_noises_dict) or just primary_noise
        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
            noise_pred, extra_noise_preds = result
        else:
            noise_pred = result
            extra_noise_preds = {}

        t_transformer = time.time() - t_start

        c_start = time.time()
        if do_cfg:
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
        guidance_interval: Optional[Tuple[float, float]] = None,
        post_step_fn: Optional[Callable] = None,
    ):
        """Execute denoising loop with optional CFG parallel and TeaCache support.

        Args:
            latents: Initial noise latents (primary stream, e.g., video)
            scheduler: Diffusion scheduler for primary stream
            prompt_embeds: Text embeddings (positive)
            guidance_scale: CFG strength (1.0 = no guidance)
            forward_fn: Transformer forward function
                       Signature: forward_fn(latents, extra_stream_latents, step_index,
                                            timestep, encoder_hidden_states,
                                            extra_tensors_dict)
                       step_index is the ordinal denoising-loop index, distinct
                       from the scheduler timestep value.
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
            guidance_interval: Optional ``(lo, hi)`` scheduler-timestep range in which CFG
                              is active. Outside the interval the effective scale is 1.0
                              (conditional prediction only); both branches still run.
            post_step_fn: Optional callable applied to latents after each scheduler step.
                         Signature: post_step_fn(latents) -> latents
                         Use for constraints that must hold throughout denoising.

        Returns:
            Single latents if no extra_streams
            Tuple (primary_latents, extra_streams_dict) if extra_streams provided
        """
        # ``predenoise`` mode: arm the profiler at the very start of denoise()
        # so the per-request pre-loop work (CFG config, scheduler refresh,
        # TeaCache reset) is captured. The window closes at the first step.
        # Note: hooked here (not at warmup() exit) to avoid leaving the profiler
        # on across the worker's IPC idle, which can interact badly with CUPTI.
        if self._predenoise_pending and not self._is_warmup:
            self._cuda_profiler_start()

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
        do_cfg = guidance_scale > 1.0
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

        # CUDA profiler scoping: "all" starts here (covers denoise + VAE),
        # step ranges start/stop at specific indices. See _parse_profile_range().
        prof = self._profile_range
        if prof == "all" and not self._is_warmup:
            self._cuda_profiler_start()
        # ``predenoise`` was started in warmup() exit; close the window now,
        # before the first denoise step kernels run. Single-shot: disarm.
        if self._predenoise_pending and not self._is_warmup:
            self._cuda_profiler_stop()
            self._predenoise_pending = False
        prof_step_starts = prof[0] if isinstance(prof, tuple) else None
        prof_step_stops = prof[1] if isinstance(prof, tuple) else None

        for i, t in enumerate(timesteps):
            if prof_step_starts is not None and i in prof_step_starts and not self._is_warmup:
                self._cuda_profiler_start()

            step_start = time.time()

            current_guidance_scale = self._resolve_step_guidance_scale(
                t,
                guidance_scale,
                guidance_interval,
                guidance_scale_2,
                boundary_timestep,
            )

            # Denoise
            with nvtx_range(f"denoise_step {i}"):
                if do_cfg_parallel:
                    timestep = t.expand(latents.shape[0])
                    (
                        noise_pred,
                        extra_noise_preds,
                        t_trans,
                        t_cfg,
                    ) = self._denoise_step_cfg_parallel(
                        latents,
                        extra_stream_latents,
                        i,
                        timestep,
                        cfg_config["local_embeds"],
                        forward_fn,
                        current_guidance_scale,
                        guidance_rescale,
                        cfg_config["seq_parallel_size"],
                        local_extras,
                    )
                else:
                    (
                        noise_pred,
                        extra_noise_preds,
                        t_trans,
                        t_cfg,
                    ) = self._denoise_step_standard(
                        latents,
                        extra_stream_latents,
                        i,
                        t,
                        prompt_embeds,
                        forward_fn,
                        current_guidance_scale,
                        guidance_rescale,
                        local_extras,
                        do_cfg=do_cfg,
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

            if post_step_fn is not None:
                latents = post_step_fn(latents)

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

            # Step-level profiler stop
            if prof_step_stops is not None and i in prof_step_stops and not self._is_warmup:
                self._cuda_profiler_stop()

        # ``postdenoise`` mode: arm the profiler now so the VAE decode (and
        # any post-denoise host work) is captured up to cleanup(). Single-shot.
        if self._postdenoise_pending and not self._is_warmup:
            self._cuda_profiler_start()
            self._postdenoise_pending = False

        if self.rank == 0:
            total_time = time.time() - start_time
            logger.info("=" * 80)
            logger.info(f"Denoising done: {total_time:.2f}s ({total_time / total_steps:.2f}s/step)")

            # Single logging site for TeaCache and Cache-DiT.
            if getattr(self, "cache_accelerator", None) and self.cache_accelerator.is_enabled():
                stats = self.cache_accelerator.get_stats()
                if stats:
                    if self.pipeline_config.cache_backend == "cache_dit":
                        logger.info("Cache-DiT stats: %s", stats)
                    elif self.pipeline_config.cache_backend == "teacache":
                        first_val = next(iter(stats.values()), None)
                        if isinstance(first_val, dict):
                            for key, s in stats.items():
                                if "hit_rate" in s:
                                    logger.info(
                                        f"TeaCache {key}: {s['hit_rate']:.1%} hit rate "
                                        f"({s['cached']}/{s['total']} steps)"
                                    )
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
        self._cuda_profiler_stop()

        for name, runner in self._cuda_graph_runners.items():
            logger.info(f"Releasing CUDA graphs for {name}")
            runner.clear()
