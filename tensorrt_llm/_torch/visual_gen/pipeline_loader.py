"""
Model loader for diffusion pipelines.

Flow:
1. Load config via DiffusionModelConfig.from_pretrained()
2. Create pipeline via AutoPipeline.from_config() with MetaInit
3. Load weights with on-the-fly quantization if dynamic_weight_quant=True
4. Call pipeline.post_load_weights()

Dynamic Quantization:
- If quant_config specifies FP8/NVFP4 and dynamic_weight_quant=True:
  - Model Linear layers are created with FP8/NVFP4 buffers
  - BF16 checkpoint weights are quantized on-the-fly during loading
  - Quantized weights are copied into model buffers
"""

import os
import time
from typing import Optional

import torch
import torch.distributed as dist

from tensorrt_llm._torch.autotuner import autotune
from tensorrt_llm._torch.models.modeling_utils import MetaInitMode
from tensorrt_llm.llmapi.utils import download_hf_model
from tensorrt_llm.logger import logger

from .config import DiffusionModelConfig, VisualGenArgs
from .mapping import VisualGenMapping
from .models import BasePipeline


class PipelineLoader:
    """
    Loader for diffusion pipelines.

    Supports dynamic quantization: when quant_config specifies FP8/NVFP4,
    model is built with quantized buffers and BF16 weights are quantized
    on-the-fly during loading.

    Example:
        args = VisualGenArgs(
            checkpoint_path="/path/to/model",
            linear=LinearConfig(type="trtllm-fp8-blockwise"),
            parallel=ParallelConfig(dit_tp_size=2),
        )
        pipeline = PipelineLoader(args).load()
    """

    def __init__(
        self,
        args: Optional[VisualGenArgs] = None,
        *,
        device: str = "cuda",
    ):
        """
        Initialize model loader.

        Args:
            args: VisualGenArgs containing all configuration (preferred)
            device: Device to load model on (fallback if args is None)
        """
        self.args = args
        self.device = torch.device(args.device if args is not None else device)

    def _resolve_checkpoint_dir(self, checkpoint_dir: str) -> str:
        """Resolve checkpoint_dir to a local directory path.

        If checkpoint_dir is an existing local path, returns it unchanged.
        Otherwise, attempts to download from HuggingFace Hub using the
        file-lock-protected ``download_hf_model`` utility (safe for
        concurrent multi-process access).

        Args:
            checkpoint_dir: Local path or HuggingFace Hub model ID.

        Returns:
            Path to local directory containing the model.

        Raises:
            ValueError: If the path cannot be resolved (invalid repo ID,
                authentication failure, offline with no cache, etc.)
        """
        if os.path.exists(checkpoint_dir):
            return checkpoint_dir

        revision = self.args.revision if self.args else None
        logger.info(
            f"'{checkpoint_dir}' not found locally; "
            f"attempting HuggingFace Hub download (revision={revision})"
        )
        try:
            local_dir = download_hf_model(checkpoint_dir, revision=revision)
        except Exception as e:
            raise ValueError(
                f"Could not resolve '{checkpoint_dir}' as a local path or "
                f"HuggingFace Hub model ID: {e}"
            ) from e
        return str(local_dir)

    def _setup_visual_gen_mapping(self, config: DiffusionModelConfig) -> None:
        if self.args is not None:
            ws = dist.get_world_size() if dist.is_initialized() else 1
            rk = dist.get_rank() if dist.is_initialized() else 0
            vgm = VisualGenMapping(
                ws,
                rk,
                cfg_size=self.args.parallel.dit_cfg_size,
                tp_size=self.args.parallel.dit_tp_size,
                ulysses_size=self.args.parallel.dit_ulysses_size,
                ring_size=self.args.parallel.dit_ring_size,
                attn2d_row_size=self.args.parallel.dit_attn2d_row_size,
                attn2d_col_size=self.args.parallel.dit_attn2d_col_size,
                parallel_vae_size=self.args.parallel.parallel_vae_size,
                order=self.args.parallel.dit_dim_order,
            )
        else:
            # Single-GPU fallback. no args = no parallelism.
            vgm = VisualGenMapping(world_size=1, rank=0)
        config.visual_gen_mapping = vgm
        config.mapping = vgm.to_llm_mapping()

    def load(
        self,
        checkpoint_dir: Optional[str] = None,
        skip_warmup: bool = False,
    ):
        """
        Load a diffusion pipeline. Returns either a `BasePipeline` (handwritten
        path) or an `AutoTransformerPipeline` (auto path) — the two are NOT
        in the same class hierarchy; callers use `isinstance(pipeline,
        BasePipeline)` to gate handwritten-only operations.

        Flow:
        1. Resolve checkpoint_dir (local path or HuggingFace Hub model ID).
        2. Load config via DiffusionModelConfig.from_pretrained().
        3. `_create_pipeline` branches on auto vs handwritten:
           - Handwritten: MetaInitMode wrap → materialize → load weights →
             load standard components → post-load hooks → torch.compile →
             warmup.
           - Auto: skip MetaInitMode (Diffusers from_pretrained loads weights
             eagerly), do all load work in __init__, then warmup. The auto
             path silently ignores `args.text_encoder_path` and
             `args.skip_components` (Diffusers' from_pretrained owns
             standard-component loading) — warnings emitted below.

        Args:
            checkpoint_dir: Local path or HF Hub model ID (uses args.checkpoint_path if not provided)
            skip_warmup: If True, skip warmup inference after loading (useful for testing)

        Returns:
            Loaded pipeline — `BasePipeline` subclass or `AutoTransformerPipeline`.
        """
        # Resolve checkpoint_dir
        checkpoint_dir = checkpoint_dir or (self.args.checkpoint_path if self.args else None)
        if not checkpoint_dir:
            raise ValueError("checkpoint_dir must be provided or set in VisualGenArgs")
        checkpoint_dir = self._resolve_checkpoint_dir(str(checkpoint_dir))

        # Get loading options from args
        skip_components = self.args.skip_components if self.args else []

        load_start = time.time()
        text_encoder_path = self.args.text_encoder_path if self.args else ""

        # =====================================================================
        # STEP 1: Load Config (includes quant config parsing)
        # Merge pretrained checkpoint config with user-provided VisualGenArgs
        # =====================================================================
        logger.info(f"Loading config from {checkpoint_dir}")
        config = DiffusionModelConfig.from_pretrained(
            checkpoint_dir,
            args=self.args,
        )

        # Log quantization settings
        if config.quant_config and config.quant_config.quant_algo:
            logger.info(f"Quantization: {config.quant_config.quant_algo.name}")
            logger.info(f"Dynamic weight quant: {config.dynamic_weight_quant}")

        # =====================================================================
        # STEP 1b: Build VisualGenMapping (must precede model creation)
        # =====================================================================
        self._setup_visual_gen_mapping(config)

        # =====================================================================
        # STEP 2: Create Pipeline
        # Pipeline type is auto-detected from model_index.json.
        # - Handwritten path: meta-init wrap so Linear layers don't allocate
        #   GPU buffers until step 2b's materialize. Weights are loaded in
        #   step 3 / step 4 below.
        # - Auto path (AutoTransformerPipeline): Diffusers'
        #   `DiffusionPipeline.from_pretrained` loads all weights eagerly in
        #   one call, which is incompatible with meta-tensor dispatch — so we
        #   skip MetaInitMode entirely for the auto branch. The auto path
        #   does ALL load work inside `__init__`.
        # =====================================================================
        pipeline = self._create_pipeline(config, checkpoint_dir)
        is_handwritten = isinstance(pipeline, BasePipeline)

        # The auto path's component-loading is owned by Diffusers'
        # `from_pretrained` (inside `AutoDiffusersPipeline.__init__`); the
        # loader's `text_encoder_path` / `skip_components` knobs do not apply.
        # Warn loudly if the user set them — they will be silently dropped.
        if not is_handwritten:
            if text_encoder_path:
                logger.warning(
                    f"AutoTransformerPipeline ignores VisualGenArgs.text_encoder_path"
                    f"={text_encoder_path!r}; Diffusers' from_pretrained loads "
                    "the text encoder from the checkpoint directory directly. "
                    "Use a Diffusers-shaped checkpoint with the desired text "
                    "encoder co-located, or override after construction via "
                    "`pipeline._inner._pipe.text_encoder = ...`."
                )
            if skip_components:
                logger.warning(
                    f"AutoTransformerPipeline ignores VisualGenArgs.skip_components"
                    f"={list(skip_components)!r}; Diffusers' from_pretrained loads "
                    "all components eagerly. To skip a component, edit the "
                    "checkpoint's `model_index.json` or replace the component "
                    "post-construction."
                )

        # ----- Handwritten-only lifecycle (steps 2b-4) ------------------------
        if is_handwritten:
            # Convert meta tensors to CUDA tensors
            self._materialize_meta_tensors(pipeline)
            pipeline.to(self.device)

            # STEP 3: Load transformer weights (handwritten checkpoint format).
            weights = pipeline.load_transformer_weights(checkpoint_dir)
            pipeline.load_weights(weights)

            # STEP 4: Load standard components (VAE, TextEncoder, etc.).
            extra_kwargs = {}
            if text_encoder_path:
                extra_kwargs["text_encoder_path"] = text_encoder_path
            pipeline.load_standard_components(
                checkpoint_dir,
                self.device,
                skip_components,
                **extra_kwargs,
            )

        logger.info(f"Model loaded successfully in {time.time() - load_start:.2f}s")

        # =====================================================================
        # STEP 5: Post-load hooks
        # Common hooks run on both paths; handwritten-only hooks gated on
        # `is_handwritten` to keep the auto path from advertising features it
        # silently no-ops.
        # =====================================================================
        t0 = time.time()
        if is_handwritten and config.parallel.parallel_vae_size > 1:
            pipeline.setup_parallel_vae()

        if hasattr(pipeline, "post_load_weights"):
            pipeline.post_load_weights()

        if is_handwritten and config.torch_compile.enable_torch_compile:
            torch._dynamo.config.cache_size_limit = 128
            pipeline.torch_compile()
        elif is_handwritten:
            logger.info("torch.compile disabled by config")

        if not skip_warmup:
            if config.torch_compile.enable_autotune:
                with autotune(
                    cache_path=os.environ.get("TLLM_AUTOTUNER_CACHE_PATH"),
                    skip_dynamic_tuning_buckets=True,
                ):
                    pipeline.warmup()
            else:
                pipeline.warmup()
            logger.info(f"Warmup completed in {time.time() - t0:.2f}s")
        else:
            logger.info("Warmup skipped (skip_warmup=True)")

        if config.pipeline.enable_layerwise_nvtx_marker:
            if not is_handwritten:
                # `transformer_components` is a `BasePipeline` concept (multi-
                # stream pipelines like WAN 2.2 declare both `transformer` and
                # `transformer_2`); the auto path has only the captured
                # GraphModule and doesn't expose the same enumeration. Skip
                # the marker on the auto path with an info log rather than
                # crashing on AttributeError.
                logger.info(
                    "enable_layerwise_nvtx_marker=True ignored on the auto path "
                    "(no `transformer_components` enumeration)"
                )
            else:
                from tensorrt_llm._torch.pyexecutor.layerwise_nvtx_marker import LayerwiseNvtxMarker

                marker = LayerwiseNvtxMarker()
                module_prefix = pipeline.__class__.__name__
                for transformer_component in pipeline.transformer_components:
                    logger.info(f"Registering layerwise NVTX markers for {transformer_component}")
                    marker.register_hooks(getattr(pipeline, transformer_component), module_prefix)

        logger.info(
            f"Pipeline loaded: {pipeline.__class__.__name__} "
            f"(total load time: {time.time() - load_start:.2f}s)"
        )
        return pipeline

    def _create_pipeline(self, config: DiffusionModelConfig, checkpoint_dir: str):
        """Build the pipeline instance, branching on auto vs handwritten path.

        The handwritten path is constructed inside `MetaInitMode` so its
        Linear layers don't allocate GPU buffers until step 2b's materialize.
        The auto path is constructed *outside* MetaInitMode because Diffusers'
        `DiffusionPipeline.from_pretrained` loads weights eagerly (and meta
        tensors confuse it).

        Dispatch (`AutoPipeline.from_config`) reuses the existing handwritten
        registry; the auto branch returns an `AutoTransformerPipeline` which is
        NOT a `BasePipeline` subclass — the caller uses `isinstance` to gate
        handwritten-only lifecycle hooks.
        """
        # Resolve which path will be taken so we know whether MetaInitMode
        # applies, without duplicating the registry's dispatch logic.
        from .auto.pipeline import AutoTransformerPipeline
        from .pipeline_registry import AutoPipeline as _AP

        target_cls = _AP.resolve_target_class(config, checkpoint_dir)
        if target_cls is AutoTransformerPipeline:
            logger.info("Creating AutoTransformerPipeline (no MetaInitMode)")
            return AutoTransformerPipeline(config, checkpoint_dir)
        logger.info(f"Creating {target_cls.__name__} with MetaInitMode")
        with MetaInitMode():
            return target_cls(config)

    def _materialize_meta_tensors(self, module: torch.nn.Module) -> None:
        """
        Convert meta tensors to CUDA tensors.

        Meta tensors are placeholders that don't allocate GPU memory.
        After model structure is defined, we materialize them to real tensors.
        """
        memo = {}

        def init_meta_tensor(t: torch.Tensor) -> torch.Tensor:
            if t.device != torch.device("meta"):
                return t
            if t not in memo:
                memo[t] = torch.empty_like(t, device="cuda")
            return memo[t]

        module._apply(init_meta_tensor)
