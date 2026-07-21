"""
Model loader for diffusion pipelines.

Flow:
1. Load config via DiffusionPipelineConfig.from_pretrained()
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
from typing import TYPE_CHECKING, List, Optional, Union

import torch
import torch.distributed as dist

from tensorrt_llm._torch.autotuner import autotune
from tensorrt_llm._torch.models.modeling_utils import MetaInitMode
from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.blackwell.video_sparse_attention import (
    CUTE_AVAILABLE,
)
from tensorrt_llm.llmapi.utils import download_hf_model
from tensorrt_llm.logger import logger
from tensorrt_llm.visual_gen.args import VisualGenArgs

from .config import DiffusionPipelineConfig
from .mapping import VisualGenMapping
from .models import AutoPipeline
from .pipeline_registry import PIPELINE_REGISTRY, PipelineComponent

if TYPE_CHECKING:
    from .models import BasePipeline


class PipelineLoader:
    """
    Loader for diffusion pipelines.

    Supports dynamic quantization: when quant_config specifies FP8/NVFP4,
    model is built with quantized buffers and BF16 weights are quantized
    on-the-fly during loading.

    Example:
        args = VisualGenArgs(
            model="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            parallel_config=ParallelConfig(ulysses_size=2),
        )
        pipeline = PipelineLoader(args).load()
    """

    def __init__(
        self,
        args: VisualGenArgs,
        *,
        device: str = "cuda",
    ):
        """
        Initialize model loader.

        Args:
            args: VisualGenArgs containing all configuration.
            device: CUDA device to load the model on (e.g., "cuda:0").
                Per-rank device routing is the caller's responsibility;
                the engine config itself is device-agnostic.
        """
        self.args = args
        self.device = torch.device(device)

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

        revision = self.args.revision
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

    def _resolve_pipeline_config(self, checkpoint_dir: str) -> dict:
        """Validate VisualGenArgs.pipeline_config against the registry and merge.

        The user-facing dict on ``VisualGenArgs.pipeline_config`` is strict:
        keys must appear in the resolved pipeline family's ``defaults``
        (the schema-by-example carried on the registry entry). Unknown
        keys raise immediately so typos surface at load time. The merged
        dict is ``{**entry.defaults, **user_dict}`` — user-supplied values
        win.
        """
        user_pipeline_config = dict(self.args.pipeline_config)

        # Detect _class_name from the resolved checkpoint, look up the
        # registry entry. If detection fails (or the class_name isn't
        # registered) leave the validation to AutoPipeline.from_config
        # so the user gets the existing "Unknown pipeline" error rather
        # than a confusing pipeline_config validation error.
        try:
            class_name = AutoPipeline._detect_from_checkpoint(checkpoint_dir)
        except ValueError:
            return user_pipeline_config

        entry = PIPELINE_REGISTRY.get(class_name)
        if entry is None:
            return user_pipeline_config

        unknown = set(user_pipeline_config) - set(entry.defaults)
        if unknown:
            raise ValueError(
                f"Unknown pipeline_config keys for {class_name} ({checkpoint_dir}): "
                f"{sorted(unknown)}. Valid keys: {sorted(entry.defaults)}"
            )
        return {**entry.defaults, **user_pipeline_config}

    def _setup_visual_gen_mapping(self, config: DiffusionPipelineConfig) -> None:
        ws = dist.get_world_size() if dist.is_initialized() else 1
        rk = dist.get_rank() if dist.is_initialized() else 0
        attn2d_row, attn2d_col = self.args.parallel_config.attn2d_size
        vgm = VisualGenMapping(
            ws,
            rk,
            cfg_size=self.args.parallel_config.cfg_size,
            ulysses_size=self.args.parallel_config.ulysses_size,
            ring_size=self.args.parallel_config.ring_size,
            attn2d_row_size=attn2d_row,
            attn2d_col_size=attn2d_col,
            tp_size=self.args.parallel_config.tp_size,
            parallel_vae_size=self.args.parallel_config.parallel_vae_size,
        )
        llm_mapping = vgm.to_llm_mapping()
        config.visual_gen_mapping = vgm
        config.mapping = llm_mapping
        for model_config in config.model_configs.values():
            model_config.visual_gen_mapping = vgm
            model_config.mapping = llm_mapping

    def load(
        self,
        checkpoint_dir: Optional[str] = None,
        skip_warmup: bool = False,
        skip_components: Optional[List[Union[str, PipelineComponent]]] = None,
    ) -> "BasePipeline":
        """
        Load a diffusion pipeline with optional dynamic quantization.

        Flow:
        1. Resolve checkpoint_dir (local path or HuggingFace Hub model ID)
        2. Load config via DiffusionPipelineConfig.from_pretrained()
        3. Create pipeline via AutoPipeline.from_config() with MetaInit
        4. Load transformer weights via pipeline.load_transformer_weights()
        5. Load auxiliary components (VAE, text_encoder)
        6. Call pipeline.post_load_weights()

        Args:
            checkpoint_dir: Local path or HF Hub model ID (uses ``args.model`` if not provided)
            skip_warmup: If True, skip warmup inference after loading (useful for testing)
            skip_components: Optional internal escape hatch — list of
                ``PipelineComponent`` values (or their string equivalents) to
                skip loading. Intended for memory-constrained unit tests
                that only exercise the transformer; not part of the public
                ``VisualGenArgs`` surface.

        Returns:
            Loaded pipeline (WanPipeline, FluxPipeline, etc.) - type auto-detected
        """
        checkpoint_dir = checkpoint_dir or self.args.model
        if not checkpoint_dir:
            raise ValueError("checkpoint_dir must be provided or set in VisualGenArgs")
        checkpoint_dir = self._resolve_checkpoint_dir(str(checkpoint_dir))

        # Strict pipeline_config validation: detect _class_name from the
        # checkpoint, look up the registry entry, reject unknown keys.
        resolved_pipeline_config = self._resolve_pipeline_config(checkpoint_dir)

        load_start = time.time()
        # text_encoder_path is an LTX-2 pipeline_config knob; it lives in
        # the merged dict that _resolve_pipeline_config produced, not on
        # VisualGenArgs directly.
        text_encoder_path = resolved_pipeline_config.get("text_encoder_path", "")

        # =====================================================================
        # STEP 1: Load Config (includes quant config parsing)
        # Merge pretrained checkpoint config with user-provided VisualGenArgs
        # =====================================================================
        logger.info(f"Loading config from {checkpoint_dir}")
        config = DiffusionPipelineConfig.from_pretrained(
            checkpoint_dir,
            args=self.args,
            pipeline_config=resolved_pipeline_config,
        )

        # Log quantization settings
        if config.quant_config and config.quant_config.quant_algo:
            logger.info(f"Quantization: {config.quant_config.quant_algo.name}")
            logger.info(f"Dynamic weight quant: {config.dynamic_weight_quant}")

        _attn_backend = config.attention.backend
        _sa_cfg = config.attention.sparse_attention_config
        if (
            _attn_backend == "CUTEDSL"
            and _sa_cfg is not None
            and getattr(_sa_cfg, "algorithm", None) == "vsa"
        ):
            kernel_path = "CuTe DSL block-sparse" if CUTE_AVAILABLE else "dense SDPA fallback"
            logger.info(
                f"Attention backend: CUTEDSL (algorithm=vsa, "
                f"sparsity={_sa_cfg.vsa_sparsity}, fine-stage={kernel_path})"
            )
        else:
            logger.info(f"Attention backend: {_attn_backend}")

        # =====================================================================
        # STEP 1b: Build VisualGenMapping (must precede model creation)
        # =====================================================================
        self._setup_visual_gen_mapping(config)

        # =====================================================================
        # STEP 2: Create Pipeline with MetaInit
        # Pipeline type is auto-detected from model_index.json
        # - Meta tensors (no GPU memory until materialization)
        # - If quant_config specifies FP8, Linear layers have FP8 weight buffers
        # =====================================================================
        logger.info("Creating pipeline with MetaInitMode")
        with MetaInitMode():
            pipeline = AutoPipeline.from_config(config, checkpoint_dir)

        # Convert meta tensors to CUDA tensors
        self._materialize_meta_tensors(pipeline)
        pipeline.to(self.device)

        # =====================================================================
        # STEP 3: Load Transformer Weights
        # Each pipeline implements load_transformer_weights() for its own
        # checkpoint format.  The default (BasePipeline) uses WeightLoader
        # for diffusers-compatible checkpoints with a transformer/ subdir.
        # If dynamic_weight_quant=True:
        #   - BF16 checkpoint weights are loaded
        #   - Quantized on-the-fly to FP8/NVFP4 by DynamicLinearWeightLoader
        #   - Copied into model's quantized buffers
        # =====================================================================
        weights = pipeline.load_transformer_weights(checkpoint_dir)
        pipeline.load_weights(weights)

        # =====================================================================
        # STEP 4: Load Standard Components (VAE, TextEncoder, etc.)
        # These are NOT quantized - loaded as-is from checkpoint
        # =====================================================================
        extra_kwargs = {}
        if text_encoder_path:
            extra_kwargs["text_encoder_path"] = text_encoder_path
        pipeline.load_standard_components(
            checkpoint_dir,
            self.device,
            skip_components=skip_components,
            **extra_kwargs,
        )
        logger.info(f"Model loaded successfully in {time.time() - load_start:.2f}s")

        # =====================================================================
        # STEP 5: Post-load Hooks (TeaCache setup, etc.)
        # =====================================================================

        t0 = time.time()
        if config.parallel.parallel_vae_size > 1:
            pipeline.setup_parallel_vae()

        if hasattr(pipeline, "post_load_weights"):
            pipeline.post_load_weights()

        if config.torch_compile.enable:
            torch._dynamo.config.cache_size_limit = 128
            pipeline.torch_compile()
        else:
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

        if config.enable_layerwise_nvtx_marker:
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
                memo[t] = torch.empty_like(t, device=self.device)
            return memo[t]

        module._apply(init_meta_tensor)
