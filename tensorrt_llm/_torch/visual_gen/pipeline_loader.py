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
from typing import TYPE_CHECKING, Optional

import torch

from tensorrt_llm._torch.models.modeling_utils import MetaInitMode
from tensorrt_llm.llmapi.utils import download_hf_model
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

from .checkpoints import WeightLoader
from .config import DiffusionArgs, DiffusionModelConfig, PipelineComponent
from .models import AutoPipeline

if TYPE_CHECKING:
    from .models import BasePipeline


class PipelineLoader:
    """
    Loader for diffusion pipelines.

    Supports dynamic quantization: when quant_config specifies FP8/NVFP4,
    model is built with quantized buffers and BF16 weights are quantized
    on-the-fly during loading.

    Example:
        args = DiffusionArgs(
            checkpoint_path="/path/to/model",
            linear=LinearConfig(type="trtllm-fp8-blockwise"),
            parallel=ParallelConfig(dit_tp_size=2),
        )
        pipeline = PipelineLoader(args).load()
    """

    def __init__(
        self,
        args: Optional[DiffusionArgs] = None,
        *,
        mapping: Optional[Mapping] = None,
        device: str = "cuda",
    ):
        """
        Initialize model loader.

        Args:
            args: DiffusionArgs containing all configuration (preferred)
            mapping: Tensor parallel mapping (fallback if args is None)
            device: Device to load model on (fallback if args is None)
        """
        self.args = args
        if args is not None:
            self.mapping = args.to_mapping()
            self.device = torch.device(args.device)
        else:
            self.mapping = mapping or Mapping()
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

    def load(
        self,
        checkpoint_dir: Optional[str] = None,
    ) -> "BasePipeline":
        """
        Load a diffusion pipeline with optional dynamic quantization.

        Flow:
        1. Resolve checkpoint_dir (local path or HuggingFace Hub model ID)
        2. Load config via DiffusionModelConfig.from_pretrained()
        3. Create pipeline via AutoPipeline.from_config() with MetaInit
        4. Load transformer weights via pipeline.load_weights()
        5. Load auxiliary components (VAE, text_encoder) via diffusers
        6. Call pipeline.post_load_weights()

        Args:
            checkpoint_dir: Local path or HF Hub model ID (uses args.checkpoint_path if not provided)

        Returns:
            Loaded pipeline (WanPipeline, FluxPipeline, etc.) - type auto-detected
        """
        # Resolve checkpoint_dir
        checkpoint_dir = checkpoint_dir or (self.args.checkpoint_path if self.args else None)
        if not checkpoint_dir:
            raise ValueError("checkpoint_dir must be provided or set in DiffusionArgs")
        checkpoint_dir = self._resolve_checkpoint_dir(str(checkpoint_dir))

        # Get loading options from args
        skip_components = self.args.skip_components if self.args else []

        # =====================================================================
        # STEP 1: Load Config (includes quant config parsing)
        # Merge pretrained checkpoint config with user-provided DiffusionArgs
        # =====================================================================
        logger.info(f"Loading config from {checkpoint_dir}")
        config = DiffusionModelConfig.from_pretrained(
            checkpoint_dir,
            args=self.args,
            mapping=self.mapping,
        )

        # Log quantization settings
        if config.quant_config and config.quant_config.quant_algo:
            logger.info(f"Quantization: {config.quant_config.quant_algo.name}")
            logger.info(f"Dynamic weight quant: {config.dynamic_weight_quant}")

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
        # If dynamic_weight_quant=True:
        #   - BF16 checkpoint weights are loaded
        #   - Quantized on-the-fly to FP8/NVFP4 by DynamicLinearWeightLoader
        #   - Copied into model's quantized buffers
        # =====================================================================
        if pipeline.transformer is None:
            raise ValueError("Pipeline has no transformer component")

        transformer_components = getattr(pipeline, "transformer_components", ["transformer"])
        logger.info(f"Transformer components: {transformer_components}")

        transformer_path = os.path.join(checkpoint_dir, PipelineComponent.TRANSFORMER)
        if not os.path.exists(transformer_path):
            raise FileNotFoundError(
                f"Transformer path does not exist: {transformer_path}. "
                f"Checkpoint directory must contain a 'transformer' subdirectory."
            )

        weight_loader = WeightLoader(components=transformer_components)
        # TODO: accelerate the cpu loading w/ multiprocessing
        weights = weight_loader.load_weights(checkpoint_dir, self.mapping)

        # Load weights into pipeline
        pipeline.load_weights(weights)

        # =====================================================================
        # STEP 4: Load Standard Components (VAE, TextEncoder via diffusers)
        # These are NOT quantized - loaded as-is from checkpoint
        # =====================================================================
        pipeline.load_standard_components(checkpoint_dir, self.device, skip_components)

        # =====================================================================
        # STEP 5: Post-load Hooks (TeaCache setup, etc.)
        # =====================================================================
        if hasattr(pipeline, "post_load_weights"):
            pipeline.post_load_weights()

        logger.info(f"Pipeline loaded: {pipeline.__class__.__name__}")
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
                memo[t] = torch.empty_like(t, device="cuda")
            return memo[t]

        module._apply(init_meta_tensor)
