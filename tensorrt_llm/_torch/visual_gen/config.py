import json
import os
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Literal, Optional, Tuple

import torch
from pydantic import BaseModel, ConfigDict, model_validator
from pydantic import Field as PydanticField

from tensorrt_llm.functional import AllReduceStrategy
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

# =============================================================================
# Pipeline component identifiers
# =============================================================================


class PipelineComponent(str, Enum):
    """Identifiers for pipeline components that can be loaded or skipped.

    Inherits from str so values compare equal to plain strings,
    e.g. ``PipelineComponent.VAE == "vae"`` is ``True``.
    """

    TRANSFORMER = "transformer"
    VAE = "vae"
    TEXT_ENCODER = "text_encoder"
    TOKENIZER = "tokenizer"
    SCHEDULER = "scheduler"
    IMAGE_ENCODER = "image_encoder"
    IMAGE_PROCESSOR = "image_processor"


# =============================================================================
# Sub-configuration classes for DiffusionArgs
# =============================================================================


class AttentionConfig(BaseModel):
    """Configuration for Attention layers."""

    backend: Literal["VANILLA", "TRTLLM"] = PydanticField(
        "VANILLA", description="Attention backend: VANILLA (PyTorch SDPA), TRTLLM"
    )


class ParallelConfig(BaseModel):
    """Configuration for distributed parallelism.

    Currently Supported:
        - dit_cfg_size: CFG (Classifier-Free Guidance) parallelism
        - dit_ulysses_size: Ulysses sequence parallelism

    Not Yet Supported:
        - dit_tp_size: Tensor parallelism (not implemented)
        - dit_ring_size: Ring attention (not implemented)
        - dit_cp_size, dit_dp_size, dit_fsdp_size: Other parallelism types

    Total world_size = dit_cfg_size × dit_ulysses_size

    Parallelism Strategy:
        - CFG Parallelism: Distributes positive/negative prompts across GPUs
        - Ulysses Parallelism: Distributes sequence within each CFG group

    Example Configurations:
        1. cfg_size=1, ulysses_size=2 -> 2 GPUs (Ulysses only)
           GPU 0-1: Single prompt, sequence parallelism across 2 GPUs

        2. cfg_size=2, ulysses_size=1 -> 2 GPUs (CFG only)
           GPU 0: Positive prompt
           GPU 1: Negative prompt

        3. cfg_size=2, ulysses_size=2 -> 4 GPUs (CFG + Ulysses)
           GPU 0-1: CFG group 0 (positive), Ulysses parallel
           GPU 2-3: CFG group 1 (negative), Ulysses parallel

        4. cfg_size=2, ulysses_size=4 -> 8 GPUs (CFG + Ulysses)
           GPU 0-3: CFG group 0 (positive), Ulysses parallel
           GPU 4-7: CFG group 1 (negative), Ulysses parallel
    """

    disable_parallel_vae: bool = False
    parallel_vae_split_dim: Literal["width", "height"] = "width"

    # DiT Parallelism
    dit_dp_size: int = PydanticField(1, ge=1)
    dit_tp_size: int = PydanticField(1, ge=1)  # Not yet supported
    dit_ulysses_size: int = PydanticField(1, ge=1)  # Supported
    dit_ring_size: int = PydanticField(1, ge=1)  # Not yet supported
    dit_cp_size: int = PydanticField(1, ge=1)
    dit_cfg_size: int = PydanticField(1, ge=1)  # Supported
    dit_fsdp_size: int = PydanticField(1, ge=1)

    # Refiner Parallelism (Optional)
    refiner_dit_dp_size: int = 1
    refiner_dit_tp_size: int = 1
    refiner_dit_ulysses_size: int = 1
    refiner_dit_ring_size: int = 1
    refiner_dit_cp_size: int = 1
    refiner_dit_cfg_size: int = 1
    refiner_dit_fsdp_size: int = 1

    t5_fsdp_size: int = 1

    def to_mapping(self) -> Mapping:
        """Convert to TRT-LLM Mapping."""
        world_size = self.dit_tp_size * self.dit_cp_size
        return Mapping(
            world_size=world_size,
            tp_size=self.dit_tp_size,
            pp_size=1,
            cp_size=self.dit_cp_size,
        )

    @model_validator(mode="after")
    def validate_parallel_sizes(self) -> "ParallelConfig":
        """Validate configuration against current environment."""
        if torch.cuda.is_available():
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            total_parallel = (
                self.dit_tp_size
                * self.dit_ulysses_size
                * self.dit_ring_size
                * self.dit_cp_size
                * self.dit_dp_size
                * self.dit_cfg_size
            )
            if total_parallel > world_size:
                raise ValueError(
                    f"Total DiT parallel size ({total_parallel}) exceeds WORLD_SIZE ({world_size})"
                )
        return self


class TeaCacheConfig(BaseModel):
    """Configuration for TeaCache runtime optimization.

    TeaCache speeds up diffusion by caching transformer outputs when timestep
    embeddings change slowly. It monitors embedding distances and reuses cached
    residuals when changes are below a threshold.

    Attributes:
        enable_teacache: Enable TeaCache optimization
        teacache_thresh: Distance threshold for cache decisions (lower = more caching)
        use_ret_steps: Use aggressive warmup mode (5 steps) vs minimal (1 step)
        coefficients: Polynomial coefficients for rescaling embedding distances
                     Applied as: rescaled_distance = poly(raw_distance)
        ret_steps: Number of warmup steps (always compute, initialized at runtime)
        cutoff_steps: Step to stop caching (always compute after, initialized at runtime)
        num_steps: Total inference steps (set at runtime)
        _cnt: Internal step counter (reset per generation)
    """

    enable_teacache: bool = False
    teacache_thresh: float = PydanticField(0.2, gt=0.0)
    use_ret_steps: bool = True

    coefficients: List[float] = PydanticField(default_factory=lambda: [1.0, 0.0])

    # Runtime state fields (initialized by TeaCacheBackend.refresh)
    ret_steps: Optional[int] = None
    cutoff_steps: Optional[int] = None
    num_steps: Optional[int] = None

    # State tracking (reset per generation)
    _cnt: int = 0

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def validate_teacache(self) -> "TeaCacheConfig":
        """Validate TeaCache configuration."""
        # Validate coefficients
        if len(self.coefficients) == 0:
            raise ValueError("TeaCache coefficients list cannot be empty")

        # Validate ret_steps if set
        if self.ret_steps is not None and self.ret_steps < 0:
            raise ValueError(f"ret_steps must be non-negative, got {self.ret_steps}")

        # Validate cutoff_steps vs num_steps if both set
        if self.cutoff_steps is not None and self.num_steps is not None:
            if self.cutoff_steps > self.num_steps:
                raise ValueError(
                    f"cutoff_steps ({self.cutoff_steps}) cannot exceed num_steps ({self.num_steps})"
                )

        return self


class PipelineConfig(BaseModel):
    """General pipeline configuration."""

    enable_torch_compile: bool = True
    torch_compile_models: str = PipelineComponent.TRANSFORMER
    torch_compile_mode: str = "default"
    fuse_qkv: bool = True

    # Offloading Config
    enable_offloading: bool = False
    offload_device: Literal["cpu", "cuda"] = "cpu"
    offload_param_pin_memory: bool = True


# =============================================================================
# DiffusionArgs - User-facing configuration (CLI / YAML)
# =============================================================================


class DiffusionArgs(BaseModel):
    """User-facing configuration for diffusion model loading and inference.

    This is the main config class used in CLI args and YAML config files.
    PipelineLoader converts this to DiffusionModelConfig internally.

    Example:
        args = DiffusionArgs(
            checkpoint_path="/path/to/model",
            quant_config={"quant_algo": "FP8_BLOCK_SCALES", "dynamic": True},
            parallel=ParallelConfig(dit_tp_size=2),
        )
        loader = PipelineLoader()
        pipeline = loader.load(args)
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Required: Path to checkpoint or HuggingFace Hub model ID
    checkpoint_path: str = PydanticField(
        "",
        description=(
            "Local directory path or HuggingFace Hub model ID "
            "(e.g., 'Wan-AI/Wan2.1-T2V-1.3B-Diffusers'). "
            "Hub models are downloaded and cached automatically."
        ),
    )

    # HuggingFace Hub options
    revision: Optional[str] = PydanticField(
        None,
        description="HuggingFace Hub revision (branch, tag, or commit SHA) to download.",
    )

    # Device/dtype options
    device: str = "cuda"
    dtype: str = "bfloat16"

    # Component loading options (use PipelineComponent enum values or plain strings)
    skip_components: List[PipelineComponent] = PydanticField(
        default_factory=list,
        description=(
            "Components to skip loading. "
            "Accepts PipelineComponent enum values or equivalent strings "
            "(e.g., [PipelineComponent.TEXT_ENCODER, PipelineComponent.VAE])"
        ),
    )

    # Sub-configs (dict input for quant_config is coerced to QuantConfig in model_validator)
    quant_config: QuantConfig = PydanticField(default_factory=QuantConfig)
    pipeline: PipelineConfig = PydanticField(default_factory=PipelineConfig)
    attention: AttentionConfig = PydanticField(default_factory=AttentionConfig)
    parallel: ParallelConfig = PydanticField(default_factory=ParallelConfig)
    teacache: TeaCacheConfig = PydanticField(default_factory=TeaCacheConfig)

    # Set by model_validator when quant_config is provided as a dict (ModelOpt format)
    dynamic_weight_quant: bool = False
    force_dynamic_quantization: bool = False

    @model_validator(mode="before")
    @classmethod
    def _parse_quant_config_dict(cls, data: Any) -> Any:
        """Parse user-facing DiffusionArgs.quant_config (dict or None) into QuantConfig and dynamic flags.

        User input is ModelOpt-format dict (e.g. {"quant_algo": "FP8", "dynamic": True}).
        We coerce it to QuantConfig + dynamic_weight_quant + force_dynamic_quantization so that
        from_pretrained() can copy them into DiffusionModelConfig (internal) without parsing again.
        """
        if not isinstance(data, dict):
            return data
        raw = data.get("quant_config")
        if raw is None:
            data = {**data, "quant_config": QuantConfig()}
            return data
        if not isinstance(raw, dict):
            return data
        qc, _, dwq, daq = DiffusionModelConfig.load_diffusion_quant_config(raw)
        data = {
            **data,
            "quant_config": qc,
            "dynamic_weight_quant": dwq,
            "force_dynamic_quantization": daq,
        }
        return data

    def to_mapping(self) -> Mapping:
        """Derive Mapping from ParallelConfig."""
        return self.parallel.to_mapping()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "DiffusionArgs":
        """Create from dictionary with automatic nested config parsing.

        Pydantic automatically handles nested configs, but we keep this method
        for backward compatibility and to filter unknown fields.
        """
        # Get valid field names for DiffusionArgs
        valid_fields = set(cls.model_fields.keys())

        # Filter to only include valid fields (ignore unknown fields)
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_fields}

        # Pydantic automatically converts nested dicts to their respective config classes
        return cls(**filtered_dict)


# =============================================================================
# Utilities
# =============================================================================


def discover_pipeline_components(checkpoint_path: Path) -> Dict[str, Path]:
    """
    Discover components from diffusers pipeline's model_index.json.

    Returns dict mapping component name to config.json path.
    """
    model_index_path = checkpoint_path / "model_index.json"
    if not model_index_path.exists():
        return {}

    with open(model_index_path) as f:
        model_index = json.load(f)

    components = {}
    for key, value in model_index.items():
        if key.startswith("_") or value is None:
            continue
        config_path = checkpoint_path / key / "config.json"
        if config_path.exists():
            components[key] = config_path

    return components


# =============================================================================
# DiffusionModelConfig - Internal configuration (merged/parsed)
# =============================================================================


class DiffusionModelConfig(BaseModel):
    """Internal ModelConfig for diffusion models.

    This is created by PipelineLoader from DiffusionArgs + checkpoint.
    Contains merged/parsed config from:
    - pretrained_config: From checkpoint/config.json
    - quant_config: From checkpoint or user quant config
    - Sub-configs: From DiffusionArgs (pipeline, attention, parallel, teacache)
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    pretrained_config: Optional[Any] = None
    mapping: Mapping = PydanticField(default_factory=Mapping)
    skip_create_weights_in_init: bool = False
    force_dynamic_quantization: bool = False
    allreduce_strategy: AllReduceStrategy = PydanticField(default=AllReduceStrategy.AUTO)
    extra_attrs: Dict = PydanticField(default_factory=dict)

    # Distributed process groups
    ulysses_process_group: Optional[torch.distributed.ProcessGroup] = None

    dynamic_weight_quant: bool = False

    # Sub-configs from DiffusionArgs (merged during from_pretrained)
    quant_config: QuantConfig = PydanticField(default_factory=QuantConfig)
    # Per-layer quant (from load_diffusion_quant_config layer_quant_config; None until mixed-precision parsing exists)
    quant_config_dict: Optional[Dict[str, QuantConfig]] = None
    pipeline: PipelineConfig = PydanticField(default_factory=PipelineConfig)
    attention: AttentionConfig = PydanticField(default_factory=AttentionConfig)
    parallel: ParallelConfig = PydanticField(default_factory=ParallelConfig)
    teacache: TeaCacheConfig = PydanticField(default_factory=TeaCacheConfig)

    @property
    def torch_dtype(self) -> "torch.dtype":
        """Get the torch dtype of the model (default: bfloat16)."""
        return torch.bfloat16

    def get_quant_config(self, name: Optional[str] = None) -> QuantConfig:
        """Get quantization config for a layer or global. Resembles LLM ModelConfig.get_quant_config."""
        if name is None or self.quant_config_dict is None:
            return self.quant_config
        if name in self.quant_config_dict:
            return self.quant_config_dict[name]
        return self.quant_config

    @staticmethod
    def load_diffusion_quant_config(
        quant_config_dict: dict,
    ) -> Tuple[QuantConfig, Optional[Dict], bool, bool]:
        """
        Parse quantization config in ModelOpt format.

        Returns: (quant_config, layer_quant_config, dynamic_weight_quant, dynamic_activation_quant)
            - quant_config: Global QuantConfig
            - layer_quant_config: Per-layer config dict (None if not using mixed precision)
            - dynamic_weight_quant: Whether to quantize weights at load time
            - dynamic_activation_quant: Whether to quantize activations dynamically
        """
        quant_algo_str = quant_config_dict.get("quant_algo")
        quant_algo = None
        if quant_algo_str:
            algo_map = {
                "FP8": QuantAlgo.FP8,
                "FP8_BLOCK_SCALES": QuantAlgo.FP8_BLOCK_SCALES,
                "NVFP4": QuantAlgo.NVFP4,
                "W4A16_AWQ": QuantAlgo.W4A16_AWQ,
                "W4A8_AWQ": QuantAlgo.W4A8_AWQ,
                "W8A8_SQ_PER_CHANNEL": QuantAlgo.W8A8_SQ_PER_CHANNEL,
            }
            quant_algo = algo_map.get(quant_algo_str)
            if quant_algo is None:
                raise ValueError(f"Unknown quant_algo: {quant_algo_str}")

        # Parse group_size and dynamic flags from config_groups
        group_size = None
        dynamic_weight_quant = False
        dynamic_activation_quant = False
        for group_config in quant_config_dict.get("config_groups", {}).values():
            weights_config = group_config.get("weights", {})
            activations_config = group_config.get("input_activations", {})
            dynamic_weight_quant = weights_config.get("dynamic", False)
            dynamic_activation_quant = activations_config.get("dynamic", False)
            # Extract group_size from weights config (e.g., NVFP4: group_size=16)
            if group_size is None:
                group_size = weights_config.get("group_size")
            break

        # Set defaults based on quant_algo if group_size not specified
        if group_size is None:
            if quant_algo in (QuantAlgo.NVFP4,):
                group_size = 16  # NVFP4 default
            elif quant_algo == QuantAlgo.FP8_BLOCK_SCALES:
                group_size = 128  # FP8 blockwise default

        # Auto-enable dynamic weight quantization if quant_algo is specified
        # but no explicit config_groups setting is present.
        # This allows simple configs like {"quant_algo": "FP8"} to work.
        if quant_algo is not None and not quant_config_dict.get("config_groups"):
            dynamic_weight_quant = quant_config_dict.get("dynamic", True)

        quant_config = QuantConfig(
            quant_algo=quant_algo,
            group_size=group_size,
            exclude_modules=quant_config_dict.get("ignore"),
        )

        # TODO: Per-layer config (None for now - future: parse mixed precision settings)
        layer_quant_config = None

        return quant_config, layer_quant_config, dynamic_weight_quant, dynamic_activation_quant

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_dir: str,
        args: Optional["DiffusionArgs"] = None,
        **kwargs,
    ) -> "DiffusionModelConfig":
        """
        Load config from pretrained checkpoint.

        Called by PipelineLoader with DiffusionArgs:
            config = DiffusionModelConfig.from_pretrained(
                checkpoint_dir=args.checkpoint_path,
                args=args,
            )

        Args:
            checkpoint_dir: Path to checkpoint
            args: DiffusionArgs containing user config (quant, pipeline, attention, parallel, teacache)
            **kwargs: Additional config options (e.g., mapping)
        """
        kwargs.pop("trust_remote_code", None)

        # Extract sub-configs from args or use defaults
        pipeline_cfg = args.pipeline if args else PipelineConfig()
        attention_cfg = args.attention if args else AttentionConfig()
        parallel_cfg = args.parallel if args else ParallelConfig()
        teacache_cfg = args.teacache if args else TeaCacheConfig()

        component = PipelineComponent.TRANSFORMER
        checkpoint_path = Path(checkpoint_dir)

        # Discover pipeline components
        components = discover_pipeline_components(checkpoint_path)

        # Determine config path
        if components:
            if component not in components:
                raise ValueError(
                    f"Component '{component}' not found. Available: {list(components.keys())}"
                )
            config_path = components[component]
        else:
            config_path = checkpoint_path / "config.json"

        if not config_path.exists():
            raise ValueError(f"Config not found at {config_path}")

        # Load pretrained_config from checkpoint
        with open(config_path) as f:
            config_dict = json.load(f)
        pretrained_config = SimpleNamespace(**config_dict)

        model_index_path = checkpoint_path / "model_index.json"
        if model_index_path.exists():
            with open(model_index_path) as f:
                model_index = json.load(f)
            if "boundary_ratio" in model_index and "transformer_2" in model_index:
                transformer_2_spec = model_index.get("transformer_2")
                if transformer_2_spec and transformer_2_spec[0] is not None:
                    pretrained_config.boundary_ratio = model_index["boundary_ratio"]

        # Resolve quant config: use args if user set quant (QuantConfig from dict), else checkpoint
        if args and args.quant_config.quant_algo is not None:
            quant_config = args.quant_config
            quant_config_dict = (
                None  # DiffusionArgs has no per-layer dict; only from checkpoint parse
            )
            dynamic_weight_quant = args.dynamic_weight_quant
            dynamic_activation_quant = args.force_dynamic_quantization
        else:
            quant_config = QuantConfig()
            quant_config_dict = None
            dynamic_weight_quant = False
            dynamic_activation_quant = False
            quant_dict = getattr(pretrained_config, "quantization_config", None)
            if isinstance(quant_dict, dict):
                quant_config, quant_config_dict, dynamic_weight_quant, dynamic_activation_quant = (
                    cls.load_diffusion_quant_config(quant_dict)
                )

        return cls(
            pretrained_config=pretrained_config,
            quant_config=quant_config,
            quant_config_dict=quant_config_dict,
            dynamic_weight_quant=dynamic_weight_quant,
            force_dynamic_quantization=dynamic_activation_quant,
            # Sub-configs from DiffusionArgs
            pipeline=pipeline_cfg,
            attention=attention_cfg,
            parallel=parallel_cfg,
            teacache=teacache_cfg,
            # Delay weight creation after apply_quant_config_exclude_modules() in __post_init__
            skip_create_weights_in_init=True,
            **kwargs,
        )
