# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated, Any, Dict, List, Literal, Optional, Tuple, Union

import torch
import yaml
from pydantic import BaseModel, ConfigDict, model_validator
from pydantic import Field as PydanticField

from tensorrt_llm._torch.visual_gen.mapping import DEFAULT_DIM_ORDER
from tensorrt_llm.functional import AllReduceStrategy
from tensorrt_llm.llmapi.utils import StrictBaseModel, set_api_status
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

# =============================================================================
# Type aliases
# =============================================================================

CacheBackendName = Literal["teacache", "cache_dit"]

# =============================================================================
# Pipeline component identifiers
# =============================================================================


class PipelineComponent(str, Enum):
    """Identifiers for pipeline components that can be loaded or skipped.

    Inherits from str so values compare equal to plain strings,
    e.g. PipelineComponent.VAE == "vae" is True.
    """

    TRANSFORMER = "transformer"
    VAE = "vae"
    TEXT_ENCODER = "text_encoder"
    TEXT_ENCODER_2 = "text_encoder_2"
    TOKENIZER = "tokenizer"
    TOKENIZER_2 = "tokenizer_2"
    SCHEDULER = "scheduler"
    IMAGE_ENCODER = "image_encoder"
    IMAGE_PROCESSOR = "image_processor"


# =============================================================================
# Sub-configuration classes for VisualGenArgs
# =============================================================================


class SageAttentionConfig(StrictBaseModel):
    """Configuration for SageAttention quantization (TRTLLM backend only).

    SageAttention quantizes Q/K/V into FP8 (or INT8 for Q/K) with per-block
    scaling factors, enabling faster attention kernels. Providing this config
    to AttentionConfig enables SageAttention; omitting it (None) disables it.

    Similar to ``sparse_attention_config`` for the base TRTLLM attention
    backend — the presence of the config object signals enablement.

    Currently these (num_elts_per_blk_q, num_elts_per_blk_k, num_elts_per_blk_v)
    combinations are enabled:
    - (1, 1, 1)
    - (1, 4, 1)
    - (1, 16, 1) [for qk_int8 == True only]
    """

    num_elts_per_blk_q: int = PydanticField(
        1, ge=0, description="Elements per quantization block for Q (0 disables)"
    )
    num_elts_per_blk_k: int = PydanticField(
        4, ge=0, description="Elements per quantization block for K (0 disables)"
    )
    num_elts_per_blk_v: int = PydanticField(
        1, ge=0, description="Elements per quantization block for V (0 disables)"
    )
    qk_int8: bool = PydanticField(True, description="Use INT8 (vs E4M3) for Q/K quantization")


class AttentionConfig(StrictBaseModel):
    """Configuration for Attention layers."""

    backend: Literal["VANILLA", "TRTLLM", "FA4"] = PydanticField(
        "VANILLA", description="Attention backend: VANILLA (PyTorch SDPA), TRTLLM, FA4"
    )
    sage_attention_config: Optional[SageAttentionConfig] = PydanticField(
        None,
        description=(
            "SageAttention config (TRTLLM backend only). "
            "Set to a SageAttentionConfig instance to enable SageAttention; "
            "leave as None to disable."
        ),
    )

    @model_validator(mode="after")
    def _validate_sage_attn_config(self) -> "AttentionConfig":
        SUPPORTED_SAGE_CONFIGS = {
            (1, 1, 1, False),
            (1, 4, 1, False),
            (1, 1, 1, True),
            (1, 4, 1, True),
            (1, 16, 1, True),
        }

        if self.sage_attention_config is not None:
            if self.backend != "TRTLLM":
                logger.critical(
                    f"sage_attention_config requires backend='TRTLLM', "
                    f"got backend='{self.backend}'. Either set backend='TRTLLM' "
                    f"or remove sage_attention_config. Disabling SageAttention."
                )
                self.sage_attention_config = None
                return self

            if (
                self.sage_attention_config.num_elts_per_blk_q,
                self.sage_attention_config.num_elts_per_blk_k,
                self.sage_attention_config.num_elts_per_blk_v,
                self.sage_attention_config.qk_int8,
            ) not in SUPPORTED_SAGE_CONFIGS:
                logger.critical(
                    f"Unsupported {self.sage_attention_config=}. Disabling SageAttention."
                )
                self.sage_attention_config = None
        return self


class ParallelConfig(StrictBaseModel):
    """Configuration for distributed parallelism.

    Currently Supported:
        - dit_cfg_size: CFG (Classifier-Free Guidance) parallelism
        - dit_ulysses_size: Ulysses head-sharding parallelism
        - dit_attn2d_row_size, dit_attn2d_col_size: Attention2D context parallelism

    Not Yet Supported:
        - dit_tp_size: Tensor parallelism (not implemented)
        - dit_ring_size: Ring attention context parallelism (not implemented)
        - dit_dp_size, dit_fsdp_size: Other parallelism types
        - Combining Ulysses and Attention2D (orthogonal in principle, not yet implemented)

    See mapping.py for more details.

    Example Configurations:
        1. cfg_size=1, ulysses_size=2 -> 2 GPUs (Ulysses only)
           GPU 0-1: Single prompt, heads sharded across 2 GPUs

        2. cfg_size=2, ulysses_size=1 -> 2 GPUs (CFG only)
           GPU 0: Positive prompt
           GPU 1: Negative prompt

        3. cfg_size=2, ulysses_size=2 -> 4 GPUs (CFG + Ulysses)
           GPU 0-1: CFG group 0 (positive), Ulysses parallel
           GPU 2-3: CFG group 1 (negative), Ulysses parallel

        4. cfg_size=2, ulysses_size=4 -> 8 GPUs (CFG + Ulysses)
           GPU 0-3: CFG group 0 (positive), Ulysses parallel
           GPU 4-7: CFG group 1 (negative), Ulysses parallel

        5. cfg_size=1, attn2d_row_size=2, attn2d_col_size=2 -> 4 GPUs (Attention2D only)
           2x2 mesh: Q gathered across row group, K/V gathered across col group
    """

    parallel_vae_size: int = PydanticField(
        1,
        ge=1,
        description="Number of ranks used for VAE parallelism. 1 disables parallel VAE.",
    )
    parallel_vae_split_dim: Literal["width", "height"] = "width"

    # DiT Parallelism
    dit_dp_size: int = PydanticField(1, ge=1)
    dit_tp_size: int = PydanticField(1, ge=1)  # Not yet supported
    dit_ulysses_size: int = PydanticField(1, ge=1)  # Supported
    dit_ring_size: int = PydanticField(1, ge=1)  # Not yet supported
    dit_attn2d_row_size: int = PydanticField(1, ge=1)  # Supported
    dit_attn2d_col_size: int = PydanticField(1, ge=1)  # Supported
    dit_cfg_size: int = PydanticField(1, ge=1)  # Supported
    dit_fsdp_size: int = PydanticField(1, ge=1)
    dit_dim_order: str = PydanticField(
        DEFAULT_DIM_ORDER,
        description=(
            "Outermost-to-innermost ordering of parallelism axes for the "
            "DeviceMesh. Innermost = most contiguous ranks."
        ),
    )

    # Refiner Parallelism (Optional)
    refiner_dit_dp_size: int = 1
    refiner_dit_tp_size: int = 1
    refiner_dit_ulysses_size: int = 1
    refiner_dit_ring_size: int = 1
    refiner_dit_cp_size: int = 1
    refiner_dit_cfg_size: int = 1
    refiner_dit_fsdp_size: int = 1

    t5_fsdp_size: int = 1

    @property
    def seq_parallel_size(self) -> int:
        """Parallelism degree over the sequence/context axis.

        Returns the active parallel degree: Attention2D total mesh size if enabled,
        ring size if ring CP is enabled, otherwise Ulysses size (head-sharding).
        Exactly one of these is active at a time; combining them is not yet implemented.
        """
        attn2d_size = self.dit_attn2d_row_size * self.dit_attn2d_col_size
        if attn2d_size > 1:
            return attn2d_size
        if self.dit_ring_size > 1:
            return self.dit_ring_size
        return self.dit_ulysses_size

    @property
    def n_workers(self) -> int:
        return self.dit_cfg_size * self.seq_parallel_size

    @property
    def total_parallel_size(self) -> int:
        return self.dit_cfg_size * self.dit_tp_size * self.seq_parallel_size

    def validate_world_size(self, world_size: int) -> None:
        if self.total_parallel_size > world_size:
            raise ValueError(
                f"total_parallel_size ({self.total_parallel_size}) "
                f"exceeds world_size ({world_size})"
            )


class BaseCacheConfig(StrictBaseModel):
    """Base class for diffusion step caching acceleration configs."""

    cache_backend: str


class TeaCacheConfig(BaseCacheConfig):
    """TeaCache step-caching acceleration config."""

    cache_backend: Literal["teacache"] = "teacache"
    teacache_thresh: float = PydanticField(0.2, gt=0.0)
    use_ret_steps: bool = False

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


class CacheDiTConfig(BaseCacheConfig):
    """Configuration for Cache-DiT (DBCache, TaylorSeer, SCM).

    Requires the cache-dit package.
    """

    cache_backend: Literal["cache_dit"] = "cache_dit"
    Fn_compute_blocks: int = PydanticField(
        1, ge=0, description="First n blocks always computed (Fn)."
    )
    Bn_compute_blocks: int = PydanticField(
        0, ge=0, description="Last n blocks use residual cache (Bn)."
    )
    max_warmup_steps: int = PydanticField(
        4,
        ge=0,
        description="Initial steps that do not use cache (default tuned for few-step runs).",
    )
    max_cached_steps: int = PydanticField(
        -1,
        description="Cap on cached steps; -1 means no limit.",
    )
    max_continuous_cached_steps: int = PydanticField(
        3,
        ge=-1,
        description="Cap on consecutive cached steps (-1 = library unlimited; default 3).",
    )
    residual_diff_threshold: float = PydanticField(
        0.24,
        ge=0.0,
        description="L1 diff threshold for DBCache (default pairs with max_continuous_cached_steps).",
    )
    enable_separate_cfg: Optional[bool] = PydanticField(
        None,
        description=(
            "If set, forwarded to DBCacheConfig.enable_separate_cfg. "
            "If None, enablers pick defaults for each pipeline (Wan: batched CFG → False)."
        ),
    )
    enable_taylorseer: bool = False
    taylorseer_order: int = PydanticField(1, ge=1, le=4)

    scm_steps_mask_policy: Optional[str] = PydanticField(
        None,
        description="Policy name for cache_dit.steps_mask (e.g. fast, medium, slow, ultra).",
    )
    scm_steps_policy: Literal["dynamic", "static"] = "dynamic"

    force_refresh_step_hint: Optional[int] = PydanticField(
        None,
        description="Optional step index hint for forced cache refresh (cache_dit DBCacheConfig).",
    )
    force_refresh_step_policy: Literal["once", "repeat"] = PydanticField(
        "once",
        description="Policy for force_refresh_step_hint: once or repeat.",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)


CacheConfig = Annotated[
    Union[TeaCacheConfig, CacheDiTConfig],
    PydanticField(discriminator="cache_backend"),
]


class TorchCompileConfig(StrictBaseModel):
    """Configuration for torch.compile and autotuning.

    Warmup shapes for torch.compile specialization are configured via
    CompilationConfig (resolutions + num_frames), not here.
    """

    enable_torch_compile: bool = True
    enable_fullgraph: bool = False
    enable_autotune: bool = True


class CudaGraphConfig(StrictBaseModel):
    """Configuration for CUDA graph capture/replay.

    Warmup shapes for CUDA graph pre-capture are configured via
    CompilationConfig (resolutions + num_frames), not here.
    """

    enable_cuda_graph: bool = False


class CompilationConfig(StrictBaseModel):
    """Configuration for torch.compile / CUDA graph warmup shapes.

    Warmup shapes are the Cartesian product of ``resolutions`` and ``num_frames``.
    For example, 2 resolutions x 2 frame counts = 4 warmup shapes.

    More warmup shapes = slower startup, but lower risk of torch.compile
    recompilation delays on first requests. Fewer shapes = faster startup,
    but first request with an un-warmed shape triggers recompilation.

    If not configured, each model pipeline uses its own defaults
    (e.g., Wan uses [(480, 832), (720, 1280)] x [33, 81]).

    YAML usage (via ``--extra_visual_gen_options``)::

        # Custom warmup: 2 resolutions x 2 frame counts = 4 shapes
        compilation:
          resolutions:
            - [480, 832]
            - [720, 1280]
          num_frames: [33, 81]

        # Only override resolutions (frame counts use model defaults)
        compilation:
          resolutions:
            - [1920, 1080]

        # Skip warmup entirely
        compilation:
          resolutions: []
          num_frames: []
    """

    resolutions: Optional[List[Tuple[int, int]]] = PydanticField(
        default=None,
        description=(
            "List of (height, width) resolutions to warmup at startup. "
            "Combined with num_frames via Cartesian product. "
            "If None, uses model-specific defaults."
        ),
    )
    num_frames: Optional[List[int]] = PydanticField(
        default=None,
        description=(
            "List of frame counts to warmup at startup. "
            "Combined with resolutions via Cartesian product. "
            "If None, uses model-specific defaults. "
            "For image models, use [1]."
        ),
    )


class PipelineConfig(StrictBaseModel):
    """Model-specific pipeline configuration."""

    fuse_qkv: bool = True
    enable_layerwise_nvtx_marker: bool = False

    # Offloading
    enable_offloading: bool = False
    offload_device: Literal["cpu", "cuda"] = "cpu"
    offload_param_pin_memory: bool = True


# =============================================================================
# VisualGenArgs - User-facing configuration (CLI / YAML)
# =============================================================================


class VisualGenArgs(StrictBaseModel):
    """User-facing configuration for diffusion model loading and inference.

    This is the main config class used in CLI args and YAML config files.
    PipelineLoader converts this to DiffusionModelConfig internally.

    Example:
        args = VisualGenArgs(
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

    # Path to the text encoder model (e.g. Gemma3 directory) used by LTX-2 pipelines.
    text_encoder_path: str = PydanticField(
        "",
        description=(
            "Path to the text encoder model directory (e.g. Gemma3). "
            "Required for LTX-2 pipelines. Must contain model weights, "
            "tokenizer files, and preprocessor config."
        ),
    )

    # Two-stage LTX-2: learned spatial upsampler checkpoint path.
    spatial_upsampler_path: str = PydanticField(
        "",
        description=(
            "Path to the learned LatentUpsampler checkpoint (.safetensors). "
            "Optional for LTX-2 two-stage pipelines. When provided, the "
            "pipeline auto-selects LTX2TwoStagesPipeline. If omitted, "
            "TensorRT-LLM tries to discover it in the checkpoint directory."
        ),
    )

    # Two-stage LTX-2: distilled LoRA checkpoint path for stage 2 refinement.
    distilled_lora_path: str = PydanticField(
        "",
        description=(
            "Path to the distilled LoRA checkpoint (.safetensors) used in "
            "the stage 2 refinement pass. If omitted, TensorRT-LLM tries to "
            "discover it in the checkpoint directory. The LoRA weights are "
            "merged into the transformer for stage 2 denoising and un-merged afterwards."
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

    # Skip warmup inference after loading (useful for testing or fast startup)
    skip_warmup: bool = False

    # Sub-configs (dict input for quant_config is coerced to QuantConfig in model_validator)
    quant_config: QuantConfig = PydanticField(default_factory=QuantConfig)
    compilation: CompilationConfig = PydanticField(default_factory=CompilationConfig)
    torch_compile: TorchCompileConfig = PydanticField(default_factory=TorchCompileConfig)
    cuda_graph: CudaGraphConfig = PydanticField(default_factory=CudaGraphConfig)
    pipeline: PipelineConfig = PydanticField(default_factory=PipelineConfig)
    attention: AttentionConfig = PydanticField(default_factory=AttentionConfig)
    parallel: ParallelConfig = PydanticField(default_factory=ParallelConfig)
    cache: Optional[CacheConfig] = None

    # Set by model_validator when quant_config is provided as a dict (ModelOpt format)
    dynamic_weight_quant: bool = False
    force_dynamic_quantization: bool = False

    @model_validator(mode="before")
    @classmethod
    def _parse_quant_config_dict(cls, data: Any) -> Any:
        """Parse user-facing VisualGenArgs.quant_config (dict or None) into QuantConfig and dynamic flags.

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

    @property
    def cache_backend(self) -> Optional[CacheBackendName]:
        return self.cache.cache_backend if self.cache is not None else None  # type: ignore[return-value]

    @property
    def teacache(self) -> Optional[TeaCacheConfig]:
        return self.cache if isinstance(self.cache, TeaCacheConfig) else None

    @property
    def cache_dit(self) -> Optional[CacheDiTConfig]:
        return self.cache if isinstance(self.cache, CacheDiTConfig) else None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.model_dump()

    @set_api_status("prototype")
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "VisualGenArgs":
        """Create from dictionary with automatic nested config parsing.

        Unknown fields cause a ValidationError (extra="forbid").
        """
        return cls(**config_dict)

    @set_api_status("prototype")
    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path], **overrides: Any) -> "VisualGenArgs":
        """Load configuration from a YAML file.

        Args:
            yaml_path: Path to the YAML configuration file.
            **overrides: Keyword arguments that override values from the YAML file.

        Returns:
            A validated VisualGenArgs instance.
        """
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f) or {}
        config_dict.update(overrides)
        return cls(**config_dict)


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


def create_attention_metadata_state() -> Dict[str, Any]:
    """Create model-scoped attention metadata state for TRTLLM visual-gen backend."""
    return {"metadata": None, "capacity": (0, 0)}


# =============================================================================
# DiffusionModelConfig - Internal configuration (merged/parsed)
# =============================================================================


class DiffusionModelConfig(BaseModel):
    """Internal ModelConfig for diffusion models.

    This is created by PipelineLoader from VisualGenArgs + checkpoint.
    Contains merged/parsed config from:
    - pretrained_config: From checkpoint/config.json
    - quant_config: From checkpoint or user quant config
    - Sub-configs: From VisualGenArgs (pipeline, attention, teacache)
    - visual_gen_mapping: Populated by setup_visual_gen_mapping() from ParallelConfig
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    pretrained_config: Optional[Any] = None
    mapping: Mapping = PydanticField(default_factory=Mapping)
    skip_create_weights_in_init: bool = False
    force_dynamic_quantization: bool = False
    allreduce_strategy: AllReduceStrategy = PydanticField(default=AllReduceStrategy.AUTO)
    extra_attrs: Dict = PydanticField(default_factory=dict)

    # Unified parallelism mapping (populated by setup_visual_gen_mapping)
    visual_gen_mapping: Optional[Any] = None  # VisualGenMapping (lazy import)

    dynamic_weight_quant: bool = False

    # Sub-configs from VisualGenArgs (merged during from_pretrained)
    quant_config: QuantConfig = PydanticField(default_factory=QuantConfig)
    # Per-layer quant (from load_diffusion_quant_config layer_quant_config; None until mixed-precision parsing exists)
    quant_config_dict: Optional[Dict[str, QuantConfig]] = None
    compilation: CompilationConfig = PydanticField(default_factory=CompilationConfig)
    torch_compile: TorchCompileConfig = PydanticField(default_factory=TorchCompileConfig)
    cuda_graph: CudaGraphConfig = PydanticField(default_factory=CudaGraphConfig)
    pipeline: PipelineConfig = PydanticField(default_factory=PipelineConfig)
    attention: AttentionConfig = PydanticField(default_factory=AttentionConfig)
    attention_metadata_state: Optional[Dict[str, Any]] = None
    parallel: ParallelConfig = PydanticField(default_factory=ParallelConfig)
    cache: Optional[CacheConfig] = None

    @property
    def cache_backend(self) -> Optional[CacheBackendName]:
        return self.cache.cache_backend if self.cache is not None else None  # type: ignore[return-value]

    @property
    def teacache(self) -> Optional[TeaCacheConfig]:
        return self.cache if isinstance(self.cache, TeaCacheConfig) else None

    @property
    def cache_dit(self) -> Optional[CacheDiTConfig]:
        return self.cache if isinstance(self.cache, CacheDiTConfig) else None

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
            # NVFP4 requires dynamic activation quantization when using dynamic mode
            # since input_scale is not calibrated
            if quant_algo == QuantAlgo.NVFP4 and dynamic_weight_quant:
                dynamic_activation_quant = True

        quant_config = QuantConfig(
            quant_algo=quant_algo,
            group_size=group_size,
            exclude_modules=quant_config_dict.get("ignore"),
        )

        # TODO: Per-layer config (None for now - future: parse mixed precision settings)
        layer_quant_config = None

        return quant_config, layer_quant_config, dynamic_weight_quant, dynamic_activation_quant

    @staticmethod
    def _convert_quantization_metadata(
        qmeta: Dict,
        tensor_keys: List[str],
    ) -> Dict:
        """
        TODO: Consider refactor this to be a utility functions.
        Convert per-layer ``_quantization_metadata`` to ModelOpt format.

        Some checkpoints (e.g. HuggingFace-quantized FP8) embed per-layer
        quantization info as::

            {"format_version": "1.0",
             "layers": {"model.diffusion_model.block.attn.to_q": {"format": "float8_e4m3fn"}, ...}}

        This converts it to the ModelOpt-compatible dict that
        :meth:`load_diffusion_quant_config` understands::

            {"quant_algo": "FP8",
             "config_groups": {"default": {"weights": {"dynamic": false}, ...}},
             "ignore": ["proj_in", "proj_out", ...]}
        """
        _FORMAT_TO_ALGO = {
            "float8_e4m3fn": "FP8",
        }

        layers = qmeta.get("layers", {})
        if not layers:
            return {}

        formats = {info.get("format") for info in layers.values()}
        if len(formats) != 1:
            logger.warning(f"_quantization_metadata has mixed formats {formats}; skipping")
            return {}

        fmt = formats.pop()
        quant_algo = _FORMAT_TO_ALGO.get(fmt)
        if quant_algo is None:
            logger.warning(f"_quantization_metadata format '{fmt}' is not supported; skipping")
            return {}

        quantized_layers = set(layers.keys())

        # Build ignore list: weight-bearing layers NOT in the quantized set.
        # Tensor keys ending with ".weight" (but not ".weight_scale") indicate
        # layers that own learnable weights.
        non_quantized = []
        for key in tensor_keys:
            if key.endswith(".weight") and not key.endswith("_scale.weight"):
                layer_name = key[: -len(".weight")]
                if layer_name not in quantized_layers:
                    non_quantized.append(layer_name)

        result = {
            "quant_algo": quant_algo,
            "config_groups": {
                "default": {
                    "weights": {"dynamic": False},
                    "input_activations": {"dynamic": False},
                }
            },
            "ignore": sorted(non_quantized),
        }
        logger.info(
            f"Converted _quantization_metadata: algo={quant_algo}, "
            f"{len(quantized_layers)} quantized layers, "
            f"{len(non_quantized)} excluded layers"
        )
        return result

    @classmethod
    def _try_load_safetensors_config(cls, checkpoint_path: Path) -> Optional[Dict]:
        """Try to read embedded config from a single-safetensors checkpoint.

        Accepts either a directory (globs for ``*.safetensors``) or a direct
        path to a ``.safetensors`` file.

        Returns the full config dict if found, ``None`` otherwise.
        """
        try:
            import safetensors.torch
        except ImportError:
            return None

        if checkpoint_path.is_file() and checkpoint_path.suffix == ".safetensors":
            sft_files = [checkpoint_path]
        else:
            sft_files = sorted(checkpoint_path.glob("*.safetensors"))

        if not sft_files:
            return None

        try:
            with safetensors.torch.safe_open(str(sft_files[0]), framework="pt") as f:
                meta = f.metadata()
                if meta and "config" in meta:
                    config = json.loads(meta["config"])
                    if "quantization_config" in meta:
                        config["quantization_config"] = json.loads(meta["quantization_config"])
                    elif "_quantization_metadata" in meta:
                        qmeta = json.loads(meta["_quantization_metadata"])
                        converted = cls._convert_quantization_metadata(qmeta, list(f.keys()))
                        if converted:
                            config["quantization_config"] = converted
                    return config
        except Exception:
            pass
        return None

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_dir: str,
        args: Optional["VisualGenArgs"] = None,
        **kwargs,
    ) -> "DiffusionModelConfig":
        """
        Load config from pretrained checkpoint.

        Called by PipelineLoader with VisualGenArgs:
            config = DiffusionModelConfig.from_pretrained(
                checkpoint_dir=args.checkpoint_path,
                args=args,
            )

        Supports two checkpoint formats:
        * **Diffusers directory layout** -- ``model_index.json`` with
          component sub-directories each containing ``config.json``.
        * **Single-safetensors** -- no ``model_index.json``; config embedded
          in the safetensors metadata header under a ``"config"`` key.  The
          transformer section is extracted as ``pretrained_config`` and the
          full dict is stored in ``extra_attrs["monolithic_safetensors_config"]``
          for use by component configurators.

        Args:
            checkpoint_dir: Path to checkpoint
            args: VisualGenArgs containing user config
                - (compilation, torch_compile, cuda_graph, pipeline, attention, parallel, teacache,
                   cache_backend, cache_dit)
            **kwargs: Additional config options (e.g., mapping)
        """
        kwargs.pop("trust_remote_code", None)

        # Extract sub-configs from args or use defaults
        compilation_cfg = args.compilation if args else CompilationConfig()
        torch_compile_cfg = args.torch_compile if args else TorchCompileConfig()
        cuda_graph_cfg = args.cuda_graph if args else CudaGraphConfig()
        pipeline_cfg = args.pipeline if args else PipelineConfig()
        attention_cfg = args.attention if args else AttentionConfig()
        parallel_cfg = args.parallel if args else ParallelConfig()
        cache_cfg = args.cache if args else None

        component = PipelineComponent.TRANSFORMER
        checkpoint_path = Path(checkpoint_dir)
        extra_attrs: Dict[str, Any] = {}

        if args:
            if args.spatial_upsampler_path:
                extra_attrs["spatial_upsampler_path"] = args.spatial_upsampler_path
            if args.distilled_lora_path:
                extra_attrs["distilled_lora_path"] = args.distilled_lora_path

        # Discover pipeline components (diffusers layout)
        components = discover_pipeline_components(checkpoint_path)

        if components:
            # ---------- Diffusers directory layout ----------
            if component not in components:
                raise ValueError(
                    f"Component '{component}' not found. Available: {list(components.keys())}"
                )
            config_path = components[component]
            if not config_path.exists():
                raise ValueError(f"Config not found at {config_path}")

            with open(config_path) as f:
                config_dict = json.load(f)
            pretrained_config = SimpleNamespace(**config_dict)

            # Ensure _name_or_path is set so TeaCache coefficient matching works.
            if not getattr(pretrained_config, "_name_or_path", None):
                pretrained_config._name_or_path = str(checkpoint_path)

            model_index_path = checkpoint_path / "model_index.json"
            if model_index_path.exists():
                with open(model_index_path) as f:
                    model_index = json.load(f)
                if "boundary_ratio" in model_index and "transformer_2" in model_index:
                    transformer_2_spec = model_index.get("transformer_2")
                    if transformer_2_spec and transformer_2_spec[0] is not None:
                        pretrained_config.boundary_ratio = model_index["boundary_ratio"]
                if "expand_timesteps" in model_index:
                    pretrained_config.expand_timesteps = bool(model_index["expand_timesteps"])
        else:
            # ---------- Single safetensors ----------
            native_config = cls._try_load_safetensors_config(checkpoint_path)

            if native_config is not None:
                transformer_dict = native_config.get("transformer", {})
                pretrained_config = SimpleNamespace(**transformer_dict)
                if not getattr(pretrained_config, "_name_or_path", None):
                    pretrained_config._name_or_path = str(checkpoint_path)
                extra_attrs["monolithic_safetensors_config"] = native_config

                # quantization_config lives as a separate safetensors metadata
                # key, not inside the transformer section. Propagate it so
                # the quant-config resolution below can pick it up.
                if "quantization_config" in native_config:
                    qc = native_config["quantization_config"]
                    # ModelOpt prefixes module names with the wrapped model
                    # attribute (e.g. "velocity_model.proj_out"). Strip that
                    # wrapper prefix so the ignore list matches TRT-LLM names.
                    _MODELOPT_WRAPPER_PREFIXES = (
                        "model.diffusion_model.",
                        "velocity_model.",
                        "denoiser.",
                        "unet.",
                        "dit.",
                    )
                    if "ignore" in qc and qc["ignore"]:
                        cleaned = []
                        for entry in qc["ignore"]:
                            for wp in _MODELOPT_WRAPPER_PREFIXES:
                                if entry.startswith(wp):
                                    entry = entry[len(wp) :]
                                    break
                            cleaned.append(entry)
                        qc["ignore"] = cleaned
                    pretrained_config.quantization_config = qc
            else:
                raise ValueError(
                    f"Config not found at {checkpoint_dir}. "
                    "Expected model_index.json (diffusers) or "
                    "safetensors with embedded config metadata."
                )

        # Resolve quant config
        if args and args.quant_config.quant_algo is not None:
            quant_config = args.quant_config
            quant_config_dict = (
                None  # VisualGenArgs has no per-layer dict; only from checkpoint parse
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

        # Enable tunable FP4 quantize for visual gen: larger activation
        # tensors (full image/video latents) amortize the AutoTuner overhead.
        if quant_config.quant_algo == QuantAlgo.NVFP4:
            from tensorrt_llm._torch.modules.linear import NVFP4LinearMethod

            NVFP4LinearMethod.use_tunable_quantize = True

        attention_metadata_state = (
            create_attention_metadata_state() if attention_cfg.backend == "TRTLLM" else None
        )

        return cls(
            pretrained_config=pretrained_config,
            quant_config=quant_config,
            quant_config_dict=quant_config_dict,
            dynamic_weight_quant=dynamic_weight_quant,
            force_dynamic_quantization=dynamic_activation_quant,
            # Sub-configs from VisualGenArgs
            compilation=compilation_cfg,
            torch_compile=torch_compile_cfg,
            cuda_graph=cuda_graph_cfg,
            pipeline=pipeline_cfg,
            attention=attention_cfg,
            attention_metadata_state=attention_metadata_state,
            parallel=parallel_cfg,
            cache=cache_cfg,
            skip_create_weights_in_init=True,
            extra_attrs=extra_attrs,
            **kwargs,
        )
