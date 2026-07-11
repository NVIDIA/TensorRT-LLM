# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Public engine-level configuration for VisualGen.

This module is the canonical public home for ``VisualGenArgs`` and the
cross-cutting sub-configs it composes. All fields carry
``Field(status="prototype")`` — the engine config surface is pre-GA and
subject to breaking changes.
"""

from pathlib import Path
from typing import Annotated, Any, Dict, List, Literal, Optional, Tuple, Union

import yaml
from pydantic import model_validator

from tensorrt_llm.llmapi.llm_args import Field
from tensorrt_llm.llmapi.utils import StrictBaseModel, set_api_status
from tensorrt_llm.models.modeling_utils import QuantConfig

from .sparse_attention import SkipSoftmaxAttentionConfig, VideoSparseAttentionConfig

# =============================================================================
# Type aliases
# =============================================================================

CacheBackendName = Literal["teacache", "cache_dit"]

# =============================================================================
# Sub-configuration classes for VisualGenArgs
# =============================================================================


class QuantAttentionConfig(StrictBaseModel):
    """Attention quantization recipe (TRTLLM / CUTEDSL backends).

    Describes user intent for quantized attention: per-bmm dtype and per-block layout for Q, K, V.
    Providing this config to AttentionConfig enables quantized attention; setting
    AttentionConfig.quant_attention_config = None disables it.

    Bare QuantAttentionConfig() is a valid Qk16Pv8 recipe.
    Unsupported recipes are rejected by AttentionConfig's validator with a ValueError.
    """

    qk_dtype: Literal["bf16", "int8", "fp8"] = Field(
        "bf16",
        status="prototype",
        description="Q/K quantization dtype; bf16 leaves Q/K unquantized.",
    )
    v_dtype: Literal["fp8"] = Field(
        "fp8",
        status="prototype",
        description="V quantization dtype. The current kernels always load V in FP8 (e4m3).",
    )
    q_block_size: int = Field(
        0,
        ge=0,
        status="prototype",
        description="Elements per quantization block for Q; 0 for per-tensor quantization.",
    )
    k_block_size: int = Field(
        0,
        ge=0,
        status="prototype",
        description="Elements per quantization block for K; 0 for per-tensor quantization.",
    )
    v_block_size: int = Field(
        0,
        ge=0,
        status="prototype",
        description="Elements per quantization block for V; 0 for per-tensor quantization.",
    )


# Discriminated union of sparse attention configs.
SparseAttentionConfig = Annotated[
    Union[SkipSoftmaxAttentionConfig, VideoSparseAttentionConfig],
    Field(discriminator="algorithm"),
]


class AttentionConfig(StrictBaseModel):
    """Configuration for Attention layers."""

    backend: Literal["VANILLA", "TRTLLM", "FA4", "CUTEDSL"] = Field(
        "VANILLA",
        status="prototype",
        description="Attention backend: VANILLA (PyTorch SDPA), TRTLLM, FA4, CUTEDSL",
    )
    quant_attention_config: Optional[QuantAttentionConfig] = Field(
        None,
        status="prototype",
        description=(
            "Quantized-attention recipe (TRTLLM / CUTEDSL backends). "
            "Set to a QuantAttentionConfig instance to enable quantized "
            "attention; leave as None to disable."
        ),
    )
    sparse_attention_config: Optional[SparseAttentionConfig] = Field(
        None,
        status="prototype",
        description=(
            "Sparse attention recipe. Discriminated by algorithm: "
            "skip_softmax (TRTLLM backend) or VSA (CUTEDSL backend)."
        ),
    )

    @model_validator(mode="after")
    def _validate_quant_attention_config(self) -> "AttentionConfig":
        # SAGE recipes target the TRTLLM backend (per-block Q/K/V scales).
        SAGE_RECIPES = {
            ("int8", "fp8", (1, 1, 1)),
            ("int8", "fp8", (1, 4, 1)),
            ("int8", "fp8", (1, 16, 1)),
            ("fp8", "fp8", (1, 1, 1)),
            ("fp8", "fp8", (1, 4, 1)),
        }
        # QK16PV8 (CUTEDSL backend): Q/K kept in bf16, V quantized to FP8.
        QK16PV8_DTYPES = {
            ("bf16", "fp8", (0, 0, 0)),
        }

        if self.quant_attention_config is None:
            return self

        q_config = self.quant_attention_config
        recipe = (
            q_config.qk_dtype,
            q_config.v_dtype,
            (q_config.q_block_size, q_config.k_block_size, q_config.v_block_size),
        )
        if self.backend == "TRTLLM":
            if recipe not in SAGE_RECIPES:
                raise ValueError(
                    f"Unsupported quant_attention_config={self.quant_attention_config!r} "
                    f"for backend='TRTLLM'. Supported SAGE recipes "
                    f"(qk_dtype, v_dtype, (q_block, k_block, v_block)): "
                    f"{sorted(SAGE_RECIPES)}."
                )
        elif self.backend == "CUTEDSL":
            if recipe not in QK16PV8_DTYPES:
                raise ValueError(
                    f"Unsupported quant_attention_config={self.quant_attention_config!r} "
                    f"for backend='CUTEDSL'. Supported (qk_dtype, v_dtype): "
                    f"{sorted(QK16PV8_DTYPES)}."
                )
        else:
            raise ValueError(
                f"quant_attention_config requires backend in ('TRTLLM', 'CUTEDSL'), "
                f"got backend='{self.backend}'. Either change backend or "
                f"remove quant_attention_config."
            )
        return self

    @model_validator(mode="after")
    def _validate_sparse_attention_config(self) -> "AttentionConfig":
        if self.sparse_attention_config is None:
            return self

        algo = self.sparse_attention_config.algorithm
        required_backend = {"skip_softmax": "TRTLLM", "vsa": "CUTEDSL"}.get(algo)
        if required_backend is None:
            return self

        if self.backend != required_backend:
            raise ValueError(
                f"sparse_attention_config with algorithm='{algo}' requires "
                f"backend='{required_backend}', got backend='{self.backend}'. "
                f"Either set backend='{required_backend}' or remove "
                f"sparse_attention_config."
            )
        return self

    @model_validator(mode="after")
    def _validate_cutedsl_quant_sparse_mutex(self) -> "AttentionConfig":
        # quant_attention_config and sparse_attention_config are mutually exclusive.
        if (
            self.backend == "CUTEDSL"
            and self.quant_attention_config is not None
            and self.sparse_attention_config is not None
        ):
            raise ValueError(
                "CUTEDSL backend: quant_attention_config and "
                "sparse_attention_config are mutually exclusive (the "
                "CuTeDSLAttention dispatcher selects either the dense path "
                "or the sparse VSA path, not both)."
            )
        return self


class ParallelConfig(StrictBaseModel):
    """Configuration for distributed parallelism across DiT-shaped models.

    The sequence axis can be sharded via Attention2D (a 2D mesh) or Ring
    Attention (a 1D mesh); these two are mutually exclusive. Either can be
    combined with Ulysses head-sharding to form an outer × inner sequence
    parallel mesh. See ``mapping.py`` for the underlying DeviceMesh layout.
    """

    parallel_vae_size: int = Field(
        1,
        ge=1,
        status="prototype",
        description="Number of ranks used for VAE parallelism. 1 disables parallel VAE.",
    )
    parallel_vae_split_dim: Literal["width", "height"] = Field(
        "width",
        status="prototype",
    )
    cfg_size: int = Field(
        1,
        ge=1,
        le=2,
        status="prototype",
        description=(
            "CFG (classifier-free guidance) batch parallelism degree. "
            "cfg_size=2 splits positive/negative prompts across 2 GPUs."
        ),
    )
    ulysses_size: int = Field(
        1,
        ge=1,
        status="prototype",
        description=("Ulysses head-sharding degree. Heads are sharded across ulysses_size GPUs."),
    )
    async_ulysses: bool = Field(
        False,
        status="prototype",
        description=(
            "Enable the async Ulysses A2A pipeline: overlap per-rank V/Q/K projection compute "
            "with cross-rank symm-mem all-to-all on a dedicated side stream. "
            "Requires ulysses_size > 1. Defaults to False."
        ),
    )
    ring_size: int = Field(
        1,
        ge=1,
        status="prototype",
        description=(
            "Ring Attention sequence-parallel degree. The sequence is sharded "
            "across ring_size GPUs and K/V blocks stream around the ring. "
            "Mutually exclusive with attn2d_size > (1, 1); combinable with "
            "ulysses_size."
        ),
    )
    attn2d_size: Tuple[Annotated[int, Field(ge=1)], Annotated[int, Field(ge=1)]] = Field(
        (1, 1),
        status="prototype",
        description=(
            "Attention2D context parallelism as (row_size, col_size). "
            "(1, 1) disables Attention2D; both entries must be >= 1. "
            "A 2x2 mesh gathers Q across the row group and K/V across the "
            "column group. Mutually exclusive with ring_size > 1."
        ),
    )
    tp_size: int = Field(
        1,
        ge=1,
        status="prototype",
        description=("Tensor parallel group size. Heads are sharded across tp_size GPUs."),
    )

    @property
    def seq_parallel_size(self) -> int:
        """Parallelism degree over the sequence/context axis.

        Combines the active context-parallel size (Attention2D mesh or ring)
        with Ulysses head-sharding. Attention2D and ring are mutually
        exclusive; either composes with Ulysses as ``cp × ulysses``.
        """
        attn2d_total = self.attn2d_size[0] * self.attn2d_size[1]
        if attn2d_total > 1:
            cp_size = attn2d_total
        elif self.ring_size > 1:
            cp_size = self.ring_size
        else:
            cp_size = 1
        return cp_size * self.ulysses_size

    @property
    def n_workers(self) -> int:
        return self.cfg_size * self.seq_parallel_size * self.tp_size

    @property
    def total_parallel_size(self) -> int:
        return self.cfg_size * self.seq_parallel_size

    @model_validator(mode="after")
    def _validate_async_ulysses(self) -> "ParallelConfig":
        if self.async_ulysses:
            if self.ulysses_size == 1:
                raise ValueError(
                    "async_ulysses=True requires ulysses_size > 1; got "
                    f"ulysses_size={self.ulysses_size}."
                )
            if self.ring_size > 1:
                raise ValueError(
                    "async_ulysses=True is incompatible with ring_size > 1: "
                    "async_ulysses forces SEPARATE_QKV which bypasses the "
                    "RingAttention wrapper. Set ring_size=1 or async_ulysses=False "
                    f"(got ring_size={self.ring_size})."
                )
        return self

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
    teacache_thresh: float = Field(0.2, gt=0.0, status="prototype")
    use_ret_steps: bool = Field(False, status="prototype")

    coefficients: Optional[List[float]] = Field(
        default=None,
        status="prototype",
        description=(
            "Polynomial coefficients used by the TeaCache decision function. "
            "None (default) uses the pipeline's built-in per-checkpoint table; "
            "an explicit list overrides the table entirely."
        ),
    )

    coefficients_2: Optional[List[float]] = Field(
        default=None,
        status="prototype",
        description=(
            "Second polynomial (Wan 2.2 dual-transformer low-noise stage only). "
            "Required together with coefficients when enabling TeaCache on Wan 2.2."
        ),
    )

    @model_validator(mode="after")
    def validate_teacache(self) -> "TeaCacheConfig":
        """Validate TeaCache configuration."""
        if self.coefficients is not None and len(self.coefficients) == 0:
            raise ValueError("TeaCache coefficients list cannot be empty")
        if self.coefficients_2 is not None and len(self.coefficients_2) == 0:
            raise ValueError("TeaCache coefficients_2 list cannot be empty")
        return self

    def is_explicit_user_override(self) -> bool:
        """Return True if coefficients were set by the user and should skip built-in table matching."""
        return self.coefficients is not None


class CacheDiTConfig(BaseCacheConfig):
    """Configuration for Cache-DiT (DBCache, TaylorSeer, SCM).

    Requires the cache-dit package.
    """

    cache_backend: Literal["cache_dit"] = "cache_dit"
    Fn_compute_blocks: int = Field(
        1,
        ge=0,
        status="prototype",
        description="First n blocks always computed (Fn).",
    )
    Bn_compute_blocks: int = Field(
        0,
        ge=0,
        status="prototype",
        description="Last n blocks use residual cache (Bn).",
    )
    max_warmup_steps: int = Field(
        4,
        ge=0,
        status="prototype",
        description="Initial steps that do not use cache (default tuned for few-step runs).",
    )
    max_cached_steps: int = Field(
        -1,
        status="prototype",
        description="Cap on cached steps; -1 means no limit.",
    )
    max_continuous_cached_steps: int = Field(
        3,
        ge=-1,
        status="prototype",
        description="Cap on consecutive cached steps (-1 = library unlimited; default 3).",
    )
    residual_diff_threshold: float = Field(
        0.24,
        ge=0.0,
        status="prototype",
        description="L1 diff threshold for DBCache (default pairs with max_continuous_cached_steps).",
    )
    enable_separate_cfg: Optional[bool] = Field(
        None,
        status="prototype",
        description=(
            "If set, forwarded to DBCacheConfig.enable_separate_cfg. "
            "If None, enablers pick defaults for each pipeline (Wan: batched CFG → False)."
        ),
    )
    enable_taylorseer: bool = Field(False, status="prototype")
    taylorseer_order: int = Field(1, ge=1, le=4, status="prototype")

    scm_steps_mask_policy: Optional[str] = Field(
        None,
        status="prototype",
        description="Policy name for cache_dit.steps_mask (e.g. fast, medium, slow, ultra).",
    )
    scm_steps_policy: Literal["dynamic", "static"] = Field(
        "dynamic",
        status="prototype",
    )

    force_refresh_step_hint: Optional[int] = Field(
        None,
        status="prototype",
        description="Optional step index hint for forced cache refresh (cache_dit DBCacheConfig).",
    )
    force_refresh_step_policy: Literal["once", "repeat"] = Field(
        "once",
        status="prototype",
        description="Policy for force_refresh_step_hint: once or repeat.",
    )


CacheConfig = Annotated[
    Union[TeaCacheConfig, CacheDiTConfig],
    Field(discriminator="cache_backend"),
]


class TorchCompileConfig(StrictBaseModel):
    """Configuration for torch.compile and autotuning.

    Warmup shapes for torch.compile specialization are configured via
    CompilationConfig (resolutions + num_frames), not here.
    """

    enable: bool = Field(True, status="prototype")
    enable_fullgraph: bool = Field(False, status="prototype")
    enable_autotune: bool = Field(True, status="prototype")


class CudaGraphConfig(StrictBaseModel):
    """Configuration for CUDA graph capture/replay.

    Warmup shapes for CUDA graph pre-capture are configured via
    CompilationConfig (resolutions + num_frames), not here.
    """

    enable: bool = Field(False, status="prototype")


class CompilationConfig(StrictBaseModel):
    """Configuration for torch.compile / CUDA graph warmup shapes.

    Warmup shapes are the Cartesian product of ``resolutions`` and
    ``num_frames``. More shapes mean slower startup but lower recompile
    risk on first requests; fewer shapes mean faster startup but a recompile
    on the first request with an un-warmed shape. When the fields are
    ``None``, each model pipeline supplies its own defaults
    (e.g. Wan uses ``[(480, 832), (720, 1280)] x [33, 81]``).
    """

    resolutions: Optional[List[Tuple[int, int]]] = Field(
        default=None,
        status="prototype",
        description=(
            "List of (height, width) resolutions to warmup at startup. "
            "Combined with num_frames via Cartesian product. "
            "If None, uses model-specific defaults."
        ),
    )
    num_frames: Optional[List[int]] = Field(
        default=None,
        status="prototype",
        description=(
            "List of frame counts to warmup at startup. "
            "Combined with resolutions via Cartesian product. "
            "If None, uses model-specific defaults. "
            "For image models, use [1]."
        ),
    )
    skip_warmup: bool = Field(
        False,
        status="prototype",
        description="Skip the post-load warmup pass (compile + capture).",
    )


# =============================================================================
# VisualGenArgs - User-facing configuration (CLI / YAML)
# =============================================================================


class VisualGenArgs(StrictBaseModel):
    """User-facing configuration for diffusion model loading and inference.

    This is the main config class used in CLI args and YAML config files.
    PipelineLoader converts this to DiffusionModelConfig internally.

    Example:
        args = VisualGenArgs(
            model="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            quant_config={"quant_algo": "FP8_BLOCK_SCALES", "dynamic": True},
            parallel_config=ParallelConfig(ulysses_size=2),
        )
        loader = PipelineLoader(args)
        pipeline = loader.load()
    """

    model: str = Field(
        "",
        status="prototype",
        description=(
            "Local directory path or HuggingFace Hub model ID "
            "(e.g., 'Wan-AI/Wan2.1-T2V-1.3B-Diffusers'). "
            "Hub models are downloaded and cached automatically."
        ),
    )

    revision: Optional[str] = Field(
        None,
        status="prototype",
        description="HuggingFace Hub revision (branch, tag, or commit SHA) to download.",
    )

    quant_config: Union[QuantConfig, Dict[str, Any]] = Field(
        default_factory=QuantConfig,
        status="prototype",
        description=(
            "Quantization config — accepts either a QuantConfig instance "
            "or a ModelOpt-format dict (e.g. ``{'quant_algo': 'FP8', "
            "'dynamic': True}``). Dict-form parsing happens lazily in "
            "DiffusionPipelineConfig.from_pretrained."
        ),
    )
    compilation_config: CompilationConfig = Field(
        default_factory=CompilationConfig,
        status="prototype",
    )
    torch_compile_config: TorchCompileConfig = Field(
        default_factory=TorchCompileConfig,
        status="prototype",
    )
    cuda_graph_config: CudaGraphConfig = Field(
        default_factory=CudaGraphConfig,
        status="prototype",
    )
    attention_config: AttentionConfig = Field(
        default_factory=AttentionConfig,
        status="prototype",
    )
    parallel_config: ParallelConfig = Field(
        default_factory=ParallelConfig,
        status="prototype",
    )
    cache_config: Optional[CacheConfig] = Field(None, status="prototype")

    pipeline_config: Dict[str, Any] = Field(
        default_factory=dict,
        status="prototype",
        description=(
            "Per-architecture pipeline runtime knobs. Strict — unknown keys "
            "raise at load time. See VisualGen.pipeline_config(model) for "
            "the legal key set for a given model."
        ),
    )

    enable_layerwise_nvtx_marker: bool = Field(
        False,
        status="prototype",
        description=(
            "If True, register per-layer NVTX hooks at load time for nsys profile readability."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_quant_config(cls, data: Any) -> Any:
        """Treat ``quant_config: null`` as ``QuantConfig()``.

        A bare ``quant_config: null`` in YAML otherwise fails union
        validation against ``Union[QuantConfig, Dict[str, Any]]``.
        """
        if isinstance(data, dict) and data.get("quant_config", "_sentinel") is None:
            data = {**data, "quant_config": QuantConfig()}
        return data

    @property
    def cache_backend(self) -> Optional[CacheBackendName]:
        return self.cache_config.cache_backend if self.cache_config is not None else None  # type: ignore[return-value]

    @property
    def teacache(self) -> Optional[TeaCacheConfig]:
        return self.cache_config if isinstance(self.cache_config, TeaCacheConfig) else None

    @property
    def cache_dit(self) -> Optional[CacheDiTConfig]:
        return self.cache_config if isinstance(self.cache_config, CacheDiTConfig) else None

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
        if not isinstance(config_dict, dict):
            raise ValueError(
                f"VisualGenArgs YAML must contain a mapping at the document root: {yaml_path}"
            )
        config_dict.update(overrides)
        return cls(**config_dict)


__all__ = [
    "QuantAttentionConfig",
    "SparseAttentionConfig",
    "SkipSoftmaxAttentionConfig",
    "VideoSparseAttentionConfig",
    "AttentionConfig",
    "ParallelConfig",
    "BaseCacheConfig",
    "TeaCacheConfig",
    "CacheDiTConfig",
    "CacheConfig",
    "TorchCompileConfig",
    "CudaGraphConfig",
    "CompilationConfig",
    "VisualGenArgs",
]
