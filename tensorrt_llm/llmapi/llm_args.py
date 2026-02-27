import ast
import functools
import json
import math
import os
import types
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, EnumMeta
from pathlib import Path
from typing import (Annotated, Any, Dict, List, Literal, Optional, Set, Tuple,
                    Type, TypeAlias, TypeVar, Union, get_args, get_origin)

import torch
import yaml
from pydantic import AliasChoices, BaseModel, ConfigDict
from pydantic import Field as PydanticField
from pydantic import (NonNegativeFloat, NonNegativeInt, PositiveInt,
                      PrivateAttr, field_validator, model_validator)
from strenum import StrEnum
from transformers import PreTrainedTokenizerBase

try:
    from ray.util.placement_group import PlacementGroup
except ImportError:
    PlacementGroup = None

from tensorrt_llm.lora_helper import (LoraConfig,
                                      get_default_trtllm_modules_to_hf_modules)

from .._utils import mpi_rank

# yapf: disable
# isort: off
from ..bindings.executor import (BatchingType as _BatchingType,
                                 CacheTransceiverBackendType as _CacheTransceiverBackendType,
                                 CacheTransceiverConfig as _CacheTransceiverConfig,
                                 CapacitySchedulerPolicy as _CapacitySchedulerPolicy,
                                 ContextChunkingPolicy as _ContextChunkingPolicy,
                                 DecodingConfig,
                                 DecodingMode,
                                 DynamicBatchConfig as _DynamicBatchConfig,
                                 EagleConfig as _EagleConfig,
                                 ExecutorConfig as _ExecutorConfig,
                                 ExtendedRuntimePerfKnobConfig as _ExtendedRuntimePerfKnobConfig,
                                 KvCacheConfig as _KvCacheConfig,
                                 LookaheadDecodingConfig as _LookaheadDecodingConfig,
                                 PeftCacheConfig as _PeftCacheConfig,
                                 SchedulerConfig as _SchedulerConfig) # isort: skip
# isort: on

# yapf: enable
from ..builder import BuildConfig, EngineConfig
from ..logger import logger
from ..mapping import CpType, Mapping
from ..models.automodel import AutoConfig
from ..models.modeling_utils import (PretrainedConfig, QuantAlgo, QuantConfig,
                                     SpeculativeDecodingMode)
from ..sampling_params import BatchedLogitsProcessor
from .build_cache import BuildCacheConfig
from .tokenizer import TokenizerBase, tokenizer_factory
from .utils import (StrictBaseModel, generate_api_docs_as_docstring,
                    get_type_repr)

TypeBaseModel = TypeVar("T", bound=BaseModel)


def Field(default: Any = ...,
          *,
          status: Optional[Literal["prototype", "beta", "deprecated"]] = None,
          **kwargs: Any) -> Any:
    """Custom Field wrapper that adds status to json_schema_extra.

    Args:
        default: The default value for the field
        status: Optional status indicator that gets added to json_schema_extra.
            - None: Stable.
            - "beta": Recommended for use per the latest documentation.
            - "prototype": Not yet stable and subject to breaking changes; intended for experimentation only.
        **kwargs: All other arguments passed to the original Pydantic Field

    Returns:
        A Pydantic FieldInfo object with the status added to json_schema_extra if provided
    """

    if status is not None:
        json_schema_extra = kwargs.get('json_schema_extra', {})
        if isinstance(json_schema_extra, dict):
            json_schema_extra['status'] = status
        else:
            # If json_schema_extra is not a dict, create a new dict with the status
            json_schema_extra = {'status': status}
        kwargs['json_schema_extra'] = json_schema_extra

    return PydanticField(default, **kwargs)


class CudaGraphConfig(StrictBaseModel):
    """
    Configuration for CUDA graphs.
    """
    # List of batch sizes to create CUDA graphs for.
    batch_sizes: Optional[List[int]] = Field(
        default=None,
        description="List of batch sizes to create CUDA graphs for.")

    max_batch_size: NonNegativeInt = Field(
        default=0, description="Maximum batch size for CUDA graphs.")

    enable_padding: bool = Field(
        default=False,
        description=
        "If true, batches are rounded up to the nearest cuda_graph_batch_size. This is usually a net win for performance."
    )

    @model_validator(mode='after')
    def validate_cuda_graph_config(self) -> 'CudaGraphConfig':
        """Validate CUDA graph configuration.

        Ensures that:
        1. If batch_sizes is provided, max_batch_size is derived as max(batch_sizes).
           If max_batch_size was already set it must be compatible (equal to max(batch_sizes));
           otherwise an error is raised.
        2. If only max_batch_size is provided, batch_sizes is generated from it.
        3. If neither is provided, a default max_batch_size of 128 is used.
        """
        if self.batch_sizes:
            self.batch_sizes = sorted(self.batch_sizes)
            derived_max = max(self.batch_sizes)
            if self.max_batch_size == 0:
                self.max_batch_size = derived_max
            elif self.max_batch_size != derived_max:
                raise ValueError(
                    "CudaGraphConfig.max_batch_size is incompatible with "
                    "CudaGraphConfig.batch_sizes. When both are provided, "
                    "max_batch_size must equal max(batch_sizes).\n"
                    f"CudaGraphConfig.batch_sizes: {self.batch_sizes}, "
                    f"max(batch_sizes): {derived_max}, "
                    f"CudaGraphConfig.max_batch_size: {self.max_batch_size}")
        else:
            max_batch_size = self.max_batch_size or 128
            generated_sizes = CudaGraphConfig._generate_cuda_graph_batch_sizes(
                max_batch_size, self.enable_padding)
            self.batch_sizes = generated_sizes
            self.max_batch_size = max_batch_size

        return self

    @staticmethod
    def _generate_cuda_graph_batch_sizes(max_batch_size: int,
                                         enable_padding: bool) -> List[int]:
        """Generate a list of batch sizes for CUDA graphs.

        Args:
            max_batch_size: Maximum batch size to generate up to
            enable_padding: Whether padding is enabled, which affects the batch size distribution

        Returns:
            List of batch sizes to create CUDA graphs for
        """
        if enable_padding:
            batch_sizes = [1, 2, 4] + [i * 8 for i in range(1, 17)]
        else:
            batch_sizes = list(range(1, 32)) + [32, 64, 128]

        # Add powers of 2 up to max_batch_size
        batch_sizes += [
            2**i for i in range(8, math.ceil(math.log(max_batch_size, 2)))
        ]

        # Filter and sort batch sizes
        batch_sizes = sorted(
            [size for size in batch_sizes if size <= max_batch_size])

        # Add max_batch_size if not already included
        if max_batch_size != batch_sizes[-1]:
            batch_sizes.append(max_batch_size)

        return batch_sizes


class GuidedDecodingConfig(StrictBaseModel):

    class GuidedDecodingBackend(Enum):
        XGRAMMAR = 0
        LLGUIDANCE = 1

    backend: GuidedDecodingBackend = Field(
        default=GuidedDecodingBackend.XGRAMMAR,
        description="The backend for guided decoding config.")
    encoded_vocab: Optional[List[str]] = Field(
        default=None,
        description="The encoded vocab for guided decoding config.")
    tokenizer_str: Optional[str] = Field(
        default=None,
        description="The tokenizer string for guided decoding config.")
    stop_token_ids: Optional[List[int]] = Field(
        default=None,
        description="The stop token ids for guided decoding config.")


class BaseSparseAttentionConfig(StrictBaseModel):
    """
    Configuration for sparse attention.
    """
    algorithm: str

    seq_len_threshold: Optional[int] = Field(
        default=None,
        description=
        "The sequence length threshold for separating short and long sequences."
    )

    def supports_backend(self, backend: str) -> bool:
        """
        Override if the speculation algorithm does not support
        a subset of the possible backends.
        """
        return True

    def get_indices_block_size(self) -> int:
        return 1

    def needs_separate_short_long_cuda_graphs(self) -> bool:
        """
        Determines whether to capture a dedicated CUDA graph for batches consisting entirely of short sequences.
        If True, capture distinct graphs for short-only batches and general cases (e.g., long or mixed batches).
        If False, capture a single unified CUDA graph for all sequences regardless of length.
        The seq_len_threshold parameter defines the cutoff boundary between short and long sequences.
        """
        return False


class RocketSparseAttentionConfig(BaseSparseAttentionConfig):
    """
    Configuration for RocketKV sparse attention.
    """
    algorithm: Literal["rocket"] = "rocket"
    window_size: Optional[int] = Field(
        default=32, description="The window size for snap KV.")
    kernel_size: Optional[int] = Field(
        default=63, description="The kernel size for snap KV.")
    topr: Optional[Union[int, float]] = Field(default=128, description="Top-r")
    topk: Optional[int] = Field(default=64, description="Top-k")
    prompt_budget: Optional[int] = Field(default=2048,
                                         description="Prompt budget")
    page_size: Optional[int] = Field(default=4, description="Page size")
    kt_cache_dtype: Optional[str] = Field(
        default='float8_e5m2',
        choices=['bfloat16', 'float8_e5m2'],
        description="KT cache dtype",
    )

    def supports_backend(self, backend: str) -> bool:
        return backend == "pytorch"

    def get_indices_block_size(self) -> int:
        return self.page_size


class DeepSeekSparseAttentionConfig(BaseSparseAttentionConfig):
    """
    Configuration for DeepSeek Sparse Attention.
    """
    algorithm: Literal["dsa"] = "dsa"
    index_n_heads: Optional[int] = Field(
        default=None, description="The number of heads for the indexer.")
    index_head_dim: Optional[int] = Field(
        default=None, description="The dimension of the indexer heads.")
    index_topk: Optional[int] = Field(default=None,
                                      description="The topk for the indexer.")
    indexer_max_chunk_size: Optional[int] = Field(
        default=None, description="The maximum chunk size for the indexer.")
    skip_indexer_for_short_seqs: bool = Field(
        default=True,
        description=
        "Whether to skip the MQA and Top-K in the indexer for short sequences.")

    def supports_backend(self, backend: str) -> bool:
        return backend == "pytorch"

    def needs_separate_short_long_cuda_graphs(self) -> bool:
        """
        Whether to capture separate CUDA graphs for short and long sequences.
        Use seq_len_threshold to determine the threshold for separating short and long sequences.
        """
        self.seq_len_threshold = self.index_topk
        return self.skip_indexer_for_short_seqs


class SkipSoftmaxAttentionConfig(BaseSparseAttentionConfig):
    """
    Configuration for skip softmax attention.
    """
    algorithm: Literal["skip_softmax"] = "skip_softmax"
    threshold_scale_factor: Optional[Union[float, Dict[str, float]]] = Field(
        default=None,
        description="The threshold scale factor for skip softmax attention.")

    def supports_backend(self, backend: str) -> bool:
        return backend == "pytorch"

    @property
    def threshold_scale_factor_prefill(self) -> Optional[float]:
        if isinstance(self.threshold_scale_factor, dict):
            return self.threshold_scale_factor.get('prefill', None)
        return self.threshold_scale_factor

    @property
    def threshold_scale_factor_decode(self) -> Optional[float]:
        if isinstance(self.threshold_scale_factor, dict):
            return self.threshold_scale_factor.get('decode', None)
        return self.threshold_scale_factor


class MoeLoadBalancerConfig(StrictBaseModel):
    """
    Pydantic configuration model for the Mixture of Experts (MoE) load balancer.

    This model holds configuration data (`num_slots`, etc.) as well as
    runtime state (`_ep_rank`, `_ep_size`) which must be set via the
    `setup()` method before use.
    """

    num_slots: Optional[int] = None
    initial_global_assignments: Optional[Dict[int, List[int]]] = Field(
        default=None,
        repr=False  # Exclude this large dict from model representation
    )
    layer_updates_per_iter: int = 0
    _ep_rank: Optional[int] = PrivateAttr(default=None)
    _ep_size: Optional[int] = PrivateAttr(default=None)

    # --- Methods ---

    def setup(self, ep_rank: int, ep_size: int) -> None:
        """
        Initializes the runtime state of the configuration.
        This must be called before accessing properties like `num_local_slots`.
        """
        self._ep_rank = ep_rank
        self._ep_size = ep_size

        # This assertion was in the original and is critical.
        if self.num_slots is None:
            raise ValueError("`num_slots` cannot be None when calling setup().")

        if self.num_slots % ep_size != 0:
            raise ValueError(
                f"`num_slots` ({self.num_slots}) must be divisible by `ep_size` ({ep_size})."
            )

    # --- Computed Properties ---
    # These properties depend on the runtime state set by setup()

    @property
    def ep_rank(self) -> int:
        """Public accessor for the private expert parallel rank."""
        if self._ep_rank is None:
            raise AttributeError("ep_rank is not set. Call setup() first.")
        return self._ep_rank

    @property
    def ep_size(self) -> int:
        """Public accessor for the private expert parallel size."""
        if self._ep_size is None:
            raise AttributeError("ep_size is not set. Call setup() first.")
        return self._ep_size

    @property
    def num_local_slots(self) -> int:
        """Calculates the number of slots local to this rank."""
        if self.num_slots is None or self._ep_size is None:
            raise ValueError(
                "Cannot calculate `num_local_slots`. "
                "`num_slots` must be set and setup() must be called.")
        return self.num_slots // self._ep_size

    @property
    def slot_start(self) -> int:
        """Calculates the starting global slot index for this rank."""
        if self._ep_rank is None:
            raise ValueError(
                "Cannot calculate `slot_start`. Call setup() first.")
        return self._ep_rank * self.num_local_slots

    @property
    def slot_end(self) -> int:
        """Calculates the ending global slot index (exclusive) for this rank."""
        return self.slot_start + self.num_local_slots

    def get_layer_initial_global_assignments(
            self, layer_idx: int) -> Optional[List[int]]:
        """
        Retrieves the initial global assignments for a specific layer.
        """
        if self.initial_global_assignments is None:
            return None

        if layer_idx not in self.initial_global_assignments:
            raise KeyError(
                f"layer_idx {layer_idx} not found in `initial_global_assignments`."
            )

        assignments = self.initial_global_assignments[layer_idx]

        if self.num_slots is None:
            raise ValueError(
                "`num_slots` is not set, cannot verify assignment length.")

        if len(assignments) != self.num_slots:
            raise ValueError(
                f"Assignment length ({len(assignments)}) for layer {layer_idx} "
                f"does not match `num_slots` ({self.num_slots}).")

        return assignments


class MoeConfig(StrictBaseModel):
    """
    Configuration for MoE.
    """
    backend: Literal[
        "AUTO", "CUTLASS", "CUTEDSL", "WIDEEP", "TRTLLM", "DEEPGEMM", "VANILLA",
        "TRITON"] = Field(
            default='AUTO',
            description="MoE backend to use. "
            "AUTO selects default backend based on model. It currently doesn\'t always give the best choice for all scenarios. The capabilities of auto selection will be improved in future releases."
        )

    max_num_tokens: Optional[int] = Field(
        default=None,
        description=
        "If set, at most max_num_tokens tokens will be sent to torch.ops.trtllm.fused_moe at the same time. If the number of tokens exceeds max_num_tokens, the input tensors will be split into chunks and a for loop will be used."
    )

    load_balancer: Optional[Union[object, str]] = Field(
        default=None,
        description="Configuration for MoE load balancing.",
        json_schema_extra={"type": "Union[MoeLoadBalancerConfig, dict, str]"})

    disable_finalize_fusion: bool = Field(
        default=False,
        description=
        "Disable FC2+finalize kernel fusion in CUTLASS MoE backend. Setting this to True recovers deterministic numerical behavior with top-k > 2."
    )

    use_low_precision_moe_combine: bool = Field(
        default=False,
        description=
        "Use low precision combine in MoE operations (only for NVFP4 quantization). When enabled, uses lower precision for combining expert outputs to improve performance."
    )


Nvfp4Backend = Literal['cutlass', 'cublaslt', 'cutedsl', 'cuda_core']


class Nvfp4GemmConfig(StrictBaseModel):
    """
    Configuration for NVFP4 GEMM backend selection.
    """
    allowed_backends: List[Nvfp4Backend] = Field(
        default_factory=lambda: ['cutlass', 'cublaslt', 'cuda_core'],
        min_length=1,
        description="List of backends to consider for auto-selection. "
        "Default excludes 'cutedsl' for faster build time. "
        "Add 'cutedsl' for extreme performance at the cost of longer server launch time."
    )


class AttentionDpConfig(StrictBaseModel):
    """
    Configuration for attention DP.
    """
    enable_balance: bool = Field(default=False,
                                 description="Whether to enable balance.")
    timeout_iters: int = Field(
        default=50, description="The number of iterations to timeout.")
    batching_wait_iters: int = Field(
        default=10,
        description="The number of iterations to wait for batching.")

    @model_validator(mode='after')
    def validate_attention_dp_config(self) -> 'AttentionDpConfig':
        if self.enable_balance:
            if self.batching_wait_iters < 0:
                raise ValueError(
                    "attention_dp_config.batching_wait_iters must be greater or equal to 0 when enable_balance is true"
                )
            if self.timeout_iters < 0:
                raise ValueError(
                    "attention_dp_config.timeout_iters must be greater or equal to 0 when enable_balance is true"
                )
        return self


class CpConfig(StrictBaseModel):
    """
    Configuration for context parallelism.
    """
    # TODO: given that multiple fields here are only used with specific cp_types, consider
    # making this a Pydantic discriminated union.
    cp_type: CpType = Field(default=CpType.ULYSSES,
                            description="Context parallel type.")
    tokens_per_block: Optional[int] = Field(
        default=None,
        description="Number of tokens per block. Used in HELIX parallelism.")
    use_nccl_for_alltoall: Optional[bool] = Field(
        default=None,
        description=
        "Whether to use NCCL for alltoall communication. Used in HELIX parallelism. Defaults to True."
    )
    fifo_version: Optional[int] = Field(
        default=None,
        description=
        "FIFO version for alltoall communication. Used in HELIX parallelism. Defaults to 2."
    )
    cp_anchor_size: Optional[int] = Field(
        default=None, description="Anchor size for STAR attention.")
    block_size: Optional[int] = Field(
        default=None, description="Block size for STAR attention.")

    @field_validator("cp_type", mode="before")
    @classmethod
    def validate_cp_type(cls, v):
        """Normalize cp_type string to uppercase."""
        if v is None:
            return None
        if isinstance(v, str):
            return v.upper()
        return v


class _ParallelConfig(StrictBaseModel):
    """The model distribution configs for LLM."""
    tp_size: int = 1
    pp_size: int = 1
    cp_size: int = 1
    gpus_per_node: int = 8
    # Set default for MoE fields to -1 to trigger auto-calculation in Mapping
    moe_cluster_size: int = -1
    moe_tp_size: int = -1
    moe_ep_size: int = -1
    cp_config: Optional[CpConfig] = Field(default=None)
    pp_partition: Optional[List[int]] = Field(default=None)
    enable_attention_dp: bool = False
    enable_lm_head_tp_in_adp: bool = False

    _devices: Optional[List[int]] = PrivateAttr(default=None)

    @property
    def devices(self) -> List[int]:
        if self._devices is None:
            return list(range(self.world_size))
        return self._devices

    @devices.setter
    def devices(self, devices: List[int]):
        if len(devices) != self.world_size:
            raise ValueError(
                f"devices {devices} should have the same length as world_size {self.world_size}"
            )
        self._devices = devices

    @property
    def world_size(self) -> int:
        return self.tp_size * self.pp_size * self.cp_size

    @property
    def world_size_per_node(self) -> int:
        world_size = self.world_size
        total_nodes = math.ceil(world_size / self.gpus_per_node)
        return world_size // total_nodes  #TODO is this right?

    @world_size.setter
    def world_size(self, world_size: int):
        if world_size != self.tp_size * self.pp_size * self.cp_size:
            raise ValueError(
                f"world_size {world_size} should be equal to tp_size * pp_size * cp_size {self.tp_size * self.pp_size * self.cp_size} "
            )

    @property
    def is_multi_gpu(self) -> bool:
        return self.world_size > 1

    def to_mapping(self) -> Mapping:
        return Mapping(
            world_size=self.world_size,
            rank=mpi_rank(),
            gpus_per_node=self.gpus_per_node,
            tp_size=self.tp_size,
            pp_size=self.pp_size,
            pp_partition=self.pp_partition,
            cp_size=self.cp_size,
            # TODO: Mapping still uses cp_config as a dict; migrate to CpConfig
            cp_config=self.cp_config.model_dump(
                exclude_none=True) if self.cp_config else {},
            enable_attention_dp=self.enable_attention_dp,
            enable_lm_head_tp_in_adp=self.enable_lm_head_tp_in_adp,
            moe_cluster_size=self.moe_cluster_size,
            moe_tp_size=self.moe_tp_size,
            moe_ep_size=self.moe_ep_size)


class CalibConfig(StrictBaseModel):
    """
    Calibration configuration.
    """
    device: Literal['cuda',
                    'cpu'] = Field(default='cuda',
                                   description="The device to run calibration.")
    calib_dataset: str = Field(
        default='cnn_dailymail',
        description="The name or local path of calibration dataset.")
    calib_batches: int = Field(
        default=512,
        description="The number of batches that the calibration runs.")
    calib_batch_size: int = Field(
        default=1, description="The batch size that the calibration runs.")
    calib_max_seq_length: int = Field(
        default=512,
        description="The maximum sequence length that the calibration runs.")
    random_seed: int = Field(
        default=1234, description="The random seed used for calibration.")
    tokenizer_max_seq_length: int = Field(
        default=2048,
        description=
        "The maximum sequence length to initialize tokenizer for calibration.")


class _ModelFormatKind(Enum):
    HF = 0
    TLLM_CKPT = 1
    TLLM_ENGINE = 2


class DecodingBaseConfig(StrictBaseModel):
    max_draft_len: Optional[NonNegativeInt] = Field(
        default=None, description="The number of drafter layers.")

    max_total_draft_tokens: Optional[int] = Field(
        default=None,
        description=
        "The number of draft tokens in the draft tokens tree. If it's a linear tree, each draft layer will "
        "only generate one draft token. In this case, max_draft_len == max_total_draft_tokens. If it's a static or "
        "dynamic tree, each draft layer may generate more than one draft token. In this case, "
        "max_total_draft_tokens >= max_draft_len.")

    speculative_model: Optional[Union[str, Path]] = Field(
        default=None,
        validation_alias=AliasChoices("speculative_model",
                                      "speculative_model_dir"),
        description=
        "The speculative (draft) model. Accepts either (1) a HuggingFace Hub model ID (e.g. 'yuhuili/EAGLE3-LLaMA3.1-Instruct-8B'),"
        "which will be automatically downloaded, or (2) a local filesystem path to a downloaded model directory."
    )

    max_concurrency: Optional[NonNegativeInt] = Field(
        default=None,
        description=
        "When specified, speculation will be disabled at batch sizes above this value. Otherwise, "
        "speculation will always be on. PyTorch backend only.")

    draft_len_schedule: Optional[dict[int, int]] = Field(
        default=None,
        description=
        "Developer interface: dynamically adjust draft length based on active batch size in runtime. "
        "Maps batch size to draft lengths. For example, {1: 4, 4: 2, 8: 0} means: "
        "batch_size >= 1 uses draft_len=4, batch_size >= 4 uses draft_len=2, "
        "batch_size >= 8 uses draft_len=0 (disable speculation). "
        "draft_len_schedule is enforced to contain batch_size=1 and its according draft_len equals "
        "max_draft_len for consistency; for example, if max_draft_len=4, the schedule must contain {1: 4}."
    )

    load_format: Optional[str] = Field(
        default=None, description="The load format of the speculative model.")

    acceptance_window: Optional[NonNegativeInt] = Field(
        default=None,
        description=
        "The rolling average window size (N) for acceptance length across completed requests. "
        "If not set or set to 0, the feature is disabled. PyTorch backend only."
    )

    acceptance_length_threshold: Optional[NonNegativeFloat] = Field(
        default=None,
        description=
        "The threshold for average acceptance length; speculation will be disabled permanently once the "
        "rolling average over the last N completed requests (N = acceptance_window) drops below this value. "
        "PyTorch backend only.")

    allow_advanced_sampling: bool = Field(
        default=False,
        status="prototype",
        description=
        "If true, allows non-greedy sampling when speculation is used. Only applicable "
        "to 1-model code paths; non-greedy sampling is always enabled on 2-model paths."
    )

    # If set, drafting is allowed to use chain drafter.
    _allow_chain_drafter: bool = PrivateAttr(True)
    # If set, drafting uses greedy sampling, irrespective of sampling parameters.
    _allow_greedy_draft_tokens: bool = PrivateAttr(True)
    # Internal: record decoding_type alias used during parsing (for warnings).
    _decoding_type_alias: Optional[str] = PrivateAttr(default=None)
    # If set, drafting will use separate KV cache in one-model speculative decoding.
    _allow_separate_draft_kv_cache: bool = PrivateAttr(True)

    @field_validator('draft_len_schedule')
    @classmethod
    def validate_draft_len_schedule_and_sort(cls, v, info):
        """Validate and sort draft_len_schedule by batch size thresholds."""
        if v is not None:
            # Validate values
            for batch_size, draft_len in v.items():
                if batch_size < 1:
                    raise ValueError(
                        f"draft_len_schedule: batch size threshold must be >= 1, got {batch_size}"
                    )
                if draft_len < 0:
                    raise ValueError(
                        f"draft_len_schedule: draft length must be >= 0, got {draft_len}"
                    )

            # Require batch_size=1 in schedule
            if 1 not in v:
                raise ValueError(
                    "draft_len_schedule must include batch_size=1. "
                    "All systems can have batch_size=1. Add {1: <max_draft_len>} to your schedule."
                )

            # Enforce schedule[1] == max_draft_len for consistency
            max_draft_len = info.data.get('max_draft_len')
            if max_draft_len is not None and v[1] != max_draft_len:
                raise ValueError(
                    f"draft_len_schedule[1] must equal max_draft_len for consistency. "
                    f"Got schedule[1]={v[1]}, but max_draft_len={max_draft_len}. "
                    f"batch_size=1 should use maximum draft length.")

            # Enforce all draft lengths <= max_draft_len
            if max_draft_len is not None:
                for batch_size, draft_len in v.items():
                    if draft_len > max_draft_len:
                        raise ValueError(
                            f"draft_len_schedule: all draft lengths must be <= max_draft_len. "
                            f"Got draft_len={draft_len} for batch_size={batch_size}, "
                            f"but max_draft_len={max_draft_len}.")

            # Return sorted dict (by batch size thresholds)
            # This ensures efficient lookup
            return dict(sorted(v.items(), key=lambda x: x[0]))
        return v

    def supports_backend(self, backend: str) -> bool:
        """
        Override if the speculation algorithm does not support
        a subset of the possible backends.
        """
        return True

    @functools.cached_property
    def spec_dec_mode(self):
        # spec_dec_mode has more functionality than the raw decoding_mode string.
        # Use an alias for the import here to avoid name collisions with the one for the
        # TRT backend.
        from tensorrt_llm._torch.speculative.interface import \
            SpeculativeDecodingMode as TorchSpeculativeDecodingMode
        return TorchSpeculativeDecodingMode.from_string(
            self.decoding_type.upper())

    @functools.cached_property
    def is_linear_tree(self) -> bool:
        return self.max_draft_len == self.max_total_draft_tokens

    @property
    def tokens_per_gen_step(self) -> int:
        """Total tokens per gen request in one spec dec iteration (including golden token)."""
        return 1 + self.max_total_draft_tokens


class KvCacheConnectorConfig(StrictBaseModel):
    """
    Configuration for the KV Cache Connector.
    """
    connector_module: str = Field(
        ...,
        description=
        "The import path to the connector module. It will be imported with `importlib.import_module`."
    )
    connector_scheduler_class: str = Field(
        ..., description="The class name of the scheduler within the module.")
    connector_worker_class: str = Field(
        ..., description="The class name of the worker within the module.")


class LayerwiseBenchmarksConfig(StrictBaseModel):
    """
    Configuration for layer-wise benchmarks calibration.
    """
    calibration_mode: Literal["NONE", "MARK", "COLLECT"] = Field(
        default="NONE",
        description=
        "Instruct the layer-wise benchmarks calibrator to work on MARK mode, or COLLECT mode",
        status="prototype")

    calibration_file_path: Optional[str] = Field(
        default=None,
        description=
        "The file path which the layer-wise benchmarks calibrator saves to or loads from",
        status="prototype")

    calibration_layer_indices: Optional[List[int]] = Field(
        default=None,
        description=
        "Layer indices to filter. If None, all layers are collected in COLLECT mode.",
        status="prototype")

    @model_validator(mode='after')
    def validate_calibration_file_path(self) -> 'LayerwiseBenchmarksConfig':
        if self.calibration_mode == "COLLECT" and not self.calibration_file_path:
            raise ValueError(
                f"Expect calibration_file_path not to be empty when work on {self.calibration_mode} mode"
            )
        return self


class MedusaDecodingConfig(DecodingBaseConfig):
    decoding_type: Literal["Medusa"] = "Medusa"
    medusa_choices: Optional[List[List[int]]] = Field(
        default=None,
        description=
        "Tree structure for Medusa draft token generation. Each sublist represents a path in the tree where elements are token indices at each level. "
        "For example, [[0], [0, 0], [1], [0, 1]] defines multiple branches.")
    num_medusa_heads: Optional[int] = Field(
        default=None,
        description=
        "Number of Medusa prediction heads to use. Each head predicts a draft token at a different position in parallel. "
        "If not specified, defaults to the 'medusa_num_heads' value from the Medusa model's config.json."
    )

    @model_validator(mode="after")
    def set_max_total_draft_tokens(self):
        self.max_total_draft_tokens = self.max_draft_len  # Current Medusa only supports linear tree
        return self

    def supports_backend(self, backend: str) -> bool:
        return backend not in ("pytorch", "_autodeploy")


class EagleDecodingConfig(DecodingBaseConfig):
    decoding_type: Literal["Eagle"] = "Eagle"
    eagle_choices: Optional[List[List[int]]] = Field(
        default=None,
        description=
        "Static tree structure for draft token generation. Each sublist represents a path in the tree. Mutually exclusive with use_dynamic_tree."
    )
    greedy_sampling: Optional[bool] = Field(
        default=True,
        description=
        "Whether to use greedy sampling (Top-1 with token equality acceptance) or typical acceptance with multinomial sampling."
    )
    posterior_threshold: Optional[float] = Field(
        default=None,
        description=
        "Minimum token probability threshold for typical acceptance. Corresponds to epsilon in https://arxiv.org/pdf/2401.10774."
    )
    use_dynamic_tree: Optional[bool] = Field(
        default=False,
        description=
        "Whether to use dynamic tree (Eagle-2 algorithm). Mutually exclusive with eagle_choices."
    )
    dynamic_tree_max_topK: Optional[int] = Field(
        default=None,
        description="The topK value for each layer when dynamic tree is enabled."
    )
    num_eagle_layers: Optional[int] = Field(
        default=None,
        description=
        "The number of eagle layers. Will not be used in pytorch flow, just for compatibility with TRT flow."
    )
    max_non_leaves_per_layer: Optional[int] = Field(
        default=None, description="The number of non-leaves in each layer.")
    eagle3_one_model: Optional[bool] = Field(
        default=True,
        description=
        "Whether to use the faster one-model implementation (draft as submodule) or the two-model implementation."
    )
    eagle3_layers_to_capture: Optional[Set[int]] = Field(
        default=None,
        description=
        "Target model layer indices to capture hidden states from for the EAGLE3 draft model. Defaults to {1, num_layers//2-1, num_layers-4}."
    )
    eagle3_model_arch: Literal["llama3", "mistral_large3"] = Field(
        default="llama3",
        description="The model architecture of the eagle3 model.")

    @field_validator('eagle_choices', mode='before')
    @classmethod
    def validate_eagle_choices(cls, v):
        if v is not None:
            logger.warning(
                "NOTE: The Draft token tree is still under development, PLEASE DO NOT USE IT !!!"
            )
            if not isinstance(v, list):
                if isinstance(v, str):
                    v = ast.literal_eval(v.replace(" ", ""))
                else:
                    raise ValueError(
                        "Wrong eagle choices type. Eagle choices should be a List[List[int]] or a string like [[0], [1], [2], [0, 0], [0, 1]]."
                    )
        return v

    @model_validator(mode='after')
    def validate_eagle_config(self) -> 'EagleDecodingConfig':
        if self.max_draft_len is None or self.max_draft_len == 0:
            raise ValueError("max_draft_len must be > 0 for Eagle")
        self.num_eagle_layers = self.max_draft_len
        self.max_total_draft_tokens = self.max_draft_len  # If using linear-tree, the max_total_draft_tokens is the same as max_draft_len

        if self.eagle3_model_arch == "mistral_large3" and self.eagle3_layers_to_capture is None:
            # FIXME find a better way to setup it.
            self.eagle3_layers_to_capture = {-1}

        # Static tree logic
        # Checks whether the input eagle choices is valid
        # and reset the max_draft_len and num_eagle_layers if necessary
        if self.eagle_choices is not None:
            if self.use_dynamic_tree:
                raise ValueError(
                    "If eagle_choices is provided, use_dynamic_tree should be False"
                )

            # Get num_eagle_layers from eagle_choices
            num_eagle_layers_from_choices = self.check_eagle_choices()
            if num_eagle_layers_from_choices != self.num_eagle_layers:
                logger.warning(
                    f"Base on the input choices, reset the num_eagle_layers(max_draft_len) from {self.num_eagle_layers} to {num_eagle_layers_from_choices}"
                )
                self.num_eagle_layers = num_eagle_layers_from_choices
                self.max_draft_len = num_eagle_layers_from_choices

            # Each draft node has a path(choice) from the root to it.
            # So the number of choices also represents the number of max draft nodes.
            self.max_total_draft_tokens = len(self.eagle_choices)

        # Dynamic tree logic
        if self.use_dynamic_tree:
            if self.eagle_choices is not None:
                raise ValueError(
                    "If use_dynamic_tree is True, eagle_choices should be None")
            if self.max_draft_len is None or self.max_draft_len <= 0:
                raise ValueError(
                    "max_draft_len should be provided, which indicates the number of drafter layers"
                )
            if self.dynamic_tree_max_topK is None or self.dynamic_tree_max_topK <= 0:
                raise ValueError(
                    "dynamic_tree_max_topK should be provided, which indicates the number of nodes to expand each time"
                )
            if self.max_total_draft_tokens is None or self.max_total_draft_tokens <= 0:
                raise ValueError(
                    "max_total_draft_tokens should be provided, which indicates the total nodes of the final draft tree. (exclude the root node)"
                )

        return self

    @model_validator(mode="after")
    def validate_speculative_model(self) -> 'EagleDecodingConfig':
        if self.speculative_model is None:
            raise ValueError("Draft model must be provided for EAGLE")
        return self

    def check_eagle_choices(self):
        # 1) Check connectivity
        unique_choices = set(
            tuple(sub_choice)
            for sub_choice in self.eagle_choices)  # remove repeated choices
        self.eagle_choices = sorted([list(t) for t in unique_choices],
                                    key=lambda x: (len(x), x))  # sort choices
        for choice in self.eagle_choices:
            if len(choice) > 1:
                assert choice[
                    0:
                    -1] in self.eagle_choices, f"Error: choice {choice} is not connected"

        # 2) Get num_eagle_layers_from_choices
        num_eagle_layers_from_choices = max(
            len(choice) for choice in self.eagle_choices)

        return num_eagle_layers_from_choices

    @functools.cached_property
    def spec_dec_mode(self):
        from tensorrt_llm._torch.speculative.interface import \
            SpeculativeDecodingMode as TorchSpeculativeDecodingMode
        if self.eagle3_one_model:
            return TorchSpeculativeDecodingMode.EAGLE3_ONE_MODEL
        return TorchSpeculativeDecodingMode.EAGLE3

    @functools.cached_property
    def num_capture_layers(self) -> int:
        """
        Returns the number of layers to capture of the target model.
        If eagle3_layers_to_capture is not None, return the length of the set.
        Otherwise, assume Eagle3 base set and return 3.
        """
        if self.eagle3_layers_to_capture is not None:
            return len(self.eagle3_layers_to_capture)
        return 3

    @functools.cached_property
    def is_linear_tree(self) -> bool:
        if self.eagle_choices is None and self.use_dynamic_tree is False:
            return True
        return False


class Eagle3DecodingConfig(EagleDecodingConfig):
    decoding_type: Literal["Eagle3"] = "Eagle3"


class SaveHiddenStatesDecodingConfig(DecodingBaseConfig):
    decoding_type: Literal["SaveState"] = "SaveState"
    output_directory: str = Field(
        description=
        "Directory path where hidden states data files will be saved. The directory is created if it does not exist."
    )
    write_interval: int = Field(
        default=20,
        description=
        "Number of requests to process before writing accumulated hidden states to disk. Lower values write more frequently but may impact performance."
    )
    file_prefix: str = Field(
        default="data",
        description=
        "Prefix for output filenames. Files are saved as '<file_prefix>_<iteration>.pt' containing input_ids and hidden_state tensors."
    )
    eagle3_layers_to_capture: Optional[Set[int]] = Field(
        default=None,
        description=
        "Set of target model layer indices to capture hidden states from for EAGLE3 draft model training. "
        "Use -1 to indicate the final post-norm hidden state. If not provided, defaults to capturing 3 intermediate layers "
        "plus the post-norm hidden state. When provided, -1 is automatically added if not present."
    )

    max_total_draft_tokens: Optional[int] = Field(
        default=1,
        init=False,
        description=
        "Internal field, not user-configurable. Fixed to 1 since this mode captures hidden states without draft token generation."
    )
    eagle_choices: Optional[List[List[int]]] = Field(
        default=None,
        init=False,
        description=
        "Internal field, not user-configurable. Always None since this mode does not use tree-based draft token structures."
    )

    _last_hidden_in_save: bool = PrivateAttr(default=True)

    def model_post_init(self, __context):
        self._last_hidden_in_save = True
        if self.eagle3_layers_to_capture is None or -1 not in self.eagle3_layers_to_capture:
            # This variable is queried to determine whether we should write the final hidden state
            # to the aux_hidden_states buffer.
            self._last_hidden_in_save = False
        elif len(self.eagle3_layers_to_capture) == 0:
            raise ValueError(
                "eagle3_layers_to_capture must be non-empty if provided")

    @functools.cached_property
    def spec_dec_mode(self):
        from tensorrt_llm._torch.speculative.interface import \
            SpeculativeDecodingMode as TorchSpeculativeDecodingMode
        return TorchSpeculativeDecodingMode.SAVE_HIDDEN_STATES

    @functools.cached_property
    def num_capture_layers(self):
        """
        Returns the number of layers to save.
        The following hidden states are saved:
        - If eagle3_layers_to_capture is None, save the eagle3 base set plus
        the post norm last hidden state.
        - Otherwise, save the specified layers plus the post norm last hidden state.

        The saved data will contain two tensors, hidden_states and aux_hidden_states.
        * hidden_states will contain the last post norm state.
        * aux_hidden_states will contain all other captured layers. The last hidden state
        will also be included in this tensor if you explicitly captured layer -1.

        Note that if you set layers to capture to {-1}, aux_hidden_states won't exist.
        """
        if self.eagle3_layers_to_capture is None:
            return 4
        return len(self.eagle3_layers_to_capture) + int(
            -1 not in self.eagle3_layers_to_capture)


class UserProvidedDecodingConfig(DecodingBaseConfig):
    decoding_type: Literal["User_Provided"] = "User_Provided"
    # Cannot use real type annotations due to circular imports
    drafter: object = Field(
        description=
        "User-provided Drafter instance implementing the prepare_draft_tokens() method for custom draft token generation. "
        "See tensorrt_llm/_torch/speculative/drafter.py for the Drafter base class interface."
    )  # Type is Drafter
    resource_manager: object = Field(
        default=None,
        description=
        "Optional user-provided BaseResourceManager instance for managing resources (memory, caches) during drafting. "
        "Called to prepare/free resources before/after target model forward passes."
    )  # Type is Optional[ResourceManager]

    @model_validator(mode="after")
    def set_max_total_draft_tokens(self):
        self.max_total_draft_tokens = self.max_draft_len  # Current UserProvided only supports linear tree
        return self


class NGramDecodingConfig(DecodingBaseConfig):
    """
    Configuration for NGram drafter speculative decoding.
    """
    decoding_type: Literal["NGram"] = "NGram"
    max_matching_ngram_size: PositiveInt = Field(
        default=2,
        description=
        "The length maximum of searching tokens (can be understood as length maximum of input tokens "
        "to search).")
    is_keep_all: bool = Field(
        default=True,
        description=
        "Whether to keep all candidate pattern-matches pairs, only one "
        "match is kept for each pattern if False.")
    is_use_oldest: bool = Field(
        default=True,
        description="Whether to provide the oldest match when pattern is hit, "
        "the newest one is provided if False.")
    is_public_pool: bool = Field(
        default=True,
        description="Whether to use a common pool for all requests, or the pool "
        "is private for each request if False.")

    @model_validator(mode="after")
    def validate_ngram_config(self):
        if self.max_draft_len is None or self.max_draft_len <= 0:
            raise ValueError("max_draft_len must be > 0 for NGram")
        self.max_total_draft_tokens = self.max_draft_len  # Current NGram only supports linear tree
        return self

    def supports_backend(self, backend: str) -> bool:
        return backend == "pytorch"


class DraftTargetDecodingConfig(DecodingBaseConfig):
    decoding_type: Literal["Draft_Target"] = "Draft_Target"

    @model_validator(mode="after")
    def validate_draft_target_config(self):
        if self.max_draft_len is None or self.max_draft_len <= 0:
            raise ValueError("max_draft_len must be > 0 for DraftTarget")
        if self.speculative_model is None:
            raise ValueError(
                "speculative_model must be specified for DraftTarget")
        self.max_total_draft_tokens = self.max_draft_len  # Current DraftTarget only supports linear tree
        return self

    def supports_backend(self, backend: str) -> bool:
        return backend == "pytorch" or backend == "_autodeploy"


class MTPDecodingConfig(DecodingBaseConfig):
    decoding_type: Literal["MTP"] = "MTP"
    num_nextn_predict_layers: PositiveInt = Field(
        default=1,
        description=
        "Number of MTP modules. Each module predicts the next token, so N modules produce N draft tokens."
    )
    use_relaxed_acceptance_for_thinking: bool = Field(
        default=False,
        description=
        "Enable relaxed acceptance during thinking phase for reasoning models. Accepts draft tokens matching any top-K candidate instead of exact top-1."
    )
    relaxed_topk: int = Field(
        default=1,
        description=
        "Number of top candidate tokens to consider for relaxed acceptance. Draft token is accepted if it matches any of these."
    )
    relaxed_delta: float = Field(
        default=0.,
        description=
        "Probability threshold for relaxed acceptance. Only candidates with prob >= (top-1 prob - delta) are kept."
    )
    use_mtp_vanilla: bool = Field(
        default=False,
        description=
        "Force vanilla MTP mode (sequential MTP layers). When False, uses EAGLE-style MTP for single-layer checkpoints."
    )
    mtp_eagle_one_model: bool = Field(
        default=True,
        description=
        "When using EAGLE-style MTP, use faster one-model implementation (drafter as submodule) vs two-model."
    )

    # TODO: remove this after distinguishing `max_draft_len` and `num_nextn_predict_layers`
    # Now we need a flag when MTPDecodingConfig is updated by PyTorchModelEngine.
    num_nextn_predict_layers_from_model_config: int = Field(
        default=1,
        init=False,
        description=
        "Internal field storing MTP layer count from model config. Used to decide decoding mode: "
        "when model has 1 layer and use_mtp_vanilla=False, uses faster EAGLE-style MTP instead of vanilla MTP."
    )

    begin_thinking_phase_token: int = Field(
        default=128798,
        description=
        "Token ID marking start of thinking phase. Relaxed acceptance only applies within this phase."
    )
    end_thinking_phase_token: int = Field(
        default=128799,
        description=
        "Token ID marking end of thinking phase. Strict acceptance resumes after this."
    )

    @model_validator(mode="after")
    def set_max_total_draft_tokens(self):
        self.max_draft_len = self.num_nextn_predict_layers
        self.max_total_draft_tokens = self.num_nextn_predict_layers  # Current MTP only supports linear tree
        return self

    @model_validator(mode="after")
    def log_two_model_deprecation_warning(self):
        if not self.mtp_eagle_one_model:
            logger.warning(
                "2-model style MTP is deprecated. The mtp_eagle_one_model flag will do nothing "
                "in release 1.3. After that, the flag will be removed entirely."
            )
        return self

    def supports_backend(self, backend: str) -> bool:
        return backend == "pytorch"

    @functools.cached_property
    def num_capture_layers(self) -> int:
        if not self.use_mtp_vanilla and not self.mtp_eagle_one_model:
            return 1
        return 0

    @functools.cached_property
    def spec_dec_mode(self):
        from tensorrt_llm._torch.speculative.interface import \
            SpeculativeDecodingMode as TorchSpeculativeDecodingMode
        if self.num_nextn_predict_layers_from_model_config == 1 and not self.use_mtp_vanilla and self.mtp_eagle_one_model:
            return TorchSpeculativeDecodingMode.MTP_EAGLE_ONE_MODEL
        elif self.num_nextn_predict_layers_from_model_config == 1 and not self.use_mtp_vanilla and not self.mtp_eagle_one_model:
            return TorchSpeculativeDecodingMode.MTP_EAGLE
        return TorchSpeculativeDecodingMode.MTP


class PARDDecodingConfig(DecodingBaseConfig):
    """Configuration for PARD (Parallel Draft) speculative decoding.

    PARD is a target-independent speculative decoding method that uses
    mask tokens to predict multiple draft tokens in parallel within a
    single forward pass.

    Key features:
    - Target-independent: doesn't use target model hidden states
    - Parallel prediction: all K draft tokens in one forward pass
    - Shared mask token: uses the same mask_token_id across all positions

    Reference: https://arxiv.org/pdf/2504.18583
    """
    mask_token_id: Optional[int] = Field(
        default=None,
        description=
        "The token ID used as a mask token for parallel draft prediction. "
        "If None, it will be read from the draft model config (typically vocab_size)."
    )

    decoding_type: Literal["PARD"] = "PARD"

    @model_validator(mode="after")
    def set_max_total_draft_tokens(self):
        self.max_total_draft_tokens = self.max_draft_len
        return self

    @property
    def tokens_per_gen_step(self) -> int:
        """PARD needs 2K tokens per gen request: K+1 accepted + K-1 masks."""
        return 2 * self.max_draft_len

    def supports_backend(self, backend: str) -> bool:
        return backend == "pytorch"

    @functools.cached_property
    def spec_dec_mode(self):
        from tensorrt_llm._torch.speculative.interface import \
            SpeculativeDecodingMode as TorchSpeculativeDecodingMode
        return TorchSpeculativeDecodingMode.PARD


class AutoDecodingConfig(DecodingBaseConfig):
    """
    Configuration for auto speculative decoding.

    This config will automatically select a good, draft-model free
    speculation algorithm with some heuristic.

    Attributes that are inherited from the base class are ignored.
    """

    decoding_type: Literal["AUTO"] = "AUTO"

    @model_validator(mode="after")
    def set_max_total_draft_tokens(self):
        self.max_total_draft_tokens = self.max_draft_len  # Current Auto only supports linear tree
        return self

    def supports_backend(self, backend: str) -> bool:
        return backend == "pytorch"


class RayPlacementConfig(StrictBaseModel):
    """
    Configuration for Ray GPU workers placement.
    Currently, this config is only used with AsyncLLM for RL scenarios.
    """
    defer_workers_init: bool = Field(
        default=False,
        description="Defer Ray worker initialization until async setup.")

    placement_groups: Optional[List[Any]] = Field(
        default=None,
        description="List of Ray placement groups, one per node. "
        "Each element must be a ray.util.placement_group.PlacementGroup instance."
    )

    placement_bundle_indices: Optional[List[List[int]]] = Field(
        default=None,
        description=
        "List of lists of bundle indices. The outer list corresponds to "
        "`placement_groups`. Each inner list specifies the bundle indices to use within "
        "that placement group. For example, if `placement_groups=[pg1, pg2]`, "
        "`[[0, 1], [0, 1]]` assigns bundles 0 and 1 from `pg1` and bundles 0 and 1 from `pg2`."
    )

    per_worker_gpu_share: Optional[float] = Field(
        default=None,
        description="GPU fraction per worker for colocation scenarios. "
        "Example: 0.1 means 10 actors can share one GPU. Defaults to 1.0 (one actor per GPU)."
    )

    @model_validator(mode='after')
    def validate_ray_placement(self) -> 'RayPlacementConfig':
        has_pgs = self.placement_groups is not None
        has_indices = self.placement_bundle_indices is not None

        if has_pgs != has_indices:
            raise ValueError(
                "placement_groups and placement_bundle_indices must be provided together"
            )

        if has_pgs:
            if len(self.placement_groups) != len(self.placement_bundle_indices):
                raise ValueError(
                    f"placement_groups length ({len(self.placement_groups)}) must equal "
                    f"placement_bundle_indices length ({len(self.placement_bundle_indices)})"
                )
            if PlacementGroup is None:
                raise ValueError(
                    "Ray must be installed to use `placement_groups`")

            for i, pg in enumerate(self.placement_groups):
                if not isinstance(pg, PlacementGroup):
                    raise TypeError(
                        f"placement_groups[{i}] must be a Ray PlacementGroup, "
                        f"got {type(pg).__name__}")

        if self.per_worker_gpu_share is not None:
            if not (0 < self.per_worker_gpu_share <= 1.0):
                raise ValueError(
                    f"per_worker_gpu_share must be between 0 and 1.0, "
                    f"got {self.per_worker_gpu_share}")

        return self


class PybindMirror(ABC):
    ''' A class containing the utilities for mirroring Python classes to
    pybinding classes.
    '''

    @abstractmethod
    def _to_pybind(self):
        pass

    @staticmethod
    def maybe_to_pybind(ins):
        if isinstance(
                ins,
                PybindMirror) or type(ins).__class__ == PybindMirrorEnumMeta:
            return ins._to_pybind()
        return ins

    @staticmethod
    def mirror_pybind_fields(pybind_class):
        """
        Class decorator that ensures Python class fields mirror those of a C++ class.

        Args:
            pybind_class: The C++ class whose fields should be mirrored

        Returns:
            A decorator function that validates field mirroring
        """

        def decorator(cls):
            assert issubclass(cls, StrictBaseModel)
            # Get all non-private fields from the C++ class
            cpp_fields = PybindMirror.get_pybind_variable_fields(pybind_class)
            python_fields = set(cls.model_fields.keys())

            # Check if all C++ fields exist in the Python class
            for field in cpp_fields:
                if field not in python_fields:
                    raise ValueError(
                        f"Field {field} is not mirrored in Python class {cls.__name__} from C++ class {pybind_class.__name__}. Please update the class."
                    )

            # Return the original class
            return cls

        return decorator

    @staticmethod
    def get_pybind_enum_fields(pybind_class):
        ''' Get all the enum fields from the pybind class. '''
        return [
            f for f in pybind_class.__members__.keys()
            if not f.startswith('_') and not callable(getattr(pybind_class, f))
        ]

    @staticmethod
    def mirror_pybind_enum(pybind_class):
        ''' Mirror the enum fields from the pybind class to the Python class. '''

        def decorator(cls):
            assert issubclass(cls, Enum)
            cpp_fields = PybindMirror.get_pybind_enum_fields(pybind_class)
            python_fields = set(cls.__members__.keys())

            for field in cpp_fields:
                if field not in python_fields:
                    raise ValueError(
                        f"Field {field} is not mirrored in Python class {cls.__name__} from C++ class {pybind_class.__name__}. Please update the class."
                    )
            return cls

        return decorator

    @staticmethod
    def get_pybind_variable_fields(config_cls):
        ''' Get all the variable fields from the pybind class. '''
        return [
            f for f in dir(config_cls)
            if not f.startswith('_') and not callable(getattr(config_cls, f))
        ]

    @staticmethod
    def pybind_equals(obj0, obj1):
        ''' Check if two pybind objects are equal. '''
        assert type(obj0) is type(obj1)
        for field in PybindMirror.get_pybind_variable_fields(type(obj0)):
            if getattr(obj0, field) != getattr(obj1, field):
                return False
        return True

    @classmethod
    def from_pybind(cls: Type[TypeBaseModel],
                    pybind_instance: "PybindMirror") -> TypeBaseModel:
        """Construct an instance of the given class from the fields in the given
        pybind class instance.

        Args:
            cls: Type of the class to construct, must be a subclass of pydantic
                 BaseModel
            pybind_instance: Instance of the pybind class to construct from its
                             fields

        Notes:
            When a field value is None in the pybind class, but it's not
            optional and has a default value in the BaseModel class, it would
            get the default value defined in the BaseModel class.

        Returns:
            Instance of the given class, populated with the fields of the given
            pybind instance
        """  # noqa: D205
        assert issubclass(cls, BaseModel)

        # Some of the fields are optional in the C++ class but in python they aren't
        # optional and have a default value, so copy the value from C++ instance
        # only if it has a value, so otherwise the default value defined in the
        # python class would be set.
        def _is_optional_type(annotation: Any) -> bool:
            """Returns True if a type annotation represents an Optional type
            (Optional[X]) or a Union type that includes None (Union[X, Y, None]
            or X | Y | None).
            """  # noqa: D205
            origin = get_origin(annotation)
            args = get_args(annotation)

            # Union is for Optional[x]
            # UnionType is for the new | operation in Python 3.10+
            return (origin is Union
                    or origin is types.UnionType) and type(None) in args

        fields_non_optional_with_default_value_in_basemodel = {
            field_name
            for field_name, field_info in cls.model_fields.items()
            if not (_is_optional_type(field_info.annotation)
                    and field_info.is_required())
        }

        kwargs = {}
        cpp_fields = PybindMirror.get_pybind_variable_fields(
            type(pybind_instance))
        for field_name in cpp_fields:
            field_value = getattr(pybind_instance, field_name)
            if field_value is not None or field_name not in fields_non_optional_with_default_value_in_basemodel:
                kwargs[field_name] = field_value
        return cls(**kwargs)


class PybindMirrorMeta(type(PybindMirror)):
    pass


class PybindMirrorEnumMeta(EnumMeta, PybindMirrorMeta):
    """
    Combined metaclass for Enum and PybindMirror.  This is crucial.
    """


@PybindMirror.mirror_pybind_enum(_BatchingType)
class BatchingType(StrEnum, metaclass=PybindMirrorEnumMeta):
    STATIC = "STATIC"
    INFLIGHT = "INFLIGHT"

    def _to_pybind(self):
        return getattr(_BatchingType, self.value)


@PybindMirror.mirror_pybind_enum(_CapacitySchedulerPolicy)
class CapacitySchedulerPolicy(StrEnum, metaclass=PybindMirrorEnumMeta):
    MAX_UTILIZATION = "MAX_UTILIZATION"
    GUARANTEED_NO_EVICT = "GUARANTEED_NO_EVICT"
    STATIC_BATCH = "STATIC_BATCH"

    def _to_pybind(self):
        return getattr(_CapacitySchedulerPolicy, self.value)


@PybindMirror.mirror_pybind_enum(_ContextChunkingPolicy)
class ContextChunkingPolicy(StrEnum, metaclass=PybindMirrorEnumMeta):
    ''' Context chunking policy. '''
    FIRST_COME_FIRST_SERVED = "FIRST_COME_FIRST_SERVED"
    EQUAL_PROGRESS = "EQUAL_PROGRESS"

    def _to_pybind(self):
        return getattr(_ContextChunkingPolicy, self.value)


class WaitingQueuePolicy(StrEnum):
    """Waiting queue scheduling policy for managing pending requests."""

    FCFS = "fcfs"  # First-Come-First-Served


@PybindMirror.mirror_pybind_fields(_DynamicBatchConfig)
class DynamicBatchConfig(StrictBaseModel, PybindMirror):
    """Dynamic batch configuration.

    Controls how batch size and token limits are dynamically adjusted at runtime.
    """
    enable_batch_size_tuning: bool = Field(
        default=True,
        description="Controls if the batch size should be tuned dynamically")

    enable_max_num_tokens_tuning: bool = Field(
        default=False,
        description="Controls if the max num tokens should be tuned dynamically"
    )

    dynamic_batch_moving_average_window: int = Field(
        default=128,
        description=
        "The window size for moving average of input and output length which is used to calculate dynamic batch size and max num tokens"
    )

    def _to_pybind(self):
        return _DynamicBatchConfig(
            enable_batch_size_tuning=self.enable_batch_size_tuning,
            enable_max_num_tokens_tuning=self.enable_max_num_tokens_tuning,
            dynamic_batch_moving_average_window=self.
            dynamic_batch_moving_average_window)


@PybindMirror.mirror_pybind_fields(_SchedulerConfig)
class SchedulerConfig(StrictBaseModel, PybindMirror):
    capacity_scheduler_policy: CapacitySchedulerPolicy = Field(
        default=CapacitySchedulerPolicy.GUARANTEED_NO_EVICT,
        description="The capacity scheduler policy to use")

    context_chunking_policy: Optional[ContextChunkingPolicy] = Field(
        default=None, description="The context chunking policy to use")

    dynamic_batch_config: Optional[DynamicBatchConfig] = Field(
        default=None,
        description=
        "The dynamic batch config to use. This only applies for the TensorRT backend and "
        "cannot currently be used with the PyTorch backend.")

    waiting_queue_policy: WaitingQueuePolicy = Field(
        default=WaitingQueuePolicy.FCFS,
        description="The waiting queue scheduling policy")

    def _to_pybind(self):
        return _SchedulerConfig(
            capacity_scheduler_policy=self.capacity_scheduler_policy._to_pybind(
            ),
            context_chunking_policy=self.context_chunking_policy._to_pybind()
            if self.context_chunking_policy else None,
            dynamic_batch_config=self.dynamic_batch_config._to_pybind()
            if self.dynamic_batch_config else None)


@PybindMirror.mirror_pybind_fields(_PeftCacheConfig)
class PeftCacheConfig(StrictBaseModel, PybindMirror):
    """
    Configuration for the PEFT cache.
    """
    num_host_module_layer: int = Field(
        default=0,
        description=
        "number of max sized 1-layer 1-module adapterSize=1 sets of weights that can be stored in host cache"
        ", affects host cache size and overrides value of host_cache_size")
    num_device_module_layer: int = Field(
        default=0,
        description=
        "number of max sized 1-layer 1-module sets of weights that can be stored in device cache"
        ", affects device cache size and overrides value of device_cache_percent"
    )
    optimal_adapter_size: int = Field(
        default=
        8,  # There are tests to keep the default value consistent with the pybind default value
        description="optimal adapter size used to set page width")
    max_adapter_size: int = Field(
        default=64,
        description="max supported adapter size. Used to compute minimum")
    num_put_workers: int = Field(
        default=1,
        description=
        "number of worker threads used to put weights into host cache")
    num_ensure_workers: int = Field(
        default=1,
        description=
        "number of worker threads used to copy weights from host to device")
    num_copy_streams: int = Field(
        default=1,
        description="number of streams used to copy weights from host to device"
    )
    max_pages_per_block_host: int = Field(
        default=24,
        description="Number of cache pages per allocation block (host)")
    max_pages_per_block_device: int = Field(
        default=8,
        description="Number of cache pages per allocation block (device)")
    device_cache_percent: float = Field(
        default=0.02,
        description=
        "Proportion of free device memory after engine load to use for cache, as a fraction from 0 to 1"
    )
    host_cache_size: int = Field(
        default=1024**3, description="size in bytes to use for host cache")
    lora_prefetch_dir: Optional[str] = Field(
        default=None,
        description=
        "folder to store the LoRA weights we hope to load during engine initialization, currently not supported"
    )

    def _to_pybind(self):
        return _PeftCacheConfig(
            num_host_module_layer=self.num_host_module_layer,
            num_device_module_layer=self.num_device_module_layer,
            optimal_adapter_size=self.optimal_adapter_size,
            max_adapter_size=self.max_adapter_size,
            num_put_workers=self.num_put_workers,
            num_ensure_workers=self.num_ensure_workers,
            num_copy_streams=self.num_copy_streams,
            max_pages_per_block_host=self.max_pages_per_block_host,
            max_pages_per_block_device=self.max_pages_per_block_device,
            device_cache_percent=self.device_cache_percent,
            host_cache_size=self.host_cache_size,
            lora_prefetch_dir=self.lora_prefetch_dir)


@PybindMirror.mirror_pybind_fields(_LookaheadDecodingConfig)
class LookaheadDecodingConfig(DecodingBaseConfig, PybindMirror):
    """
    Configuration for lookahead speculative decoding.
    """

    decoding_type: Literal["Lookahead"] = "Lookahead"
    max_window_size: PositiveInt = Field(
        default=_LookaheadDecodingConfig.get_default_lookahead_decoding_window(
        ),
        description="Number of NGrams in lookahead branch per step.")
    max_ngram_size: PositiveInt = Field(
        default=_LookaheadDecodingConfig.get_default_lookahead_decoding_ngram(),
        description="Number of tokens per NGram.")
    max_verification_set_size: PositiveInt = Field(
        default=_LookaheadDecodingConfig.
        get_default_lookahead_decoding_verification_set(),
        description="Number of NGrams in verification branch per step.")

    @model_validator(mode="after")
    def set_max_total_draft_tokens(self):
        self.max_total_draft_tokens = self.max_draft_len  # Current Lookahead only supports linear tree
        return self

    def calculate_speculative_resource(self):
        return _LookaheadDecodingConfig.calculate_speculative_resource_tuple(
            self.max_window_size, self.max_ngram_size,
            self.max_verification_set_size)

    def _to_pybind(self):
        return _LookaheadDecodingConfig(self.max_window_size,
                                        self.max_ngram_size,
                                        self.max_verification_set_size)

    def supports_backend(self, backend: str) -> bool:
        return backend not in ("pytorch", "_autodeploy")


SpeculativeConfig: TypeAlias = Annotated[
    Union[
        DraftTargetDecodingConfig,
        EagleDecodingConfig,
        Eagle3DecodingConfig,
        LookaheadDecodingConfig,
        MedusaDecodingConfig,
        MTPDecodingConfig,
        NGramDecodingConfig,
        UserProvidedDecodingConfig,
        SaveHiddenStatesDecodingConfig,
        PARDDecodingConfig,
        AutoDecodingConfig,
    ],
    Field(discriminator="decoding_type"),
]

SparseAttentionConfig: TypeAlias = Annotated[
    Union[
        RocketSparseAttentionConfig,
        DeepSeekSparseAttentionConfig,
        SkipSoftmaxAttentionConfig,
    ],
    Field(discriminator="algorithm"),
]


@PybindMirror.mirror_pybind_fields(_KvCacheConfig)
class KvCacheConfig(StrictBaseModel, PybindMirror):
    """
    Configuration for the KV cache.
    """
    enable_block_reuse: bool = Field(
        default=True,
        description=
        "Controls if KV cache blocks can be reused for different requests.")
    max_tokens: Optional[int] = Field(
        default=None,
        description=
        "The maximum number of tokens that should be stored in the KV cache. If both `max_tokens` and `free_gpu_memory_fraction` are specified, memory corresponding to the minimum will be used."
    )
    max_attention_window: Optional[List[PositiveInt]] = Field(
        default=None,
        min_length=1,
        description=
        "Size of the attention window for each sequence. Only the last tokens will be stored in the KV cache. If the number of elements in `max_attention_window` is less than the number of layers, `max_attention_window` will be repeated multiple times to the number of layers."
    )
    sink_token_length: Optional[int] = Field(
        default=None,
        description=
        "Number of sink tokens (tokens to always keep in attention window).")
    free_gpu_memory_fraction: Optional[float] = Field(
        default=0.9,
        ge=0,
        le=1,
        description=
        "The fraction of GPU memory fraction that should be allocated for the KV cache. Default is 90%. If both `max_tokens` and `free_gpu_memory_fraction` are specified, memory corresponding to the minimum will be used."
    )
    host_cache_size: Optional[int] = Field(
        default=None,
        description=
        "Size of the host cache in bytes. If both `max_tokens` and `host_cache_size` are specified, memory corresponding to the minimum will be used."
    )
    onboard_blocks: bool = Field(
        default=True, description="Controls if blocks are onboarded.")
    cross_kv_cache_fraction: Optional[float] = Field(
        default=None,
        description=
        "The fraction of the KV Cache memory should be reserved for cross attention. If set to p, self attention will use 1-p of KV Cache memory and cross attention will use p of KV Cache memory. Default is 50%. Should only be set when using encoder-decoder model."
    )
    secondary_offload_min_priority: Optional[int] = Field(
        default=None,
        description=
        "Only blocks with priority > mSecondaryOfflineMinPriority can be offloaded to secondary memory."
    )
    event_buffer_max_size: int = Field(
        default=0,
        description=
        "Maximum size of the event buffer. If set to 0, the event buffer will not be used."
    )
    attention_dp_events_gather_period_ms: int = Field(
        default=5,
        description=
        "The period in milliseconds to gather attention DP events across ranks."
    )
    enable_partial_reuse: bool = Field(
        default=True,
        description=
        "Whether blocks that are only partially matched can be reused.")
    copy_on_partial_reuse: bool = Field(
        default=True,
        description=
        "Whether partially matched blocks that are in use can be reused after copying them."
    )
    use_uvm: bool = Field(default=False,
                          description="Whether to use UVM for the KV cache.")
    max_gpu_total_bytes: NonNegativeInt = Field(
        default=0,
        description=
        "The maximum size in bytes of GPU memory that can be allocated for the KV cache. If both `max_gpu_total_bytes` and `free_gpu_memory_fraction` are specified, memory corresponding to the minimum will be allocated."
    )

    # This is a pure python field, not a pybind field. It is only for the Pytorch backend.
    dtype: str = Field(default="auto",
                       description="The data type to use for the KV cache.")

    # This is a pure python field, not a pybind field. It is only for the Pytorch backend.
    mamba_ssm_cache_dtype: Literal[
        "auto", "float16", "bfloat16", "float32"] = Field(
            default="auto",
            description=
            "The data type to use for the Mamba SSM cache. If set to 'auto', the data type will be inferred from the model config."
        )

    tokens_per_block: int = Field(default=32,
                                  description="The number of tokens per block.")

    use_kv_cache_manager_v2: bool = Field(
        default=False,
        status="prototype",
        description="Whether to use the KV cache manager v2 (experimental).")

    max_util_for_resume: float = Field(
        default=0.95,
        ge=0,
        le=1,
        status="prototype",
        description=
        "The maximum utilization of the KV cache for resume. Default is 95%. Only used when using KV cache manager v2 (experimental)."
    )

    def _to_pybind(self):
        return _KvCacheConfig(
            enable_block_reuse=self.enable_block_reuse,
            max_tokens=self.max_tokens,
            max_attention_window=self.max_attention_window,
            sink_token_length=self.sink_token_length,
            free_gpu_memory_fraction=self.free_gpu_memory_fraction,
            host_cache_size=self.host_cache_size,
            onboard_blocks=self.onboard_blocks,
            cross_kv_cache_fraction=self.cross_kv_cache_fraction,
            secondary_offload_min_priority=self.secondary_offload_min_priority,
            event_buffer_max_size=self.event_buffer_max_size,
            enable_partial_reuse=self.enable_partial_reuse,
            copy_on_partial_reuse=self.copy_on_partial_reuse,
            use_uvm=self.use_uvm,
            attention_dp_events_gather_period_ms=self.
            attention_dp_events_gather_period_ms,
            max_gpu_total_bytes=self.max_gpu_total_bytes)


@PybindMirror.mirror_pybind_fields(_ExtendedRuntimePerfKnobConfig)
class ExtendedRuntimePerfKnobConfig(StrictBaseModel, PybindMirror):
    """
    Configuration for extended runtime performance knobs.
    """

    multi_block_mode: bool = Field(
        default=True, description="Whether to use multi-block mode.")

    enable_context_fmha_fp32_acc: bool = Field(
        default=False,
        description="Whether to enable context FMHA FP32 accumulation.")

    cuda_graph_mode: bool = Field(default=False,
                                  description="Whether to use CUDA graph mode.")

    cuda_graph_cache_size: int = Field(
        default=0,
        description=
        "Number of cuda graphs to be cached in the runtime. The larger the cache, the better the perf, but more GPU memory is consumed."
    )

    def _to_pybind(self):
        res = _ExtendedRuntimePerfKnobConfig(
            multi_block_mode=self.multi_block_mode,
            enable_context_fmha_fp32_acc=self.enable_context_fmha_fp32_acc)
        res.cuda_graph_mode = self.cuda_graph_mode
        res.cuda_graph_cache_size = self.cuda_graph_cache_size
        return res


@PybindMirror.mirror_pybind_fields(_CacheTransceiverConfig)
class CacheTransceiverConfig(StrictBaseModel, PybindMirror):
    """
    Configuration for the cache transceiver.
    """

    backend: Optional[Literal[
        "DEFAULT", "UCX", "NIXL", "MOONCAKE", "MPI"]] = Field(
            default=None,
            description=
            "The communication backend type to use for the cache transceiver.")

    transceiver_runtime: Optional[Literal["CPP", "PYTHON"]] = Field(
        default=None,
        description=
        "The runtime implementation. 'CPP' for C++ transceiver (default when not set), 'PYTHON' for Python transceiver."
    )

    max_tokens_in_buffer: Optional[int] = Field(
        default=None,
        description="The max number of tokens the transfer buffer can fit.")

    kv_transfer_timeout_ms: Optional[PositiveInt] = Field(
        default=None,
        description=
        "Timeout in milliseconds for KV cache transfer. Requests exceeding this timeout will be cancelled."
    )

    kv_transfer_sender_future_timeout_ms: Optional[PositiveInt] = Field(
        default=1000,
        description=
        "Timeout in milliseconds to wait for the sender future to be ready when scheduled batch size is 0. This allows the request to be eventually cancelled by the user or because of kv_transfer_timeout_ms"
    )

    def _to_pybind(self):
        return _CacheTransceiverConfig(
            backend=_CacheTransceiverBackendType.from_string(self.backend),
            max_tokens_in_buffer=self.max_tokens_in_buffer,
            kv_transfer_timeout_ms=self.kv_transfer_timeout_ms,
            kv_transfer_sender_future_timeout_ms=self.
            kv_transfer_sender_future_timeout_ms)


@dataclass
class _ModelWrapper:
    model: Union[str, Path]

    def __post_init__(self):
        if not self.model:
            raise ValueError("model should be provided.")
        assert isinstance(self.model,
                          (str, Path)), f"Invalid model: {self.model}"

        model_dir = Path(self.model)

        if model_dir.exists() and model_dir.is_dir():
            self.model = model_dir

    @property
    def is_hub_model(self) -> bool:
        return not self.is_local_model

    @property
    def is_local_model(self) -> bool:
        return isinstance(self.model, Path)

    @property
    def model_dir(self) -> Path:
        assert self.is_local_model, f"model_dir is only available for local model, {self.model}."
        return self.model

    @model_dir.setter
    def model_dir(self, model_dir: Union[str, Path]):
        model_dir = Path(model_dir)
        assert model_dir.exists() and model_dir.is_dir(
        ), f"model_dir is not a valid path, {model_dir}"
        self.model = model_dir

    @property
    def model_name(self) -> Union[str, Path]:
        return self.model if isinstance(self.model, str) else None


class BaseLlmArgs(StrictBaseModel):
    """
    Base class for both TorchLlmArgs and TrtLlmArgs. It contains all the arguments that are common to both.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    # Explicit arguments
    model: Union[str, Path] = Field(
        description=
        "The path to the model checkpoint or the model name from the Hugging Face Hub."
    )

    tokenizer: Optional[Union[
        str, Path, TokenizerBase, PreTrainedTokenizerBase]] = Field(
            description=
            "The path to the tokenizer checkpoint or the tokenizer name from the Hugging Face Hub.",
            default=None)

    tokenizer_mode: Literal['auto', 'slow'] = Field(
        default='auto',
        description="The mode to initialize the tokenizer.",
        json_schema_extra={"type": "Literal['auto', 'slow']"})

    custom_tokenizer: Optional[str] = Field(
        default=None,
        description="Specify a custom tokenizer implementation. Accepts either: "
        "(1) a built-in alias (e.g., 'deepseek_v32'), or "
        "(2) a Python import path (e.g., 'tensorrt_llm.tokenizer.deepseek_v32.DeepseekV32Tokenizer'). "
        "The tokenizer class must implement 'from_pretrained(path, **kwargs)' and the TokenizerBase interface.",
        status="prototype")

    skip_tokenizer_init: bool = Field(
        default=False,
        description="Whether to skip the tokenizer initialization.")

    trust_remote_code: bool = Field(
        default=False, description="Whether to trust the remote code.")

    tensor_parallel_size: int = Field(default=1,
                                      description="The tensor parallel size.")

    dtype: str = Field(default="auto",
                       description="The data type to use for the model.")

    revision: Optional[str] = Field(
        default=None, description="The revision to use for the model.")

    tokenizer_revision: Optional[str] = Field(
        default=None, description="The revision to use for the tokenizer.")

    # Below are all remaining arguments

    model_kwargs: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional parameters overriding model config defaults. "
        "Precedence: (1) model_kwargs, (2) model config file, (3) model config class defaults. "
        "Unknown keys are ignored",
        status="prototype")

    pipeline_parallel_size: int = Field(
        default=1, description="The pipeline parallel size.")

    context_parallel_size: int = Field(default=1,
                                       description="The context parallel size.")

    gpus_per_node: Optional[int] = Field(
        default=None,
        description="The number of GPUs per node.",
        status="beta",
        validate_default=True)

    moe_cluster_parallel_size: Optional[int] = Field(
        default=None,
        description="The cluster parallel size for MoE models's expert weights.",
        status="beta")

    moe_tensor_parallel_size: Optional[int] = Field(
        default=None,
        description="The tensor parallel size for MoE models's expert weights.")

    moe_expert_parallel_size: Optional[int] = Field(
        default=None,
        description="The expert parallel size for MoE models's expert weights.")

    enable_attention_dp: bool = Field(
        default=False,
        description="Enable attention data parallel.",
        status="beta")

    enable_lm_head_tp_in_adp: bool = Field(
        default=False,
        description="Enable LM head TP in attention dp.",
        status="prototype")

    pp_partition: Optional[List[int]] = Field(
        default=None,
        description=
        "Pipeline parallel partition, a list of each rank's layer number.",
        status="prototype")

    cp_config: Optional[CpConfig] = Field(
        default=None,
        description="Context parallel config.",
        status="prototype")

    load_format: Literal['auto', 'dummy'] = Field(
        default='auto',
        description="The format to load the model.",
        json_schema_extra={"type": "Literal['auto', 'dummy']"})

    # LoRA arguments
    enable_lora: bool = Field(default=False, description="Enable LoRA.")

    lora_config: Optional[LoraConfig] = Field(
        default=None, description="LoRA configuration for the model.")

    # Several options from ExecutorConfig, expanded here for less hierarchy
    kv_cache_config: KvCacheConfig = Field(default_factory=KvCacheConfig,
                                           description="KV cache config.")

    enable_chunked_prefill: bool = Field(default=False,
                                         description="Enable chunked prefill.")

    guided_decoding_backend: Optional[Literal["xgrammar", "llguidance"]] = Field(
        default=None,
        description=
        "Guided decoding backend. llguidance is supported in PyTorch backend only."
    )

    batched_logits_processor: Optional[object] = Field(
        default=None,
        description="Batched logits processor.",
        json_schema_extra={
            "type": f"Optional[{get_type_repr(BatchedLogitsProcessor)}]"
        })

    iter_stats_max_iterations: Optional[int] = Field(
        default=None,
        description="The maximum number of iterations for iter stats.",
        status="prototype")

    request_stats_max_iterations: Optional[int] = Field(
        default=None,
        description="The maximum number of iterations for request stats.",
        status="prototype")

    # A handful of options from PretrainedConfig
    peft_cache_config: Optional[PeftCacheConfig] = Field(
        default=None, description="PEFT cache config.", status="prototype")

    scheduler_config: SchedulerConfig = Field(default_factory=SchedulerConfig,
                                              description="Scheduler config.",
                                              status="prototype")

    cache_transceiver_config: Optional[CacheTransceiverConfig] = Field(
        default=None,
        description="Cache transceiver config.",
        status="prototype")

    # Sparse attention config
    sparse_attention_config: Optional[SparseAttentionConfig] = Field(
        default=None,
        description="Sparse attention config.",
        status="prototype")

    # Speculative decoding parameters
    speculative_config: Optional[SpeculativeConfig] = Field(
        default=None, description="Speculative decoding config.")

    max_batch_size: Optional[int] = Field(default=2048,
                                          description="The maximum batch size.")

    # generation constraints
    max_input_len: Optional[int] = Field(
        default=1024, description="The maximum input length.")

    max_seq_len: Optional[int] = Field(
        default=None, description="The maximum sequence length.")

    max_beam_width: Optional[int] = Field(default=1,
                                          description="The maximum beam width.")

    max_num_tokens: Optional[int] = Field(
        default=8192, description="The maximum number of tokens.")

    gather_generation_logits: bool = Field(
        default=False,
        description="Gather generation logits.",
        status="prototype")

    # private fields those are unstable and just for internal use
    num_postprocess_workers: int = Field(
        default=0,
        description=
        "The number of processes used for postprocessing the generated tokens, including detokenization.",
        status="prototype")

    postprocess_tokenizer_dir: Optional[str] = Field(
        default=None,
        description="The path to the tokenizer directory for postprocessing.",
        status="prototype")

    reasoning_parser: Optional[str] = Field(
        default=None,
        description="The parser to separate reasoning content from output.",
        status="prototype")

    # TODO[Superjomn]: To deprecate this config.
    decoding_config: Optional[object] = Field(
        default=None,
        description="The decoding config.",
        json_schema_extra={
            "type": "Optional[tensorrt_llm.llmapi.llm_args.DecodingConfig]"
        },
        status="deprecated",
        deprecated="Use speculative_config instead.",
    )

    mpi_session: Optional[object] = Field(
        default=None,
        description="The optional MPI session to use for this LLM instance.",
        json_schema_extra={"type": "Optional[MpiSession]"},
        exclude=True,
        alias="_mpi_session")

    otlp_traces_endpoint: Optional[str] = Field(
        default=None,
        description="Target URL to which OpenTelemetry traces will be sent.",
        alias="otlp_traces_endpoint",
        status="prototype")

    backend: Optional[str] = Field(
        default=None,
        description="The backend to use for this LLM instance.",
        exclude_json_schema=True,  # hide from API references
        validate_default=True,
        status="deprecated",
    )

    return_perf_metrics: bool = Field(default=False,
                                      description="Return perf metrics.",
                                      status="prototype")

    perf_metrics_max_requests: NonNegativeInt = Field(
        default=0,
        description=
        "The maximum number of requests for perf metrics. Must also set return_perf_metrics to true to get perf metrics.",
        status="prototype")

    orchestrator_type: Optional[Literal["rpc", "ray"]] = Field(
        default=None,
        description=
        "The orchestrator type to use. Defaults to None, which uses MPI.",
        status="prototype",
    )

    env_overrides: Optional[Dict[str, str]] = Field(
        default=None,
        description=
        "[EXPERIMENTAL] Environment variable overrides. NOTE: import-time-cached env vars in the code won't update unless the code fetches them from os.environ on demand.",
        status="prototype")

    @field_validator('env_overrides', mode='before')
    @classmethod
    def coerce_env_overrides_to_str(cls, v):
        """Coerce env_overrides values to strings for os.environ compatibility."""
        if v is None:
            return v
        return {str(k): str(val) for k, val in v.items()}

    _parallel_config: Optional[_ParallelConfig] = PrivateAttr(default=None)
    _model_format: Optional[_ModelFormatKind] = PrivateAttr(default=None)

    @property
    def parallel_config(self) -> _ParallelConfig:
        return self._parallel_config

    @property
    def model_format(self) -> _ModelFormatKind:
        return self._model_format

    @property
    def speculative_model(self) -> Optional[Union[str, Path]]:
        return self.speculative_config.speculative_model if self.speculative_config is not None else None

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]):
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    @field_validator("dtype")
    @classmethod
    def validate_dtype(cls, v, info):
        if torch.cuda.get_device_properties(0).major < 8:
            if v == 'auto':
                v = 'float16'
            if v == 'bfloat16':
                raise RuntimeError("Pre SM 80 GPUs do not support bfloat16")
        return v

    @field_validator("gpus_per_node", mode='before')
    @classmethod
    def validate_gpus_per_node(cls, v, info):
        if os.getenv("RAY_LOCAL_WORLD_SIZE") is not None:
            return info.data.get("tensor_parallel_size")
        if v is None:
            logger.warning(
                f"Using default gpus_per_node: {torch.cuda.device_count()}")
            v = torch.cuda.device_count()
        return v

    @model_validator(mode="after")
    def normalize_optional_fields_to_defaults(self):
        """Normalize certain fields to their declared default values in case a user explicitly sets them to None.

        This is necessary because downstream code expects these fields to be non-None.
        At the same time, we still need to accept None as a valid value to avoid a breaking change.
        """
        for field_name in (
                "max_batch_size",
                "max_input_len",
                "max_beam_width",
                "max_num_tokens",
        ):
            if getattr(self, field_name) is None:
                field_info = self.__class__.model_fields.get(field_name)
                if field_info is not None and field_info.default is not None:
                    setattr(self, field_name, field_info.default)
        return self

    @model_validator(mode="after")
    def validate_parallel_config(self):
        if self.moe_cluster_parallel_size is None:
            self.moe_cluster_parallel_size = -1

        if self.moe_tensor_parallel_size is None:
            self.moe_tensor_parallel_size = -1

        if self.moe_expert_parallel_size is None:
            self.moe_expert_parallel_size = -1

        self._parallel_config = _ParallelConfig(
            tp_size=self.tensor_parallel_size,
            pp_size=self.pipeline_parallel_size,
            cp_size=self.context_parallel_size,
            gpus_per_node=self.gpus_per_node,
            moe_cluster_size=self.moe_cluster_parallel_size,
            moe_tp_size=self.moe_tensor_parallel_size,
            moe_ep_size=self.moe_expert_parallel_size,
            enable_attention_dp=self.enable_attention_dp,
            enable_lm_head_tp_in_adp=self.enable_lm_head_tp_in_adp,
            pp_partition=self.pp_partition,
            cp_config=self.cp_config)
        return self

    @model_validator(mode="after")
    def validate_and_init_tokenizer(self):
        """Initialize tokenizer based on configuration."""
        if self.skip_tokenizer_init:
            self.tokenizer = None
        elif self.custom_tokenizer:
            # If tokenizer is already a tokenizer object, custom_tokenizer is not compatible
            if isinstance(self.tokenizer,
                          (TokenizerBase, PreTrainedTokenizerBase)):
                raise ValueError(
                    "Cannot use custom_tokenizer when tokenizer is already a tokenizer object. "
                    "Please specify a tokenizer path or leave it as None to load from model path."
                )

            # Support short aliases for built-in tokenizers
            TOKENIZER_ALIASES = {
                'deepseek_v32':
                'tensorrt_llm.tokenizer.deepseek_v32.DeepseekV32Tokenizer',
            }

            tokenizer_path = TOKENIZER_ALIASES.get(self.custom_tokenizer,
                                                   self.custom_tokenizer)

            # Dynamically import and use custom tokenizer
            from importlib import import_module
            try:
                module_path, class_name = tokenizer_path.rsplit('.', 1)
                module = import_module(module_path)
                tokenizer_class = getattr(module, class_name)
                # Use tokenizer path if specified, otherwise use model path
                load_path = self.tokenizer if self.tokenizer else self.model
                self.tokenizer = tokenizer_class.from_pretrained(
                    load_path,
                    trust_remote_code=self.trust_remote_code,
                    use_fast=self.tokenizer_mode != 'slow')
            except (ValueError, ImportError, AttributeError) as e:
                raise ValueError(
                    f"Failed to load custom tokenizer '{self.custom_tokenizer}': {e}. "
                    "Expected format: 'module.path.ClassName' or a recognized alias."
                ) from e
        else:
            self.tokenizer = tokenizer_factory(
                self.tokenizer,
                trust_remote_code=self.trust_remote_code,
                use_fast=self.tokenizer_mode != 'slow')
        return self

    @model_validator(mode="after")
    def validate_runtime_args(self):
        if self.max_batch_size is not None and self.max_num_tokens is not None:
            if self.max_batch_size > self.max_num_tokens:
                logger.warning(
                    f"max_batch_size [{self.max_batch_size}] should be less than or equal to max_num_tokens [{self.max_num_tokens}]"
                )
        return self

    @model_validator(mode="after")
    def validate_lora_config_consistency(self):
        if self.lora_config:
            if len(self.lora_config.lora_dir) == 0:
                # TODO [TRTLLM-5173]
                logger.warning(
                    "lora_dir is empty, so custom embedding or lm head will not be applied."
                )

        if self.enable_lora and self.lora_config is not None and self.backend in [
                'pytorch', '_autodeploy'
        ]:
            logger.warning(
                f"enable_lora is ignored when lora_config is provided for {self.backend} backend."
            )

        if self.lora_config is not None:
            if len(self.lora_config.lora_dir) == 0 and len(
                    self.lora_config.lora_target_modules) == 0:
                logger.warning(
                    "Both lora_dir and lora_target_modules are empty, so all LoRA modules will be expected. "
                    "This will lead to serious memory consumption. Please provide either lora_dir or lora_target_modules if this behavior is not what you expect."
                )
                default_trtllm_modules_to_hf_modules = get_default_trtllm_modules_to_hf_modules(
                )
                self.lora_config.lora_target_modules = list(
                    default_trtllm_modules_to_hf_modules.keys())
        return self

    @model_validator(mode="after")
    def validate_peft_cache_config(self):
        if self.peft_cache_config is not None and self.peft_cache_config.lora_prefetch_dir is not None:
            raise ValueError(
                f"lora_prefetch_dir was set to '{self.peft_cache_config.lora_prefetch_dir}' "
                "while LoRA prefetch is not supported")
        return self

    def get_runtime_sizes(self, ) -> Tuple[int, int, int, int]:
        return (
            self.max_beam_width,
            self.max_num_tokens,
            self.max_seq_len,
            self.max_batch_size,
        )


class TrtLlmArgs(BaseLlmArgs):
    enable_tqdm: bool = Field(default=False,
                              description="Enable tqdm for progress bar.")

    workspace: Optional[str] = Field(default=None,
                                     description="The workspace for the model.")

    fail_fast_on_attention_window_too_large: bool = Field(
        default=False,
        description=
        "Fail fast when attention window is too large to fit even a single sequence in the KV cache.",
        status="prototype")

    # Once set, the model will reuse the build_cache
    enable_build_cache: Union[BuildCacheConfig,
                              bool] = Field(default=False,
                                            description="Enable build cache.")

    extended_runtime_perf_knob_config: Optional[
        ExtendedRuntimePerfKnobConfig] = Field(
            default=None, description="Extended runtime perf knob config.")

    # Quantization and calibration configurations
    calib_config: CalibConfig = Field(default_factory=CalibConfig,
                                      description="Calibration config.")

    quant_config: QuantConfig = Field(default_factory=QuantConfig,
                                      description="Quantization config.")

    embedding_parallel_mode: Literal[
        'NONE', 'SHARDING_ALONG_VOCAB', 'SHARDING_ALONG_HIDDEN'] = Field(
            default='SHARDING_ALONG_VOCAB',
            description="The embedding parallel mode.")

    fast_build: bool = Field(default=False, description="Enable fast build.")

    # BuildConfig is introduced to give users a familiar interface to configure the model building.
    build_config: Optional[BuildConfig] = Field(default=None,
                                                description="Build config.")

    # Prompt adapter arguments
    enable_prompt_adapter: bool = Field(default=False,
                                        description="Enable prompt adapter.")

    max_prompt_adapter_token: int = Field(
        default=0, description="The maximum number of prompt adapter tokens.")

    batching_type: Optional[BatchingType] = Field(default=None,
                                                  description="Batching type.")

    normalize_log_probs: bool = Field(
        default=False, description="Normalize log probabilities.")

    # Private attributes
    # This is used to hold the options for convert_checkpoint
    _convert_checkpoint_options: Dict[str,
                                      Any] = PrivateAttr(default_factory=dict)

    @model_validator(mode="after")
    def init_build_config(self):
        """
        Creating a default BuildConfig if none is provided
        """
        build_config = getattr(self, "build_config", None)
        if build_config is None:
            kwargs = {}
            if self.max_batch_size:
                kwargs["max_batch_size"] = self.max_batch_size
            if self.max_num_tokens:
                kwargs["max_num_tokens"] = self.max_num_tokens
            if self.max_seq_len:
                kwargs["max_seq_len"] = self.max_seq_len
            if self.max_beam_width:
                kwargs["max_beam_width"] = self.max_beam_width
            if self.max_input_len:
                kwargs["max_input_len"] = self.max_input_len
            self.build_config = BuildConfig(**kwargs)
        return self

    @model_validator(mode="after")
    def validate_build_config_remaining(self):
        is_trt_llm_args = isinstance(self, TrtLlmArgs)

        # TODO: remove the checker when manage weights support all data types
        if is_trt_llm_args and self.fast_build and (self.quant_config.quant_algo
                                                    is QuantAlgo.FP8):
            self.build_config.plugin_config.manage_weights = True

        if self.parallel_config.world_size == 1 and self.build_config:
            self.build_config.plugin_config.nccl_plugin = None

        if self.enable_lora and self.backend != 'pytorch':
            self.build_config.plugin_config.lora_plugin = 'auto'
            if self.lora_config is not None:
                self.build_config.lora_config.max_lora_rank = self.lora_config.max_lora_rank

        if hasattr(self,
                   'enable_prompt_adapter') and self.enable_prompt_adapter:
            self.build_config.max_prompt_embedding_table_size = self.max_prompt_adapter_token * self.build_config.max_batch_size

        return self

    @model_validator(mode="after")
    def validate_speculative_config(self):
        if self.speculative_config:
            if not self.speculative_config.supports_backend(self.backend):
                raise ValueError(
                    f"Speculation type {self.speculative_config.decoding_type} does not "
                    f"support backend {self.backend}")

            # Below, we only need to set speculative_decoding_mode/decoding_config for speculation
            # on the TRT backend.
            if isinstance(self.speculative_config, LookaheadDecodingConfig):
                max_draft_len = self.speculative_config.calculate_speculative_resource(
                )[2]
                assert max_draft_len > 0
                self.build_config.speculative_decoding_mode = SpeculativeDecodingMode.LOOKAHEAD_DECODING
                self.build_config.max_draft_len = max(
                    self.build_config.max_draft_len, max_draft_len)
                self.decoding_config = DecodingConfig(
                    decoding_mode=DecodingMode.Lookahead(),
                    lookahead_decoding_config=PybindMirror.maybe_to_pybind(
                        self.speculative_config))

            elif isinstance(self.speculative_config, MedusaDecodingConfig):
                assert self.speculative_config.max_draft_len > 0
                self.build_config.speculative_decoding_mode = SpeculativeDecodingMode.MEDUSA
                self.build_config.max_draft_len = self.speculative_config.max_draft_len
                self.decoding_config = DecodingConfig(
                    decoding_mode=DecodingMode.Medusa(),
                    medusa_choices=self.speculative_config.medusa_choices)

            elif isinstance(self.speculative_config, Eagle3DecodingConfig):
                raise ValueError(
                    "speculative_config.decoding_type 'Eagle3' is only supported on the PyTorch backend. "
                    "Use decoding_type 'Eagle' for the TensorRT backend.")

            elif isinstance(self.speculative_config, EagleDecodingConfig):
                assert self.speculative_config.max_draft_len > 0
                assert self.speculative_config.speculative_model is not None, "EAGLE3 draft model must be specified."
                self.build_config.max_draft_len = self.speculative_config.max_draft_len
                self.build_config.speculative_decoding_mode = SpeculativeDecodingMode.EAGLE
                eagle_config = _EagleConfig(
                    self.speculative_config.eagle_choices,
                    self.speculative_config.greedy_sampling,
                    self.speculative_config.posterior_threshold,
                    self.speculative_config.use_dynamic_tree,
                    self.speculative_config.dynamic_tree_max_topK)
                self.decoding_config = DecodingConfig(
                    decoding_mode=DecodingMode.Eagle(),
                    eagle_config=eagle_config)
            elif isinstance(self.speculative_config, PARDDecodingConfig):
                raise ValueError(
                    "speculative_config.decoding_type 'PARD' is only supported on the PyTorch backend."
                )
            else:
                raise ValueError(
                    f"Unrecognized speculative config type {type(self.speculative_config)}"
                )

        else:
            self.decoding_config = None

        return self

    def _load_config_from_engine(self, engine_dir: Path):
        engine_config = EngineConfig.from_json_file(engine_dir / "config.json")
        self._pretrained_config = engine_config.pretrained_config
        self.build_config = engine_config.build_config

        # load and check parallel_config
        mapping = self._pretrained_config.mapping
        if self.parallel_config.tp_size not in (1, mapping.tp_size):
            raise ValueError(
                f"tp_size {self.parallel_config.tp_size} is not consistent with the engine's tp_size {mapping.tp_size}"
            )
        if self.parallel_config.pp_size not in (1, mapping.pp_size):
            raise ValueError(
                f"pp_size {self.parallel_config.pp_size} is not consistent with the engine's pp_size {mapping.pp_size}"
            )
        if self.parallel_config.cp_size not in (1, mapping.cp_size):
            raise ValueError(
                f"cp_size {self.parallel_config.cp_size} is not consistent with the engine's cp_size {mapping.cp_size}"
            )
        self._parallel_config = _ParallelConfig(
            tp_size=mapping.tp_size,
            pp_size=mapping.pp_size,
            cp_size=mapping.cp_size,
            gpus_per_node=mapping.gpus_per_node,
            moe_cluster_size=mapping.moe_cluster_size,
            moe_tp_size=mapping.moe_tp_size,
            moe_ep_size=mapping.moe_ep_size)

    def _load_config_from_ckpt(self, ckpt_dir: Path):
        pretrained_config = PretrainedConfig.from_json_file(ckpt_dir /
                                                            "config.json")
        tp_size = pretrained_config.mapping.tp_size
        pp_size = pretrained_config.mapping.pp_size
        cp_size = pretrained_config.mapping.cp_size
        moe_cluster_size = pretrained_config.mapping.moe_cluster_size
        moe_tp_size = pretrained_config.mapping.moe_tp_size
        moe_ep_size = pretrained_config.mapping.moe_ep_size
        gpus_per_node = pretrained_config.mapping.gpus_per_node
        # load parallel_config
        if self.parallel_config.tp_size != 1 and self.parallel_config.tp_size != tp_size:
            raise ValueError(
                f"tp_size {self.parallel_config.tp_size} is not consistent with the checkpoint's tp_size {tp_size}"
            )
        if self.parallel_config.pp_size != 1 and self.parallel_config.pp_size != pp_size:
            raise ValueError(
                f"pp_size {self.parallel_config.pp_size} is not consistent with the checkpoint's pp_size {pp_size}"
            )
        if self.parallel_config.cp_size != 1 and self.parallel_config.cp_size != cp_size:
            raise ValueError(
                f"cp_size {self.parallel_config.cp_size} is not consistent with the checkpoint's cp_size {cp_size}"
            )
        self._parallel_config = _ParallelConfig(
            tp_size=tp_size,
            pp_size=pp_size,
            cp_size=cp_size,
            gpus_per_node=gpus_per_node,
            moe_cluster_size=moe_cluster_size,
            moe_tp_size=moe_tp_size,
            moe_ep_size=moe_ep_size)

    @model_validator(mode="after")
    def validate_model_format_misc(self):
        '''
        Load the model format, and do the following:

        1. Load the build_config if got an engine.
        2. Load the parallel_config if got a checkpoint.
        '''
        model_obj = _ModelWrapper(self.model)

        if model_obj.is_local_model and self.backend not in [
                'pytorch', '_autodeploy'
        ]:
            # Load parallel_config from the engine.
            model_format = get_model_format(
                self.model, trust_remote_code=self.trust_remote_code)

            if model_format is _ModelFormatKind.TLLM_ENGINE:
                if self.build_config is not None:
                    logger.warning(
                        "The build_config is ignored for model format of TLLM_ENGINE."
                    )
                self._load_config_from_engine(model_obj.model_dir)
                runtime_defaults = self._pretrained_config.runtime_defaults
                if runtime_defaults:
                    self.kv_cache_config.fill_empty_fields_from_runtime_defaults(
                        runtime_defaults)

            # Load parallel_config from the checkpoint.
            elif model_format is _ModelFormatKind.TLLM_CKPT:
                # We need to create a temporary instance to call _load_config_from_ckpt
                self._load_config_from_ckpt(model_obj.model_dir)
        else:
            model_format = _ModelFormatKind.HF

        # Store the model format in the values
        self._model_format = model_format
        return self

    @model_validator(mode="after")
    def validate_build_config_with_runtime_params(self):
        """Sync runtime parameters with build_config limits.

        This validator runs AFTER validate_model_format_misc so that when
        loading from an engine, we have the real build_config loaded.
        """
        if self.build_config is None:
            raise ValueError("build_config is not initialized")

        # These can be lower than build_config limits
        for field in ("max_batch_size", "max_num_tokens"):
            runtime_val = getattr(self, field)
            build_val = getattr(self.build_config, field)
            if runtime_val is not None and runtime_val > build_val:
                logger.warning(
                    f"{field} [{runtime_val}] clamped to build_config.{field} [{build_val}]"
                )
                setattr(self, field, build_val)

        # These must match build_config exactly
        for field in ("max_seq_len", "max_beam_width", "max_input_len"):
            runtime_val = getattr(self, field)
            build_val = getattr(self.build_config, field)
            if runtime_val is not None and runtime_val != build_val:
                logger.warning(
                    f"{field} [{runtime_val}] overridden by build_config.{field} [{build_val}]"
                )
                setattr(self, field, build_val)

        return self

    @model_validator(mode="after")
    def setup_embedding_parallel_mode(self):
        if self.embedding_parallel_mode == 'NONE':
            self._convert_checkpoint_options['use_parallel_embedding'] = False
        elif self.embedding_parallel_mode == 'SHARDING_ALONG_VOCAB':
            self._convert_checkpoint_options['use_parallel_embedding'] = True
            self._convert_checkpoint_options['embedding_sharding_dim'] = 0
        elif self.embedding_parallel_mode == 'SHARDING_ALONG_HIDDEN':
            self._convert_checkpoint_options['use_parallel_embedding'] = True
            self._convert_checkpoint_options['embedding_sharding_dim'] = 1
        # No else clause needed since validation already happened
        return self

    @model_validator(mode="after")
    def validate_enable_build_cache(self):
        if not self.enable_build_cache:
            return self
        self.enable_build_cache = BuildCacheConfig() if isinstance(
            self.enable_build_cache, bool) else self.enable_build_cache
        return self

    @model_validator(mode="after")
    def validate_kv_cache_dtype(self):
        assert self.kv_cache_config.dtype == "auto", "KvCacheConfig.dtype is not supported by the TensorRT backend."
        return self


class LoadFormat(Enum):
    AUTO = 0
    # Initialize all weights randomly.
    DUMMY = 1
    # Only load the multimodal(vision) encoder weights
    VISION_ONLY = 2


class SamplerType(StrEnum):
    """Enum for sampler type options."""
    TRTLLMSampler = "TRTLLMSampler"
    TorchSampler = "TorchSampler"
    auto = "auto"


class TorchCompileConfig(StrictBaseModel):
    """
    Configuration for torch.compile.
    """
    enable_fullgraph: bool = Field(
        default=True,
        description="Enable full graph compilation in torch.compile.")

    enable_inductor: bool = Field(
        default=False, description="Enable inductor backend in torch.compile.")

    enable_piecewise_cuda_graph: bool = Field(
        default=False,
        description="Enable piecewise CUDA graph in torch.compile.")

    capture_num_tokens: Optional[List[PositiveInt]] = Field(
        default=None,
        description=
        "List of num of tokens to capture the piecewise CUDA graph for. If not provided, the number of tokens will be the same as cuda_graph_config.batch_sizes."
    )

    @field_validator('capture_num_tokens')
    @classmethod
    def validate_capture_num_tokens(cls, v):
        if v is None:
            return v
        return sorted(set(v), reverse=True)

    enable_userbuffers: bool = Field(
        default=True,
        description=
        "When torch compile is enabled, userbuffers is enabled by default.")

    max_num_streams: PositiveInt = Field(
        default=1,
        description=
        "The maximum number of CUDA streams to use for torch.compile.")

    @model_validator(mode='after')
    def set_default_capture_num_tokens(self) -> 'TorchCompileConfig':
        if self.enable_piecewise_cuda_graph and self.capture_num_tokens is None:
            self.capture_num_tokens = [2**i for i in range(8)
                                       ] + [i for i in range(256, 3073, 256)]
        return self


class TorchLlmArgs(BaseLlmArgs):
    # PyTorch backend specific configurations
    garbage_collection_gen0_threshold: int = Field(
        default=20000,
        description=
        "Threshold for Python garbage collection of generation 0 objects."
        "Lower values trigger more frequent garbage collection.",
        status="beta")

    cuda_graph_config: Optional[CudaGraphConfig] = Field(
        default_factory=CudaGraphConfig,
        description="CUDA graph config. If true, use CUDA graphs for decoding. \
        CUDA graphs are only created for the batch sizes in cuda_graph_config.batch_sizes, \
        and are enabled for batches that consist of decoding requests *only* \
        (the reason is that it's hard to capture a single graph with prefill requests \
        since the input shapes are a function of the sequence lengths).\
         Note that each CUDA graph can use up to 200 MB of extra memory.",
        status="beta")

    attention_dp_config: Optional[AttentionDpConfig] = Field(
        default=None,
        description="Optimized load-balancing for the DP Attention scheduler.",
        status="beta")

    disable_overlap_scheduler: bool = Field(
        default=False,
        description="Disable the overlap scheduler.",
        status="beta")

    moe_config: MoeConfig = Field(default_factory=MoeConfig,
                                  description="MoE config.",
                                  status="beta")

    nvfp4_gemm_config: Nvfp4GemmConfig = Field(
        default_factory=Nvfp4GemmConfig,
        description="NVFP4 GEMM backend config.",
        status="beta")

    attn_backend: str = Field(default='TRTLLM',
                              description="Attention backend to use.",
                              status="beta")

    sampler_type: Union[str, SamplerType] = Field(
        default=SamplerType.auto,
        description=
        "The type of sampler to use. Options are TRTLLMSampler, TorchSampler or auto. Defaults to auto, which will use TorchSampler unless BeamSearch is requested.",
        status="beta")

    sampler_force_async_worker: bool = Field(
        default=False,
        description="Force usage of the async worker in the sampler for D2H "
        "copies, even if confidential compute is not active. Normally, the "
        "async worker should only be used when confidential compute is active. "
        "This argument is provided to enable it for testing purposes, "
        "irrespective of confidential compute state.",
        status="prototype")

    enable_iter_perf_stats: bool = Field(
        default=False,
        description="Enable iteration performance statistics.",
        status="prototype")

    enable_iter_req_stats: bool = Field(
        default=False,
        description=
        "If true, enables per request stats per iteration. Must also set enable_iter_perf_stats to true to get request stats.",
        status="prototype")

    print_iter_log: bool = Field(default=False,
                                 description="Print iteration logs.",
                                 status="beta")

    batch_wait_timeout_ms: NonNegativeFloat = Field(
        default=0,
        description=
        "If greater than 0, the request queue might wait up to batch_wait_timeout_ms to receive max_batch_size requests, if fewer than max_batch_size requests are currently available. If 0, no waiting occurs.",
        status="prototype")

    batch_wait_timeout_iters: NonNegativeInt = Field(
        default=0,
        description=
        "Maximum number of iterations the scheduler will wait to accumulate new coming requests for improved GPU utilization efficiency. If greater than 0, the scheduler will delay batch processing to gather more requests up to the specified iteration limit. If 0, disables timeout-iters-based batching delays.",
        status="prototype")

    batch_wait_max_tokens_ratio: float = Field(
        default=0,
        ge=0,
        le=1,
        description=
        "Token accumulation threshold ratio for batch scheduling optimization. If greater than 0, the scheduler will accumulate requests locally until the total token count reaches batch_wait_max_tokens_ratio * max_num_tokens. This mechanism enhances GPU utilization efficiency by ensuring adequate batch sizes.If 0 disables token-based batching delays.",
        status="prototype")

    torch_compile_config: Optional[TorchCompileConfig] = Field(
        default=None, description="Torch compile config.", status="prototype")

    enable_autotuner: bool = Field(
        default=True,
        description=
        "Enable autotuner for all tunable ops. This flag is for debugging purposes only, and the performance may significantly degrade if set to false.",
        status="prototype")

    enable_layerwise_nvtx_marker: bool = Field(
        default=False,
        description="If true, enable layerwise nvtx marker.",
        status="beta")

    load_format: Union[str, LoadFormat] = Field(
        default=LoadFormat.AUTO,
        description=
        "How to load the model weights. By default, detect the weight type from the model checkpoint."
    )

    enable_min_latency: bool = Field(
        default=False,
        description=
        "If true, enable min-latency mode. Currently only used for Llama4.",
        status="beta",
    )

    # TODO: make this a per-request parameter
    stream_interval: PositiveInt = Field(
        default=1,
        description=
        "The iteration interval to create responses under the streaming mode. "
        "Set this to a larger value when the batch size is large, which helps reduce the streaming overhead.",
    )

    force_dynamic_quantization: bool = Field(
        default=False,
        description="If true, force dynamic quantization. Defaults to False.",
        status="prototype",
    )

    allreduce_strategy: Optional[Literal[
        'AUTO', 'NCCL', 'UB', 'MINLATENCY', 'ONESHOT', 'TWOSHOT',
        'LOWPRECISION', 'MNNVL',
        'NCCL_SYMMETRIC']] = Field(default='AUTO',
                                   description="Allreduce strategy to use.",
                                   status="beta")

    checkpoint_loader: Optional[object] = Field(
        default=None,
        description=
        "The checkpoint loader to use for this LLM instance. You may use a custom checkpoint loader by subclassing "
        "`BaseCheckpointLoader` and providing an instance of the subclass here to load weights from a custom "
        "checkpoint format.\n"
        "If neither checkpoint_format nor checkpoint_loader are provided, checkpoint_format will be set to HF "
        "and the default HfCheckpointLoader will be used.\n"
        "If checkpoint_format and checkpoint_loader are both provided, checkpoint_loader will be ignored.",
        json_schema_extra={
            "type":
            "Optional[tensorrt_llm._torch.models.checkpoints.BaseCheckpointLoader]"
        },
        status="prototype",
    )

    checkpoint_format: Optional[str] = Field(
        default=None,
        description=
        "The format of the provided checkpoint. You may use a custom checkpoint format by subclassing "
        "`BaseCheckpointLoader` and registering it with `register_checkpoint_loader`.\n"
        "If neither checkpoint_format nor checkpoint_loader are provided, checkpoint_format will be set to HF "
        "and the default HfCheckpointLoader will be used.\n"
        "If checkpoint_format and checkpoint_loader are both provided, checkpoint_loader will be ignored.",
        status="prototype",
    )

    kv_connector_config: Optional[KvCacheConnectorConfig] = Field(
        default=None,
        description="The config for KV cache connector.",
        status="prototype",
    )

    mm_encoder_only: bool = Field(
        default=False,
        description=
        "Only load/execute the vision encoder part of the full model. Defaults to False.",
        status="prototype",
    )

    ray_worker_extension_cls: Optional[str] = Field(
        default=None,
        description="The full worker extension class name including module path."
        "Allows users to extend the functions of the RayGPUWorker class.",
        status="prototype")

    ray_placement_config: Optional[RayPlacementConfig] = Field(
        default=None,
        description=
        "Placement config for RayGPUWorker. Only used with AsyncLLM and orchestrator_type='ray'.",
        exclude=True,
        status="prototype")

    enable_sleep: bool = Field(
        default=False,
        description=
        "Enable LLM sleep feature. Sleep feature requires extra setup that may slowdown model loading."
        "Only enable it if you intend to use this feature.",
        status="prototype")

    # fp8 cute dsl configs
    use_cute_dsl_blockscaling_mm: bool = Field(
        default=False,
        description="If true, use CuTe DSL fp8 blockscaling mm implementation.",
        status="prototype",
    )
    use_cute_dsl_blockscaling_bmm: bool = Field(
        default=False,
        description="If true, use CuTe DSL fp8 blockscaling bmm implementation.",
        status="prototype",
    )

    # PrivateVars
    _quant_config: Optional[QuantConfig] = PrivateAttr(default=None)

    disable_flashinfer_sampling: bool = Field(
        default=False,
        description=
        "Disable the use of FlashInfer.sampling. This option is likely to be removed in the future.",
        status="prototype",
    )

    max_stats_len: int = Field(
        default=1000,
        description="The max number of performance statistic entries.",
        status="prototype",
    )

    layer_wise_benchmarks_config: LayerwiseBenchmarksConfig = Field(
        default_factory=LayerwiseBenchmarksConfig,
        description="Configuration for layer-wise benchmarks calibration.",
        status="prototype")

    @property
    def quant_config(self) -> QuantConfig:
        if self._quant_config is None:
            self._quant_config = QuantConfig()
        return self._quant_config

    @quant_config.setter
    def quant_config(self, value: QuantConfig):
        self._quant_config = value

    # TODO: remove backend later
    backend: Literal["pytorch"] = Field(
        default="pytorch",
        description="The backend to use for this LLM instance.",
        exclude_json_schema=True,
        status="deprecated")

    @field_validator('load_format', mode='before')
    @classmethod
    def convert_load_format(cls, v):
        if isinstance(v, LoadFormat):
            return v
        load_format = v.upper()
        if load_format not in LoadFormat.__members__:
            raise ValueError(f"Invalid LoadFormat: {v}")
        return LoadFormat[load_format]

    # Extra resource managers to use in addition to the KV cache manager.
    # Each manager's prepare_resources method is called before the forward pass,
    # and update_resources() is called after the pass finishes. free_resources()
    # is called when a request finishes. The KV cache manager is guaranteed to
    # be invoked after all of these extra managers in all stages.
    _extra_resource_managers: Dict[str,
                                   object] = PrivateAttr(default_factory=dict, )

    @property
    def extra_resource_managers(self) -> Dict[str, object]:
        return self._extra_resource_managers

    @extra_resource_managers.setter
    def extra_resource_managers(self, value: Dict[str, object]) -> None:
        self._extra_resource_managers = value

    @model_validator(mode="after")
    def set_model_format(self):
        self._model_format = _ModelFormatKind.HF
        return self

    @model_validator(mode="after")
    def validate_speculative_config(self):
        if self.speculative_config:
            if not self.speculative_config.supports_backend(self.backend):
                raise ValueError(
                    f"Speculation type {self.speculative_config.decoding_type} does not "
                    f"support backend {self.backend}")

            # If user passed decoding_type: Eagle on pytorch, convert to Eagle3 with warning
            if type(self.speculative_config) is EagleDecodingConfig:
                logger.warning(
                    "speculative_config.decoding_type 'Eagle' is not supported on the PyTorch backend; only 'Eagle3' is supported. "
                    "'Eagle' is treated as 'Eagle3' for backward compatibility. "
                    "EAGLE (v1/v2) draft checkpoints are incompatible with Eagle3—use an Eagle3 draft model."
                )
                # Convert EagleDecodingConfig to Eagle3DecodingConfig
                eagle_data = self.speculative_config.model_dump(
                    exclude={"decoding_type"})
                self.speculative_config = Eagle3DecodingConfig(**eagle_data)

            if isinstance(self.speculative_config, PARDDecodingConfig):
                assert self.speculative_config.max_draft_len > 0, "PARD max_draft_len must be > 0"

            if isinstance(self.speculative_config,
                          SaveHiddenStatesDecodingConfig):
                logger.warning(
                    "SaveHiddenStatesDecodingConfig is active, setting max_batch_size to 1, disabling overlap scheduler, and setting cuda_graph_config to None"
                )
                self.max_batch_size = 1
                self.disable_overlap_scheduler = True
                self.cuda_graph_config = None
                self.speculative_config.max_draft_len = 1

        else:
            self.decoding_config = None

        return self

    @model_validator(mode="after")
    def validate_checkpoint_format(self):
        if self.checkpoint_format is not None and self.checkpoint_loader is not None:
            logger.warning(
                "checkpoint_format and checkpoint_loader are both provided, "
                "checkpoint_loader will be ignored.")
            self.checkpoint_loader = None

        if self.checkpoint_format is None and self.checkpoint_loader is None:
            logger.info(
                "neither checkpoint_format nor checkpoint_loader were provided, "
                "checkpoint_format will be set to HF.")
            self.checkpoint_format = "HF"

        return self

    @model_validator(mode="after")
    def validate_load_balancer(self) -> 'TorchLlmArgs':
        if isinstance(self.moe_config.load_balancer, str):
            if not os.path.exists(self.moe_config.load_balancer):
                raise FileNotFoundError(
                    f"MoE load balancer config file not found: {self.moe_config.load_balancer}"
                )
            try:
                with open(self.moe_config.load_balancer) as f:
                    moe_load_balancer_config = yaml.safe_load(f)
                self.moe_config.load_balancer = MoeLoadBalancerConfig(
                    **moe_load_balancer_config)
            except Exception as e:
                raise ValueError(
                    f"Failed to load MoE load balancer config file: {self.moe_config.load_balancer}"
                ) from e
        elif isinstance(self.moe_config.load_balancer, dict):
            try:
                self.moe_config.load_balancer = MoeLoadBalancerConfig(
                    **self.moe_config.load_balancer)
            except Exception as e:
                raise ValueError(
                    f"Failed to load MoE load balancer config: {self.moe_config.load_balancer}"
                ) from e
        return self

    @model_validator(mode='after')
    def sync_quant_config_with_kv_cache_config_dtype(self) -> 'TorchLlmArgs':
        if self.kv_cache_config is None:
            return self

        assert self.quant_config is not None
        if self.kv_cache_config.dtype == "auto":
            return self
        elif self.kv_cache_config.dtype == 'fp8':
            self.quant_config.kv_cache_quant_algo = QuantAlgo.FP8
        else:
            logger.warning(
                f"Cannot sync quant_config.kv_cache_quant_algo with kv_cache_config.dtype of {self.kv_cache_config.dtype}, "
                "please update the validator")

        return self

    @model_validator(mode='after')
    def validate_helix_tokens_per_block(self) -> 'TorchLlmArgs':
        """Validate that cp_config.tokens_per_block matches kv_cache_config.tokens_per_block when HELIX parallelism is active."""
        if self.context_parallel_size == 1 or self.cp_config is None:
            return self

        cp_type = self.cp_config.cp_type
        if cp_type == CpType.HELIX:
            cp_tokens_per_block = self.cp_config.tokens_per_block
            if cp_tokens_per_block is not None:
                kv_tokens_per_block = self.kv_cache_config.tokens_per_block
                assert cp_tokens_per_block == kv_tokens_per_block, (
                    f"When HELIX parallelism is active, cp_config.tokens_per_block ({cp_tokens_per_block}) "
                    f"must match kv_cache_config.tokens_per_block ({kv_tokens_per_block})."
                )

        return self

    def warn_on_unstable_feature_usage(self) -> 'TorchLlmArgs':
        """Warn on unstable feature usage."""
        set_fields = self.model_dump(exclude_unset=True).keys()

        for field_name in set_fields:
            field_info = self.model_fields.get(field_name)

            if not field_info or not field_info.json_schema_extra:
                continue

            status = field_info.json_schema_extra.get('status', None)

            if status in ('beta', 'prototype'):
                logger.warning(
                    f"The '{field_name}' knob is a '{status}' feature. "
                    "It is not recommended for production use and may change or be removed.",
                )

        return self

    @model_validator(mode='after')
    def validate_ray_worker_extension_cls(self) -> 'TorchLlmArgs':
        if self.ray_worker_extension_cls is not None and self.orchestrator_type != "ray":
            raise ValueError(
                f"ray_worker_extension_cls is only supported with orchestrator_type='ray'"
            )
        return self

    @model_validator(mode='after')
    def validate_ray_placement_config(self) -> 'TorchLlmArgs':
        if self.ray_placement_config is not None and self.orchestrator_type != "ray":
            raise ValueError(
                "ray_placement_config is only supported with orchestrator_type='ray'"
            )
        return self

    def get_executor_config(
        self,
        _hf_model_dir: Optional[Path] = None,
        tokenizer: Optional[TokenizerBase] = None,
    ) -> _ExecutorConfig:
        executor_config = super().get_executor_config(_hf_model_dir, tokenizer)
        executor_config.mm_encoder_only = self.mm_encoder_only
        return executor_config


def update_llm_args_with_extra_dict(
        llm_args: Dict,
        llm_args_dict: Dict,
        extra_llm_api_options: Optional[str] = None) -> Dict:

    # Deep merge kv_cache_config to prevent partial YAML kv_cache_config from replacing the complete kv_cache_config
    if 'kv_cache_config' in llm_args and 'kv_cache_config' in llm_args_dict:
        # Convert KvCacheConfig object to dict if necessary
        base_kv_config = llm_args['kv_cache_config']
        if isinstance(base_kv_config, KvCacheConfig):
            base_kv_config = base_kv_config.model_dump(exclude_unset=True)
        llm_args_dict['kv_cache_config'] = base_kv_config | llm_args_dict[
            'kv_cache_config']

    field_mapping = {
        "quant_config": QuantConfig,
        "calib_config": CalibConfig,
        "build_config": BuildConfig,
        "decoding_config": DecodingConfig,
        "enable_build_cache": BuildCacheConfig,
        "lora_config": LoraConfig,
        "moe_config": MoeConfig,
        "nvfp4_gemm_config": Nvfp4GemmConfig,
        "attention_dp_config": AttentionDpConfig,
        "kv_cache_config": KvCacheConfig,
    }
    for field_name, field_type in field_mapping.items():
        if field_name in llm_args_dict:
            llm_args_dict[field_name] = field_type(**llm_args_dict[field_name])
            extra_llm_str = f"because it's specified in {extra_llm_api_options}" if extra_llm_api_options else ""
            logger.warning(f"Overriding {field_name} {extra_llm_str}")

    llm_args = llm_args | llm_args_dict

    # build_config only works for TensorRT backend, it will be ignored in PyTorch backend
    if "build_config" in llm_args:
        # Ensure build_config is a BuildConfig object, not a dict
        if isinstance(llm_args["build_config"], dict):
            llm_args["build_config"] = BuildConfig(**llm_args["build_config"])

        for key in [
                "max_batch_size",
                "max_num_tokens",
                "max_beam_width",
                "max_seq_len",
        ]:
            if key in llm_args_dict:
                logger.info(
                    f"Overriding {key} from build_config to {llm_args_dict[key]}"
                )
                setattr(llm_args["build_config"], key, llm_args_dict[key])

    return llm_args


def update_llm_args_with_extra_options(llm_args: Dict,
                                       extra_llm_api_options: str) -> Dict:
    if extra_llm_api_options is not None:
        with open(extra_llm_api_options, 'r') as f:
            llm_args_dict = yaml.safe_load(f)
            llm_args = update_llm_args_with_extra_dict(llm_args, llm_args_dict,
                                                       extra_llm_api_options)
    return llm_args


def get_model_format(model_dir: str,
                     trust_remote_code: bool = False) -> _ModelFormatKind:
    ''' Get the format of the model.  '''
    if not (Path(model_dir) / 'config.json').exists():
        raise ValueError(
            f"Failed to infer model format because no config.json exists in {model_dir}"
        )

    with open(Path(model_dir) / 'config.json') as f:
        config = json.load(f)

    try:
        if 'pretrained_config' in config and 'build_config' in config:
            model_format = _ModelFormatKind.TLLM_ENGINE
            EngineConfig.from_json_file(Path(model_dir) / 'config.json')
        elif 'architecture' in config and 'dtype' in config:
            model_format = _ModelFormatKind.TLLM_CKPT
            PretrainedConfig.from_checkpoint(model_dir)
        else:
            model_format = _ModelFormatKind.HF
            AutoConfig.from_hugging_face(model_dir,
                                         trust_remote_code=trust_remote_code)
    except Exception as e:
        raise ValueError(
            f"Inferred model format {model_format}, but failed to load config.json: {e}"
        )
    else:
        return model_format


LlmArgs = TorchLlmArgs

TRT_LLMARGS_EXPLICIT_DOCSTRING = generate_api_docs_as_docstring(TrtLlmArgs,
                                                                indent=' ' * 4)
TORCH_LLMARGS_EXPLICIT_DOCSTRING = generate_api_docs_as_docstring(TorchLlmArgs,
                                                                  indent=' ' *
                                                                  4)
