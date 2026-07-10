# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import ast
import functools
import json
import math
import os
import types
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum, EnumMeta
from pathlib import Path
from typing import (TYPE_CHECKING, Annotated, Any, ClassVar, Dict, List,
                    Literal, Optional, Set, Tuple, Type, TypeAlias, TypeVar,
                    Union, get_args, get_origin)

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

from tensorrt_llm.bindings.internal.batch_manager import LinearCacheType
from tensorrt_llm.lora_helper import (LoraConfig,
                                      get_default_trtllm_modules_to_hf_modules)

from .._utils import (_str_to_torch_dtype_dict, is_sm_100f, mpi_rank,
                      prefer_pinned)

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
from ..bindings.internal.algorithms import AgentTreeConfig as _AgentTreeConfig  # isort: skip
# isort: on

# yapf: enable
from ..builder import BuildConfig, EngineConfig
from ..logger import logger
from ..mapping import CpType, Mapping
from ..models.automodel import AutoConfig
from ..models.modeling_utils import (PretrainedConfig, QuantAlgo, QuantConfig,
                                     SpeculativeDecodingMode)
from ..sampling_params import BatchedLogitsProcessor
from ..usage.config import UsageContext  # noqa: F401
from ..usage.config import TelemetryConfig, TelemetryField
from .build_cache import BuildCacheConfig
from .tokenizer import TokenizerBase, tokenizer_factory
from .utils import (StrictBaseModel, generate_api_docs_as_docstring,
                    get_type_repr)

TypeBaseModel = TypeVar("T", bound=BaseModel)
_TRTLLM_JSON_SCHEMA_EXTRA_ATTR = "_trtllm_json_schema_extra"

if TYPE_CHECKING:
    from tensorrt_llm._torch.virtual_memory import \
        RestoreMode as _VirtualMemoryRestoreMode
else:
    _VirtualMemoryRestoreMode = Enum


def Field(default: Any = ...,
          *,
          status: Optional[Literal["prototype", "beta", "deprecated"]] = None,
          telemetry: Optional[Union[bool, Dict[str, Any],
                                    TelemetryField]] = None,
          **kwargs: Any) -> Any:
    """Custom Field wrapper that adds status and telemetry metadata.

    Args:
        default: The default value for the field
        status: Optional status indicator that gets added to json_schema_extra.
            - None: Stable.
            - "beta": Recommended for use per the latest documentation.
            - "prototype": Not yet stable and subject to breaking changes; intended for experimentation only.
        telemetry: Optional field-local telemetry override for LLM API config
            capture. Type-safe fields (categorical/numeric) auto-enroll; pass
            telemetry=TelemetryField.categorical(...) to opt a free-form str/Any field
            in via an allowlist, or telemetry=False to opt a type-safe field out.
        **kwargs: All other arguments passed to the original Pydantic Field

    Returns:
        A Pydantic FieldInfo object with extra metadata added to
        json_schema_extra if provided.
    """
    telemetry_explicit_exclude = telemetry is False
    telemetry_requested = telemetry is not None and not telemetry_explicit_exclude

    if status is not None or telemetry_requested or telemetry_explicit_exclude:
        trtllm_schema_extra: dict[str, Any] = {}
        json_schema_extra = kwargs.get('json_schema_extra', {})
        if status is not None:
            trtllm_schema_extra['status'] = status
        if telemetry_explicit_exclude:
            # Honored opt-out sentinel: excludes a type-safe-but-sensitive field
            # from capture. Consumed by build_capture_manifest's selection rule.
            trtllm_schema_extra['telemetry'] = {"exclude": True}
        elif telemetry_requested:
            if isinstance(telemetry, TelemetryField):
                telemetry_metadata = telemetry.as_json_schema_extra()
            elif telemetry is True:
                telemetry_metadata = {"kind": "value"}
            elif isinstance(telemetry, dict):
                telemetry_metadata = dict(telemetry)
            else:
                raise TypeError(
                    "telemetry must be bool, dict, or TelemetryField")
            trtllm_schema_extra['telemetry'] = telemetry_metadata
        if isinstance(json_schema_extra, dict):
            json_schema_extra = {**json_schema_extra, **trtllm_schema_extra}
        elif callable(json_schema_extra):
            original_json_schema_extra = json_schema_extra

            def merged_json_schema_extra(schema: dict[str, Any]) -> None:
                original_extra = original_json_schema_extra(schema)
                if isinstance(original_extra, dict):
                    schema.update(original_extra)
                schema.update(trtllm_schema_extra)

            setattr(merged_json_schema_extra, _TRTLLM_JSON_SCHEMA_EXTRA_ATTR,
                    trtllm_schema_extra)
            json_schema_extra = merged_json_schema_extra
        else:
            json_schema_extra = trtllm_schema_extra
        kwargs['json_schema_extra'] = json_schema_extra

    return PydanticField(default, **kwargs)


def _get_trtllm_json_schema_extra(field_info: Any) -> dict[str, Any]:
    json_schema_extra = getattr(field_info, "json_schema_extra", None)
    if callable(json_schema_extra):
        json_schema_extra = getattr(json_schema_extra,
                                    _TRTLLM_JSON_SCHEMA_EXTRA_ATTR, None)
    if isinstance(json_schema_extra, dict):
        return json_schema_extra
    return {}


class BaseCudaGraphConfig(StrictBaseModel):
    """Common configuration for CUDA graphs."""
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
    def validate_base_cuda_graph_config(self) -> 'BaseCudaGraphConfig':
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
            generated_sizes = self._generate_cuda_graph_batch_sizes(
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
            # Start with [1, 2, 4, 8, 16, 24, ..., 128] (multiples of 8)
            batch_sizes = [1, 2, 4] + [i * 8 for i in range(1, 17)]
            # Sliding 64: extend by increments of 64 up to max_batch_size
            while batch_sizes[-1] + 64 <= max_batch_size:
                batch_sizes.append(batch_sizes[-1] + 64)
        else:
            batch_sizes = list(range(1, 32)) + [32, 64, 128]
            # Add powers of 2 up to max_batch_size
            batch_sizes += [
                2**i for i in range(8, math.ceil(math.log(max_batch_size, 2)))
            ]

        # Filter and sort batch sizes for both branches
        batch_sizes = sorted(
            [size for size in batch_sizes if size <= max_batch_size])

        # Add max_batch_size if not already included
        if not batch_sizes or max_batch_size != batch_sizes[-1]:
            batch_sizes.append(max_batch_size)

        return batch_sizes


class DecodeCudaGraphConfig(BaseCudaGraphConfig):
    """CUDA graph configuration for decode requests."""

    mode: Literal["decode"] = Field(
        default="decode", description="CUDA graph configuration mode.")

    @staticmethod
    def _merge_schedule_keys(batch_sizes: List[int],
                             schedule: dict[int, int]) -> List[int]:
        """Merge draft_len_schedule keys into batch_sizes so that each schedule threshold has a corresponding CUDA graph.

        e.g. draft_len_schedule={100:4, 200:3, 300:2} adds 100, 200, 300
        into batch_sizes.

        Args:
            batch_sizes: Sorted list of existing CUDA graph batch sizes.
            schedule: draft_len_schedule mapping batch-size thresholds to
                draft lengths.

        Returns:
            Sorted, deduplicated list of batch sizes.
        """
        max_bs = batch_sizes[-1]
        extra = sorted(bs for bs in schedule if bs <= max_bs)
        if not extra:
            return batch_sizes

        merged = []
        i, j = 0, 0
        while i < len(batch_sizes) and j < len(extra):
            if batch_sizes[i] < extra[j]:
                merged.append(batch_sizes[i])
                i += 1
            elif batch_sizes[i] > extra[j]:
                merged.append(extra[j])
                j += 1
            else:
                merged.append(batch_sizes[i])
                i += 1
                j += 1
        merged.extend(batch_sizes[i:])
        merged.extend(extra[j:])
        return merged


class EncodeCudaGraphConfig(BaseCudaGraphConfig):
    """CUDA graph configuration for encode-only requests."""

    mode: Literal["encode"] = Field(
        default="encode", description="CUDA graph configuration mode.")

    num_tokens: Optional[List[PositiveInt]] = Field(
        default=None,
        min_length=1,
        description=
        "List of total token counts (sum of all per-request sequence lengths "
        "in a batch) to create encoder CUDA graphs for.")

    max_num_token: NonNegativeInt = Field(
        default=0,
        description="Maximum total number of tokens for encoder CUDA graphs. If "
        "`num_tokens` is provided, must equal max(num_tokens); otherwise "
        "`num_tokens` is generated from this value.")

    seq_lens: Optional[List[PositiveInt]] = Field(
        default=None,
        min_length=1,
        description=
        "List of max per-request sequence lengths to create encoder CUDA "
        "graphs for.")

    max_seq_len: NonNegativeInt = Field(
        default=0,
        description=
        "Maximum per-request sequence length for encoder CUDA graphs. If "
        "`seq_lens` is provided, must equal max(seq_lens); otherwise "
        "`seq_lens` is generated from this value.")

    @model_validator(mode='after')
    def validate_encoder_cuda_graph_config(self) -> 'EncodeCudaGraphConfig':
        # Encoder fields — only generate defaults when the user opted in by
        # setting at least one of num_tokens / max_num_token. Leaving both at
        # the defaults keeps encoder CUDA graphs disabled. Same for seq_lens / max_seq_len.
        if self.num_tokens:
            self.num_tokens = sorted(self.num_tokens)
            derived_max_nt = max(self.num_tokens)
            if self.max_num_token == 0:
                self.max_num_token = derived_max_nt
            elif self.max_num_token != derived_max_nt:
                raise ValueError(
                    "CudaGraphConfig.max_num_token is incompatible with "
                    "CudaGraphConfig.num_tokens. When both are provided, "
                    "max_num_token must equal max(num_tokens).\n"
                    f"CudaGraphConfig.num_tokens: {self.num_tokens}, "
                    f"max(num_tokens): {derived_max_nt}, "
                    f"CudaGraphConfig.max_num_token: {self.max_num_token}")
        elif self.max_num_token > 0:
            self.num_tokens = self._generate_cuda_graph_num_tokens(
                self.max_num_token, self.enable_padding)

        if self.seq_lens:
            self.seq_lens = sorted(self.seq_lens)
            derived_max_sl = max(self.seq_lens)
            if self.max_seq_len == 0:
                self.max_seq_len = derived_max_sl
            elif self.max_seq_len != derived_max_sl:
                raise ValueError(
                    "CudaGraphConfig.max_seq_len is incompatible with "
                    "CudaGraphConfig.seq_lens. When both are provided, "
                    "max_seq_len must equal max(seq_lens).\n"
                    f"CudaGraphConfig.seq_lens: {self.seq_lens}, "
                    f"max(seq_lens): {derived_max_sl}, "
                    f"CudaGraphConfig.max_seq_len: {self.max_seq_len}")
        elif self.max_seq_len > 0:
            self.seq_lens = self._generate_cuda_graph_seq_lens(
                self.max_seq_len, self.enable_padding)

        return self

    @staticmethod
    def _generate_cuda_graph_num_tokens(max_num_token: int,
                                        enable_padding: bool) -> List[int]:
        """Generate a list of total token counts for encoder CUDA graphs.

        Args:
            max_num_token: Maximum total tokens to generate up to.
            enable_padding: Whether padding is enabled, which affects the
                size distribution.

        Returns:
            List of total token counts to create encoder CUDA graphs for.
        """
        if enable_padding:
            # Coarser: aligned with piecewise CUDA graph capture sizes.
            sizes = [2**i for i in range(8)]  # 1, 2, 4 .. 128
            sizes += list(range(256, 3073, 256))  # 256, 512, ..., 3072
        else:
            # Finer: progressively coarser steps.
            sizes = [2**i for i in range(5)]  # 1, 2, 4 .. 16
            sizes += list(range(16, 65, 16))  # 16, 32, ..., 64
            sizes += list(range(96, 257, 32))  # 96, 128, ..., 256
            sizes += list(range(384, 1025, 128))  # 384, 512, ..., 1024
            sizes += list(range(1280, 3073, 256))  # 1280, 1536, ..., 3072

        # Beyond the base range: powers of 2 up to max_num_token.
        p = 4096
        while p < max_num_token:
            sizes.append(p)
            p *= 2

        sizes = sorted(set(s for s in sizes if s <= max_num_token))
        if not sizes or sizes[-1] != max_num_token:
            sizes.append(max_num_token)

        return sizes

    @staticmethod
    def _generate_cuda_graph_seq_lens(max_seq_len: int,
                                      enable_padding: bool) -> List[int]:
        """
        Generate a list of max per-request sequence lengths for encoder CUDA graphs.

        Args:
            max_seq_len: Maximum per-request sequence length to generate up to.
            enable_padding: Whether padding is enabled, which affects the
                size distribution.

        Returns:
            List of max sequence lengths to create encoder CUDA graphs for.
        """
        if enable_padding:
            # Coarser buckets for rounding up.
            sizes = [2**i for i in range(7)]  # 1, 2, 4 .. 64
            sizes += list(range(128, 1025, 128))  # 128, 256, ..., 1024
        else:
            # Finer: progressive steps for exact matching.
            sizes = list(range(1, 9))  # 1 .. 8
            sizes += list(range(16, 65, 8))  # 16, 24, ..., 64
            sizes += list(range(96, 257, 32))  # 96, 128, ..., 256
            sizes += list(range(320, 513, 64))  # 320, 384, 448, 512
            sizes += list(range(640, 1025, 128))  # 640, 768, 896, 1024

        # Beyond the base range: powers of 2 up to max_seq_len.
        p = 2048
        while p < max_seq_len:
            sizes.append(p)
            p *= 2

        sizes = sorted(set[Any | int](s for s in sizes if s <= max_seq_len))
        if not sizes or sizes[-1] != max_seq_len:
            sizes.append(max_seq_len)

        return sizes


# For CudaGraphConfig's backward compatibility
CudaGraphConfig = DecodeCudaGraphConfig

CudaGraphConfigType: TypeAlias = Annotated[
    Union[DecodeCudaGraphConfig, EncodeCudaGraphConfig],
    Field(discriminator="mode"),
]


class MultimodalEncoderCudaGraphConfig(StrictBaseModel):
    """CUDA graph capture for multimodal vision / audio encoders.

    One graph is captured per `buckets` entry; replay falls back to eager when the live workload's
    padded shape does not match any bucket.

    NOTE: enabling this will lead to higher GPU memory usage. It is usually more beneficial for
    `buckets` corresponding to lesser compute.
    """

    buckets: list[tuple[PositiveInt, PositiveInt]] = Field(
        min_length=1,
        description=
        ("Explicit encoder graph buckets as `(total_tokens, num_contexts)` pairs. Each pair "
         "captures one graph sized for up to `total_tokens` real encoder tokens across exactly "
         "`num_contexts` real contexts."),
        status="prototype",
    )

    enable_padding: bool = Field(
        default=True,
        description=
        ("Capture a padded variant of each bucket so partial buckets can replay. Padding works "
         "only when the request has exactly the bucket's `num_contexts` real contexts and no "
         "more than the bucket's `total_tokens` real encoder tokens. The runner appends one "
         "dummy padding context and one reserved padding token internally, so bucket values "
         "should describe only the real workload."),
        status="prototype",
    )

    warmup_steps: PositiveInt = Field(
        default=2,
        description="Warmup iterations to run per bucket before capturing.",
        status="prototype",
    )

    enable_replay_stats: bool = Field(
        default=False,
        description=
        ("Log diagnostic details for encoder CUDA graph bucket hits and misses. "
         "When enabled, each request's bucket decision is logged at INFO."),
        status="prototype",
    )

    @model_validator(mode='after')
    def _normalize(self) -> 'MultimodalEncoderCudaGraphConfig':
        for total_tokens, num_contexts in self.buckets:
            if total_tokens < num_contexts:
                raise ValueError("`buckets` entries must have total_tokens >= "
                                 "num_contexts.")
        self.buckets = sorted(set(self.buckets))
        return self


class MultimodalConfig(StrictBaseModel):
    """Multimodal model configuration."""

    encoder_cuda_graph: dict[str, MultimodalEncoderCudaGraphConfig] | None = Field(
        default=None,
        description=
        ("CUDA graph capture for multimodal encoders, keyed by modality name. "
         "This config is not applied automatically - each model must read "
         "`model_config.multimodal_config.encoder_cuda_graph` and implement capture + replay "
         "via `MultimodalEncoderCudaGraphRunner` (see "
         "`tensorrt_llm/_torch/models/multimodal_encoder_graph.py`)."),
        status="prototype",
    )

    encoder_side_stream_max_ahead: NonNegativeInt = Field(
        default=0,
        description=
        ("Maximum number of pending multimodal requests whose encoder work can be prefetched "
         "on a side CUDA stream ahead of admission. 0 disables side-stream prefetch. "
         "Incompatible with encoder_cuda_graph because graph replay uses static buffers."
         ),
        status="prototype",
    )

    video_pruning_rate: Optional[float] = Field(
        default=None,
        ge=0.0,
        lt=1.0,
        description=
        ("Pruning rate for video frames in multimodal models for Efficient Video Sampling (EVS). "
         "NOTE: this is currently only implemented in nemotron multimodal models. "
         "None (default) disables EVS, values in [0, 1) enable pruning."),
        status="prototype",
    )

    @model_validator(mode='after')
    def validate_side_stream_cuda_graph_exclusive(self) -> 'MultimodalConfig':
        if (self.encoder_cuda_graph is not None
                and self.encoder_side_stream_max_ahead > 0):
            raise ValueError(
                "multimodal_config.encoder_cuda_graph and "
                "multimodal_config.encoder_side_stream_max_ahead > 0 are "
                "mutually exclusive. Disable side-stream MM prefetch or "
                "disable MM encoder CUDA graphs.")
        return self


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
    """Configuration for sparse attention."""
    algorithm: str

    def supports_backend(self, backend: str) -> bool:
        """Override if the sparse attention algorithm does not support
        a subset of the possible backends.
        """
        return True

    def to_sparse_params(self, **kwargs):
        """Lower user-facing config into SparseParams."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement SparseParams lowering.")

    def to_sparse_metadata_params(self, **kwargs):
        """Lower user-facing config into SparseMetadataParams."""
        return None


class SeqLenAwareSparseAttentionConfig(BaseSparseAttentionConfig):
    """Sparse attention config with sequence-length dependent behavior."""

    seq_len_threshold: Optional[int] = Field(
        default=None,
        description=
        "The sequence length threshold for separating short and long sequences."
    )

    def get_indices_block_size(self) -> int:
        return 1

    def needs_separate_short_long_cuda_graphs(self) -> bool:
        """Whether to capture separate CUDA graphs for short and long sequences."""
        return False


class MiniMaxM3SparseAttentionConfig(BaseSparseAttentionConfig):
    """Configuration for MiniMax-M3 block-sparse attention.

    Drives the two-step sparse attention used by MiniMax-M3 layers 3..N:

      1. An index attention branch projects a per-head Q vector and a
         **single replicated** K vector, scores main K/V cache blocks,
         and selects the top-``topk`` blocks per ``(num_kv_heads, q_token)``
         pair (with ``init_blocks`` forced at the head and ``local_blocks``
         forced at the tail).
      2. A sparse GQA attention runs only over the selected blocks.

    The selected backend at runtime uses
    :class:`tensorrt_llm._torch.attention_backend.sparse.minimax_m3.MiniMaxM3SparseAttention`
    on top of a :class:`MiniMaxM3KVCacheManagerV2` that allocates a
    paged side index-K cache (``[num_slots, 1, sparse_index_dim]``)
    parallel to the main K/V cache. The M3 checkpoint sets
    ``disable_index_value=True`` on every sparse layer so no index V
    cache is allocated for the bring-up.
    """

    algorithm: Literal["minimax_m3"] = "minimax_m3"
    sparse_num_index_heads: int = Field(
        default=4,
        description="Number of index-attention heads (per TP rank's view).",
    )
    sparse_index_dim: int = Field(
        default=128,
        description="Per-head index Q/K dimension.",
    )
    sparse_block_size: int = Field(
        default=128,
        description="Block size used by per-block scoring + top-k selection.",
    )
    sparse_topk_blocks: int = Field(
        default=16,
        description="Number of top-k blocks per (kv_head, q_token).")
    sparse_init_blocks: int = Field(
        default=0,
        description=
        "Number of leading blocks forced into the top-k regardless of score.",
    )
    sparse_local_blocks: int = Field(
        default=1,
        description=
        "Number of trailing blocks forced into the top-k regardless of score.",
    )
    sparse_score_type: Literal["max"] = Field(
        default="max",
        description="Per-block score reduction; the M3 checkpoint sets 'max'.",
    )
    sparse_disable_index_value: bool = Field(
        default=True,
        description="If True, skip the index V branch (M3 checkpoint default).",
    )

    def supports_backend(self, backend: str) -> bool:
        return backend == "pytorch"

    def get_indices_block_size(self) -> int:
        return self.sparse_block_size

    def to_sparse_params(self, **kwargs):
        from tensorrt_llm._torch.attention_backend.sparse.minimax_m3.metadata import \
            MiniMaxM3SparseParams

        return MiniMaxM3SparseParams(
            num_index_heads=self.sparse_num_index_heads,
            sparse_index_dim=self.sparse_index_dim,
            block_size=self.sparse_block_size,
            topk=self.sparse_topk_blocks,
            init_blocks=self.sparse_init_blocks,
            local_blocks=self.sparse_local_blocks,
            score_type=self.sparse_score_type,
            disable_index_value=self.sparse_disable_index_value,
        )


class RocketSparseAttentionConfig(SeqLenAwareSparseAttentionConfig):
    """Configuration for RocketKV sparse attention."""
    algorithm: Literal["rocket"] = Field(default="rocket")
    window_size: Optional[int] = Field(
        default=32, description="The window size for RocketKV.")
    kernel_size: Optional[int] = Field(
        default=63, description="The kernel size for RocketKV.")
    topr: Optional[Union[int, float]] = Field(default=128, description="Top-r")
    topk: Optional[int] = Field(default=64, description="Top-k")
    prompt_budget: Optional[int] = Field(default=2048,
                                         description="Prompt budget")
    page_size: Optional[int] = Field(default=4, description="Page size")
    kt_cache_dtype: Optional[str] = Field(default='float8_e5m2',
                                          choices=['bfloat16', 'float8_e5m2'],
                                          description="KT cache dtype",
                                          telemetry=TelemetryField.categorical(
                                              'bfloat16', 'float8_e5m2'))

    def supports_backend(self, backend: str) -> bool:
        return backend == "pytorch"

    def get_indices_block_size(self) -> int:
        return self.page_size

    def to_sparse_params(self, **kwargs):
        from tensorrt_llm._torch.attention_backend.sparse.rocket import \
            RocketKVParams

        def _value(name: str, default):
            value = getattr(self, name)
            return default if value is None else value

        return RocketKVParams(
            window_size=_value("window_size", 32),
            kernel_size=_value("kernel_size", 63),
            topr=_value("topr", 128),
            topk=_value("topk", 64),
            prompt_budget=_value("prompt_budget", 2048),
            page_size=_value("page_size", 4),
            kt_cache_dtype=_value("kt_cache_dtype", "float8_e5m2"),
        )

    def to_sparse_metadata_params(self, **kwargs):
        from tensorrt_llm._torch.attention_backend.sparse.rocket import \
            RocketKVMetadataParams

        def _value(name: str, default):
            value = getattr(self, name)
            return default if value is None else value

        return RocketKVMetadataParams(
            prompt_budget=_value("prompt_budget", 2048),
            window_size=_value("window_size", 32),
            page_size=_value("page_size", 4),
            topk=_value("topk", 64),
        )


class DeepSeekSparseAttentionConfig(SeqLenAwareSparseAttentionConfig):
    """Configuration for DeepSeek Sparse Attention."""
    algorithm: Literal["dsa"] = Field(default="dsa")
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
    use_cute_dsl_topk: bool = Field(
        default=False,
        description=
        "Whether to use CuTE DSL top-k kernel instead of the CUDA C++ indexer_topk_decode."
    )
    use_cute_dsl_paged_mqa_logits: bool = Field(
        default=False,
        description=
        "Whether to use CuTE DSL paged MQA logits kernel on SM100 instead of C++ DeepGEMM."
    )
    q_split_threshold: int = Field(
        default=8192,
        description=
        "If number of packed tokens in prefill chunk exceeds this threshold, \
            q tokens will be evenly distributed across ranks for indexer computation. \
            If negative, q split will always be disabled.")
    indexer_rope_interleave: bool = Field(
        default=False,
        description="Whether to use interleaved RoPE layout for the indexer.")
    enable_heuristic_topk: bool = Field(
        default=False,
        description=
        "Whether to enable Guess-Verify-Refine (GVR) Top-K for the DSA decode "
        "indexer. GVR reuses previous-step Top-K indices as hints to reduce "
        "threshold search iterations. Currently supported for index_topk ∈ "
        "{512, 1024, 2048} on Blackwell (SM100+), with compress_ratio ∈ {1, 4} "
        "(DSv3.2 + DSv4 indexers). Falls back to the production insertion/"
        "radix Top-K path when prerequisites are not met.")
    indexer_k_dtype: Literal["fp8", "fp4"] = Field(
        default="fp8",
        description=
        "Data type used for the indexer K cache. `fp8` stores one FP8 E4M3 "
        "byte per element with a per-128 float32 scale; `fp4` packs two FP4 "
        "E2M1 codes per byte with a per-32 UE8M0 exponent, halving the "
        "per-token indexer K footprint (132 B to 68 B at index_head_dim=128). "
        "`fp4` requires Blackwell+ (SM>=100) at runtime and "
        "index_head_dim=128.",
    )

    @model_validator(mode="after")
    def _validate_indexer_k_dtype(self):
        """Reject fp4 on pre-Blackwell or non-128 index_head_dim.

        DeepGEMM's fp8_fp4_mqa_logits / fp8_fp4_paged_mqa_logits kernels
        require SM>=100, and invokeFusedCatFp4 hard-asserts head_dim==128.
        Surface explicitly requested invalid configs as Pydantic errors so
        they fail fast instead of with a cryptic kernel-launch failure. The
        DeepSeek-V4 default falls back to fp8 on pre-Blackwell GPUs.

        The SM check is skipped when CUDA is unavailable (config
        construction on CPU-only hosts or at doc-gen time), leaving the
        runtime kernel assertion as the final line of defense.
        """
        if self.indexer_k_dtype == "fp4":
            if self.index_head_dim is not None and self.index_head_dim != 128:
                raise ValueError(
                    f"indexer_k_dtype='fp4' requires index_head_dim=128, "
                    f"got {self.index_head_dim}. Set indexer_k_dtype='fp8' "
                    f"for non-128 indexer head dims.")
            if torch.cuda.is_available():
                from tensorrt_llm._utils import get_sm_version
                sm = get_sm_version()
                if sm < 100:
                    if 'indexer_k_dtype' not in self.model_fields_set:
                        logger.warning(
                            "DeepSeek-V4 defaults indexer_k_dtype to 'fp4', "
                            f"but the current device is SM{sm}; falling back "
                            "to 'fp8'.")
                        self.indexer_k_dtype = "fp8"
                    else:
                        raise ValueError(
                            f"indexer_k_dtype='fp4' requires SM>=100 "
                            f"(Blackwell); current device is SM{sm}. Set "
                            f"indexer_k_dtype='fp8' for non-Blackwell GPUs.")
        return self

    @model_validator(mode="after")
    def _warn_heuristic_topk_unsupported(self):
        """Warn when GVR Top-K cannot accelerate the configured index_topk.

        This warning does not reject the configuration.

        The C++ ``indexer_topk_decode`` dispatcher silently falls back to the
        radix Top-K path for unsupported K, so without this warning a user may
        believe GVR is active when it is not. ``index_topk`` may still be None
        here (it is filled from the checkpoint later), so only validate
        concrete values.
        """
        supported_topk = (512, 1024, 2048)
        if (self.enable_heuristic_topk and self.index_topk is not None
                and self.index_topk not in supported_topk):
            logger.warning(
                f"enable_heuristic_topk=True but index_topk={self.index_topk} "
                f"is not in the GVR-supported set {supported_topk}; the indexer "
                f"will silently fall back to the radix Top-K path. Set "
                f"index_topk to one of {supported_topk} to use GVR.")
        return self

    def supports_backend(self, backend: str) -> bool:
        return backend == "pytorch"

    def needs_separate_short_long_cuda_graphs(self) -> bool:
        """Whether to capture separate CUDA graphs for short and long sequences.
        Use seq_len_threshold to determine the threshold for separating short and long sequences.
        """
        self.seq_len_threshold = self.index_topk
        return self.skip_indexer_for_short_seqs

    @staticmethod
    def _is_full_indexer_layer(pretrained_config, layer_idx) -> bool:
        """Whether a DSA layer runs its own indexer ("full") or reuses the
        previous full layer's top-k ("shared") -- cross-layer indexer sharing.

        Resolved from the HF config: index_topk_pattern, else
        index_topk_freq/index_skip_topk_offset. The MTP/nextn layer
        (>= num_hidden_layers) is always full. Defaults to full (a dense
        per-layer indexer, e.g. DeepSeek-V3.2).
        """
        if pretrained_config is None or layer_idx is None:
            return True
        n = getattr(pretrained_config, "num_hidden_layers", None)
        if n is not None and layer_idx >= n:
            return True  # MTP/nextn layer
        pattern = getattr(pretrained_config, "index_topk_pattern", None)
        if pattern is not None:
            is_full = not (layer_idx < len(pattern)
                           and str(pattern[layer_idx]).upper() == "S")
        else:
            freq = max(getattr(pretrained_config, "index_topk_freq", 1) or 1, 1)
            offset = getattr(pretrained_config, "index_skip_topk_offset", 2)
            is_full = (max(layer_idx - offset + 1, 0) % freq) == 0
        if layer_idx == 0 and not is_full:
            logger.warning(
                "DSA layer 0 resolved to 'shared' but has no prior full layer's "
                "top-k to reuse; forcing it to 'full'. Check index_topk_pattern."
            )
            is_full = True
        return is_full

    def to_sparse_params(self, **kwargs):
        from tensorrt_llm._torch.attention_backend.sparse.dsa import DSAParams

        pretrained_config = kwargs.get("pretrained_config", None)

        def _value(name: str, default=None):
            value = getattr(self, name)
            if value is not None:
                return value
            if pretrained_config is not None:
                return getattr(pretrained_config, name, default)
            return default

        return DSAParams(
            index_n_heads=_value("index_n_heads"),
            index_head_dim=_value("index_head_dim"),
            index_topk=_value("index_topk"),
            indexer_max_chunk_size=self.indexer_max_chunk_size,
            skip_indexer_for_short_seqs=self.skip_indexer_for_short_seqs,
            use_cute_dsl_topk=self.use_cute_dsl_topk,
            use_cute_dsl_paged_mqa_logits=self.use_cute_dsl_paged_mqa_logits,
            q_split_threshold=self.q_split_threshold,
            indexer_rope_interleave=self.indexer_rope_interleave,
            enable_heuristic_topk=self.enable_heuristic_topk,
            indexer_k_dtype=self.indexer_k_dtype,
            is_full_indexer_layer=self._is_full_indexer_layer(
                pretrained_config, kwargs.get("layer_idx")),
        )

    def to_sparse_metadata_params(self, **kwargs):
        from tensorrt_llm._torch.attention_backend.sparse.dsa import \
            DSAMetadataParams

        pretrained_config = kwargs.get("pretrained_config", None)

        def _value(name: str, default=None):
            value = getattr(self, name)
            if value is not None:
                return value
            if pretrained_config is not None:
                return getattr(pretrained_config, name, default)
            return default

        return DSAMetadataParams(
            indexer_max_chunk_size=self.indexer_max_chunk_size or 32768,
            max_sparse_topk=_value("index_topk"),
            enable_indexer_skip=self.skip_indexer_for_short_seqs,
            enable_heuristic_topk=self.enable_heuristic_topk,
            use_cute_dsl_paged_mqa_logits=(self.use_cute_dsl_paged_mqa_logits),
            q_split_threshold=self.q_split_threshold,
        )


class DeepSeekV4SparseAttentionConfig(DeepSeekSparseAttentionConfig):
    """Configuration for DeepSeek-V4 Sparse Attention."""
    algorithm: Literal["deepseek_v4"] = "deepseek_v4"
    index_head_dim: Optional[int] = Field(
        default=128,
        description="The dimension of the DeepSeek-V4 indexer heads.")
    indexer_k_dtype: Literal["fp8", "fp4"] = Field(
        default="fp4",
        description=
        "Data type used for the indexer K cache. DeepSeek-V4 defaults to "
        "`fp4` to reduce the per-token indexer K footprint on Blackwell+ "
        "(SM>=100). Set to `fp8` for the legacy FP8 indexer K cache path.",
    )
    skip_indexer_for_short_seqs: bool = Field(
        default=False,
        description=
        "Whether to skip the MQA and Top-K in the indexer for short sequences.")

    compress_ratios: List[int] = Field(
        default_factory=lambda: [1, 1, 4, 128, 4, 128, 4],
        description="The compress ratios of each layer. DeepSeek-V4 uses 0 "
        "for uncompressed/SWA-only layers; the LLM API config normalizes "
        "0 to 1, while checkpoint-facing semantics remain unchanged.")
    window_size: int = Field(
        default=128,
        description="The sliding window size in tokens for SWA layers.")
    index_topk: Optional[int] = Field(default=512,
                                      description="The top-k for the indexer.")

    @field_validator("index_head_dim")
    @classmethod
    def validate_index_head_dim(cls, index_head_dim):
        if index_head_dim is None:
            raise ValueError(
                "index_head_dim is required for DeepSeek-V4 sparse attention.")
        return index_head_dim

    @field_validator("compress_ratios")
    @classmethod
    def normalize_compress_ratios(cls, compress_ratios):
        if not compress_ratios:
            raise ValueError("compress_ratios must not be empty.")
        if any(ratio < 0 for ratio in compress_ratios):
            raise ValueError("compress_ratios must be non-negative.")
        return [1 if ratio == 0 else ratio for ratio in compress_ratios]

    def supports_backend(self, backend: str) -> bool:
        return backend == "pytorch"

    def needs_separate_short_long_cuda_graphs(self) -> bool:
        # DeepSeek-V4 does not support short/long CUDA graph separation.
        return False

    def to_sparse_params(self, **kwargs):
        from tensorrt_llm._torch.attention_backend.sparse.deepseek_v4 import \
            DeepSeekV4Params

        pretrained_config = kwargs.get("pretrained_config", None)

        def _value(name: str, default=None):
            value = getattr(self, name)
            if value is not None:
                return value
            if pretrained_config is not None:
                return getattr(pretrained_config, name, default)
            return default

        return DeepSeekV4Params(
            index_n_heads=_value("index_n_heads"),
            index_head_dim=_value("index_head_dim"),
            index_topk=_value("index_topk"),
            indexer_max_chunk_size=self.indexer_max_chunk_size,
            skip_indexer_for_short_seqs=self.skip_indexer_for_short_seqs,
            use_cute_dsl_topk=self.use_cute_dsl_topk,
            use_cute_dsl_paged_mqa_logits=self.use_cute_dsl_paged_mqa_logits,
            q_split_threshold=self.q_split_threshold,
            indexer_rope_interleave=self.indexer_rope_interleave,
            enable_heuristic_topk=self.enable_heuristic_topk,
            indexer_k_dtype=self.indexer_k_dtype,
            compress_ratios=self.compress_ratios,
            window_size=self.window_size,
        )

    def to_sparse_metadata_params(self, **kwargs):
        from tensorrt_llm._torch.attention_backend.sparse.deepseek_v4 import \
            DeepSeekV4MetadataParams

        pretrained_config = kwargs.get("pretrained_config", None)

        def _value(name: str, default=None):
            value = getattr(self, name)
            if value is not None:
                return value
            if pretrained_config is not None:
                return getattr(pretrained_config, name, default)
            return default

        return DeepSeekV4MetadataParams(
            indexer_max_chunk_size=self.indexer_max_chunk_size or 32768,
            max_sparse_topk=_value("index_topk"),
            enable_indexer_skip=self.skip_indexer_for_short_seqs,
            enable_heuristic_topk=self.enable_heuristic_topk,
            use_cute_dsl_paged_mqa_logits=(self.use_cute_dsl_paged_mqa_logits),
            q_split_threshold=self.q_split_threshold,
            compress_ratios=self.compress_ratios,
            window_size=self.window_size,
        )


class SkipSoftmaxAttentionConfig(BaseSparseAttentionConfig):
    """Configuration for skip softmax attention."""
    algorithm: Literal["skip_softmax"] = Field(default="skip_softmax")
    threshold_scale_factor: Optional[Union[float, Dict[str, float]]] = Field(
        default=None,
        description="The threshold scale factor for skip softmax attention.")
    target_sparsity: Optional[Union[float, Dict[str, float]]] = Field(
        default=None,
        description="Target sparsity for prefill and/or decode phases. "
        "Requires formula coefficients in the model's config.json. "
        "Ignored if threshold_scale_factor is also set.")

    @field_validator("target_sparsity")
    @classmethod
    def validate_target_sparsity(cls, target_sparsity):
        values = target_sparsity.values() if isinstance(
            target_sparsity, dict) else (target_sparsity, )
        for value in values:
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(
                    "target_sparsity must be in [0, 1] for Skip Softmax Attention."
                )
        return target_sparsity

    def supports_backend(self, backend: str) -> bool:
        return backend == "pytorch"

    def to_sparse_params(self, **kwargs):
        import fnmatch

        from tensorrt_llm._torch.attention_backend.sparse.skip_softmax import (
            SkipSoftmaxParams, SkipSoftmaxScheduler,
            skip_softmax_ignore_from_ckpt_sparse_attention_config,
            skip_softmax_target_sparsity_from_ckpt_sparse_attention_config)

        ckpt_sparse_attention_config = kwargs.get(
            "ckpt_sparse_attention_config", None)
        checkpoint_config = kwargs.get("checkpoint_config", None)
        if ckpt_sparse_attention_config is None and isinstance(
                checkpoint_config, dict):
            ckpt_sparse_attention_config = checkpoint_config.get(
                "sparse_attention_config", checkpoint_config)
        pretrained_config = kwargs.get("pretrained_config", None)
        if ckpt_sparse_attention_config is None and isinstance(
                pretrained_config, dict):
            ckpt_sparse_attention_config = pretrained_config.get(
                "sparse_attention_config", pretrained_config)
        if ckpt_sparse_attention_config is None and pretrained_config is not None:
            ckpt_sparse_attention_config = getattr(pretrained_config,
                                                   "sparse_attention_config",
                                                   None)
        module_names = []
        module_name = kwargs.get("module_name", None)
        if module_name is not None:
            module_names.append(str(module_name))
        layer_idx = kwargs.get("layer_idx", None)
        if layer_idx is not None:
            layer_idx = str(layer_idx)
            module_names.extend((
                f"model.layers.{layer_idx}.self_attn",
                f"model.layers.{layer_idx}.attention",
                f"layers.{layer_idx}.self_attn",
                f"layers.{layer_idx}.attention",
            ))
        if module_names:
            ignore = (skip_softmax_ignore_from_ckpt_sparse_attention_config(
                ckpt_sparse_attention_config) or ())
            if any(
                    fnmatch.fnmatch(name, pattern) for pattern in ignore
                    for name in module_names):
                return None

        target_sparsity = self.target_sparsity
        if target_sparsity is None:
            target_sparsity = skip_softmax_target_sparsity_from_ckpt_sparse_attention_config(
                ckpt_sparse_attention_config)
        if (self.threshold_scale_factor is None and target_sparsity is not None
                and not isinstance(ckpt_sparse_attention_config, dict)):
            raise ValueError(
                "sparse_attention_config with target_sparsity requires formula "
                "coefficients in the model's config.json "
                "(sparse_attention_config.config_groups.*."
                "threshold_scale_factor.{prefill,decode}.{a,b}), "
                "but sparse_attention_config was not found or was not dict type in config.json."
            )
        scheduler = (
            SkipSoftmaxScheduler.from_threshold_scale_factor(
                self.threshold_scale_factor) if self.threshold_scale_factor
            is not None else SkipSoftmaxScheduler.from_target_sparsity(
                target_sparsity,
                ckpt_sparse_attention_config=ckpt_sparse_attention_config))
        return SkipSoftmaxParams(scheduler=scheduler)


class MoeLoadBalancerConfig(StrictBaseModel):
    """Pydantic configuration model for the Mixture of Experts (MoE) load balancer.

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
        """Initializes the runtime state of the configuration.
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
        """Retrieves the initial global assignments for a specific layer."""
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
    """Configuration for MoE."""
    backend: Literal[
        "AUTO", "CUTLASS", "CUTEDSL", "WIDEEP", "TRTLLM", "DEEPGEMM",
        "DENSEGEMM", "VANILLA", "TRITON", "MARLIN", "MEGAMOE_DEEPGEMM"] = Field(
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


Nvfp4Backend = Literal['cutlass', 'cublaslt', 'cutedsl', 'cuda_core', 'marlin']

# Short aliases for built-in custom tokenizers.
# Maps alias → full import path (module.ClassName).
TOKENIZER_ALIASES = {
    'deepseek_v32': 'tensorrt_llm.tokenizer.deepseek_v32.DeepseekV32Tokenizer',
    'deepseek_v4': 'tensorrt_llm.tokenizer.deepseek_v4.DeepseekV4Tokenizer',
}


class Nvfp4GemmConfig(StrictBaseModel):
    """Configuration for NVFP4 GEMM backend selection."""
    allowed_backends: List[Nvfp4Backend] = Field(
        default_factory=lambda: ['cutlass', 'cublaslt', 'cuda_core'],
        min_length=1,
        description="List of backends to consider for auto-selection. "
        "Default excludes 'cutedsl' for faster build time. "
        "Add 'cutedsl' for extreme performance at the cost of longer server launch time."
    )


class AttentionDpConfig(StrictBaseModel):
    """Configuration for attention DP."""
    enable_balance: bool = Field(default=False,
                                 description="Whether to enable balance.")
    timeout_iters: int = Field(
        default=50, description="The number of iterations to timeout.")
    batching_wait_iters: int = Field(
        default=10,
        description="The number of iterations to wait for batching.")
    enable_kv_cache_aware_routing: bool = Field(
        default=False,
        description="Enable internal KV cache-aware routing for attention DP. "
        "When enabled, distributes requests among ranks within a single "
        "instance's attention DP group, routing them to the rank with the "
        "matching prefix KV cache to reduce redundant prefill computation.")
    kv_cache_routing_load_balance_weight: float = Field(
        default=1.0,
        description=
        "Weight (beta) for the load-balance term in KV cache-aware routing. "
        "Higher values prioritize load balance over cache affinity. "
        "Only used when enable_kv_cache_aware_routing is True.")
    kv_cache_routing_match_rate_threshold: float = Field(
        default=0.1,
        description=
        "Cache-affinity gate in KV cache-aware routing. For each request, "
        "match_len contributes to scoring only when max(match_len) / "
        "request_tokens across eligible ranks is strictly above this "
        "threshold; otherwise match_len is forced to 0 so routing is driven "
        "purely by load. Default 0.1 requires at least a 10% hit rate before "
        "cache affinity kicks in, which prevents a small universal prefix "
        "(e.g. a shared system prompt) from pinning all traffic to the "
        "first warm ranks. Set to 0.0 to honour any nonzero match. "
        "Only used when enable_kv_cache_aware_routing is True.")
    kv_cache_routing_fair_share_multiplier: float = Field(
        default=2.0,
        description=
        "Loose per-rank active-request cap in KV cache-aware routing, "
        "expressed as a multiplier of the ceil fair-share "
        "(ceil((total_active + new) / tp_size)). Once a rank hits this cap "
        "within a scheduling batch it is removed from the eligible set for "
        "the remainder of the batch. Default 2.0 permits a 2x slack so "
        "cache affinity can dominate while preventing runaway concentration "
        "on a single rank. Set to 1.0 for strict fair share. "
        "Only used when enable_kv_cache_aware_routing is True.")
    kv_cache_routing_cold_start_warmup: bool = Field(
        default=False,
        description=
        "Cold-start mitigation in KV cache-aware routing. When True, the "
        "first tp_size relaxed requests after router init are round-robined "
        "across ranks (bypassing cache-affinity scoring) so every rank "
        "caches the shared system prompt before scoring would otherwise pin "
        "all traffic to the first warm rank. Only useful when requests "
        "share a long system prefix; for diverse-prompt workloads this can "
        "scatter requests that would otherwise consolidate on a single warm "
        "rank, wasting prefill. Default False preserves pre-warmup routing. "
        "Only used when enable_kv_cache_aware_routing is True.")
    kv_cache_routing_account_for_in_transfer: bool = Field(
        default=False,
        description=
        "In-transfer load accounting in KV cache-aware routing. When True, "
        "requests still streaming KV to the GEN worker (tracked by the "
        "PyExecutor AsyncTransferManager but no longer in active_requests) "
        "are folded back into the per-rank load reported via RankState. "
        "This can improve balance under heavy disagg traffic but inflates "
        "num_active_requests reported upstream, which lets the inference "
        "loop's idle-fetch wait expire even when no requests are runnable "
        "and causes empty fetch cycles. Default False preserves the prior "
        "behaviour (fetch blocks when truly idle). "
        "Only used when enable_kv_cache_aware_routing is True.")
    kv_cache_routing_conversation_affinity: bool = Field(
        default=False,
        description=
        "Enable explicit conversation-affinity routing for attention DP. When "
        "True, the first request of each conversation is round-robined across "
        "ranks and every subsequent request carrying the same "
        "conversation_params.conversation_id is pinned to that conversation's "
        "first-turn rank. OpenAI requests use the body conversation_params as "
        "canonical; the serve edge only creates conversation_params from the "
        "X-Session-ID header when the body does not provide it. This keeps a "
        "multi-turn conversation's KV-cache prefix on one rank (maximizing "
        "block reuse, minimizing cross-rank migration). Unlike "
        "enable_kv_cache_aware_routing (affinity inferred from prefix-match "
        "length, which is lost when blocks are evicted), the conversation->rank "
        "map is explicit and survives eviction. Falls back to load-balanced "
        "round-robin when no conversation_id is available. Takes precedence "
        "over enable_kv_cache_aware_routing when both are set.")
    kv_cache_routing_max_sessions: int = Field(
        default=65536,
        description=
        "LRU cap on the conversation->rank map used by conversation-affinity "
        "routing. The oldest conversations are evicted once more than this many "
        "are tracked, bounding memory on long-running servers. Only used when "
        "kv_cache_routing_conversation_affinity is True.")

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
    """Configuration for context parallelism."""
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
    """Calibration configuration."""
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
        default=None, description="The maximum number of draft tokens.")

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
        "The speculative (draft) model. Accepts either (1) a HuggingFace Hub model ID (e.g. 'yuhuili/EAGLE3-LLaMA3.1-Instruct-8B'), "
        "which will be automatically downloaded, or (2) a local filesystem path to a downloaded model directory."
    )

    max_concurrency: Optional[PositiveInt] = Field(
        default=None,
        description=
        "When specified (>0), speculation will be disabled at batch sizes above this value. Otherwise, "
        "speculation will always be on. PyTorch backend only. "
        "Mutually exclusive with max_concurrency since draft_len_schedule implicitly supports max concurrency control."
    )

    draft_len_schedule: Optional[dict[int, int]] = Field(
        default=None,
        description=
        "Developer interface: dynamically adjust draft length based on active batch size in runtime."
        "Maps batch size to draft lengths."
        "For example: draft_len_schedule = {4:4, 8:2, 32:1}"
        " - Batch sizes 1-4:   use draft_len=4"
        " - Batch sizes 5-8:   use draft_len=2"
        " - Batch sizes 9-32:  use draft_len=1"
        " - Batch sizes 33+:   use draft_len=0 (implicit, speculation disabled). "
        "Mutually exclusive with max_concurrency since draft_len_schedule implicitly support max concurrency control."
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

    use_rejection_sampling: bool = Field(
        default=False,
        status="prototype",
        description=
        "If true, enables rejection sampling for one-model speculative decoding "
        "paths when the batch contains any non-greedy request. All-greedy batches "
        "always take the argmax fast path regardless of this flag. Set to false "
        "(default) to use exact-match verification on non-greedy batches. "
        "The non-dynamic-tree one-model path requires FlashInfer.")

    allow_advanced_sampling: bool = Field(
        default=False,
        status="deprecated",
        description=
        "DEPRECATED: no-op kept for backward compatibility. Will be removed "
        "in a future release. Non-greedy sampling is now auto-detected per "
        "request; this flag no longer has any effect.")

    # If set, drafting is allowed to use chain drafter.
    _allow_chain_drafter: bool = PrivateAttr(True)
    # If set, drafting uses greedy sampling, irrespective of sampling parameters.
    _allow_greedy_draft_tokens: bool = PrivateAttr(True)
    # Internal: record decoding_type alias used during parsing (for warnings).
    _decoding_type_alias: Optional[str] = PrivateAttr(default=None)
    # If set, drafting will use separate KV cache in one-model speculative decoding.
    _allow_separate_draft_kv_cache: bool = PrivateAttr(True)
    # Internal: true when draft_len_schedule was auto-translated from max_concurrency.
    _translated_from_max_concurrency: bool = PrivateAttr(False)

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

            # Enforce smallest schedule key maps to max_draft_len for consistency.
            smallest_batch_size = min(v.keys())
            max_draft_len = info.data.get('max_draft_len')
            if max_draft_len is not None and v[
                    smallest_batch_size] != max_draft_len:
                raise ValueError(
                    f"draft_len_schedule[{smallest_batch_size}] must equal max_draft_len "
                    f"because it is the smallest batch-size key. "
                    f"Got schedule[{smallest_batch_size}]={v[smallest_batch_size]}, "
                    f"but max_draft_len={max_draft_len}.")

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

    @model_validator(mode='after')
    def validate_rejection_sampling_config(self):
        """Disable rejection sampling when SA-enhanced configurations are
        active, since SA may override the proposed draft tokens. This is a
        silent fallback so the new default (True) does not break sa_config
        users.
        """
        if self.use_rejection_sampling and getattr(self, 'sa_config',
                                                   None) is not None:
            self.use_rejection_sampling = False
        return self

    @model_validator(mode='before')
    @classmethod
    def _warn_deprecated_allow_advanced_sampling(cls, data):
        """Warn when users set the deprecated allow_advanced_sampling flag.

        Non-greedy sampling is now auto-detected per request and always
        available, so the flag is a no-op; warn loudly so callers update
        their configs before the flag is removed.
        """
        if isinstance(data, dict) and 'allow_advanced_sampling' in data:
            logger.warning(
                "DecodingBaseConfig: 'allow_advanced_sampling' is deprecated "
                "and will be removed in a future release. The flag has no "
                "effect — non-greedy sampling is now auto-detected per "
                "request.")
        return data

    @model_validator(mode='after')
    # 1. Validate that max_concurrency and draft_len_schedule are mutually exclusive.
    # 2. If max_concurrency is set, translate it to the corresponding draft_len_schedule.
    def validate_max_concurrency_and_draft_len_schedule_mutually_exclusive(
            self) -> "DecodingBaseConfig":
        if self.max_concurrency is not None and self.draft_len_schedule is not None:
            # Avoid ValueError during nested re-validation when only max_concurrency is set and draft_len_schedule is translated from max_concurrency
            if self._translated_from_max_concurrency:
                return self
            raise ValueError(
                "max_concurrency and draft_len_schedule are mutually exclusive. "
                "Use max_concurrency for a simple speculation cutoff, or "
                "draft_len_schedule for dynamic draft-length control.")

        if self.max_concurrency is None:
            return self

        if (self.max_draft_len is None
                or not self.spec_dec_mode.support_dynamic_draft_len()):
            return self

        self.draft_len_schedule = {
            int(self.max_concurrency): int(self.max_draft_len)
        }
        self._translated_from_max_concurrency = True

        return self

    def supports_backend(self, backend: str) -> bool:
        """Override if the speculation algorithm does not support
        a subset of the possible backends.
        """
        return True

    @property
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

    def get_runtime_tokens_per_gen_step(self, runtime_draft_len: int) -> int:
        """Total tokens per gen request for the current runtime draft length."""
        return 1 + runtime_draft_len

    def num_capture_layers(self) -> int:
        return 0


class KvCacheConnectorConfig(StrictBaseModel):
    """Configuration for the KV Cache Connector.

    Can be configured either by specifying a named preset via ``connector``
    (e.g. ``"lmcache"``), or by providing explicit ``connector_module``,
    ``connector_scheduler_class``, and ``connector_worker_class`` fields.
    When ``connector`` is set, the module/class fields are auto-populated
    from the preset registry and can be omitted.
    """
    connector: Optional[str] = Field(
        None,
        description="Named connector preset (e.g. 'lmcache'). "
        "When set, connector_module/scheduler_class/worker_class are "
        "auto-populated from the preset registry.",
        telemetry=TelemetryField.categorical('lmcache', 'lmcache-mp', 'kvbm'))
    connector_module: Optional[str] = Field(
        None,
        description=
        "The import path to the connector module. It will be imported with `importlib.import_module`."
    )
    connector_scheduler_class: Optional[str] = Field(
        None, description="The class name of the scheduler within the module.")
    connector_worker_class: Optional[str] = Field(
        None, description="The class name of the worker within the module.")
    server_url: Optional[str] = Field(
        None,
        description="URL for an external connector server "
        "(e.g. 'tcp://localhost:5555'). Connectors that run in "
        "multi-process mode use this to reach the cache server.")

    @model_validator(mode="after")
    def _resolve_preset(self) -> "KvCacheConnectorConfig":
        from tensorrt_llm._torch.pyexecutor.connectors.registry import \
            CONNECTOR_REGISTRY
        if self.connector is not None:
            preset = CONNECTOR_REGISTRY.get(self.connector)
            if preset is None:
                raise ValueError(
                    f"Unknown connector preset: {self.connector!r}. "
                    f"Known presets: {list(CONNECTOR_REGISTRY)}")
            for k, v in preset.items():
                if getattr(self, k) is None:
                    object.__setattr__(self, k, v)
        if self.connector_module is None:
            raise ValueError(
                "connector_module is required (set 'connector' to use a "
                "named preset, or provide connector_module explicitly)")
        if self.connector_scheduler_class is None:
            raise ValueError("connector_scheduler_class is required")
        if self.connector_worker_class is None:
            raise ValueError("connector_worker_class is required")
        return self


class LayerwiseBenchmarksConfig(StrictBaseModel):
    """Configuration for layer-wise benchmarks calibration."""
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
    decoding_type: Literal["Medusa"] = Field(default="Medusa")
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
    decoding_type: Literal["Eagle"] = Field(default="Eagle")
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
        "Deprecated TensorRT-only field with different semantics from draft model "
        "layer count. Do not use on the PyTorch backend.")
    _num_draft_hidden_layers: Optional[int] = PrivateAttr(default=None)
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
                "The eagle_choices/static tree feature is deprecated and will be removed in release 1.4."
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
        if not self.eagle3_one_model:
            logger.warning(
                "Eagle3 2-model is deprecated and will be removed in release 1.4."
            )

        self.num_eagle_layers = self.max_draft_len

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
                    f"Based on the input choices, reset the num_eagle_layers(max_draft_len) from {self.num_eagle_layers} to {num_eagle_layers_from_choices}"
                )
                self.num_eagle_layers = num_eagle_layers_from_choices
                self.max_draft_len = num_eagle_layers_from_choices

            # Each draft node has a path(choice) from the root to it.
            # So the number of choices also represents the number of max draft nodes.
            self.max_total_draft_tokens = len(self.eagle_choices)

        # Dynamic tree logic
        if self.use_dynamic_tree or self.dynamic_tree_max_topK is not None:
            self.use_dynamic_tree = True
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

            default_max_total_draft_tokens = self.dynamic_tree_max_topK * self.max_draft_len

            if self.max_total_draft_tokens is None:
                self.max_total_draft_tokens = default_max_total_draft_tokens
                logger.warning(
                    f"max_total_draft_tokens is not provided, use the default value {default_max_total_draft_tokens} (default_max_total_draft_tokens = dynamic_tree_max_topK * max_draft_len)"
                )
            else:
                if self.max_total_draft_tokens < self.max_draft_len:
                    raise ValueError(
                        f"max_total_draft_tokens ({self.max_total_draft_tokens}) should be >= max_draft_len ({self.max_draft_len})"
                    )
                if self.max_total_draft_tokens > self.dynamic_tree_max_topK * self.max_draft_len:
                    raise ValueError(
                        f"max_total_draft_tokens ({self.max_total_draft_tokens}) should be <= "
                        f"dynamic_tree_max_topK * max_draft_len ({self.dynamic_tree_max_topK * self.max_draft_len})"
                    )

        # Linear tree
        if self.max_total_draft_tokens is None:
            self.max_total_draft_tokens = self.max_draft_len

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
        """Returns the number of layers to capture of the target model.
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


class SAEnhancerConfig(StrictBaseModel):
    """Configuration for the Suffix Automaton (SA) draft enhancer.

    Use this to combine SA pattern-matching drafting with another speculative
    decoding method (Eagle3, MTP, PARD).  When provided as ``sa_config`` on a
    decoding config, SA drafting is enabled and may override neural draft
    tokens when the suffix match length meets the *threshold*.

    For standalone SA speculative decoding (no neural drafter), use
    :class:`SADecodingConfig` instead.
    """

    threshold: PositiveInt = Field(
        default=4,
        description="Minimum suffix match length required for the SA output "
        "to override neural draft tokens.")
    enable_global_pool: bool = Field(
        default=False,
        description="When True, each request searches all active SA states "
        "for the longest match, not just its own. Improves acceptance rates "
        "when requests share common patterns.")


class Eagle3DecodingConfig(EagleDecodingConfig):
    decoding_type: Literal["Eagle3"] = Field(default="Eagle3")

    # Backs the dynamic-tree worker's pre-allocated, batch-indexed CUDA buffers
    # (draft_tokens_buffer, history_*_buffer, tree_mask_buffer, etc. in
    # Eagle3OneModelDynamicTreeWorker.__init__). This MUST equal the global
    # max_batch_size: the worker indexes those buffers with batch_idx in
    # [0, global_max_batch_size) at runtime with no bounds check, so any value
    # smaller than the global will OOB during warmup or generation as soon as
    # batch_idx exceeds this capacity (illegal memory access). It is therefore
    # exposed as a PrivateAttr -- not a user-tunable knob -- and is
    # auto-populated by py_executor_creator from the global max_batch_size.
    _max_batch_size: Optional[int] = PrivateAttr(default=None)

    sa_config: Optional[SAEnhancerConfig] = Field(
        default=None,
        status="beta",
        description="Optional Suffix Automaton configuration. When set, "
        "combines SA drafting with Eagle3 speculative decoding.")


class SaveHiddenStatesDecodingConfig(DecodingBaseConfig):
    decoding_type: Literal["SaveState"] = Field(default="SaveState")
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
        """Returns the number of layers to save.
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
    decoding_type: Literal["User_Provided"] = Field(default="User_Provided")
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
    """Configuration for NGram drafter speculative decoding."""
    decoding_type: Literal["NGram"] = Field(default="NGram")
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


class SADecodingConfig(DecodingBaseConfig):
    """Configuration for standalone Suffix Automaton (SA) speculative decoding.

    Uses a GPU-native suffix automaton for pattern matching. Drafting runs
    inside the target model forward; supports CUDA graph and overlap scheduler.

    To combine SA with a neural drafter (Eagle3, MTP, PARD) instead of using
    it standalone, pass :class:`SAEnhancerConfig` via ``sa_config``.
    """
    decoding_type: Literal["SA"] = Field(default="SA")
    max_matching_ngram_size: int = Field(
        default=-1,
        description="Positive value (e.g., 3): fixed-size ngram matching. "
        "-1: longest possible match via suffix automaton. 0 is invalid.")
    enable_global_pool: bool = Field(
        default=False,
        description="When True, each request searches all active SA states "
        "for the longest match, not just its own. Improves acceptance rates "
        "when requests share common patterns. "
        "Limitations: at most 1024 concurrent slots; suffix matching is "
        "capped at 64 tokens per request.")

    global_pool_size: Optional[PositiveInt] = Field(
        default=None,
        description="Number of SA slots in the global pool. "
        "When None and enable_global_pool=True, defaults to "
        "max(64, max_batch_size) — a fixed-size pool independent of batch size. "
        "When set explicitly, must be >= max_batch_size. "
        "Completed requests' SA states are retained in the pool for "
        "cross-request search until the pool is full, at which point "
        "the oldest completed request is evicted. "
        "Only effective when enable_global_pool=True.")

    @model_validator(mode='after')
    def validate_sa_config(self):
        if self.max_matching_ngram_size == 0:
            raise ValueError(
                "max_matching_ngram_size must be > 0 (fixed ngram) or -1 (longest match). "
                "Got 0.")
        if self.enable_global_pool and self.max_matching_ngram_size != -1 and not (
                1 <= self.max_matching_ngram_size <= 64):
            raise ValueError(
                "max_matching_ngram_size must be -1 (longest match) or in [1, 64] "
                "when enable_global_pool is True. "
                f"Got {self.max_matching_ngram_size}.")
        if self.max_draft_len is None or self.max_draft_len <= 0:
            raise ValueError("max_draft_len must be > 0 for SA")
        if self.global_pool_size is not None:
            if self.global_pool_size < 1:
                raise ValueError("global_pool_size must be >= 1")
            if not self.enable_global_pool:
                raise ValueError(
                    "global_pool_size requires enable_global_pool=True")
        self.max_total_draft_tokens = self.max_draft_len
        return self

    def supports_backend(self, backend: str) -> bool:
        return backend == "pytorch"


class DraftTargetDecodingConfig(DecodingBaseConfig):
    decoding_type: Literal["Draft_Target"] = Field(default="Draft_Target")
    _draft_target_one_model: bool = PrivateAttr(True)

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

    @functools.cached_property
    def spec_dec_mode(self):
        from tensorrt_llm._torch.speculative.interface import \
            SpeculativeDecodingMode as TorchSpeculativeDecodingMode
        if self._draft_target_one_model:
            return TorchSpeculativeDecodingMode.DRAFT_TARGET_ONE_MODEL
        return TorchSpeculativeDecodingMode.DRAFT_TARGET


class MTPDecodingConfig(DecodingBaseConfig):
    decoding_type: Literal["MTP"] = Field(default="MTP")
    use_relaxed_acceptance_for_thinking: bool = Field(
        default=False,
        description=
        "Enable relaxed acceptance during thinking phase for reasoning models. Accepts draft tokens matching any top-K candidate instead of exact top-1."
    )
    relaxed_topk: PositiveInt = Field(
        default=1,
        description=
        "Number of top candidate tokens to consider for relaxed acceptance. Draft token is accepted if it matches any of these."
    )
    relaxed_delta: NonNegativeFloat = Field(
        default=0.0,
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

    sa_config: Optional[SAEnhancerConfig] = Field(
        default=None,
        status="beta",
        description="Optional Suffix Automaton configuration. When set, "
        "combines SA drafting with MTP speculative decoding.")

    # Internal field: number of MTP layers in the model checkpoint.
    # Auto-populated from pretrained_config by update_spec_config_from_model_config.
    # Do not set manually.
    num_nextn_predict_layers: Optional[int] = Field(
        default=None,
        init=False,
        description="Number of MTP layers in the model checkpoint. "
        "Auto-populated from the model's pretrained config. Do not set manually."
    )

    begin_thinking_phase_token: NonNegativeInt = Field(
        default=128798,
        description=
        "Token ID marking start of thinking phase. Relaxed acceptance only applies within this phase."
    )
    end_thinking_phase_token: NonNegativeInt = Field(
        default=128799,
        description=
        "Token ID marking end of thinking phase. Strict acceptance resumes after this."
    )

    @model_validator(mode="before")
    @classmethod
    def _remap_deprecated_num_nextn_predict_layers(cls, data):
        if isinstance(data, dict) and "num_nextn_predict_layers" in data:
            logger.warning(
                "MTPDecodingConfig: 'num_nextn_predict_layers' is deprecated and will be "
                "removed in a future release. Use 'max_draft_len' instead.")
            if "max_draft_len" not in data:
                data = dict(data)
                data["max_draft_len"] = data.pop("num_nextn_predict_layers")
            else:
                data = dict(data)
                data.pop("num_nextn_predict_layers")
        return data

    @model_validator(mode="after")
    def set_max_total_draft_tokens(self):
        # Leave max_draft_len as None ("use the model's num_nextn_predict_layers")
        # when the user doesn't set it; update_spec_config_from_model_config
        # resolves it from the checkpoint before the model runs. When the user
        # does set it, validate and mirror to max_total_draft_tokens (current MTP
        # only supports a linear tree).
        if self.max_draft_len is not None:
            if self.max_draft_len <= 0:
                raise ValueError("max_draft_len must be > 0 for MTP")
            self.max_total_draft_tokens = self.max_draft_len
        return self

    @model_validator(mode="after")
    def log_two_model_deprecation_warning(self):
        if not self.mtp_eagle_one_model:
            logger.warning(
                "2-model style MTP is deprecated and will be removed in release 1.4."
            )
        return self

    def supports_backend(self, backend: str) -> bool:
        return backend in ("pytorch", "_autodeploy")

    @property
    def num_capture_layers(self) -> int:
        # MTP_EAGLE (two-model) feeds captured target hidden states into the
        # separate draft engine, so the shared Eagle3ResourceManager must
        # allocate a hidden_states buffer for it. MTP_EAGLE_ONE_MODEL passes
        # the target model's hidden_states straight to the MTP layer
        # (see Eagle3OneModelWorker.prepare_1st_drafter_inputs / _run_draft_forward,
        # both gated on self.is_mtp_eagle), so no capture buffer is needed
        # and we should skip allocation to avoid disabling post-MLP/MoE
        # fusion via the layer-capture hook.
        return 1 if self.spec_dec_mode.is_mtp_eagle() else 0

    @property
    def spec_dec_mode(self):
        from tensorrt_llm._torch.speculative.interface import \
            SpeculativeDecodingMode as TorchSpeculativeDecodingMode

        # num_nextn_predict_layers is set from the model's pretrained config by
        # update_spec_config_from_model_config. Treat None (before model load) as 1.
        n = self.num_nextn_predict_layers if self.num_nextn_predict_layers is not None else 1
        if n == 1 and not self.use_mtp_vanilla and self.mtp_eagle_one_model:
            return TorchSpeculativeDecodingMode.MTP_EAGLE_ONE_MODEL
        elif n == 1 and not self.use_mtp_vanilla and not self.mtp_eagle_one_model:
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

    decoding_type: Literal["PARD"] = Field(default="PARD")

    sa_config: Optional[SAEnhancerConfig] = Field(
        default=None,
        status="beta",
        description="Optional Suffix Automaton configuration. When set, "
        "combines SA drafting with PARD speculative decoding.")

    @model_validator(mode="after")
    def set_max_total_draft_tokens(self):
        self.max_total_draft_tokens = self.max_draft_len
        return self

    @property
    def tokens_per_gen_step(self) -> int:
        """PARD needs 2K tokens per gen request: K+1 accepted + K-1 masks."""
        return 2 * self.max_draft_len

    def get_runtime_tokens_per_gen_step(self, runtime_draft_len: int) -> int:
        """PARD needs 2K runtime tokens per gen request for logical draft length K."""
        return 1 if runtime_draft_len == 0 else 2 * runtime_draft_len

    def supports_backend(self, backend: str) -> bool:
        return backend == "pytorch"

    @functools.cached_property
    def spec_dec_mode(self):
        from tensorrt_llm._torch.speculative.interface import \
            SpeculativeDecodingMode as TorchSpeculativeDecodingMode
        return TorchSpeculativeDecodingMode.PARD


class DFlashDecodingConfig(DecodingBaseConfig):
    """Configuration for DFlash speculative decoding.

    DFlash is a target-dependent speculative decoding method that uses
    hidden states from specific target model layers as cross-attention
    context in the draft model to predict multiple draft tokens in parallel.

    Key features:
    - Target-dependent: uses hidden states from target model layers
    - Parallel prediction: all K draft tokens in one forward pass
    - Cross-attention: draft model attends to target hidden states

    Reference: https://arxiv.org/pdf/2602.06036
    """
    mask_token_id: Optional[int] = Field(
        default=None,
        description=
        "The token ID used as a mask token for parallel draft prediction. "
        "If None, it will be read from the draft model config (dflash_config.mask_token_id)."
    )

    target_layer_ids: Optional[List[int]] = Field(
        default=None,
        description=
        "List of target model layer indices whose hidden states are captured "
        "for cross-attention in the draft model. If None, read from the draft "
        "model config (dflash_config.target_layer_ids).")

    decoding_type: Literal["DFlash"] = Field(default="DFlash")

    @model_validator(mode="after")
    def set_max_total_draft_tokens(self):
        self.max_total_draft_tokens = self.max_draft_len
        return self

    @property
    def tokens_per_gen_step(self) -> int:
        """DFlash only needs K+1 tokens per gen request (K drafts + 1 bonus).

        The draft produces its own mask queries internally; passing mask
        fillers through the target is pure wasted work at large batch size.
        """
        return self.max_draft_len + 1

    def get_runtime_tokens_per_gen_step(self, runtime_draft_len: int) -> int:
        """DFlash needs K+1 runtime tokens per gen request (K drafts + 1 bonus)."""
        return 1 + runtime_draft_len

    def supports_backend(self, backend: str) -> bool:
        return backend == "pytorch"

    @functools.cached_property
    def spec_dec_mode(self):
        from tensorrt_llm._torch.speculative.interface import \
            SpeculativeDecodingMode as TorchSpeculativeDecodingMode
        return TorchSpeculativeDecodingMode.DFLASH


class AutoDecodingConfig(DecodingBaseConfig):
    """Configuration for auto speculative decoding.

    This config will automatically select a good, draft-model free
    speculation algorithm with some heuristic.

    Attributes that are inherited from the base class are ignored.
    """

    decoding_type: Literal["AUTO"] = Field(default="AUTO")

    @model_validator(mode="after")
    def set_max_total_draft_tokens(self):
        self.max_total_draft_tokens = self.max_draft_len  # Current Auto only supports linear tree
        return self

    def supports_backend(self, backend: str) -> bool:
        return backend == "pytorch"


class PrometheusMetricsConfig(StrictBaseModel):
    """Configuration for Prometheus metrics collection.

    Groups all Prometheus-related parameters including custom histogram bucket
    boundaries for latency metrics.
    """

    e2e_request_latency_buckets: Optional[List[float]] = Field(
        default=None,
        description=
        "Custom histogram bucket boundaries (in seconds) for trtllm_e2e_request_latency_seconds. "
        "Defaults to built-in values when unset.",
        status="prototype")

    time_to_first_token_buckets: Optional[List[float]] = Field(
        default=None,
        description=
        "Custom histogram bucket boundaries (in seconds) for trtllm_time_to_first_token_seconds. "
        "Defaults to built-in values when unset.",
        status="prototype")

    time_per_output_token_buckets: Optional[List[float]] = Field(
        default=None,
        description=
        "Custom histogram bucket boundaries (in seconds) for trtllm_time_per_output_token_seconds. "
        "Defaults to built-in values when unset.",
        status="prototype")

    request_queue_time_buckets: Optional[List[float]] = Field(
        default=None,
        description=
        "Custom histogram bucket boundaries (in seconds) for trtllm_request_queue_time_seconds. "
        "Defaults to built-in values when unset.",
        status="prototype")

    request_prefill_time_buckets: Optional[List[float]] = Field(
        default=None,
        description=
        "Custom histogram bucket boundaries (in seconds) for trtllm_request_prefill_time_seconds. "
        "Defaults to built-in values when unset.",
        status="prototype")

    request_decode_time_buckets: Optional[List[float]] = Field(
        default=None,
        description=
        "Custom histogram bucket boundaries (in seconds) for trtllm_request_decode_time_seconds. "
        "Defaults to built-in values when unset.",
        status="prototype")

    request_inference_time_buckets: Optional[List[float]] = Field(
        default=None,
        description=
        "Custom histogram bucket boundaries (in seconds) for trtllm_request_inference_time_seconds. "
        "Defaults to built-in values when unset.",
        status="prototype")

    @field_validator(
        "e2e_request_latency_buckets",
        "time_to_first_token_buckets",
        "time_per_output_token_buckets",
        "request_queue_time_buckets",
        "request_prefill_time_buckets",
        "request_decode_time_buckets",
        "request_inference_time_buckets",
    )
    @classmethod
    def validate_histogram_buckets(cls, v: Optional[List[float]],
                                   info) -> Optional[List[float]]:
        """Validate that histogram bucket lists are non-empty and strictly increasing."""
        if v is None:
            return v
        if len(v) == 0:
            raise ValueError(
                f"{info.field_name} must not be empty when provided.")
        if any(a >= b for a, b in zip(v, v[1:])):
            raise ValueError(
                f"{info.field_name} must be strictly increasing, got {v}.")
        return v


class RayPlacementConfig(StrictBaseModel):
    """Configuration for Ray GPU workers placement.
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


class ExecutorMemoryType(StrEnum):
    """Types of GPU memory used by executor.

    These are used by the sleep/wakeup feature to target specific type of memory.
    """
    SAMPLER = "sampler"
    DRAFTER = "drafter"
    GUIDED_DECODER = "guided_decoder"
    SPEC_RESOURCES = "spec_resource_manager"
    INIT_KV_CACHE = "_no_capture_init_kv_cache"
    INIT_EXTRA_RESOURCES = "_no_capture_init_extra_resources"
    MODEL_EXTRA = "model_extra"
    EXTRA_RESOURCES = "executor_extra"
    KV_CACHE = "kv_cache"
    MODEL_ENGINE_MAIN = "model"
    MODEL_ENGINE_DRAFT = "draft_model"
    MODEL_WEIGHTS_MAIN = "model_weights"
    MODEL_WEIGHTS_DRAFT = "draft_model_weights"


@dataclass
class _SleepConfigDefaultFactory:
    """Picklable replacement for ``lambda: default_mode`` in SleepConfig's defaultdict."""

    default_mode: Any

    def __call__(self) -> Any:
        return self.default_mode


class SleepConfig(StrictBaseModel):
    """Configuration for the LLM sleep/wakeup feature."""

    restore_modes: dict[
        ExecutorMemoryType, Literal["NONE", "MEMSET", "CPU", "PINNED"]
        | _VirtualMemoryRestoreMode] = Field(
            default_factory=lambda: SleepConfig._make_defaulted_restore_modes(),
            description="Per-component RestoreMode for the sleep feature. "
            "Keys are ExecutorMemoryType values (e.g. 'model', 'kv_cache'), "
            "values can be RestoreMode names (NONE, MEMSET, CPU, PINNED) or "
            "RestoreMode enum values. "
            "Unlisted entries default to the suitable mode selected between "
            "PINNED and CPU.")

    DEFAULT_RESTORE_MODES: ClassVar[dict[str, str]] = {
        ExecutorMemoryType.KV_CACHE: "NONE",
    }

    @staticmethod
    def _normalize_restore_mode(
            value: str | _VirtualMemoryRestoreMode
    ) -> _VirtualMemoryRestoreMode:
        from tensorrt_llm._torch.virtual_memory import RestoreMode
        if isinstance(value, RestoreMode):
            return value
        if isinstance(value, str):
            try:
                return RestoreMode[value]
            except KeyError as e:
                valid = ", ".join(mode.name for mode in RestoreMode)
                raise ValueError(
                    f"invalid restore_mode: {value}. Expected one of: {valid}"
                ) from e
        raise ValueError(f"invalid restore_mode type: {type(value).__name__}")

    @staticmethod
    def _normalize_executor_memory_type(
            key: ExecutorMemoryType | str) -> ExecutorMemoryType:
        if isinstance(key, ExecutorMemoryType):
            return key
        if isinstance(key, str):
            try:
                return ExecutorMemoryType(key)
            except ValueError as e:
                valid = ", ".join(member.value for member in ExecutorMemoryType)
                raise ValueError(
                    f"invalid executor memory type: {key}. Expected one of: {valid}"
                ) from e
        raise ValueError(
            f"executor memory type must be ExecutorMemoryType or str, got {type(key).__name__}"
        )

    @classmethod
    def _make_defaulted_restore_modes(
        cls,
        cases: Optional[dict[ExecutorMemoryType,
                             str | _VirtualMemoryRestoreMode]] = None,
        *,
        default_mode: Optional[_VirtualMemoryRestoreMode] = None
    ) -> defaultdict[ExecutorMemoryType, _VirtualMemoryRestoreMode]:
        from tensorrt_llm._torch.virtual_memory import RestoreMode
        default_mode: _VirtualMemoryRestoreMode = default_mode or (
            RestoreMode.PINNED if prefer_pinned() else RestoreMode.CPU)

        if cases is None:
            cases = cls.DEFAULT_RESTORE_MODES
        normalized_cases = {
            cls._normalize_executor_memory_type(key):
            cls._normalize_restore_mode(value)
            for key, value in cases.items()
        }
        factory = _SleepConfigDefaultFactory(default_mode)
        return defaultdict(factory, normalized_cases)

    @field_validator('restore_modes', mode='plain')
    @classmethod
    def _validate_restore_modes(cls, v):
        if not isinstance(v, dict):
            raise ValueError(
                f"restore_modes must be dict, got {type(v).__name__}")

        default_mode = None
        if isinstance(v, defaultdict) and v.default_factory is not None:
            try:
                default_mode = cls._normalize_restore_mode(v.default_factory())
            except Exception as e:
                raise ValueError(
                    "restore_modes defaultdict default_factory must return a valid RestoreMode"
                ) from e

        return cls._make_defaulted_restore_modes(v, default_mode=default_mode)


class PybindMirror(ABC):
    """A class containing the utilities for mirroring Python classes to
    pybind classes.
    """

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
        """Class decorator that ensures Python class fields mirror those of a C++ class.

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
        """Get all the enum fields from the pybind class."""
        return [
            f for f in pybind_class.__members__.keys()
            if not f.startswith('_') and not callable(getattr(pybind_class, f))
        ]

    @staticmethod
    def mirror_pybind_enum(pybind_class):
        """Mirror the enum fields from the pybind class to the Python class."""

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
        """Get all the variable fields from the pybind class."""
        return [
            f for f in dir(config_cls)
            if not f.startswith('_') and not callable(getattr(config_cls, f))
        ]

    @staticmethod
    def pybind_equals(obj0, obj1):
        """Check if two pybind objects are equal."""
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
    """Combined metaclass for Enum and PybindMirror.  This is crucial."""


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
    """Context chunking policy."""
    FIRST_COME_FIRST_SERVED = "FIRST_COME_FIRST_SERVED"
    EQUAL_PROGRESS = "EQUAL_PROGRESS"
    FORCE_CHUNK = "FORCE_CHUNK"

    def _to_pybind(self):
        return getattr(_ContextChunkingPolicy, self.value)


class WaitingQueuePolicy(StrEnum):
    """Waiting queue scheduling policy for managing pending requests."""

    FCFS = "fcfs"  # First-Come-First-Served
    PRIORITY = "priority"  # Higher request.priority value is served first; ties broken by FCFS


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

    use_python_scheduler: bool = Field(
        default=False,
        description="Use pure-Python scheduler instead of C++ scheduler.")

    enable_prefix_aware_scheduling: bool = Field(
        default=True,
        description=("Use KV prefix-reuse estimates for scheduler admission, "
                     "duplicate-request deferral, and token-budget decisions. "
                     "This is orthogonal to "
                     "kv_cache_config.enable_block_reuse."))

    def _to_pybind(self) -> _SchedulerConfig:
        return _SchedulerConfig(
            capacity_scheduler_policy=self.capacity_scheduler_policy._to_pybind(
            ),
            context_chunking_policy=self.context_chunking_policy._to_pybind()
            if self.context_chunking_policy else None,
            dynamic_batch_config=self.dynamic_batch_config._to_pybind()
            if self.dynamic_batch_config else None,
            enable_prefix_aware_scheduling=self.enable_prefix_aware_scheduling)


@PybindMirror.mirror_pybind_fields(_PeftCacheConfig)
class PeftCacheConfig(StrictBaseModel, PybindMirror):
    """Configuration for the PEFT cache."""
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
    """Configuration for lookahead speculative decoding."""

    decoding_type: Literal["Lookahead"] = Field(default="Lookahead")
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
        Eagle3DecodingConfig,  # Must be before EagleDecodingConfig since it's a subclass
        EagleDecodingConfig,
        LookaheadDecodingConfig,
        MedusaDecodingConfig,
        MTPDecodingConfig,
        NGramDecodingConfig,
        SADecodingConfig,
        UserProvidedDecodingConfig,
        SaveHiddenStatesDecodingConfig,
        PARDDecodingConfig,
        DFlashDecodingConfig,
        AutoDecodingConfig,
    ],
    Field(discriminator="decoding_type"),
]

SparseAttentionConfig: TypeAlias = Annotated[
    Union[
        RocketSparseAttentionConfig,
        DeepSeekSparseAttentionConfig,
        DeepSeekV4SparseAttentionConfig,
        SkipSoftmaxAttentionConfig,
        MiniMaxM3SparseAttentionConfig,
    ],
    Field(discriminator="algorithm"),
]


class KvCacheCompressionConfig(StrictBaseModel):
    """Config for KV-cache compression: a compression manager runs a KV-reduction
    algorithm (e.g. periodic token eviction) alongside KVCacheManagerV2.

    Kept separate from SparseAttentionConfig by design -- compression changes
    which KV is stored, not the attention computation. The manager is registered
    as a resource manager in create_py_executor (_util.py), like the KV cache
    manager itself. Concrete algorithms subclass this and add their parameters.
    """
    algorithm: str = Field(
        description=
        "Name of the KV-cache compression algorithm to run; selects which "
        "compression manager is built. Concrete algorithm configs subclass this "
        "and set the value.")


@PybindMirror.mirror_pybind_fields(_AgentTreeConfig)
class AgentTreeConfig(StrictBaseModel, PybindMirror):
    """Configuration for agent tree scheduling.

    Controls how agent requests are scheduled relative to regular chat requests.
    """
    agent_percentage: float = Field(
        default=0.0,
        description=
        "The percentage of agent requests to schedule. Defaults to 0.0. "
        "Should be between 0.0 and 1.0. -1.0 means random schedule between agent and chatbot."
    )
    agent_types: Optional[List[str]] = Field(
        default=None,
        description=
        "Types of agents to schedule (e.g. 'AgentDeepResearch', 'Researcher', 'MultiroundChat')."
    )
    agent_inflight_seq_num: int = Field(
        default=2**31 - 1,
        description="Max number of inflight sequences for agent requests.")

    def _to_pybind(self):
        return _AgentTreeConfig(
            agent_percentage=self.agent_percentage,
            agent_types=self.agent_types,
            agent_inflight_seq_num=self.agent_inflight_seq_num,
        )


class ReorderRequestPolicyConfig(StrictBaseModel):
    """Configuration for request reordering policy."""
    policy_name: Optional[Literal["AgentTree"]] = Field(
        default=None, description="The name of the request reordering policy.")
    policy_args: AgentTreeConfig = Field(
        default_factory=AgentTreeConfig,
        description="The arguments of the request reordering policy.")


@PybindMirror.mirror_pybind_fields(_KvCacheConfig)
class KvCacheConfig(StrictBaseModel, PybindMirror):
    """Configuration for the KV cache."""
    enable_block_reuse: bool = Field(
        default=True,
        description=
        "Controls if KV cache blocks can be reused for different requests.")
    max_tokens: Optional[int] = Field(
        default=None,
        description=
        "The maximum number of tokens that should be stored in the KV cache. If both `max_tokens` and `free_gpu_memory_fraction` are specified, memory corresponding to the minimum will be used."
    )
    max_attention_window: Optional[List[int]] = Field(
        default=None,
        min_length=1,
        description=
        "Size of the attention window for each sequence. Only the last tokens will be stored in the KV cache. If the number of elements in `max_attention_window` is less than the number of layers, `max_attention_window` will be repeated multiple times to the number of layers."
    )
    sink_token_length: Optional[int] = Field(
        default=None,
        description=
        "Deprecated and ignored on the PyTorch backend. StreamingLLM is not supported "
        "by the PyTorch attention kernels — any non-None value has no effect and "
        "will be silently dropped before reaching the executor.",
        deprecated=True)
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
    disk_cache_size: Optional[NonNegativeInt] = Field(
        default=None,
        description=
        "Size of the disk cache in bytes. Only used by KV cache manager v2 in the PyTorch backend."
    )
    disk_cache_path: Optional[str] = Field(
        default=None,
        description=
        "Directory used for disk KV cache files. Must be set when `disk_cache_size` is positive."
    )
    cross_kv_cache_fraction: Optional[float] = Field(
        default=None,
        description=
        "The fraction of the KV Cache memory should be reserved for cross attention. If set to p, self attention will use 1-p of KV Cache memory and cross attention will use p of KV Cache memory. Defaults to None (unset); must be set when using an encoder-decoder model and must not be set otherwise."
    )
    secondary_offload_min_priority: Optional[int] = Field(
        default=None,
        description=
        "Only blocks with priority > secondary_offload_min_priority can be offloaded to secondary memory."
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
    iteration_stats_interval: PositiveInt = Field(
        default=1,
        description=
        "How often (in iterations) to collect per-iteration KV cache statistics. "
        "A value of 1 means every iteration; a value of N means every Nth iteration. "
        "Between collections, the C++ deltas accumulate, so the reported deltas cover N iterations."
    )

    # This is a pure python field, not a pybind field. It is only for the Pytorch backend.
    dtype: str = Field(
        default="auto",
        description=
        "The data type for the KV cache. 'auto' (default) leaves the checkpoint's "
        "own KV-cache quantization metadata untouched (quant_config.kv_cache_quant_algo "
        "is inherited as-is); 'fp8' or 'nvfp4' override it explicitly. Resolved at "
        "LLM-construction time, including when set via trtllm-serve "
        "--extra_llm_api_options.",
        telemetry=TelemetryField.categorical("auto", "float16", "bfloat16",
                                             "float32", "fp8", "nvfp4"))

    # This is a pure python field, not a pybind field. It is only for the Pytorch backend.
    mamba_ssm_cache_dtype: Literal[
        "auto", "float16", "bfloat16", "float32"] = Field(
            default="auto",
            description=
            "The data type to use for the Mamba SSM cache. If set to 'auto', the data type will be inferred from the model config."
        )

    # This is a pure python field, not a pybind field. It is only for the Pytorch backend.
    mamba_ssm_stochastic_rounding: bool = Field(
        default=False,
        description=
        "Enable stochastic rounding for Mamba SSM state updates. Only applicable with float16 cache dtype."
    )

    # This is a pure python field, not a pybind field. It is only for the Pytorch backend.
    mamba_ssm_philox_rounds: int = Field(
        default=10,
        ge=1,
        description=
        "Number of Philox rounds for stochastic rounding PRNG. Higher values give better randomness "
        "but increase compute cost. Only used when mamba_ssm_stochastic_rounding is enabled."
    )

    tokens_per_block: int = Field(default=32,
                                  description="The number of tokens per block.")

    # This is a pure python field, not a pybind field. It is only for the Pytorch backend.
    mamba_state_cache_interval: PositiveInt = Field(
        default=256,
        description=
        "The number of tokens between cache steps in the Mamba prefix cache.")

    use_kv_cache_manager_v2: bool = Field(
        default=False,
        status="prototype",
        description="Whether to use the KV cache manager v2 (experimental).")

    # This is a pure python field, not a pybind field. It is only for the Pytorch backend.
    enable_swa_scratch_reuse: bool = Field(
        default=False,
        status="prototype",
        description=
        "Whether KV cache manager v2 uses SWA scratch reuse during prefill.")

    kv_cache_event_hash_algo: Literal[
        "auto", "v1_block_key", "v2_sha256", "v2_sha256_64"] = Field(
            default="auto",
            status="prototype",
            description=
            "The block hash algorithm used by KV cache manager events. "
            "'auto' uses the native hash for each KV cache manager. "
            "Explicit V2 hash choices are ignored with a warning by the V1 "
            "KV cache manager.")

    max_util_for_resume: float = Field(
        default=0.95,
        ge=0,
        le=1,
        status="prototype",
        description=
        "The maximum utilization of the KV cache for resume. Default is 95%. Only used when using KV cache manager v2 (experimental)."
    )

    enable_kv_pool_rebalance: bool = Field(
        default=False,
        status="prototype",
        description=
        "Opt in to the KVCacheManagerV2 auto-tuner (``adjust()``) for "
        "rebalancing pool-group ratios between iterations. When True the "
        "PyExecutor calls ``adjust()`` opportunistically; the auto-tuner "
        "itself remains gated by V2's internal 2000-sample / 120s cooldown. "
        "When False (default) the rebalance hook is skipped entirely and "
        "pool ratios remain at their warmup-derived values. Beta: enable at "
        "your own risk. Only used when using KV cache manager v2 "
        "(experimental).")

    disk_prefetch_num_reqs: int = Field(
        default=0,
        ge=0,
        description=
        "Number of queued context requests to prefetch disk-tier KV cache blocks to host for. "
        "Set to 0 to disable prefetch. Only effective with KV cache manager v2 and block reuse enabled."
    )

    # This is a pure python field, not a pybind field. It is only for the Pytorch backend.
    pool_ratio: Optional[List[float]] = Field(
        default=None,
        min_length=1,
        status="prototype",
        description=
        "Initial pool ratios for KV cache manager v2. When used by DeepSeek-V4, "
        "values map to KVCacheManagerV2 pool_group_id order and must sum to 1.0. "
        "When set, DeepSeek-V4 uses this directly and avg_seq_len does not take effect."
    )

    # This is a pure python field, not a pybind field. It is only for the Pytorch backend.
    avg_seq_len: Optional[PositiveInt] = Field(
        default=None,
        status="prototype",
        description=
        "Average sequence length used by DeepSeek-V4 to build the KV cache manager v2 "
        "typical step. If unset, max_seq_len is used. This does not take effect when "
        "pool_ratio is set.")

    # This is a pure python field, not a pybind field. It is only for the Pytorch backend.
    block_reuse_policy: Literal["all_reusable", "per_request"] = Field(
        default="all_reusable",
        status="prototype",
        description="KV cache manager v2 block reuse policy. "
        "With SWA scratch reuse and 'all_reusable', only non-scratch "
        "blocks are saved for reuse.")

    def _to_pybind(self):
        config = _KvCacheConfig(
            enable_block_reuse=self.enable_block_reuse,
            max_tokens=self.max_tokens,
            max_attention_window=self.max_attention_window,
            free_gpu_memory_fraction=self.free_gpu_memory_fraction,
            host_cache_size=self.host_cache_size,
            cross_kv_cache_fraction=self.cross_kv_cache_fraction,
            secondary_offload_min_priority=self.secondary_offload_min_priority,
            event_buffer_max_size=self.event_buffer_max_size,
            enable_partial_reuse=self.enable_partial_reuse,
            copy_on_partial_reuse=self.copy_on_partial_reuse,
            use_uvm=self.use_uvm,
            attention_dp_events_gather_period_ms=self.
            attention_dp_events_gather_period_ms,
            max_gpu_total_bytes=self.max_gpu_total_bytes)
        return config

    @field_validator('free_gpu_memory_fraction')
    @classmethod
    def validate_free_gpu_memory_fraction(cls, v: float):
        """Validates that the fraction is between 0.0 and 1.0."""
        if not 0 <= v <= 1:
            raise ValueError(
                "kv_cache_config.free_gpu_memory_fraction must be a float between 0 and 1"
            )
        return v

    @field_validator('cross_kv_cache_fraction')
    @classmethod
    def validate_cross_kv_cache_fraction(cls, v: Optional[float]):
        if v is None:
            return v
        if not 0 <= v <= 1:
            raise ValueError(
                "kv_cache_config.cross_kv_cache_fraction must be a float between 0 and 1"
            )
        return v

    @field_validator('dtype')
    @classmethod
    def validate_dtype(cls, v: str):
        v = v.lower()
        if v in ("auto", "fp8",
                 "nvfp4") or v in _str_to_torch_dtype_dict.keys():
            return v

        raise ValueError(
            'kv_cache_config.dtype must be one of "auto", "fp8", "nvfp4", or valid torch.dtype string'
        )

    @field_validator('max_gpu_total_bytes')
    @classmethod
    def validate_max_gpu_total_bytes(cls, v: int):
        if v < 0:
            raise ValueError(
                "kv_cache_config.max_gpu_total_bytes must be non-negative")
        return v

    @model_validator(mode='after')
    def validate_disk_cache_config(self):
        if self.disk_cache_size is not None and self.disk_cache_size > 0:
            if not self.disk_cache_path:
                raise ValueError(
                    "kv_cache_config.disk_cache_path must be set when disk_cache_size is positive"
                )
            if not os.path.isdir(self.disk_cache_path):
                raise ValueError(
                    f"kv_cache_config.disk_cache_path {self.disk_cache_path} does not exist or is not a directory"
                )
        return self

    @field_validator('max_attention_window')
    @classmethod
    def validate_max_attention_window(cls, v: Optional[List[int]]):
        if v is None:
            return v

        if not isinstance(v, list) or len(v) == 0:
            raise ValueError(
                "kv_cache_config.max_attention_window must be a non-empty list of positive integers"
            )
        for i in v:
            if not isinstance(i, int):
                raise ValueError(
                    "kv_cache_config.max_attention_window must contain only integers"
                )
            if i <= 0 and i not in [LinearCacheType.RECURRENT_STATES.value]:
                raise ValueError(
                    "kv_cache_config.max_attention_window values must be positive or LinearCacheType.RECURRENT_STATES.value"
                )
        return v

    @field_validator('max_util_for_resume')
    @classmethod
    def validate_max_util_for_resume(cls, v: float):
        if not 0 <= v <= 1:
            raise ValueError(
                "kv_cache_config.max_util_for_resume must be between 0 and 1")
        return v

    @field_validator('pool_ratio')
    @classmethod
    def validate_pool_ratio(cls, v: Optional[List[float]]):
        if v is None:
            return v
        if any(r <= 0 for r in v):
            raise ValueError(
                "kv_cache_config.pool_ratio values must be positive")
        if not math.isclose(sum(v), 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(
                "kv_cache_config.pool_ratio values must sum to 1.0")
        return v


@PybindMirror.mirror_pybind_fields(_ExtendedRuntimePerfKnobConfig)
class ExtendedRuntimePerfKnobConfig(StrictBaseModel, PybindMirror):
    """Configuration for extended runtime performance knobs."""

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
    """Configuration for the cache transceiver."""

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
        default=60000,
        description=
        "Timeout in milliseconds for KV cache transfer. Requests exceeding this timeout will be cancelled."
    )

    kv_transfer_sender_future_timeout_ms: Optional[PositiveInt] = Field(
        default=1000,
        description=
        "Timeout in milliseconds to wait for the sender future to be ready when scheduled batch size is 0. This allows the request to be eventually cancelled by the user or because of kv_transfer_timeout_ms"
    )

    kv_transfer_poll_interval_ms: Optional[PositiveInt] = Field(
        default=5000,
        description=
        "Bounded wait interval in milliseconds for polling KV transfer "
        "progress when active transfers block disaggregated admission.")

    kv_cache_bounce_size_mb: int = Field(
        default=0,
        ge=0,
        description=
        "Per-region size in MiB of the native-disagg KV-cache bounce buffer (one for send, one for recv). Bounce coalesces a request's scattered per-block KV into one contiguous fabric-VMM buffer and issues a single multi-rail NIXL write. The size doubles as the on/off switch: 0 (default) keeps the per-block path, >0 enables bounce at that capacity. Only used by the Python (v2) transceiver."
    )

    def _to_pybind(self):
        return _CacheTransceiverConfig(
            backend=_CacheTransceiverBackendType.from_string(self.backend),
            max_tokens_in_buffer=self.max_tokens_in_buffer,
            kv_transfer_timeout_ms=self.kv_transfer_timeout_ms,
            kv_transfer_sender_future_timeout_ms=self.
            kv_transfer_sender_future_timeout_ms,
            kv_transfer_poll_interval_ms=self.kv_transfer_poll_interval_ms)


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


class DwdpConfig(StrictBaseModel):
    """Configuration for Distributed Weight Data Parallelism (DWDP).

    DWDP accelerates the context (prefill) phase of disaggregated MoE serving
    by combining data parallelism with NVLink-based expert weight sharing.
    Each worker holds a subset of experts locally and asynchronously prefetches
    the remaining experts from peer workers via CUDA VMM + MNNVL fabric
    handles (composite virtual-address layout with zero-copy local mapping and
    double-buffered P2P remote regions), enabling fully asynchronous execution
    across ranks without synchronization barriers.

    Currently supported with the CuteDSL MoE backend and NVFP4 quantization
    on MNNVL-connected multi-GPU systems (e.g., GB200).
    """
    dwdp_size: int = Field(default=1,
                           description="The number of GPUs per DWDP group.")
    num_groups: int = Field(
        default=1,
        description=
        "The number of DWDP groups. Total workers = num_groups * dwdp_size.")
    num_experts_per_worker: int = Field(
        default=0, description="The number of experts per worker.")
    num_prefetch_experts: int = Field(
        default=0, description="The number of prefetch experts per worker.")
    contention_opt: bool = Field(
        default=False,
        description=
        "Enable limited-contention prefetch optimization. Uses batched "
        "memcpy with round-robin slice ordering across peers and a 2 MiB "
        "slice granularity to reduce NVLink contention.")


class BaseLlmArgs(StrictBaseModel):
    """Base class for both TorchLlmArgs and TrtLlmArgs. It contains all the arguments that are common to both."""
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

    post_processor_hook: Optional[str] = Field(
        default=None,
        description=
        "Python import path of a user post-processing hook applied after "
        "detokenization and before the per-endpoint response formatter (e.g. "
        "'my_pkg.guardrail.MyPostProcessorHook'). The class must be importable and "
        "picklable, take no constructor arguments, and be callable as "
        "'__call__(chunk) -> verdict' (see tensorrt_llm.executor.postprocessor_hook). "
        "It runs once per output, per streaming chunk, and may rewrite, "
        "suppress, or terminate the output; it owns its own per-request state.",
        status="prototype")

    skip_tokenizer_init: bool = Field(
        default=False,
        description="Whether to skip the tokenizer initialization.")

    trust_remote_code: bool = Field(
        default=False, description="Whether to trust the remote code.")

    tensor_parallel_size: int = Field(default=1,
                                      description="The tensor parallel size.")

    dtype: str = Field(
        default="auto",
        description="The data type to use for the model. When 'auto' "
        "(default), it is read from the HF config.json ('dtype', or the "
        "deprecated 'torch_dtype'); for composite/VLM configs it falls "
        "back to the nested text_config.dtype. Defaults to bfloat16 if "
        "none is found, and is overridden to float16 on GPUs with compute "
        "capability < 8.0 (pre-Ampere).",
        telemetry=TelemetryField.categorical("auto", "float16", "bfloat16",
                                             "float32"))

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
        description="The cluster parallel size for MoE model's expert weights.",
        status="deprecated")

    moe_tensor_parallel_size: Optional[int] = Field(
        default=None,
        description="The tensor parallel size for MoE model's expert weights.")

    moe_expert_parallel_size: Optional[int] = Field(
        default=None,
        description="The expert parallel size for MoE model's expert weights.")

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
        ge=-1,
        description=
        "The maximum number of iterations for iter stats. Set to -1 to keep all iteration stats. "
        "Set to 0 to disable iteration stats in the TensorRT executor.",
        status="prototype")

    request_stats_max_iterations: Optional[int] = Field(
        default=None,
        ge=-1,
        description=
        "The maximum number of iterations for request stats. Set to -1 to keep all request stats. "
        "Set to 0 to disable request stats.",
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

    # KV cache compression config (separate from sparse attention: changes which
    # KV is stored, not the attention computation)
    kv_cache_compression_config: Optional[KvCacheCompressionConfig] = Field(
        default=None,
        description="KV-cache compression config; None disables compression.",
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
        status="prototype",
        telemetry=TelemetryField.categorical('auto', 'deepseek-r1', 'laguna',
                                             'qwen3', 'qwen3_5', 'minimax_m2',
                                             'minimax_m2_append_think',
                                             'nano-v3', 'gemma4', 'kimi_k2',
                                             'kimi_k25'))

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
        telemetry=TelemetryField.categorical('pytorch', 'tensorrt',
                                             '_autodeploy'))

    return_perf_metrics: bool = Field(default=False,
                                      description="Return perf metrics.",
                                      status="prototype")

    perf_metrics_max_requests: NonNegativeInt = Field(
        default=0,
        description=
        "The maximum number of requests for perf metrics. Must also set return_perf_metrics to true to get perf metrics.",
        status="prototype")

    prometheus_metrics_config: Optional[PrometheusMetricsConfig] = Field(
        default=None,
        description="Configuration for Prometheus metrics collection, including "
        "custom histogram bucket boundaries.",
        status="prototype")

    enable_energy_metrics: bool = Field(
        default=False,
        description=
        "Enable GPU energy monitoring via NVML. When enabled, the server exposes an /energy_metrics endpoint that reports cumulative GPU energy consumption in joules.",
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

    telemetry_config: TelemetryConfig = Field(
        default_factory=TelemetryConfig,
        description="Telemetry configuration (opt-out, usage context).",
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
            # The post-processing hook is a text-based guardrail
            # and needs detokenized text to inspect; without a tokenizer it could
            # never run, so reject the combination rather than silently disabling
            # the guardrail (mirrors the harmony fail-fast in OpenAIServer).
            if self.post_processor_hook is not None:
                raise ValueError(
                    "post_processor_hook is not supported together with "
                    "skip_tokenizer_init: the post-processing hook operates on "
                    "detokenized text, which is unavailable when the tokenizer "
                    "is skipped.")
            self.tokenizer = None
        elif self.custom_tokenizer:
            # IPC workers receive the tokenizer object that was already loaded
            # in the parent LLM process. Reuse TRT-LLM tokenizer wrappers as-is.
            if isinstance(self.tokenizer, TokenizerBase):
                return self
            # A raw HF tokenizer object would bypass the requested custom
            # wrapper, so keep rejecting that combination.
            if isinstance(self.tokenizer, PreTrainedTokenizerBase):
                raise ValueError(
                    "Cannot use custom_tokenizer when tokenizer is already a tokenizer object. "
                    "Please specify a tokenizer path or leave it as None to load from model path."
                )

            # Resolve short aliases via the module-level TOKENIZER_ALIASES.
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
        default=True,
        description=
        "Fail fast when attention window is too large to fit even a single sequence in the KV cache.",
        status="deprecated")

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
        """Creating a default BuildConfig if none is provided"""
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
                assert self.speculative_config.speculative_model is not None, "EAGLE draft model must be specified."
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
            elif isinstance(self.speculative_config, DFlashDecodingConfig):
                raise ValueError(
                    "speculative_config.decoding_type 'DFlash' is only supported on the PyTorch backend."
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
        """Load the model format, and do the following:

        1. Load the build_config if got an engine.
        2. Load the parallel_config if got a checkpoint.
        """
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
    # Load weights through GPU Memory Service.
    GMS = 3


class ModelExpressConfig(StrictBaseModel):
    """Prototype configuration for ModelExpress (MX) weight transfer."""

    server_url: Optional[str] = Field(
        default=None,
        description="URL of the MX (ModelExpress) server for P2P weight "
        "transfer. When set together with checkpoint_format='MX', enables "
        "GPU-to-GPU weight transfer from a running source instance, bypassing "
        "disk I/O. When the server is unreachable, loading falls back to the "
        "standard HuggingFace checkpoint path.",
        status="prototype",
    )

    server_query_timeout_s: Optional[NonNegativeInt] = Field(
        default=None,
        description="Timeout in seconds for upstream MxLiveWeightLoader source "
        "discovery. When unset, TRT-LLM first probes for existing sources: "
        "no source uses a short 30-second fallback cap, while an existing "
        "source uses modelexpress's default wait for long donor loads.",
        status="prototype")

    preshard_strategy: str = Field(
        default="per_module",
        description="How to inform TRT-LLM that MX-delivered weights are already "
        "TP-sharded for the local rank. Only 'per_module' is supported in "
        "this MX-only PR; 'global' requires LoadFormat.PRESHARDED.",
        status="prototype",
        telemetry=TelemetryField.categorical('per_module'))

    @model_validator(mode="after")
    def validate_preshard_strategy(self) -> 'ModelExpressConfig':
        if self.preshard_strategy not in ("per_module", "global"):
            raise ValueError(
                f"mx_config.preshard_strategy must be 'per_module' or 'global', "
                f"got '{self.preshard_strategy}'.")
        if self.preshard_strategy == "global":
            raise ValueError(
                "mx_config.preshard_strategy='global' requires "
                "LoadFormat.PRESHARDED, which is not yet available in "
                "TRT-LLM main. Use preshard_strategy='per_module' instead.")
        return self


class GmsConfig(StrictBaseModel):
    """Prototype configuration for GPU Memory Service (GMS) weight sharing.

    Composes orthogonally with ``checkpoint_format`` on ``TorchLlmArgs``:
    GMS controls *where* weights live (a node-shared GPU memory pool
    so multiple TRT-LLM instances zero-copy share the same bytes),
    while ``checkpoint_format`` controls *how* they get there (HF disk,
    MX P2P, etc.). Effective only when ``TorchLlmArgs.load_format ==
    LoadFormat.GMS``; setting any non-default field here without that
    load_format triggers a warning via :meth:`TorchLlmArgs.validate_gms_config`.

    Two roles are decided at connect time by the GMS daemon, not by
    config:

    - **RW (writer)**: the first instance loads weights into the pool
      via the normal checkpoint-loader pipeline, then commits them.
    - **RO (reader)**: subsequent instances zero-copy materialize the
      already-committed layout — no disk I/O, no per-instance copies.

    See :class:`tensorrt_llm._torch.memory.gpu_memory_backend.GMSBackend`
    for the integration adapter and the ``LoadFormat.GMS`` branch in
    ``ModelLoader.load`` for orchestration.
    """

    socket_path: Optional[str] = Field(
        default=None,
        description="Unix domain socket path of the GPU Memory Service. "
        "When unset, the GMS library resolves its default per-device path.",
        status="prototype",
    )

    mode: Literal["auto", "rw", "ro"] = Field(
        default="auto",
        description="GMS operating mode: 'auto' requests RW or RO, 'rw' "
        "requires writer mode, and 'ro' requires read-only mode.",
        status="prototype")

    tag: str = Field(
        default="weights",
        description="Tag identifying the model weight set in the GMS memory "
        "pool. Defaults to 'weights' to match the GMS library convention. "
        "The tag must uniquely identify a compatible model/parallelism "
        "layout for a given GMS daemon; reusing a tag for different weights "
        "can make RO readers attach to the wrong pool.",
        status="prototype",
    )

    @model_validator(mode="after")
    def validate_gms_config(self) -> 'GmsConfig':
        """Enforce non-empty :attr:`tag`.

        :attr:`mode` is enforced by its ``Literal`` type annotation —
        Pydantic rejects out-of-set values at field validation, before
        this validator runs, so the JSON schema/docs are self-describing
        (enum) without a duplicated whitelist here.  This validator only
        enforces the non-empty / non-whitespace contract on :attr:`tag`,
        which the type system can't express.

        Raises:
            ValueError: If ``tag`` is empty / whitespace-only.

        Returns:
            ``self`` (Pydantic ``model_validator`` contract).
        """
        if not self.tag or not self.tag.strip():
            raise ValueError(
                f"gms_config.tag must be a non-empty string, got {self.tag!r}.")
        return self


class SamplerType(StrEnum):
    """Enum for sampler type options."""
    TRTLLMSampler = "TRTLLMSampler"
    TorchSampler = "TorchSampler"
    auto = "auto"


class TorchCompileConfig(StrictBaseModel):
    """Configuration for torch.compile."""
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
        "Threshold for Python garbage collection of generation 0 objects. "
        "Lower values trigger more frequent garbage collection.",
        status="beta")

    cuda_graph_config: Optional[CudaGraphConfigType] = Field(
        default_factory=CudaGraphConfig,
        description="CUDA graph config. If true, use CUDA graphs for decoding. \
        CUDA graphs are only created for the batch sizes in cuda_graph_config.batch_sizes, \
        and are enabled for batches that consist of decoding requests *only* \
        (the reason is that it's hard to capture a single graph with prefill requests \
        since the input shapes are a function of the sequence lengths).\
         Note that each CUDA graph can use up to 200 MB of extra memory.",
        status="beta")

    @field_validator('cuda_graph_config', mode='before')
    @classmethod
    def infer_cuda_graph_config_mode(cls, v):
        if isinstance(v, dict) and "mode" not in v:
            encoder_keys = {
                "num_tokens", "max_num_token", "seq_lens", "max_seq_len"
            }
            v = dict(v)
            v["mode"] = "encode" if any(k in v and v[k] not in (None, 0)
                                        for k in encoder_keys) else "decode"
        return v

    multimodal_config: MultimodalConfig = Field(
        default_factory=MultimodalConfig,
        description="Multimodal model configuration.",
        status="prototype")

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

    dwdp_config: Optional[DwdpConfig] = Field(
        default=None,
        description="DWDP (Distributed Weight Data Parallelism) config.",
        status="prototype")

    encoder_max_batch_size: Optional[int] = Field(
        default=None,
        description=(
            "Maximum batch size for the multimodal encoder's AttentionMetadata. "
            "Falls back to `max_batch_size` when unset. This budget is shared "
            "proportionately across all modalities the model encodes, not set "
            "per modality; per-modality knobs may be added later."),
        status="prototype")

    encoder_max_num_tokens: Optional[int] = Field(
        default=None,
        description=(
            "Maximum number of tokens for the multimodal encoder's "
            "AttentionMetadata. Falls back to `max_num_tokens` when unset. This "
            "budget is shared proportionately across all modalities the model "
            "encodes, not set per modality; per-modality knobs may be added "
            "later."),
        status="prototype")

    @field_validator("encoder_max_batch_size", "encoder_max_num_tokens")
    @classmethod
    def validate_encoder_runtime_sizes(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v <= 0:
            raise ValueError("must be a positive integer when set")
        return v

    attn_backend: str = Field(
        default='TRTLLM',
        description="Attention backend to use.",
        status="beta",
        # Recognized values mirror get_attention_backend dispatch in
        # tensorrt_llm/_torch/attention_backend/utils.py.
        telemetry=TelemetryField.categorical("VANILLA", "TRTLLM", "FLASHINFER",
                                             "FLASHINFER_STAR_ATTENTION"))

    sampler_type: Union[str, SamplerType] = Field(
        default=SamplerType.auto,
        description=
        "The type of sampler to use. Options are TRTLLMSampler, TorchSampler or auto. Defaults to auto, which will use TorchSampler. "
        "TRTLLMSampler is deprecated and will be removed in release 1.4.",
        status="deprecated",
        deprecated=
        "This parameter will be removed in release 1.4. TorchSampler will be the default sampler.",
        telemetry=TelemetryField.categorical('TRTLLMSampler', 'TorchSampler',
                                             'auto'))

    sampler_force_async_worker: bool = Field(
        default=False,
        description="Force usage of the async worker in the sampler for D2H "
        "copies, even if confidential compute is not active. Normally, the "
        "async worker should only be used when confidential compute is active. "
        "This argument is provided to enable it for testing purposes, "
        "irrespective of confidential compute state.",
        status="prototype")

    enable_speculative_beam_history_d2h: bool = Field(
        default=False,
        description="Opt-in beam-search optimization: skip per-step "
        "beam-history D2H copies on likely-non-terminal steps via a "
        "host-side predictor and route the remaining copies through a "
        "private side stream. Mispredictions fall back to a synchronous "
        ".cpu(), preserving correctness but breaking overlap on that step. "
        "Incompatible with the async D2H worker "
        "(sampler_force_async_worker=True or confidential compute).",
        status="prototype")

    enable_early_first_token_response: bool = Field(
        default=False,
        description=
        "Under the overlap scheduler, emit the first-token response ahead of "
        "the next sample step to reduce TTFT. No effect when the overlap "
        "scheduler is disabled.",
        status="prototype")

    enable_low_latency_host_dispatch: bool = Field(
        default=False,
        description="Use low-latency spin-wait mode for CUDA host task dispatch "
        "(cudaLaunchHostFunc_v2 with cudaHostTaskSpinWait). "
        "Reduces callback latency at the cost of a CPU core spinning while "
        "waiting for the GPU event. Requires CUDA 13.2+; on older CUDA "
        "versions, falls back to the default blocking mode and logs a "
        "one-time warning.",
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
        "Token accumulation threshold ratio for batch scheduling optimization. If greater than 0, the scheduler will accumulate requests locally until the total token count reaches batch_wait_max_tokens_ratio * max_num_tokens. This mechanism enhances GPU utilization efficiency by ensuring adequate batch sizes. If 0, disables token-based batching delays.",
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
        "How to load the model weights. By default, detect the weight type from the model checkpoint.",
        telemetry=TelemetryField.categorical("auto", "dummy", "vision_only",
                                             "gms"))

    enable_min_latency: bool = Field(
        default=False,
        description=
        "If true, enable min-latency mode. Currently only used for Llama4.",
        status="beta")

    # TODO: make this a per-request parameter
    stream_interval: PositiveInt = Field(
        default=1,
        description=
        "The iteration interval to create responses under the streaming mode. "
        "Set this to a larger value when the batch size is large, which helps reduce the streaming overhead."
    )

    force_dynamic_quantization: bool = Field(
        default=False,
        description="If true, force dynamic quantization. Defaults to False.",
        status="prototype")

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

    mx_config: ModelExpressConfig = Field(
        default_factory=ModelExpressConfig,
        description="ModelExpress (MX) P2P checkpoint loading config.",
        status="prototype",
    )

    gms_config: GmsConfig = Field(
        default_factory=GmsConfig,
        description="GPU Memory Service (GMS) weight sharing config.",
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
        status="prototype")

    encode_only: bool = Field(
        default=False,
        description=
        "Set to True to use the batch-forward encode() path, which runs a "
        "single forward pass and returns the model output directly, bypassing "
        "the scheduler and autoregressive loop. Works for encoder-only "
        "models (BERT, RoBERTa, reward models) and decoder models used in "
        "single-prefill mode (e.g., extracting embeddings). When False "
        "(default), uses the standard generate() path.",
        status="prototype")

    ray_worker_extension_cls: Optional[str] = Field(
        default=None,
        description="The full worker extension class name including module path. "
        "Allows users to extend the functions of the RayGPUWorker class.",
        status="prototype")

    ray_placement_config: Optional[RayPlacementConfig] = Field(
        default=None,
        description=
        "Placement config for RayGPUWorker. Only used with AsyncLLM and orchestrator_type='ray'.",
        exclude=True,
        status="prototype")

    ray_worker_nsight_options: Optional[dict[str, str]] = Field(
        default=None,
        description="Nsight options.",
        status="prototype",
    )

    sleep_config: Optional[SleepConfig] = Field(
        default=None,
        description="Configuration for the LLM sleep feature. "
        "Sleep feature requires extra setup that may slow down model loading. "
        "Only enable it if you intend to use this feature.",
        status="prototype")

    reorder_policy_config: Optional[ReorderRequestPolicyConfig] = Field(
        default=None,
        description="The request reordering policy to use.",
        status="prototype",
    )

    enable_resource_governor: bool = Field(
        default=False,
        description="Enable the resource governor for runtime cache management "
        "operations such as KV cache truncation. This adds a per-iteration "
        "broadcast collective.",
        status="prototype")

    # fp8 cute dsl configs
    use_cute_dsl_blockscaling_mm: bool = Field(
        default=False,
        description="If true, use CuTe DSL fp8 blockscaling mm implementation.",
        status="prototype")
    use_cute_dsl_blockscaling_bmm: bool = Field(
        default=False,
        description="If true, use CuTe DSL fp8 blockscaling bmm implementation.",
        status="prototype")
    # bf16 cute dsl configs
    use_cute_dsl_bf16_bmm: bool = Field(
        default=False,
        description=
        "If true, use CuTe DSL bf16 persistent GEMM for BMM on Blackwell.",
        status="prototype")
    use_cute_dsl_bf16_gemm: bool = Field(
        default=False,
        description=
        "If true, use CuTe DSL bf16 persistent GEMM for Linear layers on Blackwell.",
        status="prototype")

    # PrivateVars
    _quant_config: Optional[QuantConfig] = PrivateAttr(default=None)

    disable_flashinfer_sampling: bool = Field(
        default=False,
        description=
        "Disable the use of FlashInfer.sampling. This option is likely to be removed in the future.",
        status="prototype")

    max_stats_len: int = Field(
        default=1000,
        ge=-1,
        description=
        "The max number of performance statistic entries. Set to -1 to keep all entries. "
        "Set to 0 to use a minimum buffer size of 1.",
        status="prototype",
    )

    @field_validator('max_stats_len')
    @classmethod
    def normalize_max_stats_len(cls, v):
        if v == -1:
            return v
        return max(v, 1)

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

    def get_encoder_runtime_sizes(self) -> Tuple[int, int]:
        """Return encoder runtime batch and token limits.

        Returns `(encoder_max_batch_size, encoder_max_num_tokens)`, falling
        back to the LLM-side `max_batch_size` / `max_num_tokens` when the
        encoder-specific knobs are not set.
        """
        return (
            self.encoder_max_batch_size
            if self.encoder_max_batch_size is not None else self.max_batch_size,
            self.encoder_max_num_tokens
            if self.encoder_max_num_tokens is not None else self.max_num_tokens,
        )

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
        # ``bool`` is a subclass of ``int`` in Python, so without an
        # explicit bool check ``load_format=True/False`` would silently
        # coerce to ``LoadFormat(1) == LoadFormat.DUMMY`` /
        # ``LoadFormat(0) == LoadFormat.AUTO``. Reject early so callers
        # who pass a misread boolean flag get an actionable error
        # instead of a silently wrong checkpoint-load mode.
        if isinstance(v, bool):
            raise ValueError(f"Invalid LoadFormat: {v}")
        if isinstance(v, int):
            return LoadFormat(v)
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
    def validate_encoder_modes(self) -> 'TorchLlmArgs':
        if self.encode_only and self.mm_encoder_only:
            raise ValueError(
                "encode_only and mm_encoder_only are mutually exclusive. "
                "Use encode_only=True for LLM.encode(), or use "
                "MultimodalEncoder/mm_encoder_only for multimodal encoder "
                "execution.")
        return self

    @model_validator(mode="after")
    def validate_encode_only_torch_compile_config(self) -> 'TorchLlmArgs':
        if (self.encode_only and self.torch_compile_config is not None
                and self.torch_compile_config.enable_piecewise_cuda_graph):
            raise ValueError(
                "encode_only does not support piecewise CUDA graph in "
                "TorchCompileConfig. Use cuda_graph_config for encoder CUDA "
                "graphs or disable enable_piecewise_cuda_graph.")
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

            if self.speculative_config.use_rejection_sampling and not isinstance(
                    self.speculative_config, Eagle3DecodingConfig):
                # Rejection sampling is only wired up for Eagle3 one-model paths.
                # Silently fall back for other spec types so the new default
                # (True) does not break them.
                # TODO: extend rejection sampling to the remaining speculative
                # decoding paths (MTP / DraftTarget / PARD / DFlash /
                # SaveHiddenStates / SA) and unify the dispatch in SpecMetadata
                # so new spec algorithms get rejection sampling for free; once
                # all paths are covered this whitelist guard can be removed.
                self.speculative_config.use_rejection_sampling = False

            if isinstance(self.speculative_config, PARDDecodingConfig):
                assert self.speculative_config.max_draft_len > 0, "PARD max_draft_len must be > 0"

            if isinstance(self.speculative_config, DFlashDecodingConfig):
                assert self.speculative_config.max_draft_len > 0, "DFlash max_draft_len must be > 0"
                # Resolve target_layer_ids and mask_token_id from draft model config if not set
                needs_target_layer_ids = self.speculative_config.target_layer_ids is None
                needs_mask_token_id = self.speculative_config.mask_token_id is None
                if (needs_target_layer_ids or needs_mask_token_id
                    ) and self.speculative_config.speculative_model is not None:
                    draft_config_path = os.path.join(
                        self.speculative_config.speculative_model,
                        "config.json")
                    if os.path.exists(draft_config_path):
                        with open(draft_config_path) as f:
                            draft_cfg = json.load(f)
                        dflash_cfg = draft_cfg.get("dflash_config", {})
                        if needs_target_layer_ids:
                            layer_ids = dflash_cfg.get("target_layer_ids")
                            if layer_ids is not None:
                                self.speculative_config.target_layer_ids = layer_ids
                        if needs_mask_token_id:
                            mask_id = dflash_cfg.get("mask_token_id")
                            if mask_id is not None:
                                self.speculative_config.mask_token_id = mask_id

            if isinstance(self.speculative_config, SADecodingConfig):
                pool_size = self.speculative_config.global_pool_size
                if pool_size is not None and self.max_batch_size is not None:
                    if pool_size < self.max_batch_size:
                        raise ValueError(
                            f"global_pool_size ({pool_size}) must be >= "
                            f"max_batch_size ({self.max_batch_size})")

            if isinstance(self.speculative_config,
                          SaveHiddenStatesDecodingConfig):
                logger.warning(
                    "SaveHiddenStatesDecodingConfig is active, setting max_batch_size to 1, disabling overlap scheduler, and setting cuda_graph_config to None"
                )
                self.max_batch_size = 1
                self.disable_overlap_scheduler = True
                self.cuda_graph_config = None
                self.speculative_config.max_draft_len = 1
            elif isinstance(self.speculative_config, DraftTargetDecodingConfig):
                assert self.speculative_config.max_draft_len > 0
                assert self.speculative_config.speculative_model is not None, "Draft model must be specified."
                if self.backend == "_autodeploy":
                    self.speculative_config._draft_target_one_model = False

            # If speculative_config.draft_len_schedule is provided, cuda_graph_config.enable_padding is automatically set to True.
            # Also we add the draft_len_schedule keys into batch_sizes for better cuda graph coverage in dynamic draft length.
            if (self.cuda_graph_config is not None
                    and self.speculative_config.draft_len_schedule is not None):
                if not self.cuda_graph_config.enable_padding:
                    logger.info(
                        "Automatically enabling cuda_graph_config.enable_padding "
                        "because draft_len_schedule is set.")
                    self.cuda_graph_config.enable_padding = True
                self.cuda_graph_config.batch_sizes = CudaGraphConfig._merge_schedule_keys(
                    self.cuda_graph_config.batch_sizes,
                    self.speculative_config.draft_len_schedule)
                logger.debug(
                    f"draft_len_schedule keys added to cuda_graph_config.batch_sizes, current batch_sizes: {self.cuda_graph_config.batch_sizes}"
                )

        else:
            self.decoding_config = None

        return self

    @model_validator(mode="after")
    def validate_early_first_token_response(self) -> 'TorchLlmArgs':
        if not self.enable_early_first_token_response:
            return self
        if self.disable_overlap_scheduler:
            logger.warning(
                "enable_early_first_token_response is relevant only when the "
                "overlap scheduler is enabled; disabling it because "
                "disable_overlap_scheduler is True.")
            self.enable_early_first_token_response = False
            return self
        is_disagg = (self.cache_transceiver_config is not None
                     and self.cache_transceiver_config.backend is not None)
        if is_disagg:
            logger.warning(
                "enable_early_first_token_response is supported only for "
                "aggregated workloads; disabling it because "
                "cache_transceiver_config is configured.")
            self.enable_early_first_token_response = False
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
    def validate_mx_config(self) -> 'TorchLlmArgs':
        # When MX is the active checkpoint format and the user did not
        # explicitly set ``mx_config.server_url``, honor the ``MODEL_EXPRESS_URL``
        # env var that the upstream ``modelexpress`` library reads. This
        # lets orchestrators configure MX via the environment while keeping
        # the resolved value visible on ``llm_args.mx_config.server_url``.
        if self.checkpoint_format == "MX" and self.mx_config.server_url is None:
            env_url = os.environ.get("MODEL_EXPRESS_URL")
            if env_url:
                logger.info(
                    "mx_config.server_url not set; using MODEL_EXPRESS_URL=%s "
                    "from environment.", env_url)
                self.mx_config.server_url = env_url

        if self.mx_config.server_url is not None and self.checkpoint_format != "MX":
            logger.warning(
                "mx_config.server_url is set but checkpoint_format is '%s', not "
                "'MX'. The MX config will be ignored. Set "
                "checkpoint_format='MX' to enable MX P2P weight transfer.",
                self.checkpoint_format)
        return self

    @model_validator(mode="after")
    def validate_gms_config(self) -> 'TorchLlmArgs':
        """Warn when GMS settings are provided without enabling GMS load.

        Catches the most common misconfiguration: a user customizes
        :attr:`gms_config` (e.g. sets a custom ``socket_path`` or
        ``mode``) but forgets to set ``load_format='GMS'``, in which
        case the entire GMS config is silently ignored. We emit a
        warning so the user notices at config time instead of
        debugging "why are my workers not zero-copy sharing weights?"
        afterwards.

        Detection is by deviation from defaults: any of
        ``socket_path != None``, ``mode != 'auto'``, or
        ``tag != 'weights'`` triggers the warning when
        ``load_format != LoadFormat.GMS``. This is intentionally a
        warning (not an error) so callers that pre-populate config
        objects from templates aren't broken.

        Returns:
            ``self`` (Pydantic ``model_validator`` contract).
        """
        gms_config_is_non_default = (self.gms_config.socket_path is not None
                                     or self.gms_config.mode != "auto"
                                     or self.gms_config.tag != "weights")
        if gms_config_is_non_default and self.load_format != LoadFormat.GMS:
            logger.warning(
                "gms_config is set but load_format is '%s', not 'GMS'. "
                "The GMS config will be ignored. Set load_format='GMS' to "
                "enable GPU Memory Service.", self.load_format.name)
        return self

    @model_validator(mode="after")
    def validate_gms_moe_compat(self) -> 'TorchLlmArgs':
        """Reject ``LoadFormat.GMS`` combined with a MoE load balancer.

        The ``MoeLoadBalancer``'s ``register_weight_slots_after_to_cuda``
        and ``finalize_model`` run AFTER the GMS RW pool is closed and
        ``finalize_write`` has committed, so any CUDA allocations they
        make land in non-GMS memory and are NOT part of the committed
        layout that RO peers receive. The result is "wrong inference,
        no error" on RO peers (broken MoE routing state). Failing at
        config-validation time is strictly better than that silent
        miscompute.

        The fix for this gap (running the MoE finalize work INSIDE
        ``mem_pool_scope`` and BEFORE ``finalize_write`` so MoE
        allocations are part of the committed layout) is tracked as
        the (MoE, GMS) follow-up; see ``model_loader.py``'s
        ``TODO(GMS-MOE-LB)`` comment.

        Returns:
            ``self`` (Pydantic ``model_validator`` contract).

        Raises:
            ValueError: When ``load_format == LoadFormat.GMS`` and
                ``moe_config.load_balancer`` is set.
        """
        if (self.load_format == LoadFormat.GMS and self.moe_config is not None
                and self.moe_config.load_balancer is not None):
            raise ValueError(
                "LoadFormat.GMS is incompatible with moe_config.load_balancer "
                "in this PR. The MoE load balancer's "
                "register_weight_slots_after_to_cuda and finalize_model run "
                "after the GMS pool closes and finalize_write commits, so "
                "their allocations land outside the committed layout. RO "
                "peers would receive a broken MoE routing state. Either "
                "disable moe_config.load_balancer or use LoadFormat.AUTO. "
                "Tracked as the (MoE, GMS) follow-up at "
                "tensorrt_llm/_torch/pyexecutor/model_loader.py "
                "(see TODO(GMS-MOE-LB)).")
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
        elif self.kv_cache_config.dtype == 'nvfp4':
            self.quant_config.kv_cache_quant_algo = QuantAlgo.NVFP4
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

            if not field_info:
                continue

            status = _get_trtllm_json_schema_extra(field_info).get(
                'status', None)

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
                "ray_worker_extension_cls is only supported with orchestrator_type='ray'"
            )
        return self

    @model_validator(mode='after')
    def validate_ray_placement_config(self) -> 'TorchLlmArgs':
        if self.ray_placement_config is not None and self.orchestrator_type != "ray":
            raise ValueError(
                "ray_placement_config is only supported with orchestrator_type='ray'"
            )
        return self

    @model_validator(mode='after')
    def validate_cute_dsl_bf16(self) -> 'TorchLlmArgs':
        if (not (self.use_cute_dsl_bf16_bmm and self.use_cute_dsl_bf16_gemm)
                and self.pipeline_parallel_size > 1 and is_sm_100f()):
            logger.info("Automatically enabling CuTe DSL BF16 BMM and GEMM for "
                        "SM100/SM103 PP.")
            self.use_cute_dsl_bf16_bmm = True
            self.use_cute_dsl_bf16_gemm = True

        if self.use_cute_dsl_bf16_bmm or self.use_cute_dsl_bf16_gemm:
            major, minor = torch.cuda.get_device_capability()
            sm = major * 10 + minor
            if sm < 100:
                raise ValueError(
                    f"use_cute_dsl_bf16_bmm and use_cute_dsl_bf16_gemm are only "
                    f"supported on Blackwell (sm >= 100), but current device has "
                    f"sm {sm}.")
        return self

    @model_validator(mode='after')
    def validate_speculative_beam_history_d2h(self) -> 'TorchLlmArgs':
        if (self.enable_speculative_beam_history_d2h
                and self.sampler_force_async_worker):
            raise ValueError(
                "enable_speculative_beam_history_d2h is incompatible with "
                "sampler_force_async_worker=True; the speculative path "
                "bypasses the sampler's async D2H worker.")
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
        extra_llm_api_options: Optional[str] = None,
        explicit_cli_keys: Optional[Set[str]] = None) -> Dict:
    """Merge YAML overrides into a CLI-derived llm_args dict.

    If `explicit_cli_keys` is provided, those CLI flag names override any
    conflicting YAML values. CLI flags whose name does not match the
    LlmArgs field name (e.g. `--free_gpu_memory_fraction` constructs
    `kv_cache_config.free_gpu_memory_fraction`) are mapped to the nested
    field they target.

    If `explicit_cli_keys` is None, YAML wins on conflicts.
    """
    # CLI scalar -> nested KvCacheConfig field. Callers add the CLI scalar
    # name to `explicit_cli_keys` to make it win over YAML's same-named
    # field inside `kv_cache_config:`.
    cli_to_kv_cache_field = {
        "free_gpu_memory_fraction": "free_gpu_memory_fraction",
        "kv_cache_dtype": "dtype",
        "enable_block_reuse": "enable_block_reuse",
    }
    # Scalars that live both at the top level of LlmArgs and inside
    # `build_config`. The build_config patch propagates the winning source
    # to the nested location.
    build_config_dual_loc_keys = (
        "max_batch_size",
        "max_num_tokens",
        "max_beam_width",
        "max_seq_len",
    )

    explicit_cli_keys = explicit_cli_keys or set()

    if 'hf_revision' in llm_args_dict:
        llm_args_dict.setdefault('revision', llm_args_dict.pop('hf_revision'))

    # Deep merge kv_cache_config to prevent partial YAML kv_cache_config from replacing the complete kv_cache_config
    if 'kv_cache_config' in llm_args and 'kv_cache_config' in llm_args_dict:
        base_kv_config = llm_args['kv_cache_config']
        if isinstance(base_kv_config, KvCacheConfig):
            base_kv_config = base_kv_config.model_dump(exclude_unset=True)
        merged = base_kv_config | llm_args_dict['kv_cache_config']
        for cli_name, kv_field in cli_to_kv_cache_field.items():
            if cli_name in explicit_cli_keys and kv_field in base_kv_config:
                merged[kv_field] = base_kv_config[kv_field]
        llm_args_dict['kv_cache_config'] = merged

    # Deep merge telemetry_config: YAML can override fields like `disabled`,
    # but `usage_context` is determined by the CLI entry point and must not
    # be overridden by user config. When `--telemetry/--no-telemetry` was
    # typed explicitly, CLI's `disabled` wins over YAML.
    if 'telemetry_config' in llm_args and 'telemetry_config' in llm_args_dict:
        yaml_tc = llm_args_dict['telemetry_config']
        if not isinstance(yaml_tc, (dict, TelemetryConfig)):
            # YAML value is null / false / etc. — drop it so the CLI default
            # is preserved by the field_mapping coercion step below.
            del llm_args_dict['telemetry_config']
        else:
            base_tc = llm_args['telemetry_config']
            if isinstance(base_tc, TelemetryConfig):
                base_tc = base_tc.model_dump(exclude_unset=True)
            if isinstance(yaml_tc, TelemetryConfig):
                yaml_tc = yaml_tc.model_dump(exclude_unset=True)
            yaml_tc.pop('usage_context', None)
            merged = base_tc | yaml_tc
            if "telemetry" in explicit_cli_keys and 'disabled' in base_tc:
                merged['disabled'] = base_tc['disabled']
            llm_args_dict['telemetry_config'] = merged

    if 'multimodal_config' in llm_args_dict:
        yaml_mm = llm_args_dict['multimodal_config']
        if yaml_mm is None:
            yaml_mm = {}
        if isinstance(yaml_mm, MultimodalConfig):
            yaml_mm = yaml_mm.model_dump(exclude_unset=True)
        if isinstance(yaml_mm, dict):
            base_mm = llm_args.get('multimodal_config', {})
            if isinstance(base_mm, MultimodalConfig):
                base_mm = base_mm.model_dump(exclude_unset=True)
            if not isinstance(base_mm, dict):
                base_mm = {}
            merged = dict(base_mm) | dict(yaml_mm)
            llm_args_dict['multimodal_config'] = merged

    # Drop YAML keys claimed by explicit CLI flags so the outer merge below
    # cannot overwrite them. Warn only when the CLI value actually differs from
    # the YAML value, so users who relied on the previous "YAML wins" behavior
    # are notified that CLI now takes precedence.
    if explicit_cli_keys:
        overridden = sorted(
            k for k in llm_args_dict
            if k in explicit_cli_keys and llm_args.get(k) != llm_args_dict[k])
        if overridden:
            logger.warning(
                f"Explicit CLI flag(s) {overridden} override the value(s) set "
                f"in the YAML config; CLI takes precedence.")
        llm_args_dict = {
            k: v
            for k, v in llm_args_dict.items() if k not in explicit_cli_keys
        }

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
        "reorder_policy_config": ReorderRequestPolicyConfig,
        "kv_cache_config": KvCacheConfig,
        "dwdp_config": DwdpConfig,
        "multimodal_config": MultimodalConfig,
        "telemetry_config": TelemetryConfig,
    }
    for field_name, field_type in field_mapping.items():
        if field_name in llm_args_dict:
            llm_args_dict[field_name] = field_type(**llm_args_dict[field_name])
            if field_name in llm_args:
                extra_llm_str = f" because it's specified in {extra_llm_api_options}" if extra_llm_api_options else ""
                logger.info(f"YAML overrides {field_name}{extra_llm_str}")

    llm_args = llm_args | llm_args_dict

    # build_config only works for TensorRT backend, it will be ignored in PyTorch backend
    if "build_config" in llm_args:
        # Ensure build_config is a BuildConfig object, not a dict
        if isinstance(llm_args["build_config"], dict):
            llm_args["build_config"] = BuildConfig(**llm_args["build_config"])

        # Propagate dual-location scalars into build_config: explicit CLI flag
        # wins; otherwise YAML's top-level scalar; otherwise leave alone. Warn
        # only when the explicit CLI value actually differs from the YAML
        # build_config value being replaced (a genuine override).
        for key in build_config_dual_loc_keys:
            if key in explicit_cli_keys and key in llm_args:
                # Warn only on a genuine override of a YAML build_config value;
                # otherwise just record where the value came from.
                if getattr(llm_args["build_config"], key) != llm_args[key]:
                    logger.warning(
                        f"Explicit CLI flag --{key}={llm_args[key]} overrides "
                        f"the value set in the YAML build_config; CLI takes "
                        f"precedence.")
                else:
                    logger.info(
                        f"build_config.{key} set to {llm_args[key]} from explicit CLI flag"
                    )
                setattr(llm_args["build_config"], key, llm_args[key])
            elif key in llm_args_dict:
                setattr(llm_args["build_config"], key, llm_args_dict[key])
                logger.info(
                    f"build_config.{key} set to {llm_args_dict[key]} from YAML top-level scalar"
                )

    return llm_args


def update_llm_args_with_extra_options(
        llm_args: Dict,
        extra_llm_api_options: str,
        explicit_cli_keys: Optional[Set[str]] = None) -> Dict:
    if extra_llm_api_options is not None:
        with open(extra_llm_api_options, 'r') as f:
            llm_args_dict = yaml.safe_load(f)
            llm_args = update_llm_args_with_extra_dict(
                llm_args,
                llm_args_dict,
                extra_llm_api_options,
                explicit_cli_keys=explicit_cli_keys)
    return llm_args


def get_model_format(model_dir: str,
                     trust_remote_code: bool = False) -> _ModelFormatKind:
    """Get the format of the model."""
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
