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
import copy
import enum
import math
import os
from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict, deque
from contextlib import nullcontext
from dataclasses import dataclass
from typing import (TYPE_CHECKING, Dict, Iterable, List, Optional, Sequence,
                    Set, Tuple, Union)

import torch

import tensorrt_llm
import tensorrt_llm.bindings
from tensorrt_llm._torch.distributed.communicator import Distributed, ReduceOp
from tensorrt_llm._torch.peft.lora.config import LoraConfig
from tensorrt_llm._torch.peft.lora.manager import LoraManager, LoraModelConfig
from tensorrt_llm._utils import (get_size_in_bytes, prefer_pinned,
                                 torch_dtype_to_binding)
from tensorrt_llm.bindings.internal.batch_manager import (
    LinearAttentionMetadata, LinearCacheType)
from tensorrt_llm.bindings.internal.runtime import TaskLayerModuleConfig
from tensorrt_llm.llmapi.llm_args import KvCacheConfig, PeftCacheConfig
from tensorrt_llm.runtime import ModelConfig as ModelConfigPython
from tensorrt_llm.runtime.kv_cache_hash import (KV_CACHE_HASH_ALGO_AUTO,
                                                KV_CACHE_HASH_ALGO_V1)

# isort: off
# isort: on
from tensorrt_llm.sampling_params import SamplingParams

from ..._utils import (binding_to_str_dtype, binding_to_torch_dtype, mpi_rank,
                       nvtx_range)
from ...logger import logger
from ...mapping import Mapping
from .config_utils import uses_vswa_kv_cache_layout
from .connectors.kv_cache_connector import KvCacheConnectorManager
from .llm_request import LlmRequest, LlmRequestState, get_draft_token_length
from .scheduler import ScheduledRequests

BufferManagerCpp = tensorrt_llm.bindings.internal.runtime.BufferManager
KVCacheManagerCpp = tensorrt_llm.bindings.internal.batch_manager.KVCacheManager
PoolConfigurationCpp = tensorrt_llm.bindings.internal.batch_manager.PoolConfiguration
CacheTypeCpp = tensorrt_llm.bindings.internal.batch_manager.CacheType
ModelConfigCpp = tensorrt_llm.bindings.ModelConfig
DataType = tensorrt_llm.bindings.DataType
KVCacheEventManagerCpp = tensorrt_llm.bindings.internal.batch_manager.KVCacheEventManager
RequestList = list[LlmRequest]
PeftCacheManagerCpp = tensorrt_llm.bindings.internal.batch_manager.PeftCacheManager
WorldConfig = tensorrt_llm.bindings.WorldConfig

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention.backends.interface import \
        AttentionMetadata
    from tensorrt_llm.llmapi.llm_args import (DecodingBaseConfig,
                                              KvCacheCompressionConfig)

    from .kv_cache_manager_v2 import KVCacheManagerV2

BlocksPerWindow = Dict[int, Tuple[
    int,
    int]]  # window_size -> (blocks_in_primary_pool, blocks_in_secondary_pool)


def _clamp_max_attention_window_vec(
    max_attention_window_vec: Optional[Sequence[int]],
    max_seq_len: int,
) -> List[int]:
    """Copy and clamp a window pattern without mutating it."""
    windows = ([max_seq_len] if max_attention_window_vec is None else
               list(max_attention_window_vec))
    return [min(max_seq_len, window) for window in windows]


def _project_max_attention_window_vec(
    max_attention_window_vec: Sequence[int],
    pp_layers: Sequence[int],
) -> List[int]:
    """Project a global window vector into cache-local layer order.

    Window vectors are repeating patterns anchored at global model layer zero.
    The projected result has one entry for every PP-local cache layer, in local
    layer order.
    """
    pattern_len = len(max_attention_window_vec)
    return [
        max_attention_window_vec[layer_idx % pattern_len]
        for layer_idx in pp_layers
    ]


def _get_minimum_blocks_per_window(
    local_blocks_per_window: BlocksPerWindow,
    rank_blocks_per_window: Sequence[BlocksPerWindow],
    rank_attention_window_sets: Sequence[Set[int]],
) -> BlocksPerWindow:
    """Conservatively align local pool capacities across distributed ranks."""
    if any(window_set != rank_attention_window_sets[0]
           for window_set in rank_attention_window_sets[1:]):
        raise RuntimeError(
            "Asymmetrical pipeline parallelism is not supported by KV cache "
            "manager V1: ranks host different attention window sets: "
            f"{rank_attention_window_sets}")

    reduced_blocks_per_window = {}
    for window_size in local_blocks_per_window:
        # Attention window sets were validated above. Recurrent-state pools
        # use different units and may be absent on some PP stages, so both
        # kinds of pool are reduced only against their matching key.
        candidate_blocks = [
            rank_blocks[window_size] for rank_blocks in rank_blocks_per_window
            if window_size in rank_blocks
        ]
        assert candidate_blocks
        reduced_blocks_per_window[window_size] = (
            min(blocks[0] for blocks in candidate_blocks),
            min(blocks[1] for blocks in candidate_blocks),
        )
    return reduced_blocks_per_window


@dataclass
class PoolConfiguration:
    """Configuration of a single KV pool.

    A pool is uniquely described by its attention ``window_size``, the
    ``head_dim`` of the layers it serves, and the cache element ``dtype``.
    A KVCacheManager is constructed from a ``list[PoolConfiguration]`` --
    one entry per pool the manager hosts.  Multiple entries with the same
    ``window_size`` are legal and reserved for future multi-pool-per-window
    cases (e.g. mixed head_dim within a single window).
    """
    window_size: int
    head_dim: int
    dtype: "DataType"


class ResourceManagerType(enum.Enum):
    KV_CACHE_MANAGER = "KV_CACHE_MANAGER"
    DRAFT_KV_CACHE_MANAGER = "DRAFT_KV_CACHE_MANAGER"
    CROSS_KV_CACHE_MANAGER = "CROSS_KV_CACHE_MANAGER"
    PEFT_CACHE_MANAGER = "PEFT_CACHE_MANAGER"
    SEQ_SLOT_MANAGER = "SEQ_SLOT_MANAGER"
    SPEC_RESOURCE_MANAGER = "SPEC_RESOURCE_MANAGER"
    KV_CACHE_COMPRESSION_MANAGER = "KV_CACHE_COMPRESSION_MANAGER"


def compute_page_count(token_count: int, tokens_per_page: int) -> int:
    return (token_count + tokens_per_page) // tokens_per_page


def _warn_if_unsupported_v1_kv_cache_event_hash_algo(hash_algo: str) -> None:
    if hash_algo in (KV_CACHE_HASH_ALGO_AUTO, KV_CACHE_HASH_ALGO_V1):
        return
    logger.warning(
        f"KVCacheManager only supports kv_cache_event_hash_algo={KV_CACHE_HASH_ALGO_V1}; "
        f"requested {hash_algo}. The V1 event manager will emit {KV_CACHE_HASH_ALGO_V1} "
        "event hashes.")


def _merge_kv_cache_pool_pointers(
    kv_cache_pool_pointers: torch.Tensor,
    block_scale_pool_pointers: torch.Tensor,
    layer_to_pool_mapping: torch.Tensor,
    layer_pool_dtypes: Sequence["DataType"],
) -> torch.Tensor:
    """Align compact scale rows with data rows in C++ physical-pool order."""
    dtype_by_physical_pool = dict(
        zip(layer_to_pool_mapping[:, 0].tolist(), layer_pool_dtypes))
    fp4_pool_indices = [
        compact_pool_idx for compact_pool_idx, physical_pool_idx in enumerate(
            sorted(dtype_by_physical_pool))
        if dtype_by_physical_pool[physical_pool_idx] == DataType.NVFP4
    ]

    aligned_scale_pool_pointers = torch.zeros_like(kv_cache_pool_pointers)
    aligned_scale_pool_pointers[fp4_pool_indices] = block_scale_pool_pointers
    return torch.stack([kv_cache_pool_pointers, aligned_scale_pool_pointers],
                       dim=-1)


class BaseResourceManager(ABC):

    @abstractmethod
    def get_max_resource_count(self) -> int:
        """Return the maximum number of real requests this manager can admit."""
        raise NotImplementedError

    @abstractmethod
    def get_needed_resource_to_completion(self, request: LlmRequest) -> int:
        raise NotImplementedError

    def get_request_kv_block_budget(
            self, request: LlmRequest) -> Optional[Tuple[int, int]]:
        """Return required and capacity KV block counts for admission.

        ``None`` indicates that this resource manager does not support the
        request-level KV block feasibility check. Opting in is deliberate:
        a manager must only return a pair once it is known that the request
        cannot possibly be served when ``required > capacity``, since the
        executor rejects such requests outright.
        """
        return None

    def kv_block_budget_applies(self) -> bool:
        """Whether ``get_request_kv_block_budget`` reports a budget at all.

        Request-independent, so startup can ask before any request exists --
        see ``PyExecutor._warn_if_kv_block_budget_unchecked``, which reports
        the pools that beam search runs against unchecked. A manager that
        opts into the check must override this too, or it will be reported as
        unchecked while in fact rejecting requests.
        """
        return False

    def add_dummy_requests(self, request_ids: List[int]):
        pass

    def prepare_resources(self, scheduled_batch: ScheduledRequests):
        pass

    def update_resources(self, scheduled_batch: ScheduledRequests):
        pass

    def free_resources(self, request: LlmRequest):
        pass

    def shutdown(self):
        pass


def get_pp_layers(
    num_layers: int,
    mapping: Mapping,
    spec_config: Optional["DecodingBaseConfig"] = None,
    layer_mask: Optional[List[bool]] = None,
) -> Tuple[List[int], int]:
    from ..speculative.utils import get_num_spec_layers

    total_num_layers = num_layers
    if layer_mask is not None:
        assert sum(layer_mask) == num_layers, (
            f"The number of enabled layers in layer_mask ({sum(layer_mask)}) "
            f"must match the number of layers ({num_layers}) "
            f"in KV cache manager, but got layer_mask: {layer_mask}")
        total_num_layers = len(layer_mask)
    # When layer_mask extends beyond pp_partition coverage (e.g., MTP draft
    # layers appended after target hidden layers), compute pp_layers for the
    # base layers, then assign extra layers to the last PP rank.
    base_num_layers = total_num_layers
    if (layer_mask is not None and mapping.pp_partition is not None
            and total_num_layers > sum(mapping.pp_partition)):
        base_num_layers = sum(mapping.pp_partition)
    pp_layers = mapping.pp_layers(base_num_layers)
    if base_num_layers < total_num_layers and mapping.is_last_pp_rank():
        pp_layers.extend(range(base_num_layers, total_num_layers))
    if layer_mask is not None:
        pp_layers = [i for i in pp_layers if layer_mask[i]]
    # Only add speculative layers when layer_mask is not provided.
    # When layer_mask is provided, the caller explicitly controls which layers
    # to include, so we should not add extra layers automatically.
    if spec_config is not None and layer_mask is None:
        num_spec_layers = get_num_spec_layers(spec_config)
        total_num_layers += num_spec_layers
        if mapping.is_last_pp_rank():
            pp_layers.extend(
                range(total_num_layers - num_spec_layers, total_num_layers))
    if len(pp_layers) == 0:
        # Don't support empty KV cache for now, provide at least 1 layer
        pp_layers.append(0)
    return pp_layers, total_num_layers


def request_context(is_draft: bool, scheduled_requests: ScheduledRequests):
    # The non-draft scope has nothing to do, and this runs on the executor's
    # per-iteration path, so return before executing the class statement below.
    if not is_draft:
        return nullcontext()

    class RequestContext:

        def __init__(self, is_draft: bool,
                     scheduled_requests: ScheduledRequests):
            self.is_draft = is_draft
            self.scheduled_requests = scheduled_requests

        def __enter__(self):
            if not self.is_draft:
                return

            for req in self.scheduled_requests.all_requests():
                req.use_draft_model = True

        def __exit__(self, exc_type, exc_val, exc_tb):
            if not self.is_draft:
                return

            # Clean up the state
            for req in self.scheduled_requests.all_requests():
                req.use_draft_model = False

    return RequestContext(is_draft, scheduled_requests)


# M-RoPE (Qwen2-VL/Qwen3-VL) splits positions across 3 axes: temporal/height/width.
_MROPE_NUM_AXES = 3


def _make_warmup_mrope_position_ids(token_num: int) -> torch.Tensor:
    """Build (_MROPE_NUM_AXES, 1, token_num) mrope_position_ids for warmup."""
    return (torch.arange(0, token_num, dtype=torch.int32,
                         device="cuda").expand(_MROPE_NUM_AXES, 1, -1).clone())


def _populate_dummy_mrope_config(req: LlmRequest, token_num: int,
                                 is_gen: bool) -> None:
    """Attach a dummy mrope_config to a warmup request's py_multimodal_data.

    Used by the dummy-request paths in both KVCacheManager and KVCacheManagerV2
    to satisfy models that consume mrope_config (e.g. Qwen2-VL) during warmup.

    TODO(TRTLLM-12045): each model should provide its own warmup dummy_data
    via an input-processor hook — this ad-hoc helper is the interim
    workaround.
    """
    mrope_config: Dict[str, torch.Tensor] = {
        "mrope_position_ids": _make_warmup_mrope_position_ids(token_num),
    }
    if is_gen:
        mrope_config["mrope_position_deltas"] = torch.zeros(
            1, dtype=torch.int32, device="cuda").unsqueeze(0)
    if req.py_multimodal_data is None:
        req.py_multimodal_data = {}
    req.py_multimodal_data["mrope_config"] = mrope_config


class KVCacheManager(BaseResourceManager):

    def __init__(
        self,
        kv_cache_config: KvCacheConfig,
        kv_cache_type: CacheTypeCpp,
        *,
        num_layers: int,
        num_kv_heads: Union[int, List[Optional[int]]],
        head_dim: int,
        tokens_per_block: int,
        # Note that max_seq_len is not necessarily equal to kv_cache_config.num_tokens.
        # It's derived from the model's BuildConfig for consistency with the C++ backend.
        max_seq_len: int,
        max_batch_size: int,
        mapping: Mapping,
        dtype: DataType = DataType.HALF,
        spec_config: Optional["DecodingBaseConfig"] = None,
        layer_mask: Optional[List[bool]] = None,
        max_num_tokens: int = 8192,
        model_config: Optional[ModelConfigCpp] = None,
        max_beam_width: int = 1,
        is_draft: bool = False,
        kv_connector_manager: Optional[KvCacheConnectorManager] = None,
        enable_indexer_k_cache: bool = False,
        indexer_k_cache_quant_block_size: int = 128,
        indexer_k_cache_index_head_dim: int = 0,
        indexer_k_cache_use_fp4: bool = False,
        # Boolean array indicating whether each layer has an indexer K cache.
        indexer_k_cache_layer_mask: Optional[List[bool]] = None,
        is_estimating_kv_cache: bool = False,
        execution_stream: Optional[torch.cuda.Stream] = None,
        linear_attention_metadata: Optional[LinearAttentionMetadata] = None,
        # Per-pool configuration list forwarded to the C++ ctor.  One entry
        # per pool the manager will host; each entry pins (window_size,
        # head_dim, dtype) for that pool.  None / empty = uniform shape
        # across all windows (default behavior); a single KVCacheManager can
        # host pools with mixed shapes when a model has heterogeneous
        # attention types (e.g. Gemma4 SWA head_dim=256 + full-attention
        # head_dim=512).
        pool_configurations: Optional[List[PoolConfiguration]] = None,
        enable_chunked_prefill: bool = False,
        **kwargs,
    ) -> None:
        self.mapping = mapping
        self.dtype = dtype
        self.kv_cache_type = kv_cache_type
        # Consumed by the disaggregation page-table builder to expose the DSA
        # indexer K cache pool as a REPLICATED pool view.
        self.enable_indexer_k_cache = enable_indexer_k_cache
        self.spec_config = spec_config
        self.pp_layers, self.num_layers = get_pp_layers(
            num_layers,
            mapping,
            spec_config=spec_config,
            layer_mask=layer_mask,
        )
        self.is_draft = is_draft
        # Retained so consumers (e.g. CUDAGraphRunner.preallocate_padding_dummies)
        # can distinguish the throwaway estimation-phase managers from the
        # final ones: the estimation cache is sized with no headroom for
        # retained dummy requests.
        self.is_estimating_kv_cache = is_estimating_kv_cache
        self.num_local_layers = len(self.pp_layers)
        self.layer_offsets = {
            idx: offset
            for offset, idx in enumerate(self.pp_layers)
        }

        if indexer_k_cache_layer_mask is not None:
            assert len(indexer_k_cache_layer_mask) >= max(self.pp_layers) + 1, (
                f"indexer_k_cache_layer_mask covers "
                f"{len(indexer_k_cache_layer_mask)} layers but this rank "
                f"manages global layer {max(self.pp_layers)}")
            self.indexer_k_cache_local_layer_mask: Optional[List[bool]] = [
                bool(indexer_k_cache_layer_mask[i]) for i in self.pp_layers
            ]
        else:
            self.indexer_k_cache_local_layer_mask = None

        self.kv_connector_manager = kv_connector_manager
        # Dummy requests can reserve their V1 sequence before they enter the
        # normal context prepare path. Track that ownership per manager so the
        # same sequence is not registered twice, while a separate draft or
        # cross-cache manager can still prepare its own copy.
        self._preprepared_dummy_request_ids: set[int] = set()

        tp_size = mapping.tp_size
        if mapping.enable_attention_dp:
            tp_size = 1

        if isinstance(num_kv_heads, int):
            self.num_kv_heads_per_layer = [
                (num_kv_heads + tp_size - 1) // tp_size
                for _ in range(self.num_local_layers)
            ]
            self.total_num_kv_heads_per_layer = [
                (num_kv_heads + tp_size - 1) // tp_size
                for _ in range(self.num_layers)
            ]
        else:
            assert len(num_kv_heads) == self.num_layers

            def append_to_kv_heads_per_layer(num_kv_heads_per_layer: List[int],
                                             kv_head: Optional[int]):
                if kv_head is not None:
                    num_kv_heads_per_layer.append(
                        (kv_head + tp_size - 1) // tp_size)
                else:
                    num_kv_heads_per_layer.append(0)

            self.num_kv_heads_per_layer = []
            if self.num_local_layers > 0:
                for i in self.pp_layers:
                    kv_head = num_kv_heads[i]
                    append_to_kv_heads_per_layer(self.num_kv_heads_per_layer,
                                                 kv_head)

            self.total_num_kv_heads_per_layer = []
            for i in range(self.num_layers):
                kv_head = num_kv_heads[i]
                append_to_kv_heads_per_layer(self.total_num_kv_heads_per_layer,
                                             kv_head)

        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        # Per-pool configuration list -- the source of truth for per-pool
        # (window_size, head_dim, dtype).  When non-empty, each pool may
        # have its own shape that differs from the manager-level scalars
        # (e.g. Gemma4 SWA head_dim=256 alongside full-attention head_dim=512).
        # Empty list means uniform shape (every window uses self.head_dim /
        # self.dtype).  Each pool's window_size is remapped after window
        # clamping in _validate_and_adjust_attention_windows; the pool
        # *indices* stay stable across that rewrite.
        self.pool_configurations: List[PoolConfiguration] = (
            list(pool_configurations) if pool_configurations else [])
        # Layer -> pool_idx mapping, built once after max_attention_window_vec
        # is initialized below.  This is the layer-centric replacement for any
        # window-keyed shape dict: multi-pool-per-window is safe because pools
        # are identified by index, not by window.
        self._layer_to_pool_idx: Dict[int, int] = {}
        self.tokens_per_block = tokens_per_block
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size
        self.kv_factor = 1 if kv_cache_type == CacheTypeCpp.SELFKONLY else 2
        # Some speculative decoding methods need to use different kv lengths for the
        # draft/target layers. Add extra tokens to handle this issue.
        # Import here to avoid circular imports
        from ..speculative import get_num_extra_kv_tokens
        self.num_extra_kv_tokens = get_num_extra_kv_tokens(spec_config)
        # Kept so prepare_resources can re-validate the per-step token budget
        # (the forward-pass scratch size enforced in _prepare_tp_inputs).
        self.max_num_tokens = max_num_tokens
        # Whether chunked prefill is enabled for this engine. Gates the shrink
        # path in fit_token_budget: a context request may only be shrunk into a
        # partial chunk when the attention backend is set up for chunked context.
        # Defaults to False (safe: leave the chunk at its scheduled size) and is
        # set to the finalized value by _create_kv_cache_manager.
        self.enable_chunked_prefill = enable_chunked_prefill
        self.event_buffer_max_size = kv_cache_config.event_buffer_max_size
        self.attention_dp_events_gather_period_ms = kv_cache_config.attention_dp_events_gather_period_ms
        self.max_draft_len = spec_config.max_draft_len if spec_config is not None else 0
        self.max_total_draft_tokens = (spec_config.tokens_per_gen_step -
                                       1) if spec_config is not None else 0
        self.linear_attention_metadata = linear_attention_metadata

        # Dynamic-tree draft manager reserves K*max_draft_len KV slots (the draft
        # loop can write that many even if max_total_draft_tokens is smaller).
        # Target manager keeps max_total_draft_tokens exactly.
        self._kv_reserve_draft_tokens = self.max_total_draft_tokens
        if (self.is_draft and spec_config is not None
                and getattr(spec_config, 'use_dynamic_tree', False)
                and getattr(spec_config, 'dynamic_tree_max_topK', 0) > 0):
            draft_loop_tokens = spec_config.dynamic_tree_max_topK * spec_config.max_draft_len
            self._kv_reserve_draft_tokens = max(self.max_total_draft_tokens,
                                                draft_loop_tokens)

        # Resolve the per-layer window vector and clamp the pool windows to
        # the same max_seq_len bound, so their window keys agree.
        self.max_attention_window_vec = self._resolve_max_attention_window_vec(
            kv_cache_config=kv_cache_config,
            max_seq_len=max_seq_len,
            pool_configurations=self.pool_configurations,
        )

        # Build layer -> pool_idx now that the windows agree. Stays valid
        # through the block-budget clamping below, which only rewrites
        # per-pool window_size fields; pool indices don't shift.
        self._layer_to_pool_idx = self._build_layer_to_pool_idx()

        # Determine if this is VSWA (Variable Sliding Window Attention).
        # The `w > 0` check excludes LinearCacheType.RECURRENT_STATES sentinel
        # values (negative) used by hybrid linear attention models.
        self.is_vswa = uses_vswa_kv_cache_layout(self.max_attention_window_vec)
        self.is_linear_attention = linear_attention_metadata is not None

        # Calculate kv cache blocks for each window size
        # FIXME: flashinfer.py accesses kv_cache_manager.blocks_in_primary_pool
        # This dependency should be adjusted as it only covers the single window
        # case and not VSWA scheme.
        if is_estimating_kv_cache:
            # If this is an estimation dry run, we have already calculated the
            # max_tokens under _util.py::try_prepare_estimation
            # Since this is a dry run, assigning the same max_tokens capacity
            # to all window sizes as they are full attentions is enough.
            self.blocks_in_primary_pool = int(kv_cache_config.max_tokens //
                                              tokens_per_block)

            host_cache_size = kv_cache_config.host_cache_size if kv_cache_config.host_cache_size else 0
            max_tokens_secondary = host_cache_size // self.get_cache_bytes_per_token(
            )
            self.blocks_in_secondary_pool = int(max_tokens_secondary //
                                                tokens_per_block)

            blocks_per_window = {
                window_size:
                (self.blocks_in_primary_pool, self.blocks_in_secondary_pool)
                for window_size in set(self.max_attention_window_vec)
            }
            if self.is_linear_attention:
                # max_tokens is already the affine-correct value computed
                # upstream (_util.py:_tokens_for_budget honors the slope +
                # intercept of CppMambaHybridCacheManager). Recurrent state
                # slots live in a separate window: at minimum the live
                # state per concurrent request, and -- when block reuse is
                # enabled -- enough room for one regular snapshot per
                # snapshot interval over the full token budget. With
                # pipeline parallelism, multiple microbatches can be
                # in-flight simultaneously on the same rank, each holding
                # up to ``max_batch_size`` sequences' Mamba state, so the
                # live-state slot count must scale with ``pp_size``.
                pp_size = self.mapping.pp_size if self.mapping is not None else 1
                live_state_slots = self.max_batch_size * pp_size
                max_snapshots = live_state_slots
                snapshot_interval = (
                    linear_attention_metadata.states_snapshot_interval)
                if (kv_cache_config.enable_block_reuse
                        and snapshot_interval is not None
                        and snapshot_interval > 0):
                    max_snapshots += (kv_cache_config.max_tokens //
                                      snapshot_interval)

                blocks_per_window[LinearCacheType.RECURRENT_STATES.value] = (
                    int(max_snapshots), 0)
            logger.info(
                f"[kv cache manager] Primary/secondary blocks for window sizes set to {blocks_per_window} for estimation dry run"
            )
        else:
            if self.is_vswa or self.is_linear_attention:
                assert isinstance(
                    kv_cache_config, KvCacheConfig
                ), "calculate_max_num_blocks_for_vswa only accepts KvCacheConfig"
                blocks_per_window = self.calculate_max_num_blocks_for_vswa(
                    kv_cache_config=kv_cache_config,
                    extra_cost_memory=0,
                )
            else:
                # Standard case: use original Python implementation
                self.blocks_in_primary_pool, self.blocks_in_secondary_pool = self.calculate_max_num_blocks(
                    kv_cache_config=kv_cache_config,
                    head_dim=head_dim,
                    tokens_per_block=tokens_per_block,
                    mapping=mapping,
                    dtype=dtype,
                    kv_factor=self.kv_factor,
                    synchronize_across_ranks=False,
                )
                blocks_per_window = {
                    self.max_attention_window_vec[0]:
                    (self.blocks_in_primary_pool, self.blocks_in_secondary_pool)
                }
            if mapping.world_size > 1:
                # PP ranks can own different window sets or cache types.
                # Gather once after either sizing path so every rank executes
                # the same collective, then align local pool capacities.
                local_blocks_per_window = blocks_per_window.copy()
                local_attention_windows = {
                    window_size
                    for window_size in self.max_attention_window_vec
                    if window_size != LinearCacheType.RECURRENT_STATES.value
                }
                rank_sizing_states = Distributed.get(mapping).allgather(
                    (blocks_per_window, local_attention_windows))
                rank_blocks_per_window = [
                    state[0] for state in rank_sizing_states
                ]
                rank_attention_window_sets = [
                    state[1] for state in rank_sizing_states
                ]
                blocks_per_window = _get_minimum_blocks_per_window(
                    local_blocks_per_window,
                    rank_blocks_per_window,
                    rank_attention_window_sets,
                )
                logger.info(
                    f"[MPI rank={mapping.rank}] Original blocks_per_window: {local_blocks_per_window}"
                )
                logger.info(
                    f"[MPI rank={mapping.rank}] Reduced blocks_per_window: {blocks_per_window}"
                )

        if len(blocks_per_window) == 1:
            window_size, pool_blocks = next(iter(blocks_per_window.items()))
            if window_size != LinearCacheType.RECURRENT_STATES.value:
                # Single-pool consumers use these legacy scalar attributes.
                # Refresh them after distributed reduction so they match the
                # capacity passed to the C++ block manager.
                (self.blocks_in_primary_pool,
                 self.blocks_in_secondary_pool) = pool_blocks

        # Validate and adjust attention windows against their upper bounds if needed
        blocks_per_window, self.max_seq_len, self.max_attention_window_vec, window_adjustments = self._validate_and_adjust_attention_windows(
            max_attention_window_vec=self.max_attention_window_vec,
            blocks_per_window=blocks_per_window,
            tokens_per_block=tokens_per_block,
            max_seq_len=self.max_seq_len,
        )

        # Rewrite each pool's window_size to match the post-clamp window.
        # Without this, a pool pinned to a pre-clamp window (e.g. 32768)
        # would be silently dropped when the validator clamps the window
        # down (e.g. to 16384), leaving the C++ side to fall back on the
        # manager-level scalar -- which is exactly the heterogeneous case
        # these per-pool configs exist to handle.  Pool *indices* are
        # preserved so self._layer_to_pool_idx stays valid.
        if window_adjustments and self.pool_configurations:
            self.pool_configurations = [
                PoolConfiguration(
                    window_size=window_adjustments.get(pc.window_size,
                                                       pc.window_size),
                    head_dim=pc.head_dim,
                    dtype=pc.dtype,
                ) for pc in self.pool_configurations
            ]

        # Hybrid linear-attention managers intentionally carry two windows
        # (the RECURRENT_STATES sentinel window plus the full attention
        # window), including with the SELFKONLY cache type (e.g. Kimi K3:
        # MLA latent cache + KDA recurrent state). The C++ WindowBlockManager
        # treats kSELFKONLY as a self cache (kv_factor=1) with regular
        # per-window pools, so the single-window normalization below (meant
        # for cross-KV caches) must not fire for them.
        if kv_cache_type != CacheTypeCpp.SELF and not self.is_linear_attention:
            assert len(
                blocks_per_window
            ) == 1, "Only one window size is supported for non-self KV cache"
            # rewrite the attention window size in blocks_per_window
            memory_pools = blocks_per_window[self.max_attention_window_vec[0]]
            blocks_per_window = {self.max_seq_len: memory_pools}
            logger.info(
                f"Adjusted attention window size to {self.max_seq_len} in blocks_per_window"
            )

        # Use the provided execution stream for proper synchronization with KVCacheTransferManager.
        # The execution stream is the stream where model forward kernels run, and KVCacheTransferManager
        # needs to synchronize with it for onboard/offload operations.
        # If no execution stream is provided, create a new one (for backward compatibility).
        self._stream = execution_stream if execution_stream is not None else torch.cuda.Stream(
        )
        logger.info(f"[KVCacheManager] execution_stream: {self._stream}")
        logger.info(f"[KVCacheManager] blocks_per_window: {blocks_per_window}")

        # The Python @dataclass PoolConfiguration is a distinct type from the
        # nanobind C++ PoolConfiguration (Python uses ``head_dim``; C++ uses
        # ``size_per_head``).  Translate at the C++ boundary so nanobind can
        # dispatch the ctor.
        pool_configurations_cpp = [
            PoolConfigurationCpp(window_size=pc.window_size,
                                 size_per_head=pc.head_dim,
                                 dtype=pc.dtype)
            for pc in self.pool_configurations
        ]

        self.enable_indexer_k_cache = enable_indexer_k_cache

        kwargs = {
            'num_kv_heads_per_layer': self.num_kv_heads_per_layer,
            'size_per_head': head_dim,
            'tokens_per_block': tokens_per_block,
            'blocks_per_window': blocks_per_window,
            'max_num_sequences': max_batch_size,
            'max_beam_width': max_beam_width,
            'max_attention_window_vec': self.max_attention_window_vec,
            'dtype': dtype,
            'sink_token_length': 0,
            'stream': self._stream.cuda_stream,  # Pass to BufferManager
            'max_sequence_length': self.max_seq_len,
            'chunk_size': min(max_num_tokens, self.max_seq_len),
            'enable_block_reuse': kv_cache_config.enable_block_reuse,
            'cache_type': kv_cache_type,
            'secondary_offload_min_priority':
            kv_cache_config.secondary_offload_min_priority,
            'enable_partial_reuse': kv_cache_config.enable_partial_reuse,
            'copy_on_partial_reuse': kv_cache_config.copy_on_partial_reuse,
            'kv_connector_manager': self.kv_connector_manager,
            'enable_indexer_k_cache': enable_indexer_k_cache,
            'indexer_k_cache_quant_block_size':
            indexer_k_cache_quant_block_size,
            'indexer_k_cache_index_head_dim': indexer_k_cache_index_head_dim,
            'indexer_k_cache_use_fp4': indexer_k_cache_use_fp4,
            'indexer_k_cache_layer_mask': self.indexer_k_cache_local_layer_mask,
            'linear_attention_metadata': linear_attention_metadata,
            # Forward the (possibly remapped) per-pool configurations.
            # window_size values are aligned with the post-clamp sizes.
            'pool_configurations': pool_configurations_cpp,
        }

        if self.event_buffer_max_size > 0:
            _warn_if_unsupported_v1_kv_cache_event_hash_algo(
                kv_cache_config.kv_cache_event_hash_algo)
            if mapping.enable_attention_dp:
                kwargs['event_manager'] = KVCacheEventManagerCpp(
                    max_kv_event_entries=self.event_buffer_max_size,
                    attention_dp_rank=mapping.rank,
                    attention_dp_size=mapping.world_size,
                    attention_dp_events_gather_period_ms=self.
                    attention_dp_events_gather_period_ms,
                )
            elif mpi_rank() == 0:
                kwargs['event_manager'] = KVCacheEventManagerCpp(
                    max_kv_event_entries=self.event_buffer_max_size)

        self.impl = KVCacheManagerCpp(**kwargs)
        # Warmup baseline for cumulative counters (set by snapshot_warmup_baseline)
        self._warmup_reused_blocks = 0
        self._warmup_missed_blocks = 0

        self.impl.allocate_pools(False)
        self.kv_cache_pool_pointers = self.impl.get_block_pool_pointers()
        self.kv_cache_pool_mapping = self.impl.get_layer_to_pool_mapping()
        kv_cache_block_scale_pool_pointers = self.impl.get_block_scale_pool_pointers(
        )
        if kv_cache_block_scale_pool_pointers.numel() > 0:
            # C++ reports one effective configuration per actual window,
            # including manager-level defaults when none were supplied.
            dtype_by_window = {
                config.window_size: config.dtype
                for config in self.impl.pool_configurations
            }
            # Match the local layer order used by C++ to build the pointer
            # mapping.
            layer_pool_dtypes = [
                dtype_by_window[self.max_attention_window_vec[layer_offset]]
                for layer_offset in range(self.num_local_layers)
            ]
            self.kv_cache_pool_pointers = _merge_kv_cache_pool_pointers(
                self.kv_cache_pool_pointers,
                kv_cache_block_scale_pool_pointers,
                self.kv_cache_pool_mapping,
                layer_pool_dtypes,
            )

        self.num_pools = self.impl.num_pools
        self.max_blocks_per_seq = self.impl.max_blocks_per_seq
        self.enable_block_reuse = kv_cache_config.enable_block_reuse
        self.enable_partial_reuse = kv_cache_config.enable_partial_reuse
        # Keep unused block offsets as safe block index 0.
        self.host_kv_cache_block_offsets = torch.zeros(
            self.num_pools,
            max_batch_size * max_beam_width,
            2,
            self.max_blocks_per_seq,
            dtype=torch.int32,
            pin_memory=prefer_pinned(),
            device='cpu')
        self.blocks_per_window = blocks_per_window

    def probe_prefix_match_length(self,
                                  input_tokens,
                                  lora_task_id=None,
                                  cache_salt=None):
        """Probe the KV cache radix tree for prefix match length.

        Returns the number of prefix tokens already cached on this rank.
        Used by KVCacheAwareADPRouter for cache-aware routing.

        ``cache_salt`` scopes the probe to the same per-request
        namespace that ``_create_kv_cache`` uses; without it, salted
        requests would be probed against the salt=None namespace and the
        router would see the wrong match length.
        """
        if not self.enable_block_reuse:
            return 0
        # is_variable_window is only defined on the concrete KVCacheManager
        # nanobind class, not on BaseKVCacheManager. Use getattr to avoid
        # AttributeError on other subclasses or mocks.
        if getattr(self.impl, 'is_variable_window', False):
            return 0
        if not input_tokens:
            return 0
        from tensorrt_llm.bindings.executor import SamplingConfig
        from tensorrt_llm.bindings.internal.batch_manager import BlockKey
        from tensorrt_llm.bindings.internal.batch_manager import \
            LlmRequest as CppLlmRequest
        block_key = BlockKey(tokens=input_tokens, lora_task_id=lora_task_id)
        unique_tokens = block_key.unique_tokens
        # buildBlockKeys() on the C++ side derives the salt id from
        # cache_salt, so the dummy request must carry the same string used by
        # the real create_kv_cache path.
        dummy_req = CppLlmRequest(request_id=0,
                                  max_new_tokens=0,
                                  input_tokens=input_tokens,
                                  sampling_config=SamplingConfig(),
                                  is_streaming=False,
                                  lora_task_id=lora_task_id,
                                  cache_salt=cache_salt)
        summary = self.impl.analyze_prefix_reuse(unique_tokens, dummy_req)
        return summary.reusable_blocks_all * self.tokens_per_block

    def shutdown(self):
        self.impl.release_pools()

    def get_max_resource_count(self) -> int:
        return self.impl.max_num_blocks

    def get_num_tokens(self, request: LlmRequest) -> int:
        # LlmRequest.get_num_tokens is out of sync with GenerationRequest when overlap scheduler is enabled.
        return self.impl.get_token_count(request.py_request_id)

    def _num_blocks_to_completion(self, prompt_len: int, max_new_tokens: int,
                                  beam_width: int) -> int:
        # Beam-aware block estimate mirroring the C++ shared/unshared cost model
        # (getNeededBlocksOneStep in kvCacheManager.cpp): the full prompt blocks
        # are shared across all beams, while the partial-last-prompt block and
        # the generated tokens are private to each beam.
        shared_context_blocks = prompt_len // self.tokens_per_block
        per_beam_tokens = prompt_len % self.tokens_per_block + max_new_tokens
        per_beam_blocks = self.get_num_kv_blocks(per_beam_tokens) * beam_width
        return shared_context_blocks + per_beam_blocks

    def kv_block_budget_applies(self) -> bool:
        # Restricted to the exact base manager: the estimate below assumes every
        # token of the sequence occupies a block of this pool. Subclasses break
        # that assumption in ways that would make it wrong in either direction
        # -- sparse/compressed KV managers (RocketKVCacheManager,
        # DSACacheManager) retain less than the full sequence and would be
        # over-estimated into false rejections, while the mamba hybrids split
        # capacity across an extra state cache this single required/capacity
        # pair cannot express. A subclass opts into the dense estimate by
        # overriding this method once its own cost model has been checked
        # against it, or supplies a cost model of its own by overriding both
        # this and `get_request_kv_block_budget`.
        if type(self) is not KVCacheManager:
            return False
        if self.kv_cache_type == CacheTypeCpp.CROSS:
            return True
        # Multi-window and linear-attention managers have per-pool capacity
        # semantics that cannot be represented by one required/capacity pair.
        # Uniform sliding-window managers also require window-aware accounting.
        return not (self.kv_cache_type != CacheTypeCpp.SELF or self.is_vswa
                    or self.is_linear_attention
                    or any(window < self.max_seq_len
                           for window in self.max_attention_window_vec))

    def get_request_kv_block_budget(
            self, request: LlmRequest) -> Optional[Tuple[int, int]]:
        """Required and available KV blocks under the dense full-attention cost
        model, or ``None`` if this manager is not covered by it."""
        if not self.kv_block_budget_applies():
            return None

        if self.kv_cache_type == CacheTypeCpp.CROSS:
            if request.encoder_output_len is None:
                logger.warning(
                    f"Encoder output length is not set for request {request.request_id}"
                )
                return None
            required_blocks = self.get_num_kv_blocks(request.encoder_output_len)
            return required_blocks, self.blocks_in_primary_pool

        extra_tokens = (self.num_extra_kv_tokens +
                        self._kv_reserve_draft_tokens)
        required_blocks = self._num_blocks_to_completion(
            request.orig_prompt_len, request.max_new_tokens + extra_tokens,
            request.py_beam_width)
        # Use GPU-primary capacity only. Secondary/offloaded blocks cannot make
        # an otherwise impossible request schedulable on the GPU.
        return required_blocks, self.blocks_in_primary_pool

    def get_needed_resource_to_completion(self, request: LlmRequest) -> int:
        # TODO: the C++ implementation of this method can be used, but the
        # Python and C++ schedulers currently do not agree on what "needed
        # resource to completion" means. The C++ one excludes already allocated
        # blocks; the Python one includes them. This should be unified, but
        # the Python scheduler needs to be fixed.
        #
        # return self.impl.get_remaining_blocks_to_completion(request)
        #
        # This intentionally uses beam_width=1 to preserve the historical
        # beam-unaware value the Python scheduler expects.
        return self._num_blocks_to_completion(request.orig_prompt_len,
                                              request.max_new_tokens,
                                              beam_width=1)

    @staticmethod
    def _has_mm_bidirectional_block(req: LlmRequest) -> bool:
        """Whether ``req`` carries a bidirectional multimodal block that makes
        re-chunking unsafe.

        Mirrors the gate in ``scheduler_v2._align_chunk_to_mm_block``:
        re-chunking a request whose boundary would split a bidirectional
        multimodal block silently breaks attention, so such requests are left
        at their scheduled chunk size rather than shrunk.
        """
        mm = getattr(req, "py_multimodal_data", None)
        return isinstance(mm, dict) and mm.get("mm_bidirectional_blocks", False)

    def _request_forward_tokens(self, req: LlmRequest, *,
                                is_context: bool) -> int:
        """Number of position ids ``req`` contributes to the forward pass.

        Exact for context requests, but **only once every resource manager has
        prepared** -- see ``fit_token_budget``. Before that,
        ``context_chunk_size`` still spans the reusable KV prefix and this
        over-counts by up to the whole cached prefix.

        Mirrors ``_prepare_tp_inputs``: a context request contributes
        ``all_prompt_tokens[pos : pos + chunk]``, which Python slicing clamps to
        what is left of the prompt, plus draft tokens on the last chunk only.
        """
        draft_len = get_draft_token_length(req)
        if is_context:
            materialized = min(req.context_chunk_size,
                               req.context_remaining_length)
            return materialized + (draft_len
                                   if req.is_last_context_chunk else 0)
        # Generation: one position per beam for the new token, plus draft tokens
        # (speculative verification) per beam.
        return req.py_beam_width * (1 + draft_len)

    def _shrink_context_chunk(self, req: LlmRequest, excess: int) -> int:
        """Shrink ``req``'s chunk by up to ``excess`` tokens. Returns the number
        of forward-pass tokens actually shed (0 if the request cannot shrink).

        The new chunk must keep ``context_current_position + context_chunk_size``
        on a block boundary: ``setPrepopulatedPromptLen`` (llmRequest.h) asserts
        that for every non-last chunk, to keep the KV cache unfragmented. So the
        chunk is trimmed to the largest block-aligned end that sheds at least
        ``excess`` tokens, and never below one block of forward progress -- a
        zero-token chunk would leave the request scheduled but computing
        nothing, which never terminates.
        """
        pos = req.context_current_position
        current = min(req.context_chunk_size, req.context_remaining_length)

        # Smallest chunk that still lands on a block boundary and makes progress.
        floor_chunk = self.tokens_per_block - (pos % self.tokens_per_block)
        target_end = ((pos + current - excess) //
                      self.tokens_per_block) * self.tokens_per_block
        new_chunk = max(floor_chunk, target_end - pos)
        if new_chunk >= current:
            return 0

        # Shrinking flips is_last_context_chunk (a computed property:
        # context_current_position + chunk_size == prompt_len) to False, so the
        # caller must re-bin the batch afterwards -- otherwise downstream treats
        # this as a final chunk and appends generation / draft tokens to it.
        req.context_chunk_size = new_chunk
        return current - new_chunk

    def fit_token_budget(self, scheduled_batch: ScheduledRequests) -> None:
        """Shrink over-budget context chunks so the forward pass cannot exceed
        ``max_num_tokens``.

        Driven by ``ResourceManager.prepare_resources`` **after** every manager
        has prepared -- see there for why that is the only correct point. The
        micro-batch scheduler admits a batch on an estimate: it charges
        ``reuse_adjusted_compute(chunk, estimated_reusable_tokens, remaining)``
        (microBatchScheduler.cpp), where ``estimated_reusable_tokens`` is a
        radix-tree guess made during capacity scheduling. The real figure is
        ``prepopulated_prompt_len``, computed later inside ``addSequence``. When
        the real reuse comes in lower than the estimate, the forward pass
        materializes more tokens than were charged and trips the
        ``total_num_tokens <= max_num_tokens`` assert in ``_prepare_tp_inputs``
        -- which fails every request in the batch and charges the executor's
        error budget (GitHub issue #13318).

        That divergence is not knowable before ``prepare_resources``: the whole
        point is that the estimate was wrong, and only ``addSequence`` knows by
        how much. Running here, ``context_current_position`` and
        ``context_chunk_size`` are final and ``_request_forward_tokens`` is
        exact.

        Shrink only, never defer. KV cache is already allocated and sequences
        are already added, so a request cannot be dropped from the batch at this
        point -- but it does not need to be. Blocks are allocated for the full
        chunk; the tokens trimmed here are simply computed on the next
        iteration. Because nothing leaves the batch, this cannot desynchronize
        the attention-DP ``_can_queue`` vote, the inflight set, or any earlier
        manager's per-request state.

        Context requests are the only work this can shed, so a batch with none
        returns immediately -- keeping the gen-only batch (the executor loop's
        hottest path) free of an O(num_generation_requests) scan.
        """
        # Tested against the two lists rather than the ``context_requests``
        # property, which concatenates them into a fresh list on every access.
        if (not scheduled_batch.context_requests_chunking
                and not scheduled_batch.context_requests_last_chunk):
            return

        budget = self.max_num_tokens
        total = sum(
            self._request_forward_tokens(req, is_context=False)
            for req in scheduled_batch.generation_requests)
        context_requests = scheduled_batch.context_requests
        total += sum(
            self._request_forward_tokens(req, is_context=True)
            for req in context_requests
            if not req.is_disagg_generation_init_state)

        excess = total - budget
        if excess <= 0:
            return

        if not self.enable_chunked_prefill:
            # A shrunk chunk is a partial context chunk, which the attention
            # backend is only set up to consume under chunked prefill; forcing
            # one produces an invalid forward pass (empty-query asserts /
            # missing quantized KV buffers / cudaErrorInvalidValue). Nothing
            # safe is left to do, so let the assert fire as it does on main.
            logger.warning(
                f"Scheduled batch needs {total} forward-pass tokens, exceeding "
                f"max_num_tokens ({budget}) by {excess}. Cannot trim: chunked "
                "prefill is disabled. See GitHub issue #13318.")
            return

        # Shed from the back. ``context_requests`` is
        # ``context_requests_chunking + context_requests_last_chunk``, so this
        # trims last-chunk requests first -- which is where the overshoot comes
        # from: only a last chunk carries a reuse discount
        # (reuse_adjusted_compute's last-chunk branch) and draft tokens, so it is
        # the request whose cost the scheduler can have under-charged. Trimming
        # it converts it back into a chunking request, which is exactly the
        # repair. Mid-prefill chunks are touched only if that is not enough.
        shed = 0
        for req in reversed(context_requests):
            if excess - shed <= 0:
                break
            # Disagg generation-init requests only allocate/transfer KV cache
            # and contribute no compute tokens, so there is nothing to shed.
            if req.is_disagg_generation_init_state:
                continue
            # Re-chunking a request whose boundary would split a bidirectional
            # multimodal block silently breaks attention.
            if self._has_mm_bidirectional_block(req):
                continue
            shed += self._shrink_context_chunk(req, excess - shed)

        if shed:
            logger.debug(
                f"fit_token_budget: trimmed {shed} context tokens from "
                f"{scheduled_batch.num_context_requests} context requests to "
                f"stay within max_num_tokens={budget}")
            # Re-bin from each request's (possibly updated)
            # is_last_context_chunk: a shrunk request is no longer a last chunk.
            scheduled_batch.reset_context_requests(context_requests)

        if shed < excess:
            logger.warning(
                f"Scheduled batch needs {total} forward-pass tokens, exceeding "
                f"max_num_tokens ({budget}); could only trim {shed}. See "
                "GitHub issue #13318.")

    def maybe_fit_token_budget(self,
                               scheduled_batch: ScheduledRequests) -> None:
        """Apply the post-allocation token-budget trim to ``scheduled_batch``."""
        if not self.is_draft:
            # The draft-model engine builds inputs with a different token shape;
            # its budget is handled separately.
            self.fit_token_budget(scheduled_batch)

    def _context_seq_len(self, req: LlmRequest,
                         is_cross: bool) -> Optional[int]:
        """Return the sequence length to pass to add_sequence_batch, or None to skip this request."""
        if req.py_request_id in self._preprepared_dummy_request_ids:
            return None
        if is_cross:
            if (getattr(req, "py_skip_cross_kv_projection", False)
                    or not req.is_first_context_chunk
                    or not self._kv_connector_should_add_sequence(req)):
                return None
            encoder_output_len = getattr(req, "encoder_output_len", None)
            if encoder_output_len is None:
                raise RuntimeError(
                    "Cross KV cache allocation requires "
                    f"encoder_output_len for request {req.py_request_id}.")
            return int(encoder_output_len)
        if not req.is_first_context_chunk or not self._kv_connector_should_add_sequence(
                req):
            return None
        return req.prompt_len

    def prepare_resources(self, scheduled_batch: ScheduledRequests):
        # Cross/encoder K/V is allocated once and never grows; handle it on a
        # dedicated path so the self-attention flow below stays unconditional.
        if self.kv_cache_type == CacheTypeCpp.CROSS:
            return self._prepare_cross_resources(scheduled_batch)

        with request_context(self.is_draft, scheduled_batch):
            # wait for all pending work to finish before launching offload/onboarding/partial copy
            self.impl.sync_transfer_manager_with_buffer_manager()

            # Collect first-chunk requests eligible for batch add_sequence_batch.
            # When block reuse is enabled, addSequenceBatch uses a two-phase
            # claim-then-onboard strategy that prevents host offloading from
            # evicting reusable blocks in the radix tree.
            batch_request_infos, batch_llm_requests = self._collect_context_sequences(
                scheduled_batch, is_cross=False)

            if batch_request_infos:
                self.impl.add_sequence_batch(batch_request_infos,
                                             batch_llm_requests)
                for req in batch_llm_requests:
                    for _ in range(self.num_extra_kv_tokens):
                        self.impl.add_token(req.py_request_id)
                    for _ in range(get_draft_token_length(req)):
                        self.impl.add_token(req.py_request_id)

                    if self.kv_connector_manager is not None:
                        block_ids = self.get_cache_indices(req)
                        self.kv_connector_manager.update_state_after_alloc(
                            req, block_ids)

            for req in scheduled_batch.generation_requests:
                if self.mapping.has_cp_helix():
                    # Distribute the decode blocks across CP ranks in a round-robin manner.
                    decode_block_id = (req.py_decoding_iter -
                                       1) // self.tokens_per_block
                    if decode_block_id % self.mapping.cp_size == self.mapping.cp_rank:
                        req.py_helix_is_inactive_rank = False
                        req.seqlen_this_rank_cp += 1
                    else:
                        req.py_helix_is_inactive_rank = True
                        # Skip allocating KV cache at decode for inactive helix ranks.
                        continue
                draft_len = get_draft_token_length(req)
                self.impl.add_token(req.py_request_id)
                for _ in range(max(draft_len, self._kv_reserve_draft_tokens)):
                    self.impl.add_token(req.py_request_id)

            # prefill and generation kernels wait for scheduled offload/onboard/partial copy work before launching
            self.impl.refresh_blocks()

        # A request may change from `context_requests_chunking` to
        # `context_requests_last_chunk` in `add_sequence` due to KV cache
        # reuse, so we rebuild the context request lists here.
        scheduled_batch.reset_context_requests()

    def report_batch_to_connector(self,
                                  scheduled_batch: ScheduledRequests) -> None:
        """Report the batch to the KV connector.

        Driven by ``ResourceManager.prepare_resources`` *after* the token-budget
        trim rather than from the end of ``prepare_resources`` above, because
        ``RequestData.num_scheduled_tokens`` is documented as "the number of
        scheduled tokens for the upcoming forward pass" and is built from
        ``context_chunk_size`` (``connectors/kv_cache_connector.py``). Reporting
        before the trim over-states it for every chunk the trim shrinks, and a
        connector that decides what to save or offload from that count would
        publish KV for tokens the forward pass never computed.
        """
        if self.kv_connector_manager is not None:
            self.kv_connector_manager.build_scheduler_output(
                scheduled_batch, self)

    def _collect_context_sequences(self, scheduled_batch: ScheduledRequests,
                                   is_cross: bool):
        """Build the (request_info, llm_request) lists for add_sequence_batch.

        Cross (encoder) sequences are sized from encoder_output_len with a beam
        width of 1 (request-scoped); self-attention sequences use the request's
        own beam width.
        """
        batch_request_infos = []
        batch_llm_requests = []
        for req in scheduled_batch.context_requests:
            seq_len = self._context_seq_len(req, is_cross)
            if seq_len is None:
                continue
            beam_width = 1 if is_cross else req.py_beam_width
            batch_request_infos.append((req.py_request_id, seq_len, beam_width))
            batch_llm_requests.append(req)
        return batch_request_infos, batch_llm_requests

    def _prepare_cross_resources(self, scheduled_batch: ScheduledRequests):
        """Allocate cross (encoder) K/V blocks.

        Encoder K/V is written once at the first decoder context step and read
        unchanged on every generation step, so it never grows: this skips the
        decode-time token growth, draft-token reserve, and scheduler bookkeeping
        that the self-attention path performs.
        """
        with request_context(self.is_draft, scheduled_batch):
            # wait for all pending work to finish before launching offload/onboarding/partial copy
            self.impl.sync_transfer_manager_with_buffer_manager()
            batch_request_infos, batch_llm_requests = self._collect_context_sequences(
                scheduled_batch, is_cross=True)
            if batch_request_infos:
                self.impl.add_sequence_batch(batch_request_infos,
                                             batch_llm_requests)
            # kernels wait for scheduled offload/onboard/partial copy work before launching
            self.impl.refresh_blocks()

    def extend_capacity_for_tokens(self, request: LlmRequest) -> None:
        """No-op for V1; interface kept consistent with V2."""

    def _kv_connector_should_add_sequence(self, request: LlmRequest) -> bool:
        return self.kv_connector_manager is None or self.kv_connector_manager.should_add_sequence(
            request)

    def add_dummy_requests(
        self,
        request_ids: List[int],
        # Note that token_nums should be past_kv_len + input_len (without
        # spec decoding). The draft tokens will be added in this function,
        # so we don't need to take care of it in the caller. When preparing
        # token_nums, we should not take the draft tokens into account, so
        # don't use the kv_cache_manager.max_seq_len, which includes both
        # extra tokens and draft tokens.
        token_nums: Optional[List[int]] = None,
        is_gen: bool = False,
        prepare_resource: bool = True,
        max_num_draft_tokens: int = 0,
        # Override of py_draft_tokens length for KV reserve (e.g. dynamic-tree
        # draft loop). Falls back to max_num_draft_tokens when None.
        kv_reserve_draft_tokens: Optional[int] = None,
        use_mrope: bool = False,
        max_beam_width: int = 1,
        encoder_output_lens: Optional[List[int]] = None,
        draft_kv_cache_manager: Optional[BaseResourceManager] = None,
        capture_sampling_params: Optional[SamplingParams] = None,
    ):
        _kv_draft = kv_reserve_draft_tokens if kv_reserve_draft_tokens is not None else max_num_draft_tokens
        available_blocks = self.get_num_free_blocks()
        # No padding if not enough KV cache space
        if available_blocks < 1:
            return None

        beam_width = max_beam_width
        requests = []
        batch_request_infos = []
        batch_llm_requests = []
        draft_batch_request_infos = []
        draft_batch_llm_requests = []
        for i, req_id in enumerate(request_ids):
            # exact choice of n can be ignored for dummy requests
            sampling_params = SamplingParams(
                n=beam_width,
                best_of=beam_width,
                use_beam_search=beam_width > 1,
                temperature=capture_sampling_params.temperature
                if capture_sampling_params is not None else None,
                top_k=capture_sampling_params.top_k
                if capture_sampling_params is not None else None,
                top_p=capture_sampling_params.top_p
                if capture_sampling_params is not None else None,
            )
            # Here 1+max_num_draft_tokens is used to extend the prompt length to
            # a non-zero number to skip illegal memory access issue in MLA kernel
            # during warmup.
            token_num = token_nums[
                i] if token_nums is not None else 1 + max_num_draft_tokens
            # Helix active rank sets past_seen_token_num = seqlen_this_rank_cp - 1
            # in _prepare_tp_inputs; need token_num >= 2 so that doesn't go negative.
            if self.mapping.has_cp_helix():
                token_num = max(token_num, 2)
            encoder_output_len = (encoder_output_lens[i]
                                  if encoder_output_lens is not None else None)
            encoder_input_tokens = ([1] * encoder_output_len
                                    if encoder_output_len is not None else None)
            # Using 1 instead of 0 prevents NaN during warmup in e.g. Deepseek
            req = LlmRequest(
                request_id=req_id,
                max_new_tokens=1,
                input_tokens=[1] * token_num,
                sampling_config=sampling_params._get_sampling_config(),
                is_streaming=False,
                encoder_input_tokens=encoder_input_tokens,
                encoder_output_len=encoder_output_len)
            req.is_dummy_request = True
            req.paged_kv_block_ids = []
            if prepare_resource:
                batch_request_infos.append((req_id, token_num, beam_width))
                batch_llm_requests.append(req)
                if draft_kv_cache_manager is not None:
                    draft_batch_request_infos.append(
                        (req_id, token_num, beam_width))
                    draft_batch_llm_requests.append(req)

            if use_mrope:
                _populate_dummy_mrope_config(req, token_num, is_gen)
            requests.append(req)

        try:
            # Use add_sequence_batch for all dummy requests, then add extra tokens.
            # This must happen before is_gen state modifications below, which may
            # set prompt_len to 0 and trigger assertion in setPrepopulatedPromptLen.
            if batch_request_infos:
                self.impl.add_sequence_batch(batch_request_infos,
                                             batch_llm_requests)
                for req_id, token_num, _ in batch_request_infos:
                    for _ in range(self.num_extra_kv_tokens):
                        self.impl.add_token(req_id)

            if draft_batch_request_infos and draft_kv_cache_manager is not None:
                draft_kv_cache_manager.impl.add_sequence_batch(
                    draft_batch_request_infos, draft_batch_llm_requests)
                for req_id, _, _ in draft_batch_request_infos:
                    for _ in range(self.num_extra_kv_tokens):
                        draft_kv_cache_manager.impl.add_token(req_id)

            # Set is_gen state after add_sequence_batch to avoid modifying
            # prompt_len before the C++ side reads it.
            if is_gen:
                for i, req in enumerate(requests):
                    token_num = token_nums[
                        i] if token_nums is not None else 1 + max_num_draft_tokens
                    if self.mapping.has_cp_helix():
                        token_num = max(token_num, 2)
                    req.state = LlmRequestState.GENERATION_IN_PROGRESS
                    req.prompt_len = token_num - 1
                    req.py_prompt_len = req.prompt_len
                    if self.mapping.has_cp_helix():
                        if self.mapping.cp_size - 1 == self.mapping.cp_rank:
                            req.py_helix_is_inactive_rank = False
                            req.prompt_len = token_num - 1
                            req.py_prompt_len = req.prompt_len
                            req.seqlen_this_rank_cp = req.prompt_len
                            req.total_input_len_cp = token_num * self.mapping.cp_size - 1
                            req.py_decoding_iter = 1
                        else:
                            req.py_helix_is_inactive_rank = True
                            req.prompt_len = token_num
                            req.py_prompt_len = req.prompt_len
                            req.seqlen_this_rank_cp = req.prompt_len
                            req.total_input_len_cp = token_num * self.mapping.cp_size - 1
                            req.py_decoding_iter = 1
                    req.py_draft_tokens = [1] * max_num_draft_tokens
                    if prepare_resource:
                        for _ in range(_kv_draft):
                            self.impl.add_token(req.request_id)
                        if draft_kv_cache_manager is not None:
                            for _ in range(_kv_draft):
                                draft_kv_cache_manager.impl.add_token(
                                    req.request_id)
        except Exception:
            # A partial allocation failure (e.g. add_token raising "no free
            # blocks left" after add_sequence_batch succeeded) must not leak
            # the sequences already registered. On the minimal KV pool built
            # for cache-size estimation, such a leak leaves too few blocks
            # for the estimation requests themselves, so the executor loop
            # spins forever without ever scheduling them and LLM startup
            # hangs (TRTLLM-14903). remove_sequence is a no-op for request
            # ids the failed batched add never registered, so every request
            # can be removed unconditionally; attempt all target and draft
            # cleanup before re-raising so one cleanup failure doesn't leak
            # the remaining sequences.
            cleanup_error = None
            for freeing_impl, freeing_requests in (
                (self.impl, batch_llm_requests),
                (draft_kv_cache_manager.impl if draft_kv_cache_manager
                 is not None else None, draft_batch_llm_requests),
            ):
                if freeing_impl is None:
                    continue
                for req in freeing_requests:
                    try:
                        freeing_impl.remove_sequence(req.py_request_id, req,
                                                     False)
                    except Exception as e:
                        cleanup_error = cleanup_error or e
            if cleanup_error is not None:
                # A failed release leaves the KV pool poisoned — the same
                # hang mechanism this cleanup exists to prevent — so it
                # must not be masked by the allocation failure alone.
                raise cleanup_error
            raise

        if batch_request_infos:
            self._preprepared_dummy_request_ids.update(
                req_id for req_id, _, _ in batch_request_infos)
        if (draft_batch_request_infos
                and isinstance(draft_kv_cache_manager, KVCacheManager)):
            draft_kv_cache_manager._preprepared_dummy_request_ids.update(
                req_id for req_id, _, _ in draft_batch_request_infos)

        return requests

    def update_resources(self,
                         scheduled_batch: ScheduledRequests,
                         attn_metadata: "AttentionMetadata" = None,
                         kv_cache_dtype_byte_size: float = None):
        # Self-attention pools rewind rejected speculative tokens each step;
        # cross/encoder K/V is immutable, so only the context-block commit below
        # applies to it.
        if self.kv_cache_type != CacheTypeCpp.CROSS:
            if not self.is_draft:
                from .kv_cache_manager_v2 import \
                    _update_kv_cache_draft_token_location

                _update_kv_cache_draft_token_location(self, scheduled_batch,
                                                      attn_metadata,
                                                      kv_cache_dtype_byte_size)

            # Rewind KV cache for requests with rejected draft tokens.
            # Skip:
            # - GENERATION_COMPLETE: finished requests
            # - CONTEXT_INIT: requests whose state was reset after being paused with KV cache freed.
            #   With overlap scheduler, the scheduler pauses a request and frees KV cache at iteration N,
            #   while the previous batch (N-1) is still trying to update the KV cache after forward pass.
            for request in scheduled_batch.generation_requests:
                if request.state in (LlmRequestState.GENERATION_COMPLETE,
                                     LlmRequestState.CONTEXT_INIT):
                    continue
                if request.py_rewind_len > 0:
                    self.rewind_kv_cache(request, request.py_rewind_len)
                # Symmetric companion to prepare_resources's reserve_slack
                # add_token loop: when _kv_reserve_draft_tokens (e.g. dynamic
                # tree's K*max_draft_len) exceeds the runtime draft length,
                # those extra slots must also be rewound, otherwise the draft
                # KV cache leaks reserve_slack tokens per generation iteration
                # and eventually overflows mCacheBlockIndices.
                runtime_draft_len = (request.py_rewind_len +
                                     request.py_num_accepted_draft_tokens)
                extra_rewind = self._kv_reserve_draft_tokens - runtime_draft_len
                if extra_rewind > 0:
                    self.rewind_kv_cache(request, extra_rewind)

        # For context requests, store completed context blocks for KV cache reuse.
        # We wait until context_remaining_length == 0 (all chunks processed) before
        # storing, so that SWA windows are safe to store — blocks won't go out-of-window
        # and be evicted while the context is still in-flight.
        for request in scheduled_batch.context_requests:
            self.impl.store_context_blocks(request)

    def free_resources(self, request: LlmRequest, pin_on_release: bool = False):
        result = self.impl.remove_sequence(request.py_request_id, request,
                                           pin_on_release)
        self._preprepared_dummy_request_ids.discard(request.py_request_id)
        return result

    def store_blocks_for_reuse(self,
                               request: LlmRequest,
                               pin_blocks: bool = False):
        return self.impl.store_blocks_for_reuse(request.py_request_id, request,
                                                pin_blocks)

    @staticmethod
    def calculate_scaling_factor_size_bytes(
            cache_size: int, quant_vector_size: int,
            scaling_factor_dtype: DataType) -> int:
        assert cache_size % quant_vector_size == 0, "NVFP4 cache size must be divisible by quant vector size"
        return get_size_in_bytes(cache_size // quant_vector_size,
                                 scaling_factor_dtype)

    def _resolve_max_attention_window_vec(
        self,
        kv_cache_config: KvCacheConfig,
        max_seq_len: int,
        pool_configurations: Optional[List["PoolConfiguration"]] = None,
    ) -> List[int]:
        """Compute the per-local-layer attention window vector.

        Two input shapes are supported and projected into local layer order:

        * ``max_attention_window is None``: use ``max_seq_len`` for every
          local layer (single-window default).
        * Otherwise: treat the user-supplied vector as a repeating pattern
          anchored at global model layer zero.

        ``pool_configurations`` (if given) are clamped in place to the same
        ``max_seq_len`` bound so their window keys stay consistent with the
        returned vector; ``_build_layer_to_pool_idx`` relies on that match.
        """
        for pc in pool_configurations or []:
            pc.window_size = min(pc.window_size, max_seq_len)
        clamped_window_pattern = _clamp_max_attention_window_vec(
            kv_cache_config.max_attention_window, max_seq_len)
        local_windows = _project_max_attention_window_vec(
            clamped_window_pattern,
            self.pp_layers,
        )
        return local_windows

    @staticmethod
    def _resolve_num_attention_layers(
        model_config: ModelConfigPython,
        mapping: Mapping,
        num_layers: Optional[int] = None,
    ) -> int:
        """Compute the effective number of attention layers for cache sizing.

        When *num_layers* is explicitly provided (e.g. for draft models whose
        HF config layer count differs from runtime), it is used directly
        without PP distribution.  Otherwise the layer count is derived from
        the model config and distributed evenly across PP ranks via
        `mapping.pp_layers`.
        """
        if num_layers is not None:
            return max(num_layers, 1)
        return max(
            len(mapping.pp_layers(model_config.get_num_attention_layers())), 1)

    # TODO: refactor get_cache_size_per_token and get_cache_bytes_per_token to use the same logic
    @staticmethod
    def get_cache_size_per_token(model_config: ModelConfigPython,
                                 mapping: Mapping,
                                 num_layers: Optional[int] = None,
                                 **kwargs):

        # get num key value heads
        config = model_config.pretrained_config
        # assert not is_hybrid_linear(config)
        num_key_value_heads = getattr(config, 'num_key_value_heads',
                                      config.num_attention_heads)
        if isinstance(num_key_value_heads, Iterable):
            num_key_value_heads = sum(num_key_value_heads) / len(
                num_key_value_heads)

        # get head dim
        mla = hasattr(config,
                      "kv_lora_rank") and config.kv_lora_rank is not None
        if mla:
            head_dim = config.kv_lora_rank + config.qk_rope_head_dim
            kv_factor = 1
        else:
            tp_size = 1 if mapping.enable_attention_dp else mapping.tp_size
            head_dim = getattr(config, "head_dim", None)
            if not isinstance(head_dim, int):
                head_dim = config.hidden_size // config.num_attention_heads
            head_dim = head_dim * num_key_value_heads // tp_size
            kv_factor = 2

        num_attention_layers = KVCacheManager._resolve_num_attention_layers(
            model_config, mapping, num_layers)
        # K and V
        mem_per_token = kv_factor * num_attention_layers * head_dim
        # The data type bytes.
        quant_config = model_config.quant_config
        if quant_config is not None and quant_config.quant_mode.has_fp8_kv_cache(
        ):
            mem_per_token *= 1
        elif quant_config is not None and quant_config.quant_mode.has_fp4_kv_cache(
        ):
            # 1 bytes for 2 elements, and SFs (fp8) per 16 elements.
            mem_per_token = math.ceil(mem_per_token / 2) + math.ceil(
                mem_per_token / 16)
        else:
            # All other cases (fp16/bf16 kv cache), we need 2 bytes per token for K and V.
            assert quant_config is None or (
                not quant_config.quant_mode.has_kv_cache_quant()
            ), "Quantized kv cache is not expected"
            mem_per_token *= 2
        return mem_per_token

    def get_cache_bytes_per_token(self):
        if isinstance(self.head_dim, list):
            # Per-layer head_dim (e.g., Gemma4 hybrid attention)
            cache_size_per_token = self.kv_factor * sum(
                kv * hd for kv, hd in zip(self.total_num_kv_heads_per_layer,
                                          self.head_dim))
        else:
            cache_size_per_token = self.kv_factor * sum(
                self.num_kv_heads_per_layer) * self.head_dim

        if self.dtype not in (DataType.FP8, DataType.HALF, DataType.BF16,
                              DataType.FLOAT, DataType.NVFP4):
            raise ValueError(f'Cannot support {self.dtype} KV cache.')

        cache_size_bytes_per_token = get_size_in_bytes(cache_size_per_token,
                                                       self.dtype)
        if self.dtype == DataType.NVFP4:
            cache_size_bytes_per_token += self.calculate_scaling_factor_size_bytes(
                cache_size_per_token,
                quant_vector_size=16,
                scaling_factor_dtype=DataType.FP8)
        return cache_size_bytes_per_token

    def calculate_max_num_blocks(
            self,
            kv_cache_config: KvCacheConfig,
            head_dim: int,
            tokens_per_block: int,
            mapping: Mapping,
            dtype: DataType,
            kv_factor: int = 2,
            synchronize_across_ranks: bool = True) -> Tuple[int, int]:
        free_mem_fraction = (kv_cache_config.free_gpu_memory_fraction
                             if kv_cache_config.free_gpu_memory_fraction
                             is not None else 0.9)

        cache_size_bytes_per_token = self.get_cache_bytes_per_token()

        free_mem, total_mem = torch.cuda.mem_get_info()

        assert free_mem_fraction < 1.0, f"Invalid freeMemFraction, freeMemFraction {free_mem_fraction} must be smaller than 1.0"
        max_tokens = free_mem_fraction * free_mem / cache_size_bytes_per_token

        # If user specified a number of tokens
        if kv_cache_config.max_tokens is not None:
            # If user also specified a free gpu memory fraction, take the min
            if kv_cache_config.free_gpu_memory_fraction is not None:
                max_tokens = min(kv_cache_config.max_tokens, max_tokens)
                logger.warning(
                    f'Both free_gpu_memory_fraction and max_tokens are set (to {free_mem_fraction} and {max_tokens} with free memory {free_mem / (1 << 30)}GiB of total memory {total_mem / (1<<30)}GiB, respectively). The smaller value will be used.'
                )
            else:
                max_tokens = kv_cache_config.max_tokens
                logger.info(
                    f"max_tokens is set by kv_cache_config.max_tokens: {max_tokens}"
                )

        if synchronize_across_ranks and mapping.world_size > 1:
            # make sure all ranks use same value for maxTokens
            dist = Distributed.get(mapping)
            max_tokens = dist.allreduce(
                max_tokens,
                op=ReduceOp.MIN,
            )

        # get number of blocks
        blocks_in_primary_pool = int(max_tokens // tokens_per_block)

        host_cache_size = kv_cache_config.host_cache_size if kv_cache_config.host_cache_size else 0
        max_tokens_secondary = host_cache_size // self.get_cache_bytes_per_token(
        )
        blocks_in_secondary_pool = int(max_tokens_secondary // tokens_per_block)

        return blocks_in_primary_pool, blocks_in_secondary_pool

    def _resolve_window_size(
            self,
            window_size: Optional[int],
            error_msg: str = "window_size must be provided for VSWA") -> int:
        if window_size is not None:
            return window_size
        if self.is_vswa:
            raise ValueError(error_msg)
        return self.max_attention_window_vec[0]

    def get_cache_indices(self,
                          request: LlmRequest,
                          window_size: Optional[int] = None) -> List[int]:
        window_size = self._resolve_window_size(window_size)
        result = self.impl.get_cache_block_ids(request.py_request_id,
                                               window_size)
        assert len(result) == 1
        return result[0]

    def get_num_front_blocks_removed(self,
                                     request_id: int,
                                     window_size: Optional[int] = None) -> int:
        """Get the number of front blocks evicted by SWA for a sequence.

        Args:
            request_id: The request id.
            window_size: Optional window size.  When supplied, returns the
                per-window eviction count (zero for non-SWA windows).  When
                omitted, defaults to ``self.max_attention_window_vec[0]`` —
                this matches the historical single-pool behavior for
                callers that never thought about windows, while keeping the
                C++ contract uniformly per-window.  VSWA callers should
                always pass ``window_size`` explicitly.
        """
        if window_size is None:
            window_size = self.max_attention_window_vec[0]
        return self.impl.get_num_front_blocks_removed(request_id,
                                                      window_size=window_size)

    def commit_and_get_block_hashes(
            self,
            request: LlmRequest,
            window_size: Optional[int] = None) -> List[int]:
        """Commit and return the chain of stored block hashes for ``request``.

        Wraps ``BaseKVCacheManager::commitAndGetBlockHashesForRequest``. The C++
        side sets each block's ``mBlockKey`` and ``mHash`` on first call so the
        hash matches what ``storeBlocks`` would later compute. Beam-width-1
        only; the connector enforces this at startup.
        """
        window_size = self._resolve_window_size(window_size)
        return list(
            self.impl.commit_and_get_block_hashes_for_request(
                request, window_size))

    def unpin_blocks_by_id(self, kv_cache_block_id: int):
        self.impl.unpin_blocks_by_id(kv_cache_block_id)

    def get_last_block_id(self, request_id: int) -> int:
        return self.impl.get_last_block_id(request_id)

    def get_priority_by_block_id(self,
                                 block_id: int,
                                 window_size: Optional[int] = None) -> int:
        """Get the retention priority of a block by its ID.

        Args:
            block_id: The ID of the block.
            window_size: The attention window size this block belongs to.
                         Required for VSWA configurations with multiple window sizes.

        Returns:
            The retention priority of the block (0-100), or default priority (35) if not found.
        """
        window_size = self._resolve_window_size(window_size)
        return self.impl.get_priority_by_block_id(block_id, window_size)

    def get_memory_pool_block_indices(self, block_ids: List[int], *,
                                      window_size: int) -> List[int]:
        # Translate logical block IDs to primary-pool slot indices. With host offload enabled,
        # a block's ID and its current pool slot can diverge after an offload/onboard cycle.
        # Every referenced block must be primary (asserted in C++): callers do primary-pool
        # pointer arithmetic and the returned index carries no residency information.
        return self.impl.get_memory_pool_block_indices(list(block_ids),
                                                       window_size)

    def get_batch_cache_indices(
        self,
        request_ids: List[int],
        layer_idx: Optional[int] = None,
        window_size: Optional[int] = None,
        beam_width: Optional[int] = 1,
        num_blocks_per_seq: Optional[Sequence[int]] = None,
    ) -> List[List[int]]:
        beam_width = beam_width or 1
        if window_size is None:
            if layer_idx is None:
                window_size = self._resolve_window_size(
                    window_size,
                    "layer_idx or window_size must be provided for VSWA")
            else:
                layer_offset = self.layer_offsets[layer_idx]
                # Explicit layer_offset -> window_size mapping (no modulo
                # masking length mismatches between pattern and num_local_layers).
                window_size = self._get_layer_offset_to_window_size(
                )[layer_offset]

        result = self.impl.get_batch_cache_block_ids(request_ids, window_size)
        for i in range(len(result)):
            beams = [list(beam) for beam in result[i]]
            assert len(beams) == beam_width, (
                f"Expected {beam_width} index arrays per request, got {len(beams)}"
            )
            result[i] = beams[
                0] if beam_width == 1 else self._pack_beam_cache_indices(beams)
            if num_blocks_per_seq is not None:
                result[i] = result[i][:num_blocks_per_seq[i]]
        return result

    def get_batch_cache_indices_flat(
        self,
        request_ids: List[int],
        num_blocks: List[int],
        layer_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """Concatenated per-request block tables, trimmed to real widths.

        Equivalent to concatenating
        ``get_batch_cache_indices(request_ids, layer_idx)[i][:num_blocks[i]]``
        over all requests into one CPU int32 tensor; matches the interface
        of ``KVCacheManagerV2.get_batch_cache_indices_flat``.
        """
        block_ids_per_seq = self.get_batch_cache_indices(
            request_ids, layer_idx=layer_idx, num_blocks_per_seq=num_blocks)
        indices_list = []
        for block_ids, n in zip(block_ids_per_seq, num_blocks):
            indices_list.extend(block_ids[:n])
        return torch.tensor(indices_list, dtype=torch.int32)

    @staticmethod
    def _pack_beam_cache_indices(beams: List[List[int]]) -> List[int]:
        """Pack beam-search blocks into a flat beam-0 layout.

        The first beam owns the shared prompt blocks. For every other beam,
        append only the final block when it differs from beam 0's final block.
        """
        if not beams:
            return []
        packed = list(beams[0])
        beam0_last = beams[0][-1] if beams[0] else None
        for beam in beams[1:]:
            if beam and beam[-1] != beam0_last:
                packed.append(beam[-1])
        return packed

    def get_num_free_blocks(self) -> int:
        if self.is_linear_attention:
            value = self.impl.get_kv_cache_stats(
            ).num_free_blocks_per_window_size[self.max_seq_len]
            logger.debug(
                f"For linear attention case, we return the number of free blocks for the kv cache (not for the recurrent states): {value}"
            )
            return value
        if self.is_vswa:
            logger.info(
                f"For {'linear attention' if self.is_linear_attention else 'VSWA'} case, we return the minimum of the number of free blocks for each window size: {self.impl.get_kv_cache_stats().num_free_blocks_per_window_size}"
            )
            return min(self.impl.get_kv_cache_stats().
                       num_free_blocks_per_window_size.values())
        else:
            return self.impl.get_kv_cache_stats().free_num_blocks

    def get_num_kv_blocks(self, num_tokens: int) -> int:
        return (num_tokens + self.tokens_per_block - 1) // self.tokens_per_block

    def get_num_available_tokens(self,
                                 token_num_upper_bound: int,
                                 max_num_draft_tokens: int = 0,
                                 **kwargs) -> int:
        free_blocks = self.get_num_free_blocks()
        result = min(
            token_num_upper_bound, free_blocks * self.tokens_per_block -
            self.num_extra_kv_tokens - max_num_draft_tokens)
        logger.debug(
            f"[get_num_available_tokens] free_blocks={free_blocks}, "
            f"tokens_per_block={self.tokens_per_block}, "
            f"num_extra_kv_tokens={self.num_extra_kv_tokens}, "
            f"token_num_upper_bound={token_num_upper_bound}, result={result}")
        return result

    def get_buffers(self,
                    layer_idx: int,
                    kv_layout: str = "NHD") -> Optional[torch.Tensor]:
        ''' Slice KV tensor for a specified layer and reshape it.

        1. Slice:
            [max_num_pages, num_layers, kv_factor, page_size * num_kv_heads * head_dim] ->
            [max_num_pages, kv_factor, page_size * num_kv_heads * head_dim]

        2. Reshape:
            kv_layout = "NHD" -> [max_num_pages, kv_factor, page_size, num_kv_heads, head_dim]
            kv_layout = "HND" -> [max_num_pages, kv_factor, num_kv_heads, page_size, head_dim]

        Note that different attention backend/implementation can have different KV layouts,
        "kv_layout" should be set accordingly to avoid surprises.

        Per-layer head_dim: when the underlying C++ manager hosts multiple pools with
        distinct head_dim (e.g., Gemma4 SWA head_dim=256 alongside full-attention
        head_dim=512), this method reads the layer's effective head_dim from the
        layer's assigned ``PoolConfiguration`` rather than the manager-level
        scalar.  Single-pool managers fall back to ``self.head_dim``.
        '''
        layer_offset = self.layer_offsets[layer_idx]
        result = self.impl.get_primary_pool_data(layer_offset)

        pool = self.get_pool_for_layer(layer_offset)
        layer_head_dim = pool.head_dim if pool else self.head_dim

        assert kv_layout in ["NHD",
                             "HND"], f"Unsupported kv_layout: {kv_layout}"
        if kv_layout == "NHD":
            return result.reshape(
                result.shape[0],
                self.kv_factor,
                self.tokens_per_block,
                self.num_kv_heads_per_layer[layer_offset],
                layer_head_dim,
            )
        else:
            return result.reshape(
                result.shape[0],
                self.kv_factor,
                self.num_kv_heads_per_layer[layer_offset],
                self.tokens_per_block,
                layer_head_dim,
            )

    def get_indexer_k_cache_pool_data(self, layer_idx: int) -> torch.Tensor:
        result = self.impl.get_indexer_k_cache_pool_data(layer_idx)
        return result.view(result.shape[0], -1)

    def check_invalid_values_in_kv_cache(self,
                                         fill_with_zero: bool = False) -> bool:
        some_checks_unavailable = False
        has_invalid_values = torch.tensor([False],
                                          dtype=torch.bool,
                                          device=torch.cuda.current_device())
        for layer_idx, layer_offset in self.layer_offsets.items():
            buffer = self.impl.get_primary_pool_data(layer_offset)
            # process in chunks of 256 pages to avoid OoM
            for i in range(0, buffer.shape[0], 256):
                buffer_slice = buffer[i:i + 256]
                try:
                    has_invalid_values.logical_or_(
                        torch.isnan(buffer_slice).any())
                    has_invalid_values.logical_or_(
                        torch.isinf(buffer_slice).any())
                except NotImplementedError:
                    some_checks_unavailable = True
            if fill_with_zero:
                buffer.zero_()
        torch.cuda.synchronize()

        if some_checks_unavailable:
            logger.warning(
                "`torch.isnan` or `torch.isinf` is not implemented for current kv cache dtype, related checks are skipped"
            )
        return bool(has_invalid_values)

    def get_unique_primary_pool(self) -> torch.Tensor:
        # returns the pool of memory that is allocated for this specific KVCacheManager instance
        # the pool is a list of block, each of which stores a fixed amount of KV cache data
        return self.impl.get_unique_primary_pool()

    def get_block_ids_per_seq(self, request_ids: List[int]) -> torch.Tensor:
        block_ids_per_seq = self.get_batch_cache_indices(request_ids)
        block_ids_per_seq_tensors = [
            torch.tensor(sublist, dtype=torch.int)
            for sublist in block_ids_per_seq
        ]
        padded_tensor = torch.nn.utils.rnn.pad_sequence(
            block_ids_per_seq_tensors, batch_first=True, padding_value=0)
        return padded_tensor

    def flush_iteration_events(self):
        self.impl.flush_iteration_events()

    def get_latest_events(self, timeout_ms: Optional[float] = 0):
        return self.impl.get_latest_events(timeout_ms)

    def get_kv_cache_stats(self):
        stats = self.impl.get_kv_cache_stats()
        # Subtract warmup baseline so cumulative counters only reflect
        # real inference traffic, not dummy requests from warmup.
        stats.reused_blocks -= self._warmup_reused_blocks
        stats.missed_blocks -= self._warmup_missed_blocks
        # Recompute cache hit rate from adjusted values.
        total = stats.reused_blocks + stats.missed_blocks
        stats.cache_hit_rate = (stats.reused_blocks /
                                total) if total > 0 else 0.0
        return stats

    def snapshot_warmup_baseline(self):
        """Snapshot cumulative reused and missed block counters so they can be subtracted later.

        Must be called after warmup completes so that get_kv_cache_stats()
        returns values that exclude warmup dummy requests.
        """
        raw = self.impl.get_kv_cache_stats()
        self._warmup_reused_blocks = raw.reused_blocks
        self._warmup_missed_blocks = raw.missed_blocks

    def get_iteration_stats(self):
        """Get per-iteration KV cache stats keyed by window size. Resets deltas on each call."""
        return self.impl.get_iteration_stats()

    def rewind_kv_cache(self, request: LlmRequest, rewind_len: int):
        self.impl.rewind_kv_cache(request.py_request_id, rewind_len)

    def calculate_cache_size_per_token(self,
                                       layers: Set[int],
                                       window_size: Optional[int] = None
                                       ) -> int:
        """Compute the (raw, dtype-agnostic) KV cache size per token for a set of layers.

        head_dim is resolved per-layer via ``get_pool_for_layer``: each
        layer's assigned ``PoolConfiguration`` supplies its ``head_dim``.
        When the manager runs in uniform-shape mode (no
        ``pool_configurations``), every layer uses ``self.head_dim``.

        Args:
            layers: Set of layer offsets.
            window_size: Accepted for backward compatibility; ignored.
                Per-layer pool lookup already covers the homogeneous case.

        Returns:
            cache size per token (number of elements, not bytes).
        """
        del window_size  # kept for compat; resolution is now per-layer
        if not self.pool_configurations:
            total_kv_heads = sum(self.num_kv_heads_per_layer[i] for i in layers)
            return total_kv_heads * self.kv_factor * self.head_dim

        total = 0
        for i in layers:
            pool = self.get_pool_for_layer(i)
            layer_head_dim = pool.head_dim if pool else self.head_dim
            total += self.num_kv_heads_per_layer[i] * layer_head_dim
        return total * self.kv_factor

    def _calculate_cache_bytes_per_token_for_layers(
            self,
            layers: Set[int],
            dtype_default: Optional[DataType] = None) -> int:
        """Compute KV cache bytes per token for a set of layers.

        Resolves head_dim and dtype per layer through
        ``get_pool_for_layer``; layers whose manager has no
        ``pool_configurations`` fall back to ``self.head_dim`` and
        ``dtype_default`` (or ``self.dtype``).  Handles NVFP4
        scaling-factor overhead, computed per dtype.
        """
        if dtype_default is None:
            dtype_default = self.dtype

        total_bytes = 0
        for i in layers:
            pool = self.get_pool_for_layer(i)
            layer_head_dim = pool.head_dim if pool else self.head_dim
            layer_dtype = pool.dtype if pool else dtype_default
            layer_elements = (self.num_kv_heads_per_layer[i] * self.kv_factor *
                              layer_head_dim)
            layer_bytes = get_size_in_bytes(layer_elements, layer_dtype)
            if layer_dtype == DataType.NVFP4:
                layer_bytes += KVCacheManager.calculate_scaling_factor_size_bytes(
                    layer_elements,
                    quant_vector_size=16,
                    scaling_factor_dtype=DataType.FP8)
            total_bytes += layer_bytes
        return total_bytes

    def _build_layer_to_pool_idx(self) -> Dict[int, int]:
        """Build the layer_offset -> pool_idx mapping for self.pool_configurations.

        Today, pool assignment is implicit via window_size: each pool has a
        unique window_size and a layer joins the pool whose window matches
        its effective window.  Multiple pools sharing a window_size would
        require an explicit per-layer pool index from the caller (a future
        ``pool_idx_per_layer`` ctor parameter); this helper raises instead
        of silently collapsing them.
        """
        if not self.pool_configurations:
            return {}
        window_to_pool_idx: Dict[int, int] = {}
        for idx, pc in enumerate(self.pool_configurations):
            if pc.window_size in window_to_pool_idx:
                raise RuntimeError(
                    f"Multiple PoolConfigurations share window_size={pc.window_size}. "
                    "Multi-pool-per-window requires an explicit layer->pool mapping, "
                    "which is not yet wired through KVCacheManager.__init__.")
            window_to_pool_idx[pc.window_size] = idx
        layer_offset_to_window_size = self._get_layer_offset_to_window_size()
        return {
            offset: window_to_pool_idx[w]
            for offset, w in layer_offset_to_window_size.items()
        }

    def get_pool_configuration(self, pool_idx: int) -> PoolConfiguration:
        """Return the PoolConfiguration at ``pool_idx``."""
        return self.pool_configurations[pool_idx]

    def get_pool_for_layer(self,
                           layer_offset: int) -> Optional[PoolConfiguration]:
        """Return the pool serving ``layer_offset``, or None for uniform managers.

        Layer-centric replacement for any window-keyed shape lookup: the
        returned pool's ``head_dim`` and ``dtype`` are authoritative for the
        given layer, regardless of how many pools share the layer's window.
        """
        if not self.pool_configurations:
            return None
        pool_idx = self._layer_to_pool_idx.get(layer_offset)
        if pool_idx is None:
            return None
        return self.pool_configurations[pool_idx]

    def _get_layer_offset_to_window_size(self) -> Dict[int, int]:
        """Inverse of _get_window_size_to_layers: layer_offset -> window_size.

        Asserts every local layer is mapped exactly once.
        """
        window_size_to_layers = self._get_window_size_to_layers()
        layer_offset_to_window_size: Dict[int, int] = {}
        for window_size, layer_offsets in window_size_to_layers.items():
            for layer_offset in layer_offsets:
                assert layer_offset not in layer_offset_to_window_size, (
                    f"layer_offset {layer_offset} mapped to multiple window "
                    f"sizes ({layer_offset_to_window_size[layer_offset]} and "
                    f"{window_size}) — window pattern is malformed.")
                layer_offset_to_window_size[layer_offset] = window_size
        assert len(layer_offset_to_window_size) == self.num_local_layers, (
            f"layer_offset_to_window_size covers "
            f"{len(layer_offset_to_window_size)} layers but num_local_layers "
            f"is {self.num_local_layers}.")
        return layer_offset_to_window_size

    def _get_window_size_to_layers(self) -> dict[int, list[int]]:
        """
        Get the window size to layers mapping.
        The returned map has window sizes as keys and lists of layer indices as values.

        max_attention_window_vec contains one entry per local cache layer.
        """
        window_size_to_layers_map = defaultdict(list)

        if not self.max_attention_window_vec:
            # This case should ideally be prevented by earlier config validation.
            # If num_local_layers is 0, an empty map is fine.
            if self.num_local_layers > 0:
                raise Exception(
                    "max_attention_window_vec cannot be empty if there are local layers."
                )
            return {
            }  # Return an empty dict if no local layers or if somehow vec is empty and no layers.

        if len(self.max_attention_window_vec) != self.num_local_layers:
            raise ValueError(
                "max_attention_window_vec must contain one entry per local "
                f"layer, got {len(self.max_attention_window_vec)} entries "
                f"for {self.num_local_layers} layers")
        for local_layer_idx, window_size in enumerate(
                self.max_attention_window_vec):
            window_size_to_layers_map[window_size].append(local_layer_idx)
        return window_size_to_layers_map

    def adjust_window_sizes_for_vswa(
        self,
        window_size_to_layers: Dict[int, List[int]],
        max_attention_window_vec: List[int],
        kv_cache_config: KvCacheConfig,
        pool_memory_bytes: int,
        kv_factor: int,
        dtype: DataType,
        is_cross_attention: bool = False,
        model_config: Optional[ModelConfigCpp] = None,
    ) -> Tuple[Dict[int, List[int]], List[int]]:

        assert is_cross_attention is False, 'Cross attention is not supported'

        max_tokens_from_config = kv_cache_config.max_tokens

        # Calculate the required memory bytes per sequence.  Each window's
        # bytes-per-token is computed with that window's effective head_dim
        # / dtype (per-window override map), falling back to the manager
        # scalars when no override is registered.
        required_mem_bytes_per_seq = 0
        for window_size in sorted(window_size_to_layers):
            layers = window_size_to_layers[window_size]
            cache_size_bytes_per_token = (
                self._calculate_cache_bytes_per_token_for_layers(
                    layers, dtype_default=dtype))
            required_mem_bytes_per_seq += window_size * cache_size_bytes_per_token
        logger.info(
            f'Required memory per sequence: {required_mem_bytes_per_seq} bytes')
        logger.info(f"Memory bytes in pool: {pool_memory_bytes}")

        if required_mem_bytes_per_seq < pool_memory_bytes:
            # No need to adjust the window sizes.
            logger.info("No need to adjust the window sizes, returning")
            return (copy.deepcopy(window_size_to_layers),
                    max_attention_window_vec)

        logger.info(
            f'Adjusting the window sizes {list(window_size_to_layers)} to fit '
            f'the memory {pool_memory_bytes} bytes.')
        adjusted_window_size_to_layers = {}

        remaining_mem_bytes = pool_memory_bytes
        remaining_layers = set(i for layers in window_size_to_layers.values()
                               for i in layers)

        accum_max_tokens = 0
        prev_window_size = 0
        adjusted_dict = {}
        adjusted_max_attention_window_vec = max_attention_window_vec.copy()

        for window_size in sorted(window_size_to_layers):
            layers = window_size_to_layers[window_size]
            if remaining_mem_bytes > 0 and remaining_layers:
                # Calculate cache size per token for remaining layers only.
                # ``remaining_layers`` may span multiple windows with
                # different head_dim / dtype, so the helper resolves each
                # layer's effective shape and dtype individually.
                cache_size_bytes_per_token = (
                    self._calculate_cache_bytes_per_token_for_layers(
                        remaining_layers, dtype_default=dtype))
                logger.debug(
                    f'Cache size per token for {len(remaining_layers)} layers: '
                    f'{cache_size_bytes_per_token} bytes')
                # Calculate max tokens that can fit in this window with remaining memory.
                max_tokens_in_window = min(
                    remaining_mem_bytes // cache_size_bytes_per_token,
                    window_size - prev_window_size)
                remaining_mem_bytes -= max_tokens_in_window * cache_size_bytes_per_token
                accum_max_tokens += max_tokens_in_window
                logger.debug(f'Remaining memory: {remaining_mem_bytes} bytes')
                logger.debug(
                    f'Max token of window {window_size}: {accum_max_tokens}')

                if accum_max_tokens < window_size:
                    logger.debug(
                        f'Max tokens ({accum_max_tokens}) cannot fill the current window ({window_size}). '
                        f'The larger windows will have the same max tokens.')
                    remaining_mem_bytes = 0

                # Clamp the sequence length if provided explicitly.
                if max_tokens_from_config is not None:
                    accum_max_tokens = min(max_tokens_from_config,
                                           accum_max_tokens)
                    # If max tokens from config is reached, stop allocating
                    # more memory. Since the maximum number of tokens is
                    # already reached, for the remaining windows maxTokens
                    # will be set by the current value of accumMaxTokens.
                    if accum_max_tokens == max_tokens_from_config:
                        remaining_mem_bytes = 0

            if accum_max_tokens not in adjusted_window_size_to_layers:
                adjusted_window_size_to_layers[accum_max_tokens] = layers.copy()
            else:
                adjusted_window_size_to_layers[accum_max_tokens].extend(layers)
            adjusted_dict[window_size] = accum_max_tokens
            # also update adjusted_max_attention_window_vec
            adjusted_max_attention_window_vec = [
                adjusted_dict.get(v, v)
                for v in adjusted_max_attention_window_vec
            ]

            remaining_layers -= set(layers)
            prev_window_size = window_size

        return (adjusted_window_size_to_layers,
                adjusted_max_attention_window_vec)

    def _calculate_max_num_blocks_for_linear_attention(
            self,
            kv_cache_config: KvCacheConfig,
            extra_cost_memory: int = 0) -> dict[int, tuple[int, int]]:
        """Python sizing for the unified hybrid mamba pool.

        Replaces the old ``KVCacheManagerCpp.calculate_max_num_blocks`` C++
        binding call. Uses the affine memory model::

            bytes(T) = slope * T + intercept
            slope     = attention_bytes_per_token + state_bytes / interval
            intercept = max_batch_size * #mamba_layers_local * state_bytes

        Recurrent state slots live in their own logical "window" keyed by
        ``LinearCacheType.RECURRENT_STATES``; attention KV blocks share the
        rest of the dict.
        """
        primary_budget = self._primary_pool_memory_bytes - extra_cost_memory
        state_bytes_per_layer = (
            self.linear_attention_metadata.all_recurrent_states_bytes)

        # max_attention_window_vec is already sharded by PP
        num_mamba_layers_local = self.max_attention_window_vec.count(
            LinearCacheType.RECURRENT_STATES.value)

        state_bytes_local = num_mamba_layers_local * state_bytes_per_layer

        attention_slope = self.get_cache_bytes_per_token()
        interval = self.linear_attention_metadata.states_snapshot_interval
        if interval is None or interval <= 0:
            mamba_slope = 0
        else:
            mamba_slope = state_bytes_local // interval
        slope = attention_slope + mamba_slope
        # STATIC_SLOTS_PER_REQUEST = 1 (live state); fixed-position
        # snapshots are not yet implemented.
        # With pipeline parallelism, multiple microbatches can be in-flight
        # simultaneously on the same rank, so each rank holds Mamba state for
        # up to ``max_batch_size * pp_size`` concurrent sequences. Mirror the
        # behaviour of KVCacheManagerV2 (see max_num_sequences calculation).
        pp_size = self.mapping.pp_size if self.mapping is not None else 1
        intercept = self.max_batch_size * pp_size * state_bytes_local

        if slope > 0:
            max_tokens = max((primary_budget - intercept) // slope, 0)
        elif primary_budget >= intercept:
            # With snapshots disabled, a rank containing only recurrent-state
            # layers has no per-token cache cost after its live slots are
            # allocated. Bound the otherwise-unlimited token count by the
            # configured capacity or by all resident sequences at max length.
            max_tokens = (kv_cache_config.max_tokens
                          if kv_cache_config.max_tokens is not None else
                          self.max_seq_len * self.max_batch_size * pp_size)
        else:
            max_tokens = 0
        if kv_cache_config.max_tokens is not None:
            max_tokens = min(kv_cache_config.max_tokens, max_tokens)
            if max_tokens < kv_cache_config.max_tokens:
                logger.warning(
                    f'The memory budget for Mamba + KV cache cannot fit the user-specified max_tokens of {kv_cache_config.max_tokens}. The calculated max_tokens based on the memory budget is {max_tokens}. Please consider adjusting max_batch_size/max_tokens/mamba_state_config.periodic_snapshot_interval.'
                )

        kv_blocks_in_primary_pool = int(max_tokens // self.tokens_per_block)

        # Secondary host pool is split in the same way as primary pool
        kv_blocks_in_secondary_pool = int(kv_blocks_in_primary_pool *
                                          (self._secondary_pool_memory_bytes /
                                           self._primary_pool_memory_bytes))

        # Recurrent state slot count: live state per concurrent request, with
        # extra room for one regular snapshot per snapshot interval over the
        # full token budget when block reuse is enabled.
        # With pipeline parallelism, multiple microbatches can be in-flight
        # simultaneously on the same rank, each holding up to ``max_batch_size``
        # sequences' Mamba state, so the live-state slot count must scale with
        # ``pp_size``. +1 is for the CUDA graph padding dummy.
        max_snapshots = self.max_batch_size * pp_size + 1
        if self.spec_config is not None:
            # cuda graph has different request ids for different draft len (CUDAGraphRunner::_get_padded_batch)
            # TODO: we can use a same slot for all these
            max_snapshots += self.spec_config.max_draft_len
        if (kv_cache_config.enable_block_reuse and interval is not None
                and interval > 0):
            max_snapshots += max_tokens // interval

        secondary_snapshots = int(max_snapshots *
                                  (self._secondary_pool_memory_bytes /
                                   self._primary_pool_memory_bytes))
        # Build per-window dict: each unique attention window gets the same
        # (primary, secondary) attention block count; the recurrent-states
        # sentinel gets the snapshot pool.
        blocks_per_window = {
            self.max_seq_len:
            (kv_blocks_in_primary_pool, kv_blocks_in_secondary_pool),
            LinearCacheType.RECURRENT_STATES.value:
            (max_snapshots, secondary_snapshots)
        }
        return blocks_per_window

    def calculate_max_num_blocks_for_vswa(
            self,
            kv_cache_config: KvCacheConfig,
            model_config: Optional[ModelConfigCpp] = None,
            extra_cost_memory: int = 0) -> dict[int, tuple[int, int]]:
        """
        Currently, this function is added to support *ONLY* VSWA.

        Args:
            kv_cache_config: The KV cache configuration object.
            model_config: The model configuration object.
            extra_cost_memory: Extra memory in bytes to exclude from available memory.

        Returns:
            A dict of (max_attention_window, (blocks_in_primary_pool, blocks_in_secondary_pool)).

        Environment variable TRTLLM_WINDOW_SIZE_SHARES is used to adjust the memory
        share of each window size. By default, we allocate equal proportion shares of
        memory for all window sizes (see the else case). With TRTLLM_WINDOW_SIZE_SHARES,
        we can override this behavior to adjust the memory share of each window size.

        For example, if we have window size of [512, 32768], then setting
        TRTLLM_WINDOW_SIZE_SHARES=0.4,0.6 will be allocating 40% of the memory to
        window size 512 and 60% of the memory to window size 32768.
        """

        # VSWA on Torch backend has not supported the cross attention.
        is_cross_attention = False

        window_size_to_layers = self._get_window_size_to_layers()
        logger.debug(f"window_size_to_layers: {window_size_to_layers}")

        free_mem, total_mem = torch.cuda.mem_get_info()
        # Respect max_gpu_total_bytes if provided
        free_gpu_memory_fraction = kv_cache_config.free_gpu_memory_fraction if kv_cache_config.free_gpu_memory_fraction else 0.9
        self._primary_pool_memory_bytes = kv_cache_config.max_gpu_total_bytes if kv_cache_config.max_gpu_total_bytes > 0 else int(
            free_mem * free_gpu_memory_fraction)
        self._secondary_pool_memory_bytes = kv_cache_config.host_cache_size if kv_cache_config.host_cache_size else 0
        logger.debug(
            f"primary_pool_memory_bytes is set to {self._primary_pool_memory_bytes/1024**3}GB, \n"
            f"secondary_pool_memory_bytes is set to {self._secondary_pool_memory_bytes/1024**3}GB"
        )

        if self.is_linear_attention:
            return self._calculate_max_num_blocks_for_linear_attention(
                kv_cache_config=kv_cache_config,
                extra_cost_memory=extra_cost_memory,
            )

        # VSWA case: adjust window sizes via Python helper that derives
        # head_dim from self.  model_config is no longer required because
        # head_size is read from self.head_dim, set during __init__.
        window_size_to_layers, max_attention_window_vec = self.adjust_window_sizes_for_vswa(
            window_size_to_layers=window_size_to_layers,
            max_attention_window_vec=self.max_attention_window_vec,
            kv_cache_config=kv_cache_config,
            pool_memory_bytes=self._primary_pool_memory_bytes,
            kv_factor=self.kv_factor,
            dtype=self.dtype,
            is_cross_attention=is_cross_attention,
        )
        self.max_attention_window_vec = max_attention_window_vec

        logger.info(
            f"Primary pool memory bytes: {self._primary_pool_memory_bytes}")
        logger.info(
            f"Secondary pool memory bytes: {self._secondary_pool_memory_bytes}")

        if os.getenv("TRTLLM_WINDOW_SIZE_SHARES") is not None:
            logger.info("Environment variable TRTLLM_WINDOW_SIZE_SHARES is set")
            window_size_shares = os.getenv("TRTLLM_WINDOW_SIZE_SHARES").split(
                ",")
            window_size_shares = [float(share) for share in window_size_shares]
            assert len(window_size_shares) == len(
                window_size_to_layers
            ), "Number of shares in TRTLLM_WINDOW_SIZE_SHARES must match number of window sizes"
            assert sum(
                window_size_shares
            ) == 1.0, "Sum of shares in TRTLLM_WINDOW_SIZE_SHARES must be 1.0"
        else:
            logger.info(
                "Using default allocation of equal proportion of memory to each window size"
            )
            window_size_shares = [
                1.0 / len(window_size_to_layers) for _ in window_size_to_layers
            ]

        logger.info(f"Derived window_size_shares: {window_size_shares}")

        blocks_per_window = {}
        for window_idx, (window_size, layers) in enumerate(
                sorted(window_size_to_layers.items())):
            # Per-window head_dim and dtype (with scalar fallback) — needed
            # for heterogeneous-attention models like Gemma4 where SWA and
            # full-attention pools have different head_dim.
            cache_size_bytes_per_token = (
                self._calculate_cache_bytes_per_token_for_layers(layers))

            primary_tokens = self._primary_pool_memory_bytes * window_size_shares[
                window_idx] / cache_size_bytes_per_token
            secondary_tokens = self._secondary_pool_memory_bytes * window_size_shares[
                window_idx] / cache_size_bytes_per_token

            if kv_cache_config.max_tokens is not None:
                if self.is_vswa:
                    logger.info(
                        f"kv_cache_config.max_tokens is not None ({kv_cache_config.max_tokens}) but we are operating on VSWA scheme. Ignoring the configuration."
                    )
                if not self.is_vswa:
                    logger.info(
                        f"kv_cache_config.max_tokens is {kv_cache_config.max_tokens}"
                    )
                    if kv_cache_config.max_tokens < primary_tokens:
                        logger.info(
                            f"kv_cache_config.max_tokens {kv_cache_config.max_tokens} is less than primary_tokens {primary_tokens}. Reducing primary_tokens to {kv_cache_config.max_tokens}"
                        )
                        primary_tokens = kv_cache_config.max_tokens

            primary_blocks = int(primary_tokens // self.tokens_per_block)
            secondary_blocks = int(secondary_tokens // self.tokens_per_block)
            logger.info(
                f"Window size = {window_size}, primary_blocks: {primary_blocks}, secondary_blocks: {secondary_blocks}"
            )
            blocks_per_window[window_size] = (primary_blocks, secondary_blocks)
        return blocks_per_window

    def _validate_and_adjust_attention_windows(
        self,
        max_attention_window_vec: List[int],
        blocks_per_window: BlocksPerWindow,
        tokens_per_block: int,
        max_seq_len: int,
    ) -> Tuple[BlocksPerWindow, int, List[int], Dict[int, int]]:
        """
        Validate and adjust attention windows against their upper bounds if needed.
        If there is no adjustment, the returned max_attention_window_vec will be the same as the input.

        Args:
            max_attention_window_vec: List of attention window sizes
            blocks_per_window: Dict mapping window size to (primary_blocks, secondary_blocks)
            tokens_per_block: Number of tokens per block
            max_seq_len: Maximum sequence length

        Returns:
            Tuple of (adjusted_blocks_per_window, adjusted_max_seq_len,
            adjusted_max_attention_window_vec, window_adjustments).
            window_adjustments maps pre_clamp -> post_clamp window size for
            every window that was clamped (empty if nothing was adjusted) so
            callers can rewrite their per-pool configurations to match the
            post-clamp window keys.
        """
        window_adjustments = {}
        # Validate each window size in blocks_per_window against its upper bound
        for window_size, (blocks_in_primary_pool,
                          _) in blocks_per_window.items():
            if window_size < 0:
                continue
            upper_bound = blocks_in_primary_pool * tokens_per_block
            assert upper_bound > 0, "Impossible to fit in any sequence in kvCache"
            if window_size > upper_bound:
                logger.warning(
                    f"Attention window size {window_size} exceeds upper bound {upper_bound} "
                    f"for available blocks. Reducing to {upper_bound}.")
                window_adjustments[window_size] = upper_bound
        # Apply adjustments to the window vector if any were needed
        if window_adjustments:
            adjusted_window_vec = [
                window_adjustments.get(window, window)
                for window in max_attention_window_vec
            ]
            logger.warning(
                f"Adjusted max_attention_window_vec to {adjusted_window_vec}")
            # update the window size in blocks_per_window if it is adjusted
            adjusted_blocks_per_window = {}
            for window_size, memory_pools in blocks_per_window.items():
                if window_size in window_adjustments:
                    adjusted_window_size = window_adjustments[window_size]
                    adjusted_blocks_per_window[
                        adjusted_window_size] = memory_pools
                    logger.warning(
                        f"Adjusted window size {window_size} to {adjusted_window_size} in blocks_per_window"
                    )
                else:
                    adjusted_blocks_per_window[window_size] = memory_pools
            # Update max_seq_len to the maximum of adjusted windows
            adjusted_max_seq_len = max(adjusted_window_vec)
            logger.warning(f"Adjusted max_seq_len to {adjusted_max_seq_len}")

            return adjusted_blocks_per_window, adjusted_max_seq_len, adjusted_window_vec, window_adjustments
        else:
            return blocks_per_window, max_seq_len, max_attention_window_vec, {}

    def pin_blocks(self, request_id: int):
        self.impl.pin_blocks(request_id)

    def copy_batch_block_offsets(self,
                                 dst_tensor: torch.Tensor,
                                 request_ids: List[int],
                                 beam_width: int,
                                 num_context: int,
                                 num_seqs: int,
                                 max_blocks: Optional[int] = None):
        # Fill the persistent host buffer in place, exactly as before. CPU-side
        # consumers read self.host_kv_cache_block_offsets directly and depend on
        # its persistent, max_batch-sized layout: DSA sparse attention, the
        # speculative-decoding draft/target swap, and the KV-cache relocation
        # path. Reassigning this attribute to a per-call, num_seqs-sized buffer
        # broke those readers (nvbug 6293536 regression), so it must stay the
        # buffer they see.
        if self.kv_cache_type == CacheTypeCpp.CROSS and beam_width > 1:
            # This branch is reached only via attribute aliasing, never a
            # direct cross_kv_cache_manager.copy_batch_block_offsets(...) call:
            # AttentionMetadata.create_cross_metadata() sets
            # cross_md.kv_cache_manager = cross_kv_cache_manager
            # (attention/backends/interface.py), and then
            # TrtllmAttentionMetadata.prepare() calls
            # self.kv_cache_manager.copy_batch_block_offsets(...)
            # (attention/backends/trtllm.py), which dispatches here on the
            # cross manager.
            num_gen_requests = len(request_ids) - num_context
            expected_num_seqs = num_context + num_gen_requests * beam_width
            assert num_seqs == expected_num_seqs, (
                f"Cross KV cache block offsets expected {expected_num_seqs} "
                f"decoder rows, got {num_seqs}.")

            # Cross KV is request-scoped: all decoder beams read the same
            # encoder K/V blocks. Populate one host row per request, then
            # expand generation rows across beams in the attention metadata
            # tensor whose rows are decoder-sequence scoped.
            self.impl.copy_batch_block_offsets(self.host_kv_cache_block_offsets,
                                               request_ids, 1, 0)
            # Snapshot the rows this call reads into a fresh pinned buffer so the
            # async H2D below has a private, immutable source (see the non-cross
            # path for the full rationale).
            num_rows = num_context + num_gen_requests
            host_block_offsets = self._stage_block_offsets_for_copy(num_rows)
            for pool_idx in range(self.num_pools):
                if num_context > 0:
                    dst_tensor[pool_idx, :num_context].copy_(
                        host_block_offsets[pool_idx, :num_context],
                        non_blocking=True)
                if num_gen_requests > 0:
                    gen_block_offsets = host_block_offsets[
                        pool_idx, num_context:num_context + num_gen_requests]
                    dst_tensor[pool_idx, num_context:num_seqs].copy_(
                        gen_block_offsets.repeat_interleave(beam_width, dim=0),
                        non_blocking=True)
            return

        self.impl.copy_batch_block_offsets(self.host_kv_cache_block_offsets,
                                           request_ids[:num_context], 1, 0)
        self.impl.copy_batch_block_offsets(self.host_kv_cache_block_offsets,
                                           request_ids[num_context:],
                                           beam_width, num_context)

        # The H2D copies below are asynchronous, so their source is read at copy
        # execution time, not enqueue time. With the overlap scheduler the CPU
        # runs an iteration ahead, so copying straight from the persistent buffer
        # let the next iteration's in-place refill clobber the source of this
        # iteration's still-pending H2D -> the attention kernel indexed another
        # batch's blocks (nvbug 6293536; the window is widened when host
        # offloading stalls the execution stream in front of the H2D). Stage the
        # async copy through a freshly-allocated pinned buffer instead; the
        # caching host allocator holds it until the consuming copy completes,
        # matching the already-safe kv_lens / block_ids_per_seq staging. The
        # persistent buffer above is untouched by this and stays valid for the
        # synchronous CPU readers.
        host_block_offsets = self._stage_block_offsets_for_copy(
            num_seqs, max_blocks)
        width = host_block_offsets.shape[-1]
        for pool_idx in range(self.num_pools):
            dst_tensor[pool_idx, :num_seqs, :, :width].copy_(
                host_block_offsets[pool_idx], non_blocking=True)

    def _stage_block_offsets_for_copy(
            self,
            num_rows: int,
            max_blocks: Optional[int] = None) -> torch.Tensor:
        """Snapshot the first ``num_rows`` rows of the persistent host block
        offset buffer into a fresh pinned buffer, to serve as the private source
        of an asynchronous H2D copy (nvbug 6293536).

        ``max_blocks`` bounds the copied block width. The buffer is laid out
        for max_seq_len (max_blocks_per_seq columns) but consumers only read
        each sequence's allocated block prefix, so a caller that knows the
        batch's maximum KV length can skip the unused tail — with a large
        max_seq_len the tail dominates the copy cost."""
        if max_blocks is None:
            width = self.max_blocks_per_seq
        else:
            width = min(max(max_blocks, 1), self.max_blocks_per_seq)
        host_block_offsets = torch.empty(self.num_pools,
                                         num_rows,
                                         2,
                                         width,
                                         dtype=torch.int32,
                                         pin_memory=prefer_pinned(),
                                         device='cpu')
        host_block_offsets.copy_(
            self.host_kv_cache_block_offsets[:, :num_rows, :, :width])
        return host_block_offsets

    def truncate_blocks(self, target_tokens: List[int],
                        num_tokens_to_keep: int):
        self.impl.truncate_blocks(target_tokens, num_tokens_to_keep)

    def reset_reuse_state(self):
        """Reset the reuse state of the KV cache manager."""
        self.impl.reset_reuse_state()


class NoFreeSlotsError(ValueError):
    """A slot-backed resource manager has no capacity for another request."""


class SlotManager:

    def __init__(self, max_num_requests: int):
        self.max_num_requests = max_num_requests
        self.slot_mapping = dict()
        self.free_slots = set(range(max_num_requests))

    def get_slot(self, request_id: int):
        return self.slot_mapping.get(request_id, None)

    def fill_slot_id_tensor(self, requests: List[LlmRequest],
                            slot_id_tensor: torch.Tensor):
        for i, request in enumerate(requests):
            slot_id = self.get_slot(request.request_id)
            if slot_id is not None:
                slot_id_tensor[i] = slot_id
            else:
                raise ValueError(f"Request {request.request_id} has no slot id")

    def add_slot(self, request_id: int):
        if request_id in self.slot_mapping:
            # CUDA graph dummy request could be added for different batches,
            # but we only need to reserve slot for it once.
            from .cuda_graph_runner import CUDA_GRAPH_DUMMY_REQUEST_ID
            assert request_id == CUDA_GRAPH_DUMMY_REQUEST_ID
            return self.slot_mapping[request_id]

        if len(self.free_slots) == 0:
            raise NoFreeSlotsError("No free slots")
        slot = self.free_slots.pop()
        self.slot_mapping[request_id] = slot
        return slot

    def remove_slot(self, request_id: int):
        if request_id in self.slot_mapping:
            slot = self.slot_mapping.pop(request_id)
            self.free_slots.add(slot)

    def shutdown(self):
        req_ids_list = list(self.slot_mapping.keys())
        for rid in req_ids_list:
            self.remove_slot(rid)
        assert len(self.slot_mapping) == 0 and len(
            self.free_slots) == self.max_num_requests


class BlockManager:

    def __init__(self, num_blocks: int, tokens_per_block: int):
        self.num_blocks = num_blocks
        self.tokens_per_block = tokens_per_block
        self.max_blocks_per_seq = self.num_blocks

        self.block_ids = dict()
        self.num_sequences = dict()
        self.free_blocks = deque(range(self.num_blocks))

    def add_tokens(self, request_id: int, num_tokens: int):
        if num_tokens > 0:
            if request_id not in self.block_ids:
                self.block_ids[request_id] = []
                self.num_sequences[request_id] = num_tokens
            else:
                self.num_sequences[request_id] += num_tokens
            block_count_needed = self.compute_block_count(
                self.num_sequences[request_id], self.tokens_per_block)
            if len(self.block_ids[request_id]) < block_count_needed:
                new_blocks = self._allocate_blocks(
                    block_count_needed - len(self.block_ids[request_id]))
                self.block_ids[request_id].extend(new_blocks)

    def copy_block_offsets(self, request_ids: List[int],
                           block_offsets: torch.Tensor) -> None:
        for i in range(len(request_ids)):
            block_ids = self.block_ids[request_ids[i]]
            block_num = len(block_ids)
            # Block ids are the page offsets directly; no lookup table needed.
            block_offsets[i, 0:block_num].copy_(
                torch.tensor(block_ids, dtype=torch.int32, device="cpu"))

    def compute_block_count(self, token_count: int,
                            tokens_per_page: int) -> int:
        return (token_count + tokens_per_page - 1) // tokens_per_page

    def free_resources(self, request: LlmRequest):
        request_id = request.py_request_id
        self._free_blocks(self.block_ids[request_id])
        del self.block_ids[request_id]
        del self.num_sequences[request_id]

    def rewind_cache(self, request: LlmRequest, rewind_len: int):
        if rewind_len == 0:
            return
        request_id = request.py_request_id
        self.num_sequences[request_id] -= rewind_len
        updated_token_num = max(self.num_sequences[request_id], 0)
        block_count_needed = self.compute_block_count(updated_token_num,
                                                      self.tokens_per_block)
        num_rewind_pages = len(self.block_ids[request_id]) - block_count_needed
        if num_rewind_pages > 0:
            self._free_blocks(self.block_ids[request_id][-num_rewind_pages:])
            self.block_ids[request_id] = self.block_ids[
                request_id][:-num_rewind_pages]
        return

    def _allocate_blocks(self, block_count: int) -> list:
        assert len(self.free_blocks) >= block_count, "Not enough blocks."
        blocks = [self.free_blocks.popleft() for _ in range(block_count)]
        return blocks

    def _free_blocks(self, block_list: list):
        self.free_blocks.extend(block_list)


# --------------------------------------------------------------------- #
# KV-cache compression framework (BaseResourceManager-based)             #
# --------------------------------------------------------------------- #


class KVCacheCompressionManager(BaseResourceManager):
    """Framework base for KV-cache compression methods in PyExecutor.

    Iteration-driven methods receive ResourceManager callbacks, while
    storage-bound methods provide a cold-page codec during cache construction.
    Subclasses coordinate through KVCacheManagerV2 without owning its pools,
    mappings, or migration lifecycle.
    """

    uses_iteration_lifecycle = True
    provides_cold_page_codec = False

    def __init__(self, config: "KvCacheCompressionConfig") -> None:
        self.config = config
        self.kv_cache_manager: Optional["KVCacheManagerV2"] = None
        self.draft_kv_cache_manager: Optional["KVCacheManagerV2"] = None

    def bind_kv_cache_managers(
        self,
        kv_cache_manager: "KVCacheManagerV2",
        draft_kv_cache_manager: Optional["KVCacheManagerV2"] = None,
    ) -> None:
        """Bind the target and optional draft KVCMs after their construction."""
        from .kv_cache_manager_v2 import KVCacheManagerV2

        if not isinstance(kv_cache_manager, KVCacheManagerV2):
            raise TypeError("KV-cache compression requires KVCacheManagerV2")
        if draft_kv_cache_manager is not None and not isinstance(
                draft_kv_cache_manager, KVCacheManagerV2):
            raise TypeError(
                "draft KV-cache compression requires KVCacheManagerV2")
        self.kv_cache_manager = kv_cache_manager
        self.draft_kv_cache_manager = draft_kv_cache_manager
        kv_cache_manager.kv_compression_manages_history = (
            self.config.changes_physical_kv_length)
        if draft_kv_cache_manager is not None:
            draft_kv_cache_manager.kv_compression_manages_history = (
                self.config.changes_physical_kv_length)

    @property
    def has_independent_draft_kv_cache(self) -> bool:
        return self.draft_kv_cache_manager is not None

    def create_cold_page_codec(
        self,
        cache_config: object,
        *,
        runtime_dtype: DataType,
        pp_layers: Sequence[int],
        num_kv_heads_per_layer: Sequence[int],
        head_dim_per_layer: Sequence[int],
        is_draft: bool = False,
    ) -> Optional[object]:
        """Create a native cold-page codec when the algorithm provides one."""
        return None

    def encode_cold_pages(
        self,
        codec_state: object,
        lifecycle_index: int,
        cold_base: int,
        page_indices: int,
        num_pages: int,
        stream: int,
    ) -> None:
        """Encode one complete KVCM migration batch into cold storage."""
        raise NotImplementedError

    def decode_cold_pages(
        self,
        codec_state: object,
        lifecycle_index: int,
        cold_base: int,
        page_indices: int,
        num_pages: int,
        stream: int,
    ) -> None:
        """Decode one complete KVCM migration batch from cold storage."""
        raise NotImplementedError

    # ================================================================== #
    # KV-cache lifecycle hooks (5, in temporal order).                   #
    # Subclasses override what they need; all default to no-op.          #
    # ================================================================== #

    def on_request_init(self, request: "LlmRequest", **kwargs) -> None:
        """Per-request init hook.

        Override to allocate per-request accumulators (e.g. per-request
        scoring buffers).
        """

    def on_context_step_end(self, requests: List["LlmRequest"],
                            **kwargs) -> None:
        """Fired once per iteration with the requests whose prefill finished
        (their final chunk) this step. Batched like the generation hook so a
        one-shot prefill-end eviction can process the cohort in one launch.
        """

    def on_generation_step_begin(
        self,
        scheduled_batch: "ScheduledRequests",
        **kwargs,
    ) -> None:
        """Fired once per generation step before this step's forward."""

    def on_generation_step_end(
        self,
        scheduled_batch: "ScheduledRequests",
        **kwargs,
    ) -> None:
        """Fired once per generation step, after every layer's forward
        completes. Override for periodic or budget-triggered eviction.
        """

    def on_request_finish(self, request: "LlmRequest", **kwargs) -> None:
        """Per-request finish / abort hook.

        Override to release per-request state allocated in
        ``on_request_init``. Underlying KV blocks are still freed by the
        ``KVCacheManagerV2``; subclasses must not free them here.
        """

    # ================================================================== #
    # BaseResourceManager interface — PyExecutor auto-invokes these each  #
    # iteration; they translate into the semantic lifecycle hooks above.  #
    # ================================================================== #

    def get_max_resource_count(self) -> int:
        """The compression manager does not own physical resources (the V2
        cache manager does). Returns 0 so PyExecutor's scheduler does not gate
        on us."""
        return 0

    def get_needed_resource_to_completion(self, request: "LlmRequest") -> int:
        """The compression manager does not own physical resources (the V2
        cache manager does). Returns 0 so PyExecutor's scheduler does not block
        on us."""
        return 0

    def prepare_resources(self, scheduled_batch: "ScheduledRequests") -> None:
        """Fire :meth:`on_request_init` once per request, on its first prefill
        chunk -- the same ``is_first_context_chunk`` gate ``KVCacheManager``
        uses, so no manager-side dedup bookkeeping is needed.
        """
        for req in scheduled_batch.context_requests:
            if req.is_first_context_chunk:
                self.on_request_init(req)
        self.on_generation_step_begin(scheduled_batch)

    def update_resources(
        self,
        scheduled_batch: "ScheduledRequests",
        attn_metadata: Optional["AttentionMetadata"] = None,
        kv_cache_dtype_byte_size: Optional[float] = None,
    ) -> None:
        """Fire :meth:`on_context_step_end` with the requests whose final
        prefill chunk ran this iteration, then :meth:`on_generation_step_end`.

        Uses the scheduler's ``context_requests_last_chunk`` split (computed at
        schedule time from ``is_last_context_chunk``) rather than tracking
        request-state transitions: it is iteration-exact and immune to a
        short-output request going straight to ``GENERATION_TO_COMPLETE``
        (which, under the overlap scheduler, never passes through
        ``GENERATION_IN_PROGRESS``). Signature matches the other resource
        managers so PyExecutor passes ``attn_metadata`` /
        ``kv_cache_dtype_byte_size`` through transparently.
        """
        if scheduled_batch.context_requests_last_chunk:
            self.on_context_step_end(
                scheduled_batch.context_requests_last_chunk)
        self.on_generation_step_end(scheduled_batch)

    def free_resources(self, request: "LlmRequest") -> None:
        """Fire :meth:`on_request_finish`."""
        self.on_request_finish(request)


class ResourceManager:

    # KV pools a request's block demand is charged against.
    _KV_BUDGET_RESOURCE_TYPES = (
        ResourceManagerType.KV_CACHE_MANAGER,
        ResourceManagerType.DRAFT_KV_CACHE_MANAGER,
        ResourceManagerType.CROSS_KV_CACHE_MANAGER,
    )

    def __init__(self, resource_managers: dict[ResourceManagerType,
                                               BaseResourceManager]):
        self.resource_managers = OrderedDict(resource_managers)

    def __call__(self, type: ResourceManagerType):
        return self.resource_managers[type]

    def register_resource_manager(self, type: ResourceManagerType,
                                  resource_manager: BaseResourceManager):
        self.resource_managers[type] = resource_manager

    def get_resource_manager(
            self, type: ResourceManagerType) -> Optional[BaseResourceManager]:
        return self.resource_managers.get(type)

    def get_request_kv_block_budgets(
            self,
            request: LlmRequest) -> List[Tuple[ResourceManagerType, int, int]]:
        budgets = []
        for resource_type in self._KV_BUDGET_RESOURCE_TYPES:
            kv_cache_manager = self.get_resource_manager(resource_type)
            if kv_cache_manager is None:
                continue
            budget = kv_cache_manager.get_request_kv_block_budget(request)
            if budget is not None:
                required_blocks, primary_capacity = budget
                budgets.append(
                    (resource_type, required_blocks, primary_capacity))
            else:
                logger.warning_once(
                    f"Resource manager {resource_type} does not provide a per-request KV block budget",
                    key=f"missing_kv_block_budget:{resource_type.value}",
                )
        return budgets

    def get_unchecked_kv_block_budget_pools(self) -> List[ResourceManagerType]:
        """KV pools present but excluded from the admission check, i.e. those
        whose ``get_request_kv_block_budget`` returns ``None`` for every
        request."""
        return [
            resource_type for resource_type in self._KV_BUDGET_RESOURCE_TYPES
            if (manager := self.get_resource_manager(resource_type)) is not None
            and not manager.kv_block_budget_applies()
        ]

    @nvtx_range("maybe_fit_token_budget")
    def maybe_fit_token_budget(self, scheduled_batch: ScheduledRequests):
        """Apply the post-allocation token-budget trim (#13318) to the batch.

        Shrinks over-budget context chunks so the forward pass cannot exceed
        max_num_tokens, mutating scheduled_batch in place. Driven from
        prepare_resources, after every manager has run, because that is the
        first point at which the batch's forward-pass token count is knowable:

        - The micro-batch scheduler admits the batch on estimated_reusable_tokens
          (a radix-tree guess). The real reuse is prepopulated_prompt_len,
          computed inside addSequence. #13318 is the case where the two differ,
          so no check running before addSequence can see the divergence.
        - Until setPrepopulatedPromptLen runs, context_chunk_size still spans the
          reusable prefix -- it is not a token count, and reading it as one
          over-charges a reuse hit by the whole cached prefix.

        Trimming this late is only safe because it shrinks rather than defers;
        nothing leaves the batch, so the attention-DP _can_queue vote, the PP
        inflight set and every earlier manager's per-request state all stay
        consistent with the batch they were computed from.
        """
        kv_cache_manager = self.resource_managers.get(
            ResourceManagerType.KV_CACHE_MANAGER)
        if kv_cache_manager is not None and hasattr(kv_cache_manager,
                                                    "maybe_fit_token_budget"):
            kv_cache_manager.maybe_fit_token_budget(scheduled_batch)

    @nvtx_range("prepare_resources")
    def prepare_resources(self, scheduled_batch: ScheduledRequests):
        for _, resource_manager in self.resource_managers.items():
            if hasattr(resource_manager, "prepare_resources"):
                resource_manager.prepare_resources(scheduled_batch)
        # After every manager, so context_current_position / context_chunk_size
        # are final. See maybe_fit_token_budget.
        self.maybe_fit_token_budget(scheduled_batch)
        # Strictly after the trim: the connector is told how many tokens the
        # forward pass will compute, which is only settled once the trim has
        # run. See KVCacheManager.report_batch_to_connector.
        kv_cache_manager = self.resource_managers.get(
            ResourceManagerType.KV_CACHE_MANAGER)
        if hasattr(kv_cache_manager, "report_batch_to_connector"):
            kv_cache_manager.report_batch_to_connector(scheduled_batch)

    @nvtx_range("update_resources")
    def update_resources(
        self,
        scheduled_batch: ScheduledRequests,
        attn_metadata: Optional["AttentionMetadata"] = None,
        kv_cache_dtype_byte_size: Optional[float] = None,
    ):
        for resource_type, resource_manager in self.resource_managers.items():
            if hasattr(resource_manager, "update_resources"):
                if resource_type == ResourceManagerType.KV_CACHE_MANAGER:
                    resource_manager.update_resources(scheduled_batch,
                                                      attn_metadata,
                                                      kv_cache_dtype_byte_size)
                else:
                    resource_manager.update_resources(scheduled_batch)

    def free_resources(self, request: LlmRequest):
        for resource_type, resource_manager in reversed(
                self.resource_managers.items()):
            if hasattr(resource_manager, "free_resources"):
                resource_manager.free_resources(request)

    def reorder_pipeline(self,
                         resource_manager_list: list[ResourceManagerType]):
        assert set(resource_manager_list) == set(self.resource_managers.keys())
        for resource_manager in resource_manager_list:
            self.resource_managers.move_to_end(resource_manager)


class PeftCacheManager(BaseResourceManager):

    def __init__(self,
                 peft_cache_config: PeftCacheConfig,
                 lora_config: LoraConfig,
                 model_config: ModelConfigCpp,
                 world_config: WorldConfig | None = None,
                 execution_stream: Optional[torch.cuda.Stream] = None,
                 lora_target_modules: Optional[List[str]] = None,
                 initial_data_type: Optional[torch.dtype] = None):
        import tensorrt_llm.bindings as _tb

        peft_cache_config = peft_cache_config._to_pybind()

        peft_cache_manager_config = _tb.PeftCacheManagerConfig(
            num_host_module_layer=peft_cache_config.num_host_module_layer,
            num_device_module_layer=peft_cache_config.num_device_module_layer,
            optimal_adapter_size=peft_cache_config.optimal_adapter_size,
            max_adapter_size=peft_cache_config.max_adapter_size,
            num_put_workers=peft_cache_config.num_put_workers,
            num_ensure_workers=peft_cache_config.num_ensure_workers,
            num_copy_streams=peft_cache_config.num_copy_streams,
            max_pages_per_block_host=peft_cache_config.max_pages_per_block_host,
            max_pages_per_block_device=peft_cache_config.
            max_pages_per_block_device,
            device_cache_percent=peft_cache_config.device_cache_percent,
            host_cache_size=peft_cache_config.host_cache_size,
            lora_prefetch_dir=peft_cache_config.lora_prefetch_dir,
        )

        if world_config is None:
            world_config = _tb.WorldConfig()

        BufferManager = tensorrt_llm.bindings.internal.runtime.BufferManager
        buffer_manager_stream = execution_stream.cuda_stream if execution_stream is not None else torch.cuda.current_stream(
        ).cuda_stream
        buffer_manager = BufferManager(buffer_manager_stream, True)
        logger.info(
            f"[PeftCacheManager] buffer_manager_stream: {buffer_manager_stream}"
        )
        self.impl = PeftCacheManagerCpp(config=peft_cache_manager_config,
                                        model_config=model_config,
                                        world_config=world_config,
                                        buffer_manager=buffer_manager)
        if initial_data_type is not None:
            self.impl.configure_data_type(
                torch_dtype_to_binding(initial_data_type))
        self._lora_config = lora_config
        self._lora_model_config = LoraModelConfig(
            lora_target_modules if lora_target_modules is not None else
            lora_config.lora_target_modules,
            lora_config.trtllm_modules_to_hf_modules, model_config.hidden_size,
            binding_to_str_dtype(model_config.data_type),
            lora_config.swap_gate_up_proj_lora_b_weight)
        mapping = Mapping(
            world_size=world_config.size,
            rank=world_config.rank,
            tp_size=world_config.tensor_parallelism,
            pp_size=world_config.pipeline_parallelism,
            gpus_per_node=world_config.gpus_per_node,
        )
        self._lora_manager = LoraManager(
            mapping=mapping,
            model_config=ModelConfigPython.from_model_config_cpp(model_config),
            cpp_peft_cache_manager=self.impl)

        self._batch_peft_table: Optional[Dict[int, list[
            TaskLayerModuleConfig]]] = None  # task_id -> layer-module-configs mapping for the current batch

    def get_lora_manager(self) -> LoraManager:
        return self._lora_manager

    @property
    def data_type(self) -> torch.dtype:
        return binding_to_torch_dtype(self.impl.data_type)

    def add_request_peft(self, request: LlmRequest):
        if request.lora_task_id is not None:
            is_task_cached = self.impl.is_task_cached(request.lora_task_id)
            if is_task_cached:
                # PeftCacheManager::addRequestPeft in CPP doesn't allow having only one of [config tensor, weights
                # tensor] without the other. Since there's no need for any of them when the LoRA adapter is already
                # cached, we can safely remove both from the request.
                request.remove_lora_tensors()
            elif request.lora_weights is None and request.py_lora_path:
                self._lora_manager.load_from_ckpt(
                    [request.py_lora_path],
                    model_config=self._lora_model_config,
                    uids=[request.lora_task_id],
                    ckpt_source=self._lora_config.lora_ckpt_source)
                uid = request.lora_task_id
                request.lora_weights = self._lora_manager.cpp_lora_weights[uid]
                if request.lora_config is None:
                    request.lora_config = self._lora_manager.cpp_lora_config[
                        uid]

            # PeftCacheManager CPP implementation expects an extra dim at index 0
            if request.lora_weights is not None:
                request.lora_weights = request.lora_weights.unsqueeze(0)
            if request.lora_config is not None:
                request.lora_config = request.lora_config.unsqueeze(0)
        self.impl.add_request_peft(request, True)

    def ensure_batch(self,
                     context_batch: List[LlmRequest],
                     generation_batch: List[LlmRequest],
                     reset_gpu_cache: bool = False) -> List[LlmRequest]:
        return self.impl.ensure_batch(context_batch, generation_batch,
                                      reset_gpu_cache)

    def get_max_resource_count(self) -> int:
        return 0

    def get_needed_resource_to_completion(self, request: LlmRequest) -> int:
        return 0

    def prepare_resources(self, scheduled_batch: ScheduledRequests):
        context_batch = scheduled_batch.context_requests
        generation_batch = scheduled_batch.generation_requests
        for req in context_batch:
            self.add_request_peft(req)

        self._batch_peft_table, _ = self.impl.ensure_batch_map_task_id(
            context_batch, generation_batch, False)

    def update_resources(self, scheduled_batch: ScheduledRequests):
        pass

    def free_resources(self, request: LlmRequest):
        self.impl.mark_request_done(request)

    def shutdown(self):
        pass

    def get_and_reset_batch_peft_table(
            self) -> Dict[int, list[TaskLayerModuleConfig]]:
        batch_peft_table = self._batch_peft_table
        self._batch_peft_table = None
        return batch_peft_table

    def is_task_cached_device(self, task_id: int) -> bool:
        return self.impl.is_task_cached_device(task_id)
