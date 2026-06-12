# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import copy
import enum
import hashlib
import math
import os
from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass
from typing import (TYPE_CHECKING, Dict, Iterable, List, NamedTuple, Optional,
                    Sequence, Set, Tuple, Union)

import torch
from mpi4py import MPI

import tensorrt_llm
import tensorrt_llm.bindings
from tensorrt_llm._torch.distributed.communicator import Distributed, ReduceOp
from tensorrt_llm._utils import (TensorWrapper, convert_to_torch_tensor,
                                 get_size_in_bytes, mpi_comm, mpi_disabled,
                                 prefer_pinned, torch_comm)
from tensorrt_llm.bindings.internal.batch_manager import (
    KvCacheStats, LinearAttentionMetadata, LinearCacheType)
from tensorrt_llm.bindings.internal.batch_manager.kv_cache_manager_v2_utils import (
    IndexMapper, copy_batch_block_offsets_to_device)
from tensorrt_llm.bindings.internal.runtime import TaskLayerModuleConfig
from tensorrt_llm.llmapi.llm_args import KvCacheConfig, PeftCacheConfig
from tensorrt_llm.lora_helper import LoraConfig
from tensorrt_llm.lora_manager import LoraManager, LoraModelConfig
from tensorrt_llm.runtime import ModelConfig as ModelConfigPython

# isort: off
from tensorrt_llm.runtime.kv_cache_manager_v2 import (
    DEFAULT_BEAM_INDEX, AttentionLayerConfig, BufferConfig, CacheTierConfig,
    DiskCacheTierConfig, GpuCacheTierConfig, HostCacheTierConfig, ReuseScope)
# isort: on
from tensorrt_llm.runtime.kv_cache_manager_v2 import \
    KVCacheManager as KVCacheManagerPy
from tensorrt_llm.runtime.kv_cache_manager_v2 import \
    KVCacheManagerConfig as KVCacheManagerConfigPy
from tensorrt_llm.runtime.kv_cache_manager_v2 import (LayerId, TokenIdExt,
                                                      _KVCache)
from tensorrt_llm.runtime.kv_cache_manager_v2._block_radix_tree import \
    gen_multimodal_cache_key_tokens
from tensorrt_llm.runtime.kv_cache_manager_v2._common import (BAD_PAGE_INDEX,
                                                              GPU_LEVEL,
                                                              BeamIndex)
from tensorrt_llm.runtime.kv_cache_manager_v2._config import DataRole
from tensorrt_llm.runtime.kv_cache_manager_v2._exceptions import OutOfPagesError
from tensorrt_llm.runtime.kv_cache_manager_v2._utils import (exact_div,
                                                             typed_range)
from tensorrt_llm.sampling_params import SamplingParams

from ..._utils import binding_to_str_dtype, mpi_rank, nvtx_range
from ...logger import logger
from ...mapping import CpType, Mapping
from .connectors.kv_cache_connector import KvCacheConnectorManager
from .llm_request import (LlmRequest, LlmRequestState, SamplingConfig,
                          get_draft_token_length)
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
    from tensorrt_llm._torch.attention_backend.interface import \
        AttentionMetadata

BlocksPerWindow = Dict[int, Tuple[
    int,
    int]]  # window_size -> (blocks_in_primary_pool, blocks_in_secondary_pool)


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
    PEFT_CACHE_MANAGER = "PEFT_CACHE_MANAGER"
    SEQ_SLOT_MANAGER = "SEQ_SLOT_MANAGER"
    SPEC_RESOURCE_MANAGER = "SPEC_RESOURCE_MANAGER"


class Role:
    KEY = DataRole("key")
    VALUE = DataRole("value")
    KEY_BLOCK_SCALE = DataRole("key_block_scale")
    VALUE_BLOCK_SCALE = DataRole("value_block_scale")
    ALL = DataRole("all")


class _MmRunMetadata(NamedTuple):
    """CPU tensors for exact multimodal run lookup.

    `num_runs` is the total number of flat multimodal runs across all
    multimodal items, i.e. `item_run_cu_offsets[-1]`. All members are 1-D CPU
    int64 tensors with shape `[num_runs]`.
    """

    # Shape [num_runs]; full-prompt start index for each flat run.
    run_positions: torch.Tensor
    # Shape [num_runs]; one-past-last full-prompt index for each flat run.
    run_ends: torch.Tensor
    # Shape [num_runs]; logical multimodal item index for each flat run.
    run_item_indices: torch.Tensor
    # Shape [num_runs]; item-local token offset where each flat run begins.
    run_item_offsets: torch.Tensor


def _hash_to_digest(hash_ints: Sequence[int]) -> bytes:
    # Convert 8 x int32 hash chunks to the 32-byte digest used by C++ block
    # keys. The byte order matches getNthByte(), which extracts MSB first.
    try:
        hash_len = len(hash_ints)
    except TypeError as exc:
        raise ValueError(
            "Expected 8 int32 hash values, got non-sized input") from exc
    if hash_len != 8:
        raise ValueError(f"Expected 8 int32 hash values, got {hash_len}")
    if not all(isinstance(value, int) for value in hash_ints):
        raise ValueError("Expected multimodal hash values to be integers")
    return b''.join(v.to_bytes(4, 'big', signed=True) for v in hash_ints)


def _ensure_int64_cpu_tensor(
        values: Sequence[int] | torch.Tensor) -> torch.Tensor:
    # Block-reuse augmentation is Python-side index math. The metadata is
    # produced by host mm preprocessing and carried in
    # MultimodalInput metadata. A non-CPU tensor here means the upstream
    # contract drifted and would introduce an unexpected sync in the
    # block-reuse path, so fail loudly.
    if isinstance(values, torch.Tensor):
        if values.device.type != "cpu":
            raise ValueError(
                "multimodal block-reuse metadata must be CPU-resident, "
                f"got {values.device}")
        return values.to(dtype=torch.int64)
    return torch.as_tensor(values, dtype=torch.int64)


def _resolve_multimodal_run_metadata(
        req: LlmRequest) -> Optional[_MmRunMetadata]:
    # TODO(perf): cache per request; block-reuse invokes this once per block,
    # repeatedly rebuilding identical tensors for the same request metadata.
    # Worked example for one logical multimodal item split by text:
    #
    #   prompt index: 0    1      2      3    4      5
    #   prompt token: T0   MM0:0  MM0:1  T1   MM0:2  MM0:3
    #   flat run:          run0   run0        run1   run1
    #
    # Inputs encode the prompt layout as:
    #
    #   item_run_cu_offsets = [0, 2]  # item 0 owns flat runs [0, 2)
    #   run_positions       = [1, 4]  # run0 starts at prompt 1, run1 at 4
    #   run_lengths         = [2, 2]  # token counts for run0 and run1
    #
    # The derived arrays below let the augmentation loop map any overlapping
    # flat run back to the item digest and to the item-local token offset:
    #
    #   item_run_counts        = [2]     # item 0 has two runs
    #   cumulative_run_lengths = [0,2,4] # tokens before run0, run1, and end
    #   item_starts            = [0]     # item 0 starts at flat run 0
    #   run_item_indices       = [0,0]   # run0 and run1 both belong to item 0
    #   run_item_offsets       = [0,2]   # run1 begins at MM0 token offset 2
    #
    # So a chunk slice covering prompt [4, 6) uses MM0's digest with token
    # offset 2; a slice covering prompt [5, 6) uses token offset 3.

    # Shape [num_items + 1]; item i owns flat runs [offsets[i], offsets[i + 1]).
    item_run_cu_offsets = req.multimodal_item_run_cu_offsets
    # Shape [num_runs]; full-prompt start index for each flat run.
    run_positions = req.multimodal_run_positions
    # Shape [num_runs]; token count for each flat run.
    run_lengths = req.multimodal_run_lengths

    if all(field is None
           for field in (item_run_cu_offsets, run_positions, run_lengths)):
        return None

    if (item_run_cu_offsets is None or run_positions is None
            or run_lengths is None):
        raise ValueError(
            "multimodal run metadata must be validated before block reuse and provided together"
        )

    item_run_cu_offsets = _ensure_int64_cpu_tensor(item_run_cu_offsets)
    run_positions = _ensure_int64_cpu_tensor(run_positions)
    run_lengths = _ensure_int64_cpu_tensor(run_lengths)

    # Shape [num_items]; number of flat runs owned by each logical item.
    item_run_counts = item_run_cu_offsets[1:] - item_run_cu_offsets[:-1]

    # Shape [num_runs + 1]; item-token count before each flat run, plus end.
    cumulative_run_lengths = torch.cat(
        (torch.zeros(1, dtype=torch.int64), torch.cumsum(run_lengths, dim=0)))

    # Shape [num_items]; first flat-run index for each logical item.
    item_starts = item_run_cu_offsets[:-1]

    # Shape [num_runs]; logical multimodal item index for each flat run.
    run_item_indices = torch.repeat_interleave(
        torch.arange(item_run_counts.numel(), dtype=torch.int64),
        item_run_counts)

    # Shape [num_runs]; item-local token offset where each flat run begins.
    run_item_offsets = (cumulative_run_lengths[:-1] -
                        cumulative_run_lengths[item_starts][run_item_indices])
    # Shape [num_runs]; one-past-last full-prompt index for each flat run.
    run_ends = run_positions + run_lengths

    return _MmRunMetadata(
        run_positions=run_positions,
        run_ends=run_ends,
        run_item_indices=run_item_indices,
        run_item_offsets=run_item_offsets,
    )


def _augment_tokens_with_mm_run_metadata(
        vocab_size: int, result: list[TokenIdExt],
        multimodal_hashes: Sequence[Sequence[int]], metadata: _MmRunMetadata,
        chunk_start: int, chunk_end: int) -> list[TokenIdExt]:
    # Only rewrite multimodal runs that overlap the materialized prompt slice.
    overlap_mask = ((metadata.run_ends > chunk_start)
                    & (metadata.run_positions < chunk_end))
    # Shape [num_overlapping_runs]; flat-run indices that touch this chunk.
    overlap_run_indices = torch.nonzero(overlap_mask).flatten()
    if overlap_run_indices.numel() == 0:
        return result

    # `result` is indexed relative to this chunk, but multimodal cache-key
    # tokens are indexed relative to the logical multimodal item. The rewrite
    # below therefore computes, for each selected run segment:
    # - chunk_result_offset = prompt position - chunk_start
    # - item_token_offset = item-local run start + intra-run prompt offset
    # Shape [num_overlapping_runs]; item index for each selected flat run.
    overlap_run_item_indices = metadata.run_item_indices[overlap_run_indices]
    # Materialize the selected run metadata once before the Python loop. These
    # are CPU tensors by construction, and the loop below is request hot-path
    # bookkeeping.
    overlap_run_positions = metadata.run_positions[overlap_run_indices]
    prompt_overlap_starts = torch.clamp(overlap_run_positions, min=chunk_start)
    prompt_overlap_ends = torch.clamp(metadata.run_ends[overlap_run_indices],
                                      max=chunk_end)
    item_token_offsets = (metadata.run_item_offsets[overlap_run_indices] +
                          prompt_overlap_starts - overlap_run_positions)
    chunk_result_offsets = prompt_overlap_starts - chunk_start
    lengths = prompt_overlap_ends - prompt_overlap_starts

    current_item_idx: Optional[int] = None
    digest = b""
    for item_idx, chunk_result_offset, item_token_offset, length in zip(
            overlap_run_item_indices.tolist(),
            chunk_result_offsets.tolist(),
            item_token_offsets.tolist(),
            lengths.tolist(),
            strict=True):
        if item_idx != current_item_idx:
            current_item_idx = item_idx
            digest = _hash_to_digest(multimodal_hashes[item_idx])
        # Feed the coarse item property (content digest) and granular run
        # properties (item-local offset and span length) into the key
        # generator, so cache keys reflect the actual multimodal tokens being
        # rewritten.
        result[chunk_result_offset:chunk_result_offset +
               length] = gen_multimodal_cache_key_tokens(
                   vocab_size, digest, length, token_offset=item_token_offset)

    return result


def _augment_tokens_with_contiguous_mm_metadata(
        vocab_size: int, result: list[TokenIdExt],
        multimodal_hashes: Sequence[Sequence[int]],
        multimodal_positions: Sequence[int] | torch.Tensor,
        multimodal_lengths: Sequence[int] | torch.Tensor, chunk_start: int,
        chunk_end: int) -> list[TokenIdExt]:
    # Legacy metadata assumes every item is one contiguous prompt span. Exact
    # run metadata is preferred for video layouts that interleave text tokens.
    positions = _ensure_int64_cpu_tensor(multimodal_positions)
    lengths = _ensure_int64_cpu_tensor(multimodal_lengths)

    item_ends = positions + lengths
    overlap_mask = (item_ends > chunk_start) & (positions < chunk_end)
    overlap_item_indices = torch.nonzero(overlap_mask).flatten()
    for item_idx_tensor in overlap_item_indices:
        item_idx = int(item_idx_tensor)
        pos = positions[item_idx].item()
        length = lengths[item_idx].item()
        overlap_start = max(pos, chunk_start)
        overlap_end = min(pos + length, chunk_end)
        source_offset = overlap_start - pos
        result_offset = overlap_start - chunk_start
        overlap_length = overlap_end - overlap_start
        result[result_offset:result_offset +
               overlap_length] = gen_multimodal_cache_key_tokens(
                   vocab_size,
                   _hash_to_digest(multimodal_hashes[item_idx]),
                   overlap_length,
                   token_offset=source_offset)

    return result


def compute_page_count(token_count: int, tokens_per_page: int) -> int:
    return (token_count + tokens_per_page) // tokens_per_page


class BaseResourceManager(ABC):

    @abstractmethod
    def get_max_resource_count(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_needed_resource_to_completion(self, request: LlmRequest) -> int:
        raise NotImplementedError

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


def _locate_accepted_draft_tokens(requests: List[LlmRequest]):
    num_accepted_draft_tokens = []
    accepted_draft_tokens_indices = []
    rewind_draft_token_separate_adjustments = []
    # for context requests, the py_num_accepted_draft_tokens = 0, and py_num_accepted_draft_tokens_indices = []
    for seq in requests:
        num_accepted_draft_tokens.append(seq.py_num_accepted_draft_tokens)
        rewind_draft_token_separate_adjustments.append(
            seq.py_rewind_draft_token_separate_adjustment)
        accepted_draft_tokens_indices.extend(
            seq.py_num_accepted_draft_tokens_indices)
    batch_size = len(requests)
    num_accepted_draft_tokens_offset = torch.zeros(batch_size + 1,
                                                   dtype=torch.int32,
                                                   device='cuda')
    num_accepted_draft_tokens_offset[1:] = torch.cumsum(torch.tensor(
        num_accepted_draft_tokens, dtype=torch.int32),
                                                        dim=0)
    accepted_draft_tokens_indices = torch.tensor(accepted_draft_tokens_indices,
                                                 dtype=torch.int32,
                                                 device='cuda')
    rewind_draft_token_separate_adjustments = torch.tensor(
        rewind_draft_token_separate_adjustments,
        dtype=torch.int32,
        device='cuda')
    return num_accepted_draft_tokens_offset, accepted_draft_tokens_indices, rewind_draft_token_separate_adjustments


# M-RoPE (Qwen2-VL/Qwen3-VL) splits positions across 3 axes: temporal/height/width.
_MROPE_NUM_AXES = 3


def _make_warmup_mrope_position_ids(token_num: int) -> torch.Tensor:
    """Build (_MROPE_NUM_AXES, 1, token_num) mrope_position_ids for warmup."""
    return (torch.arange(0, token_num,
                         dtype=torch.int32).expand(_MROPE_NUM_AXES, 1,
                                                   -1).clone())


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
            1, dtype=torch.int32).unsqueeze(0)
    if req.py_multimodal_data is None:
        req.py_multimodal_data = {}
    req.py_multimodal_data["mrope_config"] = mrope_config


def _update_kv_cache_draft_token_location(cache_manager,
                                          scheduled_batch: ScheduledRequests,
                                          attn_metadata: "AttentionMetadata",
                                          kv_cache_dtype_byte_size: float):
    run_kv_cache_relocation = False
    for request in scheduled_batch.generation_requests:
        if request.state != LlmRequestState.GENERATION_COMPLETE:
            if request.py_num_accepted_draft_tokens > 0 and len(
                    request.py_num_accepted_draft_tokens_indices) > 0:
                run_kv_cache_relocation = True
    if not run_kv_cache_relocation:
        return
    requests = scheduled_batch.all_requests()
    accepted_draft_token_offsets, packed_accepted_draft_tokens_indices, rewind_draft_token_separate_adjustments = _locate_accepted_draft_tokens(
        requests)
    past_key_value_lengths = attn_metadata.kv_lens_cuda[:len(requests)]
    if attn_metadata.kv_cache_block_offsets is not None and attn_metadata.host_kv_cache_pool_pointers is not None and attn_metadata.host_kv_cache_pool_mapping is not None:
        use_paged_kv_cache = True
    else:
        use_paged_kv_cache = False
    assert use_paged_kv_cache, "Only paged kv cache is supported"
    assert len(
        cache_manager.max_attention_window_vec
    ) == 1, "Currently, only one max attention window size is supported."

    if use_paged_kv_cache:
        assert len(set(cache_manager.num_kv_heads_per_layer)) == 1, \
            "update_kv_cache_draft_token_location requires uniform num_kv_heads across all layers, " \
            f"but got {cache_manager.num_kv_heads_per_layer}"
        torch.ops.tensorrt_llm.update_kv_cache_draft_token_location(
            accepted_draft_token_offsets,
            packed_accepted_draft_tokens_indices,
            past_key_value_lengths,
            True,
            cache_manager.num_layers,
            # Use TP-sharded num_kv_heads (per-rank) instead of the unsharded
            # total so the C++ kernel computes correct strides and grid dims.
            cache_manager.num_kv_heads_per_layer[0],
            int(cache_manager.head_dim * kv_cache_dtype_byte_size),
            cache_manager.max_total_draft_tokens,
            cache_manager.max_attention_window_vec[0],
            rewind_draft_token_separate_adjustments,
            None,
            cache_manager.kv_cache_pool_pointers,
            attn_metadata.kv_cache_block_offsets,
            cache_manager.max_blocks_per_seq,
            cache_manager.tokens_per_block,
            None,
        )


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
        **kwargs,
    ) -> None:
        self.mapping = mapping
        self.dtype = dtype
        self.kv_cache_type = kv_cache_type
        self.spec_config = spec_config
        self.pp_layers, self.num_layers = get_pp_layers(
            num_layers,
            mapping,
            spec_config=spec_config,
            layer_mask=layer_mask,
        )
        self.is_draft = is_draft
        self.num_local_layers = len(self.pp_layers)
        self.layer_offsets = {
            idx: offset
            for offset, idx in enumerate(self.pp_layers)
        }

        self.kv_connector_manager = kv_connector_manager

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
            num_layers=num_layers,
            layer_mask=layer_mask,
            pool_configurations=self.pool_configurations,
        )

        # Build layer -> pool_idx now that the windows agree. Stays valid
        # through the block-budget clamping below, which only rewrites
        # per-pool window_size fields; pool indices don't shift.
        self._layer_to_pool_idx = self._build_layer_to_pool_idx()

        # Determine if this is VSWA (Variable Sliding Window Attention).
        # The `w > 0` check excludes LinearCacheType.RECURRENT_STATES sentinel
        # values (negative) used by hybrid linear attention models.
        self.is_vswa = len(set(self.max_attention_window_vec)) > 1 and all(
            w > 0 for w in self.max_attention_window_vec)
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
                if kv_cache_config.enable_block_reuse:
                    max_snapshots += (
                        kv_cache_config.max_tokens //
                        linear_attention_metadata.states_snapshot_interval)

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
                if mapping.world_size > 1:
                    # make sure all ranks use the same number of primary/secondary blocks
                    if mpi_disabled():
                        for window_size, (
                                primary_blocks,
                                secondary_blocks) in blocks_per_window.items():
                            reduced_primary_blocks = torch_comm().allreduce(
                                primary_blocks,
                                op=torch.distributed.ReduceOp.MIN)
                            reduced_secondary_blocks = torch_comm().allreduce(
                                secondary_blocks,
                                op=torch.distributed.ReduceOp.MIN)
                            blocks_per_window[window_size] = (
                                reduced_primary_blocks,
                                reduced_secondary_blocks)
                    else:
                        for window_size, (
                                primary_blocks,
                                secondary_blocks) in blocks_per_window.items():
                            reduced_primary_blocks = mpi_comm().allreduce(
                                primary_blocks, op=MPI.MIN)
                            reduced_secondary_blocks = mpi_comm().allreduce(
                                secondary_blocks, op=MPI.MIN)
                            blocks_per_window[window_size] = (
                                reduced_primary_blocks,
                                reduced_secondary_blocks)
                    logger.info(
                        f"[MPI rank={mapping.rank}] Original blocks_per_window: {blocks_per_window}"
                    )
                    logger.info(
                        f"[MPI rank={mapping.rank}] Reduced blocks_per_window: {blocks_per_window}"
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
                )
                blocks_per_window = {
                    self.max_attention_window_vec[0]:
                    (self.blocks_in_primary_pool, self.blocks_in_secondary_pool)
                }

        # Validate and adjust attention windows against their upper bounds if needed
        blocks_per_window, self.max_seq_len, self.max_attention_window_vec, window_adjustments = self._validate_and_adjust_attention_windows(
            max_attention_window_vec=self.max_attention_window_vec,
            blocks_per_window=blocks_per_window,
            tokens_per_block=tokens_per_block,
            max_seq_len=self.max_seq_len,
            max_beam_width=max_beam_width,
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

        if kv_cache_type != CacheTypeCpp.SELF:
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
            'enable_partial_reuse': kv_cache_config.enable_partial_reuse,
            'copy_on_partial_reuse': kv_cache_config.copy_on_partial_reuse,
            'kv_connector_manager': self.kv_connector_manager,
            'enable_indexer_k_cache': enable_indexer_k_cache,
            'indexer_k_cache_quant_block_size':
            indexer_k_cache_quant_block_size,
            'indexer_k_cache_index_head_dim': indexer_k_cache_index_head_dim,
            'indexer_k_cache_use_fp4': indexer_k_cache_use_fp4,
            'linear_attention_metadata': linear_attention_metadata,
            # Forward the (possibly remapped) per-pool configurations.
            # window_size values are aligned with the post-clamp sizes.
            'pool_configurations': pool_configurations_cpp,
        }

        if self.event_buffer_max_size > 0:
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
        kv_cache_block_scale_pool_pointers = self.impl.get_block_scale_pool_pointers(
        )
        if kv_cache_block_scale_pool_pointers.numel() > 0:
            self.kv_cache_pool_pointers = torch.stack([
                self.kv_cache_pool_pointers, kv_cache_block_scale_pool_pointers
            ],
                                                      dim=-1)

        self.kv_cache_pool_mapping = self.impl.get_layer_to_pool_mapping()
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

    def probe_prefix_match_length(self, input_tokens, lora_task_id=None):
        """Probe the KV cache radix tree for prefix match length.

        Returns the number of prefix tokens already cached on this rank.
        Used by KVCacheAwareADPRouter for cache-aware routing.
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
        from tensorrt_llm.bindings import SamplingConfig
        from tensorrt_llm.bindings.internal.batch_manager import BlockKey
        from tensorrt_llm.bindings.internal.batch_manager import \
            LlmRequest as CppLlmRequest
        block_key = BlockKey(tokens=input_tokens, lora_task_id=lora_task_id)
        unique_tokens = block_key.unique_tokens
        dummy_req = CppLlmRequest(request_id=0,
                                  max_new_tokens=0,
                                  input_tokens=input_tokens,
                                  sampling_config=SamplingConfig(),
                                  is_streaming=False,
                                  lora_task_id=lora_task_id)
        summary = self.impl.analyze_prefix_reuse(unique_tokens, dummy_req)
        return summary.reusable_blocks_all * self.tokens_per_block

    def shutdown(self):
        self.impl.release_pools()

    def get_max_resource_count(self) -> int:
        return self.impl.max_num_blocks

    def get_num_tokens(self, request: LlmRequest) -> int:
        # LlmRequest.get_num_tokens is out of sync with GenerationRequest when overlap scheduler is enabled.
        return self.impl.get_token_count(request.py_request_id)

    def get_needed_resource_to_completion(self, request: LlmRequest) -> int:
        # TODO: the C++ implementation of this method can be used, but the
        # Python and C++ schedulers currently do not agree on what "needed
        # resource to completion" means. The C++ one excludes already allocated
        # blocks; the Python one includes them. This should be unified, but
        # the Python scheduler needs to be fixed.
        #
        # return self.impl.get_remaining_blocks_to_completion(request)
        context_token_count = request.orig_prompt_len
        num_context_blocks = context_token_count // self.tokens_per_block
        remaining_tokens = context_token_count + request.max_new_tokens - num_context_blocks * self.tokens_per_block
        need_blocks = num_context_blocks + math.ceil(
            remaining_tokens / self.tokens_per_block)
        return need_blocks

    def prepare_resources(self, scheduled_batch: ScheduledRequests):
        with request_context(self.is_draft, scheduled_batch):
            # wait for all pending work to finish before launching offload/onboarding/partial copy
            self.impl.sync_transfer_manager_with_buffer_manager()

            # Collect first-chunk requests eligible for add_sequence_batch.
            # When block reuse is enabled, addSequenceBatch uses a two-phase
            # claim-then-onboard strategy that prevents host offloading from
            # evicting reusable blocks in the radix tree.
            batch_request_infos = []
            batch_llm_requests = []
            batch_ctx_requests = []

            # allocate KV Cache
            is_star_cp = 'cp_type' in self.mapping.cp_config and CpType.STAR == self.mapping.cp_config[
                'cp_type']

            for req in scheduled_batch.context_requests:
                req_beam_width = req.py_beam_width
                if is_star_cp:
                    if req.ctx_iters == 0:
                        seq_len = sum(
                            len(ctx_block) for ctx_block in req.ctx_blocks)
                        prompt_len = seq_len + (
                            len(req.query_id) if self.mapping.cp_rank
                            == self.mapping.cp_size - 1 else 0)
                        batch_request_infos.append(
                            (req.py_request_id, prompt_len, req_beam_width))
                        batch_llm_requests.append(req)
                        batch_ctx_requests.append(req)
                else:
                    if req.is_first_context_chunk and self._kv_connector_should_add_sequence(
                            req):
                        # Batch path: two-phase claim-then-onboard
                        batch_request_infos.append(
                            (req.py_request_id, req.prompt_len, req_beam_width))
                        batch_llm_requests.append(req)
                        batch_ctx_requests.append(req)

            if batch_request_infos:
                self.impl.add_sequence_batch(batch_request_infos,
                                             batch_llm_requests)
                for req in batch_ctx_requests:
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

        if self.kv_connector_manager is not None:
            self.kv_connector_manager.build_scheduler_output(
                scheduled_batch, self)

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
        # For capturable drafting loops. During normal inference, the draft model always
        # has enough KV cache space to fit all of our draft tokens. During warmup, however,
        # we need to make the KV cache manager aware that multiple autoregressive steps will
        # occur.
        num_extra_decoding_steps: int = 0,
        draft_kv_cache_manager: Optional[BaseResourceManager] = None,
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
            sampling_params = SamplingParams(n=beam_width,
                                             best_of=beam_width,
                                             use_beam_search=beam_width > 1)
            # Here 1+max_num_draft_tokens is used to extend the prompt length to
            # a non-zero number to skip illegal memory access issue in MLA kernel
            # during warmup.
            token_num = token_nums[
                i] if token_nums is not None else 1 + max_num_draft_tokens
            # Helix active rank sets past_seen_token_num = seqlen_this_rank_cp - 1
            # in _prepare_tp_inputs; need token_num >= 2 so that doesn't go negative.
            if self.mapping.has_cp_helix():
                token_num = max(token_num, 2)
            encoder_input_tokens = [
                1
            ] * token_num if self.impl.cross_kv else None
            # Using 1 instead of 0 prevents NaN during warmup in e.g. Deepseek
            req = LlmRequest(request_id=req_id,
                             max_new_tokens=1,
                             input_tokens=[1] * token_num,
                             sampling_config=SamplingConfig(
                                 sampling_params._get_sampling_config()),
                             is_streaming=False,
                             encoder_input_tokens=encoder_input_tokens)
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

        # Use add_sequence_batch for all dummy requests, then add extra tokens.
        # This must happen before is_gen state modifications below, which may
        # set prompt_len to 0 and trigger assertion in setPrepopulatedPromptLen.
        if batch_request_infos:
            self.impl.add_sequence_batch(batch_request_infos,
                                         batch_llm_requests)
            for req_id, token_num, _ in batch_request_infos:
                for _ in range(self.num_extra_kv_tokens):
                    self.impl.add_token(req_id)
                for _ in range(num_extra_decoding_steps):
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

        return requests

    def update_resources(self,
                         scheduled_batch: ScheduledRequests,
                         attn_metadata: "AttentionMetadata" = None,
                         kv_cache_dtype_byte_size: float = None):
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
        return self.impl.remove_sequence(request.py_request_id, request,
                                         pin_on_release)

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
        num_layers: int,
        layer_mask: Optional[List[bool]],
        pool_configurations: Optional[List["PoolConfiguration"]] = None,
    ) -> List[int]:
        """Compute the per-local-layer attention window vector.

        Three input shapes are supported:

        * ``max_attention_window is None``: use ``max_seq_len`` as the only
          entry (single-window default).
        * ``len(max_attention_window) == num_layers``: the user supplied a
          global per-layer pattern. Shard it down to this PP rank using
          ``layer_mask`` + ``self.pp_layers`` / ``self.layer_offsets``,
          clamping each entry to ``max_seq_len``.
        * Otherwise: use the user-supplied vector verbatim, clamped
          element-wise to ``max_seq_len`` so the largest window can't skew
          the KV cache pool sizing.

        ``pool_configurations`` (if given) are clamped in place to the same
        ``max_seq_len`` bound so their window keys stay consistent with the
        returned vector; ``_build_layer_to_pool_idx`` relies on that match.
        """
        for pc in pool_configurations or []:
            pc.window_size = min(pc.window_size, max_seq_len)
        if kv_cache_config.max_attention_window is None:
            return [max_seq_len]
        if len(kv_cache_config.max_attention_window) == num_layers:
            if layer_mask is not None:
                global_enabled_layers = [
                    layer_idx for layer_idx in range(len(layer_mask))
                    if layer_mask[layer_idx]
                ]
            else:
                global_enabled_layers = list(range(num_layers))
            pp_rank_offset = global_enabled_layers.index(self.pp_layers[0])
            sharded = []
            for layer_idx in self.pp_layers:
                if layer_mask is not None and not layer_mask[layer_idx]:
                    continue
                window_size = kv_cache_config.max_attention_window[
                    pp_rank_offset + self.layer_offsets[layer_idx]]
                sharded.append(min(window_size, max_seq_len))
            return sharded
        # General case: clamp each user-supplied entry to max_seq_len.
        return [
            min(max_seq_len, w) for w in kv_cache_config.max_attention_window
        ]

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

    def calculate_max_num_blocks(self,
                                 kv_cache_config: KvCacheConfig,
                                 head_dim: int,
                                 tokens_per_block: int,
                                 mapping: Mapping,
                                 dtype: DataType,
                                 kv_factor: int = 2):
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

        if mapping.world_size > 1:
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

    def get_max_atten_window_upper_bound(self, blocks_in_primary_pool,
                                         tokens_per_block, max_beam_width,
                                         max_seq_len: Optional[int]):
        token_capacity = blocks_in_primary_pool * tokens_per_block
        max_blocks_per_seq = math.floor(token_capacity /
                                        (max_beam_width * tokens_per_block))
        assert max_blocks_per_seq > 0, "Impossible to fit in any sequence in kvCache"

        max_atten_window_upper_bound = max_blocks_per_seq * tokens_per_block
        if max_seq_len is not None and max_seq_len > max_atten_window_upper_bound and max_beam_width > 1:
            max_atten_window_upper_bound -= tokens_per_block
        assert max_atten_window_upper_bound > 0, "Impossible to fit in any sequence in kvCache"
        return max_atten_window_upper_bound

    def get_cache_indices(self,
                          request: LlmRequest,
                          window_size: Optional[int] = None) -> List[int]:
        if window_size is None:
            if len(self.max_attention_window_vec) > 1:
                raise ValueError("window_size must be provided for VSWA")
            window_size = self.max_attention_window_vec[0]

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
        if window_size is None:
            # ``is_vswa`` (distinct window sizes) is the real VSWA signal; a
            # uniform per-layer vector such as ``[4096, 4096, ...]`` has
            # ``len > 1`` yet a single effective window, so keying off the
            # length would spuriously reject it for connector callers that omit
            # ``window_size``.
            if self.is_vswa:
                raise ValueError("window_size must be provided for VSWA")
            window_size = self.max_attention_window_vec[0]

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
        if window_size is None:
            if len(self.max_attention_window_vec) > 1:
                raise ValueError("window_size must be provided for VSWA")
            window_size = self.max_attention_window_vec[0]
        return self.impl.get_priority_by_block_id(block_id, window_size)

    def get_batch_cache_indices(
        self,
        request_ids: List[int],
        layer_idx: Optional[int] = None,
        window_size: Optional[int] = None,
    ) -> List[List[int]]:
        if window_size is None:
            if layer_idx is None:
                if len(self.max_attention_window_vec) > 1:
                    raise ValueError(
                        "layer_idx or window_size must be provided for VSWA")
                window_size = self.max_attention_window_vec[0]
            else:
                layer_offset = self.layer_offsets[layer_idx]
                # Explicit layer_offset -> window_size mapping (no modulo
                # masking length mismatches between pattern and num_local_layers).
                window_size = self._get_layer_offset_to_window_size(
                )[layer_offset]

        result = self.impl.get_batch_cache_block_ids(request_ids, window_size)
        for i in range(len(result)):
            assert (len(result[i])) == 1
            result[i] = result[i][0]
        return result

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

        Asserts every local layer is mapped exactly once.  This is the
        explicit, length-mismatch-safe replacement for
        ``max_attention_window_vec[layer_offset % len(max_attention_window_vec)]``
        — that modulo silently masks length mismatches between the window
        pattern and num_local_layers; this helper catches them via the
        assert below.
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

        max_attention_window_vec is treated as a repeating pattern.
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

        # Treat max_attention_window_vec as a repeating pattern.
        pattern_len = len(
            self.max_attention_window_vec
        )  # `sliding_window_pattern`, in HF config terms, e.g. https://huggingface.co/google/gemma-3-1b-it/blob/main/config.json#L32
        # early return if max_attention_window_vec is a single value(SWA)
        if pattern_len == 1:
            return {
                self.max_attention_window_vec[0]:
                list(range(self.num_local_layers))
            }
        for local_layer_idx in range(self.num_local_layers):
            global_layer_idx = self.pp_layers[local_layer_idx]
            window_size = self.max_attention_window_vec[global_layer_idx %
                                                        pattern_len]
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

        max_tokens = max((primary_budget - intercept) // slope, 0)
        if kv_cache_config.max_tokens is not None:
            max_tokens = min(kv_cache_config.max_tokens, max_tokens)
            if max_tokens < kv_cache_config.max_tokens:
                logger.warning(
                    f'The memory budget for Mamba + KV cache cannot fit the user-specified max_tokens of {kv_cache_config.max_tokens}. The calculated max_tokens based on the memory budget is {max_tokens}. Please consider adjusting max_batch_size/max_tokens/mamba_state_cache_interval.'
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
        max_beam_width: int,
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
            upper_bound = self.get_max_atten_window_upper_bound(
                blocks_in_primary_pool=blocks_in_primary_pool,
                tokens_per_block=tokens_per_block,
                max_beam_width=max_beam_width,
                max_seq_len=max_seq_len)
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

    def copy_batch_block_offsets(self, dst_tensor: torch.Tensor,
                                 request_ids: List[int], beam_width: int,
                                 num_context: int, num_seqs: int):
        self.impl.copy_batch_block_offsets(self.host_kv_cache_block_offsets,
                                           request_ids[:num_context], 1, 0)
        self.impl.copy_batch_block_offsets(self.host_kv_cache_block_offsets,
                                           request_ids[num_context:],
                                           beam_width, num_context)

        for pool_idx in range(self.host_kv_cache_block_offsets.shape[0]):
            dst_tensor[pool_idx, :num_seqs].copy_(
                self.host_kv_cache_block_offsets[pool_idx, :num_seqs],
                non_blocking=True)

    def truncate_blocks(self, target_tokens: List[int],
                        num_tokens_to_keep: int):
        self.impl.truncate_blocks(target_tokens, num_tokens_to_keep)

    def reset_reuse_state(self):
        """Reset the reuse state of the KV cache manager."""
        self.impl.reset_reuse_state()


class KVCacheManagerV2(BaseResourceManager):

    def __init__(
        self,
        kv_cache_config: KvCacheConfig,
        kv_cache_type: CacheTypeCpp,
        *,
        num_layers: int,
        num_kv_heads: Union[int, List[Optional[int]]],
        head_dim: Union[int, List[int]],
        tokens_per_block: int,
        # Note that max_seq_len is not necessarily equal to kv_cache_config.num_tokens.
        # It's derived from the model's BuildConfig for consistency with the C++ backend.
        max_seq_len: int,
        max_batch_size: int,
        mapping: Mapping,
        dtype: DataType = DataType.HALF,
        spec_config=None,
        layer_mask: Optional[List[bool]] = None,
        vocab_size: int = None,
        max_num_tokens: int = 8192,
        model_config: Optional[ModelConfigCpp] = None,
        max_beam_width: int = 1,
        is_draft: bool = False,
        kv_connector_manager: Optional[KvCacheConnectorManager] = None,
        execution_stream: Optional[torch.cuda.Stream] = None,
        is_disagg: bool = False,
        **kwargs,
    ) -> None:
        self.mapping = mapping
        self.dtype = dtype
        self.is_disagg = is_disagg

        assert kv_connector_manager is None, "kv_connector_manager is not supported for KVCacheManagerV2"
        assert not (mapping.cp_config.get('cp_type') == CpType.STAR), \
            "Star attention is not supported for KVCacheManagerV2"

        self.kv_cache_type = kv_cache_type
        self.pp_layers, self.num_layers = get_pp_layers(
            num_layers,
            mapping,
            spec_config=spec_config,
            layer_mask=layer_mask,
        )
        self.is_draft = is_draft
        self.num_local_layers = len(self.pp_layers)
        self.layer_offsets = {
            idx: offset
            for offset, idx in enumerate(self.pp_layers)
        }
        self.max_beam_width = max_beam_width

        tp_size = mapping.tp_size
        if mapping.enable_attention_dp:
            tp_size = 1

        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.tokens_per_block = tokens_per_block
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size
        self.kv_factor = 1 if kv_cache_type == CacheTypeCpp.SELFKONLY else 2
        from ..speculative import get_num_extra_kv_tokens
        self.num_extra_kv_tokens = get_num_extra_kv_tokens(spec_config)
        self.max_total_draft_tokens = spec_config.max_total_draft_tokens if spec_config is not None else 0

        # Mirror V1's KV reserve sizing (see V1 __init__ for rationale).
        self._kv_reserve_draft_tokens = self.max_total_draft_tokens
        if (self.is_draft and spec_config is not None
                and getattr(spec_config, 'use_dynamic_tree', False)
                and getattr(spec_config, 'dynamic_tree_max_topK', 0) > 0):
            draft_loop_tokens = spec_config.dynamic_tree_max_topK * spec_config.max_draft_len
            self._kv_reserve_draft_tokens = max(self.max_total_draft_tokens,
                                                draft_loop_tokens)

        self.event_buffer_max_size = kv_cache_config.event_buffer_max_size

        assert self.event_buffer_max_size == 0, "event_buffer_max_size must be 0"

        self._stream = execution_stream if execution_stream is not None else torch.cuda.current_stream(
        )
        logger.info(f"[KVCacheManager] execution_stream: {self._stream}")

        # Determine max_attention_window_vec
        if kv_cache_config.max_attention_window is not None:

            self.max_attention_window_vec = kv_cache_config.max_attention_window.copy(
            )  # Make a copy to avoid modifying original
            # Clamp all window sizes to max_seq_len before calculating the
            # number of KV cache blocks. This prevents the KV cache pool from
            # being skewed by the largest window values.
            self.max_attention_window_vec = [
                min(max_seq_len, w) for w in self.max_attention_window_vec
            ]

            self.max_attention_window_vec = [
                None if w == max_seq_len else w
                for w in self.max_attention_window_vec
            ]

        else:
            self.max_attention_window_vec = [None]

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

        # Build per-layer head_dim (similar to num_kv_heads_per_layer)
        if isinstance(head_dim, int):
            self.head_dim_per_layer = [
                head_dim for _ in range(self.num_local_layers)
            ]
        else:
            assert len(head_dim) == self.num_layers, \
                f"head_dim list length ({len(head_dim)}) must match num_layers ({self.num_layers})"
            self.head_dim_per_layer = []
            if self.num_local_layers > 0:
                for i in self.pp_layers:
                    self.head_dim_per_layer.append(head_dim[i])
            if len(set(self.head_dim_per_layer)) > 1:
                logger.info(
                    f"Per-layer head_dim: {len(self.head_dim_per_layer)} layers, "
                    f"unique values={set(self.head_dim_per_layer)}")

        self.is_vswa = len(set(self.max_attention_window_vec)) > 1

        quota = float('inf')
        if kv_cache_config.max_gpu_total_bytes is not None and kv_cache_config.max_gpu_total_bytes > 0:
            quota = int(kv_cache_config.max_gpu_total_bytes)
            logger.info(
                f"max_gpu_total_bytes is provided. New quota is {quota / (1 << 30)}GiB"
            )
        if kv_cache_config.max_tokens is not None:
            quota_from_max_tokens = int(
                math.ceil(
                    self._get_quota_from_max_tokens(kv_cache_config.max_tokens)
                    / kv_cache_config.max_util_for_resume))
            quota = min(quota, quota_from_max_tokens)
            logger.info(
                f"max_tokens {kv_cache_config.max_tokens} is provided. Allowed quota from max_tokens is {quota_from_max_tokens / (1 << 30)}GiB. New quota is {quota / (1 << 30)}GiB"
            )

        assert quota != float(
            'inf'
        ), "Quota not set. Check kv_cache_config.max_tokens or kv_cache_config.max_gpu_total_bytes"

        # Sync KV cache token capacity across ranks so all ranks allocate
        # the same number of tokens and the scheduler produces identical
        # batches.  Normalize to token count before the allreduce because
        # bytes_per_token varies across PP ranks (different local layers).
        if mapping.world_size > 1:
            dist = Distributed.get(mapping)
            bytes_per_token = self.get_cache_bytes_per_token()
            max_tokens = quota / bytes_per_token
            max_tokens = dist.allreduce(max_tokens, op=ReduceOp.MIN)
            quota = max_tokens * bytes_per_token

        logger.info(
            f"KV cache manager v2 device quota set to {quota / (1 << 30)}GiB")

        cache_tiers: List[CacheTierConfig] = [GpuCacheTierConfig(quota=quota)]
        if kv_cache_config.host_cache_size is not None and kv_cache_config.host_cache_size > 0:
            host_quota = kv_cache_config.host_cache_size
        else:
            # The V2 MAX_UTILIZATION scheduler relies on suspend/resume to
            # evict and later restore KV cache pages.  Without a host tier,
            # suspended pages have nowhere to be offloaded and resume()
            # always fails, causing a scheduling deadlock where no
            # generation request can ever make progress.
            #
            # Automatically provision a host tier matching the GPU quota so
            # suspend/resume works out of the box.  Cap at available host
            # memory to avoid allocation failures.
            try:
                mem_available = os.sysconf('SC_PAGE_SIZE') * os.sysconf(
                    'SC_AVPHYS_PAGES')
            except (ValueError, OSError):
                mem_available = float('inf')
            host_quota = min(quota, int(mem_available * 0.5))
            if host_quota <= 0:
                host_quota = quota
        if host_quota > 0:
            cache_tiers.append(HostCacheTierConfig(quota=host_quota))
            logger.info(
                f"KV cache manager v2 host cache quota set to {host_quota / (1 << 30):.2f}GiB"
            )
        disk_cache_size = kv_cache_config.disk_cache_size
        if disk_cache_size is not None and disk_cache_size > 0:
            disk_cache_path = kv_cache_config.disk_cache_path
            assert disk_cache_path is not None
            cache_tiers.append(
                DiskCacheTierConfig(quota=disk_cache_size,
                                    path=disk_cache_path))
            logger.info(
                f"KV cache manager v2 disk cache quota set to {disk_cache_size / (1 << 30):.2f}GiB at {disk_cache_path}"
            )

        self.vocab_size = vocab_size

        config = self._build_cache_config(
            kv_cache_config,
            tokens_per_block=tokens_per_block,
            vocab_size=vocab_size,
            cache_tiers=cache_tiers,
        )

        self.kv_cache_manager_py_config = config

        self.impl = KVCacheManagerPy(config)

        self.num_pools = len(self.impl.layer_grouping)

        num_layers = len(config.layers)
        self.layer_to_pool_mapping_dict: dict[int, int] = {
            layer_id: self.impl.get_layer_group_id(layer_id)
            for layer_id in typed_range(LayerId(num_layers))
        }

        (self.kv_cache_pool_pointers,
         self.kv_cache_pool_mapping) = self._build_pool_mapping_tensors()

        self.kv_cache_map: dict[int, _KVCache] = {}

        # Tracks the draft length allocated by try_allocate_generation per
        # request.  Used by extend_capacity_for_tokens to compute the exact
        # padding delta instead of blindly extending, which would cause
        # unbounded capacity growth.
        self._allocated_draft_lens: dict[int, int] = {}

        # Defensive cap for get_num_available_tokens: when host cache is
        # enabled, clamp_max_seq_len_for_mem may return a value that spans
        # both GPU and host tiers.  Storing the explicit max_tokens (if set)
        # lets us cap the result to GPU-only capacity so callers like CUDA
        # graph warmup don't over-allocate beyond the GPU pool.
        # None when max_tokens is not explicitly configured — other config
        # paths (max_gpu_total_bytes, free_gpu_memory_fraction) are already
        # bounded by the GPU quota passed to GpuCacheTierConfig.
        self._gpu_max_tokens = kv_cache_config.max_tokens

        max_num_tokens = self.get_num_available_tokens(
            token_num_upper_bound=max_seq_len)

        if max_seq_len > max_num_tokens:
            logger.warning(
                f"max_seq_len {max_seq_len} is greater than max_num_tokens {max_num_tokens} that can be allocated in kv cache manager, setting max_seq_len to {max_num_tokens}"
            )
            # max_num_tokens is a float from clamp_max_seq_len_for_mem; cast
            # so downstream int-only consumers (torch.randint size, range)
            # stay int.
            self.max_seq_len = int(max_num_tokens)

        # Pad max_blocks_per_seq to next multiple of 4 (copy_block_offsets kernel).
        # Account for max single-sequence capacity = seq_len + extra KV tokens +
        # _kv_reserve_draft_tokens (see __init__) + 1 base decode token.
        max_seq_capacity = self.max_seq_len + self.num_extra_kv_tokens + self._kv_reserve_draft_tokens + 1
        self.max_blocks_per_seq = (max_seq_capacity + tokens_per_block -
                                   1) // tokens_per_block
        if self.max_blocks_per_seq % 4 != 0:
            self.max_blocks_per_seq = ((self.max_blocks_per_seq + 3) // 4) * 4

        self.enable_block_reuse = kv_cache_config.enable_block_reuse
        self.enable_partial_reuse = kv_cache_config.enable_partial_reuse

        # With pipeline parallelism, multiple microbatches can be in-flight
        # simultaneously, so we need slots for all concurrent sequences.
        # Plus 1 for cuda graph dummy request.
        # In disaggregated mode, use a coefficient of 2: at any moment up to
        # `max_num_sequences` requests can be actively generating while another
        # up to `max_num_sequences` requests are still in KV transfer
        # (TRANS_IN_PROGRESS) and continue to hold their index slots. The 2x
        # capacity lets the next batch of active requests acquire slots without
        # waiting for the previous batch's transfers to finish.
        max_num_sequences = max_batch_size * mapping.pp_size
        index_mapper_capacity = max_num_sequences * (2 if is_disagg else 1) + 1
        logger.info(
            f"KVCacheManagerV2: IndexMapper capacity={index_mapper_capacity} "
            f"(max_num_sequences={max_num_sequences}, is_disagg={is_disagg}, max_beam_width={max_beam_width})"
        )
        self.index_mapper = IndexMapper(index_mapper_capacity, max_beam_width)
        self._early_freed_index_requests: set[int] = set()
        self.index_scales = torch.empty(self.num_pools,
                                        dtype=torch.int32,
                                        pin_memory=prefer_pinned(),
                                        device='cpu')
        self.kv_offset = torch.empty(self.num_pools,
                                     dtype=torch.int32,
                                     pin_memory=prefer_pinned(),
                                     device='cpu')
        for pool_id in range(self.num_pools):
            layer_id = self.impl.layer_grouping[pool_id][0]
            self.index_scales[pool_id] = self.impl.get_page_index_scale(
                layer_id, Role.KEY)
            if self.kv_cache_type != CacheTypeCpp.SELFKONLY:
                self.kv_offset[pool_id] = exact_div(
                    self.impl.get_mem_pool_base_address(layer_id, Role.VALUE) -
                    self.impl.get_mem_pool_base_address(layer_id, Role.KEY),
                    self.impl.get_page_stride(layer_id, Role.KEY))
            else:
                self.kv_offset[pool_id] = 0

        # Keep unused block offsets as safe block index 0.
        self.host_kv_cache_block_offsets = torch.zeros(
            self.num_pools,
            index_mapper_capacity * max_beam_width,
            2,  # key and value
            self.max_blocks_per_seq,
            dtype=torch.int32,
            pin_memory=prefer_pinned(),
            device='cpu')

    def _get_quota_from_max_tokens(self, max_tokens: int) -> int:
        return int(max_tokens * self.get_cache_bytes_per_token())

    def _build_pool_mapping_tensors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        kv_cache_pool_pointers = torch.tensor([[
            self.impl.get_mem_pool_base_address(
                self.impl.layer_grouping[pool_id][0], Role.KEY), 0
        ] for pool_id in range(self.num_pools)],
                                              dtype=torch.int64,
                                              device="cpu",
                                              pin_memory=prefer_pinned())

        if self.dtype == DataType.NVFP4:
            kv_cache_pool_pointers = torch.stack([
                kv_cache_pool_pointers,
                torch.tensor([[
                    self.impl.get_mem_pool_base_address(
                        self.impl.layer_grouping[pool_id][0],
                        Role.KEY_BLOCK_SCALE), 0
                ] for pool_id in range(self.num_pools)],
                             dtype=torch.int64,
                             device="cpu",
                             pin_memory=prefer_pinned())
            ],
                                                 dim=-1)

        kv_cache_pool_mapping_list = []
        for layer_id in typed_range(LayerId(self.num_local_layers)):
            layer_group_id = self.impl.get_layer_group_id(layer_id)
            if self.dtype != DataType.NVFP4:
                addr_offset = self.impl.get_mem_pool_base_address(
                    layer_id, Role.KEY) - int(
                        kv_cache_pool_pointers[layer_group_id][0])
            else:
                addr_offset = self.impl.get_mem_pool_base_address(
                    layer_id, Role.KEY) - int(
                        kv_cache_pool_pointers[layer_group_id][0][0])
                block_scale_addr_offset = self.impl.get_mem_pool_base_address(
                    layer_id, Role.KEY_BLOCK_SCALE) - int(
                        kv_cache_pool_pointers[layer_group_id][0][1])
                block_scale_offset = exact_div(
                    block_scale_addr_offset,
                    self.get_layer_bytes_per_token(
                        layer_id, Role.KEY_BLOCK_SCALE) * self.kv_factor *
                    self.tokens_per_block)
            offset = exact_div(
                addr_offset,
                self.get_layer_bytes_per_token(layer_id, Role.KEY) *
                self.kv_factor * self.tokens_per_block)

            if self.dtype == DataType.NVFP4:
                assert block_scale_offset == offset, "Block scale offset and offset should be the same"

            kv_cache_pool_mapping_list.append([layer_group_id, offset])

        kv_cache_pool_mapping = torch.tensor(kv_cache_pool_mapping_list,
                                             dtype=torch.int32,
                                             device="cpu",
                                             pin_memory=prefer_pinned())
        return kv_cache_pool_pointers, kv_cache_pool_mapping

    def _build_cache_config(
        self,
        kv_cache_config: KvCacheConfig,
        *,
        tokens_per_block: int,
        vocab_size: Optional[int],
        cache_tiers: List[CacheTierConfig],
    ) -> KVCacheManagerConfigPy:
        buffer_type = [Role.KEY]
        if self.kv_cache_type != CacheTypeCpp.SELFKONLY:
            buffer_type.append(Role.VALUE)
        if kv_cache_config.dtype == "nvfp4":
            for layer_idx, hd in enumerate(self.head_dim_per_layer):
                assert hd % 2 == 0, \
                    f"head_dim must be divisible by 2 for nvfp4 kv cache, but layer {layer_idx} has head_dim={hd}"
            buffer_type.append(Role.KEY_BLOCK_SCALE)
            if self.kv_cache_type != CacheTypeCpp.SELFKONLY:
                buffer_type.append(Role.VALUE_BLOCK_SCALE)

        enable_partial_reuse = (kv_cache_config.enable_partial_reuse
                                and self.max_beam_width == 1)

        return KVCacheManagerConfigPy(
            tokens_per_block=tokens_per_block,
            vocab_size=vocab_size,
            cache_tiers=cache_tiers,
            max_util_for_resume=kv_cache_config.max_util_for_resume,
            enable_partial_reuse=enable_partial_reuse,
            layers=[
                AttentionLayerConfig(
                    layer_id=layer_id,
                    buffers=[
                        BufferConfig(
                            role=role,
                            size=self.get_layer_bytes_per_token(
                                local_layer_idx=layer_id, data_role=role) *
                            tokens_per_block,
                        ) for role in buffer_type
                    ],
                    sliding_window_size=self.max_attention_window_vec[
                        self.pp_layers[layer_id] %
                        len(self.max_attention_window_vec)],
                    num_sink_tokens=None,
                ) for layer_id in typed_range(LayerId(self.num_local_layers))
            ],
        )

    @property
    def blocks_in_primary_pool(self) -> int:
        """
        Get the number of blocks in the primary pool.
        """
        return self.impl.get_page_index_upper_bound(0, Role.KEY)

    def get_buffers(self,
                    layer_idx: int,
                    kv_layout: str = "NHD") -> Optional[torch.Tensor]:
        layer_offset = self.layer_offsets[layer_idx]
        addr_key = self.impl.get_mem_pool_base_address(layer_offset, Role.KEY)
        if self.kv_cache_type != CacheTypeCpp.SELFKONLY:
            addr_value = self.impl.get_mem_pool_base_address(
                layer_offset, Role.VALUE)
            page_size_key = self.impl.get_page_stride(layer_offset, Role.KEY)
            page_size_value = self.impl.get_page_stride(layer_offset,
                                                        Role.VALUE)

            assert addr_key + page_size_value == addr_value and page_size_key == page_size_value

        assert kv_layout in ["NHD",
                             "HND"], f"Unsupported kv_layout: {kv_layout}"

        element_per_container = 1
        dtype = self.dtype
        if dtype == DataType.NVFP4:
            element_per_container = 2
            dtype = torch.int8

        layer_head_dim = self.head_dim_per_layer[layer_offset]
        if kv_layout == "NHD":
            shape = [
                self.impl.get_page_index_upper_bound(layer_offset, Role.KEY) //
                self.kv_factor,
                self.kv_factor,
                self.tokens_per_block,
                self.num_kv_heads_per_layer[layer_offset],
                layer_head_dim // element_per_container,
            ]
        else:
            shape = [
                self.impl.get_page_index_upper_bound(layer_offset, Role.KEY) //
                self.kv_factor,
                self.kv_factor,
                self.num_kv_heads_per_layer[layer_offset],
                self.tokens_per_block,
                layer_head_dim // element_per_container,
            ]

        return convert_to_torch_tensor(TensorWrapper(
            addr_key,
            dtype,
            shape,
        ))

    def get_num_available_tokens(self,
                                 *,
                                 token_num_upper_bound: int,
                                 batch_size: int = 1,
                                 max_num_draft_tokens: int = 0) -> int:
        extra_tokens = self.num_extra_kv_tokens + max_num_draft_tokens
        # Token num upper bound is the maximum number of tokens that can be allocated in the kv cache manager.
        # We need to add extra tokens to the token num upper bound to account for the extra tokens.
        clamped = self.impl.clamp_max_seq_len_for_mem(
            batch_size, token_num_upper_bound + extra_tokens) - extra_tokens
        # clamp_max_seq_len_for_mem considers all tiers (GPU + host).  When
        # max_tokens is explicitly set, cap by GPU-only capacity so callers
        # (e.g. CUDA graph warmup) don't exceed the GPU pool.
        if self._gpu_max_tokens is not None:
            clamped = min(clamped, self._gpu_max_tokens - extra_tokens)
        return clamped

    def get_num_free_blocks(self) -> int:
        # NOTE This method is used to get the number of blocks in the primary pool not the FREE blocks.
        # However, since we only use this function when the kv cache manager is empty, so it is safe to do so.
        assert len(
            self.kv_cache_map
        ) == 0, "get_num_free_blocks is only used when the kv cache manager is empty"
        max_num_pages = max([
            self.impl.get_page_index_upper_bound(layer_id, Role.KEY)
            for layer_id in typed_range(LayerId(self.num_local_layers))
        ])
        return max_num_pages // self.kv_factor

    # ---- Scheduling API (called by KVCacheV2Scheduler) ----

    def is_request_active(self, request_id: int) -> bool:
        """Return True if *request_id* has a live, non-suspended KV cache."""
        kv_cache = self.kv_cache_map.get(request_id)
        return kv_cache is not None and kv_cache.is_active

    def _required_gen_capacity(self, req: LlmRequest,
                               current_capacity: int) -> int:
        """Compute generation KV cache capacity for a request.

        Grows *current_capacity* by 1 + draft tokens.
        """
        draft_len = get_draft_token_length(req)
        return current_capacity + 1 + draft_len

    def try_allocate_generation(self, req: LlmRequest) -> bool:
        """Try to allocate one additional KV cache slot for a generation request.

        Resumes from suspended state if needed, then resizes capacity by 1 (+
        draft tokens). Returns True on success, False if allocation failed.
        """
        kv_cache = self.kv_cache_map.get(req.py_request_id)
        if kv_cache is None:
            return False

        if not kv_cache.is_active:
            if not kv_cache.resume(self._stream.cuda_stream):
                return False
            self._restore_page_index_bufs(req.py_request_id, kv_cache)

        if not self._ensure_generation_beam_width(req, kv_cache):
            return False

        draft_len = get_draft_token_length(req)
        self._allocated_draft_lens[req.py_request_id] = draft_len
        return kv_cache.resize(
            self._required_gen_capacity(req, kv_cache.capacity))

    def revert_allocate_generation(self, req: LlmRequest) -> None:
        """Undo the capacity growth from try_allocate_generation.

        When attention DP causes can_queue=False after scheduling, the
        forward pass is skipped but the scheduler already grew each
        generation request's KV cache capacity by 1 (+draft tokens).
        This method shrinks capacity back to undo that spurious growth
        so it does not accumulate across iterations and overflow the
        host page-index buffer.
        """
        kv_cache = self.kv_cache_map.get(req.py_request_id)
        if kv_cache is None or not kv_cache.is_active:
            return
        draft_len = get_draft_token_length(req)
        reverted_cap = kv_cache.capacity - 1 - draft_len
        if reverted_cap < 0:
            return
        if not kv_cache.resize(reverted_cap):
            raise RuntimeError(
                f"Failed to revert KV cache capacity for request "
                f"{req.py_request_id} from {kv_cache.capacity} to "
                f"{reverted_cap}")

    def revert_allocate_context(self, req: LlmRequest) -> None:
        """Undo the capacity growth from this iter's ``resize_context``.

        When delay batching (``_balance_adp_requests`` /
        ``_waiting_requests``) defers a context request after V2
        scheduling, the forward pass is skipped for that request but the
        scheduler already grew its KV cache capacity to cover the chunk.
        This shrinks capacity back to the pre-resize value so the
        freshly-allocated pages can be reused during the wait window —
        important for long contexts where one deferred request can hold
        GBs of KV.
        """
        pre_cap = getattr(req, "py_ctx_pre_resize_cap", None)
        if pre_cap is None:
            return
        # Mark as consumed even if the resize below is skipped, so a
        # later iter does not see a stale snapshot.
        req.py_ctx_pre_resize_cap = None
        kv_cache = self.kv_cache_map.get(req.py_request_id)
        if kv_cache is None or not kv_cache.is_active:
            return
        if pre_cap >= kv_cache.capacity:
            return
        if not kv_cache.resize(pre_cap):
            raise RuntimeError(
                f"Failed to revert KV cache capacity for context "
                f"request {req.py_request_id} from "
                f"{kv_cache.capacity} to {pre_cap}")
        if pre_cap > 0:
            kv_cache.suspend()

    def _set_page_index_bufs(self, request_id: int, kv_cache: _KVCache) -> None:
        assert kv_cache.beam_width <= self.max_beam_width
        index = self.index_mapper.get_index(request_id)
        for i in range(int(kv_cache.beam_width)):
            for pool_idx in range(self.num_pools):
                buffer: torch.Tensor = self.host_kv_cache_block_offsets[
                    pool_idx, index * self.max_beam_width + i, 0]
                kv_cache.set_base_page_index_buf(BeamIndex(i), pool_idx,
                                                 memoryview(buffer.numpy()))

    def _ensure_generation_beam_width(self, req: LlmRequest,
                                      kv_cache: _KVCache) -> bool:
        target_beam_width = BeamIndex(req.py_beam_width)
        assert 1 <= target_beam_width <= self.max_beam_width
        if kv_cache.beam_width == target_beam_width:
            return True

        try:
            kv_cache.beam_width = target_beam_width
        except OutOfPagesError:
            return False

        self._set_page_index_bufs(req.py_request_id, kv_cache)
        return True

    def _restore_page_index_bufs(self, request_id: int,
                                 kv_cache: _KVCache) -> None:
        """Re-connect host page-index buffers after resume().

        suspend() clears the base_page_index_buf pointers (sets them to
        None) so the KV cache stops writing page indices to the host
        buffer.  After resume(), the KV cache has re-locked pages but
        copy_batch_block_offsets still reads from the host buffer, so we
        must re-connect the buffers to avoid stale/zero page indices that
        would cause illegal memory accesses during the forward pass.
        """
        self._set_page_index_bufs(request_id, kv_cache)

    def _resume_and_restore(self, req_id: int, kv_cache) -> bool:
        """Resume a suspended KV cache and restore its page index buffers.

        Returns True if the cache is (or becomes) active, False on failure.
        """
        if kv_cache.is_active:
            return True
        if not kv_cache.resume(self._stream.cuda_stream):
            return False
        self._restore_page_index_bufs(req_id, kv_cache)
        return True

    def prepare_context(self, req: LlmRequest) -> bool:
        """Create _KVCache, handle block reuse, and resume. Does NOT resize.

        For first chunk: creates _KVCache (with block reuse lookup if enabled),
        sets context_current_position, and resumes from suspended state.
        For subsequent chunks: verifies existing cache is active.
        Returns True on success, False if preparation failed.
        """
        if req.is_first_context_chunk:
            kv_cache = self.kv_cache_map.get(req.py_request_id)
            if kv_cache is None:
                # Last token cannot be recovered, so we don't include it in
                # the input tokens to look up for the block that can be reused.
                if self.enable_block_reuse:
                    all_tokens = req.get_tokens(DEFAULT_BEAM_INDEX)
                    tokens = self._augment_tokens_for_block_reuse(
                        all_tokens, req, end=len(all_tokens) - 1)
                else:
                    tokens = None
                kv_cache = self._create_kv_cache(req.py_request_id,
                                                 req.lora_task_id,
                                                 tokens,
                                                 cache_salt=req.cache_salt)
                if kv_cache is None:
                    return False
                kv_cache.cuda_stream = self._stream.cuda_stream

            if not self.enable_block_reuse:
                kv_cache.stop_committing()
            else:
                req.context_current_position = kv_cache.num_committed_tokens
                req.set_prepopulated_prompt_len(kv_cache.num_committed_tokens,
                                                self.tokens_per_block)

            return self._resume_and_restore(req.py_request_id, kv_cache)
        else:
            # Subsequent chunk: cache must exist from first chunk.
            # It may be suspended (e.g., evicted between chunks), so
            # _resume_and_restore handles reactivation.
            kv_cache = self.kv_cache_map.get(req.py_request_id)
            assert kv_cache is not None, (
                f"KV cache missing for non-first context chunk, request {req.py_request_id}"
            )
            return self._resume_and_restore(req.py_request_id, kv_cache)

    def resize_context(self, req: LlmRequest, num_tokens: int) -> bool:
        """Resize KV cache to cover context_current_position + num_tokens.

        num_tokens is the number of tokens to be processed (i.e.,
        context_remaining_length or a chunk thereof). The target capacity is
        computed as context_current_position + num_tokens so that block reuse
        overlaps with existing capacity are handled correctly.
        Returns True on success, False if resize failed (first chunk is
        suspended on failure).

        Snapshots the pre-resize capacity on ``req.py_ctx_pre_resize_cap``
        when growth happens so ``revert_allocate_context`` can undo it if
        delay batching defers the request.
        """
        kv_cache = self.kv_cache_map.get(req.py_request_id)
        if kv_cache is None:
            return False

        target = req.context_current_position + num_tokens + self.num_extra_kv_tokens
        capacity = max(kv_cache.capacity, target)
        pre_cap = kv_cache.capacity

        if not kv_cache.resize(capacity):
            if req.is_first_context_chunk:
                kv_cache.suspend()
            return False

        # None means "no growth this iter, nothing to revert"; this also
        # invalidates a stale snapshot from a prior iter on the same req.
        req.py_ctx_pre_resize_cap = pre_cap if capacity > pre_cap else None
        return True

    def extend_capacity_for_tokens(self, request: LlmRequest) -> None:
        """Extend KV cache capacity for the CUDA-graph padding delta.

        ``try_allocate_generation`` allocated capacity for the schedule-reduced
        draft length.  After padding restores ``py_draft_tokens`` to the static
        max, we must extend by exactly the difference so that the subsequent
        rewind (which operates on the padded length) does not underflow.

        The delta is computed from ``_allocated_draft_lens`` (recorded by
        ``try_allocate_generation``) vs the current draft length (post-padding).
        """
        allocated = self._allocated_draft_lens.pop(request.py_request_id, None)
        if allocated is None:
            return
        current_draft_len = get_draft_token_length(request)
        delta = current_draft_len - allocated
        if delta <= 0:
            return
        kv_cache = self.kv_cache_map[request.py_request_id]
        new_capacity = kv_cache.capacity + delta
        success = kv_cache.resize(new_capacity)
        if not success:
            raise ValueError(
                f"Failed to extend capacity of KV cache for request "
                f"{request.py_request_id} by {delta} tokens "
                f"(target capacity {new_capacity})")

    def suspend_request(self, req: LlmRequest) -> None:
        """Suspend a request's KV cache (move to host tier)."""
        kv_cache = self.kv_cache_map.get(req.py_request_id)
        if kv_cache is not None and kv_cache.is_active:
            kv_cache.suspend()

    def resume_request(self, req: LlmRequest) -> bool:
        """Resume a previously-suspended KV cache for *req*.

        Returns True if the cache is (or becomes) active on GPU, False if
        resume was refused (e.g. GPU pressure above max_util_for_resume)
        or no cache exists for the request.
        """
        kv_cache = self.kv_cache_map.get(req.py_request_id)
        if kv_cache is None:
            return False
        return self._resume_and_restore(req.py_request_id, kv_cache)

    # ---- prepare_resources ----

    @nvtx_range("prepare_resources_kv_cache_manager_v2")
    def prepare_resources(self, scheduled_batch: ScheduledRequests):
        if self.is_draft:
            # Draft V2 manager: mirror the main manager by creating/resizing
            # KV caches for scheduled requests (the main V2 scheduler does not
            # know about the draft manager).
            self._prepare_draft_resources(scheduled_batch)
            return

    def _prepare_draft_resources(self, scheduled_batch: ScheduledRequests):
        """Create/resize KV caches in the draft V2 manager for scheduled requests.

        The main V2 scheduler only manages the primary KV cache manager.
        The draft manager must mirror context/generation allocations so that
        its IndexMapper contains the correct request IDs for
        copy_batch_block_offsets().
        """
        with request_context(True, scheduled_batch):
            for req in scheduled_batch.context_requests:
                kv_cache = self.kv_cache_map.get(req.py_request_id)
                if kv_cache is None:
                    kv_cache = self._create_kv_cache(req.py_request_id,
                                                     req.lora_task_id,
                                                     None,
                                                     cache_salt=req.cache_salt)
                    kv_cache.stop_committing()
                if not self._resume_and_restore(req.py_request_id, kv_cache):
                    raise RuntimeError(
                        f"Failed to resume draft KV cache for request {req.py_request_id}"
                    )
                draft_len = get_draft_token_length(req)
                capacity = (req.context_current_position +
                            req.context_chunk_size + draft_len +
                            self.num_extra_kv_tokens)
                if not kv_cache.resize(capacity):
                    raise RuntimeError(
                        f"Draft KV cache context resize failed for request "
                        f"{req.py_request_id}: could not resize to {capacity} tokens"
                    )

            for req in scheduled_batch.generation_requests:
                kv_cache = self.kv_cache_map.get(req.py_request_id)
                if kv_cache is None:
                    raise RuntimeError(
                        f"Missing draft KV cache for generation request {req.py_request_id}"
                    )
                if not self._resume_and_restore(req.py_request_id, kv_cache):
                    raise RuntimeError(
                        f"Failed to resume draft KV cache for request {req.py_request_id}"
                    )
                if not self._ensure_generation_beam_width(req, kv_cache):
                    raise RuntimeError(
                        f"Failed to expand draft KV cache beam width for request {req.py_request_id}"
                    )
                new_cap = self._required_gen_capacity(req, kv_cache.capacity)
                # Pad the resize up to _kv_reserve_draft_tokens (see __init__);
                # no-op when reserve == draft_token_length.
                reserve_slack = (self._kv_reserve_draft_tokens -
                                 get_draft_token_length(req))
                if reserve_slack > 0:
                    new_cap += reserve_slack
                if not kv_cache.resize(new_cap):
                    raise RuntimeError(
                        f"Draft KV cache generation resize failed for request "
                        f"{req.py_request_id}: could not resize to {new_cap} tokens"
                    )

    def _augment_tokens_for_block_reuse(
            self,
            tokens: Sequence[int],
            req: LlmRequest,
            start: int = 0,
            end: int | None = None) -> Sequence[TokenIdExt]:
        """Augment token sequence with multimodal content digests for block reuse.

        Multimodal placeholder tokens (e.g. image_token_id) share the same ID
        regardless of the underlying content. This method replaces each
        multimodal token region with TokenIdExt values produced by
        gen_multimodal_cache_key_tokens(), embedding the content digest
        (Blake3 hash) into the token sequence so that the radix tree can
        distinguish blocks belonging to different images/videos.

        When *start*/*end* are given, they define the chunk bounds; only
        `tokens[start:end]` is materialized and returned. This avoids
        re-augmenting the full prompt on every chunk during chunked prefill.

        For text-only requests this is a no-op.
        """
        if end is None:
            end = len(tokens)
        chunk_start = start
        chunk_end = end
        is_sliced = chunk_start != 0 or chunk_end != len(tokens)

        if (req.multimodal_hashes is None or req.multimodal_positions is None
                or req.multimodal_lengths is None):
            return tokens[chunk_start:chunk_end] if is_sliced else tokens

        result: list[TokenIdExt] = list(tokens[chunk_start:chunk_end])
        run_metadata = _resolve_multimodal_run_metadata(req)
        if run_metadata is not None:
            return _augment_tokens_with_mm_run_metadata(self.vocab_size, result,
                                                        req.multimodal_hashes,
                                                        run_metadata,
                                                        chunk_start, chunk_end)

        return _augment_tokens_with_contiguous_mm_metadata(
            self.vocab_size, result, req.multimodal_hashes,
            req.multimodal_positions, req.multimodal_lengths, chunk_start,
            chunk_end)

    def get_kv_cache_stats(self):
        kv_cache_stats = KvCacheStats()
        kv_cache_stats.allocated_bytes = self.impl.get_quota(GPU_LEVEL)

        return kv_cache_stats

    def get_iteration_stats(self):
        """V2 does not support per-iteration stats yet."""
        return None

    def get_block_ids_per_seq(self, request_ids: List[int]) -> torch.Tensor:
        block_ids_per_seq = self.get_batch_cache_indices(request_ids)
        block_ids_per_seq_tensors = [
            torch.tensor([
                i // self.num_local_layers if i != BAD_PAGE_INDEX else 0
                for i in sublist
            ],
                         dtype=torch.int) for sublist in block_ids_per_seq
        ]
        padded_tensor = torch.nn.utils.rnn.pad_sequence(
            block_ids_per_seq_tensors, batch_first=True, padding_value=0)
        return padded_tensor

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
            kv_reserve_draft_tokens: Optional[int] = None,
            use_mrope: bool = False,
            max_beam_width: int = 1,
            num_extra_decoding_steps: int = 0,
            draft_kv_cache_manager: Optional['BaseResourceManager'] = None):
        _kv_draft = kv_reserve_draft_tokens if kv_reserve_draft_tokens is not None else max_num_draft_tokens

        beam_width = max_beam_width
        requests = []

        def release_resources(current_request: LlmRequest,
                              free_draft_resources: bool = False) -> None:
            for req in requests:
                self.free_resources(req)
            self.free_resources(current_request)
            if draft_kv_cache_manager is not None:
                for req in requests:
                    draft_kv_cache_manager.free_resources(req)
                if free_draft_resources:
                    draft_kv_cache_manager.free_resources(current_request)

        for i, req_id in enumerate(request_ids):
            # exact choice of n can be ignored for dummy requests
            sampling_params = SamplingParams(n=beam_width,
                                             best_of=beam_width,
                                             use_beam_search=beam_width > 1)
            # Here 1+max_num_draft_tokens is used to extend the prompt length to
            # a non-zero number to skip illegal memory access issue in MLA kernel
            # during warmup.
            token_num = token_nums[
                i] if token_nums is not None else 1 + max_num_draft_tokens
            # token_num - 1 is the past history length in generation.
            history_hint = max(0, token_num - 1) if is_gen else None
            # TODO: support cross attention
            encoder_input_tokens = None
            # Using 1 instead of 0 prevents NaN during warmup in e.g. Deepseek
            input_tokens = [1 for _ in range(token_num)]
            req = LlmRequest(request_id=req_id,
                             max_new_tokens=1,
                             input_tokens=input_tokens,
                             sampling_config=SamplingConfig(
                                 sampling_params._get_sampling_config()),
                             is_streaming=False,
                             encoder_input_tokens=encoder_input_tokens)
            req.is_dummy_request = True
            req.paged_kv_block_ids = []
            if prepare_resource:
                # Dummy/warmup request. ``stop_committing()`` below blocks all
                # writes to the radix tree, so the choice of branch does not
                # affect committed state. ``cache_salt`` is left defaulted
                # to None to avoid coupling synthetic data to any salted branch.
                kv_cache = self._create_kv_cache(req.py_request_id,
                                                 req.lora_task_id, input_tokens)
                assert kv_cache.num_committed_tokens == 0
                success = kv_cache.resume(self._stream.cuda_stream)
                if not success:
                    release_resources(req)
                    return None
                kv_cache.stop_committing()
                dummy_capacity = token_num + self.num_extra_kv_tokens + num_extra_decoding_steps
                # Need to hint the committed history to activate stale-block
                # optimization and match the solver's pool budget.
                success = kv_cache.resize(dummy_capacity,
                                          history_length=history_hint)
                if not success:
                    release_resources(req)
                    return None
                draft_kv_cache = None
                if draft_kv_cache_manager is not None:
                    draft_kv_cache = draft_kv_cache_manager._create_kv_cache(
                        req.py_request_id, req.lora_task_id, input_tokens)
                    # Dummy path: see comment above, no salt.
                    success = draft_kv_cache.resume(
                        draft_kv_cache_manager._stream.cuda_stream)
                    if not success:
                        release_resources(req, free_draft_resources=True)
                        return None
                    draft_kv_cache.stop_committing()
                    success = draft_kv_cache.resize(dummy_capacity)
                    if not success:
                        release_resources(req, free_draft_resources=True)
                        return None

            if is_gen:
                req.state = LlmRequestState.GENERATION_IN_PROGRESS
                req.prompt_len = token_num - 1
                req.py_prompt_len = req.prompt_len
                req.py_draft_tokens = [1] * max_num_draft_tokens
                if prepare_resource:
                    if not self._ensure_generation_beam_width(req, kv_cache):
                        release_resources(req,
                                          free_draft_resources=draft_kv_cache
                                          is not None)
                        return None
                    new_capacity = kv_cache.capacity + _kv_draft + 1
                    success = kv_cache.resize(new_capacity,
                                              history_length=history_hint)
                    if not success:
                        release_resources(req,
                                          free_draft_resources=draft_kv_cache
                                          is not None)
                        return None
                    if draft_kv_cache is not None:
                        if not draft_kv_cache_manager._ensure_generation_beam_width(
                                req, draft_kv_cache):
                            release_resources(req, free_draft_resources=True)
                            return None
                        success = draft_kv_cache.resize(new_capacity)
                        if not success:
                            release_resources(req, free_draft_resources=True)
                            return None

            if use_mrope:
                _populate_dummy_mrope_config(req, token_num, is_gen)
            requests.append(req)

        return requests

    def try_commit_blocks_for_reuse(self, request: LlmRequest,
                                    kv_cache) -> None:
        if (self.enable_block_reuse and not self.is_draft
                and not request.is_dummy_request
                and request.context_current_position
                > kv_cache.num_committed_tokens):
            tokens = self._augment_tokens_for_block_reuse(
                request.get_tokens(DEFAULT_BEAM_INDEX),
                request,
                start=kv_cache.num_committed_tokens,
                end=request.context_current_position)
            if tokens:
                kv_cache.commit(tokens)
            kv_cache.stop_committing()

    def release_index_slot(self, request_id: int) -> None:
        """Release IndexMapper slot early while keeping KV cache blocks allocated.

        After prefill completes on a context-only worker, the IndexMapper slot
        (used for host_kv_cache_block_offsets during model forward) is no longer
        needed.  Releasing it early allows new requests to be scheduled while
        the KV cache blocks are still being transferred via NIXL/UCX.
        """
        self.index_mapper.remove_sequence(request_id)
        self._early_freed_index_requests.add(request_id)

    def free_resources(self, request: LlmRequest, pin_on_release: bool = False):
        self._allocated_draft_lens.pop(request.py_request_id, None)
        kv_cache = self.kv_cache_map.pop(request.py_request_id, None)
        if kv_cache is None:
            return
        self.try_commit_blocks_for_reuse(request, kv_cache)
        kv_cache.close()
        if request.py_request_id in self._early_freed_index_requests:
            self._early_freed_index_requests.discard(request.py_request_id)
        else:
            self.index_mapper.remove_sequence(request.py_request_id)

    def get_batch_cache_indices(
            self,
            request_ids: List[int],
            layer_idx: Optional[int] = None) -> List[List[int]]:
        if layer_idx is None:
            pool_id = 0
        else:
            pool_id = self.layer_to_pool_mapping_dict[
                self.layer_offsets[layer_idx]]
        return self._get_batch_cache_indices_by_pool_id(request_ids,
                                                        pool_id=pool_id,
                                                        is_kv_aggregate=True)

    def _get_batch_cache_indices_by_pool_id(
            self,
            request_ids: List[int],
            *,
            pool_id: int = 0,
            is_kv_aggregate: bool = True) -> List[List[int]]:

        if is_kv_aggregate:
            # Div by kv_factor to index kv cache with size [num_blocks, kv_factor, tokens_per_block, num_kv_heads, head_dim]
            div_factor = self.kv_factor
        else:
            div_factor = 1

        res = []

        for req_id in request_ids:
            idx_tensor = torch.as_tensor(
                self.kv_cache_map[req_id].get_base_page_indices(pool_id))
            res.append((torch.where(
                idx_tensor != BAD_PAGE_INDEX,
                idx_tensor * self.index_scales[pool_id] // div_factor,
                BAD_PAGE_INDEX)).tolist())

        return res

    def get_cache_bytes_per_token(self) -> int:
        data_roles = [Role.KEY]
        if self.kv_cache_type != CacheTypeCpp.SELFKONLY:
            data_roles.append(Role.VALUE)
        if self.dtype == DataType.NVFP4:
            data_roles.append(Role.KEY_BLOCK_SCALE)
            if self.kv_cache_type != CacheTypeCpp.SELFKONLY:
                data_roles.append(Role.VALUE_BLOCK_SCALE)

        return sum(
            self.get_layer_bytes_per_token(local_layer_idx=local_layer_idx,
                                           data_role=data_role)
            for local_layer_idx in range(self.num_local_layers)
            for data_role in data_roles)

    def get_layer_bytes_per_token(self, local_layer_idx: int, data_role: Role):
        if self.dtype not in (
                DataType.FP8,
                DataType.HALF,
                DataType.BF16,
                DataType.FLOAT,
                DataType.NVFP4,
        ):
            raise ValueError(f"Cannot support {self.dtype} KV cache.")

        if data_role == Role.ALL:
            kv_factor = self.kv_factor
        elif data_role in [
                Role.KEY, Role.VALUE, Role.KEY_BLOCK_SCALE,
                Role.VALUE_BLOCK_SCALE
        ]:
            if data_role in [Role.KEY_BLOCK_SCALE, Role.VALUE_BLOCK_SCALE]:
                assert self.dtype == DataType.NVFP4, "NVFP4 is the only supported dtype for block quant data roles"
            if data_role == Role.VALUE:
                assert self.kv_cache_type != CacheTypeCpp.SELFKONLY, "VALUE data role is not supported for SELFKONLY cache type"
            kv_factor = 1
        else:
            raise ValueError(f"Invalid data role: {data_role}")

        cache_size_per_token = kv_factor * self.num_kv_heads_per_layer[
            local_layer_idx] * self.head_dim_per_layer[local_layer_idx]

        cache_size_bytes_per_token = get_size_in_bytes(cache_size_per_token,
                                                       self.dtype)

        if data_role in [Role.KEY, Role.VALUE]:
            return cache_size_bytes_per_token

        quant_size_per_token = 0

        if self.dtype == DataType.NVFP4:
            quant_size_per_token = self.calculate_scaling_factor_size_bytes(
                cache_size_per_token,
                quant_vector_size=16,
                scaling_factor_dtype=DataType.FP8,
            )

        if data_role in [Role.KEY_BLOCK_SCALE, Role.VALUE_BLOCK_SCALE]:
            return quant_size_per_token

        # Role.ALL combines both
        return cache_size_bytes_per_token + quant_size_per_token

    @staticmethod
    def calculate_scaling_factor_size_bytes(
            cache_size: int, quant_vector_size: int,
            scaling_factor_dtype: DataType) -> int:
        assert cache_size % quant_vector_size == 0, "NVFP4 cache size must be divisible by quant vector size"
        return get_size_in_bytes(cache_size // quant_vector_size,
                                 scaling_factor_dtype)

    def check_invalid_values_in_kv_cache(self,
                                         fill_with_zero: bool = False) -> bool:
        some_checks_unavailable = False
        has_invalid_values = torch.tensor([False],
                                          dtype=torch.bool,
                                          device=torch.cuda.current_device())
        pool_handled = set()

        # Handle each layer from start to end to traverse the whole KV cache.
        for layer_id, layer_offset in self.layer_offsets.items():
            pool_id = self.layer_to_pool_mapping_dict[layer_offset]
            if pool_id in pool_handled:
                continue
            buffer = self.get_buffers(layer_id)
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
            pool_handled.add(pool_id)
        torch.cuda.synchronize()

        if some_checks_unavailable:
            logger.warning(
                "`torch.isnan` or `torch.isinf` is not implemented for current kv cache dtype, related checks are skipped"
            )
        return bool(has_invalid_values)

    def shutdown(self):
        for kv_cache in self.kv_cache_map.values():
            kv_cache.close()
        self.kv_cache_map.clear()
        self.impl.shutdown()

    def get_max_resource_count(self) -> int:
        # TODO: implement this
        return 1

    def get_needed_resource_to_completion(self, request: LlmRequest) -> int:
        # TODO: implement this
        # context_token_count = request.orig_prompt_len
        # num_context_blocks = context_token_count // self.tokens_per_block
        # remaining_tokens = context_token_count + request.max_new_tokens - num_context_blocks * self.tokens_per_block
        # need_blocks = num_context_blocks + math.ceil(
        #     remaining_tokens / self.tokens_per_block)
        # return need_blocks
        return 0

    # TODO: refactor get_cache_size_per_token and get_cache_bytes_per_token to use the same logic
    @staticmethod
    def get_cache_size_per_token(model_config: ModelConfigPython,
                                 mapping: Mapping,
                                 num_layers: Optional[int] = None,
                                 **kwargs):
        # get kv cache dtype bytes
        mem_per_token = 2
        quant_config = model_config.quant_config
        if quant_config is not None and quant_config.quant_mode.has_fp8_kv_cache(
        ):
            mem_per_token = 1

        # get num key value heads
        config = model_config.pretrained_config
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
        mem_per_token *= num_attention_layers * head_dim

        # K and V
        mem_per_token *= kv_factor
        return mem_per_token

    def update_resources(self,
                         scheduled_batch: ScheduledRequests,
                         attn_metadata: "AttentionMetadata" = None,
                         kv_cache_dtype_byte_size: float = None):
        if not self.is_draft:
            _update_kv_cache_draft_token_location(self, scheduled_batch,
                                                  attn_metadata,
                                                  kv_cache_dtype_byte_size)
        for req in scheduled_batch.context_requests:
            if req.py_request_id not in self.kv_cache_map:
                continue
            kv_cache = self.kv_cache_map[req.py_request_id]
            # In the overlap scheduler, iteration N+1's eviction may
            # suspend a ctx request's KV cache while iteration N's
            # update_resources still needs to process it.  Skip the
            # resize — the request will be resumed by the scheduler
            # on the next iteration.
            if not kv_cache.is_active:
                continue
            if self.enable_block_reuse and not self.is_draft and not req.is_dummy_request:
                if req.context_current_position > kv_cache.num_committed_tokens:
                    tokens = self._augment_tokens_for_block_reuse(
                        req.get_tokens(DEFAULT_BEAM_INDEX),
                        req,
                        start=kv_cache.num_committed_tokens,
                        end=req.context_current_position)
                    kv_cache.commit(tokens)
                if req.context_remaining_length == 0:
                    kv_cache.stop_committing()
            else:
                success = kv_cache.resize(None, req.context_current_position)
                if not success:
                    raise ValueError(
                        f"Failed to resize history length of KV cache for request {req.py_request_id} to {req.context_current_position} tokens at context update"
                    )

        for req in scheduled_batch.generation_requests:
            if req.py_request_id not in self.kv_cache_map:
                continue
            kv_cache = self.kv_cache_map[req.py_request_id]
            # In the overlap scheduler, the scheduler for iteration N+1
            # may suspend a gen request's KV cache (via self-eviction or
            # victim eviction) while iteration N's update_resources still
            # needs to process it. Skip suspended caches — the request
            # will be resumed by the scheduler on the next iteration.
            if not kv_cache.is_active:
                continue
            new_capacity = None if req.state in (
                LlmRequestState.GENERATION_COMPLETE,
                LlmRequestState.CONTEXT_INIT
            ) else kv_cache.capacity - req.py_rewind_len
            success = kv_cache.resize(new_capacity, req.max_beam_num_tokens - 1)
            if not success:
                raise ValueError(
                    f"Failed to resize KV cache for request {req.py_request_id} to capacity {new_capacity} and history length {req.max_beam_num_tokens - 1} tokens at generation update"
                )

    def copy_batch_block_offsets(self, dst_tensor: torch.Tensor,
                                 request_ids: List[int], beam_width: int,
                                 num_contexts: int, num_seqs: int):
        assert beam_width <= self.max_beam_width

        copy_idx = self.index_mapper.get_copy_index(request_ids, num_contexts,
                                                    beam_width)
        assert copy_idx.shape[0] == num_seqs

        copy_batch_block_offsets_to_device(self.host_kv_cache_block_offsets,
                                           dst_tensor, copy_idx,
                                           self.index_scales, self.kv_offset,
                                           self._stream.cuda_stream)

    def _create_kv_cache(self,
                         request_id: int,
                         lora_task_id: int | None,
                         input_tokens: Sequence[TokenIdExt] | None,
                         cache_salt: str | None = None):
        assert request_id not in self.kv_cache_map, f"KV cache for request {request_id} already exists"
        if self.index_mapper.num_free_slots() == 0:
            logger.warning(
                "No free IndexMapper slots for request %s "
                "(%d/%d slots in use, likely held by DISAGG_GENERATION_TRANS_IN_PROGRESS requests). "
                "Skipping KV cache creation; request will retry next iteration.",
                request_id, self.index_mapper.size(), self.index_mapper.size())
            return None
        # ReuseScope.salt is int|None; derive a deterministic int from the
        # cache_salt string so the same string yields the same reuse namespace
        # across processes (matches C++ blockKey hashing on cacheSalt).
        salt_int = (int.from_bytes(
            hashlib.sha256(cache_salt.encode("utf-8")).digest()[:8], "little")
                    if cache_salt is not None else None)
        kv_cache = self.impl.create_kv_cache(
            ReuseScope(lora_id=lora_task_id, salt=salt_int),
            input_tokens,
        )
        self.kv_cache_map[request_id] = kv_cache
        self.index_mapper.add_new_sequence(request_id)
        self._set_page_index_bufs(request_id, kv_cache)
        return kv_cache

    def reset_reuse_state(self):
        self.impl.clear_reusable_blocks()


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
            raise ValueError("No free slots")
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

        self.base_block_offsets = torch.arange(self.num_blocks,
                                               device="cpu",
                                               dtype=torch.int32)

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
            block_offsets[i, 0:block_num].copy_(
                self.base_block_offsets[torch.tensor(block_ids,
                                                     dtype=torch.int32,
                                                     device="cpu")])

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


class ResourceManager:

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

    @nvtx_range("prepare_resources")
    def prepare_resources(self, scheduled_batch: ScheduledRequests):
        for _, resource_manager in self.resource_managers.items():
            if hasattr(resource_manager, "prepare_resources"):
                resource_manager.prepare_resources(scheduled_batch)

    @nvtx_range("update_resources")
    def update_resources(
        self,
        scheduled_batch: ScheduledRequests,
        attn_metadata: Optional["AttentionMetadata"] = None,
        kv_cache_dtype_byte_size: Optional[float] = None,
    ):
        for _, resource_manager in self.resource_managers.items():
            if hasattr(resource_manager, "update_resources"):
                if isinstance(resource_manager, KVCacheManager):
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
                 lora_target_modules: Optional[List[str]] = None):
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
