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
import hashlib
import math
import os
from collections import OrderedDict, defaultdict
from dataclasses import fields
from typing import TYPE_CHECKING, Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple, Union

import torch
from strenum import StrEnum

from tensorrt_llm._torch.distributed.communicator import Distributed, ReduceOp
from tensorrt_llm._utils import (
    TensorWrapper,
    convert_to_torch_tensor,
    get_size_in_bytes,
    prefer_pinned,
)
from tensorrt_llm.bindings.internal.batch_manager import KvCacheIterationStats, KvCacheStats
from tensorrt_llm.bindings.internal.batch_manager.kv_cache_manager_v2_utils import (
    IndexMapper,
    copy_batch_block_offsets_to_device,
)
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.runtime.kv_cache_hash import get_effective_kv_cache_event_hash_algo
from tensorrt_llm.runtime.kv_cache_manager_v2 import (
    DEFAULT_BEAM_INDEX,
    AttentionLayerConfig,
    BufferConfig,
    CacheTierConfig,
    DiskCacheTierConfig,
    GpuCacheTierConfig,
    HostCacheTierConfig,
    KVCacheIterationStatsDelta,
    LayerId,
    PageIndexMode,
    ReuseScope,
    SwaScratchReuseConfig,
    TokenIdExt,
    _KVCache,
)
from tensorrt_llm.runtime.kv_cache_manager_v2 import KVCacheManager as KVCacheManagerPy
from tensorrt_llm.runtime.kv_cache_manager_v2 import KVCacheManagerConfig as KVCacheManagerConfigPy
from tensorrt_llm.runtime.kv_cache_manager_v2._block_radix_tree import (
    gen_multimodal_cache_key_tokens,
)
from tensorrt_llm.runtime.kv_cache_manager_v2._common import (
    BAD_PAGE_INDEX,
    CACHE_LEVEL1,
    GPU_LEVEL,
    CacheLevel,
)
from tensorrt_llm.runtime.kv_cache_manager_v2._config import DataRole
from tensorrt_llm.runtime.kv_cache_manager_v2._event_manager import KVCacheEventManager
from tensorrt_llm.runtime.kv_cache_manager_v2._exceptions import CuError
from tensorrt_llm.runtime.kv_cache_manager_v2._exceptions import (
    OutOfMemoryError as KVCacheOutOfMemoryError,
)
from tensorrt_llm.runtime.kv_cache_manager_v2._life_cycle_registry import AttnLifeCycle, LifeCycleId
from tensorrt_llm.runtime.kv_cache_manager_v2._utils import exact_div, typed_range
from tensorrt_llm.sampling_params import SamplingParams

from ..._utils import binding_to_torch_dtype, mpi_rank, nvtx_range, str_dtype_to_torch
from ...logger import logger
from ...mapping import CpType, Mapping
from ..utils import maybe_compile
from .connectors.kv_cache_connector import KvCacheConnectorManager
from .kv_cache_stats import (
    KVCacheV2IterationStatsReport,
    KVCacheV2LifeCycleIterationStats,
    KVCacheV2PoolGroupIterationStats,
)
from .llm_request import LlmRequest, LlmRequestState, SamplingConfig, get_draft_token_length
from .resource_manager import (
    BaseResourceManager,
    CacheTypeCpp,
    DataType,
    KVCacheManager,
    ModelConfigCpp,
    ModelConfigPython,
    _populate_dummy_mrope_config,
    get_pp_layers,
    request_context,
)
from .scheduler import ScheduledRequests

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention_backend.interface import AttentionMetadata

KV_CACHE_ITERATION_STATS_DELTA_FIELDS = tuple(
    field.name for field in fields(KVCacheIterationStatsDelta)
)
KV_CACHE_ITERATION_STATS_REUSE_FIELDS = (
    "iter_reused_blocks",
    "iter_full_reused_blocks",
    "iter_partial_reused_blocks",
    "iter_missed_blocks",
)
KV_CACHE_ITERATION_STATS_POOL_GROUP_FIELDS = tuple(
    field_name
    for field_name in KV_CACHE_ITERATION_STATS_DELTA_FIELDS
    if field_name not in KV_CACHE_ITERATION_STATS_REUSE_FIELDS
)


class Role:
    KEY = DataRole("key")
    VALUE = DataRole("value")
    KEY_BLOCK_SCALE = DataRole("key_block_scale")
    VALUE_BLOCK_SCALE = DataRole("value_block_scale")
    # Sparse-attention per-layer index-K cache (MiniMax-M3 and similar
    # sparse-block-selection backends). Registered as a native V2
    # BufferConfig on sparse layers via the extra_buffers_per_layer hook on
    # _build_cache_config, so allocation, free, slot reuse, and prefix
    # reuse share the lifecycle of the main K/V buffers for the same layer.
    INDEX_KEY = DataRole("index_key")
    ALL = DataRole("all")


class BlockReusePolicy(StrEnum):
    ALL_REUSABLE = "all_reusable"
    PER_REQUEST = "per_request"


def _estimate_full_attn_size_per_token(
    layer_sizes: Sequence[int], attention_windows: Sequence[Optional[int]]
) -> int:
    return sum(
        layer_size
        for layer_size, window_size in zip(layer_sizes, attention_windows)
        if window_size is None or window_size <= 0
    )


def _estimate_swa_cache_size(
    layer_sizes: Sequence[int],
    attention_windows: Sequence[Optional[int]],
    tokens_per_block: int,
    *,
    context: bool,
    scratch: bool,
) -> tuple[int, int]:
    tokens_per_block = int(tokens_per_block)
    size_per_token = 0
    size_per_request = 0
    scratch_keys = set()
    for layer_size, window_size in zip(layer_sizes, attention_windows):
        if window_size is not None and window_size > 0:
            window_tokens = math.ceil(window_size / tokens_per_block) * tokens_per_block
            if not context:
                size_per_request += window_tokens * layer_size
            elif not scratch:
                size_per_token += layer_size
            else:
                scratch_key = (int(window_size), layer_size)
                if scratch_key in scratch_keys:
                    size_per_request += window_tokens * layer_size
                else:
                    scratch_keys.add(scratch_key)
                    size_per_token += layer_size
    return size_per_token, size_per_request


def _get_static_cache_size_layer_components(
    model_config: ModelConfigPython,
    mapping: Mapping,
    num_layers: Optional[int] = None,
    **kwargs,
) -> tuple[List[int], List[Optional[int]]]:
    config = model_config.pretrained_config
    max_seq_len = kwargs.get("max_seq_len")
    kv_cache_config = kwargs.get("kv_cache_config")

    num_key_value_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
    if isinstance(num_key_value_heads, Iterable):
        num_key_value_heads = sum(num_key_value_heads) / len(num_key_value_heads)

    mla = hasattr(config, "kv_lora_rank") and config.kv_lora_rank is not None
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

    cache_size_per_token = kv_factor * head_dim
    quant_config = model_config.quant_config
    if quant_config is not None and quant_config.quant_mode.has_fp8_kv_cache():
        layer_size = cache_size_per_token
    elif quant_config is not None and quant_config.quant_mode.has_fp4_kv_cache():
        layer_size = math.ceil(cache_size_per_token / 2) + math.ceil(cache_size_per_token / 16)
    else:
        assert quant_config is None or (not quant_config.quant_mode.has_kv_cache_quant()), (
            "Quantized kv cache is not expected"
        )
        layer_size = cache_size_per_token * 2

    num_attention_layers = KVCacheManager._resolve_num_attention_layers(
        model_config, mapping, num_layers
    )
    layer_sizes = [layer_size] * num_attention_layers
    window_pattern = kv_cache_config.max_attention_window if kv_cache_config is not None else None

    def get_window_size(layer_idx: int) -> Optional[int]:
        if window_pattern is None or not isinstance(window_pattern, (list, tuple)):
            return None
        window_size = window_pattern[layer_idx % len(window_pattern)]
        if window_size is None or window_size <= 0:
            return None
        window_size = int(window_size)
        if max_seq_len is not None and window_size == int(max_seq_len):
            return None
        return window_size

    attention_windows = [get_window_size(layer_idx) for layer_idx in range(num_attention_layers)]
    return layer_sizes, attention_windows


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
        raise ValueError("Expected 8 int32 hash values, got non-sized input") from exc
    if hash_len != 8:
        raise ValueError(f"Expected 8 int32 hash values, got {hash_len}")
    if not all(isinstance(value, int) for value in hash_ints):
        raise ValueError("Expected multimodal hash values to be integers")
    return b"".join(v.to_bytes(4, "big", signed=True) for v in hash_ints)


def _ensure_int64_cpu_tensor(values: Sequence[int] | torch.Tensor) -> torch.Tensor:
    # Block-reuse augmentation is Python-side index math. The metadata is
    # produced by host mm preprocessing and carried in
    # MultimodalInput metadata. A non-CPU tensor here means the upstream
    # contract drifted and would introduce an unexpected sync in the
    # block-reuse path, so fail loudly.
    if isinstance(values, torch.Tensor):
        if values.device.type != "cpu":
            raise ValueError(
                f"multimodal block-reuse metadata must be CPU-resident, got {values.device}"
            )
        return values.to(dtype=torch.int64)
    return torch.as_tensor(values, dtype=torch.int64)


def _resolve_multimodal_run_metadata(req: LlmRequest) -> Optional[_MmRunMetadata]:
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

    if all(field is None for field in (item_run_cu_offsets, run_positions, run_lengths)):
        return None

    if item_run_cu_offsets is None or run_positions is None or run_lengths is None:
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
        (torch.zeros(1, dtype=torch.int64), torch.cumsum(run_lengths, dim=0))
    )

    # Shape [num_items]; first flat-run index for each logical item.
    item_starts = item_run_cu_offsets[:-1]

    # Shape [num_runs]; logical multimodal item index for each flat run.
    run_item_indices = torch.repeat_interleave(
        torch.arange(item_run_counts.numel(), dtype=torch.int64), item_run_counts
    )

    # Shape [num_runs]; item-local token offset where each flat run begins.
    run_item_offsets = (
        cumulative_run_lengths[:-1] - cumulative_run_lengths[item_starts][run_item_indices]
    )
    # Shape [num_runs]; one-past-last full-prompt index for each flat run.
    run_ends = run_positions + run_lengths

    return _MmRunMetadata(
        run_positions=run_positions,
        run_ends=run_ends,
        run_item_indices=run_item_indices,
        run_item_offsets=run_item_offsets,
    )


def _augment_tokens_with_mm_run_metadata(
    vocab_size: int,
    result: list[TokenIdExt],
    multimodal_hashes: Sequence[Sequence[int]],
    metadata: _MmRunMetadata,
    chunk_start: int,
    chunk_end: int,
) -> list[TokenIdExt]:
    # Only rewrite multimodal runs that overlap the materialized prompt slice.
    overlap_mask = (metadata.run_ends > chunk_start) & (metadata.run_positions < chunk_end)
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
    prompt_overlap_ends = torch.clamp(metadata.run_ends[overlap_run_indices], max=chunk_end)
    item_token_offsets = (
        metadata.run_item_offsets[overlap_run_indices]
        + prompt_overlap_starts
        - overlap_run_positions
    )
    chunk_result_offsets = prompt_overlap_starts - chunk_start
    lengths = prompt_overlap_ends - prompt_overlap_starts

    current_item_idx: Optional[int] = None
    digest = b""
    for item_idx, chunk_result_offset, item_token_offset, length in zip(
        overlap_run_item_indices.tolist(),
        chunk_result_offsets.tolist(),
        item_token_offsets.tolist(),
        lengths.tolist(),
        strict=True,
    ):
        if item_idx != current_item_idx:
            current_item_idx = item_idx
            digest = _hash_to_digest(multimodal_hashes[item_idx])
        # Feed the coarse item property (content digest) and granular run
        # properties (item-local offset and span length) into the key
        # generator, so cache keys reflect the actual multimodal tokens being
        # rewritten.
        result[chunk_result_offset : chunk_result_offset + length] = (
            gen_multimodal_cache_key_tokens(
                vocab_size, digest, length, token_offset=item_token_offset
            )
        )

    return result


def _augment_tokens_with_contiguous_mm_metadata(
    vocab_size: int,
    result: list[TokenIdExt],
    multimodal_hashes: Sequence[Sequence[int]],
    multimodal_positions: Sequence[int] | torch.Tensor,
    multimodal_lengths: Sequence[int] | torch.Tensor,
    chunk_start: int,
    chunk_end: int,
) -> list[TokenIdExt]:
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
        result[result_offset : result_offset + overlap_length] = gen_multimodal_cache_key_tokens(
            vocab_size,
            _hash_to_digest(multimodal_hashes[item_idx]),
            overlap_length,
            token_offset=source_offset,
        )

    return result


def _locate_accepted_draft_tokens(requests: List[LlmRequest]):
    num_accepted_draft_tokens = []
    accepted_draft_tokens_indices = []
    rewind_draft_token_separate_adjustments = []
    # for context requests, the py_num_accepted_draft_tokens = 0, and py_num_accepted_draft_tokens_indices = []
    for seq in requests:
        num_accepted_draft_tokens.append(seq.py_num_accepted_draft_tokens)
        rewind_draft_token_separate_adjustments.append(
            seq.py_rewind_draft_token_separate_adjustment
        )
        accepted_draft_tokens_indices.extend(seq.py_num_accepted_draft_tokens_indices)
    batch_size = len(requests)
    num_accepted_draft_tokens_offset = torch.zeros(batch_size + 1, dtype=torch.int32, device="cuda")
    num_accepted_draft_tokens_offset[1:] = torch.cumsum(
        torch.tensor(num_accepted_draft_tokens, dtype=torch.int32), dim=0
    )
    accepted_draft_tokens_indices = torch.tensor(
        accepted_draft_tokens_indices, dtype=torch.int32, device="cuda"
    )
    rewind_draft_token_separate_adjustments = torch.tensor(
        rewind_draft_token_separate_adjustments, dtype=torch.int32, device="cuda"
    )
    return (
        num_accepted_draft_tokens_offset,
        accepted_draft_tokens_indices,
        rewind_draft_token_separate_adjustments,
    )


def _update_kv_cache_draft_token_location(
    cache_manager,
    scheduled_batch: ScheduledRequests,
    attn_metadata: "AttentionMetadata",
    kv_cache_dtype_byte_size: float,
):
    run_kv_cache_relocation = False
    for request in scheduled_batch.generation_requests:
        if request.state != LlmRequestState.GENERATION_COMPLETE:
            if (
                request.py_num_accepted_draft_tokens > 0
                and len(request.py_num_accepted_draft_tokens_indices) > 0
            ):
                run_kv_cache_relocation = True
    if not run_kv_cache_relocation:
        return
    requests = scheduled_batch.all_requests()
    (
        accepted_draft_token_offsets,
        packed_accepted_draft_tokens_indices,
        rewind_draft_token_separate_adjustments,
    ) = _locate_accepted_draft_tokens(requests)
    past_key_value_lengths = attn_metadata.kv_lens_cuda[: len(requests)]
    if (
        attn_metadata.kv_cache_block_offsets is not None
        and attn_metadata.host_kv_cache_pool_pointers is not None
        and attn_metadata.host_kv_cache_pool_mapping is not None
    ):
        use_paged_kv_cache = True
    else:
        use_paged_kv_cache = False
    assert use_paged_kv_cache, "Only paged kv cache is supported"
    assert len(cache_manager.max_attention_window_vec) == 1, (
        "Currently, only one max attention window size is supported."
    )

    if use_paged_kv_cache:
        assert len(set(cache_manager.num_kv_heads_per_layer)) == 1, (
            "update_kv_cache_draft_token_location requires uniform num_kv_heads across all layers, "
            f"but got {cache_manager.num_kv_heads_per_layer}"
        )
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


@maybe_compile(options={"max-autotune": True})
def _copy_swa_block_offsets_with_scratch_compiled(
    block_offsets: torch.Tensor,
    copy_idx: torch.Tensor,
    pool_ids: torch.Tensor,
    scales: torch.Tensor,
    layer_offsets: torch.Tensor,
    scratch_pages: torch.Tensor,
    block_positions: torch.Tensor,
    scratch_begs: torch.Tensor,
    scratch_ends: torch.Tensor,
    scratch_slots: torch.Tensor,
    num_contexts: torch.Tensor,
    output: torch.Tensor,
) -> None:
    base = block_offsets[pool_ids[:, :, None], copy_idx[None, None, :], 0, :]
    converted = torch.where(
        base == BAD_PAGE_INDEX,
        BAD_PAGE_INDEX,
        base * scales[:, :, None, None] + layer_offsets[:, :, None, None],
    )

    context_positions = torch.arange(
        scratch_begs.shape[1],
        dtype=torch.int32,
        device=scratch_begs.device,
    )
    active_context = context_positions < num_contexts
    scratch_mask_by_pool = (
        (block_positions >= scratch_begs[:, :, None])
        & (block_positions < scratch_ends[:, :, None])
        & active_context[None, :, None]
    )
    range_index = torch.where(
        scratch_mask_by_pool,
        block_positions - scratch_begs[:, :, None],
        0,
    )
    total_offset = range_index[pool_ids] * scratch_pages[:, :, None, None]
    slot_idx = (total_offset // scales[:, :, None, None]).clamp(max=scratch_slots.shape[-1] - 1)
    slot_id = scratch_slots[pool_ids].gather(-1, slot_idx.long())
    offset = total_offset % scales[:, :, None, None]
    scratch_index = (
        slot_id * scales[:, :, None, None]
        + (offset + layer_offsets[:, :, None, None]) % scales[:, :, None, None]
    )
    scratch_mask = scratch_mask_by_pool[pool_ids]
    converted = torch.where(scratch_mask, scratch_index, converted)

    output.copy_(converted.permute(0, 2, 1, 3))


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
        enable_stats: bool = False,
        **kwargs,
    ) -> None:
        self.mapping = mapping
        self.dtype = dtype
        self.is_disagg = is_disagg

        assert kv_connector_manager is None, (
            "kv_connector_manager is not supported for KVCacheManagerV2"
        )
        assert max_beam_width == 1, "max_beam_width must be 1 for KVCacheManagerV2"
        assert not (mapping.cp_config.get("cp_type") == CpType.STAR), (
            "Star attention is not supported for KVCacheManagerV2"
        )

        self.kv_cache_type = kv_cache_type
        self.pp_layers, self.num_layers = get_pp_layers(
            num_layers,
            mapping,
            spec_config=spec_config,
            layer_mask=layer_mask,
        )
        self.is_draft = is_draft
        self.enable_swa_scratch_reuse = (
            kv_cache_config.enable_swa_scratch_reuse and not self.is_draft
        )
        self.block_reuse_policy = BlockReusePolicy(kv_cache_config.block_reuse_policy)
        self.num_local_layers = len(self.pp_layers)
        self.layer_offsets = {idx: offset for offset, idx in enumerate(self.pp_layers)}
        self.max_beam_width = max_beam_width

        tp_size = mapping.tp_size
        if mapping.enable_attention_dp:
            tp_size = 1

        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.tokens_per_block = tokens_per_block
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size
        self.max_num_tokens = max_num_tokens
        self.kv_factor = 1 if kv_cache_type == CacheTypeCpp.SELFKONLY else 2
        from ..speculative import get_num_extra_kv_tokens

        self.num_extra_kv_tokens = get_num_extra_kv_tokens(spec_config)
        # Mirror V1: expose max_draft_len so the native disagg AuxBuffer
        # (_make_aux_buffer's getattr fallback) is sized for MTP/spec decoding.
        self.max_draft_len = spec_config.max_draft_len if spec_config is not None else 0
        self.max_total_draft_tokens = (
            spec_config.max_total_draft_tokens if spec_config is not None else 0
        )

        # Mirror V1's KV reserve sizing (see V1 __init__ for rationale).
        self._kv_reserve_draft_tokens = self.max_total_draft_tokens
        if (
            self.is_draft
            and spec_config is not None
            and getattr(spec_config, "use_dynamic_tree", False)
            and getattr(spec_config, "dynamic_tree_max_topK", 0) > 0
        ):
            draft_loop_tokens = spec_config.dynamic_tree_max_topK * spec_config.max_draft_len
            self._kv_reserve_draft_tokens = max(self.max_total_draft_tokens, draft_loop_tokens)

        self.event_buffer_max_size = kv_cache_config.event_buffer_max_size
        self.enable_stats = enable_stats
        kv_cache_event_hash_algo = get_effective_kv_cache_event_hash_algo(
            kv_cache_config.kv_cache_event_hash_algo,
            use_kv_cache_manager_v2=True,
        )

        self._stream = (
            execution_stream if execution_stream is not None else torch.cuda.current_stream()
        )
        logger.info(f"[KVCacheManager] execution_stream: {self._stream}")

        # Determine max_attention_window_vec
        if kv_cache_config.max_attention_window is not None:
            self.max_attention_window_vec = (
                kv_cache_config.max_attention_window.copy()
            )  # Make a copy to avoid modifying original
            # Clamp all window sizes to max_seq_len before calculating the
            # number of KV cache blocks. This prevents the KV cache pool from
            # being skewed by the largest window values.
            self.max_attention_window_vec = [
                min(max_seq_len, w) for w in self.max_attention_window_vec
            ]

            self.max_attention_window_vec = [
                None if w == max_seq_len else w for w in self.max_attention_window_vec
            ]

        else:
            self.max_attention_window_vec = [None]

        event_window_size = max(
            self.max_seq_len if window_size is None else int(window_size)
            for window_size in self.max_attention_window_vec
        )
        self.event_manager: Optional[KVCacheEventManager] = None
        if self.event_buffer_max_size > 0:
            if mapping.enable_attention_dp:
                self.event_manager = KVCacheEventManager(
                    self.event_buffer_max_size,
                    window_size=event_window_size,
                    attention_dp_rank=mapping.rank,
                    attention_dp_gather=Distributed.get(mapping).allgather,
                    hash_algo=kv_cache_event_hash_algo,
                )
            elif mpi_rank() == 0:
                self.event_manager = KVCacheEventManager(
                    self.event_buffer_max_size,
                    window_size=event_window_size,
                    hash_algo=kv_cache_event_hash_algo,
                )

        if isinstance(num_kv_heads, int):
            self.num_kv_heads_per_layer = [
                (num_kv_heads + tp_size - 1) // tp_size for _ in range(self.num_local_layers)
            ]
            self.total_num_kv_heads_per_layer = [
                (num_kv_heads + tp_size - 1) // tp_size for _ in range(self.num_layers)
            ]
        else:
            assert len(num_kv_heads) == self.num_layers

            def append_to_kv_heads_per_layer(
                num_kv_heads_per_layer: List[int], kv_head: Optional[int]
            ):
                if kv_head is not None:
                    num_kv_heads_per_layer.append((kv_head + tp_size - 1) // tp_size)
                else:
                    num_kv_heads_per_layer.append(0)

            self.num_kv_heads_per_layer = []
            if self.num_local_layers > 0:
                for i in self.pp_layers:
                    kv_head = num_kv_heads[i]
                    append_to_kv_heads_per_layer(self.num_kv_heads_per_layer, kv_head)

            self.total_num_kv_heads_per_layer = []
            for i in range(self.num_layers):
                kv_head = num_kv_heads[i]
                append_to_kv_heads_per_layer(self.total_num_kv_heads_per_layer, kv_head)

        # Build per-layer head_dim (similar to num_kv_heads_per_layer)
        if isinstance(head_dim, int):
            self.head_dim_per_layer = [head_dim for _ in range(self.num_local_layers)]
        else:
            assert len(head_dim) == self.num_layers, (
                f"head_dim list length ({len(head_dim)}) must match num_layers ({self.num_layers})"
            )
            self.head_dim_per_layer = []
            if self.num_local_layers > 0:
                for i in self.pp_layers:
                    self.head_dim_per_layer.append(head_dim[i])
            if len(set(self.head_dim_per_layer)) > 1:
                logger.info(
                    f"Per-layer head_dim: {len(self.head_dim_per_layer)} layers, "
                    f"unique values={set(self.head_dim_per_layer)}"
                )

        self.is_vswa = len(set(self.max_attention_window_vec)) > 1

        quota = float("inf")
        if (
            kv_cache_config.max_gpu_total_bytes is not None
            and kv_cache_config.max_gpu_total_bytes > 0
        ):
            quota = int(kv_cache_config.max_gpu_total_bytes)
            logger.info(f"max_gpu_total_bytes is provided. New quota is {quota / (1 << 30)}GiB")
        if kv_cache_config.max_tokens is not None:
            quota_from_max_tokens = int(
                math.ceil(
                    self._get_quota_from_max_tokens(kv_cache_config.max_tokens)
                    / kv_cache_config.max_util_for_resume
                )
            )
            quota = min(quota, quota_from_max_tokens)
            logger.info(
                f"max_tokens {kv_cache_config.max_tokens} is provided. "
                f"Allowed quota from max_tokens is {quota_from_max_tokens / (1 << 30)}GiB. "
                f"New quota is {quota / (1 << 30)}GiB"
            )

        assert quota != float("inf"), (
            "Quota not set. Check kv_cache_config.max_tokens or kv_cache_config.max_gpu_total_bytes"
        )

        # Sync KV cache token capacity across ranks so all ranks allocate
        # the same number of tokens and the scheduler produces identical
        # batches.  Normalize to token count before the allreduce because
        # bytes_per_token varies across PP ranks (different local layers).
        if mapping.world_size > 1:
            dist = Distributed.get(mapping)
            max_tokens = self._get_max_tokens_from_quota(quota)
            max_tokens = dist.allreduce(max_tokens, op=ReduceOp.MIN)
            # inf max_tokens means all layers are SWA and every rank quota can
            # fit all SWA fixed cache.
            if not math.isinf(max_tokens):
                quota = self._get_quota_from_max_tokens(max_tokens)

        logger.info(f"KV cache manager v2 device quota set to {quota / (1 << 30)}GiB")

        cache_tiers: List[CacheTierConfig] = [GpuCacheTierConfig(quota=quota)]
        if kv_cache_config.host_cache_size is not None and kv_cache_config.host_cache_size >= 0:
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
            # memory and pinnable memory limit to avoid allocation failures.
            import resource

            try:
                mem_available = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES")
            except (ValueError, OSError):
                mem_available = float("inf")
            try:
                _soft, _hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
                memlock_limit = _soft if _soft != resource.RLIM_INFINITY else float("inf")
            except (ValueError, OSError):
                memlock_limit = float("inf")
            candidates = [quota]
            if mem_available != float("inf"):
                candidates.append(int(mem_available * 0.5))
            if memlock_limit != float("inf"):
                candidates.append(int(memlock_limit * 0.8))
            host_quota = min(candidates)
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
            cache_tiers.append(DiskCacheTierConfig(quota=disk_cache_size, path=disk_cache_path))
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

        try:
            self.impl = KVCacheManagerPy(config, event_manager=self.event_manager)
        except (CuError, KVCacheOutOfMemoryError):
            if len(cache_tiers) > 1:
                logger.warning(
                    "Failed to initialize KV cache manager with host cache "
                    "tier (cuMemHostRegister may have failed). "
                    "Retrying without host cache tier."
                )
                cache_tiers_gpu_only = [t for t in cache_tiers if isinstance(t, GpuCacheTierConfig)]
                config = self._build_cache_config(
                    kv_cache_config,
                    tokens_per_block=tokens_per_block,
                    vocab_size=vocab_size,
                    cache_tiers=cache_tiers_gpu_only,
                )
                cache_tiers = cache_tiers_gpu_only
                self.kv_cache_manager_py_config = config
                self.impl = KVCacheManagerPy(config, event_manager=self.event_manager)
            else:
                raise
        if self.event_manager is not None:
            self.event_manager.set_layer_group_window_sizes(
                self._get_event_window_sizes_by_layer_group()
            )
            self.event_manager.add_created_event(
                self._get_event_num_blocks_per_cache_level(cache_tiers, tokens_per_block),
                self._get_event_layer_group_ids(),
            )

        self.num_pools = len(self.impl.layer_grouping)
        # num_pools is the physical pool count owned by the KV cache manager.
        # With SWA scratch reuse, scratch slot IDs are only valid with
        # per-layer page indices, so the attention op sees one virtual pool per
        # local layer while the underlying manager can still group layers.
        if self.enable_swa_scratch_reuse:
            self.num_attention_op_pools = self.num_local_layers
        else:
            self.num_attention_op_pools = self.num_pools

        num_layers = len(config.layers)
        self.layer_to_pool_mapping_dict: dict[int, int] = {
            layer_id: self.impl.get_layer_group_id(layer_id)
            for layer_id in typed_range(LayerId(num_layers))
        }

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

        max_num_tokens = self.get_num_available_tokens(token_num_upper_bound=max_seq_len)

        if max_seq_len > max_num_tokens:
            logger.warning(
                f"max_seq_len {max_seq_len} is greater than max_num_tokens {max_num_tokens} "
                "that can be allocated in kv cache manager, setting "
                f"max_seq_len to {max_num_tokens}"
            )
            # max_num_tokens is a float from clamp_max_seq_len_for_mem; cast
            # so downstream int-only consumers (torch.randint size, range)
            # stay int.
            self.max_seq_len = int(max_num_tokens)

        # Pad max_blocks_per_seq to next multiple of 4 (copy_block_offsets kernel).
        # Account for max single-sequence capacity = seq_len + extra KV tokens +
        # _kv_reserve_draft_tokens (see __init__) + 1 base decode token.
        max_seq_capacity = (
            self.max_seq_len + self.num_extra_kv_tokens + self._kv_reserve_draft_tokens + 1
        )
        self.max_blocks_per_seq = (max_seq_capacity + tokens_per_block - 1) // tokens_per_block
        if self.max_blocks_per_seq % 4 != 0:
            self.max_blocks_per_seq = ((self.max_blocks_per_seq + 3) // 4) * 4

        self.enable_block_reuse = kv_cache_config.enable_block_reuse
        self.enable_partial_reuse = kv_cache_config.enable_partial_reuse
        self.disk_prefetch_num_reqs = kv_cache_config.disk_prefetch_num_reqs

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
        self._prepare_page_table_tensor(index_mapper_capacity)

        self._log_kv_cache_pool_lifecycle_mapping()

    def _build_pool_mapping_tensors(self):
        """Build the ``(kv_cache_pool_pointers, kv_cache_pool_mapping)`` tensors.

        Extracted into an overridable hook: subclasses whose pools
        coalesce extra per-layer buffers alongside K/V (e.g.
        MiniMax-M3 merges ``Role.INDEX_KEY`` into the K/V pool) cannot
        derive the per-layer mapping offset via ``exact_div`` of the
        byte address delta, because a layer no longer contributes
        exactly K+V. They override this method to compute the offset
        from the layer's position within its pool group instead.
        """
        kv_cache_pool_pointers_list = []
        kv_cache_pool_mapping_list = []
        block_scale_pool_pointers_list = []
        if self.enable_swa_scratch_reuse:
            for layer_id in typed_range(LayerId(self.num_local_layers)):
                kv_cache_pool_pointers_list.append(
                    [
                        self.impl.get_mem_pool_base_address(
                            layer_id, Role.KEY, PageIndexMode.PER_LAYER
                        ),
                        0,
                    ]
                )
                if self.dtype == DataType.NVFP4:
                    block_scale_pool_pointers_list.append(
                        [
                            self.impl.get_mem_pool_base_address(
                                layer_id, Role.KEY_BLOCK_SCALE, PageIndexMode.PER_LAYER
                            ),
                            0,
                        ]
                    )
                kv_cache_pool_mapping_list.append([int(layer_id), 0])
        else:
            for pool_id in range(self.num_pools):
                layer_id = self.impl.layer_grouping[pool_id][0]
                kv_cache_pool_pointers_list.append(
                    [
                        self.impl.get_mem_pool_base_address(
                            layer_id, Role.KEY, PageIndexMode.SHARED
                        ),
                        0,
                    ]
                )
                if self.dtype == DataType.NVFP4:
                    block_scale_pool_pointers_list.append(
                        [
                            self.impl.get_mem_pool_base_address(
                                layer_id, Role.KEY_BLOCK_SCALE, PageIndexMode.SHARED
                            ),
                            0,
                        ]
                    )

            for layer_id in typed_range(LayerId(self.num_local_layers)):
                layer_group_id = self.impl.get_layer_group_id(layer_id)
                if self.dtype != DataType.NVFP4:
                    key_base_addr = kv_cache_pool_pointers_list[layer_group_id][0]
                    addr_offset = (
                        self.impl.get_mem_pool_base_address(
                            layer_id, Role.KEY, PageIndexMode.SHARED
                        )
                        - key_base_addr
                    )
                else:
                    key_base_addr = kv_cache_pool_pointers_list[layer_group_id][0]
                    block_scale_base_addr = block_scale_pool_pointers_list[layer_group_id][0]
                    addr_offset = (
                        self.impl.get_mem_pool_base_address(
                            layer_id, Role.KEY, PageIndexMode.SHARED
                        )
                        - key_base_addr
                    )
                    block_scale_addr_offset = (
                        self.impl.get_mem_pool_base_address(
                            layer_id, Role.KEY_BLOCK_SCALE, PageIndexMode.SHARED
                        )
                        - block_scale_base_addr
                    )
                    block_scale_offset = exact_div(
                        block_scale_addr_offset,
                        self.get_layer_bytes_per_token(layer_id, Role.KEY_BLOCK_SCALE)
                        * self.kv_factor
                        * self.tokens_per_block,
                    )
                offset = exact_div(
                    addr_offset,
                    self.get_layer_bytes_per_token(layer_id, Role.KEY)
                    * self.kv_factor
                    * self.tokens_per_block,
                )

                if self.dtype == DataType.NVFP4:
                    assert block_scale_offset == offset, (
                        "Block scale offset and offset should be the same"
                    )

                kv_cache_pool_mapping_list.append([layer_group_id, offset])

        if self.dtype == DataType.NVFP4:
            for pool_id, block_scale_pool_pointers in enumerate(block_scale_pool_pointers_list):
                pool_pointers = kv_cache_pool_pointers_list[pool_id]
                kv_cache_pool_pointers_list[pool_id] = [
                    [pool_pointers[0], block_scale_pool_pointers[0]],
                    [pool_pointers[1], block_scale_pool_pointers[1]],
                ]

        kv_cache_pool_pointers = torch.tensor(
            kv_cache_pool_pointers_list,
            dtype=torch.int64,
            device="cpu",
            pin_memory=prefer_pinned(),
        )
        kv_cache_pool_mapping = torch.tensor(
            kv_cache_pool_mapping_list,
            dtype=torch.int32,
            device="cpu",
            pin_memory=prefer_pinned(),
        )
        return kv_cache_pool_pointers, kv_cache_pool_mapping

    def _prepare_page_table_tensor(self, index_mapper_capacity: int) -> None:
        self.kv_cache_pool_pointers, self.kv_cache_pool_mapping = self._build_pool_mapping_tensors()
        self.index_scales = torch.empty(
            self.num_pools, dtype=torch.int32, pin_memory=prefer_pinned(), device="cpu"
        )
        self.kv_offset = torch.empty(
            self.num_pools, dtype=torch.int32, pin_memory=prefer_pinned(), device="cpu"
        )
        for pool_id in range(self.num_pools):
            layer_id = self.impl.layer_grouping[pool_id][0]
            self.index_scales[pool_id] = self.impl.get_page_index_scale(layer_id, Role.KEY)
            if self.kv_cache_type != CacheTypeCpp.SELFKONLY:
                self.kv_offset[pool_id] = exact_div(
                    self.impl.get_mem_pool_base_address(layer_id, Role.VALUE, PageIndexMode.SHARED)
                    - self.impl.get_mem_pool_base_address(layer_id, Role.KEY, PageIndexMode.SHARED),
                    self.impl.get_page_stride(layer_id, Role.KEY),
                )
            else:
                self.kv_offset[pool_id] = 0

        # Keep unused block offsets as safe block index 0.
        self.host_kv_cache_block_offsets = torch.zeros(
            self.num_pools,
            index_mapper_capacity * self.max_beam_width,
            2,  # key and value
            self.max_blocks_per_seq,
            dtype=torch.int32,
            pin_memory=prefer_pinned(),
            device="cpu",
        )
        if self.enable_swa_scratch_reuse:
            self._prepare_swa_scratch_copy_tensors(index_mapper_capacity)

    def _get_runtime_cache_size_layer_components(self) -> tuple[List[int], List[Optional[int]]]:
        layer_sizes = []
        attention_windows = []
        pattern_len = len(self.max_attention_window_vec)
        for local_layer_idx in range(self.num_local_layers):
            layer_sizes.append(
                self.get_layer_bytes_per_token(local_layer_idx=local_layer_idx, data_role=Role.ALL)
            )
            attention_windows.append(
                self.max_attention_window_vec[self.pp_layers[local_layer_idx] % pattern_len]
            )
        return layer_sizes, attention_windows

    def _get_max_tokens_from_quota(self, quota: int) -> float:
        layer_sizes, attention_windows = self._get_runtime_cache_size_layer_components()
        full_attn_size_per_token = _estimate_full_attn_size_per_token(
            layer_sizes, attention_windows
        )
        context_swa_size_per_token, _ = _estimate_swa_cache_size(
            layer_sizes,
            attention_windows,
            self.tokens_per_block,
            context=True,
            scratch=self.enable_swa_scratch_reuse,
        )
        (
            generation_swa_size_per_token,
            generation_swa_size_per_request,
        ) = _estimate_swa_cache_size(
            layer_sizes, attention_windows, self.tokens_per_block, context=False, scratch=False
        )
        size_per_batch = self.max_batch_size * generation_swa_size_per_request
        if quota < size_per_batch:
            return 0
        context_size_per_token = full_attn_size_per_token + context_swa_size_per_token
        context_limit_quota = self.max_num_tokens * context_size_per_token + size_per_batch
        if quota <= context_limit_quota:
            if context_size_per_token <= 0:
                return float("inf")
            return (quota - size_per_batch) / context_size_per_token

        generation_size_per_token = full_attn_size_per_token + generation_swa_size_per_token
        if generation_size_per_token <= 0:
            return float("inf")
        return self.max_num_tokens + (quota - context_limit_quota) / generation_size_per_token

    def _get_quota_from_max_tokens(self, max_tokens: int) -> int:
        layer_sizes, attention_windows = self._get_runtime_cache_size_layer_components()
        full_attn_size_per_token = _estimate_full_attn_size_per_token(
            layer_sizes, attention_windows
        )
        (
            context_swa_size_per_token,
            _,
        ) = _estimate_swa_cache_size(
            layer_sizes,
            attention_windows,
            self.tokens_per_block,
            context=True,
            scratch=self.enable_swa_scratch_reuse,
        )
        (
            generation_swa_size_per_token,
            generation_swa_size_per_request,
        ) = _estimate_swa_cache_size(
            layer_sizes, attention_windows, self.tokens_per_block, context=False, scratch=False
        )
        context_tokens = min(max_tokens, self.max_num_tokens)
        generation_tokens = max_tokens - context_tokens
        generation_quota = (
            max_tokens * full_attn_size_per_token
            + generation_tokens * generation_swa_size_per_token
            + self.max_batch_size * generation_swa_size_per_request
        )
        context_extra_quota = context_tokens * context_swa_size_per_token
        return int(generation_quota + context_extra_quota)

    def _get_event_num_blocks_per_cache_level(
        self,
        cache_tiers: List[CacheTierConfig],
        tokens_per_block: int,
    ) -> List[int]:
        bytes_per_block = self.get_cache_bytes_per_token() * tokens_per_block
        if bytes_per_block <= 0:
            return []
        return [int(tier.quota // bytes_per_block) for tier in cache_tiers]

    def _get_event_layer_group_ids(self) -> List[int]:
        return [int(layer_group_id) for layer_group_id in range(len(self.impl.layer_grouping))]

    def _get_event_window_sizes_by_layer_group(self) -> Dict[int, int]:
        # Assumes every layer in a group shares the same sliding_window_size,
        # which is how `impl.layer_grouping` partitions layers today. Only the
        # first layer's window is read; if the grouping policy ever permits
        # mixed windows in one group, this needs to fan out per-layer.

        def get_event_window_size(layer_id: int) -> int:
            window_size = self.kv_cache_manager_py_config.layers[layer_id].sliding_window_size
            return self.max_seq_len if window_size is None else int(window_size)

        return {
            int(layer_group_id): get_event_window_size(int(layer_ids[0]))
            for layer_group_id, layer_ids in enumerate(self.impl.layer_grouping)
        }

    def _format_kv_cache_pool_lifecycle_entry(self, layer_id: LayerId, role: DataRole) -> str:
        attr = self.impl._storage.get_buffer_attr(layer_id, role)
        pool_group_id = self.impl._storage.get_pool_group_index(attr.life_cycle_id)
        lifecycle = self.impl._life_cycles.get_life_cycle(attr.life_cycle_id)
        return (
            f"role={str(role)}, pool_group_id={int(pool_group_id)}, "
            f"lifecycle_id={int(attr.life_cycle_id)}, "
            f"lifecycle={lifecycle}"
        )

    def _log_kv_cache_pool_lifecycle_mapping(self) -> None:
        entries = OrderedDict()
        for layer in self.kv_cache_manager_py_config.layers:
            for buffer in layer.buffers:
                entries.setdefault(
                    self._format_kv_cache_pool_lifecycle_entry(layer.layer_id, buffer.role), None
                )

        if not entries:
            return

        logger.info(f"{type(self).__name__} role-to-pool/lifecycle mapping:")
        for entry in entries:
            logger.info(entry)

    def _prepare_swa_scratch_copy_tensors(self, index_mapper_capacity: int) -> None:
        pool_ids = torch.empty(
            self.num_attention_op_pools,
            2,
            dtype=torch.long,
            device="cpu",
        )
        scales = torch.empty(
            self.num_attention_op_pools,
            2,
            dtype=torch.int32,
            device="cpu",
        )
        layer_offsets = torch.empty_like(scales)
        scratch_pages = torch.empty_like(scales)
        for local_layer_idx in range(self.num_local_layers):
            layer_id = LayerId(local_layer_idx)
            pool_id = self.layer_to_pool_mapping_dict[layer_id]
            roles = [Role.KEY, Role.VALUE]
            if self.kv_cache_type == CacheTypeCpp.SELFKONLY:
                roles[1] = Role.KEY
            for role_idx, role in enumerate(roles):
                converter = self.impl.get_page_index_converter(layer_id, role)
                if converter.expansion != 1:
                    raise NotImplementedError(
                        "SWA scratch block-table conversion does not support "
                        f"expanded page indices yet: layer={layer_id}, role={role}, "
                        f"expansion={converter.expansion}"
                    )
                pool_ids[local_layer_idx, role_idx] = pool_id
                scales[local_layer_idx, role_idx] = int(converter.scale)
                layer_offsets[local_layer_idx, role_idx] = int(converter.layer_offset)
                scratch_pages[local_layer_idx, role_idx] = int(converter.scratch_pages_per_block)

        staging_capacity = index_mapper_capacity * self.max_beam_width
        device = torch.device("cuda", torch.cuda.current_device())
        self._device_kv_cache_block_offsets_input = torch.empty_like(
            self.host_kv_cache_block_offsets,
            device=device,
        )
        self._device_attention_op_block_offsets_staging = torch.empty(
            self.num_attention_op_pools,
            staging_capacity,
            2,
            self.max_blocks_per_seq,
            dtype=torch.int32,
            device=device,
        )
        self._device_copy_idx_staging = torch.zeros(
            staging_capacity,
            dtype=torch.long,
            device=device,
        )
        self._device_num_contexts = torch.empty((), dtype=torch.int32, device=device)
        self._device_attention_op_pool_ids = pool_ids.to(device=device)
        self._device_attention_op_scales = scales.to(device=device)
        self._device_attention_op_layer_offsets = layer_offsets.to(device=device)
        self._device_attention_op_scratch_pages = scratch_pages.to(device=device)
        self._device_block_positions = torch.arange(
            self.max_blocks_per_seq,
            dtype=torch.int32,
            device=device,
        )

        min_scale = int(scales.min().item()) if scales.numel() > 0 else 1
        max_scratch_pages = int(scratch_pages.max().item()) if scratch_pages.numel() > 0 else 1
        self._max_scratch_slots = max(
            1,
            (self.max_blocks_per_seq * max_scratch_pages + min_scale - 1) // min_scale,
        )
        scratch_slots_shape = (
            self.num_pools,
            staging_capacity,
            self._max_scratch_slots,
        )
        self._host_scratch_begs_staging = torch.zeros(
            self.num_pools,
            staging_capacity,
            dtype=torch.int32,
            pin_memory=prefer_pinned(),
            device="cpu",
        )
        self._host_scratch_ends_staging = torch.zeros(
            self.num_pools,
            staging_capacity,
            dtype=torch.int32,
            pin_memory=prefer_pinned(),
            device="cpu",
        )
        self._host_scratch_slots_staging = torch.zeros(
            scratch_slots_shape,
            dtype=torch.int32,
            pin_memory=prefer_pinned(),
            device="cpu",
        )
        self._device_scratch_begs_staging = torch.zeros(
            self.num_pools,
            staging_capacity,
            dtype=torch.int32,
            device=device,
        )
        self._device_scratch_ends_staging = torch.zeros(
            self.num_pools,
            staging_capacity,
            dtype=torch.int32,
            device=device,
        )
        self._device_scratch_slots_staging = torch.zeros(
            scratch_slots_shape,
            dtype=torch.int32,
            device=device,
        )

    def _copy_idx_to_device(self, copy_idx: torch.Tensor) -> torch.Tensor:
        self._device_copy_idx_staging[: copy_idx.size(0)].copy_(copy_idx, non_blocking=True)
        return self._device_copy_idx_staging

    def _copy_scratch_metadata_to_device(
        self,
        request_ids: List[int],
        num_contexts: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        host_begs = self._host_scratch_begs_staging[:, :num_contexts]
        host_ends = self._host_scratch_ends_staging[:, :num_contexts]
        host_slots = self._host_scratch_slots_staging[:, :num_contexts, :]
        host_begs.zero_()
        host_ends.zero_()
        host_slots.zero_()

        for pool_id in range(self.num_pools):
            for context_idx, request_id in enumerate(request_ids[:num_contexts]):
                desc = self.kv_cache_map[request_id].get_scratch_desc(pool_id)
                if desc is None:
                    continue
                slot_ids = desc.slot_ids
                if len(slot_ids) > self._max_scratch_slots:
                    raise RuntimeError(
                        f"Scratch slot count {len(slot_ids)} exceeds staging capacity "
                        f"{self._max_scratch_slots}"
                    )
                host_begs[pool_id, context_idx] = int(desc.range.beg)
                host_ends[pool_id, context_idx] = int(desc.range.end)
                for slot_idx, slot_id in enumerate(slot_ids):
                    host_slots[pool_id, context_idx, slot_idx] = int(slot_id)

        self._device_scratch_begs_staging.copy_(
            self._host_scratch_begs_staging,
            non_blocking=True,
        )
        self._device_scratch_ends_staging.copy_(
            self._host_scratch_ends_staging,
            non_blocking=True,
        )
        self._device_scratch_slots_staging.copy_(
            self._host_scratch_slots_staging,
            non_blocking=True,
        )
        return (
            self._device_scratch_begs_staging,
            self._device_scratch_ends_staging,
            self._device_scratch_slots_staging,
        )

    def _copy_batch_block_offsets_per_layer(
        self,
        dst_tensor: torch.Tensor,
        request_ids: List[int],
        copy_idx: torch.Tensor,
        num_contexts: int,
        num_seqs: int,
    ) -> None:
        device_copy_idx = self._copy_idx_to_device(copy_idx)
        self._device_kv_cache_block_offsets_input.copy_(
            self.host_kv_cache_block_offsets,
            non_blocking=True,
        )
        scratch_begs, scratch_ends, scratch_slots = self._copy_scratch_metadata_to_device(
            request_ids,
            num_contexts,
        )
        self._device_num_contexts.fill_(num_contexts)
        _copy_swa_block_offsets_with_scratch_compiled(
            self._device_kv_cache_block_offsets_input,
            device_copy_idx,
            self._device_attention_op_pool_ids,
            self._device_attention_op_scales,
            self._device_attention_op_layer_offsets,
            self._device_attention_op_scratch_pages,
            self._device_block_positions,
            scratch_begs,
            scratch_ends,
            scratch_slots,
            self._device_num_contexts,
            self._device_attention_op_block_offsets_staging,
        )
        dst_tensor[: self.num_attention_op_pools, :num_seqs].copy_(
            self._device_attention_op_block_offsets_staging[
                : self.num_attention_op_pools,
                :num_seqs,
            ],
            non_blocking=True,
        )

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
                assert hd % 2 == 0, (
                    f"head_dim must be divisible by 2 for nvfp4 kv cache, but layer {layer_idx} has head_dim={hd}"
                )
            buffer_type.append(Role.KEY_BLOCK_SCALE)
            if self.kv_cache_type != CacheTypeCpp.SELFKONLY:
                buffer_type.append(Role.VALUE_BLOCK_SCALE)

        scratch_reuse_config = None
        if self.enable_swa_scratch_reuse:
            # Context requests allocate num_extra_kv_tokens for spec decoding.
            # They should not count toward the scratch range.
            scratch_reuse_config = SwaScratchReuseConfig(max_rewind_len=self.num_extra_kv_tokens)

        # Subclasses (e.g. MiniMax-M3 sparse cache) can register additional
        # per-layer BufferConfig entries — for example a sparse index-K
        # buffer — without overriding the K/V/NVFP4 scale wiring above.
        # The dict maps local layer id -> list of extra BufferConfig. Each
        # extra buffer's role must be unique within the layer (asserted by
        # AttentionLayerConfig.__post_init__) and its size must be in bytes
        # per block (= bytes_per_token * tokens_per_block).
        extra_buffers_per_layer = (
            self._extra_buffers_per_layer(tokens_per_block=tokens_per_block) or {}
        )

        layer_configs: List[AttentionLayerConfig] = []
        for layer_id in typed_range(LayerId(self.num_local_layers)):
            buffers = [
                BufferConfig(
                    role=role,
                    size=self.get_layer_bytes_per_token(local_layer_idx=layer_id, data_role=role)
                    * tokens_per_block,
                )
                for role in buffer_type
            ]
            for extra in extra_buffers_per_layer.get(int(layer_id), ()):
                assert extra.role not in buffer_type, (
                    f"extra buffer role {extra.role!r} for layer "
                    f"{int(layer_id)} duplicates a standard K/V/scale role"
                )
                buffers.append(extra)
            layer_configs.append(
                AttentionLayerConfig(
                    layer_id=layer_id,
                    buffers=buffers,
                    sliding_window_size=self.max_attention_window_vec[
                        self.pp_layers[layer_id] % len(self.max_attention_window_vec)
                    ],
                    num_sink_tokens=None,
                )
            )

        return KVCacheManagerConfigPy(
            tokens_per_block=tokens_per_block,
            vocab_size=vocab_size,
            cache_tiers=cache_tiers,
            max_util_for_resume=kv_cache_config.max_util_for_resume,
            enable_stats=self.enable_stats,
            swa_scratch_reuse=scratch_reuse_config,
            initial_pool_ratio=kv_cache_config.pool_ratio,
            layers=layer_configs,
        )

    def _extra_buffers_per_layer(
        self, *, tokens_per_block: int
    ) -> Optional[dict[int, List[BufferConfig]]]:
        """Return per-local-layer extra BufferConfig entries to register
        alongside the standard K/V/NVFP4 scale buffers.

        Default implementation returns ``None``. Subclasses override this
        to register additional buffers — for example, MiniMax-M3 registers
        a sparse index-K buffer for each sparse local layer. Each
        ``BufferConfig.size`` is interpreted as bytes per block (i.e.,
        ``bytes_per_token * tokens_per_block``), matching the standard
        buffers built in :meth:`_build_cache_config`. The block storage
        groups buffers by lifecycle and size with an opaque role key, so
        new roles do not require C++ changes.
        """
        return None

    @property
    def blocks_in_primary_pool(self) -> int:
        """
        Get the number of blocks in the primary pool.
        """
        return self.impl.get_page_index_upper_bound(0, Role.KEY)

    def get_buffers(self, layer_idx: int, kv_layout: str = "NHD") -> Optional[torch.Tensor]:
        layer_offset = self.layer_offsets[layer_idx]
        addr_key = self.impl.get_mem_pool_base_address(layer_offset, Role.KEY, PageIndexMode.SHARED)
        if self.kv_cache_type != CacheTypeCpp.SELFKONLY:
            addr_value = self.impl.get_mem_pool_base_address(
                layer_offset, Role.VALUE, PageIndexMode.SHARED
            )
            page_size_key = self.impl.get_page_stride(layer_offset, Role.KEY)
            page_size_value = self.impl.get_page_stride(layer_offset, Role.VALUE)

            assert addr_key + page_size_value == addr_value and page_size_key == page_size_value

        assert kv_layout in ["NHD", "HND"], f"Unsupported kv_layout: {kv_layout}"

        element_per_container = 1
        dtype = self.dtype
        if dtype == DataType.NVFP4:
            element_per_container = 2
            dtype = torch.int8

        layer_head_dim = self.head_dim_per_layer[layer_offset]
        if kv_layout == "NHD":
            shape = [
                self.impl.get_page_index_upper_bound(layer_offset, Role.KEY) // self.kv_factor,
                self.kv_factor,
                self.tokens_per_block,
                self.num_kv_heads_per_layer[layer_offset],
                layer_head_dim // element_per_container,
            ]
        else:
            shape = [
                self.impl.get_page_index_upper_bound(layer_offset, Role.KEY) // self.kv_factor,
                self.kv_factor,
                self.num_kv_heads_per_layer[layer_offset],
                self.tokens_per_block,
                layer_head_dim // element_per_container,
            ]

        return convert_to_torch_tensor(
            TensorWrapper(
                addr_key,
                dtype,
                shape,
            )
        )

    def get_index_k_buffer(
        self,
        layer_idx: int,
        *,
        num_heads: int = 1,
        head_dim: int,
        dtype: Union[torch.dtype, "DataType", str] = torch.bfloat16,
    ) -> Optional[torch.Tensor]:
        """Return a torch view over the V2-managed paged ``Role.INDEX_KEY``
        buffer for ``layer_idx``, or ``None`` when the layer has no
        INDEX_KEY buffer registered (e.g. dense layers in a sparse model,
        or non-local layers on the current PP rank).

        The view has shape ``[num_pages, tokens_per_block, num_heads,
        head_dim]`` where ``num_pages == impl.get_page_index_upper_bound(
        layer_idx, Role.INDEX_KEY)``. Sparse modeling code addresses
        entries by ``(page, within_page, head, dim)`` after decomposing
        the per-token slot id used by the main paged K/V cache into
        ``(page, within_page)``.

        Because :class:`BufferConfig` only carries an opaque byte ``size``
        per block, the dtype and head shape are caller-side contracts.
        The caller must pass the same ``num_heads``, ``head_dim``, and
        ``dtype`` it used to compute the registered
        ``size = num_heads * head_dim * dtype_bytes * tokens_per_block``
        in ``_extra_buffers_per_layer``. The accessor validates
        ``num_heads * head_dim * dtype_bytes * tokens_per_block ==
        page_stride`` so a wiring mismatch fails loudly instead of
        returning a view with silently wrong stride.

        Follows the same ``TensorWrapper`` / ``convert_to_torch_tensor``
        pattern as :meth:`get_buffers`. The returned tensor is a
        zero-copy view over V2-managed pool memory and stays valid for
        the lifetime of the cache manager — writes through the view
        propagate to the pool, and successive calls return views over
        the same backing storage.
        """
        if layer_idx not in self.layer_offsets:
            return None
        layer_offset = self.layer_offsets[layer_idx]
        try:
            addr = self.impl.get_mem_pool_base_address(layer_offset, Role.INDEX_KEY)
            page_stride = self.impl.get_page_stride(layer_offset, Role.INDEX_KEY)
            page_upper = self.impl.get_page_index_upper_bound(layer_offset, Role.INDEX_KEY)
            converter = self.impl.get_page_index_converter(layer_offset, Role.INDEX_KEY)
        except KeyError:
            # INDEX_KEY not registered for this layer (default V2 manager
            # registers only K/V/scale; sparse subclasses register
            # INDEX_KEY only on sparse layers via
            # ``_extra_buffers_per_layer``).
            return None

        if isinstance(dtype, DataType):
            torch_dtype = binding_to_torch_dtype(dtype)
        elif isinstance(dtype, str):
            torch_dtype = str_dtype_to_torch(dtype)
        else:
            torch_dtype = dtype

        elem_bytes = torch.tensor([], dtype=torch_dtype).element_size()
        expected_stride = num_heads * head_dim * elem_bytes * self.tokens_per_block
        assert page_stride == expected_stride, (
            f"INDEX_KEY page stride mismatch for layer {layer_idx}: "
            f"V2 reports page_stride={page_stride}, but the caller "
            f"supplied num_heads={num_heads} * head_dim={head_dim} * "
            f"elem_bytes={elem_bytes} * tokens_per_block="
            f"{self.tokens_per_block} = {expected_stride}. Re-check "
            f"the BufferConfig.size used to register Role.INDEX_KEY "
            f"in _extra_buffers_per_layer."
        )

        # Multi-layer coalescing: when INDEX_KEY shares the V2 storage
        # pool with K/V (production MiniMax-M3 at TP=8 makes K, V, and
        # INDEX_KEY all 256 bytes/token so V2 coalesces them by
        # ``(life_cycle_id, single_buffer_size)``), ``page_upper`` is
        # not the per-layer page count — it is the max page index from
        # this buffer's base in the coalesced pool. Decompose it into
        # ``num_slots = (page_upper + layer_offset_pages) // scale`` so
        # the per-layer view has the right shape. ``scale`` here is the
        # buffers-per-slot count; for a non-coalesced INDEX_KEY pool it
        # equals 1 and ``num_slots == page_upper``, matching the legacy
        # ``shape[0] == page_upper`` behavior.
        scale = int(converter.scale)
        layer_offset_pages = int(converter.layer_offset)
        num_slots_total = page_upper + layer_offset_pages
        assert num_slots_total % scale == 0, (
            f"V2 storage inconsistency for INDEX_KEY of layer "
            f"{layer_idx}: page_upper + layer_offset_pages = "
            f"{num_slots_total} is not divisible by scale = {scale}."
        )
        num_slots = num_slots_total // scale

        if scale == 1:
            # Non-coalesced INDEX_KEY pool: the per-buffer stride is
            # the entire page, so ``[page_upper, tokens_per_block,
            # num_heads, head_dim]`` is the correct contiguous view.
            shape = [page_upper, self.tokens_per_block, num_heads, head_dim]
            return convert_to_torch_tensor(TensorWrapper(addr, torch_dtype, shape))

        # Coalesced pool: build a ``[num_slots, scale, tokens_per_block,
        # num_heads, head_dim]`` view at INDEX_KEY's base, then slice
        # ``[:, 0]`` to extract this layer's INDEX_KEY data. The slice
        # preserves dim-0 stride = ``scale * page_stride`` bytes, so
        # ``view[s, w, h, d]`` lands on the correct byte for any
        # ``s`` in [0, num_slots).
        full_slot_shape = [num_slots, scale, self.tokens_per_block, num_heads, head_dim]
        full_view = convert_to_torch_tensor(TensorWrapper(addr, torch_dtype, full_slot_shape))
        return full_view[:, 0]

    def get_num_available_tokens(
        self, *, token_num_upper_bound: int, batch_size: int = 1, max_num_draft_tokens: int = 0
    ) -> int:
        extra_tokens = self.num_extra_kv_tokens + max_num_draft_tokens
        # Token num upper bound is the maximum number of tokens that can be allocated in the kv cache manager.
        # We need to add extra tokens to the token num upper bound to account for the extra tokens.
        clamped = (
            self.impl.clamp_max_seq_len_for_mem(batch_size, token_num_upper_bound + extra_tokens)
            - extra_tokens
        )
        # clamp_max_seq_len_for_mem considers all tiers (GPU + host).  When
        # max_tokens is explicitly set, cap by GPU-only capacity so callers
        # (e.g. CUDA graph warmup) don't exceed the GPU pool.
        if self._gpu_max_tokens is not None:
            clamped = min(clamped, self._gpu_max_tokens - extra_tokens)
        return clamped

    def get_num_free_blocks(self) -> int:
        # NOTE This method is used to get the number of blocks in the primary pool not the FREE blocks.
        # However, since we only use this function when the kv cache manager is empty, so it is safe to do so.
        assert len(self.kv_cache_map) == 0, (
            "get_num_free_blocks is only used when the kv cache manager is empty"
        )
        max_num_pages = max(
            [
                self.impl.get_page_index_upper_bound(layer_id, Role.KEY)
                for layer_id in typed_range(LayerId(self.num_local_layers))
            ]
        )
        return max_num_pages // self.kv_factor

    def commit_scheduled_kv_cache_stats(self, scheduled_batch: ScheduledRequests) -> None:
        if self.is_draft or not self.enable_stats:
            return
        dirty_req_ids = self.impl.get_dirty_stats_kv_cache_ids()
        for req in scheduled_batch.all_requests():
            if req.py_request_id in dirty_req_ids:
                kv_cache = self.kv_cache_map.get(req.py_request_id)
                if kv_cache is None:
                    continue
                request_stats = kv_cache.commit_pending_stats()
                if not req.is_dummy and not request_stats.empty:
                    req.update_kv_cache_perf_metrics(
                        request_stats.alloc_total_blocks,
                        request_stats.alloc_new_blocks,
                        request_stats.reused_blocks,
                        request_stats.missed_blocks,
                    )

    # ---- Scheduling API (called by KVCacheV2Scheduler) ----

    def is_request_active(self, request_id: int) -> bool:
        """Return True if *request_id* has a live, non-suspended KV cache."""
        kv_cache = self.kv_cache_map.get(request_id)
        return kv_cache is not None and kv_cache.is_active

    def _effective_draft_len(self, req: LlmRequest) -> int:
        """Draft token length to use for next-step KV capacity calculation.

        For a disagg gen request whose KV transmission just completed
        (state == DISAGG_GENERATION_TRANS_COMPLETE), py_draft_tokens is
        still [] when the scheduler asks for capacity, because it gets
        mirrored from context_phase_params.draft_tokens later in
        _prepare_disagg_gen_transmission_complete (which runs AFTER the
        scheduler in the executor loop). Without compensating here, the
        first gen forward writes 1 + len(ctx_draft_tokens) tokens into
        KV cache but only +1 was reserved, OOB-ing the KV block table at
        the next tokens_per_block-aligned boundary.
        """
        draft_len = get_draft_token_length(req)
        if (
            draft_len == 0
            and req.is_disagg_generation_transmission_complete
            and req.context_phase_params is not None
        ):
            ctx_draft_tokens = req.context_phase_params.draft_tokens
            if ctx_draft_tokens is not None:
                draft_len = len(ctx_draft_tokens)
        return draft_len

    def _required_gen_capacity(self, req: LlmRequest, current_capacity: int) -> int:
        """Compute generation KV cache capacity for a request.

        Grows *current_capacity* by 1 + draft tokens.
        """
        return current_capacity + 1 + self._effective_draft_len(req)

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

        draft_len = self._effective_draft_len(req)
        self._allocated_draft_lens[req.py_request_id] = draft_len
        return kv_cache.resize(self._required_gen_capacity(req, kv_cache.capacity))

    def revert_allocate_generation(self, req: LlmRequest) -> None:
        """Undo the capacity growth from try_allocate_generation.

        When attention DP causes can_queue=False after scheduling, the
        forward pass is skipped but the scheduler already grew each
        generation request's KV cache capacity by 1 (+draft tokens).
        This method shrinks capacity back to undo that spurious growth
        so it does not accumulate across iterations and overflow the
        host page-index buffer.

        Mirror the effective draft length used in _required_gen_capacity
        so disagg-gen-trans-complete revert stays symmetric.
        """
        kv_cache = self.kv_cache_map.get(req.py_request_id)
        if kv_cache is None or not kv_cache.is_active:
            return
        draft_len = self._allocated_draft_lens.pop(
            req.py_request_id, self._effective_draft_len(req)
        )
        reverted_cap = kv_cache.capacity - 1 - draft_len
        if reverted_cap < 0:
            return
        if not kv_cache.resize(reverted_cap):
            raise RuntimeError(
                f"Failed to revert KV cache capacity for request "
                f"{req.py_request_id} from {kv_cache.capacity} to "
                f"{reverted_cap}"
            )

    def revert_allocate_context(self, req: LlmRequest) -> None:
        """Undo the capacity growth from this iteration's context resize."""
        pre_cap = getattr(req, "py_ctx_pre_resize_cap", None)
        if pre_cap is None:
            return
        req.py_ctx_pre_resize_cap = None
        kv_cache = self.kv_cache_map.get(req.py_request_id)
        if kv_cache is None or not kv_cache.is_active:
            return
        if pre_cap >= kv_cache.capacity:
            return
        if kv_cache.history_length > pre_cap:
            self.free_resources(req)
            return
        history_length = min(kv_cache.history_length, pre_cap)
        if not kv_cache.resize(pre_cap, history_length):
            raise RuntimeError(
                f"Failed to revert KV cache capacity for context "
                f"request {req.py_request_id} from {kv_cache.capacity} "
                f"to {pre_cap}"
            )
        if pre_cap > 0:
            kv_cache.suspend()

    def _restore_page_index_bufs(self, request_id: int, kv_cache) -> None:
        """Re-connect host page-index buffers after resume().

        suspend() clears the base_page_index_buf pointers (sets them to
        None) so the KV cache stops writing page indices to the host
        buffer.  After resume(), the KV cache has re-locked pages but
        copy_batch_block_offsets still reads from the host buffer, so we
        must re-connect the buffers to avoid stale/zero page indices that
        would cause illegal memory accesses during the forward pass.
        """
        index = self.index_mapper.get_index(request_id)
        for i in range(self.max_beam_width):
            for pool_idx in range(self.num_pools):
                buffer: torch.Tensor = self.host_kv_cache_block_offsets[
                    pool_idx, index * self.max_beam_width + i, 0
                ]
                kv_cache.set_base_page_index_buf(i, pool_idx, memoryview(buffer.numpy()))

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
        assert not req.is_disagg_generation_init_state, (
            f"req {req.py_request_id}: use prepare_disagg_gen_init"
        )
        return self._prepare_context_impl(req)

    def _prepare_context_impl(self, req: LlmRequest) -> bool:
        if req.is_first_context_chunk:
            kv_cache = self.kv_cache_map.get(req.py_request_id)
            if kv_cache is None:
                all_tokens = req.get_tokens(DEFAULT_BEAM_INDEX)
                # Last token cannot be recovered, so we don't include it in
                # the input tokens to look up for the block that can be reused.
                if self.enable_block_reuse:
                    tokens = self._augment_tokens_for_block_reuse(
                        all_tokens, req, end=len(all_tokens) - 1
                    )
                else:
                    tokens = None
                kv_cache = self._create_kv_cache(
                    req.py_request_id,
                    req.lora_task_id,
                    tokens,
                    cache_salt=req.cache_salt,
                    is_dummy=req.is_dummy,
                    expected_prompt_length=req.prompt_len - 1,
                )
                if kv_cache is None:
                    return False
                kv_cache.cuda_stream = self._stream.cuda_stream

            if not self.enable_block_reuse:
                kv_cache.stop_committing()
            else:
                req.context_current_position = kv_cache.num_committed_tokens
                req.set_prepopulated_prompt_len(
                    kv_cache.num_committed_tokens, self.tokens_per_block
                )

            if req.is_disagg_generation_init_state:
                # Disagg generation receives prompt KV from the context worker;
                # scratch blocks are only valid for local prefill chunks.
                kv_cache.enable_swa_scratch_reuse = False
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

        Returns True on success, False if resize failed (first chunk is
        suspended on failure).
        """
        assert not req.is_disagg_generation_init_state, (
            f"req {req.py_request_id}: use prepare_disagg_gen_init"
        )
        kv_cache = self.kv_cache_map.get(req.py_request_id)
        if kv_cache is None:
            return False

        target = req.context_current_position + num_tokens + self.num_extra_kv_tokens
        capacity = max(kv_cache.capacity, target)
        pre_cap = kv_cache.capacity

        success = kv_cache.resize(capacity)
        if not success:
            if req.is_first_context_chunk:
                kv_cache.suspend()
            return False
        req.py_ctx_pre_resize_cap = pre_cap if capacity > pre_cap else None
        return True

    def prepare_disagg_gen_init(self, req: LlmRequest) -> bool:
        """Prepare KV cache for a disagg generation init request.

        Allocates capacity for the full prompt (+ draft) and sets
        ``kv_cache.history_length`` to ``prompt_len``. Returns True on
        success, False if preparation or resize failed (cache is suspended
        on resize failure).
        """
        if not self._prepare_context_impl(req):
            return False

        kv_cache = self.kv_cache_map.get(req.py_request_id)
        if kv_cache is None:
            return False

        # prompt_len is the full incoming prompt length, robust to block
        # reuse (which may leave a non-zero context_current_position).
        target = req.prompt_len + get_draft_token_length(req) + self.num_extra_kv_tokens
        capacity = max(kv_cache.capacity, target)
        pre_cap = kv_cache.capacity

        success = kv_cache.resize(capacity, req.prompt_len)
        if not success:
            if req.is_first_context_chunk:
                kv_cache.suspend()
            return False
        req.py_ctx_pre_resize_cap = pre_cap if capacity > pre_cap else None
        return True

    def get_history_length(self, req: LlmRequest) -> int | None:
        """Return the cache's current history_length, or None if no cache.

        Exposes the per-request SWA history watermark so callers
        (e.g., the disagg transceiver) can verify scheduler/cache contracts
        without reaching into ``kv_cache_map`` directly.
        """
        kv_cache = self.kv_cache_map.get(req.py_request_id)
        if kv_cache is None:
            return None
        return kv_cache.history_length

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
                f"(target capacity {new_capacity})"
            )

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
                    kv_cache = self._create_kv_cache(
                        req.py_request_id,
                        req.lora_task_id,
                        None,
                        cache_salt=req.cache_salt,
                        is_dummy=req.is_dummy,
                    )
                    kv_cache.stop_committing()
                if not self._resume_and_restore(req.py_request_id, kv_cache):
                    raise RuntimeError(
                        f"Failed to resume draft KV cache for request {req.py_request_id}"
                    )
                draft_len = get_draft_token_length(req)
                capacity = (
                    req.context_current_position
                    + req.context_chunk_size
                    + draft_len
                    + self.num_extra_kv_tokens
                )
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
                new_cap = self._required_gen_capacity(req, kv_cache.capacity)
                # Pad the resize up to _kv_reserve_draft_tokens (see __init__);
                # no-op when reserve == draft_token_length.
                reserve_slack = self._kv_reserve_draft_tokens - get_draft_token_length(req)
                if reserve_slack > 0:
                    new_cap += reserve_slack
                if not kv_cache.resize(new_cap):
                    raise RuntimeError(
                        f"Draft KV cache generation resize failed for request "
                        f"{req.py_request_id}: could not resize to {new_cap} tokens"
                    )

    def _augment_tokens_for_block_reuse(
        self, tokens: Sequence[int], req: LlmRequest, start: int = 0, end: int | None = None
    ) -> Sequence[TokenIdExt]:
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

        if (
            req.multimodal_hashes is None
            or req.multimodal_positions is None
            or req.multimodal_lengths is None
        ):
            return tokens[chunk_start:chunk_end] if is_sliced else tokens

        result: list[TokenIdExt] = list(tokens[chunk_start:chunk_end])
        run_metadata = _resolve_multimodal_run_metadata(req)
        if run_metadata is not None:
            return _augment_tokens_with_mm_run_metadata(
                self.vocab_size, result, req.multimodal_hashes, run_metadata, chunk_start, chunk_end
            )

        return _augment_tokens_with_contiguous_mm_metadata(
            self.vocab_size,
            result,
            req.multimodal_hashes,
            req.multimodal_positions,
            req.multimodal_lengths,
            chunk_start,
            chunk_end,
        )

    def _stats_window_size(self, window_size: Optional[int]) -> int:
        return self.max_seq_len if window_size is None else int(window_size)

    def _stats_life_cycle_window_size(self, life_cycle) -> Optional[int]:
        if not isinstance(life_cycle, AttnLifeCycle):
            return None
        return self._stats_window_size(life_cycle.window_size)

    def _storage_pool_groups_by_window(self) -> dict[int, set[int]]:
        pool_groups_by_window: dict[int, set[int]] = defaultdict(set)
        for life_cycle_id, life_cycle in self.impl._life_cycles.attention_life_cycles():
            pool_group_id = self.impl._storage.get_pool_group_index(life_cycle_id)
            pool_groups_by_window[self._stats_window_size(life_cycle.window_size)].add(
                int(pool_group_id)
            )
        return pool_groups_by_window

    @staticmethod
    def _windows_by_pool_group(
        pool_groups_by_window: dict[int, set[int]],
    ) -> dict[int, tuple[int, ...]]:
        windows_by_pool_group: dict[int, set[int]] = defaultdict(set)
        for window_size, pool_group_ids in pool_groups_by_window.items():
            for pool_group_id in pool_group_ids:
                windows_by_pool_group[pool_group_id].add(window_size)
        return {
            pool_group_id: tuple(sorted(window_sizes))
            for pool_group_id, window_sizes in windows_by_pool_group.items()
        }

    @staticmethod
    def _filter_iteration_stats_delta(delta, field_names) -> KVCacheIterationStatsDelta:
        filtered = KVCacheIterationStatsDelta()
        for field_name in field_names:
            setattr(filtered, field_name, getattr(delta, field_name))
        return filtered

    @staticmethod
    def _add_iteration_stats_delta(
        bucket: dict[int, KVCacheIterationStatsDelta], key: int, delta: KVCacheIterationStatsDelta
    ) -> None:
        if delta.empty:
            return
        if key not in bucket:
            bucket[key] = delta.copy()
            return
        bucket[key].add(delta)

    @staticmethod
    def _iteration_cache_hit_rate(stats) -> float:
        total = stats.iter_reused_blocks + stats.iter_missed_blocks
        if stats.iter_reused_blocks == 0 or total == 0:
            return 0.0
        return stats.iter_reused_blocks / total

    @staticmethod
    def _apply_iteration_stats_delta(
        stats, delta, field_names=KV_CACHE_ITERATION_STATS_DELTA_FIELDS
    ) -> None:
        if delta is None:
            return
        for field_name in field_names:
            setattr(stats, field_name, getattr(delta, field_name))
        stats.iter_cache_hit_rate = KVCacheManagerV2._iteration_cache_hit_rate(stats)

    def _build_iteration_stats(
        self,
        pool_group_ids: Iterable[int],
        primary_stats,
        secondary_stats_by_level,
        primary_peak_stats,
        secondary_peak_stats_by_level,
        delta,
        field_names=KV_CACHE_ITERATION_STATS_DELTA_FIELDS,
    ):
        pool_group_ids = tuple(pool_group_ids)
        stats = KvCacheIterationStats()
        stats.primary_max_num_blocks = sum(
            primary_stats[pool_group_id].total for pool_group_id in pool_group_ids
        )
        stats.primary_free_num_blocks = sum(
            primary_stats[pool_group_id].available for pool_group_id in pool_group_ids
        )
        stats.primary_used_num_blocks = stats.primary_max_num_blocks - stats.primary_free_num_blocks
        stats.primary_evictable_num_blocks = sum(
            primary_stats[pool_group_id].evictable for pool_group_id in pool_group_ids
        )
        stats.primary_peak_free_num_blocks = sum(
            primary_peak_stats[pool_group_id].available for pool_group_id in pool_group_ids
        )
        stats.primary_peak_used_num_blocks = sum(
            primary_peak_stats[pool_group_id].unavailable for pool_group_id in pool_group_ids
        )
        stats.primary_peak_evictable_num_blocks = sum(
            primary_peak_stats[pool_group_id].evictable for pool_group_id in pool_group_ids
        )
        stats.secondary_max_num_blocks = sum(
            level_stats[pool_group_id].total
            for level_stats in secondary_stats_by_level
            for pool_group_id in pool_group_ids
        )
        stats.secondary_free_num_blocks = sum(
            level_stats[pool_group_id].available
            for level_stats in secondary_stats_by_level
            for pool_group_id in pool_group_ids
        )
        stats.secondary_used_num_blocks = (
            stats.secondary_max_num_blocks - stats.secondary_free_num_blocks
        )
        stats.secondary_evictable_num_blocks = sum(
            level_stats[pool_group_id].evictable
            for level_stats in secondary_stats_by_level
            for pool_group_id in pool_group_ids
        )
        stats.secondary_peak_free_num_blocks = sum(
            peak_stats[pool_group_id].available
            for peak_stats in secondary_peak_stats_by_level
            for pool_group_id in pool_group_ids
        )
        stats.secondary_peak_used_num_blocks = sum(
            peak_stats[pool_group_id].unavailable
            for peak_stats in secondary_peak_stats_by_level
            for pool_group_id in pool_group_ids
        )
        stats.secondary_peak_evictable_num_blocks = sum(
            peak_stats[pool_group_id].evictable
            for peak_stats in secondary_peak_stats_by_level
            for pool_group_id in pool_group_ids
        )
        self._apply_iteration_stats_delta(stats, delta, field_names)
        return stats

    def _collect_iteration_stats_deltas(
        self, raw_iteration_stats, storage
    ) -> tuple[dict, dict, dict, dict]:
        reuse_deltas_by_window: dict[int, KVCacheIterationStatsDelta] = {}
        reuse_deltas_by_life_cycle: dict[int, KVCacheIterationStatsDelta] = {}
        pool_group_deltas_by_window: dict[int, KVCacheIterationStatsDelta] = {}
        pool_group_deltas: dict[int, KVCacheIterationStatsDelta] = {}

        for life_cycle_id, delta in raw_iteration_stats.items():
            life_cycle = self.impl._life_cycles.get_life_cycle(life_cycle_id)
            pool_group_id = int(storage.get_pool_group_index(life_cycle_id))
            window_size = self._stats_life_cycle_window_size(life_cycle)

            pool_group_delta = self._filter_iteration_stats_delta(
                delta, KV_CACHE_ITERATION_STATS_POOL_GROUP_FIELDS
            )
            self._add_iteration_stats_delta(pool_group_deltas, pool_group_id, pool_group_delta)
            if window_size is not None:
                self._add_iteration_stats_delta(
                    pool_group_deltas_by_window, window_size, pool_group_delta
                )

            reuse_delta = self._filter_iteration_stats_delta(
                delta, KV_CACHE_ITERATION_STATS_REUSE_FIELDS
            )
            if reuse_delta.empty:
                continue
            reuse_deltas_by_life_cycle[int(life_cycle_id)] = reuse_delta.copy()
            if window_size is not None:
                self._add_iteration_stats_delta(reuse_deltas_by_window, window_size, reuse_delta)

        return (
            reuse_deltas_by_window,
            reuse_deltas_by_life_cycle,
            pool_group_deltas_by_window,
            pool_group_deltas,
        )

    def _build_window_iteration_stats(
        self,
        window_size: int,
        pool_groups_by_window: dict[int, set[int]],
        windows_by_pool_group: dict[int, tuple[int, ...]],
        primary_stats,
        secondary_stats_by_level,
        primary_peak_stats,
        secondary_peak_stats_by_level,
        pool_group_delta,
        reuse_delta,
    ):
        pool_group_ids = tuple(
            pool_group_id
            for pool_group_id in pool_groups_by_window.get(window_size, set())
            if windows_by_pool_group.get(pool_group_id) == (window_size,)
        )
        stats = self._build_iteration_stats(
            pool_group_ids,
            primary_stats,
            secondary_stats_by_level,
            primary_peak_stats,
            secondary_peak_stats_by_level,
            pool_group_delta,
            KV_CACHE_ITERATION_STATS_POOL_GROUP_FIELDS,
        )
        self._apply_iteration_stats_delta(stats, reuse_delta, KV_CACHE_ITERATION_STATS_REUSE_FIELDS)
        return stats

    def _build_pool_group_iteration_stats(
        self,
        pool_group_id: int,
        windows_by_pool_group: dict[int, tuple[int, ...]],
        primary_stats,
        secondary_stats_by_level,
        primary_peak_stats,
        secondary_peak_stats_by_level,
        pool_group_delta,
    ) -> KVCacheV2PoolGroupIterationStats:
        return KVCacheV2PoolGroupIterationStats(
            pool_group_id=pool_group_id,
            slot_size=tuple(primary_stats[pool_group_id].slot_size),
            window_sizes=windows_by_pool_group.get(pool_group_id, ()),
            stats=self._build_iteration_stats(
                (pool_group_id,),
                primary_stats,
                secondary_stats_by_level,
                primary_peak_stats,
                secondary_peak_stats_by_level,
                pool_group_delta,
                KV_CACHE_ITERATION_STATS_POOL_GROUP_FIELDS,
            ),
        )

    def _build_life_cycle_iteration_stats(
        self,
        life_cycle_id: int,
        storage,
        primary_stats,
        secondary_stats_by_level,
        primary_peak_stats,
        secondary_peak_stats_by_level,
        reuse_delta,
    ) -> KVCacheV2LifeCycleIterationStats:
        typed_life_cycle_id = LifeCycleId(life_cycle_id)
        life_cycle = self.impl._life_cycles.get_life_cycle(typed_life_cycle_id)
        pool_group_id = int(storage.get_pool_group_index(typed_life_cycle_id))
        return KVCacheV2LifeCycleIterationStats(
            life_cycle_id=life_cycle_id,
            pool_group_id=pool_group_id,
            window_size=self._stats_life_cycle_window_size(life_cycle),
            kind="attention" if isinstance(life_cycle, AttnLifeCycle) else "ssm",
            stats=self._build_iteration_stats(
                (),
                primary_stats,
                secondary_stats_by_level,
                primary_peak_stats,
                secondary_peak_stats_by_level,
                reuse_delta,
                KV_CACHE_ITERATION_STATS_REUSE_FIELDS,
            ),
        )

    def get_kv_cache_stats(self):
        kv_cache_stats = KvCacheStats()
        pool_group_stats = self.impl._storage.get_statistics(GPU_LEVEL)
        max_num_blocks = sum(stat.total for stat in pool_group_stats)
        free_num_blocks = sum(stat.available for stat in pool_group_stats)
        committed_stats = self.impl.get_committed_stats()

        kv_cache_stats.max_num_blocks = max_num_blocks
        kv_cache_stats.free_num_blocks = free_num_blocks
        kv_cache_stats.used_num_blocks = max_num_blocks - free_num_blocks
        kv_cache_stats.tokens_per_block = self.tokens_per_block
        kv_cache_stats.alloc_total_blocks = committed_stats.alloc_total_blocks
        kv_cache_stats.alloc_new_blocks = committed_stats.alloc_new_blocks
        kv_cache_stats.reused_blocks = committed_stats.reused_blocks
        kv_cache_stats.missed_blocks = committed_stats.missed_blocks
        total = kv_cache_stats.reused_blocks + kv_cache_stats.missed_blocks
        kv_cache_stats.cache_hit_rate = (
            0.0
            if kv_cache_stats.reused_blocks == 0 or total == 0
            else kv_cache_stats.reused_blocks / total
        )
        kv_cache_stats.num_free_blocks_per_window_size = {
            window_size: sum(
                pool_group_stats[pool_group_id].available for pool_group_id in pool_group_ids
            )
            for window_size, pool_group_ids in self._storage_pool_groups_by_window().items()
        }
        kv_cache_stats.allocated_bytes = self.impl.get_quota(GPU_LEVEL)

        return kv_cache_stats

    def flush_iteration_events(self):
        if self.event_manager is not None:
            self.event_manager.flush_iteration_events()

    def get_latest_events(self, timeout_ms: Optional[float] = None):
        if self.event_manager is None:
            return []
        return self.event_manager.get_latest_events(timeout_ms)

    def get_iteration_stats(self):
        if not self.enable_stats:
            return None

        storage = self.impl._storage
        pool_groups_by_window = self._storage_pool_groups_by_window()
        windows_by_pool_group = self._windows_by_pool_group(pool_groups_by_window)
        raw_iteration_stats = self.impl.get_and_reset_iteration_stats()
        primary_peak_stats = self.impl.get_and_reset_iteration_peak_block_stats(GPU_LEVEL)
        secondary_peak_stats_by_level = [
            self.impl.get_and_reset_iteration_peak_block_stats(CacheLevel(level))
            for level in range(1, int(storage.num_cache_levels))
        ]
        (
            reuse_deltas_by_window,
            reuse_deltas_by_life_cycle,
            pool_group_deltas_by_window,
            pool_group_deltas,
        ) = self._collect_iteration_stats_deltas(raw_iteration_stats, storage)

        windows = set(pool_groups_by_window)
        windows.update(reuse_deltas_by_window)
        windows.update(pool_group_deltas_by_window)
        primary_stats = storage.get_statistics(GPU_LEVEL)
        secondary_stats_by_level = [
            storage.get_statistics(CacheLevel(level))
            for level in range(1, int(storage.num_cache_levels))
        ]

        stats_by_window = {
            window_size: self._build_window_iteration_stats(
                window_size,
                pool_groups_by_window,
                windows_by_pool_group,
                primary_stats,
                secondary_stats_by_level,
                primary_peak_stats,
                secondary_peak_stats_by_level,
                pool_group_deltas_by_window.get(window_size),
                reuse_deltas_by_window.get(window_size),
            )
            for window_size in sorted(windows)
        }

        pool_group_ids = sorted(set(windows_by_pool_group) | set(pool_group_deltas))
        stats_by_pool_group = {
            pool_group_id: self._build_pool_group_iteration_stats(
                pool_group_id,
                windows_by_pool_group,
                primary_stats,
                secondary_stats_by_level,
                primary_peak_stats,
                secondary_peak_stats_by_level,
                pool_group_deltas.get(pool_group_id),
            )
            for pool_group_id in pool_group_ids
        }

        stats_by_life_cycle = {
            life_cycle_id: self._build_life_cycle_iteration_stats(
                life_cycle_id,
                storage,
                primary_stats,
                secondary_stats_by_level,
                primary_peak_stats,
                secondary_peak_stats_by_level,
                reuse_delta,
            )
            for life_cycle_id, reuse_delta in sorted(reuse_deltas_by_life_cycle.items())
        }

        return KVCacheV2IterationStatsReport(
            stats_by_window, stats_by_pool_group, stats_by_life_cycle
        )

    def get_block_ids_per_seq(self, request_ids: List[int]) -> torch.Tensor:
        block_ids_per_seq = self.get_batch_cache_indices(request_ids)
        block_ids_per_seq_tensors = [
            torch.tensor(
                [i // self.num_local_layers if i != BAD_PAGE_INDEX else 0 for i in sublist],
                dtype=torch.int,
            )
            for sublist in block_ids_per_seq
        ]
        padded_tensor = torch.nn.utils.rnn.pad_sequence(
            block_ids_per_seq_tensors, batch_first=True, padding_value=0
        )
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
        encoder_output_lens: Optional[List[int]] = None,
        num_extra_decoding_steps: int = 0,
        draft_kv_cache_manager: Optional["BaseResourceManager"] = None,
    ):
        _kv_draft = (
            kv_reserve_draft_tokens if kv_reserve_draft_tokens is not None else max_num_draft_tokens
        )

        beam_width = max_beam_width
        requests = []

        def release_resources(
            current_request: LlmRequest, free_draft_resources: bool = False
        ) -> None:
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
            sampling_params = SamplingParams(
                n=beam_width, best_of=beam_width, use_beam_search=beam_width > 1
            )
            # Here 1+max_num_draft_tokens is used to extend the prompt length to
            # a non-zero number to skip illegal memory access issue in MLA kernel
            # during warmup.
            token_num = token_nums[i] if token_nums is not None else 1 + max_num_draft_tokens
            # token_num - 1 is the past history length in generation.
            history_hint = max(0, token_num - 1) if is_gen else None
            encoder_output_len = encoder_output_lens[i] if encoder_output_lens is not None else None
            encoder_input_tokens = (
                [1] * encoder_output_len if encoder_output_len is not None else None
            )
            # Using 1 instead of 0 prevents NaN during warmup in e.g. Deepseek
            input_tokens = [1 for _ in range(token_num)]
            req = LlmRequest(
                request_id=req_id,
                max_new_tokens=1,
                input_tokens=input_tokens,
                sampling_config=SamplingConfig(sampling_params._get_sampling_config()),
                is_streaming=False,
                encoder_input_tokens=encoder_input_tokens,
                encoder_output_len=encoder_output_len,
            )
            req.is_dummy_request = True
            req.paged_kv_block_ids = []
            if prepare_resource:
                # Dummy/warmup request. ``stop_committing()`` below blocks all
                # writes to the radix tree, so the choice of branch does not
                # affect committed state. ``cache_salt`` is left defaulted
                # to None to avoid coupling synthetic data to any salted branch.
                kv_cache = self._create_kv_cache(
                    req.py_request_id, req.lora_task_id, input_tokens, is_dummy=req.is_dummy
                )
                # Saturated IndexMapper (e.g. disagg gen trans in progress)
                # returns None; retry next iter.
                if kv_cache is None:
                    release_resources(req)
                    return None
                assert kv_cache.num_committed_tokens == 0
                success = kv_cache.resume(self._stream.cuda_stream)
                if not success:
                    release_resources(req)
                    return None
                kv_cache.stop_committing()
                dummy_capacity = token_num + self.num_extra_kv_tokens + num_extra_decoding_steps
                if is_gen:
                    kv_cache.enable_swa_scratch_reuse = False
                # Need to hint the committed history to activate stale-block
                # optimization and match the solver's pool budget.
                success = kv_cache.resize(dummy_capacity, history_length=history_hint)
                if not success:
                    release_resources(req)
                    return None
                draft_kv_cache = None
                if draft_kv_cache_manager is not None:
                    draft_kv_cache = draft_kv_cache_manager._create_kv_cache(
                        req.py_request_id, req.lora_task_id, input_tokens, is_dummy=req.is_dummy
                    )
                    # Dummy path: see comment above, no salt.
                    if draft_kv_cache is None:
                        release_resources(req)
                        return None
                    success = draft_kv_cache.resume(draft_kv_cache_manager._stream.cuda_stream)
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
                    new_capacity = kv_cache.capacity + _kv_draft + 1
                    success = kv_cache.resize(new_capacity, history_length=history_hint)
                    if not success:
                        release_resources(req, free_draft_resources=draft_kv_cache is not None)
                        return None
                    if draft_kv_cache is not None:
                        success = draft_kv_cache.resize(new_capacity)
                        if not success:
                            release_resources(req, free_draft_resources=True)
                            return None

            if use_mrope:
                _populate_dummy_mrope_config(req, token_num, is_gen)
            requests.append(req)

        return requests

    def try_commit_blocks(self, request: LlmRequest) -> None:
        should_block_reuse = (
            self.enable_block_reuse and not self.is_draft and not request.is_dummy_request
        )
        if not should_block_reuse:
            return

        kv_cache = self.kv_cache_map.get(request.py_request_id)
        if kv_cache is None:
            return

        if request.context_current_position > kv_cache.num_committed_tokens:
            tokens = self._augment_tokens_for_block_reuse(
                request.get_tokens(DEFAULT_BEAM_INDEX),
                request,
                start=kv_cache.num_committed_tokens,
                end=request.context_current_position,
            )
            kv_cache.commit(tokens)
        if request.context_remaining_length == 0:
            kv_cache.stop_committing()

    def release_index_slot(self, request_id: int) -> None:
        """Release IndexMapper slot early while keeping KV cache blocks allocated.

        After prefill completes on a context-only worker, the IndexMapper slot
        (used for host_kv_cache_block_offsets during model forward) is no longer
        needed.  Releasing it early allows new requests to be scheduled while
        the KV cache blocks are still being transferred via NIXL/UCX.
        """
        kv_cache = self.kv_cache_map.get(request_id)
        if kv_cache is not None:
            for i in range(self.max_beam_width):
                for pool_idx in range(self.num_pools):
                    kv_cache.set_base_page_index_buf(i, pool_idx, None)
        self.index_mapper.remove_sequence(request_id)
        self._early_freed_index_requests.add(request_id)

    def free_resources(self, request: LlmRequest, pin_on_release: bool = False):
        self._allocated_draft_lens.pop(request.py_request_id, None)
        kv_cache = self.kv_cache_map.pop(request.py_request_id, None)
        if kv_cache is None:
            self.impl.clear_stats_excluded(request.py_request_id)
            return
        kv_cache.discard_pending_stats()
        kv_cache.close()
        self.impl.clear_stats_excluded(request.py_request_id)
        if request.py_request_id in self._early_freed_index_requests:
            self._early_freed_index_requests.discard(request.py_request_id)
        else:
            self.index_mapper.remove_sequence(request.py_request_id)

    def get_batch_cache_indices(
        self,
        request_ids: List[int],
        layer_idx: Optional[int] = None,
        num_blocks_per_seq: Optional[Sequence[int]] = None,
    ) -> List[List[int]]:
        if layer_idx is None:
            pool_id = 0
        else:
            pool_id = self.layer_to_pool_mapping_dict[self.layer_offsets[layer_idx]]
        return self._get_batch_cache_indices_by_pool_id(
            request_ids,
            pool_id=pool_id,
            is_kv_aggregate=True,
            num_blocks_per_seq=num_blocks_per_seq,
        )

    def _get_batch_cache_indices_by_pool_id(
        self,
        request_ids: List[int],
        *,
        pool_id: int = 0,
        is_kv_aggregate: bool = True,
        num_blocks_per_seq: Optional[Sequence[int]] = None,
    ) -> List[List[int]]:
        if is_kv_aggregate:
            # Div by kv_factor to index kv cache with size
            # [num_blocks, kv_factor, tokens_per_block, num_kv_heads, head_dim]
            div_factor = self.kv_factor
        else:
            div_factor = 1

        index_scale = int(self.index_scales[pool_id])
        res = []

        for req_idx, req_id in enumerate(request_ids):
            kv_cache = self.kv_cache_map[req_id]
            # Zero-copy page-index buffers are padded to max_blocks_per_seq.
            # Only convert blocks owned by this request; attention callers
            # discard the padded tail immediately.
            num_blocks = kv_cache.num_blocks
            if num_blocks_per_seq is not None:
                num_blocks = min(num_blocks, num_blocks_per_seq[req_idx])
            base_page_indices = kv_cache.get_base_page_indices(pool_id)[:num_blocks]
            res.append(
                [
                    base_page_index * index_scale // div_factor
                    if base_page_index != BAD_PAGE_INDEX
                    else BAD_PAGE_INDEX
                    for base_page_index in base_page_indices
                ]
            )

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
            self.get_layer_bytes_per_token(local_layer_idx=local_layer_idx, data_role=data_role)
            for local_layer_idx in range(self.num_local_layers)
            for data_role in data_roles
        )

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
        elif data_role in [Role.KEY, Role.VALUE, Role.KEY_BLOCK_SCALE, Role.VALUE_BLOCK_SCALE]:
            if data_role in [Role.KEY_BLOCK_SCALE, Role.VALUE_BLOCK_SCALE]:
                assert self.dtype == DataType.NVFP4, (
                    "NVFP4 is the only supported dtype for block quant data roles"
                )
            if data_role == Role.VALUE:
                assert self.kv_cache_type != CacheTypeCpp.SELFKONLY, (
                    "VALUE data role is not supported for SELFKONLY cache type"
                )
            kv_factor = 1
        else:
            raise ValueError(f"Invalid data role: {data_role}")

        cache_size_per_token = (
            kv_factor
            * self.num_kv_heads_per_layer[local_layer_idx]
            * self.head_dim_per_layer[local_layer_idx]
        )

        cache_size_bytes_per_token = get_size_in_bytes(cache_size_per_token, self.dtype)

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
        cache_size: int, quant_vector_size: int, scaling_factor_dtype: DataType
    ) -> int:
        assert cache_size % quant_vector_size == 0, (
            "NVFP4 cache size must be divisible by quant vector size"
        )
        return get_size_in_bytes(cache_size // quant_vector_size, scaling_factor_dtype)

    def check_invalid_values_in_kv_cache(self, fill_with_zero: bool = False) -> bool:
        some_checks_unavailable = False
        has_invalid_values = torch.tensor(
            [False], dtype=torch.bool, device=torch.cuda.current_device()
        )
        pool_handled = set()

        # Handle each layer from start to end to traverse the whole KV cache.
        for layer_id, layer_offset in self.layer_offsets.items():
            pool_id = self.layer_to_pool_mapping_dict[layer_offset]
            if pool_id in pool_handled:
                continue
            buffer = self.get_buffers(layer_id)
            # process in chunks of 256 pages to avoid OoM
            for i in range(0, buffer.shape[0], 256):
                buffer_slice = buffer[i : i + 256]
                try:
                    has_invalid_values.logical_or_(torch.isnan(buffer_slice).any())
                    has_invalid_values.logical_or_(torch.isinf(buffer_slice).any())
                except NotImplementedError:
                    some_checks_unavailable = True
            if fill_with_zero:
                buffer.zero_()
            pool_handled.add(pool_id)
        torch.cuda.synchronize()

        if some_checks_unavailable:
            logger.warning(
                "`torch.isnan` or `torch.isinf` is not implemented for current "
                "kv cache dtype, related checks are skipped"
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
    def get_cache_size_per_token(
        model_config: ModelConfigPython,
        mapping: Mapping,
        num_layers: Optional[int] = None,
        **kwargs,
    ):
        layer_sizes, attention_windows = _get_static_cache_size_layer_components(
            model_config, mapping, num_layers=num_layers, **kwargs
        )
        full_attn_size_per_token = _estimate_full_attn_size_per_token(
            layer_sizes, attention_windows
        )
        swa_size_per_token, swa_size_per_request = _estimate_swa_cache_size(
            layer_sizes,
            attention_windows,
            kwargs["tokens_per_block"],
            context=False,
            scratch=False,
        )
        max_batch_size = int(kwargs.get("max_batch_size") or 0)
        return (
            full_attn_size_per_token + swa_size_per_token,
            swa_size_per_request * max_batch_size,
        )

    def update_context_resources(self, scheduled_batch: ScheduledRequests):
        """Update KV cache for context requests in the current batch.

        This is separated from update_resources (which handles generation
        requests only) because the overlap executor needs context KV cache
        updates to happen before next batch scheduling. Otherwise, the scheduler
        would under-estimate available KV cache for sliding-window attention
        layer. In non-overlap scheduler, you should call it together with
        update_resources().
        """
        for req in scheduled_batch.context_requests:
            if req.py_request_id not in self.kv_cache_map:
                continue
            kv_cache = self.kv_cache_map[req.py_request_id]
            # In the overlap scheduler, iteration N+1's eviction may
            # suspend a ctx request's KV cache while iteration N's
            # update still needs to process it.  Skip the resize — the
            # request will be resumed by the scheduler on the next
            # iteration.
            if not kv_cache.is_active:
                continue
            should_block_reuse = (
                self.enable_block_reuse and not self.is_draft and not req.is_dummy_request
            )
            is_all_reusable = self.block_reuse_policy == BlockReusePolicy.ALL_REUSABLE
            should_resize = not should_block_reuse or not is_all_reusable
            should_commit = is_all_reusable or req.context_remaining_length == 0

            if should_resize:
                success = kv_cache.resize(None, req.context_current_position)
                if not success:
                    raise ValueError(
                        "Failed to resize history length of KV cache for request "
                        f"{req.py_request_id} to {req.context_current_position} tokens "
                        "at context update"
                    )
            if should_commit:
                self.try_commit_blocks(req)
            if req.context_remaining_length == 0:
                # Scratch blocks are only for prefill chunks. Disable them at
                # the context/generation boundary so generation uses normal KV
                # pages before the first generation allocation.
                kv_cache.enable_swa_scratch_reuse = False

    def update_resources(
        self,
        scheduled_batch: ScheduledRequests,
        attn_metadata: "AttentionMetadata" = None,
        kv_cache_dtype_byte_size: float = None,
    ):
        if not self.is_draft:
            _update_kv_cache_draft_token_location(
                self, scheduled_batch, attn_metadata, kv_cache_dtype_byte_size
            )
        # Context request KV cache updates are handled by
        # update_context_resources, called separately from the executor loop.
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
            new_capacity = (
                None
                if req.state in (LlmRequestState.GENERATION_COMPLETE, LlmRequestState.CONTEXT_INIT)
                else kv_cache.capacity - req.py_rewind_len
            )
            success = kv_cache.resize(new_capacity, req.max_beam_num_tokens - 1)
            if not success:
                raise ValueError(
                    f"Failed to resize KV cache for request {req.py_request_id} "
                    f"to capacity {new_capacity} and history length "
                    f"{req.max_beam_num_tokens - 1} tokens at generation update"
                )

    def copy_batch_block_offsets(
        self,
        dst_tensor: torch.Tensor,
        request_ids: List[int],
        beam_width: int,
        num_contexts: int,
        num_seqs: int,
    ):
        assert beam_width == 1, "beam_width must be 1 for KVCacheManagerV2"

        copy_idx = self.index_mapper.get_copy_index(request_ids, num_contexts, beam_width)
        assert copy_idx.shape[0] == num_seqs

        if self.enable_swa_scratch_reuse:
            self._copy_batch_block_offsets_per_layer(
                dst_tensor, request_ids, copy_idx, num_contexts, num_seqs
            )
            return

        copy_batch_block_offsets_to_device(
            self.host_kv_cache_block_offsets,
            dst_tensor,
            copy_idx,
            self.index_scales,
            self.kv_offset,
            self._stream.cuda_stream,
        )

    @staticmethod
    def _derive_reuse_salt(cache_salt: str | None) -> int | None:
        """Derive ``ReuseScope.salt`` (int|None) from the ``cache_salt`` string.

        Deterministic so the same string yields the same reuse namespace across
        processes (matches C++ blockKey hashing on cacheSalt). Shared by cache
        creation, prefetch, and the cache-aware router probe so they all hit the
        same radix-tree namespace.
        """
        if cache_salt is None:
            return None
        digest = hashlib.sha256(cache_salt.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "little")

    def _create_kv_cache(
        self,
        request_id: int,
        lora_task_id: int | None,
        input_tokens: Sequence[TokenIdExt] | None,
        *,
        cache_salt: str | None = None,
        is_dummy: bool = False,
        expected_prompt_length: int | None = None,
    ):
        assert request_id not in self.kv_cache_map, (
            f"KV cache for request {request_id} already exists"
        )
        if self.index_mapper.num_free_slots() == 0:
            logger.warning(
                "No free IndexMapper slots for request %s "
                "(%d/%d slots in use, likely held by DISAGG_GENERATION_TRANS_IN_PROGRESS requests). "
                "Skipping KV cache creation; request will retry next iteration.",
                request_id,
                self.index_mapper.size(),
                self.index_mapper.size(),
            )
            return None
        salt_int = self._derive_reuse_salt(cache_salt)
        kv_cache = self.impl.create_kv_cache(
            ReuseScope(lora_id=lora_task_id, salt=salt_int),
            input_tokens,
            id=request_id,
            expected_prompt_length=expected_prompt_length,
        )
        self.kv_cache_map[request_id] = kv_cache
        if is_dummy:
            self.impl.mark_stats_excluded(request_id)
            kv_cache.discard_pending_stats()
        index = self.index_mapper.add_new_sequence(request_id)
        for i in range(self.max_beam_width):
            for pool_idx in range(self.num_pools):
                buffer: torch.Tensor = self.host_kv_cache_block_offsets[
                    pool_idx, index * self.max_beam_width + i, 0
                ]
                kv_cache.set_base_page_index_buf(i, pool_idx, memoryview(buffer.numpy()))
        return kv_cache

    def probe_prefix_match_length(self, input_tokens, lora_task_id=None, cache_salt=None):
        """Probe the v2 KV cache radix tree for prefix match length.

        Returns the number of prefix tokens already cached on this rank,
        without acquiring page ownership. Mirrors the v1 ``KVCacheManager``
        adapter so ``KVCacheAwareADPRouter`` works on both KV cache backends.

        ``cache_salt`` (and ``lora_task_id``) must match the values used by
        ``_create_kv_cache``; the salt is derived from the ``cache_salt``
        string the same way, so the probe queries the same reuse namespace.
        Otherwise the router would see an incorrect match length.
        """
        if not self.enable_block_reuse:
            return 0
        if not input_tokens:
            return 0
        salt_int = self._derive_reuse_salt(cache_salt)
        return self.impl.probe_reuse(
            ReuseScope(lora_id=lora_task_id, salt=salt_int),
            input_tokens,
        )

    def prefetch_for_context_tokens(self, requests: list) -> bool:
        """Prefetch radix-tree blocks from disk→host for upcoming context requests.

        Returns True if all prefetches succeeded, False if any failed.
        """
        if not self.enable_block_reuse:
            return False
        # Prefetch via a transient KV cache that holds the reuse-matched blocks,
        # prefetches disk->host, then closes. Holding blocks costs no GPU space
        # (never resumed) and close() needs no stream sync. The transient cache
        # is NOT registered in kv_cache_map / IndexMapper.
        success = True
        for req in requests:
            all_tokens = req.get_tokens(DEFAULT_BEAM_INDEX)
            tokens = self._augment_tokens_for_block_reuse(all_tokens, req, end=len(all_tokens) - 1)
            # Use the same salt derivation as _create_kv_cache so the transient
            # cache hits the same radix-tree blocks.
            salt_int = self._derive_reuse_salt(req.cache_salt)
            kv_cache = self.impl.create_kv_cache(
                ReuseScope(lora_id=req.lora_task_id, salt=salt_int), tokens
            )
            # Prefetch to the first tier below GPU (host if present, otherwise
            # disk). prefetch() is a best-effort hint either way.
            if not kv_cache.prefetch(CACHE_LEVEL1):
                logger.warning("prefetch failed for request %s", req.py_request_id)
                success = False
            kv_cache.close()
        return success

    def reset_reuse_state(self):
        self.impl.clear_reusable_blocks()
