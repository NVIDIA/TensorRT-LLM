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

"""TriAttention KV-cache compression: periodic physical KV eviction.

Every ``beta`` confirmed generation tokens TriAttention scores each cached token with a
trigonometric importance score (computed from offline-calibrated statistics of
the model's pre-RoPE query vectors) and physically deletes the tokens below the
top-B keep set. There is no context-phase work and no per-step attention mask:
the eviction runs in the compression manager's final
``on_generation_step_end`` hook.

TriAttention is a :class:`BaseKVCacheCompressionManager` and nothing more -- it
has no attention backend of its own; decode runs the model's standard dense
kernel over the compacted cache. TriAttention derives each request's effective
confirmed physical length after V2's native update/rewind and publishes the
cumulative evicted count on ``LlmRequest.py_num_compressed_tokens``; attention
metadata reads it back through the KV cache manager on the next step.
Physical reclaim uses V2's existing resize path directly after compaction. An
already-enqueued speculative suffix is excluded from scoring and appended
unchanged to the retained prefix by the same per-layer compact operation.
With one-model speculative decoding, the separate draft KV cache is compacted
in the same round with the target's kept token set (union mode only), so
target and draft always share one physical KV length.

Structure: ``init_eviction_buffers`` is the ONE one-time constructor for the
whole eviction stack -- it validates the geometry, allocates every buffer,
compiles the mode-needed CuTe entries eagerly, and builds the C++ compaction
launch data (``init_compaction_buffers``). The result is a plain namespace of
tensors, events, and ints. ``stage_eviction_cohort`` and
``run_eviction_round`` are the per-round flow: straight-line module functions
that feed the kernels directly.

KV layout: the decode kernel stores keys in HND layout
``[num_pages, kv_factor, num_kv_heads, tokens_per_block, head_dim]``. The Python
gather / score / compact code MUST read ``get_buffers`` with ``kv_layout="HND"``;
reading the default NHD silently swaps the token and head axes and scrambles the
cache.

Position handling: kept keys retain their original RoPE rotation (no re-RoPE on
compaction). The model engine keeps the decode query at its true absolute
position while the attention metadata uses the compacted physical length, so a
query against a kept key at its original rotation still yields the correct
relative distance.

Calibration is NOT computed here: the user calibrates with the official tool
(github.com/WeianMao/triattention) and passes that .pt via ``calibration_path``;
the manager converts it to our runtime schema at load (see _resolve_calibration).
The scoring math follows the same upstream reference (``methods/pruning_utils.py``).
"""

from types import SimpleNamespace
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

import torch

from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2, Role
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState, get_draft_token_length
from tensorrt_llm._torch.pyexecutor.resource_manager import BaseKVCacheCompressionManager
from tensorrt_llm._utils import nvtx_range, nvtx_range_debug, prefer_pinned
from tensorrt_llm.bindings.internal.batch_manager.kv_cache_manager_v2_utils import (
    copy_batch_block_offsets_to_device,
)
from tensorrt_llm.logger import logger
from tensorrt_llm.runtime.kv_cache_manager_v2 import AttentionLayerConfig

from ..compaction import init_compaction_buffers
from .triattention_kernels import (
    _settle_ties_and_pack_compaction_sources_kernel,
    build_mean_phase_table,
    gather_mean_phases,
    grow_mean_phase_table,
    prepare_per_head_scores,
)

if TYPE_CHECKING:
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
    from tensorrt_llm._torch.pyexecutor.scheduler.scheduler import ScheduledRequests


# Required keys for the calibration ``.pt`` consumed by TriAttention.
_REQUIRED_CALIBRATION_KEYS = frozenset({"E_q", "E_q_norm", "omega", "freq_scale_sq"})

# Upper bound of the geometric integration offset ladder [1, 2, 4, ...]; no
# caller ever tuned it, so it is a constant rather than a constructor knob.
_OFFSET_MAX_LENGTH = 65536


# Stream-affinity contract: the staged buffers and compiled launches are bound
def _page_table_slot_layout(
    page_representatives: List[int],
    page_table_keys: List[object],
) -> Tuple[Dict[int, int], int]:
    """Map representative layers to page-table snapshot slots."""
    use_pool_ids = all(
        isinstance(key, tuple)
        and len(key) == 2
        and key[0] == "pool"
        and isinstance(key[1], int)
        and key[1] >= 0
        for key in page_table_keys
    )
    unique_slots = []
    key_to_slot = {}
    representative_slots = {}
    for representative, key in zip(page_representatives, page_table_keys):
        slot = key_to_slot.get(key)
        if slot is None:
            slot = int(key[1]) if use_pool_ids else len(key_to_slot)
            key_to_slot[key] = slot
            unique_slots.append(slot)
        representative_slots[representative] = slot
    slot_count = max(unique_slots, default=-1) + 1
    return representative_slots, slot_count


def _protected_tail_capacity(manager: KVCacheManagerV2, what: str) -> int:
    """The V2 tail (extra KV + draft reserve + 1) moved with every compaction."""
    capacity = int(manager.num_extra_kv_tokens) + int(manager._kv_reserve_draft_tokens) + 1
    if capacity <= 0:
        raise RuntimeError(f"{what}KVCacheManagerV2 exposes an invalid protected-tail capacity")
    return capacity


def _allocate_page_table_plane(
    layer_pools: List[torch.Tensor],
    page_representatives: List[int],
    page_table_keys: List[object],
    num_page_table_slots: Optional[int],
    token_capacity: int,
    max_requests: int,
    device: torch.device,
    what: str,
) -> Tuple[Dict[int, int], int, torch.Tensor, torch.Tensor]:
    """Allocate one staged block-offset plane (host pinned + device)."""
    representative_slots, minimum_slots = _page_table_slot_layout(
        page_representatives, page_table_keys
    )
    if num_page_table_slots is None:
        num_page_table_slots = minimum_slots
    tokens_per_block = int(layer_pools[page_representatives[0]].shape[3])
    page_count = (token_capacity + tokens_per_block - 1) // tokens_per_block
    copy_block_count = (page_count + 3) // 4 * 4
    plane_shape = (num_page_table_slots, max_requests, 2, copy_block_count)
    host = torch.empty(plane_shape, dtype=torch.int32, device="cpu", pin_memory=prefer_pinned())
    dev = torch.empty(plane_shape, dtype=torch.int32, device=device)
    return representative_slots, copy_block_count, host, dev


def init_eviction_buffers(
    *,
    eviction_mode: str,
    layer_pools: List[torch.Tensor],
    dense_groups: List[List[int]],
    dense_layers: List[int],
    swa_layers: Sequence[int] = (),
    swa_window: Optional[int] = None,
    layer_group_representative: Optional[Dict[int, int]] = None,
    layer_pool_keys: Optional[List[object]] = None,
    page_representatives: List[int],
    max_requests: int,
    seq_len: int,
    num_q_heads: int,
    num_freqs: int,
    keep_count: int,
    q_real: torch.Tensor,
    q_imag: torch.Tensor,
    mlr_coef: torch.Tensor,
    freq_scale_sq: torch.Tensor,
    offsets: torch.Tensor,
    omega: torch.Tensor,
    phase: Optional[Dict[str, object]] = None,
    page_table_keys: Optional[List[object]] = None,
    num_page_table_slots: Optional[int] = None,
    decode_width: Optional[int] = None,
    page_table_token_capacity: Optional[int] = None,
    protected_tail_capacity: int = 0,
    draft_layer_pools: Optional[List[torch.Tensor]] = None,
    draft_layers: Optional[List[int]] = None,
    draft_layer_group_representative: Optional[Dict[int, int]] = None,
    draft_layer_pool_keys: Optional[List[object]] = None,
    draft_page_representatives: Optional[List[int]] = None,
    draft_page_table_keys: Optional[List[object]] = None,
    draft_num_page_table_slots: Optional[int] = None,
    draft_page_table_token_capacity: Optional[int] = None,
    draft_protected_tail_capacity: int = 0,
) -> SimpleNamespace:
    """Build the ONE plain namespace of buffers for the whole eviction stack.

    The single one-time constructor: buffer staging, eager CuTe compilation
    for exactly the entries the eviction mode launches, selection buffers,
    and the C++ compaction launch data. Runs outside CUDA graph capture
    (compilation allocates and synchronizes); the CuTe runner validates its
    own geometry contract and raises loudly -- there is no fallback path.
    The returned namespace holds tensors, events, compiled runners, and
    ints; all flow logic lives in ``stage_eviction_cohort`` and
    ``run_eviction_round``. The buffers retain references to every scored
    layer pool: the compiled kernels encode immutable TMA descriptors from
    their raw device addresses, so the pools must stay alive and stay put.
    """
    from .triattention_cute_score_fused import TriAttentionCuteScoreRunner

    device = layer_pools[page_representatives[0]].device
    max_requests = int(max_requests)
    seq_len = int(seq_len)
    if page_table_token_capacity is None:
        page_table_token_capacity = seq_len
    page_table_token_capacity = int(page_table_token_capacity)
    # Decode-width capacity of the score buffers; per-request prompt lengths
    # are staged runtime metadata.
    if decode_width is None:
        decode_width = seq_len
    decode_width = int(decode_width)
    keep_count = int(keep_count)

    q_real = q_real.to(device=device, dtype=torch.float32).contiguous()
    q_imag = q_imag.to(device=device, dtype=torch.float32).contiguous()
    mlr_coef = mlr_coef.to(device=device, dtype=torch.float32).contiguous()
    freq_scale_sq = freq_scale_sq.to(device=device, dtype=torch.float32).contiguous()
    offsets = offsets.to(device=device, dtype=torch.float32).contiguous()
    omega = omega.to(device=device, dtype=torch.float32).contiguous()
    if page_table_keys is None:
        page_table_keys = list(range(len(page_representatives)))

    bufs = SimpleNamespace()
    bufs.eviction_mode = eviction_mode
    bufs.device = device
    bufs.max_requests = max_requests
    bufs.bucket_seq_len = seq_len
    bufs.decode_width = decode_width
    bufs.keep_count = keep_count
    bufs.page_table_token_capacity = page_table_token_capacity

    # ---- staged page-table planes (target, plus the co-compressed draft) ---
    (
        bufs.representative_slots,
        bufs.copy_block_count,
        bufs._bulk_offsets_src,
        bufs.block_offsets_device,
    ) = _allocate_page_table_plane(
        layer_pools,
        page_representatives,
        page_table_keys,
        num_page_table_slots,
        page_table_token_capacity,
        max_requests,
        device,
        "",
    )
    # The draft is never scored: these offsets feed only the draft compacts.
    bufs.draft_block_offsets_device = None
    bufs._draft_bulk_offsets_src = None
    bufs.draft_representative_slots = {}
    bufs.draft_copy_block_count = 0
    if draft_layer_pools is not None:
        (
            bufs.draft_representative_slots,
            bufs.draft_copy_block_count,
            bufs._draft_bulk_offsets_src,
            bufs.draft_block_offsets_device,
        ) = _allocate_page_table_plane(
            draft_layer_pools,
            draft_page_representatives,
            draft_page_table_keys,
            draft_num_page_table_slots,
            int(draft_page_table_token_capacity),
            max_requests,
            device,
            "draft ",
        )

    # ---- per-round metadata table: ONE host-to-device copy per round -------
    # Three metadata rows (logical position, valid length, prompt length) plus
    # one move-offsets row per compacted cache family; offsets rows have
    # request_capacity + 1 entries, hence the extra column.
    bufs.request_metadata_host = torch.empty(
        (6, max_requests + 1), dtype=torch.int32, device="cpu", pin_memory=prefer_pinned()
    )
    bufs._bulk_copy_idx_src = torch.arange(
        max_requests, dtype=torch.int32, device="cpu", pin_memory=prefer_pinned()
    )
    # Zero-filled so an unstaged cohort gathers the phase table's row 0
    # instead of indexing it with uninitialized round starts.
    bufs.request_metadata_device = torch.zeros(
        (6, max_requests + 1), dtype=torch.int32, device=device
    )
    bufs.round_starts_device = bufs.request_metadata_device[0, :max_requests]
    bufs.valid_seq_lens_device = bufs.request_metadata_device[1, :max_requests]
    # Per-request pinned prompt lengths: the score kernel starts each
    # request's decode window here, so one bucket may mix prompt lengths.
    bufs.token_starts_device = bufs.request_metadata_device[2, :max_requests]
    bufs.dense_move_offsets = bufs.request_metadata_device[3]
    bufs.swa_move_offsets = bufs.request_metadata_device[4]
    bufs.draft_move_offsets = bufs.request_metadata_device[5]
    bufs.mean_cos = torch.empty((max_requests, num_freqs), dtype=torch.float32, device=device)
    bufs.mean_sin = torch.empty_like(bufs.mean_cos)
    # The phase table depends only on the shared calibration; the manager
    # shares one dict with every buffer namespace (tests may pass None for a
    # private table).
    if phase is None:
        phase = build_mean_phase_table(offsets, omega, initial_rows=seq_len)
    bufs.phase = phase

    # ---- score state: ONE fused group across ALL dense layers --------------
    # Segments carry their own page-table slot, so distinct per-layer
    # storages/block tables share a single launch. Pool geometry, dtype, and
    # layout are validated by the CuTe runner itself below.
    p0 = layer_pools[dense_layers[0]]
    _, kv_factor, num_kv_heads, tokens_per_block, head_dim = p0.shape
    bufs.num_layers = len(dense_layers)
    bufs.num_q_heads = int(num_q_heads)
    bufs.num_kv_heads = int(num_kv_heads)
    bufs.num_freqs = int(num_freqs)
    bufs.tokens_per_block = int(tokens_per_block)
    # Calibration tables span every model layer; segments index them by
    # ABSOLUTE layer id ON DEVICE where they cannot be range-checked, so
    # validate the extent once here, loudly.
    num_calibrated_layers = q_real.numel() // (bufs.num_q_heads * bufs.num_freqs)
    if min(dense_layers) < 0 or max(dense_layers) >= num_calibrated_layers:
        raise ValueError("scored layer index exceeds the calibrated layer extent")
    _rep_of = {layer: layers[0] for layers in dense_groups for layer in layers}
    page_table_slots = [bufs.representative_slots[_rep_of[layer]] for layer in dense_layers]
    bufs.seg_req = torch.arange(max_requests, dtype=torch.int32, device=device).repeat_interleave(
        bufs.num_layers
    )
    seg_layer = torch.tensor(list(dense_layers), dtype=torch.int32, device=device).repeat(
        max_requests
    )
    block_offsets = bufs.block_offsets_device
    slots_t = torch.tensor(page_table_slots, dtype=torch.int64, device=device)
    req_idx = torch.arange(max_requests, dtype=torch.int64, device=device).repeat_interleave(
        bufs.num_layers
    )
    slot_idx = slots_t.repeat(max_requests)
    seg_page_off = slot_idx * block_offsets.stride(0) + req_idx * block_offsets.stride(1)

    max_segments = max_requests * bufs.num_layers
    # Head axis pads to the MMA tile N=8; one padded scratch plane must stay
    # 32-bit indexable (wraparound = silent wild reads, not a clean error).
    if (8 - 1) * max_segments * seq_len >= 2**31:
        raise ValueError(
            f"score bucket overflows the 32-bit scratch plane: {(8 - 1) * max_segments * seq_len}"
        )
    # The kernel scores each request's window into a head-major scratch;
    # all buffers below are persistent because the compiled kernels capture
    # their device pointers.
    bufs.cute_scratch = torch.empty(
        bufs.num_kv_heads * 8 * max_segments * seq_len, dtype=torch.float32, device=device
    )
    seg_out_offset = (torch.arange(max_segments, dtype=torch.int64, device=device) * seq_len).to(
        torch.int32
    )
    bufs.gather_columns = torch.arange(decode_width, dtype=torch.int64, device=device).view(
        1, 1, 1, 1, -1
    )
    # Compile the SM100 CuTe entries this mode launches -- HERE at buffer
    # construction, outside any CUDA graph capture (compilation allocates and
    # synchronizes). Union rounds run the fused score+stats+union pipeline;
    # the per-head modes run the score-only entry. There is deliberately no
    # other score path and no fallback.
    union = eviction_mode == "union"
    bufs.union_rows = None
    if union:
        # Union output rows are sized by the whole bucket (the widest
        # possible window); consumers mask by the per-request widths.
        bufs.union_rows = torch.empty((max_requests, seq_len), dtype=torch.float32, device=device)
    try:
        bufs.runner = TriAttentionCuteScoreRunner(
            layer_pools=list(layer_pools),
            layer_indices=[int(layer) for layer in dense_layers],
            max_requests=max_requests,
            num_layers=bufs.num_layers,
            seq_len=seq_len,
            num_q_heads=bufs.num_q_heads,
            num_kv_heads=bufs.num_kv_heads,
            num_freqs=bufs.num_freqs,
            tokens_per_block=bufs.tokens_per_block,
            page_ids=block_offsets.view(-1),
            seg_page_off=seg_page_off,
            seg_req_id=bufs.seg_req,
            seg_layer_id=seg_layer,
            # The kernels read per-request lengths and window starts straight
            # from the staged metadata rows (pointer capture; the rows are
            # int32 views into the persistent metadata table, so the round
            # needs no per-round re-marshaling copies for them).
            valid_seq_lens=bufs.valid_seq_lens_device,
            seg_out_offset=seg_out_offset,
            token_starts=bufs.token_starts_device,
            q_real=q_real.view(-1),
            q_imag=q_imag.view(-1),
            mlr_coef=mlr_coef.view(-1),
            mean_cos=bufs.mean_cos,
            mean_sin=bufs.mean_sin,
            freq_scale_sq=freq_scale_sq,
            output=bufs.cute_scratch,
            enable_partial_stats=union,
        )
    except (ImportError, RuntimeError, TypeError, ValueError, AssertionError) as error:
        raise RuntimeError(
            "TriAttention CuTe score setup failed and no other score path exists"
        ) from error
    logger.info(
        f"TriAttention CuTe score enabled: {bufs.num_q_heads}q/{bufs.num_kv_heads}kv heads, "
        f"{bufs.num_freqs} freqs, {bufs.tokens_per_block}-token pages"
    )

    # ---- selection buffers --------------------------------------------------
    # Per-request valid decode widths, refreshed each round from the staged
    # lengths; prompt offsets alias the staged per-request prompt lengths so
    # the values are written once per round.
    bufs.valid_widths = torch.full((max_requests,), decode_width, dtype=torch.int32, device=device)
    bufs.prompt_offsets = bufs.token_starts_device
    if union:
        bufs.selection_rows_per_request = 1
        bufs.row_prompt_offsets = bufs.prompt_offsets
        # The fused pipeline writes normalized per-request union rows straight
        # into ``combined``; only the top-k settle-and-pack stage remains.
        bufs.combined = torch.empty(
            (max_requests, decode_width), dtype=torch.float32, device=device
        )
        bufs.final_indices = torch.empty(
            (max_requests, keep_count), dtype=torch.int32, device=device
        )
        # Kept decode ordinals only: rows are prompt-length independent, so
        # one buffer namespace serves cohorts with mixed prompt lengths.
        bufs.keep = torch.empty((max_requests, keep_count), dtype=torch.int32, device=device)
        # Row-major views consumed by the top-k settle launch.
        bufs.selection_scores_rows = bufs.combined
        bufs.selection_row_lengths = bufs.valid_widths
        bufs.provisional_rows = bufs.final_indices
        bufs.keep_rows = bufs.keep
        # Padded rows carry zero valid width; their provisional TopK entries
        # must still be in-range ordinals for the finalizer's score gather.
        bufs.final_indices.zero_()
        bufs.score_output = None
    else:
        selection_rows = (
            bufs.num_kv_heads
            if eviction_mode == "per_head"
            else bufs.num_layers * bufs.num_kv_heads
        )
        bufs.selection_rows_per_request = selection_rows
        bufs.row_prompt_offsets = torch.zeros(
            (max_requests * selection_rows,), dtype=torch.int32, device=device
        )
        # Decode-only per-head scores gathered from the CuTe scratch, the
        # ``[request, layer, head, token]`` layout the reduce kernels read.
        bufs.score_output = torch.empty(
            max_requests,
            bufs.num_layers,
            bufs.num_q_heads,
            decode_width,
            dtype=torch.float32,
            device=device,
        )
        score_shape = (max_requests, bufs.num_layers, bufs.num_q_heads, 1)
        bufs.row_mean = torch.empty(score_shape, dtype=torch.float32, device=device)
        bufs.row_std = torch.empty_like(bufs.row_mean)
        bufs.selection_scores = torch.empty(
            (max_requests, selection_rows, decode_width), dtype=torch.float32, device=device
        )
        bufs.row_seq_lens = torch.full(
            (max_requests, selection_rows), decode_width, dtype=torch.int32, device=device
        )
        selection_shape = (max_requests, selection_rows, keep_count)
        bufs.top_indices_i32 = torch.empty(selection_shape, dtype=torch.int32, device=device)
        bufs.keep = torch.empty(selection_shape, dtype=torch.int32, device=device)
        bufs.selection_scores_rows = bufs.selection_scores.view(
            max_requests * selection_rows, decode_width
        )
        bufs.selection_row_lengths = bufs.row_seq_lens.view(-1)
        bufs.provisional_rows = bufs.top_indices_i32.view(-1, keep_count)
        bufs.keep_rows = bufs.keep.view(-1, keep_count)
        bufs.top_indices_i32.zero_()

    # ---- compaction launch data + settle/pack fusion ------------------------
    # One settle program per (request, selection row).
    bufs.settle_grid = (max_requests, bufs.selection_rows_per_request)
    draft_kwargs = {}
    if draft_layers:
        draft_kwargs = dict(
            draft_layer_pools=draft_layer_pools,
            draft_layers=list(draft_layers),
            draft_layer_group_representative=draft_layer_group_representative,
            draft_layer_pool_keys=draft_layer_pool_keys,
            draft_protected_tail_capacity=int(draft_protected_tail_capacity),
            draft_kv_block_offsets=bufs.draft_block_offsets_device,
            draft_page_table_slots=bufs.draft_representative_slots,
            draft_move_offsets=bufs.draft_move_offsets,
        )
    compaction = init_compaction_buffers(
        eviction_mode=eviction_mode,
        layer_pools=layer_pools,
        dense_layers=list(dense_layers),
        swa_layers=list(swa_layers),
        layer_group_representative=layer_group_representative,
        kept_token_ordinals=bufs.keep,
        valid_sequence_lengths=bufs.valid_seq_lens_device,
        kv_block_offsets=bufs.block_offsets_device,
        page_table_slots=bufs.representative_slots,
        request_count=max_requests,
        prompt_offsets=bufs.token_starts_device,
        decode_keep_count=keep_count,
        swa_window=swa_window,
        layer_pool_keys=list(layer_pool_keys),
        protected_tail_capacity=int(protected_tail_capacity),
        # Tails vary per round (in-flight growth), so the per-family move
        # offsets ride the staged metadata rows each round.
        dense_move_offsets=bufs.dense_move_offsets,
        swa_move_offsets=bufs.swa_move_offsets,
        **draft_kwargs,
    )
    # Flatten the launch data to plain fields: ONE fused launch settles the
    # kept ordinals and packs the dense/SWA move sources
    # (``settle_top_tokens``); ``run_eviction_round`` fires the draft pack
    # and every family's C++ moves directly.
    bufs.compaction_families = compaction["families"]
    bufs.settle_pack_tensors = compaction["settle_pack_tensors"]
    bufs.settle_pack_shape = compaction["settle_pack_shape"]
    bufs.draft_pack = compaction["draft_pack"]
    bufs.swa_destination_bases = compaction["swa_destination_bases"]
    bufs.swa_rebase_delta = compaction["swa_rebase_delta"]

    # ---- round-ordering events ----------------------------------------------
    bufs.copy_done = torch.cuda.Event()
    # First record publishes constructor allocations to the V2 copy stream;
    # later records protect pinned metadata before the next cohort reuses it.
    bufs.copy_done.record(torch.cuda.current_stream(device))
    bufs.bulk_copy_done = torch.cuda.Event()
    bufs.bulk_consume_done = torch.cuda.Event()
    bufs.copy_pending = False
    bufs.page_tables_active = False
    return bufs


def _stage_block_offsets(
    bufs: SimpleNamespace,
    manager: KVCacheManagerV2,
    request_ids: List[int],
    current_stream: torch.cuda.Stream,
    source: torch.Tensor,
    destination: torch.Tensor,
    copy_block_count: int,
) -> None:
    """Copy one request group's V2 block offsets before live compaction.

    Uses the V2 block-offset kernel with an immutable pinned snapshot of the
    selected host-table rows: this enqueues asynchronous host-memory reads,
    and TriAttention later resizes the same cache, which mutates the
    manager's table in place. The IndexMapper synchronously resolves request
    slots and gathers only their beam-0 K block offsets, decoupling both live
    inputs before the native asynchronous copy consumes the snapshot with
    identity indices. ``dst[pool, r, 0(K), :]`` holds ``base_page *
    index_scales``; score and compact decode that K plane inline.
    """
    if bufs.copy_pending and not bufs.copy_done.query():
        bufs.copy_done.synchronize()
    # The native device copy reads only K and derives V with kv_offset.
    manager.index_mapper.gather_k_block_offsets(
        manager.host_kv_cache_block_offsets,
        source,
        request_ids,
        copy_block_count,
    )
    manager._stream.wait_event(bufs.copy_done)
    copy_batch_block_offsets_to_device(
        source,
        destination,
        bufs._bulk_copy_idx_src[: len(request_ids)],
        manager.index_scales,
        manager.kv_offset,
        manager._stream.cuda_stream,
    )
    bufs.bulk_copy_done.record(manager._stream)
    current_stream.wait_event(bufs.bulk_copy_done)


def stage_eviction_cohort(
    bufs: SimpleNamespace,
    manager: KVCacheManagerV2,
    request_ids: List[int],
    round_starts: List[int],
    token_starts: List[int],
    seq_lens: Optional[List[int]] = None,
    draft_manager: Optional[KVCacheManagerV2] = None,
    dense_move_offsets: Optional[List[int]] = None,
    swa_move_offsets: Optional[List[int]] = None,
    draft_move_offsets: Optional[List[int]] = None,
) -> None:
    """Copy one eviction cohort into the reusable device buffers.

    ``token_starts`` carries each request's pinned prompt length; the score
    kernel starts that request's decode window there, so the cohort may mix
    prompt lengths.
    """
    request_count = len(request_ids)
    stream = torch.cuda.current_stream(bufs.device)
    # Reuse guard: staging over a cohort whose pages are still being read
    # would silently corrupt the in-flight compaction.
    if bufs.page_tables_active:
        raise RuntimeError("previous page-table cohort is still active")
    if seq_lens is None:
        seq_lens = [bufs.bucket_seq_len] * request_count
    request_metadata = torch.as_tensor((round_starts, seq_lens, token_starts), dtype=torch.int32)
    # Grow the phase table while this cohort's round starts are still host
    # integers: a stale-capacity gather is an out-of-bounds index_select on
    # the device.
    grow_mean_phase_table(bufs.phase, int(max(round_starts)) + 1)
    _stage_block_offsets(
        bufs,
        manager,
        request_ids,
        stream,
        bufs._bulk_offsets_src,
        bufs.block_offsets_device,
        bufs.copy_block_count,
    )
    if draft_manager is not None:
        _stage_block_offsets(
            bufs,
            draft_manager,
            request_ids,
            stream,
            bufs._draft_bulk_offsets_src,
            bufs.draft_block_offsets_device,
            bufs.draft_copy_block_count,
        )
    bufs.request_metadata_host[:3, :request_count].copy_(request_metadata)
    # Rows past this cohort are padding: zero lengths keep the score kernel
    # and selection inert for them.
    bufs.request_metadata_host[:3, request_count:].zero_()
    # This round's per-family move offsets ride the same table, so the single
    # device copy below carries them too.
    for row, family_offsets in (
        (3, dense_move_offsets),
        (4, swa_move_offsets),
        (5, draft_move_offsets),
    ):
        if family_offsets is not None:
            bufs.request_metadata_host[row, : len(family_offsets)].copy_(
                torch.as_tensor(family_offsets, dtype=torch.int32)
            )
    try:
        # Copy the fixed backing once. Only the first ``request_count``
        # columns are consumed by this cohort.
        bufs.request_metadata_device.copy_(bufs.request_metadata_host, non_blocking=True)
    finally:
        # Guard the pinned metadata until its asynchronous copies complete.
        # Page-table device-buffer reuse is guarded separately after compact.
        bufs.copy_done.record(stream)
        bufs.copy_pending = True
    bufs.page_tables_active = True
    # The staged per-request prompt lengths are shared with the selection;
    # per-head modes re-expand them into their row-major view here.
    if bufs.row_prompt_offsets is not bufs.prompt_offsets:
        bufs.row_prompt_offsets.view(bufs.max_requests, bufs.selection_rows_per_request).copy_(
            bufs.prompt_offsets.unsqueeze(1).expand(-1, bufs.selection_rows_per_request)
        )


def mark_page_tables_consumed(bufs: SimpleNamespace, *manager_streams: torch.cuda.Stream) -> None:
    """Order V2 page-table reuse and resize after this cohort's compact.

    Every passed manager stream (target, and the draft when co-compressed)
    waits on one event recorded after the compact launches, so neither cache
    can free or reallocate pages this cohort is still reading.
    """
    if not bufs.page_tables_active:
        raise RuntimeError("TriAttention page tables were not staged")
    bufs.bulk_consume_done.record(torch.cuda.current_stream(bufs.device))
    for manager_stream in manager_streams:
        manager_stream.wait_event(bufs.bulk_consume_done)
    bufs.page_tables_active = False


def settle_top_tokens(bufs: SimpleNamespace) -> None:
    """Pick the top-k with the CuTE selector, then settle its output.

    The CuTE top-k is fast but breaks score ties arbitrarily and emits
    indices in arbitrary order; the settle kernel recomputes the threshold
    membership with lowest-index-wins ties, rebases each row by its prompt
    offset, and writes sorted ordinals. The same launch packs each request's
    dense/SWA compaction move sources from the ordinals it just settled
    (buffers built without compaction compile the pack half away).
    """
    # The trailing 1 is next_n: decode scores one query token per request.
    torch.ops.trtllm.cute_dsl_indexer_topk_decode(
        bufs.selection_scores_rows,
        bufs.selection_row_lengths,
        bufs.provisional_rows,
        bufs.keep_count,
        1,
    )
    _settle_ties_and_pack_compaction_sources_kernel[bufs.settle_grid](
        bufs.selection_scores_rows,
        bufs.selection_row_lengths,
        bufs.row_prompt_offsets,
        bufs.provisional_rows,
        bufs.keep_rows,
        *bufs.settle_pack_tensors,
        WIDTH=bufs.decode_width,
        KEEP_COUNT=bufs.keep_count,
        SELECTION_ROWS=bufs.selection_rows_per_request,
        **bufs.settle_pack_shape,
        HAS_SETTLE=True,
        BLOCK=256,
        num_warps=4,
    )


def run_eviction_round(bufs: SimpleNamespace, normalize_scores: bool) -> None:
    """One staged eviction round, kernels fired directly in sequence.

    Union: phase gather (also derives valid widths), fused score+stats+union
    (two CuTe launches), top-k, settle-and-pack, C++ compacts. Per-head
    modes: phase gather, score-only CuTe launch, decode-window gather,
    stats+reduce kernels, top-k, settle-and-pack, C++ compacts. A
    co-compressed draft adds its own pack launch before its C++ moves.
    Every launch covers the full request capacity; padded rows past the
    staged cohort carry zero lengths and stay inert.
    """
    request_count = bufs.max_requests
    union = bufs.eviction_mode == "union"
    with nvtx_range("triattention.score", color="blue"):
        # mean_cos/mean_sin feed the compiled score launches, which captured
        # their device pointers: refresh them in place from this round's
        # staged round starts. The same launch derives the per-request valid
        # decode widths; the compiled kernels read valid lengths and window
        # starts straight from the staged metadata rows (pointer capture).
        gather_mean_phases(
            bufs.phase,
            bufs.round_starts_device,
            bufs.mean_cos,
            bufs.mean_sin,
            bufs.valid_seq_lens_device,
            bufs.token_starts_device,
            bufs.valid_widths,
            request_count,
        )
        if union:
            bufs.runner.launch_union_fusion(
                request_count, bufs.mean_cos, bufs.mean_sin, bufs.union_rows[:request_count]
            )
            columns = min(bufs.union_rows.shape[1], bufs.combined.shape[1])
            bufs.combined[:request_count, :columns].copy_(bufs.union_rows[:request_count, :columns])
        else:
            bufs.runner.launch(request_count, bufs.mean_cos, bufs.mean_sin)
            # The kernel wrote each request's window scores (from its pinned
            # prompt length) into the head-major scratch, padded to the MMA
            # tile N=8 per KV head. Gather each request's decode window into
            # the [request, layer, head, token] layout the reduce kernels
            # read; columns past a request's valid width carry unscored
            # scratch data masked by ``valid_widths``.
            group_size = bufs.num_q_heads // bufs.num_kv_heads
            num_segments = request_count * bufs.num_layers
            source = (
                bufs.cute_scratch[: bufs.num_kv_heads * 8 * num_segments * bufs.bucket_seq_len]
                .view(bufs.num_kv_heads, 8, request_count, bufs.num_layers, bufs.bucket_seq_len)[
                    :, :group_size
                ]
                .permute(2, 3, 0, 1, 4)
            )
            columns = (
                bufs.token_starts_device[:request_count].to(torch.int64).view(-1, 1, 1, 1, 1)
                + bufs.gather_columns
            )
            columns = columns.clamp_(max=bufs.bucket_seq_len - 1).expand(
                request_count, bufs.num_layers, bufs.num_kv_heads, group_size, bufs.decode_width
            )
            torch.gather(
                source,
                4,
                columns,
                out=bufs.score_output[:request_count].view(
                    request_count, bufs.num_layers, bufs.num_kv_heads, group_size, bufs.decode_width
                ),
            )
    with nvtx_range("triattention.select", color="yellow"):
        if not union:
            prepare_per_head_scores(
                bufs.score_output[:request_count],
                bufs.valid_widths,
                bufs.row_mean,
                bufs.row_std,
                bufs.selection_scores,
                bufs.row_seq_lens,
                request_count,
                num_kv_heads=bufs.num_kv_heads,
                per_layer=bufs.eviction_mode == "per_layer_perhead",
                normalize_scores=normalize_scores,
            )
        settle_top_tokens(bufs)
    with nvtx_range("triattention.compact", color="purple"):
        if bufs.swa_destination_bases is not None:
            # The prompt offsets may have been re-staged since construction;
            # rebase the SWA landing positions for this round.
            torch.add(bufs.prompt_offsets, bufs.swa_rebase_delta, out=bufs.swa_destination_bases)
        for family in bufs.compaction_families:
            if family["name"] == "draft":
                # One more pack launch broadcasts the target keep set over
                # the draft KV heads and appends the draft's own tail
                # ordinals. HAS_SETTLE=False compiles the settle half away
                # (the ordinals arrive pre-settled), so any well-formed
                # tensor stands in for the settle-side pointer arguments.
                _settle_ties_and_pack_compaction_sources_kernel[(bufs.max_requests, 1)](
                    bufs.keep,
                    bufs.valid_seq_lens_device,
                    bufs.draft_pack["offsets"],
                    bufs.keep,
                    bufs.keep,
                    bufs.valid_seq_lens_device,
                    bufs.draft_pack["offsets"],
                    bufs.draft_pack["indices"],
                    bufs.draft_pack["offsets"],
                    bufs.draft_pack["indices"],
                    WIDTH=bufs.keep_count,
                    KEEP_COUNT=bufs.keep_count,
                    SELECTION_ROWS=1,
                    DENSE_TOTAL=bufs.draft_pack["dense_total"],
                    SWA_TOTAL=0,
                    MOVE_CAPACITY=bufs.draft_pack["move_capacity"],
                    NUM_KV_HEADS=bufs.draft_pack["num_kv_heads"],
                    SWA_WINDOW=0,
                    UNION=True,
                    PER_LAYER=False,
                    HAS_SWA=False,
                    HAS_SETTLE=False,
                    BLOCK=256,
                    num_warps=4,
                )
            for group in family["groups"]:
                torch.ops.trtllm.sparse_kv_cache_compact_layers(
                    group["pools"],
                    group["pool_pointers"],
                    group["page_table"],
                    family["source"],
                    family["offsets"],
                    family["destination_bases"],
                    group["source_layer_indices"],
                )


class TriAttention(BaseKVCacheCompressionManager):
    """Periodic physical KV eviction driven by trigonometric importance scoring.

    Overrides ``on_generation_step_end``: every ``beta`` confirmed generation tokens it
    reads the cached keys through the ``KVCacheManagerV2``, scores each token
    with offline-calibrated stats, and physically evicts the tokens below the
    keep set. Full-attention layers are scored; kernel-masked SWA layers preserve
    their latest window in the same compacted prefix. Every layer ends with the
    same request-wide cached length.
    """

    adjusts_generation_kv_length = True

    def __init__(
        self,
        kv_cache_manager: KVCacheManagerV2,
        budget: int,
        draft_kv_cache_manager: Optional[KVCacheManagerV2] = None,
        beta: int = 128,
        model_path: Optional[str] = None,
        calibration_path: Optional[str] = None,
        eviction_mode: str = "union",
        normalize_scores: bool = True,
        pin_prefill: bool = True,
        count_prompt_tokens: bool = False,
    ):
        super().__init__(kv_cache_manager, draft_kv_cache_manager)
        self.budget = budget
        self.beta = beta
        if self.budget <= 0 or self.beta <= 0:
            raise ValueError("TriAttention budget and beta must both be positive")
        # Which token set each eviction round keeps. The user-facing meaning of
        # each mode is documented on TriAttentionKvCacheCompressionConfig
        # (llm_args); implementation notes live above the selection helpers.
        self.eviction_mode = eviction_mode
        if self.eviction_mode not in ("union", "per_head", "per_layer_perhead"):
            raise ValueError(
                f"Unknown eviction_mode {self.eviction_mode!r}; expected one of "
                "'union', 'per_head', 'per_layer_perhead'"
            )
        self.normalize_scores = bool(normalize_scores)
        if self.eviction_mode == "union" and not self.normalize_scores:
            # The fused score+stats+union CuTe pipeline is THE union path
            # and always z-normalizes.
            raise ValueError(
                "TriAttention union eviction requires normalize_scores=True: "
                "the fused union pipeline always z-normalizes score rows"
            )
        self.pin_prefill = bool(pin_prefill)
        # cpt=False (default): budget counts DECODE tokens only (pinned prompt is
        # extra). cpt=True: budget INCLUDES the pinned prompt.
        self.count_prompt_tokens = bool(count_prompt_tokens)
        if not self.pin_prefill or self.count_prompt_tokens:
            raise ValueError(
                "TriAttention physical KV reclaim requires pin_prefill=True and "
                "count_prompt_tokens=False so finalized prompt KV is preserved"
            )
        # All physical moves use the C++ V2 compaction operation.
        # No other compaction path exists.
        # Calibration is the OFFICIAL TriAttention .pt (passed via
        # calibration_path), resolved + converted on the first request
        # (on_request_init). TRT-LLM does NOT compute calibration; model_path is
        # used for RoPE tables and local layer_types/sliding_window metadata.
        self.model_path = model_path
        if self.model_path is None:
            raise ValueError(
                "TriAttention requires model_path so kernel-masked "
                "sliding-attention layers can be classified safely"
            )
        self.calibration_path = calibration_path
        self.calibration: Optional[Dict[str, torch.Tensor]] = None
        self._calibrated = False
        # Calibration-derived dims + stats, filled in on_request_init.
        self._H: Optional[int] = None
        self._F: Optional[int] = None
        self._freq_scale_sq: Optional[torch.Tensor] = None

        # Geometric integration offsets (built lazily on first eviction so the
        # device matches the cache pool).
        self._offsets: Optional[torch.Tensor] = None
        # Mean-phase table dict, shared by reference with every buffer
        # namespace so it persists across buffer rebuilds.
        self._phase: Optional[Dict[str, object]] = None

        # Request presence records successful initialization. Each value is a
        # plain dict {generation_steps, evicted_tokens, confirmed_kv_length}.
        self._request_states: Dict[int, Dict[str, object]] = {}
        # The overlap executor prepares B(n) before finalizing B(n-1). Keep the
        # exact fixed-linear generation width for that currently in-flight
        # batch as ``(batch, {request_id: growth})``; the final hook treats
        # those slots as an opaque suffix.
        self._prepared_generation_batch: Optional[Tuple[object, Dict[int, int]]] = None
        # The eviction buffers are built once at the first eviction, sized to
        # capacity bounds, and reused for the manager's lifetime.
        self._buffers: Optional[SimpleNamespace] = None
        self._buffers_fingerprint: Optional[tuple] = None
        self._local_to_global_layers_cache: Optional[List[int]] = None
        self._attention_layer_partition_cache: Optional[
            Tuple[List[int], List[int], Optional[int]]
        ] = None
        self._runtime_kv_layout_cache: Optional[Dict[str, object]] = None
        self._draft_runtime_kv_layout_cache: Optional[Dict[str, object]] = None

    def on_request_init(self, request: "LlmRequest", **kwargs) -> None:
        """Mark capacity-only decode and resolve calibration once.

        Loads the user-supplied OFFICIAL calibration .pt and converts it to our
        runtime schema (see _resolve_calibration). TRT-LLM does not calibrate.
        """
        request_id = request.py_request_id
        if request_id not in self._request_states:
            self._validate_v2_compatibility()
            self._validate_request_capacity(request)
            num_layers = self._num_layers_from_manager()
            self._attention_layer_partition(num_layers)
            self._request_states[request_id] = {
                "generation_steps": 0,
                "evicted_tokens": 0,
                "confirmed_kv_length": None,
            }
        self._ensure_calibrated()

    def _validate_request_capacity(self, request: "LlmRequest") -> None:
        """Require enough target page-table capacity to reach first eviction."""
        manager = self.kv_cache_manager
        # V2 mirrors the resolved speculative draft length (0 without spec).
        speculative_overshoot = int(manager.max_draft_len)
        first_eviction_decode_length = (
            self.budget // self.beta + 1
        ) * self.beta + speculative_overshoot
        decode_capacity = min(int(request.py_max_new_tokens), first_eviction_decode_length)
        confirmed_capacity = int(request.py_prompt_len) + decode_capacity
        protected_tail_capacity = self._configured_protected_tail_capacity()
        required_capacity = confirmed_capacity + protected_tail_capacity
        pool_confirmed_capacity = manager.get_num_available_tokens(
            token_num_upper_bound=confirmed_capacity,
            max_num_draft_tokens=int(manager._kv_reserve_draft_tokens) + 1,
        )
        table_capacity = manager.max_blocks_per_seq * manager.tokens_per_block
        if confirmed_capacity > pool_confirmed_capacity or required_capacity > table_capacity:
            raise ValueError(
                "TriAttention target KV capacity is too small to reach its first "
                f"eviction: request requires {required_capacity} tokens "
                f"(prompt={request.py_prompt_len}, budget={self.budget}, "
                f"beta={self.beta}, decode before eviction or completion="
                f"{decode_capacity}, speculative overshoot="
                f"{speculative_overshoot}, protected tail="
                f"{protected_tail_capacity}), "
                f"but the V2 pool covers {pool_confirmed_capacity + protected_tail_capacity} "
                f"tokens and its page table covers {table_capacity} tokens"
            )
        draft_manager = self.draft_kv_cache_manager
        if draft_manager is None:
            return
        draft_protected_tail = self._draft_protected_tail_capacity()
        draft_required_capacity = confirmed_capacity + draft_protected_tail
        draft_pool_capacity = draft_manager.get_num_available_tokens(
            token_num_upper_bound=confirmed_capacity,
            max_num_draft_tokens=int(draft_manager._kv_reserve_draft_tokens) + 1,
        )
        draft_table_capacity = draft_manager.max_blocks_per_seq * draft_manager.tokens_per_block
        if (
            confirmed_capacity > draft_pool_capacity
            or draft_required_capacity > draft_table_capacity
        ):
            raise ValueError(
                "TriAttention draft KV capacity is too small to reach the first "
                f"co-compression: request requires {draft_required_capacity} "
                f"tokens (prompt={request.py_prompt_len}, budget={self.budget}, "
                f"beta={self.beta}, decode before eviction or completion="
                f"{decode_capacity}, draft protected tail={draft_protected_tail}), "
                f"but the draft V2 pool covers "
                f"{draft_pool_capacity + draft_protected_tail} tokens and its "
                f"page table covers {draft_table_capacity} tokens"
            )

    def _draft_protected_tail_capacity(self) -> int:
        """Return the draft tail moved and re-reserved by every co-compression."""
        return _protected_tail_capacity(self.draft_kv_cache_manager, "draft ")

    def _ensure_calibrated(self) -> None:
        """Resolve calibration once for the first request."""
        if self._calibrated:
            return
        self.calibration = self._resolve_calibration()
        self._H = int(self.calibration["E_q"].shape[1])
        self._F = int(self.calibration["E_q"].shape[2])
        # Squared per-frequency RoPE scaling factor (required calibration key).
        self._freq_scale_sq = self.calibration["freq_scale_sq"].to(dtype=torch.float32)
        # Pre-split query stats + MLR coefficient for the score kernel so
        # it doesn't recompute (E_q_norm - |E_q|) per call. Shapes [L, H, F].
        _Eq = self.calibration["E_q"]
        self._triattn_q_real = _Eq.real.to(torch.float32).contiguous()
        self._triattn_q_imag = _Eq.imag.to(torch.float32).contiguous()
        self._triattn_mlr_coef = (
            self.calibration["E_q_norm"].to(torch.float32) - _Eq.abs().to(torch.float32)
        ).contiguous()
        self._calibrated = True

    def _validate_v2_compatibility(self) -> None:
        """Reject runtime modes outside the V2 physical-compaction contract."""
        manager = self.kv_cache_manager
        if not isinstance(manager, KVCacheManagerV2):
            raise ValueError("TriAttention physical eviction requires KVCacheManagerV2")
        if manager.kv_factor != 2:
            raise ValueError(
                "TriAttention requires a standard key/value KV cache; "
                "MLA/SELFKONLY caches are not supported"
            )
        if manager.mapping.enable_attention_dp:
            raise ValueError("TriAttention does not support attention DP")
        if manager.is_disagg:
            raise ValueError("TriAttention does not support disaggregated serving")
        if manager.max_beam_width != 1:
            raise ValueError("TriAttention requires beam-width-one decoding")
        if manager.enable_swa_scratch_reuse:
            raise RuntimeError("TriAttention does not support V2 SWA scratch page-table remapping")
        # Speculative feature gates (resolved draft length, linear drafting,
        # mode whitelist) run in the factory, where spec_config lives. The
        # draft cache itself is validated here whenever one is attached.
        draft_manager = self.draft_kv_cache_manager
        if draft_manager is not None:
            if not draft_manager.is_draft:
                raise ValueError(
                    "TriAttention speculative compatibility requires the actual "
                    "separate draft KV cache manager"
                )
            if draft_manager.kv_factor != 2:
                raise ValueError(
                    "TriAttention compresses the draft KV cache together with "
                    "the target, so the draft cache must be a standard "
                    "key/value cache"
                )
            if draft_manager.enable_swa_scratch_reuse:
                raise RuntimeError(
                    "TriAttention does not support V2 SWA scratch page-table remapping"
                )
            if self.eviction_mode != "union":
                raise ValueError(
                    "TriAttention draft KV co-compression supports only "
                    "eviction_mode='union'; per-head keep sets are not defined "
                    "for draft layers, which are never scored"
                )
            if any(window is not None for window in draft_manager.max_attention_window_vec) or any(
                not isinstance(layer, AttentionLayerConfig) or layer.sliding_window_size is not None
                for layer in draft_manager.kv_cache_manager_py_config.layers
            ):
                raise ValueError(
                    "TriAttention draft KV co-compression requires full-attention "
                    "draft V2 lifecycles"
                )
        if any(window is not None for window in manager.max_attention_window_vec) or any(
            not isinstance(layer, AttentionLayerConfig) or layer.sliding_window_size is not None
            for layer in manager.kv_cache_manager_py_config.layers
        ):
            raise ValueError(
                "TriAttention requires full-attention V2 lifecycles; native SWA, "
                "VSWA, and SSM pools are not supported"
            )

    # The framework drives all request-lifecycle hooks. TriAttention resolves
    # calibration on request init, evicts periodically at generation-step end,
    # and removes per-request state at finish. It scores from offline
    # calibration, not from live queries or attention scores, so it needs no
    # per-layer attention hook: the whole eviction runs once per period in
    # on_generation_step_end, which loops the layers and reads each layer's keys
    # straight from the KV pool.

    def on_generation_step_end(self, scheduled_batch: "ScheduledRequests", **kwargs) -> None:
        """Compact after native KV-cache updates have finalized this iteration.

        The compression manager is ordered after KVCacheManagerV2, so capacity
        already reflects the written token and any rewind. The overlap scheduler
        may already have enqueued the next forward; CUDA stream ordering keeps
        compaction after that reader. The resize happens only after compaction;
        it detaches the compacted tail without blocking the host, while V2's
        per-slot finish events prevent early page reuse.
        """
        with nvtx_range_debug("triattention.generation_step_end", color="blue"):
            self._periodic_evict(scheduled_batch)

    def on_generation_step_begin(self, scheduled_batch: "ScheduledRequests", **kwargs) -> None:
        """Snapshot fixed-linear target growth; mutation remains in final update."""
        generation_growth = {}
        for request in scheduled_batch.generation_requests:
            request_id = request.py_request_id
            growth = 1 + max(
                get_draft_token_length(request),
                self.kv_cache_manager._kv_reserve_draft_tokens,
            )
            generation_growth[request_id] = growth
        self._prepared_generation_batch = (scheduled_batch, generation_growth)

    def _inflight_generation_growth(
        self, scheduled_batch: "ScheduledRequests", request_id: int
    ) -> int:
        """Return exact newer target allocation width under overlap scheduling."""
        prepared = self._prepared_generation_batch
        if prepared is None or scheduled_batch is prepared[0]:
            return 0
        return prepared[1].get(request_id, 0)

    def _periodic_evict(
        self,
        scheduled_batch: "ScheduledRequests",
    ) -> None:
        """Count confirmed tokens; every ``beta`` tokens score the cache
        and physically evict to the pinned prompt plus top-B decode tokens."""
        gen_requests = scheduled_batch.generation_requests
        if not gen_requests:
            return
        mgr = self.kv_cache_manager
        resolved_requests = []
        for request in gen_requests:
            if request.is_dummy or request.state in (
                LlmRequestState.GENERATION_COMPLETE,
                LlmRequestState.CONTEXT_INIT,
            ):
                continue
            request_id = request.py_request_id
            kv_cache = mgr.kv_cache_map.get(request_id)
            if kv_cache is None:
                continue
            if not kv_cache.is_active:
                raise RuntimeError(
                    "TriAttention cannot finalize a suspended target KV cache; "
                    f"request {request_id} must be resumed before "
                    "the final update hook"
                )
            if request_id not in self._request_states:
                raise RuntimeError(
                    f"request {request_id} reached generation without on_request_init"
                )
            resolved_requests.append((request, request_id, kv_cache))
        if not resolved_requests:
            return
        protected_tails: Dict[int, int] = {}
        prepared: List[Dict[str, object]] = []
        protected_tail_capacity = self._configured_protected_tail_capacity()

        # Resolve every active target cache before changing cadence state (the
        # captured cache objects also avoid repeating the V2 map lookup here),
        # building the due cohort's per-request eviction metadata in the same
        # pass -- ``_evict_requests`` trusts it as-is.
        with nvtx_range("triattention.metadata", color="cyan"):
            for request, request_id, kv_cache in resolved_requests:
                raw_capacity = int(kv_cache.capacity)
                # One-engine speculative decoding keeps a fixed reserve E.
                # Under overlap, B(n) is allocated/enqueued before finalizing
                # B(n-1), so its exact scheduler growth Q is also opaque. Both
                # spans are contiguous after the stable target prefix and move
                # byte-for-byte.
                protected_tail = int(mgr.num_extra_kv_tokens) + self._inflight_generation_growth(
                    scheduled_batch, request_id
                )
                seq_len = raw_capacity - protected_tail
                if seq_len < 0 or protected_tail < 0:
                    raise RuntimeError(
                        f"Request {request_id} has an inconsistent protected target tail: "
                        f"confirmed={seq_len}, capacity={raw_capacity}, "
                        f"protected_tail={protected_tail}"
                    )
                if protected_tail > protected_tail_capacity:
                    raise RuntimeError(
                        f"Request {request_id} protected tail {protected_tail} exceeds "
                        f"configured capacity {protected_tail_capacity}"
                    )
                if seq_len < kv_cache.history_length:
                    raise RuntimeError(
                        f"Request {request_id} KV length {seq_len} is below finalized "
                        f"history {kv_cache.history_length}"
                    )
                request_state = self._request_states[request_id]
                request_state["confirmed_kv_length"] = seq_len
                previous_step = request_state["generation_steps"]
                confirmed_delta = 1 + int(request.py_num_accepted_draft_tokens)
                step = previous_step + confirmed_delta
                request_state["generation_steps"] = step
                if previous_step // self.beta >= step // self.beta:
                    continue
                expected_keep_count = self._minimum_evictable_length(request, seq_len)
                if seq_len <= expected_keep_count:
                    continue
                if self.draft_kv_cache_manager is not None:
                    draft_kv_cache = self.draft_kv_cache_manager.kv_cache_map.get(request_id)
                    if draft_kv_cache is None or not draft_kv_cache.is_active:
                        raise RuntimeError(
                            "TriAttention cannot co-compress a missing or "
                            f"suspended draft KV cache; request {request_id} must "
                            "be resumed before the final update hook"
                        )
                protected_tails[request_id] = protected_tail
                prepared.append(
                    {
                        "request": request,
                        "request_id": request_id,
                        "seq_len": int(seq_len),
                        # Restore the uncompressed confirmed logical position
                        # from the physical prefix and cumulative eviction
                        # count.
                        "round_start": int(seq_len + request_state["evicted_tokens"]),
                        "prompt_len": min(int(request.py_prompt_len), int(seq_len)),
                        "expected_keep_count": expected_keep_count,
                        "protected_tail": protected_tail,
                    }
                )

        # Compact all affected dense and kernel-masked SWA layers, then release
        # the unreachable tail directly through V2's public resize primitive.
        # Prompt lengths and tails are per-request metadata, so the whole due
        # cohort runs as one batched round (the buffers hold max_batch_size
        # requests, which bounds any generation batch).
        if not prepared:
            return
        num_layers = self._num_layers_from_manager()
        # Ungated NVTX with the due count in the message, so any nsys capture
        # shows how many requests each eviction round carries. This path runs
        # outside CUDA-graph capture, so the dynamic message is safe; the cost
        # is one host-side f-string per eviction round.
        with nvtx_range(
            f"triattention.evict_request_group reqs={len(prepared)}",
            color="purple",
        ):
            capacity_targets = self._evict_requests(prepared, num_layers)
        self._resize_compacted_requests(capacity_targets, protected_tails)

    def _resize_compacted_requests(self, capacity_targets, protected_tails) -> None:
        if not capacity_targets:
            return
        mgr = self.kv_cache_manager
        draft_manager = self.draft_kv_cache_manager
        with nvtx_range("triattention.resize", color="red"):
            with nvtx_range_debug("triattention.v2_resize", color="red"):
                for rid, target_capacity in capacity_targets:
                    kv_cache = mgr.kv_cache_map.get(rid)
                    if kv_cache is None or not kv_cache.is_active:
                        continue
                    if target_capacity > kv_cache.capacity:
                        raise RuntimeError(
                            f"Request {rid} compacted capacity {target_capacity} exceeds "
                            f"current capacity {kv_cache.capacity}"
                        )
                    protected_tail = protected_tails[rid]
                    resized_capacity = target_capacity + protected_tail
                    if not kv_cache.resize(resized_capacity, None):
                        raise RuntimeError(
                            f"Failed to resize compacted KV cache for request {rid} "
                            f"to {resized_capacity} tokens"
                        )
                if draft_manager is not None:
                    # The draft cache was compacted with the same kept token
                    # set, so it shrinks to the same retained length plus its
                    # own protected tail.
                    draft_protected_tail = self._draft_protected_tail_capacity()
                    for rid, target_capacity in capacity_targets:
                        draft_kv_cache = draft_manager.kv_cache_map.get(rid)
                        if draft_kv_cache is None or not draft_kv_cache.is_active:
                            continue
                        draft_capacity = target_capacity + draft_protected_tail
                        if not draft_kv_cache.resize(draft_capacity, None):
                            raise RuntimeError(
                                "Failed to resize co-compressed draft KV cache "
                                f"for request {rid} to {draft_capacity} tokens"
                            )

    def _minimum_evictable_length(self, request: "LlmRequest", seq_len: int) -> int:
        """Return the largest cache length for which selection is an identity.

        With a decode-only budget, pinned prompt tokens do not consume ``budget``.
        Selection therefore keeps every token until the cache exceeds
        ``prompt_len + budget``. The constructor guarantees the decode-only
        budget (``pin_prefill=True``, ``count_prompt_tokens=False``).
        """
        prompt_len = min(int(request.py_prompt_len), seq_len)
        return prompt_len + self.budget

    def _local_score_calibration(
        self,
        num_layers: int,
        global_layers: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return calibration tensors indexed in this PP rank's local layer order."""
        if global_layers and max(global_layers) >= self._triattn_q_real.shape[0]:
            raise ValueError(
                f"TriAttention calibration has {self._triattn_q_real.shape[0]} layers, "
                f"but this PP rank references global layer {max(global_layers)}"
            )
        if global_layers == list(range(global_layers[0], global_layers[0] + num_layers)):
            layer_slice = slice(global_layers[0], global_layers[0] + num_layers)
            return (
                self._triattn_q_real[layer_slice],
                self._triattn_q_imag[layer_slice],
                self._triattn_mlr_coef[layer_slice],
            )
        layer_ids = torch.as_tensor(
            global_layers,
            device=self._triattn_q_real.device,
            dtype=torch.long,
        )
        return (
            self._triattn_q_real.index_select(0, layer_ids),
            self._triattn_q_imag.index_select(0, layer_ids),
            self._triattn_mlr_coef.index_select(0, layer_ids),
        )

    def _configured_protected_tail_capacity(self) -> int:
        """Return the largest target tail reserved by the native V2 lifecycle."""
        return _protected_tail_capacity(self.kv_cache_manager, "")

    def on_request_finish(self, request: "LlmRequest", **kwargs) -> None:
        """Drop this request's per-request length and eviction state."""
        request_id = request.py_request_id
        self._request_states.pop(request_id, None)
        prepared = self._prepared_generation_batch
        if prepared is not None:
            prepared[1].pop(request_id, None)
        # The buffers stay resident across idle periods: their memory is a
        # deliberate one-time cost and rebuilding it per burst would reintroduce
        # allocation on the decode hot path.

    # ================================================================== #
    # Helpers (eviction / scoring / V2 cache access / calibration)       #
    # ================================================================== #

    def _local_to_global_layers(self, num_layers: int) -> List[int]:
        """Return V2's global layer id for every local TriAttention layer slot."""
        cached = self._local_to_global_layers_cache
        if cached is not None:
            if len(cached) != num_layers:
                raise ValueError(
                    f"TriAttention layer count changed from {len(cached)} to {num_layers}"
                )
            return cached

        global_layers = [int(layer) for layer in self.kv_cache_manager.pp_layers]
        if len(global_layers) != num_layers:
            raise ValueError(
                f"KVCacheManagerV2 exposes {len(global_layers)} PP layers, "
                f"but TriAttention received {num_layers} local layers"
            )
        self._local_to_global_layers_cache = global_layers
        return global_layers

    @staticmethod
    def _has_sliding_window_signal(config: Dict[str, object]) -> bool:
        """Return whether config metadata hints at sliding attention."""
        use_sliding_window = config.get("use_sliding_window")
        if isinstance(use_sliding_window, bool):
            return use_sliding_window
        for field in (
            "sliding_window",
            "sliding_window_size",
            "sliding_window_pattern",
            "max_window_layers",
        ):
            value = config.get(field)
            if isinstance(value, bool):
                if value:
                    return True
            elif isinstance(value, (int, float)):
                if value > 0:
                    return True
            elif value:
                return True
        return False

    def _attention_layer_partition(
        self, num_layers: int
    ) -> Tuple[List[int], List[int], Optional[int]]:
        """Return dense layers, kernel-masked SWA layers, and the SWA window.

        TriAttention initialization has already rejected real V2 windowed
        lifecycles. A sliding layer found here is therefore stored at full length
        and applies its window only in the attention kernel.
        """
        cached = self._attention_layer_partition_cache
        if cached is not None:
            return cached

        model_path = self.model_path
        if model_path is None:
            raise ValueError("TriAttention requires model_path")

        try:
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(
                model_path, trust_remote_code=True, local_files_only=True
            )
        except Exception as exc:
            raise ValueError(
                f"TriAttention could not load the local model config from {model_path!r}"
            ) from exc
        config_values = config.get_text_config().to_dict()
        layer_types = config_values.get("layer_types")
        if not layer_types:
            if self._has_sliding_window_signal(config_values):
                raise ValueError(
                    "Model config exposes sliding-window metadata but no layer_types; "
                    "TriAttention cannot classify kernel-masked SWA layers safely"
                )
            result = (list(range(num_layers)), [], None)
            self._attention_layer_partition_cache = result
            return result
        global_layers = self._local_to_global_layers(num_layers)
        if global_layers and max(global_layers) >= len(layer_types):
            raise ValueError(
                f"Model config has {len(layer_types)} layer_types entries, "
                f"but this PP rank references global layer {max(global_layers)}"
            )

        swa_layers = [
            local_layer
            for local_layer, global_layer in enumerate(global_layers)
            if "sliding" in str(layer_types[global_layer]).lower()
        ]
        swa_set = set(swa_layers)
        dense_layers = [layer for layer in range(num_layers) if layer not in swa_set]
        window_size = None
        if swa_layers:
            raw_window = config_values.get("sliding_window")
            if not isinstance(raw_window, int) or raw_window <= 0:
                raise ValueError(
                    "TriAttention requires a positive integer model sliding_window "
                    "when layer_types contains sliding attention"
                )
            if self.budget < raw_window:
                raise ValueError(
                    f"TriAttention budget={self.budget} must be at least "
                    f"the kernel-masked SWA window size {raw_window}"
                )
            window_size = raw_window
        result = (dense_layers, swa_layers, window_size)
        self._attention_layer_partition_cache = result
        return result

    def _runtime_kv_layout(self, num_layers: int) -> Dict[str, object]:
        """Return stable V2 pool views and layer groups for eviction.

        KVCacheManagerV2 keeps GPU virtual addresses and layer geometry stable,
        while opt-in pool rebalance can change the page dimension. Cache all
        layer views, then query the live page count for one representative per
        physical pool before reuse (fail-closed rebalance check).
        """
        cached = self._runtime_kv_layout_cache
        manager = self.kv_cache_manager
        if cached is not None:
            if cached["num_layers"] != num_layers:
                raise ValueError(
                    f"TriAttention layer count changed from {cached['num_layers']} to {num_layers}"
                )
            if cached["manager"] is not manager:
                raise RuntimeError("TriAttention target KV cache manager changed at runtime")
            current_page_counts = self._pool_page_counts(
                manager,
                cached["global_layers"],
                cached["pool_representatives"],
            )
            if current_page_counts != cached["pool_page_counts"]:
                raise RuntimeError(
                    "TriAttention V2 pool layout changed after the layout was built; "
                    "KV pool rebalance is not supported"
                )
            return cached

        global_layers = self._local_to_global_layers(num_layers)
        dense_layers, swa_layers, swa_window = self._attention_layer_partition(num_layers)
        if not dense_layers:
            raise ValueError("TriAttention requires at least one full-attention layer")
        layout = self._build_runtime_kv_layout(
            manager,
            global_layers,
            dense_layers=dense_layers,
            swa_layers=swa_layers,
            swa_window=swa_window,
            dense_storage_groups=self._dense_layer_pool_groups(dense_layers, global_layers),
            what="",
        )
        self._runtime_kv_layout_cache = layout
        return layout

    def _build_runtime_kv_layout(
        self,
        manager: KVCacheManagerV2,
        global_layers: List[int],
        *,
        dense_layers: List[int],
        swa_layers: List[int],
        swa_window: Optional[int],
        dense_storage_groups: Optional[Dict[object, List[int]]],
        what: str,
    ) -> Dict[str, object]:
        """Build the manager-lifetime layer and pool views one eviction reads.

        ``dense_storage_groups`` restricts the compaction groups to the dense
        layers (target cache); None groups every layer (draft cache, which has
        no SWA partition). ``what`` prefixes error messages ("" or "draft ").
        """
        num_layers = len(global_layers)
        maybe_layer_pools = [manager.get_buffers(layer, kv_layout="HND") for layer in global_layers]
        if any(pool is None for pool in maybe_layer_pools):
            missing = [
                layer for layer, pool in zip(global_layers, maybe_layer_pools) if pool is None
            ]
            raise RuntimeError(f"Missing {what}KV pools for attention layers {missing}")
        layer_pools = [pool for pool in maybe_layer_pools if pool is not None]
        all_layers = list(range(num_layers))
        layer_pool_keys = tuple(
            self._page_table_pool_keys(all_layers, global_layers, manager=manager)
        )
        all_storage_groups: Dict[object, List[int]] = {}
        for layer, pool_key in zip(all_layers, layer_pool_keys):
            all_storage_groups.setdefault(pool_key, []).append(layer)
        storage_groups = (
            dense_storage_groups if dense_storage_groups is not None else all_storage_groups
        )
        layer_group_representative = {
            layer: layers[0] for layers in storage_groups.values() for layer in layers
        }
        pool_representatives = tuple(layers[0] for layers in all_storage_groups.values())
        return dict(
            manager=manager,
            num_layers=num_layers,
            global_layers=global_layers,
            layer_pools=layer_pools,
            dense_layers=dense_layers,
            swa_layers=swa_layers,
            swa_window=swa_window,
            storage_groups=storage_groups,
            layer_group_representative=layer_group_representative,
            layer_pool_keys=layer_pool_keys,
            pool_representatives=pool_representatives,
            pool_page_counts=tuple(
                int(layer_pools[layer].shape[0]) for layer in pool_representatives
            ),
            pool_view_fingerprint=self._pool_view_fingerprint(
                [layer_pools[layer] for layer in pool_representatives]
            ),
        )

    def _draft_runtime_kv_layout(self) -> Dict[str, object]:
        """Return stable draft V2 pool views, mirroring ``_runtime_kv_layout``.

        The draft cache is compacted with the target's kept token set, so its
        layout has no scoring role: every draft layer is dense and there is no
        SWA partition.
        """
        manager = self.draft_kv_cache_manager
        if manager is None:
            raise RuntimeError("TriAttention has no draft KV cache manager to lay out")
        cached = self._draft_runtime_kv_layout_cache
        if cached is not None:
            if cached["manager"] is not manager:
                raise RuntimeError("TriAttention draft KV cache manager changed at runtime")
            current_page_counts = self._pool_page_counts(
                manager,
                cached["global_layers"],
                cached["pool_representatives"],
            )
            if current_page_counts != cached["pool_page_counts"]:
                raise RuntimeError(
                    "TriAttention draft V2 pool layout changed after the layout "
                    "was built; KV pool rebalance is not supported"
                )
            return cached

        global_layers = [int(layer) for layer in manager.pp_layers]
        if not global_layers:
            raise RuntimeError("TriAttention draft KV cache manager exposes no layers")
        layout = self._build_runtime_kv_layout(
            manager,
            global_layers,
            dense_layers=list(range(len(global_layers))),
            swa_layers=[],
            swa_window=None,
            dense_storage_groups=None,
            what="draft ",
        )
        self._draft_runtime_kv_layout_cache = layout
        return layout

    @staticmethod
    def _pool_page_counts(
        manager: KVCacheManagerV2,
        global_layers: Sequence[int],
        pool_representatives: Sequence[int],
    ) -> Tuple[int, ...]:
        """Read the only pool-view dimension that V2 rebalance can change."""
        return tuple(
            int(
                manager.impl.get_page_index_upper_bound(
                    manager.layer_offsets[global_layers[layer]],
                    Role.KEY,
                )
            )
            // int(manager.kv_factor)
            for layer in pool_representatives
        )

    @staticmethod
    def _pool_view_fingerprint(pools: List[torch.Tensor]) -> Tuple[tuple, ...]:
        """Identify the V2 pool properties consumed by score and compact kernels."""
        return tuple(
            (
                pool.data_ptr(),
                tuple(int(value) for value in pool.shape),
                tuple(int(value) for value in pool.stride()),
                pool.dtype,
                pool.device,
            )
            for pool in pools
        )

    def _buffers_for(
        self,
        layout: Dict[str, object],
        prepared: Sequence[Dict[str, object]],
    ) -> SimpleNamespace:
        """Return the eviction buffers, building them once at first use.

        The request capacity follows the executor's max batch size (memory
        scales linearly with it) and the decode-width capacity follows the
        eviction bound (compaction keeps the scored decode region near
        ``budget`` plus one period of growth), so one set of buffers serves
        every round. It is rebuilt only when the pool views change or a
        round outgrows it.
        """
        if not prepared:
            raise ValueError("TriAttention eviction requires at least one request")
        needed_width = max(item["seq_len"] - item["prompt_len"] for item in prepared)
        needed_page_tokens = max(item["seq_len"] + item["protected_tail"] for item in prepared)
        needed_requests = len(prepared)
        draft_fingerprint = None
        if self.draft_kv_cache_manager is not None:
            draft_layout = self._draft_runtime_kv_layout()
            draft_fingerprint = (
                draft_layout["pool_page_counts"],
                draft_layout["pool_view_fingerprint"],
            )
        fingerprint = (
            self.eviction_mode,
            self.budget,
            tuple(layout["dense_layers"]),
            layout["pool_view_fingerprint"],
            draft_fingerprint,
        )
        bufs = self._buffers
        if bufs is not None:
            if (
                self._buffers_fingerprint == fingerprint
                and needed_width <= bufs.decode_width
                and needed_page_tokens <= bufs.page_table_token_capacity
                and needed_requests <= bufs.max_requests
            ):
                return bufs
            # Pools changed or this round outgrew the buffers: rebuild.
            self._buffers = None

        mgr = self.kv_cache_manager
        tail_capacity = self._configured_protected_tail_capacity()
        request_capacity = max(needed_requests, int(mgr.max_batch_size))
        decode_width = max(
            needed_width,
            self.budget + 2 * self.beta + int(mgr.max_total_draft_tokens or 0),
        )
        # Bucket the score scratch by what cohorts actually present instead of
        # pinning it to max_seq_len: with pinned prompts the post-compaction
        # length is bounded by prompt + budget + slack, so one power-of-two
        # bucket serves the steady state, and a cohort that outgrows it simply
        # rebuilds the buffers through the capacity check above. A
        # max_seq_len floor would make the scratch unindexable in 32 bits and
        # tens of GiB at large batch for work that never scores past ~1K
        # tokens per request.
        seq_capacity = max(int(needed_page_tokens), 1024)
        seq_capacity = 1 << (seq_capacity - 1).bit_length()
        seq_capacity = min(seq_capacity, max(int(mgr.max_seq_len), int(needed_page_tokens)))
        # The CuTe score kernel stores full compute tiles into a scratch
        # strided by this bucket capacity, so the capacity must be
        # tile-aligned (the geometry gate rejects anything else). Rounding up
        # costs at most one tile of scratch per segment.
        score_tile_tokens = max(64, int(mgr.tokens_per_block))
        seq_capacity = -(-seq_capacity // score_tile_tokens) * score_tile_tokens
        # Mis-tiled buckets stripe the score scratch silently; the builder
        # guarantees alignment right here.
        assert seq_capacity % score_tile_tokens == 0
        page_table_token_capacity = max(needed_page_tokens, seq_capacity + tail_capacity)

        dense_groups = list(layout["storage_groups"].values())
        representatives = [group[0] for group in dense_groups]
        representatives.extend(
            layer for layer in layout["swa_layers"] if layer not in representatives
        )
        draft_kwargs = {}
        if self.draft_kv_cache_manager is not None:
            draft_layout = self._draft_runtime_kv_layout()
            draft_tail_capacity = self._draft_protected_tail_capacity()
            draft_representatives = list(draft_layout["pool_representatives"])
            draft_kwargs = dict(
                draft_layer_pools=draft_layout["layer_pools"],
                draft_layers=draft_layout["dense_layers"],
                draft_layer_group_representative=draft_layout["layer_group_representative"],
                draft_layer_pool_keys=list(draft_layout["layer_pool_keys"]),
                draft_page_representatives=draft_representatives,
                draft_page_table_keys=[
                    draft_layout["layer_pool_keys"][layer] for layer in draft_representatives
                ],
                draft_num_page_table_slots=self.draft_kv_cache_manager.num_pools,
                draft_page_table_token_capacity=seq_capacity + draft_tail_capacity,
                draft_protected_tail_capacity=draft_tail_capacity,
            )

        first_pool = layout["layer_pools"][layout["dense_layers"][0]]
        if self._offsets is None:
            # Upstream pruning_utils.build_geometric_offsets: [1, 2, 4, ... <= max].
            self._offsets = torch.tensor(
                [float(1 << i) for i in range(_OFFSET_MAX_LENGTH.bit_length())],
                device=first_pool.device,
                dtype=torch.float32,
            )
        if self._phase is None:
            self._phase = build_mean_phase_table(
                self._offsets,
                self.calibration["omega"]
                .to(device=first_pool.device, dtype=torch.float32)
                .contiguous(),
                initial_rows=seq_capacity,
            )
        q_real, q_imag, mlr_coef = self._local_score_calibration(
            layout["num_layers"], layout["global_layers"]
        )
        bufs = init_eviction_buffers(
            eviction_mode=self.eviction_mode,
            layer_pools=layout["layer_pools"],
            dense_groups=dense_groups,
            dense_layers=layout["dense_layers"],
            swa_layers=layout["swa_layers"],
            swa_window=layout["swa_window"],
            layer_group_representative=layout["layer_group_representative"],
            layer_pool_keys=list(layout["layer_pool_keys"]),
            page_representatives=representatives,
            max_requests=request_capacity,
            seq_len=seq_capacity,
            num_q_heads=int(self._H),
            num_freqs=int(self._F),
            keep_count=self.budget,
            q_real=q_real,
            q_imag=q_imag,
            mlr_coef=mlr_coef,
            freq_scale_sq=self._freq_scale_sq,
            offsets=self._offsets,
            omega=self.calibration["omega"],
            phase=self._phase,
            page_table_keys=self._page_table_pool_keys(representatives, layout["global_layers"]),
            num_page_table_slots=layout["manager"].num_pools,
            decode_width=decode_width,
            page_table_token_capacity=page_table_token_capacity,
            protected_tail_capacity=tail_capacity,
            **draft_kwargs,
        )
        self._buffers = bufs
        self._buffers_fingerprint = fingerprint
        return bufs

    def _move_offsets_for(
        self,
        layout: Dict[str, object],
        prepared: Sequence[Dict[str, object]],
        capacity: int,
    ) -> Tuple[List[int], Optional[List[int]], Optional[List[int]]]:
        """Build this round's per-family move offsets, padded to the capacity.

        Rows past the cohort repeat the final offset, so padded requests move
        nothing in the pack kernel and the C++ compact launches.
        """

        def padded_offsets(moves_per_request: List[int]) -> List[int]:
            offsets = [0]
            for moves in moves_per_request:
                offsets.append(offsets[-1] + moves)
            offsets.extend(offsets[-1:] * (capacity - len(moves_per_request)))
            return offsets

        tails = [item["protected_tail"] for item in prepared]
        dense = padded_offsets([self.budget + tail for tail in tails])
        swa = None
        if layout["swa_layers"] and layout["swa_window"]:
            swa = padded_offsets([int(layout["swa_window"]) + tail for tail in tails])
        draft = None
        if self.draft_kv_cache_manager is not None:
            draft_tail = self._draft_protected_tail_capacity()
            draft = padded_offsets([self.budget + draft_tail] * len(prepared))
        return dense, swa, draft

    def _page_table_pool_keys(
        self,
        representatives: List[int],
        global_layers: List[int],
        manager: Optional[KVCacheManagerV2] = None,
    ) -> List[object]:
        """Return stable V2-pool keys for the representative layers."""
        if manager is None:
            manager = self.kv_cache_manager
        layer_offsets = manager.layer_offsets
        layer_to_pool = manager.layer_to_pool_mapping_dict
        try:
            return [
                ("pool", int(layer_to_pool[layer_offsets[global_layers[layer]]]))
                for layer in representatives
            ]
        except (IndexError, KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("KVCacheManagerV2 exposes an invalid layer-to-pool mapping") from exc

    def _dense_layer_pool_groups(
        self,
        dense_layers: List[int],
        global_layers: List[int],
    ) -> Dict[object, List[int]]:
        """Group layers that use the same V2 page table."""
        groups: Dict[object, List[int]] = {}
        for layer, pool_key in zip(
            dense_layers,
            self._page_table_pool_keys(dense_layers, global_layers),
        ):
            groups.setdefault(pool_key, []).append(layer)
        return groups

    def _evict_requests(
        self,
        prepared: List[Dict[str, object]],
        num_layers: int,
    ) -> List[Tuple[int, int]]:
        """Score and compact a prepared cohort, returning ``(request_id, capacity)`` targets.

        ``prepared`` carries the per-request eviction metadata resolved by
        ``_periodic_evict`` (every entry is due and evictable). Only
        full-attention layers participate in scoring. For kernel-masked SWA
        layers, the latest model window is rebased to the tail of the common
        compacted prefix before the request-wide capacity is reduced.
        """
        with nvtx_range_debug("triattention.resolve_layout", color="blue"):
            layout = self._runtime_kv_layout(num_layers)
        with nvtx_range_debug("triattention.staging_lookup", color="blue"):
            bufs = self._buffers_for(layout, prepared)
            if layout["swa_layers"] and layout["swa_window"]:
                # SWA landing positions are prompt-dependent; reject a request
                # whose retained span cannot cover the model window this round.
                for item in prepared:
                    if item["prompt_len"] + self.budget < int(layout["swa_window"]):
                        raise ValueError(
                            f"Request {item['request_id']} retains "
                            f"{item['prompt_len'] + self.budget} tokens, below the "
                            f"sliding window {layout['swa_window']}"
                        )
        with nvtx_range_debug("triattention.page_table_stage", color="orange"):
            dense_offsets, swa_offsets, draft_offsets = self._move_offsets_for(
                layout, prepared, bufs.max_requests
            )
            stage_eviction_cohort(
                bufs,
                self.kv_cache_manager,
                [item["request_id"] for item in prepared],
                [item["round_start"] for item in prepared],
                [item["prompt_len"] for item in prepared],
                [item["seq_len"] for item in prepared],
                draft_manager=self.draft_kv_cache_manager,
                dense_move_offsets=dense_offsets,
                swa_move_offsets=swa_offsets,
                draft_move_offsets=draft_offsets,
            )

        try:
            run_eviction_round(bufs, self.normalize_scores)
        finally:
            consumer_streams = [self.kv_cache_manager._stream]
            if self.draft_kv_cache_manager is not None:
                consumer_streams.append(self.draft_kv_cache_manager._stream)
            mark_page_tables_consumed(bufs, *consumer_streams)

        capacity_targets = []
        for item in prepared:
            keep_count = item["expected_keep_count"]
            evicted = item["seq_len"] - keep_count
            if evicted <= 0:
                raise RuntimeError("TriAttention attempted an identity compaction")
            request_state = self._request_states[item["request_id"]]
            request_state["evicted_tokens"] += evicted
            request_state["confirmed_kv_length"] = keep_count
            # Publish the cumulative count on the request: this is the
            # manager's only channel to the runtime. The model engine
            # reads it back where it builds num_cached_tokens_per_seq,
            # so the kernels see the compacted KV length next step.
            item["request"].py_num_compressed_tokens = request_state["evicted_tokens"]
            capacity_targets.append((item["request_id"], keep_count))
        return capacity_targets

    def _num_layers_from_manager(self) -> int:
        return len(self.kv_cache_manager.pp_layers)

    # ------------------------------------------------------------------ #
    # Helpers: calibration loading                                       #
    # ------------------------------------------------------------------ #

    def _resolve_calibration(self) -> Dict[str, torch.Tensor]:
        """Load the user-supplied calibration .pt and return our runtime schema.

        TriAttention does NOT compute calibration -- the user calibrates with the
        official tool (github.com/WeianMao/triattention) and passes that file via
        ``calibration_path``; we only run inference. Both the official R-KV layout
        (``{metadata, stats{"layerLL_headHH": {q_mean_real, q_mean_imag,
        q_abs_mean}}}``) and our already-converted flat layout are accepted -- the
        official one is converted here. Calibration resolves lazily on the
        first request (``on_request_init``), not at manager construction."""
        if self.calibration_path is None:
            raise ValueError(
                "TriAttention requires `calibration_path`: a calibration .pt from "
                "the official tool (github.com/WeianMao/triattention). TRT-LLM does "
                "not compute calibration -- see examples/ for the Qwen3-8B file and "
                "the official calibration instructions."
            )
        raw = torch.load(self.calibration_path, map_location="cpu", weights_only=False)
        if isinstance(raw, dict) and _REQUIRED_CALIBRATION_KEYS <= set(raw):
            return {k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in raw.items()}
        if isinstance(raw, dict) and {"metadata", "stats"} <= set(raw):
            return self._convert_official_calibration(raw)
        got = sorted(raw.keys()) if isinstance(raw, dict) else type(raw).__name__
        raise ValueError(
            f"Unrecognized calibration at {self.calibration_path}: expected the "
            f"official {{metadata, stats}} layout or "
            f"{sorted(_REQUIRED_CALIBRATION_KEYS)}; got {got}."
        )

    def _convert_official_calibration(self, raw) -> Dict[str, torch.Tensor]:
        """Convert the official per-(layer, head) stats to our flat runtime schema.

        ``E_q[l,h] = q_mean_real + i*q_mean_imag`` and ``E_q_norm[l,h] =
        q_abs_mean`` are the same statistic, just restacked into ``[L, H, F]``.
        ``omega`` / ``freq_scale_sq`` are not in the official file (its runtime
        recomputes them from the model rotary), so we derive them from the model
        config -- model-intrinsic and corpus-independent."""
        stats = raw["stats"]
        meta = raw.get("metadata", {})
        if "sampled_heads" in meta:
            heads = [(int(a), int(b)) for a, b in meta["sampled_heads"]]
        else:
            heads = [
                (int(k[len("layer") : k.index("_head")]), int(k[k.index("_head") + len("_head") :]))
                for k in stats
            ]
        num_layers = max(layer for layer, _ in heads) + 1
        num_heads = max(h for _, h in heads) + 1
        freq_count = int(next(iter(stats.values()))["q_mean_real"].numel())
        E_q = torch.zeros(num_layers, num_heads, freq_count, dtype=torch.complex64)
        E_q_norm = torch.zeros(num_layers, num_heads, freq_count, dtype=torch.float32)
        for layer, h in heads:
            s = stats[f"layer{layer:02d}_head{h:02d}"]
            E_q[layer, h] = torch.complex(s["q_mean_real"].float(), s["q_mean_imag"].float())
            E_q_norm[layer, h] = s["q_abs_mean"].float()
        omega, freq_scale_sq = self._rope_tables(freq_count)
        calib = {
            "E_q": E_q.to("cuda"),
            "E_q_norm": E_q_norm.to("cuda"),
            "omega": omega.to("cuda"),
            "freq_scale_sq": freq_scale_sq.to("cuda"),
        }
        logger.info(
            f"TriAttention: converted official calibration {self.calibration_path}"
            f" -> E_q[L={num_layers}, H={num_heads}, F={freq_count}]"
        )
        return calib

    def _rope_tables(self, freq_count: int):
        """RoPE ``omega`` (inv_freq) + ``freq_scale_sq`` (squared position-0
        amplitude) from the model config -- model-intrinsic, corpus-independent
        (the official file does not store them). Reads both config generations:
        transformers>=5.5 ``rope_parameters`` (rope_theta folded inside) and the
        legacy top-level ``rope_scaling``/``rope_theta``. Plain RoPE uses the
        standard formula with the resolved theta (attention_factor 1); scaled
        variants (yarn, llama3, ...) go through transformers' rope-init so their
        attention_factor is honored. The analytic fallback survives ONLY for
        ImportError (rope-init module absent); every other failure raises."""
        import transformers
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True).get_text_config()
        config_values = cfg.to_dict()
        head_dim = freq_count * 2
        rope_params = (
            config_values.get("rope_parameters") or config_values.get("rope_scaling") or {}
        )
        if rope_params and all(isinstance(v, dict) for v in rope_params.values()):
            raise ValueError(
                f"TriAttention: layer-type-keyed rope_parameters are not supported for "
                f"calibration conversion (model {self.model_path}); got {rope_params!r}."
            )
        rope_type = rope_params.get("rope_type") or rope_params.get("type") or "default"
        theta_seen = rope_params.get("rope_theta", config_values.get("rope_theta"))
        base = float(theta_seen) if theta_seen is not None else 10000.0

        def analytic_inv_freq():
            idx = torch.arange(0, head_dim, 2, dtype=torch.float32)
            return (1.0 / (base ** (idx / head_dim)))[:freq_count].clone()

        if rope_type == "default":
            # The original RoPE formula (transformers>=5.5 computes it per-model
            # and no longer keys "default" in ROPE_INIT_FUNCTIONS).
            omega, scale_sq = analytic_inv_freq(), 1.0
        else:
            try:
                from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
            except ImportError:
                logger.warning(
                    f"TriAttention: transformers rope-init unavailable; using the analytic "
                    f"inv_freq with theta={base} for {self.model_path} and IGNORING "
                    f"rope_type={rope_type!r} scaling corrections."
                )
                return analytic_inv_freq(), torch.ones(freq_count, dtype=torch.float32)
            if rope_type not in ROPE_INIT_FUNCTIONS:
                raise ValueError(
                    f"TriAttention: unknown rope_type {rope_type!r} for {self.model_path} "
                    f"(transformers {transformers.__version__} provides "
                    f"{sorted(ROPE_INIT_FUNCTIONS)}); rope config seen: {rope_params!r}."
                )
            try:
                inv_freq, attention_factor = ROPE_INIT_FUNCTIONS[rope_type](cfg, device="cpu")
            except Exception as exc:
                raise ValueError(
                    f"TriAttention: rope-init {rope_type!r} failed for {self.model_path}; "
                    f"rope config seen: {rope_params!r}."
                ) from exc
            omega = inv_freq.to(torch.float32)[:freq_count].clone()
            scale_sq = float(attention_factor) ** 2
        return omega, torch.full((freq_count,), scale_sq, dtype=torch.float32)
