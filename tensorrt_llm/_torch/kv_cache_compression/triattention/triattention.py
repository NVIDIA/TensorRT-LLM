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

"""TriAttention KV-cache compression: periodic physical KV eviction during generation.

Every ``beta`` confirmed tokens, cached tokens are scored with a trigonometric
importance score from offline calibration and tokens outside the top-``budget``
keep set are physically deleted; decode runs the model's standard attention over
the compacted cache. Kept keys keep their original RoPE rotation (no re-RoPE).
KV pools must be read with ``kv_layout="HND"``. Calibration comes from the
official tool (github.com/WeianMao/triattention) and is converted at load.
"""

from types import SimpleNamespace
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

import torch
import triton

from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2, Role
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm._torch.pyexecutor.resource_manager import BaseKVCacheCompressionManager
from tensorrt_llm._utils import nvtx_range, nvtx_range_debug, prefer_pinned
from tensorrt_llm.bindings.internal.batch_manager.kv_cache_manager_v2_utils import (
    copy_batch_block_offsets_to_device,
)
from tensorrt_llm.logger import logger
from tensorrt_llm.runtime.kv_cache_manager_v2 import AttentionLayerConfig

from ..compaction import compact, init_compaction_buffers
from .triattention_kernels import (
    _gather_mean_phase_kernel,
    _settle_ties_kernel,
    grow_mean_phase_table,
    prepare_per_head_scores,
)

if TYPE_CHECKING:
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
    from tensorrt_llm._torch.pyexecutor.scheduler.scheduler import ScheduledRequests


# Required keys for the calibration ``.pt`` consumed by TriAttention.
_REQUIRED_CALIBRATION_KEYS = frozenset({"E_q", "E_q_norm", "omega", "freq_scale_sq"})

# Generation requests skipped by every eviction step.
_SKIP_REQUEST_STATES = (
    LlmRequestState.GENERATION_COMPLETE,
    LlmRequestState.CONTEXT_INIT,
)

# Upper bound of the geometric integration offset ladder [1, 2, 4, ...].
_OFFSET_MAX_LENGTH = 65536


def _allocate_block_offset_staging(
    anchor_pool: torch.Tensor,
    *,
    num_pools: int,
    max_requests: int,
    token_capacity: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """One pinned host snapshot + persistent device table pair in the native
    V2 ``[pool, request, K/V, block]`` layout (block width 4-aligned for the
    ``PackedInt`` copy ABI); the device follows the anchor KV pool."""
    tokens_per_block = int(anchor_pool.shape[3])
    page_count = (token_capacity + tokens_per_block - 1) // tokens_per_block
    staged_blocks_per_seq = (page_count + 3) // 4 * 4
    shape = (num_pools, max_requests, 2, staged_blocks_per_seq)
    host = torch.empty(shape, dtype=torch.int32, device="cpu", pin_memory=prefer_pinned())
    device_table = torch.empty(shape, dtype=torch.int32, device=anchor_pool.device)
    return host, device_table


def init_eviction_buffers(
    *,
    eviction_mode: str,
    layout: Dict[str, object],
    calibration: Dict[str, torch.Tensor],
    phase: Dict[str, object],
    capacities: Dict[str, int],
    draft: Optional[Dict[str, object]] = None,
    normalize_scores: bool = True,
) -> SimpleNamespace:
    """Build the one namespace of buffers, compiled launches, and compaction data
    (the compiled kernels capture raw pool addresses: scored pools must stay alive and stay put).

    ``layout`` is the runtime KV layout dict, passed whole; ``calibration``
    carries the local q_real/q_imag/mlr_coef [L, H, F] slices and
    freq_scale_sq; ``draft`` is one all-or-none resolved branch (its layout
    dict plus tail/page-table capacities); ``capacities`` the capacity numbers.
    Static round policy (``normalize_scores``, mode) binds here, once.
    """
    from .triattention_cute_score_fused import N as PADDED_HEAD_COLUMNS
    from .triattention_cute_score_fused import TriAttentionCuteScoreRunner

    layer_pools = layout["layer_pools"]
    dense_layers = list(layout["dense_layers"])
    swa_layers = list(layout["swa_layers"])
    swa_window = layout["swa_window"]
    layer_group_representative = layout["layer_group_representative"]
    # Canonical layer -> V2 pool id tuple; it IS the staged plane slot map.
    layer_pool_ids = tuple(layout["layer_pool_ids"])
    dense_groups = list(layout["storage_groups"].values())
    page_representatives = [group[0] for group in dense_groups]
    page_representatives.extend(layer for layer in swa_layers if layer not in page_representatives)
    num_page_table_slots = int(layout["manager"].num_pools)

    device = layer_pools[page_representatives[0]].device
    max_requests = int(capacities["max_requests"])
    seq_len = int(capacities["bucket_seq_len"])
    page_table_token_capacity = int(capacities["page_table_token_capacity"])
    decode_width = int(capacities["decode_width"])
    keep_count = int(capacities["keep_count"])
    protected_tail_capacity = int(capacities["protected_tail_capacity"])

    q_real, q_imag, mlr_coef, freq_scale_sq = (
        calibration[key].to(device=device, dtype=torch.float32).contiguous()
        for key in ("q_real", "q_imag", "mlr_coef", "freq_scale_sq")
    )
    num_q_heads = int(q_real.shape[1])
    num_freqs = int(q_real.shape[2])

    bufs = SimpleNamespace()
    bufs.eviction_mode = eviction_mode
    bufs.normalize_scores = bool(normalize_scores)
    bufs.max_requests = max_requests
    bufs.bucket_seq_len = seq_len
    bufs.decode_width = decode_width
    bufs.keep_count = keep_count
    bufs.page_table_token_capacity = page_table_token_capacity

    # ---- block-offset staging (target, plus the co-compressed draft) -------
    bufs.block_offsets_host, bufs.block_offsets_device = _allocate_block_offset_staging(
        layer_pools[page_representatives[0]],
        num_pools=num_page_table_slots,
        max_requests=max_requests,
        token_capacity=page_table_token_capacity,
    )
    # The draft is never scored: these offsets feed only the draft compacts.
    bufs.draft_block_offsets_device = None
    bufs.draft_block_offsets_host = None
    bufs.draft_protected_tail_capacity = None
    if draft is not None:
        draft_layout = draft["layout"]
        draft_representatives = list(draft_layout["pool_representatives"])
        draft_anchor_pool = draft_layout["layer_pools"][draft_representatives[0]]
        # Construction-boundary invariant: the round shares one stream/event
        # contract, so the draft pools must live on the target device.
        if draft_anchor_pool.device != device:
            raise RuntimeError("TriAttention draft KV pools must share the target KV pool device")
        bufs.draft_block_offsets_host, bufs.draft_block_offsets_device = (
            _allocate_block_offset_staging(
                draft_anchor_pool,
                num_pools=int(draft_layout["manager"].num_pools),
                max_requests=max_requests,
                token_capacity=int(draft["page_table_token_capacity"]),
            )
        )
        bufs.draft_protected_tail_capacity = int(draft["protected_tail_capacity"])

    # ---- per-round metadata table: one H2D copy; move-offsets rows need the +1 column ----
    bufs.request_metadata_host = torch.empty(
        (6, max_requests + 1), dtype=torch.int32, device="cpu", pin_memory=prefer_pinned()
    )
    # numpy view over the pinned rows: per-round staging writes lists in place.
    bufs.request_metadata_host_np = bufs.request_metadata_host.numpy()
    bufs.identity_copy_indices_host = torch.arange(
        max_requests, dtype=torch.int32, device="cpu", pin_memory=prefer_pinned()
    )
    # Zero-filled: an unstaged cohort must gather the phase table's row 0.
    bufs.request_metadata_device = torch.zeros(
        (6, max_requests + 1), dtype=torch.int32, device=device
    )
    bufs.round_starts_device = bufs.request_metadata_device[0, :max_requests]
    bufs.valid_seq_lens_device = bufs.request_metadata_device[1, :max_requests]
    # Per-request pinned prompt lengths (per-request decode window starts).
    bufs.token_starts_device = bufs.request_metadata_device[2, :max_requests]
    dense_move_offsets_row = bufs.request_metadata_device[3]
    swa_move_offsets_row = bufs.request_metadata_device[4]
    draft_move_offsets_row = bufs.request_metadata_device[5]
    # SWA staging geometry, bound once (the compaction plans stay opaque):
    # the phase gather rebases each request's SWA destination base in place.
    bufs.swa_window = int(swa_window) if swa_layers else None
    bufs.swa_destination_bases = torch.empty_like(bufs.token_starts_device) if swa_layers else None
    bufs.swa_rebase_delta = keep_count - bufs.swa_window if swa_layers else 0
    bufs.mean_cos = torch.empty((max_requests, num_freqs), dtype=torch.float32, device=device)
    bufs.mean_sin = torch.empty_like(bufs.mean_cos)
    bufs.phase = phase
    bufs.phase_num_freqs = int(phase["omega"].numel())
    bufs.phase_f_block = triton.next_power_of_2(bufs.phase_num_freqs)

    # ---- score state: one fused group across all dense layers --------------
    p0 = layer_pools[dense_layers[0]]
    _, kv_factor, num_kv_heads, tokens_per_block, head_dim = p0.shape
    bufs.num_layers = len(dense_layers)
    bufs.num_q_heads = int(num_q_heads)
    bufs.num_kv_heads = int(num_kv_heads)
    bufs.num_freqs = int(num_freqs)
    bufs.tokens_per_block = int(tokens_per_block)
    _rep_of = {layer: layers[0] for layers in dense_groups for layer in layers}
    dense_layer_slots = [layer_pool_ids[_rep_of[layer]] for layer in dense_layers]
    seg_req_id = torch.arange(max_requests, dtype=torch.int32, device=device).repeat_interleave(
        bufs.num_layers
    )
    seg_layer_id = torch.tensor(list(dense_layers), dtype=torch.int32, device=device).repeat(
        max_requests
    )
    block_offsets = bufs.block_offsets_device
    slots_t = torch.tensor(dense_layer_slots, dtype=torch.int64, device=device)
    req_idx = seg_req_id.to(torch.int64)
    slot_idx = slots_t.repeat(max_requests)
    seg_page_off = slot_idx * block_offsets.stride(0) + req_idx * block_offsets.stride(1)

    max_segments = max_requests * bufs.num_layers
    # The score plane must stay 32-bit indexable (wraparound = silent wild read).
    if (PADDED_HEAD_COLUMNS - 1) * max_segments * seq_len >= 2**31:
        raise ValueError(
            "score bucket overflows the 32-bit score plane: "
            f"{(PADDED_HEAD_COLUMNS - 1) * max_segments * seq_len}"
        )
    # Persistent buffers: the compiled kernels capture their device pointers.
    bufs.padded_head_columns = PADDED_HEAD_COLUMNS
    bufs.score_scratch = torch.empty(
        bufs.num_kv_heads * PADDED_HEAD_COLUMNS * max_segments * seq_len,
        dtype=torch.float32,
        device=device,
    )
    # int32 is safe here: covered by the 2^31 score-plane audit above.
    seg_out_offset = (torch.arange(max_segments, dtype=torch.int64, device=device) * seq_len).to(
        torch.int32
    )
    bufs.gather_columns = torch.arange(decode_width, dtype=torch.int64, device=device).view(
        1, 1, 1, 1, -1
    )
    # Compile the mode's SM100 CuTe entries; no other score path, no fallback.
    union = eviction_mode == "union"
    # Persistent gather index (per-head modes): per round only the
    # token-start base is re-added in place; the expanded view is fixed.
    bufs.gather_index_base = None
    bufs.gather_index = None
    if not union:
        num_kv_heads_early = int(layer_pools[dense_layers[0]].shape[2])
        bufs.gather_index_base = torch.empty(
            (max_requests, 1, 1, 1, decode_width), dtype=torch.int64, device=device
        )
        bufs.gather_index = bufs.gather_index_base.expand(
            max_requests,
            len(dense_layers),
            num_kv_heads_early,
            num_q_heads // num_kv_heads_early,
            decode_width,
        )
    bufs.union_scores = None
    if union:
        # Bucket-wide rows; consumers mask by the per-request widths.
        bufs.union_scores = torch.empty((max_requests, seq_len), dtype=torch.float32, device=device)
    # THE score path (no fallback): construction failures raise the runner's
    # own dtype/shape/TMA error.
    bufs.runner = TriAttentionCuteScoreRunner(
        layer_pools=list(layer_pools),
        layer_indices=[int(layer) for layer in dense_layers],
        max_requests=max_requests,
        seq_len=seq_len,
        num_q_heads=bufs.num_q_heads,
        num_freqs=bufs.num_freqs,
        page_ids=block_offsets.view(-1),
        seg_page_off=seg_page_off,
        seg_req_id=seg_req_id,
        seg_layer_id=seg_layer_id,
        # Pointer capture of the staged metadata rows.
        valid_seq_lens=bufs.valid_seq_lens_device,
        seg_out_offset=seg_out_offset,
        token_starts=bufs.token_starts_device,
        q_real=q_real.view(-1),
        q_imag=q_imag.view(-1),
        mlr_coef=mlr_coef.view(-1),
        mean_cos=bufs.mean_cos,
        mean_sin=bufs.mean_sin,
        freq_scale_sq=freq_scale_sq,
        output=bufs.score_scratch,
        union_scores=bufs.union_scores,
        enable_partial_stats=union,
    )
    logger.info(
        f"TriAttention CuTe score enabled: {bufs.num_q_heads}q/{bufs.num_kv_heads}kv heads, "
        f"{bufs.num_freqs} freqs, {bufs.tokens_per_block}-token pages"
    )

    # ---- selection buffers (canonical row-major, one name per storage) -----
    bufs.valid_widths = torch.full((max_requests,), decode_width, dtype=torch.int32, device=device)
    if union:
        bufs.selection_rows_per_request = 1
        bufs.selection_scores_rows = torch.empty(
            (max_requests, decode_width), dtype=torch.float32, device=device
        )
        # One selection row per request: its length IS the staged valid width.
        bufs.selection_row_lengths = bufs.valid_widths
        # Padded rows still need in-range ordinals for the finalizer's gather.
        bufs.provisional_rows = torch.zeros(
            (max_requests, keep_count), dtype=torch.int32, device=device
        )
        # Kept decode ordinals only (prompt-length independent rows).
        bufs.kept_ordinal_rows = torch.empty(
            (max_requests, keep_count), dtype=torch.int32, device=device
        )
        bufs.score_output = None
    else:
        selection_rows = (
            bufs.num_kv_heads
            if eviction_mode == "per_head"
            else bufs.num_layers * bufs.num_kv_heads
        )
        # Both rectangles must stay 32-bit indexable (wraparound = wild reads).
        score_rect = max_requests * bufs.num_layers * bufs.num_q_heads * decode_width
        selection_rect = max_requests * selection_rows * max(decode_width, keep_count)
        if max(score_rect, selection_rect) >= 2**31:
            raise ValueError(
                f"per-head score rectangles overflow 32-bit indexing: "
                f"scores {score_rect}, selection {selection_rect}"
            )
        bufs.selection_rows_per_request = selection_rows
        # [request, layer, head, token] layout read by the reduce kernels.
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
        bufs.row_inv_std = torch.empty_like(bufs.row_mean)
        bufs.selection_scores_rows = torch.empty(
            (max_requests * selection_rows, decode_width), dtype=torch.float32, device=device
        )
        bufs.selection_row_lengths = torch.full(
            (max_requests * selection_rows,), decode_width, dtype=torch.int32, device=device
        )
        bufs.provisional_rows = torch.zeros(
            (max_requests * selection_rows, keep_count), dtype=torch.int32, device=device
        )
        bufs.kept_ordinal_rows = torch.empty(
            (max_requests * selection_rows, keep_count), dtype=torch.int32, device=device
        )

    # ---- compaction plans + decision-materialization prebinds ---------------
    per_layer = eviction_mode == "per_layer_perhead"
    draft_contract = None
    if draft is not None:
        draft_layout = draft["layout"]
        draft_contract = dict(
            layer_pools=draft_layout["layer_pools"],
            dense_layers=list(draft_layout["dense_layers"]),
            layer_group_representative=draft_layout["layer_group_representative"],
            layer_pool_ids=tuple(draft_layout["layer_pool_ids"]),
            kv_block_offsets=bufs.draft_block_offsets_device,
            dense_move_offsets=draft_move_offsets_row,
            protected_tail_capacity=int(draft["protected_tail_capacity"]),
        )
    # Opaque launch plans: only compact() interprets them.
    bufs.compaction_plan = init_compaction_buffers(
        target=dict(
            layer_pools=layer_pools,
            dense_layers=list(dense_layers),
            swa_layers=list(swa_layers),
            swa_window=swa_window,
            layer_group_representative=layer_group_representative,
            layer_pool_ids=layer_pool_ids,
            kv_block_offsets=bufs.block_offsets_device,
            token_starts=bufs.token_starts_device,
            swa_destination_bases=bufs.swa_destination_bases,
            # Per-round tails: the move offsets ride the staged metadata rows.
            dense_move_offsets=dense_move_offsets_row,
            swa_move_offsets=swa_move_offsets_row,
            per_layer_sources=per_layer,
            # The decision rows the plans pack into move sources.
            kept_ordinal_rows=bufs.kept_ordinal_rows,
            decision_rows=bufs.selection_rows_per_request,
            valid_seq_lens=bufs.valid_seq_lens_device,
        ),
        capacities=dict(
            max_requests=max_requests,
            keep_count=keep_count,
            protected_tail_capacity=int(protected_tail_capacity),
        ),
        draft=draft_contract,
    )
    # The decision side: the settle launch materializes the kept-ordinal rows.
    bufs.settle_args = (
        bufs.selection_scores_rows,
        bufs.selection_row_lengths,
        bufs.token_starts_device,
        bufs.provisional_rows,
        bufs.kept_ordinal_rows,
    )
    bufs.settle_kwargs = dict(
        WIDTH=decode_width,
        KEEP_COUNT=keep_count,
        SELECTION_ROWS=bufs.selection_rows_per_request,
    )

    # ---- round-ordering events ----------------------------------------------
    # Host staging (pinned metadata + snapshots) reuse fence.
    bufs.staging_reuse_event = torch.cuda.Event()
    bufs.staging_reuse_event.record(torch.cuda.current_stream(device))
    # Manager-stream H2D of the block-offset tables has completed.
    bufs.block_offsets_ready_event = torch.cuda.Event()
    # This cohort's compact is done: manager may resize/reuse pages.
    bufs.compaction_done_event = torch.cuda.Event()
    bufs.copy_pending = False
    return bufs


def _stage_block_offsets(
    bufs: SimpleNamespace,
    manager: KVCacheManagerV2,
    request_ids: List[int],
    host_block_offsets: torch.Tensor,
    device_block_offsets: torch.Tensor,
) -> None:
    """Gather the pinned snapshot before the async device copy: resize mutates
    the live host table. The round owner has already fenced host-staging reuse."""
    manager.index_mapper.gather_k_block_offsets(
        manager.host_kv_cache_block_offsets,
        host_block_offsets,
        request_ids,
        host_block_offsets.shape[-1],
    )
    manager._stream.wait_event(bufs.staging_reuse_event)
    copy_batch_block_offsets_to_device(
        host_block_offsets,
        device_block_offsets,
        bufs.identity_copy_indices_host[: len(request_ids)],
        manager.index_scales,
        manager.kv_offset,
        manager._stream.cuda_stream,
    )
    bufs.block_offsets_ready_event.record(manager._stream)
    torch.cuda.current_stream(device_block_offsets.device).wait_event(
        bufs.block_offsets_ready_event
    )


def _cohort_move_offsets(
    bufs: SimpleNamespace,
    prepared: Sequence[Dict[str, object]],
) -> Tuple[List[int], Optional[List[int]], Optional[List[int]]]:
    """Cumulative dense/SWA/draft move offsets for one prepared cohort (keep
    set plus protected tail per request; rows past the cohort repeat the final
    offset and contribute no moves)."""

    def padded_offsets(moves_per_request: List[int]) -> List[int]:
        offsets = [0]
        for moves in moves_per_request:
            offsets.append(offsets[-1] + moves)
        offsets.extend(offsets[-1:] * (bufs.max_requests - len(moves_per_request)))
        return offsets

    tails = [int(item["protected_tail"]) for item in prepared]
    dense = padded_offsets([bufs.keep_count + tail for tail in tails])
    swa = None
    if bufs.swa_window is not None:
        swa = padded_offsets([bufs.swa_window + tail for tail in tails])
    draft = None
    if bufs.draft_protected_tail_capacity is not None:
        draft = padded_offsets(
            [bufs.keep_count + bufs.draft_protected_tail_capacity] * len(prepared)
        )
    return dense, swa, draft


def settle_top_tokens(bufs: SimpleNamespace, request_count: int) -> None:
    """Pick the top-k and settle ties into the kept-ordinal decision rows
    (the compaction contract packs them into move sources)."""
    rows = request_count * bufs.selection_rows_per_request
    # The trailing 1 is next_n: decode scores one query token per request.
    torch.ops.trtllm.cute_dsl_indexer_topk_decode(
        bufs.selection_scores_rows[:rows],
        bufs.selection_row_lengths[:rows],
        bufs.provisional_rows[:rows],
        bufs.keep_count,
        1,
    )
    _settle_ties_kernel[(request_count, bufs.selection_rows_per_request)](
        *bufs.settle_args, **bufs.settle_kwargs
    )


def execute_eviction_round(
    bufs: SimpleNamespace,
    manager: KVCacheManagerV2,
    prepared: Sequence[Dict[str, object]],
    draft_manager: Optional[KVCacheManagerV2] = None,
) -> None:
    """Run one eviction round over the prepared cohort: stage the page-table
    snapshots and round metadata, then score, select, settle, and compact, and
    finally order the manager streams after this cohort's compact (every launch
    covers the full request capacity; padded rows carry zero lengths and stay
    inert)."""
    with nvtx_range_debug("triattention.page_table_stage", color="orange"):
        request_ids = [item["request_id"] for item in prepared]
        round_starts = [item["round_start"] for item in prepared]
        token_starts = [item["prompt_len"] for item in prepared]
        seq_lens = [item["seq_len"] for item in prepared]
        dense_move_offsets, swa_move_offsets, draft_move_offsets = _cohort_move_offsets(
            bufs, prepared
        )
        stream = torch.cuda.current_stream(bufs.block_offsets_device.device)
        # int32 gate before any buffer or device work: the in-place numpy writes below wrap silently.
        max_round_start = max(round_starts)
        rows = (
            (0, round_starts),
            (1, seq_lens),
            (2, token_starts),
            (3, dense_move_offsets),
            (4, swa_move_offsets),
            (5, draft_move_offsets),
        )
        for row, values in rows:
            if values is not None and not -0x80000000 <= min(values) <= max(values) <= 0x7FFFFFFF:
                raise ValueError(f"staged metadata row {row} exceeds the int32 range")
        # The one host-staging reuse fence: the previous cohort's async copies
        # must complete before the pinned metadata rows AND the pinned
        # target/draft block-offset snapshots are rewritten.
        if bufs.copy_pending and not bufs.staging_reuse_event.query():
            bufs.staging_reuse_event.synchronize()
        host_table = bufs.request_metadata_host_np
        for row, values in rows:
            if values is not None:
                host_table[row, : len(values)] = values
        # Zero lengths keep the score kernel and selection inert for padded rows.
        host_table[:3, len(prepared) :] = 0
        grow_mean_phase_table(bufs.phase, int(max_round_start) + 1)
        _stage_block_offsets(
            bufs,
            manager,
            request_ids,
            bufs.block_offsets_host,
            bufs.block_offsets_device,
        )
        if draft_manager is not None:
            _stage_block_offsets(
                bufs,
                draft_manager,
                request_ids,
                bufs.draft_block_offsets_host,
                bufs.draft_block_offsets_device,
            )
        try:
            bufs.request_metadata_device.copy_(bufs.request_metadata_host, non_blocking=True)
        finally:
            # Guards the pinned staging until the asynchronous copies complete.
            bufs.staging_reuse_event.record(stream)
            bufs.copy_pending = True
    request_count = len(prepared)
    union = bufs.eviction_mode == "union"
    try:
        with nvtx_range("triattention.score", color="blue"):
            # In-place refresh: the compiled score launches captured these pointers.
            _gather_mean_phase_kernel[(request_count,)](
                bufs.round_starts_device,
                bufs.phase["cos"],
                bufs.phase["sin"],
                bufs.phase["rows"],
                bufs.valid_seq_lens_device,
                bufs.token_starts_device,
                bufs.mean_cos,
                bufs.mean_sin,
                bufs.valid_widths,
                bufs.swa_destination_bases,
                bufs.swa_rebase_delta,
                NUM_FREQS=bufs.phase_num_freqs,
                F_BLOCK=bufs.phase_f_block,
                HAS_SWA=bufs.swa_destination_bases is not None,
                num_warps=1,
            )
            if union:
                # The runner's ctor bound the mean/union buffers; the launch
                # takes only the active cohort size.
                bufs.runner.launch_union_fusion(request_count)
                columns = min(bufs.union_scores.shape[1], bufs.selection_scores_rows.shape[1])
                bufs.selection_scores_rows[:request_count, :columns].copy_(
                    bufs.union_scores[:request_count, :columns]
                )
            else:
                bufs.runner.launch(request_count)
                # Gather each decode window into the [request, layer, head, token] layout of the reduces.
                group_size = bufs.num_q_heads // bufs.num_kv_heads
                num_segments = request_count * bufs.num_layers
                pad = bufs.padded_head_columns
                source = (
                    bufs.score_scratch[
                        : bufs.num_kv_heads * pad * num_segments * bufs.bucket_seq_len
                    ]
                    .view(
                        bufs.num_kv_heads, pad, request_count, bufs.num_layers, bufs.bucket_seq_len
                    )[:, :group_size]
                    .permute(2, 3, 0, 1, 4)
                )
                torch.add(
                    bufs.token_starts_device[:request_count].view(-1, 1, 1, 1, 1),
                    bufs.gather_columns,
                    out=bufs.gather_index_base[:request_count],
                )
                bufs.gather_index_base[:request_count].clamp_(max=bufs.bucket_seq_len - 1)
                columns = bufs.gather_index[:request_count]
                torch.gather(
                    source,
                    4,
                    columns,
                    out=bufs.score_output[:request_count].view(
                        request_count,
                        bufs.num_layers,
                        bufs.num_kv_heads,
                        group_size,
                        bufs.decode_width,
                    ),
                )
        with nvtx_range("triattention.select", color="yellow"):
            if not union:
                prepare_per_head_scores(
                    bufs.score_output[:request_count],
                    bufs.valid_widths,
                    bufs.row_mean,
                    bufs.row_inv_std,
                    bufs.selection_scores_rows,
                    bufs.selection_row_lengths,
                    per_layer=bufs.eviction_mode == "per_layer_perhead",
                    normalize_scores=bufs.normalize_scores,
                )
            settle_top_tokens(bufs, request_count)
        with nvtx_range("triattention.compact", color="purple"):
            compact(bufs.compaction_plan, request_count)
    finally:
        # Order V2 page-table reuse and resize after this cohort's compact.
        bufs.compaction_done_event.record(stream)
        manager._stream.wait_event(bufs.compaction_done_event)
        if draft_manager is not None:
            draft_manager._stream.wait_event(bufs.compaction_done_event)


class TriAttention(BaseKVCacheCompressionManager):
    """Periodic physical KV eviction driven by trigonometric importance scoring."""

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
    ):
        super().__init__(kv_cache_manager, draft_kv_cache_manager)
        # budget/beta positivity and the eviction_mode literal are validated at
        # the config boundary (TriAttentionKvCacheCompressionConfig).
        self.budget = budget
        self.beta = beta
        self.eviction_mode = eviction_mode
        self.normalize_scores = bool(normalize_scores)
        if self.eviction_mode == "union" and not self.normalize_scores:
            raise ValueError(
                "TriAttention union eviction requires normalize_scores=True: "
                "the fused union pipeline always z-normalizes score rows"
            )
        # Hard-coded semantics: the prompt is always pinned and the budget
        # counts decode tokens only (physical KV reclaim requires both).
        # Calibration is the official TriAttention .pt; TRT-LLM does not
        # compute calibration. The config boundary requires both paths.
        self.model_path = model_path
        self.calibration_path = calibration_path
        self.calibration: Optional[Dict[str, torch.Tensor]] = None
        self._calibrated = False
        self._freq_scale_sq: Optional[torch.Tensor] = None

        # Mean-phase table dict, shared by reference with every buffer namespace.
        self._phase: Optional[Dict[str, object]] = None

        # Per-request {generation_steps, evicted_tokens}.
        self._request_states: Dict[int, Dict[str, object]] = {}
        # In-flight overlap batch reference; membership resolves lazily.
        self._prepared_generation_batch: Optional[object] = None
        self._prepared_generation_ids: Optional[set] = None
        # Manager-lifetime capability gates: everything read there is fixed at
        # construction, so validation runs once here.
        self._validate_v2_compatibility()
        # Manager-lifetime constants (V2 fixes every input at construction):
        # protected tails are num_extra + reserved draft width + 1 sampled token.
        self._protected_tail_capacity = (
            int(kv_cache_manager.num_extra_kv_tokens)
            + int(kv_cache_manager._kv_reserve_draft_tokens)
            + 1
        )
        self._draft_protected_tail_capacity: Optional[int] = None
        if draft_kv_cache_manager is not None:
            self._draft_protected_tail_capacity = (
                int(draft_kv_cache_manager.num_extra_kv_tokens)
                + int(draft_kv_cache_manager._kv_reserve_draft_tokens)
                + 1
            )
        self._generation_growth = 1 + int(kv_cache_manager._kv_reserve_draft_tokens)
        # Built once at the first eviction, reused for the manager's lifetime.
        self._buffers: Optional[SimpleNamespace] = None
        self._local_to_global_layers_cache: Optional[List[int]] = None
        self._attention_layer_partition_cache: Optional[
            Tuple[List[int], List[int], Optional[int]]
        ] = None
        self._runtime_kv_layout_cache: Optional[Dict[str, object]] = None
        self._draft_runtime_kv_layout_cache: Optional[Dict[str, object]] = None

    def on_request_init(self, request: "LlmRequest", **kwargs) -> None:
        """Track the request and resolve the official calibration on first use."""
        request_id = request.py_request_id
        if request_id not in self._request_states:
            self._validate_request_capacity(request)
            self._attention_layer_partition()
            self._request_states[request_id] = {
                "generation_steps": 0,
                "evicted_tokens": 0,
            }
        self._ensure_calibrated()

    def _validate_request_capacity(self, request: "LlmRequest") -> None:
        manager = self.kv_cache_manager
        speculative_overshoot = int(manager.max_draft_len)
        first_eviction_decode_length = (
            self.budget // self.beta + 1
        ) * self.beta + speculative_overshoot
        decode_capacity = min(int(request.py_max_new_tokens), first_eviction_decode_length)
        confirmed_capacity = int(request.py_prompt_len) + decode_capacity
        protected_tail_capacity = self._protected_tail_capacity
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
        draft_protected_tail = self._draft_protected_tail_capacity
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

    def _ensure_calibrated(self) -> None:
        if self._calibrated:
            return
        self.calibration = self._resolve_calibration()
        self._freq_scale_sq = self.calibration["freq_scale_sq"].to(dtype=torch.float32)
        # Pre-split query stats + MLR coefficient, shapes [L, H, F].
        _Eq = self.calibration["E_q"]
        self._triattn_q_real = _Eq.real.to(torch.float32).contiguous()
        self._triattn_q_imag = _Eq.imag.to(torch.float32).contiguous()
        self._triattn_mlr_coef = (
            self.calibration["E_q_norm"].to(torch.float32) - _Eq.abs().to(torch.float32)
        ).contiguous()
        self._calibrated = True

    def _validate_v2_compatibility(self) -> None:
        # The base manager already enforces KVCacheManagerV2 target/draft types,
        # and V2 construction guarantees beam width one.
        manager = self.kv_cache_manager
        if manager.kv_factor != 2:
            raise ValueError(
                "TriAttention requires a standard key/value KV cache; "
                "MLA/SELFKONLY caches are not supported"
            )
        if manager.mapping.enable_attention_dp:
            raise ValueError("TriAttention does not support attention DP")
        if manager.is_disagg:
            raise ValueError("TriAttention does not support disaggregated serving")
        if manager.enable_swa_scratch_reuse:
            raise RuntimeError("TriAttention does not support V2 SWA scratch page-table remapping")
        # Speculative feature gates run in the factory; the draft cache itself
        # is validated here (V2 already forces scratch reuse off for drafts).
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

    def on_generation_step_end(self, scheduled_batch: "ScheduledRequests", **kwargs) -> None:
        """Compact after native KV-cache updates have finalized this iteration
        (must run after KVCacheManagerV2 so capacity reflects the written token and any rewind)."""
        with nvtx_range_debug("triattention.generation_step_end", color="blue"):
            self._periodic_evict(scheduled_batch)

    def on_generation_step_begin(self, scheduled_batch: "ScheduledRequests", **kwargs) -> None:
        """Snapshot the prepared batch; mutation remains in final update."""
        self._prepared_generation_batch = scheduled_batch
        self._prepared_generation_ids = None

    def _inflight_generation_growth(
        self, scheduled_batch: "ScheduledRequests", request_id: int
    ) -> int:
        prepared = self._prepared_generation_batch
        if prepared is None or scheduled_batch is prepared:
            return 0
        member_ids = self._prepared_generation_ids
        if member_ids is None:
            member_ids = {request.py_request_id for request in prepared.generation_requests}
            self._prepared_generation_ids = member_ids
        if request_id not in member_ids:
            return 0
        return self._generation_growth

    def _periodic_evict(
        self,
        scheduled_batch: "ScheduledRequests",
    ) -> None:
        gen_requests = scheduled_batch.generation_requests
        if not gen_requests:
            return
        mgr = self.kv_cache_manager
        resolved_requests = []
        for request in gen_requests:
            if request.is_dummy or request.state in _SKIP_REQUEST_STATES:
                continue
            request_id = request.py_request_id
            kv_cache = mgr.kv_cache_map.get(request_id)
            if kv_cache is None:
                continue
            if not kv_cache.is_active:
                # Overlap scheduling may suspend a cache mid-flight; defer this
                # request (pre-launch) instead of failing the whole batch.
                continue
            resolved_requests.append((request, request_id, kv_cache))
        if not resolved_requests:
            return
        prepared: List[Dict[str, object]] = []

        # The resolved cache objects thread all the way to resize.
        with nvtx_range("triattention.metadata", color="cyan"):
            for request, request_id, kv_cache in resolved_requests:
                # Cadence gate first; capacity math and consistency raises run in the due branch.
                request_state = self._request_states[request_id]
                previous_step = request_state["generation_steps"]
                step = previous_step + 1 + int(request.py_num_accepted_draft_tokens)
                request_state["generation_steps"] = step
                if previous_step // self.beta >= step // self.beta:
                    continue
                raw_capacity = int(kv_cache.capacity)
                # Speculative reserve + in-flight overlap growth: contiguous tail moved byte-for-byte.
                protected_tail = int(mgr.num_extra_kv_tokens) + self._inflight_generation_growth(
                    scheduled_batch, request_id
                )
                seq_len = raw_capacity - protected_tail
                if seq_len < kv_cache.history_length:
                    raise RuntimeError(
                        f"Request {request_id} KV length {seq_len} is below finalized "
                        f"history {kv_cache.history_length}"
                    )
                expected_keep_count = self._minimum_evictable_length(request, seq_len)
                if seq_len <= expected_keep_count:
                    continue
                draft_kv_cache = None
                if self.draft_kv_cache_manager is not None:
                    draft_kv_cache = self.draft_kv_cache_manager.kv_cache_map.get(request_id)
                    if draft_kv_cache is None:
                        # A missing draft cache is a wiring/lifecycle bug.
                        raise RuntimeError(
                            "TriAttention cannot co-compress a missing draft KV "
                            f"cache for request {request_id}"
                        )
                    if not draft_kv_cache.is_active:
                        # Target and draft defer together (pre-launch).
                        continue
                prepared.append(
                    {
                        "request": request,
                        "request_id": request_id,
                        "kv_cache": kv_cache,
                        "draft_kv_cache": draft_kv_cache,
                        "seq_len": int(seq_len),
                        # Uncompressed logical position (prefix + evicted).
                        "round_start": int(seq_len + request_state["evicted_tokens"]),
                        "prompt_len": min(int(request.py_prompt_len), int(seq_len)),
                        "expected_keep_count": expected_keep_count,
                        "protected_tail": protected_tail,
                    }
                )

        if not prepared:
            return
        # Ungated NVTX: the due count in the message shows each round's size.
        with nvtx_range(
            f"triattention.evict_request_group reqs={len(prepared)}",
            color="purple",
        ):
            compacted = self._evict_requests(prepared)
        self._resize_compacted_requests(compacted)

    def _resize_compacted_requests(self, prepared) -> None:
        if not prepared:
            return
        with nvtx_range("triattention.resize", color="red"):
            with nvtx_range_debug("triattention.v2_resize", color="red"):
                for item in prepared:
                    kv_cache = item["kv_cache"]
                    if not kv_cache.is_active:
                        # Bytes already moved: skipping the ledger resize here
                        # would leave silent corruption. The compact-to-resize
                        # window is owned by this hook; a suspension inside it
                        # breaks the lifecycle contract.
                        raise RuntimeError(
                            f"Request {item['request_id']} target KV cache was "
                            "suspended between compact and resize"
                        )
                    resized_capacity = item["expected_keep_count"] + item["protected_tail"]
                    if not kv_cache.resize(resized_capacity, None):
                        raise RuntimeError(
                            f"Failed to resize compacted KV cache for request "
                            f"{item['request_id']} to {resized_capacity} tokens"
                        )
                if self.draft_kv_cache_manager is not None:
                    # Same kept set: the draft shrinks to the same retained length plus its own tail.
                    draft_protected_tail = self._draft_protected_tail_capacity
                    for item in prepared:
                        draft_kv_cache = item["draft_kv_cache"]
                        if not draft_kv_cache.is_active:
                            raise RuntimeError(
                                f"Request {item['request_id']} draft KV cache was "
                                "suspended between compact and resize"
                            )
                        draft_capacity = item["expected_keep_count"] + draft_protected_tail
                        if not draft_kv_cache.resize(draft_capacity, None):
                            raise RuntimeError(
                                "Failed to resize co-compressed draft KV cache for "
                                f"request {item['request_id']} to {draft_capacity} tokens"
                            )

    def _minimum_evictable_length(self, request: "LlmRequest", seq_len: int) -> int:
        """Return the largest cache length for which selection is an identity."""
        prompt_len = min(int(request.py_prompt_len), seq_len)
        return prompt_len + self.budget

    def _local_score_calibration(
        self,
        global_layers: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_layers = len(global_layers)
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

    def on_request_finish(self, request: "LlmRequest", **kwargs) -> None:
        """Drop this request's eviction state; the buffers stay resident."""
        self._request_states.pop(request.py_request_id, None)

    # ---- helpers (eviction / scoring / V2 cache access / calibration) ----

    def _local_to_global_layers(self) -> List[int]:
        cached = self._local_to_global_layers_cache
        if cached is None:
            cached = [int(layer) for layer in self.kv_cache_manager.pp_layers]
            self._local_to_global_layers_cache = cached
        return cached

    @staticmethod
    def _has_sliding_window_signal(config: Dict[str, object]) -> bool:
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

    def _attention_layer_partition(self) -> Tuple[List[int], List[int], Optional[int]]:
        """SWA layers here are stored at full length; the window applies only in the kernel."""
        cached = self._attention_layer_partition_cache
        if cached is not None:
            return cached

        model_path = self.model_path
        global_layers = self._local_to_global_layers()
        num_layers = len(global_layers)

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

    def _runtime_kv_layout(self) -> Dict[str, object]:
        # The manager identity and layer count are manager-lifetime owner
        # contracts; only the pool page counts are polled (stale-pointer
        # safety until V2 exposes a layout epoch).
        manager = self.kv_cache_manager
        cached = self._runtime_kv_layout_cache
        if cached is not None:
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

        global_layers = self._local_to_global_layers()
        dense_layers, swa_layers, swa_window = self._attention_layer_partition()
        if not dense_layers:
            raise ValueError("TriAttention requires at least one full-attention layer")
        layout = self._build_runtime_kv_layout(
            manager,
            global_layers,
            dense_layers=dense_layers,
            swa_layers=swa_layers,
            swa_window=swa_window,
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
        what: str,
    ) -> Dict[str, object]:
        maybe_layer_pools = [manager.get_buffers(layer, kv_layout="HND") for layer in global_layers]
        if any(pool is None for pool in maybe_layer_pools):
            missing = [
                layer for layer, pool in zip(global_layers, maybe_layer_pools) if pool is None
            ]
            raise RuntimeError(f"Missing {what}KV pools for attention layers {missing}")
        layer_pools = [pool for pool in maybe_layer_pools if pool is not None]
        # Canonical pool IDs, resolved once; every grouping derives from them.
        layer_pool_ids = self._page_table_pool_ids(manager, global_layers)
        all_storage_groups: Dict[int, List[int]] = {}
        for layer, pool_id in enumerate(layer_pool_ids):
            all_storage_groups.setdefault(pool_id, []).append(layer)
        # Scored/compacted groups cover the dense layers only; SWA layers
        # stage and compact as their own representatives.
        storage_groups: Dict[int, List[int]] = {}
        for layer in dense_layers:
            storage_groups.setdefault(layer_pool_ids[layer], []).append(layer)
        layer_group_representative = {
            layer: layers[0] for layers in storage_groups.values() for layer in layers
        }
        pool_representatives = tuple(layers[0] for layers in all_storage_groups.values())
        return dict(
            manager=manager,
            global_layers=global_layers,
            layer_pools=layer_pools,
            dense_layers=dense_layers,
            swa_layers=swa_layers,
            swa_window=swa_window,
            storage_groups=storage_groups,
            layer_group_representative=layer_group_representative,
            layer_pool_ids=layer_pool_ids,
            pool_representatives=pool_representatives,
            pool_page_counts=tuple(
                int(layer_pools[layer].shape[0]) for layer in pool_representatives
            ),
        )

    def _draft_runtime_kv_layout(self) -> Dict[str, object]:
        # Production callers gate on ``draft_kv_cache_manager is not None``.
        manager = self.draft_kv_cache_manager
        cached = self._draft_runtime_kv_layout_cache
        if cached is not None:
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

    def _buffers_for(
        self,
        layout: Dict[str, object],
        prepared: Sequence[Dict[str, object]],
    ) -> SimpleNamespace:
        # Empty cohorts never reach here: _periodic_evict no-ops pre-launch.
        needed_width = max(item["seq_len"] - item["prompt_len"] for item in prepared)
        needed_page_tokens = max(item["seq_len"] + item["protected_tail"] for item in prepared)
        needed_requests = len(prepared)
        if self.draft_kv_cache_manager is not None:
            # The cached layout lookup enforces draft V2 pool page-count
            # stability every round, exactly like the target's lookup.
            self._draft_runtime_kv_layout()
        bufs = self._buffers
        if bufs is not None:
            if (
                needed_width <= bufs.decode_width
                and needed_page_tokens <= bufs.page_table_token_capacity
                and needed_requests <= bufs.max_requests
            ):
                return bufs
            # This round outgrew the buffers: rebuild.
            self._buffers = None

        mgr = self.kv_cache_manager
        tail_capacity = self._protected_tail_capacity
        request_capacity = max(needed_requests, int(mgr.max_batch_size))
        decode_width = max(
            needed_width,
            self.budget + 2 * self.beta + int(mgr.max_total_draft_tokens or 0),
        )
        # Bucket sized by the presented cohorts, NOT max_seq_len (a floor there breaks 32-bit indexing).
        seq_capacity = max(int(needed_page_tokens), 1024)
        seq_capacity = 1 << (seq_capacity - 1).bit_length()
        seq_capacity = min(seq_capacity, max(int(mgr.max_seq_len), int(needed_page_tokens)))
        # The bucket capacity must be tile-aligned (mis-tiling stripes the
        # score scratch silently); the ceiling division constructs that fact.
        score_tile_tokens = max(64, int(mgr.tokens_per_block))
        seq_capacity = -(-seq_capacity // score_tile_tokens) * score_tile_tokens
        page_table_token_capacity = max(needed_page_tokens, seq_capacity + tail_capacity)

        draft = None
        if self.draft_kv_cache_manager is not None:
            draft_tail_capacity = self._draft_protected_tail_capacity
            draft = dict(
                layout=self._draft_runtime_kv_layout(),
                protected_tail_capacity=draft_tail_capacity,
                page_table_token_capacity=seq_capacity + draft_tail_capacity,
            )

        first_pool = layout["layer_pools"][layout["dense_layers"][0]]
        if self._phase is None:
            # Upstream geometric offsets [1, 2, 4, ... <= max]: the table
            # builder consumes them as host floats only (no device copy).
            self._phase = {
                "omega": self.calibration["omega"]
                .to(device=first_pool.device, dtype=torch.float32)
                .contiguous(),
                "offset_values": [float(1 << i) for i in range(_OFFSET_MAX_LENGTH.bit_length())],
                "cos": None,
                "sin": None,
                "rows": 0,
            }
            grow_mean_phase_table(self._phase, max(int(seq_capacity), 1))
        q_real, q_imag, mlr_coef = self._local_score_calibration(layout["global_layers"])
        bufs = init_eviction_buffers(
            eviction_mode=self.eviction_mode,
            layout=layout,
            calibration=dict(
                q_real=q_real,
                q_imag=q_imag,
                mlr_coef=mlr_coef,
                freq_scale_sq=self._freq_scale_sq,
            ),
            phase=self._phase,
            capacities=dict(
                max_requests=request_capacity,
                bucket_seq_len=seq_capacity,
                decode_width=decode_width,
                page_table_token_capacity=page_table_token_capacity,
                keep_count=self.budget,
                protected_tail_capacity=tail_capacity,
            ),
            draft=draft,
            normalize_scores=self.normalize_scores,
        )
        self._buffers = bufs
        return bufs

    @staticmethod
    def _page_table_pool_ids(
        manager: KVCacheManagerV2,
        global_layers: List[int],
    ) -> Tuple[int, ...]:
        """Canonical local layer -> V2 pool id tuple (the staged plane slots).

        V2 owns the mapping; its own lookup errors are the precise ones."""
        layer_offsets = manager.layer_offsets
        layer_to_pool = manager.layer_to_pool_mapping_dict
        return tuple(
            int(layer_to_pool[layer_offsets[global_layer]]) for global_layer in global_layers
        )

    def _evict_requests(
        self,
        prepared: List[Dict[str, object]],
    ) -> List[Dict[str, object]]:
        with nvtx_range_debug("triattention.resolve_layout", color="blue"):
            layout = self._runtime_kv_layout()
        with nvtx_range_debug("triattention.staging_lookup", color="blue"):
            # Retained spans always cover the model window (construction rejects budget < window).
            bufs = self._buffers_for(layout, prepared)
        execute_eviction_round(
            bufs,
            self.kv_cache_manager,
            prepared,
            self.draft_kv_cache_manager,
        )
        for item in prepared:
            # Identity cohorts were filtered pre-launch (_periodic_evict).
            evicted = item["seq_len"] - item["expected_keep_count"]
            request_state = self._request_states[item["request_id"]]
            request_state["evicted_tokens"] += evicted
            # The manager's only channel to the runtime (feeds num_cached_tokens_per_seq).
            item["request"].py_num_compressed_tokens = request_state["evicted_tokens"]
        return prepared

    # ---- helpers: calibration loading ----

    def _resolve_calibration(self) -> Dict[str, torch.Tensor]:
        """Load the user-supplied calibration .pt and return our runtime schema.

        TriAttention does NOT compute calibration -- the user calibrates with the
        official tool (github.com/WeianMao/triattention) and passes that file via
        ``calibration_path``; we only run inference. Both the official R-KV layout
        (``{metadata, stats{"layerLL_headHH": {q_mean_real, q_mean_imag,
        q_abs_mean}}}``) and our already-converted flat layout are accepted -- the
        official one is converted here. Calibration resolves lazily on the
        first request (``on_request_init``), not at manager construction, and
        stays on CPU: runtime construction moves it to the pool device once."""
        raw = torch.load(self.calibration_path, map_location="cpu", weights_only=False)
        if isinstance(raw, dict) and _REQUIRED_CALIBRATION_KEYS <= set(raw):
            return raw
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
        # CPU schema: runtime construction moves tensors to the pool device once.
        calib = {
            "E_q": E_q,
            "E_q_norm": E_q_norm,
            "omega": omega,
            "freq_scale_sq": freq_scale_sq,
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
            # transformers>=5.5 no longer keys "default" in ROPE_INIT_FUNCTIONS: use the formula.
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
