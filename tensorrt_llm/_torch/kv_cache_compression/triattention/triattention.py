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

from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2, Role
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm._torch.pyexecutor.resource_manager import BaseKVCacheCompressionManager
from tensorrt_llm._utils import nvtx_range, nvtx_range_debug, prefer_pinned
from tensorrt_llm.bindings.internal.batch_manager.kv_cache_manager_v2_utils import (
    copy_batch_block_offsets_to_device,
)
from tensorrt_llm.logger import logger
from tensorrt_llm.runtime.kv_cache_manager_v2 import AttentionLayerConfig

from ..compaction import init_compaction_buffers
from .triattention_kernels import (
    SETTLE_PACK_BLOCK,
    SETTLE_PACK_NUM_WARPS,
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

# Generation requests skipped by every eviction step.
_SKIP_REQUEST_STATES = (
    LlmRequestState.GENERATION_COMPLETE,
    LlmRequestState.CONTEXT_INIT,
)

# Upper bound of the geometric integration offset ladder [1, 2, 4, ...].
_OFFSET_MAX_LENGTH = 65536


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
    num_page_table_slots: int,
    token_capacity: int,
    max_requests: int,
    device: torch.device,
) -> Tuple[Dict[int, int], int, torch.Tensor, torch.Tensor]:
    """Allocate one staged block-offset plane (host pinned + device).

    The ``("pool", id)`` page-table keys are the snapshot slot numbering.
    """
    representative_slots = {
        representative: int(key[1])
        for representative, key in zip(page_representatives, page_table_keys)
    }
    tokens_per_block = int(layer_pools[page_representatives[0]].shape[3])
    page_count = (token_capacity + tokens_per_block - 1) // tokens_per_block
    copy_block_count = (page_count + 3) // 4 * 4
    plane_shape = (num_page_table_slots, max_requests, 2, copy_block_count)
    host = torch.empty(plane_shape, dtype=torch.int32, device="cpu", pin_memory=prefer_pinned())
    dev = torch.empty(plane_shape, dtype=torch.int32, device=device)
    return representative_slots, copy_block_count, host, dev


def attach_compaction_bundle(bufs: SimpleNamespace, compaction: Dict[str, object]) -> None:
    """Flatten one compaction bundle into the prebound launch fields the round fires."""
    bufs.compaction_families = compaction["families"]
    bufs.settle_pack_tensors = compaction["settle_pack_tensors"]
    bufs.settle_pack_shape = compaction["settle_pack_shape"]
    bufs.draft_pack = compaction["draft_pack"]
    bufs.swa_destination_bases = compaction["swa_destination_bases"]
    bufs.swa_rebase_delta = compaction["swa_rebase_delta"]

    def group_args(family):
        return tuple(
            (
                group["pools"],
                group["pool_pointers"],
                group["page_table"],
                family["source"],
                family["offsets"],
                family["destination_bases"],
                group["source_layer_indices"],
            )
            for group in family["groups"]
        )

    target_args: List[tuple] = []
    draft_args: Tuple[tuple, ...] = ()
    for family in compaction["families"]:
        if family["name"] == "draft":
            # Kept separate: the draft pack launch must precede the draft moves.
            draft_args = group_args(family)
        else:
            target_args.extend(group_args(family))
    bufs.compact_launch_args = tuple(target_args)
    bufs.draft_compact_launch_args = draft_args
    draft_pack = compaction["draft_pack"]
    if draft_pack is None:
        bufs.draft_pack_args = None
        bufs.draft_pack_kwargs = None
        return
    # Pack-only launch (HAS_SETTLE=False): settle-side pointers are None.
    bufs.draft_pack_args = (
        None,
        None,
        None,
        None,
        bufs.keep,
        bufs.valid_seq_lens_device,
        draft_pack["offsets"],
        draft_pack["indices"],
        None,
        None,
    )
    bufs.draft_pack_kwargs = dict(
        WIDTH=bufs.keep_count,
        KEEP_COUNT=bufs.keep_count,
        SELECTION_ROWS=1,
        DENSE_TOTAL=draft_pack["dense_total"],
        SWA_TOTAL=0,
        MOVE_CAPACITY=draft_pack["move_capacity"],
        NUM_KV_HEADS=draft_pack["num_kv_heads"],
        SWA_WINDOW=0,
        UNION=True,
        PER_LAYER=False,
        HAS_SWA=False,
        HAS_SETTLE=False,
        BLOCK=SETTLE_PACK_BLOCK,
        num_warps=SETTLE_PACK_NUM_WARPS,
    )


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
    page_table_keys: List[object],
    num_page_table_slots: int,
    decode_width: int,
    page_table_token_capacity: int,
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
    """Build the one namespace of buffers, compiled launches, and compaction data.

    Runs once per geometry, outside CUDA graph capture. The compiled kernels
    capture raw pool addresses, so the scored pools must stay alive and stay put.
    """
    from .triattention_cute_score_fused import N as PADDED_HEAD_COLUMNS
    from .triattention_cute_score_fused import TriAttentionCuteScoreRunner

    device = layer_pools[page_representatives[0]].device
    max_requests = int(max_requests)
    seq_len = int(seq_len)
    page_table_token_capacity = int(page_table_token_capacity)
    decode_width = int(decode_width)
    keep_count = int(keep_count)

    q_real, q_imag, mlr_coef, freq_scale_sq, offsets, omega = (
        tensor.to(device=device, dtype=torch.float32).contiguous()
        for tensor in (q_real, q_imag, mlr_coef, freq_scale_sq, offsets, omega)
    )

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
    )
    # The draft is never scored: these offsets feed only the draft compacts.
    bufs.draft_block_offsets_device = None
    bufs._draft_bulk_offsets_src = None
    bufs.draft_copy_block_count = 0
    draft_page_slots: Dict[int, int] = {}
    if draft_layer_pools is not None:
        (
            draft_page_slots,
            bufs.draft_copy_block_count,
            bufs._draft_bulk_offsets_src,
            bufs.draft_block_offsets_device,
        ) = _allocate_page_table_plane(
            draft_layer_pools,
            draft_page_representatives,
            draft_page_table_keys,
            int(draft_num_page_table_slots),
            int(draft_page_table_token_capacity),
            max_requests,
            device,
        )

    # ---- per-round metadata table: one host-to-device copy per round -------
    # Rows: logical position, valid length, prompt length, then one
    # move-offsets row per family (offsets rows need the +1 column).
    bufs.request_metadata_host = torch.empty(
        (6, max_requests + 1), dtype=torch.int32, device="cpu", pin_memory=prefer_pinned()
    )
    # numpy view over the pinned rows: per-round staging writes lists in place.
    bufs.request_metadata_host_np = bufs.request_metadata_host.numpy()
    bufs._bulk_copy_idx_src = torch.arange(
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
    bufs.mean_cos = torch.empty((max_requests, num_freqs), dtype=torch.float32, device=device)
    bufs.mean_sin = torch.empty_like(bufs.mean_cos)
    # ``phase=None`` is the one documented test seam (private table built here).
    if phase is None:
        phase = build_mean_phase_table(offsets, omega, initial_rows=seq_len)
    bufs.phase = phase

    # ---- score state: one fused group across all dense layers --------------
    p0 = layer_pools[dense_layers[0]]
    _, kv_factor, num_kv_heads, tokens_per_block, head_dim = p0.shape
    bufs.num_layers = len(dense_layers)
    bufs.num_q_heads = int(num_q_heads)
    bufs.num_kv_heads = int(num_kv_heads)
    bufs.num_freqs = int(num_freqs)
    bufs.tokens_per_block = int(tokens_per_block)
    # Segments index calibration by absolute layer id on device, where it
    # cannot be range-checked: validate the extent here.
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
    req_idx = bufs.seg_req.to(torch.int64)
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
    bufs.cute_scratch = torch.empty(
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
    bufs.union_rows = None
    if union:
        # Bucket-wide rows; consumers mask by the per-request widths.
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
    bufs.valid_widths = torch.full((max_requests,), decode_width, dtype=torch.int32, device=device)
    bufs.prompt_offsets = bufs.token_starts_device
    if union:
        bufs.selection_rows_per_request = 1
        bufs.row_prompt_offsets = bufs.prompt_offsets
        bufs.combined = torch.empty(
            (max_requests, decode_width), dtype=torch.float32, device=device
        )
        bufs.final_indices = torch.empty(
            (max_requests, keep_count), dtype=torch.int32, device=device
        )
        # Kept decode ordinals only (prompt-length independent rows).
        bufs.keep = torch.empty((max_requests, keep_count), dtype=torch.int32, device=device)
        # Row-major views consumed by the top-k settle launch.
        bufs.selection_scores_rows = bufs.combined
        bufs.selection_row_lengths = bufs.valid_widths
        bufs.provisional_rows = bufs.final_indices
        bufs.keep_rows = bufs.keep
        # Padded rows still need in-range ordinals for the finalizer's gather.
        bufs.final_indices.zero_()
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
        bufs.row_prompt_offsets = torch.zeros(
            (max_requests * selection_rows,), dtype=torch.int32, device=device
        )
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
            draft_page_table_slots=draft_page_slots,
            draft_move_offsets=draft_move_offsets_row,
        )
    compaction = init_compaction_buffers(
        union=union,
        per_layer=eviction_mode == "per_layer_perhead",
        layer_pools=layer_pools,
        dense_layers=list(dense_layers),
        swa_layers=list(swa_layers),
        layer_group_representative=layer_group_representative,
        valid_sequence_lengths=bufs.valid_seq_lens_device,
        kv_block_offsets=bufs.block_offsets_device,
        page_table_slots=bufs.representative_slots,
        request_count=max_requests,
        prompt_offsets=bufs.token_starts_device,
        decode_keep_count=keep_count,
        swa_window=swa_window,
        layer_pool_keys=list(layer_pool_keys),
        protected_tail_capacity=int(protected_tail_capacity),
        # Per-round tails: the move offsets ride the staged metadata rows.
        dense_move_offsets=dense_move_offsets_row,
        swa_move_offsets=swa_move_offsets_row,
        **draft_kwargs,
    )
    attach_compaction_bundle(bufs, compaction)

    # ---- round-ordering events ----------------------------------------------
    bufs.copy_done = torch.cuda.Event()
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

    Gathers an immutable pinned snapshot of the beam-0 K block offsets before
    the asynchronous device copy (resize later mutates the live host table).
    """
    if bufs.copy_pending and not bufs.copy_done.query():
        bufs.copy_done.synchronize()
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
    seq_lens: List[int],
    draft_manager: Optional[KVCacheManagerV2] = None,
    dense_move_offsets: Optional[List[int]] = None,
    swa_move_offsets: Optional[List[int]] = None,
    draft_move_offsets: Optional[List[int]] = None,
) -> None:
    """Copy one eviction cohort into the reusable device buffers.

    ``token_starts`` carries each request's pinned prompt length (per-request
    decode window start), so one cohort may mix prompt lengths.
    """
    request_count = len(request_ids)
    stream = torch.cuda.current_stream(bufs.device)
    if bufs.page_tables_active:
        raise RuntimeError("previous page-table cohort is still active")
    # int32 gate first (before buffers or device work): the in-place numpy
    # writes below can wrap silently, and round starts and move offsets are
    # the two proven overflow families.
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
    # The previous cohort's metadata H2D must complete before the pinned
    # rows are rewritten (same guard _stage_block_offsets applies later).
    if bufs.copy_pending and not bufs.copy_done.query():
        bufs.copy_done.synchronize()
    host_table = bufs.request_metadata_host_np
    for row, values in rows:
        if values is not None:
            host_table[row, : len(values)] = values
    # Zero lengths keep the score kernel and selection inert for padded rows.
    host_table[:3, request_count:] = 0
    grow_mean_phase_table(bufs.phase, int(max_round_start) + 1)
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
    try:
        bufs.request_metadata_device.copy_(bufs.request_metadata_host, non_blocking=True)
    finally:
        # Guards the pinned metadata until the asynchronous copies complete.
        bufs.copy_done.record(stream)
        bufs.copy_pending = True
    bufs.page_tables_active = True
    # Per-head modes re-expand the prompt lengths into their row-major view.
    if bufs.row_prompt_offsets is not bufs.prompt_offsets:
        bufs.row_prompt_offsets.view(bufs.max_requests, bufs.selection_rows_per_request).copy_(
            bufs.prompt_offsets.unsqueeze(1).expand(-1, bufs.selection_rows_per_request)
        )


def mark_page_tables_consumed(bufs: SimpleNamespace, *manager_streams: torch.cuda.Stream) -> None:
    """Order V2 page-table reuse and resize after this cohort's compact."""
    if not bufs.page_tables_active:
        raise RuntimeError("TriAttention page tables were not staged")
    bufs.bulk_consume_done.record(torch.cuda.current_stream(bufs.device))
    for manager_stream in manager_streams:
        manager_stream.wait_event(bufs.bulk_consume_done)
    bufs.page_tables_active = False


def settle_top_tokens(bufs: SimpleNamespace) -> None:
    """Pick the top-k, settle ties to sorted ordinals, and pack the move sources.

    The settle kernel resolves the selector's arbitrary tie-breaks with
    lowest-index-wins and rebases each row by its prompt offset.
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
        BLOCK=SETTLE_PACK_BLOCK,
        num_warps=SETTLE_PACK_NUM_WARPS,
    )


def run_eviction_round(bufs: SimpleNamespace, normalize_scores: bool) -> None:
    """Fire one staged eviction round: score, select, settle, compact.

    Every launch covers the full request capacity; padded rows carry zero
    lengths and stay inert.
    """
    request_count = bufs.max_requests
    union = bufs.eviction_mode == "union"
    with nvtx_range("triattention.score", color="blue"):
        # In-place refresh: the compiled score launches captured these pointers.
        gather_mean_phases(
            bufs.phase,
            bufs.round_starts_device,
            bufs.mean_cos,
            bufs.mean_sin,
            bufs.valid_seq_lens_device,
            bufs.token_starts_device,
            bufs.valid_widths,
            request_count,
            swa_destination_bases=bufs.swa_destination_bases,
            rebase_delta=bufs.swa_rebase_delta,
        )
        if union:
            bufs.runner.launch_union_fusion(
                request_count, bufs.mean_cos, bufs.mean_sin, bufs.union_rows[:request_count]
            )
            columns = min(bufs.union_rows.shape[1], bufs.combined.shape[1])
            bufs.combined[:request_count, :columns].copy_(bufs.union_rows[:request_count, :columns])
        else:
            bufs.runner.launch(request_count, bufs.mean_cos, bufs.mean_sin)
            # Gather each decode window from the head-major scratch into the
            # [request, layer, head, token] layout the reduce kernels read;
            # columns past a request's valid width are masked downstream.
            group_size = bufs.num_q_heads // bufs.num_kv_heads
            num_segments = request_count * bufs.num_layers
            pad = bufs.padded_head_columns
            source = (
                bufs.cute_scratch[: bufs.num_kv_heads * pad * num_segments * bufs.bucket_seq_len]
                .view(bufs.num_kv_heads, pad, request_count, bufs.num_layers, bufs.bucket_seq_len)[
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
        # Prebound calls; the draft pack launch precedes the draft moves.
        for args in bufs.compact_launch_args:
            torch.ops.trtllm.sparse_kv_cache_compact_layers(*args)
        if bufs.draft_pack_args is not None:
            _settle_ties_and_pack_compaction_sources_kernel[(bufs.max_requests, 1)](
                *bufs.draft_pack_args, **bufs.draft_pack_kwargs
            )
            for args in bufs.draft_compact_launch_args:
                torch.ops.trtllm.sparse_kv_cache_compact_layers(*args)


class TriAttention(BaseKVCacheCompressionManager):
    """Periodic physical KV eviction driven by trigonometric importance scoring.

    Scores full-attention layers every ``beta`` confirmed tokens and evicts
    below the keep set; kernel-masked SWA layers keep their latest window.
    Every layer ends with the same request-wide cached length.
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
        self.eviction_mode = eviction_mode
        if self.eviction_mode not in ("union", "per_head", "per_layer_perhead"):
            raise ValueError(
                f"Unknown eviction_mode {self.eviction_mode!r}; expected one of "
                "'union', 'per_head', 'per_layer_perhead'"
            )
        self.normalize_scores = bool(normalize_scores)
        if self.eviction_mode == "union" and not self.normalize_scores:
            raise ValueError(
                "TriAttention union eviction requires normalize_scores=True: "
                "the fused union pipeline always z-normalizes score rows"
            )
        self.pin_prefill = bool(pin_prefill)
        # False (default): the budget counts decode tokens only.
        self.count_prompt_tokens = bool(count_prompt_tokens)
        if not self.pin_prefill or self.count_prompt_tokens:
            raise ValueError(
                "TriAttention physical KV reclaim requires pin_prefill=True and "
                "count_prompt_tokens=False so finalized prompt KV is preserved"
            )
        # Calibration is the official TriAttention .pt, converted on the
        # first request; TRT-LLM does not compute calibration.
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

        # Geometric integration offsets, built lazily on first eviction.
        self._offsets: Optional[torch.Tensor] = None
        # Mean-phase table dict, shared by reference with every buffer namespace.
        self._phase: Optional[Dict[str, object]] = None

        # Per-request {generation_steps, evicted_tokens}.
        self._request_states: Dict[int, Dict[str, object]] = {}
        # In-flight overlap batch reference; membership id-set and the growth
        # constant (1 + reserved draft width) resolve lazily.
        self._prepared_generation_batch: Optional[object] = None
        self._prepared_generation_ids: Optional[set] = None
        self._generation_growth: Optional[int] = None
        # Memoized manager invariants.
        self._v2_validated = False
        self._protected_tail_cache: Optional[int] = None
        # Built once at the first eviction, reused for the manager's lifetime.
        self._buffers: Optional[SimpleNamespace] = None
        self._buffers_fingerprint: Optional[tuple] = None
        self._local_to_global_layers_cache: Optional[List[int]] = None
        self._attention_layer_partition_cache: Optional[
            Tuple[List[int], List[int], Optional[int]]
        ] = None
        self._runtime_kv_layout_cache: Optional[Dict[str, object]] = None
        self._draft_runtime_kv_layout_cache: Optional[Dict[str, object]] = None

    def on_request_init(self, request: "LlmRequest", **kwargs) -> None:
        """Validate once and resolve the official calibration for the first request."""
        request_id = request.py_request_id
        if request_id not in self._request_states:
            self._validate_v2_compatibility()
            self._validate_request_capacity(request)
            num_layers = self._num_layers_from_manager()
            self._attention_layer_partition(num_layers)
            self._request_states[request_id] = {
                "generation_steps": 0,
                "evicted_tokens": 0,
            }
        self._ensure_calibrated()

    def _validate_request_capacity(self, request: "LlmRequest") -> None:
        """Require enough target page-table capacity to reach first eviction."""
        manager = self.kv_cache_manager
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
        """Reject runtime modes outside the V2 physical-compaction contract (memoized)."""
        if self._v2_validated:
            return
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
        # Speculative feature gates run in the factory; the draft cache itself
        # is validated here.
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
        self._v2_validated = True

    def on_generation_step_end(self, scheduled_batch: "ScheduledRequests", **kwargs) -> None:
        """Compact after native KV-cache updates have finalized this iteration.

        Runs after KVCacheManagerV2 (capacity reflects the written token and
        any rewind); CUDA stream ordering keeps compaction after any already
        enqueued overlap forward.
        """
        with nvtx_range_debug("triattention.generation_step_end", color="blue"):
            self._periodic_evict(scheduled_batch)

    def on_generation_step_begin(self, scheduled_batch: "ScheduledRequests", **kwargs) -> None:
        """Snapshot the prepared batch; mutation remains in final update."""
        self._prepared_generation_batch = scheduled_batch
        self._prepared_generation_ids = None

    def _inflight_generation_growth(
        self, scheduled_batch: "ScheduledRequests", request_id: int
    ) -> int:
        """Return the in-flight allocation width (1 + reserved draft) under overlap."""
        prepared = self._prepared_generation_batch
        if prepared is None or scheduled_batch is prepared:
            return 0
        member_ids = self._prepared_generation_ids
        if member_ids is None:
            member_ids = {request.py_request_id for request in prepared.generation_requests}
            self._prepared_generation_ids = member_ids
        if request_id not in member_ids:
            return 0
        growth = self._generation_growth
        if growth is None:
            growth = 1 + int(self.kv_cache_manager._kv_reserve_draft_tokens)
            self._generation_growth = growth
        return growth

    def _periodic_evict(
        self,
        scheduled_batch: "ScheduledRequests",
    ) -> None:
        """Every ``beta`` confirmed tokens, evict to the pinned prompt + top-B set."""
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

        # The resolved cache objects thread all the way to resize; the due
        # cohort's metadata is built in the same pass.
        with nvtx_range("triattention.metadata", color="cyan"):
            for request, request_id, kv_cache in resolved_requests:
                # Cadence gate first; capacity/tail math and the consistency
                # raises run in the due branch.
                request_state = self._request_states[request_id]
                previous_step = request_state["generation_steps"]
                step = previous_step + 1 + int(request.py_num_accepted_draft_tokens)
                request_state["generation_steps"] = step
                if previous_step // self.beta >= step // self.beta:
                    continue
                raw_capacity = int(kv_cache.capacity)
                # Speculative reserve + in-flight overlap growth: contiguous
                # after the stable prefix, moved byte-for-byte.
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
                expected_keep_count = self._minimum_evictable_length(request, seq_len)
                if seq_len <= expected_keep_count:
                    continue
                draft_kv_cache = None
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
        num_layers = self._num_layers_from_manager()
        # Ungated NVTX: the due count in the message shows each round's size.
        with nvtx_range(
            f"triattention.evict_request_group reqs={len(prepared)}",
            color="purple",
        ):
            capacity_targets = self._evict_requests(prepared, num_layers)
        self._resize_compacted_requests(capacity_targets, protected_tails)

    def _resize_compacted_requests(self, capacity_targets, protected_tails) -> None:
        """Release each compacted tail through the resolved cache objects."""
        if not capacity_targets:
            return
        with nvtx_range("triattention.resize", color="red"):
            with nvtx_range_debug("triattention.v2_resize", color="red"):
                for rid, kv_cache, _, target_capacity in capacity_targets:
                    if not kv_cache.is_active:
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
                if self.draft_kv_cache_manager is not None:
                    # Same kept set: the draft shrinks to the same retained
                    # length plus its own tail.
                    draft_protected_tail = self._draft_protected_tail_capacity()
                    for rid, _, draft_kv_cache, target_capacity in capacity_targets:
                        if not draft_kv_cache.is_active:
                            continue
                        draft_capacity = target_capacity + draft_protected_tail
                        if not draft_kv_cache.resize(draft_capacity, None):
                            raise RuntimeError(
                                "Failed to resize co-compressed draft KV cache "
                                f"for request {rid} to {draft_capacity} tokens"
                            )

    def _minimum_evictable_length(self, request: "LlmRequest", seq_len: int) -> int:
        """Return the largest cache length for which selection is an identity
        (decode-only budget: everything is kept up to ``prompt_len + budget``)."""
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
        if self._protected_tail_cache is None:
            self._protected_tail_cache = _protected_tail_capacity(self.kv_cache_manager, "")
        return self._protected_tail_cache

    def on_request_finish(self, request: "LlmRequest", **kwargs) -> None:
        """Drop this request's eviction state; the buffers stay resident."""
        self._request_states.pop(request.py_request_id, None)

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

        SWA layers here are stored at full length; the window applies only in
        the attention kernel.
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

        Cached; re-checks live pool page counts before reuse (pool rebalance
        is rejected fail-closed).
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

        ``dense_storage_groups=None`` groups every layer (draft cache);
        ``what`` prefixes error messages.
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
        """Return stable draft V2 pool views, mirroring ``_runtime_kv_layout``."""
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

        Rebuilt only when the pool views change or a round outgrows them.
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
        # Power-of-two bucket sized by the presented cohorts, NOT max_seq_len
        # (a max_seq_len floor would break 32-bit indexing at large batch).
        seq_capacity = max(int(needed_page_tokens), 1024)
        seq_capacity = 1 << (seq_capacity - 1).bit_length()
        seq_capacity = min(seq_capacity, max(int(mgr.max_seq_len), int(needed_page_tokens)))
        # The bucket capacity must be tile-aligned (mis-tiled buckets stripe
        # the score scratch silently).
        score_tile_tokens = max(64, int(mgr.tokens_per_block))
        seq_capacity = -(-seq_capacity // score_tile_tokens) * score_tile_tokens
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
            # Upstream geometric offsets [1, 2, 4, ... <= max].
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
        """Build this round's per-family move offsets, padded to the capacity
        (padded rows repeat the final offset and move nothing)."""

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
        """Score and compact a prepared cohort, returning the resize targets.

        Only full-attention layers are scored; kernel-masked SWA layers keep
        their latest window rebased into the compacted prefix.
        """
        with nvtx_range_debug("triattention.resolve_layout", color="blue"):
            layout = self._runtime_kv_layout(num_layers)
        with nvtx_range_debug("triattention.staging_lookup", color="blue"):
            # Retained spans always cover the model window: construction
            # rejects budget < window, and the pinned prompt only adds.
            bufs = self._buffers_for(layout, prepared)
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
            # The manager's only channel to the runtime: the engine subtracts
            # it where it builds num_cached_tokens_per_seq.
            item["request"].py_num_compressed_tokens = request_state["evicted_tokens"]
            capacity_targets.append(
                (item["request_id"], item["kv_cache"], item["draft_kv_cache"], keep_count)
            )
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
