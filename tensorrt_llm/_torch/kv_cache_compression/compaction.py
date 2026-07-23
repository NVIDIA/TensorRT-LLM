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

"""Batched physical KV-cache compaction: an algorithm-neutral mover.

``init_compaction_buffers`` agrees on the decision rows (kept ordinals;
move offsets ride the caller's staged rows) once per geometry and returns
opaque launch plans. The caller materializes its keep decision into the
agreed rows each round, then ``compact`` loops the plans: each packs its
family's move sources and fires its native launches. This module knows
cache-family geometry and the decision format only; the plans' internals
are private to it.
"""

from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _pack_move_sources_kernel(
    kept_ordinal_rows,
    valid_seq_lens,
    dense_move_offsets,
    dense_move_indices,
    swa_move_offsets,
    swa_move_indices,
    KEEP_COUNT: tl.constexpr,
    DECISION_ROWS: tl.constexpr,
    MOVE_CAPACITY: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    PER_LAYER: tl.constexpr,
    DENSE_TOTAL: tl.constexpr,
    SWA_TOTAL: tl.constexpr,
    SWA_WINDOW: tl.constexpr,
    BLOCK: tl.constexpr = 256,
):
    """Pack one decision row into one family's move sources (increasing
    kept ordinals; C++ in-place copy contract): dense rows forward the row
    content verbatim for the first KEEP_COUNT moves, then append the
    protected tail; SWA rows write latest-window ordinals once per KV head."""
    BROADCAST: tl.constexpr = DECISION_ROWS == 1
    HAS_SWA: tl.constexpr = SWA_TOTAL > 0
    request = tl.program_id(0)
    decision_row = tl.program_id(1)
    row = request * DECISION_ROWS + decision_row
    kept_row = kept_ordinal_rows + row * KEEP_COUNT
    dense_begin = tl.load(dense_move_offsets + request)
    dense_end = tl.load(dense_move_offsets + request + 1)
    dense_count = dense_end - dense_begin
    valid_len = tl.load(valid_seq_lens + request)
    if HAS_SWA:
        swa_begin = tl.load(swa_move_offsets + request)
        swa_end = tl.load(swa_move_offsets + request + 1)
        swa_count = swa_end - swa_begin
    for move_start in tl.static_range(0, MOVE_CAPACITY, BLOCK):
        move = move_start + tl.arange(0, BLOCK)
        kept = tl.load(
            kept_row + move,
            mask=move < KEEP_COUNT,
            other=0,
        )
        dense_source = tl.where(move < KEEP_COUNT, kept, valid_len + move - KEEP_COUNT)
        if BROADCAST:
            # The one decision row per request feeds every KV head's packed row.
            for head in tl.static_range(0, NUM_KV_HEADS):
                tl.store(
                    dense_move_indices + head * DENSE_TOTAL + dense_begin.to(tl.int64) + move,
                    dense_source,
                    mask=move < dense_count,
                )
        else:
            dense_output = decision_row.to(tl.int64) * DENSE_TOTAL + dense_begin.to(tl.int64) + move
            tl.store(dense_move_indices + dense_output, dense_source, mask=move < dense_count)
        if HAS_SWA:
            swa_source = valid_len - SWA_WINDOW + move
            if BROADCAST:
                for head in tl.static_range(0, NUM_KV_HEADS):
                    tl.store(
                        swa_move_indices + head * SWA_TOTAL + swa_begin.to(tl.int64) + move,
                        swa_source,
                        mask=move < swa_count,
                    )
            else:
                swa_mask = move < swa_count
                if PER_LAYER:
                    # SWA has one shared row per head; the first layer's decision rows write it.
                    swa_mask = swa_mask & (decision_row < NUM_KV_HEADS)
                head = decision_row % NUM_KV_HEADS
                swa_output = head.to(tl.int64) * SWA_TOTAL + swa_begin.to(tl.int64) + move
                tl.store(
                    swa_move_indices + swa_output,
                    swa_source,
                    mask=swa_mask,
                )


def _make_move_indices(
    index_prefix: Tuple[int, ...],
    moves_per_request: int,
    max_requests: int,
    device: torch.device,
) -> torch.Tensor:
    return torch.empty(
        (*index_prefix, moves_per_request * max_requests), dtype=torch.int32, device=device
    )


def _make_compact_launches(
    entries: List[Tuple[int, torch.Tensor, torch.Tensor]],
    layer_pool_ids: Tuple[int, ...],
    *,
    move_indices: torch.Tensor,
    move_offsets: torch.Tensor,
    destination_bases: torch.Tensor,
    per_layer_slots: Optional[Dict[int, int]] = None,
) -> Tuple[Dict[str, object], ...]:
    """Batch layers into one launch record per uniform V2 pool (grouped by
    canonical pool id plus pool/page-table geometry), carrying the native
    ``sparse_kv_cache_compact_layers`` operands as plain named fields."""
    device = entries[0][1].device
    grouped = OrderedDict()
    for layer, pool, page_table in entries:
        key = (
            layer_pool_ids[layer],
            str(pool.dtype),
            str(pool.device),
            tuple(int(value) for value in pool.shape[1:]),
            tuple(int(value) for value in page_table.shape),
        )
        grouped.setdefault(key, []).append((layer, pool, page_table))

    launches = []
    for group_entries in grouped.values():
        layers = tuple(entry[0] for entry in group_entries)
        pools = list(entry[1] for entry in group_entries)
        page_tables = tuple(entry[2] for entry in group_entries)
        source_layer_indices = None
        if per_layer_slots is not None:
            source_layer_indices = torch.tensor(
                [per_layer_slots[layer] for layer in layers],
                dtype=torch.int32,
                device=device,
            )
        launches.append(
            dict(
                pools=pools,
                pool_pointers=torch.tensor(
                    [pool.data_ptr() for pool in pools],
                    dtype=torch.int64,
                    device=device,
                ),
                page_table=page_tables[0],
                move_indices=move_indices,
                move_offsets=move_offsets,
                destination_bases=destination_bases,
                source_layer_indices=source_layer_indices,
            )
        )
    return tuple(launches)


def init_compaction_buffers(
    *,
    target: Dict[str, object],
    capacities: Dict[str, int],
    draft: Optional[Dict[str, object]] = None,
) -> Tuple[Dict[str, object], ...]:
    """Agree on the decision rows and return opaque launch plans per geometry.

    Move sources must be increasing kept ordinals with
    destination_bases[request] + move <= source[move] (C++ in-place copy
    contract). ``target`` carries the resolved dense/SWA grouping inputs from
    the runtime layout (``per_layer_sources`` selects 3-D per-layer move rows;
    ``layer_pool_ids`` the canonical layer -> V2 pool id tuple that indexes
    ``kv_block_offsets``; ``swa_destination_bases`` the caller-owned SWA
    rebase row) plus the decision inputs :func:`compact` packs each round:
    ``kept_ordinal_rows`` (``max_requests * decision_rows`` rows of
    ``keep_count`` int32 kept ordinals, forwarded verbatim), the
    per-request ``decision_rows`` count (1 = one shared row broadcast over
    every KV head), and the staged per-request ``valid_seq_lens`` the
    protected tail rides after. ``draft`` is one all-or-none resolved branch
    (its dense-only moves broadcast the one shared decision row over the
    draft's own KV heads); ``capacities`` the request/keep/tail capacity
    numbers.

    Returns one plan per compacted cache family (target, then the optional
    draft): plain contract fields -- decision inputs, move-index buffers,
    pack geometry ints, and the grouped native launch records -- that only
    :func:`compact` interprets.
    """
    layer_pools = target["layer_pools"]
    dense_layers = tuple(int(layer) for layer in target["dense_layers"])
    swa_layers = tuple(int(layer) for layer in target["swa_layers"])
    layer_pool_ids = tuple(int(pool_id) for pool_id in target["layer_pool_ids"])
    kv_block_offsets = target["kv_block_offsets"]
    layer_group_representative = target["layer_group_representative"]
    token_starts = target["token_starts"]
    dense_move_offsets = target["dense_move_offsets"]
    swa_move_offsets = target["swa_move_offsets"]
    swa_window = target["swa_window"]
    per_layer_sources = bool(target["per_layer_sources"])
    kept_ordinal_rows = target["kept_ordinal_rows"]
    decision_rows = int(target["decision_rows"])
    valid_seq_lens = target["valid_seq_lens"]

    device = layer_pools[dense_layers[0]].device
    max_requests = int(capacities["max_requests"])
    keep_count = int(capacities["keep_count"])
    protected_tail_capacity = int(capacities["protected_tail_capacity"])

    # Pool shape [pages, K/V, heads, tokens, dim].
    num_kv_heads = int(layer_pools[dense_layers[0]].shape[2])
    dense_index_prefix = (len(dense_layers), num_kv_heads) if per_layer_sources else (num_kv_heads,)
    dense_move_indices = _make_move_indices(
        dense_index_prefix,
        keep_count + protected_tail_capacity,
        max_requests,
        device,
    )
    dense_entries = [
        (
            layer,
            layer_pools[layer],
            kv_block_offsets[layer_pool_ids[layer_group_representative[layer]], :max_requests, 0],
        )
        for layer in dense_layers
    ]

    swa_move_indices = None
    if not swa_layers:
        swa_move_offsets = None
        swa_window = 0
    else:
        swa_window = int(swa_window)
        swa_move_indices = _make_move_indices(
            (num_kv_heads,),
            swa_window + protected_tail_capacity,
            max_requests,
            device,
        )

    dense_slots = (
        {layer: slot for slot, layer in enumerate(dense_layers)} if per_layer_sources else None
    )
    # Widest per-request move count any staged offsets may express.
    move_capacity = keep_count + protected_tail_capacity
    if swa_move_indices is not None:
        move_capacity = max(move_capacity, swa_window + protected_tail_capacity)

    target_launches = list(
        _make_compact_launches(
            dense_entries,
            layer_pool_ids,
            move_indices=dense_move_indices,
            move_offsets=dense_move_offsets,
            destination_bases=token_starts,
            per_layer_slots=dense_slots,
        )
    )
    if swa_layers:
        # SWA layers stage against their own page-table slots.
        swa_entries = [
            (
                layer,
                layer_pools[layer],
                kv_block_offsets[layer_pool_ids[layer], :max_requests, 0],
            )
            for layer in swa_layers
        ]
        target_launches.extend(
            _make_compact_launches(
                swa_entries,
                layer_pool_ids,
                move_indices=swa_move_indices,
                move_offsets=swa_move_offsets,
                destination_bases=target["swa_destination_bases"],
            )
        )

    target_plan = dict(
        kept_ordinal_rows=kept_ordinal_rows,
        valid_seq_lens=valid_seq_lens,
        decision_rows=decision_rows,
        per_layer_sources=per_layer_sources,
        dense_move_offsets=dense_move_offsets,
        dense_move_indices=dense_move_indices,
        swa_move_offsets=swa_move_offsets,
        swa_move_indices=swa_move_indices,
        swa_window=swa_window,
        keep_count=keep_count,
        move_capacity=move_capacity,
        num_kv_heads=num_kv_heads,
        dense_total=int(dense_move_indices.shape[-1]),
        swa_total=int(swa_move_indices.shape[-1]) if swa_move_indices is not None else 0,
        launches=tuple(target_launches),
    )

    plans = [target_plan]
    if draft is not None:
        if decision_rows != 1:
            raise ValueError(
                "draft packing broadcasts one shared decision row per request; "
                f"got {decision_rows} decision rows"
            )
        draft_layer_pools = draft["layer_pools"]
        draft_dense_layers = tuple(int(layer) for layer in draft["dense_layers"])
        draft_layer_pool_ids = tuple(int(pool_id) for pool_id in draft["layer_pool_ids"])
        draft_tail = int(draft["protected_tail_capacity"])
        # Own launch groups: the draft may use a different KV-head count.
        draft_num_kv_heads = int(draft_layer_pools[draft_dense_layers[0]].shape[2])
        draft_move_indices = _make_move_indices(
            (draft_num_kv_heads,),
            keep_count + draft_tail,
            max_requests,
            device,
        )
        draft_entries = [
            (
                layer,
                draft_layer_pools[layer],
                draft["kv_block_offsets"][
                    draft_layer_pool_ids[draft["layer_group_representative"][layer]],
                    :max_requests,
                    0,
                ],
            )
            for layer in draft_dense_layers
        ]
        draft_launches = _make_compact_launches(
            draft_entries,
            draft_layer_pool_ids,
            move_indices=draft_move_indices,
            move_offsets=draft["dense_move_offsets"],
            destination_bases=token_starts,
        )
        draft_plan = dict(
            kept_ordinal_rows=kept_ordinal_rows,
            valid_seq_lens=valid_seq_lens,
            decision_rows=1,
            per_layer_sources=False,
            dense_move_offsets=draft["dense_move_offsets"],
            dense_move_indices=draft_move_indices,
            swa_move_offsets=None,
            swa_move_indices=None,
            swa_window=0,
            keep_count=keep_count,
            move_capacity=keep_count + draft_tail,
            num_kv_heads=draft_num_kv_heads,
            dense_total=int(draft_move_indices.shape[-1]),
            swa_total=0,
            launches=draft_launches,
        )
        plans.append(draft_plan)
    return tuple(plans)


def compact(
    plans: Tuple[Dict[str, object], ...],
    request_count: int,
) -> None:
    """Pack each plan's move sources and fire its native compacts, in plan order.

    Pure mover: the caller has already materialized its kept ordinals into the
    agreed decision rows for the active ``request_count`` cohort, and the
    caller owns the completion ordering of the whole round. Every launch
    argument comes straight off the plan's init-built contract fields; the
    round constructs no containers and runs no torch ops of its own.
    """
    for plan in plans:
        _pack_move_sources_kernel[(request_count, plan["decision_rows"])](
            plan["kept_ordinal_rows"],
            plan["valid_seq_lens"],
            plan["dense_move_offsets"],
            plan["dense_move_indices"],
            plan["swa_move_offsets"],
            plan["swa_move_indices"],
            KEEP_COUNT=plan["keep_count"],
            DECISION_ROWS=plan["decision_rows"],
            MOVE_CAPACITY=plan["move_capacity"],
            NUM_KV_HEADS=plan["num_kv_heads"],
            PER_LAYER=plan["per_layer_sources"],
            DENSE_TOTAL=plan["dense_total"],
            SWA_TOTAL=plan["swa_total"],
            SWA_WINDOW=plan["swa_window"],
        )
        for launch in plan["launches"]:
            torch.ops.trtllm.sparse_kv_cache_compact_layers(
                launch["pools"],
                launch["pool_pointers"],
                launch["page_table"],
                launch["move_indices"],
                launch["move_offsets"],
                launch["destination_bases"],
                launch["source_layer_indices"],
            )
