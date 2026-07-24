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

``build_compaction_plans`` returns opaque launch plans once per geometry;
each round the caller writes its keep decision into the agreed rows and
``compact`` packs every plan's move sources and fires its native launches.
"""

from typing import Dict, List, Optional, Tuple, TypedDict

import torch
import triton
import triton.language as tl


class CompactionLaunch(TypedDict):
    pools: List[torch.Tensor]
    pool_pointers: torch.Tensor
    page_table: torch.Tensor
    move_indices: torch.Tensor
    move_offsets: torch.Tensor
    destination_bases: torch.Tensor
    source_layer_indices: Optional[torch.Tensor]


class CompactionPlan(TypedDict):
    decision_rows: int
    pack_args: Tuple[Optional[torch.Tensor], ...]
    pack_constexprs: Dict[str, object]
    launches: Tuple[CompactionLaunch, ...]


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
    """Pack one decision row into one family's move sources: dense rows emit the
    kept tokens then the protected tail; SWA rows emit the latest window
    (ascending order: the native copy moves in place)."""
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


def build_compaction_plans(
    *,
    target: Dict[str, object],
    draft: Optional[Dict[str, object]] = None,
) -> Tuple[CompactionPlan, ...]:
    """One plan per compacted cache family (target, then the optional draft);
    only :func:`compact` reads the plans' contents at launch."""
    plans = [_build_plan(target)]
    if draft is not None:
        # The draft compacts under the target's decision rows and token starts.
        plans.append(
            _build_plan(
                dict(
                    draft,
                    token_starts=target["token_starts"],
                    kept_ordinal_rows=target["kept_ordinal_rows"],
                    valid_seq_lens=target["valid_seq_lens"],
                )
            )
        )
    return tuple(plans)


def _build_plan(cache: Dict[str, object]) -> CompactionPlan:
    layer_pools = cache["layer_pools"]
    dense_layers = tuple(int(layer) for layer in cache["dense_layers"])
    swa_layers = tuple(int(layer) for layer in cache.get("swa_layers", ()))
    layer_pool_ids = tuple(int(pool_id) for pool_id in cache["layer_pool_ids"])
    kv_block_offsets = cache["kv_block_offsets"]
    layer_group_representative = cache["layer_group_representative"]
    token_starts = cache["token_starts"]
    dense_move_offsets = cache["dense_move_offsets"]
    per_layer_sources = bool(cache.get("per_layer_sources", False))
    kept_ordinal_rows = cache["kept_ordinal_rows"]
    valid_seq_lens = cache["valid_seq_lens"]
    protected_tail_capacity = int(cache["protected_tail_capacity"])

    first_pool = layer_pools[dense_layers[0]]
    device = first_pool.device
    max_requests = int(valid_seq_lens.shape[0])
    keep_count = int(kept_ordinal_rows.shape[1])
    decision_rows = int(kept_ordinal_rows.shape[0]) // max_requests
    # Pool shape [pages, K/V, heads, tokens, dim].
    num_kv_heads = int(first_pool.shape[2])
    dense_index_prefix = (len(dense_layers), num_kv_heads) if per_layer_sources else (num_kv_heads,)
    dense_move_indices = torch.empty(
        (*dense_index_prefix, (keep_count + protected_tail_capacity) * max_requests),
        dtype=torch.int32,
        device=device,
    )
    dense_entries = [
        (
            layer,
            layer_pools[layer],
            kv_block_offsets[layer_pool_ids[layer_group_representative[layer]], :max_requests, 0],
        )
        for layer in dense_layers
    ]
    dense_slots = (
        {layer: slot for slot, layer in enumerate(dense_layers)} if per_layer_sources else None
    )

    swa_move_indices = None
    swa_move_offsets = None
    swa_window = 0
    # One move group per family axis (dense / SWA): the layers + the tensors driving their moves.
    move_groups = [
        (dense_entries, dense_move_indices, dense_move_offsets, token_starts, dense_slots),
    ]
    if swa_layers:
        swa_move_offsets = cache["swa_move_offsets"]
        swa_window = int(cache["swa_window"])
        swa_move_indices = torch.empty(
            (num_kv_heads, (swa_window + protected_tail_capacity) * max_requests),
            dtype=torch.int32,
            device=device,
        )
        # SWA layers stage against their own page-table slots.
        swa_entries = [
            (
                layer,
                layer_pools[layer],
                kv_block_offsets[layer_pool_ids[layer], :max_requests, 0],
            )
            for layer in swa_layers
        ]
        move_groups.append(
            (swa_entries, swa_move_indices, swa_move_offsets, cache["swa_destination_bases"], None)
        )

    # Widest per-request move count any staged offsets may express.
    move_capacity = keep_count + protected_tail_capacity
    if swa_layers:
        move_capacity = max(move_capacity, swa_window + protected_tail_capacity)

    launches: List[CompactionLaunch] = []
    for entries, move_indices, move_offsets, destination_bases, slots in move_groups:
        grouped = {}
        for layer, pool, page_table in entries:
            key = (
                layer_pool_ids[layer],
                str(pool.dtype),
                str(pool.device),
                tuple(int(value) for value in pool.shape[1:]),
                tuple(int(value) for value in page_table.shape),
            )
            grouped.setdefault(key, []).append((layer, pool, page_table))
        for group_entries in grouped.values():
            layers = tuple(entry[0] for entry in group_entries)
            pools = list(entry[1] for entry in group_entries)
            source_layer_indices = None
            if slots is not None:
                source_layer_indices = torch.tensor(
                    [slots[layer] for layer in layers],
                    dtype=torch.int32,
                    device=device,
                )
            launches.append(
                CompactionLaunch(
                    pools=pools,
                    pool_pointers=torch.tensor(
                        [pool.data_ptr() for pool in pools],
                        dtype=torch.int64,
                        device=device,
                    ),
                    page_table=group_entries[0][2],
                    move_indices=move_indices,
                    move_offsets=move_offsets,
                    destination_bases=destination_bases,
                    source_layer_indices=source_layer_indices,
                )
            )

    return CompactionPlan(
        decision_rows=decision_rows,
        pack_args=(
            kept_ordinal_rows,
            valid_seq_lens,
            dense_move_offsets,
            dense_move_indices,
            swa_move_offsets,
            swa_move_indices,
        ),
        pack_constexprs=dict(
            KEEP_COUNT=keep_count,
            DECISION_ROWS=decision_rows,
            MOVE_CAPACITY=move_capacity,
            NUM_KV_HEADS=num_kv_heads,
            PER_LAYER=per_layer_sources,
            DENSE_TOTAL=int(dense_move_indices.shape[-1]),
            SWA_TOTAL=int(swa_move_indices.shape[-1]) if swa_move_indices is not None else 0,
            SWA_WINDOW=swa_window,
        ),
        launches=tuple(launches),
    )


def compact(
    plans: Tuple[CompactionPlan, ...],
    request_count: int,
) -> None:
    """Pack each plan's move sources and fire its native compacts, in plan order
    (pure mover: the caller owns the decision rows and the round's completion ordering)."""
    for plan in plans:
        _pack_move_sources_kernel[(request_count, plan["decision_rows"])](
            *plan["pack_args"], **plan["pack_constexprs"]
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
