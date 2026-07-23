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

``init_compaction_buffers`` agrees on the decision buffers (move-source
indices; offsets ride the caller's staged rows) once per geometry and retains
one launch contract. The caller materializes its keep decision into those
buffers each round, then ``compact`` fires the native target and draft
launches. This module knows cache-family geometry and the decision format
only; the contract's launch tuples are private to it.
"""

from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch


def _make_move_indices(
    index_prefix: Tuple[int, ...],
    moves_per_request: int,
    request_count: int,
    device: torch.device,
) -> torch.Tensor:
    return torch.empty(
        (*index_prefix, moves_per_request * request_count), dtype=torch.int32, device=device
    )


def _compact_groups(
    entries: List[Tuple[int, torch.Tensor, torch.Tensor]],
    pool_keys: Tuple[object, ...],
    device: torch.device,
    per_layer_slots: Optional[Dict[int, int]] = None,
) -> Tuple[Dict[str, object], ...]:
    """Batch layers into one ``sparse_kv_cache_compact_layers`` launch per uniform V2 pool."""
    grouped = OrderedDict()
    for layer, pool, page_table in entries:
        key = (
            pool_keys[layer],
            str(pool.dtype),
            str(pool.device),
            tuple(int(value) for value in pool.shape[1:]),
            tuple(int(value) for value in page_table.shape),
        )
        grouped.setdefault(key, []).append((layer, pool, page_table))

    result = []
    for group_entries in grouped.values():
        layers = tuple(entry[0] for entry in group_entries)
        pools = tuple(entry[1] for entry in group_entries)
        page_tables = tuple(entry[2] for entry in group_entries)
        source_layer_indices = None
        if per_layer_slots is not None:
            source_layer_indices = torch.tensor(
                [per_layer_slots[layer] for layer in layers],
                dtype=torch.int32,
                device=device,
            )
        result.append(
            dict(
                pools=list(pools),
                page_table=page_tables[0],
                pool_pointers=torch.tensor(
                    [pool.data_ptr() for pool in pools],
                    dtype=torch.int64,
                    device=device,
                ),
                source_layer_indices=source_layer_indices,
            )
        )
    return tuple(result)


def _launch_tuples(
    groups: Tuple[Dict[str, object], ...],
    source: torch.Tensor,
    offsets: torch.Tensor,
    destination_bases: torch.Tensor,
) -> Tuple[tuple, ...]:
    return tuple(
        (
            group["pools"],
            group["pool_pointers"],
            group["page_table"],
            source,
            offsets,
            destination_bases,
            group["source_layer_indices"],
        )
        for group in groups
    )


def init_compaction_buffers(
    *,
    target: Dict[str, object],
    capacities: Dict[str, int],
    draft: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """Agree on the decision buffers and retain one launch contract per geometry.

    Move sources must be increasing kept ordinals with
    destination_bases[request] + move <= source[move] (C++ in-place copy
    contract). ``target`` carries the resolved dense/SWA grouping inputs from
    the runtime layout (``per_layer_sources`` selects 3-D per-layer move rows);
    ``draft`` is one all-or-none resolved branch; ``capacities`` the
    request/keep/tail capacity numbers. The returned contract exposes the
    agreed move-source buffers and geometry constants; its launch tuples are
    private to :func:`compact`.
    """
    layer_pools = target["layer_pools"]
    dense_layers = tuple(int(layer) for layer in target["dense_layers"])
    swa_layers = tuple(int(layer) for layer in target["swa_layers"])
    layer_pool_keys = tuple(target["layer_pool_keys"])
    kv_block_offsets = target["kv_block_offsets"]
    page_table_slots = target["page_table_slots"]
    layer_group_representative = target["layer_group_representative"]
    prompt_offsets = target["prompt_offsets"]
    dense_move_offsets = target["dense_move_offsets"]
    swa_move_offsets = target["swa_move_offsets"]
    swa_window = target["swa_window"]
    per_layer_sources = bool(target["per_layer_sources"])

    device = layer_pools[dense_layers[0]].device
    request_count = int(capacities["request_capacity"])
    decode_keep_count = int(capacities["decode_keep_count"])
    protected_tail_capacity = int(capacities["protected_tail_capacity"])

    # Pool shape [pages, K/V, heads, tokens, dim].
    num_kv_heads = int(layer_pools[dense_layers[0]].shape[2])
    dense_index_prefix = (len(dense_layers), num_kv_heads) if per_layer_sources else (num_kv_heads,)
    dense_move_indices = _make_move_indices(
        dense_index_prefix,
        decode_keep_count + protected_tail_capacity,
        request_count,
        device,
    )
    dense_entries = [
        (
            layer,
            layer_pools[layer],
            kv_block_offsets[
                page_table_slots[layer_group_representative[layer]], :request_count, 0
            ],
        )
        for layer in dense_layers
    ]

    swa_destination_bases = None
    swa_move_indices = None
    swa_entries = []
    if not swa_layers:
        swa_move_offsets = None
        swa_window = 0
    else:
        swa_window = int(swa_window)
        swa_destination_bases = torch.empty_like(prompt_offsets)
        swa_move_indices = _make_move_indices(
            (num_kv_heads,),
            swa_window + protected_tail_capacity,
            request_count,
            device,
        )
        # SWA layers are staged as their own page-table representatives.
        swa_entries = [
            (
                layer,
                layer_pools[layer],
                kv_block_offsets[page_table_slots[layer], :request_count, 0],
            )
            for layer in swa_layers
        ]

    dense_slots = (
        {layer: slot for slot, layer in enumerate(dense_layers)} if per_layer_sources else None
    )
    has_swa = swa_move_indices is not None
    swa_total = int(swa_move_indices.shape[-1]) if has_swa else 0
    # Widest per-request move count any staged offsets may express.
    move_capacity = decode_keep_count + protected_tail_capacity
    if has_swa:
        move_capacity = max(move_capacity, swa_window + protected_tail_capacity)

    target_launches = list(
        _launch_tuples(
            _compact_groups(dense_entries, layer_pool_keys, device, dense_slots),
            dense_move_indices,
            dense_move_offsets,
            prompt_offsets,
        )
    )
    if swa_layers:
        target_launches.extend(
            _launch_tuples(
                _compact_groups(swa_entries, layer_pool_keys, device),
                swa_move_indices,
                swa_move_offsets,
                swa_destination_bases,
            )
        )

    draft_launches: Tuple[tuple, ...] = ()
    draft_move_indices = None
    if draft is not None:
        draft_layer_pools = draft["layer_pools"]
        draft_layers = tuple(int(layer) for layer in draft["layers"])
        draft_tail = int(draft["protected_tail_capacity"])
        # Own launch groups: the draft may use a different KV-head count.
        draft_num_kv_heads = int(draft_layer_pools[draft_layers[0]].shape[2])
        draft_move_indices = _make_move_indices(
            (draft_num_kv_heads,),
            decode_keep_count + draft_tail,
            request_count,
            device,
        )
        draft_entries = [
            (
                layer,
                draft_layer_pools[layer],
                draft["kv_block_offsets"][
                    draft["page_table_slots"][draft["layer_group_representative"][layer]],
                    :request_count,
                    0,
                ],
            )
            for layer in draft_layers
        ]
        draft_launches = _launch_tuples(
            _compact_groups(draft_entries, tuple(draft["layer_pool_keys"]), device),
            draft_move_indices,
            draft["move_offsets"],
            prompt_offsets,
        )

    return dict(
        # Agreed decision buffers and geometry constants (the public interface).
        dense_move_indices=dense_move_indices,
        swa_move_indices=swa_move_indices,
        draft_move_indices=draft_move_indices,
        dense_total=int(dense_move_indices.shape[-1]),
        swa_total=swa_total,
        move_capacity=move_capacity,
        num_kv_heads=num_kv_heads,
        swa_window=swa_window,
        has_swa=has_swa,
        swa_destination_bases=swa_destination_bases,
        # Per-round SWA destination rebase delta.
        swa_rebase_delta=decode_keep_count - swa_window,
        # Completion event: compact() records it after the last native launch.
        consume_done=torch.cuda.Event(),
        # Private launch tuples: only compact() interprets these.
        target_launches=tuple(target_launches),
        draft_launches=draft_launches,
        has_draft=draft is not None,
    )


def compact(compaction: Dict[str, object], request_count: int) -> None:
    """Fire the native target compacts, then the draft compacts, and record completion.

    Pure mover: the caller has already materialized its keep decision into the
    agreed move-source buffers for the active ``request_count`` cohort.
    """
    for launch in compaction["target_launches"]:
        torch.ops.trtllm.sparse_kv_cache_compact_layers(*launch)
    for launch in compaction["draft_launches"]:
        torch.ops.trtllm.sparse_kv_cache_compact_layers(*launch)
    compaction["consume_done"].record(torch.cuda.current_stream())
