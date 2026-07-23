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

"""Batched physical KV-cache compaction for eviction-based compression.

``init_compaction_buffers`` allocates plain-dict launch data once per
geometry; the eviction driver fires the bundle's fields directly each round.
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
    """Packed source-index buffer sized for the widest per-request moves
    (the move offsets are caller-owned rows)."""
    return torch.empty(
        (*index_prefix, moves_per_request * request_count), dtype=torch.int32, device=device
    )


def _compact_groups(
    entries: List[Tuple[int, torch.Tensor, torch.Tensor]],
    pool_keys: Tuple[object, ...],
    device: torch.device,
    per_layer_slots: Optional[Dict[int, int]] = None,
) -> Tuple[Dict[str, object], ...]:
    """Batch layers into one ``sparse_kv_cache_compact_layers`` launch per
    uniform V2 pool. ``per_layer_slots`` maps layers to selection rows
    (per-layer eviction only)."""
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


def init_compaction_buffers(
    *,
    union: bool,
    per_layer: bool,
    layer_pools: List[torch.Tensor],
    dense_layers: List[int],
    swa_layers: List[int],
    layer_group_representative: Dict[int, int],
    valid_sequence_lengths: torch.Tensor,
    kv_block_offsets: torch.Tensor,
    page_table_slots: Dict[int, int],
    request_count: int,
    prompt_offsets: torch.Tensor,
    decode_keep_count: int,
    swa_window: Optional[int],
    layer_pool_keys: List[object],
    protected_tail_capacity: int = 0,
    draft_layer_pools: Optional[List[torch.Tensor]] = None,
    draft_layers: Optional[List[int]] = None,
    draft_layer_group_representative: Optional[Dict[int, int]] = None,
    draft_layer_pool_keys: Optional[List[object]] = None,
    draft_protected_tail_capacity: Optional[int] = None,
    draft_kv_block_offsets: Optional[torch.Tensor] = None,
    draft_page_table_slots: Optional[Dict[int, int]] = None,
    dense_move_offsets: torch.Tensor,
    swa_move_offsets: Optional[torch.Tensor] = None,
    draft_move_offsets: Optional[torch.Tensor] = None,
) -> Dict[str, object]:
    """Allocate per-geometry compaction launch data for the driver's round.

    Selection rows: union = one per request; per_layer = one per (layer, KV
    head); else one per KV head. Move sources must be increasing kept ordinals
    with destination_bases[request] + move <= source[move] (C++ in-place copy
    contract). Returns the launch bundle the round function fires directly.
    """
    device = layer_pools[dense_layers[0]].device
    request_count = int(request_count)
    decode_keep_count = int(decode_keep_count)
    protected_tail_capacity = int(protected_tail_capacity)
    dense_layers = tuple(int(layer) for layer in dense_layers)
    swa_layers = tuple(int(layer) for layer in swa_layers)
    layer_pool_keys = tuple(layer_pool_keys)

    # Pool shape [pages, K/V, heads, tokens, dim].
    num_kv_heads = int(layer_pools[dense_layers[0]].shape[2])
    dense_index_prefix = (len(dense_layers), num_kv_heads) if per_layer else (num_kv_heads,)
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

    dense_slots = {layer: slot for slot, layer in enumerate(dense_layers)} if per_layer else None
    # Without SWA layers the SWA pointer args are None (HAS_SWA=False).
    has_swa = swa_move_indices is not None
    swa_total = int(swa_move_indices.shape[-1]) if has_swa else 0
    # Widest per-request move count any staged offsets may express.
    move_capacity = decode_keep_count + protected_tail_capacity
    if has_swa:
        move_capacity = max(move_capacity, swa_window + protected_tail_capacity)
    settle_pack_tensors = (
        valid_sequence_lengths,
        dense_move_offsets,
        dense_move_indices,
        swa_move_offsets if has_swa else None,
        swa_move_indices if has_swa else None,
    )
    settle_pack_shape = dict(
        DENSE_TOTAL=int(dense_move_indices.shape[-1]),
        SWA_TOTAL=swa_total,
        MOVE_CAPACITY=move_capacity,
        NUM_KV_HEADS=num_kv_heads,
        SWA_WINDOW=swa_window,
        UNION=union,
        PER_LAYER=per_layer,
        HAS_SWA=has_swa,
    )
    families = [
        dict(
            name="dense",
            groups=_compact_groups(dense_entries, layer_pool_keys, device, dense_slots),
            source=dense_move_indices,
            offsets=dense_move_offsets,
            destination_bases=prompt_offsets,
        )
    ]
    if swa_layers:
        families.append(
            dict(
                name="swa",
                groups=_compact_groups(swa_entries, layer_pool_keys, device),
                source=swa_move_indices,
                offsets=swa_move_offsets,
                destination_bases=swa_destination_bases,
            )
        )

    if draft_layers:
        draft_tail = int(draft_protected_tail_capacity or 0)
        draft_layers = tuple(int(layer) for layer in draft_layers)
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
                draft_kv_block_offsets[
                    draft_page_table_slots[draft_layer_group_representative[layer]],
                    :request_count,
                    0,
                ],
            )
            for layer in draft_layers
        ]
        # Geometry constants for the draft's own pack launch.
        draft_pack = dict(
            indices=draft_move_indices,
            offsets=draft_move_offsets,
            dense_total=int(draft_move_indices.shape[-1]),
            move_capacity=decode_keep_count + draft_tail,
            num_kv_heads=draft_num_kv_heads,
        )
        families.append(
            dict(
                name="draft",
                groups=_compact_groups(draft_entries, tuple(draft_layer_pool_keys), device),
                source=draft_move_indices,
                offsets=draft_move_offsets,
                destination_bases=prompt_offsets,
            )
        )
    else:
        draft_pack = None

    return dict(
        families=families,
        settle_pack_tensors=settle_pack_tensors,
        settle_pack_shape=settle_pack_shape,
        draft_pack=draft_pack,
        swa_destination_bases=swa_destination_bases,
        # Per-round SWA destination rebase delta.
        swa_rebase_delta=decode_keep_count - swa_window,
    )
