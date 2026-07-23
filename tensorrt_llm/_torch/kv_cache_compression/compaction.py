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

A general post-eviction component: given each request's kept-token ordinals,
its valid sequence length, and the staged V2 block offsets, the surviving KV
moves in place through batched C++ compact launches, fed by per-request move
indices packed on device. Everything is plain tensors and dicts:
``init_compaction_buffers`` allocates the launch data once per geometry
(called by ``triattention.init_eviction_buffers``) and returns one bundle
whose fields the eviction driver fires directly each round -- the target's
dense/SWA packing rides the driver's fused settle launch (described by the
bundle's ``settle_pack_tensors``/``settle_pack_shape``), the co-compressed
draft's own pack launch and every family's C++ moves are inlined in
``triattention.run_eviction_round``.
"""

from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Tuple

import torch


def _make_move_buffers(
    index_prefix: Tuple[int, ...],
    moves_per_request: List[int],
    device: torch.device,
    external_offsets: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Allocate the packed source-index buffer and its per-request offsets.

    ``external_offsets`` shares a caller-owned device row (refreshed together
    with the round metadata in one copy) instead of allocating one here; the
    index buffer is always sized for the widest per-request move counts.
    """
    offsets = [0]
    for count in moves_per_request:
        offsets.append(offsets[-1] + count)
    indices = torch.empty((*index_prefix, offsets[-1]), dtype=torch.int32, device=device)
    if external_offsets is not None:
        return indices, external_offsets
    return indices, torch.tensor(offsets, dtype=torch.int32, device=device)


def _page_table_provider(
    page_table_slots: Dict[int, int],
    kv_block_offsets: torch.Tensor,
    device: torch.device,
    request_count: int,
    what: str,
) -> Callable[[int], torch.Tensor]:
    """Return per-slot K block-offset views, cached per slot."""
    tables: Dict[int, torch.Tensor] = {}

    def page_table_for(representative: int) -> torch.Tensor:
        slot = page_table_slots[representative]
        if slot not in tables:
            tables[slot] = kv_block_offsets[slot, :request_count, 0]
        return tables[slot]

    return page_table_for


def _compact_groups(
    entries: List[Tuple[int, torch.Tensor, torch.Tensor]],
    pool_keys: Tuple[object, ...],
    device: torch.device,
    per_layer_slots: Optional[Dict[int, int]] = None,
) -> Tuple[Dict[str, object], ...]:
    """Batch layers into one C++ launch per uniform V2 pool.

    Each returned dict is the plain launch data for one
    ``sparse_kv_cache_compact_layers`` call. ``per_layer_slots`` maps each
    layer to its selection row; it is only set when every dense layer keeps
    its own token set (per-layer eviction).
    """
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
    eviction_mode: str,
    layer_pools: List[torch.Tensor],
    dense_layers: List[int],
    swa_layers: List[int],
    layer_group_representative: Dict[int, int],
    kept_token_ordinals: torch.Tensor,
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
    dense_move_offsets: Optional[torch.Tensor] = None,
    swa_move_offsets: Optional[torch.Tensor] = None,
    draft_move_offsets: Optional[torch.Tensor] = None,
) -> Dict[str, object]:
    """Allocate the per-geometry compaction launch data as one plain dict.

    Dense layers keep the prompt in place and compact the selected decode
    tokens plus any target KV reserved for the next overlapped forward;
    kernel-masked SWA layers keep the latest window plus the same protected
    tail. A co-compressed draft cache reuses the target's kept token ordinals
    (broadcast over the draft's own KV-head count, union mode only) plus the
    draft's own protected tail, landing at the same destination base.

    ``kept_token_ordinals`` carries increasing kept decode ordinals (absolute
    positions) per request; prompt tokens never move, so the rectangle is
    prompt-length independent and ``prompt_offsets`` carries each request's
    pinned prompt length. ``kv_block_offsets`` is the staged V2 snapshot
    ``[slot, request, K/V, block]`` (offset = ``2*page + plane``);
    ``protected_tail_capacity`` is the widest per-request tail this geometry
    must support -- actual per-round lengths arrive through the staged
    move-offset rows. Nothing launches here: the target's dense/SWA packing
    rides the driver's fused settle launch (``settle_pack_tensors`` +
    ``settle_pack_shape`` describe it) and the driver's round function fires
    the draft pack (``draft_pack``) and every family's C++ moves directly.

    Returns one plain bundle dict: ``families`` (each ``{"name", "groups",
    "source", "offsets", "destination_bases"}``), the settle/pack launch data
    above, ``draft_pack`` (``None`` or ``{"indices", "offsets",
    "dense_total", "move_capacity", "num_kv_heads"}``), ``swa_rebase_delta``
    (per-round SWA destination rebase), and the geometry scalars
    (``selection_rows``, ``request_count``, ``decode_keep_count``, ...).
    """
    device = layer_pools[dense_layers[0]].device
    request_count = int(request_count)
    decode_keep_count = int(decode_keep_count)
    protected_tail_capacity = int(protected_tail_capacity)
    dense_layers = tuple(int(layer) for layer in dense_layers)
    swa_layers = tuple(int(layer) for layer in swa_layers)
    layer_pool_keys = tuple(layer_pool_keys)

    per_layer = eviction_mode == "per_layer_perhead"
    # The C++ compact op takes the KV-head count from each launch's pool
    # shape [pages, K/V, heads, tokens, dim].
    num_kv_heads = int(layer_pools[dense_layers[0]].shape[2])
    dense_index_prefix = (len(dense_layers), num_kv_heads) if per_layer else (num_kv_heads,)
    dense_move_indices, dense_move_offsets = _make_move_buffers(
        dense_index_prefix,
        [decode_keep_count + protected_tail_capacity] * request_count,
        device,
        external_offsets=dense_move_offsets,
    )
    page_table_for = _page_table_provider(
        page_table_slots, kv_block_offsets, device, request_count, "compaction"
    )
    dense_entries = [
        (layer, layer_pools[layer], page_table_for(layer_group_representative[layer]))
        for layer in dense_layers
    ]

    swa_destination_bases = None
    swa_move_indices = None
    swa_entries = []
    if not swa_layers:
        # No SWA family: drop the unused offsets row. With SWA layers the
        # argument must stay live so the family reads the per-round staged
        # offsets instead of its construction-time sizes.
        swa_move_offsets = None
        swa_window = 0
    else:
        # Per-request window validity (prompt + decode keep >= window) is
        # prompt-dependent and checked by the caller each round.
        swa_window = int(swa_window)
        swa_destination_bases = torch.empty_like(prompt_offsets)
        swa_move_indices, swa_move_offsets = _make_move_buffers(
            (num_kv_heads,),
            [swa_window + protected_tail_capacity] * request_count,
            device,
            external_offsets=swa_move_offsets,
        )
        # SWA layers are staged as their own page-table representatives.
        swa_entries = [(layer, layer_pools[layer], page_table_for(layer)) for layer in swa_layers]

    dense_slots = {layer: slot for slot, layer in enumerate(dense_layers)} if per_layer else None
    union = eviction_mode == "union"
    if union:
        selection_rows = 1
    elif per_layer:
        selection_rows = len(dense_layers) * num_kv_heads
    else:
        selection_rows = num_kv_heads
    # Selection rows carry decode-only kept ordinals (already absolute), so
    # the rectangle is prompt-length independent. HAS_SWA specializes the SWA
    # loads and stores away, so without SWA layers the dense buffers stand in
    # for the compiled-away SWA pointer arguments.
    has_swa = swa_move_indices is not None
    swa_total = int(swa_move_indices.shape[-1]) if has_swa else 0
    # Widest per-request move count any staged offsets may express; the
    # packing loop covers exactly this many move slots per packed row.
    move_capacity = decode_keep_count + protected_tail_capacity
    if has_swa:
        move_capacity = max(move_capacity, swa_window + protected_tail_capacity)
    settle_pack_tensors = (
        valid_sequence_lengths,
        dense_move_offsets,
        dense_move_indices,
        swa_move_offsets if has_swa else dense_move_offsets,
        swa_move_indices if has_swa else dense_move_indices,
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
            # The fused selection-side settle launch packs the dense/SWA move
            # sources when it finalizes the kept ordinals; only the C++ moves
            # consume these buffers. Each round then packs exactly once.
            groups=_compact_groups(dense_entries, layer_pool_keys, device, dense_slots),
            source=dense_move_indices,
            offsets=dense_move_offsets,
            destination_bases=prompt_offsets,
        )
    ]
    if swa_layers:
        # The fused dense pack fills the SWA move buffers in the same launch.
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
        # The draft forms its own launch groups so it may use a different
        # KV-head count than the target.
        draft_num_kv_heads = int(draft_layer_pools[draft_layers[0]].shape[2])
        draft_move_indices, draft_move_offsets = _make_move_buffers(
            (draft_num_kv_heads,),
            [decode_keep_count + draft_tail] * request_count,
            device,
            external_offsets=draft_move_offsets,
        )
        draft_page_table_for = _page_table_provider(
            draft_page_table_slots, draft_kv_block_offsets, device, request_count, "draft"
        )
        draft_entries = [
            (
                layer,
                draft_layer_pools[layer],
                draft_page_table_for(draft_layer_group_representative[layer]),
            )
            for layer in draft_layers
        ]
        # In union mode the pack kernel reads selection row 0 for every packed
        # row, so the driver fires one more pack launch broadcasting the
        # target keep set over the draft KV heads and appending the draft's
        # own tail ordinals; these are that launch's geometry constants.
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
        kept_token_ordinals=kept_token_ordinals,
        valid_sequence_lengths=valid_sequence_lengths,
        selection_rows=selection_rows,
        settle_pack_tensors=settle_pack_tensors,
        settle_pack_shape=settle_pack_shape,
        draft_pack=draft_pack,
        num_kv_heads=num_kv_heads,
        swa_window=swa_window,
        swa_destination_bases=swa_destination_bases,
        # The prompt offsets may be re-staged each round; the driver rebases
        # the SWA landing positions with this delta before the moves.
        swa_rebase_delta=decode_keep_count - swa_window,
        prompt_offsets=prompt_offsets,
        decode_keep_count=decode_keep_count,
        request_count=request_count,
        protected_tail_capacity=protected_tail_capacity,
        draft_protected_tail_capacity=(
            int(draft_protected_tail_capacity or 0) if draft_layers else 0
        ),
    )
