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

Given each request's kept-token ordinals, its valid sequence length, and the
staged V2 block offsets, this module packs per-request move indices with one
Triton launch per compacted cache (one launch covers the target's dense and
SWA families; a co-compressed draft adds a second) and then moves the
surviving KV in place with batched C++ compact launches. Everything is plain
tensors and dicts: ``build_cache_compactions`` allocates the launch data once
per geometry (called by ``triattention.prepare_eviction_workspace``) and
``run_cache_compactions`` fires the kernels directly each round. A driver
that finalizes the keep set in its own GPU launch takes the target's
dense/SWA packing over into that launch (``fuse_dense_pack_into_selection``);
the draft always packs here.
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


# Launch shape of the move-index packing kernel: tokens per program along the
# move axis, and its warp count.
_PACK_BLOCK_TOKENS = 256
_PACK_NUM_WARPS = 4


def build_move_pack_arguments(
    kept_token_ordinals: torch.Tensor,
    valid_sequence_lengths: torch.Tensor,
    move_source_offsets: torch.Tensor,
    move_source_indices: torch.Tensor,
    *,
    eviction_mode: str,
    decode_keep_count: int,
    num_dense_layers: int,
    num_kv_heads: int,
    max_protected_tail: int,
    swa_window: int,
    swa_move_source_offsets: Optional[torch.Tensor],
    swa_move_source_indices: Optional[torch.Tensor],
) -> Dict[str, object]:
    """Describe one move-index packing as plain kernel launch data.

    The packing kernel reads the kept-token ordinals and each request's valid
    length and writes the packed per-(layer, head) move source indices
    consumed by the C++ compact launches. ``launch_move_pack`` fires it
    standalone; a fused selection-side settle launch consumes the same dict.
    """
    per_layer = eviction_mode == "per_layer_perhead"
    union = eviction_mode == "union"
    request_count = int(kept_token_ordinals.shape[0]) if kept_token_ordinals.ndim else 0
    if union:
        selection_rows = 1
    elif per_layer:
        selection_rows = num_dense_layers * num_kv_heads
    else:
        selection_rows = num_kv_heads
    # Selection rows carry decode-only kept ordinals (already absolute), so
    # the rectangle is prompt-length independent.
    if swa_move_source_indices is not None:
        swa_offsets_arg = swa_move_source_offsets
        swa_indices_arg = swa_move_source_indices
        swa_total = int(swa_move_source_indices.shape[-1])
    else:
        # HAS_SWA specializes all corresponding loads and stores away.
        swa_offsets_arg = move_source_offsets
        swa_indices_arg = move_source_indices
        swa_total = 0

    max_move = decode_keep_count + max_protected_tail
    if swa_total:
        max_move = max(max_move, swa_window + max_protected_tail)
    return dict(
        kept_token_ordinals=kept_token_ordinals,
        valid_sequence_lengths=valid_sequence_lengths,
        dense_offsets=move_source_offsets,
        dense_indices=move_source_indices,
        swa_offsets=swa_offsets_arg,
        swa_indices=swa_indices_arg,
        dense_total=int(move_source_indices.shape[-1]),
        swa_total=swa_total,
        selection_rows=selection_rows,
        keep_count=decode_keep_count,
        request_count=request_count,
        num_kv_heads=num_kv_heads,
        swa_window=swa_window,
        # Widest per-request move count any staged offsets may express; the
        # packing loop covers exactly this many move slots per packed row.
        move_capacity=max_move,
        union=union,
        per_layer=per_layer,
        has_swa=swa_total > 0,
    )


def launch_move_pack(pack: Dict[str, object]) -> None:
    """Fire one standalone move-index packing launch.

    One program per (request, selection row); the settle half is compiled
    away because the ordinals arrive pre-settled (the draft flow reuses the
    target's keep set verbatim), so only the pack half runs. The settle-side
    pointer arguments are compiled away with it; any well-formed tensor
    stands in for them.
    """
    from .triattention_kernels import _settle_ties_and_pack_compaction_sources_kernel

    kept = pack["kept_token_ordinals"]
    _settle_ties_and_pack_compaction_sources_kernel[
        (pack["request_count"], pack["selection_rows"])
    ](
        kept,
        pack["valid_sequence_lengths"],
        pack["dense_offsets"],
        kept,
        kept,
        pack["valid_sequence_lengths"],
        pack["dense_offsets"],
        pack["dense_indices"],
        pack["swa_offsets"],
        pack["swa_indices"],
        WIDTH=pack["keep_count"],
        KEEP_COUNT=pack["keep_count"],
        SELECTION_ROWS=pack["selection_rows"],
        DENSE_TOTAL=pack["dense_total"],
        SWA_TOTAL=pack["swa_total"],
        MOVE_CAPACITY=pack["move_capacity"],
        NUM_KV_HEADS=pack["num_kv_heads"],
        SWA_WINDOW=pack["swa_window"],
        UNION=pack["union"],
        PER_LAYER=pack["per_layer"],
        HAS_SWA=pack["has_swa"],
        HAS_SETTLE=False,
        BLOCK=_PACK_BLOCK_TOKENS,
        num_warps=_PACK_NUM_WARPS,
    )


def build_cache_compactions(
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
    fuse_dense_pack_into_selection: bool = False,
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
    move-offset rows. With ``fuse_dense_pack_into_selection`` the target's
    dense/SWA packing is left to the caller's fused settle launch (the
    returned ``dense_pack`` dict describes it) and only the C++ moves run
    here; the draft always keeps its own pack launch.

    Returns ``{"families": [...], "dense_pack": ..., "num_kv_heads": ...,
    "swa_window": ..., "swa_destination_bases": ...}`` where each family is
    ``{"pack": dict|None, "groups": (...), "source": t, "offsets": t,
    "destination_bases": t}``.
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
    dense_pack = build_move_pack_arguments(
        kept_token_ordinals,
        valid_sequence_lengths,
        dense_move_offsets,
        dense_move_indices,
        eviction_mode=eviction_mode,
        decode_keep_count=decode_keep_count,
        num_dense_layers=len(dense_layers),
        num_kv_heads=num_kv_heads,
        max_protected_tail=protected_tail_capacity,
        swa_window=swa_window,
        swa_move_source_offsets=swa_move_offsets,
        swa_move_source_indices=swa_move_indices,
    )
    families = [
        dict(
            name="dense",
            # A fused selection-side settle launch packs the dense/SWA move
            # sources when it finalizes the kept ordinals; only the C++ moves
            # stay here. Each round then packs exactly once.
            pack=None if fuse_dense_pack_into_selection else dense_pack,
            groups=_compact_groups(dense_entries, layer_pool_keys, device, dense_slots),
            source=dense_move_indices,
            offsets=dense_move_offsets,
            destination_bases=prompt_offsets,
        )
    ]
    if swa_layers:
        # The dense pack call fills the SWA move buffers in the same run.
        families.append(
            dict(
                name="swa",
                pack=None,
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
        # row, so one more pack launch broadcasts the target keep set over the
        # draft KV heads and appends the draft's own tail ordinals.
        draft_pack = build_move_pack_arguments(
            kept_token_ordinals,
            valid_sequence_lengths,
            draft_move_offsets,
            draft_move_indices,
            eviction_mode="union",
            decode_keep_count=decode_keep_count,
            num_dense_layers=1,
            num_kv_heads=draft_num_kv_heads,
            max_protected_tail=draft_tail,
            swa_window=0,
            swa_move_source_offsets=None,
            swa_move_source_indices=None,
        )
        families.append(
            dict(
                name="draft",
                pack=draft_pack,
                groups=_compact_groups(draft_entries, tuple(draft_layer_pool_keys), device),
                source=draft_move_indices,
                offsets=draft_move_offsets,
                destination_bases=prompt_offsets,
            )
        )

    return dict(
        families=families,
        dense_pack=dense_pack,
        num_kv_heads=num_kv_heads,
        swa_window=swa_window,
        swa_destination_bases=swa_destination_bases,
        prompt_offsets=prompt_offsets,
        decode_keep_count=decode_keep_count,
        request_count=request_count,
        protected_tail_capacity=protected_tail_capacity,
        draft_protected_tail_capacity=(
            int(draft_protected_tail_capacity or 0) if draft_layers else 0
        ),
    )


def run_cache_compactions(compaction: Dict[str, object]) -> None:
    """Pack the move indices, then run every cache family's C++ compacts."""
    swa_destination_bases = compaction["swa_destination_bases"]
    if swa_destination_bases is not None:
        # The prompt offsets may have been re-staged since construction;
        # rebase the SWA landing positions for this round.
        torch.add(
            compaction["prompt_offsets"],
            compaction["decode_keep_count"] - compaction["swa_window"],
            out=swa_destination_bases,
        )
    for family in compaction["families"]:
        if family["pack"] is not None:
            launch_move_pack(family["pack"])
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
