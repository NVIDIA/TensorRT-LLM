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
prepared Triton launch and then moves the surviving KV in place with batched
C++ compact launches. Inputs are plain tensors, so any eviction method that
produces a kept-token set per request can drive it.
"""

from collections import OrderedDict
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple

import torch

from .prepared_launch import PreparedTritonKernelLaunch

_SUPPORTED_POOL_DTYPES = (torch.bfloat16, torch.float16, torch.float32)

# Launch shape of the move-index packing kernel: tokens per program along the
# move axis, and its warp count.
_PACK_BLOCK_TOKENS = 256
_PACK_NUM_WARPS = 4


class _CppCompactGroup(NamedTuple):
    """One layered sparse-KV updater launch over pools sharing a block table."""

    pools: Tuple[torch.Tensor, ...]
    page_table: torch.Tensor
    pool_pointers: torch.Tensor
    source_layer_indices: Optional[torch.Tensor]

    def launch(
        self, source: torch.Tensor, offsets: torch.Tensor, destination_bases: torch.Tensor
    ) -> None:
        torch.ops.trtllm.sparse_kv_cache_compact_layers(
            list(self.pools),
            self.pool_pointers,
            self.page_table,
            source,
            offsets,
            self.source_layer_indices,
            destination_bases,
        )


class _SingleCacheCompaction(NamedTuple):
    """One compacted cache family (target dense, target SWA, or draft).

    Holds the prepared launch that packs this family's move indices (None
    when an earlier family's pack launch fills them in the same call), the
    C++ launch groups that consume them, and the destination base the moved
    tokens land at.
    """

    prepared_move_index_pack: Optional[PreparedTritonKernelLaunch]
    cpp_launch_groups: Tuple[_CppCompactGroup, ...]
    move_source_indices: torch.Tensor
    move_source_offsets: torch.Tensor
    # Per-request landing positions; may alias staged prompt lengths so the
    # values track the current round without a refresh.
    destination_bases: torch.Tensor

    def launch(self) -> None:
        if self.prepared_move_index_pack is not None:
            self.prepared_move_index_pack()
        for group in self.cpp_launch_groups:
            group.launch(
                self.move_source_indices, self.move_source_offsets, self.destination_bases
            )


def _cuda_int32_contiguous(tensors: Tuple[torch.Tensor, ...], device: torch.device) -> bool:
    return all(
        tensor.is_cuda
        and tensor.dtype == torch.int32
        and tensor.device == device
        and tensor.is_contiguous()
        for tensor in tensors
    )


def _validated_kv_head_count(
    pools: List[torch.Tensor],
    layers: Tuple[int, ...],
    device: torch.device,
    what: str,
) -> int:
    """Return the common KV-head count of one launch side's pools.

    The C++ compact op reads the interleaved V2 layout
    ``[page, K/V, head, token, dim]`` and takes the KV-head count from each
    launch's pool shape, so every layer on one side must agree on it.
    """
    first = pools[layers[0]]
    if first.ndim != 5 or first.shape[2] <= 0:
        raise ValueError(
            f"{what} pools must be 5-D interleaved V2 pools "
            f"[pages, K/V, heads, tokens, dim]; layer {layers[0]} has shape "
            f"{tuple(first.shape)}"
        )
    num_kv_heads = int(first.shape[2])
    if not all(
        pools[layer].ndim == 5
        and pools[layer].shape[1] == 2
        and pools[layer].device == device
        and int(pools[layer].shape[2]) == num_kv_heads
        and pools[layer].is_contiguous()
        and pools[layer].dtype in _SUPPORTED_POOL_DTYPES
        for layer in layers
    ):
        raise ValueError(
            f"{what} requires contiguous interleaved BF16/FP16/FP32 pools "
            "with one common KV-head count"
        )
    return num_kv_heads


def _make_move_buffers(
    index_prefix: Tuple[int, ...],
    moves_per_request: List[int],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Allocate the packed source-index buffer and its per-request offsets."""
    offsets = [0]
    for count in moves_per_request:
        offsets.append(offsets[-1] + count)
    indices = torch.empty((*index_prefix, offsets[-1]), dtype=torch.int32, device=device)
    return indices, torch.tensor(offsets, dtype=torch.int32, device=device)


def _validated_tail_lengths(
    tail_lengths: Optional[List[int]],
    request_count: int,
    what: str,
) -> Tuple[int, ...]:
    if tail_lengths is None:
        tail_lengths = [0] * request_count
    if len(tail_lengths) != request_count:
        raise ValueError(f"{what} lengths must match the request count")
    if any(length < 0 for length in tail_lengths):
        raise ValueError(f"{what} lengths must be non-negative")
    return tuple(int(length) for length in tail_lengths)


def _page_table_provider(
    page_table_slots: Dict[int, int],
    kv_block_offsets: torch.Tensor,
    device: torch.device,
    request_count: int,
    what: str,
) -> Callable[[int], torch.Tensor]:
    """Return validated per-slot K block-offset views, cached per slot."""
    tables: Dict[int, torch.Tensor] = {}

    def page_table_for(representative: int) -> torch.Tensor:
        slot = page_table_slots[representative]
        if slot not in tables:
            block_offsets = kv_block_offsets[slot, :request_count, 0]
            if block_offsets.device != device or block_offsets.dtype != torch.int32:
                raise ValueError(f"{what} block offsets must be int32 tensors on the pool device")
            if block_offsets.ndim != 2 or block_offsets.stride(1) != 1:
                raise ValueError(f"{what} K block offsets must have a contiguous block dimension")
            tables[slot] = block_offsets
        return tables[slot]

    return page_table_for


def _compact_groups(
    entries: List[Tuple[int, torch.Tensor, torch.Tensor]],
    pool_keys: Tuple[object, ...],
    device: torch.device,
    per_layer_slots: Optional[Dict[int, int]] = None,
) -> Tuple[_CppCompactGroup, ...]:
    """Batch layers into one C++ launch per uniform V2 pool.

    ``per_layer_slots`` maps each layer to its selection row; it is only set
    when every dense layer keeps its own token set (per-layer eviction).
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
        if len({int(pool.data_ptr()) for pool in pools}) != len(pools):
            raise ValueError("layered compaction requires a distinct pool view for every layer")
        if len({int(page_table.data_ptr()) for page_table in page_tables}) != 1:
            raise ValueError("layers in one V2 pool must share one block-offset table")
        source_layer_indices = None
        if per_layer_slots is not None:
            source_layer_indices = torch.tensor(
                [per_layer_slots[layer] for layer in layers],
                dtype=torch.int32,
                device=device,
            )
        result.append(
            _CppCompactGroup(
                pools=pools,
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


def _prepared_move_index_pack_launch(
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
) -> PreparedTritonKernelLaunch:
    """Build one prepared launch of the move-index packing kernel.

    The kernel reads the kept-token ordinals and each request's valid length
    and writes the packed per-(layer, head) move source indices consumed by
    the C++ compact launches. Only the caller-provided selection tensors are
    validated here; the move buffers are allocated by this module.
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
    selection_prefix = (request_count,) if union else (request_count, selection_rows)
    # Selection rows carry decode-only kept ordinals (already absolute), so
    # the rectangle is prompt-length independent.
    expected_selection = (*selection_prefix, decode_keep_count)
    if (
        request_count <= 0
        or tuple(kept_token_ordinals.shape) != expected_selection
        or valid_sequence_lengths.shape != (request_count,)
    ):
        raise ValueError(
            f"prepared compaction packing expects kept ordinals of shape "
            f"{expected_selection} and one valid length per request; got "
            f"{tuple(kept_token_ordinals.shape)} and "
            f"{tuple(valid_sequence_lengths.shape)}"
        )

    device = kept_token_ordinals.device
    if not _cuda_int32_contiguous((kept_token_ordinals, valid_sequence_lengths), device):
        raise ValueError("prepared compaction packing requires contiguous CUDA int32 tensors")

    if swa_move_source_indices is not None:
        swa_offsets_arg = swa_move_source_offsets
        swa_indices_arg = swa_move_source_indices
        swa_total = int(swa_move_source_indices.shape[-1])
    else:
        # HAS_SWA specializes all corresponding loads and stores away.
        swa_offsets_arg = move_source_offsets
        swa_indices_arg = move_source_indices
        swa_total = 0

    from .triattention_kernels import _pack_compaction_sources_kernel

    max_move = decode_keep_count + max_protected_tail
    if swa_total:
        max_move = max(max_move, swa_window + max_protected_tail)
    packed_row_count = num_dense_layers * num_kv_heads if per_layer else num_kv_heads
    grid = (
        request_count,
        packed_row_count,
        (max_move + _PACK_BLOCK_TOKENS - 1) // _PACK_BLOCK_TOKENS,
    )
    bound_tensors = (
        kept_token_ordinals,
        valid_sequence_lengths,
        move_source_offsets,
        move_source_indices,
        swa_offsets_arg,
        swa_indices_arg,
    )
    # Ordered to match the kernel's constexpr parameter declaration: the
    # prepared launch replays these by position.
    constexpr_values = dict(
        DENSE_TOTAL=int(move_source_indices.shape[-1]),
        SWA_TOTAL=swa_total,
        SELECTION_ROWS=selection_rows,
        SELECTION_STRIDE=decode_keep_count,
        KEEP_COUNT=decode_keep_count,
        NUM_KV_HEADS=num_kv_heads,
        SWA_WINDOW=swa_window,
        UNION=union,
        PER_LAYER=per_layer,
        HAS_SWA=swa_total > 0,
        BLOCK=_PACK_BLOCK_TOKENS,
    )
    return PreparedTritonKernelLaunch(
        _pack_compaction_sources_kernel,
        bound_tensors,
        constexpr_values,
        grid=grid,
        num_warps=_PACK_NUM_WARPS,
    )


class BatchedKVCacheCompaction:
    """Batched physical compaction of the KV caches for one fixed geometry.

    Dense layers keep the prompt in place and compact the selected decode
    tokens plus any target KV reserved for the next overlapped forward;
    kernel-masked SWA layers keep the latest window plus the same protected
    tail. A co-compressed draft cache reuses the target's kept token ordinals
    (broadcast over the draft's own KV-head count) plus the draft's own
    protected tail, landing at the same destination base.

    Key constructor inputs:
        `kept_token_ordinals`: increasing kept decode ordinals (absolute
            positions) per request; shape `[requests, keep]` for `union`,
            with a selection-row dimension in between for the per-head
            modes. Prompt tokens never move, so the rectangle is
            prompt-length independent and one cohort may mix prompt sizes;
            `prompt_offsets` carries each request's pinned prompt length.
        `kv_block_offsets`: the staged V2 block-offset snapshot laid out as
            `[slot, request, K/V, block]`, where a block offset encodes
            page and K/V plane as `2*page + plane`.
        `page_table_slots` / `layer_group_representative`: map each layer's
            group representative to its snapshot slot; layers that share a
            slot must share one block-offset table.
        `protected_tail_lengths`: per-request KV positions past the valid
            length reserved for a forward already in flight; they move with
            the kept tokens.
        `draft_*`: co-compressed draft-cache layout (union mode only); the
            draft reuses the target keep set and pins the same prompt.
    """

    def __init__(
        self,
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
        protected_tail_lengths: Optional[List[int]] = None,
        layer_pool_keys: Optional[List[object]] = None,
        draft_layer_pools: Optional[List[torch.Tensor]] = None,
        draft_layers: Optional[List[int]] = None,
        draft_layer_group_representative: Optional[Dict[int, int]] = None,
        draft_layer_pool_keys: Optional[List[object]] = None,
        draft_protected_tail_lengths: Optional[List[int]] = None,
        draft_kv_block_offsets: Optional[torch.Tensor] = None,
        draft_page_table_slots: Optional[Dict[int, int]] = None,
    ) -> None:
        if eviction_mode not in ("union", "per_head", "per_layer_perhead"):
            raise ValueError(f"unsupported compaction mode: {eviction_mode}")
        if request_count <= 0 or decode_keep_count <= 0:
            raise ValueError("batched compaction requires requests and retained tokens")
        if not dense_layers:
            raise ValueError("batched compaction requires at least one dense layer")
        if draft_layers and eviction_mode != "union":
            raise ValueError("draft co-compaction supports only union eviction")
        if draft_layer_pools is not None and not draft_layers:
            raise ValueError("draft pools were given without any draft layers")
        if not swa_layers and swa_window:
            raise ValueError("swa_window was given without any SWA layers")

        self.eviction_mode = eviction_mode
        self.device = layer_pools[dense_layers[0]].device
        # The move buffers are allocated on the pool device, so the selection
        # tensors feeding the pack kernel must already live there.
        if kept_token_ordinals.device != self.device:
            raise ValueError("kept-token ordinals must live on the pool device")
        self.request_count = int(request_count)
        # Per-request pinned prompt lengths; this usually aliases the staged
        # prompt buffer, so the values track the current round. Only the
        # geometry is validated here.
        if (
            prompt_offsets.shape != (self.request_count,)
            or prompt_offsets.dtype != torch.int32
            or prompt_offsets.device != self.device
            or not prompt_offsets.is_contiguous()
        ):
            raise ValueError("per-request prompt offsets do not match the cohort")
        self.prompt_offsets = prompt_offsets
        self.decode_keep_count = int(decode_keep_count)
        self.protected_tail_lengths = _validated_tail_lengths(
            protected_tail_lengths, self.request_count, "protected-tail"
        )
        self.max_protected_tail = max(self.protected_tail_lengths, default=0)
        self.dense_layers = tuple(int(layer) for layer in dense_layers)
        self.swa_layers = tuple(int(layer) for layer in swa_layers)
        if layer_pool_keys is None:
            layer_pool_keys = [("layer", layer) for layer in range(len(layer_pools))]
        if len(layer_pool_keys) != len(layer_pools):
            raise ValueError("pool keys must match the layer-pool count")
        self.layer_pool_keys = tuple(layer_pool_keys)

        per_layer = self.eviction_mode == "per_layer_perhead"
        self.num_kv_heads = _validated_kv_head_count(
            layer_pools,
            (*self.dense_layers, *self.swa_layers),
            self.device,
            "batched compaction",
        )
        dense_index_prefix = (
            (len(self.dense_layers), self.num_kv_heads) if per_layer else (self.num_kv_heads,)
        )
        dense_move_indices, dense_move_offsets = _make_move_buffers(
            dense_index_prefix,
            [self.decode_keep_count + length for length in self.protected_tail_lengths],
            self.device,
        )
        page_table_for = _page_table_provider(
            page_table_slots,
            kv_block_offsets,
            self.device,
            self.request_count,
            "compaction",
        )
        dense_entries = [
            (layer, layer_pools[layer], page_table_for(layer_group_representative[layer]))
            for layer in self.dense_layers
        ]

        self.swa_window = 0
        self.swa_destination_bases = None
        swa_move_indices = None
        swa_move_offsets = None
        swa_entries = []
        if self.swa_layers:
            if swa_window is None or swa_window <= 0:
                raise ValueError("SWA compaction requires a valid retained window")
            # Per-request window validity (prompt + decode keep >= window) is
            # prompt-dependent and checked by the caller each round.
            self.swa_window = int(swa_window)
            self.swa_destination_bases = torch.empty_like(self.prompt_offsets)
            swa_move_indices, swa_move_offsets = _make_move_buffers(
                (self.num_kv_heads,),
                [self.swa_window + length for length in self.protected_tail_lengths],
                self.device,
            )
            # SWA layers are staged as their own page-table representatives.
            swa_entries = [
                (layer, layer_pools[layer], page_table_for(layer)) for layer in self.swa_layers
            ]

        dense_slots = (
            {layer: slot for slot, layer in enumerate(self.dense_layers)} if per_layer else None
        )
        dense_pack = _prepared_move_index_pack_launch(
            kept_token_ordinals,
            valid_sequence_lengths,
            dense_move_offsets,
            dense_move_indices,
            eviction_mode=self.eviction_mode,
            decode_keep_count=self.decode_keep_count,
            num_dense_layers=len(self.dense_layers),
            num_kv_heads=self.num_kv_heads,
            max_protected_tail=self.max_protected_tail,
            swa_window=self.swa_window,
            swa_move_source_offsets=swa_move_offsets,
            swa_move_source_indices=swa_move_indices,
        )
        self.target_dense_compaction = _SingleCacheCompaction(
            prepared_move_index_pack=dense_pack,
            cpp_launch_groups=_compact_groups(
                dense_entries, self.layer_pool_keys, self.device, dense_slots
            ),
            move_source_indices=dense_move_indices,
            move_source_offsets=dense_move_offsets,
            destination_bases=self.prompt_offsets,
        )
        # The dense pack launch fills the SWA move buffers in the same call.
        self.target_swa_compaction = None
        if self.swa_layers:
            self.target_swa_compaction = _SingleCacheCompaction(
                prepared_move_index_pack=None,
                cpp_launch_groups=_compact_groups(swa_entries, self.layer_pool_keys, self.device),
                move_source_indices=swa_move_indices,
                move_source_offsets=swa_move_offsets,
                destination_bases=self.swa_destination_bases,
            )

        self.draft_compaction = None
        if draft_layers:
            self.draft_compaction = self._build_draft_compaction(
                kept_token_ordinals,
                valid_sequence_lengths,
                draft_layer_pools=draft_layer_pools,
                draft_layers=draft_layers,
                draft_layer_group_representative=draft_layer_group_representative,
                draft_layer_pool_keys=draft_layer_pool_keys,
                draft_protected_tail_lengths=draft_protected_tail_lengths,
                draft_kv_block_offsets=draft_kv_block_offsets,
                draft_page_table_slots=draft_page_table_slots,
            )

        self.cache_compactions = tuple(
            compaction
            for compaction in (
                self.target_dense_compaction,
                self.target_swa_compaction,
                self.draft_compaction,
            )
            if compaction is not None
        )

    def _build_draft_compaction(
        self,
        kept_token_ordinals: torch.Tensor,
        valid_sequence_lengths: torch.Tensor,
        *,
        draft_layer_pools: Optional[List[torch.Tensor]],
        draft_layers: List[int],
        draft_layer_group_representative: Optional[Dict[int, int]],
        draft_layer_pool_keys: Optional[List[object]],
        draft_protected_tail_lengths: Optional[List[int]],
        draft_kv_block_offsets: Optional[torch.Tensor],
        draft_page_table_slots: Optional[Dict[int, int]],
    ) -> _SingleCacheCompaction:
        """Build the co-compressed draft cache's own pack and launch groups.

        The draft forms its own launch groups so it may use a different
        KV-head count than the target. Union-only eviction is enforced by
        the constructor before dense groups are built.
        """
        if (
            draft_layer_pools is None
            or draft_layer_group_representative is None
            or draft_layer_pool_keys is None
        ):
            raise ValueError("draft co-compaction requires the full draft layout")
        if draft_kv_block_offsets is None or draft_page_table_slots is None:
            raise ValueError("draft co-compaction requires staged draft page tables")
        draft_tail_lengths = _validated_tail_lengths(
            draft_protected_tail_lengths, self.request_count, "draft protected-tail"
        )
        if len(draft_layer_pool_keys) != len(draft_layer_pools):
            raise ValueError("draft pool keys must match the draft layer-pool count")
        draft_layers = tuple(int(layer) for layer in draft_layers)
        draft_num_kv_heads = _validated_kv_head_count(
            draft_layer_pools,
            draft_layers,
            self.device,
            "draft co-compaction",
        )
        draft_move_indices, draft_move_offsets = _make_move_buffers(
            (draft_num_kv_heads,),
            [self.decode_keep_count + length for length in draft_tail_lengths],
            self.device,
        )
        draft_page_table_for = _page_table_provider(
            draft_page_table_slots,
            draft_kv_block_offsets,
            self.device,
            self.request_count,
            "draft",
        )
        draft_entries = [
            (
                layer,
                draft_layer_pools[layer],
                draft_page_table_for(draft_layer_group_representative[layer]),
            )
            for layer in draft_layers
        ]
        # In union mode the pack kernel reads selection row 0 for every
        # packed row, so one more prepared launch broadcasts the target keep
        # set over the draft KV heads and appends the draft's own tail
        # ordinals (valid_seq_len + 0..tail-1).
        draft_pack = _prepared_move_index_pack_launch(
            kept_token_ordinals,
            valid_sequence_lengths,
            draft_move_offsets,
            draft_move_indices,
            eviction_mode="union",
            decode_keep_count=self.decode_keep_count,
            num_dense_layers=1,
            num_kv_heads=draft_num_kv_heads,
            max_protected_tail=max(draft_tail_lengths, default=0),
            swa_window=0,
            swa_move_source_offsets=None,
            swa_move_source_indices=None,
        )
        return _SingleCacheCompaction(
            prepared_move_index_pack=draft_pack,
            cpp_launch_groups=_compact_groups(
                draft_entries, tuple(draft_layer_pool_keys), self.device
            ),
            move_source_indices=draft_move_indices,
            move_source_offsets=draft_move_offsets,
            destination_bases=self.prompt_offsets,
        )

    def launch(self) -> None:
        """Pack the move indices, then run every cache family's C++ compacts."""
        if self.swa_destination_bases is not None:
            # The prompt offsets may have been re-staged since construction;
            # rebase the SWA landing positions for this round.
            torch.add(
                self.prompt_offsets,
                self.decode_keep_count - self.swa_window,
                out=self.swa_destination_bases,
            )
        for compaction in self.cache_compactions:
            compaction.launch()
