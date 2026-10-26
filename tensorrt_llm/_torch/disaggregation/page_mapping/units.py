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
"""Which units of a request move, by the unit of their layer group (``LayerGroupDesc.unit``).

A "block" group's unit ids are blocks by ordinal (``CacheKind.PAGED``): ordinal ``i`` covers tokens
``[i * tokens_per_page, (i+1) * tokens_per_page)`` and both sides index their per-request tables by
ordinal. A "slot" group (``CacheKind.STATE``) holds one slot per request with every layer's state at
the end of the prompt, so it moves once, after the last chunk. Nothing here depends on the model;
how bytes inside a unit correspond is ``correspondence``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional, Set, Tuple

import numpy as np

from tensorrt_llm import logger
from tensorrt_llm._torch.disaggregation.base import CacheKind
from tensorrt_llm._torch.pyexecutor.kv_cache.page_layout import LayerGroupDesc

from .rank_pages import RankPages
from .topology import shard_grid

if TYPE_CHECKING:
    from tensorrt_llm._torch.disaggregation.native.rank_info import RankInfo

EMPTY = np.array([], dtype=np.int64)


@dataclass(frozen=True)
class LocalUnitsContext:
    """What :func:`local_units` may consult to list a request's units on this rank."""

    request: Any
    group_index: int
    pages: RankPages
    reuse_adapter: Any
    transfer_len: int
    """Prompt tokens the request moves (``KVCacheManagerV2.get_transfer_length``)."""
    cached_tokens: int
    """Block-aligned prefix the receiver already holds for this layer group (0 on the sender)."""


@dataclass(frozen=True)
class ChunkContext:
    """One prompt chunk, in block ordinals."""

    start_block: int
    end_block: int
    is_last: bool
    prompt_len: int


@dataclass(frozen=True)
class PairContext:
    """What :func:`pair_units` may consult when pairing one request's units."""

    self_ri: "RankInfo"
    peer_ri: "RankInfo"
    group: LayerGroupDesc
    """The sender's layer group being paired."""
    dst_start_token: Optional[int]


@dataclass(frozen=True)
class RegistrationRegion:
    """One device allocation to register with the transfer agent.

    ``within_known`` lets a region be skipped when it lies inside one already registered (pools
    that share a single allocation, registered lowest base first).
    """

    base_address: int
    size: int
    within_known: bool = False


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def local_units(ctx: LocalUnitsContext) -> np.ndarray:
    """This rank's unit ids for one request in one layer group; ``-1`` where it holds none.

    PAGED: ``ceil(transfer_len / tpb)`` ordinals, padded where fewer blocks are allocated than the
    prompt spans (incremental context allocation), without the speculative / first-token tail, and
    with the receiver's cached prefix masked. Block lists carry beam 0 only: beam-search attention
    reads prompt positions through beam 0. STATE: the request's slot in the group, or none.
    """
    if ctx.pages.kind(ctx.group_index) == CacheKind.STATE:
        slot = ctx.reuse_adapter.get_state_slot(ctx.request, ctx.group_index)
        return EMPTY if slot < 0 else np.array([slot], dtype=np.int64)
    tpb = ctx.reuse_adapter.tokens_per_block
    prompt_blocks = (ctx.transfer_len + tpb - 1) // tpb
    ordinals = ctx.reuse_adapter.get_block_ordinals(ctx.request, ctx.group_index)
    window = np.full(prompt_blocks, -1, dtype=np.int64)
    n = min(int(ordinals.size), prompt_blocks)
    window[:n] = ordinals[:n]
    window[: min(ctx.cached_tokens // tpb, prompt_blocks)] = -1
    return window


def chunk_units(pages: RankPages, g: int, units: np.ndarray, ctx: ChunkContext) -> np.ndarray:
    """The part of :func:`local_units` one prompt chunk sends; ``-1`` blanks the rest."""
    window = pages.group(g).kept_window
    if pages.kind(g) == CacheKind.STATE or (window is not None and window < ctx.prompt_len):
        # State holds the end of the prompt; SWA blocks can leave the active window between chunks.
        # Either moves once, with the last chunk.
        return units if ctx.is_last else np.full_like(units, -1)
    # Positional: keep the chunk's ordinals, blank everything else.
    chunk = np.full_like(units, -1)
    chunk[ctx.start_block : ctx.end_block] = units[ctx.start_block : ctx.end_block]
    return chunk


def pair_ordinals(
    src_units: np.ndarray, dst_units: np.ndarray, dst_start_block: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """Pair positional block tables ordinal by ordinal.

    Both tables are indexed by block ordinal (see ``Chunk``); ``-1`` marks an ordinal that side has
    nothing for. A block moves only where both sides hold it, so SWA eviction, prefix reuse, a wider
    speculative window and a chunk's range bounds all fall out of one intersection. Ordinals below
    ``dst_start_block`` are excluded as well.
    """
    if src_units.size != dst_units.size:
        raise ValueError(
            f"positional block tables differ in length: src={src_units.size} "
            f"dst={dst_units.size}; both sides must span the same prompt"
        )
    mask = (src_units >= 0) & (dst_units >= 0)
    if dst_start_block > 0:
        mask[:dst_start_block] = False
    idx = np.flatnonzero(mask)
    return src_units[idx], dst_units[idx]


def pair_units(
    src_units: np.ndarray, dst_units: np.ndarray, ctx: PairContext
) -> Tuple[np.ndarray, np.ndarray]:
    """Equal-length sender / receiver unit ids that move; both empty when nothing does."""
    src_units = np.asarray(src_units, dtype=np.int64)
    dst_units = np.asarray(dst_units, dtype=np.int64)
    if ctx.group.unit == "slot":
        if src_units.size == 0 or dst_units.size == 0:
            return EMPTY, EMPTY
        return src_units[:1], dst_units[:1]
    if ctx.group.blocks_by_cp and ctx.peer_ri.cp_size > 1 and ctx.self_ri.cp_size == 1:
        # Helix: the receiver owns global blocks [cp_rank::cp_size] (same protocol as
        # partition_context_for_helix), so its table is the strided subset of ours; block reuse is
        # rejected under helix.
        src_units = src_units[ctx.peer_ri.cp_rank :: ctx.peer_ri.cp_size]
    tokens_per_page = ctx.self_ri.pages.layout.tokens_per_page
    dst_start_block = (
        ctx.dst_start_token // tokens_per_page if ctx.dst_start_token is not None else 0
    )
    return pair_ordinals(src_units, dst_units, dst_start_block=dst_start_block)


def transfer_bytes(kind: CacheKind, buffer_bytes: int, units: np.ndarray) -> int:
    """Bytes a ``buffer_bytes`` buffer moves for ``units``: once per block, or once per request."""
    if kind == CacheKind.STATE:
        return buffer_bytes if bool((units >= 0).any()) else 0
    return int((units >= 0).sum()) * buffer_bytes


def sends_incrementally(kind: CacheKind) -> bool:
    """Whether a layer group's units can move in token chunks while the prompt is still running."""
    return kind == CacheKind.PAGED


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------


def registration_regions(pages: RankPages, g: int) -> List[RegistrationRegion]:
    """Device allocations backing one layer group, to register with the transfer agent."""
    pools = pages.group_pools(g)
    if pages.kind(g) == CacheKind.PAGED:
        # Each pool is an independent allocation spanning all of its slots.
        return [RegistrationRegion(p.base, p.num_slots * p.slot_bytes) for p in pools]
    # State roles may share one allocation (interleaved, equal sizes) or be separate. Lowest base
    # first, so a covering region registers before roles that fall inside it.
    return [
        RegistrationRegion(p.base, p.num_slots * p.slot_bytes, True)
        for p in sorted(pools, key=lambda p: p.base)
    ]


def bounce_unit_bytes(pages: RankPages, g: int) -> Optional[int]:
    """Bytes per unit the bounce buffer reserves per ordinal of a PAGED group; None for a STATE
    group, sized by :func:`receiver_extra_bytes` instead."""
    if pages.kind(g) == CacheKind.STATE:
        return None
    return sum(p.slot_bytes for p in pages.group_pools(g))


def _state_keys(pages: RankPages) -> Set[Tuple[int, str]]:
    return {
        (buffer.layer, buffer.role)
        for g in range(pages.num_layer_groups)
        if pages.kind(g) == CacheKind.STATE
        for buffer in pages.buffers(g)
    }


def receiver_extra_bytes(
    sender_pages: RankPages, receiver_pages: RankPages, g: int, dst_units: np.ndarray
) -> int:
    """Bytes landing in a STATE group's slot on the receiver: its own buffer bytes over buffers the
    sender holds. Receiver-local, whatever the sender-side shard pairing, so it needs no simulation
    of which sender ranks write (RankInfoServer only describes rank 0)."""
    if receiver_pages.kind(g) == CacheKind.PAGED or dst_units.size == 0:
        return 0
    sender_keys = _state_keys(sender_pages)
    return sum(
        buffer.size
        for buffer in receiver_pages.buffers(g)
        if (buffer.layer, buffer.role) in sender_keys
    )


# ---------------------------------------------------------------------------
# Compatibility
# ---------------------------------------------------------------------------


def log_incompatible(reason: str, **details) -> None:
    text = ", ".join(f"{k}={v!r}" for k, v in details.items())
    logger.warning("%s", f"incompatible peer: {reason}" + (f"; {text}" if text else ""))


def check_rank_compatible(self_ri: "RankInfo", peer_ri: "RankInfo") -> bool:
    """Whether the two ranks' block ordinals can correspond; logs why and returns False when not.

    Block ordinal ``i`` must cover the same tokens on both sides, so a rank holding block units
    pairs only with a peer of the same ``tokens_per_page``. Helix pairs a context instance without
    context parallelism with a generation instance using it, never two context-parallel instances.
    How bytes inside a unit correspond is checked buffer by buffer
    (``correspondence.validate_peer_compatible``).
    """
    if self_ri.cp_size != 1 and peer_ri.cp_size != 1:
        log_incompatible(
            "cp_size must be 1 on at least one side (helix pairs a cp=1 context instance "
            "with a cp=N generation instance)",
            local=self_ri.cp_size,
            peer=peer_ri.cp_size,
        )
        return False
    pages, peer_pages = self_ri.pages, peer_ri.pages
    if pages is None or peer_pages is None:
        return True
    has_blocks = any(d.unit == "block" for d in pages.layout.layer_groups) and any(
        d.unit == "block" for d in peer_pages.layout.layer_groups
    )
    local, peer = pages.layout.tokens_per_page, peer_pages.layout.tokens_per_page
    if has_blocks and local != peer:
        log_incompatible(
            "tokens_per_page mismatch: block ordinals would cover different tokens",
            local=local,
            peer=peer,
        )
        return False
    return True


def validate_state(self_pages: RankPages, self_ri: "RankInfo", peer_ri: "RankInfo", pairs) -> None:
    """Reject a peer whose recurrent state cannot be resharded into ours.

    Analogue of the C++ ``rnnCacheFormatter inquireSupport`` gate. The core invariant is the
    *global* (grid-aggregated) state size: for a sharded segment ``per_rank_bytes * grid`` is
    TP-invariant, so it must match even when the two sides' TP differ. A model that keeps a
    replicated full-size state while sharding over a grid > 1 violates it under heterogeneous TP --
    exactly where resharding would read past the slot -- and is rejected here.
    """
    for g, buffer, pg, peer_buffer in pairs:
        if self_pages.kind(g) != CacheKind.STATE:
            continue
        role = f"role {buffer.role!r} (layer {buffer.layer})"
        for i, (a, b) in enumerate(zip(buffer.segments, peer_buffer.segments)):
            if a.split_axis is None:
                if a.nbytes != b.nbytes:
                    raise ValueError(
                        f"replicated state size differs for {role} segment {i} "
                        f"(local={a.nbytes}, peer={b.nbytes})"
                    )
                continue
            grid = shard_grid(self_pages.group(g).grid, self_ri).size
            peer_grid = shard_grid(peer_ri.pages.group(pg).grid, peer_ri).size
            if a.nbytes * grid != b.nbytes * peer_grid:
                raise ValueError(
                    f"global (TP-aggregated) size of {role} segment {i} differs: local "
                    f"{a.nbytes} bytes/rank x state grid {grid} vs peer {b.nbytes} bytes/rank x "
                    f"state grid {peer_grid}. Per-rank state sizes are inconsistent with a sharded "
                    "layout across the two sides; either the state shape/dtype differs, or the "
                    "model keeps a replicated (non-sharded) per-rank recurrent state while "
                    "sharding over a grid > 1, which supports heterogeneous ctx/gen TP only with "
                    "attention-DP enabled on both sides."
                )
            if a.split_axis == "head" and a.unit_bytes != b.unit_bytes:
                # TP-invariant head_dim * d_state * element_size: a different state shape or dtype.
                raise ValueError(
                    f"bytes per head of {role} differ (local={a.unit_bytes}, "
                    f"peer={b.unit_bytes}); check head_dim / d_state / mamba_ssm_cache_dtype"
                )


def warn_row_fragments(fragments_per_unit: int, local_split: int, peer_split: int) -> None:
    """Buffers split differently on the two sides move one fragment per row; say so once."""
    if fragments_per_unit:
        logger.warning_once(
            "Disaggregated KV transfer between ranks that split their buffers differently "
            f"moves approximately {fragments_per_unit} NIXL descriptors per transferred unit "
            "per peer, one per row of each multi-row segment "
            f"(local split extent={local_split}, peer split extent={peer_split}). "
            "Long-context TEP/DEP transfers may have high latency.",
            key=f"native-row-fragments-{local_split}-{peer_split}-{fragments_per_unit}",
        )
