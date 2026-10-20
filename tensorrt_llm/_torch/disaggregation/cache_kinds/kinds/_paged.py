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
"""Rules shared by paged units: blocks indexed by ordinal.

Block ordinal ``i`` covers tokens ``[i * tpb, (i+1) * tpb)`` and both sides index their per-request
tables by ordinal. K/V buffers are sharded by KV head over attention TP (HND: head-major, one slice
per buffer; NHD: token-major, one slice per token); replicated roles move whole from one elected
sender.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from tensorrt_llm import logger

from ..correspondence import log_incompatible
from ..page_table import BufferDecl, PageTable, UnitSpace
from ..spec import ChunkContext, LocalUnitsContext, PairContext, RegistrationRegion
from ..topology import ATTENTION_TP

HEAD_MAJOR = "HND"
"""K/V stored ``[head, token, dim]`` per block: a head range is one contiguous slice per buffer."""

TOKEN_MAJOR = "NHD"
"""K/V stored ``[token, head, dim]`` per block: a head range is a slice inside every token."""


def kv_heads_per_rank(manager) -> int:
    """The rank's KV heads, as ``RankInfo.attention`` reports them (0: no attention cache)."""
    return next((h for h in manager.num_kv_heads_per_layer if h > 0), 0)


def params(tokens_per_block: int, window: Optional[int]) -> Dict:
    return {"tokens_per_block": int(tokens_per_block), "window": window}


def replicated(nbytes: int) -> BufferDecl:
    """A buffer every attention-TP rank holds identically (index-key caches, MLA side tails)."""
    return BufferDecl(nbytes=nbytes, domain=ATTENTION_TP)


def kv(nbytes: int, heads: int, tokens_per_block: int, layout: str, context: str) -> BufferDecl:
    """A K/V buffer sharded by KV head over attention TP, in ``HEAD_MAJOR`` or ``TOKEN_MAJOR``."""
    if heads <= 0:
        raise ValueError(f"{context}: attention role on a rank that reports no KV heads")
    if layout == TOKEN_MAJOR:
        axes: Tuple[Tuple[str, int], ...] = (("token", tokens_per_block), ("head", heads))
    elif layout == HEAD_MAJOR:
        axes = (("head", heads),)
    else:
        raise ValueError(f"{context}: unknown K/V layout {layout!r}")
    return BufferDecl(nbytes=nbytes, domain=ATTENTION_TP, axes=axes, split_axis="head")


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


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


def local_units(ctx: LocalUnitsContext) -> np.ndarray:
    """``ceil(transfer_len / tpb)`` ordinals: the local slot, or -1 where this side has nothing.

    Pads when fewer blocks are allocated than the prompt spans (incremental ctx allocation), drops
    the speculative / first-token tail, and masks the receiver's cached prefix. Block lists carry
    beam 0 only: beam-search attention reads prompt positions through beam 0.
    """
    tpb = ctx.reuse_adapter.tokens_per_block
    prompt_blocks = (ctx.transfer_len + tpb - 1) // tpb
    ordinals = ctx.reuse_adapter.get_block_ordinals(ctx.request, ctx.space_index)
    window = np.full(prompt_blocks, -1, dtype=np.int64)
    n = min(int(ordinals.size), prompt_blocks)
    window[:n] = ordinals[:n]
    window[: min(ctx.cached_tokens // tpb, prompt_blocks)] = -1
    return window


def chunk_units(space: UnitSpace, units: np.ndarray, ctx: ChunkContext) -> np.ndarray:
    window = space.params.get("window")
    if window is not None and window < ctx.prompt_len:
        # SWA blocks can leave the active window between chunks: defer the space and send its
        # complete final active window at once.
        return units if ctx.is_last else np.full_like(units, -1)
    # Positional: keep the chunk's ordinals, blank everything else.
    chunk = np.full_like(units, -1)
    chunk[ctx.start_block : ctx.end_block] = units[ctx.start_block : ctx.end_block]
    return chunk


def pair_units(
    src_units: np.ndarray, dst_units: np.ndarray, ctx: PairContext
) -> Tuple[np.ndarray, np.ndarray]:
    src_units = np.asarray(src_units, dtype=np.int64)
    dst_units = np.asarray(dst_units, dtype=np.int64)
    if ctx.peer_ri.cp_size > 1 and ctx.self_ri.cp_size == 1:
        # Helix: the receiver owns global blocks [cp_rank::cp_size] (same protocol as
        # partition_context_for_helix), so its table is the strided subset of ours; block reuse is
        # rejected under helix.
        src_units = src_units[ctx.peer_ri.cp_rank :: ctx.peer_ri.cp_size]
    tokens_per_block = ctx.space.params["tokens_per_block"]
    dst_start_block = (
        ctx.dst_start_token // tokens_per_block if ctx.dst_start_token is not None else 0
    )
    return pair_ordinals(src_units, dst_units, dst_start_block=dst_start_block)


def transfer_bytes(region_bytes: int, units: np.ndarray) -> int:
    return int((units >= 0).sum()) * region_bytes


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------


def registration_regions(page_table: PageTable, space_index: int) -> List[RegistrationRegion]:
    # Each pool is an independent allocation spanning all of its slots.
    regions = []
    for pool_idx in page_table.spaces[space_index].pools:
        pool = page_table.pools[pool_idx]
        regions.append(
            RegistrationRegion(pool.base_address, pool.num_slots * pool.slot_stride_bytes)
        )
    return regions


def bounce_unit_bytes(page_table: PageTable, space_index: int) -> int:
    # Several regions of one physical pool contribute that pool once.
    return sum(
        int(page_table.pools[pool_idx].slot_bytes)
        for pool_idx in page_table.spaces[space_index].pools
    )


# ---------------------------------------------------------------------------
# Compatibility
# ---------------------------------------------------------------------------


def _uses_exact_tpb_regions(ri) -> bool:
    """Token-major and replicated regions address bytes inside a block, so their geometry only
    lines up when both sides use the same tokens_per_block."""
    if ri.page_table is None:
        return False
    return any(
        region.replicated or any("token" in buf.axis_names for buf in region.buffers)
        for space in ri.page_table.spaces
        for region in space.regions
    )


def _tpb_incompatible(kind: str, self_ri, peer_ri) -> bool:
    local, peer = self_ri.attention.tokens_per_block, peer_ri.attention.tokens_per_block
    if local == peer:
        return False
    if _uses_exact_tpb_regions(self_ri) or _uses_exact_tpb_regions(peer_ri):
        log_incompatible(
            kind, "tokens_per_block mismatch for NHD/replicated pools", local=local, peer=peer
        )
        return True
    if max(local, peer) % min(local, peer) != 0:
        log_incompatible(kind, "tokens_per_block not divisible", local=local, peer=peer)
        return True
    logger.warning(
        f"{kind}: tokens_per_block mismatch (local={local}, peer={peer}); "
        "KV transfer proceeds — ensure block boundaries align during transfer."
    )
    return False


def check_rank_compatible(kind: str, self_ri, peer_ri) -> bool:
    """Attention topology both sides must share for paged blocks to correspond."""
    a, b = self_ri.attention, peer_ri.attention

    def fail(cond: bool, reason: str, **details) -> bool:
        if cond:
            log_incompatible(kind, reason, **details)
        return cond

    heads_match = a.is_mla or a.kv_heads_per_rank == b.kv_heads_per_rank
    return not (
        fail(a.is_mla != b.is_mla, "is_mla mismatch", local=a.is_mla, peer=b.is_mla)
        or fail(
            self_ri.cp_size != 1 and peer_ri.cp_size != 1,
            "cp_size must be 1 on at least one side (helix pairs a cp=1 context instance "
            "with a cp=N generation instance)",
            local=self_ri.cp_size,
            peer=peer_ri.cp_size,
        )
        or fail(
            (self_ri.cp_size != 1 or peer_ri.cp_size != 1)
            and a.tokens_per_block != b.tokens_per_block,
            "helix block-interleaved transfer requires equal tokens_per_block on both sides "
            "(block boundaries must coincide for [cp_rank::cp_size] ownership)",
            local=a.tokens_per_block,
            peer=b.tokens_per_block,
        )
        or fail(
            a.element_bytes != b.element_bytes,
            "element_bytes mismatch",
            local=a.element_bytes,
            peer=b.element_bytes,
        )
        or fail(
            not heads_match
            and not float(a.tokens_per_block * a.dims_per_head * a.element_bytes).is_integer(),
            "sub-byte head slicing is not byte-aligned",
            tokens_per_block=a.tokens_per_block,
            dims_per_head=a.dims_per_head,
            element_bytes=a.element_bytes,
        )
        or _tpb_incompatible(kind, self_ri, peer_ri)
        or fail(
            a.dims_per_head != b.dims_per_head,
            "dims_per_head mismatch",
            local=a.dims_per_head,
            peer=b.dims_per_head,
        )
        or fail(
            a.is_mla and (a.kv_heads_per_rank != 1 or b.kv_heads_per_rank != 1),
            "MLA requires kv_heads_per_rank == 1 for both ranks",
            local=a.kv_heads_per_rank,
            peer=b.kv_heads_per_rank,
        )
    )


def warn_token_major_fragments(self_ri, peer_ri, groups) -> None:
    """Head-mismatched token-major regions move one fragment per token; say so once."""
    a, b = self_ri.attention, peer_ri.attention
    if a is None or b is None or a.is_mla or a.kv_heads_per_rank == b.kv_heads_per_rank:
        return
    page_table = self_ri.page_table
    per_token = sum(
        1
        for group in groups
        for ri in group.regions
        for buf in page_table.spaces[group.self_space].regions[ri].buffers
        if "token" in buf.axis_names
    )
    if per_token:
        logger.warning_once(
            "NHD head-mismatched disaggregated KV transfer has no contiguous staging path and "
            f"will emit approximately {per_token} NIXL descriptors per transferred token per "
            "peer, excluding block-level replicated pools "
            f"(local_kv_heads={a.kv_heads_per_rank}, peer_kv_heads={b.kv_heads_per_rank}). "
            "Long-context TEP/DEP transfers may have high latency.",
            key=f"native-nhd-head-mismatch-{a.kv_heads_per_rank}-{b.kv_heads_per_rank}-{per_token}",
        )
