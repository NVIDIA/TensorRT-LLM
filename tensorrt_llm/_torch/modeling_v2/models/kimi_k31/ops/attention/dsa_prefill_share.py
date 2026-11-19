# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sharing the DSA indexer prefill of long-prefix chunks across attention-DP ranks.

Under attention DP the ranks run their own requests in lockstep, so a context step lasts as long as the rank with
the largest indexer work: its query rows x keys, which grows with the request's cached prefix. On the DEP perf
target (``_k31_perf`` metadata under attention DP, TP > 1, no Helix CP) on a server that runs context chunks (an
aggregated or a disaggregated context server; not a disaggregated generation server, ``TRTLLM_DISAGG_ROLE`` =
``generation`` as the serve launcher exports it) every rank offers up to ``_SHARE_SLOTS`` (2) indexer prefill chunks
with at least
``_SHARE_MIN_KEYS`` (200000) keys and ``_SHARE_MIN_ROWS`` (1024) query rows. The offers and the rank's total indexer
work ride on the step's attention-DP count exchange (``offer_counts`` / ``take_plan``, called by the metadata's
``get_adp_token_counts`` / ``set_adp_token_counts``); a rank's offers are taken only while its work exceeds
``_SHARE_IMBALANCE`` (1.25) times the step's mean.

The taken chunks are cut into pieces inside the owner's query tiles (the tiles the owner's own indexer would run:
``prefill_q_tile``): ``_SHARE_PIECE_ROWS`` (512) rows from each tile's start. 512 rows is the logits kernel's row
block: a launch costs the same for 128 or 512 rows, so pieces of whole blocks keep the kernel's per-row rate. The
pieces go, largest first, to the rank with the least work so far among the ranks below the step's mean work (each
owner's own load counted without its taken chunks), so the ranks with spare capacity absorb them and a loaded rank
that kept its own chunk takes none. Each indexer layer then:

1. all-gathers the taken chunks' indexer inputs from their owners: the chunk's gathered index keys and key scales,
   its index queries, head weights and row windows (``start_shared_indexer``, on a side stream while this rank
   scores its own chunks);
2. on every rank, runs the indexer (logits + prefill Top-K, the layer's own kernels) for its pieces;
3. all-gathers the Top-K rows scored for other owners; each owner writes its chunks' rows into the layer's Top-K
   buffer, the rows of the pieces it scored itself directly (``finish_shared_indexer``).

Query rows are independent, and every piece runs with the Top-K plan of its whole owner tile (``tier_rows``), so
the rows come back bit for bit as the owner alone would write them. A step shares nothing unless every rank runs a
context indexer in it (no rank skips it, none runs a CUDA graph or a query split); the plan is a host function of
the exchanged counts, the same on every rank. Elsewhere (the general target, attention TP, a disaggregated
generation server) the exchange carries no extra counts and the indexer is unchanged.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, List, Optional, Sequence, Tuple

import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.comm.allgather_module import (
    allgather_module,
)
from tensorrt_llm._utils import nvtx_range
from tensorrt_llm.logger import logger

if TYPE_CHECKING:
    from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention.dsa_indexer import K31Indexer
    from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention.dsa_metadata import (
        K31DSAtrtllmAttentionMetadata,
    )

#: The disaggregated worker role the serve launcher exports to every rank (unset on an aggregated server).
_DISAGG_ROLE_ENV = "TRTLLM_DISAGG_ROLE"


def disagg_role() -> Optional[str]:
    """The serving role the runtime published for this server's ranks, or None on an aggregated server.

    ``trtllm-serve --server_role`` and the disaggregated MPI launch export ``TRTLLM_DISAGG_ROLE`` (``context`` or
    ``generation``) to every rank, and the workers apply it before the model is built. The family's one reader of
    that variable: the targets and routing take the role from here.
    """
    return os.environ.get(_DISAGG_ROLE_ENV)


_ALIGN = 256
#: Unit of the exchanged indexer work: 2^20 query-key pairs.
_PAIRS_UNIT = 1 << 20

#: Indexer prefill chunks a rank offers per step.
_SHARE_SLOTS = 2
#: Keys and query rows an offered chunk has at least.
_SHARE_MIN_KEYS = 200000
_SHARE_MIN_ROWS = 1024
#: A rank's offered chunks are taken only while its indexer work exceeds this multiple of the step's mean over the
#: ranks, so a balanced step pays no communication.
_SHARE_IMBALANCE = 1.25
#: Rows of a piece within an owner tile.
_SHARE_PIECE_ROWS = 512


def owner_tile(rows: int, keys: int) -> int:
    """The owner's prefill query tile of a chunk: ``K31Indexer.prefill_q_tile`` (the logits element budget)."""
    from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention import dsa_indexer

    return max(1, min(rows, dsa_indexer._INDEXER_MQA_LOGITS_ELEM_BUDGET // max(1, keys)))


def num_descriptor_counts() -> int:
    """Extra per-rank counts in the attention-DP exchange: [eligible flag, indexer work in 2^20 query-key pairs] +
    (rows, keys) per slot."""
    return 2 + 2 * _SHARE_SLOTS


def _pad(n: int) -> int:
    return -(-n // _ALIGN) * _ALIGN


@dataclass
class ShareEntry:
    """One taken chunk: its owner rank, the owner's offer slot, its query rows and keys."""

    owner: int
    slot: int
    rows: int
    keys: int


@dataclass
class SharePiece:
    """Rows ``[lo, hi)`` of taken chunk ``entry``, inside an owner tile of ``tier`` rows, scored by ``rank``."""

    entry: int
    lo: int
    hi: int
    tier: int
    rank: int


@dataclass
class SharePlan:
    """The step's taken chunks and their pieces (the same on every rank), and this rank's indices of its own taken
    chunks into ``metadata.indexer_prefill_chunks``, by slot."""

    world: int
    rank: int
    entries: List[ShareEntry]
    local_chunks: List[int] = field(default_factory=list)
    pieces: List[SharePiece] = field(default_factory=list)

    def pieces_of_rank(self, r: int) -> List[SharePiece]:
        return [p for p in self.pieces if p.rank == r]

    def sent_by(self, r: int) -> List[SharePiece]:
        """The pieces rank ``r`` scores for other owners (an owner keeps the rows of its own pieces)."""
        return [p for p in self.pieces if p.rank == r and self.entries[p.entry].owner != r]

    def rows_sent_by(self, r: int) -> int:
        return sum(p.hi - p.lo for p in self.sent_by(r))


_LOGITS_ROW_BLOCK = 512


def chunk_pieces(rows: int, keys: int, piece_rows: int) -> List[Tuple[int, int, int]]:
    """``(lo, hi, rows of the enclosing owner tile)`` of a taken chunk's pieces, in row order."""
    tile = owner_tile(rows, keys)
    out = []
    for t0 in range(0, rows, tile):
        t1 = min(t0 + tile, rows)
        out += [(p0, min(p0 + piece_rows, t1), t1 - t0) for p0 in range(t0, t1, piece_rows)]
    return out


def _piece_cost(rows: int, keys: int) -> int:
    """Scoring cost of a piece in query-key pairs: the logits kernel runs whole 512-row blocks."""
    return -(-rows // _LOGITS_ROW_BLOCK) * _LOGITS_ROW_BLOCK * keys


def assign_pieces(
    entries: List[ShareEntry], work: Sequence[int], world: int, piece_rows: int
) -> List[SharePiece]:
    """The pieces of ``entries`` and their ranks: largest first, each to the least-loaded rank below the mean work
    (``work`` per rank in 2^20 query-key pairs; an owner's load counts without its taken chunks). Ties go to the
    lower rank, so every rank derives the same assignment."""
    load = [int(w) * _PAIRS_UNIT for w in work]
    for e in entries:
        load[e.owner] = max(0, load[e.owner] - e.rows * e.keys)
    mean = sum(int(w) for w in work) * _PAIRS_UNIT / world
    pieces = []
    for i, e in enumerate(entries):
        pieces += [(i, lo, hi, tier) for lo, hi, tier in chunk_pieces(e.rows, e.keys, piece_rows)]
    order = sorted(
        range(len(pieces)),
        key=lambda j: (-_piece_cost(pieces[j][2] - pieces[j][1], entries[pieces[j][0]].keys), j),
    )
    takers = [r for r in range(world) if load[r] < mean] or list(range(world))
    rank_of = [0] * len(pieces)
    for j in order:
        i, lo, hi, _ = pieces[j]
        r = min(takers, key=lambda t: (load[t], t))
        rank_of[j] = r
        load[r] += _piece_cost(hi - lo, entries[i].keys)
    return [SharePiece(i, lo, hi, tier, rank_of[j]) for j, (i, lo, hi, tier) in enumerate(pieces)]


def _metadata_shares(metadata: "K31DSAtrtllmAttentionMetadata") -> bool:
    """Whether the step shares: the DEP perf target's metadata (``_k31_perf``, attention DP over TP > 1 ranks, no
    Helix CP) on a server that runs context chunks, i.e. not a disaggregated generation server."""
    mapping = getattr(metadata, "mapping", None)
    return (
        getattr(metadata, "_k31_perf", False)
        and mapping is not None
        and mapping.enable_attention_dp
        and mapping.tp_size > 1
        and not mapping.has_cp_helix()
        and disagg_role() != "generation"
    )


def local_descriptors(metadata: "K31DSAtrtllmAttentionMetadata") -> Tuple[List[int], List[int]]:
    """This rank's [eligible, work, rows0, keys0, rows1, keys1, ...] and the offered chunk indices."""
    slots = _SHARE_SLOTS
    chunks = metadata.indexer_prefill_chunks or []
    eligible = int(
        metadata.num_contexts > 0
        and metadata.num_ctx_tokens > 0
        and not metadata.skip_indexer_for_ctx_reqs
        and metadata.indexer_prefill_q_split is None
        and not metadata.is_cuda_graph
    )
    cands = []
    work = 0
    if eligible:
        min_keys, min_rows = _SHARE_MIN_KEYS, _SHARE_MIN_ROWS
        for i, c in enumerate(chunks):
            rows = c.token_end - c.token_start
            keys = c.k_token_end - c.k_token_start
            work += rows * keys
            if keys >= min_keys and rows >= min_rows:
                cands.append((rows * keys, i, rows, keys))
        cands.sort(reverse=True)
        cands = cands[:slots]
        # Offered chunks keep their batch order (the all-gather layout).
        cands.sort(key=lambda t: t[1])
    desc = [eligible, -(-work // _PAIRS_UNIT)]
    for _, _, rows, keys in cands:
        desc += [rows, keys]
    desc += [0, 0] * (slots - len(cands))
    return desc, [i for _, i, _, _ in cands]


def build_plan(
    columns: Sequence[Sequence[int]], rank: int, local_chunks: List[int]
) -> Optional[SharePlan]:
    """The step's plan from the exchanged descriptor columns (``columns[i][r]`` = count ``i`` of rank ``r``)."""
    world = len(columns[0])
    if world <= 1 or not all(columns[0]):
        return None
    work = columns[1]
    bar = _SHARE_IMBALANCE * sum(work) / world
    entries = []
    for r in range(world):
        if work[r] <= bar:
            continue
        for s in range(_SHARE_SLOTS):
            rows, keys = columns[2 + 2 * s][r], columns[3 + 2 * s][r]
            if rows > 0:
                entries.append(ShareEntry(r, s, int(rows), int(keys)))
    if not entries:
        return None
    owned_slots = [e.slot for e in entries if e.owner == rank]
    pieces = assign_pieces(entries, work, world, _SHARE_PIECE_ROWS)
    logger.info_once(
        f"DSA prefill share active: {len(entries)} chunk(s) over {world} ranks "
        f"(rows x keys {[(e.rows, e.keys) for e in entries]}), {len(pieces)} piece(s) on ranks "
        f"{sorted({p.rank for p in pieces})}",
        key="k31_dsa_prefill_share_active",
    )
    return SharePlan(world, rank, entries, [local_chunks[s] for s in owned_slots], pieces)


def offer_counts(metadata: "K31DSAtrtllmAttentionMetadata") -> List[int]:
    """The counts this rank appends to the attention-DP exchange: none unless sharing is on."""
    metadata.indexer_share_plan = None
    if not _metadata_shares(metadata):
        return []
    desc, metadata._share_local_chunks = local_descriptors(metadata)
    return desc


def take_plan(
    metadata: "K31DSAtrtllmAttentionMetadata", columns: List[List[int]]
) -> List[List[int]]:
    """Strip this module's columns off the exchanged counts and set the step's plan; the columns unchanged unless
    sharing is on."""
    if not _metadata_shares(metadata):
        return columns
    n = num_descriptor_counts()
    metadata.indexer_share_plan = build_plan(
        columns[-n:], metadata.mapping.tp_rank, getattr(metadata, "_share_local_chunks", [])
    )
    return columns[:-n]


def score_piece(
    indexer: "K31Indexer",
    q: torch.Tensor,
    k_data: torch.Tensor,
    k_scale: torch.Tensor,
    weights: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    out: torch.Tensor,
    lo: int,
    hi: int,
    tier: int,
) -> None:
    """Prefill Top-K of rows ``[lo, hi)`` of a chunk (``q`` ... ``row_ends`` the whole chunk's rows), inside an owner
    tile of ``tier`` rows, into ``out`` (``hi - lo`` rows): one launch with the owner tile's Top-K plan."""
    indexer.forward_prepared(
        q[lo:hi],
        k_data,
        k_scale,
        weights[lo:hi],
        row_starts[lo:hi],
        row_ends[lo:hi],
        out,
        tier_rows=tier,
    )


def _entry_layout(
    rows: int, keys: int, key_bytes: int, q_row_bytes: int, w_row_bytes: int
) -> Tuple[dict, int]:
    """Byte offsets of one taken chunk's parts in its owner's all-gather contribution, and its padded size."""
    parts = (
        ("k", keys * key_bytes),
        ("s", keys * 4),
        ("q", rows * q_row_bytes),
        ("w", rows * w_row_bytes),
        ("ks", rows * 4),
        ("ke", rows * 4),
    )
    offs, o = {}, 0
    for name, nbytes in parts:
        offs[name] = o
        o += _pad(nbytes)
    return offs, o


_SIDE_STREAMS: dict = {}


def _side_stream(device: torch.device) -> torch.cuda.Stream:
    st = _SIDE_STREAMS.get(device.index)
    if st is None:
        st = _SIDE_STREAMS[device.index] = torch.cuda.Stream(device=device)
    return st


@dataclass
class PendingShare:
    """An indexer layer's shared prefill between ``start_shared_indexer`` and ``finish_shared_indexer``."""

    plan: SharePlan
    layouts: list
    contrib: List[int]
    recv: torch.Tensor
    done: torch.cuda.Event
    q_dtype: torch.dtype
    q_shape: Tuple[int, ...]
    w_dtype: torch.dtype
    w_cols: int
    key_bytes: int
    q_row_bytes: int
    w_row_bytes: int


@nvtx_range("k31_dsa_prefill_share_start")
def start_shared_indexer(
    indexer: "K31Indexer",
    metadata: "K31DSAtrtllmAttentionMetadata",
    q: torch.Tensor,
    weights: torch.Tensor,
    gather_keys: Callable[[int, int], Tuple[torch.Tensor, torch.Tensor]],
) -> PendingShare:
    """Step 1 of the module docstring (a collective over the attention-DP group). ``gather_keys(k_token_start,
    num_keys)`` returns a chunk's index keys and key scales from this layer's index-K cache."""
    plan: SharePlan = metadata.indexer_share_plan
    me = plan.rank
    dev = q.device
    key_bytes = indexer.head_dim
    q_row_bytes = q[0].numel() * q.element_size()
    w_row_bytes = weights.shape[1] * weights.element_size()
    layouts = [
        _entry_layout(e.rows, e.keys, key_bytes, q_row_bytes, w_row_bytes) for e in plan.entries
    ]
    contrib = [0] * plan.world
    for e, (_, size) in zip(plan.entries, layouts):
        contrib[e.owner] += size

    # This rank's taken chunks, packed in slot order.
    send = torch.empty(contrib[me], dtype=torch.uint8, device=dev)
    o = 0
    mine = [(e, lay) for e, lay in zip(plan.entries, layouts) if e.owner == me]
    for (e, (offs, size)), ci in zip(mine, plan.local_chunks):
        chunk = metadata.indexer_prefill_chunks[ci]
        k_data, k_scale = gather_keys(chunk.k_token_start, e.keys)
        t0, t1 = chunk.token_start, chunk.token_end
        parts = [
            ("k", k_data),
            ("s", k_scale),
            ("q", q[t0:t1]),
            ("w", weights[t0:t1]),
            ("ks", chunk.cu_seqlen_ks),
            ("ke", chunk.cu_seqlen_ke),
        ]
        for name, src in parts:
            flat = src.contiguous().view(torch.uint8).reshape(-1)
            send[o + offs[name] : o + offs[name] + flat.numel()].copy_(flat)
        o += size
    side = _side_stream(dev)
    side.wait_stream(torch.cuda.current_stream(dev))
    with torch.cuda.stream(side):
        recv = allgather_module(send, metadata.mapping, 0, contrib)
        send.record_stream(side)
        done = torch.cuda.Event()
        done.record(side)
    return PendingShare(
        plan,
        layouts,
        contrib,
        recv,
        done,
        q.dtype,
        tuple(q.shape[1:]),
        weights.dtype,
        weights.shape[1],
        key_bytes,
        q_row_bytes,
        w_row_bytes,
    )


@nvtx_range("k31_dsa_prefill_share_finish")
def finish_shared_indexer(
    indexer: "K31Indexer",
    metadata: "K31DSAtrtllmAttentionMetadata",
    pending: PendingShare,
    topk_indices_buffer: torch.Tensor,
) -> None:
    """Steps 2-3 of the module docstring (a collective over the attention-DP group)."""
    plan, layouts, contrib, recv = pending.plan, pending.layouts, pending.contrib, pending.recv
    world, me = plan.world, plan.rank
    dev = topk_indices_buffer.device
    main = torch.cuda.current_stream(dev)
    main.wait_event(pending.done)
    recv.record_stream(main)
    starts = [sum(contrib[:r]) for r in range(world)]
    seen = [0] * world
    views = []
    for e, (offs, size) in zip(plan.entries, layouts):
        base = starts[e.owner] + seen[e.owner]
        seen[e.owner] += size

        def view(name, nbytes, dtype, shape, base=base, offs=offs):
            return recv[base + offs[name] : base + offs[name] + nbytes].view(dtype).view(shape)

        views.append(
            (
                view(
                    "q", e.rows * pending.q_row_bytes, pending.q_dtype, (e.rows, *pending.q_shape)
                ),
                view(
                    "k",
                    e.keys * pending.key_bytes,
                    torch.float8_e4m3fn,
                    (e.keys, pending.key_bytes),
                ),
                view("s", e.keys * 4, torch.float32, (e.keys, 1)),
                view("w", e.rows * pending.w_row_bytes, pending.w_dtype, (e.rows, pending.w_cols)),
                view("ks", e.rows * 4, torch.int32, (e.rows,)),
                view("ke", e.rows * 4, torch.int32, (e.rows,)),
            )
        )
    k = indexer.index_topk
    slot_chunk = dict(zip((e.slot for e in plan.entries if e.owner == me), plan.local_chunks))

    def owner_rows(p: SharePiece) -> torch.Tensor:
        chunk = metadata.indexer_prefill_chunks[slot_chunk[plan.entries[p.entry].slot]]
        return topk_indices_buffer[chunk.token_start + p.lo : chunk.token_start + p.hi]

    # This rank's pieces: its own chunks' rows straight into its Top-K buffer, the others' to send.
    sent = plan.sent_by(me)
    mine_out = torch.empty(sum(p.hi - p.lo for p in sent), k, dtype=torch.int32, device=dev)
    o = 0
    for p in plan.pieces_of_rank(me):
        q, k_data, k_scale, w, ks, ke = views[p.entry]
        if plan.entries[p.entry].owner == me:
            out = owner_rows(p)
        else:
            out = mine_out[o : o + p.hi - p.lo]
            o += p.hi - p.lo
        score_piece(indexer, q, k_data, k_scale, w, ks, ke, out, p.lo, p.hi, p.tier)

    # The other ranks' rows back to the owners.
    rows_per_rank = [plan.rows_sent_by(r) for r in range(world)]
    if not any(rows_per_rank):
        return
    allout = allgather_module(mine_out, metadata.mapping, 0, rows_per_rank)
    offset = [sum(rows_per_rank[:r]) for r in range(world)]
    for p in plan.pieces:
        owner = plan.entries[p.entry].owner
        if owner == p.rank:
            continue
        n = p.hi - p.lo
        if owner == me:
            owner_rows(p).copy_(allout[offset[p.rank] : offset[p.rank] + n])
        offset[p.rank] += n
