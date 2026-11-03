# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sharing the DSA indexer prefill of long-prefix chunks across attention-DP ranks.

Under attention DP the ranks run their own requests in lockstep, so a step lasts as long as the rank with the
largest indexer work: its query rows x keys, which grows with the request's cached prefix. On a Kimi K3.1
disaggregated context server (``prefill_share_enabled``) every rank offers up to ``TRTLLM_DSA_PREFILL_SHARE_MAX``
indexer prefill chunks with at least ``TRTLLM_DSA_PREFILL_SHARE_MIN_KEYS`` keys and
``TRTLLM_DSA_PREFILL_SHARE_MIN_ROWS`` query rows, carried in the per-step attention-DP count exchange with its total
indexer work; a rank's offers are taken only while that work exceeds ``TRTLLM_DSA_PREFILL_SHARE_IMBALANCE`` times the
step's mean. Each layer then:

1. all-gathers the offered chunks' indexer inputs (the gathered index-K bytes and scales of the chunk, its FP8
   queries, fp32 head weights and row windows) from their owners;
2. on every rank, runs the indexer (logits + Top-K, the layer's own kernels) for rank ``r``'s ``1 / W`` slice of
   each offered chunk's rows;
3. all-gathers the Top-K rows; each owner writes its chunks' rows into the layer's Top-K buffer.

Query rows are independent, so every row's selected set is the one the owner would compute. The self-sampling
Top-K engine picks its plan by the launch's row count and a row's index order depends on the plan, so each rank
walks its rows through the owner's query tiles and runs every piece with the plan of its owner tile
(``tier_rows``): the rows come back bit for bit as the owner alone writes them.
A step shares nothing unless every rank has context tokens (all of them then run the context indexer of every
layer), none skips its context indexer and none runs in a CUDA graph (the plan and the collective sizes are host
values). FP8 and MXFP4 index keys (the MXFP4 query scales travel with the queries).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch

from tensorrt_llm._utils import nvtx_range

from .params import disagg_role, is_kimi_k31_dsa

if TYPE_CHECKING:
    from .indexer import Indexer
    from .metadata import DSAtrtllmAttentionMetadata

_ALIGN = 256


def prefill_share_enabled() -> bool:
    """Whether attention-DP ranks share long indexer prefill chunks: Kimi K3.1 on a disaggregated context server
    (the attention-DP layout gate is ``DSAtrtllmAttentionMetadata._shares_prefill``)."""
    return is_kimi_k31_dsa() and disagg_role() == "context"


def share_slots() -> int:
    return int(os.environ.get("TRTLLM_DSA_PREFILL_SHARE_MAX", "2"))


def share_min_keys() -> int:
    return int(os.environ.get("TRTLLM_DSA_PREFILL_SHARE_MIN_KEYS", "200000"))


def share_min_rows() -> int:
    return int(os.environ.get("TRTLLM_DSA_PREFILL_SHARE_MIN_ROWS", "1024"))


def share_imbalance() -> float:
    """TRTLLM_DSA_PREFILL_SHARE_IMBALANCE (default 1.25; 0 = no gate): a rank's offered chunks are shared only
    while its indexer work (query rows x keys over all its prefill chunks) exceeds this multiple of the step's mean
    over the ranks, so a balanced step pays no communication."""
    return float(os.environ.get("TRTLLM_DSA_PREFILL_SHARE_IMBALANCE", "1.25"))


_PAIRS_UNIT = 1 << 20


def num_descriptor_counts() -> int:
    """Extra per-rank counts in the attention-DP exchange: [eligible flag, indexer work in 2^20 query-key pairs] +
    (rows, keys) per slot + a bit mask of the slots whose rows all start at the chunk's first key."""
    return 3 + 2 * share_slots()


def _pad(n: int) -> int:
    return -(-n // _ALIGN) * _ALIGN


@dataclass
class ShareEntry:
    owner: int
    slot: int
    rows: int
    keys: int
    # Every row's window starts at the chunk's first key (one request): a kernel-choice hint for the indexer.
    row_starts_zero: bool = False


@dataclass
class SharePlan:
    world: int
    rank: int
    entries: List[ShareEntry]
    # Indices into metadata.indexer_prefill_chunks of this rank's offered chunks, by slot.
    local_chunks: List[int] = field(default_factory=list)

    def row_range(self, e: ShareEntry, r: int) -> Tuple[int, int]:
        return e.rows * r // self.world, e.rows * (r + 1) // self.world

    def rows_of_rank(self, r: int) -> int:
        return sum(hi - lo for lo, hi in (self.row_range(e, r) for e in self.entries))


def entry_layout(
    rows: int, keys: int, key_bytes: int, q_row_bytes: int, w_row_bytes: int, qs_row_bytes: int = 0
) -> Tuple[dict, int]:
    """Byte offsets of one offered chunk's parts in its owner's all-gather contribution, and its padded size
    (``qs``: the per-row query scales of MXFP4 index queries, else empty)."""
    parts = (
        ("k", keys * key_bytes),
        ("s", keys * 4),
        ("q", rows * q_row_bytes),
        ("w", rows * w_row_bytes),
        ("ks", rows * 4),
        ("ke", rows * 4),
        ("qs", rows * qs_row_bytes),
    )
    offs, o = {}, 0
    for name, nbytes in parts:
        offs[name] = o
        o += _pad(nbytes)
    return offs, o


def local_descriptors(metadata: "DSAtrtllmAttentionMetadata") -> Tuple[List[int], List[int]]:
    """This rank's [eligible, work, rows0, keys0, rows1, keys1, ..., zero-start slot mask] and the offered chunk
    indices."""
    slots = share_slots()
    chunks = metadata.indexer_prefill_chunks or []
    eligible = int(
        metadata.num_contexts > 0
        and metadata.num_ctx_tokens > 0
        and not metadata.skip_indexer_for_ctx_reqs
        and metadata.indexer_prefill_q_split is None
        and not getattr(metadata, "is_cuda_graph", False)
    )
    cands = []
    work = 0
    if eligible:
        min_keys, min_rows = share_min_keys(), share_min_rows()
        for i, c in enumerate(chunks):
            rows = c.token_end - c.token_start
            keys = c.k_token_end - c.k_token_start
            work += rows * keys
            if keys >= min_keys and rows >= min_rows:
                cands.append((rows * keys, i, rows, keys))
        cands.sort(reverse=True)
        cands = cands[:slots]
        # Offered chunks keep their batch order (any order works; it is the all-gather layout).
        cands.sort(key=lambda t: t[1])
    desc = [eligible, -(-work // _PAIRS_UNIT)]
    for _, _, rows, keys in cands:
        desc += [rows, keys]
    desc += [0, 0] * (slots - len(cands))
    desc.append(
        sum(
            1 << s
            for s, (_, i, _, _) in enumerate(cands)
            if getattr(chunks[i], "row_starts_zero", False)
        )
    )
    return desc, [i for _, i, _, _ in cands]


def build_plan(columns: List[List[int]], rank: int, local_chunks: List[int]) -> Optional[SharePlan]:
    """The step's plan from the exchanged descriptor columns (``columns[i][r]`` = count ``i`` of rank ``r``)."""
    world = len(columns[0])
    if world <= 1 or not all(columns[0]):
        return None
    work = columns[1]
    gate = share_imbalance()
    bar = gate * sum(work) / world
    entries = []
    zero_mask = columns[2 + 2 * share_slots()]
    for r in range(world):
        if gate > 0 and work[r] <= bar:
            continue
        for s in range(share_slots()):
            rows, keys = columns[2 + 2 * s][r], columns[3 + 2 * s][r]
            if rows > 0:
                entries.append(ShareEntry(r, s, rows, keys, bool(zero_mask[r] >> s & 1)))
    if not entries:
        return None
    if not any(e.owner == rank for e in entries):
        local_chunks = []
    from tensorrt_llm.logger import logger

    logger.info_once(
        f"DSA prefill share active: {len(entries)} chunk(s) over {world} ranks "
        f"(rows x keys {[(e.rows, e.keys) for e in entries]})",
        key="dsa_prefill_share_active",
    )
    return SharePlan(world, rank, entries, list(local_chunks))


def share_overlap() -> bool:
    """TRTLLM_DSA_PREFILL_SHARE_OVERLAP (default 1): the input all-gather runs on a side stream while this rank's own
    (unshared) chunks are scored."""
    return os.environ.get("TRTLLM_DSA_PREFILL_SHARE_OVERLAP", "1") == "1"


_SIDE_STREAMS: dict = {}


def _side_stream(device: torch.device) -> torch.cuda.Stream:
    st = _SIDE_STREAMS.get(device.index)
    if st is None:
        st = _SIDE_STREAMS[device.index] = torch.cuda.Stream(device=device)
    return st


@dataclass
class _Pending:
    plan: SharePlan
    layouts: list
    contrib: List[int]
    recv: torch.Tensor
    done: Optional[torch.cuda.Event]
    heads: int
    dim: int
    q_dtype: torch.dtype
    w_dtype: torch.dtype
    w_cols: int
    q_row_bytes: int
    w_row_bytes: int
    key_bytes: int
    qs_row_bytes: int = 0
    qs_shape: Tuple[int, ...] = ()
    qs_dtype: Optional[torch.dtype] = None


@nvtx_range("dsa_prefill_share_start")
def start_shared_indexer(
    indexer: "Indexer",
    metadata: "DSAtrtllmAttentionMetadata",
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    q_scale: Optional[torch.Tensor] = None,
) -> _Pending:
    """Step 1 of the module docstring (collective over the attention-DP group); with share_overlap() the all-gather
    is left running on a side stream."""
    from tensorrt_llm._torch.distributed import allgather

    from .kernels import gather_k_cache_paged

    plan: SharePlan = metadata.indexer_share_plan
    world, me = plan.world, plan.rank
    dev = q_fp8.device
    key_bytes = indexer.k_data_bytes_per_token
    heads, dim = q_fp8.shape[1], q_fp8.shape[2]
    q_row_bytes = heads * dim * q_fp8.element_size()
    w_row_bytes = weights.shape[1] * weights.element_size()
    qs_row_bytes = 0 if q_scale is None else q_scale[0].numel() * q_scale.element_size()
    layouts = [
        entry_layout(e.rows, e.keys, key_bytes, q_row_bytes, w_row_bytes, qs_row_bytes)
        for e in plan.entries
    ]
    contrib = [0] * world
    for e, (_, size) in zip(plan.entries, layouts):
        contrib[e.owner] += size

    # This rank's offered chunks, packed in slot order.
    send = torch.empty(contrib[me], dtype=torch.uint8, device=dev)
    o = 0
    k_cache_4d = metadata.kv_cache_manager.get_indexer_k_cache_buffers(indexer.layer_idx)
    mine = [(e, lay) for e, lay in zip(plan.entries, layouts) if e.owner == me]
    for (e, (offs, size)), ci in zip(mine, plan.local_chunks):
        chunk = metadata.indexer_prefill_chunks[ci]
        gathered = gather_k_cache_paged(
            k_cache_4d,
            metadata.slot_mapping_fp8_fullkv,
            metadata.slot_mapping_scale_fullkv,
            chunk.k_token_start,
            e.keys,
            key_bytes,
        )
        if gathered is None:
            gathered = torch.ops.trtllm.indexer_k_cache_gather_op(
                k_cache_4d,
                metadata.slot_mapping_fp8_fullkv,
                metadata.slot_mapping_scale_fullkv,
                chunk.k_token_start,
                e.keys,
                key_bytes,
            )
        k_data, k_scale = gathered
        t0, t1 = chunk.token_start, chunk.token_end
        parts = [
            ("k", k_data),
            ("s", k_scale),
            ("q", q_fp8[t0:t1]),
            ("w", weights[t0:t1]),
            ("ks", chunk.cu_seqlen_ks),
            ("ke", chunk.cu_seqlen_ke),
        ]
        if q_scale is not None:
            parts.append(("qs", q_scale[t0:t1]))
        for name, src in parts:
            flat = src.contiguous().view(torch.uint8).reshape(-1)
            send[o + offs[name] : o + offs[name] + flat.numel()].copy_(flat)
        o += size
    done = None
    if share_overlap():
        main = torch.cuda.current_stream(dev)
        side = _side_stream(dev)
        side.wait_stream(main)
        with torch.cuda.stream(side):
            recv = allgather(send, metadata.mapping, dim=0, sizes=contrib)
            send.record_stream(side)
            done = torch.cuda.Event()
            done.record(side)
    else:
        recv = allgather(send, metadata.mapping, dim=0, sizes=contrib)
    return _Pending(
        plan,
        layouts,
        contrib,
        recv,
        done,
        heads,
        dim,
        q_fp8.dtype,
        weights.dtype,
        weights.shape[1],
        q_row_bytes,
        w_row_bytes,
        key_bytes,
        qs_row_bytes,
        tuple(q_scale.shape[1:]) if q_scale is not None else (),
        q_scale.dtype if q_scale is not None else None,
    )


@nvtx_range("dsa_prefill_share_finish")
def finish_shared_indexer(
    indexer: "Indexer",
    metadata: "DSAtrtllmAttentionMetadata",
    pending: _Pending,
    topk_indices_buffer: torch.Tensor,
) -> None:
    """Steps 2-3 of the module docstring (collective over the attention-DP group)."""
    from tensorrt_llm._torch.distributed import allgather

    plan, layouts, contrib, recv = pending.plan, pending.layouts, pending.contrib, pending.recv
    world, me = plan.world, plan.rank
    dev = topk_indices_buffer.device
    if pending.done is not None:
        main = torch.cuda.current_stream(dev)
        main.wait_event(pending.done)
        recv.record_stream(main)
    key_bytes, q_row_bytes, w_row_bytes = (
        pending.key_bytes,
        pending.q_row_bytes,
        pending.w_row_bytes,
    )
    starts = [sum(contrib[:r]) for r in range(world)]
    seen = [0] * world
    outs = []
    k = indexer.index_topk
    for e, (offs, size) in zip(plan.entries, layouts):
        base = starts[e.owner] + seen[e.owner]
        seen[e.owner] += size

        def view(name, nbytes, dtype, shape, base=base, offs=offs):
            return recv[base + offs[name] : base + offs[name] + nbytes].view(dtype).view(shape)

        k_data = view("k", e.keys * key_bytes, torch.float8_e4m3fn, (e.keys, key_bytes))
        k_scale = view("s", e.keys * 4, torch.float32, (e.keys, 1))
        q = view("q", e.rows * q_row_bytes, pending.q_dtype, (e.rows, pending.heads, pending.dim))
        w = view("w", e.rows * w_row_bytes, pending.w_dtype, (e.rows, pending.w_cols))
        ks = view("ks", e.rows * 4, torch.int32, (e.rows,))
        ke = view("ke", e.rows * 4, torch.int32, (e.rows,))
        qs = (
            view("qs", e.rows * pending.qs_row_bytes, pending.qs_dtype, (e.rows, *pending.qs_shape))
            if pending.qs_row_bytes
            else None
        )
        lo, hi = plan.row_range(e, me)
        out = torch.empty(hi - lo, k, dtype=torch.int32, device=dev)
        # Walk this rank's rows through the owner's query tiles of the chunk: every piece runs the Top-K plan of
        # its whole owner tile, so each row's indices (order included) are the ones the owner alone would write.
        tile = indexer.prefill_q_tile(e.rows, e.keys, q.is_cuda)
        for t0, t1, tier_rows in indexer.prefill_tile_pieces(lo, hi, e.rows, tile):
            indexer.forward_prepared(
                q[t0:t1],
                k_data,
                k_scale,
                w[t0:t1],
                ks[t0:t1],
                ke[t0:t1],
                out[t0 - lo : t1 - lo],
                qs[t0:t1] if qs is not None else None,
                tier_rows=tier_rows,
                row_starts_zero=e.row_starts_zero,
            )
        outs.append(out)
    mine_out = torch.cat(outs) if outs else torch.empty(0, k, dtype=torch.int32, device=dev)

    # Every rank's rows back to the owners.
    rows_per_rank = [plan.rows_of_rank(r) for r in range(world)]
    allout = allgather(mine_out, metadata.mapping, dim=0, sizes=rows_per_rank)
    row_starts = [sum(rows_per_rank[:r]) for r in range(world)]
    taken = [0] * world
    owned = [e for e in plan.entries if e.owner == me]
    slot_chunk = dict(zip((e.slot for e in owned), plan.local_chunks))
    for e in plan.entries:
        for r in range(world):
            lo, hi = plan.row_range(e, r)
            if e.owner == me and hi > lo:
                src = allout[row_starts[r] + taken[r] : row_starts[r] + taken[r] + hi - lo]
                chunk = metadata.indexer_prefill_chunks[slot_chunk[e.slot]]
                topk_indices_buffer[chunk.token_start + lo : chunk.token_start + hi].copy_(src)
            taken[r] += hi - lo
