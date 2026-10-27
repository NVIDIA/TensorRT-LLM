# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-rank top-k of the drafter's vocab-sharded logits, reduced to ``top_k`` keys per rank.

``draft_shard_topk`` keeps the top ``top_k`` sort keys of every ``chunk``
columns of a rank's ``[M, shard]`` slice, ``ceil(shard / chunk) * top_k`` keys
per row (208 at TP4), and the draft head exchanges all of them. Only the
rank's own top ``top_k`` can be among the full vocabulary's top ``top_k``: a
key's position in its rank's order (value desc, column asc) is at most its
position in the global order (value desc, vocab id asc), since vocab id =
rank * shard + column orders a rank's columns the same way. So this entry
runs ``draft_shard_topk``'s per-chunk selection and then keeps each row's
``top_k`` largest keys, largest first, in the all-reduce layout: this rank's
slot holds them as exact 16-bit halves in fp32, every other rank's slot +0.0.
``draft_topk_merge`` over the sum of all ranks' outputs gives the result it
gives over the per-chunk keys, bit for bit (selection, tie order and the
selector's operands), from 1/13 of the bytes at TP4.

``canon`` and ``canon_band`` (a drafter logit transform's canonical keys) go to
the per-chunk stage unchanged; the rank's top ``top_k`` of canonical keys holds
every key of the global top ``top_k`` under the transformed order for the same
reason.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from .draft_shard_topk import CHUNK, FULL_BAND, NUM_WARPS, SHARD, TOP_K, DraftShardTopK

#: Per-chunk keys one row program reduces.
MAX_KEYS = 4096

_draft_shard_topk = DraftShardTopK()


@triton.jit
def _draft_rank_topk_kernel(
    keys_ptr,
    out_ptr,
    num_keys,
    keys_row_stride,
    out_row_stride,
    rank,
    num_ranks,
    TOP_K: tl.constexpr,
    KEYS_PAD: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, KEYS_PAD)
    key = tl.load(
        keys_ptr + row.to(tl.int64) * keys_row_stride + cols,
        mask=cols < num_keys,
        other=-2147483648,
    )
    slots = tl.arange(0, TOP_K)
    out = tl.full([TOP_K], -2147483648, tl.int32)
    # Keys are unique per element; EMPTY_KEY (the minimum) only pads.
    for j in tl.static_range(TOP_K):
        best = tl.max(key, axis=0)
        out = tl.where(slots == j, best, out)
        key = tl.where(key == best, -2147483648, key)
    base = out_ptr + row.to(tl.int64) * out_row_stride
    col = (rank * TOP_K + slots) * 2
    tl.store(base + col, ((out >> 16) & 0xFFFF).to(tl.float32))
    tl.store(base + col + 1, (out & 0xFFFF).to(tl.float32))
    zeros = tl.zeros([2 * TOP_K], tl.float32)
    span = tl.arange(0, 2 * TOP_K)
    for other in range(num_ranks):
        if other != rank:
            tl.store(base + other * TOP_K * 2 + span, zeros)


class DraftRankTopK(OpWrapper):
    """A rank's top ``top_k`` sort keys of a bf16 logits slice, in the all-reduce layout.

        logits     [M, shard] bf16, stride(-1) == 1, row stride free; shard <= 65536
        top_k      candidates kept per rank (a power of two)
        chunk      columns per program of the per-chunk stage (a power of two >= top_k)
        num_warps  warps per program of the per-chunk stage
        rank, num_ranks
        canon, canon_band
                   ``draft_shard_topk``'s canonical-key table and its identity band

    Returns ``[M, num_ranks * top_k * 2]`` fp32: rank ``rank``'s keys at columns
    ``[rank * top_k * 2, (rank + 1) * top_k * 2)`` as (high, low) 16-bit halves,
    largest key first (``draft_shard_topk``'s key encoding), ``EMPTY_KEY`` past a
    shard of fewer than ``top_k`` columns; +0.0 in every other rank's slot.
    Integer selection: exact.
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})
    CELLS: tuple[Cell, ...] = (
        Cell(
            "five requests at TP8 (35 rows, the fewest wide_draft_topk takes)",
            {"rows": 35, "shard": SHARD, "transform": None},
        ),
        Cell(
            "five requests at TP8, the Muse Spark Open drafter transform's canon",
            {"rows": 35, "shard": SHARD, "transform": "musespark"},
        ),
        Cell(
            "24 requests at TP4 (a 50512-column shard), the drafter transform",
            {"rows": 168, "shard": 2 * SHARD, "transform": "musespark"},
        ),
    )
    note = (
        "Used with draft_topk_merge after the draft head's MNNVL fp32 all-reduce "
        "(wide_draft_topk's RankTopKExchange, from 33 rows): the sum of "
        "all ranks' outputs is the concatenation of their top_k lists. Two launches: "
        "draft_shard_topk's per-chunk stage into an int32 scratch, then one program per "
        "row that keeps the row's top_k and writes this rank's slot and the zeros."
    )

    def __call__(
        self,
        logits: torch.Tensor,
        top_k: int = TOP_K,
        chunk: int = CHUNK,
        num_warps: int = NUM_WARPS,
        rank: int = 0,
        num_ranks: int = 1,
        canon: torch.Tensor | None = None,
        canon_band: tuple[int, int] = FULL_BAND,
    ) -> torch.Tensor:
        rows = logits.shape[0]
        keys = _draft_shard_topk(
            logits, top_k, chunk, num_warps, canon=canon, canon_band=canon_band
        )
        out = torch.empty((rows, num_ranks * top_k * 2), dtype=torch.float32, device=logits.device)
        _draft_rank_topk_kernel[(rows,)](
            keys,
            out,
            keys.shape[1],
            keys.stride(0),
            out.stride(0),
            rank,
            num_ranks,
            TOP_K=top_k,
            KEYS_PAD=triton.next_power_of_2(keys.shape[1]),
            num_warps=4,
        )
        return out

    def reference(
        self,
        logits: torch.Tensor,
        top_k: int = TOP_K,
        chunk: int = CHUNK,
        num_warps: int = NUM_WARPS,
        rank: int = 0,
        num_ranks: int = 1,
        canon: torch.Tensor | None = None,
        canon_band: tuple[int, int] = FULL_BAND,
    ) -> torch.Tensor:
        """The rank's top ``top_k`` keys over the whole shard, from torch ops; geometry ignored."""
        rows = logits.shape[0]
        # One chunk spanning the shard: draft_shard_topk's encoding of the rank's top_k.
        keys = _draft_shard_topk.reference(
            logits, top_k, 1 << 16, canon=canon, canon_band=canon_band
        ).to(torch.int64)
        unsigned = torch.where(keys < 0, keys + 2**32, keys)
        halves = torch.stack(((unsigned >> 16) & 0xFFFF, unsigned & 0xFFFF), dim=-1).float()
        packed = torch.zeros((rows, num_ranks, top_k, 2), device=logits.device)
        packed[:, rank] = halves
        return packed.view(rows, -1)

    def is_valid(
        self,
        logits: torch.Tensor,
        top_k: int = TOP_K,
        chunk: int = CHUNK,
        num_warps: int = NUM_WARPS,
        rank: int = 0,
        num_ranks: int = 1,
        canon: torch.Tensor | None = None,
        canon_band: tuple[int, int] = FULL_BAND,
        **unused,
    ) -> None:
        _draft_shard_topk.is_valid(
            logits, top_k, chunk, num_warps, canon=canon, canon_band=canon_band
        )
        if not 0 <= rank < num_ranks:
            raise ValueError(f"rank {rank} outside [0, {num_ranks})")
        if triton.cdiv(logits.shape[1], chunk) * top_k > MAX_KEYS:
            raise ValueError(f"more than {MAX_KEYS} per-chunk keys per row")


draft_rank_topk = DraftRankTopK()
