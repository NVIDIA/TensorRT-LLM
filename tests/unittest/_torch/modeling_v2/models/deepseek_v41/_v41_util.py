# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared inputs of the deepseek_v41 entry tests (not a test module)."""

import zlib

import torch

from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_pack_rows import (
    csa2_pack_rows,
    csa2_unpack_rows,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_rows_io import (
    csa2_gather_rows_footer,
    csa2_row_addr,
    csa2_scatter_rows,
    csa2_scatter_rows_footer,
)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
IS_SM103 = torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 3)


def gen_for(cell, salt: int = 0) -> torch.Generator:
    """A generator seeded from the cell's reason, so a cell's inputs are the same on every run."""
    return torch.Generator(device=DEV).manual_seed((zlib.crc32(cell.why.encode()) + salt) % 2**31)


def index_pool(ratio: int, nkeys: int, gen: torch.Generator):
    """An index-key pool holding `nkeys` random keys of one request, in shuffled pages.

    Built with the *references* of the pack and scatter entries, so a bug in the ops under test
    cannot hide in the fixture. Returns `(pool, table, dequantized_keys)`.
    """
    rpp = 128 // ratio
    pages = (nkeys + rpp - 1) // rpp + 1
    keys = torch.randn(nkeys, 128, generator=gen, device=DEV)
    pool = torch.zeros(pages + 2, rpp, 68, dtype=torch.uint8, device=DEV)
    table = (
        torch.arange(pages, dtype=torch.int32, device=DEV).flip(0).unsqueeze(0)
    )  # shuffled pages
    rid = torch.zeros(nkeys, dtype=torch.long, device=DEV)
    addr = csa2_row_addr.reference(table, rid, torch.arange(nkeys, device=DEV), rpp)
    packed = csa2_pack_rows.reference(keys, "index")
    csa2_scatter_rows_footer.reference(
        pool, addr, packed, torch.ones(nkeys, dtype=torch.bool, device=DEV)
    )
    return pool, table, csa2_unpack_rows.reference(packed, "index", torch.float32)


def fp4_grid(q: torch.Tensor) -> torch.Tensor:
    """`q` after the index quantize-dequantize round trip the model applies before scoring."""
    flat = q.reshape(-1, q.shape[-1])
    return csa2_unpack_rows.reference(
        csa2_pack_rows.reference(flat, "index"), "index", q.dtype
    ).view_as(q)


def request_index_pools(ratio: int, lens, seed: int = 0):
    """Index-key pool and page table of several requests (`lens[r]` tokens each), keys random, pages shuffled.

    Built with the entries' own callables for addressing and packing, like `index_pool`. Returns
    `(pool [pages, 128 // ratio, 68] uint8, table int32 [requests, blocks])`.
    """
    torch.manual_seed(seed)
    rpp = 128 // ratio
    nreq = len(lens)
    nkeys = [n // ratio + 1 for n in lens]
    nb = max(nkeys) // rpp + 2
    pool = torch.zeros(nreq * nb + 2, rpp, 68, dtype=torch.uint8, device=DEV)
    table = torch.arange(nreq * nb, dtype=torch.int32, device=DEV).view(nreq, nb).flip(1) + 1
    for r in range(nreq):
        rows = torch.arange(nkeys[r], device=DEV)
        addr = csa2_row_addr(table, torch.full_like(rows, r), rows, rpp)
        packed = csa2_pack_rows(torch.randn(nkeys[r], 128, device=DEV), "index")
        csa2_scatter_rows_footer(pool, addr, packed, torch.ones_like(addr, dtype=torch.bool))
    return pool, table


def overlap(got: torch.Tensor, ref: torch.Tensor) -> float:
    """Share of the reference's reachable picks that the op also picked (rows of -1 padded positions)."""
    inter = sum(len(set(g[g >= 0].tolist()) & set(r[r >= 0].tolist())) for g, r in zip(got, ref))
    return inter / max(1, sum(int((r >= 0).sum()) for r in ref))


def check_consumer_picks(picks, q, w, rid, pool, table, ratio, topk, cand_positions, c_len, span):
    """The gate of a candidate consumer: every pick lies in the candidate set and scores within the kernel's
    output precision of the exact k-th best candidate.

    The sparse paged logits kernel returns bf16 scores (relative error <= 2^-8) and takes bf16 head weights
    (another 2^-8). Both scores of a comparison carry that error, the pick's upward and the k-th best's
    downward, so a pick may rank below the exact k-th best by up to 2 * 2 * 2^-8 = 2^-6 of the row's largest;
    no tighter band can hold with random scores packed this densely. A row with fewer
    reachable candidates than k keeps all of them, otherwise it fills k picks.
    """
    from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_indexer_decode import (
        dequantize_queries,
    )

    qb = dequantize_queries(q).float()
    n = qb.shape[0]
    cols = torch.arange(span, device=DEV)
    addr = csa2_row_addr(
        table, rid.long().unsqueeze(-1).expand(n, span), cols.expand(n, span), 128 // ratio
    )
    keys = csa2_unpack_rows(csa2_gather_rows_footer(pool, addr), "index", torch.float32)
    s = (torch.einsum("nhd,ntd->nht", qb, keys).relu() * w.bfloat16().float().unsqueeze(-1)).sum(1)
    s = s.masked_fill(cols >= c_len.long().unsqueeze(-1), float("-inf"))
    pc = cand_positions.long()
    cm = torch.zeros(n, span + 1, dtype=torch.bool, device=DEV)
    cm.scatter_(
        1, torch.where((pc >= 0) & (pc < span), pc, span), torch.ones_like(pc, dtype=torch.bool)
    )
    s = s.masked_fill(~cm[:, :span], float("-inf"))
    k = min(picks.shape[1], span)
    kth = s.topk(min(topk, span), dim=-1).values[:, -1:]
    got = picks[:, :k].long()
    live = got >= 0
    assert bool((cm[:, :span].gather(1, got.clamp(min=0)) | ~live).all()), (
        "pick outside the candidate set"
    )
    ps = s.gather(1, got.clamp(min=0))
    slack = 2.0**-6 * s.amax(dim=-1, keepdim=True).clamp(min=1e-6)
    assert bool(((ps >= kth - slack) | ~live).all()), (
        "pick scores below the k-th best candidate by more than bf16 precision"
    )
    assert torch.equal(live.sum(1), torch.isfinite(s).sum(1).clamp(max=k))


__all__ = [
    "DEV",
    "IS_SM103",
    "gen_for",
    "index_pool",
    "fp4_grid",
    "csa2_scatter_rows",
    "request_index_pools",
    "overlap",
    "check_consumer_picks",
]
