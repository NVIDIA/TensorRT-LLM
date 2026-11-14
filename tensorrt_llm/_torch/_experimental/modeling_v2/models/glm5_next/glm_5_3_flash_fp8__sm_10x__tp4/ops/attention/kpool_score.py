# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pool scores of the GLM-5.3-Flash indexer over the paged pool keys, BF16 tensor-core dots, in
one launch.

Row ``r`` scores each of its ``kv_lens[r] // kpool`` complete pools ``j``:

    score[r, j] = sum_h (w[r, h] * w_scale) * relu(q_scale * q[r, h, :] . key_j)

where ``key_j`` is the pool key cached in the last ``head_dim`` columns of the indexer row of the
pool's first member (``kpool_update`` maintains it). BF16 x BF16 products are exact in FP32, so
each dot is an FP32 accumulation of exact products. Pools a row cannot see yet get the FP32
minimum, so a following top-k never prefers them.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)

#: The score of a pool a row cannot see.
FP32_MIN = torch.finfo(torch.float32).min


@triton.jit
def _kpool_score_kernel(
    Q,
    W,
    POOL,
    BT,
    KV_LENS,
    OUT,
    w_row_stride,
    w_head_stride,
    slot_stride,
    row_stride,
    bt_stride,
    tpb,
    num_pools_max,
    q_scale,
    w_scale,
    min_value,
    HD: tl.constexpr,
    KPOOL: tl.constexpr,
    H: tl.constexpr,
    HP: tl.constexpr,
    BP: tl.constexpr,
):
    """Score ``BP`` pools of row ``r`` against block table ``r``: ``sum_h w[h] * relu(q_h . key_j
    * s)``, BF16 operands into the tensor-core dot with FP32 accumulation.

    Pools whose last member is not yet visible to the row (``j >= kv_len // KPOOL``) get
    ``min_value`` so a following top-k never picks them ahead of a candidate; a block of pools
    past every visible one stores ``min_value`` without gathering keys.
    """
    r = tl.program_id(0).to(tl.int64)
    j0 = tl.program_id(1) * BP
    j = j0 + tl.arange(0, BP)
    out_row = OUT + r * num_pools_max
    num_cand = tl.load(KV_LENS + r).to(tl.int64) // KPOOL
    if j0 >= num_cand:
        tl.store(out_row + j, tl.full([BP], min_value, tl.float32), mask=j < num_pools_max)
    else:
        d = tl.arange(0, HD)
        h = tl.arange(0, HP)
        hmask = h < H
        valid = j < num_cand
        pos = j.to(tl.int64) * KPOOL
        page = pos // tpb
        off = pos - page * tpb
        slot = tl.load(BT + r * bt_stride + page, mask=valid, other=0).to(tl.int64)
        kptr = POOL + slot * slot_stride + off * row_stride + 2 * HD
        keys = tl.load(kptr[:, None] + d[None, :], mask=valid[:, None], other=0.0).to(tl.bfloat16)
        q = tl.load(
            Q + r * H * HD + h[:, None] * HD + d[None, :], mask=hmask[:, None], other=0.0
        ).to(tl.bfloat16)
        scores = tl.dot(q, tl.trans(keys), input_precision="ieee")  # [HP, BP]
        scores = tl.maximum(scores * q_scale, 0.0)
        w = tl.load(W + r * w_row_stride + h * w_head_stride, mask=hmask, other=0.0)
        mixed = tl.sum(scores * (w.to(tl.float32) * w_scale)[:, None], axis=0)
        tl.store(out_row + j, tl.where(valid, mixed, min_value), mask=j < num_pools_max)


class KpoolScore(OpWrapper):
    """``q`` BF16 ``[N, H, head_dim]`` (contiguous), ``weights`` ``[N, H]`` (any strides) ->
    FP32 ``[N, num_pools_max]`` pool scores.

    Row ``r`` reads block table ``r`` and sees ``kv_lens[r]`` positions: columns ``[0,
    kv_lens[r] // kpool)`` are its scores, every other column is the FP32 minimum.
    """

    ARCHS = frozenset({Arch.SM_100})
    # The indexer has 32 heads of 128 and scores pools of 4 over 32-token blocks. The tables
    # are 4,100 pages wide, so the capacity is num_pools_max = 32,800 pools. The cells give the
    # requests' KV length after the step; a verify step scores its rows on per-row copies of
    # the generation tables (`GenerationRows.tables`).
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="MTP-3 verify of one 13K-token request",
            spec=dict(requests=1, tokens=4, kv=13312),
        ),
        Cell(
            why="MTP-3 verify of eight 13K-token requests",
            spec=dict(requests=8, tokens=4, kv=13312),
        ),
        Cell(
            why="MTP-3 verify of 64 requests, the largest CUDA-graph batch",
            spec=dict(requests=64, tokens=4, kv=13312),
        ),
        Cell(
            why="single-token decode of eight 13K-token requests",
            spec=dict(requests=8, tokens=1, kv=13312),
        ),
        Cell(
            why="one row: an MTP draft step of one 13K-token request",
            spec=dict(requests=1, tokens=1, kv=13312),
        ),
    )

    note = """
    The target scores its generation rows here (`MlaLayer.generation_partial`); packed context
    rows are scored by `kpool_score_fp8`.

    Only BF16 queries and BF16 pool keys: the dot reads both as BF16, so an FP32 query would
    be rounded to BF16 silently. `is_valid` refuses it.
    """

    def __call__(
        self,
        q: torch.Tensor,
        weights: torch.Tensor,
        index_pool: torch.Tensor,
        block_tables: torch.Tensor,
        kv_lens: torch.Tensor,
        tokens_per_block: int,
        *,
        num_pools_max: int,
        head_dim: int,
        kpool: int,
        q_scale: float,
        w_scale: float,
    ) -> torch.Tensor:
        n, num_heads, hd = q.shape
        if hd != head_dim:
            raise ValueError(f"kpool_score: q head_dim {hd} != {head_dim}")
        out = torch.empty(n, num_pools_max, dtype=torch.float32, device=q.device)
        if n == 0:
            return out
        bp = 64
        _kpool_score_kernel[(n, triton.cdiv(num_pools_max, bp))](
            q,
            weights,
            index_pool,
            block_tables,
            kv_lens,
            out,
            weights.stride(0),
            weights.stride(1),
            index_pool.stride(0),
            index_pool.stride(1),
            block_tables.stride(0),
            tokens_per_block,
            num_pools_max,
            q_scale,
            w_scale,
            FP32_MIN,
            HD=head_dim,
            KPOOL=kpool,
            H=num_heads,
            HP=max(16, triton.next_power_of_2(num_heads)),
            BP=bp,
            num_warps=4,
        )
        return out

    def reference(
        self,
        q: torch.Tensor,
        weights: torch.Tensor,
        index_pool: torch.Tensor,
        block_tables: torch.Tensor,
        kv_lens: torch.Tensor,
        tokens_per_block: int,
        *,
        num_pools_max: int,
        head_dim: int,
        kpool: int,
        q_scale: float,
        w_scale: float,
    ) -> torch.Tensor:
        """The scores in FP32 from the BF16 operands, every invisible column the FP32 minimum."""
        n = q.shape[0]
        dev = q.device
        out = torch.full((n, num_pools_max), FP32_MIN, dtype=torch.float32, device=dev)
        if n == 0:
            return out
        table = torch.arange(n, device=dev)
        pools = (kv_lens.long() // kpool).clamp(max=num_pools_max)
        pool_keys = index_pool[:, :, 2 * head_dim : 3 * head_dim]
        chunk = 32
        for r0 in range(0, n, chunk):
            rows = slice(r0, min(r0 + chunk, n))
            width = int(pools[rows].max())
            if width == 0:
                continue
            first = torch.arange(width, device=dev) * kpool
            slot = block_tables[table[rows, None], (first // tokens_per_block)[None, :]]
            keys = pool_keys[slot.long(), (first % tokens_per_block)[None, :]].float()
            with true_fp32_matmul():
                dots = torch.einsum("rhd,rpd->rhp", q[rows].float(), keys)
            mixed = (dots * q_scale).relu() * (weights[rows].float() * w_scale)[:, :, None]
            scores = mixed.sum(dim=1)
            col = torch.arange(width, device=dev)
            out[rows, :width] = torch.where(col < pools[rows, None], scores, FP32_MIN)
        return out

    def is_valid(
        self,
        q: torch.Tensor,
        weights: torch.Tensor,
        index_pool: torch.Tensor,
        block_tables: torch.Tensor,
        kv_lens: torch.Tensor,
        tokens_per_block: int,
        *,
        num_pools_max: int,
        head_dim: int,
        kpool: int,
        q_scale: float,
        w_scale: float,
    ) -> None:
        assert q.dtype == torch.bfloat16 and index_pool.dtype == torch.bfloat16, (
            f"the dot reads q and the pool keys as BF16; got {q.dtype}, {index_pool.dtype}"
        )
        # The kernel reads q at r * H * head_dim + h * head_dim + d.
        assert q.dim() == 3 and q.is_contiguous(), f"q must be a contiguous [N, H, D]; {q.stride()}"
        n, num_heads, _ = q.shape
        assert head_dim & (head_dim - 1) == 0, f"head_dim {head_dim} must be a power of two"
        assert weights.shape == (n, num_heads)
        assert index_pool.dim() == 3 and index_pool.stride(2) == 1
        assert index_pool.shape[1] == tokens_per_block and index_pool.shape[2] >= 3 * head_dim
        assert block_tables.dim() == 2 and block_tables.stride(1) == 1
        assert kv_lens.shape == (n,)
        if n == 0:
            return
        # Row r reads table r.
        assert block_tables.stride(0) == 0 or block_tables.shape[0] >= n
        widest = int(kv_lens.max())
        # A row with more pools than num_pools_max loses the rest without a sign.
        assert widest // kpool <= num_pools_max, f"{widest // kpool} pools > {num_pools_max}"
        assert widest <= block_tables.shape[1] * tokens_per_block, (
            f"a visible length of {widest} exceeds the tables' "
            f"{block_tables.shape[1] * tokens_per_block} positions"
        )


kpool_score = KpoolScore()
