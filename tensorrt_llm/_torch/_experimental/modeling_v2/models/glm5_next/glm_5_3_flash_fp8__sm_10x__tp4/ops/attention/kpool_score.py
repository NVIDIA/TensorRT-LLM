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
    REQ,
    OUT,
    num_rows,
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
    ROWS: tl.constexpr,
    PRECISION: tl.constexpr,
    HAS_REQ: tl.constexpr,
    DOT_DTYPE: tl.constexpr = tl.float32,
):
    """Score ``BP`` pools for ``ROWS`` consecutive rows: ``sum_h w[h] * relu(q_h . key_j * s)``.

    Pools whose last member is not yet visible to a row (``j >= kv_len //
    KPOOL``) get ``min_value`` so a following top-k never picks them ahead of a
    candidate. Row ``r`` reads block table ``REQ[r]`` when ``HAS_REQ`` (packed
    context rows of several requests), else table ``r``. ``ROWS > 1`` is for
    query tokens that share a block table (one request, or one request per
    group under ``HAS_REQ``): the ``BP`` pool keys are gathered once and reused
    for every row of the program. A group straddling two requests (only at
    request boundaries) gathers per row instead.
    """
    # The FP32 score matrix can exceed 2**31 elements during long-context prefill.
    r0 = tl.program_id(0).to(tl.int64) * ROWS
    j0 = tl.program_id(1) * BP
    j = j0 + tl.arange(0, BP)
    d = tl.arange(0, HD)
    h = tl.arange(0, HP)
    hmask = h < H
    r_last = tl.minimum(r0 + ROWS - 1, num_rows - 1)
    pos = j.to(tl.int64) * KPOOL
    page = pos // tpb
    off = pos - page * tpb
    if HAS_REQ:
        t0 = tl.load(REQ + r0).to(tl.int64)
        t_last = tl.load(REQ + r_last).to(tl.int64)
        shared = t0 == t_last
    else:
        t0 = r0.to(tl.int64)
        shared = True
    if shared:
        # The last row of the group has the largest visible length (rows of
        # one request are in position order); a block past its candidates is
        # past every row's.
        kv_last = tl.load(KV_LENS + r_last).to(tl.int64)
        if j0 >= kv_last // KPOOL:
            for i in tl.static_range(ROWS):
                r = r0 + i
                tl.store(
                    OUT + r * num_pools_max + j,
                    tl.full([BP], min_value, tl.float32),
                    mask=(j < num_pools_max) & (r < num_rows),
                )
        else:
            valid_any = j < kv_last // KPOOL
            slot = tl.load(BT + t0 * bt_stride + page, mask=valid_any, other=0).to(tl.int64)
            kptr = POOL + slot * slot_stride + off * row_stride + 2 * HD
            keys = tl.load(kptr[:, None] + d[None, :], mask=valid_any[:, None], other=0.0).to(
                DOT_DTYPE
            )
            for i in tl.static_range(ROWS):
                r = r0 + i
                in_range = r < num_rows
                kv_len = tl.load(KV_LENS + r, mask=in_range, other=0).to(tl.int64)
                valid = j < kv_len // KPOOL
                q = tl.load(
                    Q + r * H * HD + h[:, None] * HD + d[None, :],
                    mask=hmask[:, None] & in_range,
                    other=0.0,
                ).to(DOT_DTYPE)
                scores = tl.dot(q, tl.trans(keys), input_precision=PRECISION)  # [HP, BP]
                scores = tl.maximum(scores * q_scale, 0.0)
                w = tl.load(
                    W + r * w_row_stride + h * w_head_stride, mask=hmask & in_range, other=0.0
                ).to(tl.float32)
                mixed = tl.sum(scores * (w * w_scale)[:, None], axis=0)
                mixed = tl.where(valid, mixed, min_value)
                tl.store(OUT + r * num_pools_max + j, mixed, mask=(j < num_pools_max) & in_range)
    else:
        for i in tl.static_range(ROWS):
            r = r0 + i
            in_range = r < num_rows
            kv_len = tl.load(KV_LENS + r, mask=in_range, other=0).to(tl.int64)
            tbl = tl.load(REQ + r, mask=in_range, other=0).to(tl.int64)
            valid = j < kv_len // KPOOL
            slot = tl.load(BT + tbl * bt_stride + page, mask=valid, other=0).to(tl.int64)
            kptr = POOL + slot * slot_stride + off * row_stride + 2 * HD
            keys = tl.load(kptr[:, None] + d[None, :], mask=valid[:, None], other=0.0).to(DOT_DTYPE)
            q = tl.load(
                Q + r * H * HD + h[:, None] * HD + d[None, :],
                mask=hmask[:, None] & in_range,
                other=0.0,
            ).to(DOT_DTYPE)
            scores = tl.dot(q, tl.trans(keys), input_precision=PRECISION)  # [HP, BP]
            scores = tl.maximum(scores * q_scale, 0.0)
            w = tl.load(
                W + r * w_row_stride + h * w_head_stride, mask=hmask & in_range, other=0.0
            ).to(tl.float32)
            mixed = tl.sum(scores * (w * w_scale)[:, None], axis=0)
            mixed = tl.where(valid, mixed, min_value)
            tl.store(OUT + r * num_pools_max + j, mixed, mask=(j < num_pools_max) & in_range)


class KpoolScore(OpWrapper):
    """``q`` BF16 ``[N, H, head_dim]`` (contiguous), ``weights`` ``[N, H]`` (any strides) ->
    FP32 ``[N, num_pools_max]`` pool scores.

    Row ``r`` reads block table ``request_ids[r]`` when given, else table ``r``, and sees
    ``kv_lens[r]`` positions: columns ``[0, kv_lens[r] // kpool)`` are its scores. Without
    ``max_kv_len`` every other column is the FP32 minimum. With ``max_kv_len`` (a host bound on
    every ``kv_lens[r]``) only the 64-pool blocks some row can see are launched; columns past
    them are left unwritten, so a reader stops at each row's ``kv_lens[r] // kpool``.
    ``rows_per_program > 1`` scores that many consecutive rows per program and gathers their
    shared pool keys once: the rows must share a table, either a stride-0 broadcast table or
    ``request_ids`` with each request's rows consecutive and in position order.
    """

    ARCHS = frozenset({Arch.SM_100})
    # The indexer has 32 heads of 128 and scores pools of 4 over 32-token blocks. The tables
    # are 4,100 pages wide, so the capacity is num_pools_max = 32,800 pools. Generation cells
    # give the requests' KV length after the step; context cells give (cached, new) per
    # request and the scored row range. A verify step scores one row per program on per-row
    # copies of the generation tables; packed context rows score 16 per program.
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
            why="one rank's sequence-parallel quarter of a 13,312-token prompt (rows 0-3327)",
            spec=dict(context=((0, 13312),), rows=(0, 3328)),
        ),
        Cell(
            why="a 13,312-token prompt's last sequence-parallel quarter (rows 9984-13311)",
            spec=dict(context=((0, 13312),), rows=(9984, 13312)),
        ),
        Cell(
            why="two packed context requests: a 16-row group straddles the request boundary",
            spec=dict(context=((4099, 1001), (0, 1313)), rows=(0, 2314)),
        ),
    )

    note = """
    The packed context rows of this model are scored by `kpool_score_fp8` on sm_100; this
    scorer takes them when the FP8 scorer has nothing to score (no complete pool) or the
    BF16 path is forced (TRTLLM_GLM_INDEXER_BF16=1 in the module path).

    Only BF16 queries and BF16 pool keys: the dot reads both as BF16, so an FP32 query would
    be rounded to BF16 silently. `is_valid` refuses it.

    `rows_per_program > 1` without `request_ids` and without a stride-0 table raises. With
    `request_ids`, a group whose first and last rows share a request but whose middle rows do
    not would score the middle rows against the wrong table: `is_valid` requires request ids
    in non-decreasing order and each request's visible lengths non-decreasing.
    """

    def __call__(
        self,
        q,
        weights,
        index_pool,
        block_tables,
        kv_lens,
        tokens_per_block,
        *,
        num_pools_max,
        head_dim,
        kpool,
        q_scale,
        w_scale,
        rows_per_program=1,
        request_ids=None,
        max_kv_len=None,
    ):
        n, num_heads, hd = q.shape
        if hd != head_dim:
            raise ValueError(f"kpool_score: q head_dim {hd} != {head_dim}")
        out = torch.empty(n, num_pools_max, dtype=torch.float32, device=q.device)
        if n == 0:
            return out
        rows = max(1, int(rows_per_program))
        if rows > 1 and n > 1 and request_ids is None and block_tables.stride(0) != 0:
            raise ValueError(
                "kpool_score: rows_per_program > 1 needs a broadcast block table or request_ids"
            )
        bp = 64
        score_blocks = triton.cdiv(num_pools_max, bp)
        if max_kv_len is not None:
            score_blocks = min(score_blocks, triton.cdiv(max(0, int(max_kv_len)) // kpool, bp))
            if score_blocks == 0:
                return out
        _kpool_score_kernel[(triton.cdiv(n, rows), score_blocks)](
            q,
            weights,
            index_pool,
            block_tables,
            kv_lens,
            kv_lens if request_ids is None else request_ids,
            out,
            n,
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
            ROWS=rows,
            # BF16 operands into the tensor-core dot, FP32 accumulation.
            PRECISION="ieee",
            DOT_DTYPE=tl.bfloat16,
            HAS_REQ=request_ids is not None,
            # Row groups are register-heavy; two warps per program keep more
            # programs resident, which hides the gathered key loads.
            num_warps=2 if rows > 1 else 4,
        )
        return out

    def reference(
        self,
        q,
        weights,
        index_pool,
        block_tables,
        kv_lens,
        tokens_per_block,
        *,
        num_pools_max,
        head_dim,
        kpool,
        q_scale,
        w_scale,
        rows_per_program=1,
        request_ids=None,
        max_kv_len=None,
    ):
        """The scores in FP32 from the BF16 operands, every invisible column the FP32 minimum.

        ``rows_per_program`` and ``max_kv_len`` change only which columns the op writes, not
        what it writes, so they are not read here.
        """
        n = q.shape[0]
        dev = q.device
        out = torch.full((n, num_pools_max), FP32_MIN, dtype=torch.float32, device=dev)
        if n == 0:
            return out
        table = torch.arange(n, device=dev) if request_ids is None else request_ids.long()
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
        q,
        weights,
        index_pool,
        block_tables,
        kv_lens,
        tokens_per_block,
        *,
        num_pools_max,
        head_dim,
        kpool,
        q_scale,
        w_scale,
        rows_per_program=1,
        request_ids=None,
        max_kv_len=None,
    ):
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
        widest = int(kv_lens.max())
        # A row with more pools than num_pools_max loses the rest without a sign.
        assert widest // kpool <= num_pools_max, f"{widest // kpool} pools > {num_pools_max}"
        assert widest <= block_tables.shape[1] * tokens_per_block, (
            f"a visible length of {widest} exceeds the tables' "
            f"{block_tables.shape[1] * tokens_per_block} positions"
        )
        if max_kv_len is not None:
            # Columns past max_kv_len // kpool are never written.
            assert widest <= max_kv_len, f"kv_lens reach {widest} > max_kv_len {max_kv_len}"
        if request_ids is None:
            assert block_tables.stride(0) == 0 or block_tables.shape[0] >= n
        else:
            assert request_ids.shape == (n,)
            assert 0 <= int(request_ids.min()) and int(request_ids.max()) < block_tables.shape[0]
        if rows_per_program > 1 and n > 1:
            # A group shares its first row's table and its last row's candidates.
            lens = kv_lens.long()
            if request_ids is None:
                assert bool((lens.diff() >= 0).all()), "rows must be in position order"
            else:
                req = request_ids.long()
                assert bool((req.diff() >= 0).all()), "each request's rows must be consecutive"
                assert bool(((lens.diff() >= 0) | (req.diff() != 0)).all()), (
                    "each request's rows must be in position order"
                )


kpool_score = KpoolScore()
