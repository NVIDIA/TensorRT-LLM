# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pool-key refresh of the paged indexer cache, one program per row.

Pool ``j`` of a request is its positions ``[j * kpool, (j + 1) * kpool)``. Its key sits in the
last ``head_dim`` columns of the indexer row ``[k | gate | pool key]`` of its first member and
is, per column ``d``,

    key[d] = sum_m softmax_m(gate[m, d] + ape[m, d]) * k[m, d]

over the members ``m`` visible at the refreshing row's position (later members are masked):
an FP32 softmax, the probabilities and the products rounded to BF16, an FP32 sum and a BF16
key, the numerics of the model's pool compression. A row refreshes the pool its position falls
in, so an incomplete pool carries the key of its visible members and the write of its last
member finalizes it.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper


@triton.jit
def _kpool_update_kernel(
    POOL,
    BT,
    POS,
    APE,
    REQ,
    slot_stride,
    row_stride,
    bt_stride,
    tpb,
    HD: tl.constexpr,
    KPOOL: tl.constexpr,
    HAS_REQ: tl.constexpr,
    ROWS_PER_TABLE: tl.constexpr,
):
    """Recompute the pool key of the pool containing ``POS[b]``.

    Members beyond ``POS[b]`` are masked (they are not visible yet), so an
    incomplete pool carries the key of its visible members and is finalized by
    the write of its last member. Row ``b`` reads block table ``REQ[b]`` when
    ``HAS_REQ`` (several rows of one request share a table), else table
    ``b // ROWS_PER_TABLE``. With ``ROWS_PER_TABLE > 1`` the rows of one table are
    consecutive positions; a row whose successor falls in the same pool is skipped,
    so each pool is written once with its last visible member, as sequential
    updates would leave it. Numerics follow ``build_pools``: fp32
    softmax over ``gate + ape``, bf16 probabilities and products, fp32 sum,
    bf16 result.
    """
    b = tl.program_id(0)
    if HAS_REQ:
        tbl = tl.load(REQ + b).to(tl.int64)
    else:
        tbl = (b // ROWS_PER_TABLE).to(tl.int64)
    pos = tl.load(POS + b).to(tl.int64)
    if ROWS_PER_TABLE > 1:
        if b % ROWS_PER_TABLE < ROWS_PER_TABLE - 1:
            if tl.load(POS + b + 1).to(tl.int64) // KPOOL == pos // KPOOL:
                return
    start = (pos // KPOOL) * KPOOL
    m = tl.arange(0, KPOOL)
    d = tl.arange(0, HD)
    mem_pos = start + m
    valid = mem_pos <= pos
    page = mem_pos // tpb
    off = mem_pos - page * tpb
    slot = tl.load(BT + tbl * bt_stride + page, mask=valid, other=0).to(tl.int64)
    row = POOL + slot * slot_stride + off * row_stride
    k = tl.load(row[:, None] + d[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
    g = tl.load(row[:, None] + HD + d[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
    ape = tl.load(APE + m[:, None] * HD + d[None, :]).to(tl.float32)
    logits = tl.where(valid[:, None], g + ape, float("-inf"))
    mx = tl.max(logits, axis=0)
    e = tl.exp(logits - mx[None, :])
    probs = e / tl.sum(e, axis=0)[None, :]
    probs = probs.to(tl.bfloat16).to(tl.float32)
    prod = (probs * k).to(tl.bfloat16).to(tl.float32)
    pool_key = tl.sum(prod, axis=0)
    page0 = start // tpb
    off0 = start - page0 * tpb
    slot0 = tl.load(BT + tbl * bt_stride + page0).to(tl.int64)
    tl.store(POOL + slot0 * slot_stride + off0 * row_stride + 2 * HD + d, pool_key.to(tl.bfloat16))


class KpoolUpdate(OpWrapper):
    """Refresh, in ``index_pool``, the key of the pool containing each row's position.

    ``index_pool`` BF16 ``[slots, tokens_per_block, >= 3 * head_dim]``, ``block_tables``
    ``[tables, pages]``, ``ape`` BF16 ``[kpool, head_dim]`` (the compression position bias).
    Two row layouts:

    - no ``request_ids``: ``positions`` ``[N, T]`` holds ``T`` consecutive positions of table
      ``n`` (a verify step's draft rows), or ``[N]`` one position of table ``n`` (single-token
      decode). The result equals refreshing the ``T`` columns in order: of several rows in one
      pool only the last writes.
    - ``request_ids`` int32 ``[N]`` with ``positions`` ``[N]``: row ``i`` reads table
      ``request_ids[i]`` (packed context rows). The caller passes one row per pool, the
      pool-final positions a step completes: two rows on one pool race.

    Writes only the pool-key columns of each refreshed pool's first-member row; returns None.
    """

    ARCHS = frozenset({Arch.SM_100})
    # The indexer key is 128 wide (cache rows of 3 * 128 BF16), pools of 4 over 32-token
    # blocks. Generation cells give the requests' KV length after the step; request b's is
    # kv - b % 4, so draft rows straddle pools differently per request. Context cells give
    # (cached, new) per request: the rows are the step's pool-final positions.
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
            why="single-token decode of eight 13K-token requests: 1-D positions",
            spec=dict(requests=8, tokens=1, kv=13312),
        ),
        Cell(
            why="a 13,312-token prompt: its 3,328 pool-final positions",
            spec=dict(context=((0, 13312),)),
        ),
        Cell(
            why="two requests: a chunk completing a cached pool, and a fresh prompt",
            spec=dict(context=((4099, 3000), (0, 1313))),
        ),
    )

    note = """
    The members of a refreshed pool are read from the `[k | gate]` columns, so a step writes
    its rows' indexer state (`mla_prologue`, `mla_context_prologue`) before refreshing.

    `request_ids` with 2-D positions raises. Several rows of one table that are not
    consecutive positions, or two rows on one pool with `request_ids`, leave a pool key that
    depends on scheduling; `is_valid` refuses both.
    """

    def __call__(
        self,
        index_pool: torch.Tensor,
        block_tables: torch.Tensor,
        positions: torch.Tensor,
        ape: torch.Tensor,
        tokens_per_block: int,
        *,
        head_dim: int,
        kpool: int,
        request_ids: torch.Tensor | None = None,
    ) -> None:
        if request_ids is not None and positions.dim() != 1:
            # The same-pool skip compares POS[b] with POS[b + 1], which is only valid
            # when consecutive rows belong to one table.
            raise ValueError("kpool_update: request_ids requires 1-D positions")
        rows_per_table = positions.shape[1] if positions.dim() == 2 else 1
        positions = positions.reshape(-1)
        n = positions.shape[0]
        if n == 0:
            return None
        _kpool_update_kernel[(n,)](
            index_pool,
            block_tables,
            positions,
            ape,
            positions if request_ids is None else request_ids,
            index_pool.stride(0),
            index_pool.stride(1),
            block_tables.stride(0),
            tokens_per_block,
            HD=head_dim,
            KPOOL=kpool,
            HAS_REQ=request_ids is not None,
            ROWS_PER_TABLE=rows_per_table,
            num_warps=4,
        )
        return None

    def reference(
        self,
        index_pool: torch.Tensor,
        block_tables: torch.Tensor,
        positions: torch.Tensor,
        ape: torch.Tensor,
        tokens_per_block: int,
        *,
        head_dim: int,
        kpool: int,
        request_ids: torch.Tensor | None = None,
    ) -> None:
        """The refresh in plain torch, in place on ``index_pool``: every row's key from the
        members visible at its position, and of the rows on one pool the last one's, as
        refreshing the rows one after another leaves it. The BF16 roundings of the
        probabilities and products are the pool key's definition, not a kernel artifact."""
        rows_per_table = positions.shape[1] if positions.dim() == 2 else 1
        pos = positions.reshape(-1).long()
        n = pos.shape[0]
        if n == 0:
            return None
        dev = pos.device
        order = torch.arange(n, device=dev)
        table = order // rows_per_table if request_ids is None else request_ids.long()
        start = pos // kpool * kpool
        member = start[:, None] + torch.arange(kpool, device=dev)  # [n, kpool]
        visible = member <= pos[:, None]
        page = (member // tokens_per_block).clamp(max=block_tables.shape[1] - 1)
        slot = block_tables[table[:, None], page].long()
        off = member % tokens_per_block
        rows = index_pool[slot, off].float()  # [n, kpool, width]
        # Members past the position are masked out of the sums, whatever their rows hold.
        rows = rows.masked_fill(~visible[:, :, None], 0.0)
        k, gate = rows[..., :head_dim], rows[..., head_dim : 2 * head_dim]
        logits = (gate + ape.float()).masked_fill(~visible[:, :, None], float("-inf"))
        probs = logits.softmax(dim=1).to(torch.bfloat16).float()
        key = (probs * k).to(torch.bfloat16).float().sum(dim=1).to(torch.bfloat16)
        # The last row on each pool's first-member row wins.
        dest = slot[:, 0] * tokens_per_block + off[:, 0]
        last = torch.full((int(dest.max()) + 1,), -1, dtype=torch.long, device=dev)
        last = last.scatter_reduce(0, dest, order, reduce="amax")
        keep = last[dest] == order
        index_pool[slot[keep, 0], off[keep, 0], 2 * head_dim : 3 * head_dim] = key[keep]
        return None

    def is_valid(
        self,
        index_pool: torch.Tensor,
        block_tables: torch.Tensor,
        positions: torch.Tensor,
        ape: torch.Tensor,
        tokens_per_block: int,
        *,
        head_dim: int,
        kpool: int,
        request_ids: torch.Tensor | None = None,
    ) -> None:
        assert index_pool.dtype == torch.bfloat16 and index_pool.dim() == 3
        assert index_pool.stride(2) == 1 and index_pool.shape[1] == tokens_per_block
        assert index_pool.shape[2] >= 3 * head_dim, "rows are [k | gate | pool key]"
        for name, width in (("head_dim", head_dim), ("kpool", kpool)):
            assert width > 0 and width & (width - 1) == 0, f"{name} {width} must be a power of 2"
        # The kernel reads ape at m * head_dim + d.
        assert ape.shape == (kpool, head_dim) and ape.is_contiguous()
        assert block_tables.dim() == 2 and block_tables.stride(1) == 1
        assert positions.dim() in (1, 2)
        pos = positions.reshape(-1).long()
        n = pos.shape[0]
        if n == 0:
            return
        rows_per_table = positions.shape[1] if positions.dim() == 2 else 1
        if positions.dim() == 2:
            # The same-pool skip lets the later of two consecutive rows write.
            assert bool((positions.diff(dim=1) == 1).all()), "a table's rows must be consecutive"
        if request_ids is None:
            table = torch.arange(n, device=pos.device) // rows_per_table
        else:
            assert request_ids.shape == (n,)
            table = request_ids.long()
        assert 0 <= int(table.min()) and int(table.max()) < block_tables.shape[0]
        assert int(pos.min()) >= 0
        assert int(pos.max()) < block_tables.shape[1] * tokens_per_block
        # One writer per pool and table: with request_ids one row, else one table's rows.
        first = pos // kpool * kpool
        page = first // tokens_per_block
        dest = block_tables[table, page].long() * tokens_per_block + first % tokens_per_block
        if request_ids is not None:
            assert dest.unique().numel() == n, "two rows refresh one pool"
        else:
            pairs = torch.stack([dest, table]).unique(dim=1)
            assert pairs.shape[1] == dest.unique().numel(), "two tables share a pool's row"


kpool_update = KpoolUpdate()
