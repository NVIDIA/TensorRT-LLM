# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compressed-indexer selection: head-weighted relu group scores -> top-k groups -> packed latent rows.

Ids leave ascending from column 0, ``-1`` past the row's length; the consumer applies no causal mask.
Scoring adapted from SGLang's ``_logits_kernel`` (``hy4_flash_indexpool.py``, Apache-2.0); radix select
and expand adapted from glm5_next's ``kpool_select.py`` (Apache-2.0).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper, true_fp32_matmul

#: Groups scored per `tl.dot` step (the dot's N; at least 16, and the register tile stays small).
BLOCK_G = 128
#: Score columns one radix pass and one emit step stream.
CHUNK = 2048
#: Columns of -1 padding written per step.
PAD_BLOCK = 512


@triton.jit
def _order_key(scores):
    """FP32 -> uint32 with the same order (larger float, larger key)."""
    bits = scores.to(tl.uint32, bitcast=True)
    return tl.where((bits >> 31) != 0, ~bits, bits | 0x80000000)


@triton.jit
def _in_prefix(key, thresh, shift):
    """Whether ``key`` agrees with ``thresh`` on the bits above ``shift + 8`` (all of them in
    the first pass; two shifts keep every shift amount below 32)."""
    return ((key >> (shift + 7)) >> 1) == ((thresh >> (shift + 7)) >> 1)


@triton.jit
def _radix_step(hist, remaining, thresh, shift):
    """Fix the 8 key bits at ``shift`` of the ``remaining``-th largest key from their histogram."""
    bins = tl.arange(0, 256)
    at_or_above = tl.cumsum(hist, 0, reverse=True)
    b = tl.max(tl.where(at_or_above >= remaining, bins, -1), 0)
    remaining -= tl.sum(tl.where(bins > b, hist, 0), 0)
    return remaining, thresh | (b.to(tl.uint32) << shift)


@triton.jit
def _select(srow, sel_row, num_cand, SELECT_K: tl.constexpr, CHUNK: tl.constexpr):
    """The ``SELECT_K`` largest of ``num_cand`` scores, ascending ids, into ``sel_row``; returns the count.

    Four 8-bit radix passes find the threshold key; ties at it resolve to the lower group id.
    """
    lane = tl.arange(0, CHUNK)
    thresh = tl.zeros([], tl.uint32)
    remaining = tl.full([], SELECT_K, tl.int32)
    if num_cand > SELECT_K:
        for p in range(4):
            shift = (24 - 8 * p).to(tl.uint32)
            hist = tl.zeros([256], tl.int32)
            for c0 in range(0, num_cand, CHUNK):
                idx = c0 + lane
                m = idx < num_cand
                key = _order_key(tl.load(srow + idx, mask=m, other=0.0))
                m = m & _in_prefix(key, thresh, shift)
                hist += tl.histogram(((key >> shift) & 255).to(tl.int32), 256, mask=m)
            remaining, thresh = _radix_step(hist, remaining, thresh, shift)
    else:
        remaining = tl.full([], 1 << 30, tl.int32)
    n_sel = tl.zeros([], tl.int32)
    ties_seen = tl.zeros([], tl.int32)
    for c0 in range(0, num_cand, CHUNK):
        idx = c0 + lane
        m = idx < num_cand
        key = _order_key(tl.load(srow + idx, mask=m, other=0.0))
        eq = m & (key == thresh)
        eq_rank = ties_seen + tl.cumsum(eq.to(tl.int32), 0)
        sel = (m & (key > thresh)) | (eq & (eq_rank <= remaining))
        dest = n_sel + tl.cumsum(sel.to(tl.int32), 0) - 1
        tl.store(sel_row + dest, idx.to(tl.int32), mask=sel)
        ties_seen += tl.sum(eq.to(tl.int32), 0)
        n_sel += tl.sum(sel.to(tl.int32), 0)
    return n_sel


@triton.jit
def _indexer_select_kernel(
    QUERY,  # [R, H, D] bf16, the indexer's rotated query
    GROUP_KEYS,  # [B, G, D] bf16, the request's rotated group keys
    HEAD_W,  # [R, H] fp32, the head-weight projection's output
    ROW_BATCH,  # [R] int32, row -> request
    POSITIONS,  # [R] int32, the row's own token position; -1 for a padded row
    LOGITS,  # [R, G] fp32 scratch
    SEL,  # [R, SELECT_K] int32 scratch
    PACKED,  # [R, OUT_W] int32
    LENGTHS,  # [R] int32
    NONEMPTY,  # [R] bool
    q_stride_r,
    g_stride_b,
    logits_stride,
    groups,
    scale,
    D: tl.constexpr,
    H: tl.constexpr,
    HP: tl.constexpr,
    RATIO: tl.constexpr,
    RATIO_P2: tl.constexpr,
    SELECT_K: tl.constexpr,
    KP2: tl.constexpr,
    OUT_W: tl.constexpr,
    BLOCK_G: tl.constexpr,
    CHUNK: tl.constexpr,
    PAD_BLOCK: tl.constexpr,
):
    """One program per row. ``pos < 0`` (padded row): one dummy id, length 1, ``nonempty`` False."""
    r = tl.program_id(0).to(tl.int64)
    pos = tl.load(POSITIONS + r).to(tl.int64)
    row = PACKED + r * OUT_W
    for c0 in range(0, OUT_W, PAD_BLOCK):
        col = c0 + tl.arange(0, PAD_BLOCK)
        tl.store(row + col, tl.full([PAD_BLOCK], -1, tl.int32), mask=col < OUT_W)
    if pos < 0:
        tl.store(row, 0)
        tl.store(LENGTHS + r, 1)
        tl.store(NONEMPTY + r, pos >= 0)
    else:
        b = tl.load(ROW_BATCH + r).to(tl.int64)
        qg = pos // RATIO
        num_cand = tl.minimum(qg, groups.to(tl.int64)).to(tl.int32)
        hs = tl.arange(0, HP)
        d = tl.arange(0, D)
        q = tl.load(
            QUERY + r * q_stride_r + hs[:, None] * D + d[None, :],
            mask=(hs < H)[:, None],
            other=0.0,
        ).to(tl.float32)
        w = tl.load(HEAD_W + r * H + hs, mask=hs < H, other=0.0) * scale
        lrow = LOGITS + r * logits_stride
        for g0 in range(0, num_cand, BLOCK_G):
            g = g0 + tl.arange(0, BLOCK_G)
            gm = g < num_cand
            k = tl.load(
                GROUP_KEYS + b * g_stride_b + g[:, None] * D + d[None, :],
                mask=gm[:, None],
                other=0.0,
            ).to(tl.float32)
            s = tl.dot(q, tl.trans(k), input_precision="ieee")
            s = tl.maximum(s, 0.0) * w[:, None]
            tl.store(lrow + g, tl.sum(s, axis=0), mask=gm)
        # The select and the expand read what this program wrote.
        tl.debug_barrier()
        n_sel = _select(lrow, SEL + r * SELECT_K, num_cand, SELECT_K, CHUNK)
        tl.debug_barrier()
        # Selected group i -> columns [i * RATIO, (i + 1) * RATIO).
        i = tl.arange(0, KP2)
        gi = tl.load(SEL + r * SELECT_K + i, mask=i < n_sel, other=0).to(tl.int64)
        m = tl.arange(0, RATIO_P2)
        ok = (i < n_sel)[:, None] & (m < RATIO)[None, :]
        tl.store(
            row + i[:, None] * RATIO + m[None, :],
            (gi[:, None] * RATIO + m[None, :]).to(tl.int32),
            mask=ok,
        )
        # The row's own group, up to its own position: the slots no later group's key covers.
        t = tl.arange(0, RATIO_P2)
        tail = (pos - qg * RATIO).to(tl.int32)
        tl.store(row + n_sel * RATIO + t, (qg * RATIO + t).to(tl.int32), mask=t <= tail)
        tl.store(LENGTHS + r, n_sel * RATIO + tail + 1)
        tl.store(NONEMPTY + r, pos >= 0)


def _reference_logits(
    query: torch.Tensor,
    group_keys: torch.Tensor,
    head_weights: torch.Tensor,
    row_batch: torch.Tensor,
    positions: torch.Tensor,
    *,
    ratio: int,
    scale: float,
) -> torch.Tensor:
    """``[R, G]`` head-weighted relu scores, ``-inf`` at and past the row's own group; in the operands' dtype."""
    rows = query.shape[0]
    batch, groups, _ = group_keys.shape
    logits = torch.full((rows, groups), float("-inf"), dtype=query.dtype, device=query.device)
    col = torch.arange(groups, device=query.device)
    rb, pos = row_batch.long(), positions.long()
    for b in range(batch):
        rows_b = (rb == b).nonzero().flatten()
        if rows_b.numel() == 0:
            continue
        qg = torch.div(pos[rows_b], ratio, rounding_mode="floor")
        for c0 in range(0, rows_b.numel(), 1024):
            part = rows_b[c0 : c0 + 1024]
            part_qg = qg[c0 : c0 + 1024]
            with true_fp32_matmul():
                scores = torch.einsum("rhd,gd->rhg", query[part], group_keys[b])
            scores = scores.relu() * (head_weights[part] * scale)[:, :, None]
            visible = col[None, :] < part_qg[:, None]
            logits[part] = torch.where(visible, scores.sum(dim=1), float("-inf"))
    return logits


class IndexerSelect(OpWrapper):
    """-> (``packed`` int32 ``[R, index_topk]``, ``lengths`` int32 ``[R]``, ``nonempty`` bool ``[R]``).

    ``query``/``group_keys`` arrive mRoPE-rotated; ``head_weights`` is unscaled; ``row_batch`` maps row -> pool.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="one decode row at the envelope's max length: 5120 tokens -> 1280 groups",
            spec=dict(rows=1, kv=5120),
        ),
        Cell(
            why="decode at max_batch_size: 64 rows x 5120 tokens, 2048 ids each",
            spec=dict(rows=64, kv=5120),
        ),
        Cell(
            why="prefill of the envelope's largest row count: 5120 rows over one 5120-token pool",
            spec=dict(rows=5120, kv=5120, mode="prefill"),
        ),
        Cell(
            why="shortest context the indexer runs on: 8 rows, 2049 tokens -> 512 groups, 511 recalled",
            spec=dict(rows=8, kv=2049),
        ),
        Cell(
            why="captured batch with two padded rows: 8 rows, 2 of them at position -1",
            spec=dict(rows=8, kv=5120, inactive=2),
        ),
    )

    def __call__(
        self,
        query: torch.Tensor,
        group_keys: torch.Tensor,
        head_weights: torch.Tensor,
        row_batch: torch.Tensor,
        positions: torch.Tensor,
        *,
        ratio: int = 4,
        index_topk: int = 2048,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rows, heads, dim = query.shape
        batch, groups, _ = group_keys.shape
        select_k = index_topk // ratio - 1
        if ratio < 1:
            raise ValueError(f"indexer_select: ratio {ratio} must be positive")
        if select_k < 1:
            raise ValueError(
                f"indexer_select: index_topk {index_topk} // ratio {ratio} must recall at "
                f"least one group"
            )
        if dim < 16 or dim & (dim - 1):
            raise ValueError(f"indexer_select: head_dim {dim} must be a power of two, at least 16")
        if heads < 1:
            raise ValueError(f"indexer_select: heads {heads} must be positive")
        packed = torch.empty((rows, index_topk), dtype=torch.int32, device=query.device)
        lengths = torch.empty(rows, dtype=torch.int32, device=query.device)
        nonempty = torch.empty(rows, dtype=torch.bool, device=query.device)
        if rows == 0:
            return packed, lengths, nonempty
        logits = torch.empty((rows, groups), dtype=torch.float32, device=query.device)
        sel = torch.empty((rows, select_k), dtype=torch.int32, device=query.device)
        _indexer_select_kernel[(rows,)](
            query,
            group_keys,
            head_weights,
            row_batch,
            positions,
            logits,
            sel,
            packed,
            lengths,
            nonempty,
            query.stride(0),
            group_keys.stride(0),
            logits.stride(0),
            groups,
            (heads * dim) ** -0.5,
            D=dim,
            H=heads,
            HP=triton.next_power_of_2(heads),
            RATIO=ratio,
            RATIO_P2=triton.next_power_of_2(ratio),
            SELECT_K=select_k,
            KP2=triton.next_power_of_2(select_k),
            OUT_W=index_topk,
            BLOCK_G=BLOCK_G,
            CHUNK=CHUNK,
            PAD_BLOCK=PAD_BLOCK,
            num_warps=8,
        )
        return packed, lengths, nonempty

    def reference(
        self,
        query: torch.Tensor,
        group_keys: torch.Tensor,
        head_weights: torch.Tensor,
        row_batch: torch.Tensor,
        positions: torch.Tensor,
        *,
        ratio: int = 4,
        index_topk: int = 2048,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The same selection in plain torch, fp32 throughout."""
        rows, heads, dim = query.shape
        _, groups, _ = group_keys.shape
        dev = query.device
        logits = _reference_logits(
            query.float(),
            group_keys.float(),
            head_weights.float(),
            row_batch,
            positions,
            ratio=ratio,
            scale=(heads * dim) ** -0.5,
        )
        select_k = min(index_topk // ratio - 1, groups)
        if select_k > 0:
            top = torch.topk(logits, select_k, dim=1).indices
            # Drop -inf (not visible) picks; survivors ascending, then -1.
            top = torch.where(torch.gather(logits, 1, top) > float("-inf"), top, -1)
            top = torch.where(top >= 0, top, torch.full_like(top, groups))
            top = top.sort(dim=1).values
            top = torch.where(top < groups, top, torch.full_like(top, -1))
        else:
            top = torch.empty((rows, 0), dtype=torch.int64, device=dev)
        k = top.shape[1]
        active = positions >= 0
        qg = torch.div(positions.long(), ratio, rounding_mode="floor").clamp(min=0)
        n_sel = (top >= 0).sum(dim=1)
        tail = torch.where(active, positions.long() - qg * ratio, torch.full_like(qg, -1))
        length = n_sel * ratio + tail + 1
        packed = torch.full((rows, index_topk), -1, dtype=torch.int32, device=dev)
        if k:
            hist = top.clamp(min=0)[:, :, None] * ratio + torch.arange(ratio, device=dev)
            hist = hist.reshape(rows, k * ratio)
            hist = torch.where(
                (top >= 0)[:, :, None].expand(rows, k, ratio).reshape(rows, k * ratio),
                hist,
                torch.full_like(hist, -1),
            )
            packed[:, : k * ratio] = hist.to(torch.int32)
        col = torch.arange(index_topk, device=dev)
        tail_idx = col[None, :] - (n_sel * ratio)[:, None]
        tail_ok = (tail_idx >= 0) & (tail_idx <= tail[:, None])
        packed = torch.where(tail_ok, (qg[:, None] * ratio + tail_idx).to(torch.int32), packed)
        empty = length == 0
        if bool(empty.any()):
            packed[empty, 0] = 0
        return packed, length.clamp(min=1).to(torch.int32), length > 0

    def is_valid(
        self,
        query: torch.Tensor,
        group_keys: torch.Tensor,
        head_weights: torch.Tensor,
        row_batch: torch.Tensor,
        positions: torch.Tensor,
        *,
        ratio: int = 4,
        index_topk: int = 2048,
    ) -> None:
        rows, heads, dim = query.shape
        _, groups, _ = group_keys.shape
        # Dense inner strides; only the row strides are taken from the tensors.
        assert tuple(query.stride()[1:]) == (dim, 1), (
            f"query must be dense within a row, [heads, head_dim]; got strides "
            f"{tuple(query.stride())} (a last-dim slice of a wider tensor reads its neighbour's "
            f"columns)"
        )
        assert tuple(group_keys.stride()[1:]) == (dim, 1), (
            f"group_keys must be dense within a request, [groups, head_dim]; got strides "
            f"{tuple(group_keys.stride())} (the kernel reads the groups `head_dim` apart, so a "
            f"pool whose groups sit `stride(1)` apart scores the wrong keys)"
        )
        assert tuple(head_weights.shape) == (rows, heads) and head_weights.stride() == (
            heads,
            1,
        ), (
            f"head_weights must be dense [rows, heads] = ({rows}, {heads}); got "
            f"{tuple(head_weights.shape)} with strides {tuple(head_weights.stride())} (a "
            f"transposed one is accepted when rows == heads and computes another weighted sum)"
        )
        assert row_batch.shape == (rows,) and positions.shape == (rows,)
        # The kernel clamps candidates to the pool, so a short pool would silently drop visible groups.
        active = positions >= 0
        if bool(active.any()):
            assert int(positions[active].max()) // ratio <= groups, (
                f"a row at position {int(positions[active].max())} has "
                f"{int(positions[active].max()) // ratio} complete groups before its own, past "
                f"the pool's {groups}: the recall would silently miss the rest"
            )
            assert (
                int(row_batch[active].min()) >= 0
                and int(row_batch[active].max()) < group_keys.shape[0]
            ), "every active row must name a request of group_keys"


indexer_select = IndexerSelect()
