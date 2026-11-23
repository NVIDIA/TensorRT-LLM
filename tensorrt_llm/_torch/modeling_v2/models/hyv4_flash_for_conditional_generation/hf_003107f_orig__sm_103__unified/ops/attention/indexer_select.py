# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compressed-indexer selection of the Hy4-Flash indexer: group scores to packed latent rows,
one launch.

A query row ``r`` scores every complete group of its request's pool that closed before it,
one weighted sum over the indexer's heads, keeps the best ``index_topk // ratio - 1`` of them
plus the group it sits in, and expands the kept groups to the token slots they stand for::

    scores[g] = relu(q_h . k_g)                                   # fp32, per head h
    logits[g] = sum_h w_h * scores[g] * (H * D)**-0.5              # fp32
    recall    = the min(group_topk - 1, visible) best logits, plus the row's own group
    ids       = each recalled group g -> g * ratio + [0, ratio), then the own group's
                slots up to the row's own position

The ids leave ascending from column 0, ``-1`` past the row's length: the ``packed`` layout
``sparse_mla_decode`` reads (its first ``lengths[r]`` columns are latent-pool rows, and it
applies no causal mask of its own).

``HYV4FlashCompressedIndexer.forward`` in the checkpoint's own implementation is
``scores``/``logits`` above, the mask ``group_ids >= query_group_index``, ``complete_topk =
min(group_topk - 1, num_groups)``, ``recalled_groups = cat(topk, query_group_index)``, the
expansion ``recalled_groups * ratio + offsets`` clamped to ``kv_length - 1``, and
``head_weights * (n_heads**-0.5 * head_dim**-0.5)``. It folds the recall into the causal
attention mask; this entry returns the ids that survive that mask, and drops the clamp because
every id it emits is at most the row's own position, which is inside the cache.

The scoring is adapted from SGLang's ``_logits_kernel``
(``python/sglang/kernels/ops/attention/dsa/hy4_flash_indexpool.py``, Apache-2.0): its
``relu(q . k)`` per head, weighted by the head weight and summed over heads. The radix select,
the barrier between the program's own write and read of scratch, and the expand-and-pad are
adapted from glm5_next's target entry ``ops/attention/kpool_select.py`` (Apache-2.0). Both are
one launch here: the scores go to a per-row scratch, the program selects from it, and it
expands from its own selection.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper, true_fp32_matmul

#: Groups scored per `tl.dot` step (the dot's N; at least 16, and the register tile stays small).
BLOCK_G = 128
#: Score columns one radix pass and one emit step stream; a row of the run's envelope (1280
#: groups) is one step, a longer pool streams.
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
    """The ``SELECT_K`` largest of the row's ``num_cand`` scores, ascending, written to
    ``sel_row``; returns their count, ``min(SELECT_K, num_cand)``.

    Four 8-bit radix passes find the threshold's exact FP32 key, then one emit pass streams the
    row again and writes every score above it plus the ties it needs, in ascending id order, so
    a tie at the threshold resolves to the lower group id. With ``num_cand <= SELECT_K`` every
    candidate is selected (the threshold stays 0, below every real key) and no radix pass runs.
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
    """One program per row: score, select, expand.

    ``pos < 0`` is a padded row of a captured batch: one dummy id, length 1, ``nonempty``
    False, so the consumer has a row it may read and a flag that says it holds nothing.
    """
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
    """``[R, G]``: the head-weighted relu score of every group, ``-inf`` at and past the row's
    own group (the checkpoint's ``masked_fill(group_ids >= query_group_index)``).

    Computed in the operands' own dtype, so a caller may pass fp32 tensors (the reference's
    precision) or fp64 ones (a recompute to measure the fp32 error against)."""
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
    """``query`` BF16 ``[R, H, D]``, ``group_keys`` BF16 ``[B, G, D]``, ``head_weights`` FP32
    ``[R, H]``, ``row_batch`` int32 ``[R]`` and ``positions`` int32 ``[R]`` ->
    (``packed`` int32 ``[R, index_topk]``, ``lengths`` int32 ``[R]``, ``nonempty`` bool ``[R]``).

    ``query`` and ``group_keys`` arrive already rotated with the indexer's own interleaved
    mRoPE (the query at its row's position, the group keys at their group's first position);
    ``head_weights`` is ``head_weight_projection``'s output before the entry's
    ``(H * D) ** -0.5`` scale. ``row_batch`` maps a row to the request whose pool it scores, so
    a decode step is ``B`` rows over ``B`` pools and a prefill chunk is its rows over one.
    """

    ARCHS = frozenset({Arch.SM_103})

    # The checkpoint's indexer: 16 heads of 128, ratio 4, index_topk 2048, so 511 groups are
    # recalled and a row's ids reach 2048 columns. The vocabulary's call sites are one row at
    # the envelope's max length and 64 rows at max_batch_size; the envelope's largest row
    # count (a 5120-token prefill chunk) and the shortest context the indexer runs on
    # (2049 tokens -> 512 groups) bound the rest.
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

    note = """
    Called by the target's `Indexer.pool_and_select` in both phases: after `indexer_pool` and
    the indexer's mRoPE produced the group keys, and after `query_projection` and the query's
    mRoPE produced the query. Its `packed`/`lengths` are the `sparse_mla_decode` selection of
    the same row, and its `nonempty` is what lets a caller skip a row with nothing visible.
    The checkpoint skips the indexer entirely at `kv_length <= index_topk` (2048) and attends
    densely; this entry is correct there too (it then recalls every visible group, i.e. the
    dense causal prefix) but the target should not call it.

    Sizes: a row's candidate count is its own complete-group count `pos // ratio` clamped to
    the pool's `G`; `min(group_topk - 1, candidates)` groups are recalled, `group_topk =
    index_topk // ratio` as the checkpoint derives it. The output is `index_topk` columns wide,
    the consumer's width, and the longest row fills all of them (511 groups x 4 + the 4 slots
    of the own group).

    Preconditions:
      - `query` and `group_keys` are read with `D` between heads or groups and 1 within a head
        or key, while only the outer strides (`query.stride(0)`, `group_keys.stride(0)`) are
        taken from the tensors. A last-dim slice of a wider buffer is accepted and scores its
        neighbour's columns; a group-axis slice is not, its groups stay `D` apart and its
        outer stride is read. `is_valid` refuses the first.
      - `head_weights` is `[R, H]` dense. A transposed one has the same shape when `R == H`
        and is accepted, computing a different weighted sum. `is_valid` refuses it.
      - The pool must hold every complete group before a row's own group: a shorter pool is
        silently missing groups from the recall (the kernel clamps its candidate count to the
        pool). `is_valid` refuses it.
    Loud (not pinned by a test): `D` a power of two of at least 16, `index_topk // ratio` at
    least 2, `ratio` at least 1, and `row_batch`/`positions` shaped `[R]`; `__call__` raises
    `ValueError` for the first three. A zero-row call returns empty tensors.

    Precision: the interior is fp32, as the checkpoint computes it (`query.float()`,
    `keys.float()`, the weighted sum in fp32), and the output is ids, so the only thing a
    rounding can move is which group sits at the top-k threshold. The kernel's dot is an IEEE
    fp32 dot (`input_precision="ieee"`) of bf16-representable operands, so its products are
    exact and only the accumulation order differs from the reference's; the test derives its
    threshold band from a float64 recompute and compares the selected ids exactly wherever the
    k-th and (k+1)-th reference logits are farther apart than that band. Ties at the threshold
    are the one place the two may legitimately differ (the kernel prefers the lower group id;
    the reference takes torch's order), and the test compares the selected logits' multiset
    there. Measured on the five cells: the fp32 reference sits at most 9.72e-07 from a float64
    recompute of the same logits (8.74e-07 at 2049 tokens), the selected logits' multiset
    differs from the reference's by 0.0 on every cell, and of the 5195 rows that select a
    group, 5175 had a boundary gap past the test's 2e-05 band and selected the reference's
    groups exactly; the other 20 sat inside a tie (one of them the eight-valued pool of the
    tie test).

    Plain `@triton.jit`, no autotune, JIT-compiled at first call; the cells drove that JIT path
    on sm_103. `D`, `H`, `RATIO`, `SELECT_K` and the tiles are `tl.constexpr`, so each geometry
    is its own specialization. Not captured here: CUDA graphs at a fixed shape (the scratch is
    allocated per call, as the target's other select entry does, so a capture would have to
    take the allocation into its pool).

    Not certified: heads other than 16, `D` other than 128, `ratio` other than 4,
    `index_topk` other than 2048, a pool with fewer groups than a row's complete groups (see
    the guard), an fp32 or fp16 `query`/`group_keys`, an fp32 `head_weights` of a different
    width, `positions` past 2**31 - 1, rows past the envelope's 5120, and batches past 64.
    """

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
        """The same selection in plain torch, fp32 throughout.

        The top-k of ``_reference_logits`` (``-inf`` at and past the row's own group), with the
        rows that see fewer than ``select_k`` groups keeping what they see; then the expansion,
        the own group's tail up to the row's position, and ``-1`` after the row's length. A row
        with nothing visible (a padded row) gets one dummy id 0 and ``nonempty`` False, as the
        kernel writes.
        """
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
            # Groups past a row's own are -inf and must not be recalled as -inf ids; the
            # survivors go first, ascending, the rest -1 after them.
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
        # Read with dense inner strides while only the row strides are taken from the tensors.
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
        # A pool shorter than a row's complete groups drops visible groups from the recall
        # without a sign: the candidate count is clamped to the pool.
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
