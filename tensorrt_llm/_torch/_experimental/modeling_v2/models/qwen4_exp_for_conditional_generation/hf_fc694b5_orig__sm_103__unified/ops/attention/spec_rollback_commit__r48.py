# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""`SpeculativeRollback`'s staging and commit of a verify step in one launch each.

Tier 5 (Triton written for this target). A verify step runs `p` rows per generation request
through the trunk; the PLE pools and the QSA planes are written for every row. The staging
launch copies what the commit rewrites from: each request's pre-step n-gram id history
followed by its rows' ids (the `stream` whose length-`context` windows are the histories
after each row), and its pre-step PLE conv state. After the draft loop the commit launch
rewrites, for each generation request that accepted `a` rows:

* its n-gram pool row to window `a` of its stream (the history after row `a - 1`);
* its PLE conv pool row to columns `[a, a + state_len)` of `[pre-step state | the conv input
  of its p rows]`;
* the QSA position plane and every attention layer's index-K plane at each of its rows of
  ordinal `>= a` to the values gathered before the trunk wrote them.

The commit runs one program per (request, block): block `b` writes the request's conv
channels `[b * 256, (b + 1) * 256)` and restores its rejected rows in index-K plane `b`; block 0
also writes its n-gram row and position-plane slots.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from .qsa_rollback_planes__r47 import ROW

#: Conv channels a program covers.
_BLOCK_C = 256


@triton.jit
def _stage_kernel(
    ngram_pool,
    ngram_stride,
    conv_pool,
    conv_s0,
    conv_s1,
    conv_s2,
    slots,
    ids,
    stream,
    conv_prev,
    channels,
    P: tl.constexpr,
    CTX: tl.constexpr,
    W_BLOCK: tl.constexpr,
    STATE: tl.constexpr,
    STATE_BLOCK: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    i = tl.program_id(0)
    cb = tl.program_id(1)
    slot = tl.load(slots + i).to(tl.int64)
    c = cb * BLOCK_C + tl.arange(0, BLOCK_C)
    s = tl.arange(0, STATE_BLOCK)
    m = (c < channels)[:, None] & (s < STATE)[None, :]
    vals = tl.load(conv_pool + slot * conv_s0 + c[:, None] * conv_s1 + s[None, :] * conv_s2, mask=m)
    tl.store(
        conv_prev + (i.to(tl.int64) * channels + c[:, None]) * STATE + s[None, :], vals, mask=m
    )
    if cb == 0:
        w = tl.arange(0, W_BLOCK)
        hist = tl.load(ngram_pool + slot * ngram_stride + w, mask=w < CTX)
        row = tl.load(ids + i * P + (w - CTX), mask=(w >= CTX) & (w < CTX + P)).to(tl.int64)
        tl.store(stream + i * (CTX + P) + w, tl.where(w < CTX, hist, row), mask=w < CTX + P)


@triton.jit
def _commit_kernel(
    accepted,
    slots,
    ngram_pool,
    ngram_stride,
    stream,
    conv_pool,
    conv_s0,
    conv_s1,
    conv_s2,
    conv_prev,
    conv_x,
    x_s0,
    x_s1,
    x_s2,
    position,
    pos_s0,
    pos_s1,
    pos_s2,
    position_old,
    pages,
    within,
    plane_ptrs,
    plane_strides,
    index_old,
    num_planes,
    rows,
    channels,
    P: tl.constexpr,
    P_BLOCK: tl.constexpr,
    CTX: tl.constexpr,
    CTX_BLOCK: tl.constexpr,
    STATE: tl.constexpr,
    STATE_BLOCK: tl.constexpr,
    BLOCK_C: tl.constexpr,
    ROW: tl.constexpr,
):
    i = tl.program_id(0)
    cb = tl.program_id(1)
    a = tl.load(accepted + i).to(tl.int32)
    slot = tl.load(slots + i).to(tl.int64)
    i64 = i.to(tl.int64)

    # PLE conv row: column `a + s` of [pre-step state | conv input of rows 0..P-1].
    c = cb * BLOCK_C + tl.arange(0, BLOCK_C)
    s = tl.arange(0, STATE_BLOCK)
    j = a + s
    m = (c < channels)[:, None] & (s < STATE)[None, :]
    from_prev = (j < STATE)[None, :]
    prev = tl.load(
        conv_prev + (i64 * channels + c[:, None]) * STATE + j[None, :], mask=m & from_prev
    )
    xin = tl.load(
        conv_x + i64 * x_s0 + (j - STATE).to(tl.int64)[None, :] * x_s1 + c[:, None] * x_s2,
        mask=m & ~from_prev,
    )
    tl.store(
        conv_pool + slot * conv_s0 + c[:, None] * conv_s1 + s[None, :] * conv_s2,
        tl.where(from_prev, prev, xin),
        mask=m,
    )

    # The rows of ordinal >= a get back the plane values gathered before the trunk wrote
    # them: index-K plane `cb` here, the position plane in block 0.
    o = tl.arange(0, P_BLOCK)
    r = i64 * P + o
    rejected = (o < P) & (o >= a)
    if cb < num_planes:
        page = tl.load(pages + r, mask=rejected, other=0)
        tok = tl.load(within + r, mask=rejected, other=0)
        e = tl.arange(0, ROW)
        base = tl.load(plane_ptrs + cb).to(tl.pointer_type(tl.bfloat16))
        stride = tl.load(plane_strides + cb)
        vals = tl.load(
            index_old + (cb * rows + r[:, None]) * ROW + e[None, :], mask=rejected[:, None]
        )
        tl.store(
            base + page[:, None] * stride + tok[:, None] * ROW + e[None, :],
            vals,
            mask=rejected[:, None],
        )

    if cb == 0:
        # n-gram row: the id window after row `a - 1` (a negative row wraps, as torch
        # indexing does).
        t = a - 1
        t = tl.where(t < 0, t + P, t)
        k = tl.arange(0, CTX_BLOCK)
        hist = tl.load(stream + i64 * (CTX + P) + t + 1 + k, mask=k < CTX)
        tl.store(ngram_pool + slot * ngram_stride + k, hist, mask=k < CTX)

        page = tl.load(pages + r, mask=rejected, other=0)
        tok = tl.load(within + r, mask=rejected, other=0)
        ax = tl.arange(0, 4)
        pm = rejected[:, None] & (ax < 3)[None, :]
        old_pos = tl.load(position_old + r[:, None] * 3 + ax[None, :], mask=pm)
        tl.store(
            position + page[:, None] * pos_s0 + tok[:, None] * pos_s1 + ax[None, :] * pos_s2,
            old_pos,
            mask=pm,
        )


def _pow2(n: int) -> int:
    return triton.next_power_of_2(max(int(n), 1))


class SpecRollbackStage(OpWrapper):
    """`(torch.cat([ngram_pool[slots], ids.reshape(n, p).to(int64)], 1),
    conv_pool[slots].contiguous())`: per generation request, its n-gram id history followed
    by its `p` rows' ids, and its PLE conv state, read before the trunk writes either pool.

    Fusion boundary: the two reads of the staging; the slots and ids are the caller's.
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS: tuple[Cell, ...] = (
        Cell(why="MTP3, one request", spec=dict(n=1, p=4)),
        Cell(why="MTP3, 128 requests", spec=dict(n=128, p=4)),
    )
    note = """
    Metadata consumed: none. Call site: `SpeculativeRollback.stage_ple_verify`, before
    `ple_ngram_embed` publishes the id history. Grid (requests, conv channel blocks).
    """

    def __call__(self, ngram_pool, conv_pool, slots, ids, p: int):
        n = slots.shape[0]
        ctx = ngram_pool.shape[1]
        channels, state = conv_pool.shape[1], conv_pool.shape[2]
        stream = torch.empty([n, ctx + p], dtype=ngram_pool.dtype, device=ngram_pool.device)
        conv_prev = torch.empty(
            [n, channels, state], dtype=conv_pool.dtype, device=conv_pool.device
        )
        if n:
            _stage_kernel[(n, triton.cdiv(channels, _BLOCK_C))](
                ngram_pool,
                ngram_pool.stride(0),
                conv_pool,
                conv_pool.stride(0),
                conv_pool.stride(1),
                conv_pool.stride(2),
                slots,
                ids,
                stream,
                conv_prev,
                channels,
                P=p,
                CTX=ctx,
                W_BLOCK=_pow2(ctx + p),
                STATE=state,
                STATE_BLOCK=_pow2(state),
                BLOCK_C=_BLOCK_C,
                num_warps=4,
            )
        return stream, conv_prev

    def reference(self, ngram_pool, conv_pool, slots, ids, p: int):
        n = slots.shape[0]
        slots = slots.to(torch.long)
        rows = torch.reshape(ids[: n * p], [n, p]).to(ngram_pool.dtype)
        return torch.cat([ngram_pool[slots], rows], 1), conv_pool[slots]

    def is_valid(self, ngram_pool, conv_pool, slots, ids, p: int) -> None:
        assert ngram_pool.dim() == 2 and ngram_pool.stride(1) == 1
        assert ngram_pool.dtype == torch.int64
        assert conv_pool.dim() == 3
        assert slots.dim() == 1 and ids.dim() == 1 and ids.is_contiguous()
        assert ids.shape[0] >= slots.shape[0] * p


class SpecRollbackCommit(OpWrapper):
    """For each generation request `i` with `a = accepted[i]` accepted rows (golden token
    included, `1 <= a <= p`) and slot `slots[i]`:

    * `ngram_pool[slot] = stream[i, a : a + context]`;
    * `conv_pool[slot] = cat([conv_prev[i], conv_x[i].T], 1)[:, a : a + state_len]`;
    * for each row `r = i * p + t` with `t >= a`: `position[pages[r], within[r]] =
      position_old[r]` and `plane_l[pages[r], within[r]] = index_old[l, r]` for every
      index-K plane `l` of `plane_table`.

    Fusion boundary: the whole commit of `SpeculativeRollback`; the staged tensors are what
    `SpecRollbackStage`, `qsa_rollback_gather` and the position gather produced.
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS: tuple[Cell, ...] = (
        Cell(why="MTP3, one request, 12 planes", spec=dict(n=1, p=4, planes=12)),
        Cell(why="MTP3, 128 requests, 12 planes", spec=dict(n=128, p=4, planes=12)),
    )
    note = """
    Metadata consumed: none. Call site: `SpeculativeRollback.commit_speculative_states`,
    after the draft loop. Grid (requests, max(conv channel blocks, planes)). Slots or plane
    slots shared by two requests (CUDA-graph padding dummies) are written by both; which one
    lands is unordered, as with the indexed writes this replaces.
    """

    def __call__(
        self,
        accepted,
        slots,
        ngram_pool,
        stream,
        conv_pool,
        conv_prev,
        conv_x,
        position,
        position_old,
        pages,
        within,
        plane_ptrs,
        plane_strides,
        index_old,
    ) -> None:
        n, p, channels = conv_x.shape
        state = conv_pool.shape[2]
        ctx = ngram_pool.shape[1]
        if not n:
            return
        blocks = max(triton.cdiv(channels, _BLOCK_C), index_old.shape[0])
        _commit_kernel[(n, blocks)](
            accepted,
            slots,
            ngram_pool,
            ngram_pool.stride(0),
            stream,
            conv_pool,
            conv_pool.stride(0),
            conv_pool.stride(1),
            conv_pool.stride(2),
            conv_prev,
            conv_x,
            conv_x.stride(0),
            conv_x.stride(1),
            conv_x.stride(2),
            position,
            position.stride(0),
            position.stride(1),
            position.stride(2),
            position_old,
            pages,
            within,
            plane_ptrs,
            plane_strides,
            index_old,
            index_old.shape[0],
            index_old.shape[1],
            channels,
            P=p,
            P_BLOCK=_pow2(p),
            CTX=ctx,
            CTX_BLOCK=_pow2(ctx),
            STATE=state,
            STATE_BLOCK=_pow2(state),
            BLOCK_C=_BLOCK_C,
            ROW=ROW,
            num_warps=4,
        )

    def reference(self, *args) -> None:
        raise NotImplementedError("test_modeling_v2_spec_rollback drives the handler end to end")

    def is_valid(
        self,
        accepted,
        slots,
        ngram_pool,
        stream,
        conv_pool,
        conv_prev,
        conv_x,
        position,
        position_old,
        pages,
        within,
        plane_ptrs,
        plane_strides,
        index_old,
    ) -> None:
        n, p, channels = conv_x.shape
        assert accepted.shape == (n,) and slots.shape == (n,)
        assert ngram_pool.dtype == stream.dtype == torch.int64 and ngram_pool.stride(1) == 1
        assert stream.shape == (n, ngram_pool.shape[1] + p) and stream.is_contiguous()
        assert conv_pool.shape[1] == channels and conv_prev.is_contiguous()
        assert conv_prev.shape == (n, channels, conv_pool.shape[2])
        assert position.dim() == 3 and position.shape[2] == 3
        assert position_old.shape == (n * p, 3) and position_old.is_contiguous()
        assert pages.shape == within.shape == (n * p,)
        assert pages.dtype == within.dtype == torch.int64
        assert index_old.is_contiguous() and index_old.shape[1:] == (n * p, ROW)
        assert plane_ptrs.numel() == plane_strides.numel() == index_old.shape[0]


spec_rollback_stage = SpecRollbackStage()
spec_rollback_commit = SpecRollbackCommit()
