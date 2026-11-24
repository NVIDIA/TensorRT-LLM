# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DFlash2 window decomposition on CPU: a float64 model, and the Triton kernels interpreted.

The float64 model runs the kernels' algorithm in plain torch: the plan (``window_plan_torch``),
each page-aligned body chunk, the head and block keys, and the select-based log-sum-exp merge,
per CP rank and then across ranks. Against the dense sliding window it must be exact to
float64 rounding for CP 1, 2, 4 and 8, blocks of 6 and 8, histories at every window boundary
and ranks that hold nothing of a row.

The Triton kernels then run under ``TRITON_INTERPRET=1``: the plan kernel must equal the torch
plan exactly, and the body and tail kernels the float64 model to FP32 rounding, at small head
and latent widths with real window boundaries; with ``vec_io`` (the hinted row offsets,
``ALIGNED`` at every stride here) the body and tail give the same bits. The interpreter has to
be chosen before triton is first imported, so those checks run this file as a script in a fresh
process that loads only the kernels module (``python test_rubin_dflash2_window.py <kernels file>
<check>``); nothing in the pytest session itself is interpreted.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import os
import subprocess
import sys

import pytest
import torch

pytestmark = pytest.mark.cpu_only

#: The kernels module: the package's under pytest, an interpreted copy in the script process.
kw = None


def _package_kernels():
    global kw
    if kw is None:
        from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.spec import (
            rubin_dflash2_window,
        )

        kw = rubin_dflash2_window
    return kw


@pytest.fixture(autouse=True)
def _kernels_module():
    _package_kernels()


_F64 = torch.float64
#: Histories at every boundary of a 1024-position window with 64-token pages: empty, one key,
#: a page edge, the window edge with the block straddling it, and several pages past it.
_EDGES = (0, 1, 63, 64, 65, *range(1022, 1032), 4097, 9000)


# a paged history per request, split over CP ranks as the draft cache stores it


@dataclasses.dataclass
class _Case:
    lengths: list[int]
    block_size: int
    window: int
    page_size: int
    cp_size: int
    query: torch.Tensor  # [R, H, D]
    pool: torch.Tensor  # [pages, P, D], every rank's pages in one tensor
    block_kv: torch.Tensor  # [R, D]
    global_table: torch.Tensor  # [B, global pages] pool page of each global page
    history: list[torch.Tensor]  # [L_b, D] per request

    def local_table(self, rank: int) -> torch.Tensor:
        """Rank ``rank``'s page table: its local page ``c`` is global page ``c * cp + rank``."""
        cols = self.global_table.shape[1] // self.cp_size
        idx = torch.arange(cols) * self.cp_size + rank
        return self.global_table[:, idx].to(torch.int32).contiguous()


def _make_case(
    lengths,
    block_size,
    *,
    window=1024,
    page_size=64,
    cp_size=1,
    heads=3,
    latent=32,
    rope=16,
    dtype=_F64,
    seed=0,
) -> _Case:
    g = torch.Generator().manual_seed(seed)
    batch = len(lengths)
    dim = latent + rope
    span = max(lengths) + block_size + page_size * cp_size
    gpages = -(-span // (page_size * cp_size)) * cp_size
    num_pages = batch * gpages + 7
    table = torch.randperm(num_pages, generator=g)[: batch * gpages].view(batch, gpages)
    pool = torch.randn(num_pages, page_size, dim, generator=g, dtype=_F64)
    history = []
    for b, length in enumerate(lengths):
        pos = torch.arange(length)
        history.append(pool[table[b, pos // page_size], pos % page_size])
    rows = batch * block_size
    query = torch.randn(rows, heads, dim, generator=g, dtype=_F64)
    block_kv = torch.randn(rows, dim, generator=g, dtype=_F64)
    return _Case(
        lengths=list(lengths),
        block_size=block_size,
        window=window,
        page_size=page_size,
        cp_size=cp_size,
        query=query.to(dtype),
        pool=pool.to(dtype),
        block_kv=block_kv.to(dtype),
        global_table=table,
        history=[h.to(dtype) for h in history],
    )


def _plan(case: _Case, rank: int, num_splits: int) -> kw.DFlash2WindowPlan:
    num_ctx = torch.tensor(case.lengths, dtype=torch.int64)
    return kw.window_plan_torch(
        num_ctx,
        case.local_table(rank),
        None,
        case.block_size,
        case.window,
        case.page_size,
        case.cp_size,
        rank,
        num_splits,
    )


# float64 model of the kernels: body chunks, tail, select-based LSE merge


def _part(q, keys, visible, scale, latent):
    """Softmax attention of ``q`` ``[blk, H, D]`` over ``keys`` ``[n, D]`` where ``visible``
    ``[blk, n]``: normalized output, natural-log max and exp sum; ``(0, -inf, 0)`` if empty."""
    scores = torch.einsum("jhd,nd->jhn", q.to(_F64), keys.to(_F64)) * scale
    scores = scores.masked_fill(~visible[:, None, :], float("-inf"))
    peak = (
        scores.amax(-1)
        if keys.shape[0]
        else torch.full(scores.shape[:2], float("-inf"), dtype=_F64)
    )
    has = torch.isfinite(peak)
    probs = torch.where(
        visible[:, None, :], torch.exp(scores - torch.where(has, peak, 0)[..., None]), 0
    )
    total = probs.sum(-1)
    out = torch.einsum("jhn,nc->jhc", probs, keys.to(_F64)[:, :latent])
    out = torch.where(has[..., None], out / torch.where(has, total, 1)[..., None], 0)
    return out, torch.where(has, peak, float("-inf")), torch.where(has, total, 0)


def _merge(parts):
    """Select-based LSE merge of ``(out, max, sum)`` parts: no weight multiplies an empty part."""
    peak = torch.stack([torch.where(s > 0, m, float("-inf")) for _, m, s in parts]).amax(0)
    safe = torch.where(torch.isfinite(peak), peak, 0)
    acc = torch.zeros_like(parts[0][0])
    total = torch.zeros_like(parts[0][2])
    for out, m, s in parts:
        live = s > 0
        w = torch.where(live, s * torch.exp(torch.where(live, m, 0) - safe), 0)
        acc = acc + torch.where(live[..., None], w[..., None] * out, 0)
        total = total + w
    has = total > 0
    out = torch.where(has[..., None], acc / torch.where(has, total, 1)[..., None], 0)
    return out, torch.where(has, peak, float("-inf")), torch.where(has, total, 0)


def _model_rank(case: _Case, plan: kw.DFlash2WindowPlan, scale: float, latent: int):
    """One rank's window partials as the kernels compute them: ``[R, H, latent]`` and stats."""
    blk, page = case.block_size, case.page_size
    pool_rows = case.pool.reshape(-1, case.pool.shape[-1])
    outs, stats = [], []
    for b in range(len(case.lengths)):
        q = case.query[b * blk : (b + 1) * blk]
        t = torch.arange(int(plan.body_len[b]))
        body_rows = plan.body_pages[b].long()[t // page] * page + t % page
        parts = []
        bounds = plan.chunk_bounds[b].tolist()
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            keys = pool_rows[body_rows[lo:hi]]
            parts.append(_part(q, keys, torch.ones(blk, hi - lo, dtype=torch.bool), scale, latent))
        head = plan.head_rows[b].long()
        hw = head.shape[0]
        head_vis = (head >= 0)[None, :] & (
            torch.arange(hw)[None, :] >= plan.head_first[b].long()[:, None]
        )
        k = torch.arange(blk)
        bits = int(plan.block_bits[b])
        owned = torch.tensor([(bits >> i) & 1 == 1 for i in range(blk)])
        block_vis = owned[None, :] & (k[None, :] <= k[:, None])
        keys = torch.cat((pool_rows[head.clamp_min(0)], case.block_kv[b * blk : (b + 1) * blk]))
        parts.append(_part(q, keys, torch.cat((head_vis, block_vis), dim=1), scale, latent))
        out, m, s = _merge(parts)
        outs.append(out)
        stats.append(torch.stack((m, s), dim=-1))
    return torch.cat(outs), torch.cat(stats)


def _dense(case: _Case, scale: float, latent: int) -> torch.Tensor:
    """The dense causal sliding window, position by position, in float64."""
    blk, window = case.block_size, case.window
    outs = []
    for b, length in enumerate(case.lengths):
        keys = torch.cat((case.history[b], case.block_kv[b * blk : (b + 1) * blk])).to(_F64)
        for j in range(blk):
            p = length + j
            k = keys[max(0, p - window + 1) : p + 1]
            scores = case.query[b * blk + j].to(_F64) @ k.T * scale
            outs.append(torch.softmax(scores, -1) @ k[:, :latent])
    return torch.stack(outs)


def _merged_over_ranks(case: _Case, num_splits: int, scale: float, latent: int):
    parts = []
    for rank in range(case.cp_size):
        out, stats = _model_rank(case, _plan(case, rank, num_splits), scale, latent)
        assert torch.isfinite(out).all(), f"rank {rank}: a NaN or inf partial"
        empty = stats[..., 1] == 0
        assert (out[empty] == 0).all() and torch.isneginf(stats[..., 0][empty]).all(), (
            f"rank {rank}: an empty row is not a zero partial with (-inf, 0)"
        )
        parts.append((out, stats[..., 0], stats[..., 1]))
    return _merge(parts)[0]


_SCALE = 0.13


@pytest.mark.parametrize("block_size", [6, 8])
@pytest.mark.parametrize("cp_size", [1, 2, 4, 8])
@pytest.mark.parametrize("num_splits", [1, 3, 16])
def test_model_equals_dense_window(cp_size, block_size, num_splits):
    case = _make_case(_EDGES, block_size, cp_size=cp_size, seed=cp_size * 10 + block_size)
    merged = _merged_over_ranks(case, num_splits, _SCALE, 32)
    dense = _dense(case, _SCALE, 32)
    torch.testing.assert_close(merged, dense, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("cp_size", [2, 4, 8])
def test_model_with_empty_shards(cp_size):
    """Histories that leave whole ranks without a key of some rows, and rows without a body."""
    lengths = [0, 1, 2, 5, 60, 64, 66, 127, 128, 130, 191, 1000, 1030]
    case = _make_case(lengths, 6, cp_size=cp_size, seed=7)
    merged = _merged_over_ranks(case, 2, _SCALE, 32)
    torch.testing.assert_close(merged, _dense(case, _SCALE, 32), rtol=1e-12, atol=1e-12)


def test_dense_reference_is_the_drafters_window():
    """The float64 dense window here is the drafter's own: ``sparse_mla_rows_eager`` over the
    index lists ``dflash2_window_rows`` builds, with the block stored at ``L + j`` as the
    general target's non-helix path stores it (FP32, so to FP32 rounding)."""
    blk, page = 8, 64
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.dflash2_window import (  # noqa: E501
        dflash2_window_rows,
        sparse_mla_rows_eager,
    )

    case = _make_case(_EDGES, blk, seed=3)
    pool = case.pool.clone()
    for b, length in enumerate(case.lengths):
        pos = torch.arange(length, length + blk)
        pool[case.global_table[b, pos // page], pos % page] = case.block_kv[b * blk : (b + 1) * blk]
    ctx_len = torch.tensor(case.lengths)
    rows, _ = dflash2_window_rows(ctx_len, case.global_table, blk, page, 1024, 1024)
    eager = sparse_mla_rows_eager(
        case.query.float(), pool.float().reshape(-1, pool.shape[-1]), rows, _SCALE, 32
    )
    torch.testing.assert_close(eager.double(), _dense(case, _SCALE, 32), rtol=2e-5, atol=2e-5)


def test_geometry_bounds_hold_at_every_length():
    """``body_page_count`` and ``head_width`` bound the plan at every history length."""
    for block_size in (1, 5, 6, 8):
        for cp_size in (1, 2, 4, 8):
            nwl = kw.body_page_count(1024, 64, block_size, cp_size)
            hw = kw.head_width(64, block_size)
            for length in range(0, 3 * 1024, 7):
                lo_max = max(0, length + block_size - 1024)
                start = min(length, -(-lo_max // 64) * 64)
                lo0 = max(0, length - 1023)
                assert start - lo0 <= hw
                pages = -(-(length - start) // 64) if start < length else 0
                assert -(-pages // cp_size) <= nwl


def test_split_count_is_static_and_bounded():
    assert kw.split_count(148, 1, 12, 8, 1024, 64, 1) == 16  # TEP8: one request fills the GPU
    assert kw.split_count(148, 12, 96, 8, 1024, 64, 1) == 2  # DEP16: 144 tiles, about one wave
    assert kw.split_count(148, 16, 96, 6, 1024, 64, 8) == 2  # helix8: a rank holds <= 2 body pages
    for batch in (1, 2, 4, 8, 12, 16):
        assert 1 <= kw.split_count(160, batch, 96, 8, 1024, 64, 1) <= 16


# the Triton kernels under the interpreter


def _plan_kernel(module, case, rank, num_splits, table_rows=None, table=None):
    batch = len(case.lengths)
    buffers = module.plan_buffers(
        batch + 2, case.block_size, case.window, case.page_size, case.cp_size, 16, "cpu"
    )
    for t in buffers:
        t.fill_(-7)  # stale values a smaller earlier step could leave
    num_ctx = torch.tensor(case.lengths, dtype=torch.int64)
    return module.launch_window_plan(
        num_ctx,
        case.local_table(rank) if table is None else table,
        table_rows,
        case.block_size,
        case.window,
        case.page_size,
        case.cp_size,
        rank,
        num_splits,
        buffers,
    )


def _check_plan(module):
    """The plan kernel equals the torch plan, field by field, on every rank."""
    for block_size in (6, 8):
        for cp_size in (1, 2, 4, 8):
            case = _make_case(_EDGES, block_size, cp_size=cp_size, seed=13 + cp_size)
            for rank in range(cp_size):
                for num_splits in (1, 3, 16):
                    got = _plan_kernel(module, case, rank, num_splits)
                    want = _plan(case, rank, num_splits)
                    for name, a, b in zip(want._fields, got, want):
                        assert torch.equal(a, b), (
                            f"blk {block_size} cp {cp_size} rank {rank} splits {num_splits}: "
                            f"{name} differs"
                        )


def _check_table_rows(interpreted):
    """With ``table_rows`` each request reads its own row of a larger shared table."""
    case = _make_case([5, 1500, 1031, 9000], 8, cp_size=2, seed=17)
    local = case.local_table(1)
    shared = torch.full((9, local.shape[1]), 10**6, dtype=torch.int32)
    rows = torch.tensor([7, 2, 5, 0])
    shared[rows] = local
    got = _plan_kernel(interpreted, case, 1, 4, table_rows=rows, table=shared)
    want = _plan(case, 1, 4)
    for name, a, b in zip(want._fields, got, want):
        assert torch.equal(a, b), f"{name} differs"


def _small_case(lengths, block_size, cp_size, seed):
    # Real window arithmetic at small widths: 16-token pages, a 64-position window.
    return _make_case(
        lengths,
        block_size,
        window=64,
        page_size=16,
        cp_size=cp_size,
        heads=3,
        latent=32,
        rope=16,
        dtype=torch.float32,
        seed=seed,
    )


_SMALL_EDGES = (0, 1, 5, 15, 16, 17, 55, 57, 58, 63, 64, 65, 71, 100, 300)


_KERNEL_CASES = (
    # block_size, cp_size, num_splits, block_m, block_dv
    (8, 1, 1, 16, 32),
    (8, 1, 3, 16, 16),
    (6, 2, 2, 16, 16),
    (6, 4, 1, 64, 32),
    (8, 4, 2, 16, 16),
)


def _bits(x: torch.Tensor) -> torch.Tensor:
    return x.view({2: torch.int16, 4: torch.int32, 8: torch.int64}[x.element_size()])


def _check_vec_io(interpreted, args, scale, latent, out, stats, num_splits, **tiles):
    """The same launch with ``vec_io``: these strides take the hinted (``ALIGNED``) path in both
    kernels, and the output and stats keep every bit."""
    query, pool, block_kv, plan = args
    batch, block_size = plan.head_first.shape
    workspace = interpreted.window_workspace(
        batch, block_size, query.shape[1], latent, num_splits, "cpu"
    )
    workspace.partial_out.fill_(float("nan"))
    workspace.partial_stats.fill_(float("nan"))
    widths = {k: v for k, v in tiles.items() if k in ("block_dv", "block_kc")}
    assert interpreted.window_vec_io_aligned(query, pool, block_kv, workspace, latent, **widths)
    got_stats = torch.full_like(stats, float("nan"))
    got = interpreted.launch_window_triton(
        *args, scale, latent, 1.0, workspace, got_stats, vec_io=True, **tiles
    )
    assert torch.equal(_bits(got), _bits(out)), "vec_io changed the output"
    assert torch.equal(_bits(got_stats), _bits(stats)), "vec_io changed the stats"


def _check_body_tail(interpreted):
    for case_args in _KERNEL_CASES:
        _check_body_tail_case(interpreted, *case_args)


def _check_body_tail_case(interpreted, block_size, cp_size, num_splits, block_m, block_dv):
    case = _small_case(_SMALL_EDGES, block_size, cp_size, seed=block_size + cp_size)
    batch, heads = len(case.lengths), case.query.shape[1]
    rows = batch * block_size
    parts = []
    for rank in range(cp_size):
        plan = _plan_kernel(interpreted, case, rank, num_splits)
        workspace = interpreted.window_workspace(batch, block_size, heads, 32, num_splits, "cpu")
        workspace.partial_out.fill_(float("nan"))  # stale workspace must never be read
        workspace.partial_stats.fill_(float("nan"))
        stats = torch.full((rows, heads, 2), float("nan"))
        out = interpreted.launch_window_triton(
            case.query,
            case.pool,
            case.block_kv,
            plan,
            _SCALE,
            32,
            1.0,
            workspace,
            stats,
            block_m=block_m,
            block_dv=block_dv,
        )
        _check_vec_io(
            interpreted,
            (case.query, case.pool, case.block_kv, plan),
            _SCALE,
            32,
            out,
            stats,
            num_splits,
            block_m=block_m,
            block_dv=block_dv,
        )
        model_out, model_stats = _model_rank(case, _plan(case, rank, num_splits), _SCALE, 32)
        assert torch.isfinite(out).all() and not torch.isnan(stats).any()
        torch.testing.assert_close(out.double(), model_out, rtol=2e-5, atol=2e-6)
        empty = model_stats[..., 1] == 0
        assert torch.isneginf(stats[..., 0][empty]).all() and (stats[..., 1][empty] == 0).all()
        assert (out[empty] == 0).all()
        torch.testing.assert_close(
            stats[~empty].double(), model_stats[~empty], rtol=2e-5, atol=2e-5
        )
        ref_stats = torch.empty_like(stats)
        ref = interpreted.window_reference(
            case.query, case.pool, case.block_kv, plan, _SCALE, 32, 1.0, ref_stats
        )
        torch.testing.assert_close(out, ref, rtol=2e-5, atol=2e-6)
        parts.append((out.double(), stats[..., 0].double(), stats[..., 1].double()))
    merged = _merge(parts)[0]
    torch.testing.assert_close(merged, _dense(case, _SCALE, 32), rtol=5e-5, atol=5e-6)


def _check_empty_bodies(interpreted):
    """Body lengths 0 .. blk - 1 (histories below the block; dummy rows of a padded graph have
    0), cut into more chunks than they have pages: every empty chunk and every empty body is
    selected away, never multiplied in, so no row turns NaN."""
    block_size = 8
    for lengths in (list(range(block_size)), [0] * block_size):
        case = _small_case(lengths, block_size, 1, seed=23)
        plan = _plan_kernel(interpreted, case, 0, 4)
        assert plan.body_len.tolist() == lengths
        workspace = interpreted.window_workspace(block_size, block_size, 3, 32, 4, "cpu")
        workspace.partial_out.fill_(float("nan"))
        workspace.partial_stats.fill_(float("nan"))
        stats = torch.full((block_size * block_size, 3, 2), float("nan"))
        out = interpreted.launch_window_triton(
            case.query,
            case.pool,
            case.block_kv,
            plan,
            _SCALE,
            32,
            1.0,
            workspace,
            stats,
            block_m=16,
        )
        args = (case.query, case.pool, case.block_kv, plan)
        _check_vec_io(interpreted, args, _SCALE, 32, out, stats, 4, block_m=16)
        assert torch.isfinite(out).all() and torch.isfinite(stats).all()
        torch.testing.assert_close(out.double(), _dense(case, _SCALE, 32), rtol=2e-5, atol=2e-6)


def _check_production_dims(interpreted):
    """The default tiles at the drafter's own widths (96 or 12 heads, latent 512, rope 64,
    64-token pages, a 1024-position window) over FP8 and BF16 pools, against the reference.

    The query is FP16 here: Triton 3.9's interpreter multiplies BF16 tiles wrongly (its own
    ``tl.dot`` on BF16 operands gives garbage, FP16 and FP32 are exact) and has no FP8 dot.
    The kernels never take an FP8 dot (keys widen to the query's dtype first), so FP16 drives
    the same code as BF16 does on the GPU.
    """
    cases = (
        # pool, heads, block, cp, rank, lengths, splits
        ("fp8", 12, 8, 1, 0, (1031, 9000), 3),
        ("fp8", 96, 6, 8, 1, (60, 1030), 1),
        ("bf16", 12, 8, 1, 0, (190000,), 16),
    )
    page, window, latent, rope = 64, 1024, 512, 64
    for pool_dtype, heads, blk, cp, rank, lengths, splits in cases:
        g = torch.Generator().manual_seed(heads + blk)
        batch, dim = len(lengths), latent + rope
        span = page * cp
        gcols = -(-(max(lengths) + blk + span) // span) * cp
        table = torch.randperm(batch * gcols, generator=g).view(batch, gcols)
        pool = 0.5 * torch.randn(batch * gcols, page, dim, generator=g)
        block_kv = 0.5 * torch.randn(batch * blk, dim, generator=g)
        wide = torch.float8_e4m3fn if pool_dtype == "fp8" else torch.bfloat16
        pool, block_kv = pool.to(wide), block_kv.to(wide)
        query = (0.5 * torch.randn(batch * blk, heads, dim, generator=g)).half()
        buffers = interpreted.plan_buffers(batch, blk, window, page, cp, splits, "cpu")
        plan = interpreted.launch_window_plan(
            torch.tensor(lengths),
            table[:, rank::cp].to(torch.int32).contiguous(),
            None,
            blk,
            window,
            page,
            cp,
            rank,
            splits,
            buffers,
        )
        workspace = interpreted.window_workspace(batch, blk, heads, latent, splits, "cpu")
        stats = torch.full((batch * blk, heads, 2), float("nan"))
        out = interpreted.launch_window_triton(
            query, pool, block_kv, plan, 0.1308608, latent, 1.0, workspace, stats
        )
        _check_vec_io(
            interpreted, (query, pool, block_kv, plan), 0.1308608, latent, out, stats, splits
        )
        ref_stats = torch.empty_like(stats)
        ref = interpreted.window_reference(
            query, pool, block_kv, plan, 0.1308608, latent, 1.0, ref_stats
        )
        torch.testing.assert_close(out.float(), ref.float(), rtol=4e-3, atol=4e-3)
        empty = ref_stats[..., 1] == 0
        assert torch.isneginf(stats[..., 0][empty]).all() and (stats[..., 1][empty] == 0).all()
        torch.testing.assert_close(stats[~empty], ref_stats[~empty], rtol=1e-4, atol=1e-4)


_INTERPRETED_CHECKS = {
    "plan": _check_plan,
    "table_rows": _check_table_rows,
    "body_tail": _check_body_tail,
    "empty_bodies": _check_empty_bodies,
    "production_dims": _check_production_dims,
}


@pytest.mark.parametrize("check", sorted(_INTERPRETED_CHECKS))
def test_triton_kernels_interpreted(check):
    """The Triton kernels in Triton's interpreter, in a fresh process (see the module doc)."""
    env = dict(os.environ, TRITON_INTERPRET="1", CUDA_VISIBLE_DEVICES="")
    proc = subprocess.run(
        [sys.executable, __file__, _package_kernels().__file__, check],
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    assert proc.returncode == 0, f"{check}:\n{proc.stdout[-6000:]}\n{proc.stderr[-6000:]}"
    assert f"ok {check}" in proc.stdout


if __name__ == "__main__":
    # The interpreted run: TRITON_INTERPRET is set before triton is first imported here.
    assert os.environ.get("TRITON_INTERPRET") == "1", "run with TRITON_INTERPRET=1"
    _spec = importlib.util.spec_from_file_location("_dflash2_window_interpreted", sys.argv[1])
    kw = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(kw)
    _INTERPRETED_CHECKS[sys.argv[2]](kw)
    print(f"ok {sys.argv[2]}")
