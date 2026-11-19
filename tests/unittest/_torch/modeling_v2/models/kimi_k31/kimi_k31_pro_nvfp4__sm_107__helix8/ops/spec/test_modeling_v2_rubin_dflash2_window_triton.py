# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DFlash2 window attention entry in Triton (``rubin_dflash2_window_triton``), on a sample of
its cells.

Every cell builds a paged draft history in which only the pages of each request's window hold
data; every other column of the page table points at a page of NaN. A read outside the window
turns the output NaN.

CPU (any host): the entry's ``reference`` over each cell's plan, merged over the CP ranks for
helix cells, is the drafter's dense sliding window, position by position, in float64.

GPU (SM107): every sampled cell against ``reference`` within the band derived below, and the helix
stats.
"""

from __future__ import annotations

import dataclasses
import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import true_fp32_matmul
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kimi_k31_pro_nvfp4__sm_107__helix8.ops.spec.rubin_dflash2_window_plan import (  # noqa: E501
    plan_buffers,
    rubin_dflash2_window_plan,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kimi_k31_pro_nvfp4__sm_107__helix8.ops.spec.rubin_dflash2_window_triton import (  # noqa: E501
    rubin_dflash2_window_triton as op,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kimi_k31_pro_nvfp4__sm_107__helix8.ops.spec.rubin_dflash2_window_triton import (  # noqa: E501
    window_workspace,
)

__extra_import_path__ = ["../../../../..", "../../.."]
from _k31_utils import on_sm, sample_cells
from _validating import assert_within_ulp, certifying  # noqa: E402,F401

gpu = on_sm(107, reason="the window entries run on SM107 (Rubin)")

#: A sample of the entry's cells: the smallest and the median batch.
CELLS = sample_cells(op.CELLS, size_key="batch")[:2]

#: Band in units of 2**-8 of each row's largest output (``assert_within_ulp``'s scale). Both
#: sides round the output to BF16 once (1 unit apart at most from that); the kernel also feeds
#: the probabilities to P.V as BF16 (2**-9 relative each) and merges the chunks in another
#: order. The same rounding steps measured 2 units per element and 0.55 .. 0.82 in relative RMS
#: for the TRTLLM-Gen window decode (ops/spec/dflash2_window_attention's test). The query stays
#: BF16 here even over an FP8 pool (keys widen exactly), so one band serves both pools.
_BAND = dict(element_ulp=4.0, rms_ulp=1.0)


@dataclasses.dataclass
class Inputs:
    spec: dict
    lengths: list
    table: torch.Tensor  # [B, global pages] int64; pool page 0 is the NaN page
    pool: torch.Tensor  # [pages, P, D], one layer of a [pages, layers, P, D] pool
    query: torch.Tensor  # [R, H, D] BF16
    block_kv: torch.Tensor  # [R, D] in the pool's dtype

    def local_table(self, rank: int) -> torch.Tensor:
        """Rank ``rank``'s table: its local page ``c`` is global page ``c * cp + rank``."""
        cp = self.spec["cp_size"]
        return self.table[:, rank::cp].to(torch.int32).contiguous()

    def num_ctx(self) -> torch.Tensor:
        return torch.tensor(self.lengths, dtype=torch.int64, device=self.pool.device)


def build(spec: dict, seed: int, device, lengths=None) -> Inputs:
    """A paged history whose window pages are real and whose other pages are NaN."""
    g = torch.Generator().manual_seed(seed)
    lengths = list(spec["lengths"] if lengths is None else lengths)
    batch, blk, page = len(lengths), spec["block_size"], spec["page_size"]
    window, cp = spec["window"], spec["cp_size"]
    dim = spec["latent"] + spec["rope"]
    span = page * cp
    gcols = -(-(max(lengths) + blk + span) // span) * cp
    table = torch.zeros(batch, gcols, dtype=torch.int64)
    count = 1
    for b, length in enumerate(lengths):
        first, last = max(0, length - window - blk) // page, (length + blk - 1) // page
        for gp in range(first, last + 1):
            table[b, gp] = count
            count += 1
    perm = torch.randperm(count - 1, generator=g) + 1
    table = torch.where(table > 0, perm[(table - 1).clamp_min(0)], table)
    base = 0.5 * torch.randn(count, spec["layers"], page, dim, generator=g)
    base[0] = float("nan")
    rows = batch * blk
    query = (0.5 * torch.randn(rows, spec["heads"], dim, generator=g)).bfloat16()
    block_kv = 0.5 * torch.randn(rows, dim, generator=g)
    if spec["pool"] == "fp8":
        base = base.clamp(-448, 448).to(torch.float8_e4m3fn)
        block_kv = block_kv.to(torch.float8_e4m3fn)
    else:
        base, block_kv = base.bfloat16(), block_kv.bfloat16()
    base = base.to(device)
    return Inputs(
        spec=spec,
        lengths=lengths,
        table=table.to(device),
        pool=base[:, spec["layers"] - 1],
        query=query.to(device),
        block_kv=block_kv.to(device),
    )


def _rows_f64(pool: torch.Tensor, pages: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    if pool.dtype == torch.float8_e4m3fn:
        return pool.view(torch.uint8)[pages, offsets].view(torch.float8_e4m3fn).double()
    return pool[pages, offsets].double()


def dense_window(inp: Inputs) -> torch.Tensor:
    """Every query row over its whole window, position by position, in float64."""
    spec = inp.spec
    blk, page, window = spec["block_size"], spec["page_size"], spec["window"]
    scale, latent = spec["scale"], spec["latent"]
    outs = []
    for b, length in enumerate(inp.lengths):
        lo = max(0, length - window + 1)
        pos = torch.arange(lo, length, device=inp.pool.device)
        hist = _rows_f64(inp.pool, inp.table[b, pos // page], pos % page)
        keys = torch.cat((hist, inp.block_kv[b * blk : (b + 1) * blk].double()))
        for j in range(blk):
            p = length + j
            k = keys[max(0, p - window + 1) - lo : p + 1 - lo]
            scores = inp.query[b * blk + j].double() @ k.T * scale
            outs.append(torch.softmax(scores, -1) @ k[:, :latent])
    return torch.stack(outs)


def merge_ranks(parts):
    """Select-based LSE merge of per-rank ``(out, stats)``: what the helix combine computes."""
    maxima = torch.stack([s[..., 0] for _, s in parts]).double()
    sums = torch.stack([s[..., 1] for _, s in parts]).double()
    outs = torch.stack([o for o, _ in parts]).double()
    live = sums > 0
    peak = torch.where(live, maxima, float("-inf")).amax(0)
    peak = torch.where(torch.isfinite(peak), peak, 0)
    w = torch.where(live, sums * torch.exp(torch.where(live, maxima, 0) - peak), 0)
    acc = torch.where(live[..., None], w[..., None] * outs, 0).sum(0)
    return acc / w.sum(0)[..., None]


def make_plan(inp: Inputs, rank: int, reference: bool = False):
    spec = inp.spec
    blk, cp, splits = spec["block_size"], spec["cp_size"], spec["num_splits"]
    args = (
        inp.num_ctx(),
        inp.local_table(rank),
        None,
        blk,
        spec["window"],
        spec["page_size"],
        cp,
        rank,
        splits,
    )
    if reference:
        return rubin_dflash2_window_plan.reference(*args, None)
    buffers = plan_buffers(
        len(inp.lengths), blk, spec["window"], spec["page_size"], cp, splits, inp.pool.device
    )
    return rubin_dflash2_window_plan(*args, buffers)


def _seed(cell) -> int:
    return zlib.crc32(cell.why.encode()) % 2**31


def _ids(cells):
    return [c.why[:56] for c in cells]


# CPU: the reference over the plan is the dense window


@pytest.mark.cpu_only
@pytest.mark.parametrize("cell", CELLS, ids=_ids(CELLS))
def test_reference_is_the_dense_window(cell):
    spec = cell.spec
    inp = build(spec, _seed(cell), "cpu")
    rows, heads = inp.query.shape[:2]
    parts = []
    for rank in range(spec["cp_size"]):
        plan = make_plan(inp, rank, reference=True)
        stats = torch.empty(rows, heads, 2, dtype=torch.float32)
        out = op.reference(
            inp.query.float(),
            inp.pool,
            inp.block_kv,
            plan,
            spec["scale"],
            spec["latent"],
            1.0,
            None,
            stats,
        )
        assert torch.isfinite(out).all(), f"rank {rank} read a page outside the window"
        empty = stats[..., 1] == 0
        assert (out[empty] == 0).all() and torch.isneginf(stats[..., 0][empty]).all()
        parts.append((out, stats))
    merged = merge_ranks(parts) if len(parts) > 1 else parts[0][0].double()
    torch.testing.assert_close(merged, dense_window(inp), rtol=2e-4, atol=2e-5)


# GPU


def run_entry(inp: Inputs, rank: int):
    spec = inp.spec
    rows, heads = inp.query.shape[:2]
    plan = make_plan(inp, rank)
    workspace = window_workspace(
        len(inp.lengths),
        spec["block_size"],
        heads,
        spec["latent"],
        spec["num_splits"],
        inp.pool.device,
    )
    stats = torch.full((rows, heads, 2), float("nan"), device=inp.pool.device)
    out = op(
        inp.query,
        inp.pool,
        inp.block_kv,
        plan,
        spec["scale"],
        spec["latent"],
        1.0,
        workspace,
        stats if spec["stats"] else None,
    )
    return out, stats, plan, workspace


def check_against_reference(inp: Inputs, out, stats, plan, workspace):
    spec = inp.spec
    ref_stats = torch.empty_like(stats)
    with true_fp32_matmul():
        ref = op.reference(
            inp.query,
            inp.pool,
            inp.block_kv,
            plan,
            spec["scale"],
            spec["latent"],
            1.0,
            workspace,
            ref_stats,
        )
    assert torch.isfinite(out).all(), "NaN: a read outside the window, or an unselected empty part"
    assert_within_ulp(out, ref, **_BAND)
    if spec["stats"]:
        empty = ref_stats[..., 1] == 0
        assert torch.isneginf(stats[..., 0][empty]).all() and (stats[..., 1][empty] == 0).all()
        assert (out[empty] == 0).all()
        torch.testing.assert_close(stats[~empty], ref_stats[~empty], rtol=1e-4, atol=1e-4)


@gpu
@pytest.mark.parametrize("cell", CELLS, ids=_ids(CELLS))
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    inp = build(cell.spec, _seed(cell), "cuda")
    rank = cell.spec["cp_rank"]
    check_against_reference(inp, *run_entry(inp, rank))
