# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DFlash2 window attention entry in Triton (``rubin_dflash2_window_triton``).

Every cell builds a paged draft history in which only the pages of each request's window hold
data; every other column of the page table points at a page of NaN. A read outside the window
turns the output NaN.

CPU (any host): the entry's ``reference`` over each cell's plan, merged over the CP ranks for
helix cells, is the drafter's dense sliding window, position by position, in float64.

GPU (SM107): every cell against ``reference`` within the band derived below; the helix stats;
one CUDA graph captured on one input set and replayed on another; body lengths 0 .. blk - 1 and
rank rows without a key give no NaN; no Triton compile after the warmup pass, with the launch
levers off or on.

CPU (any host): the launch levers are keyword only and off by default, the entry hands them to the
launcher unchanged, and ``vec_io`` takes the vector path (``ALIGNED``) exactly where every stride
and base allows it: at helix8's strides, not on a pool whose page stride or base is 8 bytes off.
Under ``pdl`` one variable sets the body's trigger, the tail's wait and the tail's PDL launch, the
body stays a normal launch, and the public tail launcher never launches with PDL.
(``test_rubin_dflash2_window_bits.py`` checks on the GPU that the levers change no output byte.)
"""

from __future__ import annotations

import dataclasses
import inspect
import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import true_fp32_matmul
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.spec import (
    rubin_dflash2_window as kw,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec import (
    rubin_dflash2_window_triton as entry_module,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.rubin_dflash2_window_plan import (  # noqa: E501
    plan_buffers,
    rubin_dflash2_window_plan,
    split_count,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.rubin_dflash2_window_triton import (  # noqa: E501
    WindowTiles,
    window_workspace,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.rubin_dflash2_window_triton import (  # noqa: E501
    rubin_dflash2_window_triton as op,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../.."]
from _validating import assert_within_ulp, certifying  # noqa: E402,F401

_ON_SM107 = torch.cuda.is_available() and get_sm_version() in (103, 107)
gpu = pytest.mark.skipif(
    not _ON_SM107, reason="the window entries run on SM103 (GB300) and SM107 (Rubin)"
)

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
@pytest.mark.parametrize("cell", op.CELLS, ids=_ids(op.CELLS))
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


# CPU: the launch levers

#: The entry's launch levers and the value each takes when on.
_LEVERS = {"vec_io": True, "pdl": True}


@pytest.mark.cpu_only
def test_the_levers_are_keyword_only_and_off_by_default():
    for method in (type(op).__call__, type(op).reference, type(op).is_valid):
        params = inspect.signature(method).parameters
        for name in _LEVERS:
            assert params[name].kind is inspect.Parameter.KEYWORD_ONLY, (method, name)
            assert params[name].default is False, (method, name)


@pytest.mark.cpu_only
def test_the_entry_hands_the_levers_to_the_launcher_unchanged(monkeypatch):
    seen = []
    monkeypatch.setattr(entry_module, "launch_window_triton", lambda *a, **k: seen.append((a, k)))
    args = tuple(range(9))
    op(*args)
    assert seen[-1] == (args, {name: False for name in _LEVERS})
    op(*args, **_LEVERS)
    assert seen[-1] == (args, _LEVERS)


class _Launches:
    """A kernel stand-in: records each launch's keyword arguments (its constexprs)."""

    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        return lambda *args, **kwargs: self.calls.append(kwargs)


def _helix8_buffers(page_stride: int = 5 * 64 * 576, pool_offset: int = 0):
    """helix8's window inputs on the CPU (never read): a request-major query, one layer of the
    five-layer FP8 pool (or a pool of the given page stride and byte offset), the block rows."""
    batch, blk, heads, dim, latent = 2, 6, 96, 576, 512
    rows = batch * blk
    query = torch.empty(rows, heads, dim, dtype=torch.bfloat16)
    raw = torch.empty(pool_offset + 4 * page_stride, dtype=torch.uint8)[pool_offset:]
    pool = raw.view(torch.float8_e4m3fn).as_strided((4, 64, dim), (page_stride, dim, 1))
    block_kv = torch.empty(rows, 1, dim, dtype=torch.float8_e4m3fn)[:, 0]
    workspace = window_workspace(batch, blk, heads, latent, 1, "cpu")
    plan = kw.plan_view(kw.plan_buffers(batch, blk, 1024, 64, 8, 1, "cpu"), batch, 1)
    stats = torch.empty(rows, heads, 2)
    return query, pool, block_kv, plan, workspace, stats


@pytest.mark.cpu_only
def test_vec_io_takes_the_vector_path_exactly_where_the_strides_allow(monkeypatch):
    """At helix8's strides both kernels launch ``ALIGNED`` under ``vec_io`` (and never without it);
    a pool page stride or base 8 bytes off a multiple of 16 runs the scalar path, warned once per
    kernel, and ``window_vec_io_aligned`` says the same as the launches."""
    from tensorrt_llm.logger import logger

    body, tail, warned = _Launches(), _Launches(), []
    monkeypatch.setattr(kw, "_window_body_kernel", body)
    monkeypatch.setattr(kw, "_window_tail_kernel", tail)
    monkeypatch.setattr(logger, "warning_once", lambda msg, *a, key=None: warned.append(key))

    query, pool, block_kv, plan, workspace, stats = _helix8_buffers()
    assert entry_module.window_vec_io_aligned(query, pool, block_kv, workspace, 512)
    for vec_io in (False, True):
        op(query, pool, block_kv, plan, 0.13, 512, 1.0, workspace, stats, vec_io=vec_io)
        assert body.calls[-1]["ALIGNED"] is vec_io and tail.calls[-1]["ALIGNED"] is vec_io
    assert not warned
    for bad in (dict(page_stride=64 * 576 + 8), dict(pool_offset=8)):
        query, pool, block_kv, plan, workspace, stats = _helix8_buffers(**bad)
        assert not entry_module.window_vec_io_aligned(query, pool, block_kv, workspace, 512)
        op(query, pool, block_kv, plan, 0.13, 512, 1.0, workspace, stats, vec_io=True)
        assert body.calls[-1]["ALIGNED"] is False and tail.calls[-1]["ALIGNED"] is False, bad
    refused = "dflash2_window_vec_io_refused_"
    assert sorted(warned) == [refused + "body"] * 2 + [refused + "tail"] * 2


@pytest.mark.cpu_only
def test_one_variable_sets_the_tails_wait_and_its_pdl_launch(monkeypatch):
    """Under PDL the body triggers its dependents from a normal launch, and the tail waits and
    launches as a dependent, from one variable; with the engine-wide switch off (or a CPU tensor)
    neither happens; ``launch_window_tail``, which may follow any kernel, never launches with PDL."""
    body, tail = _Launches(), _Launches()
    monkeypatch.setattr(kw, "_window_body_kernel", body)
    monkeypatch.setattr(kw, "_window_tail_kernel", tail)
    query, pool, block_kv, plan, workspace, stats = _helix8_buffers()
    assert kw._pdl_on(query) is False  # a CPU tensor: never PDL
    for switch in (True, False):  # the engine-wide switch on an SM107 GPU, then TRTLLM_ENABLE_PDL=0
        monkeypatch.setattr(kw, "_pdl_on", lambda q, on=switch: on)
        for pdl in (False, True):
            op(query, pool, block_kv, plan, 0.13, 512, 1.0, workspace, stats, pdl=pdl)
            want = pdl and switch
            assert body.calls[-1]["PDL_TRIGGER"] is want and "launch_pdl" not in body.calls[-1]
            assert tail.calls[-1]["PDL_WAIT"] is want and tail.calls[-1]["launch_pdl"] is want
    monkeypatch.setattr(kw, "_pdl_on", lambda q: True)
    partial = (workspace.partial_out[:2, :1], workspace.partial_stats[:2, :1])
    kw.launch_window_tail(query, pool, block_kv, plan, *partial, 0.13, 512, 1.0, stats)
    assert tail.calls[-1]["PDL_WAIT"] is False and tail.calls[-1]["launch_pdl"] is False
    assert "pdl" not in inspect.signature(kw.launch_window_tail).parameters


# GPU


def run_entry(inp: Inputs, rank: int, entry=op, workspace=None, plan=None, levers=None):
    spec = inp.spec
    rows, heads = inp.query.shape[:2]
    plan = make_plan(inp, rank) if plan is None else plan
    if workspace is None:
        workspace = window_workspace(
            len(inp.lengths),
            spec["block_size"],
            heads,
            spec["latent"],
            spec["num_splits"],
            inp.pool.device,
        )
    stats = torch.full((rows, heads, 2), float("nan"), device=inp.pool.device)
    # A cell with "tiles" runs the entry at a target's launch tiles (WindowTiles fields).
    launch = {"tiles": WindowTiles(**spec["tiles"])} if spec.get("tiles") else {}
    out = entry(
        inp.query,
        inp.pool,
        inp.block_kv,
        plan,
        spec["scale"],
        spec["latent"],
        1.0,
        workspace,
        stats if spec["stats"] else None,
        **(levers or {}),
        **launch,
    )
    return out, stats, plan, workspace


def check_against_reference(
    inp: Inputs, rank: int, out, stats, plan, workspace, entry=op, band=None
):
    spec = inp.spec
    ref_stats = torch.empty_like(stats)
    with true_fp32_matmul():
        ref = entry.reference(
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
    assert_within_ulp(out, ref, **(band or _BAND))
    if spec["stats"]:
        empty = ref_stats[..., 1] == 0
        assert torch.isneginf(stats[..., 0][empty]).all() and (stats[..., 1][empty] == 0).all()
        assert (out[empty] == 0).all()
        torch.testing.assert_close(stats[~empty], ref_stats[~empty], rtol=1e-4, atol=1e-4)


@gpu
@pytest.mark.parametrize("cell", op.CELLS, ids=_ids(op.CELLS))
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    inp = build(cell.spec, _seed(cell), "cuda")
    rank = cell.spec["cp_rank"]
    out, stats, plan, workspace = run_entry(inp, rank)
    check_against_reference(inp, rank, out, stats, plan, workspace)


@pytest.mark.parametrize(
    "tiles",
    [
        dict(block_m=64, block_dv=512, num_warps=4),
        dict(block_m=128, block_dv=256),
        dict(block_dv=48),
        dict(num_warps=3),
    ],
)
def test_window_tiles_refuse_bad_launches(tiles):
    """A tile larger than the default's (64 x 512 at 4 or 8 warps faults with a misaligned
    address on GB300), an edge that is not a power of two, or a warp count Triton does not take
    is refused when the tiles are made."""
    with pytest.raises(ValueError):
        WindowTiles(**tiles)


_GRAPH_SPEC = dict(op.CELLS[0].spec, batch=4, stats=True, cp_size=2, cp_rank=1, num_splits=3)


@gpu
@pytest.mark.parametrize("cp_size", [1, 2])
def test_capture_on_one_input_set_replay_on_another(cp_size):
    """Capture plan + window on inputs A, copy inputs B into the same buffers, replay: the
    replay equals an eager run on B bit for bit (rows with an empty body included)."""
    spec = dict(_GRAPH_SPEC, cp_size=cp_size, cp_rank=cp_size - 1)
    sets = [
        build(spec, 1, "cuda", lengths=(1031, 9000, 0, 65)),
        build(spec, 2, "cuda", lengths=(190000, 1, 4097, 7)),
    ]
    rank = spec["cp_rank"]
    batch, blk, heads = 4, spec["block_size"], spec["heads"]
    cols = max(s.local_table(rank).shape[1] for s in sets)
    pages = max(s.pool.shape[0] for s in sets)
    dim = spec["latent"] + spec["rope"]
    dtype = sets[0].pool.dtype
    num_ctx = torch.zeros(batch, dtype=torch.int64, device="cuda")
    table = torch.zeros(batch, cols, dtype=torch.int32, device="cuda")
    pool = torch.zeros(pages, spec["page_size"], dim, dtype=dtype, device="cuda")
    query = torch.zeros(batch * blk, heads, dim, dtype=torch.bfloat16, device="cuda")
    block_kv = torch.zeros(batch * blk, dim, dtype=dtype, device="cuda")
    stats = torch.zeros(batch * blk, heads, 2, device="cuda")
    buffers = plan_buffers(batch, blk, spec["window"], spec["page_size"], cp_size, 3, "cuda")
    workspace = window_workspace(batch, blk, heads, spec["latent"], 3, "cuda")

    def load(s: Inputs):
        num_ctx.copy_(s.num_ctx())
        local = s.local_table(rank)
        table.zero_()
        table[:, : local.shape[1]] = local
        pool.view(torch.uint8).zero_()
        pool[: s.pool.shape[0]] = s.pool
        query.copy_(s.query)
        block_kv.copy_(s.block_kv)

    def step():
        plan = rubin_dflash2_window_plan(
            num_ctx, table, None, blk, spec["window"], spec["page_size"], cp_size, rank, 3, buffers
        )
        return op(query, pool, block_kv, plan, spec["scale"], spec["latent"], 1.0, workspace, stats)

    load(sets[0])
    step()  # eager warmup: every kernel compiled before capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out = step()
    load(sets[1])
    graph.replay()
    replayed, replayed_stats = graph_out.clone(), stats.clone()
    eager = step()
    assert torch.isfinite(replayed).all()
    assert torch.equal(replayed, eager) and torch.equal(replayed_stats, stats)
    plan = make_plan(sets[1], rank)
    check_against_reference(sets[1], rank, replayed, replayed_stats, plan, workspace)


@gpu
@pytest.mark.parametrize("block_size", [6, 8])
@pytest.mark.parametrize("cp_size,rank", [(1, 0), (8, 0), (8, 1)])
def test_body_lengths_below_the_block_give_no_nan(block_size, cp_size, rank):
    """Body lengths 0 .. blk - 1 (dummy rows have 0); under helix a rank may hold block keys
    of a row and no body page, or nothing of it at all."""
    spec = dict(
        op.CELLS[0].spec,
        batch=block_size + 2,
        block_size=block_size,
        cp_size=cp_size,
        cp_rank=rank,
        stats=cp_size > 1,
        num_splits=2,
    )
    lengths = [*range(block_size), 60, 66]
    inp = build(spec, 5, "cuda", lengths=lengths)
    out, stats, plan, workspace = run_entry(inp, rank)
    check_against_reference(inp, rank, out, stats, plan, workspace)


def _compiled_variants() -> int:
    total = 0
    for fn in (kw._window_plan_kernel, kw._window_body_kernel, kw._window_tail_kernel):
        caches = getattr(fn, "device_caches", None)
        if caches is None:
            pytest.skip("this Triton keeps no per-device kernel cache to count")
        total += sum(len(entry[0]) for entry in caches.values())
    return total


@gpu
def test_no_compile_after_the_warmup_pass():
    """New lengths, new pool contents and new batch rows never compile: only the static shapes
    of a graph (batch bucket, split count) and the launch levers select a kernel variant."""
    spec = dict(op.CELLS[0].spec, stats=False)
    buckets = (1, 2, 4)
    rounds = [
        [(1031,), (9000, 1), (0, 63, 4097, 190000)],
        [(65,), (190000, 4097), (1, 1022, 65536, 5)],
    ]
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    counts = []
    for lengths_per_bucket in rounds:
        for batch, lengths in zip(buckets, lengths_per_bucket):
            splits = split_count(num_sms, batch, spec["heads"], spec["block_size"], 1024, 64, 1)
            inp = build(dict(spec, num_splits=splits), batch, "cuda", lengths=lengths)
            for levers in ({}, _LEVERS):
                run_entry(inp, 0, levers=levers)
        torch.cuda.synchronize()
        counts.append(_compiled_variants())
    assert counts[1] == counts[0], f"compiled {counts[1] - counts[0]} variants after warmup"
