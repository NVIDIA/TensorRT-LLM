# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the musespark_open router_shard entry.

The entry claims the built-in router projection's bits for its rank's 32
columns: for every rank, the concatenation of the eight ranks' outputs is
``MuseSparkOpenMoE.router_logits`` bit for bit, at every row count 1..8 and at
the warp count the built-in launch uses for that row count (with and without
the MoE prepare stream), with or without programmatic dependent launch and the
early trigger. The cells are also gated against the entry's float64 reference.

The plain-launch entry ``router_shard_side`` (``router_fork``) claims the same bits
into a caller-allocated output: checked the same way, and in a CUDA graph where it
runs on a side stream forked behind a producer and is joined before a
programmatic-dependent-launch consumer -- the captured edge into it must be a full
one and the edge out of it a programmatic one, and the consumer must still read its
output (a long sleep in front of the side kernel), replay after replay.

``MODELING_V2_TIMING=1`` also runs the microbenchmark (printed, never gated).
"""

import os
import zlib
from types import SimpleNamespace

import pytest
import torch
import triton
import triton.language as tl
from torch import nn

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.moe.router_shard import (  # noqa: E501
    EXPERTS,
    HIDDEN,
    LOCAL_EXPERTS,
    MAX_ROWS,
    RANKS,
    builtin_warps,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.moe.router_shard import (  # noqa: E501
    router_shard as op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.moe.router_shard import (  # noqa: E501
    router_shard_side as side_op,
)
from tensorrt_llm._torch.models.modeling_musespark_open import MuseSparkOpenMoE

__extra_import_path__ = ["../../../../..", "../.."]
from _validating import (  # noqa: E402,F401 — path above; fixture by name
    assert_within_ulp,
    certifying,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="router_shard needs a GPU")

DEV = "cuda"
#: The checkpoint's metap multiplier: (hidden / base_width) ** -0.5 with base_width 256.
CKPT_SCALE = (HIDDEN / 256.0) ** -0.5
#: (enable_pdl, early_trigger) launch modes; none changes a bit.
LAUNCHES = ((False, False), (True, False), (True, True))


def _builtin_moe(prepare_stream: bool, weight: torch.Tensor) -> MuseSparkOpenMoE:
    """A built-in MoE as far as ``router_logits`` reads it."""
    moe = MuseSparkOpenMoE.__new__(MuseSparkOpenMoE)
    nn.Module.__init__(moe)
    moe.layer_idx = 0
    moe.num_experts = EXPERTS
    moe.hidden_size = HIDDEN
    moe.metap_multiplier = CKPT_SCALE
    moe._bf16x3_router_enabled = False
    moe.router = nn.Linear(HIDDEN, EXPERTS, bias=False, device=DEV, dtype=torch.float32)
    with torch.no_grad():
        moe.router.weight.copy_(weight)
    # ``router_logits`` only asks whether a prepare stream exists; a placeholder
    # keeps the hundreds of stand-ins below from creating CUDA streams.
    moe._prepare_stream = _PREPARE_STREAM if prepare_stream else None
    moe.experts = SimpleNamespace(comm=None)
    return moe


_PREPARE_STREAM = object()


def _inputs(rows: int, seed: int):
    """BF16 rows at a log-normal per-row scale (sigma 1.5) and an N(0, 0.02^2) FP32 weight."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    x = torch.randn(rows, HIDDEN, generator=g, device=DEV)
    x = (x * torch.exp(1.5 * torch.randn(rows, 1, generator=g, device=DEV))).to(torch.bfloat16)
    weight = torch.randn(EXPERTS, HIDDEN, generator=g, device=DEV) * 0.02
    return x, weight


def _all_ranks(x, weight, num_warps, enable_pdl=False, early_trigger=False):
    return torch.cat(
        [
            op(x, weight, rank, CKPT_SCALE, num_warps, enable_pdl, early_trigger)
            for rank in range(RANKS)
        ],
        dim=1,
    )


def _assert_bits_equal(got: torch.Tensor, want: torch.Tensor, what: str) -> None:
    assert got.dtype == want.dtype == torch.float32, what
    assert got.shape == want.shape, (what, got.shape, want.shape)
    diff = got.view(torch.int32) != want.view(torch.int32)
    if diff.any():
        rows, cols = diff.nonzero(as_tuple=True)
        raise AssertionError(
            f"{what}: {int(diff.sum())} logit(s) differ from the built-in projection, first at "
            f"row {int(rows[0])} expert {int(cols[0])}: {got[rows[0], cols[0]].item()!r} vs "
            f"{want[rows[0], cols[0]].item()!r}"
        )


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:48] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    certifying(op)
    rows, num_warps = cell.spec["rows"], cell.spec["num_warps"]
    x, weight = _inputs(rows, zlib.crc32(cell.why.encode()))
    before = (x.clone(), weight.clone())
    prepare_stream = num_warps == 16
    assert builtin_warps(rows, prepare_stream) == num_warps, "a cell the built-in never launches"
    want = MuseSparkOpenMoE.router_logits(_builtin_moe(prepare_stream, weight), x)
    for rank in range(RANKS):
        out = op(x, weight, rank, CKPT_SCALE, num_warps)
        assert out.is_contiguous() and out.shape == (rows, LOCAL_EXPERTS)
        # Within 16 FP32 ulps of the row's largest logit, 2 ulps RMS (router_gemv's band:
        # the same 8192-term sums in the same order).
        assert_within_ulp(
            out,
            op.reference(x, weight, rank, CKPT_SCALE, num_warps).to(out.dtype),
            element_ulp=16.0,
            rms_ulp=2.0,
        )
        first = rank * LOCAL_EXPERTS
        _assert_bits_equal(
            out, want[:, first : first + LOCAL_EXPERTS], f"rank {rank} rows={rows} w={num_warps}"
        )
    assert torch.equal(x, before[0]) and torch.equal(weight, before[1]), "inputs are read-only"


@pytest.mark.parametrize("prepare_stream", (True, False), ids=("prepare_stream", "no_prepare"))
@pytest.mark.parametrize("rows", range(1, MAX_ROWS + 1))
def test_bit_identity_with_the_builtin_projection(rows, prepare_stream) -> None:
    """Every row count, both launch layouts, every launch mode, 16 draws each."""
    num_warps = builtin_warps(rows, prepare_stream)
    for seed in range(16):
        x, weight = _inputs(rows, seed=1000 * rows + seed + 7 * prepare_stream)
        want = MuseSparkOpenMoE.router_logits(_builtin_moe(prepare_stream, weight), x)
        for enable_pdl, early in LAUNCHES:
            _assert_bits_equal(
                _all_ranks(x, weight, num_warps, enable_pdl, early),
                want,
                f"rows={rows} seed={seed} pdl={enable_pdl} early={early}",
            )


def test_cuda_graph_capture_and_replay_with_pdl() -> None:
    """Captured with PDL and the early trigger behind a producer kernel of its input."""
    x, weight = _inputs(8, seed=8)
    source = torch.empty_like(x)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            source.copy_(x)
            _all_ranks(source, weight, 16, True, True)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        source.copy_(x)
        outs = [op(source, weight, r, CKPT_SCALE, 16, True, True) for r in range(RANKS)]
    for seed in range(4):
        new_x, new_weight = _inputs(8, seed=18 + seed)
        x.copy_(new_x)
        weight.copy_(new_weight)
        graph.replay()
        torch.cuda.synchronize()
        want = MuseSparkOpenMoE.router_logits(_builtin_moe(True, weight), x)
        _assert_bits_equal(torch.cat(outs, dim=1), want, f"graph replay {seed}")


def _side_all_ranks(x, weight, num_warps):
    outs = [torch.full((x.shape[0], LOCAL_EXPERTS), float("nan"), device=DEV) for _ in range(RANKS)]
    for rank, out in enumerate(outs):
        assert side_op(x, weight, rank, CKPT_SCALE, num_warps, out) is out
    return torch.cat(outs, dim=1)


@pytest.mark.parametrize("cell", side_op.CELLS, ids=[c.why[:48] for c in side_op.CELLS])
def test_side_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    certifying(side_op)
    rows, num_warps = cell.spec["rows"], cell.spec["num_warps"]
    x, weight = _inputs(rows, zlib.crc32(cell.why.encode()) + 1)
    before = (x.clone(), weight.clone())
    prepare_stream = num_warps == 16
    assert builtin_warps(rows, prepare_stream) == num_warps, "a cell the built-in never launches"
    want = MuseSparkOpenMoE.router_logits(_builtin_moe(prepare_stream, weight), x)
    for rank in range(RANKS):
        out = torch.full((rows, LOCAL_EXPERTS), float("nan"), device=DEV)
        side_op(x, weight, rank, CKPT_SCALE, num_warps, out)
        assert_within_ulp(
            out,
            side_op.reference(x, weight, rank, CKPT_SCALE, num_warps, out).to(out.dtype),
            element_ulp=16.0,
            rms_ulp=2.0,
        )
        first = rank * LOCAL_EXPERTS
        _assert_bits_equal(
            out, want[:, first : first + LOCAL_EXPERTS], f"side rank {rank} rows={rows}"
        )
    assert torch.equal(x, before[0]) and torch.equal(weight, before[1]), "inputs are read-only"


@pytest.mark.parametrize("prepare_stream", (True, False), ids=("prepare_stream", "no_prepare"))
@pytest.mark.parametrize("rows", range(1, MAX_ROWS + 1))
def test_side_bit_identity_with_the_shard_and_the_builtin(rows, prepare_stream) -> None:
    """The plain launch gives router_shard's bits (every launch mode) and the built-in's."""
    num_warps = builtin_warps(rows, prepare_stream)
    for seed in range(16):
        x, weight = _inputs(rows, seed=3000 * rows + seed + 7 * prepare_stream)
        want = MuseSparkOpenMoE.router_logits(_builtin_moe(prepare_stream, weight), x)
        got = _side_all_ranks(x, weight, num_warps)
        _assert_bits_equal(got, want, f"side rows={rows} seed={seed}")
        for enable_pdl, early in LAUNCHES:
            _assert_bits_equal(
                got,
                _all_ranks(x, weight, num_warps, enable_pdl, early),
                f"side vs shard rows={rows} seed={seed} pdl={enable_pdl} early={early}",
            )


def _edges(graph) -> list:
    """``(source kernel, target kernel, programmatic)`` of every kernel-to-kernel edge."""
    from cuda.bindings import driver

    def name(node) -> str:
        _, kind = driver.cuGraphNodeGetType(node)
        if kind != driver.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL:
            return f"<{kind.name}>"
        _, params = driver.cuGraphKernelNodeGetParams(node)
        status, raw = driver.cuFuncGetName(params.func)
        if status != driver.CUresult.CUDA_SUCCESS:
            status, raw = driver.cuKernelGetName(params.kern)
        return raw.decode() if isinstance(raw, bytes) else str(raw)

    handle = driver.CUgraph(graph.raw_cuda_graph())
    get = getattr(driver, "cuGraphGetEdges_v2", None) or driver.cuGraphGetEdges
    total = int(get(handle, 0)[-1])
    _, sources, targets, data, _ = get(handle, total)
    programmatic = int(driver.CUgraphDependencyType.CU_GRAPH_DEPENDENCY_TYPE_PROGRAMMATIC)
    return [
        (name(a), name(b), int(d.type) == programmatic) for a, b, d in zip(sources, targets, data)
    ]


@triton.jit
def _globaltimer(dummy):
    return tl.inline_asm_elementwise(
        "mov.u64 $0, %globaltimer;", "=l,r", [dummy], dtype=tl.int64, is_pure=False, pack=1
    )


@triton.jit
def _hop_stand_in(src, dst, ns, BLOCK: tl.constexpr):
    # The early-trigger hop's shape: release the dependents early, then (after the wait) spend a while
    # before the stores the forked side kernel reads.
    tl.extra.cuda.gdc_launch_dependents()
    tl.extra.cuda.gdc_wait()
    limit = tl.load(ns)
    pid = tl.program_id(0)
    start = _globaltimer(pid)
    now = start
    while now - start < limit:
        now = _globaltimer(pid)
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(dst + offsets, tl.load(src + offsets))


@triton.jit
def _pdl_consumer(src, dst, N: tl.constexpr):
    # The routed pool kernel's shape: nothing read before the wait.
    tl.extra.cuda.gdc_wait()
    offsets = tl.arange(0, N)
    tl.store(dst + offsets, tl.load(src + offsets))


def test_side_launch_forked_and_joined_in_a_cuda_graph() -> None:
    """The router fork's shape in one graph: a hop stand-in that triggers its dependents at entry and
    stores its output 200 us later -> fork -> [side: (sleep), router_shard_side] and, on
    the capturing stream, a PDL GEMV stand-in (router_shard with PDL) -> join -> a PDL
    consumer that reads the side output after its wait. The captured edge into the side
    kernel is a full one (it reads the hop's output with no wait of its own), the edge out
    of it a programmatic one; replays with new inputs give the built-in bits, also with a
    2M-cycle sleep in front of the side kernel."""
    from _graph_edges import dedicated_stream

    x, weight = _inputs(8, seed=31)
    pool_weight = _inputs(8, seed=32)[1]
    source = torch.zeros_like(x)
    logits = torch.empty(8, LOCAL_EXPERTS, device=DEV)
    consumed = torch.empty_like(logits)
    delay_ns = torch.tensor([200_000], dtype=torch.int64, device=DEV)
    side = dedicated_stream()
    block = 1024

    def step(sleep: bool):
        _hop_stand_in[(x.numel() // block,)](x, source, delay_ns, BLOCK=block, launch_pdl=True)
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            if sleep:
                torch.cuda._sleep(2_000_000)
            side_op(source, weight, 3, CKPT_SCALE, 16, logits)
        pooled = op(source, pool_weight, 5, CKPT_SCALE, 16, True, False)
        torch.cuda.current_stream().wait_stream(side)
        _pdl_consumer[(1,)](logits, consumed, N=8 * LOCAL_EXPERTS, launch_pdl=True)
        return pooled

    for sleep in (False, True):
        for _ in range(2):
            step(sleep)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph):
            pooled = step(sleep)
        edges = _edges(graph)
        into = [e for e in edges if e[1] == "_router_shard_side_kernel"]
        out_of = [e for e in edges if e[0] == "_router_shard_side_kernel"]
        # torch.cuda._sleep launches ATen's spin_kernel.
        source_kernel = "spin_kernel" if sleep else "_hop_stand_in"
        assert len(into) == 1 and source_kernel in into[0][0] and not into[0][2], into
        if sleep:
            assert not any(prog for _, b, prog in edges if "spin_kernel" in b), edges
        assert out_of == [("_router_shard_side_kernel", "_pdl_consumer", True)], out_of
        assert ("_hop_stand_in", "_router_shard_kernel", True) in edges, edges
        for seed in range(4):
            new_x, new_weight = _inputs(8, seed=41 + seed + 10 * sleep)
            x.copy_(new_x)
            weight.copy_(new_weight)
            source.zero_()
            logits.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            want = MuseSparkOpenMoE.router_logits(_builtin_moe(True, weight), x)
            what = f"sleep={sleep} replay {seed}"
            _assert_bits_equal(logits, want[:, 3 * LOCAL_EXPERTS : 4 * LOCAL_EXPERTS], what)
            _assert_bits_equal(consumed, logits, f"consumer {what}")
            _assert_bits_equal(pooled, op(x, pool_weight, 5, CKPT_SCALE, 16), f"pool {what}")


@pytest.mark.parametrize("rows", (1, 3, 8))
def test_side_is_valid_admits_the_served_inputs_and_refuses_others(rows):
    x, weight = _inputs(rows, seed=rows + 50)
    out = torch.empty(rows, LOCAL_EXPERTS, device=DEV)
    side_op.is_valid(x, weight, 3, CKPT_SCALE, 16, out)
    for bad_out in (
        torch.empty(rows, LOCAL_EXPERTS, device=DEV, dtype=torch.bfloat16),
        torch.empty(rows + 1, LOCAL_EXPERTS, device=DEV),
        torch.empty(rows, 2 * LOCAL_EXPERTS, device=DEV)[:, ::2],
        torch.empty(rows, LOCAL_EXPERTS),
    ):
        with pytest.raises(ValueError):
            side_op.is_valid(x, weight, 3, CKPT_SCALE, 16, bad_out)
    with pytest.raises(ValueError):
        side_op.is_valid(x, weight, 3, CKPT_SCALE, 8, out)


@pytest.mark.parametrize("rows", (1, 3, 8))
def test_is_valid_admits_the_served_inputs_and_refuses_others(rows):
    x, weight = _inputs(rows, seed=rows)
    op.is_valid(x, weight, 3, CKPT_SCALE, 16)
    for bad in (
        (x.float(), weight, 3, CKPT_SCALE, 16),
        (x, weight.to(torch.bfloat16), 3, CKPT_SCALE, 16),
        (x, weight.t(), 3, CKPT_SCALE, 16),
        (x, weight, 8, CKPT_SCALE, 16),
        (x, weight, 3, CKPT_SCALE, 8),
    ):
        with pytest.raises(ValueError):
            op.is_valid(*bad)
    with pytest.raises(ValueError):
        op.is_valid(torch.cat([x] * 9)[: rows + 8].contiguous(), weight, 3, CKPT_SCALE, 16)


@pytest.mark.skipif(os.environ.get("MODELING_V2_TIMING") != "1", reason="set MODELING_V2_TIMING=1")
def test_microbenchmark() -> None:
    """µs per call in a CUDA graph of 62 calls over 62 distinct weights (every call
    reads its weight from HBM), the built-in full projection vs one rank's shard.
    Printed, never gated."""
    layers = 62
    weights = [_inputs(1, seed=100 + i)[1] for i in range(layers)]
    builtin = [_builtin_moe(True, w) for w in weights]

    def timed(fn) -> float:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        samples = []
        for _ in range(30):
            start.record()
            graph.replay()
            end.record()
            torch.cuda.synchronize()
            samples.append(start.elapsed_time(end) * 1000.0 / layers)
        return sorted(samples)[len(samples) // 2]

    print()
    for rows in (1, 8):
        x = _inputs(rows, seed=rows)[0]
        old = timed(lambda: [MuseSparkOpenMoE.router_logits(m, x) for m in builtin])
        new = timed(lambda: [op(x, w, 3, CKPT_SCALE, 16) for w in weights])
        new_pdl = timed(lambda: [op(x, w, 3, CKPT_SCALE, 16, True) for w in weights])
        print(
            f"  rows={rows}: built-in {old:.2f} us/call, router_shard {new:.2f} us/call, "
            f"with PDL {new_pdl:.2f} us/call"
        )
