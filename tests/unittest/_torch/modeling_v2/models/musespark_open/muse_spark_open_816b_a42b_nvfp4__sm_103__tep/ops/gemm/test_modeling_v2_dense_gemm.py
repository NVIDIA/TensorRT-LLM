# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the musespark_open dense_gemm and l2_prefetch entries, and their ``dense_gemms``
wiring and named forks.

``dense_gemm`` is not bit-identical to the built-in dispatcher -- which is not
one kernel: each rank keeps its own autotuned winner -- so besides the cells
(gated against the entry's float64 reference within a stated band) the planned
kernel of every site and row count is held to a fixed error budget, relative to
the kernels the ranks' built-in dispatch runs for the same rows, on the same inputs:

* B1' -- its largest distance from the correctly rounded product, in BF16 ulps,
  is no larger than the worst built-in kernel's;
* B2 -- its rate of outputs that differ from the correctly rounded product is at
  most 1.25 x the worst built-in kernel's;
* B3 -- its largest error normalized by ``sum_k |x_k w_k|`` is at most 1.25 x the
  worst built-in kernel's;
* B4 -- repeated calls, both launch modes and CUDA-graph replays give the same bits.

A named fork (``DenseGemmPlan.add_fork``) is checked in a captured graph: its
kernel follows its predecessor by a full edge and precedes the GEMM of its join
role by a programmatic edge, beside that GEMM's own programmatic predecessor.

``MODELING_V2_TIMING=1`` also runs the microbenchmark (printed, never gated).
"""

import logging
import os
import random
import types

import pytest
import torch
import triton
import triton.language as tl

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.dense_gemms import (  # noqa: E501
    dispatcher as dense_gemms,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.gemm.dense_gemm import (  # noqa: E501
    MAX_ROWS,
    O_PROJ_SHAPE,
    POOL_SHAPE,
    QKV_SHAPE,
    SPLITK_CTA_K,
    UNPOOL_SHAPE,
    post_wait_columns,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.gemm.dense_gemm import (  # noqa: E501
    dense_gemm as op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.gemm.dense_gemm import (  # noqa: E501
    l2_prefetch as prefetch_op,
)
from tensorrt_llm._torch.modules import low_m_gemm
from tensorrt_llm._torch.modules.linear import Linear

__extra_import_path__ = ["../../../../.."]
from _validating import (  # noqa: E402,F401 — path above; fixture by name
    assert_within_ulp,
    certifying,
)

assert torch.cuda.is_available(), "dense_gemm requires a CUDA device"

DEV = "cuda"
SITE_SHAPES = {"qkv": QKV_SHAPE, "o_proj": O_PROJ_SHAPE, "pool": POOL_SHAPE, "unpool": UNPOOL_SHAPE}
_CUB, _S10, _S11, _S12 = "cublas", "splitk:64,8,4,10", "splitk:64,8,4,11", "splitk:64,8,4,12"
_D256_1, _D256_2, _D256_4 = "direct:256,1", "direct:256,2", "direct:256,4"
_D128_2, _D128_4, _D64_1, _D64_2 = "direct:128,2", "direct:128,4", "direct:64,1", "direct:64,2"
#: The kernel the built-in dispatch picks on ranks 0..7, per site and row count, as one
#: autotuner cache of the built-in model at K=7 records them. Ranks 1..7 run cuBLAS for QKV
#: and o_proj at every row count.
S0_KERNELS = {
    "qkv": {
        m: (rank0,) + (_CUB,) * 7
        for m, rank0 in enumerate(
            (
                _D256_2,
                "splitk:64,16,4,10",
                "splitk:128,16,4,5",
                "splitk:64,16,4,10",
                _S11,
                _S11,
                _S10,
                "splitk:64,16,4,10",
            ),
            start=1,
        )
    },
    "o_proj": {
        m: (rank0,) + (_CUB,) * 7
        for m, rank0 in enumerate(
            (
                "direct:128,4",
                "splitk:128,16,1,6",
                "splitk:128,32,1,5",
                "splitk:128,8,1,5",
                "splitk:128,8,1,6",
                "splitk:64,16,1,11",
                "splitk:128,16,1,6",
                "splitk:128,8,2,5",
            ),
            start=1,
        )
    },
    "pool": {
        1: (_D256_2, _D256_2, _D128_2, _D256_4, _D256_2, _D256_2, _D256_2, _D256_4),
        2: (_D256_2, _D256_2, _D256_2, _D256_1, _D256_2, _D256_2, _D256_2, _D256_2),
        3: (_D256_2, _D256_2, _D128_2, _D256_2, _D128_2, _D256_2, _D256_2, _D128_2),
        4: (_D256_2, _D256_2, _D128_2, _D256_2, _D128_2, _D256_2, _D64_1, _D256_2),
        5: (_D128_2, _D256_2, _D128_2, _D128_2, _D128_2, _D128_2, _D128_2, _D256_2),
        6: (_D128_2, _D64_1, _D64_1, _D64_1, _D128_2, _D128_2, _D128_2, _D128_2),
        7: (_S11, _D128_2, _S10, _D128_2, _S12, _S11, _S12, _S10),
        8: (_S11, _S10, _S12, _S11, _S12, _S12, _S12, _S11),
    },
    "unpool": {
        1: (_D256_2, _D256_4, _D128_4, _D128_2, _D256_4, _D256_2, _D256_4, _D256_2),
        2: (_D256_2, _D256_4, _D256_2, _D256_2, _D256_2, _D256_2, _D64_2, _D128_2),
        3: (_D128_2, _D128_2, _D256_2, _D128_2, _D128_2, _D256_2, _D128_2, _D64_2),
        4: (_D64_2, _D128_4, _D128_2, _D128_2, _D128_4, _D64_2, _D128_4, _D64_2),
        5: (_D64_2, _D128_2, _D64_2, _D128_2, _S10, _D64_2, _D128_2, _D128_2),
        6: (_S10, _D64_2, _S11, _S12, _S10, _S10, _S12, _S11),
        7: (_S11, _D128_4, _S10, _S10, _S10, _S10, _S12, _S10),
        8: (_S10, "splitk:64,16,4,8", _S11, _S10, _S10, _S12, _S12, _S10),
    },
}
#: The ``dense_gemms`` plan of this device: ``PLAN[site][rows - 1]``.
PLAN = dense_gemms.default_kernels(capability=dense_gemms.device_capability())


def _operands(m, k, n, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    x = torch.randn(m, k, generator=g, device=DEV).to(torch.bfloat16)
    w = (0.02 * torch.randn(n, k, generator=g, device=DEV)).to(torch.bfloat16)
    return x, w


# --- the entry's cells ------------------------------------------------------------------


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:48] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    """Every cell against the float64 reference.

    Band: the kernels accumulate in FP32 and round once, so an output differs
    from the correctly rounded product by at most one BF16 ulp of itself plus an
    FP32 summation error far below that (``K * 2^-24`` relative to the row's
    terms); one ulp of an element is at most one ulp of its row's largest
    magnitude, hence 1 element ulp. A rounding flip moves a value by one ulp; at
    a measured flip rate below 2e-3 the RMS over a tensor stays far below 0.05 ulp.
    """
    certifying(op)
    s = cell.spec
    x, w = _operands(s["m"], s["k"], s["n"], seed=11)
    out = op(x, w, s["kernel"])
    ref = op.reference(x, w, s["kernel"])
    assert out.shape == (s["m"], s["n"]) and out.dtype == torch.bfloat16
    assert_within_ulp(out, ref, element_ulp=1.0, rms_ulp=0.05)


@pytest.mark.parametrize("cell", prefetch_op.CELLS, ids=[c.why[:48] for c in prefetch_op.CELLS])
def test_prefetch_cells_change_nothing(cell, certifying) -> None:  # noqa: F811
    """A prefetch leaves the tensor and the sink as they were, and the GEMM after it the same."""
    certifying(prefetch_op)
    s = cell.spec
    g = torch.Generator(device=DEV).manual_seed(5)
    base = torch.randn(s["row_offset"] + s["rows"], s["cols"], generator=g, device=DEV).to(
        torch.bfloat16
    )
    view = base[s["row_offset"] :]
    before = base.clone()
    sink = torch.zeros(1, dtype=torch.int32, device=DEV)
    assert prefetch_op(view, sink) is None
    assert prefetch_op.reference(view, sink) is None
    torch.cuda.synchronize()
    assert torch.equal(base, before) and int(sink.item()) == 0


def test_the_compiled_prefetch_issues_bulk_l2_prefetches() -> None:
    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.gemm import (  # noqa: E501
        dense_gemm as module,
    )

    w = torch.zeros(64, 1024, dtype=torch.bfloat16, device=DEV)
    sink = torch.zeros(1, dtype=torch.int32, device=DEV)
    compiled = module._l2_prefetch_kernel[(1,)](
        w.data_ptr(), w.numel() * 2, sink, CHUNK=module.PREFETCH_CHUNK, LANES=module.PREFETCH_LANES
    )
    ptx = compiled.asm["ptx"]
    assert "cp.async.bulk.prefetch.L2.global" in ptx
    assert "ld.global" not in ptx, "a prefetch must not read the tensor itself"
    # It never triggers its dependents: a PDL successor launches only after its last CTA exits.
    assert "griddepcontrol" not in ptx


def test_post_wait_columns_use_the_split_k_kernels_k_tile() -> None:
    """``post_wait_columns`` counts k-tiles of flashinfer's ``_CTA_K`` columns."""
    from flashinfer.gemm.kernels import dense_bf16_gemm_sm100_splitk as splitk

    assert splitk._CTA_K == SPLITK_CTA_K
    k = QKV_SHAPE[0]
    assert post_wait_columns(PLAN["qkv"][7], k) == (1536, 512, 2048, 4)
    assert post_wait_columns(PLAN["qkv"][0], k) is None  # the direct kernel waits first


# --- the error budget -------------------------------------------------------------------


def _ordered(bf16):
    """BF16 bit patterns on a line where adjacent values are adjacent integers."""
    bits = bf16.view(torch.int16).int()
    return torch.where(bits < 0, -32768 - bits, bits)


def _budget_stats(kernel, k, n, m, draws, outliers):
    """flip rate, max ulp distance, max normalized error vs float64, over seeded draws.

    With ``outliers``, 1 % of the activation channels are scaled by 30, as the
    residual stream's outlier channels are.
    """
    flips = total = 0
    max_ulp = 0
    max_norm = 0.0
    for d in range(draws):
        x, w = _operands(m, k, n, seed=1000 + d)
        if outliers:
            g = torch.Generator(device=DEV).manual_seed(4000 + d)
            channels = torch.randperm(k, generator=g, device=DEV)[: k // 100]
            x = x.float()
            x[:, channels] *= 30.0
            x = x.to(torch.bfloat16)
        exact = x.double() @ w.double().t()
        rounded = exact.float().to(torch.bfloat16)
        out = op(x, w, kernel)
        flips += int((out != rounded).sum())
        total += out.numel()
        max_ulp = max(max_ulp, int((_ordered(out) - _ordered(rounded)).abs().max()))
        scale = x.double().abs() @ w.double().abs().t()
        max_norm = max(max_norm, float(((out.double() - exact).abs() / scale).max()))
    return flips / total, max_ulp, max_norm


@pytest.mark.parametrize("outliers", (False, True), ids=("normal", "outliers"))
@pytest.mark.parametrize("m", range(1, 9))
@pytest.mark.parametrize("site", tuple(SITE_SHAPES))
def test_the_planned_kernel_is_within_the_error_budget(site, m, outliers) -> None:
    """The plan's kernel against every kernel some built-in rank runs for the same rows."""
    k, n = SITE_SHAPES[site]
    draws = 12
    planned = _budget_stats(PLAN[site][m - 1], k, n, m, draws, outliers)
    builtin = [
        _budget_stats(kernel, k, n, m, draws, outliers)
        for kernel in sorted(set(S0_KERNELS[site][m]))
    ]
    worse_flip = max(b[0] for b in builtin)
    worse_ulp = max(b[1] for b in builtin)
    worse_norm = max(b[2] for b in builtin)
    print(
        f"[budget] {site} M={m} outliers={outliers}: planned {PLAN[site][m - 1]} {planned} built-in "
        f"{sorted(set(S0_KERNELS[site][m]))} {builtin}"
    )
    assert planned[0] <= 1.25 * worse_flip, ("B2", planned, builtin)
    assert planned[1] <= worse_ulp, ("B1'", planned, builtin)
    assert planned[2] <= 1.25 * worse_norm, ("B3", planned, builtin)


@pytest.mark.parametrize(
    "site, pinned, siblings",
    [
        ("qkv", "splitk:64,8,4,12", ("splitk:64,16,4,10", "splitk:128,8,4,6")),
        ("pool", "splitk:64,8,4,12", ("splitk:64,8,4,10", "splitk:64,8,4,11")),
        (
            "unpool",
            "splitk:64,8,4,12",
            ("splitk:64,8,4,10", "splitk:64,8,4,11", "splitk:64,16,4,8"),
        ),
    ],
)
def test_split_k4_siblings_compute_the_same_bits(site, pinned, siblings) -> None:
    """Split-K (64,8,4,12) sums the same K quarters in the same order as the built-in split-K-4
    tactics. At 8 rows the plan therefore gives rank 0's QKV bits and every rank's pooler and
    unpooler bits."""
    assert PLAN[site][7] == pinned
    assert {kernel for kernel in S0_KERNELS[site][8] if kernel.startswith("splitk")} <= {
        pinned,
        *siblings,
    }
    k, n = SITE_SHAPES[site]
    for m in (1, 3, 8):
        for seed in range(4):
            x, w = _operands(m, k, n, seed=200 + seed)
            out = op(x, w, pinned)
            for sibling in siblings:
                assert torch.equal(out, op(x, w, sibling)), (site, m, seed, sibling)


#: TP4 (one tray) per-rank shapes ``(K, N)`` by site; ``*_replicated``: the pooler and unpooler
#: on the whole weights, as the built-in model runs them at TP4 without sharded projections.
TP4_SITE_SHAPES = {
    "qkv": (8192, 4608),
    "o_proj": (2048, 8192),
    "pool": (8192, 1024),
    "unpool": (4096, 2048),
    "pool_replicated": (8192, 4096),
    "unpool_replicated": (4096, 8192),
}
#: The TP4 plan per site and row count (``*_replicated``: the replicated layout's table).
TP4_PLAN = dict(
    dense_gemms.default_kernels(4, sharded=True),
    pool_replicated=dense_gemms.default_kernels(4, sharded=False)["pool"],
    unpool_replicated=dense_gemms.default_kernels(4, sharded=False)["unpool"],
)
_K2_11, _K2_12, _K1_11, _K1_12 = (
    "splitk:64,8,2,11",
    "splitk:64,8,2,12",
    "splitk:64,8,1,11",
    "splitk:64,8,1,12",
)
#: The kernel one sm103 rank's built-in autotuner picks for the TP4 shapes, rows 1..8 (from
#: a tactic sweep); cuBLAS is the other kernel the built-in race keeps on some rank.
TP4_AUTOTUNED = {
    "qkv": (_D256_2, _K2_12, _K2_12, _K2_12, _K2_12, "splitk:128,8,4,6", _K2_12, _K2_11),
    "o_proj": ("direct:128,4", _K1_12, _K1_11, _K1_12, _K1_11, _K1_12, _K1_11, _K1_12),
    "pool": (_D128_2, _D64_2, _D64_2, _S11, _S12, _S12, _S12, _S12),
    "unpool": (
        _D128_4,
        _D128_2,
        _D128_4,
        _K2_11,
        "splitk:128,8,4,6",
        _K2_12,
        "splitk:64,8,2,10",
        "splitk:128,8,4,6",
    ),
    "pool_replicated": (_D256_2,) + (_K2_12,) * 7,
    "unpool_replicated": (_D128_2, _K1_12, _K1_12, _K1_12, _K1_12, _K1_12, _K1_11, _K1_12),
}


def test_the_tp4_plan_validates_on_its_shapes() -> None:
    for site, (k, n) in TP4_SITE_SHAPES.items():
        for m in range(1, MAX_ROWS + 1):
            assert dense_gemms.kernel_error(TP4_PLAN[site][m - 1], m, k, n) is None, (site, m)


@pytest.mark.parametrize("m", (1, 2, 3, 5, 8))
@pytest.mark.parametrize("site", tuple(TP4_SITE_SHAPES))
def test_the_tp4_planned_kernel_is_within_the_error_budget(site, m) -> None:
    """The TP4 plan's kernel against the kernels a built-in rank runs for the same rows at TP4
    (its autotuned pick and cuBLAS), with the TP8 budget: B1' no worse in ulps, B2 / B3 within 1.25x."""
    k, n = TP4_SITE_SHAPES[site]
    draws = 8
    planned = _budget_stats(TP4_PLAN[site][m - 1], k, n, m, draws, False)
    builtin = [
        _budget_stats(kernel, k, n, m, draws, False)
        for kernel in sorted({TP4_AUTOTUNED[site][m - 1], "cublas"})
    ]
    print(
        f"[budget tp4] {site} M={m}: planned {TP4_PLAN[site][m - 1]} {planned} built-in {builtin}"
    )
    assert planned[0] <= 1.25 * max(b[0] for b in builtin), ("B2", planned, builtin)
    assert planned[1] <= max(b[1] for b in builtin), ("B1'", planned, builtin)
    assert planned[2] <= 1.25 * max(b[2] for b in builtin), ("B3", planned, builtin)


@pytest.mark.parametrize("rows", (1, 4, 8))
@pytest.mark.parametrize("site", tuple(TP4_SITE_SHAPES))
def test_tp4_repeat_pdl_and_graph_replay_give_the_same_bits(site, rows) -> None:
    k, n = TP4_SITE_SHAPES[site]
    kernel = TP4_PLAN[site][rows - 1]
    x, w = _operands(rows, k, n, seed=31)
    first = op(x, w, kernel, enable_pdl=True)
    assert torch.equal(first, op(x, w, kernel, enable_pdl=True))
    assert torch.equal(first, op(x, w, kernel, enable_pdl=False))
    static_x = x.clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        op(static_x, w, kernel)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = op(static_x, w, kernel)
    for seed in range(3):
        fresh, _ = _operands(rows, k, n, seed=60 + seed)
        static_x.copy_(fresh)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(captured, op(fresh, w, kernel))


def test_cublas_is_the_builtin_cublas_runner() -> None:
    k, n = O_PROJ_SHAPE
    x, w = _operands(8, k, n, seed=3)
    runner = low_m_gemm._CuBLASGemmRunner(has_bias=False)
    assert torch.equal(op(x, w, "cublas"), runner([x, w.t()], tactic=runner._TACTIC))


@pytest.mark.parametrize("rows", (1, 8))
@pytest.mark.parametrize("site", tuple(SITE_SHAPES))
def test_repeat_pdl_and_graph_replay_give_the_same_bits(site, rows) -> None:
    k, n = SITE_SHAPES[site]
    kernel = PLAN[site][rows - 1]
    x, w = _operands(rows, k, n, seed=21)
    first = op(x, w, kernel, enable_pdl=True)
    assert torch.equal(first, op(x, w, kernel, enable_pdl=True))
    assert torch.equal(first, op(x, w, kernel, enable_pdl=False))
    static_x = x.clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        op(static_x, w, kernel)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = op(static_x, w, kernel)
    for seed in range(3):
        fresh, _ = _operands(rows, k, n, seed=50 + seed)
        static_x.copy_(fresh)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(captured, op(fresh, w, kernel))


def test_malformed_operands_raise_rather_than_return() -> None:
    k, n = QKV_SHAPE
    x, w = _operands(8, k, n, seed=1)
    with pytest.raises(ValueError):
        op(x, w, "splitk:64,8,4")  # not a tactic
    with pytest.raises(ValueError):
        op(x, w, "nvjet")
    with pytest.raises(ValueError):
        op(x, w, "direct:256")  # not a tactic
    with pytest.raises(ValueError):
        op(x[:, :1000].contiguous(), w[:, :1000].contiguous(), "splitk:64,8,4,12")  # K % 128
    with pytest.raises(ValueError):
        op(x[:, :1024].contiguous(), w[:, :1024].contiguous(), "direct:256,1")  # K % 2048


def test_is_valid_refuses_what_the_kernels_would_answer_wrongly() -> None:
    k, n = UNPOOL_SHAPE
    x, w = _operands(8, k, n, seed=1)
    op.is_valid(x, w, "cublas")
    for bad in (
        (x.float(), w.float(), "cublas"),  # torch.mm would return an FP32 product
        (torch.cat([x, x]), w, "cublas"),  # 16 rows: outside the certified rows
        (x, w.t().contiguous(), "cublas"),  # the weight's transpose
    ):
        with pytest.raises(AssertionError):
            op.is_valid(*bad)


# --- the dense_gemms wiring on real modules ------------------------------------------------


def _linear(n, k, seed) -> Linear:
    """A built-in ``Linear`` as far as class and the dispatch go, holding a real BF16 weight."""
    linear = Linear.__new__(Linear)
    torch.nn.Module.__init__(linear)
    _, w = _operands(1, k, n, seed)
    linear.weight = torch.nn.Parameter(w, requires_grad=False)
    linear.register_parameter("bias", None)
    linear.use_custom_cublas_mm = False
    return linear


def _real_linear(n, k, seed) -> Linear:
    """The built-in ``Linear`` as its constructor builds it, holding a seeded BF16 weight."""
    linear = Linear(k, n, bias=False, dtype=torch.bfloat16).to(DEV)
    _, w = _operands(1, k, n, seed)
    with torch.no_grad():
        linear.weight.copy_(w)
    return linear


def _model(num_layers, tp_rank=3, tp_size=8, make=_linear):
    """Layers holding the four projections of a tep8 rank (pooler/unpooler replicated, as built)."""
    layers = []
    for i in range(num_layers):
        layer = torch.nn.Module()
        layer.self_attn = torch.nn.Module()
        layer.self_attn.qkv_proj = make(QKV_SHAPE[1], QKV_SHAPE[0], 10 * i + 1)
        layer.self_attn.o_proj = make(O_PROJ_SHAPE[1], O_PROJ_SHAPE[0], 10 * i + 2)
        layer.feed_forward = torch.nn.Module()
        layer.feed_forward.pooler = make(POOL_SHAPE[1] * tp_size, POOL_SHAPE[0], 10 * i + 3)
        layer.feed_forward.unpooler = make(UNPOOL_SHAPE[1] * tp_size, UNPOOL_SHAPE[0], 10 * i + 4)
        layer.feed_forward.mapping = types.SimpleNamespace(tp_size=tp_size, tp_rank=tp_rank)
        # The built-in model gives every MoE these workspaces when it shards the projections.
        layer.feed_forward._pool_gather_workspace = object()
        layer.feed_forward._unpool_gather_workspace = object()
        layers.append(layer)
    return types.SimpleNamespace(layers=layers)


def _slice(linear, tp_rank=3, tp_size=8):
    cols = linear.weight.shape[0] // tp_size
    return linear.weight[tp_rank * cols : (tp_rank + 1) * cols]


@torch.inference_mode()
def _step(model, inputs, prep_stream):
    """One emulated decode step's projections, in the order and on the streams the model uses.

    Under inference mode, as the engine runs a forward: with autograd on, the
    built-in dispatch and this feature's both leave the GEMM to the plain path.
    """
    outs = []
    main = torch.cuda.current_stream()
    for layer, (xq, xo, xp, xu) in zip(model.layers, inputs):
        attention, moe = layer.self_attn, layer.feed_forward
        outs.append(
            low_m_gemm.apply_low_m_gemm(attention.qkv_proj, xq, attention.qkv_proj.weight, None)
        )
        outs.append(
            low_m_gemm.apply_low_m_gemm(attention.o_proj, xo, attention.o_proj.weight, None)
        )
        prep_stream.wait_stream(main)
        with torch.cuda.stream(prep_stream):
            outs.append(
                low_m_gemm.apply_low_m_gemm(
                    moe.pooler, xp, _slice(moe.pooler), None, force_active=True
                )
            )
        main.wait_stream(prep_stream)
        outs.append(
            low_m_gemm.apply_low_m_gemm(
                moe.unpooler, xu, _slice(moe.unpooler), None, force_active=True
            )
        )
    return outs


def _expected(model, inputs):
    outs = []
    for layer, (xq, xo, xp, xu) in zip(model.layers, inputs):
        attention, moe = layer.self_attn, layer.feed_forward
        r = int(xq.shape[0]) - 1
        outs.append(op(xq, attention.qkv_proj.weight, PLAN["qkv"][r]))
        outs.append(op(xo, attention.o_proj.weight, PLAN["o_proj"][r]))
        outs.append(op(xp, _slice(moe.pooler), PLAN["pool"][r]))
        outs.append(op(xu, _slice(moe.unpooler), PLAN["unpool"][r]))
    return outs


def _inputs(num_layers, rows, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    return [
        tuple(
            torch.randn(rows, shape[0], generator=g, device=DEV).to(torch.bfloat16)
            for shape in (QKV_SHAPE, O_PROJ_SHAPE, POOL_SHAPE, UNPOOL_SHAPE)
        )
        for _ in range(num_layers)
    ]


@pytest.mark.parametrize("rows", (8, 2))
@pytest.mark.parametrize("prefetch", (False, True), ids=("dense_gemms", "gemm_prefetch"))
def test_the_installed_projections_run_the_entry_eagerly_and_in_a_graph(
    prefetch, rows, monkeypatch
):
    """Through ``apply_low_m_gemm`` -- what the built-in Linear and the sharded projections call --
    an installed model's four projections give the plan's kernel's bits for the row count,
    before and after the engine attaches its own dispatcher, eagerly and under CUDA-graph
    capture with prefetch forks."""
    monkeypatch.setattr(low_m_gemm, "LOW_M_GEMM_ACTIVE", True)
    model = _model(2)
    plan = dense_gemms.install(model, prefetch=prefetch)
    engine = low_m_gemm.LowMGemmDispatcher()
    engine.attach(
        torch.nn.ModuleList(
            [torch.nn.ModuleList([layer.self_attn, layer.feed_forward]) for layer in model.layers]
        ),
        force=True,
    )
    for layer in model.layers:
        for linear in (
            layer.self_attn.qkv_proj,
            layer.self_attn.o_proj,
            layer.feed_forward.pooler,
            layer.feed_forward.unpooler,
        ):
            assert isinstance(linear._low_m_gemm_dispatcher, dense_gemms.DenseGemmDispatcher)
            assert linear._low_m_gemm_dispatcher.builtin is engine
    prep = torch.cuda.Stream()
    inputs = _inputs(2, rows, seed=7)
    eager = _step(model, inputs, prep)
    torch.cuda.synchronize()
    assert not plan.pending
    for got, want in zip(eager, _expected(model, inputs)):
        assert torch.equal(got, want)

    static = _inputs(2, rows, seed=8)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _step(model, static, prep)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = _step(model, static, prep)
    assert not plan.pending
    for seed in (9, 10):
        fresh = _inputs(2, rows, seed=seed)
        for s_layer, f_layer in zip(static, fresh):
            for s, f in zip(s_layer, f_layer):
                s.copy_(f)
        graph.replay()
        torch.cuda.synchronize()
        for got, want in zip(captured, _expected(model, fresh)):
            assert torch.equal(got, want)


def test_calls_the_entry_does_not_serve_go_to_the_engine_dispatcher(monkeypatch):
    monkeypatch.setattr(low_m_gemm, "LOW_M_GEMM_ACTIVE", True)
    model = _model(1)
    dense_gemms.install(model, prefetch=False)
    qkv = model.layers[0].self_attn.qkv_proj
    seen = []

    class _Engine:
        def apply(self, module, input_tensor, weight, bias, force_active=False):
            seen.append((module, input_tensor.shape, force_active))
            return "builtin"

    qkv._low_m_gemm_dispatcher = _Engine()
    x8 = torch.randn(8, QKV_SHAPE[0], device=DEV).to(torch.bfloat16)
    # With autograd on (outside the engine's inference mode).
    assert low_m_gemm.apply_low_m_gemm(qkv, x8, qkv.weight, None) == "builtin"
    with torch.inference_mode():
        x9 = torch.randn(9, QKV_SHAPE[0], device=DEV).to(torch.bfloat16)
        assert low_m_gemm.apply_low_m_gemm(qkv, x9, qkv.weight, None) == "builtin"
        assert (
            low_m_gemm.apply_low_m_gemm(
                qkv, x8, qkv.weight, torch.zeros(QKV_SHAPE[1], device=DEV).to(torch.bfloat16)
            )
            == "builtin"
        )
        assert low_m_gemm.apply_low_m_gemm(qkv, x8.float(), qkv.weight, None) == "builtin"
        assert [s[1][0] for s in seen] == [8, 9, 8, 8]
        out = low_m_gemm.apply_low_m_gemm(qkv, x8, qkv.weight, None)
    assert torch.equal(out, op(x8, qkv.weight, PLAN["qkv"][7]))
    assert len(seen) == 4


def test_a_decode_call_that_does_not_fit_says_so(monkeypatch):
    """A BF16 inference call of at most 8 rows that the entry cannot serve goes to the engine,
    whose kernel is chosen per rank, and the dispatcher warns (once per site, by key); calls the
    engine serves anyway (more rows, FP32, autograd on) do not warn."""
    monkeypatch.setattr(low_m_gemm, "LOW_M_GEMM_ACTIVE", True)
    model = _model(1)
    dense_gemms.install(model, prefetch=False)
    qkv = model.layers[0].self_attn.qkv_proj
    warnings = []
    monkeypatch.setattr(
        dense_gemms.logger, "warning_once", lambda *msg, key: warnings.append((key, msg))
    )

    class _Engine:
        def apply(self, module, input_tensor, weight, bias, force_active=False):
            return "builtin"

    qkv._low_m_gemm_dispatcher = _Engine()
    strided = torch.randn(8, 2 * QKV_SHAPE[0], device=DEV).to(torch.bfloat16)[:, ::2]
    with torch.inference_mode():
        for _ in range(2):
            assert low_m_gemm.apply_low_m_gemm(qkv, strided, qkv.weight, None) == "builtin"
        rows9 = torch.randn(9, QKV_SHAPE[0], device=DEV).to(torch.bfloat16)
        assert low_m_gemm.apply_low_m_gemm(qkv, rows9, qkv.weight, None) == "builtin"
        assert low_m_gemm.apply_low_m_gemm(qkv, strided.float(), qkv.weight, None) == "builtin"
    assert low_m_gemm.apply_low_m_gemm(qkv, strided, qkv.weight, None) == "builtin"  # autograd on
    assert [key for key, _ in warnings] == ["musespark_open.dense_gemms.decline_qkv"] * 2
    assert "contiguous=False" in warnings[0][1][0]


# --- real Linear modules, CUDA graphs and the prefetch fork --------------------------------------


@torch.inference_mode()
def _forward_step(model, plan, inputs, prep_stream, between=None):
    """One emulated decode step through the built-in ``Linear.forward`` for QKV and o_proj (what
    the attention calls) and ``apply_low_m_gemm`` on the rank's slices for the pooler (on the
    preparation stream) and the unpooler, ending with the join the target makes at the end of
    its forward. ``between`` runs on o_proj's input between the two attention projections."""
    outs = []
    main = torch.cuda.current_stream()
    for layer, (xq, xo, xp, xu) in zip(model.layers, inputs):
        attention, moe = layer.self_attn, layer.feed_forward
        outs.append(attention.qkv_proj(xq))
        outs.append(attention.o_proj(xo if between is None else between(xo)))
        prep_stream.wait_stream(main)
        with torch.cuda.stream(prep_stream):
            outs.append(
                low_m_gemm.apply_low_m_gemm(
                    moe.pooler, xp, _slice(moe.pooler), None, force_active=True
                )
            )
        main.wait_stream(prep_stream)
        outs.append(
            low_m_gemm.apply_low_m_gemm(
                moe.unpooler, xu, _slice(moe.unpooler), None, force_active=True
            )
        )
    plan.join()
    return outs


def _refill(static, fresh):
    for s_layer, f_layer in zip(static, fresh):
        for s, f in zip(s_layer, f_layer):
            s.copy_(f)


def test_real_linear_modules_replay_graphs_of_every_row_count(monkeypatch):
    """Built-in ``Linear`` modules as constructed, the engine's dispatcher attached after
    adoption, with the prefetch: one CUDA graph per row count 1..8 on one plan, replayed in shuffled order
    with fresh inputs and eager steps of other row counts in between, all bit-equal to the
    plan's kernels."""
    monkeypatch.setattr(low_m_gemm, "LOW_M_GEMM_ACTIVE", True)
    num_layers = 3
    model = _model(num_layers, make=_real_linear)
    plan = dense_gemms.install(model, prefetch=True)
    engine = low_m_gemm.prepare_low_m_gemm(
        torch.nn.ModuleList(
            [torch.nn.ModuleList([layer.self_attn, layer.feed_forward]) for layer in model.layers]
        )
    )
    for layer in model.layers:
        assert type(layer.self_attn.qkv_proj) is dense_gemms.MuseSparkOpenTepDenseLinear
        assert layer.self_attn.qkv_proj._low_m_gemm_dispatcher.builtin is engine
    prep, capture_stream = torch.cuda.Stream(), torch.cuda.Stream()
    graphs, static_inputs, captured, pool = {}, {}, {}, None
    for rows in range(1, MAX_ROWS + 1):
        eager_in = _inputs(num_layers, rows, seed=100 + rows)
        eager = _forward_step(model, plan, eager_in, prep)
        for got, want in zip(eager, _expected(model, eager_in)):
            assert torch.equal(got, want), ("eager", rows)
        static_inputs[rows] = _inputs(num_layers, rows, seed=200 + rows)
        capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(capture_stream):
            _forward_step(model, plan, static_inputs[rows], prep)
        torch.cuda.synchronize()
        graphs[rows] = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graphs[rows], stream=capture_stream, pool=pool):
            captured[rows] = _forward_step(model, plan, static_inputs[rows], prep)
        pool = graphs[rows].pool()
        assert not plan.pending
    order = list(range(1, MAX_ROWS + 1)) * 2
    random.Random(5).shuffle(order)
    for i, rows in enumerate(order):
        fresh = _inputs(num_layers, rows, seed=1000 + i)
        _refill(static_inputs[rows], fresh)
        graphs[rows].replay()
        other = _inputs(num_layers, rows % MAX_ROWS + 1, seed=5000 + i)
        eager = _forward_step(model, plan, other, prep)
        torch.cuda.synchronize()
        for got, want in zip(captured[rows], _expected(model, fresh)):
            assert torch.equal(got, want), ("replay", rows, i)
        for got, want in zip(eager, _expected(model, other)):
            assert torch.equal(got, want), ("eager between replays", rows, i)
    assert plan.dropped == 0


def _one_layer_gemm_prefetch(monkeypatch):
    monkeypatch.setattr(low_m_gemm, "LOW_M_GEMM_ACTIVE", True)
    model = _model(1, make=_real_linear)
    plan = dense_gemms.install(model, prefetch=True)
    return plan, model.layers[0].self_attn.qkv_proj


def _capture_and_replay_qkv(plan, qkv, static):
    """Capture a QKV call, whose fork has no consumer in the capture, and the end-of-forward
    join; replay with fresh rows and compare with the plan's kernel."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.inference_mode(), torch.cuda.graph(graph, stream=stream):
        captured = qkv(static)
        capture_id = dense_gemms.current_capture_id()
        assert capture_id is not None
        assert plan.pending and plan._fork_capture == capture_id
        plan.join()
    assert not plan.pending
    rows = int(static.shape[0])
    for seed in (61, 62):
        fresh, _ = _operands(rows, QKV_SHAPE[0], QKV_SHAPE[1], seed=seed)
        static.copy_(fresh)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(captured, op(fresh, qkv.weight, PLAN["qkv"][rows - 1]))


def test_a_fork_left_by_an_eager_step_is_dropped_by_the_next_capture(monkeypatch):
    """Review r2 case (a): an eager forward ended between a fork and its consumer. The first
    dispatcher inside the next capture must not wait on that eager work."""
    plan, qkv = _one_layer_gemm_prefetch(monkeypatch)
    x, _ = _operands(8, QKV_SHAPE[0], QKV_SHAPE[1], seed=60)
    with torch.inference_mode():
        qkv(x)
    assert plan.pending and plan._fork_capture is None
    torch.cuda.synchronize()
    _capture_and_replay_qkv(plan, qkv, x.clone())
    assert plan.dropped == 1


def test_a_fork_inside_a_capture_is_joined_by_the_end_of_the_forward(monkeypatch):
    """Review r2 case (b): a fork whose consumer does not run inside the capture (a sibling
    feature's GEMM, a shorter forward) is joined by the target's end-of-forward join."""
    plan, qkv = _one_layer_gemm_prefetch(monkeypatch)
    x, _ = _operands(8, QKV_SHAPE[0], QKV_SHAPE[1], seed=60)
    with torch.inference_mode():
        qkv(x)
    plan.join()
    torch.cuda.synchronize()
    assert not plan.pending and plan.dropped == 0
    _capture_and_replay_qkv(plan, qkv, x.clone())
    assert plan.dropped == 0
    with torch.inference_mode():  # eager again afterwards
        assert torch.equal(qkv(x), op(x, qkv.weight, PLAN["qkv"][7]))
    plan.join()
    torch.cuda.synchronize()


@pytest.mark.parametrize("rows", (8, 3))
def test_gemm_prefetch_in_a_breakable_graph(monkeypatch, caplog, rows):
    """A prefill graph breaks at the attention op, between QKV and o_proj. The segment end joins
    QKV's fork; o_proj, in the next segment, drops it instead of waiting on a stream outside its
    capture. Replays are bit-equal to the plan's kernels."""
    from tensorrt_llm._torch.pyexecutor.breakable_cuda_graph import (
        BreakableCUDAGraph,
        BreakableCUDAGraphCapture,
        eager_on_graph,
    )

    @eager_on_graph
    def attention_break(t):
        return t * 1

    monkeypatch.setattr(low_m_gemm, "LOW_M_GEMM_ACTIVE", True)
    num_layers = 2
    model = _model(num_layers, make=_real_linear)
    plan = dense_gemms.install(model, prefetch=True)
    prep = torch.cuda.Stream()
    static = _inputs(num_layers, rows, seed=300 + rows)
    _forward_step(model, plan, static, prep, between=attention_break)
    torch.cuda.synchronize()
    graph = BreakableCUDAGraph()
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with caplog.at_level(logging.WARNING):
        with BreakableCUDAGraphCapture(graph, stream=capture_stream):
            captured = _forward_step(model, plan, static, prep, between=attention_break)
    torch.cuda.synchronize()
    assert graph.num_breaks == num_layers and not plan.pending
    assert plan.dropped == num_layers
    assert not [r for r in caplog.records if "Dropping a wait" in r.getMessage()]
    for seed in (401, 402):
        fresh = _inputs(num_layers, rows, seed=seed)
        _refill(static, fresh)
        with torch.inference_mode():  # the break's replay copies into inference tensors
            graph.replay()
        torch.cuda.synchronize()
        for got, want in zip(captured, _expected(model, fresh)):
            assert torch.equal(got, want)


def _kernel_edges(graph):
    """``(from node, to node, programmatic)`` of every edge of a kept graph, and each node's
    kernel name."""
    from cuda.bindings import driver

    def check(result):
        status, *values = result
        assert status == driver.CUresult.CUDA_SUCCESS, status
        return values

    handle = driver.CUgraph(graph.raw_cuda_graph())
    # CUDA 13's cuGraphGetEdges returns the edge data (CUDA 12's _v2).
    get = getattr(driver, "cuGraphGetEdges_v2", None) or driver.cuGraphGetEdges
    total = int(check(get(handle, 0))[-1])
    sources, targets, data, _ = check(get(handle, total))
    programmatic = int(driver.CUgraphDependencyType.CU_GRAPH_DEPENDENCY_TYPE_PROGRAMMATIC)
    edges = [
        (int(a), int(b), int(e.type) == programmatic) for a, b, e in zip(sources, targets, data)
    ]
    names = {}
    for node in list(sources) + list(targets):
        (kind,) = check(driver.cuGraphNodeGetType(node))
        if kind != driver.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL:
            names[int(node)] = f"<{kind.name}>"
            continue
        (params,) = check(driver.cuGraphKernelNodeGetParams(node))
        status, raw = driver.cuFuncGetName(params.func)
        if status != driver.CUresult.CUDA_SUCCESS:
            status, raw = driver.cuKernelGetName(params.kern)
        names[int(node)] = raw.decode() if isinstance(raw, bytes) else str(raw)
    return edges, names


def test_a_named_fork_is_a_programmatic_predecessor_of_its_join_roles_gemm(monkeypatch):
    """GEMM A (layer 0's pooler), a fork of an L2 prefetch of layer 1's QKV weight, GEMM B
    (layer 0's unpooler, another role), GEMM C (layer 1's QKV, the fork's join role), all
    split-K with programmatic dependent launch. In the captured graph the prefetch kernel P
    has one edge in, a full one from A, and one out, a programmatic one to C; C's
    predecessors are P and B, both programmatic; B does not wait for P. The fork's stream
    is not the capturing stream, and every output is the plan's kernel's bits, eagerly and
    in replays."""
    monkeypatch.setattr(low_m_gemm, "LOW_M_GEMM_ACTIVE", True)
    monkeypatch.setenv("TRTLLM_ENABLE_PDL", "1")
    model = _model(2, make=_real_linear)
    plan = dense_gemms.install(model, prefetch=False)
    assert plan.enable_pdl and PLAN["qkv"][7] == PLAN["pool"][7] == PLAN["unpool"][7]
    fork = plan.add_fork("test_qkv", "qkv")
    first, second = model.layers
    weight = second.self_attn.qkv_proj.weight

    def prefetch():
        prefetch_op(weight, fork.sink)

    @torch.inference_mode()
    def step(xp, xu, xq):
        moe = first.feed_forward
        a = low_m_gemm.apply_low_m_gemm(moe.pooler, xp, _slice(moe.pooler), None, True)
        assert fork.fork(prefetch)
        b = low_m_gemm.apply_low_m_gemm(moe.unpooler, xu, _slice(moe.unpooler), None, True)
        assert fork.pending, "only the QKV dispatcher joins the fork"
        c = second.self_attn.qkv_proj(xq)
        assert not fork.pending
        return a, b, c

    def inputs(seed):
        g = torch.Generator(device=DEV).manual_seed(seed)
        return [
            torch.randn(8, shape[0], generator=g, device=DEV).to(torch.bfloat16)
            for shape in (POOL_SHAPE, UNPOOL_SHAPE, QKV_SHAPE)
        ]

    def expected(xp, xu, xq):
        return (
            op(xp, _slice(first.feed_forward.pooler), PLAN["pool"][7]),
            op(xu, _slice(first.feed_forward.unpooler), PLAN["unpool"][7]),
            op(xq, weight, PLAN["qkv"][7]),
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    static = inputs(1)
    with torch.cuda.stream(stream):
        eager = step(*static)
    torch.cuda.synchronize()
    for got, want in zip(eager, expected(*static)):
        assert torch.equal(got, want)
    assert fork.stream is not None and fork.stream.cuda_stream != stream.cuda_stream
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph, stream=stream):
        captured = step(*static)
    assert (fork.forks, fork.joins, fork.dropped, fork.skipped) == (2, 2, 0, 0)

    edges, names = _kernel_edges(graph)
    prefetches = [node for node, name in names.items() if "l2_prefetch" in name]
    assert len(prefetches) == 1, names
    p = prefetches[0]
    into_p = [(a, prog) for a, b, prog in edges if b == p]
    out_of_p = [(b, prog) for a, b, prog in edges if a == p]
    assert len(into_p) == 1 and not into_p[0][1], "P follows A by a full edge"
    assert len(out_of_p) == 1 and out_of_p[0][1], "P precedes C by a programmatic edge"
    a_node, c_node = into_p[0][0], out_of_p[0][0]
    into_c = {a: prog for a, b, prog in edges if b == c_node}
    assert len(into_c) == 2 and p in into_c and all(into_c.values()), (into_c, names)
    (b_node,) = [node for node in into_c if node != p]
    assert {a: prog for a, b, prog in edges if b == b_node} == {a_node: True}, "A -> B, not P"
    for node in (a_node, b_node, c_node):
        assert "l2_prefetch" not in names[node]

    for seed in (2, 3):
        fresh = inputs(seed)
        for s, f in zip(static, fresh):
            s.copy_(f)
        graph.replay()
        torch.cuda.synchronize()
        for got, want in zip(captured, expected(*fresh)):
            assert torch.equal(got, want), seed


# --- microbenchmark (MODELING_V2_TIMING=1) -----------------------------------------------


@triton.jit
def _globaltimer():
    z = tl.full([1], 0, tl.int32)
    t = tl.inline_asm_elementwise(
        "mov.u64 $0, %globaltimer;", "=l,r", [z], dtype=tl.int64, is_pure=False, pack=1
    )
    return tl.max(t, 0)


@triton.jit
def _spin_then_copy_kernel(src, dst, n_elem, spin_ns, BLOCK: tl.constexpr):
    """A latency-bound predecessor that produces x and releases its dependents at exit, like the model's."""
    t0 = _globaltimer()
    t = t0
    while t - t0 < spin_ns:
        t = _globaltimer()
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(dst + offs, tl.load(src + offs, mask=offs < n_elem), mask=offs < n_elem)


def _spin_copy(src, dst, spin_ns):
    n = src.numel()
    _spin_then_copy_kernel[(triton.cdiv(n, 1024),)](src, dst, n, spin_ns, BLOCK=1024, num_warps=4)


def _graph_us(fn, reps=20):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        fn()
    for _ in range(3):
        graph.replay()
    samples = []
    for _ in range(5):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(reps):
            graph.replay()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / reps)
    return sorted(samples)[2]


_BENCH_SITES = ("qkv", "o_proj", "pool", "unpool")
#: Spin before each projection: unpool gather + boundary; prep, KV append, attention and
#: output gate; all-reduce + boundary; router, experts and finalize all-reduce.
_BENCH_SPIN_NS = {"qkv": 6000, "o_proj": 15000, "pool": 8000, "unpool": 25000}


def _bench_rows(rows, num_layers):
    """GEMM microseconds per layer of each variant: (step graph - spins-only graph) / layers."""
    model = _model(num_layers)
    weights = [
        (
            layer.self_attn.qkv_proj.weight,
            layer.self_attn.o_proj.weight,
            _slice(layer.feed_forward.pooler),
            _slice(layer.feed_forward.unpooler),
        )
        for layer in model.layers
    ]
    src = _inputs(1, rows, seed=1)[0]
    bufs = [torch.empty_like(s) for s in src]

    def spins_only():
        for _ in weights:
            for index, site in enumerate(_BENCH_SITES):
                _spin_copy(src[index], bufs[index], _BENCH_SPIN_NS[site])

    def builtin_step(kernels):
        for layer_weights in weights:
            for index, site in enumerate(_BENCH_SITES):
                _spin_copy(src[index], bufs[index], _BENCH_SPIN_NS[site])
                op(bufs[index], layer_weights[index], kernels[site])

    @torch.inference_mode()
    def installed_step(projections):
        served = 0
        for modules in projections:
            for index, site in enumerate(_BENCH_SITES):
                _spin_copy(src[index], bufs[index], _BENCH_SPIN_NS[site])
                linear = modules[index]
                weight = linear.weight if index < 2 else _slice(linear)
                out = low_m_gemm.apply_low_m_gemm(
                    linear, bufs[index], weight, None, force_active=True
                )
                served += out is not None
        # A call left to the plain path would time nothing here.
        assert served == len(projections) * len(_BENCH_SITES)

    spins = _graph_us(spins_only)
    results = {}
    ranks_of = {}
    for rank in range(8):
        kernels = tuple(S0_KERNELS[site][rows][rank] for site in _BENCH_SITES)
        ranks_of.setdefault(kernels, []).append(rank)
    for kernels, ranks in ranks_of.items():
        by_site = dict(zip(_BENCH_SITES, kernels))
        name = "built-in rank " + ",".join(map(str, ranks))
        results[name] = _graph_us(lambda k=by_site: builtin_step(k))
    for name, prefetch, overrides in _BENCH_INSTALLED:
        installed = _model(num_layers)
        dense_gemms.install(installed, prefetch=prefetch, kernels=_kernels(overrides))
        projections = [
            (
                layer.self_attn.qkv_proj,
                layer.self_attn.o_proj,
                layer.feed_forward.pooler,
                layer.feed_forward.unpooler,
            )
            for layer in installed.layers
        ]
        results[name] = _graph_us(lambda p=projections: installed_step(p))
    return {name: (us - spins) / num_layers for name, us in results.items()}


def _kernels(overrides=()):
    """The default kernel table with ``(site, rows, kernel)`` overrides."""
    kernels = {site: list(by_rows) for site, by_rows in dense_gemms.default_kernels().items()}
    for site, rows, kernel in overrides:
        for row in rows:
            kernels[site][row - 1] = kernel
    return {site: tuple(by_rows) for site, by_rows in kernels.items()}


_ALL_ROWS = range(1, 9)

#: Installed variants: name, prefetch, kernel overrides.
_BENCH_INSTALLED = (
    ("dense_gemms", False, ()),
    ("gemm_prefetch", True, ()),
    (
        "dense_gemms split-K for qkv/pool/unpool",
        False,
        (
            ("qkv", _ALL_ROWS, "splitk:64,8,4,12"),
            ("o_proj", _ALL_ROWS, "cublas"),
            ("pool", _ALL_ROWS, "splitk:64,8,4,12"),
            ("unpool", _ALL_ROWS, "splitk:64,8,4,12"),
        ),
    ),
    ("dense_gemms direct unpool from 3 rows", False, (("unpool", range(3, 9), "direct:64,2"),)),
    ("dense_gemms direct pool from 3 rows", False, (("pool", range(3, 9), "direct:128,2"),)),
)


@pytest.mark.skipif(
    os.environ.get("MODELING_V2_TIMING") != "1", reason="MODELING_V2_TIMING=1 runs it"
)
def test_microbenchmark_against_the_builtin_kernels(monkeypatch) -> None:
    """A 62-layer emulated decode step of the four projections behind latency-bound predecessors.

    Per layer: spin 6 us (unpool gather + boundary) -> QKV -> spin 15 us (prep,
    KV append, attention, output gate) -> o_proj -> spin 8 us (all-reduce,
    boundary) -> pool -> spin 25 us (router, experts, finalize all-reduce) ->
    unpool. Each spin kernel produces the next GEMM's x and releases it at exit.
    Variants: the built-in kernels of each group of ranks that runs the same ones
    (their autotuned winners, called directly), and the installed dispatchers:
    ``dense_gemms``, with the prefetch, and with one or more sites' kernels
    overridden, to show each choice.
    The pooler runs on the main stream here (the model overlaps it with the router).
    """
    monkeypatch.setattr(low_m_gemm, "LOW_M_GEMM_ACTIVE", True)
    num_layers = 62
    for rows in (8, 1):
        per_layer = _bench_rows(rows, num_layers)
        slowest = max(us for name, us in per_layer.items() if name.startswith("built-in"))
        for name, us in per_layer.items():
            gain = us - slowest
            print(
                f"[bench] rows={rows} {name:32s} GEMM {us:6.2f} us/layer, vs the slowest built-in rank "
                f"{gain:+6.2f} us/layer = {gain * num_layers / 1000:+.3f} ms/step"
            )
        torch.cuda.empty_cache()
