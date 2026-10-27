# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the low_m_bf16_mm entry.

Every cell names the branch `plan` must pick, and a host-only check pins the
branch of the table's other shapes and M buckets, so a table edit that moves a
shape to another kernel fails here before it can change a number silently.
Beyond the cells: CUDA-graph capture and replay after warmup (the way the
decode forward runs it), and the guard against a padded `mat_b`, which the
cuBLAS branch would read as dense.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.gemm.low_m_bf16_mm import (  # noqa: E501
    low_m_bf16_mm as op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.gemm.low_m_bf16_mm import (  # noqa: E501
    plan,
)

__extra_import_path__ = ["../../../.."]
from _validating import (  # noqa: E402,F401 — path above; fixture by name
    ULP,
    assert_within_ulp,
    certifying,
    validating,
)

assert torch.cuda.is_available(), "low_m_bf16_mm requires a CUDA device"


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """Accumulation order only: 4 bf16 ulp element-wise against the row scale,
    2 ulp of relative RMS -- the band `cublas_mm` uses, for the same reason.
    Every branch sums the same bf16 products in fp32 and rounds once; what is
    left between kernel and fp32 reference is the final bf16 rounding (half an
    ulp) plus the summation order."""
    assert_within_ulp(out, ref, element_ulp=4.0, rms_ulp=2.0, ulp=ULP[torch.bfloat16])


def _build(m, n, k, seed):
    """`mat_b` is a `[N, K]` weight's `.t()`, as a target stores its projections."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    mat_a = torch.randn(m, k, generator=g, device="cuda").to(torch.bfloat16)
    weight = (torch.randn(n, k, generator=g, device="cuda") * k**-0.5).to(torch.bfloat16)
    return mat_a, weight.t()


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811
    certifying(op)
    s = cell.spec
    assert plan(s["m"], s["n"], s["k"]).kind == s["kind"], plan(s["m"], s["n"], s["k"])
    mat_a, mat_b = _build(s["m"], s["n"], s["k"], seed=zlib.crc32(cell.why.encode()) % 2**31)
    out = op(mat_a, mat_b, s["pdl"])
    assert out.shape == (s["m"], s["n"]) and out.dtype == torch.bfloat16 and out.is_contiguous()
    _compare(out, op.reference(mat_a, mat_b, s["pdl"]))


#: `(M, N, K, branch)` of the table's shapes and M buckets the cells do not drive:
#: what `plan` picks for each, held without a launch.
_PLAN_PINS = (
    (1, 10240, 2560, "direct"),
    (1, 48, 2560, "direct"),
    (32, 48, 2560, "direct"),
    (8, 2560, 6144, "splitk"),
    (2, 16480, 2560, "cublas"),
    (1, 2560, 6144, "direct"),
    (3, 12288, 2560, "splitk"),
    (8, 12288, 2560, "splitk"),
    (16, 12288, 2560, "cublas"),
    (16, 512, 2560, "direct"),
    (24, 512, 2560, "splitk"),
    (16, 640, 2560, "splitk"),
    (4, 1280, 2560, "direct"),
    (8, 1296, 2560, "splitk"),
    (1, 1, 2560, "direct"),
    (1, 16480, 2560, "direct"),
    (2, 14080, 2560, "direct"),
    (3, 14080, 2560, "splitk"),
    (8, 14080, 2560, "splitk"),
    (16, 14080, 2560, "cublas"),
    (1, 2560, 2560, "direct"),
    (4, 2560, 2560, "direct"),
    (8, 2560, 2560, "splitk"),
    (1, 2560, 640, "cublas"),
    (8192, 2560, 6144, "cublas"),
    (1, 1024, 2048, "cublas"),
)


def test_plan_pins_the_branch_of_the_other_table_shapes() -> None:
    picked = [(m, n, k, want, plan(m, n, k).kind) for m, n, k, want in _PLAN_PINS]
    wrong = [p for p in picked if p[3] != p[4]]
    assert not wrong, f"(M, N, K, pinned, picked): {wrong}"


def test_graph_replay_after_warmup() -> None:
    """Decode runs this under CUDA graphs, with PDL: compile in warmup, capture,
    then replay with new activations copied into the captured input. One call per
    branch, in a chain, so a FlashInfer kernel follows cuBLAS and vice versa."""
    pdl = True
    shapes = [(1, 10240, 2560), (8, 2560, 6144), (1, 2560, 640), (8, 12288, 2560)]
    assert [plan(*s).kind for s in shapes] == ["direct", "splitk", "cublas", "splitk"]
    ins = [_build(m, n, k, seed=i) for i, (m, n, k) in enumerate(shapes)]
    static = [a.clone() for a, _ in ins]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            [op(a, b, pdl) for a, (_, b) in zip(static, ins)]
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outs = [op(a, b, pdl) for a, (_, b) in zip(static, ins)]
    for trial in range(3):
        g = torch.Generator(device="cuda").manual_seed(100 + trial)
        for a in static:
            a.copy_(torch.randn(a.shape, generator=g, device="cuda").to(a.dtype))
        graph.replay()
        torch.cuda.synchronize()
        for a, (_, b), out in zip(static, ins, outs):
            _compare(out, op.reference(a, b))


def _padded(m, n, k):
    """A row-padded mat_a and a column-padded mat_b: dense along K, leading
    dimension 2K. Both are views a caller gets by slicing a wider tensor."""
    a = torch.randn(m, 2 * k, dtype=torch.bfloat16, device="cuda")[:, :k]
    w = (torch.randn(n, 2 * k, device="cuda") * k**-0.5).to(torch.bfloat16)[:, :k]
    return a, w.t()


def test_guard_refuses_a_padded_mat_b() -> None:
    """Why the guard exists: the cuBLAS branch ignores mat_b's leading dimension
    and silently computes a different product."""
    a, b = _padded(4, 640, 640)
    a = a.contiguous()
    assert plan(4, 640, 640).kind == "cublas" and b.stride() == (1, 1280)
    with pytest.raises(AssertionError):
        _compare(op(a, b), op.reference(a, b))
    with pytest.raises(AssertionError, match="column-major"):
        with validating(op):
            op(a, b)
