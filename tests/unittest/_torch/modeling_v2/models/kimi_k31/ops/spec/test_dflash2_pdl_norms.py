# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 DFlash2 early-release norms (CuTe DSL on FlashInfer's norm kernels).

Every certified cell is gated bit for bit against the entry's reference: FlashInfer's ``rmsnorm``
for the plain norm, and ``dflash2_grouped_conv`` followed by FlashInfer's ``fused_add_rmsnorm``
for the fused one (output and updated residual). The inputs are laid out as the helix16 drafter
produces them: the q_a norm reads a row-strided view of the [q_a | kv_a] GEMM's output, the delta
is one side's strided view of the kernel projection's output. The PDL launches are also driven
between the split-K GEMMs that feed and read them, captured in one CUDA graph.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.gemm.rubin_ull_bf16_low_m_gemm import (  # noqa: E501
    bf16_low_m_gemm,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.dflash2_conv_add_rmsnorm import (  # noqa: E501
    dflash2_conv_add_rmsnorm as fused_op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.dflash2_conv_add_rmsnorm import (  # noqa: E501
    dflash2_rmsnorm as norm_op,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() not in (100, 103, 107),
    reason="the Kimi K3.1 DFlash2 kernels are certified on SM10x",
)

HIDDEN, TAPS, GROUP, EPS = 7168, 2, 16, 1e-6


def _bitwise(out: torch.Tensor, ref: torch.Tensor, what: str) -> None:
    assert out.shape == ref.shape and out.dtype == ref.dtype, (what, out.shape, ref.shape)
    bad = int((out.contiguous().view(torch.int16) != ref.contiguous().view(torch.int16)).sum())
    assert bad == 0, f"{what}: {bad}/{out.numel()} elements differ from the reference"


def _norm_args(spec, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    rows, hidden, pitch = spec["rows"], spec["hidden"], spec["pitch"]
    bank = (3 * torch.randn(rows, pitch, generator=g, device="cuda")).bfloat16()
    weight = (1 + 0.1 * torch.randn(hidden, generator=g, device="cuda")).bfloat16()
    return bank[:, :hidden], weight, EPS, spec["pdl"]


def _fused_raw(spec, seed):
    """``(a, projection, base, residual, weight)`` as the drafter holds them."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    rows, groups = spec["batch"] * spec["width"], HIDDEN // GROUP
    a = torch.randn(rows, HIDDEN, generator=g, device="cuda")
    projection = 0.25 * torch.randn(rows, 2 * TAPS * groups, generator=g, device="cuda")
    base = 0.5 * torch.randn(2, TAPS, HIDDEN, generator=g, device="cuda")
    residual = 4 * torch.randn(rows, HIDDEN, generator=g, device="cuda")
    weight = 1 + 0.1 * torch.randn(HIDDEN, generator=g, device="cuda")
    if spec["fill"] == "nan_prev":
        width = spec["width"]
        a[2 * width : 3 * width] = float("nan")
        projection[2 * width : 3 * width] = float("nan")
    elif spec["fill"] == "zeros":
        keep = torch.rand(rows, HIDDEN, generator=g, device="cuda") < 0.5
        sign = torch.where(torch.rand(rows, HIDDEN, generator=g, device="cuda") < 0.5, 1.0, -1.0)
        a = torch.where(keep, a, 0.0 * sign)
        projection = -projection.abs()
        base = -base.abs()
    return tuple(t.bfloat16() for t in (a, projection, base, residual, weight))


def _fused_args(spec, a, projection, base, residual, weight):
    delta = projection.view(a.shape[0], 2, TAPS, HIDDEN // GROUP)[:, spec["side"]]
    return (
        a,
        delta,
        base[spec["side"]],
        residual,
        weight,
        EPS,
        spec["width"],
        GROUP,
        spec["pdl"],
        spec["early"],
        spec.get("ctas", 0),
    )


@pytest.mark.parametrize("cell", norm_op.CELLS, ids=[c.why[:48] for c in norm_op.CELLS])
def test_norm_cells(cell, certifying) -> None:  # noqa: F811
    certifying(norm_op)
    args = _norm_args(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    _bitwise(norm_op(*args), norm_op.reference(*args), "rmsnorm")


@pytest.mark.parametrize("cell", fused_op.CELLS, ids=[c.why[:48] for c in fused_op.CELLS])
def test_fused_cells(cell, certifying) -> None:  # noqa: F811
    certifying(fused_op)
    raw = _fused_raw(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    out, residual = fused_op(*_fused_args(cell.spec, *(t.clone() for t in raw)))
    ref_out, ref_residual = fused_op.reference(*_fused_args(cell.spec, *(t.clone() for t in raw)))
    _bitwise(residual, ref_residual, "residual")
    if cell.spec.get("ctas", 0):
        # The cluster sums the squares in another order: rstd differs in the last FP32 bits.
        ok = torch.isfinite(ref_out.float())
        diff = (out.float() - ref_out.float()).abs()[ok]
        assert torch.equal(torch.isfinite(out.float()), ok), "non-finite pattern differs"
        assert diff.max().item() <= 2**-6 * ref_out.float().abs()[ok].max().item(), diff.max()
        assert (diff == 0).float().mean().item() > 0.9, "most outputs should still be bitwise"
    else:
        _bitwise(out, ref_out, "output")
    if cell.spec["fill"] == "nan_prev":
        width = cell.spec["width"]
        assert torch.isfinite(out[3 * width : 4 * width].float()).all(), "NaN leaked"


@pytest.mark.skipif(get_sm_version() != 103, reason="the drafter's low-M GEMM tactics are SM103's")
def test_pdl_between_drafter_gemms() -> None:
    """The helix16 drafter's chains in one CUDA graph, all on PDL with the early release:
    o_proj-like GEMM -> conv + add + norm -> conv_proj GEMM, and the [q_a | kv_a] GEMM -> q_a norm
    (row-strided) -> q_b GEMM. Replayed on new inputs, every output is bitwise the eager
    reference chain (FlashInfer's norms, the separate convolution)."""
    spec = dict(batch=1, width=8, side=1, pdl=True, early=True, fill="randn")
    rows, tactic = 8, ("splitk", 64, 8, 4, 12)
    g = torch.Generator(device="cuda").manual_seed(9)
    a, projection, base, residual, weight = _fused_raw(spec, seed=10)
    w_in = (0.01 * torch.randn(HIDDEN, 7168, generator=g, device="cuda")).bfloat16()
    w_conv = (0.01 * torch.randn(1792, HIDDEN, generator=g, device="cuda")).bfloat16()
    w_qkv = (0.01 * torch.randn(2112, HIDDEN, generator=g, device="cuda")).bfloat16()
    w_qb = (0.01 * torch.randn(18432 // 16, 1536, generator=g, device="cuda")).bfloat16()
    q_norm = (1 + 0.1 * torch.randn(1536, generator=g, device="cuda")).bfloat16()
    x = torch.randn(rows, 7168, generator=g, device="cuda").bfloat16()
    a_buf = torch.empty(rows, HIDDEN, dtype=torch.bfloat16, device="cuda")
    coef = torch.empty(rows, 1792, dtype=torch.bfloat16, device="cuda")
    q_kv = torch.empty(rows, 2112, dtype=torch.bfloat16, device="cuda")
    q_b = torch.empty(rows, w_qb.shape[0], dtype=torch.bfloat16, device="cuda")
    res = residual.clone()

    def chain(op, use_reference):
        run = op.reference if use_reference else op
        bf16_low_m_gemm(x, w_in, a_buf, tactic, True)
        args = _fused_args(spec, a_buf, projection, base, res, weight)
        out, _ = (fused_op.reference if use_reference else fused_op)(*args)
        bf16_low_m_gemm(out, w_conv, coef, tactic, True)
        bf16_low_m_gemm(out, w_qkv, q_kv, tactic, True)
        qa = run(q_kv[:, :1536], q_norm, EPS, True)
        bf16_low_m_gemm(qa, w_qb, q_b, ("splitk", 128, 8, 1, 6), True)
        return out, qa

    chain(norm_op, False)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outs = chain(norm_op, False)
    for seed in (11, 12):
        x.copy_(torch.randn(rows, 7168, generator=g, device="cuda").bfloat16())
        res.copy_(residual)
        graph.replay()
        torch.cuda.synchronize()
        got = [t.clone() for t in (*outs, res, coef, q_b)]
        res.copy_(residual)
        want_outs = chain(norm_op, True)
        torch.cuda.synchronize()
        for name, u, v in zip(
            ("conv-add-norm", "q_a norm", "residual", "conv_proj", "q_b"),
            got,
            (*want_outs, res, coef, q_b),
        ):
            _bitwise(u, v, name)


def test_one_launch_each() -> None:
    raw = _fused_raw(fused_op.CELLS[0].spec, seed=1)
    args = _fused_args(fused_op.CELLS[0].spec, *raw)
    nargs = _norm_args(norm_op.CELLS[1].spec, seed=1)
    fused_op(*args), norm_op(*nargs)
    torch.cuda.synchronize()
    for fn, a in ((fused_op, args), (norm_op, nargs)):
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            fn(*a)
            torch.cuda.synchronize()
        n = sum(1 for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA)
        assert n == 1, (type(fn).__name__, n)


def test_guard_refuses_overlap() -> None:
    raw = _fused_raw(fused_op.CELLS[0].spec, seed=5)
    args = list(_fused_args(fused_op.CELLS[0].spec, *raw))
    args[3] = args[0]
    with validating(fused_op), pytest.raises(AssertionError, match="overlap"):
        fused_op(*args)
