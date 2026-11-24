# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 DFlash2 grouped-convolution entry (CuTe DSL).

Every certified cell is gated bit for bit against the entry's reference (the target's torch
convolution): same bytes, signed zeros and NaN payloads included. The inputs are laid out as the
drafter produces them: the delta is one side's strided view of the kernel projection's
``[rows, 2 * taps * groups]`` output, the base kernel one side of ``[2, taps, hidden]``. The PDL
launch (the helix16 drafter's) is also driven between the GEMMs that feed and read it in the drafter,
captured in a CUDA graph.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.gemm.rubin_ull_bf16_low_m_gemm import (  # noqa: E501
    bf16_low_m_gemm,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.dflash2_grouped_conv import (  # noqa: E501
    K31DFlash2BlockConv,
    _torch_dflash2_grouped_conv,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.dflash2_grouped_conv import (  # noqa: E501
    dflash2_grouped_conv as op,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() not in (100, 103, 107),
    reason="the Kimi K3.1 DFlash2 kernels are certified on SM10x",
)


def _build_raw(spec, seed):
    """``(x, projection, base)``: activations, kernel projection output, both sides' base."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    batch, width = spec["batch"], spec["width"]
    hidden, taps, group = spec["hidden"], spec["taps"], spec["group"]
    rows, groups = batch * width, hidden // group
    x = torch.randn(rows, hidden, generator=g, device="cuda")
    projection = 0.25 * torch.randn(rows, 2 * taps * groups, generator=g, device="cuda")
    base = 0.5 * torch.randn(2, taps, hidden, generator=g, device="cuda")
    if spec["fill"] == "nan_prev":
        # Request 2 is a padding request full of NaN; request 3's block starts right after it.
        x[2 * width : 3 * width] = float("nan")
        projection[2 * width : 3 * width] = float("nan")
    elif spec["fill"] == "zeros":
        keep = torch.rand(rows, hidden, generator=g, device="cuda") < 0.5
        sign = torch.where(torch.rand(rows, hidden, generator=g, device="cuda") < 0.5, 1.0, -1.0)
        x = torch.where(keep, x, 0.0 * sign)
        projection = -projection.abs()
        base = -base.abs()
    return x.to(torch.bfloat16), projection.to(torch.bfloat16), base.to(torch.bfloat16)


def _args(spec, x, projection, base):
    """The op's arguments: one side's delta view and base kernel, as the drafter passes them."""
    taps, groups = spec["taps"], spec["hidden"] // spec["group"]
    delta = projection.view(x.shape[0], 2, taps, groups)[:, spec["side"]]
    return x, delta, base[spec["side"]], spec["width"], spec["group"], spec.get("pdl", False)


def _build(spec, seed):
    return _args(spec, *_build_raw(spec, seed))


def _assert_bitwise(out: torch.Tensor, ref: torch.Tensor) -> None:
    assert out.shape == ref.shape and out.dtype == ref.dtype, (out.shape, ref.shape)
    bits, ref_bits = out.view(torch.int16), ref.view(torch.int16)
    mismatched = int((bits != ref_bits).sum())
    assert mismatched == 0, f"{mismatched}/{out.numel()} elements differ from the reference"


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:48] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811
    certifying(op)
    args = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    out = op(*args)
    assert out.data_ptr() != args[0].data_ptr(), "the op must return a new tensor"
    _assert_bitwise(out, op.reference(*args))
    if cell.spec["fill"] == "nan_prev":
        width = cell.spec["width"]
        assert torch.isfinite(out[3 * width : 4 * width].float()).all(), "NaN leaked"


def test_one_launch_per_call() -> None:
    """The op is one kernel; the reference is the launch count it replaces (printed)."""
    args = _build(dict(op.CELLS[0].spec, batch=4), seed=1)
    op(*args), op.reference(*args)  # compile and warm both
    torch.cuda.synchronize()
    counts = {}
    for name, fn in (("op", op), ("reference", op.reference)):
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            fn(*args)
            torch.cuda.synchronize()
        counts[name] = sum(
            1 for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA
        )
    print(f"device launches per call: {counts}")
    assert counts["op"] == 1, counts


@pytest.mark.parametrize("pdl", [False, True], ids=["plain", "pdl"])
def test_cuda_graph_replay(pdl) -> None:
    """Captured after one eager call, replayed on new inputs: still bitwise."""
    spec = dict(op.CELLS[1].spec, batch=4, pdl=pdl)
    x, projection, base = _build_raw(spec, seed=2)
    args = _args(spec, x, projection, base)
    op(*args)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = op(*args)
    for seed in (3, 4):
        for static, new in zip((x, projection, base), _build_raw(spec, seed=seed)):
            static.copy_(new)
        graph.replay()
        torch.cuda.synchronize()
        _assert_bitwise(out, op.reference(*args))


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda a: (a[0].float(),) + a[1:], "BF16/FP16"),
        (lambda a: (a[0].t().contiguous().t(),) + a[1:], "contiguous"),
        (lambda a: (a[0], a[1], a[2], a[3], 12), "multiple of"),
    ],
    ids=["fp32-activations", "column-major", "group-not-multiple-of-8"],
)
def test_guard_refuses(mutate, message) -> None:
    args = mutate(_build(op.CELLS[0].spec, seed=5))
    with validating(op), pytest.raises(AssertionError, match=message):
        op(*args)


def test_block_conv_holder_prepare_finish_sides():
    """The ``K31DFlash2BlockConv`` holder: ``prepare`` projects its input once (``kernel_projection``), convolves it
    with side 0 and hands ``finish`` side 1's coefficients; both convolutions are the op, bitwise its torch
    reference."""
    g = torch.Generator(device="cuda").manual_seed(0)
    block_size, hidden, group, taps = 5, 7168, 16, 2
    n = 4 * block_size
    base = torch.randn(2, taps, hidden, generator=g, device="cuda").to(torch.bfloat16)
    proj = (0.02 * torch.randn(2 * taps * hidden // group, hidden, generator=g, device="cuda")).to(
        torch.bfloat16
    )
    conv = K31DFlash2BlockConv(base, proj, taps, group, hidden)
    h = torch.randn(n, hidden, generator=g, device="cuda").to(torch.bfloat16)
    a = torch.randn(n, hidden, generator=g, device="cuda").to(torch.bfloat16)
    out_in, coef_out = conv.prepare(h, block_size)
    coef = torch.nn.functional.linear(h, proj).view(n, 2, taps, hidden // group)
    ref_in = _torch_dflash2_grouped_conv(h, coef[:, 0], base[0], block_size, group)
    assert torch.equal(out_in.view(torch.int16), ref_in.view(torch.int16))
    assert torch.equal(coef_out, coef[:, 1])
    out_out = conv.finish(a, coef_out, block_size)
    ref_out = _torch_dflash2_grouped_conv(a, coef[:, 1], base[1], block_size, group)
    assert torch.equal(out_out.view(torch.int16), ref_out.view(torch.int16))


@pytest.mark.skipif(get_sm_version() != 103, reason="the drafter's low-M GEMM tactics are SM103's")
def test_pdl_between_drafter_gemms() -> None:
    """The helix16 drafter's chain in one CUDA graph, all on PDL: the kernel projection's GEMM
    writes the delta, the convolution (PDL) reads it, the next projection's GEMM reads the
    convolution's output. Replayed on new inputs, the convolution's output and the GEMM's are
    bitwise the same chain run eagerly with the plain convolution."""
    spec = dict(op.CELLS[1].spec, batch=1, side=0)
    taps, groups, hidden = spec["taps"], spec["hidden"] // spec["group"], spec["hidden"]
    rows = spec["batch"] * spec["width"]
    tactic = ("splitk", 64, 8, 4, 12)
    g = torch.Generator(device="cuda").manual_seed(6)
    x = torch.randn(rows, hidden, generator=g, device="cuda").bfloat16()
    w_proj = (0.01 * torch.randn(2 * taps * groups, hidden, generator=g, device="cuda")).bfloat16()
    w_next = (0.01 * torch.randn(1536, hidden, generator=g, device="cuda")).bfloat16()
    base = (0.5 * torch.randn(2, taps, hidden, generator=g, device="cuda")).bfloat16()
    proj = torch.empty(rows, w_proj.shape[0], dtype=torch.bfloat16, device="cuda")
    nxt = torch.empty(rows, w_next.shape[0], dtype=torch.bfloat16, device="cuda")

    def chain(pdl):
        bf16_low_m_gemm(x, w_proj, proj, tactic, True)
        delta = proj.view(rows, 2, taps, groups)[:, 0]
        conv = op(x, delta, base[0], spec["width"], spec["group"], pdl)
        bf16_low_m_gemm(conv, w_next, nxt, tactic, True)
        return conv

    chain(True)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        conv = chain(True)
    for seed in (7, 8):
        x.copy_(torch.randn(rows, hidden, generator=g, device="cuda").bfloat16())
        graph.replay()
        torch.cuda.synchronize()
        got_conv, got_next = conv.clone(), nxt.clone()
        want_conv = chain(False)
        torch.cuda.synchronize()
        _assert_bitwise(got_conv, want_conv)
        _assert_bitwise(got_next, nxt)
