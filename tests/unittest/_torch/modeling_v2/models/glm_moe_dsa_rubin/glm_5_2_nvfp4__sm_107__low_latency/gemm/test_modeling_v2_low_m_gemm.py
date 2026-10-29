# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the glm_moe_dsa tp16 low_m_gemm entry."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa_rubin.glm_5_2_nvfp4__sm_107__low_latency.ops.gemm.low_m_gemm import (  # noqa: E501
    low_m_gemm as op,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../.."]
from _validating import assert_within_ulp, certifying, validating  # noqa: E402,F401

pytestmark = pytest.mark.skipif(
    get_sm_version() != 107, reason="the glm_moe_dsa tp16 entries are certified on sm_107"
)


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """One GEMM's worth of accumulation-order difference.

    Kernel and reference multiply the same BF16 operands exactly in FP32 and differ only in the
    order of the FP32 sums: the kernel's per-slice tensor-core accumulation and its fixed DSMEM
    reduction of the K slices against torch's FP32 matmul. That is about 1e-6 relative to a
    row's scale, which moves the one BF16 rounding of an output only where the FP32 sum sits
    next to a rounding midpoint: rare one-ulp flips. One BF16 ulp of the row maximum is 2 units
    of the band's 2**-8, so 4 units element-wise admits any such flip with margin, and 2 units of
    relative RMS sits far above what a handful of flips adds. A lost or doubled K slice, a
    wrong tile or a missing tail is off by the size of the values themselves, hundreds of units.
    """
    assert_within_ulp(out, ref, element_ulp=4.0, rms_ulp=2.0)


def _build(spec: dict, seed: int):
    """Operands of one cell: unit-variance activations, weights scaled to unit-variance outputs.

    `out` starts as NaN, so an output the kernel never stores fails the comparison.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    m, k, n = spec["m"], spec["k"], spec["n"]
    x = torch.randn(m, k, generator=g, device="cuda").to(torch.bfloat16)
    weight = (torch.randn(n, k, generator=g, device="cuda") * k**-0.5).to(torch.bfloat16)
    out = torch.full((m, n), float("nan"), dtype=torch.bfloat16, device="cuda")
    return x, weight, out


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    x, weight, out = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    x_before, weight_before = x.clone(), weight.clone()
    # Identity, not equality: the op writes the caller's buffer and returns it.
    assert op(x, weight, out) is out, "the op writes in place and returns what it wrote"
    assert torch.equal(x, x_before) and torch.equal(weight, weight_before)
    _compare(out, op.reference(x, weight, out))


@pytest.mark.parametrize(
    "spec, one_wave",
    [
        (dict(m=8, k=6144, n=256), False),
        (dict(m=8, k=6144, n=19360), True),
        (dict(m=8, k=6144, n=3072), False),
        (dict(m=8, k=1536, n=6144), False),
    ],
)
def test_cuda_graph_capture_and_replay(spec, one_wave) -> None:
    """The target captures the whole decode step: the op must replay on new contents of the
    same buffers. Warm-up compiles the kernel outside the capture."""
    x, weight, out = _build(spec, seed=31)

    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        op(x, weight, out, one_wave=one_wave)
    torch.cuda.current_stream().wait_stream(warmup)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(x, weight, out, one_wave=one_wave)

    fresh, _, _ = _build(spec, seed=32)
    x.copy_(fresh)
    out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    _compare(out, op.reference(x, weight, out))


def _overlapping(spec: dict, operand: str):
    """A cell's operands with `out` placed over `operand`'s memory."""
    x, weight, _ = _build(spec, seed=33)
    m, n = spec["m"], spec["n"]
    source = x if operand == "x" else weight
    buffer = torch.empty(source.numel() + m * n, dtype=torch.bfloat16, device="cuda")
    buffer[: source.numel()].copy_(source.view(-1))
    placed = buffer[: source.numel()].view(source.shape)
    out = buffer[: m * n].view(m, n)
    return (placed, weight, out) if operand == "x" else (x, placed, out)


@pytest.mark.parametrize("operand", ["x", "weight"])
def test_guard_refuses_an_out_over_an_input(operand) -> None:
    """The owner CTAs store while other clusters still read the inputs: silently wrong."""
    x, weight, out = _overlapping(op.CELLS[0].spec, operand)
    with pytest.raises(AssertionError, match=f"out must not share memory with {operand}"):
        with validating(op):
            op(x, weight, out)


@pytest.mark.parametrize("rows", [1, 2, 3, 6, 12, 18])
def test_batched_v_b_strides_and_graph(rows):
    from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import true_fp32_matmul
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa_rubin.glm_5_2_nvfp4__sm_107__low_latency.ops.gemm._low_m_splitk import (  # noqa: E501
        bmm_prewait_tactic,
        run_splitk_bmm,
    )

    gen = torch.Generator(device="cuda").manual_seed(400 + rows)
    latent = torch.randn(rows, 16, 512, device="cuda", generator=gen).bfloat16()
    weight = (torch.randn(16, 256, 512, device="cuda", generator=gen) / 512**0.5).bfloat16()
    storage = torch.full((rows + 2, 16, 256), float("nan"), device="cuda", dtype=torch.bfloat16)
    output = storage[1:-1]
    tactic = bmm_prewait_tactic(rows, 256, 512)

    def run():
        return run_splitk_bmm(
            latent.transpose(0, 1),
            weight,
            output.transpose(0, 1),
            True,
            tactic,
            reserve_full_smem=False,
        )

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for _ in range(3):
        latent.copy_(torch.randn(latent.shape, device="cuda", generator=gen).bfloat16())
        graph.replay()
        torch.cuda.synchronize()
        with true_fp32_matmul():
            ref = torch.bmm(latent.transpose(0, 1).float(), weight.float().transpose(1, 2))
        _compare(output, ref.transpose(0, 1).bfloat16())
        assert storage[0].isnan().all() and storage[-1].isnan().all()


@pytest.mark.parametrize(
    "rows, vocab", [(3, 38720), (1, 19360), (2, 19360), (4, 19360), (8, 19360)]
)
def test_one_wave_lm_head_numeric(rows, vocab):
    x, weight, out = _build(dict(m=rows, k=6144, n=vocab), seed=715)
    op(x, weight, out, one_wave=True)
    _compare(out, op.reference(x, weight, out))


@pytest.mark.parametrize(
    "rows, inter", [(6, 3072), (12, 3072), (18, 3072), (1, 1536), (2, 1536), (4, 1536), (8, 1536)]
)
def test_dense_mlp_composition(rows, inter):
    from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import true_fp32_matmul
    from tensorrt_llm._torch.modules.swiglu import swiglu

    x, up, mid = _build(dict(m=rows, k=6144, n=2 * inter), seed=800 + rows)
    _, down, out = _build(dict(m=rows, k=inter, n=6144), seed=900 + rows)
    got = op(swiglu(op(x, up, mid)), down, out)
    with true_fp32_matmul():
        gate_up = (x.float() @ up.float().t()).bfloat16()
        gate, value = gate_up.float().chunk(2, dim=-1)
        activated = (torch.nn.functional.silu(gate) * value).bfloat16()
        ref = (activated.float() @ down.float().t()).bfloat16()
    _compare(got, ref)
