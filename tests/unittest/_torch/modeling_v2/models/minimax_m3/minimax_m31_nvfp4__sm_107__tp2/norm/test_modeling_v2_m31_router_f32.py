# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for TP2AGG_ROUTER_F32: the MoE boundary's CTXNORMQ _f32 variant writes the norm widened to float32, and the
prefill router reads it instead of casting. Every output must equal the base op's bit for bit, the fp32 copy must be
norm.float() bit for bit, and the router logits must equal m3_fp32_linear's. Skipped when either op is not built."""

import types

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2 import (
    modeling as m,
)

assert torch.cuda.is_available(), "TP2AGG_ROUTER_F32 requires a CUDA device"

H, E = 6144, 128
EPS = 1e-6
FP4_SCALE = 448.0 * 6.0 / 8.0

op = m._ctxnormq_op()
f32_op = m._ctxnormq_f32_op()
pytestmark = pytest.mark.skipif(
    op is None or f32_op is None, reason="the CTXNORMQ op or its _f32 variant is not built"
)


def _operands(rows, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    inter = (
        torch.randn(rows, H, generator=g, device="cuda")
        * torch.pow(2.0, torch.randint(-6, 7, (rows, 1), generator=g, device="cuda").float())
    ).to(torch.bfloat16)
    residual = (2 * torch.randn(rows, H, generator=g, device="cuda")).to(torch.bfloat16)
    weight = (1.0 + 0.1 * torch.randn(H, generator=g, device="cuda")).to(torch.bfloat16)
    router = 0.02 * torch.randn(E, H, generator=g, device="cuda")
    return inter, residual, weight, router


def _bytes_equal(a, b):
    return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


@pytest.mark.parametrize("rows", [1, 127, 1024, 1160, 4096, 12697])
@pytest.mark.parametrize("fp4", [False, True], ids=["nofp4", "moe"])
def test_f32_variant_equals_the_base_op(rows, fp4) -> None:
    inter, residual, weight, _ = _operands(rows, 20261031 + rows)
    scale = torch.tensor([FP4_SCALE], device="cuda", dtype=torch.float32) if fp4 else None
    inter_a, inter_b = inter.clone(), inter.clone()
    base = op(inter_a, residual, weight, EPS, True, scale, False)
    outs = f32_op(inter_b, residual, weight, EPS, scale, False)
    assert _bytes_equal(inter_a, inter_b)
    for a, b in zip(base, outs[:5]):
        assert _bytes_equal(a, b)
    assert outs[5].dtype == torch.float32
    assert torch.equal(outs[5].view(torch.int32), base[0].float().view(torch.int32))


def test_router_reads_the_boundary_f32() -> None:
    """_router_logits takes the published fp32 copy for the same tensor (bitwise m3_fp32_linear), once; any other
    tensor falls back to the cast."""
    rows = 2048
    inter, residual, weight, router = _operands(rows, 5)
    scale = torch.tensor([FP4_SCALE], device="cuda", dtype=torch.float32)
    normed, _, _, _, _, normed_f32 = f32_op(inter, residual, weight, EPS, scale, False)
    core = types.SimpleNamespace(w={"l3_router": router})
    ref = m.m3_fp32_linear(normed, router)
    m._CTXNORMQ_F32.publish(normed, normed_f32, None)
    got = m._router_logits(core, 3, normed)
    assert torch.equal(got.view(torch.int32), ref.view(torch.int32))
    assert m._CTXNORMQ_F32.pending is None
    m._CTXNORMQ_F32.publish(normed, normed_f32, None)
    other = normed.clone()
    assert torch.equal(m._router_logits(core, 3, other).view(torch.int32), ref.view(torch.int32))
    assert m._CTXNORMQ_F32.pending is None


def test_fake_impl_matches_the_op() -> None:
    from torch._subclasses.fake_tensor import FakeTensorMode

    inter, residual, weight, _ = _operands(1160, 7)
    scale = torch.tensor([FP4_SCALE], device="cuda")
    real = f32_op(inter.clone(), residual, weight, EPS, scale, False)
    with FakeTensorMode(allow_non_fake_inputs=True) as mode:
        fake = f32_op(
            mode.from_tensor(inter),
            mode.from_tensor(residual),
            mode.from_tensor(weight),
            EPS,
            mode.from_tensor(scale),
            False,
        )
    assert [(tuple(t.shape), t.dtype) for t in fake] == [(tuple(t.shape), t.dtype) for t in real]
