# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for CTXNORMQ: the prefill boundary's residual add + RMSNorm that also writes the next projection's MXFP8
activation and the routed experts' NVFP4 input (torch.ops.trtllm.minimax_m31_ctx_residual_rmsnorm_mxfp8).

The quantized outputs must equal mxfp8_quantize / fp4_quantize of the op's own norm bit for bit; the norm and the
updated residual are checked against an fp32 reference. Skipped when the op is not built (no overlay library).
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_103__tp4 import (
    modeling as m,
)

assert torch.cuda.is_available(), "CTXNORMQ requires a CUDA device"

H = 6144
EPS = 1e-6
FP4_SCALE = 448.0 * 6.0 / 8.0

op = m._ctxnormq_op()
pytestmark = pytest.mark.skipif(
    op is None, reason="minimax_m31_ctx_residual_rmsnorm_mxfp8 is not built"
)


def _operands(rows, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    inter = torch.randn(rows, H, generator=g, device="cuda").to(torch.bfloat16)
    residual = (2 * torch.randn(rows, H, generator=g, device="cuda")).to(torch.bfloat16)
    weight = (1.0 + 0.1 * torch.randn(H, generator=g, device="cuda")).to(torch.bfloat16)
    return inter, residual, weight


@pytest.mark.parametrize("rows", [1024, 1160, 4096])
@pytest.mark.parametrize("fp4", [False, True], ids=["attn", "moe"])
def test_quantized_outputs_equal_quantizing_the_norm(rows, fp4) -> None:
    inter, residual, weight = _operands(rows, 20261021 + rows)
    scale = torch.tensor([FP4_SCALE], device="cuda", dtype=torch.float32) if fp4 else None
    ref_updated = (inter.float() + residual.float()).to(torch.bfloat16)
    normed, q, sf, f4, f4_sf = op(inter, residual, weight, EPS, True, scale, False)
    # The updated residual is written in place into inter: one bf16 rounding of the fp32 sum, exact.
    assert torch.equal(inter.view(torch.int16), ref_updated.view(torch.int16))
    u = ref_updated.float()
    ref_normed = u * torch.rsqrt(u.pow(2).mean(-1, keepdim=True) + EPS) * weight.float()
    # fp32 sum of squares + fast rsqrt vs torch's: at most one bf16 rounding apart.
    torch.testing.assert_close(normed.float(), ref_normed, rtol=2**-7, atol=2**-7)
    q_ref, sf_ref = torch.ops.trtllm.mxfp8_quantize(normed, True)
    assert torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8))
    assert torch.equal(sf.view(torch.uint8)[: sf_ref.numel()], sf_ref.view(torch.uint8).flatten())
    if fp4:
        f4_ref, f4_sf_ref = torch.ops.trtllm.fp4_quantize(normed, scale, 16, False, False)
        assert torch.equal(f4.view(torch.uint8).flatten(), f4_ref.view(torch.uint8).flatten())
        assert torch.equal(
            f4_sf.view(torch.uint8).flatten(),
            f4_sf_ref.view(torch.uint8).flatten()[: f4_sf.numel()],
        )
        # _routed_chunks splits the NVFP4 scales per row chunk.
        assert f4_sf.numel() == rows * H // 16


def test_deterministic() -> None:
    outs = []
    for _ in range(3):
        inter, residual, weight = _operands(2048, 7)
        scale = torch.tensor([FP4_SCALE], device="cuda", dtype=torch.float32)
        outs.append((inter,) + tuple(op(inter, residual, weight, EPS, True, scale, False)))
    for o in outs[1:]:
        assert all(
            torch.equal(a.view(torch.uint8), b.view(torch.uint8)) for a, b in zip(outs[0], o)
        )


def test_small_boundaries_keep_the_fused_allreduce() -> None:
    """Below CTXNORMQ_MIN_ROWS the boundary is not taken (the microbench crossover); decided before any collective."""
    x = torch.zeros(m.CTXNORMQ_MIN_ROWS - 1, H, dtype=torch.bfloat16, device="cuda")
    assert m._ctxnormq_allreduce_norm(None, x, x, x[0], None) is None


@pytest.mark.parametrize("fp4", [False, True], ids=["attn", "moe"])
@pytest.mark.parametrize("write_norm", [False, True])
def test_fake_impl_matches_the_op(fp4, write_norm) -> None:
    """The registered fake (catalog m31_fake_ops) gives every output the op's shape and dtype."""
    from torch._subclasses.fake_tensor import FakeTensorMode

    from tensorrt_llm._torch._experimental.modeling_v2.catalog.fakes import m31_fake_ops

    assert "minimax_m31_ctx_residual_rmsnorm_mxfp8" in m31_fake_ops._REGISTERED
    inter, residual, weight = _operands(1160, 7)
    scale = torch.tensor([FP4_SCALE], device="cuda") if fp4 else None
    real = op(inter.clone(), residual, weight, EPS, write_norm, scale, False)
    with FakeTensorMode(allow_non_fake_inputs=True) as mode:
        fake = op(
            mode.from_tensor(inter),
            mode.from_tensor(residual),
            mode.from_tensor(weight),
            EPS,
            write_norm,
            None if scale is None else mode.from_tensor(scale),
            False,
        )
    assert [(tuple(t.shape), t.dtype) for t in fake] == [(tuple(t.shape), t.dtype) for t in real]
