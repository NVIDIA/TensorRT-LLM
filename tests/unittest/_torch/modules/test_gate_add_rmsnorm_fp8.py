# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared-expert gate + residual add + RMSNorm + E4M3 quant in one launch (kernels/blackwell/gate_add_rmsnorm_fp8.py):
bitwise equal to the three launches of the attention-DP post-MoE chain it replaces."""

import pytest
import torch

import tensorrt_llm  # noqa: F401
from tensorrt_llm._torch.moe.fused_shared_expert import fused_sigmoid_gate_mul_add
from tensorrt_llm._utils import is_blackwell_datacenter

pytestmark = pytest.mark.skipif(
    not is_blackwell_datacenter(), reason="the post-MoE fold runs on SM 10.0 / 10.3"
)


def _inputs(rows, hidden, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*shape, scale=1.0):
        return (torch.randn(*shape, generator=g, device="cuda") * scale).to(torch.bfloat16)

    return dict(
        routed=randn(rows, hidden),
        shared=randn(rows, hidden, scale=0.5),
        gate_logits=randn(rows, 1, scale=2.0),
        residual=randn(rows, hidden, scale=4.0),
        weight=randn(hidden, scale=0.1),
        scale=torch.rand(1, generator=g, device="cuda") * 0.05 + 0.01,
    )


def _reference(t, eps):
    """fused_sigmoid_gate_mul_add, FlashInfer's gemma fused add + RMSNorm (as RMSNorm runs it), then the next FP8
    projection's static_quantize_e4m3_per_tensor."""
    x = fused_sigmoid_gate_mul_add(t["routed"].clone(), t["gate_logits"], t["shared"])
    residual = t["residual"].clone()
    torch.ops.trtllm.flashinfer_gemma_fused_add_rmsnorm(x, residual, t["weight"], eps)
    q, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(x, t["scale"])
    return x, residual, q


@pytest.mark.parametrize("write_y", [True, False])
@pytest.mark.parametrize("rows", [1, 7, 48 * 7, 52 * 7, 1001])
def test_gate_add_rmsnorm_fp8_bitwise(rows, write_y):
    from tensorrt_llm._torch.kernels.blackwell.gate_add_rmsnorm_fp8 import gate_add_rmsnorm_fp8

    hidden, eps = 4096, 1e-6
    t = _inputs(rows, hidden, seed=rows)
    y_ref, residual_ref, q_ref = _reference(t, eps)

    routed, residual = t["routed"].clone(), t["residual"].clone()
    y, q = gate_add_rmsnorm_fp8(
        routed,
        t["shared"],
        t["gate_logits"],
        residual,
        t["weight"],
        eps,
        t["scale"],
        write_y=write_y,
    )
    torch.cuda.synchronize()

    assert y is routed
    assert torch.equal(residual.view(torch.int16), residual_ref.view(torch.int16))
    assert torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8))
    if write_y:
        assert torch.equal(y.view(torch.int16), y_ref.view(torch.int16))
    else:
        assert torch.equal(routed.view(torch.int16), t["routed"].view(torch.int16))


def test_supports_hidden():
    from tensorrt_llm._torch.kernels.blackwell.gate_add_rmsnorm_fp8 import supports_hidden

    assert supports_hidden(4096)
    assert not supports_hidden(4096 + 8)
