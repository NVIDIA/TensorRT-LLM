# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bare router projections must honor the global deterministic arithmetic mode."""

import pytest
import torch

from tensorrt_llm._torch.models.modeling_qwen3_moe import Qwen3Gate
from tensorrt_llm._torch.models.modeling_qwen3_next import Qwen3NextGate
from tensorrt_llm._torch.modules.deterministic_linear import deterministic_linear_enabled
from tensorrt_llm._torch.moe.fused_moe.trtllm_gen import decode_front
from tensorrt_llm._utils import get_sm_version

pytestmark = pytest.mark.skipif(
    get_sm_version() != 100, reason="Deterministic BF16 GEMM is SM100 only"
)


@pytest.mark.parametrize("gate_cls", [Qwen3NextGate, Qwen3Gate])
@pytest.mark.parametrize("seed", [17, 42, 99])
def test_qwen_gate_deterministic_batch_invariance(monkeypatch, gate_cls, seed):
    monkeypatch.setenv("FORCE_DETERMINISTIC", "1")
    torch.manual_seed(seed)
    gate = gate_cls(2048, 256, 8, dtype=torch.bfloat16).cuda()
    gate.weight.copy_(torch.randn_like(gate.weight) * 0.02)
    x = torch.randn(640, 2048, device="cuda", dtype=torch.bfloat16)
    full = gate(x)
    assert full.dtype == torch.bfloat16
    for row in (0, 127, 511, 639):
        torch.testing.assert_close(gate(x[row : row + 1]), full[row : row + 1], atol=0, rtol=0)
    torch.testing.assert_close(gate(x[:512]), full[:512], atol=0, rtol=0)
    expected = x.double() @ gate.weight.double().T
    torch.testing.assert_close(full.float(), expected.float(), atol=5e-5, rtol=0.008)


@pytest.mark.parametrize("gate_cls", [Qwen3NextGate, Qwen3Gate])
def test_qwen_gate_float_output_preserved(monkeypatch, gate_cls):
    monkeypatch.setenv("FORCE_DETERMINISTIC", "1")
    gate = gate_cls(2048, 256, 8, dtype=torch.bfloat16).cuda()
    # Each logit is 2047 + 2**-8, which FP32 holds and BF16 rounds to 2048: a
    # BF16 result converted to FP32 would not pass.
    gate.weight.fill_(1.0)
    gate.weight[:, -1] = 2.0**-8
    gate.out_dtype = torch.float32
    output = gate(torch.ones(3, 2048, device="cuda", dtype=torch.bfloat16))
    assert output.dtype == torch.float32
    assert torch.all(output == 2047.00390625)


def test_decode_front_leaves_the_router_to_the_fixed_k_gate(monkeypatch):
    """The decode front runs a router GEMM of its own, so it declines the calls
    whose gate takes the fixed-K kernel and still considers the others."""
    asked = []
    monkeypatch.setattr(decode_front, "front_eligible", lambda *args: asked.append(args) and False)
    gate = Qwen3NextGate(2048, 256, 8, dtype=torch.bfloat16).cuda()
    x = torch.randn(4, 2048, device="cuda", dtype=torch.bfloat16)
    front = decode_front.DecodeFront(frozenset())

    monkeypatch.setenv("FORCE_DETERMINISTIC", "1")
    deterministic_linear_enabled.cache_clear()
    assert front(gate, None, x) is None
    assert not asked

    monkeypatch.delenv("FORCE_DETERMINISTIC")
    deterministic_linear_enabled.cache_clear()
    assert front(gate, None, x) is None
    assert len(asked) == 1
