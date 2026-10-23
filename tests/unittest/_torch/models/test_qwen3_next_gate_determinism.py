# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bare router projections must honor the global deterministic arithmetic mode."""

import pytest
import torch

from tensorrt_llm._torch.models.modeling_qwen3_next import Qwen3NextGate


@pytest.mark.parametrize("seed", [17, 42, 99])
def test_qwen_gate_deterministic_batch_invariance(monkeypatch, seed):
    monkeypatch.setenv("FORCE_DETERMINISTIC", "1")
    torch.manual_seed(seed)
    gate = Qwen3NextGate(2048, 256, 8, dtype=torch.bfloat16).cuda()
    gate.weight.copy_(torch.randn_like(gate.weight) * 0.02)
    x = torch.randn(640, 2048, device="cuda", dtype=torch.bfloat16)
    full = gate(x)
    for row in (0, 127, 511, 639):
        torch.testing.assert_close(gate(x[row : row + 1]), full[row : row + 1], atol=0, rtol=0)
    torch.testing.assert_close(gate(x[:512]), full[:512], atol=0, rtol=0)
    expected = x.double() @ gate.weight.double().T
    torch.testing.assert_close(full.float(), expected.float(), atol=5e-5, rtol=0.008)


def test_qwen_gate_float_output_preserved(monkeypatch):
    monkeypatch.setenv("FORCE_DETERMINISTIC", "1")
    gate = Qwen3NextGate(2048, 256, 8, dtype=torch.bfloat16).cuda()
    gate.weight.zero_()
    gate.out_dtype = torch.float32
    output = gate(torch.ones(3, 2048, device="cuda", dtype=torch.bfloat16))
    assert output.dtype == torch.float32
    assert torch.count_nonzero(output) == 0
