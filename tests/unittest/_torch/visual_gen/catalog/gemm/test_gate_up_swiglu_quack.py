# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.visual_gen.catalog.gemm import gate_up_swiglu_quack as ops
from tensorrt_llm._utils import get_sm_version


@pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() not in (100, 103),
    reason="QuACK BF16 SwiGLU requires SM100 or SM103",
)
@pytest.mark.parametrize("cell", ops.gate_up_swiglu_quack_bf16.CELLS, ids=lambda cell: cell.why)
def test_quack_epilogue(cell):
    spec = cell.spec
    gen = torch.Generator(device="cuda").manual_seed(42)
    x = torch.randn(
        (spec["tokens"], spec["hidden"]), dtype=torch.bfloat16, device="cuda", generator=gen
    )
    weight = torch.randn(
        (2 * spec["intermediate"], spec["hidden"]),
        dtype=torch.bfloat16,
        device="cuda",
        generator=gen,
    )
    weight = weight * spec["hidden"] ** -0.5
    assert ops.gate_up_swiglu_quack_available()
    compiled = torch.compile(ops.gate_up_swiglu_quack_bf16, fullgraph=True)
    actual = compiled(x, weight)
    assert ops.gate_up_swiglu_quack_available()
    expected = ops.gate_up_swiglu_quack_bf16.reference(x, weight)
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=2e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="uses the CUDA op")
def test_provider_failure_preserves_unfused_result(monkeypatch):
    calls = []

    def fail(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("test JIT failure")

    monkeypatch.setattr(ops, "_quack_gemm_act", lambda: fail)
    monkeypatch.setattr(ops, "_provider_failed", False)
    gen = torch.Generator(device="cuda").manual_seed(7)
    x = torch.randn((3, 32), dtype=torch.bfloat16, device="cuda", generator=gen)
    weight = torch.randn((32, 32), dtype=torch.bfloat16, device="cuda", generator=gen)
    gate, up = F.linear(x, weight).chunk(2, dim=-1)
    expected = F.silu(gate) * up
    for _ in range(2):
        torch.testing.assert_close(
            ops.gate_up_swiglu_quack_bf16(x, weight), expected, atol=0, rtol=0
        )
    assert len(calls) == 1
