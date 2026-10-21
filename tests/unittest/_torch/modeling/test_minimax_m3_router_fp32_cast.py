# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 router input cast: the SM107 Triton path is bitwise ``.to(torch.float32)``, and its gate."""

import pytest
import torch

from tensorrt_llm._torch.models import modeling_minimaxm3 as m3

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA op")


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int32)


@cuda
@pytest.mark.parametrize("tokens", [2048, 2049, 8192, 16384])
def test_router_cast_bitwise(tokens):
    g = torch.Generator(device="cuda").manual_seed(tokens)
    x = torch.randn(tokens, 6144, device="cuda", generator=g).to(torch.bfloat16)
    x[0, :6] = torch.tensor(
        [float("inf"), float("-inf"), float("nan"), -0.0, 1e-40, -3e38], dtype=torch.bfloat16
    )
    out = m3.minimax_m3_router_input_fp32(x)
    assert out.dtype == torch.float32 and out.shape == x.shape
    assert torch.equal(_bits(out), _bits(x.to(torch.float32)))


@cuda
def test_router_cast_fallbacks():
    x = torch.randn(4096, 2 * 6144, device="cuda").to(torch.bfloat16)
    for t in (x[:, :6144], x[:1024, :6144].contiguous(), x[:, :6144].half()):
        out = m3.minimax_m3_router_input_fp32(t)
        assert torch.equal(_bits(out), _bits(t.to(torch.float32)))


@pytest.mark.parametrize(
    "sm_version,enabled,expected", [(107, True, True), (103, True, False), (107, False, False)]
)
def test_router_cast_gate(monkeypatch, sm_version, enabled, expected):
    monkeypatch.setattr(m3, "get_sm_version", lambda: sm_version)
    monkeypatch.setattr(m3, "_SM107_ROUTER_FP32_CAST", enabled)
    gate = m3.MiniMaxM3Gate(hidden_size=64, num_experts=8, top_k=2, routed_scaling_factor=1.0)
    assert gate._sm107_fp32_cast is expected
