# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FP32 router logits keep their precision and arithmetic across token counts."""

import pytest
import torch

from tensorrt_llm._torch.models.modeling_deepseekv3 import DeepseekV3Gate


@pytest.mark.parametrize("hidden,experts", [(2688, 128), (4096, 256), (7168, 256)])
def test_deepseek_gate_deterministic_fp32_projection(monkeypatch, hidden, experts):
    monkeypatch.setenv("FORCE_DETERMINISTIC", "1")
    torch.manual_seed(99)
    gate = DeepseekV3Gate(
        hidden,
        experts,
        top_k=8,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=1.0,
        dtype=torch.bfloat16,
    ).cuda()
    gate.weight.copy_(torch.randn_like(gate.weight) * 0.02)
    x = torch.randn(640, hidden, device="cuda", dtype=torch.bfloat16)
    full = gate(x)
    assert full.dtype == torch.float32
    rows = [0, 127, 511, 639]
    for row in rows:
        torch.testing.assert_close(gate(x[row : row + 1]), full[row : row + 1], atol=0, rtol=0)
    torch.testing.assert_close(gate(x[:512]), full[:512], atol=0, rtol=0)
    reference = x[rows].double() @ gate.weight.double().T
    # FP32 accumulation of up to 7168 products incurs rounding error;
    # cross-shape equality above remains bitwise exact.
    torch.testing.assert_close(full[rows].double(), reference, atol=5e-5, rtol=1e-5)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = gate(x[:7])
    x.mul_(0.875)
    graph.replay()
    torch.testing.assert_close(captured, gate(x[:7]), atol=0, rtol=0)
    torch.testing.assert_close(captured[:1], gate(x[:1]), atol=0, rtol=0)
