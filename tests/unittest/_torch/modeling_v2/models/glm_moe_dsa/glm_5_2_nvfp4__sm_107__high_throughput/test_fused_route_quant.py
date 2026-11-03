# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The fused producer must preserve routing and linear-layout NVFP4 bytes."""

import pytest
import torch

import tensorrt_llm  # noqa: F401

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="The high-throughput GLM target requires SM107",
)


@pytest.mark.parametrize("rows", [1, 7, 20, 56, 60, 120])
@pytest.mark.parametrize("topk", [8, 16])
def test_fused_route_quant_matches_separate_ops(rows: int, topk: int):
    torch.manual_seed(123)
    scores = torch.randn(rows, 256, dtype=torch.float32, device="cuda")
    bias = torch.randn(256, dtype=torch.float32, device="cuda")
    hidden = torch.randn(rows, 6144, dtype=torch.bfloat16, device="cuda")
    scale = torch.tensor([448.0 * 6.0 / 8.0], dtype=torch.float32, device="cuda")
    values = torch.empty(rows, topk, dtype=torch.float32, device="cuda")
    indices = torch.empty(rows, topk, dtype=torch.int32, device="cuda")
    packed = torch.empty(rows, 6144 // 2, dtype=torch.uint8, device="cuda")
    scales = torch.empty(rows, 6144 // 16, dtype=torch.uint8, device="cuda")

    def fused():
        torch.ops.trtllm.glm_rubin_noaux_tc_fp4_quantize_out(
            scores, bias, hidden, scale, 1, 1, topk, 2.5, values, indices, packed, scales
        )

    fused()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fused()
    # Replay with new inputs; zeros also exercise routing tie handling.
    for tied in (False, True):
        scores.copy_(torch.zeros_like(scores) if tied else torch.randn_like(scores))
        bias.copy_(torch.zeros_like(bias) if tied else torch.randn_like(bias))
        hidden.copy_(torch.randn_like(hidden))
        graph.replay()
        expected_values, expected_indices = torch.ops.trtllm.noaux_tc_op(
            scores, bias, 1, 1, topk, 2.5
        )
        expected_packed, expected_scales = torch.ops.trtllm.fp4_quantize(
            hidden, scale, 16, False, False
        )
        torch.testing.assert_close(indices, expected_indices, rtol=0, atol=0)
        torch.testing.assert_close(values, expected_values, rtol=0, atol=0)
        torch.testing.assert_close(packed, expected_packed.view(torch.uint8), rtol=0, atol=0)
        torch.testing.assert_close(
            scales.reshape(-1), expected_scales.view(torch.uint8).reshape(-1), rtol=0, atol=0
        )
