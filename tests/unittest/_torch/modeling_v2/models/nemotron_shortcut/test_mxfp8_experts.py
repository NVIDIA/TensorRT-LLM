# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression for native block scales and slot-weighted FC2 during graph replay."""

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.catalog.quantization.mxfp8_quantize import mxfp8_quantize
from tensorrt_llm._torch.modeling_v2.models.nemotron_shortcut.moe import grouped_quantized_experts
from tensorrt_llm._torch.modeling_v2.models.nemotron_shortcut.mxfp8__sm_103__default.ops import (
    Mxfp8Projection,
)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="native MXFP8 projections require Blackwell SM100/SM103",
)
def test_mxfp8_block_scales_and_slot_weights_across_graph_replay():
    device = torch.device("cuda")
    experts, width, padded = 3, 160, 256
    channels = torch.arange(width, device=device)
    matrices = torch.zeros(experts, padded, padded, device=device)
    swizzled, columns, factors = [], [], []
    for expert in range(experts):
        col = (channels + 17 * expert) % width
        matrices[expert, channels, col] = 1
        exponent = (channels[:, None] + torch.arange(width // 32, device=device) + expert) % 4 - 2
        scale = torch.zeros(padded, padded // 32, dtype=torch.uint8, device=device)
        scale[:width, : width // 32] = (exponent + 127).byte()
        swizzled.append(torch.ops.trtllm.block_scale_interleave(scale).flatten())
        columns.append(col)
        factors.append(torch.exp2(exponent[channels, col // 32].float()))
    proj = Mxfp8Projection(matrices.to(torch.float8_e4m3fn), torch.stack(swizzled), width)

    def project_ref(x, expert):
        # Each row selects exactly one input channel, so the independent reference
        # has no accumulation-order error and checks each selected scale exactly.
        quant, sf = mxfp8_quantize.reference(x, swizzled_layout=False)
        dequant = quant.float().view(-1, width // 32, 32) * torch.exp2(
            sf.float().view(-1, width // 32, 1) - 127
        )
        return (dequant.view(-1, width)[:, columns[expert]] * factors[expert]).bfloat16()

    tokens = 7
    x = ((torch.arange(tokens * width, device=device).view(tokens, width) % 97 - 39) / 8).bfloat16()
    ids = torch.tensor([[0, 2], [3, 0], [2, 3], [0, 3], [2, 0], [3, 2], [2, 0]], device=device)
    scores = torch.tensor([0.27, 0.73], device=device).expand(tokens, 2).contiguous()
    dense = Mxfp8Projection(proj.weight[2], proj.block_scales[2], width)
    torch.testing.assert_close(dense.apply(x), project_ref(x, 2), rtol=0, atol=0)
    grouped_quantized_experts(x, ids, scores, proj, proj, 0, 16)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = grouped_quantized_experts(x, ids, scores, proj, proj, 0, 16)
    # Expert 1 was empty at capture and now receives local assignments.
    ids.copy_((ids + 1) % 4)
    graph.replay()
    slots = x.new_zeros((tokens, 2, width))
    for expert in range(experts):
        rows, indices = (ids == expert).nonzero(as_tuple=True)
        hidden = project_ref(x[rows], expert)
        limited = (16 * torch.tanh(hidden.float() / 16)).bfloat16()
        weighted = (limited.relu().square().float() * scores[rows, indices, None]).bfloat16()
        slots[rows, indices] = project_ref(weighted, expert)
    torch.testing.assert_close(captured, slots.float().sum(1), rtol=0, atol=0)
