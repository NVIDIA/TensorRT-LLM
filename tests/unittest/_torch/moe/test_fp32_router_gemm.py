# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from tensorrt_llm._torch.moe.fused_moe.fp32_router_gemm import fp32_router_gemm


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("rows", [1, 2, 4, 8])
@pytest.mark.parametrize("strided", [False, True])
def test_fp32_router_logits_and_selection_replay(rows: int, strided: bool) -> None:
    generator = torch.Generator(device="cuda").manual_seed(6810832)
    weight = torch.randn(288, 4096, device="cuda", generator=generator) * 0.02
    bias = torch.randn(288, device="cuda", generator=generator) * 0.01
    storage = torch.randn(rows * 2, 4096, dtype=torch.bfloat16, device="cuda", generator=generator)
    x = storage[::2] if strided else storage[:rows]
    fp32_router_gemm(x, weight)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = fp32_router_gemm(x, weight)
    for _ in range(3):
        x.copy_(torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=generator))
        graph.replay()
        # Explicit FP32 products avoid any TF32 setting affecting the oracle.
        expected = (x.float()[:, None, :] * weight[None, :, :]).sum(dim=-1)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
        actual_ids = torch.topk(actual.sigmoid() + bias, 8, dim=-1).indices
        expected_ids = torch.topk(expected.sigmoid() + bias, 8, dim=-1).indices
        assert torch.equal(actual_ids, expected_ids)
