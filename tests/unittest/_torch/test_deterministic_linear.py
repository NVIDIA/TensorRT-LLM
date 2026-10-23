# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise runtime BF16 linear dispatch across token counts and layouts."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modules.linear import UnquantizedLinearMethod

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")


@pytest.mark.parametrize("layout", ["contiguous", "row_stride", "column_stride"])
@pytest.mark.parametrize("bias_enabled", [False, True])
@pytest.mark.parametrize("tp_size", [1, 4])
def test_linear_token_count_invariance(monkeypatch, layout, bias_enabled, tp_size):
    monkeypatch.setenv("FORCE_DETERMINISTIC", "1")
    torch.manual_seed(1942)
    n, k = 513, 1031
    weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(n, device="cuda", dtype=torch.bfloat16) if bias_enabled else None
    layer = SimpleNamespace(
        weight=weight,
        tp_size=tp_size,
        use_custom_cublas_mm=False,
        use_cute_dsl_bf16_gemm=False,
        mapping=None,
    )
    method = UnquantizedLinearMethod()
    target = torch.randn(8, k, device="cuda", dtype=torch.bfloat16)
    expected = torch.cat([method.apply(layer, row[None], bias) for row in target])
    reference = target.double() @ weight.double().T
    if bias is not None:
        reference += bias.double()
    torch.testing.assert_close(
        expected.float(), reference.bfloat16().float(), rtol=0.008, atol=0.001
    )
    for m in (8, 31, 32, 63, 64, 127, 128, 129, 255, 256, 513):
        width = k if layout == "contiguous" else 2 * k
        storage = torch.randn(m, width, device="cuda", dtype=torch.bfloat16)
        inputs = (
            storage
            if layout == "contiguous"
            else (storage[:, :k] if layout == "row_stride" else storage[:, ::2])
        )
        indices = torch.linspace(0, m - 1, 8, device="cuda").long()
        inputs[indices] = target
        actual = method.apply(layer, inputs, bias)
        torch.testing.assert_close(actual[indices], expected, rtol=0, atol=0)


def test_linear_cuda_graph_reads_updated_input(monkeypatch):
    monkeypatch.setenv("FORCE_DETERMINISTIC", "1")
    torch.manual_seed(17)
    layer = SimpleNamespace(
        weight=torch.randn(97, 257, device="cuda", dtype=torch.bfloat16),
        tp_size=1,
        use_custom_cublas_mm=False,
        use_cute_dsl_bf16_gemm=False,
        mapping=None,
    )
    x = torch.randn(2, 4, 257, device="cuda", dtype=torch.bfloat16)
    method = UnquantizedLinearMethod()
    expected = method.apply(layer, x, None)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = method.apply(layer, x, None)
    graph.replay()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    x.mul_(0.5)
    expected = method.apply(layer, x, None)
    graph.replay()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
