# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise runtime BF16 linear dispatch across token counts and layouts."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modules.embedding import LMHead
from tensorrt_llm._torch.modules.linear import UnquantizedLinearMethod
from tensorrt_llm._utils import get_sm_version

pytestmark = pytest.mark.skipif(
    get_sm_version() not in (100, 103), reason="Deterministic BF16 GEMM is SM100/SM103 only"
)


@pytest.mark.parametrize("layout", ["contiguous", "row_stride", "column_stride"])
@pytest.mark.parametrize("bias_enabled", [False, True])
def test_linear_token_count_invariance(monkeypatch, layout, bias_enabled):
    monkeypatch.setenv("FORCE_DETERMINISTIC", "1")
    torch.manual_seed(1942)
    n, k = 513, 1031
    weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(n, device="cuda", dtype=torch.bfloat16) if bias_enabled else None
    layer = SimpleNamespace(
        weight=weight,
        tp_size=1,
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


@pytest.mark.parametrize("output_dtype", [None, torch.float32], ids=["bf16", "fp32"])
@torch.inference_mode()
def test_lm_head_token_count_invariance(monkeypatch, output_dtype):
    monkeypatch.setenv("FORCE_DETERMINISTIC", "1")
    torch.manual_seed(11)
    hidden_size, vocab_size = 4096, 131072
    head = LMHead(vocab_size, hidden_size, dtype=torch.bfloat16, output_dtype=output_dtype).cuda()
    head.weight.copy_(
        torch.randn(vocab_size, hidden_size, device="cuda", dtype=torch.bfloat16) * 0.02
    )
    target = torch.randn(8, hidden_size, device="cuda", dtype=torch.bfloat16)
    expected = torch.cat([head(row[None]) for row in target])
    assert expected.dtype == (output_dtype or torch.bfloat16)
    reference = target.double() @ head.weight[:4096].double().T
    if output_dtype is None:
        torch.testing.assert_close(
            expected[:, :4096].float(), reference.bfloat16().float(), rtol=0.008, atol=0.001
        )
    else:
        torch.testing.assert_close(expected[:, :4096].double(), reference, rtol=1e-4, atol=1e-4)
    for m in (8, 31, 64, 257, 513):
        inputs = torch.randn(m, hidden_size, device="cuda", dtype=torch.bfloat16)
        indices = torch.linspace(0, m - 1, 8, device="cuda").long()
        inputs[indices] = target
        torch.testing.assert_close(head(inputs)[indices], expected, rtol=0, atol=0)
