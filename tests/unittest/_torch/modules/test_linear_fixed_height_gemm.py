# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Batch-shape-deterministic Linear layers run every GEMM at one height."""

import pytest
import torch

import tensorrt_llm._torch.modules.linear as linear_module
from tensorrt_llm._torch.modules.linear import FIXED_GEMM_HEIGHT, Linear, _fixed_height_linear

ROW_COUNTS = [
    1,
    2,
    FIXED_GEMM_HEIGHT - 1,
    FIXED_GEMM_HEIGHT,
    FIXED_GEMM_HEIGHT + 1,
    3 * FIXED_GEMM_HEIGHT + 5,
]
SWITCHES = ["FORCE_DETERMINISTIC", "FORCE_KV_REUSE_DETERMINISTIC"]


@pytest.mark.cpu_only
@pytest.mark.parametrize("rows", ROW_COUNTS)
def test_every_gemm_has_the_fixed_height(monkeypatch, rows):
    real_linear = torch.nn.functional.linear
    heights = []

    def spy(chunk, weight, bias=None):
        heights.append(chunk.shape[0])
        return real_linear(chunk, weight, bias)

    monkeypatch.setattr(linear_module.F, "linear", spy)
    inputs = torch.randn(rows, 8)
    weight = torch.randn(4, 8)
    bias = torch.randn(4)

    outputs = _fixed_height_linear(inputs, weight, bias)

    assert set(heights) == {FIXED_GEMM_HEIGHT}
    assert torch.allclose(outputs, real_linear(inputs, weight, bias))


@pytest.mark.cpu_only
@pytest.mark.parametrize("shape", [(8,), (3, 8), (2, 5, 8), (0, 8)])
def test_input_batch_dimensions_are_preserved(shape):
    outputs = _fixed_height_linear(torch.randn(*shape), torch.randn(4, 8), None)

    assert outputs.shape == (*shape[:-1], 4)


@pytest.mark.cpu_only
def test_fixed_height_is_opt_in(monkeypatch):
    for switch in SWITCHES:
        monkeypatch.delenv(switch, raising=False)

    assert not Linear(8, 4, bias=False, dtype=torch.bfloat16).use_fixed_height_gemm


@pytest.mark.cpu_only
@pytest.mark.parametrize("switch", SWITCHES)
def test_each_switch_enables_fixed_height(monkeypatch, switch):
    monkeypatch.setenv(switch, "1")

    assert Linear(8, 4, bias=False, dtype=torch.bfloat16).use_fixed_height_gemm


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuBLAS")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("in_features,out_features", [(2048, 3072), (4096, 6144), (2048, 128256)])
@torch.inference_mode()
def test_row_bits_do_not_depend_on_batch_height(monkeypatch, dtype, in_features, out_features):
    monkeypatch.setenv("FORCE_KV_REUSE_DETERMINISTIC", "1")
    torch.manual_seed(42)
    with torch.device("cuda"):
        linear = Linear(in_features, out_features, bias=False, dtype=dtype)
    weight = torch.randn(out_features, in_features, device="cuda") * 0.02
    linear.load_weights([{"weight": weight.to(dtype)}])
    linear.post_load_weights()
    heights = [1, 2, 3, 8, 16, 17, 32, 64, 65, 127, 128, 129, 256]
    rows = torch.randn(max(heights), in_features, dtype=dtype, device="cuda")
    target = rows[0].clone()
    reference = linear(target.unsqueeze(0))[0].view(torch.int16)

    for height in heights:
        batch = rows[:height].clone()
        batch[height - 1] = target
        assert torch.equal(linear(batch)[height - 1].view(torch.int16), reference), height
