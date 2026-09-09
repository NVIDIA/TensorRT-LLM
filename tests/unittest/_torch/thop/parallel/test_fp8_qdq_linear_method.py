# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tensorrt_llm._torch.modules.linear import FP8QDQLinearMethod


def _make_module(enable_cuda_core: bool) -> SimpleNamespace:
    return SimpleNamespace(
        all_reduce=None,
        dtype=torch.bfloat16,
        enable_cuda_core=enable_cuda_core,
        force_dynamic_quantization=True,
        input_scale=torch.tensor(1.0),
        mapping=None,
        weight=torch.zeros((4, 3), dtype=torch.bfloat16),
        weight_scale=torch.tensor(1.0),
    )


def test_cublas_scaled_mm_fuses_bias_without_post_add(monkeypatch: pytest.MonkeyPatch) -> None:
    input_tensor = torch.zeros((9, 3), dtype=torch.bfloat16)
    bias = torch.tensor([1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0], dtype=torch.bfloat16)[::2]
    gemm_output = torch.full((9, 4), 10.0, dtype=torch.bfloat16)
    received_bias: list[torch.Tensor] = []

    def fake_cublas_scaled_mm(
        _input: torch.Tensor, _weight: torch.Tensor, *, bias: torch.Tensor, **_kwargs: object
    ) -> torch.Tensor:
        received_bias.append(bias)
        return gemm_output + bias

    quantize = Mock(return_value=(input_tensor, torch.tensor(0.5)))
    cublas_scaled_mm = Mock(side_effect=fake_cublas_scaled_mm)
    cuda_scaled_mm = Mock()
    monkeypatch.setattr(torch.ops.tensorrt_llm, "quantize_e4m3_per_tensor", quantize)
    monkeypatch.setattr(torch.ops.trtllm, "cublas_scaled_mm", cublas_scaled_mm)
    monkeypatch.setattr(torch.ops.trtllm, "cuda_scaled_mm", cuda_scaled_mm)

    output = FP8QDQLinearMethod().apply(_make_module(enable_cuda_core=True), input_tensor, bias)

    torch.testing.assert_close(output, gemm_output + bias)
    assert len(received_bias) == 1
    assert received_bias[0].is_contiguous()
    torch.testing.assert_close(received_bias[0], bias)
    cublas_scaled_mm.assert_called_once()
    cuda_scaled_mm.assert_not_called()


def test_small_m_cuda_scaled_mm_adds_bias_once(monkeypatch: pytest.MonkeyPatch) -> None:
    input_tensor = torch.zeros((8, 3), dtype=torch.bfloat16)
    bias = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.bfloat16)
    gemm_output = torch.full((8, 4), 10.0, dtype=torch.bfloat16)

    def fake_cuda_scaled_mm(
        _input: torch.Tensor, _weight: torch.Tensor, *, bias: torch.Tensor | None, **_kwargs: object
    ) -> torch.Tensor:
        assert bias is None
        return gemm_output

    quantize = Mock(return_value=(input_tensor, torch.tensor(0.5)))
    cublas_scaled_mm = Mock()
    cuda_scaled_mm = Mock(side_effect=fake_cuda_scaled_mm)
    monkeypatch.setattr(torch.ops.tensorrt_llm, "quantize_e4m3_per_tensor", quantize)
    monkeypatch.setattr(torch.ops.trtllm, "cublas_scaled_mm", cublas_scaled_mm)
    monkeypatch.setattr(torch.ops.trtllm, "cuda_scaled_mm", cuda_scaled_mm)

    output = FP8QDQLinearMethod().apply(_make_module(enable_cuda_core=True), input_tensor, bias)

    torch.testing.assert_close(output, gemm_output + bias)
    cuda_scaled_mm.assert_called_once()
    cublas_scaled_mm.assert_not_called()
