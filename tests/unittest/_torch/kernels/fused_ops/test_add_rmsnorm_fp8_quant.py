# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fused add + RMSNorm + FP8 quantize vs. flashinfer add + RMSNorm then static quantize."""

import pytest
import torch
from utils.util import skip_pre_ada

import tensorrt_llm  # noqa: F401  (registers torch.ops.tensorrt_llm)
from tensorrt_llm._torch.flashinfer_utils import IS_FLASHINFER_AVAILABLE
from tensorrt_llm._torch.kernels.fused_ops.add_rmsnorm_fp8_quant import add_rmsnorm_fp8_quant
from tensorrt_llm._torch.modules.rms_norm import RMSNorm

pytestmark = [
    skip_pre_ada,
    pytest.mark.skipif(not IS_FLASHINFER_AVAILABLE, reason="flashinfer unavailable"),
]

_EPS = 1e-6


def _static_quant(x, scale):
    return torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(x, scale)[0]


def _flashinfer_add_norm(x, residual, weight, use_gemma):
    import tensorrt_llm._torch.custom_ops.flashinfer_custom_ops  # noqa: F401

    x, residual = x.clone(), residual.clone()
    op = (
        torch.ops.trtllm.flashinfer_gemma_fused_add_rmsnorm
        if use_gemma
        else torch.ops.trtllm.flashinfer_fused_add_rmsnorm
    )
    op(x, residual, weight, _EPS)
    return x, residual


def _inputs(num_tokens, hidden_size, seed=0):
    torch.manual_seed(seed)
    x = torch.randn((num_tokens, hidden_size), dtype=torch.bfloat16, device="cuda")
    residual = 4 * torch.randn_like(x)
    weight = 0.1 * torch.randn(hidden_size, dtype=torch.bfloat16, device="cuda")
    scale = torch.tensor([0.0213], dtype=torch.float32, device="cuda")
    return x, residual, weight, scale


def _assert_same_bits(actual, expected):
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


@pytest.mark.parametrize("hidden_size", (2048, 4096, 5120))
@pytest.mark.parametrize("num_tokens", (1, 7, 512, 2048))
@pytest.mark.parametrize("use_gemma", (True, False))
def test_add_rmsnorm_fp8_quant_matches_unfused(hidden_size, num_tokens, use_gemma):
    x, residual, weight, scale = _inputs(num_tokens, hidden_size)
    ref_normed, ref_residual = _flashinfer_add_norm(x, residual, weight, use_gemma)
    ref_fp8 = _static_quant(ref_normed, scale)

    fp8 = add_rmsnorm_fp8_quant(x, residual, weight, scale, _EPS, use_gemma=use_gemma)

    _assert_same_bits(residual, ref_residual)
    _assert_same_bits(fp8, _static_quant(x, scale))
    torch.testing.assert_close(x, ref_normed, atol=0, rtol=2**-7)
    assert (x != ref_normed).float().mean().item() < 1e-3
    assert (fp8.view(torch.uint8) != ref_fp8.view(torch.uint8)).float().mean().item() < 1e-3


def test_add_rmsnorm_fp8_quant_empty():
    x, residual, weight, scale = _inputs(0, 4096)
    fp8 = add_rmsnorm_fp8_quant(x, residual, weight, scale, _EPS, use_gemma=True)
    assert fp8.shape == x.shape and fp8.dtype == torch.float8_e4m3fn


def _gemma_norm(hidden_size):
    norm = RMSNorm(hidden_size=hidden_size, eps=_EPS, dtype=torch.bfloat16, use_gemma=True)
    with torch.no_grad():
        norm.weight.copy_(0.1 * torch.randn(hidden_size, dtype=torch.bfloat16))
    return norm.cuda()


@pytest.mark.parametrize("num_tokens", (1, 300))
def test_rms_norm_emits_fp8_copy(num_tokens):
    hidden_size = 1024
    norm = _gemma_norm(hidden_size)
    x, residual, _, scale = _inputs(num_tokens, hidden_size, seed=num_tokens)
    ref_normed, ref_residual = _flashinfer_add_norm(x, residual, norm.weight, use_gemma=True)

    norm.fp8_scale = scale
    (fp8, normed), residual_out = norm(x, residual)

    _assert_same_bits(residual_out, ref_residual)
    _assert_same_bits(fp8, _static_quant(normed, scale))
    torch.testing.assert_close(normed, ref_normed, atol=0, rtol=2**-7)


def test_rms_norm_without_residual_or_scale_keeps_plain_output():
    hidden_size = 1024
    norm = _gemma_norm(hidden_size)
    x, residual, _, scale = _inputs(5, hidden_size)
    ref_normed, ref_residual = _flashinfer_add_norm(x, residual, norm.weight, use_gemma=True)

    out, residual_out = norm(x.clone(), residual.clone())
    _assert_same_bits(out, ref_normed)
    _assert_same_bits(residual_out, ref_residual)

    norm.fp8_scale = scale
    assert isinstance(norm(x.clone()), torch.Tensor)
