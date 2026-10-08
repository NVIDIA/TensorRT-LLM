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

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm.rocm import ops

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("width", [1, 31, 32, 33, 255, 256, 257, 769])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_norms_tail_shapes_and_residual_rounding(width, dtype) -> None:
    x = torch.randn(3, width).to(dtype)
    residual = torch.randn_like(x)
    weight = torch.randn(width).to(dtype)
    bias = torch.randn_like(weight)
    before = x.clone()
    actual, updated = ops.fused_add_rms_norm(x, residual, weight)
    assert torch.equal(updated, x + residual)
    torch.testing.assert_close(actual, ops.rms_norm(x + residual, weight))
    expected = F.layer_norm(x.float(), (width,), weight.float(), bias.float()).to(dtype)
    torch.testing.assert_close(ops.layer_norm(x, weight, bias), expected)
    assert torch.equal(x, before)


@pytest.mark.parametrize("interleaved", [False, True])
def test_rotary_partial_tail_and_zero_angles(interleaved) -> None:
    x = torch.randn(5, 2, 34)
    cos, sin = torch.ones(5, 13), torch.zeros(5, 13)
    assert torch.equal(ops.rotary_embedding(x, cos, sin, interleaved), x)
    angles = torch.randn_like(cos)
    result = ops.rotary_embedding(x, angles.cos(), angles.sin(), interleaved)
    assert torch.equal(result[..., 26:], x[..., 26:])
    torch.testing.assert_close(
        result[..., :26].square().sum(-1), x[..., :26].square().sum(-1), rtol=2e-6, atol=2e-6
    )


def test_attention_matches_sdpa_and_supports_right_aligned_gqa() -> None:
    q, k, v = [torch.randn(2, 4, 5, 16) for _ in range(3)]
    torch.testing.assert_close(
        ops.attention(q, k, v), F.scaled_dot_product_attention(q, k, v), rtol=1e-5, atol=1e-6
    )
    torch.testing.assert_close(
        ops.attention(q, k, v, causal=True),
        F.scaled_dot_product_attention(q, k, v, is_causal=True),
        rtol=1e-5,
        atol=1e-6,
    )
    q = q[:, :, -1:]
    k, v = k[:, :2], v[:, :2]
    expected = F.scaled_dot_product_attention(
        q, k.repeat_interleave(2, 1), v.repeat_interleave(2, 1)
    )
    torch.testing.assert_close(ops.attention(q, k, v, causal=True), expected, rtol=1e-5, atol=1e-6)
    mask = torch.zeros((2, 1, 1, 5), dtype=torch.bool)
    assert torch.equal(ops.attention(q, k, v, mask=mask), torch.zeros_like(q))


@pytest.mark.parametrize("epsilon", [0, -1, float("inf"), float("nan")])
def test_invalid_normalization_epsilon(epsilon) -> None:
    with pytest.raises(ValueError, match="epsilon"):
        ops.rms_norm(torch.ones(2, 3), torch.ones(3), epsilon)


def test_operator_input_validation() -> None:
    with pytest.raises(ValueError, match="weight"):
        ops.rms_norm(torch.ones(2, 3), torch.ones(2))
    with pytest.raises(ValueError, match="identical"):
        ops.gated_activation(torch.ones(3), torch.ones(4))
    with pytest.raises(ValueError, match="FP32"):
        ops.rotary_embedding(torch.ones(2, 1, 4), torch.ones(2, 2).half(), torch.zeros(2, 2).half())
    with pytest.raises(ValueError, match="GQA"):
        ops.attention(torch.ones(1, 3, 2, 4), torch.ones(1, 2, 2, 4), torch.ones(1, 2, 2, 4))


@pytest.mark.parametrize("epsilon", [1e-50, 1e40])
def test_normalization_epsilon_must_be_positive_finite_fp32(epsilon) -> None:
    with pytest.raises(ValueError, match="FP32"):
        ops.rms_norm(torch.ones(1, 4), torch.ones(4), epsilon)


def test_native_rms_adapter_preserves_hidden_states_keyword(monkeypatch) -> None:
    from transformers.models.llama.modeling_llama import LlamaRMSNorm

    from tensorrt_llm.rocm import ops

    norm = LlamaRMSNorm(4)
    hidden = torch.randn(2, 4)
    expected = norm(hidden_states=hidden)
    reference = ops.rms_norm
    monkeypatch.setattr(
        ops, "rms_norm", lambda input, weight, epsilon, backend: reference(input, weight, epsilon)
    )
    assert ops.apply_native_norms(norm) == 1
    torch.testing.assert_close(norm(hidden_states=hidden), expected)
