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

from tensorrt_llm._torch.modules.attention import Attention
from tensorrt_llm._torch.modules.fused_ops.fused_qk_norm_rope_gate import (
    fused_qkv_gemma_rmsnorm_rope_gate,
    fused_sigmoid_mul,
)


def _make_cos_sin(max_positions: int, rotary_dim: int, theta: float = 1_000_000.0):
    inverse_frequency = 1.0 / (
        theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32, device="cuda") / rotary_dim)
    )
    positions = torch.arange(max_positions, dtype=torch.float32, device="cuda")
    frequency = torch.outer(positions, inverse_frequency)
    return torch.stack((frequency.cos(), frequency.sin()), dim=1).contiguous()


def _reference(
    qkv,
    q_weight,
    k_weight,
    cos_sin,
    positions,
    eps,
    num_q_heads,
    num_kv_heads,
    head_dim,
    rotary_dim,
    mrope_section=None,
):
    num_tokens = qkv.shape[0]
    q_size = num_q_heads * head_dim
    kv_size = num_kv_heads * head_dim
    q_gate, k, v = qkv.split((2 * q_size, kv_size, kv_size), dim=-1)
    q_gate = q_gate.view(num_tokens, num_q_heads, 2 * head_dim)
    q = q_gate[..., :head_dim]
    gate = q_gate[..., head_dim:]
    k = k.view(num_tokens, num_kv_heads, head_dim)

    def _gemma_norm(x, weight):
        x_float = x.float()
        inverse_rms = torch.rsqrt(x_float.square().mean(dim=-1, keepdim=True) + eps)
        return (x_float * inverse_rms * (weight.float() + 1.0)).to(qkv.dtype)

    q = _gemma_norm(q, q_weight)
    k = _gemma_norm(k, k_weight)
    half_rotary = rotary_dim // 2
    if mrope_section is None:
        selected = cos_sin[positions.long()]
        cos = selected[:, 0].unsqueeze(1)
        sin = selected[:, 1].unsqueeze(1)
    else:
        positions = positions.reshape(3, num_tokens).long()
        rotary_offsets = torch.arange(half_rotary, device=qkv.device)
        sections = torch.zeros_like(rotary_offsets)
        sections[(rotary_offsets % 3 == 1) & (rotary_offsets < mrope_section[1] * 3)] = 1
        sections[(rotary_offsets % 3 == 2) & (rotary_offsets < mrope_section[2] * 3)] = 2
        token_offsets = torch.arange(num_tokens, device=qkv.device).unsqueeze(1)
        selected_positions = positions[sections.unsqueeze(0), token_offsets]
        cos = cos_sin[selected_positions, 0, rotary_offsets].unsqueeze(1)
        sin = cos_sin[selected_positions, 1, rotary_offsets].unsqueeze(1)

    def _rope(x):
        x_first = x[..., :half_rotary].float()
        x_second = x[..., half_rotary:rotary_dim].float()
        first = x_first * cos - x_second * sin
        second = x_second * cos + x_first * sin
        return torch.cat((first, second, x[..., rotary_dim:].float()), dim=-1).to(qkv.dtype)

    q = _rope(q).reshape(num_tokens, q_size)
    k = _rope(k).reshape(num_tokens, kv_size)
    return torch.cat((q, k, v), dim=-1), gate


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize(
    "num_tokens,num_q_heads,num_kv_heads,head_dim,rotary_dim",
    (
        (1, 8, 2, 128, 128),
        (7, 8, 2, 256, 128),
        (65, 16, 4, 256, 256),
        (333, 16, 4, 256, 256),
    ),
)
def test_fused_qkv_gemma_rmsnorm_rope_gate_matches_reference(
    dtype, num_tokens, num_q_heads, num_kv_heads, head_dim, rotary_dim
):
    torch.manual_seed(1234)
    q_size = num_q_heads * head_dim
    kv_size = num_kv_heads * head_dim
    width = 2 * q_size + 2 * kv_size
    storage = torch.randn((num_tokens, width + 37), dtype=dtype, device="cuda")
    qkv = storage[:, :width]
    q_weight = torch.randn((head_dim,), dtype=dtype, device="cuda") * 0.1
    k_weight = torch.randn((head_dim,), dtype=dtype, device="cuda") * 0.1
    cos_sin = _make_cos_sin(2048, rotary_dim)
    positions = torch.randint(0, 2048, (num_tokens,), dtype=torch.int32, device="cuda")
    eps = 1e-6

    actual_qkv, actual_gate = fused_qkv_gemma_rmsnorm_rope_gate(
        qkv,
        q_weight,
        k_weight,
        cos_sin,
        positions,
        eps,
        num_q_heads,
        num_kv_heads,
        head_dim,
        rotary_dim,
    )
    expected_qkv, expected_gate = _reference(
        qkv,
        q_weight,
        k_weight,
        cos_sin,
        positions,
        eps,
        num_q_heads,
        num_kv_heads,
        head_dim,
        rotary_dim,
    )

    torch.testing.assert_close(actual_qkv, expected_qkv, atol=0.02, rtol=0.02)
    torch.testing.assert_close(actual_gate, expected_gate, atol=0, rtol=0)
    torch.testing.assert_close(actual_qkv[:, q_size + kv_size :], qkv[:, 2 * q_size + kv_size :])


@pytest.mark.parametrize("inplace", (False, True))
def test_fused_sigmoid_mul_supports_strided_gate(inplace):
    torch.manual_seed(7)
    num_tokens, num_heads, head_dim = 17, 8, 128
    attention = torch.randn((num_tokens, num_heads * head_dim), dtype=torch.bfloat16, device="cuda")
    gate_storage = torch.randn(
        (num_tokens, num_heads, 2 * head_dim), dtype=torch.bfloat16, device="cuda"
    )
    gate = gate_storage[..., :head_dim]
    expected = attention.float() * torch.sigmoid(gate.reshape(num_tokens, -1).float())
    actual = fused_sigmoid_mul(attention.clone(), gate, inplace=inplace)
    torch.testing.assert_close(actual.float(), expected, atol=0.02, rtol=0.02)


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
def test_fused_qkv_gemma_rmsnorm_rope_gate_supports_interleaved_mrope(dtype):
    torch.manual_seed(2026)
    num_tokens, num_q_heads, num_kv_heads = 37, 16, 2
    head_dim, rotary_dim = 256, 64
    mrope_section = (11, 11, 10)
    q_size = num_q_heads * head_dim
    kv_size = num_kv_heads * head_dim
    qkv = torch.randn((num_tokens, 2 * q_size + 2 * kv_size), dtype=dtype, device="cuda")
    q_weight = torch.randn((head_dim,), dtype=dtype, device="cuda") * 0.1
    k_weight = torch.randn((head_dim,), dtype=dtype, device="cuda") * 0.1
    cos_sin = _make_cos_sin(4096, rotary_dim, theta=10_000_000.0)
    base_positions = torch.randint(0, 4000, (num_tokens,), dtype=torch.int32, device="cuda")
    positions = torch.stack(
        (base_positions, base_positions + 3, base_positions + 7), dim=0
    ).unsqueeze(1)
    eps = 1e-6

    actual_qkv, actual_gate = fused_qkv_gemma_rmsnorm_rope_gate(
        qkv,
        q_weight,
        k_weight,
        cos_sin,
        positions,
        eps,
        num_q_heads,
        num_kv_heads,
        head_dim,
        rotary_dim,
        mrope_section,
    )
    expected_qkv, expected_gate = _reference(
        qkv,
        q_weight,
        k_weight,
        cos_sin,
        positions,
        eps,
        num_q_heads,
        num_kv_heads,
        head_dim,
        rotary_dim,
        mrope_section,
    )

    torch.testing.assert_close(actual_qkv, expected_qkv, atol=0.02, rtol=0.02)
    torch.testing.assert_close(actual_gate, expected_gate, atol=0, rtol=0)


@pytest.mark.parametrize("num_tokens", (1, 37, 333))
def test_fused_qkv_gemma_rmsnorm_rope_gate_matches_production_thop(num_tokens):
    torch.manual_seed(9027)
    num_q_heads, num_kv_heads = 16, 2
    head_dim, rotary_dim = 256, 64
    mrope_section = (11, 11, 10)
    q_size = num_q_heads * head_dim
    kv_size = num_kv_heads * head_dim
    qkv = torch.randn(
        (num_tokens, 2 * q_size + 2 * kv_size),
        dtype=torch.bfloat16,
        device="cuda",
    )
    q_weight = torch.randn((head_dim,), dtype=torch.bfloat16, device="cuda") * 0.1
    k_weight = torch.randn((head_dim,), dtype=torch.bfloat16, device="cuda") * 0.1
    cos_sin = _make_cos_sin(4096, rotary_dim, theta=10_000_000.0)
    base_positions = torch.randint(0, 4000, (num_tokens,), dtype=torch.int32, device="cuda")
    positions = torch.stack(
        (base_positions, base_positions + 3, base_positions + 7), dim=0
    ).contiguous()

    q_gate, k, v = qkv.split((2 * q_size, kv_size, kv_size), dim=-1)
    q, expected_gate = [
        value.reshape(num_tokens, -1)
        for value in torch.chunk(q_gate.view(num_tokens, num_q_heads, 2 * head_dim), 2, dim=-1)
    ]
    expected_qkv = torch.cat((q, k, v), dim=-1)
    torch.ops.trtllm.fused_qk_norm_rope(
        expected_qkv,
        num_q_heads,
        num_kv_heads,
        num_kv_heads,
        head_dim,
        rotary_dim,
        1e-6,
        q_weight,
        k_weight,
        10_000_000.0,
        True,
        positions,
        1.0,
        0.0,
        0.0,
        1.0,
        True,
        True,
        True,
        mrope_section[1],
        mrope_section[2],
    )

    actual_qkv, actual_gate = fused_qkv_gemma_rmsnorm_rope_gate(
        qkv,
        q_weight,
        k_weight,
        cos_sin,
        positions,
        1e-6,
        num_q_heads,
        num_kv_heads,
        head_dim,
        rotary_dim,
        mrope_section,
    )

    torch.testing.assert_close(actual_qkv, expected_qkv, atol=0.02, rtol=0.02)
    torch.testing.assert_close(actual_gate.reshape(num_tokens, -1), expected_gate, atol=0, rtol=0)


def test_output_gate_fallback_flattens_fused_gate():
    torch.manual_seed(17)
    attention = torch.randn((5, 256), dtype=torch.bfloat16, device="cuda")
    gate = torch.randn((5, 2, 128), dtype=torch.bfloat16, device="cuda")
    expected = attention * torch.sigmoid(gate.reshape_as(attention))
    actual = Attention.apply_output_gate(None, attention, gate)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_fused_sigmoid_mul_supports_flat_gate():
    torch.manual_seed(11)
    attention = torch.randn((9, 512), dtype=torch.float16, device="cuda")
    gate = torch.randn_like(attention)
    expected = attention.float() * torch.sigmoid(gate.float())
    actual = fused_sigmoid_mul(attention, gate)
    torch.testing.assert_close(actual.float(), expected, atol=0.005, rtol=0.005)
