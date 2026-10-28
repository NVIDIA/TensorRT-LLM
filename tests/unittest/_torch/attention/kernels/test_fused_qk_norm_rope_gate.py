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

import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

import tensorrt_llm._torch.attention.kernels.fused_qk_norm_rope_gate as _kernel_module
from tensorrt_llm._torch.attention.attention import Attention
from tensorrt_llm._torch.attention.kernels.fused_qk_norm_rope_gate import (
    fused_qkv_gemma_rmsnorm_rope_gate,
    fused_sigmoid_mul_inplace,
)


@pytest.fixture(params=(False, True), ids=("stock", "tiled"))
def qkv_launch(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    # Exercise both kernels on every GPU, independently of production eligibility.
    monkeypatch.setattr(_kernel_module, "_use_tiled_qkv_rope", lambda *args: request.param)


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
@pytest.mark.usefixtures("qkv_launch")
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


def test_fused_sigmoid_mul_inplace_supports_strided_gate():
    torch.manual_seed(7)
    num_tokens, num_heads, head_dim = 17, 8, 128
    attention = torch.randn((num_tokens, num_heads * head_dim), dtype=torch.bfloat16, device="cuda")
    gate_storage = torch.randn(
        (num_tokens, num_heads, 2 * head_dim), dtype=torch.bfloat16, device="cuda"
    )
    gate = gate_storage[..., :head_dim]
    expected = attention.float() * torch.sigmoid(gate.reshape(num_tokens, -1).float())
    actual = attention.clone()
    fused_sigmoid_mul_inplace(actual, gate)
    torch.testing.assert_close(actual.float(), expected, atol=0.02, rtol=0.02)


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.usefixtures("qkv_launch")
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
@pytest.mark.usefixtures("qkv_launch")
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


def test_fused_sigmoid_mul_inplace_supports_flat_gate():
    torch.manual_seed(11)
    attention = torch.randn((9, 512), dtype=torch.float16, device="cuda")
    gate = torch.randn_like(attention)
    expected = attention.float() * torch.sigmoid(gate.float())
    actual = attention.clone()
    fused_sigmoid_mul_inplace(actual, gate)
    torch.testing.assert_close(actual.float(), expected, atol=0.005, rtol=0.005)


@pytest.mark.usefixtures("qkv_launch")
@pytest.mark.parametrize("backend", ("eager", "inductor"))
def test_fused_qkv_gemma_rmsnorm_rope_gate_supports_torch_compile(backend: str) -> None:
    torch.manual_seed(2027)
    num_tokens, num_q_heads, num_kv_heads = 7, 8, 2
    head_dim, rotary_dim = 128, 128
    q_size = num_q_heads * head_dim
    kv_size = num_kv_heads * head_dim
    qkv = torch.randn(
        (num_tokens, 2 * q_size + 2 * kv_size),
        dtype=torch.bfloat16,
        device="cuda",
    )
    q_weight = torch.randn((head_dim,), dtype=torch.bfloat16, device="cuda") * 0.1
    k_weight = torch.randn((head_dim,), dtype=torch.bfloat16, device="cuda") * 0.1
    cos_sin = _make_cos_sin(128, rotary_dim)
    positions = torch.arange(num_tokens, dtype=torch.int32, device="cuda")

    compiled = torch.compile(
        fused_qkv_gemma_rmsnorm_rope_gate,
        backend=backend,
        fullgraph=True,
    )
    actual = compiled(
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
    )
    expected = fused_qkv_gemma_rmsnorm_rope_gate(
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
    )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_fused_sigmoid_mul_inplace_supports_torch_compile():
    attention = torch.randn((7, 512), dtype=torch.bfloat16, device="cuda")
    gate = torch.randn((7, 4, 128), dtype=torch.bfloat16, device="cuda")
    expected = attention.clone()
    fused_sigmoid_mul_inplace(expected, gate)

    def apply_gate(attention_output, gate_input):
        fused_sigmoid_mul_inplace(attention_output, gate_input)
        return attention_output

    compiled = torch.compile(apply_gate, backend="eager", fullgraph=True)
    actual = compiled(attention.clone(), gate)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("num_tokens", (0, 1, 7, 8, 9, 255, 256, 257, 513))
@pytest.mark.parametrize(
    "q_heads,kv_heads,head_dim,rotary_dim",
    ((8, 2, 128, 128), (16, 2, 256, 64), (8, 1, 256, 64), (8, 2, 192, 96), (8, 2, 192, 192)),
)
def test_tiled_qkv_rope_matches_stock_and_replays(
    dtype: torch.dtype, num_tokens: int, q_heads: int, kv_heads: int, head_dim: int, rotary_dim: int
) -> None:
    torch.manual_seed(2028)
    width = 2 * (q_heads + kv_heads) * head_dim
    qkv = torch.randn((num_tokens, width + 37), dtype=dtype, device="cuda")[:, :width]
    qw = torch.randn(head_dim, dtype=dtype, device="cuda") * 0.1
    kw = torch.randn_like(qw) * 0.1
    table = _make_cos_sin(128, rotary_dim)
    positions = torch.randint(0, 128, (3, num_tokens), dtype=torch.int64, device="cuda")
    half = rotary_dim // 2
    sections = (half - 2 * (half // 3), half // 3, half // 3)

    def run() -> tuple[torch.Tensor, torch.Tensor]:
        return fused_qkv_gemma_rmsnorm_rope_gate(
            qkv,
            qw,
            kw,
            table,
            positions,
            1e-6,
            q_heads,
            kv_heads,
            head_dim,
            rotary_dim,
            sections,
        )

    with patch.object(_kernel_module, "_use_tiled_qkv_rope", return_value=False):
        expected = run()
    with patch.object(_kernel_module, "_use_tiled_qkv_rope", return_value=True):
        actual = run()
        torch.testing.assert_close(actual[0], expected[0], atol=0.02, rtol=0.02)
        torch.testing.assert_close(actual[1], expected[1], atol=0, rtol=0)
        v_start = (q_heads + kv_heads) * head_dim
        torch.testing.assert_close(actual[0][:, v_start:], expected[0][:, v_start:], atol=0, rtol=0)
        if num_tokens == 0:
            return
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = run()
        for _ in range(3):
            qkv.add_(0.125)
            expected = run()
            graph.replay()
            torch.testing.assert_close(captured, expected, atol=0, rtol=0)


@pytest.mark.parametrize("enabled", (False, True))
def test_qkv_rope_tiling_dispatch_is_architecture_independent(
    monkeypatch: pytest.MonkeyPatch, enabled: bool
) -> None:
    monkeypatch.setattr(_kernel_module, "_QKV_ROPE_TILING_ENABLED", enabled)
    qkv = torch.empty((256, 1024), device="cuda")
    with patch.object(torch.cuda, "get_device_properties") as query:
        assert _kernel_module._use_tiled_qkv_rope(qkv, 16, 2, 256, 64) == enabled
        query.assert_not_called()


@pytest.mark.parametrize(
    "num_tokens,head_dim,rotary_dim",
    ((0, 256, 64), (255, 256, 64), (256, 128, 64), (256, 192, 96), (256, 256, 256)),
)
def test_qkv_rope_tiling_excluded_shapes(
    monkeypatch: pytest.MonkeyPatch, num_tokens: int, head_dim: int, rotary_dim: int
) -> None:
    monkeypatch.setattr(_kernel_module, "_QKV_ROPE_TILING_ENABLED", True)
    qkv = torch.empty((num_tokens, 1024), device="cuda")
    assert not _kernel_module._use_tiled_qkv_rope(qkv, 16, 2, head_dim, rotary_dim)


@pytest.mark.parametrize("enabled", (False, True))
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("q_heads,dim,rotary", ((16, 256, 64), (8, 128, 128)))
@pytest.mark.parametrize("tokens", (257, 32737, 32768, 32769))
def test_qkv_rope_tiling_production_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    enabled: bool,
    dtype: torch.dtype,
    q_heads: int,
    dim: int,
    rotary: int,
    tokens: int,
) -> None:
    monkeypatch.setattr(_kernel_module, "_QKV_ROPE_TILING_ENABLED", enabled)
    kv_heads = 2
    qkv = torch.randn((tokens, 2 * (q_heads + kv_heads) * dim), device="cuda", dtype=dtype)
    weight = torch.randn(dim, device="cuda", dtype=dtype) * 0.1
    table = _make_cos_sin(tokens, rotary)
    positions = torch.arange(tokens, device="cuda", dtype=torch.int32)

    def run() -> tuple[torch.Tensor, torch.Tensor]:
        return fused_qkv_gemma_rmsnorm_rope_gate(
            qkv, weight, weight, table, positions, 1e-6, q_heads, kv_heads, dim, rotary
        )

    run()
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as profile:
        actual = run()
        torch.cuda.synchronize()
    trace_file = tmp_path / "dispatch.json"
    profile.export_chrome_trace(str(trace_file))
    launches = [
        event
        for event in json.loads(trace_file.read_text())["traceEvents"]
        if event.get("cat") == "kernel"
        and "_fused_qkv_gemma_rmsnorm_rope_gate_kernel" in event["name"]
    ]
    block_tokens = 8 if enabled else 1
    assert len(launches) == 1
    assert launches[0]["args"]["grid"] == [
        (tokens + block_tokens - 1) // block_tokens,
        q_heads + 2 * kv_heads,
        1,
    ]
    expected = _reference(
        qkv, weight, weight, table, positions, 1e-6, q_heads, kv_heads, dim, rotary
    )
    torch.testing.assert_close(actual[0], expected[0], atol=0.02, rtol=0.02)
    torch.testing.assert_close(actual[1], expected[1], atol=0, rtol=0)
    v_start = (q_heads + kv_heads) * dim
    torch.testing.assert_close(actual[0][:, v_start:], expected[0][:, v_start:], atol=0, rtol=0)
    if tokens == 32737:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = run()
        qkv.add_(0.125)
        expected = run()
        graph.replay()
        torch.testing.assert_close(captured, expected, atol=0, rtol=0)


@pytest.mark.parametrize(
    "tokens,q_heads,kv_heads,expected",
    (
        (256, 16, 2, True),
        (256, 8, 1, False),
        (512, 8, 1, True),
        (256, 32, 4, True),
        (512, 4, 1, True),
        (1024, 1, 1, True),
    ),
)
def test_qkv_rope_tiling_accepts_tp_head_counts(
    monkeypatch: pytest.MonkeyPatch, tokens: int, q_heads: int, kv_heads: int, expected: bool
) -> None:
    monkeypatch.setattr(_kernel_module, "_QKV_ROPE_TILING_ENABLED", True)
    qkv = torch.empty((tokens, 2 * (q_heads + kv_heads) * 256), device="cuda", dtype=torch.bfloat16)
    assert _kernel_module._use_tiled_qkv_rope(qkv, q_heads, kv_heads, 256, 64) == expected


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("q_heads,head_dim,rotary_dim", ((8, 128, 128), (16, 256, 64)))
def test_qkv_rope_tiling_batch_boundaries(
    monkeypatch: pytest.MonkeyPatch,
    dtype: torch.dtype,
    q_heads: int,
    head_dim: int,
    rotary_dim: int,
) -> None:
    monkeypatch.setattr(_kernel_module, "_QKV_ROPE_TILING_ENABLED", True)
    torch.manual_seed(2030)
    kv_heads, prefix_tokens = 2, 255
    qkv = torch.randn((32769, 2 * (q_heads + kv_heads) * head_dim), device="cuda", dtype=dtype)
    weight = torch.randn(head_dim, device="cuda", dtype=dtype) * 0.1
    table = _make_cos_sin(128, rotary_dim)
    positions = torch.randint(0, 128, (3, 32769), device="cuda", dtype=torch.int32)
    half = rotary_dim // 2
    sections = (half - 2 * (half // 3), half // 3, half // 3)
    reference = None
    for tokens in (255, 256, 257, 16384, 16385, 32737, 32768, 32769):
        actual = fused_qkv_gemma_rmsnorm_rope_gate(
            qkv[:tokens],
            weight,
            weight,
            table,
            positions[:, :tokens].contiguous(),
            1e-6,
            q_heads,
            kv_heads,
            head_dim,
            rotary_dim,
            sections,
        )
        prefix = tuple(value[:prefix_tokens].clone() for value in actual)
        if reference is None:
            reference = prefix
        else:
            # The opt-in permits Q/K rounding changes at dispatch boundaries.
            torch.testing.assert_close(prefix[0], reference[0], atol=0.02, rtol=0.02)
            torch.testing.assert_close(prefix[1], reference[1], atol=0, rtol=0)
            v_start = (q_heads + kv_heads) * head_dim
            torch.testing.assert_close(
                prefix[0][:, v_start:], reference[0][:, v_start:], atol=0, rtol=0
            )
