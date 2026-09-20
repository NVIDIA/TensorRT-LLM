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
"""Validate attention norm/gate rounding and strided gate addressing."""

import pytest
import torch

from tensorrt_llm._torch.models.modeling_musespark_open import MuseSparkOpenAttention
from tensorrt_llm._torch.modules.rms_norm import RMSNorm


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("tokens", [1, 2, 16, 128, 512, 4096, 32768])
@pytest.mark.parametrize("strided", [False, True])
def test_norm_gate_and_graph_replay(tokens, strided):
    torch.manual_seed(950)
    with torch.inference_mode():
        mod = MuseSparkOpenAttention.__new__(MuseSparkOpenAttention)
        torch.nn.Module.__init__(mod)
        mod.head_dim = 64
        mod._fuse_qk_norm_rope_gate = False
        mod._large_norm_gate_enabled = True
        mod.o_head_norm = RMSNorm(
            hidden_size=64, eps=1e-5, dtype=torch.bfloat16, has_weights=False
        ).cuda()
        x = torch.randn(tokens, 2048, device="cuda", dtype=torch.bfloat16)
        gate = (
            torch.randn(tokens, 32, 128, device="cuda", dtype=torch.bfloat16)[:, :, 64:]
            if strided
            else torch.randn_like(x)
        )
        normed = mod.o_head_norm(x.reshape(-1, 64)).reshape_as(x)
        expected = normed * gate.reshape_as(x).sigmoid()
        got = mod.apply_output_gate(x, gate)
        rtol, atol = (0, 0) if tokens >= 512 else (0.016, 0.002)
        torch.testing.assert_close(got, expected, rtol=rtol, atol=atol)
        assert (got.float() - expected.float()).norm() / expected.float().norm() < 0.003
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            got = mod.apply_output_gate(x, gate)
        x.normal_()
        gate.normal_()
        expected = mod.apply_output_gate(x, gate)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(got, expected, rtol=0, atol=0)
        reference = mod.o_head_norm(x.reshape(-1, 64)).reshape_as(x) * gate.reshape_as(x).sigmoid()
        torch.testing.assert_close(got, reference, rtol=rtol, atol=atol)
        assert (got.float() - reference.float()).norm() / reference.float().norm() < 0.003


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("heads", [1, 8, 32])
@pytest.mark.parametrize("strided", [False, True])
@torch.inference_mode()
def test_reference_order_dynamic_shapes(heads, strided):
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.rmsnorm_gate import (
        _compile_norm_gate,
        rmsnorm_gate,
    )

    norm = RMSNorm(hidden_size=64, eps=1e-5, dtype=torch.bfloat16, has_weights=False).cuda()
    compiled = _compile_norm_gate(heads)
    misses = _compile_norm_gate.cache_info().misses
    for tokens in (513, 4099, 515):
        torch.manual_seed(5300 + tokens)
        x = torch.randn(tokens, heads * 64, device="cuda", dtype=torch.bfloat16)
        gate = torch.randn(
            tokens, heads, 128 if strided else 64, device="cuda", dtype=torch.bfloat16
        )[..., -64:]
        actual = rmsnorm_gate(x, gate, 1e-5)
        expected = norm(x.reshape(-1, 64)).reshape_as(x) * gate.reshape_as(x).sigmoid()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = rmsnorm_gate(x, gate, 1e-5)
        x.normal_()
        gate.normal_().mul_(3)
        expected = norm(x.reshape(-1, 64)).reshape_as(x) * gate.reshape_as(x).sigmoid()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert _compile_norm_gate(heads) is compiled
        assert _compile_norm_gate.cache_info().misses == misses


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_reference_order_all_finite_bf16_gates():
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.rmsnorm_gate import rmsnorm_gate

    gates = torch.arange(65536, device="cuda").to(torch.uint16).view(torch.bfloat16)
    gates = gates[torch.isfinite(gates)].reshape(-1, 64)
    x = torch.ones_like(gates)
    actual = rmsnorm_gate(x, gates, 1e-5)
    torch.testing.assert_close(actual, gates.sigmoid(), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("unaligned", ["input_offset", "gate_offset", "gate_stride"])
@torch.inference_mode()
def test_unaligned_large_gate_uses_fallback(unaligned, monkeypatch):
    import tensorrt_llm._torch.cute_dsl_kernels.blackwell.rmsnorm_gate as kernel

    def unexpected_kernel(*args, **kwargs):
        pytest.fail("Unaligned tensors must use the original norm/gate path")

    monkeypatch.setattr(kernel, "rmsnorm_gate", unexpected_kernel)
    mod = MuseSparkOpenAttention.__new__(MuseSparkOpenAttention)
    torch.nn.Module.__init__(mod)
    mod.head_dim = 64
    mod._fuse_qk_norm_rope_gate = False
    mod.o_head_norm = RMSNorm(
        hidden_size=64, eps=1e-5, dtype=torch.bfloat16, has_weights=False
    ).cuda()
    x = torch.randn(512, 2048, device="cuda", dtype=torch.bfloat16)
    gate = torch.randn_like(x)
    if unaligned == "input_offset":
        x = torch.randn(512 * 2048 + 1, device="cuda", dtype=torch.bfloat16)[1:].view_as(x)
    elif unaligned == "gate_offset":
        gate = torch.randn(512 * 2048 + 1, device="cuda", dtype=torch.bfloat16)[1:].view_as(gate)
    else:
        gate = torch.randn(512, 2049, device="cuda", dtype=torch.bfloat16)[:, :2048]
    if unaligned == "input_offset":
        # The reference FlashInfer norm also rejects misaligned input data.
        # Enabling this optimization must preserve that validation rather
        # than dispatching a vectorized kernel with invalid assumptions.
        for enabled in (False, True):
            mod._large_norm_gate_enabled = enabled
            with pytest.raises(ValueError, match="Misaligned Tensor data"):
                mod.apply_output_gate(x, gate)
        return
    mod._large_norm_gate_enabled = False
    expected = mod.apply_output_gate(x, gate)
    mod._large_norm_gate_enabled = True
    actual = mod.apply_output_gate(x, gate)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
