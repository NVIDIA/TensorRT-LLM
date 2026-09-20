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
"""Validate residual/prenorm rounding and graph replay at serving widths."""

import pytest
import torch

from tensorrt_llm._torch.models.modeling_musespark_open import MuseSparkOpenGatedResidual
from tensorrt_llm._torch.modules.rms_norm import RMSNorm

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _case(tokens, hidden):
    torch.manual_seed(719)
    residual = torch.randn(tokens, hidden, device="cuda", dtype=torch.float32)
    branch = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
    gate = MuseSparkOpenGatedResidual(hidden, torch.bfloat16).cuda()
    norm = RMSNorm(hidden_size=hidden, eps=1e-5, dtype=torch.bfloat16, use_gemma=True).cuda()
    with torch.no_grad():
        gate.gate.normal_(0, 0.5)
        norm.weight.normal_(0, 0.02)
    gate.cache_derived_state()
    return residual, branch, gate, norm


@pytest.mark.parametrize("with_gain", [False, True])
@pytest.mark.parametrize("tokens", [1, 2, 16, 128, 8192])
@pytest.mark.parametrize("hidden", [256, 8192])
def test_boundary_fusion_preserves_residual_and_prenorm(tokens, hidden, with_gain, monkeypatch):
    monkeypatch.setenv("MUSESPARK_OPEN_FUSED_RESIDUAL", "1")
    with torch.inference_mode():
        residual, branch, gate, norm = _case(tokens, hidden)
        gain = (
            (torch.randn(hidden, device="cuda", dtype=torch.bfloat16) * 0.02 + 1)
            if with_gain
            else None
        )
        expected_residual = gate.add_after_rmsnorm(residual, branch, 1e-5, gain)
        expected_norm = norm(expected_residual.to(torch.bfloat16))
        actual_residual, actual_norm = gate._add_after_rmsnorm_and_norm(
            residual, branch, 1e-5, norm, gain
        )
        torch.testing.assert_close(actual_residual, expected_residual, rtol=2e-5, atol=2e-5)
        torch.testing.assert_close(actual_norm, expected_norm, rtol=0.016, atol=0.001)
        assert actual_residual.dtype == torch.float32
        assert actual_norm.dtype == torch.bfloat16
        relative_l2 = (
            actual_norm.float() - expected_norm.float()
        ).norm() / expected_norm.float().norm()
        assert relative_l2 < 1e-3


def test_boundary_fusion_graph_replay(monkeypatch):
    monkeypatch.setenv("MUSESPARK_OPEN_FUSED_RESIDUAL", "1")
    with torch.inference_mode():
        residual, branch, gate, norm = _case(2, 8192)
        gate._add_after_rmsnorm_and_norm(residual, branch, 1e-5, norm)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = gate._add_after_rmsnorm_and_norm(residual, branch, 1e-5, norm)
        residual.normal_()
        branch.normal_()
        expected = gate._add_after_rmsnorm_and_norm(residual, branch, 1e-5, norm)
        graph.replay()
        torch.cuda.synchronize()
        for got, want in zip(actual, expected):
            torch.testing.assert_close(got, want, rtol=0, atol=0)


@pytest.mark.parametrize("tokens", [1, 2, 16])
def test_three_layer_boundary_carry_and_graph(tokens):
    import contextlib

    from tensorrt_llm._torch.models.modeling_musespark_open import (
        MuseSparkOpenDecoderLayer,
        MuseSparkOpenModel,
        MuseSparkOpenRMSNormWeightFirst,
    )

    with torch.inference_mode():
        torch.manual_seed(951)
        hidden = 256
        model = MuseSparkOpenModel.__new__(MuseSparkOpenModel)
        torch.nn.Module.__init__(model)
        model.dtype = torch.bfloat16
        model.m_emb = 1.0
        model._prefill_sp_enabled = False
        model.embed_norm = RMSNorm(
            hidden_size=hidden, eps=1e-5, dtype=torch.bfloat16, has_weights=False
        ).cuda()
        model.norm = RMSNorm(hidden_size=hidden, eps=1e-5, dtype=torch.bfloat16).cuda()
        layers = []
        for layer_idx in range(3):
            layer = MuseSparkOpenDecoderLayer.__new__(MuseSparkOpenDecoderLayer)
            torch.nn.Module.__init__(layer)
            layer.layer_idx = layer_idx
            layer.dtype = torch.bfloat16
            layer.nvtx = lambda name: contextlib.nullcontext()
            layer.pre_qkv_proj_norm = RMSNorm(
                hidden_size=hidden, eps=1e-5, dtype=torch.bfloat16, use_gemma=True
            ).cuda()
            layer.pre_ffn_norm = RMSNorm(
                hidden_size=hidden, eps=1e-5, dtype=torch.bfloat16, use_gemma=True
            ).cuda()
            layer.post_attn_norm = RMSNorm(
                hidden_size=hidden, eps=1e-5, dtype=torch.bfloat16, has_weights=False
            ).cuda()
            layer.post_ffn_norm = MuseSparkOpenRMSNormWeightFirst(hidden, 1e-5, torch.bfloat16).cuda()
            layer.add_attn = MuseSparkOpenGatedResidual(hidden, torch.bfloat16).cuda()
            layer.add_ffn = MuseSparkOpenGatedResidual(hidden, torch.bfloat16).cuda()
            layer.self_attn = lambda hidden_states, **kwargs: hidden_states * 0.7
            layer.feed_forward = lambda x, metadata: x * 0.9
            for param in layer.parameters():
                param.normal_(0, 0.03)
            layers.append(layer)
        model.layers = torch.nn.ModuleList(layers)
        for module in model.modules():
            if hasattr(module, "cache_derived_state"):
                module.cache_derived_state()
        x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
        residual = model.embed_norm(x).to(torch.float32)
        for layer in model.layers:
            residual = layer(position_ids=None, hidden_states=residual, attn_metadata=None)
        expected = model.norm(residual.to(torch.bfloat16))
        from types import SimpleNamespace

        captures = []
        metadata = SimpleNamespace(
            maybe_capture_hidden_states=lambda idx, state, residual: captures.append(
                (idx, state, residual)
            )
        )
        got = model(attn_metadata=None, inputs_embeds=x, spec_metadata=metadata)
        assert [entry[0] for entry in captures] == [0, 1, 2]
        assert all(
            state.dtype == torch.float32 and residual is None for _, state, residual in captures
        )
        torch.testing.assert_close(got, expected, rtol=0.04, atol=0.004)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            got = model(attn_metadata=None, inputs_embeds=x)
        x.normal_()
        expected = model(attn_metadata=None, inputs_embeds=x)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(got, expected, rtol=0, atol=0)
