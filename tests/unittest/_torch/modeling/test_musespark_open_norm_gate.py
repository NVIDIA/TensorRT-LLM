# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused attention output norm and gate: rounding, strided gates, fallbacks and graph replay."""

import pytest
import torch

from tensorrt_llm._torch.models import musespark_open_kernels
from tensorrt_llm._torch.models.modeling_musespark_open import MuseSparkOpenAttention
from tensorrt_llm._torch.modules.rms_norm import RMSNorm

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _attention():
    mod = MuseSparkOpenAttention.__new__(MuseSparkOpenAttention)
    torch.nn.Module.__init__(mod)
    mod.head_dim = 64
    mod._fuse_qk_norm_rope_gate = False
    mod.o_head_norm = RMSNorm(
        hidden_size=64, eps=1e-5, dtype=torch.bfloat16, has_weights=False
    ).cuda()
    return mod


def _reference(mod, x, gate):
    return mod.o_head_norm(x.reshape(-1, 64)).reshape_as(x) * gate.reshape_as(x).sigmoid()


def _inputs(tokens, strided):
    x = torch.randn(tokens, 2048, device="cuda", dtype=torch.bfloat16)
    if strided:
        # The gate half of a per-head interleaved projection.
        gate = torch.randn(tokens, 32, 128, device="cuda", dtype=torch.bfloat16)[:, :, 64:]
    else:
        gate = torch.randn_like(x)
    return x, gate


@pytest.mark.parametrize("tokens", [1, 2, 16, 128])
@pytest.mark.parametrize("strided", [False, True])
@torch.inference_mode()
def test_norm_gate_and_graph_replay(tokens, strided):
    torch.manual_seed(950)
    mod = _attention()
    x, gate = _inputs(tokens, strided)
    got = mod.apply_output_gate(x, gate)
    expected = _reference(mod, x, gate)
    # The fused norm reduces in another order, which can move single values
    # across a BF16 rounding boundary.
    torch.testing.assert_close(got, expected, rtol=0.016, atol=0.002)
    assert (got.float() - expected.float()).norm() / expected.float().norm() < 0.003
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        got = mod.apply_output_gate(x, gate)
    x.normal_()
    gate.normal_()
    eager = mod.apply_output_gate(x, gate)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(got, eager, rtol=0, atol=0)
    reference = _reference(mod, x, gate)
    torch.testing.assert_close(got, reference, rtol=0.016, atol=0.002)


@pytest.mark.parametrize("tokens", [129, 512])
@torch.inference_mode()
def test_unsupported_batches_use_the_reference(tokens, monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("an unsupported batch reached the fused norm and gate")

    monkeypatch.setattr(musespark_open_kernels, "attention_norm_gate", unexpected)
    torch.manual_seed(951)
    mod = _attention()
    x, gate = _inputs(tokens, strided=True)
    torch.testing.assert_close(mod.apply_output_gate(x, gate), _reference(mod, x, gate))
