# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused residual joins, alone and with the next prenorm: rounding, graphs, layer carry."""

import contextlib
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.models.modeling_musespark_open import (
    MuseSparkOpenDecoderLayer,
    MuseSparkOpenGatedResidual,
    MuseSparkOpenModel,
    MuseSparkOpenRMSNormWeightFirst,
)
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
def test_boundary_fusion_preserves_residual_and_prenorm(tokens, hidden, with_gain):
    with torch.inference_mode():
        residual, branch, gate, norm = _case(tokens, hidden)
        gain = (
            (torch.randn(hidden, device="cuda", dtype=torch.bfloat16) * 0.02 + 1)
            if with_gain
            else None
        )
        expected_residual = gate.add_after_rmsnorm(residual, branch, 1e-5, gain)
        expected_norm = norm(expected_residual.to(torch.bfloat16))
        actual_residual, actual_norm = gate.add_after_rmsnorm_and_norm(
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


# The plain residual join: `gated_add(residual, rmsnorm(branch))`, checked
# against the composition it replaces, including the real `RMSNorm` module on
# the attention side.


def _join_case(tokens, hidden, dtype, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    residual = torch.randn(tokens, hidden, generator=gen, dtype=torch.float32, device="cuda")
    branch = torch.randn(tokens, hidden, generator=gen, dtype=torch.float32, device="cuda").to(
        dtype
    )
    gate = MuseSparkOpenGatedResidual(hidden, dtype).cuda()
    with torch.no_grad():
        gate.gate.normal_(0, 0.5, generator=gen)
    gate.cache_derived_state()
    return residual, branch, gate


def _weight_first_norm(hidden, eps, dtype):
    norm = MuseSparkOpenRMSNormWeightFirst(hidden, eps, dtype).cuda()
    with torch.no_grad():
        norm.weight.normal_(0, 0.3)
    norm.cache_derived_state()
    return norm


def _weightless_norm(eps):
    def norm(x):
        x = x.to(torch.float32)
        return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)).to(torch.bfloat16)

    return norm


@pytest.mark.parametrize("tokens", [1, 5, 64])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_fused_join_matches_the_weightless_norm_path(tokens, dtype):
    """Attention side: `add_attn(h, post_attn_norm(y))` with a real weightless `RMSNorm`."""
    hidden, eps = 256, 1e-6
    residual, branch, gate = _join_case(tokens, hidden, dtype)
    reference_norm = RMSNorm(hidden_size=hidden, eps=eps, dtype=dtype, has_weights=False).cuda()

    fused = gate.add_after_rmsnorm(residual, branch, eps)
    reference = gate(residual, reference_norm(branch))

    assert fused.dtype is torch.float32
    assert (fused - reference).abs().max() < 1e-4 * reference.abs().max()
    torch.testing.assert_close(fused, reference, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("tokens", [1, 5, 64])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_fused_join_matches_the_weight_first_norm_path(tokens, dtype):
    """FFN side, with the cached `1 + w` passed as the gain, as the decoder layer does."""
    hidden, eps = 256, 1e-6
    residual, branch, gate = _join_case(tokens, hidden, dtype)
    norm = _weight_first_norm(hidden, eps, dtype)

    fused = gate.add_after_rmsnorm(residual, branch, eps, norm.gain())
    reference = gate(residual, norm(branch))

    assert (fused - reference).abs().max() < 1e-4 * reference.abs().max()
    torch.testing.assert_close(fused, reference, rtol=2e-5, atol=2e-5)
    # Dropping the gain is a different function, so a kernel ignoring it fails here.
    assert not torch.allclose(fused, gate.add_after_rmsnorm(residual, branch, eps), atol=1e-2)


def test_a_gain_of_another_dtype_takes_the_composition():
    """The composition promotes `gain * branch` across dtypes and skips a round the kernel takes.

    Bit-exact equality with the composition shows the kernel did not run: it
    reduces the variance in another order.
    """
    hidden, eps, tokens = 128, 1e-6, 4
    residual, branch, gate = _join_case(tokens, hidden, torch.float16)
    gain = torch.empty(hidden, dtype=torch.float32, device="cuda").normal_(0, 0.3) + 1.0

    got = gate.add_after_rmsnorm(residual, branch, eps, gain)

    scaled = (gain * branch).to(torch.float32)
    normed = (scaled * torch.rsqrt(scaled.pow(2).mean(-1, keepdim=True) + eps)).to(branch.dtype)
    torch.testing.assert_close(got, gate(residual, normed), atol=0, rtol=0)


def test_the_fused_join_gives_the_same_answer_with_or_without_the_gain_cache():
    hidden, eps, tokens = 128, 1e-6, 4
    residual, branch, gate = _join_case(tokens, hidden, torch.bfloat16)
    norm = MuseSparkOpenRMSNormWeightFirst(hidden, eps, torch.bfloat16).cuda()
    with torch.no_grad():
        norm.weight.normal_(0, 0.3)

    assert norm._weight_plus_one is None
    uncached = gate.add_after_rmsnorm(residual, branch, eps, norm.gain())
    norm.cache_derived_state()
    cached = gate.add_after_rmsnorm(residual, branch, eps, norm.gain())

    torch.testing.assert_close(cached, uncached, atol=0, rtol=0)


def test_the_fused_join_replays_from_a_cuda_graph():
    hidden, eps, tokens = 256, 1e-6, 8
    residual, branch, gate = _join_case(tokens, hidden, torch.bfloat16)
    norm = _weight_first_norm(hidden, eps, torch.bfloat16)

    gate.add_after_rmsnorm(residual, branch, eps, norm.gain())
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out = gate.add_after_rmsnorm(residual, branch, eps, norm.gain())

    fresh_residual, fresh_branch, _ = _join_case(tokens, hidden, torch.bfloat16, seed=7)
    residual.copy_(fresh_residual)
    branch.copy_(fresh_branch)
    graph.replay()
    torch.cuda.synchronize()

    assert torch.isfinite(static_out).all()
    torch.testing.assert_close(static_out, gate(residual, norm(branch)), rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("tokens", [1, 4])
def test_the_fused_join_differs_from_the_composition_only_by_reduction_order(tokens):
    """At the checkpoint's width, within a bound only the variance's reduction order can explain.

    The kernel keeps both of the composition's rounds to the activation dtype
    (the gain product and the normalized branch); one that dropped either would
    land further from the composition than this bound. On larger inputs the
    other reduction order eventually moves a value across a BF16 rounding
    boundary, which the tolerance tests above allow for.
    """
    hidden, eps = 8192, 1e-6
    residual, branch, gate = _join_case(tokens, hidden, torch.bfloat16, seed=3)
    norm = _weight_first_norm(hidden, eps, torch.bfloat16)

    for gain, reference_norm in ((norm.gain(), norm), (None, _weightless_norm(eps))):
        fused = gate.add_after_rmsnorm(residual, branch, eps, gain)
        unfused = gate(residual, reference_norm(branch))
        assert torch.isfinite(fused).all()
        assert (fused - unfused).abs().max() < 1e-6 * unfused.abs().max()


def test_boundary_fusion_graph_replay():
    with torch.inference_mode():
        residual, branch, gate, norm = _case(2, 8192)
        gate.add_after_rmsnorm_and_norm(residual, branch, 1e-5, norm)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = gate.add_after_rmsnorm_and_norm(residual, branch, 1e-5, norm)
        residual.normal_()
        branch.normal_()
        expected = gate.add_after_rmsnorm_and_norm(residual, branch, 1e-5, norm)
        graph.replay()
        torch.cuda.synchronize()
        for got, want in zip(actual, expected):
            torch.testing.assert_close(got, want, rtol=0, atol=0)


def _stub_layer(layer_idx, hidden):
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
    return layer


@pytest.mark.parametrize("tokens", [1, 2, 16])
def test_three_layer_boundary_carry_and_graph(tokens):
    """Each layer hands the next its attention input; the result matches layer by layer."""
    with torch.inference_mode():
        torch.manual_seed(951)
        hidden = 256
        model = MuseSparkOpenModel.__new__(MuseSparkOpenModel)
        torch.nn.Module.__init__(model)
        model.dtype = torch.bfloat16
        model.m_emb = 1.0
        model.embed_norm = RMSNorm(
            hidden_size=hidden, eps=1e-5, dtype=torch.bfloat16, has_weights=False
        ).cuda()
        model.norm = RMSNorm(hidden_size=hidden, eps=1e-5, dtype=torch.bfloat16).cuda()
        model.layers = torch.nn.ModuleList([_stub_layer(index, hidden) for index in range(3)])
        for module in model.modules():
            if hasattr(module, "cache_derived_state"):
                module.cache_derived_state()
        x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
        residual = model.embed_norm(x).to(torch.float32)
        for layer in model.layers:
            residual = layer(position_ids=None, hidden_states=residual, attn_metadata=None)
        expected = model.norm(residual.to(torch.bfloat16))

        captures = []
        metadata = SimpleNamespace(
            maybe_capture_hidden_states=lambda idx, state, residual: captures.append(
                (idx, state, residual)
            )
        )
        got = model(attn_metadata=None, inputs_embeds=x, spec_metadata=metadata)
        # Drafters capture every layer's FP32 residual stream.
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
