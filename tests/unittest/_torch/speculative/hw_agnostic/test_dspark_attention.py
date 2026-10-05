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
"""DSpark attention math and managed-page addressing tests."""

import types
from unittest.mock import Mock

import pytest
import torch

import tensorrt_llm._torch.models.modeling_dspark as modeling_dspark
from tensorrt_llm._torch.models.modeling_dspark import (
    DSv4DSparkDraftModel,
    apply_dspark_rotary,
    precompute_dspark_freqs_cis,
)

__extra_import_path__ = [".."]


# The captured-context attention primitives were folded into
# modeling_dspark; keep the historical alias so monkeypatch targets
# below read unchanged.
dspark_attention = modeling_dspark


def test_rmsnorm_rope_fallback_applies_weight_without_rmsnorm(monkeypatch):
    monkeypatch.setattr(dspark_attention, "IS_CUTLASS_DSL_AVAILABLE", False)
    torch.manual_seed(11)
    x = torch.randn(2, 3, 64, dtype=torch.bfloat16)
    weight = torch.randn(64, dtype=torch.bfloat16)
    freqs_cis = torch.empty(2, 3, 1, dtype=torch.complex64)

    actual = dspark_attention._rmsnorm_rope_batched(
        x,
        weight,
        1e-6,
        0,
        freqs_cis,
        apply_weight=True,
        apply_rmsnorm=False,
    )

    expected = (x.float() * weight.float()).to(x.dtype)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_rmsnorm_rope_skips_frequency_view_on_unsupported_arch(monkeypatch):
    monkeypatch.setattr(dspark_attention, "IS_CUTLASS_DSL_AVAILABLE", True)
    monkeypatch.setattr(dspark_attention, "is_sm_100f", lambda: False)
    view_as_real = Mock(side_effect=AssertionError("frequency view should be skipped"))
    monkeypatch.setattr(torch, "view_as_real", view_as_real)
    x = torch.randn(2, 3, 64, dtype=torch.bfloat16)

    actual = dspark_attention._rmsnorm_rope_batched(
        x,
        torch.ones(64, dtype=torch.bfloat16),
        1e-6,
        0,
        torch.empty(2, 3, 1, dtype=torch.complex64),
        apply_weight=False,
        apply_rmsnorm=False,
    )

    assert actual is x
    view_as_real.assert_not_called()


def test_batched_attention_rejects_mismatched_window_size():
    from test_dspark_cute_dsl_attention import _make_attn_inputs, _run

    g = _make_attn_inputs(device="cpu")
    g["window"] = 64
    with pytest.raises(ValueError, match="128-token window"):
        _run(g)


def test_rope_table_is_cached_once_per_device():
    model = types.SimpleNamespace(
        _attn_params={"rope_head_dim": 16},
        _freqs_cap=64,
        _rope_theta=10000.0,
        _freqs_table_cache={},
    )

    first = DSv4DSparkDraftModel._dspark_freqs_table(model, torch.device("cpu"))
    second = DSv4DSparkDraftModel._dspark_freqs_table(model, torch.device("cpu"))

    assert first.data_ptr() == second.data_ptr()
    assert len(model._freqs_table_cache) == 1
    positions = torch.tensor([1, 17, 63])
    expected = precompute_dspark_freqs_cis(16, 64, rope_theta=10000.0)
    torch.testing.assert_close(first[positions], expected[positions])


def test_dspark_block_uses_stage_id_as_attention_layer_idx(monkeypatch):
    captured = {}

    def fake_decoder_layer_init(
        self,
        model_config,
        layer_idx,
        aux_stream_dict,
        attention_layer_idx=None,
        mapping_with_cp=None,
        disable_post_moe_fusion=False,
    ):
        torch.nn.Module.__init__(self)
        self.model_config = model_config
        self.config = model_config.pretrained_config
        self.layer_idx = layer_idx
        captured.update(
            layer_idx=layer_idx,
            attention_layer_idx=attention_layer_idx,
            aux_stream_dict=aux_stream_dict,
            mapping_with_cp=mapping_with_cp,
            disable_post_moe_fusion=disable_post_moe_fusion,
        )

    monkeypatch.setattr(
        modeling_dspark.DeepseekV4DecoderLayer,
        "__init__",
        fake_decoder_layer_init,
    )
    model_config = types.SimpleNamespace(
        pretrained_config=types.SimpleNamespace(vocab_size=128, hc_mult=2),
        spec_config=None,
    )

    block = modeling_dspark.DSv4DSparkBlock(
        model_config,
        layer_idx=10,
        aux_stream_dict={},
        stage_id=1,
        num_stages=3,
        num_capture_layers=0,
    )

    assert block.layer_idx == captured["layer_idx"] == 10
    assert captured["attention_layer_idx"] == block.stage_id == 1
    assert captured["disable_post_moe_fusion"] is True


@pytest.mark.parametrize("enable_fused_hc", [True, False])
def test_forward_stage_honors_enable_fused_hc(monkeypatch, enable_fused_hc):
    """The draft stage must use the inherited fused-HC rollback setting."""
    torch.manual_seed(71)
    num_requests, block_size, hc_mult, hidden_size = 1, 2, 2, 3
    h = torch.randn(num_requests, block_size, hc_mult, hidden_size)
    attention_input = torch.randn(num_requests, block_size, hidden_size)
    attention_output = torch.randn_like(attention_input)
    mid_residual = torch.randn_like(h)
    attention_post_mix = torch.randn(num_requests, block_size, hc_mult, 1)
    attention_comb_mix = torch.randn(num_requests, block_size, hc_mult, hc_mult)
    ffn_post_mix = torch.randn_like(attention_post_mix)
    ffn_comb_mix = torch.randn_like(attention_comb_mix)
    raw_ffn_input = torch.randn_like(attention_input)
    normed_ffn_input = torch.randn_like(attention_input)
    moe_output = torch.randn(num_requests * block_size, hidden_size)
    final_h = torch.randn_like(h)
    events = []

    def record(name, result):
        def call(*args, **kwargs):
            events.append(name)
            return result

        return call

    monkeypatch.setattr(
        modeling_dspark,
        "dspark_attention_forward",
        Mock(return_value=attention_output),
    )

    hc_attn = types.SimpleNamespace(
        pre_mapping=Mock(return_value=(attention_post_mix, attention_comb_mix, attention_input)),
        post_mapping=Mock(side_effect=record("attention_post", mid_residual)),
    )
    hc_ffn = types.SimpleNamespace(
        fused_hc=Mock(
            side_effect=record(
                "fused",
                (mid_residual, ffn_post_mix, ffn_comb_mix, normed_ffn_input),
            )
        ),
        pre_mapping=Mock(
            side_effect=record("ffn_pre", (ffn_post_mix, ffn_comb_mix, raw_ffn_input))
        ),
        post_mapping=Mock(side_effect=record("ffn_post", final_h)),
    )
    post_attention_layernorm = Mock(side_effect=record("ffn_norm", normed_ffn_input))
    post_attention_layernorm.weight = torch.ones(hidden_size)
    post_attention_layernorm.variance_epsilon = 1e-6
    stage = types.SimpleNamespace(
        enable_fused_hc=enable_fused_hc,
        hc_attn=hc_attn,
        hc_ffn=hc_ffn,
        input_layernorm=Mock(side_effect=lambda tensor: tensor),
        post_attention_layernorm=post_attention_layernorm,
        mlp=Mock(return_value=moe_output),
        _dspark_attn={},
    )
    model = types.SimpleNamespace(
        use_real_mla=False,
        _attn_params={"window_size": 2, "head_dim": 1},
        model_config=types.SimpleNamespace(
            mapping=types.SimpleNamespace(enable_attention_dp=False, tp_size=8)
        ),
    )

    actual = DSv4DSparkDraftModel._forward_stage(
        model,
        stage,
        h,
        torch.randn(num_requests, hidden_size),
        torch.ones(num_requests, dtype=torch.long),
        torch.empty(0),
        torch.zeros(num_requests, block_size, dtype=torch.long),
        torch.empty(1, 128, 1),
        torch.zeros(num_requests, 1, dtype=torch.int32),
        torch.full((num_requests,), 128, dtype=torch.long),
        torch.ones(num_requests, dtype=torch.long),
    )

    assert actual is final_h
    torch.testing.assert_close(
        stage.mlp.call_args.args[0],
        normed_ffn_input.reshape(num_requests * block_size, hidden_size),
    )
    # Non-attention-DP multi-GPU (enable_attention_dp=False, tp_size>1): the draft
    # MoE must all-reduce its TP-sharded output, mirroring the target MoE. The
    # attention-DP and single-GPU paths keep it disabled.
    assert stage.mlp.call_args.kwargs["final_all_reduce_params"].enable_allreduce is True
    if enable_fused_hc:
        assert events == ["fused", "ffn_post"]
        hc_ffn.fused_hc.assert_called_once()
        hc_attn.post_mapping.assert_not_called()
        hc_ffn.pre_mapping.assert_not_called()
        post_attention_layernorm.assert_not_called()
        fused_kwargs = hc_ffn.fused_hc.call_args.kwargs
        assert fused_kwargs["norm_weight"] is post_attention_layernorm.weight
        assert fused_kwargs["norm_eps"] == post_attention_layernorm.variance_epsilon
    else:
        assert events == ["attention_post", "ffn_pre", "ffn_norm", "ffn_post"]
        hc_ffn.fused_hc.assert_not_called()
        hc_attn.post_mapping.assert_called_once()
        hc_ffn.pre_mapping.assert_called_once_with(mid_residual)
        post_attention_layernorm.assert_called_once_with(raw_ffn_input)


def _ref_precompute_freqs_cis(dim, seqlen, base):
    """DeepSpec precompute_freqs_cis with original_seq_len == 0 (no YaRN)."""
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def _ref_apply_rotary_emb(x, freqs_cis, inverse=False):
    """DeepSpec apply_rotary_emb (returns a fresh tensor instead of in-place)."""
    xc = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    if xc.ndim == 3:
        fc = freqs_cis.view(1, xc.size(1), xc.size(-1))
    else:
        fc = freqs_cis.view(1, xc.size(1), 1, xc.size(-1))
    return torch.view_as_real(xc * fc).flatten(-2).to(x.dtype)


@pytest.mark.parametrize("rope_head_dim,seqlen", [(64, 16), (64, 1), (128, 8)])
def test_precompute_freqs_cis_matches_reference(rope_head_dim, seqlen):
    got = precompute_dspark_freqs_cis(rope_head_dim, seqlen, rope_theta=10000.0)
    ref = _ref_precompute_freqs_cis(rope_head_dim, seqlen, 10000.0)
    torch.testing.assert_close(got, ref)


@pytest.mark.parametrize("ndim", [3, 4])
def test_apply_rotary_matches_reference(ndim):
    torch.manual_seed(0)
    b, s, h, rd = 2, 5, 4, 64
    x = torch.randn(b, s, h, rd) if ndim == 4 else torch.randn(b, s, rd)
    fc = precompute_dspark_freqs_cis(rd, s)
    got = apply_dspark_rotary(x, fc)
    ref = _ref_apply_rotary_emb(x, fc)
    torch.testing.assert_close(got, ref)


@pytest.mark.parametrize("ndim", [3, 4])
def test_apply_rotary_inverse_roundtrip(ndim):
    """De-rotation (inverse) must undo the forward rotation (property test)."""
    torch.manual_seed(1)
    b, s, h, rd = 2, 6, 3, 64
    x = torch.randn(b, s, h, rd) if ndim == 4 else torch.randn(b, s, rd)
    fc = precompute_dspark_freqs_cis(rd, s)
    roundtrip = apply_dspark_rotary(apply_dspark_rotary(x, fc), fc, inverse=True)
    torch.testing.assert_close(roundtrip, x, rtol=1e-5, atol=1e-5)
