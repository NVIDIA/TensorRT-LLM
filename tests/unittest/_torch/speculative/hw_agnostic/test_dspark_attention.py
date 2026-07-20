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
"""Unit tests for the DSpark captured-context attention primitives (CPU).

The two primitives are validated against fully independent computations:
``dspark_sparse_attn`` vs (a) a per-element loop reference of the kernel formula
and (b) ``torch.nn.functional.scaled_dot_product_attention`` for the no-sink,
all-valid case; ``get_dspark_topk_idxs`` vs the reference index formula.
"""

import types
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F

import tensorrt_llm._torch.models.modeling_dspark as modeling_dspark
from tensorrt_llm._torch.models.dspark.attention import (
    apply_dspark_rotary,
    dspark_attention_forward,
    dspark_sparse_attn,
    get_dspark_topk_idxs,
    precompute_dspark_freqs_cis,
)
from tensorrt_llm._torch.models.modeling_dspark import DSparkDraftModel


def test_rope_table_is_cached_once_per_device():
    model = types.SimpleNamespace(
        _attn_params={"rope_head_dim": 16},
        _freqs_cap=64,
        _rope_theta=10000.0,
        _freqs_table_cache={},
    )

    first = DSparkDraftModel._dspark_freqs_table(model, torch.device("cpu"))
    second = DSparkDraftModel._dspark_freqs_table(model, torch.device("cpu"))

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

    block = modeling_dspark.DSparkBlock(
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
    )

    actual = DSparkDraftModel._forward_stage(
        model,
        stage,
        h,
        torch.randn(num_requests, hidden_size),
        1,
        torch.empty(0),
        torch.zeros(num_requests, block_size, dtype=torch.long),
    )

    assert actual is final_h
    torch.testing.assert_close(
        stage.mlp.call_args.args[0],
        normed_ffn_input.reshape(num_requests * block_size, hidden_size),
    )
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


def _loop_reference(q, kv, attn_sink, topk_idxs, scale):
    """Obvious, slow per-(b,m,h) reference of the exact kernel math."""
    b, m, h, d = q.shape
    out = torch.zeros(b, m, h, d, dtype=torch.float32)
    qf, kvf, sink = q.float(), kv.float(), attn_sink.float()
    for bi in range(b):
        for mi in range(m):
            idxs = topk_idxs[bi, mi].tolist()
            for hi in range(h):
                scores, vecs = [], []
                for j in idxs:
                    if j < 0:
                        continue
                    k = kvf[bi, j]
                    scores.append(torch.dot(qf[bi, mi, hi], k) * scale)
                    vecs.append(k)
                if not scores:
                    continue
                s = torch.stack(scores)
                smax = s.max()
                p = torch.exp(s - smax)
                denom = p.sum() + torch.exp(sink[hi] - smax)
                num = (p.unsqueeze(-1) * torch.stack(vecs)).sum(0)
                out[bi, mi, hi] = num / denom
    return out


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_sparse_attn_matches_loop_reference(seed):
    torch.manual_seed(seed)
    b, m, h, d, n, topk = 2, 5, 3, 16, 40, 12
    q = torch.randn(b, m, h, d)
    kv = torch.randn(b, n, d)
    attn_sink = torch.randn(h)
    idx = torch.stack(
        [torch.stack([torch.randperm(n)[:topk] for _ in range(m)]) for _ in range(b)]
    ).int()
    scale = d**-0.5
    got = dspark_sparse_attn(q, kv, attn_sink, idx, scale).float()
    ref = _loop_reference(q, kv, attn_sink, idx, scale)
    torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-4)


def test_sparse_attn_no_sink_matches_sdpa():
    """With a -inf sink (no sink mass) and all-valid contiguous indices, the
    primitive must equal standard scaled-dot-product attention over the gathered
    KV — an independent implementation."""
    torch.manual_seed(0)
    b, m, h, d, topk = 2, 4, 5, 16, 9
    q = torch.randn(b, m, h, d)
    kv = torch.randn(b, topk, d)  # n == topk, attend to all
    attn_sink = torch.full((h,), float("-inf"))
    idx = torch.arange(topk).view(1, 1, -1).expand(b, m, topk).int()
    scale = d**-0.5
    got = dspark_sparse_attn(q, kv, attn_sink, idx, scale).float()

    # SDPA: q [b,h,m,d], k/v [b,h,topk,d] (broadcast the shared KV over heads).
    qh = q.permute(0, 2, 1, 3)
    kvh = kv.unsqueeze(1).expand(b, h, topk, d)
    ref = F.scaled_dot_product_attention(qh, kvh, kvh, scale=scale).permute(0, 2, 1, 3)
    torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-4)


def test_sparse_attn_sink_reduces_mass():
    """A finite sink must strictly shrink the attention output magnitude vs an
    infinitely-negative (disabled) sink, because it adds denominator mass only."""
    torch.manual_seed(0)
    b, m, h, d, topk = 1, 2, 2, 16, 6
    q = torch.randn(b, m, h, d)
    kv = torch.randn(b, topk, d)
    idx = torch.arange(topk).view(1, 1, -1).expand(b, m, topk).int()
    scale = d**-0.5
    no_sink = dspark_sparse_attn(q, kv, torch.full((h,), float("-inf")), idx, scale)
    with_sink = dspark_sparse_attn(q, kv, torch.zeros(h), idx, scale)
    assert with_sink.abs().sum() < no_sink.abs().sum()


def test_sparse_attn_masked_indices_excluded():
    """An index of -1 must be excluded exactly (equiv. to dropping that column)."""
    torch.manual_seed(0)
    b, m, h, d = 1, 1, 2, 16
    q = torch.randn(b, m, h, d)
    kv = torch.randn(b, 5, d)
    sink = torch.full((h,), float("-inf"))
    scale = d**-0.5
    full = torch.tensor([[[0, 1, 2, 3]]]).int()
    masked = torch.tensor([[[0, 1, 2, -1]]]).int()
    drop3 = torch.tensor([[[0, 1, 2]]]).int()
    got_masked = dspark_sparse_attn(q, kv, sink, masked, scale)
    got_drop = dspark_sparse_attn(q, kv, sink, drop3, scale)
    torch.testing.assert_close(got_masked, got_drop, rtol=1e-5, atol=1e-5)
    # And masking genuinely changes the result vs attending to position 3.
    got_full = dspark_sparse_attn(q, kv, sink, full, scale)
    assert not torch.allclose(got_full, got_masked, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize(
    "start_pos,window,block", [(1, 128, 5), (3, 128, 5), (10, 4, 5), (200, 128, 6)]
)
def test_get_dspark_topk_idxs_matches_reference(start_pos, window, block):
    bsz = 3
    got = get_dspark_topk_idxs(window, bsz, block, start_pos)
    # Reference formula (DeepSpec get_dspark_topk_idxs).
    ctx = torch.arange(min(window, start_pos + 1))
    blk = window + torch.arange(block)
    ref_row = torch.cat([ctx, blk]).int()
    assert got.shape == (bsz, block, ref_row.numel())
    for bi in range(bsz):
        for mi in range(block):
            torch.testing.assert_close(got[bi, mi], ref_row)


def test_get_dspark_topk_idxs_requires_generation():
    with pytest.raises(AssertionError):
        get_dspark_topk_idxs(128, 1, 5, 0)


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


def _make_attn_inputs(seed=0):
    """Small synthetic DSpark attention inputs/weights (CPU bf16)."""
    torch.manual_seed(seed)
    dim, n_heads, head_dim, rd = 12, 4, 8, 4
    q_lora, o_lora, n_groups = 6, 5, 2
    window, block, start_pos = 8, 3, 5
    b = 2
    g = dict(
        dim=dim,
        n_heads=n_heads,
        head_dim=head_dim,
        rope_head_dim=rd,
        q_lora=q_lora,
        o_lora=o_lora,
        n_groups=n_groups,
        window=window,
        block=block,
        start_pos=start_pos,
        b=b,
        eps=1e-6,
        softmax_scale=head_dim**-0.5,
    )
    bf = torch.bfloat16
    g["x"] = torch.randn(b, block, dim, dtype=bf)
    g["main_x"] = torch.randn(b, 1, dim, dtype=bf)
    g["kv_cache0"] = torch.randn(b, window, head_dim, dtype=bf)
    g["wq_a"] = torch.randn(q_lora, dim, dtype=bf) * 0.1
    g["wq_b"] = torch.randn(n_heads * head_dim, q_lora, dtype=bf) * 0.1
    g["wkv"] = torch.randn(head_dim, dim, dtype=bf) * 0.1
    g["wo_a"] = torch.randn(n_groups * o_lora, n_heads * head_dim // n_groups, dtype=bf) * 0.1
    g["wo_b"] = torch.randn(dim, n_groups * o_lora, dtype=bf) * 0.1
    g["q_norm"] = torch.ones(q_lora)
    g["kv_norm"] = torch.ones(head_dim)
    g["attn_sink"] = torch.randn(n_heads)
    g["freqs"] = precompute_dspark_freqs_cis(rd, start_pos + 1 + block + 2)
    return g


def _run(g):
    return dspark_attention_forward(
        g["x"],
        g["main_x"],
        g["start_pos"],
        g["kv_cache0"],
        wq_a=g["wq_a"],
        q_norm_w=g["q_norm"],
        wq_b=g["wq_b"],
        wkv=g["wkv"],
        kv_norm_w=g["kv_norm"],
        wo_a=g["wo_a"],
        wo_b=g["wo_b"],
        attn_sink=g["attn_sink"],
        n_heads=g["n_heads"],
        head_dim=g["head_dim"],
        rope_head_dim=g["rope_head_dim"],
        n_groups=g["n_groups"],
        o_lora_rank=g["o_lora"],
        window_size=g["window"],
        eps=g["eps"],
        softmax_scale=g["softmax_scale"],
        freqs_cis=g["freqs"],
    )


def test_attention_forward_shape_and_determinism():
    g = _make_attn_inputs()
    o = _run(g)
    assert tuple(o.shape) == (g["b"], g["block"], g["dim"])
    assert torch.isfinite(o.float()).all()
    torch.testing.assert_close(o, _run(g))  # deterministic


def test_attention_forward_does_not_mutate_kv_cache():
    """The rolling window write must be functional (cache cloned, not mutated)."""
    g = _make_attn_inputs()
    before = g["kv_cache0"].clone()
    _run(g)
    torch.testing.assert_close(g["kv_cache0"], before)
