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
"""Unit tests for the DSpark draft model: captured-context attention primitives,
the batched CUDA-graph-safe attention path, the draft I/O proposal stage, and
the draft-network heads (markov + confidence). CPU except per-test GPU gates.
"""

import types
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F

import tensorrt_llm._torch.models.modeling_dspark as modeling_dspark
from tensorrt_llm._torch.models.dspark.attention import (
    apply_dspark_rotary,
    apply_dspark_rotary_batched,
    dspark_attention_forward,
    dspark_attention_forward_batched,
    dspark_sparse_attn,
    get_dspark_topk_idxs,
    get_dspark_topk_idxs_batched,
    precompute_dspark_freqs_cis,
)
from tensorrt_llm._torch.models.dspark.draft import build_draft_input_ids, dspark_propose
from tensorrt_llm._torch.models.dspark.heads import (
    DSparkConfidenceHead,
    RNNHead,
    VanillaMarkov,
    build_markov_head,
)
from tensorrt_llm._torch.models.modeling_dspark import DSparkDraftModel

VOCAB, RANK, HID, BLK = 257, 16, 32, 5
DRAFT_B, HEADS_B = 2, 3
NOISE_ID = 199

# ---------------------------------------------------------------------------
# Captured-context attention primitives (CPU)
# ---------------------------------------------------------------------------


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
        model_config=types.SimpleNamespace(
            mapping=types.SimpleNamespace(enable_attention_dp=False, tp_size=8)
        ),
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


def test_sparse_attn_matches_loop_reference():
    torch.manual_seed(0)
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
    """With a -inf sink and all-valid contiguous indices, the primitive must
    equal standard scaled-dot-product attention over the gathered KV."""
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


@pytest.mark.parametrize("start_pos,window,block", [(3, 128, 5), (10, 4, 5)])
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
    # De-rotation (inverse) must undo the forward rotation (property test).
    roundtrip = apply_dspark_rotary(got, fc, inverse=True)
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
    before = g["kv_cache0"].clone()
    o = _run(g)
    assert tuple(o.shape) == (g["b"], g["block"], g["dim"])
    assert torch.isfinite(o.float()).all()
    torch.testing.assert_close(o, _run(g))  # deterministic
    # The rolling window write must be functional (cache cloned, not mutated).
    torch.testing.assert_close(g["kv_cache0"], before)


# ---------------------------------------------------------------------------
# Batched draft-attention path (CUDA-graph safety)
# ---------------------------------------------------------------------------


def _make_batched_inputs(seed=0, start_positions=(1, 3, 20)):
    """Per-request DSpark attention inputs/weights (shared weights, distinct
    start_pos and pre-seeded windows) for batched-vs-scalar comparison."""
    torch.manual_seed(seed)
    dim, n_heads, head_dim, rd = 12, 4, 8, 4
    q_lora, o_lora, n_groups = 6, 5, 2
    window, block = 8, 3
    G = len(start_positions)
    bf = torch.bfloat16
    # A single fixed RoPE table covering every request's positions (both paths
    # index/gather the same values, so freqs are identical across paths).
    maxlen = max(start_positions) + 1 + block + 4
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
        G=G,
        start_positions=list(start_positions),
        eps=1e-6,
        softmax_scale=head_dim**-0.5,
        x=torch.randn(G, block, dim, dtype=bf),
        main_x=torch.randn(G, 1, dim, dtype=bf),
        # Distinct, non-zero seeded window per request to exercise context reads.
        kv_cache=torch.randn(G, window, head_dim, dtype=bf),
        wq_a=torch.randn(q_lora, dim, dtype=bf) * 0.1,
        wq_b=torch.randn(n_heads * head_dim, q_lora, dtype=bf) * 0.1,
        wkv=torch.randn(head_dim, dim, dtype=bf) * 0.1,
        wo_a=torch.randn(n_groups * o_lora, n_heads * head_dim // n_groups, dtype=bf) * 0.1,
        wo_b=torch.randn(dim, n_groups * o_lora, dtype=bf) * 0.1,
        q_norm=torch.ones(q_lora),
        kv_norm=torch.ones(head_dim),
        attn_sink=torch.randn(n_heads),
        freqs=precompute_dspark_freqs_cis(rd, maxlen),
    )
    return g


def _attn_kwargs(g):
    return dict(
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


def _scalar_reference(g):
    """Run the validated scalar attention once per request and stack the outputs."""
    outs = []
    for i, sp in enumerate(g["start_positions"]):
        out_i = dspark_attention_forward(
            g["x"][i : i + 1],
            g["main_x"][i : i + 1],
            int(sp),
            g["kv_cache"][i : i + 1].clone(),
            **_attn_kwargs(g),
        )
        outs.append(out_i)
    return torch.cat(outs, dim=0)


def _batched(g, persist=False):
    G = g["G"]
    start_pos = torch.tensor(g["start_positions"], dtype=torch.long)
    slots = torch.arange(G, dtype=torch.long)
    return dspark_attention_forward_batched(
        g["x"],
        g["main_x"],
        start_pos,
        g["kv_cache"].clone(),
        slots,
        persist=persist,
        **_attn_kwargs(g),
    )


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_batched_attention_matches_scalar_per_request(seed):
    """Batched attention == per-request scalar attention at distinct start_pos."""
    g = _make_batched_inputs(seed=seed)
    ref = _scalar_reference(g)
    got = _batched(g)
    assert tuple(got.shape) == (g["G"], g["block"], g["dim"])
    torch.testing.assert_close(got, ref, rtol=2e-2, atol=2e-2)


def test_batched_attention_persist_writes_through_window():
    """persist gates the window write: False mutates nothing, True writes
    main_kv into exactly the start_pos%window row of each request."""
    g = _make_batched_inputs(seed=3)
    G, win = g["G"], g["window"]
    start_pos = torch.tensor(g["start_positions"], dtype=torch.long)
    slots = torch.arange(G, dtype=torch.long)
    cache = g["kv_cache"].clone()
    before = cache.clone()

    dspark_attention_forward_batched(
        g["x"], g["main_x"], start_pos, cache, slots, persist=False, **_attn_kwargs(g)
    )
    # persist=False must not mutate the caller's window (functional).
    torch.testing.assert_close(cache, before)

    dspark_attention_forward_batched(
        g["x"], g["main_x"], start_pos, cache, slots, persist=True, **_attn_kwargs(g)
    )
    # Exactly the start_pos%win row of each request changed.
    for i, sp in enumerate(g["start_positions"]):
        changed = (cache[i] != before[i]).any(dim=-1)
        expected = torch.zeros(win, dtype=torch.bool)
        expected[sp % win] = True
        assert torch.equal(changed, expected), f"req {i}: wrong window row written"


@pytest.mark.parametrize("start_positions", [(1, 3, 20), (5, 5, 5), (2, 7, 200)])
def test_batched_topk_matches_scalar(start_positions):
    """Fixed-size masked batched topk == scalar topk per request (valid slots)."""
    window, block = 8, 3
    start_pos = torch.tensor(start_positions, dtype=torch.long)
    batched = get_dspark_topk_idxs_batched(window, block, start_pos)
    assert tuple(batched.shape) == (len(start_positions), block, window + block)
    for i, sp in enumerate(start_positions):
        scalar = get_dspark_topk_idxs(window, 1, block, int(sp))[0]  # [block, topk_i]
        # The batched row drops the -1-masked context slots to recover the scalar
        # (variable-width) index list; both must then be identical.
        for m in range(block):
            valid = batched[i, m][batched[i, m] >= 0]
            torch.testing.assert_close(valid, scalar[m].to(valid.dtype))


@pytest.mark.parametrize("ndim", [3, 4])
def test_batched_rotary_matches_scalar_per_row(ndim):
    """apply_dspark_rotary_batched (per-row freqs) == scalar applied row by row."""
    torch.manual_seed(0)
    G, s, h, rd = 3, 4, 2, 8
    x = torch.randn(G, s, h, rd) if ndim == 4 else torch.randn(G, s, rd)
    table = precompute_dspark_freqs_cis(rd, 64)
    # Per-row absolute start positions -> per-row freq windows.
    starts = [1, 9, 30]
    per_row = torch.stack([table[sp : sp + s] for sp in starts], dim=0)  # [G, s, rd/2]
    got = apply_dspark_rotary_batched(x, per_row)
    for i, sp in enumerate(starts):
        ref_i = apply_dspark_rotary(x[i : i + 1], table[sp : sp + s])
        torch.testing.assert_close(got[i : i + 1], ref_i)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph capture needs a GPU")
def test_batched_attention_cuda_graph_capture_replay():
    """The batched attention captures + replays and matches eager output,
    proving the path is free of capture-illegal ops."""
    g = _make_batched_inputs(seed=0)
    dev = "cuda"
    G = g["G"]
    start_pos = torch.tensor(g["start_positions"], dtype=torch.long, device=dev)
    slots = torch.arange(G, dtype=torch.long, device=dev)
    # Static input tensors the graph reads/writes.
    x = g["x"].to(dev)
    main_x = g["main_x"].to(dev)
    cache = g["kv_cache"].to(dev)
    kw = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in _attn_kwargs(g).items()}

    def run(persist):
        return dspark_attention_forward_batched(
            x, main_x, start_pos, cache, slots, persist=persist, **kw
        )

    eager = run(persist=False)

    # Warmup (PyTorch CUDA-graph semantics) then capture on a non-persist call so
    # the comparison isn't perturbed by the window write-through.
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            run(persist=False)
    torch.cuda.current_stream().wait_stream(s)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = run(persist=False)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out, eager, rtol=2e-2, atol=2e-2)


# Confidence-scheduled verification: the scoring path runs inside the target's
# captured graph, so it must itself be graph-capturable.


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph capture needs a GPU")
@pytest.mark.parametrize("with_markov", [False, True])
def test_confidence_head_is_cuda_graph_capturable(with_markov):
    B, BLK, HID, RANK = 4, 5, 32, 16
    head = (
        DSparkConfidenceHead(
            hidden_size=HID, markov_rank=RANK, with_markov=with_markov, block_size=BLK
        )
        .cuda()
        .eval()
    )
    hid = torch.randn(B, BLK, HID, device="cuda")
    prev = torch.randn(B, BLK, RANK, device="cuda") if with_markov else None
    out = torch.empty(B, BLK, device="cuda", dtype=torch.float32)

    def run():
        return head.apply_sts(head(hid, prev_embeddings=prev))

    # Warm up on a side stream, as CUDA graph capture requires.
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out.copy_(run())

    eager = run()
    g.replay()
    torch.cuda.synchronize()
    assert torch.allclose(out, eager, atol=1e-5), "replay diverged from eager"
    assert torch.all((out >= 0.0) & (out <= 1.0)), "calibrated confidence must be a probability"

    # New inputs must flow through on replay -- a graph that captured constants
    # would keep returning the old scores and silently freeze the scheduler.
    hid.copy_(torch.randn_like(hid))
    fresh = run()
    g.replay()
    torch.cuda.synchronize()
    assert torch.allclose(out, fresh, atol=1e-5), "replay ignored updated inputs"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph capture needs a GPU")
def test_sts_table_update_is_visible_to_a_captured_graph():
    """STS must be updated in place; rebinding the buffer changes its data_ptr
    and a graph captured earlier would keep reading the old storage."""
    B, BLK, HID = 2, 5, 32
    head = DSparkConfidenceHead(hidden_size=HID, block_size=BLK).cuda().eval()
    hid = torch.randn(B, BLK, HID, device="cuda")
    out = torch.empty(B, BLK, device="cuda", dtype=torch.float32)

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            head.apply_sts(head(hid))
    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out.copy_(head.apply_sts(head(hid)))

    g.replay()
    torch.cuda.synchronize()
    before = out.clone()

    storage_before = head.sts_temperatures.data_ptr()
    head.load_sts_temperatures(torch.full((BLK,), 4.0))
    assert head.sts_temperatures.data_ptr() == storage_before, (
        "load_sts_temperatures must write in place; rebinding hides it from the graph"
    )

    g.replay()
    torch.cuda.synchronize()
    assert not torch.allclose(out, before), (
        "the captured graph did not see the new STS temperatures"
    )
    assert torch.allclose(out, torch.sigmoid(head(hid) / 4.0), atol=1e-5)


def test_apply_sts_accepts_host_logits_from_a_device_head():
    """apply_sts must handle host logits while sts_temperatures lives on the
    device: the planner calibrates on CPU-staged rows from a CUDA head."""
    BLK = 5
    head = DSparkConfidenceHead(hidden_size=8, block_size=BLK)
    if torch.cuda.is_available():
        head = head.cuda()
    head.load_sts_temperatures(torch.full((BLK,), 2.0))

    host_logits = torch.zeros(3, BLK)  # CPU, as the planner stages them
    probs = head.apply_sts(host_logits)
    assert probs.device.type == "cpu"
    assert torch.allclose(probs, torch.full((3, BLK), 0.5), atol=1e-6)


# Ragged verification runs inside the same captured graph, so its scatter and
# accept-count primitives must also capture without host syncs.


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph capture needs a GPU")
def test_ragged_scatter_is_cuda_graph_capturable():
    """scatter_ragged_to_padded and count_accepted_ragged capture and replay
    inside the target's graph without host syncs."""
    from tensorrt_llm._torch.speculative.dspark_ragged import (
        build_qo_indptr,
        count_accepted_ragged,
        scatter_ragged_to_padded,
    )

    lens = torch.tensor([3, 1, 2], dtype=torch.int32, device="cuda")
    indptr = build_qo_indptr(lens)
    flat = torch.tensor([10, 11, 12, 20, 30, 31], dtype=torch.int32, device="cuda")
    draft = torch.tensor([[1, 2, 3, 9], [4, 9, 9, 9], [5, 6, 9, 9]], device="cuda")
    target = torch.tensor([[1, 2, 7, 9], [4, 9, 9, 9], [5, 6, 9, 9]], device="cuda")
    out = torch.empty(3, 4, dtype=torch.int32, device="cuda")
    accepted = torch.empty(3, dtype=torch.int64, device="cuda")

    def run():
        padded = scatter_ragged_to_padded(
            flat, verify_lens=lens, qo_indptr=indptr, max_len=4, pad_value=-1
        )
        counts = count_accepted_ragged(
            draft_tokens=draft, target_tokens=target, verify_lens=lens
        )
        return padded, counts

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        padded, counts = run()
        out.copy_(padded)
        accepted.copy_(counts)

    g.replay()
    torch.cuda.synchronize()
    assert out.tolist() == [[10, 11, 12, -1], [20, -1, -1, -1], [30, 31, -1, -1]]
    # req 0 matches 2 then diverges; req 1 matches its single position; req 2
    # matches both. Column 3 is padding for every row and must never count.
    assert accepted.tolist() == [2, 1, 2]

    # New packed contents must flow through on replay.
    flat.copy_(torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.int32, device="cuda"))
    g.replay()
    torch.cuda.synchronize()
    assert out.tolist() == [[1, 2, 3, -1], [4, -1, -1, -1], [5, 6, -1, -1]]


# ---------------------------------------------------------------------------
# Draft I/O proposal stage
# ---------------------------------------------------------------------------


def test_build_draft_input_ids():
    bonus = torch.tensor([7, 9])
    ids = build_draft_input_ids(bonus, block_size=BLK, noise_token_id=NOISE_ID)
    assert ids.shape == (DRAFT_B, BLK)
    assert torch.equal(ids[:, 0], bonus)
    assert torch.all(ids[:, 1:] == NOISE_ID)


def test_dspark_propose_full_block_no_confidence():
    torch.manual_seed(0)
    markov = build_markov_head(
        markov_head_type="rnn", vocab_size=VOCAB, markov_rank=RANK, hidden_size=HID
    ).eval()
    base = torch.randn(DRAFT_B, BLK, VOCAB)
    bonus = torch.randint(0, VOCAB, (DRAFT_B,))
    hid = torch.randn(DRAFT_B, BLK, HID)
    with torch.no_grad():
        tokens, confidence = dspark_propose(
            base,
            bonus_token_ids=bonus,
            block_hidden=hid,
            markov_head=markov,
            confidence_head=None,
            block_size=BLK,
        )
    assert tokens.shape == (DRAFT_B, BLK)
    assert confidence is None
    # Tokens match the markov head's own greedy block sampling.
    ref_tokens, _ = markov.sample_block_tokens(
        base, first_prev_token_ids=bonus, hidden_states=hid, temperature=0.0
    )
    assert torch.equal(tokens, ref_tokens)


def _propose_with_confidence(conf, markov, *, return_confidence):
    base = torch.randn(1, BLK, VOCAB)
    bonus = torch.randint(0, VOCAB, (1,))
    hid = torch.ones(1, BLK, HID)
    with torch.no_grad():
        return dspark_propose(
            base,
            bonus_token_ids=bonus,
            block_hidden=hid,
            markov_head=markov,
            confidence_head=conf,
            block_size=BLK,
            return_confidence=return_confidence,
        )


def test_dspark_propose_scores_without_shortening_the_block():
    """The block is always proposed in full; confidence only scores it, and
    only when asked (return_confidence is the run-constant opt-in)."""
    torch.manual_seed(1)
    markov = build_markov_head(
        markov_head_type="vanilla", vocab_size=VOCAB, markov_rank=RANK, hidden_size=HID
    ).eval()
    conf = DSparkConfidenceHead(hidden_size=HID, block_size=BLK).eval()
    # The confidence proj is bias-free, so drive the logit via a constant weight
    # against a constant hidden: logit = weight_val * HID per position.
    with torch.no_grad():
        conf.proj.weight.fill_(-5.0 / HID)  # sigmoid ~ 0.0067, i.e. hopeless

    tokens, confidence = _propose_with_confidence(conf, markov, return_confidence=True)
    assert tokens.shape == (1, BLK)
    assert confidence.shape == (1, BLK)
    # Low confidence must NOT shorten the proposal -- that decision belongs to
    # the verification scheduler, not the drafter.
    assert torch.all(confidence < 0.0)
    assert torch.all(conf.apply_sts(confidence) < 0.5)

    # Opt-in: with the flag off the head is not consulted at all.
    _, confidence = _propose_with_confidence(conf, markov, return_confidence=False)
    assert confidence is None


def test_dspark_propose_is_free_of_host_syncs():
    """No ``.item()``/``nonzero`` on this path: it runs inside the target's graph."""
    import inspect
    import io
    import tokenize

    from tensorrt_llm._torch.models.dspark import draft as draft_mod

    src = inspect.getsource(draft_mod.dspark_propose)
    # Strip comments and string literals so the check reads the code, not the
    # prose describing it (the implementation comments name these very calls).
    code = "".join(
        tok.string if tok.type not in (tokenize.COMMENT, tokenize.STRING) else " "
        for tok in tokenize.generate_tokens(io.StringIO(src).readline)
    )
    for banned in (".item(", "nonzero", "range("):
        assert banned not in code, f"dspark_propose must stay capture-safe: found {banned!r}"


# ---------------------------------------------------------------------------
# Draft-network heads (markov + confidence, CPU)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("head_type", ["vanilla", "gated", "rnn"])
def test_markov_block_sampling_shapes_and_determinism(head_type):
    torch.manual_seed(0)
    head = build_markov_head(
        markov_head_type=head_type, vocab_size=VOCAB, markov_rank=RANK, hidden_size=HID
    ).eval()
    base = torch.randn(HEADS_B, BLK, VOCAB)
    first = torch.randint(0, VOCAB, (HEADS_B,))
    hid = torch.randn(HEADS_B, BLK, HID)
    with torch.no_grad():
        tok, logits = head.sample_block_tokens(
            base, first_prev_token_ids=first, hidden_states=hid, temperature=0.0
        )
        tok2, _ = head.sample_block_tokens(
            base, first_prev_token_ids=first, hidden_states=hid, temperature=0.0
        )
    assert tok.shape == (HEADS_B, BLK)
    assert logits.shape == (HEADS_B, BLK, VOCAB)
    # Greedy is deterministic.
    assert torch.equal(tok, tok2)
    # Each sampled token is the argmax of its (bias-corrected) step logits.
    assert torch.equal(tok, logits.argmax(dim=-1))


def test_markov_bias_is_additive_low_rank():
    # bias = W2(W1[token]); the corrected first-step logits == base + bias.
    torch.manual_seed(1)
    head = VanillaMarkov(vocab_size=VOCAB, markov_rank=RANK).eval()
    base = torch.randn(HEADS_B, BLK, VOCAB)
    first = torch.randint(0, VOCAB, (HEADS_B,))
    with torch.no_grad():
        _, corrected = head.sample_block_tokens(
            base, first_prev_token_ids=first, hidden_states=None, temperature=0.0
        )
        expected0 = base[:, 0] + head.markov_w2(head.markov_w1(first))
    assert torch.allclose(corrected[:, 0], expected0, atol=1e-5)


def test_rnn_state_carries_across_positions():
    torch.manual_seed(2)
    head = RNNHead(vocab_size=VOCAB, markov_rank=RANK, hidden_size=HID).eval()
    initial_state = torch.zeros(1, RANK)
    prev_embedding = head.get_prev_embeddings(torch.zeros(1, dtype=torch.long))
    prefix_hidden = torch.randn(1, HID)
    current_hidden = torch.randn(1, HID)

    with torch.no_grad():
        state_a, _ = head._rnn_step(initial_state, prev_embedding, prefix_hidden)
        state_b, _ = head._rnn_step(initial_state, prev_embedding, -prefix_hidden)
        _, bias_a = head._rnn_step(state_a, prev_embedding, current_hidden)
        _, bias_b = head._rnn_step(state_b, prev_embedding, current_hidden)

    assert not torch.allclose(state_a, state_b)
    assert not torch.allclose(bias_a, bias_b)


def test_build_markov_head_rank_zero_returns_none():
    assert (
        build_markov_head(
            markov_head_type="vanilla", vocab_size=VOCAB, markov_rank=0, hidden_size=HID
        )
        is None
    )


def test_confidence_head_emits_raw_logits():
    head = DSparkConfidenceHead(hidden_size=HID, block_size=BLK)
    conf = head(torch.randn(HEADS_B, BLK, HID))
    assert conf.shape == (HEADS_B, BLK)
    assert conf.dtype == torch.float32


def test_apply_sts_uses_per_position_temperatures():
    head = DSparkConfidenceHead(hidden_size=HID, block_size=BLK)
    # Default (all-ones) calibration is a plain sigmoid, bounded in [0, 1].
    raw = torch.tensor([[2.0, 0.0, -2.0, 1.0, -1.0]])
    assert torch.allclose(head.apply_sts(raw), torch.sigmoid(raw), atol=1e-6)
    assert torch.all((head.apply_sts(raw) >= 0.0) & (head.apply_sts(raw) <= 1.0))

    temps = torch.tensor([1.0, 2.0, 4.0, 1.0, 1.0])
    head.load_sts_temperatures(temps)
    raw = torch.full((1, BLK), 4.0)
    got = head.apply_sts(raw)
    # A larger temperature pulls the probability toward 0.5.
    assert got[0, 0] > got[0, 1] > got[0, 2]
    assert torch.allclose(got, torch.sigmoid(raw / temps), atol=1e-6)


def test_load_sts_temperatures_updates_in_place():
    """The STS table is calibration, not a checkpoint weight: it must update
    in place, stay out of state_dict, and refuse malformed tables."""
    head = DSparkConfidenceHead(hidden_size=HID, block_size=BLK)
    assert "sts_temperatures" not in head.state_dict()
    with pytest.raises(ValueError, match="one per block position"):
        head.load_sts_temperatures(torch.ones(BLK + 1))
    with pytest.raises(ValueError, match="strictly positive"):
        head.load_sts_temperatures(torch.zeros(BLK))
    before = head.sts_temperatures
    head.load_sts_temperatures(torch.full((BLK,), 2.0))
    assert head.sts_temperatures is before
    assert torch.allclose(before, torch.full((BLK,), 2.0))


def test_confidence_head_load_weights_rejects_a_dropped_bias():
    """A checkpoint bias silently ignored would shift every confidence score."""
    head = DSparkConfidenceHead(hidden_size=HID, block_size=BLK)
    good = {"proj.weight": torch.randn(1, HID)}
    head.load_weights([good])
    assert torch.allclose(head.proj.weight, good["proj.weight"])

    with pytest.raises(ValueError, match="bias=True"):
        head.load_weights([{**good, "proj.bias": torch.zeros(1)}])

    # Constructed with bias=True, the same checkpoint bias is accepted.
    biased = DSparkConfidenceHead(hidden_size=HID, block_size=BLK, bias=True)
    w = {"proj.weight": torch.randn(1, HID), "proj.bias": torch.randn(1)}
    biased.load_weights([w])
    assert torch.allclose(biased.proj.bias, w["proj.bias"])


def test_confidence_head_with_markov_concat_dim():
    head = DSparkConfidenceHead(hidden_size=HID, markov_rank=RANK, with_markov=True)
    hid = torch.randn(HEADS_B, BLK, HID)
    prev_emb = torch.randn(HEADS_B, BLK, RANK)
    out = head(hid, prev_embeddings=prev_emb)
    assert out.shape == (HEADS_B, BLK)
    with pytest.raises(AssertionError):
        head(hid)  # with_markov requires prev_embeddings
