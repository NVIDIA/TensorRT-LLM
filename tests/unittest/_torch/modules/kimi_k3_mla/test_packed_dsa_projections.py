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
"""KimiK3MLAAttention._pack_dsa_projections: which layers pack and what the packed weights hold."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.modules.kimi_k3_mla.kimi_k3_mla_attention import KimiK3MLAAttention


def _linear(out_features, in_features, dtype=torch.bfloat16, quant=False):
    return SimpleNamespace(
        weight=torch.nn.Parameter(
            torch.randn(out_features, in_features).to(dtype), requires_grad=False
        ),
        bias=None,
        has_any_quant=quant,
        lora=None,
    )


def _attention(*, indexer=True, kv_a_quant=False, qa_fused_scale=None, gate=False):
    idx = None
    if indexer:
        idx = SimpleNamespace(
            projection_free=False,
            wk=_linear(128, 256, dtype=torch.float32),
            weights_proj=_linear(32, 256, dtype=torch.float32),
            wq_b=_linear(512, 64),
        )
    return SimpleNamespace(
        mqa=SimpleNamespace(indexer=idx),
        kv_a_proj_with_mqa=_linear(96, 256, quant=kv_a_quant),
        q_b_proj=_linear(192, 64),
        _resolve_qa_fused_scale=lambda: qa_fused_scale,
        use_output_gate=gate,
        g_proj=_linear(48, 256) if gate else None,
        _k31_packed_gate=False,
    )


def test_packs_both_inputs_into_row_views():
    attn = _attention()
    kv_a, q_b = attn.kv_a_proj_with_mqa.weight.clone(), attn.q_b_proj.weight.clone()
    wk = attn.mqa.indexer.wk.weight.to(torch.bfloat16)
    wp = attn.mqa.indexer.weights_proj.weight.to(torch.bfloat16)
    wq_b = attn.mqa.indexer.wq_b.weight.clone()

    KimiK3MLAAttention._pack_dsa_projections(attn)

    hidden, q = attn.dsa_packed_hidden_weight, attn.dsa_packed_q_weight
    assert hidden.shape == (96 + 128 + 32, 256) and q.shape == (192 + 512, 64)
    torch.testing.assert_close(hidden, torch.cat([kv_a, wk, wp]), rtol=0, atol=0)
    torch.testing.assert_close(q, torch.cat([q_b, wq_b]), rtol=0, atol=0)
    # The module weights alias the packed rows, so the pack adds no memory.
    assert attn.kv_a_proj_with_mqa.weight.data_ptr() == hidden.data_ptr()
    assert attn.q_b_proj.weight.data_ptr() == q.data_ptr()
    assert attn.mqa.indexer.wq_b.weight.data_ptr() == q[192:].data_ptr()
    # One GEMM over the packed weight equals the separate projections.
    x = torch.randn(3, 256).to(torch.bfloat16)
    out = F.linear(x, hidden).split([96, 128, 32], -1)
    for got, w in zip(out, (kv_a, wk, wp)):
        torch.testing.assert_close(got, F.linear(x, w))


def test_gate_stays_separate_by_default():
    attn = _attention(gate=True)
    g = attn.g_proj.weight
    KimiK3MLAAttention._pack_dsa_projections(attn)
    assert attn.dsa_packed_hidden_weight.shape == (96 + 128 + 32, 256)
    assert attn.dsa_packed_gate_rows == 0 and attn.g_proj.weight is g


def test_packed_gate_rows_follow_the_indexer_rows():
    """Packed gate: g_proj's rows close the packed hidden-state weight; g_proj's weight becomes their row
    view."""
    attn = _attention(gate=True)
    attn._k31_packed_gate = True
    kv_a, g = attn.kv_a_proj_with_mqa.weight.clone(), attn.g_proj.weight.clone()
    wk = attn.mqa.indexer.wk.weight.to(torch.bfloat16)
    wp = attn.mqa.indexer.weights_proj.weight.to(torch.bfloat16)

    KimiK3MLAAttention._pack_dsa_projections(attn)

    hidden = attn.dsa_packed_hidden_weight
    assert hidden.shape == (96 + 128 + 32 + 48, 256) and attn.dsa_packed_gate_rows == 48
    torch.testing.assert_close(hidden, torch.cat([kv_a, wk, wp, g]), rtol=0, atol=0)
    assert attn.g_proj.weight.data_ptr() == hidden[96 + 128 + 32 :].data_ptr()
    torch.testing.assert_close(attn.g_proj.weight, g, rtol=0, atol=0)


@pytest.mark.parametrize(
    "kwargs",
    [dict(indexer=False), dict(kv_a_quant=True), dict(qa_fused_scale=torch.ones(1))],
    ids=["no_indexer_k3", "quantized_kv_a", "fused_qa_scale"],
)
def test_leaves_unpackable_layers_alone(kwargs):
    attn = _attention(**kwargs)
    KimiK3MLAAttention._pack_dsa_projections(attn)
    assert attn.dsa_packed_hidden_weight is None and attn.dsa_packed_q_weight is None


def test_quantized_q_b_packs_only_the_hidden_input():
    attn = _attention()
    attn.q_b_proj.has_any_quant = True
    KimiK3MLAAttention._pack_dsa_projections(attn)
    assert attn.dsa_packed_hidden_weight is not None and attn.dsa_packed_q_weight is None


def test_packed_indexer_q_rows_are_token_major():
    """A column slice of the packed [q_b | wq_b] output must reach the FP8 cat-quantize kernel as
    [tokens * heads, head_dim] rows with one row stride (its only stride argument)."""
    from tensorrt_llm._torch.attention.backends.sparse.dsa.indexer import Indexer

    n_heads, head_dim, tokens = 2, 128, 3
    indexer = SimpleNamespace(
        n_heads=n_heads, head_dim=head_dim, rotary_emb=None, k_norm=lambda k: k, wq_b=None
    )
    packed_out = torch.randn(tokens, 192 + n_heads * head_dim)
    q_slice = packed_out[:, 192:]
    q_pe, q_nope, _, _ = Indexer._qk_projection_and_rope(
        indexer, None, torch.randn(tokens, head_dim), None, q_proj_out=q_slice
    )
    rows = q_nope.reshape(-1, head_dim // 2)
    assert q_nope.stride(-2) == head_dim and q_nope.stride(0) == n_heads * head_dim
    torch.testing.assert_close(
        rows, q_slice.reshape(tokens * n_heads, head_dim)[:, head_dim // 2 :]
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("tokens", [128, 1000, 4096, 8192])
def test_split_q_projection_matches_packed_gemm(tokens):
    """From _DSA_SPLIT_Q_PROJ_MIN_TOKENS tokens forward_dsa_proj runs the packed [q_b_proj | wq_b] projection
    as two GEMMs on row slices of the packed weight; at Kimi K3.1's shape (q_lora 1536, 96 x 192 MLA q columns,
    32 x 128 indexer q columns) both outputs equal the packed GEMM's column slices bitwise."""
    from tensorrt_llm._torch.attention.backends.sparse.dsa import module as dsa_module

    assert tokens >= dsa_module._DSA_SPLIT_Q_PROJ_MIN_TOKENS
    torch.manual_seed(tokens)
    mla_n, idx_n, q_lora = 96 * 192, 32 * 128, 1536
    packed = (torch.randn(mla_n + idx_n, q_lora, device="cuda") * q_lora**-0.5).to(torch.bfloat16)
    qr = torch.randn(tokens, q_lora, device="cuda").to(torch.bfloat16)
    q_ref, idx_ref = F.linear(qr, packed).split([mla_n, idx_n], -1)
    q = F.linear(qr, packed[:mla_n])
    idx_q = F.linear(qr, packed[mla_n:])
    assert q.is_contiguous() and idx_q.is_contiguous()
    assert torch.equal(q, q_ref) and torch.equal(idx_q, idx_ref)
