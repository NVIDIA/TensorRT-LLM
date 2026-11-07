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
"""Tests for the MiniMax-H3 Q|K projection GEMM with the per-head RMSNorm + three-axis RoPE epilogue."""

import pytest
import torch
import torch.nn.functional as F
from test_minimax_h3_transformer import (  # noqa: I001  (sibling test module, pytest prepends its dir)
    _make_model_config,
)

import tensorrt_llm._torch.visual_gen.models.minimax_h3.transformer_minimax_h3 as h3
from tensorrt_llm._utils import get_sm_version

requires_sm100 = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() not in (100, 103),
    reason="The QuACK Q|K epilogue GEMM runs on SM100/SM103 only",
)

HEADS = 2
HEAD_DIM = 128
HIDDEN = 256
SEQ = 301
ROPE_FREQ_DIM = 16
ROTARY_DIM = 6 * ROPE_FREQ_DIM


def _epilogue_class():
    """Imported inside the tests: the module builds QuACK's SM100 epilogue at import."""
    from tensorrt_llm._torch.visual_gen.models.minimax_h3.kernels.qk_epilogue import (
        MiniMaxH3QKEpilogue,
    )

    return MiniMaxH3QKEpilogue


def _attention_and_rope() -> tuple[h3.MiniMaxH3Attention, h3.MiniMaxH3RotaryPosEmbed]:
    attn = h3.MiniMaxH3Attention(
        hidden_size=HIDDEN,
        num_attention_heads=HEADS,
        attention_head_dim=HEAD_DIM,
        qk_norm_eps=1e-5,
        model_config=_make_model_config(),
        layer_idx=0,
        module_name="test.attn",
    ).to("cuda")
    attn.requires_grad_(False)
    generator = torch.Generator(device="cuda").manual_seed(0)
    for name, parameter in attn.named_parameters():
        noise = torch.randn(parameter.shape, device="cuda", generator=generator)
        # Non-uniform norm weights, so a norm weight left in checkpoint order is visible.
        parameter.copy_(1 + 0.1 * noise if name.startswith("norm_") else noise * HIDDEN**-0.5)
    rope = h3.MiniMaxH3RotaryPosEmbed(rope_freq_dim=ROPE_FREQ_DIM).to("cuda")
    return attn, rope


def _inputs(batch: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(1)
    x = torch.randn((batch, SEQ, HIDDEN), device="cuda", generator=generator).to(torch.bfloat16)
    position_ids = torch.randint(0, 512, (SEQ, 3), device="cuda", generator=generator)
    return x, position_ids


def _pair_layout() -> torch.Tensor:
    """Head column order the epilogue emits for Q and K: split-half RoPE pairs (i, i + 48) adjacent."""
    half = ROTARY_DIM // 2
    rotary = [column for i in range(half) for column in (i, i + half)]
    return torch.tensor(rotary + list(range(ROTARY_DIM, HEAD_DIM)), device="cuda")


def _reference_qkv(attn, cos, sin, x):
    """The unfused path's Q/K/V math in FP64 on the checkpoint weights: QKV projection, per-head
    RMSNorm, then ``apply_minimax_h3_rotary_emb`` with the rope module's FP32 cos/sin."""
    qkv = F.linear(x.double(), attn.qkv_proj.weight.double())
    q, k, v = qkv.split([attn.local_q_dim, attn.local_kv_dim, attn.local_kv_dim], dim=-1)

    def norm_rope(states, norm):
        states = states.unflatten(-1, (-1, HEAD_DIM))
        variance = states.pow(2).mean(-1, keepdim=True)
        states = states * torch.rsqrt(variance + norm.variance_epsilon) * norm.weight.double()
        return h3.apply_minimax_h3_rotary_emb(states, cos, sin)

    return norm_rope(q, attn.norm_q), norm_rope(k, attn.norm_k), v


def _assert_single_bf16_rounding(actual: torch.Tensor, reference: torch.Tensor) -> None:
    """The epilogue rounds its FP32 result to BF16 once (half an ulp); its fast-math sin/cos and the
    reference's FP32 angle tables add errors orders of magnitude smaller, so one BF16 ulp of the
    output scale bounds the difference."""
    assert actual.shape == reference.shape
    assert (actual.double() - reference).abs().max() <= reference.abs().max() * 2**-7


@requires_sm100
def test_qk_epilogue_matches_norm_rope_in_pair_layout():
    """Q and K equal the unfused path's normalized, rotated heads with columns in the pair layout
    (the same permutation of both leaves every Q.K product unchanged); V keeps the checkpoint layout."""
    attn, rope = _attention_and_rope()
    x, position_ids = _inputs(batch=1)
    rope.with_positions = True
    rotary_emb = rope(position_ids)
    q_ref, k_ref, v_ref = _reference_qkv(attn, *rotary_emb[:2], x)

    epilogue = _epilogue_class()(attn, rope.inv_freq, ROTARY_DIM, torch.device("cuda"))
    q, k, v = epilogue(x[0], attn.qkv_proj.weight, rotary_emb[2:])

    layout = _pair_layout()
    _assert_single_bf16_rounding(q.view(SEQ, HEADS, HEAD_DIM), q_ref[0][..., layout])
    _assert_single_bf16_rounding(k.view(SEQ, HEADS, HEAD_DIM), k_ref[0][..., layout])
    _assert_single_bf16_rounding(v, v_ref[0])


@requires_sm100
@pytest.mark.parametrize(
    "batch,with_positions,uses_epilogue",
    [(1, True, True), (2, True, False), (1, False, False)],
    ids=["epilogue", "batch2_fallback", "cos_sin_fallback"],
)
def test_attention_with_qk_epilogue_matches_checkpoint_order(
    monkeypatch, batch, with_positions, uses_epilogue
):
    """With the Q/K weights permuted for the epilogue, attention equals the unfused path on the
    checkpoint weights, both when the epilogue runs and when forward falls back to un-permuting Q, K."""
    epilogue_class = _epilogue_class()
    attn, rope = _attention_and_rope()
    x, position_ids = _inputs(batch)
    attn.fuse_qk_norm_rope = False
    expected = attn(x, rope(position_ids))

    # The production flag: the fused QK-norm kernel must not run on the permuted weights.
    attn.fuse_qk_norm_rope = True
    attn.h3_qk_epilogue = epilogue_class(attn, rope.inv_freq, ROTARY_DIM, torch.device("cuda"))
    rope.with_positions = with_positions
    calls = []
    epilogue_call = epilogue_class.__call__

    def tracked(self, *args):
        calls.append(True)
        return epilogue_call(self, *args)

    monkeypatch.setattr(epilogue_class, "__call__", tracked)
    actual = attn(x, rope(position_ids))

    assert bool(calls) == uses_epilogue
    # Single vs per-step Q/K rounding, or a permuted RMSNorm reduction, is BF16 noise on the output;
    # Q and K in mismatched head layouts change the attention scores outright.
    expected = expected.float()
    assert (actual.float() - expected).abs().max() <= expected.abs().max() * 2**-6
