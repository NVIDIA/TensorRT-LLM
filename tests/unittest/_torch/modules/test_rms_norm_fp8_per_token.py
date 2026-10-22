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
"""Fused (gated) RMSNorm + per-token FP8 quant vs norm followed by per-token quant."""

from types import SimpleNamespace

import pytest
import torch
from utils.util import getSMVersion

from tensorrt_llm._torch.models.modeling_qwen3_next import Qwen3NextLinearDecoderLayer
from tensorrt_llm._torch.models.modeling_utils import EagerFusionConfig
from tensorrt_llm._torch.modules.mamba.gdn_mixer import Qwen3NextGatedDeltaNet
from tensorrt_llm._torch.modules.mamba.layernorm_gated import rms_norm_gated_token_major
from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm._torch.modules.rms_norm_fp8_per_token import (
    add_rms_norm_fp8_per_token,
    rms_norm_gated_fp8_per_token,
)
from tensorrt_llm._torch.utils import EventType, Fp8PerTokenQuantizedTensor

pytestmark = pytest.mark.skipif(getSMVersion() < 89, reason="FP8 requires SM89+")


def _per_token_quant(y: torch.Tensor):
    return torch.ops.tensorrt_llm.vectorized_per_token_fp8_quant(y)


def _assert_quant_matches(q, scale, y):
    # Must match quantizing the kernel's own normed output exactly.
    q_ref, scale_ref = _per_token_quant(y)
    torch.testing.assert_close(scale, scale_ref, rtol=0, atol=0)
    assert torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8))


@pytest.mark.parametrize("num_tokens", [1, 50, 3000])
@pytest.mark.parametrize("use_gemma", [False, True])
@pytest.mark.parametrize("with_residual", [False, True])
def test_add_rms_norm_fp8_per_token(num_tokens, use_gemma, with_residual):
    torch.manual_seed(0)
    hidden, eps = 2560, 1e-6
    x = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device="cuda") * 3
    residual = torch.randn_like(x) * 10 if with_residual else None
    weight = torch.randn(hidden, dtype=torch.bfloat16, device="cuda") * 0.2

    ref_in = x.float() + residual.float() if with_residual else x.float()
    gain = weight.float() + 1 if use_gemma else weight.float()
    ref = ref_in * torch.rsqrt(ref_in.pow(2).mean(-1, keepdim=True) + eps) * gain

    res_out = residual.clone() if with_residual else None
    y, q, scale = add_rms_norm_fp8_per_token(x, res_out, weight, eps, use_gemma)

    if with_residual:
        assert torch.equal(res_out, ref_in.to(torch.bfloat16))
    torch.testing.assert_close(y.float(), ref, rtol=1e-2, atol=1e-2)
    assert scale.shape == (num_tokens, 1) and scale.dtype == torch.float32
    _assert_quant_matches(q, scale, y)


@pytest.mark.parametrize("num_tokens", [1, 50, 3000])
def test_rms_norm_gated_fp8_per_token(num_tokens):
    torch.manual_seed(0)
    heads, head_dim, eps = 32, 128, 1e-6
    x = torch.randn(num_tokens * heads, head_dim, dtype=torch.bfloat16, device="cuda")
    # z as a column slice of a wider per-token projection, as GDN passes it.
    z = torch.randn(num_tokens, heads * head_dim + 512, dtype=torch.bfloat16, device="cuda")
    z = z[:, : heads * head_dim].view(num_tokens, heads, head_dim)
    weight = torch.randn(head_dim, dtype=torch.bfloat16, device="cuda") * 0.3 + 1

    q, scale = rms_norm_gated_fp8_per_token(x, z, weight, eps)

    # The reference norm's reduction order can differ by a BF16 ulp: allow rare 1-code diffs.
    y = rms_norm_gated_token_major(x, z, weight, eps).view(num_tokens, heads * head_dim)
    q_ref, scale_ref = _per_token_quant(y)
    torch.testing.assert_close(scale, scale_ref, rtol=1e-2, atol=0)
    steps = (q.view(torch.uint8).short() - q_ref.view(torch.uint8).short()).abs()
    assert steps.max() <= 1
    assert (steps != 0).float().mean() < 1e-5


def test_rms_norm_gated_fp8_per_token_rejects_unknown_gate():
    x = torch.randn(32, 128, dtype=torch.bfloat16, device="cuda")
    z = torch.randn(1, 32, 128, dtype=torch.bfloat16, device="cuda")
    weight = torch.ones(128, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="gate_activation"):
        rms_norm_gated_fp8_per_token(x, z, weight, 1e-6, "gelu")


def test_rmsnorm_fp8_per_token_out():
    torch.manual_seed(0)
    hidden = 2560
    norm = RMSNorm(
        hidden_size=hidden, eps=1e-6, dtype=torch.bfloat16, device="cuda", use_gemma=True
    )
    with torch.no_grad():
        norm.weight.normal_(0, 0.2)
    norm.fp8_per_token_out = True
    x = torch.randn(50, hidden, dtype=torch.bfloat16, device="cuda")
    residual = torch.randn_like(x)

    out, res_out = norm(x, residual.clone())

    assert isinstance(out, Fp8PerTokenQuantizedTensor)
    assert out.shape == x.shape
    y_ref, q_ref, scale_ref = add_rms_norm_fp8_per_token(
        x, residual.clone(), norm.weight, 1e-6, True
    )
    assert torch.equal(out.unquantized_hidden_states, y_ref)
    assert torch.equal(out.fp8_tensor.view(torch.uint8), q_ref.view(torch.uint8))
    assert torch.equal(out.per_token_scale, scale_ref)
    assert torch.equal(res_out, (x.float() + residual.float()).to(torch.bfloat16))


def test_gdn_in_proj_fp8_per_token_handoff():
    hidden, num_tokens = 2560, 50
    heads, head_dim, conv_dim = 4, 128, 1536
    norm = RMSNorm(
        hidden_size=hidden, eps=1e-6, dtype=torch.bfloat16, device="cuda", use_gemma=True
    )
    norm.fp8_per_token_out = True
    x = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device="cuda")
    normed, _ = norm(x, torch.randn_like(x))

    seen = {}

    def projection(name, out_features):
        def forward(inp):
            seen[name] = inp
            return torch.zeros(num_tokens, out_features, dtype=torch.bfloat16, device="cuda")

        return forward

    gdn = SimpleNamespace(
        in_proj_qkvz=projection("qkvz", conv_dim + heads * head_dim),
        in_proj_ba=projection("ba", 2 * heads),
        event_dict={key: torch.cuda.Event() for key in [EventType.Main, EventType.Attention]},
        aux_stream=None,
        conv_dim_per_tp=conv_dim,
        num_v_heads_per_tp=heads,
        head_v_dim=head_dim,
    )
    Qwen3NextGatedDeltaNet._compute_tokenwise_inputs(gdn, normed)

    # in_proj_qkvz gets the FP8 tensor and scale; in_proj_ba gets the BF16 normed output.
    assert isinstance(normed, Fp8PerTokenQuantizedTensor)
    assert seen["qkvz"] is normed
    assert seen["ba"] is normed.unquantized_hidden_states


@pytest.mark.parametrize("with_residual", [False, True])
def test_linear_decoder_layer_empty_input_with_fp8_per_token_norms(with_residual):
    hidden = 2560

    def fp8_norm():
        norm = RMSNorm(
            hidden_size=hidden, eps=1e-6, dtype=torch.bfloat16, device="cuda", use_gemma=True
        )
        norm.fp8_per_token_out = True
        return norm

    def skipped_linear_attn(*args, **kwargs):
        raise AssertionError("linear attention must be skipped for zero tokens")

    mlp_inputs = []

    def mlp(hidden_states, attn_metadata, **kwargs):
        mlp_inputs.append(hidden_states)
        return hidden_states

    layer = SimpleNamespace(
        input_layernorm=fp8_norm(),
        post_attention_layernorm=fp8_norm(),
        next_layer_layernorm=fp8_norm(),
        linear_attn=skipped_linear_attn,
        mlp=mlp,
        fusion_config=EagerFusionConfig(),
        mapping=SimpleNamespace(tp_size=1),
    )
    x = torch.empty(0, hidden, dtype=torch.bfloat16, device="cuda")
    residual = torch.empty_like(x) if with_residual else None

    hidden_states, residual = Qwen3NextLinearDecoderLayer.forward(layer, None, x, None, residual)

    # Zero-token norms emit plain high-precision tensors, not FP8 wrappers.
    for out in (mlp_inputs[0], hidden_states, residual):
        assert type(out) is torch.Tensor
        assert out.shape == (0, hidden) and out.dtype == torch.bfloat16
