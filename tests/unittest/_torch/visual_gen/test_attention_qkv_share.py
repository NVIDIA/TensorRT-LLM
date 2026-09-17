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
"""Tests for Attention's opt-in shared QKV input quantization.

Static FP8 checkpoints keep q/k/v as separate calibrated projections, so a fused
QKV Linear would re-quantize two of them onto a third's scale. Splitting them
costs three quantizations of one identical activation instead of one, which
``share_qkv_input_quant=True`` recovers by quantizing once and handing the same
FP8 tensor to all three.

The flag is opt-in: enabling it also commits the caller to running
``post_load_weights()``, which is where the equal-input_scale invariant that
makes the sharing sound is actually checked. These tests pin both halves.
"""

import pytest
import torch

from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig
from tensorrt_llm._torch.visual_gen.modules.attention import Attention, QKVMode
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

HIDDEN, HEADS, KV_HEADS, HEAD_DIM, TOKENS = 512, 8, 8, 64, 128
INPUT_SCALE, WEIGHT_SCALE = 1e-2, 1.1e-3

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _make(*, share, qkv_mode=QKVMode.SEPARATE_QKV, quant_algo=QuantAlgo.FP8, force_dynamic=False):
    config = DiffusionModelConfig(
        quant_config=QuantConfig(quant_algo=quant_algo) if quant_algo else QuantConfig(),
        force_dynamic_quantization=force_dynamic,
    )
    return Attention(
        hidden_size=HIDDEN,
        num_attention_heads=HEADS,
        num_key_value_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        qkv_mode=qkv_mode,
        qk_norm=False,
        bias=False,
        config=config,
        enable_sequence_parallel=False,
        share_qkv_input_quant=share,
    ).cuda()


def _set_scales(linear, input_scale=INPUT_SCALE):
    linear.weight_scale.data.fill_(WEIGHT_SCALE)
    linear.input_scale.data.fill_(input_scale)
    linear.inv_input_scale.data.fill_(1.0 / input_scale)


def _calibrate(attn, input_scale=INPUT_SCALE):
    for name in ("to_q", "to_k", "to_v"):
        _set_scales(getattr(attn, name), input_scale)


@requires_cuda
def test_default_does_not_share():
    """Every existing caller relies on the unshared default."""
    attn = _make(share=False)
    assert attn.share_qkv_input_quant is False
    _calibrate(attn)
    assert attn._shares_qkv_input_quant() is False


@requires_cuda
def test_fused_qkv_rejects_sharing():
    """A fused QKV projection already quantizes its input exactly once."""
    with pytest.raises(ValueError, match="SEPARATE_QKV"):
        _make(share=True, qkv_mode=QKVMode.FUSE_QKV)


@requires_cuda
def test_forced_dynamic_rejects_sharing():
    with pytest.raises(ValueError, match="force_dynamic_quantization"):
        _make(share=True, force_dynamic=True)


@requires_cuda
def test_unquantized_attention_does_not_share():
    """Without FP8 weights there is no static scale to quantize against."""
    attn = _make(share=True, quant_algo=None)
    assert attn._shares_qkv_input_quant() is False


@requires_cuda
def test_shared_input_is_quantized_once():
    """All three projections must receive the *same* already-quantized tensor.

    Three independent quantizations of one activation would also yield FP8 at
    every input, so dtype alone proves nothing -- storage identity is what
    discriminates.
    """
    attn = _make(share=True)
    _calibrate(attn)
    attn.post_load_weights()

    seen = {}

    def record(name):
        def hook(_module, inputs):
            seen[name] = (inputs[0].dtype, inputs[0].data_ptr())

        return hook

    for name in ("to_q", "to_k", "to_v"):
        getattr(attn, name).register_forward_pre_hook(record(name))

    x = torch.randn(1, TOKENS, HIDDEN, device="cuda", dtype=torch.bfloat16) * 0.02
    attn.get_qkv(x)

    assert {dtype for dtype, _ in seen.values()} == {torch.float8_e4m3fn}
    assert len({ptr for _, ptr in seen.values()}) == 1, (
        f"q/k/v should share one quantized activation, got {seen}"
    )


@requires_cuda
def test_sharing_matches_unshared_result():
    """Sharing must be a pure launch-count optimization, not a numerical change.

    Each Linear applies its own input_scale in the epilogue, and all three agree
    here, so quantizing once must be bit-identical to quantizing three times.
    """
    shared, unshared = _make(share=True), _make(share=False)
    torch.manual_seed(0)
    for name in ("to_q", "to_k", "to_v"):
        out_features = getattr(shared, name).out_features
        weight = (torch.randn(out_features, HIDDEN, device="cuda", dtype=torch.bfloat16) * 0.05).to(
            torch.float8_e4m3fn
        )
        for attn in (shared, unshared):
            getattr(attn, name).weight.data.copy_(weight)
            _set_scales(getattr(attn, name))
    shared.post_load_weights()

    x = torch.randn(1, TOKENS, HIDDEN, device="cuda", dtype=torch.bfloat16) * 0.02
    for actual, expected in zip(shared.get_qkv(x), unshared.get_qkv(x)):
        assert torch.equal(actual, expected)


@requires_cuda
def test_cross_attention_source_is_not_shared():
    """k/v read a different tensor than q, so there is no single activation."""
    attn = _make(share=True)
    _calibrate(attn)
    attn.post_load_weights()

    x = torch.randn(1, TOKENS, HIDDEN, device="cuda", dtype=torch.bfloat16) * 0.02
    encoder = torch.randn(1, TOKENS, HIDDEN, device="cuda", dtype=torch.bfloat16) * 0.02
    assert attn._can_share_qkv_quantize(x, encoder) is False

    seen = {}
    attn.to_q.register_forward_pre_hook(lambda _m, inputs: seen.update(q=inputs[0].dtype))
    attn.get_qkv(x, encoder_hidden_states=encoder)
    assert seen["q"] == torch.bfloat16


@requires_cuda
def test_post_load_weights_rejects_mismatched_input_scales():
    """A checkpoint violating the shared-scale invariant must fail loudly.

    to_k's GEMM would apply its own scale to activations quantized with to_q's,
    which silently corrupts the projection rather than erroring.
    """
    attn = _make(share=True)
    _calibrate(attn)
    _set_scales(attn.to_k, input_scale=INPUT_SCALE * 2)

    with pytest.raises(ValueError, match="same calibrated input_scale"):
        attn.post_load_weights()


@requires_cuda
def test_post_load_weights_is_a_noop_when_not_sharing():
    attn = _make(share=False)
    _calibrate(attn)
    _set_scales(attn.to_k, input_scale=INPUT_SCALE * 2)
    attn.post_load_weights()


@requires_cuda
def test_prequantized_input_is_passed_through():
    """An upstream fused norm+quant already produced FP8; do not re-quantize."""
    attn = _make(share=True)
    _calibrate(attn)
    attn.post_load_weights()

    x = torch.randn(TOKENS, HIDDEN, device="cuda", dtype=torch.bfloat16)
    assert attn._can_share_qkv_quantize(x.to(torch.float8_e4m3fn), None) is False


@requires_cuda
@pytest.mark.parametrize("shape", [(TOKENS, HIDDEN), (2, TOKENS, HIDDEN)], ids=["rank2", "rank3"])
def test_rank_is_preserved(shape):
    """The quantize reshape is a view; projections must still see their rank."""
    attn = _make(share=True)
    _calibrate(attn)
    attn.post_load_weights()

    x = torch.randn(*shape, device="cuda", dtype=torch.bfloat16) * 0.02
    q, k, v = attn.get_qkv(x)
    assert q.shape == (*shape[:-1], HEADS * HEAD_DIM)
    assert k.shape == v.shape == (*shape[:-1], KV_HEADS * HEAD_DIM)
