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
"""Tests for the TransformerEngine FP8 visual-gen attention backend."""

import pytest
import torch
import torch.nn.functional as F
from pydantic import ValidationError

from tensorrt_llm._torch.attention.backends.interface import PredefinedAttentionMask
from tensorrt_llm._torch.visual_gen.attention_backend import create_attention
from tensorrt_llm._torch.visual_gen.attention_backend.interface import AttentionTensorLayout
from tensorrt_llm.visual_gen.args import AttentionConfig, QuantAttentionConfig

try:
    import transformer_engine.pytorch  # noqa: F401

    _te_available = True
except ImportError:
    _te_available = False

# (name, batch, num_heads, num_kv_heads, seq_len_q, seq_len_kv, head_dim)
SHAPES = [
    ("mha", 2, 8, 8, 512, 512, 128),
    ("gqa", 1, 16, 4, 1024, 1024, 128),
    ("cross", 1, 8, 8, 1024, 512, 128),
]


# Set from Rubin measurements (see the flat-softmax tests below).
MAX_FLAT_SOFTMAX_REL_ERR = 0.06
STALE_TO_FIXED_MIN_RATIO = 3.0
# Small V keeps the attention output (~std(v) / sqrt(seq)) far below E4M3's normal
# range at scale 1.0; per-call scales are invariant to this magnitude.
FLAT_SOFTMAX_V_AMP = 0.1


def _require_te_fp8() -> None:
    if not torch.cuda.is_available():
        pytest.skip("TE FP8 attention requires CUDA.")
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("TE FP8 fused attention test targets Blackwell/Rubin-class GPUs (sm100+).")
    if not _te_available:
        pytest.fail("TransformerEngine is not importable; the TE backend tests need it.")


def _make_qkv(batch, num_heads, num_kv_heads, seq_q, seq_kv, head_dim, amp=1.0, v_amp=1.0):
    torch.manual_seed(0)
    dev = torch.device("cuda")
    q = amp * torch.randn(batch, seq_q, num_heads, head_dim, device=dev, dtype=torch.bfloat16)
    k = amp * torch.randn(batch, seq_kv, num_kv_heads, head_dim, device=dev, dtype=torch.bfloat16)
    v = v_amp * torch.randn(batch, seq_kv, num_kv_heads, head_dim, device=dev, dtype=torch.bfloat16)
    return q, k, v


def _reference(q, k, v, is_causal):
    """FP32 SDPA in NHD layout."""
    rep = q.shape[2] // k.shape[2]
    k = k.repeat_interleave(rep, dim=2)
    v = v.repeat_interleave(rep, dim=2)
    out = F.scaled_dot_product_attention(
        q.float().transpose(1, 2),
        k.float().transpose(1, 2),
        v.float().transpose(1, 2),
        is_causal=is_causal,
    )
    return out.transpose(1, 2)


def _rel_err(out, ref):
    return ((out.float() - ref).norm() / ref.norm()).item()


def _make_te(num_heads, num_kv_heads, head_dim):
    return create_attention(
        backend="TE",
        layer_idx=0,
        num_heads=num_heads,
        head_dim=head_dim,
        num_kv_heads=num_kv_heads,
        dtype=torch.bfloat16,
        attention_config=AttentionConfig(backend="TE"),
    )


def test_te_backend_config_accepted():
    assert AttentionConfig(backend="TE").backend == "TE"


def test_te_backend_rejects_quant_attention_config():
    """TE owns its FP8 recipe; an explicit quant_attention_config is a config mistake."""
    with pytest.raises(ValidationError, match="quant_attention_config requires backend"):
        AttentionConfig(
            backend="TE", quant_attention_config=QuantAttentionConfig(qk_dtype="fp8", v_dtype="fp8")
        )


def test_te_backend_wiring():
    _require_te_fp8()
    attention = _make_te(8, 8, 128)
    assert type(attention).__name__ == "TEAttention"
    assert attention.preferred_layout == AttentionTensorLayout.NHD
    assert not attention.support_fused_qkv() and not attention.support_lse()


@pytest.mark.parametrize("shape", SHAPES, ids=[s[0] for s in SHAPES])
@pytest.mark.parametrize("is_causal", [False, True])
def test_te_fp8_attention_accuracy(shape, is_causal):
    _require_te_fp8()
    _, batch, num_heads, num_kv_heads, seq_q, seq_kv, head_dim = shape
    if is_causal and seq_q != seq_kv:
        pytest.skip("causal attention is only exercised for self-attention shapes")
    q, k, v = _make_qkv(batch, num_heads, num_kv_heads, seq_q, seq_kv, head_dim)
    attention = _make_te(num_heads, num_kv_heads, head_dim)
    mask = PredefinedAttentionMask.CAUSAL if is_causal else PredefinedAttentionMask.FULL

    with torch.no_grad():
        out = attention.forward(q, k, v, attention_mask=mask)
    ref = _reference(q, k, v, is_causal)

    assert out.shape == q.shape
    cos = F.cosine_similarity(out.float().flatten(), ref.flatten(), dim=0).item()
    assert cos > 0.995, f"cosine {cos} vs FP32 reference"


def _run_te(attention, q, k, v, grad_mode):
    ctx = torch.no_grad() if grad_mode == "no_grad" else torch.inference_mode()
    with ctx:
        return attention.forward(q, k, v)


@pytest.mark.parametrize("grad_mode", ["no_grad", "inference_mode"])
def test_te_fp8_scales_track_inputs_without_autograd(grad_mode):
    """Grad-free inference must write per-call FP8 scales, not keep the initial 1.0.

    Small Q/K give a near-uniform softmax over 16k keys, so each output row
    averages many V rows and is small (~std(v) / sqrt(seq)). At scale 1.0 such
    outputs fall into E4M3's subnormal range and lose precision; per-call
    scales keep them in range. (Large Q/K instead give a peaked softmax whose
    error is dominated by E4M3 rounding of Q and K, which no scale changes, so
    that regime cannot detect the bug.) The two calls differ 2x in magnitude,
    so the checked scales must come from the current inputs.
    """
    _require_te_fp8()
    num_heads, head_dim, seq = 8, 128, 16384
    attention = _make_te(num_heads, num_heads, head_dim)

    for amp in (0.25, 0.5):
        q, k, v = _make_qkv(
            1, num_heads, num_heads, seq, seq, head_dim, amp=amp, v_amp=FLAT_SOFTMAX_V_AMP
        )
        err = _rel_err(_run_te(attention, q, k, v, grad_mode), _reference(q, k, v, False))
        assert err < MAX_FLAT_SOFTMAX_REL_ERR, f"amp={amp}: relative error {err}"

        scale = attention._attn_op.fp8_meta["scaling_fwd"].scale.float()
        amax_qkv = max(t.abs().amax().item() for t in (q, k, v))
        assert scale[2].item() == pytest.approx(448.0 / amax_qkv, rel=1e-3)
        assert scale[3].item() == pytest.approx(448.0 / v.abs().amax().item(), rel=1e-3)
        assert scale[8].item() == pytest.approx(448.0)


def test_te_fp8_stale_scales_lose_flat_softmax(monkeypatch):
    """Negative control: without per-call scales the same inputs lose accuracy.

    Shows the flat-softmax regime above actually detects stale scales.
    """
    _require_te_fp8()
    num_heads, head_dim, seq = 8, 128, 16384
    q, k, v = _make_qkv(
        1, num_heads, num_heads, seq, seq, head_dim, amp=0.25, v_amp=FLAT_SOFTMAX_V_AMP
    )
    ref = _reference(q, k, v, False)
    fixed = _rel_err(_run_te(_make_te(num_heads, num_heads, head_dim), q, k, v, "no_grad"), ref)

    from tensorrt_llm._torch.visual_gen.attention_backend import te as te_module

    monkeypatch.setattr(te_module.TEAttention, "_set_forward_scales", lambda self, q, k, v: None)
    stale = _rel_err(_run_te(_make_te(num_heads, num_heads, head_dim), q, k, v, "no_grad"), ref)
    assert stale > STALE_TO_FIXED_MIN_RATIO * fixed, f"stale {stale} vs per-call {fixed}"


def test_te_rejects_key_padding_mask():
    _require_te_fp8()
    q, k, v = _make_qkv(1, 8, 8, 256, 256, 128)
    attention = _make_te(8, 8, 128)
    with pytest.raises(NotImplementedError, match="key_padding_mask"):
        attention.forward(
            q, k, v, key_padding_mask=torch.ones(1, 256, dtype=torch.bool, device="cuda")
        )
