# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Tests for the TransformerEngine FP8 attention backend (TEAttention).

Three tests, each covering one failure mode of the backend:
  1. numerics of ``forward``            -- output matches a full-precision reference
  2. mask handling                      -- FULL / None / CAUSAL / unsupported
  3. object contract                    -- layout, no-LSE, op cache, config rejection

FP8 tolerances: TEAttention quantizes Q/K/V, so it sits at roughly 0.08 relative
error against an fp32 reference (measured on GB300 across S=128..1024 for
B=1 H=10 D=128). Comparisons against a full-precision reference therefore use
cosine similarity plus a loose relative error rather than elementwise
assert_close.
"""

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.attention_backend.interface import PredefinedAttentionMask
from tensorrt_llm._torch.visual_gen.attention_backend.interface import AttentionTensorLayout
from tensorrt_llm._torch.visual_gen.attention_backend.te import TEAttention

try:
    import transformer_engine  # noqa: F401

    _te_available = True
except ImportError:
    _te_available = False

pytestmark = pytest.mark.skipif(
    not _te_available or not torch.cuda.is_available(),
    reason="transformer_engine and GPU required",
)

FP8_COS_SIM = 0.99
FP8_REL_ERR = 0.12


@pytest.fixture
def make_te_attn():
    def _make(num_heads=4, head_dim=64, num_kv_heads=None):
        return TEAttention(
            layer_idx=0,
            num_heads=num_heads,
            head_dim=head_dim,
            num_kv_heads=num_kv_heads or num_heads,
        )

    return _make


def _qkv(B, S, H, D, Hkv=None, seed=0):
    torch.manual_seed(seed)
    kw = dict(device="cuda", dtype=torch.bfloat16)
    return (
        torch.randn(B, S, H, D, **kw),
        torch.randn(B, S, Hkv or H, D, **kw),
        torch.randn(B, S, Hkv or H, D, **kw),
    )


def _reference(q, k, v, D, is_causal=False, enable_gqa=False):
    """Full-precision SDPA reference in [B, S, H, D] layout."""
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2).float(),
        k.transpose(1, 2).float(),
        v.transpose(1, 2).float(),
        scale=1.0 / (D**0.5),
        is_causal=is_causal,
        enable_gqa=enable_gqa,
    )
    return out.transpose(1, 2).to(q.dtype)


def _assert_close_fp8(out, ref, what):
    o, r = out.reshape(-1).float(), ref.reshape(-1).float()
    cos = F.cosine_similarity(o, r, dim=0).item()
    rel = (torch.linalg.vector_norm(o - r) / torch.linalg.vector_norm(r)).item()
    assert cos > FP8_COS_SIM, f"{what}: cosine similarity {cos:.4f} <= {FP8_COS_SIM}"
    assert rel < FP8_REL_ERR, f"{what}: relative error {rel:.4f} >= {FP8_REL_ERR}"


@pytest.mark.parametrize("S,Hkv", [(64, None), (256, None), (1024, None), (256, 2)])
def test_forward_matches_reference(make_te_attn, S, Hkv):
    """forward() returns a well-formed [B, S, H, D] tensor matching full-precision SDPA.

    Parametrized over sequence length and over GQA (Hkv=2 against H=8), because
    the GQA path takes a different branch in _parse_inputs and previously did not
    agree with the non-GQA one.
    """
    B, H, D = 1, 8, 64
    attn = make_te_attn(num_heads=H, head_dim=D, num_kv_heads=Hkv)
    q, k, v = _qkv(B, S, H, D, Hkv, seed=42)

    with torch.no_grad():
        out = attn(q, k, v)

    assert out.shape == (B, S, H, D)
    assert torch.isfinite(out).all(), "output contains NaN or Inf"
    _assert_close_fp8(out, _reference(q, k, v, D, enable_gqa=Hkv is not None), f"S={S} Hkv={Hkv}")


def test_attention_mask_handling(make_te_attn):
    """FULL and None mean no mask, CAUSAL masks, anything else is rejected.

    None is the no-mask convention across the backends -- VanillaAttention derives
    causality as `attention_mask == CAUSAL`, and Attention2DAttention explicitly
    permits None and forwards it to the inner backend unchanged.
    """
    B, S, H, D = 1, 64, 4, 64
    attn = make_te_attn(num_heads=H, head_dim=D)
    q, k, v = _qkv(B, S, H, D, seed=3)

    with torch.no_grad():
        out_full = attn(q, k, v, attention_mask=PredefinedAttentionMask.FULL)
        out_none = attn(q, k, v, attention_mask=None)
        out_causal = attn(q, k, v, attention_mask=PredefinedAttentionMask.CAUSAL)

    # FP8 tolerance, not exact: current scaling rederives the scale on every call.
    _assert_close_fp8(out_none, out_full, "attention_mask=None must mean FULL")
    _assert_close_fp8(out_causal, _reference(q, k, v, D, is_causal=True), "causal")
    assert not torch.allclose(out_full, out_causal, atol=1e-3), "causal must differ from full"

    with pytest.raises(NotImplementedError, match="key_padding_mask"):
        attn(q, k, v, key_padding_mask=torch.ones(B, S, device="cuda", dtype=torch.bool))
    with pytest.raises(NotImplementedError, match="attention_mask"):
        attn(q, k, v, attention_mask="bidirectional")


def test_backend_contract(make_te_attn):
    """Layout/LSE declarations, per-trait op caching, and config rejection."""
    attn = make_te_attn(num_heads=4, head_dim=64)
    assert attn.preferred_layout == AttentionTensorLayout.NHD
    assert TEAttention.support_fused_qkv() is False

    # No LSE: this is the flag Attention2D/Ring read to reject TE as an inner backend.
    assert TEAttention.support_lse() is False

    # Each trait keeps its own instance; rebuilding on a mask switch would reset FP8 calibration.
    q, k, v = _qkv(1, 64, 4, 64, seed=5)
    with torch.no_grad():
        attn(q, k, v, attention_mask=PredefinedAttentionMask.FULL)
        attn(q, k, v, attention_mask=PredefinedAttentionMask.CAUSAL)
    op_full = attn._get_attn_op(None, "no_mask")
    op_causal = attn._get_attn_op(None, "causal")
    assert op_full is not op_causal
    assert len(attn._attn_ops) == 2
    with torch.no_grad():  # switching back reuses, never rebuilds
        attn(q, k, v, attention_mask=PredefinedAttentionMask.FULL)
    assert attn._get_attn_op(None, "no_mask") is op_full
    assert len(attn._attn_ops) == 2

    # TE runs its own FP8 recipe and must not silently absorb quant_attention_config.
    with pytest.raises(NotImplementedError, match="quant_attention_config"):
        TEAttention(num_heads=4, head_dim=64, quant_attention_config=object())
