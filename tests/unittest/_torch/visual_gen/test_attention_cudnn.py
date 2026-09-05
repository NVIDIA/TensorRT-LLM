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

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.attention_backend.interface import PredefinedAttentionMask
from tensorrt_llm._torch.utils import unswizzle_sf
from tensorrt_llm._torch.visual_gen.attention_backend import create_attention
from tensorrt_llm._torch.visual_gen.attention_backend.cudnn import (
    CuDNNAttention,
    _quantize_mxfp8_qk,
    _quantize_mxfp8_v,
)
from tensorrt_llm._torch.visual_gen.attention_backend.interface import AttentionTensorLayout
from tensorrt_llm.visual_gen.args import AttentionConfig, QuantAttentionConfig

# Recipe name -> quant_attention_config accepted by AttentionConfig.
RECIPES = {
    "no_quant": None,
    "fp8": QuantAttentionConfig(qk_dtype="fp8", v_dtype="fp8"),
    "mxfp8": QuantAttentionConfig(qk_dtype="mxfp8", v_dtype="mxfp8"),
}

# (name, batch, num_heads, num_kv_heads, seq_len_q, seq_len_kv, head_dim)
SHAPES = [
    ("mha", 2, 8, 8, 512, 512, 128),
    ("gqa", 1, 16, 4, 1024, 1024, 128),
    ("mqa", 1, 16, 1, 256, 256, 64),
    ("cross_gqa", 1, 8, 2, 512, 333, 128),
]

# Cosine similarity against an FP32 reference. The quantized recipes are bounded by
# the FP8 P*V GEMM that cuDNN performs internally on both quantized paths.
MIN_COSINE = {"no_quant": 0.9999, "fp8": 0.995, "mxfp8": 0.995}


def _require_cudnn(recipe: str) -> None:
    if not torch.cuda.is_available():
        pytest.skip("cuDNN attention backend requires CUDA.")
    if recipe != "no_quant" and torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("cuDNN FP8/MXFP8 SDPA requires a Blackwell-class GPU (sm100+).")


def _make_qkv(batch, num_heads, num_kv_heads, seq_q, seq_kv, head_dim, device):
    torch.manual_seed(0)
    q = torch.randn(batch, seq_q, num_heads, head_dim, device=device, dtype=torch.bfloat16)
    k = torch.randn(batch, seq_kv, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    v = torch.randn(batch, seq_kv, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    return q, k, v


def _reference(q, k, v, is_causal):
    """FP32 SDPA reference plus its log-sum-exp."""
    q, k, v = (t.transpose(1, 2) for t in (q, k, v))
    num_heads, num_kv_heads = q.shape[1], k.shape[1]
    out = F.scaled_dot_product_attention(
        q.float(), k.float(), v.float(), is_causal=is_causal, enable_gqa=num_heads != num_kv_heads
    )
    k_rep = k.float().repeat_interleave(num_heads // num_kv_heads, dim=1)
    logits = (q.float() @ k_rep.transpose(-1, -2)) * q.shape[-1] ** -0.5
    if is_causal:
        seq_q, seq_kv = q.shape[2], k.shape[2]
        causal_mask = torch.ones(seq_q, seq_kv, device=q.device, dtype=torch.bool)
        logits = logits.masked_fill(causal_mask.triu(seq_kv - seq_q + 1), float("-inf"))
    return out.transpose(1, 2), torch.logsumexp(logits, dim=-1).transpose(1, 2)


@pytest.mark.parametrize("recipe", list(RECIPES))
@pytest.mark.parametrize("shape", SHAPES, ids=[s[0] for s in SHAPES])
@pytest.mark.parametrize("is_causal", [False, True])
def test_cudnn_attention(recipe, shape, is_causal):
    """Output and LSE match an FP32 SDPA reference for every recipe and mask."""
    _require_cudnn(recipe)
    _, batch, num_heads, num_kv_heads, seq_q, seq_kv, head_dim = shape
    if is_causal and seq_q != seq_kv:
        pytest.skip("Causal masking is only meaningful for self attention.")

    device = torch.device("cuda")
    q, k, v = _make_qkv(batch, num_heads, num_kv_heads, seq_q, seq_kv, head_dim, device)
    attention = CuDNNAttention(
        num_heads=num_heads,
        head_dim=head_dim,
        num_kv_heads=num_kv_heads,
        dtype=torch.bfloat16,
        quant_attention_config=RECIPES[recipe],
    )
    mask = PredefinedAttentionMask.CAUSAL if is_causal else PredefinedAttentionMask.FULL
    output, lse = attention.forward_with_lse(q, k, v, attention_mask=mask)

    ref_out, ref_lse = _reference(q, k, v, is_causal)
    assert output.shape == (batch, seq_q, num_heads, head_dim)
    assert output.dtype == torch.bfloat16
    assert lse.shape == (batch, seq_q, num_heads)

    cosine = F.cosine_similarity(output.float().flatten(), ref_out.flatten(), dim=0).item()
    assert cosine > MIN_COSINE[recipe], f"{recipe}: cosine similarity {cosine} too low"
    # LSE is computed in FP32 by cuDNN on all recipes; only the quantized logits differ.
    torch.testing.assert_close(lse, ref_lse, atol=0.5 if recipe != "no_quant" else 1e-2, rtol=0.0)


@pytest.mark.parametrize("seq_len", [128, 1000])
@pytest.mark.parametrize("head_dim", [32, 64, 128])
def test_mxfp8_scale_factor_layout_roundtrip(seq_len, head_dim):
    """Dequantizing with cuDNN's documented layout reproduces the input tensor.

    Guards the two scale-factor layouts the MXFP8 recipe relies on: Q/K block along
    the head dim (``[B, H, S_padded, D_scale]``, ``stride[3] == 1``) and V blocks
    along the sequence dim (``[B, H, S_scale, D_padded]``, ``stride[2] == 1``).
    Both come out of ``torch.ops.trtllm.mxfp8_quantize``'s swizzled (``F8_128x4``)
    buffer, so a layout regression shows up here rather than as a silent accuracy
    loss inside cuDNN.
    """
    _require_cudnn("mxfp8")
    device = torch.device("cuda")
    batch, num_heads = 2, 3
    torch.manual_seed(0)

    def dequant_e8m0(scale_factors):
        return torch.exp2(scale_factors.float() - 127.0)

    # Block-aligned dynamic range: a wrong scale mapping cannot average out.
    channel_gain = 10.0 ** (torch.arange(head_dim, device=device) // 32 % 7 - 3)
    x_qk = (
        torch.randn(batch, num_heads, seq_len, head_dim, device=device) * channel_gain
    ).bfloat16()
    x_q, scale_factors = _quantize_mxfp8_qk(x_qk)
    seq_padded, d_scale = scale_factors.shape[2], scale_factors.shape[3]
    linear_sf = unswizzle_sf(
        scale_factors.reshape(-1), batch * num_heads * seq_padded, d_scale * 32, 32
    ).view(batch, num_heads, seq_padded, d_scale)[:, :, :seq_len, : head_dim // 32]
    dequantized = x_q.float() * dequant_e8m0(linear_sf).repeat_interleave(32, dim=-1)
    rel_err = ((dequantized - x_qk.float()).norm() / x_qk.float().norm()).item()
    assert rel_err < 0.05, f"Q/K MXFP8 round-trip error {rel_err} exceeds e4m3 block noise"

    token_gain = (10.0 ** (torch.arange(seq_len, device=device) // 32 % 7 - 3)).view(seq_len, 1)
    x_v = (torch.randn(batch, num_heads, seq_len, head_dim, device=device) * token_gain).bfloat16()
    v_q, v_sf = _quantize_mxfp8_v(x_v)
    s_scale, d_padded = v_sf.shape[2], v_sf.shape[3]
    assert v_sf.stride(2) == 1, "cuDNN requires descale_v to have a contiguous S-scale dimension"
    linear_v_sf = unswizzle_sf(
        v_sf.permute(0, 1, 3, 2).reshape(-1), batch * num_heads * d_padded, s_scale * 32, 32
    ).view(batch, num_heads, d_padded, s_scale)[:, :, :head_dim, :]
    v_scale = dequant_e8m0(linear_v_sf).repeat_interleave(32, dim=-1)[..., :seq_len]
    dequantized_v = v_q.float() * v_scale.permute(0, 1, 3, 2)
    rel_err_v = ((dequantized_v - x_v.float()).norm() / x_v.float().norm()).item()
    assert rel_err_v < 0.05, f"V MXFP8 round-trip error {rel_err_v} exceeds e4m3 block noise"


def test_cudnn_graph_cache_reuses_plans():
    """Repeated calls with the same geometry reuse one compiled graph."""
    _require_cudnn("no_quant")
    device = torch.device("cuda")
    q, k, v = _make_qkv(1, 4, 4, 256, 256, 64, device)
    attention = CuDNNAttention(num_heads=4, head_dim=64, dtype=torch.bfloat16)

    CuDNNAttention.clear_graph_cache()
    attention.forward(q, k, v)
    assert len(CuDNNAttention._graph_cache) == 1
    attention.forward(q, k, v)
    assert len(CuDNNAttention._graph_cache) == 1
    # A different mask needs its own plan.
    attention.forward(q, k, v, attention_mask=PredefinedAttentionMask.CAUSAL)
    assert len(CuDNNAttention._graph_cache) == 2


def test_cudnn_backend_wires_validated_recipes():
    """create_attention("CUDNN") wires the validated recipe into the backend."""
    _require_cudnn("mxfp8")
    attention = create_attention(
        backend="CUDNN",
        layer_idx=0,
        num_heads=8,
        head_dim=128,
        num_kv_heads=8,
        dtype=torch.bfloat16,
        attention_config=AttentionConfig(backend="CUDNN", quant_attention_config=RECIPES["mxfp8"]),
    )
    assert isinstance(attention, CuDNNAttention)
    assert attention.recipe == "mxfp8"
    assert attention.preferred_layout == AttentionTensorLayout.NHD
    assert attention.support_lse() and not attention.support_fused_qkv()


@pytest.mark.parametrize("num_kv_heads", [8, 2], ids=["mha", "gqa"])
def test_cudnn_fp8_attention_with_fused_qkv(num_kv_heads):
    """Packed Q/K/V are quantized through the shared-scale path; separate ones are not.

    The shared scale shifts the output slightly, so the two paths are compared at the
    suite's FP8 tolerance rather than for equality.
    """
    _require_cudnn("fp8")
    device = torch.device("cuda")
    batch, num_heads, seq_len, head_dim = 1, 8, 256, 64
    q_dim, kv_dim = num_heads * head_dim, num_kv_heads * head_dim

    # Q/K/V as views into one buffer, the way get_qkv splits under FUSE_QKV.
    torch.manual_seed(0)
    qkv = torch.randn(batch, seq_len, q_dim + 2 * kv_dim, device=device, dtype=torch.bfloat16)
    q, k, v = (
        x.view(batch, seq_len, -1, head_dim) for x in qkv.split([q_dim, kv_dim, kv_dim], dim=-1)
    )

    assert CuDNNAttention._is_fused_qkv(q, k, v)
    fused_qkv = CuDNNAttention._as_fused_qkv(q, k, v)
    assert fused_qkv.data_ptr() == qkv.data_ptr(), "fused view must alias, not copy"
    torch.testing.assert_close(fused_qkv, qkv, atol=0.0, rtol=0.0)

    # Cloning breaks the shared storage, so equal values take the per-tensor path.
    separate = tuple(x.contiguous() for x in (q, k, v))
    assert not CuDNNAttention._is_fused_qkv(*separate)

    attention = CuDNNAttention(
        num_heads=num_heads,
        head_dim=head_dim,
        num_kv_heads=num_kv_heads,
        dtype=torch.bfloat16,
        quant_attention_config=RECIPES["fp8"],
    )
    fused_out = attention.forward(q, k, v)
    separate_out = attention.forward(*separate)
    ref_out, _ = _reference(q, k, v, is_causal=False)

    for what, a, b in (
        ("fused vs reference", fused_out.float(), ref_out),
        ("separate vs reference", separate_out.float(), ref_out),
        ("fused vs separate", fused_out.float(), separate_out.float()),
    ):
        cosine = F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
        assert cosine > MIN_COSINE["fp8"], f"{what}: cosine similarity {cosine} too low"
