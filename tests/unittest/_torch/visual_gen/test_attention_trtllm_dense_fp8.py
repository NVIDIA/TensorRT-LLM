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
"""TRTLLM per-tensor FP8 attention on the trtllm-gen dense FP8 FMHA library."""

import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.attention.backends.fmha.trtllm_gen_dense_fp8 import TrtllmGenDenseFp8Fmha
from tensorrt_llm._torch.visual_gen.attention.attention import Attention, QKVMode
from tensorrt_llm._torch.visual_gen.attention.backends.cudnn import CuDNNAttention
from tensorrt_llm._torch.visual_gen.attention.backends.utils import create_attention
from tensorrt_llm._torch.visual_gen.attention.backends.vanilla import VanillaAttention
from tensorrt_llm._torch.visual_gen.config import (
    DiffusionModelConfig,
    create_attention_metadata_state,
)
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.visual_gen.args import AttentionConfig, QuantAttentionConfig

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() != 107,
    reason="trtllm-gen dense FP8 attention serves the per-tensor FP8 recipe on SM107",
)

E4M3 = torch.float8_e4m3fn
E4M3_MAX = 448.0
HEAD_DIM = 128
FP8_RECIPE = QuantAttentionConfig(qk_dtype="fp8", v_dtype="fp8")


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


def _sdpa(q, k, v):
    """[B, S, H, D] reference in FP32."""
    out = F.scaled_dot_product_attention(*(t.float().transpose(1, 2) for t in (q, k, v)))
    return out.transpose(1, 2)


def _qkv(batch, seq, heads, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return [
        torch.randn(batch, seq, heads, HEAD_DIM, device="cuda", generator=g).bfloat16()
        for _ in range(3)
    ]


def _backend(backend, heads, quant=FP8_RECIPE):
    return create_attention(
        backend=backend,
        layer_idx=0,
        num_heads=heads,
        head_dim=HEAD_DIM,
        dtype=torch.bfloat16,
        attention_config=AttentionConfig(backend=backend, quant_attention_config=quant),
        attention_metadata_state=create_attention_metadata_state(),
    )


def _model_config(backend, quant=FP8_RECIPE, heads=12):
    config = DiffusionModelConfig(
        pretrained_config=SimpleNamespace(
            hidden_size=heads * HEAD_DIM,
            num_attention_heads=heads,
            attention_head_dim=HEAD_DIM,
            eps=1e-6,
        ),
        attention=AttentionConfig(backend=backend, quant_attention_config=quant),
    )
    config.attention_metadata_state = create_attention_metadata_state()
    return config


def _wan_self_attention(backend, quant=FP8_RECIPE, heads=12, **kwargs):
    attn = Attention(
        hidden_size=heads * HEAD_DIM,
        num_attention_heads=heads,
        qkv_mode=QKVMode.FUSE_QKV,
        qk_norm=True,
        qk_norm_mode="full",
        fuse_qk_norm_rope=True,
        config=_model_config(backend, quant, heads),
        enable_sequence_parallel=False,
        **kwargs,
    ).cuda()
    g = torch.Generator(device="cuda").manual_seed(7)
    with torch.no_grad():
        for name, p in attn.named_parameters():
            if "norm" in name:
                p.copy_(1.0 + 0.1 * torch.randn(p.shape, device="cuda", generator=g))
            else:
                p.copy_(0.02 * torch.randn(p.shape, device="cuda", generator=g))
    return attn


def _wan_rope(seq):
    """Wan-style [1, S, 1, D] cos/sin tables with interleaved pairs."""
    pos = torch.arange(seq, dtype=torch.float64)
    inv = 1.0 / (10000 ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float64) / HEAD_DIM))
    angles = torch.outer(pos, inv)
    cos = angles.cos().repeat_interleave(2, dim=-1).float()
    sin = angles.sin().repeat_interleave(2, dim=-1).float()
    return cos.view(1, seq, 1, HEAD_DIM).cuda(), sin.view(1, seq, 1, HEAD_DIM).cuda()


@pytest.mark.parametrize("batch,seq", [(1, 1560), (2, 1560)])
@pytest.mark.parametrize("heads", [10, 40])
def test_bf16_input_matches_sdpa_and_cudnn(batch, seq, heads):
    trtllm = _backend("TRTLLM", heads)
    assert any(isinstance(f, TrtllmGenDenseFp8Fmha) for f in trtllm._fmha_manager.fmha_libs)
    q, k, v = _qkv(batch, seq, heads)
    out = trtllm.forward(q=q, k=k, v=v, batch_size=batch, seq_len=seq)
    assert out.shape == (batch, seq, heads * HEAD_DIM) and out.dtype == torch.bfloat16
    ref = _sdpa(q, k, v).reshape(batch, seq, -1)
    assert _cosine(out, ref) > 0.99

    cudnn = _backend("CUDNN", heads)
    out_cudnn = cudnn.forward(q=q, k=k, v=v, batch_size=batch, seq_len=seq)
    assert _cosine(out, out_cudnn.reshape(batch, seq, -1)) > 0.998


@pytest.mark.parametrize("out_fp8", [False, True])
def test_prequantized_matches_sdpa(out_fp8):
    batch, seq, heads = 2, 1560, 40
    trtllm = _backend("TRTLLM", heads)
    assert trtllm.accepts_fp8_qkv()
    q, k, v = _qkv(batch, seq, heads, seed=1)
    qk_mul = math.sqrt(math.log2(math.e) / math.sqrt(HEAD_DIM))
    v_descale = (v.float().abs().amax() / E4M3_MAX).reshape(1)
    q8, k8 = ((t.float() * qk_mul).to(E4M3) for t in (q, k))
    v8 = (v.float() / v_descale).to(E4M3)
    ref = _sdpa(q, k, v)
    out_scale = (ref.abs().amax() / E4M3_MAX).reshape(1) if out_fp8 else None

    out = trtllm.forward_fp8(q8, k8, v8, v_descale, batch, seq, out_scale=out_scale)
    assert out.shape == (batch, seq, heads, HEAD_DIM)
    assert out.dtype == (E4M3 if out_fp8 else torch.bfloat16)
    dequant = out.float() * out_scale if out_fp8 else out.float()
    assert _cosine(dequant, ref) > 0.99


def test_attention_module_uses_fp8_producer():
    batch, seq, heads = 2, 1560, 12
    attn = _wan_self_attention("TRTLLM", heads=heads)
    assert attn._fp8_qkv_producer
    hidden = torch.randn(batch, seq, heads * HEAD_DIM, device="cuda").bfloat16()
    freqs = _wan_rope(seq)
    with torch.no_grad():
        out = attn(hidden, freqs=freqs)
        # Same module through the BF16 QK norm + RoPE and the backend's own quantization.
        attn._fp8_qkv_producer = False
        out_backend_quant = attn(hidden, freqs=freqs)
        vanilla = _wan_self_attention("VANILLA", quant=None, heads=heads)
        vanilla.load_state_dict(attn.state_dict())
        out_bf16 = vanilla(hidden, freqs=freqs)
    assert out.shape == out_bf16.shape
    assert _cosine(out, out_backend_quant) > 0.995
    assert _cosine(out, out_bf16) > 0.99


def test_attention_module_fp8_producer_under_compile():
    batch, seq, heads = 1, 1560, 12
    attn = _wan_self_attention("TRTLLM", heads=heads)
    hidden = torch.randn(batch, seq, heads * HEAD_DIM, device="cuda").bfloat16()
    freqs = _wan_rope(seq)
    with torch.no_grad():
        eager = attn(hidden, freqs=freqs)
        compiled = torch.compile(attn, fullgraph=True)(hidden, freqs=freqs)
    torch.testing.assert_close(compiled, eager, rtol=0.02, atol=0.02)
    assert _cosine(compiled, eager) > 0.9999


@pytest.mark.parametrize(
    "backend,kwargs",
    [
        ("TRTLLM", {"qk_norm_mode": "per_head"}),
        ("TRTLLM", {"fuse_qk_norm_rope": False}),
        ("CUDNN", {}),
    ],
)
def test_fp8_producer_off(backend, kwargs):
    defaults = dict(
        qkv_mode=QKVMode.FUSE_QKV, qk_norm=True, qk_norm_mode="full", fuse_qk_norm_rope=True
    )
    defaults.update(kwargs)
    attn = Attention(
        hidden_size=12 * HEAD_DIM,
        num_attention_heads=12,
        config=_model_config(backend),
        enable_sequence_parallel=False,
        **defaults,
    )
    assert not attn._fp8_qkv_producer


def test_cross_attention_runs_on_cudnn_fp8(monkeypatch):
    def cross_attention():
        return Attention(
            hidden_size=12 * HEAD_DIM,
            num_attention_heads=12,
            qkv_mode=QKVMode.SEPARATE_QKV,
            qk_norm=True,
            config=_model_config("TRTLLM"),
            enable_sequence_parallel=False,
        )

    assert isinstance(cross_attention().attn, CuDNNAttention)
    # PrimTS serves the recipe when enabled; cross-attention keeps the VANILLA fallback.
    monkeypatch.setenv("TLLM_FMHA_LIBS", "+prims_ts")
    assert isinstance(cross_attention().attn, VanillaAttention)


def test_dense_fp8_library_needs_the_recipe():
    trtllm = _backend("TRTLLM", 10, quant=None)
    assert not any(isinstance(f, TrtllmGenDenseFp8Fmha) for f in trtllm._fmha_manager.fmha_libs)
    assert not trtllm.accepts_fp8_qkv()
