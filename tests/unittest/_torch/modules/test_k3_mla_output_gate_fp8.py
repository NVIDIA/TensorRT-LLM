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
"""Kimi K3 MLA output gate: the fused gate + FP8 quant and the overlapped g_proj.

* ``gate_mul_fp8_r128c4(a, g)`` must equal, byte for byte, the E4M3 codes and the whole R128c4
  UE8M0 scale buffer (padding rows included) of
  ``fp8_quantize_1x128_cutedsl_ue8m0(a * g.sigmoid())``, and the gated o_proj output through
  the K3 FP8 weight-read projection must be unchanged (Rubin only).
* ``TRTLLM_K3_MLA_GATE_FP8`` / ``TRTLLM_K3_MLA_GATE_OVERLAP`` dispatch in
  ``KimiK3MLAAttention``: same output as the unfused, serial path, eagerly and in a CUDA graph.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

_rubin = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="the R128c4 quantizer and the CuTe DSL MXFP8 GEMM need an SM107 (Rubin) GPU",
)

HIDDEN = 7168


def _inputs(m: int, k: int, seed: int):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    channel = torch.exp(torch.randn(k, device="cuda", generator=gen))
    a = (torch.randn((m, k), device="cuda", generator=gen) * channel).to(torch.bfloat16)
    g = (torch.randn((m, k), device="cuda", generator=gen) * 4).to(torch.bfloat16)
    if m > 4:
        a[0, :128] = 0  # an all-zero block: amax floored at 1e-10
        g[1, :64] = 90.0  # saturated gates
        g[2, :64] = -90.0
        a[3, 128:256] *= 2.0**100  # a large block (scale exponent far from the others)
        a[4, 256:384] *= 2.0**-100  # a tiny block
    return a, g


def _quantize_ref(y: torch.Tensor):
    return torch.ops.trtllm.fp8_quantize_1x128_cutedsl_ue8m0(y.contiguous())


@_rubin
@pytest.mark.parametrize("k", [1536, 12288])
@pytest.mark.parametrize("m", [1, 6, 12, 64, 127, 128, 129, 224, 256])
def test_bitwise_vs_unfused(m, k):
    from tensorrt_llm._torch.modules.kimi_k3_mla.output_gate_fp8 import gate_mul_fp8_r128c4

    a, g = _inputs(m, k, seed=7 * m + k)
    codes, scales = gate_mul_fp8_r128c4(a, g)
    ref_codes, ref_scales = _quantize_ref(a * g.sigmoid())
    assert torch.equal(codes.view(torch.uint8), ref_codes.view(torch.uint8))
    assert scales.dtype == ref_scales.dtype == torch.uint8
    assert scales.numel() == ref_scales.numel() == (m + 127) // 128 * 128 * (k // 32)
    assert torch.equal(scales, ref_scales.reshape(-1))


@_rubin
@pytest.mark.parametrize("inf", [float("inf"), -float("inf")])
def test_inf_amax_saturates_like_the_quantizer(inf):
    """An inf amax gives UE8M0 254; the inf code saturates to +-448 (0x7E / 0xFE) as in C++."""
    from tensorrt_llm._torch.modules.kimi_k3_mla.output_gate_fp8 import gate_mul_fp8_r128c4

    m, k = 6, 1536
    a, g = _inputs(m, k, seed=11)
    a[2, 5] = inf
    a[2, 7] = 3.0
    codes, scales = gate_mul_fp8_r128c4(a, g)
    ref_codes, ref_scales = _quantize_ref(a * g.sigmoid())
    assert torch.equal(codes.view(torch.uint8), ref_codes.view(torch.uint8))
    assert torch.equal(scales, ref_scales.reshape(-1))


@_rubin
@pytest.mark.parametrize("m", [6, 129])
def test_sigmoid_flush_to_zero_band(m):
    """Gates in (-88.72, -87.34): PyTorch's sigmoid is denormal there, Triton's libdevice
    flushes it to zero; the products are far below the quantizer's 1e-10 amax floor either way."""
    from tensorrt_llm._torch.modules.kimi_k3_mla.output_gate_fp8 import gate_mul_fp8_r128c4

    k = 1536
    a, g = _inputs(m, k, seed=13)
    band = torch.linspace(-88.9, -87.1, k, device="cuda").to(torch.bfloat16)
    g[:, : k // 2] = band[: k // 2]
    g[:, k // 2 :] = band[: k // 2]
    codes, scales = gate_mul_fp8_r128c4(a, g)
    ref_codes, ref_scales = _quantize_ref(a * g.sigmoid())
    assert torch.equal(codes.view(torch.uint8), ref_codes.view(torch.uint8))
    assert torch.equal(scales, ref_scales.reshape(-1))


@_rubin
def test_strided_rows():
    from tensorrt_llm._torch.modules.kimi_k3_mla.output_gate_fp8 import gate_mul_fp8_r128c4

    m, k = 200, 1536
    a_full, g_full = _inputs(m, 2 * k, seed=5)
    a, g = a_full[:, :k], g_full[:, k:]
    codes, scales = gate_mul_fp8_r128c4(a, g)
    ref_codes, ref_scales = _quantize_ref(a * g.sigmoid())
    assert torch.equal(codes.view(torch.uint8), ref_codes.view(torch.uint8))
    assert torch.equal(scales, ref_scales.reshape(-1))


def _fp8_linear(out_features: int, in_features: int, seed: int):
    from tensorrt_llm._torch.models.modeling_kimi_linear import (
        _Fp8BlockScaleWeightReadLinear as Lin,
    )

    gen = torch.Generator(device="cuda").manual_seed(seed)
    w = (torch.randn((out_features, in_features), device="cuda", generator=gen) * 0.02).to(
        torch.bfloat16
    )
    return Lin(*Lin.quantize_weight(w), out_features)


@_rubin
@pytest.mark.parametrize("m,k", [(6, 1536), (12, 1536), (224, 12288)])
def test_gated_o_proj_unchanged(m, k):
    from tensorrt_llm._torch.modules.kimi_k3_mla.output_gate_fp8 import gate_mul_fp8_r128c4

    o_proj = _fp8_linear(HIDDEN, k, seed=m)
    assert o_proj.supports_prequantized_input
    a, g = _inputs(m, k, seed=m + 1)
    ref = o_proj(a * g.sigmoid())
    out = o_proj.forward_prequantized(*gate_mul_fp8_r128c4(a, g))
    assert torch.equal(out.view(torch.int16), ref.view(torch.int16))


class _Linear(torch.nn.Module):
    """A BF16 projection standing in for g_proj / o_proj (no prequantized entry)."""

    def __init__(self, out_features: int, in_features: int, seed: int):
        super().__init__()
        gen = torch.Generator(device="cuda").manual_seed(seed)
        self.weight = torch.nn.Parameter(
            (torch.randn((out_features, in_features), device="cuda", generator=gen) * 0.02).to(
                torch.bfloat16
            ),
            requires_grad=False,
        )

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight)


def _bare_mla(g_proj, o_proj, gate_stream=None):
    """A KimiK3MLAAttention with only what the output-gate path reads."""
    from tensorrt_llm._torch.modules.kimi_k3_mla.kimi_k3_mla_attention import KimiK3MLAAttention

    mla = KimiK3MLAAttention.__new__(KimiK3MLAAttention)
    torch.nn.Module.__init__(mla)
    mla.use_output_gate = True
    mla.g_proj = g_proj
    mla.o_proj = o_proj
    mla._gate_stream = gate_stream
    mla.g_fused_a_proj = None  # TRTLLM_K3_MLA_FUSE_GATE_A off: forward() takes the g_proj path
    mla._gate_fork_event = torch.cuda.Event()
    mla._gate_join_event = torch.cuda.Event()
    return mla


@_rubin
@pytest.mark.parametrize("m", [6, 129])
def test_module_gate_fp8_dispatch(m, monkeypatch):
    from tensorrt_llm._torch.modules.kimi_k3_mla import kimi_k3_mla_attention as k3mla
    from tensorrt_llm._torch.modules.kimi_k3_mla import output_gate_fp8

    k = 1536
    mla = _bare_mla(_fp8_linear(k, HIDDEN, seed=1), _fp8_linear(HIDDEN, k, seed=2))
    hidden = (torch.randn((m, HIDDEN), device="cuda") * 0.5).to(torch.bfloat16)
    attn_out, _ = _inputs(m, k, seed=3)
    calls = []
    fused = output_gate_fp8.gate_mul_fp8_r128c4

    def spy(*args):
        calls.append(args[0].shape)
        return fused(*args)

    monkeypatch.setattr(output_gate_fp8, "gate_mul_fp8_r128c4", spy)
    monkeypatch.setattr(k3mla, "_K3_MLA_GATE_FP8", False)
    ref = mla._apply_output_gate_and_o_proj(hidden, attn_out)
    assert not calls
    monkeypatch.setattr(k3mla, "_K3_MLA_GATE_FP8", True)
    out = mla._apply_output_gate_and_o_proj(hidden, attn_out)
    assert calls == [attn_out.shape]
    assert torch.equal(out.view(torch.int16), ref.view(torch.int16))
    # A projection without a prequantized entry keeps the unfused path.
    plain = _bare_mla(_Linear(k, HIDDEN, seed=1), _Linear(HIDDEN, k, seed=2))
    plain_ref = plain.o_proj(attn_out * plain.g_proj(hidden).sigmoid())
    calls.clear()
    assert torch.equal(plain._apply_output_gate_and_o_proj(hidden, attn_out), plain_ref)
    assert not calls


class _StreamSpy(torch.nn.Module):
    """Wraps a projection and records the stream it is called on."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.streams = []

    def forward(self, x):
        self.streams.append(torch.cuda.current_stream())
        return self.inner(x)


@pytest.mark.parametrize("capture", [False, True])
@pytest.mark.parametrize("fp8", [False, pytest.param(True, marks=_rubin)])
def test_module_gate_overlap(capture, fp8, monkeypatch):
    """g_proj on the side stream (under multi-stream, as during CUDA graph capture) gives the
    serial result; the attention stand-in writes a function of hidden_states into its output.
    With ``fp8`` the projections are the Rubin FP8 weight-read GEMMs and the gate emits the FP8
    o_proj operand (TRTLLM_K3_MLA_GATE_FP8 with TRTLLM_K3_MLA_GATE_OVERLAP)."""
    from tensorrt_llm._torch.modules.kimi_k3_mla import kimi_k3_mla_attention as k3mla
    from tensorrt_llm._torch.modules.mla import MLA
    from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream

    monkeypatch.setattr(k3mla, "_K3_MLA_GATE_FP8", fp8)

    m, k = 12, 1536
    w_attn = (torch.randn((k, HIDDEN), device="cuda") * 0.02).to(torch.bfloat16)

    def fake_forward_impl(self, position_ids, hidden_states, attn_metadata, attn_output):
        torch.matmul(hidden_states, w_attn.t(), out=attn_output[0])

    def fake_create_outputs(self, hidden_states, attn_metadata):
        return [hidden_states.new_empty((hidden_states.shape[0], k))]

    monkeypatch.setattr(MLA, "forward_impl", fake_forward_impl)
    monkeypatch.setattr(MLA, "_create_outputs", fake_create_outputs, raising=False)
    if fp8:
        g_proj, o_proj = _fp8_linear(k, HIDDEN, seed=4), _fp8_linear(HIDDEN, k, seed=5)
    else:
        g_proj, o_proj = _Linear(k, HIDDEN, seed=4), _Linear(HIDDEN, k, seed=5)
    g_spy = _StreamSpy(g_proj)
    side = torch.cuda.Stream()
    serial = _bare_mla(g_spy, o_proj)
    overlap = _bare_mla(g_spy, o_proj, gate_stream=side)
    hidden = (torch.randn((m, HIDDEN), device="cuda") * 0.5).to(torch.bfloat16)
    ref = serial.forward(hidden, None)
    assert g_spy.streams[-1] == torch.cuda.current_stream()
    g_spy.streams.clear()
    if not capture:
        with with_multi_stream(True):
            out = overlap.forward(hidden, None)
    else:
        with with_multi_stream(True):
            overlap.forward(hidden, None)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = overlap.forward(hidden, None)
        new_hidden = (torch.randn((m, HIDDEN), device="cuda") * 0.5).to(torch.bfloat16)
        hidden.copy_(new_hidden)
        graph.replay()
        ref = serial.forward(new_hidden, None)
    torch.cuda.synchronize()
    assert torch.equal(out.view(torch.int16), ref.view(torch.int16))
    # g_proj ran on the side stream in every overlapped forward (eager and captured).
    assert g_spy.streams and all(s == side for s in g_spy.streams[:1])
