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
"""Fused RMSNorm + FP8 1x128 (CuTe UE8M0 R128c4) quantization.

``rmsnorm_fp8_quantize_1x128_cutedsl_ue8m0`` replaces the Kimi K3 attention-DP
routed latent tail ``FlashInfer RMSNorm -> fp8_quantize_1x128_cutedsl_ue8m0``.
The fused kernel rounds the normalized row to BF16 before quantizing, exactly
like that sequence, so the only legitimate difference is the last bit of rstd
(FP32 order of the row's sum of squares; FlashInfer's fast-math division). The
tests pin:
  * the scale layout (byte-for-byte equal to the plain quantizer on the same
    BF16 rows, zero scales for padded rows),
  * near-bitwise agreement with the unfused sequence,
  * row-strided inputs, CUDA graph capture, and the Rubin GEMM consumer.
"""

import pytest
import torch
from _torch.helpers import calc_diff

import tensorrt_llm  # noqa: F401  (registers torch.ops.trtllm)

SM100_FAMILY = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="needs an SM100-family GPU (Blackwell / Rubin)",
)

EPS = 1e-6
# Kimi K3 latent (3584), MLA q_lora (1536) / kv_lora (512), hidden (7168).
KS = [512, 1536, 3584, 7168]
# Decode graph buckets at DEP16 (48/64/80/96 tokens), small M, and >128 rows.
MS = [1, 7, 48, 80, 96, 128, 200]


def _inputs(m, k, seed=0):
    torch.manual_seed(seed)
    x = torch.randn((m, k), device="cuda", dtype=torch.float32)
    # Heavy-tailed rows/blocks so the per-block UE8M0 scales vary.
    x[::5] *= 37.0
    x[:, 128:256] *= 0.01
    x = x.to(torch.bfloat16)
    weight = (1.0 + 0.1 * torch.randn((k,), device="cuda")).to(torch.bfloat16)
    # The fused op reads the weight before its PDL wait (fine for model weights);
    # make these freshly written test tensors visible before the launch.
    torch.cuda.synchronize()
    return x, weight


def _rmsnorm_bf16(x, weight, eps=EPS):
    """FlashInfer-style single-rounding RMSNorm: bf16((x * rstd) * w) in FP32."""
    xf = x.float()
    rstd = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return ((xf * rstd) * weight.float()).to(torch.bfloat16)


def _dequant(values, flat_scale, m, k):
    """Dequantize the CuTe R128c4 layout (one UE8M0 byte per 32 values)."""
    padded_m = (m + 127) // 128 * 128
    num_k_tiles = (k // 32 + 3) // 4
    rows = torch.arange(m, device=values.device)
    cols = torch.arange(k // 32, device=values.device)
    kt, kin = cols // 4, cols % 4
    offset = (
        (rows[:, None] // 128) * num_k_tiles * 512
        + kt[None, :] * 512
        + (rows[:, None] % 32) * 16
        + ((rows[:, None] % 128) // 32) * 4
        + kin[None, :]
    )
    assert flat_scale.numel() == padded_m * num_k_tiles * 4
    exponent = flat_scale.long()[offset]
    scale = torch.pow(2.0, (exponent - 127).float())
    return values.float() * scale.repeat_interleave(32, dim=1)


@SM100_FAMILY
@pytest.mark.parametrize("k", KS)
@pytest.mark.parametrize("m", MS)
def test_matches_unfused_sequence(m, k):
    x, weight = _inputs(m, k)
    values, scale = torch.ops.trtllm.rmsnorm_fp8_quantize_1x128_cutedsl_ue8m0(x, weight, EPS)

    ref_values, ref_scale = torch.ops.trtllm.fp8_quantize_1x128_cutedsl_ue8m0(
        _rmsnorm_bf16(x, weight)
    )

    assert values.shape == (m, k) and values.dtype == torch.float8_e4m3fn
    assert scale.shape == ref_scale.shape and scale.dtype == torch.uint8

    # rstd may differ in its last bit, which can move a block's amax across a
    # power of two in rare cases; anything more than that is a bug.
    scale_mismatch = (scale != ref_scale).float().mean().item()
    assert scale_mismatch < 2e-3, f"scale mismatch rate {scale_mismatch}"
    code_mismatch = (values.view(torch.uint8) != ref_values.view(torch.uint8)).float().mean().item()
    assert code_mismatch < 1e-2, f"fp8 code mismatch rate {code_mismatch}"

    deq = _dequant(values, scale, m, k)
    ref_deq = _dequant(ref_values, ref_scale, m, k)
    assert torch.isfinite(deq).all()
    assert calc_diff(deq, ref_deq) < 1e-4
    # Both against the exact BF16 normalized rows: plain FP8 1x128 accuracy.
    assert calc_diff(deq, _rmsnorm_bf16(x, weight).float()) < 5e-3


@SM100_FAMILY
def test_padded_rows_get_zero_scales():
    m, k = 80, 3584
    x, weight = _inputs(m, k, seed=3)
    _, scale = torch.ops.trtllm.rmsnorm_fp8_quantize_1x128_cutedsl_ue8m0(x, weight, EPS)
    _, ref_scale = torch.ops.trtllm.fp8_quantize_1x128_cutedsl_ue8m0(_rmsnorm_bf16(x, weight))
    padded_m = 128
    num_k_tiles = k // 128
    rows = torch.arange(m, padded_m, device="cuda")
    kt = torch.arange(num_k_tiles, device="cuda")
    base = kt[None, :] * 512 + (rows[:, None] % 32) * 16 + ((rows[:, None] % 128) // 32) * 4
    padded = torch.stack([scale.long()[base + i] for i in range(4)])
    assert (padded == 0).all()
    assert torch.equal(scale.long()[base], ref_scale.long()[base])


@SM100_FAMILY
def test_row_strided_input():
    """A column slice (e.g. MLA q_a out of the fused_a output) is accepted."""
    m, k, ld = 48, 1536, 2112
    torch.manual_seed(1)
    wide = torch.randn((m, ld), device="cuda").to(torch.bfloat16)
    x = wide[:, :k]
    assert x.stride(0) == ld
    weight = (1.0 + 0.1 * torch.randn((k,), device="cuda")).to(torch.bfloat16)
    torch.cuda.synchronize()  # the op reads the weight before its PDL wait
    values, scale = torch.ops.trtllm.rmsnorm_fp8_quantize_1x128_cutedsl_ue8m0(x, weight, EPS)
    contiguous_values, contiguous_scale = torch.ops.trtllm.rmsnorm_fp8_quantize_1x128_cutedsl_ue8m0(
        x.contiguous(), weight, EPS
    )
    assert torch.equal(values.view(torch.uint8), contiguous_values.view(torch.uint8))
    assert torch.equal(scale, contiguous_scale)


@SM100_FAMILY
def test_cuda_graph_replay_matches_eager():
    m, k = 96, 3584
    x, weight = _inputs(m, k, seed=5)
    eager_values, eager_scale = torch.ops.trtllm.rmsnorm_fp8_quantize_1x128_cutedsl_ue8m0(
        x, weight, EPS
    )
    static_x = x.clone()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        # Warm up outside capture.
        torch.ops.trtllm.rmsnorm_fp8_quantize_1x128_cutedsl_ue8m0(static_x, weight, EPS)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_values, graph_scale = torch.ops.trtllm.rmsnorm_fp8_quantize_1x128_cutedsl_ue8m0(
            static_x, weight, EPS
        )
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(graph_values.view(torch.uint8), eager_values.view(torch.uint8))
    assert torch.equal(graph_scale, eager_scale)


@SM100_FAMILY
@pytest.mark.parametrize("k", [100, 3600])
def test_rejects_k_not_multiple_of_128(k):
    x = torch.zeros((4, k), device="cuda", dtype=torch.bfloat16)
    weight = torch.ones((k,), device="cuda", dtype=torch.bfloat16)
    with pytest.raises(RuntimeError):
        torch.ops.trtllm.rmsnorm_fp8_quantize_1x128_cutedsl_ue8m0(x, weight, EPS)


@SM100_FAMILY
def test_empty_batch():
    weight = torch.ones((3584,), device="cuda", dtype=torch.bfloat16)
    x = torch.zeros((0, 3584), device="cuda", dtype=torch.bfloat16)
    values, scale = torch.ops.trtllm.rmsnorm_fp8_quantize_1x128_cutedsl_ue8m0(x, weight, EPS)
    assert values.shape == (0, 3584)
    assert scale.numel() == 0


def _rubin_mxfp8():
    from tensorrt_llm._torch.models.modeling_kimi_linear import _rubin_mxfp8_is_available

    return _rubin_mxfp8_is_available()


@pytest.mark.skipif(
    not torch.cuda.is_available() or not _rubin_mxfp8(),
    reason="needs SM107 with the CuTe DSL Rubin helpers",
)
@pytest.mark.parametrize("m", [16, 80])
def test_latent_up_projection_matches_unfused(m):
    """Kimi K3 latent tail: norm -> FP8 weight-read up-projection (3584 -> 7168)."""
    from tensorrt_llm._torch.models.modeling_kimi_linear import (
        _Fp8BlockScaleWeightReadLinear as K3Fp8Linear,
    )
    from tensorrt_llm._torch.modules.rms_norm import RMSNorm

    k, n = 3584, 7168
    torch.manual_seed(7)
    w_up = (torch.randn((n, k), device="cuda") / k**0.5).to(torch.bfloat16)
    weight, weight_scale = K3Fp8Linear.quantize_weight(w_up)
    up = K3Fp8Linear(weight, weight_scale, n)
    norm = RMSNorm(hidden_size=k, eps=EPS, dtype=torch.bfloat16).cuda()
    with torch.no_grad():
        norm.weight.copy_((1.0 + 0.1 * torch.randn((k,), device="cuda")).to(torch.bfloat16))
    x, _ = _inputs(m, k, seed=11)

    unfused = up(norm(x))
    activation, activation_scale = torch.ops.trtllm.rmsnorm_fp8_quantize_1x128_cutedsl_ue8m0(
        x, norm.weight, norm.variance_epsilon
    )
    fused = up.forward_prequantized(activation, activation_scale)

    assert fused.shape == unfused.shape == (m, n)
    assert calc_diff(fused, unfused) < 1e-4


@pytest.mark.skipif(
    not torch.cuda.is_available() or not _rubin_mxfp8(),
    reason="needs SM107 with the CuTe DSL Rubin helpers",
)
def test_kimi_k3_latent_gate(monkeypatch):
    """The model only takes the fused path when opted in and the consumer fits."""
    from types import SimpleNamespace

    import tensorrt_llm._torch.models.modeling_kimi_linear as kimi
    from tensorrt_llm._torch.modules.rms_norm import RMSNorm

    k, n = 3584, 7168
    w_up = (torch.randn((n, k), device="cuda") / k**0.5).to(torch.bfloat16)
    up = kimi._Fp8BlockScaleWeightReadLinear(
        *kimi._Fp8BlockScaleWeightReadLinear.quantize_weight(w_up), n
    )
    norm = RMSNorm(hidden_size=k, eps=EPS, dtype=torch.bfloat16).cuda()
    owner = SimpleNamespace(routed_expert_up_proj=up, routed_expert_norm=norm)
    y = torch.randn((80, k), device="cuda").to(torch.bfloat16)

    monkeypatch.setattr(kimi, "_K3_FUSED_LATENT_NORM_QUANT", False)
    assert not kimi.KimiK3MoERuntime._can_fuse_latent_norm_quant(owner, y)
    monkeypatch.setattr(kimi, "_K3_FUSED_LATENT_NORM_QUANT", True)
    assert kimi.KimiK3MoERuntime._can_fuse_latent_norm_quant(owner, y)
    assert not kimi.KimiK3MoERuntime._can_fuse_latent_norm_quant(owner, y.float())
    bf16_up = SimpleNamespace(
        routed_expert_up_proj=torch.nn.Linear(k, n, bias=False), routed_expert_norm=norm
    )
    assert not kimi.KimiK3MoERuntime._can_fuse_latent_norm_quant(bf16_up, y)
    # A (1 + w) Gemma-style norm is a different formula: stay on the unfused path.
    gemma = SimpleNamespace(
        routed_expert_up_proj=up,
        routed_expert_norm=RMSNorm(
            hidden_size=k, eps=EPS, dtype=torch.bfloat16, use_gemma=True
        ).cuda(),
    )
    assert not kimi.KimiK3MoERuntime._can_fuse_latent_norm_quant(gemma, y)
