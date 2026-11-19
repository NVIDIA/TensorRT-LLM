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

"""L1 parity for the SM120/SM121 FLUX 3 NVFP4 GEMM fused epilogues.

The boundary is the GEMM output (or packed output and scales). The reference
is eager FP32 math on the same decoded NVFP4 operands. The BF16 linear and qkv
modes are T1; the requantized packed mode is T2. CPU cases check which GEMM
``Sm120NVFP4LinearMethod`` dispatches to.
"""

import pytest
import torch
import torch.nn.functional as F
from flux3_test_utils import assert_cuda_graph_replays, assert_parity

from tensorrt_llm._torch.modules.linear import Linear, NVFP4LinearMethod
from tensorrt_llm._torch.utils import Fp4QuantizedTensor, unswizzle_sf
from tensorrt_llm._torch.visual_gen.models.flux3 import sm120_nvfp4
from tensorrt_llm._torch.visual_gen.models.flux3.fused_gemm_common import nvfp4_scale_bytes
from tensorrt_llm.math_utils import pad_up
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

requires_sm120 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((12, 0), (12, 1)),
    reason="requires SM120 or SM121 GPU",
)


def _decode(data, scales, global_scale):
    levels = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=data.device)
    codes = torch.stack((data & 15, data >> 4), dim=-1).flatten(1).long()
    values = levels[codes & 7] * torch.where(codes & 8 != 0, -1, 1)
    rows, columns = values.shape
    scale = unswizzle_sf(scales, pad_up(rows, 128), columns)[:rows]
    scale = scale.view(torch.float8_e4m3fn).float()
    return values * scale.repeat_interleave(16, dim=1) * global_scale


def _packed_reference(values, inverse_scale):
    rows, columns = values.shape
    shaped = values.reshape(rows, columns // 16, 16)
    scales = (shaped.abs().amax(-1) * inverse_scale / 6.0).to(torch.float8_e4m3fn)
    scaled = shaped * (inverse_scale / scales.float().clamp_min(1e-12)).unsqueeze(-1)
    levels = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=values.device)
    nearest = (scaled.abs()[..., None] - levels).abs().argmin(-1).to(torch.uint8)
    signed = nearest | ((scaled < 0).to(torch.uint8) << 3)
    packed = (signed[..., 0::2] | (signed[..., 1::2] << 4)).reshape(rows, columns // 2)
    return packed, torch.ops.trtllm.block_scale_interleave(scales.view(torch.uint8))


@requires_sm120
@pytest.mark.parametrize(
    "mode,shape",
    [
        ("linear", (256, 256, 256)),
        ("linear", (200, 256, 256)),
        ("qkv", (256, 768, 512)),
        ("qkv", (200, 768, 512)),
        ("packed", (256, 512, 3072)),
    ],
)
def test_flux3_sm120_nvfp4_epilogue(mode, shape):
    from flashinfer import SfLayout, nvfp4_quantize

    from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.sm120 import flux3_sm120_nvfp4

    rows, columns, reduction = shape
    torch.manual_seed(2026)
    activation = torch.randn((rows, reduction), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((columns, reduction), device="cuda", dtype=torch.bfloat16)
    unit_scale = torch.ones(1, device="cuda", dtype=torch.float32)
    activation_packed, activation_scale = nvfp4_quantize(
        activation, unit_scale, sfLayout=SfLayout.layout_128x4, do_shuffle=False, backend="cute-dsl"
    )
    weight_packed, weight_scale = nvfp4_quantize(
        weight, unit_scale, sfLayout=SfLayout.layout_128x4, do_shuffle=False, backend="cute-dsl"
    )
    raw = (
        _decode(activation_packed, activation_scale, 1.0)
        @ _decode(weight_packed, weight_scale, 1.0).T
    )
    q_norm = torch.rand(128, device="cuda", dtype=torch.bfloat16)
    k_norm = torch.rand(128, device="cuda", dtype=torch.bfloat16)
    angles = torch.randn((rows, 64), device="cuda")
    cos_sin = torch.cat((angles.cos(), angles.sin()), dim=-1)
    positions = torch.arange(rows - 1, -1, -1, device="cuda", dtype=torch.int64)
    qkv_scale = torch.tensor([0.125, 0.25, 0.375], device="cuda")
    inverse_scale = torch.tensor([2.0], device="cuda")
    alpha = torch.tensor([0.125], device="cuda")

    def run():
        return flux3_sm120_nvfp4.run(
            mode,
            activation_packed,
            weight_packed,
            activation_scale,
            weight_scale,
            alpha,
            output_inverse_scale=inverse_scale,
            q_norm=q_norm,
            k_norm=k_norm,
            cos_sin=cos_sin,
            positions=positions,
            qkv_scale=qkv_scale,
        )

    actual = assert_cuda_graph_replays(run)

    if mode == "linear":
        expected = (raw * 0.125).bfloat16()
    elif mode == "qkv":
        heads = columns // 384
        expected = (
            (raw.reshape(rows, 3, heads, 128) * qkv_scale[None, :, None, None]).bfloat16().float()
        )
        cosine = cos_sin[positions, :64][:, None, :]
        sine = cos_sin[positions, 64:][:, None, :]
        for part, norm in ((0, q_norm), (1, k_norm)):
            values = F.rms_norm(expected[:, part], (128,), norm.float(), 1e-6)
            even, odd = values[..., 0::2], values[..., 1::2]
            expected[:, part] = torch.stack(
                (even * cosine - odd * sine, odd * cosine + even * sine), dim=-1
            ).flatten(-2)
        expected = expected.reshape(rows, columns).bfloat16()
    else:
        expected = (raw[:, ::2] * 0.125) * F.silu(raw[:, 1::2] * 0.125)
        actual, actual_scales = actual
        expected_packed, expected_scales = _packed_reference(expected, inverse_scale)
        actual = _decode(actual, actual_scales, 0.5)
        expected = _decode(expected_packed, expected_scales, 0.5)
        assert_parity(
            actual_scales.view(torch.float8_e4m3fn),
            expected_scales.view(torch.float8_e4m3fn),
            relative_l2=1e-2,
            cosine=0.9999,
        )

    assert_parity(actual, expected, relative_l2=1e-2, cosine=0.9999)


def _sm120_linear(out_features=128, in_features=256, bias=False):
    layer = Linear(
        in_features,
        out_features,
        bias=bias,
        dtype=torch.bfloat16,
        quant_config=QuantConfig(quant_algo=QuantAlgo.NVFP4),
    )
    layer.quant_method = sm120_nvfp4.Sm120NVFP4LinearMethod()
    return layer


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "case", ["off_sm120", "bias", "autograd", "fp16_input", "unaligned_n", "unswizzled_fp4"]
)
def test_sm120_linear_method_falls_back_to_trtllm(monkeypatch, case):
    """What the SM120 kernel cannot take runs NVFP4LinearMethod.apply through Linear.forward."""
    calls = []
    monkeypatch.setattr(
        NVFP4LinearMethod, "apply", lambda self, module, x, bias: calls.append(x) or "trtllm"
    )
    if case != "off_sm120":
        # Only the case under test keeps the layer off the kernel.
        monkeypatch.setattr(sm120_nvfp4, "sm120_device", lambda device: True)
    layer = _sm120_linear(out_features=96 if case == "unaligned_n" else 128, bias=case == "bias")
    x = torch.zeros(2, 4, 256, dtype=torch.float16 if case == "fp16_input" else torch.bfloat16)
    if case == "unswizzled_fp4":
        scales = torch.zeros(nvfp4_scale_bytes(4, 256), dtype=torch.uint8)
        x = Fp4QuantizedTensor(torch.zeros(4, 128, dtype=torch.uint8), scales, False)
    with torch.set_grad_enabled(case == "autograd"):
        assert layer(x) == "trtllm"
    assert len(calls) == 1 and calls[0] is x


@pytest.mark.cpu_only
def test_sm120_linear_method_feeds_the_kernel_the_layer_operands(monkeypatch):
    """A packed SwiGLU output reaches the SM120 op with the layer's weights and device alpha."""
    calls = []

    def kernel(data, scales, weight, weight_scale, alpha):
        calls.append((data, scales, weight, weight_scale, alpha))
        return torch.zeros(data.shape[0], weight.shape[0], dtype=torch.bfloat16)

    monkeypatch.setattr(sm120_nvfp4, "sm120_device", lambda device: True)
    monkeypatch.setattr(torch.ops.trtllm, "flux3_sm120_nvfp4_linear", kernel)
    layer = _sm120_linear()
    data = torch.zeros(4, 128, dtype=torch.uint8)
    scales = torch.zeros(nvfp4_scale_bytes(4, 256), dtype=torch.uint8)
    with torch.inference_mode():
        output = layer(Fp4QuantizedTensor(data, scales, is_sf_swizzled=True))
    assert output.shape == (4, 128) and output.dtype == torch.bfloat16
    ((got_data, got_scales, weight, weight_scale, alpha),) = calls
    assert got_data is data and torch.equal(got_scales, scales)
    assert weight is layer.weight and weight_scale is layer.weight_scale and alpha is layer.alpha
