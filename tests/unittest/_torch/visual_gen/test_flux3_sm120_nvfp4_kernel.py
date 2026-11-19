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
is eager FP32 math on the same decoded NVFP4 operands. The BF16 qkv mode is T1;
the requantized packed mode is T2.
"""

import pytest
import torch
import torch.nn.functional as F
from flux3_test_utils import assert_cuda_graph_replays, assert_parity

from tensorrt_llm._torch.utils import unswizzle_sf
from tensorrt_llm.math_utils import pad_up

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
    [("qkv", (256, 768, 512)), ("qkv", (200, 768, 512)), ("packed", (256, 512, 3072))],
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

    if mode == "qkv":
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
