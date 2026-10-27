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

"""L1 parity for SM120/SM121 Flux.3 NVFP4 GEMM fused epilogues.

The boundary is the GEMM output (or packed output and scales). The reference
is eager FP32 math on the same decoded NVFP4 operands. BF16 modes are T1;
the requantized packed mode is T2.
"""

import pytest
import torch
import torch.nn.functional as F


def _scale_indices(rows, scale_columns, device):
    row = torch.arange(rows, device=device)[:, None]
    column = torch.arange(scale_columns, device=device)[None, :]
    return (
        column % 4
        + column // 4 * 512
        + row % 32 * 16
        + row % 128 // 32 * 4
        + row // 128 * 128 * scale_columns
    )


def _decode(data, scales, global_scale):
    levels = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=data.device)
    codes = torch.stack((data & 15, data >> 4), dim=-1).flatten(1).long()
    values = levels[codes & 7] * torch.where(codes & 8 != 0, -1, 1)
    rows, columns = values.shape
    scale = scales.flatten()[_scale_indices(rows, columns // 16, data.device)]
    scale = scale.contiguous().view(torch.float8_e4m3fn).float()
    return values * scale.repeat_interleave(16, dim=1) * global_scale


def _metrics(actual, expected):
    actual = actual.float().flatten()
    expected = expected.float().flatten()
    error = (actual - expected).abs()
    norm = expected.norm()
    return {
        "relative_l2": (error.norm() / norm).item() if norm.item() else error.norm().item(),
        "cosine": F.cosine_similarity(actual, expected, dim=0).item() if norm.item() else 1.0,
        "max_abs": error.max().item(),
        "p99_abs": torch.quantile(error, 0.99).item(),
    }


def _packed_reference(values, inverse_scale):
    rows, columns = values.shape
    shaped = values.reshape(rows, columns // 16, 16)
    scales = (shaped.abs().amax(-1) * inverse_scale / 6.0).to(torch.float8_e4m3fn)
    scaled = shaped * (inverse_scale / scales.float().clamp_min(1e-12)).unsqueeze(-1)
    levels = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=values.device)
    nearest = (scaled.abs()[..., None] - levels).abs().argmin(-1).to(torch.uint8)
    signed = nearest | ((scaled < 0).to(torch.uint8) << 3)
    packed = (signed[..., 0::2] | (signed[..., 1::2] << 4)).reshape(rows, columns // 2)
    layout = _scale_indices(rows, columns // 16, values.device)
    packed_scales = torch.empty(rows * columns // 16, device=values.device, dtype=torch.uint8)
    packed_scales.scatter_(0, layout.flatten(), scales.view(torch.uint8).flatten())
    return packed, packed_scales


@pytest.mark.parametrize(
    "mode,shape",
    [
        ("linear", (256, 256, 256)),
        ("swiglu", (256, 256, 256)),
        ("qkv", (256, 768, 512)),
        ("packed", (256, 512, 3072)),
    ],
)
def test_flux3_sm120_nvfp4_epilogue(mode, shape):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (12, 0),
        (12, 1),
    ):
        pytest.skip("requires SM120 or SM121 GPU")

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
    actual = flux3_sm120_nvfp4.run(
        mode,
        activation_packed,
        weight_packed,
        activation_scale,
        weight_scale,
        0.125,
        1.0,
        output_inverse_scale=inverse_scale,
        q_norm=q_norm,
        k_norm=k_norm,
        cos_sin=cos_sin,
        positions=positions,
        qkv_scale=qkv_scale,
    )
    assert (
        torch.cuda.current_device(),
        torch.cuda.get_device_capability(),
        mode,
        rows,
        columns,
        reduction,
    ) in flux3_sm120_nvfp4._COMPILED
    if mode == "qkv":
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            replayed = flux3_sm120_nvfp4.run(
                mode,
                activation_packed,
                weight_packed,
                activation_scale,
                weight_scale,
                0.125,
                1.0,
                output_inverse_scale=inverse_scale,
                q_norm=q_norm,
                k_norm=k_norm,
                cos_sin=cos_sin,
                positions=positions,
                qkv_scale=qkv_scale,
            )
        graph.replay()
        torch.testing.assert_close(replayed, actual, rtol=1e-3, atol=1e-3)

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
        if mode == "swiglu":
            expected = expected.bfloat16()
        else:
            actual, actual_scales = actual
            expected_packed, expected_scales = _packed_reference(expected, inverse_scale)
            actual = _decode(actual, actual_scales, 0.5)
            expected = _decode(expected_packed, expected_scales, 0.5)
            scale_stats = _metrics(
                actual_scales.view(torch.float8_e4m3fn).float(),
                expected_scales.view(torch.float8_e4m3fn).float(),
            )
            assert scale_stats["relative_l2"] < 1e-2, (mode, shape, scale_stats)

    stats = _metrics(actual, expected)
    print(f"{mode} {shape}: {stats}")
    assert stats["relative_l2"] < 1e-2 and stats["cosine"] > 0.9999, (mode, shape, stats)


@pytest.mark.parametrize(
    "capability, sm120",
    [((12, 0), True), ((12, 1), True), ((10, 0), False), ((9, 0), False)],
)
def test_flux3_sm120_dispatch_is_default(
    monkeypatch: pytest.MonkeyPatch, capability: tuple[int, int], sm120: bool
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA GPU")

    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_primsts, primsts

    weight = torch.empty((128, 64), device="cuda", dtype=torch.uint8)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device=None: capability)
    assert flux3_primsts.is_sm120(weight.device) is sm120
    assert primsts._supported_nvfp4_device(weight) is (capability != (9, 0))
    assert not flux3_primsts.is_sm120(weight.cpu().device)
