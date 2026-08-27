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

from tensorrt_llm._torch.visual_gen.models.wan.fp4_fused_quant import (
    rmsnorm_silu_nvfp4_quant,
    silu_nvfp4_quant,
)
from tensorrt_llm._torch.visual_gen.models.wan.wan_vae import (
    NVFP4WanCausalConv3d,
    WanCausalConv3d,
    _fp4_compile_cache,
    _supports_nvfp4_device,
)


def _require_supported_gpu() -> None:
    if not _supports_nvfp4_device(torch.device("cuda")):
        pytest.skip("NVFP4 Wan VAE requires an SM100, SM103, or SM120 GPU")


@pytest.mark.parametrize("channels", [128, 192, 512])
def test_fused_silu_nvfp4_quant_matches_trtllm_quantize(channels: int) -> None:
    """Compare fused bytes, including activation blocks beyond calibration."""
    _require_supported_gpu()
    torch.manual_seed(0)
    activation = torch.randn((17, channels), device="cuda", dtype=torch.bfloat16)
    activation[0, :16] = 20.0
    activation[1, 0] = -0.0
    global_scale = torch.tensor([448.0 * 6.0], device="cuda", dtype=torch.float32)

    expected = torch.ops.trtllm.fp4_quantize(
        F.silu(activation),
        global_scale,
        16,
        False,
        False,
    )
    actual = silu_nvfp4_quant(activation, global_scale)

    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected[1].reshape_as(actual[1]), rtol=0, atol=0)


def test_fused_rmsnorm_silu_nvfp4_quant_matches_trtllm_quantize() -> None:
    _require_supported_gpu()
    torch.manual_seed(0)
    rows, channels = 17, 192
    activation = torch.randn((rows, channels), device="cuda", dtype=torch.bfloat16)
    gamma = torch.randn((channels,), device="cuda", dtype=torch.bfloat16)
    global_scale = torch.tensor([448.0 * 6.0 / 4.0], device="cuda", dtype=torch.float32)
    scale = channels**0.5
    normalized = F.normalize(activation.float(), dim=1).to(torch.bfloat16)
    normalized = normalized * scale
    normalized = normalized * gamma
    expected = torch.ops.trtllm.fp4_quantize(
        F.silu(normalized),
        global_scale,
        16,
        False,
        False,
    )
    actual = rmsnorm_silu_nvfp4_quant(
        activation,
        global_scale,
        gamma,
        scale,
    )

    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected[1].reshape_as(actual[1]), rtol=0, atol=0)


def test_nvfp4_wan_conv_bias_residual_and_spatial_padding() -> None:
    """Run the VAE wrapper through its CuTe ABI on a partial output tile."""
    _require_supported_gpu()
    torch.manual_seed(0)
    base = WanCausalConv3d(192, 96, 3, padding=1).cuda().to(torch.bfloat16).eval()
    with torch.no_grad():
        base.weight.zero_()
        base.bias.normal_()
    conv = NVFP4WanCausalConv3d(base).cuda().to(torch.bfloat16).eval()
    activation = torch.randn((1, 192, 1, 8, 10), device="cuda", dtype=torch.bfloat16)
    residual = torch.randn((1, 96, 1, 8, 8), device="cuda", dtype=torch.bfloat16)

    actual = conv(
        activation,
        spatial_padding=(1, 0),
        residual=residual,
    )
    expected = residual + conv.bias.view(1, -1, 1, 1, 1)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    # Exercise weight prequantization, KTRSC layout, SFB swizzle, and
    # alpha scaling with a nonzero convolution while reusing the same tactic.
    base = WanCausalConv3d(192, 96, 3, padding=1).cuda().to(torch.bfloat16).eval()
    with torch.no_grad():
        base.weight.normal_(std=0.02)
        base.bias.zero_()
    conv = NVFP4WanCausalConv3d(base).cuda().to(torch.bfloat16).eval()
    residual.zero_()
    actual = conv(
        activation,
        spatial_padding=(1, 0),
        residual=residual,
    ).float()
    # Build the asymmetric-padding BF16 reference explicitly so this wrapper
    # test does not depend on the parallel halo optimization's Conv3d API.
    padded_activation = F.pad(activation, (0, 0, 0, 0, 2, 0))
    expected = F.conv3d(
        padded_activation,
        conv.weight,
        conv.bias,
        stride=conv.stride,
        padding=(0, 1, 0),
        dilation=conv.dilation,
    ).float()
    relative_l2_error = torch.linalg.vector_norm(actual - expected) / torch.linalg.vector_norm(
        expected
    )
    cosine_similarity = F.cosine_similarity(actual.flatten(), expected.flatten(), dim=0)

    assert relative_l2_error.item() < 0.25
    assert cosine_similarity.item() > 0.96


def test_nvfp4_wan_conv_reuses_cubin_across_runtime_shapes() -> None:
    """One compiled kernel serves compatible channel and spatial shapes."""
    _require_supported_gpu()
    saved_cache = dict(_fp4_compile_cache)
    try:
        _fp4_compile_cache.clear()
        initial_size = len(_fp4_compile_cache)
        for input_channels, height, width in ((256, 4, 6), (512, 5, 7)):
            base = (
                WanCausalConv3d(input_channels, 256, 3, padding=1).cuda().to(torch.bfloat16).eval()
            )
            with torch.no_grad():
                base.weight.zero_()
                base.bias.normal_()
            conv = NVFP4WanCausalConv3d(base).cuda().to(torch.bfloat16).eval()
            activation = torch.randn(
                (1, input_channels, 1, height, width),
                device="cuda",
                dtype=torch.bfloat16,
            )

            actual = conv(activation)
            expected = conv.bias.view(1, -1, 1, 1, 1).expand_as(actual)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

        assert len(_fp4_compile_cache) - initial_size == 1
    finally:
        _fp4_compile_cache.clear()
        _fp4_compile_cache.update(saved_cache)
