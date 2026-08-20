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


def _require_sm100() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required")
    major, minor = torch.cuda.get_device_capability()
    if major != 10:
        pytest.skip(f"NVFP4 Conv3d requires SM100-family, got sm_{major}{minor}")


def test_nvfp4_conv3d_bias_residual_epilogue_matches_reference() -> None:
    """Exercise the product tactic and BF16 epilogue against the kernel reference."""
    _require_sm100()
    cutlass = pytest.importorskip("cutlass")
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.conv.dense_blockscaled_implicit_gemm_fprop import (
        run,
    )

    runtime_us = run(
        ncdhw=(1, 128, 3, 12, 16),
        ktrs=(256, 3, 3, 3),
        stride_dhw=(1, 1, 1),
        upper_pad_dhw=(0, 1, 1),
        lower_pad_dhw=(0, 1, 1),
        dil_dhw=(1, 1, 1),
        ab_dtype=cutlass.Float4E2M1FN,
        d_dtype=cutlass.BFloat16,
        acc_dtype=cutlass.Float32,
        sf_dtype=cutlass.Float8E4M3FN,
        sf_vec_size=16,
        mma_tiler_mn=(256, 256),
        preferred_cluster_shape_mn=(2, 1),
        fallback_cluster_shape_mn=(2, 1),
        use_2cta_instrs=True,
        use_bias=True,
        beta=1.0,
        tolerance=1e-2,
        warmup_iterations=1,
        iterations=2,
        skip_ref_check=False,
    )

    assert runtime_us is not None and runtime_us > 0


@pytest.mark.parametrize("channels", [128, 192, 512])
def test_fused_silu_nvfp4_quant_matches_trtllm_quantize(channels: int) -> None:
    """Compare fused bytes, including activation blocks beyond calibration."""
    _require_sm100()
    pytest.importorskip("triton")
    from tensorrt_llm._torch.visual_gen.models.wan.fp4_fused_quant import silu_nvfp4_quant

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
    _require_sm100()
    pytest.importorskip("triton")
    from tensorrt_llm._torch.visual_gen.models.wan.fp4_fused_quant import rmsnorm_silu_nvfp4_quant

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


def test_nvfp4_wan_conv_product_path_bias_residual_and_spatial_padding() -> None:
    """Run the VAE wrapper through its real CuTe ABI on a partial output tile."""
    _require_sm100()
    pytest.importorskip("cutlass")
    from tensorrt_llm._torch.visual_gen.models.wan.wan_vae import (
        NVFP4WanCausalConv3d,
        WanCausalConv3d,
    )

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
    expected = residual + base.bias.view(1, -1, 1, 1, 1)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    # Exercise product weight prequantization, KTRSC layout, SFB swizzle, and
    # alpha scaling with a nonzero convolution while reusing the same tactic.
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
    # Build the asymmetric-padding BF16 reference explicitly so this kernel
    # test does not depend on the parallel halo optimization's Conv3d API.
    padded_activation = F.pad(activation, (0, 0, 0, 0, 2, 0))
    expected = F.conv3d(
        padded_activation,
        base.weight,
        base.bias,
        stride=base.stride,
        padding=(0, 1, 0),
        dilation=base.dilation,
    ).float()
    relative_l2_error = torch.linalg.vector_norm(actual - expected) / torch.linalg.vector_norm(
        expected
    )
    cosine_similarity = F.cosine_similarity(actual.flatten(), expected.flatten(), dim=0)

    assert relative_l2_error.item() < 0.25
    assert cosine_similarity.item() > 0.96


def test_nvfp4_wan_conv_reuses_cubin_across_runtime_shapes() -> None:
    """The provider's runtime-shape kernel serves compatible C and H/W values."""
    _require_sm100()
    pytest.importorskip("cutlass")
    from tensorrt_llm._torch.visual_gen.models.wan.wan_vae import (
        NVFP4WanCausalConv3d,
        WanCausalConv3d,
        _fp4_compile_cache,
    )

    _fp4_compile_cache.clear()
    for input_channels, height, width in ((256, 4, 6), (512, 5, 7)):
        base = WanCausalConv3d(input_channels, 256, 3, padding=1).cuda().to(torch.bfloat16).eval()
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
        expected = base.bias.view(1, -1, 1, 1, 1).expand_as(actual)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    assert len(_fp4_compile_cache) == 1
