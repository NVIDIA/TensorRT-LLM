# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import os
import subprocess
import sys

import pytest
import torch
from _torch.helpers import (calc_diff, per_block_cast_to_fp8,
                            per_block_cast_to_fp8_e8m0)
from utils.util import getSMVersion, isSM100Family

import tensorrt_llm.quantization.utils.fp8_utils as fp8_utils
from tensorrt_llm._torch.autotuner import AutoTuner, autotune
from tensorrt_llm._torch.cute_dsl_utils import (IS_CUTLASS_DSL_AVAILABLE,
                                                IS_CUTLASS_DSL_RUBIN_AVAILABLE)


@pytest.mark.skipif(
    not isSM100Family(),
    reason="The test is for Blackwell only. Current SM is %d." % getSMVersion(),
)
@pytest.mark.parametrize(
    "k, n",
    [(7168, 2112), (1536, 24576), (512, 32768), (16384, 7168), (7168, 4096),
     (2048, 7168), (1024, 1024)],
)
@pytest.mark.parametrize(
    "m",
    [7, 64, 128, 4096],
)
@pytest.mark.parametrize(
    "dtype",
    [torch.bfloat16],
)
def test_fp8_block_scale_deep_gemm(dtype, m, k, n):
    torch.random.manual_seed(0)
    a = torch.randn((m, k), device='cuda', dtype=dtype)
    b = torch.randn((n, k), device='cuda', dtype=dtype)

    act_b_fp8, act_b_sf = per_block_cast_to_fp8_e8m0(b)

    output_expected = a @ b.t()

    with autotune():
        output = torch.ops.trtllm.fp8_swap_ab_gemm(
            a,
            act_b_fp8,
            act_b_sf,
        )

    diff = calc_diff(output, output_expected)
    assert diff < 1e-2


@pytest.mark.skipif(
    getSMVersion() not in (100, 107, 89, 120),
    reason="The test is for Blackwell and Ada only. Current SM is %d." %
    getSMVersion(),
)
@pytest.mark.parametrize(
    "k, n",
    [(7168, 2112), (1536, 24576), (512, 32768), (16384, 7168), (7168, 4096),
     (2048, 7168), (1024, 1024)],
)
@pytest.mark.parametrize(
    "m",
    [7, 64, 128, 4096],
)
@pytest.mark.parametrize(
    "dtype",
    [torch.bfloat16],
)
def test_fp8_block_scale_gemm(dtype, m, k, n):

    torch.random.manual_seed(0)
    a = torch.randn((m, k), device='cuda', dtype=dtype) / k
    b = torch.randn((n, k), device='cuda', dtype=dtype) / k

    if getSMVersion() == 120:
        act_a_fp8, act_a_sf = fp8_utils.per_token_quant_and_transform(a)
        act_b_fp8, act_b_sf = per_block_cast_to_fp8_e8m0(b)
        act_b_sf = fp8_utils.transform_sf_into_required_layout(
            act_b_sf,
            mn=act_b_fp8.shape[0],
            k=act_b_fp8.shape[1],
            recipe=(1, 128, 128),
            is_sfa=False)
    else:
        act_a_fp8, act_a_sf = torch.ops.trtllm.fp8_quantize_1x128(a)
        act_b_fp8, act_b_sf = per_block_cast_to_fp8(b)

    output_expected = a @ b.t()

    output = torch.ops.trtllm.fp8_block_scaling_gemm(act_a_fp8, act_b_fp8,
                                                     act_a_sf, act_b_sf)
    diff = calc_diff(output, output_expected)
    assert diff < 1e-3
    torch.testing.assert_close(output, output_expected, atol=1e-3, rtol=1e-3)


@pytest.mark.skipif(
    getSMVersion() not in (100, 103),
    reason="The test is for SM100 and SM103 only. Current SM is %d." %
    getSMVersion(),
)
@pytest.mark.parametrize(
    "k, n",
    [(7168, 2112), (1536, 24576), (512, 32768), (16384, 7168), (7168, 4096),
     (2048, 7168), (1024, 1024)],
)
@pytest.mark.parametrize(
    "m",
    [7, 64, 128, 4096],
)
@pytest.mark.parametrize(
    "dtype",
    [torch.bfloat16],
)
@pytest.mark.parametrize(
    "use_tvm_ffi",
    [True, False],
)
def test_cute_dsl_fp8_block_scale_gemm(dtype, m, k, n, use_tvm_ffi):

    torch.random.manual_seed(0)
    a = torch.randn((m, k), device='cuda', dtype=dtype) / k
    b = torch.randn((n, k), device='cuda', dtype=dtype) / k

    act_a_fp8, act_a_sf = torch.ops.trtllm.fp8_quantize_1x128(a)
    act_b_fp8, act_b_sf = per_block_cast_to_fp8(b)

    output_expected = a @ b.t()

    with autotune():
        cute_dsl_output = torch.ops.trtllm.cute_dsl_fp8_gemm_blackwell(
            act_a_fp8, act_b_fp8, act_a_sf, act_b_sf, use_tvm_ffi=use_tvm_ffi)

    # test Cute DSL kernel
    cute_dsl_output = torch.ops.trtllm.cute_dsl_fp8_gemm_blackwell(
        act_a_fp8, act_b_fp8, act_a_sf, act_b_sf, use_tvm_ffi=use_tvm_ffi)

    diff = calc_diff(cute_dsl_output, output_expected)
    assert diff < 1e-3
    torch.testing.assert_close(cute_dsl_output,
                               output_expected,
                               atol=1e-3,
                               rtol=1e-3)


@pytest.mark.skipif(
    getSMVersion() != 90 and getSMVersion() != 89 and getSMVersion() != 120,
    reason="The test is for Hopper and Ada only. Current SM is %d." %
    getSMVersion(),
)
@pytest.mark.parametrize(
    "k, n",
    [(7168, 2112), (512, 32768), (16384, 7168), (2048, 7168)],
)
@pytest.mark.parametrize(
    "m",
    [7, 64, 128],
)
@pytest.mark.parametrize(
    "num_groups",
    [4, 8, 16],
)
@pytest.mark.parametrize(
    "dtype",
    [torch.bfloat16],
)
def test_fp8_block_scale_bmm(dtype, m, k, n, num_groups):

    torch.random.manual_seed(0)
    a = torch.randn((m, num_groups, k), device='cuda', dtype=dtype) / k
    b = torch.randn((num_groups, n, k), device='cuda', dtype=dtype) / k

    if getSMVersion() == 120:
        a_fp8, a_scales = fp8_utils.per_token_quant_and_transform(
            a, need_permute102=True)
        b_fp8, b_scales = per_block_cast_to_fp8_e8m0(b)
        b_scales = fp8_utils.transform_sf_into_required_layout(
            b_scales,
            mn=n,
            k=k,
            recipe=(1, 128, 128),
            num_groups=num_groups,
            is_sfa=False)
    else:
        a_fp8, a_scales = torch.ops.trtllm.fp8_batched_quantize_1x128_permute102(
            a)

        b_fp8 = torch.zeros_like(b, device='cuda', dtype=torch.float8_e4m3fn)
        b_scales = torch.zeros((num_groups, (n + 127) // 128, (k + 127) // 128),
                               device='cuda',
                               dtype=torch.float)

        for i in range(num_groups):
            b_fp8[i], b_scales[i] = per_block_cast_to_fp8(b[i])

    output_expected = torch.einsum('mgk,gnk->gmn', a, b)
    output = torch.empty((num_groups, m, n), device='cuda', dtype=dtype)

    torch.ops.trtllm.fp8_block_scaling_bmm_out(a_fp8, b_fp8, a_scales, b_scales,
                                               output)
    diff = calc_diff(output, output_expected)
    assert diff < 1e-3
    torch.testing.assert_close(output, output_expected, atol=1e-3, rtol=1e-3)


@pytest.mark.skipif(
    getSMVersion() != 120,
    reason="The test is for SM120 only. Current SM is %d." % getSMVersion(),
)
@pytest.mark.parametrize(
    "k, n",
    [(7168, 2112), (2048, 7168)],
)
@pytest.mark.parametrize(
    "num_rows",
    [7, 64, 128],
)
@pytest.mark.parametrize(
    "num_experts, top_k",
    [(4, 3), (8, 4), (16, 5)],
)
@pytest.mark.parametrize(
    "dtype",
    [torch.bfloat16],
)
def test_fp8_block_scale_moe_gemm(dtype, num_rows, top_k, num_experts, k, n):

    def mock_moe_fc1(num_rows: int, top_k: int, num_experts: int, k: int,
                     n: int):
        assert top_k <= num_experts, 'top_k must be less than or equal to num_experts'
        # routing and selecting
        expert_ids = torch.randint(0,
                                   num_experts, (num_rows, top_k),
                                   device="cuda",
                                   dtype=torch.int)
        token_per_expert = torch.bincount(expert_ids.flatten(),
                                          minlength=num_experts).long()
        token_offset = torch.cumsum(token_per_expert, dim=0)  # int64
        token_offset = torch.cat(
            [torch.zeros((1, ), device="cuda", dtype=torch.long), token_offset],
            dim=0)

        total_rows = token_per_expert.sum()
        expanded_tokens = torch.randn(
            (total_rows, k), device="cuda", dtype=dtype) / k
        experts_weights = torch.randn(
            (num_experts, n, k), device="cuda", dtype=dtype) / k
        fc1_ref = torch.zeros(total_rows, n, device="cuda", dtype=dtype)

        # moe fc1 compute
        for i in range(num_experts):
            start = token_offset[i]
            end = token_offset[i + 1]
            if start < end:
                fc1_ref[start:end] = expanded_tokens[
                    start:end] @ experts_weights[i].t()

        return token_offset, expanded_tokens, experts_weights, fc1_ref

    token_offset, a, b, output_expected = mock_moe_fc1(num_rows, top_k,
                                                       num_experts, k, n)
    fp8_b, sf_b = per_block_cast_to_fp8_e8m0(b)
    sf_b = fp8_utils.transform_sf_into_required_layout(
        sf=sf_b,
        mn=b.shape[-2],
        k=b.shape[-1],
        recipe=(1, 128, 128),
        num_groups=num_experts,
        is_sfa=False,
    )
    b_fp8 = (fp8_b, sf_b)

    dummy_sfa = torch.zeros((1, 1), dtype=torch.int32, device="cuda")
    output = torch.ops.trtllm.fp8_block_scaling_moe_gemm(
        a, b_fp8[0], dummy_sfa, b_fp8[1], token_offset)
    diff = calc_diff(output, output_expected)
    print(
        f"num_rows={num_rows}, top_k={top_k}, num_experts={num_experts}, k={k}, n={n}, diff={diff:.5f}"
    )
    assert diff < 1e-3
    torch.testing.assert_close(output, output_expected, atol=1e-3, rtol=1e-3)


@pytest.mark.skipif(
    getSMVersion() not in (100, 103),
    reason="The test is for SM100 and SM103 only. Current SM is %d." %
    getSMVersion(),
)
@pytest.mark.parametrize(
    "k, n",
    [(7168, 2112), (512, 32768), (16384, 7168), (2048, 7168)],
)
@pytest.mark.parametrize(
    "m",
    [7, 64, 128],
)
@pytest.mark.parametrize(
    "num_groups",
    [4, 8, 16],
)
@pytest.mark.parametrize(
    "dtype",
    [torch.bfloat16],
)
@pytest.mark.parametrize(
    "use_tvm_ffi",
    [True, False],
)
def test_cute_dsl_fp8_block_scale_bmm(dtype, m, k, n, num_groups, use_tvm_ffi):

    torch.random.manual_seed(0)
    a = torch.randn((m, num_groups, k), device='cuda', dtype=dtype) / k
    a_fp8, a_scales = torch.ops.trtllm.fp8_batched_quantize_1x128_permute102(a)

    b = torch.randn((num_groups, n, k), device='cuda', dtype=dtype) / k
    b_fp8 = torch.zeros_like(b, device='cuda', dtype=torch.float8_e4m3fn)
    b_scales = torch.zeros((num_groups, (n + 127) // 128, (k + 127) // 128),
                           device='cuda',
                           dtype=torch.float)

    for i in range(num_groups):
        b_fp8[i], b_scales[i] = per_block_cast_to_fp8(b[i])

    output_expected = torch.einsum('mgk,gnk->gmn', a, b)
    output = torch.empty((num_groups, m, n), device='cuda', dtype=dtype)
    # tune
    with autotune():
        torch.ops.trtllm.cute_dsl_fp8_bmm_blackwell(a_fp8,
                                                    b_fp8,
                                                    a_scales,
                                                    b_scales,
                                                    output,
                                                    use_tvm_ffi=use_tvm_ffi)
    # run the tuned kernel
    torch.ops.trtllm.cute_dsl_fp8_bmm_blackwell(a_fp8,
                                                b_fp8,
                                                a_scales,
                                                b_scales,
                                                output,
                                                use_tvm_ffi=use_tvm_ffi)
    diff = calc_diff(output, output_expected)
    assert diff < 1e-3
    torch.testing.assert_close(output, output_expected, atol=1e-3, rtol=1e-3)


@pytest.mark.skipif(not IS_CUTLASS_DSL_AVAILABLE,
                    reason="The test requires CuTe DSL support.")
def test_cute_dsl_fp8_block_scale_bmm_autotune_profiles() -> None:
    from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import \
        CuteDSLFp8BlackwellBmmRunner

    batch_size, n, k = 2, 128, 128

    def make_inputs(m: int) -> list[torch.Tensor]:
        scale_m = (m + 3) // 4 * 4
        return [
            torch.empty((batch_size, m, k)),
            torch.empty((batch_size, n, k)),
            torch.empty((batch_size, k // 128, scale_m)),
            torch.empty((batch_size, n // 128, k // 128)),
            torch.empty((batch_size, m, n)),
        ]

    tuning_config = CuteDSLFp8BlackwellBmmRunner.tuning_config
    profiles = AutoTuner()._optimization_profiles(tuning_config,
                                                  make_inputs(m=7))
    profile_shapes = [profile.get_opt_shapes() for profile in profiles]
    assert [(shapes[0][1], shapes[2][2], shapes[4][1])
            for shapes in profile_shapes] == [
                (1, 4, 1),
                (2, 4, 2),
                (4, 4, 4),
                (7, 8, 7),
            ]

    def cache_profile(m: int) -> tuple[tuple[int, ...], ...]:
        input_shapes = tuple(tensor.shape for tensor in make_inputs(m))
        return AutoTuner._find_nearest_profile(
            input_shapes,
            tuning_config.dynamic_tensor_specs,
            tuning_config.constraint_specs,
            tuning_config.tune_max_num_tokens,
        )

    assert cache_profile(5) == cache_profile(7)


@pytest.mark.skipif(
    getSMVersion() != 107 or not IS_CUTLASS_DSL_RUBIN_AVAILABLE,
    reason="The test requires SM107 and SM107 CuTe DSL support.",
)
@pytest.mark.parametrize(
    "k, n",
    [(7168, 2112), (1536, 24576), (512, 32768), (16384, 7168), (7168, 4096),
     (2048, 7168), (1024, 1024)],
)
@pytest.mark.parametrize(
    "m",
    [7, 64, 128, 4096],
)
@pytest.mark.parametrize(
    "dtype",
    [torch.bfloat16],
)
@pytest.mark.parametrize(
    "use_tvm_ffi",
    [True, False],
)
def test_cute_dsl_fp8_block_scale_gemm_rubin(dtype, m, k, n, use_tvm_ffi):

    torch.random.manual_seed(0)
    a = torch.randn((m, k), device='cuda', dtype=dtype) / k
    b = torch.randn((n, k), device='cuda', dtype=dtype) / k

    act_a_fp8, act_a_sf = torch.ops.trtllm.fp8_quantize_1x128(a)
    act_b_fp8, act_b_sf = per_block_cast_to_fp8(b)

    output_expected = a @ b.t()

    with autotune():
        cute_dsl_output = torch.ops.trtllm.cute_dsl_fp8_gemm_rubin(
            act_a_fp8, act_b_fp8, act_a_sf, act_b_sf, use_tvm_ffi=use_tvm_ffi)

    cute_dsl_output = torch.ops.trtllm.cute_dsl_fp8_gemm_rubin(
        act_a_fp8, act_b_fp8, act_a_sf, act_b_sf, use_tvm_ffi=use_tvm_ffi)

    diff = calc_diff(cute_dsl_output, output_expected)
    assert diff < 1e-3
    torch.testing.assert_close(cute_dsl_output,
                               output_expected,
                               atol=1e-3,
                               rtol=1e-3)


@pytest.mark.skipif(
    getSMVersion() != 107 or not IS_CUTLASS_DSL_RUBIN_AVAILABLE,
    reason="The test requires SM107 and SM107 CuTe DSL support.",
)
def test_cute_dsl_mxfp8_gemm_rubin_k128_replicated_scales():
    """Connect K128 quantization to the dense kernel's K32 SF contract."""
    m, n, k = 1, 128, 512
    torch.random.manual_seed(0)
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16) / k
    b = torch.randn((n, k), device="cuda", dtype=torch.bfloat16) / k

    a_fp8, a_sf = torch.ops.trtllm.fp8_quantize_1x128_packed_ue8m0(a)
    b_fp8, b_sf_k128 = per_block_cast_to_fp8_e8m0(b)
    b_sf = fp8_utils.transform_k128_scales_to_cutedsl_mxfp8_layout(b_sf_k128,
                                                                   mn=n,
                                                                   k=k)

    with autotune():
        output = torch.ops.trtllm.cute_dsl_mxfp8_gemm_rubin(
            a_fp8, b_fp8, a_sf, b_sf)

    output = torch.ops.trtllm.cute_dsl_mxfp8_gemm_rubin(a_fp8, b_fp8, a_sf,
                                                        b_sf)
    expected = a @ b.t()

    diff = calc_diff(output, expected)
    assert diff < 1e-3
    torch.testing.assert_close(output, expected, atol=1e-3, rtol=1e-3)


@pytest.mark.skipif(
    getSMVersion() != 107 or not IS_CUTLASS_DSL_RUBIN_AVAILABLE,
    reason="The test requires SM107 and SM107 CuTe DSL support.",
)
@pytest.mark.parametrize("m", [1, 128])
def test_cute_dsl_dsv4_qb_gemm_fused_rmsnorm_rope_quant(m):
    """Validate production decode shapes, including CUDA Graph replay."""
    num_heads, head_dim, nope_dim, k = 16, 512, 448, 1536
    n = num_heads * head_dim
    eps = 1e-6
    torch.random.manual_seed(17)
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((n, k), device="cuda", dtype=torch.bfloat16) / k**0.5

    a_fp8, a_sf = torch.ops.trtllm.fp8_quantize_1x128_packed_ue8m0(a)
    ref_a_fp8, ref_a_sf = torch.ops.trtllm.fp8_quantize_1x128(a, use_ue8m0=True)
    a_sf_k128 = ref_a_sf[:, :m].t().contiguous().float()
    assert torch.equal(a_fp8.view(torch.uint8), ref_a_fp8.view(torch.uint8))
    b_fp8, b_sf_k128 = per_block_cast_to_fp8_e8m0(b)
    b_sf = fp8_utils.transform_k128_scales_to_cutedsl_mxfp8_layout(b_sf_k128,
                                                                   mn=n,
                                                                   k=k)

    position_ids = torch.arange(m, dtype=torch.int32, device="cuda")
    cu_q_seqlens = torch.tensor([0, m], dtype=torch.int32, device="cuda")
    kv_cache_lengths = torch.tensor([m], dtype=torch.int32, device="cuda")
    quant_scale_qkv = torch.ones(1, dtype=torch.float32, device="cuda")

    rope_dim = head_dim - nope_dim
    freq_idx = torch.arange(0, rope_dim, 2, dtype=torch.float32, device="cuda")
    inv_freq = 1.0 / (10000.0**(freq_idx / rope_dim))
    angles = torch.outer(position_ids.float(), inv_freq)
    duplicated_angles = torch.cat([angles, angles], dim=-1)
    cos_sin_cache = torch.stack(
        [duplicated_angles.cos(),
         duplicated_angles.sin()], dim=-1).contiguous()

    def run_fusion():
        return torch.ops.trtllm.cute_dsl_dsv4_qb_gemm_fused_rmsnorm_rope_quant(
            a_fp8,
            b_fp8,
            a_sf,
            b_sf,
            cos_sin_cache,
            cu_q_seqlens,
            kv_cache_lengths,
            position_ids,
            quant_scale_qkv,
            eps,
        )

    run_fusion()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = run_fusion()
    quant_scale_qkv.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.count_nonzero(output) == 0
    quant_scale_qkv.fill_(1.0)
    graph.replay()
    torch.cuda.synchronize()
    output = output.view(m, num_heads, head_dim)

    a_scale = a_sf_k128.repeat_interleave(128, dim=1)[:, :k]
    b_scale = b_sf_k128.repeat_interleave(128, dim=0)[:n].repeat_interleave(
        128, dim=1)[:, :k]
    q_proj = ((a_fp8.float() * a_scale) @ (b_fp8.float() * b_scale).t()).to(
        torch.bfloat16)
    q_proj = q_proj.view(m, num_heads, head_dim).float()
    inv_rms = torch.rsqrt(q_proj.square().mean(dim=-1, keepdim=True) + eps)
    normed = q_proj * inv_rms

    expected_nope = normed[..., :nope_dim].to(torch.float8_e4m3fn)
    q_pe = normed[..., nope_dim:].to(torch.bfloat16).float()
    coefficient = cos_sin_cache[position_ids.long(), :rope_dim //
                                2].unsqueeze(1)
    q0, q1 = q_pe[..., 0::2], q_pe[..., 1::2]
    cos, sin = coefficient[..., 0], coefficient[..., 1]
    rotated = torch.stack([cos * q0 - sin * q1, cos * q1 + sin * q0],
                          dim=-1).flatten(-2)
    expected_rope = rotated.to(torch.bfloat16).to(torch.float8_e4m3fn)
    expected = torch.cat([expected_nope, expected_rope], dim=-1)

    torch.testing.assert_close(
        output[..., :nope_dim].float(),
        expected[..., :nope_dim].float(),
        rtol=0.25,
        atol=2.0**-8,
    )
    torch.testing.assert_close(
        output[..., nope_dim:].float(),
        expected[..., nope_dim:].float(),
        rtol=0.25,
        atol=2.0**-5,
    )


@pytest.mark.skipif(
    not IS_CUTLASS_DSL_RUBIN_AVAILABLE,
    reason="The test requires SM107 CuTe DSL support.",
)
def test_cute_dsl_blockscaled_rubin_runner_hierarchy():
    from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import (
        CuteDSLBlockScaledRubinLinear, CuteDSLMXFP8RubinLinear,
        CuteDSLNVFP4RubinLinear)
    from tensorrt_llm._torch.cute_dsl_kernels.rubin.dense_blockscaled_gemm_persistent import \
        Sm107BlockScaledPersistentDenseGemmMixedClustersKernel

    assert CuteDSLNVFP4RubinLinear.__bases__ == (
        CuteDSLBlockScaledRubinLinear, )
    assert CuteDSLMXFP8RubinLinear.__bases__ == (
        CuteDSLBlockScaledRubinLinear, )
    assert (CuteDSLNVFP4RubinLinear.kernel_cache
            is not CuteDSLMXFP8RubinLinear.kernel_cache)
    assert "_compute_grid" not in (
        Sm107BlockScaledPersistentDenseGemmMixedClustersKernel.__dict__)
    assert "_compute_mixed_cluster_grid" in (
        Sm107BlockScaledPersistentDenseGemmMixedClustersKernel.__dict__)


@pytest.mark.skipif(
    getSMVersion() != 107 or not IS_CUTLASS_DSL_RUBIN_AVAILABLE,
    reason="The test requires SM107 and SM107 CuTe DSL support.",
)
@pytest.mark.parametrize(
    "m,n,k,mma_tiler,mma_inst_shape,use_prefetch",
    [
        (1024, 7168, 2048, (256, 192, 128), (256, 192, 64), True),
        (1024, 768, 7168, (256, 64, 128), (256, 64, 64), True),
        (1024, 8192, 1536, (256, 192, 128), (256, 192, 64), True),
        (2048, 7168, 2048, (256, 192, 128), (256, 192, 64), True),
        (2048, 7168, 384, (256, 192, 128), (256, 192, 64), True),
        (2048, 8192, 1536, (256, 192, 128), (256, 192, 64), True),
        (4096, 7168, 2048, (256, 192, 128), (256, 192, 64), True),
        (4096, 7168, 384, (128, 192, 128), (128, 192, 64), True),
        (4096, 8192, 1536, (256, 192, 128), (256, 192, 64), True),
        (512, 7168, 2048, (128, 192, 128), (128, 192, 64), True),
        (512, 7168, 2048, (256, 192, 128), (256, 192, 64), True),
        (8192, 7168, 2048, (256, 256, 128), (256, 256, 64), False),
        (8192, 7168, 384, (128, 192, 128), (128, 192, 64), True),
        (8192, 8192, 1536, (256, 256, 128), (256, 256, 64), False),
    ],
)
def test_cute_dsl_mxfp8_gemm_rubin_mixed_clusters_multi_wave(
        m, n, k, mma_tiler, mma_inst_shape, use_prefetch):
    """Mixed cluster shapes must share one persistent tile partition."""
    from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import \
        CuteDSLMXFP8RubinLinear

    torch.random.manual_seed(41)
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((n, k), device="cuda", dtype=torch.bfloat16) * 0.02

    a_fp8, a_sf = torch.ops.trtllm.fp8_quantize_1x128_packed_ue8m0(a)
    b_fp8, b_sf_k128 = per_block_cast_to_fp8_e8m0(b)
    b_sf = fp8_utils.transform_k128_scales_to_cutedsl_mxfp8_layout(b_sf_k128,
                                                                   mn=n,
                                                                   k=k)
    alpha = torch.ones((), dtype=torch.float32, device="cuda")
    inputs = [a_fp8, b_fp8, a_sf, b_sf, alpha]

    base_tactic = (
        "base",
        mma_tiler,
        mma_inst_shape,
        (2, 1),
        False,
        use_prefetch,
        "static",
        "m",
        1,
    )
    mixed_tactic = (
        "mixed_clusters",
        mma_tiler,
        mma_inst_shape,
        (4, 2),
        (2, 1),
        False,
        use_prefetch,
        "static",
        "m",
    )
    runner = CuteDSLMXFP8RubinLinear(output_dtype=torch.bfloat16,
                                     use_tvm_ffi=True)
    tactics = runner.get_valid_tactics(inputs, None)
    assert base_tactic in tactics
    assert mixed_tactic in tactics

    base_output = runner(inputs, tactic=base_tactic)
    mixed_output = runner(inputs, tactic=mixed_tactic)
    torch.cuda.synchronize()

    expected = a @ b.t()
    assert torch.isfinite(mixed_output).all()
    assert calc_diff(base_output, expected) < 1e-3
    assert calc_diff(mixed_output, expected) < 1e-3
    torch.testing.assert_close(mixed_output, base_output, rtol=0, atol=0)


@pytest.mark.skipif(
    getSMVersion() != 107 or not IS_CUTLASS_DSL_RUBIN_AVAILABLE,
    reason="The test requires SM107 and SM107 CuTe DSL support.",
)
def test_cute_dsl_mxfp8_gemm_rubin_mixed_clusters_clc_dynamic_prefetch_multi_wave(
):
    """Mixed-cluster MXFP8 supports CLC scheduling with prefetch enabled."""
    from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import \
        CuteDSLMXFP8RubinLinear

    m, n, k = 4096, 2048, 512
    torch.random.manual_seed(42)
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((n, k), device="cuda", dtype=torch.bfloat16) * 0.02

    a_fp8, a_sf = torch.ops.trtllm.fp8_quantize_1x128_packed_ue8m0(a)
    b_fp8, b_sf_k128 = per_block_cast_to_fp8_e8m0(b)
    b_sf = fp8_utils.transform_k128_scales_to_cutedsl_mxfp8_layout(b_sf_k128,
                                                                   mn=n,
                                                                   k=k)
    alpha = torch.ones((), dtype=torch.float32, device="cuda")
    inputs = [a_fp8, b_fp8, a_sf, b_sf, alpha]

    static_tactic = (
        "mixed_clusters",
        (256, 256, 128),
        (256, 256, 64),
        (4, 2),
        (2, 1),
        False,
        True,
        "static",
        "m",
    )
    dynamic_tactic = (*static_tactic[:-2], "clc_dynamic", "m")
    runner = CuteDSLMXFP8RubinLinear(output_dtype=torch.bfloat16,
                                     use_tvm_ffi=True)
    valid_tactics = runner.get_valid_tactics(inputs, None)
    for raster_order in ("m", "n"):
        tactic = (*static_tactic[:-2], "static", raster_order)
        assert tactic in valid_tactics
    assert dynamic_tactic in valid_tactics
    unsafe_dynamic_tactic = (*static_tactic[:-2], "clc_dynamic", "n")
    assert unsafe_dynamic_tactic not in valid_tactics

    static_output = runner(inputs, tactic=static_tactic)
    dynamic_output = runner(inputs, tactic=dynamic_tactic)
    torch.cuda.synchronize()

    expected = a @ b.t()
    assert torch.isfinite(dynamic_output).all()
    assert calc_diff(static_output, expected) < 1e-3
    assert calc_diff(dynamic_output, expected) < 1e-3
    torch.testing.assert_close(dynamic_output, static_output, rtol=0, atol=0)


@pytest.mark.skipif(
    getSMVersion() != 107 or not IS_CUTLASS_DSL_RUBIN_AVAILABLE,
    reason="The test requires SM107 and SM107 CuTe DSL support.",
)
def test_cute_dsl_mxfp8_gemm_rubin_clc_dynamic_prefetch_multi_wave():
    """CLC dynamic scheduling supports Boolean-enabled prefetch."""
    from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import \
        CuteDSLMXFP8RubinLinear

    m, n, k = 2048, 8192, 1536
    torch.random.manual_seed(43)
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((n, k), device="cuda", dtype=torch.bfloat16) * 0.02

    a_fp8, a_sf = torch.ops.trtllm.fp8_quantize_1x128_packed_ue8m0(a)
    b_fp8, b_sf_k128 = per_block_cast_to_fp8_e8m0(b)
    b_sf = fp8_utils.transform_k128_scales_to_cutedsl_mxfp8_layout(b_sf_k128,
                                                                   mn=n,
                                                                   k=k)
    alpha = torch.ones((), dtype=torch.float32, device="cuda")
    inputs = [a_fp8, b_fp8, a_sf, b_sf, alpha]

    static_tactic = (
        "base",
        (256, 256, 128),
        (256, 256, 64),
        (4, 2),
        False,
        True,
        "static",
        "m",
        1,
    )
    dynamic_tactic = (
        "base",
        (256, 256, 128),
        (256, 256, 64),
        (4, 2),
        False,
        True,
        "clc_dynamic",
        "n",
        1,
    )
    split_k_tactic = (*dynamic_tactic[:-1], 4)
    runner = CuteDSLMXFP8RubinLinear(output_dtype=torch.bfloat16,
                                     use_tvm_ffi=True)
    valid_tactics = runner.get_valid_tactics(inputs, None)
    assert static_tactic in valid_tactics
    assert dynamic_tactic in valid_tactics
    assert split_k_tactic in valid_tactics
    for scheduler_mode in ("static", "clc_dynamic"):
        for raster_order in ("m", "n"):
            tactic = (*static_tactic[:-3], scheduler_mode, raster_order, 1)
            assert tactic in valid_tactics

    static_output = runner(inputs, tactic=static_tactic)
    dynamic_output = runner(inputs, tactic=dynamic_tactic)
    split_k_output = runner(inputs, tactic=split_k_tactic)
    torch.cuda.synchronize()

    expected = a @ b.t()
    assert torch.isfinite(dynamic_output).all()
    assert torch.isfinite(split_k_output).all()
    assert calc_diff(static_output, expected) < 1e-3
    assert calc_diff(dynamic_output, expected) < 1e-3
    assert calc_diff(split_k_output, expected) < 1e-3
    torch.testing.assert_close(dynamic_output, static_output, rtol=0, atol=0)


@pytest.mark.skipif(
    getSMVersion() != 107 or not IS_CUTLASS_DSL_RUBIN_AVAILABLE,
    reason="The test requires SM107 and SM107 CuTe DSL support.",
)
@pytest.mark.parametrize("m", [1, 7, 128])
@pytest.mark.parametrize("use_tvm_ffi", [True, False])
def test_cute_dsl_fp8_block_scale_bmm_rubin(m, use_tvm_ffi):
    k = 4096
    n = 1024
    num_groups = 8
    dtype = torch.bfloat16

    torch.random.manual_seed(0)
    a = torch.randn((m, num_groups, k), device="cuda", dtype=dtype) / k
    a_fp8, a_scales = torch.ops.trtllm.fp8_batched_quantize_1x128_permute102(a)

    b = torch.randn((num_groups, n, k), device="cuda", dtype=dtype) / k
    b_fp8 = torch.empty_like(b, dtype=torch.float8_e4m3fn)
    b_scales = torch.empty(
        (num_groups, n // 128, k // 128),
        device="cuda",
        dtype=torch.float32,
    )
    for group_idx in range(num_groups):
        b_fp8[group_idx], b_scales[group_idx] = per_block_cast_to_fp8(
            b[group_idx])

    output_expected = torch.einsum("mgk,gnk->gmn", a, b)
    output = torch.empty((num_groups, m, n), device="cuda", dtype=dtype)
    torch.ops.trtllm.cute_dsl_fp8_bmm_rubin(
        a_fp8,
        b_fp8,
        a_scales,
        b_scales,
        output,
        use_tvm_ffi=use_tvm_ffi,
    )

    diff = calc_diff(output, output_expected)
    assert diff < 1e-3
    torch.testing.assert_close(output, output_expected, atol=1e-3, rtol=1e-3)


def _allocate_fp8_bmm_quant_outputs(a_fp8, b_fp8):
    num_groups, m, _ = a_fp8.shape
    n = b_fp8.shape[1]
    flat_n = num_groups * n
    fp8_output = torch.empty((m, flat_n),
                             dtype=torch.float8_e4m3fn,
                             device=a_fp8.device)
    fp8_batched = fp8_output.view(m, num_groups, n).permute(1, 0, 2)
    scale_numel = ((m + 127) // 128 * 128 *
                   (((flat_n + 31) // 32 + 3) // 4 * 4))
    packed_scale = torch.empty((scale_numel, ),
                               dtype=torch.uint8,
                               device=a_fp8.device)
    return fp8_output, fp8_batched, packed_scale


def _valid_fp8_bmm_quant_scale_indices(m, flat_n, device):
    sf_k_padded = ((flat_n + 31) // 32 + 3) // 4 * 4
    indices = []
    for row in range(m):
        for sf_n_base in range(0, flat_n, 128):
            sf_k_base = sf_n_base // 32
            for sf_replica in range(4):
                sf_k = sf_k_base + sf_replica
                byte_offset = ((
                    ((row // 128) *
                     (sf_k_padded // 4) + sf_k // 4) * 32 + row % 32) * 4 +
                               (row % 128) // 32) * 4 + sf_k % 4
                indices.append(byte_offset)
    return torch.tensor(indices, dtype=torch.int64, device=device)


def _assert_valid_fp8_bmm_quant_scales_equal(fused_scale, ref_scale, m, flat_n):
    indices = _valid_fp8_bmm_quant_scale_indices(m, flat_n, fused_scale.device)
    assert torch.equal(
        fused_scale.view(-1)[indices],
        ref_scale.contiguous().view(-1)[indices])


@pytest.mark.skipif(
    getSMVersion() != 107 or not IS_CUTLASS_DSL_RUBIN_AVAILABLE,
    reason="The test requires SM107 and SM107 CuTe DSL support.",
)
@pytest.mark.parametrize("m", [1, 16, 128, 256])
def test_cute_dsl_fp8_bmm_quantize_rubin_matches_separate(m):
    k, n, num_groups = 4096, 1024, 8
    torch.manual_seed(314)
    a = torch.randn((m, num_groups, k), device="cuda", dtype=torch.bfloat16) / k
    a_fp8, a_scales = torch.ops.trtllm.fp8_batched_quantize_1x128_permute102(a)
    b = torch.randn((num_groups, n, k), device="cuda", dtype=torch.bfloat16) / k
    b_fp8 = torch.empty_like(b, dtype=torch.float8_e4m3fn)
    b_scales = torch.empty(
        (num_groups, n // 128, k // 128),
        device="cuda",
        dtype=torch.float32,
    )
    for group_idx in range(num_groups):
        b_fp8[group_idx], b_scales[group_idx] = per_block_cast_to_fp8(
            b[group_idx])

    bf16_output = torch.empty((num_groups, m, n),
                              device="cuda",
                              dtype=torch.bfloat16)
    torch.ops.trtllm.cute_dsl_fp8_bmm_rubin(a_fp8, b_fp8, a_scales, b_scales,
                                            bf16_output)
    flat_bf16 = bf16_output.permute(1, 0, 2).reshape(m, num_groups * n)
    ref_fp8, ref_scale = (
        torch.ops.trtllm.fp8_quantize_1x128_packed_ue8m0(flat_bf16))

    fused_fp8, fused_batched, fused_scale = _allocate_fp8_bmm_quant_outputs(
        a_fp8, b_fp8)
    torch.ops.trtllm.cute_dsl_fp8_bmm_quantize_rubin_out(
        a_fp8,
        b_fp8,
        a_scales,
        b_scales,
        fused_batched,
        fused_scale,
    )
    torch.cuda.synchronize()

    _assert_valid_fp8_bmm_quant_scales_equal(fused_scale, ref_scale, m,
                                             num_groups * n)
    torch.testing.assert_close(fused_fp8.float(),
                               ref_fp8.float(),
                               rtol=0.0,
                               atol=1.0 / 16.0)


@pytest.mark.skipif(
    getSMVersion() != 107 or not IS_CUTLASS_DSL_RUBIN_AVAILABLE,
    reason="The test requires SM107 and SM107 CuTe DSL support.",
)
def test_cute_dsl_fp8_bmm_quantize_rubin_cuda_graph():
    m, n, k, num_groups = 7, 1024, 4096, 8
    a = torch.randn((m, num_groups, k), device="cuda", dtype=torch.bfloat16) / k
    a_fp8, a_scales = torch.ops.trtllm.fp8_batched_quantize_1x128_permute102(a)
    b = torch.randn((num_groups, n, k), device="cuda", dtype=torch.bfloat16) / k
    b_fp8 = torch.empty_like(b, dtype=torch.float8_e4m3fn)
    b_scales = torch.empty(
        (num_groups, n // 128, k // 128),
        device="cuda",
        dtype=torch.float32,
    )
    for group_idx in range(num_groups):
        b_fp8[group_idx], b_scales[group_idx] = per_block_cast_to_fp8(
            b[group_idx])

    fused_fp8, fused_batched, fused_scale = _allocate_fp8_bmm_quant_outputs(
        a_fp8, b_fp8)
    args = (
        a_fp8,
        b_fp8,
        a_scales,
        b_scales,
        fused_batched,
        fused_scale,
    )
    torch.ops.trtllm.cute_dsl_fp8_bmm_quantize_rubin_out(*args)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        torch.ops.trtllm.cute_dsl_fp8_bmm_quantize_rubin_out(*args)
    graph.replay()
    torch.cuda.synchronize()

    bf16_output = torch.empty((num_groups, m, n),
                              device="cuda",
                              dtype=torch.bfloat16)
    torch.ops.trtllm.cute_dsl_fp8_bmm_rubin(a_fp8, b_fp8, a_scales, b_scales,
                                            bf16_output)
    flat_bf16 = bf16_output.permute(1, 0, 2).reshape(m, num_groups * n)
    ref_fp8, ref_scale = (
        torch.ops.trtllm.fp8_quantize_1x128_packed_ue8m0(flat_bf16))
    _assert_valid_fp8_bmm_quant_scales_equal(fused_scale, ref_scale, m,
                                             num_groups * n)
    torch.testing.assert_close(fused_fp8.float(),
                               ref_fp8.float(),
                               rtol=0.0,
                               atol=1.0 / 16.0)


@pytest.mark.skipif(
    getSMVersion() != 107 or not IS_CUTLASS_DSL_RUBIN_AVAILABLE,
    reason="The test requires SM107 and SM107 CuTe DSL support.",
)
def test_cute_dsl_fp8_block_scale_bmm_rubin_cuda_graph():
    m, n, k, num_groups = 7, 1024, 4096, 8
    a = torch.randn((m, num_groups, k), device="cuda", dtype=torch.bfloat16) / k
    a_fp8, a_scales = torch.ops.trtllm.fp8_batched_quantize_1x128_permute102(a)
    b = torch.randn((num_groups, n, k), device="cuda", dtype=torch.bfloat16) / k
    b_fp8 = torch.empty_like(b, dtype=torch.float8_e4m3fn)
    b_scales = torch.empty(
        (num_groups, n // 128, k // 128),
        device="cuda",
        dtype=torch.float32,
    )
    for group_idx in range(num_groups):
        b_fp8[group_idx], b_scales[group_idx] = per_block_cast_to_fp8(
            b[group_idx])

    output = torch.empty((num_groups, m, n),
                         device="cuda",
                         dtype=torch.bfloat16)
    args = (a_fp8, b_fp8, a_scales, b_scales, output)
    torch.ops.trtllm.cute_dsl_fp8_bmm_rubin(*args)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        torch.ops.trtllm.cute_dsl_fp8_bmm_rubin(*args)
    output.fill_(float("nan"))
    graph.replay()

    output_expected = torch.einsum("mgk,gnk->gmn", a, b)
    torch.testing.assert_close(output, output_expected, atol=1e-3, rtol=1e-3)


def deepSeekFp8ComputeGemmReference(mM, mN, mK, valsC, dqSfsC, valsA, dqSfsA,
                                    valsB, dqSfsB, quantizeOutput, tileSize):
    for mi in range(mM):
        for ni in range(0, mN, tileSize):
            acc = torch.zeros(tileSize, dtype=torch.float32)
            for nj in range(tileSize):
                nk = ni + nj
                for ki in range(0, mK, tileSize):
                    '''
                    tmp = 0.0
                    for kj in range(tileSize):
                        kk = ki + kj
                        a = valsA[mi, kk]
                        b = valsB[nk, kk]
                        tmp += a * b
                    '''
                    tmp = valsA[mi, ki:ki + tileSize] @ valsB[nk,
                                                              ki:ki + tileSize]
                    dpSfA = dqSfsA[ki // tileSize, mi]
                    dpSfB = dqSfsB[ni // tileSize, ki // tileSize]
                    acc[nj] += (dpSfA * dpSfB) * tmp
            aMax = -float("inf")
            for nj in range(tileSize):
                aMax = max(aMax, abs(acc[nj]))
            E4m3MaxVal = 448
            if dqSfsC is not None:
                dqSfsC[ni // tileSize, mi] = aMax / E4m3MaxVal
            for nj in range(tileSize):
                val = acc[nj]
                if quantizeOutput:
                    val = val / aMax * E4m3MaxVal
                valsC[mi, ni + nj] = val


def fp8_block_scaling_gemm_reference(a, b, a_scale, b_scale, tile_size=128):
    m, k = a.shape
    n = b.shape[0]
    assert b.shape[1] == k
    assert k % tile_size == 0
    assert n % tile_size == 0
    assert a_scale.shape == (k // tile_size, m)
    assert b_scale.shape == (n // tile_size, k // tile_size)
    c = torch.zeros((m, n), dtype=torch.float32)

    a = a.to(torch.float32).cpu()
    b = b.to(torch.float32).cpu()
    a_scale = a_scale.cpu()
    b_scale = b_scale.cpu()
    deepSeekFp8ComputeGemmReference(m, n, k, c, None, a, a_scale, b, b_scale,
                                    False, tile_size)
    return c


@pytest.mark.skipif(
    getSMVersion() != 100,
    reason="The kernel only supports Blackwell. Current SM is %d." %
    getSMVersion(),
)
def test_fp8_blockscale_gemm_reference():
    torch.random.manual_seed(0)

    m, k, n = 3, 6, 4
    tile_size = 2
    a = torch.randn((m, k), dtype=torch.float32)
    b = torch.randn((n, k), dtype=torch.float32)
    a_scale = torch.ones((k // tile_size, m), dtype=torch.float32)
    b_scale = torch.ones((n // tile_size, k // tile_size), dtype=torch.float32)
    c = fp8_block_scaling_gemm_reference(a, b, a_scale, b_scale, tile_size)
    torch.testing.assert_close(c, a @ b.t(), atol=1e-1, rtol=1e-2)

    m, k, n = 4, 4, 4
    tile_size = 2
    a = torch.randn((m, k), dtype=torch.float32)
    b = torch.randn((n, k), dtype=torch.float32)
    a_scale = torch.randint(1, 8, (k // tile_size, m), dtype=torch.float32)
    b_scale = torch.randint(1,
                            8, (n // tile_size, k // tile_size),
                            dtype=torch.float32)
    c = fp8_block_scaling_gemm_reference(a, b, a_scale, b_scale, tile_size)
    c_expected = torch.zeros_like(c)
    for i in range(m):
        for j in range(n):
            for kk in range(k):
                a_current_scale = a_scale[kk // tile_size, i]
                b_current_scale = b_scale[j // tile_size, kk // tile_size]
                c_expected[i, j] += a[i, kk] * b[
                    j, kk] * a_current_scale * b_current_scale
    torch.testing.assert_close(c, c_expected, atol=1e-1, rtol=1e-2)


@pytest.mark.skipif(
    getSMVersion() not in (100, 107),
    reason="The kernel only supports Blackwell. Current SM is %d." %
    getSMVersion(),
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_fp8_blockscale_gemm_trtllmgen(dtype):
    torch.random.manual_seed(0)

    m, k, n = 128, 512, 512
    tile_size = 128
    if dtype == torch.float8_e4m3fn:
        a = torch.randn((m, k), device='cuda',
                        dtype=torch.float32).to(torch.float8_e4m3fn)
        a_scale = 2 * torch.randn(
            (k // tile_size, m), device='cuda').to(torch.float)

    else:
        a = torch.randn((m, k), device='cuda', dtype=dtype)
        a, a_scale = torch.ops.trtllm.fp8_quantize_1x128(a)
        a_scale = a_scale.view(-1, a.shape[0])

    b = torch.randn((n, k), device='cuda',
                    dtype=torch.float32).to(torch.float8_e4m3fn)
    b_scale = 2 * torch.randn(
        (n // tile_size, k // tile_size), device='cuda').to(torch.float)

    c_expected = fp8_block_scaling_gemm_reference(a, b, a_scale, b_scale,
                                                  tile_size)
    c_actual = torch.ops.trtllm.fp8_block_scaling_gemm(a, b, a_scale, b_scale)
    torch.testing.assert_close(c_actual.cpu().to(torch.float32),
                               c_expected,
                               atol=1e-1,
                               rtol=1e-2)


def run_test_in_subprocess(env, test_file):
    # Create a copy of the current environment
    process_env = os.environ.copy()

    # Update with the new environment variables
    process_env.update(env)

    # Run the test in a subprocess
    result = subprocess.run([sys.executable, '-m', 'pytest', test_file, '-v'],
                            capture_output=True,
                            text=True,
                            env=process_env)

    # Print the output
    print(result.stdout)
    if result.stderr:
        print(result.stderr)

    # Return the exit code
    return result.returncode


@pytest.mark.skipif(
    getSMVersion() != 90,
    reason="The test is for Hopper only. Current SM is %d." % getSMVersion(),
)
@pytest.mark.parametrize("env", [
    {},
    {
        'TRTLLM_DG_JIT_USE_NVCC': '1'
    },
])
def test_deep_gemm_in_subprocess(env):
    # Get the directory of the current file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Specify the target test file in the same directory
    test_file = os.path.join(current_dir, "deep_gemm_tests.py")

    exit_code = run_test_in_subprocess(env, test_file)
    assert exit_code == 0, f"Test for env {env} failed with exit code {exit_code}"
