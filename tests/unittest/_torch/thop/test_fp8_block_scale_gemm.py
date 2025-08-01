# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
                            per_block_cast_to_fp8_e8m0,
                            per_token_cast_to_fp8_e8m0)
from utils.util import getSMVersion


@pytest.mark.skipif(
    getSMVersion() != 100,
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

    act_a_fp8, act_a_sf = per_token_cast_to_fp8_e8m0(a)
    act_b_fp8, act_b_sf = per_block_cast_to_fp8_e8m0(b)

    output_expected = a @ b.t()
    import deep_gemm
    output = torch.empty((act_a_fp8.shape[0], act_b_fp8.shape[0]),
                         device=act_a_fp8.device,
                         dtype=torch.bfloat16)

    deep_gemm.fp8_gemm_nt((act_a_fp8, act_a_sf), (act_b_fp8, act_b_sf), output)
    diff = calc_diff(output, output_expected)
    assert diff < 1e-2


@pytest.mark.skipif(
    getSMVersion() != 100 and getSMVersion() != 89,
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

    act_a_fp8, act_a_sf = torch.ops.trtllm.fp8_quantize_1x128(a)
    act_b_fp8, act_b_sf = per_block_cast_to_fp8(b)

    output_expected = a @ b.t()

    output = torch.ops.trtllm.fp8_block_scaling_gemm(act_a_fp8, act_b_fp8,
                                                     act_a_sf, act_b_sf)
    diff = calc_diff(output, output_expected)
    assert diff < 1e-3
    torch.testing.assert_close(output, output_expected, atol=1e-3, rtol=1e-3)


@pytest.mark.skipif(
    getSMVersion() != 90 and getSMVersion() != 89,
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

    a_fp8, a_scales = torch.ops.trtllm.fp8_batched_quantize_1x128_permute102(a)

    b = torch.randn((num_groups, n, k), device='cuda', dtype=dtype) / k
    b_fp8 = torch.zeros_like(b, device='cuda', dtype=torch.float8_e4m3fn)
    b_scales = torch.zeros((num_groups, (n + 127) // 128, (k + 127) // 128),
                           device='cuda',
                           dtype=torch.float)

    for i in range(num_groups):
        b_fp8[i], b_scales[i] = per_block_cast_to_fp8(b[i])

    output_expected = torch.einsum('mgk,gnk->gmn', a, b)
    output = torch.empty((num_groups, m, n),
                         device='cuda',
                         dtype=torch.bfloat16)

    torch.ops.trtllm.fp8_block_scaling_bmm_out(a_fp8, b_fp8, a_scales, b_scales,
                                               output)
    diff = calc_diff(output, output_expected)
    assert diff < 1e-3
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
    getSMVersion() != 100,
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
    {
        'TRTLLM_DG_ENABLED': '0'
    },
])
def test_deep_gemm_in_subprocess(env):
    # Get the directory of the current file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Specify the target test file in the same directory
    test_file = os.path.join(current_dir, "deep_gemm_tests.py")

    exit_code = run_test_in_subprocess(env, test_file)
    assert exit_code == 0, f"Test for env {env} failed with exit code {exit_code}"
