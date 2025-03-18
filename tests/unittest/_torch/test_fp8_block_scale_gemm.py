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
import itertools
from typing import List, Tuple

import pytest
import torch
from helpers import calc_diff, ceil_div, per_block_cast_to_fp8
from utils.util import getSMVersion


@pytest.mark.skipif(
    getSMVersion() != 90,
    reason="Op only supported on Hopper",
)
@pytest.mark.parametrize(
    "k, n",
    [(7168, 2112), (1536, 24576), (512, 32768), (16384, 7168), (7168, 4096),
     (2048, 7168), (1024, 1024)],
)
@pytest.mark.parametrize(
    "m",
    [64, 128, 4096],
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

    output = torch.ops.trtllm.fp8_block_scaling_gemm(act_a_fp8, act_b_fp8,
                                                     act_a_sf, act_b_sf)

    output_expected = a @ b.t()
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
        a_scale = 2 * torch.rand(
            (k // tile_size, m), device='cuda').to(torch.float)

    else:
        a = torch.randn((m, k), device='cuda', dtype=dtype)
        a, a_scale = torch.ops.trtllm.fp8_quantize_1x128(a)
        a_scale = a_scale.view(-1, a.shape[0])

    b = torch.randn((n, k), device='cuda',
                    dtype=torch.float32).to(torch.float8_e4m3fn)
    b_scale = 2 * torch.rand(
        (n // tile_size, k // tile_size), device='cuda').to(torch.float)

    c_expected = fp8_block_scaling_gemm_reference(a, b, a_scale, b_scale,
                                                  tile_size)
    c_actual = torch.ops.trtllm.fp8_block_scaling_gemm(a, b, a_scale, b_scale)
    torch.testing.assert_close(c_actual.cpu().to(torch.float32),
                               c_expected,
                               atol=1e-1,
                               rtol=1e-2)


def change_to_offset_layout(
    ms: List[int],
    x_fp8: torch.Tensor,
    x_scale: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    x_list = []
    x_scale_list = []
    shape_m_total = 0
    num_problems = len(ms)
    m_acc = [0] + list(itertools.accumulate(ms))

    for i in range(num_problems):
        ms[i]
        x_list.append(x_fp8[m_acc[i]:m_acc[i + 1]])
        x_scale_padded = x_scale[m_acc[i]:m_acc[i + 1]]
        if x_scale_padded.shape[0] % 32 != 0:
            x_empty = torch.zeros(
                [32 - (x_scale_padded.shape[0] % 32), x_scale_padded.shape[1]],
                dtype=x_scale_padded.dtype,
                device=x_scale_padded.device,
            )
            x_scale_padded = torch.cat([x_scale_padded, x_empty])
        x_scale_list.append(x_scale_padded)

    shape_m_total = m_acc[-1]
    ret_x = torch.cat(x_list)
    ret_x_scale = torch.cat(x_scale_list)
    ret_x_scale = ret_x_scale.t().contiguous()
    pad_target = ceil_div(shape_m_total + num_problems * 31, 32) * 32
    pad_target -= ret_x_scale.shape[1]
    ret_x_scale = torch.nn.functional.pad(ret_x_scale, (0, pad_target),
                                          mode='constant',
                                          value=0)

    return ret_x, ret_x_scale


def construct_grouped(
    ms: List[int], k: int, n: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor]:
    assert all(m % 4 == 0 for m in ms), f'TMA alignment error: {ms}'
    torch.random.manual_seed(0)
    num_groups = len(ms)
    x = torch.randn((sum(ms), k), device='cuda', dtype=torch.bfloat16) / k
    y = torch.randn((num_groups, n, k), device='cuda', dtype=torch.bfloat16) / k
    m_acc = [0] + list(itertools.accumulate(ms))
    ref_out = torch.empty((sum(ms), n), device='cuda', dtype=torch.bfloat16)
    for i in range(num_groups):
        ref_out[m_acc[i]:m_acc[i + 1]] = torch.einsum('mk,nk->mn',
                                                      x[m_acc[i]:m_acc[i + 1]],
                                                      y[i])

    x_fp8, x_scale = (torch.empty_like(x, dtype=torch.float8_e4m3fn),
                      torch.empty((sum(ms), k // 128),
                                  device='cuda',
                                  dtype=torch.float))
    y_fp8, y_scale = (torch.empty_like(y, dtype=torch.float8_e4m3fn),
                      torch.empty((num_groups, (n + 127) // 128, k // 128),
                                  device='cuda',
                                  dtype=torch.float))

    for i in range(num_groups):
        xi = x[m_acc[i]:m_acc[i + 1]]
        yi = y[i]
        x_fp8_i, x_scale_i = torch.ops.trtllm.fp8_quantize_1x128(xi)
        x_fp8[m_acc[i]:m_acc[i + 1]] = x_fp8_i.view(
            x_fp8[m_acc[i]:m_acc[i + 1]].shape)
        x_scale[m_acc[i]:m_acc[i + 1]] = x_scale_i.view(
            x_scale[m_acc[i]:m_acc[i + 1]].shape[::-1]).t().contiguous()

        y_fp8_i, y_scale_i = per_block_cast_to_fp8(yi)
        y_fp8[i] = y_fp8_i.view(y_fp8[i].shape)
        y_scale[i] = y_scale_i.view(y_scale[i].shape)

    return x_fp8, x_scale, y_fp8, y_scale, ref_out


def construct_batched(
    num_batches: int, m: int, k: int, n: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor]:
    assert m % 4 == 0, f'TMA alignment error: {m}'

    torch.random.manual_seed(0)
    x = torch.randn(
        (num_batches, m, k), device='cuda', dtype=torch.bfloat16) / k
    y = torch.randn(
        (num_batches, n, k), device='cuda', dtype=torch.bfloat16) / k
    ref_out = torch.einsum('bmk,bnk->bmn', x, y)

    x_fp8, x_scale = (torch.empty_like(x, dtype=torch.float8_e4m3fn),
                      torch.empty((num_batches, m, k // 128),
                                  device='cuda',
                                  dtype=torch.float))
    y_fp8, y_scale = (torch.empty_like(y, dtype=torch.float8_e4m3fn),
                      torch.empty((num_batches, (n + 127) // 128, k // 128),
                                  device='cuda',
                                  dtype=torch.float))

    for i in range(num_batches):
        x_fp8[i], x_scale_i = torch.ops.trtllm.fp8_quantize_1x128(x[i])
        x_scale[i] = x_scale_i.view(x_scale[i].shape)
        y_fp8[i], y_scale[i] = per_block_cast_to_fp8(y[i])

    return x_fp8, x_scale, y_fp8, y_scale, ref_out


@pytest.mark.skipif(
    getSMVersion() != 90,
    reason="Op only supported on Hopper",
)
@pytest.mark.parametrize("ms", [[256, 256], [128, 64, 64], [16, 24, 48]])
@pytest.mark.parametrize("k, n", [(7168, 4096), (2048, 7168)])
def test_fp8_block_scaling_moe_gemm(ms, k, n):
    offset_cpu = [0] + list(itertools.accumulate(ms))
    offset = torch.tensor(offset_cpu, device='cuda', dtype=torch.int64)
    x_fp8, x_scale, y_fp8, y_scale, ref_out = construct_grouped(ms, k, n)
    x_fp8, x_scale = change_to_offset_layout(ms, x_fp8, x_scale)
    out = torch.ops.trtllm.fp8_block_scaling_moe_gemm(x_fp8, y_fp8, x_scale,
                                                      y_scale, offset)
    diff = calc_diff(out, ref_out)
    assert diff < 1e-3
    torch.testing.assert_close(out, ref_out, atol=1e-3, rtol=1e-3)


@pytest.mark.skipif(
    getSMVersion() != 90,
    reason="Op only supported on Hopper",
)
@pytest.mark.parametrize("batch_size, m", [(1, 1024), (2, 512), (4, 256)])
@pytest.mark.parametrize("k, n", [(7168, 4096), (2048, 7168)])
def test_fp8_block_scaling_bmm(batch_size, m, k, n):
    torch.random.manual_seed(0)
    x_fp8, x_scale, y_fp8, y_scale, ref_out = construct_batched(
        batch_size, m, k, n)
    output = torch.ops.trtllm.fp8_block_scaling_bmm(x_fp8, y_fp8, x_scale,
                                                    y_scale)
    diff = calc_diff(output, ref_out)
    assert diff < 1e-3
    torch.testing.assert_close(output, ref_out, atol=1e-3, rtol=1e-3)
