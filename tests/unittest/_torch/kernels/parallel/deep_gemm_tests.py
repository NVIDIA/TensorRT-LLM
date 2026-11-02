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
from _torch.helpers import calc_diff, per_block_cast_to_fp8
from utils.util import getSMVersion


@pytest.mark.skipif(
    getSMVersion() != 90,
    reason="The test is for Hopper only. Current SM is %d." % getSMVersion(),
)
@pytest.mark.parametrize(
    "k, n",
    [(7168, 2112), (2048, 7168)],
)
@pytest.mark.parametrize(
    "m",
    [7, 16, 64, 128, 4096],
)
def test_fp8_block_scale_gemm(m, k, n):
    torch.random.manual_seed(0)
    a = torch.randn((m, k), device='cuda', dtype=torch.bfloat16) / k
    b = torch.randn((n, k), device='cuda', dtype=torch.bfloat16) / k

    act_a_fp8, act_a_sf = torch.ops.trtllm.fp8_quantize_1x128(a)
    act_b_fp8, act_b_sf = per_block_cast_to_fp8(b)

    output = torch.ops.trtllm.fp8_block_scaling_gemm(act_a_fp8, act_b_fp8,
                                                     act_a_sf, act_b_sf)

    output_expected = a @ b.t()
    diff = calc_diff(output, output_expected)
    assert diff < 1e-3
    torch.testing.assert_close(output, output_expected, atol=1e-3, rtol=1e-3)


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

    # Need to keep the same as the one in cpp/include/tensorrt_llm/deep_gemm/scheduler.cuh
    def compute_padded_offset(offset, idx_problem, alignment=32):
        return (offset + idx_problem * (alignment - 1)) // alignment * alignment

    offset = 0
    for i in range(num_problems):
        ms[i]
        x_list.append(x_fp8[m_acc[i]:m_acc[i + 1]])
        offset_next = compute_padded_offset(m_acc[i + 1], i + 1)
        size_padded = (offset_next - offset) - (m_acc[i + 1] - m_acc[i])
        x_scale_padded = torch.cat([
            x_scale[m_acc[i]:m_acc[i + 1]],
            torch.zeros(
                [size_padded, *x_scale.shape[1:]],
                dtype=x_scale.dtype,
                device=x_scale.device,
            ),
        ])
        x_scale_list.append(x_scale_padded)
        offset = offset_next

    shape_m_total = m_acc[-1]
    ret_x = torch.cat(x_list)
    ret_x_scale = torch.cat(x_scale_list)
    ret_x_scale = ret_x_scale.t().contiguous()
    pad_target = compute_padded_offset(shape_m_total, num_problems)
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
    reason="Op only supported on Hopper, current SM is %d." % getSMVersion(),
)
@pytest.mark.parametrize(
    "ms", [[256, 256], [128, 64, 64], [16, 24, 48], [4, 8, 16, 32]])
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
    reason="Op only supported on Hopper, current SM is %d." % getSMVersion(),
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
