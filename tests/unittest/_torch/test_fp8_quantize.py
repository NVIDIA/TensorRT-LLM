# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import math

import pytest
import torch
from utils.util import getSMVersion


def _dequant_fp8(input, scale, transpose_scale, block_m, block_n):
    input = input.to(torch.float)
    scale = scale.to(torch.float)
    if transpose_scale:
        scale = scale.t()
    output = torch.zeros_like(input)
    m, n = input.shape
    m_tile = 128 if block_m else 1
    n_tile = 128 if block_n else 1

    if m_tile == 1:
        assert n % 16 == 0, "n must be divisible by 16"
        total_blocks = math.ceil(n / 128)
        for block in range(total_blocks):
            # Calculate start position in 2D array
            start_col = block * 128
            end_col = min(start_col + 128, n)
            output[:, start_col:
                   end_col] = input[:, start_col:end_col] * scale[:,
                                                                  block].view(
                                                                      -1, 1)

    elif n_tile == 1:
        assert m % 16 == 0, "m must be divisible by 16"
        total_blocks = math.ceil(m / 128)
        for block in range(total_blocks):
            # Calculate start position in 2D array
            start_row = block * 128
            end_row = min(start_row + 128, m)
            output[start_row:end_row, :] = input[start_row:end_row, :] * scale[
                block, :]
    else:
        assert n % 16 == 0, "n must be divisible by 16"
        assert m % 16 == 0, "m must be divisible by 16"
        n_blocks = math.ceil(n / 128)
        m_blocks = math.ceil(m / 128)
        for i in range(n_blocks):
            for j in range(m_blocks):
                start_row = j * 128
                end_row = min(start_row + 128, m)
                start_col = i * 128
                end_col = min(start_col + 128, n)
                output[start_row:end_row,
                       start_col:end_col] = input[start_row:end_row,
                                                  start_col:end_col] * scale[j,
                                                                             i]
    return output


@pytest.mark.skipif(
    getSMVersion() != 100 and getSMVersion() != 90,
    reason="Only test on Blackwell and Hopper",
)
@pytest.mark.parametrize("k", [576, 256, 32])
@pytest.mark.parametrize(
    "m",
    [4, 16, 256],
)
@pytest.mark.parametrize(
    "dtype",
    [torch.bfloat16],
)
def test_fp8_quantize_blackwell(dtype, m, k):
    torch.random.manual_seed(0)
    # TODO: make sure there is no padding for now
    assert m % 4 == 0, "Disable padding for now"
    a = torch.randn((m, k), device='cuda', dtype=dtype)
    fp8_a, fp8_a_scale = torch.ops.trtllm.fp8_quantize_1x128(a)
    fp8_a_scale = fp8_a_scale.view(-1,
                                   fp8_a.shape[0])  # transpose the scale view
    a_dequant = _dequant_fp8(fp8_a, fp8_a_scale, True, False, True)

    torch.testing.assert_close(a_dequant.cpu().to(torch.float32),
                               a.cpu().to(torch.float32),
                               atol=1e-1,
                               rtol=1e-1)
