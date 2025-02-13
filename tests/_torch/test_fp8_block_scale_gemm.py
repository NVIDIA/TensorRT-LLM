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
import os
import sys

import pytest
import torch
from utils.util import getSMVersion

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


@pytest.mark.skipif(
    getSMVersion() < 90,
    reason="Op only supported on Hopper and above",
)
@pytest.mark.parametrize("k_n", [(8192, 1024), (128, 256), (16, 32)])
@pytest.mark.parametrize(
    "m",
    [1024, 100, 13],
)
@pytest.mark.parametrize(
    "dtype",
    # TODO: add FP8 tests when we add a quantize op as well.
    [torch.bfloat16],
)
def test_fp8_block_scale_gemm(dtype, m, k_n):
    torch.random.manual_seed(0)
    k, n = k_n
    a = torch.randn((m, k), device='cuda', dtype=dtype) / k
    b = torch.randn((n, k), device='cuda', dtype=dtype) / k

    output = torch.ops.trtllm.fp8_block_scaling_gemm(a, b, None, None)

    output_expected = a @ b.t()

    torch.testing.assert_allclose(output, output_expected, atol=1e-1, rtol=1e-2)
