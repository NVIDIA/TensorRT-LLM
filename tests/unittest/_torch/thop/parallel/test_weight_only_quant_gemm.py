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

import pytest
import torch
from _torch.helpers import calc_diff, calc_woq_tolerence


def weight_only_quant_gemm_reference(a, b, b_scales):
    a_dtype = a.dtype
    a = a.to(dtype=torch.float)
    b = b.to(dtype=torch.float)
    b_scales = b_scales.to(dtype=torch.float)
    # Do matmul
    ref = torch.matmul(a, b * b_scales)

    return ref.to(dtype=a_dtype)


@pytest.mark.parametrize(
    "k, n",
    [(7168, 2112), (1536, 24576), (512, 32768), (16384, 7168), (1024, 1024)],
)
@pytest.mark.parametrize(
    "m",
    [7, 64, 4096],
)
@pytest.mark.parametrize(
    "a_dtype",
    [torch.float16, torch.bfloat16],
)
@pytest.mark.parametrize(
    "b_dtype",
    [torch.int8, torch.quint4x2],
)
def test_weight_only_quant_gemm(a_dtype, b_dtype, m, k, n):
    import tensorrt_llm  # noqa: F401

    torch.random.manual_seed(0)

    # generate a, int4/int8 b, and scales
    a = torch.randn((m, k), dtype=a_dtype, device="cuda")
    b = torch.rand((k, n), dtype=a_dtype, device="cuda") * 2 - 1.0
    b, processed_b, b_scales = torch.ops.trtllm._symmetric_quantize_last_axis_of_batched_matrix(
        b.cpu(), b_dtype)
    if b_dtype == torch.quint4x2:
        b = torch.ops.trtllm.unpack_int4_packed_tensor_to_int8(b.cpu())

    output = torch.ops.trtllm.weight_only_quant_gemm(a, processed_b.cuda(),
                                                     b_dtype, b_scales.cuda(),
                                                     a_dtype)

    output_ref = weight_only_quant_gemm_reference(a, b.cuda(), b_scales.cuda())

    # check accuracy
    diff = calc_diff(output, output_ref)
    assert diff < 1e-3, f"Difference {diff} >= 1e-3"
    atol = calc_woq_tolerence(output_ref, b_dtype)
    torch.testing.assert_close(output_ref, output, atol=atol, rtol=1e-7)
