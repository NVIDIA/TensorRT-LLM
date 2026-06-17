# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


def router_gemm_ref(input, weight, bias, dtype):
    logits_ref = torch.matmul(input, weight)
    return logits_ref


@pytest.mark.parametrize(
    "num_tokens", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
@pytest.mark.parametrize("num_experts", [256])
@pytest.mark.parametrize("hidden_size",
                         [7168, 6144, 4096])  # 4096 will enter fallback kernel
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_router_gemm_run(num_tokens, num_experts, hidden_size, dtype):
    torch.manual_seed(24)
    torch.cuda.manual_seed(24)

    device = torch.device("cuda")
    input = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
    weight = torch.randn((num_experts, hidden_size), dtype=dtype, device=device)
    bias = None
    logits = torch.ops.trtllm.dsv3_router_gemm_op(input, weight.t(), bias,
                                                  torch.float32)
    logtis_ref = router_gemm_ref(input.float(),
                                 weight.t().float(), bias, torch.float32)
    assert torch.allclose(logits, logtis_ref, rtol=5e-2)
