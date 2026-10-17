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


def make_operand(rows, cols, layout, device):
    # "unaligned" starts 8 bytes past a 16-byte boundary; "padded" slices rows
    # out of wider rows.
    if layout == "unaligned":
        return torch.randn(rows * cols + 4, dtype=torch.bfloat16,
                           device=device)[4:].view(rows, cols)
    if layout == "padded":
        return torch.randn(rows, cols + 64, dtype=torch.bfloat16,
                           device=device)[:, :cols]
    return torch.randn(rows, cols, dtype=torch.bfloat16, device=device)


@pytest.mark.parametrize(
    "num_tokens", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
@pytest.mark.parametrize("num_experts,hidden_size", [(256, 7168), (256, 6144),
                                                     (256, 4096), (288, 4096)])
@pytest.mark.parametrize("layout", ["packed", "unaligned", "padded"])
def test_router_gemm_run(num_tokens, num_experts, hidden_size, layout):
    torch.manual_seed(24)
    torch.cuda.manual_seed(24)

    device = torch.device("cuda")
    input = make_operand(num_tokens, hidden_size, layout, device)
    weight = make_operand(num_experts, hidden_size, layout, device)
    logits = torch.ops.trtllm.dsv3_router_gemm_op(input, weight.t(), None,
                                                  torch.float32)
    logits_ref = router_gemm_ref(input.float(),
                                 weight.t().float(), None, torch.float32)
    # BF16 products are exact in FP32; only the accumulation order differs.
    torch.testing.assert_close(logits, logits_ref, atol=1e-2, rtol=1e-3)
