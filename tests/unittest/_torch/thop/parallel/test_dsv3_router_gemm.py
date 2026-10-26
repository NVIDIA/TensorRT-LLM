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
@pytest.mark.parametrize("num_experts,hidden_size", [(256, 7168), (256, 6144),
                                                     (256, 4096), (384, 5120)])
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


@pytest.mark.skipif(not torch.cuda.is_available()
                    or torch.cuda.get_device_capability() not in ((10, 0),
                                                                  (10, 3)),
                    reason="The small-M kernel runs on SM100/SM103")
@pytest.mark.parametrize("m", [1, 6, 12, 64])
@pytest.mark.parametrize("num_experts,hidden_size", [(384, 5120), (128, 5120),
                                                     (256, 4096)])
def test_router_gemm_small_m_matches_reference(m, num_experts, hidden_size):
    """The small-M tactic equals the exact product; the race may pick either kernel."""
    from tensorrt_llm._torch.autotuner import AutoTuner, autotune
    from tensorrt_llm._torch.custom_ops.torch_custom_ops import (
        RouterGemmRunner, router_gemm_autotuned)

    torch.manual_seed(0)
    x = torch.randn((m, hidden_size), dtype=torch.bfloat16, device="cuda")
    weight = torch.randn((num_experts, hidden_size),
                         dtype=torch.bfloat16,
                         device="cuda")
    reference = (x.double() @ weight.double().t()).float()

    runner = RouterGemmRunner()
    assert runner.small_m_applies([x, weight])
    torch.testing.assert_close(runner([x, weight], tactic=0),
                               reference,
                               rtol=1e-3,
                               atol=1e-2)

    AutoTuner.get().clear_cache()
    with autotune():
        tuned = router_gemm_autotuned(x, weight)
    served = router_gemm_autotuned(x, weight)
    torch.testing.assert_close(served, tuned, rtol=0, atol=0)
    torch.testing.assert_close(served, reference, rtol=1e-3, atol=1e-2)
