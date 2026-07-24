# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


@pytest.mark.parametrize("num_tokens", [1, 7, 64, 100, 8192])
@pytest.mark.parametrize("num_experts", [128, 256])
@pytest.mark.parametrize("hidden_size", [6144, 4096])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_moe_router_gemm(num_tokens, num_experts, hidden_size, dtype):
    torch.manual_seed(24)
    torch.cuda.manual_seed(24)

    device = torch.device("cuda")
    act = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
    # Router weight is stored in fp32 (routing-critical precision preserved).
    weight = torch.randn((num_experts, hidden_size), dtype=torch.float32, device=device)

    logits = torch.ops.trtllm.moe_router_gemm_op(act, weight, out_dtype=torch.float32)

    assert logits.shape == (num_tokens, num_experts)
    assert logits.dtype == torch.float32

    # Ground truth in fp64. The bf16/fp16 activation upcasts losslessly, so the
    # only difference from the kernel is the kernel's fp32 accumulation over the
    # hidden dim. An eager fp32 F.linear is not a valid reference here: when TF32
    # tensor cores are allowed it truncates inputs to a 10-bit mantissa, which is
    # less accurate than the kernel and flags spurious mismatches at M >= 7 (where
    # cuBLASLt selects the TF32 tensorop kernel rather than the SIMT fp32 SGEMM
    # used for tiny M).
    truth = torch.matmul(act.double(), weight.double().t())

    kernel_err = (logits.double() - truth).abs().max().item()

    # Tolerance covers fp32 accumulation over the hidden dim while staying far
    # below the value scale (std ~ sqrt(hidden_size)), so a real logic bug still
    # produces an out-of-tolerance error.
    torch.testing.assert_close(logits.double(), truth, rtol=1e-2, atol=1e-1)

    # The kernel must be at least as accurate as the eager cast plus linear,
    # documenting that routing precision is not downgraded.
    eager = torch.nn.functional.linear(act.to(torch.float32), weight)
    eager_err = (eager.double() - truth).abs().max().item()
    assert kernel_err <= eager_err + 1e-3, (
        f"router GEMM kernel ({kernel_err:.3e}) is less accurate than the eager "
        f"cast plus linear ({eager_err:.3e})"
    )
