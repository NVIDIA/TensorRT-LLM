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
# 256 experts: DeepSeek-V3/V4, GLM-5 (scalar FFMA kernel, all three hidden sizes).
# 288 experts: GLM-5.3-Flash (scalar FFMA kernel, hidden 4096).
# 896 experts: Kimi K3 (tensor-core kernel for hidden 7168; cuBLAS fallback otherwise).
@pytest.mark.parametrize("num_experts,hidden_size", [(256, 7168), (256, 6144),
                                                     (256, 4096), (288, 4096),
                                                     (896, 7168), (896, 6144),
                                                     (896, 4096)])
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
    # The router output only matters through the top-k selection that follows it.
    top_k = 16 if num_experts == 896 else 8
    assert torch.equal(
        torch.topk(logits, top_k, dim=-1).indices.sort(dim=-1).values,
        torch.topk(logits_ref, top_k, dim=-1).indices.sort(dim=-1).values)


@pytest.mark.skipif(not torch.cuda.is_available()
                    or torch.cuda.get_device_capability() != (10, 7),
                    reason="requires SM107")
@pytest.mark.parametrize("num_tokens", [1, 8, 16])
def test_router_latent_gemm_matches_separate_ops(num_tokens: int) -> None:
    from types import SimpleNamespace

    from tensorrt_llm._torch.models.modeling_kimi_linear import KimiK3MoEGate

    torch.manual_seed(24)
    torch.cuda.manual_seed(24)
    config = SimpleNamespace(hidden_size=7168,
                             num_experts=896,
                             num_experts_per_token=16,
                             routed_scaling_factor=2.5,
                             moe_router_activation_func="sigmoid",
                             moe_renormalize=True)
    gate = KimiK3MoEGate(config,
                         logits_gemm_dtype=torch.bfloat16,
                         device=torch.device("cuda"))
    down = torch.nn.Linear(config.hidden_size,
                           3584,
                           bias=False,
                           device="cuda",
                           dtype=torch.bfloat16)
    hidden = torch.randn(num_tokens,
                         config.hidden_size,
                         dtype=torch.bfloat16,
                         device="cuda")
    with torch.no_grad():
        gate.weight.normal_()
        down.weight.normal_()
        expected_logits = gate.compute_logits(hidden)
        expected_projection = down(hidden)
        precise_projection = (hidden.double() @ down.weight.double().t()).to(
            torch.bfloat16)
        logits, projection = torch.ops.trtllm.dsv3_router_latent_gemm_op(
            hidden, gate.weight, down.weight)

    torch.testing.assert_close(logits, expected_logits, rtol=0, atol=0)
    assert torch.isfinite(logits).all()
    assert projection.shape == expected_projection.shape
    assert projection.dtype == expected_projection.dtype == torch.bfloat16
    # Unit-normal K=7168 dot products need an FP32 accumulation allowance near
    # zero; one BF16 step alone can reject even the correctly rounded result.
    accumulation_atol = 1e-3
    for actual, reference in ((projection, expected_projection),
                              (projection, precise_projection),
                              (expected_projection, precise_projection)):
        lower = torch.nextafter(reference,
                                torch.full_like(reference, -torch.inf)).float()
        upper = torch.nextafter(reference,
                                torch.full_like(reference, torch.inf)).float()
        within_tolerance = (torch.isfinite(actual)
                            & torch.isfinite(reference)
                            & (actual.float() >= lower - accumulation_atol)
                            & (actual.float() <= upper + accumulation_atol))
        mismatches = (~within_tolerance).sum().item()
        max_abs = (actual.float() - reference.float()).abs().max().item()
        assert mismatches == 0, (
            f"{mismatches}/{actual.numel()} projection values exceed one BF16 "
            f"step + {accumulation_atol} or are nonfinite; max_abs={max_abs}")
