# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bit equality of the Kimi K3 MoE input glue at context and generation row counts.

- ``router_input_fp32``: the FP32 copy of the router input (vectorized kernel from 512 rows).
- ``KimiK3MoEGate.compute_logits``: the FP32 router logits.
"""

import dataclasses

import pytest
import torch

from tensorrt_llm._torch.models import modeling_kimi_linear as mkl
from tensorrt_llm._torch.modules.kimi_k3_router_input import router_input_fp32

HIDDEN = 7168
# Context rows: one MoE chunk and a whole 8192-token context iteration of a rank.
CTX_ROWS = [2048, 8192]
# Generation rows: decode batches.
GEN_ROWS = [1, 64, 200]

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int32) if t.dtype == torch.float32 else t.view(torch.int16)


@CUDA
@pytest.mark.parametrize("rows", GEN_ROWS + [511, 512, 513] + CTX_ROWS)
def test_router_input_fp32_is_exact(rows):
    torch.manual_seed(rows)
    x = (torch.randn(rows, HIDDEN, device="cuda") * 3).to(torch.bfloat16)
    specials = torch.tensor(
        [0.0, -0.0, float("inf"), -float("inf"), 9.2e-41, -3.0e38, float("nan")],
        dtype=torch.bfloat16,
        device="cuda",
    )
    x[0, : specials.numel()] = specials
    out = router_input_fp32(x)
    ref = x.to(torch.float32)
    assert out.dtype == torch.float32 and out.shape == ref.shape
    nan = torch.isnan(ref)
    assert torch.equal(torch.isnan(out), nan)
    assert torch.equal(_bits(out)[~nan], _bits(ref)[~nan])


@dataclasses.dataclass
class _K3GateConfig:
    hidden_size: int = HIDDEN
    num_experts: int = 896
    num_experts_per_token: int = 16
    moe_renormalize: bool = True
    moe_router_activation_func: str = "sigmoid"
    routed_scaling_factor: float = 1.0
    num_expert_group: int = 1
    topk_group: int = 1


@CUDA
@pytest.mark.parametrize("rows", GEN_ROWS + CTX_ROWS)
def test_router_logits_unchanged(rows):
    torch.manual_seed(0)
    gate = mkl.KimiK3MoEGate(_K3GateConfig()).cuda()
    with torch.no_grad():
        gate.weight.normal_(0.0, 0.02)
    x = torch.randn(rows, HIDDEN, device="cuda").to(torch.bfloat16)
    with torch.no_grad():
        logits = gate.compute_logits(x)
        ref = torch.nn.functional.linear(x.type(torch.float32), gate.weight.type(torch.float32))
    assert torch.equal(_bits(logits), _bits(ref))
