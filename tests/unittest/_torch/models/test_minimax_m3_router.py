# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 prefill router: the Triton path equals the cuBLAS router GEMM bit for bit."""

import pytest
import torch

from tensorrt_llm._torch.models import minimax_m3_router
from tensorrt_llm._torch.models.minimax_m3_router import (
    router_split_applies,
    router_split_logits,
)
from tensorrt_llm._torch.models.modeling_minimaxm3 import MiniMaxM3Gate, split_router_weight_bf16
from tensorrt_llm._utils import get_sm_version

HIDDEN, EXPERTS = 6144, 128


def _stock_logits(x, split_weight):
    terms = split_weight.shape[1] // x.shape[1]
    return torch.ops.trtllm.cublas_mm(
        torch.cat([x] * terms, dim=-1), split_weight.t(), bias=None, out_dtype=torch.float32
    )


@pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() not in minimax_m3_router._SUPPORTED_SM,
    reason="bitwise equality is verified on SM103 only",
)
@pytest.mark.parametrize("rows", [8625, 9000, 12288, 16266, 16322, 16384])
def test_router_split_logits_bitwise(rows):
    torch.manual_seed(rows)
    weight = torch.randn(EXPERTS, HIDDEN, device="cuda") * 0.02
    split_weight = split_router_weight_bf16(weight)
    x = (torch.randn(rows, HIDDEN, device="cuda") * 1.5).to(torch.bfloat16)
    assert router_split_applies(x, split_weight, HIDDEN)
    assert torch.equal(router_split_logits(x, split_weight, HIDDEN), _stock_logits(x, split_weight))

    gate = MiniMaxM3Gate(HIDDEN, EXPERTS, top_k=4, routed_scaling_factor=2.0).cuda()
    gate.load_weights([{"weight": weight}])
    assert torch.equal(gate(x), _stock_logits(x, split_weight))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_router_split_skips_rows_outside_range(monkeypatch):
    monkeypatch.setattr(minimax_m3_router, "ROUTER_TRITON_ROWS", ((100, 200),))
    split_weight = torch.zeros(EXPERTS, 3 * HIDDEN, dtype=torch.bfloat16, device="cuda")
    x = torch.zeros(99, HIDDEN, dtype=torch.bfloat16, device="cuda")
    assert not router_split_applies(x, split_weight, HIDDEN)
    monkeypatch.setattr(minimax_m3_router, "ROUTER_TRITON_ROWS", ())
    assert not router_split_applies(x[:1].expand(150, HIDDEN), split_weight, HIDDEN)
