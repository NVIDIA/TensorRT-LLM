# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The dep2 routed experts (the CuTe-DSL grouped GEMMs) take fp32 routing weights, as stock's CuTe-DSL grouped GEMMs do
(`routing_scales_dtype=torch.float32`; `try_fused_route` routes in fp32). Weights rounded to bf16 moved every row's
layer-0 FFN output by ~2e-3 (relative) against stock, which later layers amplified (ctx17). Host-only: the real
`_route_quant` around stub ops."""

import importlib
from types import SimpleNamespace as NS

import pytest
import torch
from torch import nn

modeling = importlib.import_module(
    "tensorrt_llm._torch.modeling_v2.models.deepseek_v41.flash_nvfp4__sm_103__dep2.modeling"
)


def _moe():
    moe = object.__new__(modeling._Moe)
    nn.Module.__init__(moe)
    moe.cfg, moe.topk = NS(routed_scaling_factor=1.5), 6
    moe.weights_dtype = torch.float32
    moe.gate_weight, moe.gate_bias, moe.g1 = torch.zeros(384, 8), torch.zeros(384), torch.ones(1)
    return moe


@pytest.fixture
def calls(monkeypatch):
    seen = []
    monkeypatch.setattr(
        modeling, "cublas_mm", lambda x, w, b, dt: torch.zeros(x.shape[0], w.shape[1], dtype=dt)
    )
    monkeypatch.setattr(
        modeling, "dsv4_route", lambda *a: seen.append(("route", a[-1])) or ("ids", "w")
    )
    monkeypatch.setattr(
        modeling, "dsv4_route_nvfp4_quant", lambda *a: seen.append(("route_quant", a[-1])) or "rq"
    )
    return seen


def test_routing_weights_are_fp32(calls):
    moe, x = _moe(), torch.zeros(3, 8, dtype=torch.bfloat16)
    moe._route_quant(x)
    moe._route_quant(x, pre=("fp4", "sf"))
    assert calls == [("route_quant", torch.float32), ("route", torch.float32)]


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the constructor creates the overlap's CUDA events"
)
def test_the_constructor_selects_fp32_routing_weights():
    """The value `_route_quant` forwards is the one `_Moe.__init__` sets (the stub above only proves the forwarding)."""
    cfg = NS(num_experts_per_tok=6, n_routed_experts=8, moe_intermediate_size=256, hidden_size=512,
             routed_scaling_factor=1.5)  # fmt: skip
    mapping = NS(
        enable_attention_dp=True,
        world_size=2,
        moe_tp_size=1,
        moe_tp_rank=0,
        moe_ep_size=2,
        moe_ep_rank=0,
    )
    assert modeling._Moe(cfg, mapping).weights_dtype == torch.float32
