# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The router of an unfused MoE past the tensor-core router's 48 rows: the fp32 router (``m3_fp32_linear``) is bit for
bit the built-in gate's generation-step arithmetic there, ``F.linear(h.to(float32), w)`` (the gate's TF32 fixed tile
is off in generation steps, its tensor-core router covers 9..16 rows, and below 2048 rows its fp32 cast is the plain
``.to(torch.float32)``), so a 49..192-row decode step routes as the agg canon does."""

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__tp2 import (  # noqa: E501
    modeling,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__tp2.ops.gemm.m3_fp32_linear import (  # noqa: E501
    m3_fp32_linear,
)
from tensorrt_llm._utils import get_sm_version

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() != 107, reason="SM107 (Rubin) only"
)

HIDDEN, EXPERTS = 6144, 128


@pytest.mark.parametrize("rows", [49, 64, 128, 192])
def test_the_fp32_router_is_the_gates_generation_arithmetic(rows: int) -> None:
    assert not modeling.ROUTER_TC_GEN.covers(rows), "the tensor-core router stops at 48 rows"
    g = torch.Generator(device="cuda").manual_seed(rows)
    h = torch.randn(rows, HIDDEN, generator=g, device="cuda").to(torch.bfloat16)
    w = torch.randn(EXPERTS, HIDDEN, generator=g, device="cuda") / HIDDEN**0.5
    got = m3_fp32_linear(h, w)
    want = F.linear(h.to(torch.float32), w)
    assert got.dtype == torch.float32
    assert torch.equal(got.view(torch.int32), want.view(torch.int32))
