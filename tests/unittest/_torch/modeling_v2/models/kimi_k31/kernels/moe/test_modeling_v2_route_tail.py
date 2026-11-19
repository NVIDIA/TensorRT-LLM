# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the family's routing-tail kernel (``route_tail.k31_route_topk``).

The kernel's claim is bitwise: the experts, their order and the BF16 weights equal
``noaux_tc_op`` (one expert group, top-16 of 896) followed by the BF16 cast of its FP32
weights, which is what the routed-expert scheduler hands a TRTLLM-Gen backend. So this
compares against the engine op itself, exactly, at the smallest and the largest launch.
"""

import pytest
import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

if not IS_CUTLASS_DSL_AVAILABLE:
    pytest.skip("the routing tail is a CuTe DSL kernel", allow_module_level=True)

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kernels.moe import route_tail  # noqa: E402

assert torch.cuda.is_available(), "the routing tail requires a CUDA device"
if torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)):
    pytest.skip("the routing tail runs on SM100 / SM103 / SM107", allow_module_level=True)

E, K = route_tail.NUM_EXPERTS, route_tail.TOP_K


def _inputs(rows: int, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    logits = torch.randn(rows, E, generator=g, device="cuda", dtype=torch.float32)
    bias = torch.randn(E, generator=g, device="cuda", dtype=torch.float32) * 0.1
    return logits, bias


def _engine(logits, bias, scale):
    """The path the kernel replaces: noaux_tc_op, then the BF16 cast of its weights."""
    weights, ids = torch.ops.trtllm.noaux_tc_op(logits, bias, 1, 1, K, scale)
    return ids, weights.to(torch.bfloat16)


def _assert_bitwise(got, want):
    ids, weights = got
    ref_ids, ref_weights = want
    assert ids.dtype is torch.int32 and weights.dtype is torch.bfloat16
    assert ids.shape == ref_ids.shape == weights.shape
    assert torch.equal(ids, ref_ids), "selected experts (or their order) differ from noaux_tc_op"
    assert torch.equal(weights.view(torch.int16), ref_weights.view(torch.int16)), (
        "BF16 weights differ from bf16(noaux_tc_op weights)"
    )


# 65, past the boundary where the attention-DP MoE switches from its fused routing to this kernel;
# 2048, the largest launch.
ROWS = (65, 2048)


@pytest.mark.parametrize("rows", ROWS)
def test_bitwise_vs_noaux_tc(rows: int) -> None:
    logits, bias = _inputs(rows, seed=rows)
    got = route_tail.k31_route_topk(logits, bias, 2.5)
    _assert_bitwise(got, _engine(logits, bias, 2.5))
