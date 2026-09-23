# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from utils.util import isSM100Family

pytestmark = pytest.mark.skipif(
    not isSM100Family(), reason="Requires Blackwell or Rubin quantization kernels"
)


def _assert_same_bytes(actual, expected):
    for actual_tensor, expected_tensor in zip(actual, expected):
        assert actual_tensor.shape == expected_tensor.shape
        assert actual_tensor.dtype == expected_tensor.dtype
        torch.testing.assert_close(
            actual_tensor.view(torch.uint8), expected_tensor.view(torch.uint8), atol=0, rtol=0
        )


def _input(rows):
    generator = torch.Generator(device="cuda").manual_seed(17)
    return torch.randn((rows, 7168), device="cuda", dtype=torch.bfloat16, generator=generator)


@pytest.mark.parametrize("rows", [129, 816, 817, 8096, 8192])
@pytest.mark.parametrize("swizzled", [False, True])
@pytest.mark.parametrize("reserved_sms", [0, 8])
def test_fp4_quantize_sm_budget_matches_default(rows, swizzled, reserved_sms):
    x = _input(rows)
    scale = torch.ones((1,), device="cuda", dtype=torch.float32)
    expected = torch.ops.trtllm.fp4_quantize(x, scale, 16, False, swizzled)
    actual = torch.ops.trtllm.fp4_quantize.sm_budget(
        x, scale, 16, False, swizzled, reserved_sms=reserved_sms
    )
    _assert_same_bytes(actual, expected)


@pytest.mark.parametrize("rows", [0, 129, 816, 817, 8096, 8192])
@pytest.mark.parametrize("reserved_sms", [0, 8])
def test_shared_fp8_quantize_sm_budget_matches_default(rows, reserved_sms):
    x = _input(rows)
    expected = torch.ops.trtllm.fp8_quantize_1x128_packed_ue8m0(x)
    actual = torch.ops.trtllm.fp8_quantize_1x128_packed_ue8m0.sm_budget(
        x, reserved_sms=reserved_sms
    )
    _assert_same_bytes(actual, expected)


@pytest.mark.parametrize("op", ["fp4", "fp8"])
@pytest.mark.parametrize("budget", ["negative", "all_sms"])
def test_quantize_sm_budget_rejects_invalid_budget(op, budget):
    x = _input(129)
    reserved_sms = (
        -1
        if budget == "negative"
        else torch.cuda.get_device_properties(x.device).multi_processor_count
    )
    with pytest.raises(RuntimeError, match="reserved_sms"):
        if op == "fp4":
            scale = torch.ones((1,), device="cuda", dtype=torch.float32)
            torch.ops.trtllm.fp4_quantize.sm_budget(
                x, scale, 16, False, False, reserved_sms=reserved_sms
            )
        else:
            torch.ops.trtllm.fp8_quantize_1x128_packed_ue8m0.sm_budget(x, reserved_sms=reserved_sms)


def test_shared_fp8_quantize_sm_budget_rejects_legacy_layout():
    with pytest.raises(RuntimeError, match="R128c4"):
        torch.ops.trtllm.fp8_quantize_1x128_packed_ue8m0.sm_budget(
            _input(129), False, reserved_sms=8
        )


@pytest.mark.parametrize("op", ["fp4", "fp8"])
@pytest.mark.parametrize("reserved_sms", [0, 8])
def test_quantize_sm_budget_fake_and_cuda_graph(op, reserved_sms):
    x = _input(129)
    if op == "fp4":
        scale = torch.ones((1,), device="cuda", dtype=torch.float32)
        quantize = torch.ops.trtllm.fp4_quantize.sm_budget
        args = (x, scale, 16, False, False)
    else:
        quantize = torch.ops.trtllm.fp8_quantize_1x128_packed_ue8m0.sm_budget
        args = (x, True)
    kwargs = {"reserved_sms": reserved_sms}
    torch.library.opcheck(quantize, args, kwargs, test_utils=("test_schema", "test_faketensor"))
    expected = quantize(*args, **kwargs)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = quantize(*args, **kwargs)
    graph.replay()
    _assert_same_bytes(actual, expected)
