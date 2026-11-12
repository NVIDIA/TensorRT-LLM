# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``fp4_quantize.into``: NVFP4 quantization into caller-owned buffers matches ``fp4_quantize`` byte for byte."""

import pytest
import torch
from utils.util import skip_pre_blackwell

pytestmark = skip_pre_blackwell


@pytest.mark.parametrize("reserved_sms", [0, 8])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("rows,bucket,hidden", [(37, 64, 7168), (64, 64, 7168), (1, 8, 1024)])
def test_fp4_quantize_into_writes_what_fp4_quantize_returns(
    rows, bucket, hidden, dtype, reserved_sms
):
    """The rows land in the first rows of a bucket's buffers, with or without SMs left to other
    work; the rest stays untouched."""
    torch.manual_seed(rows)
    x = torch.randn(rows, hidden, device="cuda", dtype=dtype)
    scale = torch.tensor([1.7], device="cuda")
    want, want_sf = torch.ops.trtllm.fp4_quantize(x, scale, 16, False, False)
    act = torch.full((bucket, hidden // 2), 9, dtype=torch.uint8, device="cuda")
    act_sf = torch.full((bucket, hidden // 16), 9, dtype=torch.uint8, device="cuda")
    torch.ops.trtllm.fp4_quantize.into(x, scale, act[:rows], act_sf[:rows], False, reserved_sms)
    assert torch.equal(act[:rows], want)
    assert torch.equal(act_sf[:rows].view(-1), want_sf.view(-1))
    assert (act[rows:] == 9).all() and (act_sf[rows:] == 9).all()
    with pytest.raises(RuntimeError):  # a buffer short of the rows
        torch.ops.trtllm.fp4_quantize.into(x, scale, act[: rows - 1], act_sf[:rows], False)


def test_fp4_quantize_into_refuses_buffers_it_would_misread():
    """A misaligned input and outputs of the right size but the wrong layout are refused."""
    rows, hidden = 4, 1024
    scale = torch.tensor([1.7], device="cuda")
    x = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
    act = torch.empty((rows, hidden // 2), dtype=torch.uint8, device="cuda")
    act_sf = torch.empty((rows, hidden // 16), dtype=torch.uint8, device="cuda")
    # One element off: contiguous, but not on the 16-byte boundary of the kernel's vector loads.
    flat = torch.randn(rows * hidden + 1, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="16-byte alignment"):
        torch.ops.trtllm.fp4_quantize.into(flat[1:].view(rows, hidden), scale, act, act_sf, False)
    # The same bytes as [M, K / 2], transposed.
    with pytest.raises(RuntimeError, match=r"out must be"):
        torch.ops.trtllm.fp4_quantize.into(x, scale, act.view(hidden // 2, rows), act_sf, False)
    # Rows padded past K / 16: large enough, but every row after the first would land at the wrong offset.
    padded_sf = torch.empty((rows, hidden // 16 + 4), dtype=torch.uint8, device="cuda")
    with pytest.raises(RuntimeError, match=r"out_sf must be"):
        torch.ops.trtllm.fp4_quantize.into(x, scale, act, padded_sf, False)
    torch.ops.trtllm.fp4_quantize.into(x, scale, act, act_sf, False)


@pytest.mark.parametrize("rows,hidden", [(37, 1056), (130, 7168)])
def test_fp4_quantize_into_swizzled_writes_what_fp4_quantize_returns(rows, hidden):
    """The swizzled layout fills the padded 128 x 4 block as ``fp4_quantize`` does, and no more."""
    torch.manual_seed(rows)
    x = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
    scale = torch.tensor([1.7], device="cuda")
    want, want_sf = torch.ops.trtllm.fp4_quantize(x, scale, 16, False, True)
    act = torch.empty((rows, hidden // 2), dtype=torch.uint8, device="cuda")
    tail = 64
    act_sf = torch.full((want_sf.numel() + tail,), 9, dtype=torch.uint8, device="cuda")
    torch.ops.trtllm.fp4_quantize.into(x, scale, act, act_sf, True)
    assert torch.equal(act, want)
    assert torch.equal(act_sf[: want_sf.numel()], want_sf.view(torch.uint8))
    assert (act_sf[want_sf.numel() :] == 9).all()
    with pytest.raises(RuntimeError, match=r"out_sf must be"):
        torch.ops.trtllm.fp4_quantize.into(x, scale, act, act_sf[: want_sf.numel() - 1], True)


def test_fp4_quantize_into_without_rows_writes_nothing():
    hidden = 1024
    x = torch.empty((0, hidden), device="cuda", dtype=torch.bfloat16)
    scale = torch.tensor([1.7], device="cuda")
    act = torch.empty((0, hidden // 2), dtype=torch.uint8, device="cuda")
    act_sf = torch.empty((0, hidden // 16), dtype=torch.uint8, device="cuda")
    torch.ops.trtllm.fp4_quantize.into(x, scale, act, act_sf, False)


@pytest.mark.parametrize("swizzled", [False, True])
def test_fp4_quantize_into_schema_and_fake(swizzled):
    rows, hidden = 5, 1024
    x = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
    scale = torch.tensor([1.7], device="cuda")
    _, want_sf = torch.ops.trtllm.fp4_quantize(x, scale, 16, False, swizzled)
    act = torch.empty((rows, hidden // 2), dtype=torch.uint8, device="cuda")
    act_sf = torch.empty(
        want_sf.numel() if swizzled else (rows, hidden // 16), dtype=torch.uint8, device="cuda"
    )
    torch.library.opcheck(
        torch.ops.trtllm.fp4_quantize.into,
        (x, scale, act, act_sf, swizzled),
        test_utils=("test_schema", "test_faketensor"),
    )
