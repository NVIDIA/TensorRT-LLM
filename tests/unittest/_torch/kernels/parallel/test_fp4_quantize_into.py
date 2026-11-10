# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NVFP4 quantization into caller-owned buffers matches ``fp4_quantize`` byte for byte."""

import pytest
import torch
from utils.util import skip_pre_blackwell

pytestmark = skip_pre_blackwell


@pytest.mark.parametrize("rows,bucket,hidden", [(37, 64, 7168), (64, 64, 7168), (1, 8, 1024)])
def test_fp4_quantize_into_writes_what_fp4_quantize_returns(rows, bucket, hidden):
    """The rows land in the first rows of a bucket's buffers; the rest stays untouched."""
    torch.manual_seed(rows)
    x = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
    scale = torch.tensor([1.7], device="cuda")
    want, want_sf = torch.ops.trtllm.fp4_quantize(x, scale, 16, False, False)
    act = torch.full((bucket, hidden // 2), 9, dtype=torch.uint8, device="cuda")
    act_sf = torch.full((bucket, hidden // 16), 9, dtype=torch.uint8, device="cuda")
    torch.ops.trtllm.fp4_quantize_into(x, scale, act[:rows], act_sf[:rows], False)
    assert torch.equal(act[:rows], want)
    assert torch.equal(act_sf[:rows].view(-1), want_sf.view(-1))
    assert (act[rows:] == 9).all() and (act_sf[rows:] == 9).all()
    with pytest.raises(RuntimeError):  # a buffer short of the rows
        torch.ops.trtllm.fp4_quantize_into(x, scale, act[: rows - 1], act_sf[:rows], False)
