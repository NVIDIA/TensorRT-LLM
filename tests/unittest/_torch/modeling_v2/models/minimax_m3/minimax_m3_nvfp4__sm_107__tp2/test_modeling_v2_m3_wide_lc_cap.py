# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The wide target's MXFP8 o_proj hand-off, and so its long-context sparse decode, stops at one 128-row scale tile.

LC2 is certified as the MXFP8 hand-off step, so the hand-off's row cap is LC2's row cap. A wide decode step past it
(33..48 requests x 4 verify rows) takes the Triton producer + merge and the o_proj quantizes its BF16 input itself
(attention/test_modeling_v2_m3_sparse_decode.py: test_plain_route_is_bitwise_the_stock_decode).
"""

import pytest
import torch

import tensorrt_llm  # noqa: F401  (registers torch.ops.trtllm)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__tp2 import (
    modeling as dispatch,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__tp2.ops import (
    triton_sparse_decode as tsd,
)


def test_the_hand_off_cap_is_one_scale_tile():
    """The MXFP8 hand-off (and so the long-context sparse decode) serves up to one 128-row scale tile; the target's
    129..192-row steps take the Triton producer + merge and the o_proj quantizes its BF16 input itself."""
    assert tsd.MergeMxfp8.MAX_ROWS == 128
    assert dispatch.WIDE_MAX_TOKENS > tsd.MergeMxfp8.MAX_ROWS


def test_lc2_certifies_only_the_hand_off_step():
    assert dispatch.SPARSE_DECODE_LC.covers("mxfp8")
    assert not dispatch.SPARSE_DECODE_LC.covers("plain")


@pytest.mark.parametrize("rows", [1, 4, 127, 128])
def test_the_hand_off_is_the_steps_own_scale_tiles(rows):
    """The published scales are the stock MXFP8 quantization's size for the step (the o_proj GEMM checks the swizzled
    scales against the step's padded rows): one 128-row tile for every step the hand-off serves."""
    width = 4096
    buffers = tsd.MERGE_MXFP8.buffers(width, torch.device("cuda"))
    q8, sf = tsd.MergeMxfp8.handoff(buffers, rows)
    _, stock_sf = torch.ops.trtllm.mxfp8_quantize(
        torch.randn(rows, width, dtype=torch.bfloat16, device="cuda"), True
    )
    assert q8.shape == (rows, width)
    assert sf.numel() == stock_sf.numel()
    assert sf.data_ptr() == buffers[1].data_ptr()
