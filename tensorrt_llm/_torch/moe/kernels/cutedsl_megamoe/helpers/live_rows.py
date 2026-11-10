# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""The leading live rows of a launch under the forward's live-row word (layout: tensorrt_llm/_torch/moe/live_rows.py)."""

import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import AddressSpace
from cutlass.cutlass_dsl import Int32

from tensorrt_llm._torch.moe.live_rows import MAX_REQUESTS as LIVE_ROW_REQUEST_MASK
from tensorrt_llm._torch.moe.live_rows import REQUEST_BITS as LIVE_ROW_REQUEST_BITS


@cute.jit
def live_row_count(live_row_word: cute.Pointer, rows: Int32) -> Int32:
    """Of ``rows`` rows: none when P is 0, all when R >= P or P does not divide them, else rows / P * R."""
    word = cute.arch.load(cute.make_ptr(Int32, live_row_word.toint(), AddressSpace.gmem, assumed_align=4), Int32)
    real = word & Int32(LIVE_ROW_REQUEST_MASK)
    requests = (word >> Int32(LIVE_ROW_REQUEST_BITS)) & Int32(LIVE_ROW_REQUEST_MASK)
    rows_per_request = rows // cutlass.max(requests, Int32(1))
    count = rows
    if requests == Int32(0):
        count = Int32(0)
    elif (real < requests) & (rows_per_request * requests == rows):
        count = rows_per_request * real
    return count
