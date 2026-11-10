# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""L2 prefetch of the expert weights the FC12 kernel will stream first.

The weights do not depend on the routed tokens, only on which (expert, output
block) tiles the scheduler hands out first, and that is known as soon as the
per-expert sizes are. Until the payload lands the kernel cannot issue a single
MMA, so HBM sits idle for the whole communication latency; warming L2 with the
first claim waves turns that idle time into weight bandwidth.

Weights are K-major, so the A rows of one CTA's 128-row slice of an output
block are one contiguous span per expert, and so is the matching block of the
atom-tiled scale factors. A prefetch never changes what is computed: a wrong or
evicted line only costs bandwidth.
"""

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import Int32, Int64


@cute.jit
def prefetch_l2_bulk(address: Int64, size: Int32) -> None:
    """``cp.async.bulk.prefetch.L2``: fire-and-forget, size a multiple of 16 bytes."""
    llvm.inline_asm(
        None,
        [address.ir_value(), size.ir_value()],
        "cp.async.bulk.prefetch.L2.global [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=0,
    )


@cute.jit
def prefetch_span(base: Int64, span_bytes: int, lane_idx: Int32, chunk_bytes: int) -> None:
    """Split one contiguous span over the 32 lanes of a warp, ``chunk_bytes`` per prefetch."""
    chunk_count = (span_bytes + chunk_bytes - 1) // chunk_bytes
    rounds = (chunk_count + 31) // 32
    for prefetch_round in cutlass.range_constexpr(rounds):
        chunk = Int32(prefetch_round * 32) + lane_idx
        if chunk < Int32(chunk_count):
            offset = chunk * Int32(chunk_bytes)
            size = Int32(chunk_bytes)
            if cutlass.const_expr(span_bytes % chunk_bytes != 0):
                if chunk == Int32(chunk_count - 1):
                    size = Int32(span_bytes - (chunk_count - 1) * chunk_bytes)
            prefetch_l2_bulk(base + Int64(offset), size)


__all__ = ["prefetch_l2_bulk", "prefetch_span"]
