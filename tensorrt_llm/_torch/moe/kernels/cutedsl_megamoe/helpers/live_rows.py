# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""The live rows of a launch on the device, from the forward's live-row word.

The word (int32) holds the real requests R in bits 0-15 and the requests P in bits 16-31. Of a launch's n rows the
leading ones are live: none when P is 0, all when R >= P or P does not divide n, else n / P * R. This is the rule of
TensorRT-LLM's ``tensorrt_llm/_torch/moe/live_rows.py`` (``live_row_count``, the host reference) and of the route
staging (``megaMoeLiveRows`` in ``cpp/tensorrt_llm/kernels/moe/cuteDsl/megaMoeStageRoutes.h``), so a bound taken here
covers exactly the rows whose routes the staging kept.
"""

import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import Int32, Int64, T, dsl_user_op

REQUEST_BITS = 16
REQUEST_MASK = (1 << REQUEST_BITS) - 1


@dsl_user_op
def _ld_nc_s32(address: Int64, *, loc=None, ip=None) -> Int32:
    """A read-only (non-coherent cache) load: the word changes only between launches. The same load as the adaptive
    route's (communication/nvlink_domain/adaptive_route.py), kept here so the reduce helpers need no communication
    import."""
    return Int32(
        llvm.inline_asm(
            T.i32(), [address.ir_value()], "ld.global.nc.b32 $0, [$1];", "=r,l", has_side_effects=False,
            asm_dialect=0, loc=loc, ip=ip,
        )
    )


@cute.jit
def live_row_count(word: Int32, rows: Int32) -> Int32:
    """The leading live rows of ``rows`` under ``word`` (module docstring)."""
    real = word & Int32(REQUEST_MASK)
    requests = (word >> Int32(REQUEST_BITS)) & Int32(REQUEST_MASK)
    live = rows
    if requests == Int32(0):
        live = Int32(0)
    elif real < requests:
        if rows % requests == Int32(0):
            live = rows // requests * real
    return live


@cute.jit
def live_rows_at(word_address: Int64, rows: Int32) -> Int32:
    """``live_row_count`` of the word at ``word_address``: a read-only load, the word changes only between launches."""
    return live_row_count(_ld_nc_s32(word_address), rows)


def live_row_count_host(word: int, rows: int) -> int:
    """The host twin of ``live_row_count`` (tests)."""
    real, requests = word & REQUEST_MASK, (word >> REQUEST_BITS) & REQUEST_MASK
    if requests == 0:
        return 0
    if real < requests and rows % requests == 0:
        return rows // requests * real
    return rows


__all__ = ["live_row_count", "live_row_count_host", "live_rows_at"]
