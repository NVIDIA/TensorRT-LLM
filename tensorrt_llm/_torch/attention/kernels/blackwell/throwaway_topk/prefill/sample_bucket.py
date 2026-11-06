# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Clamp a descending-key distance before using it as a shared-memory index."""
import cutlass
from cutlass.cutlass_dsl import dsl_user_op
from cutlass._mlir.dialects import llvm

@dsl_user_op
def unsigned_bucket_index(distance,shift,*,loc=None,ip=None):
    # Clamp ordered-key distances with unsigned PTX: -inf padding can
    # exceed 2**31 units from a positive range. The final bin is signed 0..255.
    result=llvm.inline_asm(cutlass.Int32.mlir_type,
        [cutlass.Uint32(distance).ir_value(loc=loc,ip=ip),cutlass.Int32(shift).ir_value(loc=loc,ip=ip)],
        "{ .reg .b32 shifted; shr.u32 shifted,$1,$2; min.u32 $0,shifted,255; }",
        "=r,r,r",has_side_effects=False,loc=loc,ip=ip)
    return cutlass.Int32(result)
