# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""L2 cache-policy constants for the indexer-K stream."""

import cutlass
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op

POL_NONE, POL_EVICT_FIRST, POL_UNCHANGED, POL_FIRST_HALF, POL_NORMAL = (0, 1, 2, 3, 4)
POL_NAMES = {
    POL_NONE: "none",
    POL_EVICT_FIRST: "ef",
    POL_UNCHANGED: "eu",
    POL_FIRST_HALF: "ef50",
    POL_NORMAL: "en",
}
_POL_ASM = {
    POL_EVICT_FIRST: "createpolicy.fractional.L2::evict_first.b64 $0, 1.0;",
    POL_UNCHANGED: "createpolicy.fractional.L2::evict_unchanged.b64 $0, 1.0;",
    POL_FIRST_HALF: "createpolicy.fractional.L2::evict_first.L2::evict_unchanged.b64 $0, 0.5;",
    POL_NORMAL: "createpolicy.fractional.L2::evict_normal.b64 $0, 1.0;",
}


@dsl_user_op
def make_l2_policy(policy: int, *, loc=None, ip=None) -> cutlass.Int64:
    if policy not in _POL_ASM:
        raise ValueError(f"no createpolicy for knob {policy}")
    return cutlass.Int64(
        llvm.inline_asm(
            cutlass.Int64.mlir_type,
            [],
            _POL_ASM[policy],
            "=l",
            has_side_effects=True,
            loc=loc,
            ip=ip,
        )
    )
