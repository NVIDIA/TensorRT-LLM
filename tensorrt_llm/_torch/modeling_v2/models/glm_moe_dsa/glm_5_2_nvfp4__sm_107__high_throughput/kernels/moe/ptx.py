# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import Int64, T, dsl_user_op


@dsl_user_op
def ld_relaxed_sys_b64(address: Int64, *, loc=None, ip=None) -> Int64:
    """Load one naturally aligned 64-bit word with a relaxed system-scope load (see ``st_relaxed_sys_b64``)."""
    return Int64(
        llvm.inline_asm(
            T.i64(),
            [address.ir_value()],
            "ld.relaxed.sys.global.b64 $0, [$1];",
            "=l,l",
            has_side_effects=True,
            asm_dialect=0,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def st_relaxed_sys_b64(address: Int64, value: Int64, *, loc=None, ip=None) -> None:
    """Store one naturally aligned 64-bit word with a relaxed system-scope store.

    A morally strong aligned 64-bit access is single-copy atomic: a relaxed system-scope load of
    the same word returns either the old or the new value in full, never a mix.
    """
    llvm.inline_asm(
        None,
        [address.ir_value(), value.ir_value()],
        "st.relaxed.sys.global.b64 [$0], $1;",
        "l,l",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )
