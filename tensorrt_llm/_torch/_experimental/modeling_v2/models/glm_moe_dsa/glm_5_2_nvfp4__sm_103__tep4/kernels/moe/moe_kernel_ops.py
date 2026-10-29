# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Kernel-side DSL operations for MoE FC2 finalize kernels.

These are GPU kernel operations (using MLIR/DSL) that don't belong in
moe_utils.py (host-only) or cutlass.experimental.primitives (framework-level).
"""

from __future__ import annotations

import cutlass.base_dsl.typing as cutlass
from cutlass import Pointer, dsl_user_op
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm as _llvm
from cutlass._mlir_helpers.dialect_proxy import DialectAutoConvertProxy
from cutlass.base_dsl.typing import Array, Numeric
from cutlass.cute.arch.nvvm_wrappers import inline_ptx
from cutlass.cutlass_dsl import T

llvm = DialectAutoConvertProxy(_llvm)


@dsl_user_op
def vectorized_atomic_add_bf16x8_relaxed(
    ptr: Array | Pointer | cutlass.Int64 | cutlass.Uint64,
    val0: Numeric,
    val1: Numeric,
    val2: Numeric,
    val3: Numeric,
    val4: Numeric,
    val5: Numeric,
    val6: Numeric,
    val7: Numeric,
    *,
    loc: ir.Location | None = None,
    ip: ir.InsertionPoint | None = None,
) -> None:
    """Emit ``red.relaxed.gpu.global.v4.bf16x2.add.noftz`` for 8 BF16.

    Callers pass 8 logical BF16 elements and this helper packs them
    into the required ``bf16x2`` payloads internally. The reduction is
    performed with relaxed memory ordering, sufficient for commutative
    scatter-reduce operations.
    """
    values = (val0, val1, val2, val3, val4, val5, val6, val7)
    for idx, value in enumerate(values):
        if not isinstance(value, cutlass.BFloat16):
            raise TypeError(
                "vectorized_atomic_add_bf16x8_relaxed expects "
                f"8 bf16 scalars; arg {idx} has type={type(value)}"
            )

    if hasattr(ptr, "data_ptr"):
        ptr_int = ptr.data_ptr().toint(cutlass.Int64)
    elif hasattr(ptr, "toint"):
        ptr_int = ptr.toint(cutlass.Int64)
    else:
        ptr_int = cutlass.Int64(ptr)

    packed_words = []
    for pair_start in range(0, len(values), 2):
        pair = cutlass.Vector.from_elements(
            (values[pair_start], values[pair_start + 1]),
            cutlass.BFloat16,
            loc=loc,
            ip=ip,
        )
        packed_words.append(llvm.bitcast(T.i32(), pair, loc=loc, ip=ip))

    inline_ptx(
        "red.relaxed.gpu.global.v4.bf16x2.add.noftz [{$r0}], {{$r1}, {$r2}, {$r3}, {$r4}};",
        read_only_args=[ptr_int, *packed_words],
        loc=loc,
        ip=ip,
    )
