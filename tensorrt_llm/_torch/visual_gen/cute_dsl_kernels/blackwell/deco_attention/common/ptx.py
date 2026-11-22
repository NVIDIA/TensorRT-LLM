# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inline PTX that both kernel families issue, and asm, the one emitter of the package's inline PTX: it takes the
instruction text, the typed operands and the result classes of a statement. The family modules (tcgen05/ptx.py,
sm120/ptx/) hold what one family alone issues.

An operation is a CuTe DSL call where the DSL offers it in the form the kernel uses, and a wrapper of these modules
otherwise. A wrapper emits exactly the text it shows; its docstring says "no side effects" where it declares none
(the compiler may then merge, move or drop the statement).

Names: a wrapper of one instruction is named after it, as operation, address space, mode, type, vector length
(ld_global_nc_f32, st_global_cs_b32_v2_pred); the type is left out where it is the argument `dtype`. Suffixes: _pred
(predicated), _imm (an immediate operand), _hint (an L2 cache policy), _trans (transposed). A conversion is
<source>_to_<result> with the element counts (f32x8_to_e2m1x8). A block of several instructions is named after what
it computes.

Here: E2M1 packing."""

from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cute.typing import Float32, Int32, Int64, Uint32
from cutlass.cutlass_dsl import T, dsl_user_op

# Per class of an operand or a result of asm: its register class in the constraint string and its IR type
_REGISTER = {Int32: "r", Uint32: "r", Float32: "f", Int64: "l"}
_IR_TYPE = {Int32: T.i32, Uint32: T.i32, Float32: T.f32, Int64: T.i64}


def operand(dtype, x):
    """x as an operand of asm of class dtype (Int32, Uint32, Float32 or Int64)."""
    return dtype, x, _REGISTER[dtype]


def i32(x):
    """x as an Int32 operand of asm."""
    return operand(Int32, x)


def u32(x):
    """x as a Uint32 operand of asm."""
    return operand(Uint32, x)


def f32(x):
    """x as a Float32 operand of asm."""
    return operand(Float32, x)


def i64(x):
    """x as an Int64 operand of asm."""
    return operand(Int64, x)


def tied(index, x):
    """The operand x of asm (an i32, u32, f32 or i64 of a value) in the register of result `index`: the block updates
    that value in place."""
    dtype, value, _ = x
    return dtype, value, str(index)


def early(dtype):
    """A result class of asm for a result that shares its register with no operand: the block writes the result while
    it still reads operands."""
    return dtype, "=&" + _REGISTER[dtype]


def asm(text, operands=(), results=None, *, side_effects, loc=None, ip=None):
    """One inline-PTX statement or block.

    text: the instructions; $0, $1, ... are the results in order, then the operands in order.
    operands: the operand values, each as i32, u32, f32 or i64 of a value (the class it is converted with; operand
        takes the class as an argument), or as tied of one of those.
    results: None, one result class (Int32, Uint32, Float32 or Int64), or a list of them; a class may be given as
        early of it.
    side_effects: whether the statement has side effects.

    Returns None, the result as its class, or the list of the results as their classes. The constraint string (per
    result and then per operand, its register class: r for a 32-bit integer, f for an f32, l for an i64), the IR
    values of the operands and the extraction of the results from the struct of several results follow from these.
    """
    several = isinstance(results, list)
    outs = results if several else [] if results is None else [results]
    outs = [r if isinstance(r, tuple) else (r, "=" + _REGISTER[r]) for r in outs]
    types = [_IR_TYPE[dtype]() for dtype, _ in outs]
    if several:
        ret = ir.Type.parse("!llvm.struct<(" + ",".join(str(t) for t in types) + ")>")
    else:
        ret = types[0] if types else None
    value = llvm.inline_asm(
        ret,
        [dtype(x).ir_value(loc=loc, ip=ip) for dtype, x, _ in operands],
        text,
        ",".join([c for _, c in outs] + [c for _, _, c in operands]),
        has_side_effects=side_effects,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    if several:
        return [
            dtype(llvm.extractvalue(t, value, [i], loc=loc, ip=ip))
            for i, ((dtype, _), t) in enumerate(zip(outs, types))
        ]
    return outs[0][0](value) if outs else None


# -------------------------------------------------------------------------------- Conversions and packed arithmetic
@dsl_user_op
def f32x8_to_e2m1x8(y, *, loc=None, ip=None):
    """Eight f32 -> u32 word of four E2M1 code bytes (cvt.rn.satfinite.e2m1x2.f32 per pair: pair p in byte p, element 2
    p in the low nibble); no side effects."""
    return asm(
        "{ .reg .b8 b0, b1, b2, b3;\n"
        " cvt.rn.satfinite.e2m1x2.f32 b0, $2, $1;\n cvt.rn.satfinite.e2m1x2.f32 b1, $4, $3;\n"
        " cvt.rn.satfinite.e2m1x2.f32 b2, $6, $5;\n cvt.rn.satfinite.e2m1x2.f32 b3, $8, $7;\n"
        " mov.b32 $0, {b0, b1, b2, b3}; }",
        [f32(v) for v in y],
        Uint32,
        side_effects=False,
        loc=loc,
        ip=ip,
    )
