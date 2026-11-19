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
"""Inline PTX that both kernel families issue, and the one emitter of the inline PTX of the package: asm, which takes
the instruction text, the typed operands and the result classes of a statement and derives the rest. The wrappers of
the PTX modules call it: this one, deco_attention/tcgen05/ptx.py and the modules of the package
deco_attention/sm120/ptx/, which hold what one family alone issues.

The rule of the package: an operation is a CuTe DSL call (cute.arch, cute.math, the typed pointers of common/mem.py,
the experimental primitives of common/dsl.py) where the DSL offers the operation in the form the kernel uses; it is
an inline-PTX wrapper of one of these modules otherwise.

Each wrapper emits exactly the instruction text it shows; whether it has side effects is part of its contract (the
compiler may merge, move or drop a statement without them), and a docstring says "no side effects" where the wrapper
declares none.

The names, in all PTX modules. A wrapper of one instruction is named after it, in the order operation, address space,
mode (the cache or ordering mode of a memory access, the rounding of an arithmetic instruction), type, vector length:
ld_global_nc_f32, st_global_cs_b32_v2_pred, mul_rn_f32. The type is left out where the wrapper takes it
as an argument (`dtype`: ld_global_cg_v4), and the address of a memory access is the parameter `addr` (dst and src
where there are two). A suffix names a variant: _pred (predicated on an operand), _imm (an immediate in place of an
operand), _hint (an L2 cache policy), _trans (transposed). A conversion is <source>_to_<result> with the element
counts (f32x8_to_e2m1x8, e2m1x8_to_f32x8). A block of several instructions is named after what it computes, and one
operation has one name in both families (tc_mma and tc_mma_f8f6f4, mma_bf16 and mma_nvf4).

The sections:

  conversions and packed arithmetic   E2M1 packing and the warp ballot
  memory access                       the vector load of global memory that is not cached in L1, the acquire load of a
                                      word"""

from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cute.typing import Float32, Int32, Int64, Uint32
from cutlass.cutlass_dsl import T, dsl_user_op

# Per class of an operand or a result of asm: its register class in the constraint string, its IR type, and the PTX
# type of a load of a value of the class
_REGISTER = {Int32: "r", Uint32: "r", Float32: "f", Int64: "l"}
_IR_TYPE = {Int32: T.i32, Uint32: T.i32, Float32: T.f32, Int64: T.i64}
_PTX_TYPE = {Int32: "b32", Uint32: "b32", Float32: "f32", Int64: "b64"}


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


@dsl_user_op
def vote_ballot(p, *, loc=None, ip=None):
    """i32 p -> u32 mask of the lanes of the warp with p != 0 (setp.ne.s32, vote.sync.ballot.b32 over all 32 lanes)."""
    return asm(
        "{ .reg .pred q; setp.ne.s32 q, $1, 0; vote.sync.ballot.b32 $0, q, 0xffffffff; }",
        [i32(p)],
        Uint32,
        side_effects=True,
        loc=loc,
        ip=ip,
    )


# -------------------------------------------------------------------------------- Memory access
@dsl_user_op
def ld_global_cg_v4(addr, dtype, *, loc=None, ip=None):
    """i64 global address -> four 32-bit words (16 bytes) as dtype, cached at the global level and not in L1 (data
    another CTA writes): by ld.global.cg.v4.b32 for Int32, by ld.global.cg.v4.f32 for Float32. The merging piece of a
    split unit reads the rows of the other pieces with it (tcgen05/fp8pv/epilogue.py, sm120/nvfp4pv/kernel.py)."""
    return asm(
        f"ld.global.cg.v4.{_PTX_TYPE[dtype]} {{$0, $1, $2, $3}}, [$4];",
        [i64(addr)],
        [dtype] * 4,
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def ld_global_acquire_b32(addr, dtype=Uint32, *, loc=None, ip=None):
    """i64 global address -> 32-bit word by ld.acquire.gpu.global.b32, as dtype (Uint32 or Int32): the coherent read
    of a word that another CTA writes, a counter or a flag."""
    return asm(
        "ld.acquire.gpu.global.b32 $0, [$1];", [i64(addr)], dtype, side_effects=True, loc=loc, ip=ip
    )
