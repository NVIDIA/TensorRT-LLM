# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One-kernel ``attn_output * gate.sigmoid()`` for the Kimi K3.1 MLA output gate (CuTe DSL).

The launcher is :func:`k31_sigmoid_gate_mul`. The result is bit-identical to the two PyTorch
BF16 kernels: each element runs the arithmetic below, written out as inline PTX so that no
compiler choice can move a bit.

* ``sigmoid(g) = 1 / (1 + expf(-g))`` in FP32. ``expf`` is the CUDA math library's flush-to-zero
  ``__nv_expf`` (range reduction by FMA, ``ex2.approx.ftz``, scale by ``2^j``), with its final
  scaling multiply fused into the ``1 +`` add. The gate is rounded to BF16.
* The reciprocal is the correctly rounded ``1 / d`` of ``rcp.rn.f32``, without its branch: for
  ``d < 2^126`` it is the compiler's own fast path for ``rcp.rn.f32`` (``rcp.approx`` and one
  FMA correction, the same instructions, so the same bits); for ``2^126 <= d < 2^128`` the
  fast path of ``d / 4`` scaled by ``1 / 4``, which lands in the denormal range with a second
  rounding; ``d = inf`` gives 0. ``rcp.rn.f32`` as written puts each element's reciprocal
  behind its own branch, which serializes the elements of a thread.
* The product is the FP32 product of the two BF16 values (exact), rounded to BF16 (round to
  nearest even). The product and its roundings keep denormals, as the PyTorch kernels do: ``g``
  in about [-103, -87.3] has a denormal sigmoid.

Since the gate is a function of one BF16 value and the product is exact before its rounding, the
tests check the whole map: every one of the 65536 gate values (the scaled path included) against
PyTorch.

Layout: ``attn_output`` is a contiguous ``[M, N]``; ``gate`` is either contiguous or a column
slice of a row-major tensor (row stride ``>= N``, unit column stride), as when the gate is a
column block of a packed GEMM output. Each thread moves 16-byte vectors (8 elements) when every
base pointer is 16-byte aligned and the row length and gate row stride are multiples of 8, and
single elements otherwise. Launches use programmatic dependent launch when
``TRTLLM_ENABLE_PDL`` is unset or ``1``: the kernel waits for its producer before its first load;
a grid of at most one wave also lets its dependent launch once its loads are issued (a larger
one leaves that to its exit, so the dependent's CTAs do not take the slots of its later waves).
"""

from __future__ import annotations

import functools

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

try:
    from cuda.bindings import driver as cuda
except ImportError:
    from cuda import cuda

#: BF16 elements per 16-byte vector.
VEC = 8
_OPTIONS = "--opt-level 2 --enable-tvm-ffi"

# ``__nv_expf`` (flush-to-zero build) and the rest of the per-element sequence, two elements per
# call: $1 holds two BF16 values of attn_output, $2 the two gate values, $0 the two results. The
# constants: 0x3BBB989D = log2(e) / 256, 0x437C0000 = 252, 0x4B400001 = 1.5 * 2^23 + 1,
# 0xCB40007F = -(1.5 * 2^23 + 127), 0x3FB8AA3B / 0x32A57060 = log2(e) split high / low.
_LANE = """
cvt.f32.bf16 a{i}, g{i};
neg.f32 a{i}, a{i};
fma.rn.ftz.f32 t{i}, a{i}, 0f3BBB989D, 0f3F000000;
cvt.ftz.sat.f32.f32 t{i}, t{i};
fma.rm.ftz.f32 j{i}, t{i}, 0f437C0000, 0f4B400001;
add.rn.f32 k{i}, j{i}, 0fCB40007F;
neg.f32 k{i}, k{i};
fma.rn.ftz.f32 r{i}, a{i}, 0f3FB8AA3B, k{i};
fma.rn.ftz.f32 r{i}, a{i}, 0f32A57060, r{i};
ex2.approx.ftz.f32 e{i}, r{i};
shl.b32 s{i}, j{i}, 23;
fma.rn.f32 d{i}, e{i}, s{i}, 0f3F800000;
setp.lt.f32 pf{i}, d{i}, 0f7E800000;
mul.rn.f32 v{i}, d{i}, 0f3E800000;
selp.b32 v{i}, d{i}, v{i}, pf{i};
rcp.approx.ftz.f32 w{i}, v{i};
fma.rn.f32 u{i}, v{i}, w{i}, 0fBF800000;
neg.f32 u{i}, u{i};
fma.rn.f32 q{i}, w{i}, u{i}, w{i};
mul.rn.f32 v{i}, q{i}, 0f3E800000;
selp.b32 q{i}, q{i}, v{i}, pf{i};
setp.eq.f32 pi{i}, d{i}, 0f7F800000;
selp.b32 q{i}, 0, q{i}, pi{i};
cvt.rn.bf16.f32 qh{i}, q{i};
cvt.f32.bf16 q{i}, qh{i};
cvt.f32.bf16 b{i}, x{i};
mul.rn.f32 p{i}, b{i}, q{i};
cvt.rn.bf16.f32 o{i}, p{i};
"""
_PAIR_ASM = (
    "{\n"
    ".reg .b16 x0, x1, g0, g1, qh0, qh1, o0, o1;\n"
    ".reg .b32 a0, a1, t0, t1, j0, j1, k0, k1, r0, r1, e0, e1, s0, s1, d0, d1, q0, q1, b0, b1, "
    "p0, p1, v0, v1, w0, w1, u0, u1;\n"
    ".reg .pred pf0, pf1, pi0, pi1;\n"
    "mov.b32 {x0, x1}, $1;\n"
    "mov.b32 {g0, g1}, $2;\n" + _LANE.format(i=0) + _LANE.format(i=1) + "mov.b32 $0, {o0, o1};\n}"
)


def _asm(res, operands, text, constraints, side_effects, *, loc=None, ip=None):
    return llvm.inline_asm(
        res,
        operands,
        text,
        constraints,
        has_side_effects=side_effects,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _gate_mul_pair(xw, gw, *, loc=None, ip=None):
    """Two BF16 results packed in one word from two packed ``attn_output`` and gate values."""
    return Int32(
        _asm(
            T.i32(),
            [Int32(xw).ir_value(loc=loc, ip=ip), Int32(gw).ir_value(loc=loc, ip=ip)],
            _PAIR_ASM,
            "=r,r,r",
            False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ld_v4(addr, pred, *, loc=None, ip=None):
    """16 bytes at ``addr`` (zeros where ``pred`` is 0, without touching memory)."""
    ty = llvm.StructType.get_literal([T.i32()] * 4)
    res = _asm(
        ty,
        [Int64(addr).ir_value(loc=loc, ip=ip), Int32(pred).ir_value(loc=loc, ip=ip)],
        "{\n.reg .pred p;\nsetp.ne.b32 p, $5, 0;\nmov.b32 $0, 0;\nmov.b32 $1, 0;\n"
        "mov.b32 $2, 0;\nmov.b32 $3, 0;\n@p ld.global.nc.v4.b32 {$0, $1, $2, $3}, [$4];\n}",
        "=r,=r,=r,=r,l,r",
        True,
        loc=loc,
        ip=ip,
    )
    return [Int32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip)) for i in range(4)]


@dsl_user_op
def _st_v4(addr, words, pred, *, loc=None, ip=None):
    _asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip)]
        + [Int32(w).ir_value(loc=loc, ip=ip) for w in words]
        + [Int32(pred).ir_value(loc=loc, ip=ip)],
        "{\n.reg .pred p;\nsetp.ne.b32 p, $5, 0;\n@p st.global.v4.b32 [$0], {$1, $2, $3, $4};\n}",
        "l,r,r,r,r,r",
        True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ld_b16(addr, pred, *, loc=None, ip=None):
    """The 2 bytes at ``addr`` in the low half of a word (0 where ``pred`` is 0)."""
    return Int32(
        _asm(
            T.i32(),
            [Int64(addr).ir_value(loc=loc, ip=ip), Int32(pred).ir_value(loc=loc, ip=ip)],
            "{\n.reg .pred p;\n.reg .b16 h;\nsetp.ne.b32 p, $2, 0;\nmov.b16 h, 0;\n"
            "@p ld.global.nc.b16 h, [$1];\ncvt.u32.u16 $0, h;\n}",
            "=r,l,r",
            True,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _st_b16(addr, word, pred, *, loc=None, ip=None):
    _asm(
        None,
        [
            Int64(addr).ir_value(loc=loc, ip=ip),
            Int32(word).ir_value(loc=loc, ip=ip),
            Int32(pred).ir_value(loc=loc, ip=ip),
        ],
        "{\n.reg .pred p;\n.reg .b16 h;\nsetp.ne.b32 p, $2, 0;\ncvt.u16.u32 h, $1;\n"
        "@p st.global.b16 [$0], h;\n}",
        "l,r,r",
        True,
        loc=loc,
        ip=ip,
    )


class SigmoidGateMulKernel:
    """``out = attn_output * sigmoid(gate)`` over ``[rows, cols]``, ``vec`` elements per step.

    A step is one 16-byte vector (``vec == 8``) or one element (``vec == 1``). Thread ``t`` of
    CTA ``b`` owns steps ``(b * unroll + u) * threads + t`` for ``u < unroll``: it issues all
    its loads, then computes and stores. Packed gate rows: the steps run over all rows as one.
    ``strided`` (the gate rows sit ``gate_row_stride`` elements apart): CTA ``(r, b)`` takes
    those steps of row ``r``.
    """

    def __init__(self, vec: int, strided: bool, threads: int, unroll: int, use_pdl: bool):
        if vec not in (1, VEC):
            raise ValueError(f"vec must be 1 or {VEC}, got {vec}")
        self.vec = vec
        self.strided = strided
        self.threads = threads
        self.unroll = unroll
        self.use_pdl = use_pdl

    @cute.jit
    def __call__(
        self,
        attn_output: cute.Tensor,
        gate: cute.Tensor,
        out: cute.Tensor,
        trigger: Int32,
        stream: cuda.CUstream,
    ):
        rows = attn_output.shape[0]
        cols_v = attn_output.shape[1] // self.vec
        per_cta = self.threads * self.unroll
        # Steps per row of CTAs: one gate row (strided) or every element (packed rows).
        steps = cols_v
        grid = [rows, (steps + per_cta - 1) // per_cta, 1]
        if cutlass.const_expr(not self.strided):
            steps = rows * attn_output.shape[1] // self.vec
            grid = [(steps + per_cta - 1) // per_cta, 1, 1]
        self.kernel(
            attn_output.iterator.toint(),
            gate.iterator.toint(),
            out.iterator.toint(),
            steps,
            Int64(gate.stride[0]),
            trigger,
        ).launch(
            grid=grid,
            block=[self.threads, 1, 1],
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        x_addr: Int64,
        g_addr: Int64,
        out_addr: Int64,
        steps: Int32,
        gate_row_stride: Int64,
        trigger: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        row = Int32(0)
        if cutlass.const_expr(self.strided):
            row, bidx, _ = cute.arch.block_idx()
        step_bytes = 2 * self.vec
        first = bidx * (self.threads * self.unroll) + tidx
        offs = []
        g_offs = []
        oks = []
        for u in cutlass.range_constexpr(self.unroll):
            v = first + u * self.threads
            oks.append(Int32(v < steps))
            if cutlass.const_expr(self.strided):
                offs.append((Int64(row) * steps + Int64(v)) * step_bytes)
                g_offs.append((Int64(row) * gate_row_stride + Int64(v) * self.vec) * 2)
            else:
                offs.append(Int64(v) * step_bytes)
                g_offs.append(Int64(v) * step_bytes)

        if cutlass.const_expr(self.use_pdl):
            cute.arch.griddepcontrol_wait()
        xs = []
        gs = []
        for u in cutlass.range_constexpr(self.unroll):
            if cutlass.const_expr(self.vec == VEC):
                xs.append(_ld_v4(x_addr + offs[u], oks[u]))
                gs.append(_ld_v4(g_addr + g_offs[u], oks[u]))
            else:
                xs.append([_ld_b16(x_addr + offs[u], oks[u])])
                gs.append([_ld_b16(g_addr + g_offs[u], oks[u])])
        if cutlass.const_expr(self.use_pdl):
            if trigger != 0:
                cute.arch.griddepcontrol_launch_dependents()

        for u in cutlass.range_constexpr(self.unroll):
            res = [_gate_mul_pair(xw, gw) for xw, gw in zip(xs[u], gs[u])]
            if cutlass.const_expr(self.vec == VEC):
                _st_v4(out_addr + offs[u], res, oks[u])
            else:
                _st_b16(out_addr + offs[u], res[0], oks[u])


@functools.cache
def _compile(vec: int, strided: bool, threads: int, unroll: int, use_pdl: bool):
    rows, cols = cute.sym_int(), cute.sym_int()
    x_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16, (rows, cols), stride_order=(1, 0)
    )
    g_fake = cute.runtime.make_fake_tensor(
        cutlass.BFloat16, (rows, cols), stride=(cute.sym_int64(), 1)
    )
    out_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16, (rows, cols), stride_order=(1, 0)
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    kernel = SigmoidGateMulKernel(vec, strided, threads, unroll, use_pdl)
    return cute.compile(kernel, x_fake, g_fake, out_fake, Int32(0), stream_fake, options=_OPTIONS)


#: Steps (vectors or elements) up to which the grid is taken for one wave: 1024 CTAs of 128
#: threads, about 7 per SM.
_ONE_WAVE_STEPS = 1024 * 128


def sigmoid_gate_mul_into(
    attn_output: torch.Tensor,
    gate: torch.Tensor,
    out: torch.Tensor,
    *,
    threads: int = 128,
    unroll: int = 1,
    use_pdl: bool | None = None,
    trigger: bool | None = None,
) -> None:
    """Write ``attn_output * gate.sigmoid()`` into ``out`` (contiguous, ``attn_output``'s shape and
    dtype; layouts as in the module docstring, not re-checked). Compiles once per layout variant
    (vector or element steps, packed or strided gate rows) on the first call, which must run
    outside CUDA-graph capture; later calls only launch. ``trigger`` (default: the grid is at
    most about one wave) is a launch argument, not a variant."""
    n = attn_output.numel()
    if n == 0:
        return
    if n >= 2**31:
        raise ValueError("sigmoid_gate_mul indexes elements with 32-bit integers")
    strided = not gate.is_contiguous()
    if strided:
        vec_ok = attn_output.shape[1] % VEC == 0 and gate.stride(0) % VEC == 0
    else:
        vec_ok = n % VEC == 0
    vec_ok = vec_ok and (attn_output.data_ptr() | gate.data_ptr() | out.data_ptr()) % 16 == 0
    vec = VEC if vec_ok else 1
    if use_pdl is None:
        use_pdl = get_env_enable_pdl()
    if trigger is None:
        trigger = n // vec <= _ONE_WAVE_STEPS
    _compile(vec, strided, threads, unroll, use_pdl)(attn_output, gate, out, int(trigger))


def _check(attn_output: torch.Tensor, gate: torch.Tensor) -> None:
    if (
        attn_output.dtype is not torch.bfloat16
        or gate.dtype is not torch.bfloat16
        or attn_output.dim() != 2
        or attn_output.shape != gate.shape
    ):
        raise ValueError("attn_output and gate must be BF16 tensors of one 2D shape")
    if not attn_output.is_cuda or gate.device != attn_output.device:
        raise ValueError("attn_output and gate must be on one CUDA device")
    if not attn_output.is_contiguous():
        raise ValueError("attn_output must be contiguous")
    if not gate.is_contiguous() and (gate.stride(1) != 1 or gate.stride(0) < gate.shape[1]):
        raise ValueError("gate must be contiguous or a column slice of a row-major tensor")


def k31_sigmoid_gate_mul(attn_output: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """``attn_output * gate.sigmoid()`` (BF16 ``[M, N]``), bit-identical to the two PyTorch
    kernels. ``attn_output`` contiguous; ``gate`` contiguous or a column slice of a row-major
    tensor."""
    _check(attn_output, gate)
    out = torch.empty_like(attn_output)
    sigmoid_gate_mul_into(attn_output, gate, out)
    return out


__all__ = ["SigmoidGateMulKernel", "k31_sigmoid_gate_mul", "sigmoid_gate_mul_into"]
