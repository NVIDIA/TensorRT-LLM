# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One-kernel ``prefix + (routed + shared)`` for the end of a Kimi K3.1 MoE layer (CuTe DSL).

The decoder layer adds the routed and the shared expert outputs (``routed_out + shared_out``) and then adds that to
the running residual (``prefix_sum + hidden_states``): two BF16 PyTorch kernels that read four and write two
``[tokens, hidden]`` tensors. This kernel reads the three inputs once and writes the result once, with the same two
roundings: every element is ``bf16(f32(prefix) + f32(bf16(f32(routed) + f32(shared))))``, round to nearest even,
denormals kept, as PyTorch's BF16 ``add`` computes it (FP32 opmath, alpha 1). Bit-identical to the two kernels on
every finite input; a NaN comes out as a NaN (its payload may differ).

Layout: three contiguous BF16 tensors of one shape. Each thread moves 16-byte vectors (8 elements) when every base
pointer is 16-byte aligned and the element count is a multiple of 8, and single elements otherwise. Launches use
programmatic dependent launch when ``TRTLLM_ENABLE_PDL`` is unset or ``1``: the kernel waits for its producer before
its first load; a grid of at most one wave also lets its dependent launch once its loads are issued.
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

# Two elements per call: $1 two BF16 values of prefix, $2 of routed, $3 of shared; $0 the two results.
_LANE = """
cvt.f32.bf16 r{i}, xr{i};
cvt.f32.bf16 s{i}, xs{i};
add.rn.f32 t{i}, r{i}, s{i};
cvt.rn.bf16.f32 th{i}, t{i};
cvt.f32.bf16 t{i}, th{i};
cvt.f32.bf16 p{i}, xp{i};
add.rn.f32 o{i}, p{i}, t{i};
cvt.rn.bf16.f32 oh{i}, o{i};
"""
_PAIR_ASM = (
    "{\n"
    ".reg .b16 xp0, xp1, xr0, xr1, xs0, xs1, th0, th1, oh0, oh1;\n"
    ".reg .b32 r0, r1, s0, s1, t0, t1, p0, p1, o0, o1;\n"
    "mov.b32 {xp0, xp1}, $1;\n"
    "mov.b32 {xr0, xr1}, $2;\n"
    "mov.b32 {xs0, xs1}, $3;\n"
    + _LANE.format(i=0)
    + _LANE.format(i=1)
    + "mov.b32 $0, {oh0, oh1};\n}"
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
def _add3_pair(pw, rw, sw, *, loc=None, ip=None):
    """Two BF16 results packed in one word from packed prefix, routed and shared values."""
    return Int32(
        _asm(
            T.i32(),
            [
                Int32(pw).ir_value(loc=loc, ip=ip),
                Int32(rw).ir_value(loc=loc, ip=ip),
                Int32(sw).ir_value(loc=loc, ip=ip),
            ],
            _PAIR_ASM,
            "=r,r,r,r",
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


class MoeResidualAddKernel:
    """``out = prefix + (routed + shared)`` over ``steps`` vectors (``vec == 8``) or elements (``vec == 1``).

    Thread ``t`` of CTA ``b`` owns steps ``(b * unroll + u) * threads + t`` for ``u < unroll``: it issues all its
    loads, then computes and stores.
    """

    def __init__(self, vec: int, threads: int, unroll: int, use_pdl: bool):
        if vec not in (1, VEC):
            raise ValueError(f"vec must be 1 or {VEC}, got {vec}")
        self.vec = vec
        self.threads = threads
        self.unroll = unroll
        self.use_pdl = use_pdl

    @cute.jit
    def __call__(
        self,
        prefix: cute.Tensor,
        routed: cute.Tensor,
        shared: cute.Tensor,
        out: cute.Tensor,
        trigger: Int32,
        stream: cuda.CUstream,
    ):
        steps = prefix.shape[0] * prefix.shape[1] // self.vec
        per_cta = self.threads * self.unroll
        self.kernel(
            prefix.iterator.toint(),
            routed.iterator.toint(),
            shared.iterator.toint(),
            out.iterator.toint(),
            steps,
            trigger,
        ).launch(
            grid=[(steps + per_cta - 1) // per_cta, 1, 1],
            block=[self.threads, 1, 1],
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        p_addr: Int64,
        r_addr: Int64,
        s_addr: Int64,
        out_addr: Int64,
        steps: Int32,
        trigger: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        step_bytes = 2 * self.vec
        first = bidx * (self.threads * self.unroll) + tidx
        offs = []
        oks = []
        for u in cutlass.range_constexpr(self.unroll):
            v = first + u * self.threads
            oks.append(Int32(v < steps))
            offs.append(Int64(v) * step_bytes)

        if cutlass.const_expr(self.use_pdl):
            cute.arch.griddepcontrol_wait()
        ps = []
        rs = []
        ss = []
        for u in cutlass.range_constexpr(self.unroll):
            if cutlass.const_expr(self.vec == VEC):
                ps.append(_ld_v4(p_addr + offs[u], oks[u]))
                rs.append(_ld_v4(r_addr + offs[u], oks[u]))
                ss.append(_ld_v4(s_addr + offs[u], oks[u]))
            else:
                ps.append([_ld_b16(p_addr + offs[u], oks[u])])
                rs.append([_ld_b16(r_addr + offs[u], oks[u])])
                ss.append([_ld_b16(s_addr + offs[u], oks[u])])
        if cutlass.const_expr(self.use_pdl):
            if trigger != 0:
                cute.arch.griddepcontrol_launch_dependents()

        for u in cutlass.range_constexpr(self.unroll):
            res = [_add3_pair(pw, rw, sw) for pw, rw, sw in zip(ps[u], rs[u], ss[u])]
            if cutlass.const_expr(self.vec == VEC):
                _st_v4(out_addr + offs[u], res, oks[u])
            else:
                _st_b16(out_addr + offs[u], res[0], oks[u])


@functools.cache
def _compile(vec: int, threads: int, unroll: int, use_pdl: bool):
    rows, cols = cute.sym_int(), cute.sym_int()
    fakes = [
        cute.runtime.make_fake_compact_tensor(cutlass.BFloat16, (rows, cols), stride_order=(1, 0))
        for _ in range(4)
    ]
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    kernel = MoeResidualAddKernel(vec, threads, unroll, use_pdl)
    return cute.compile(kernel, *fakes, Int32(0), stream_fake, options=_OPTIONS)


#: Steps (vectors or elements) up to which the grid is taken for one wave: 1024 CTAs of 128 threads.
_ONE_WAVE_STEPS = 1024 * 128


def moe_residual_add_into(
    prefix: torch.Tensor,
    routed: torch.Tensor,
    shared: torch.Tensor,
    out: torch.Tensor,
    *,
    threads: int = 128,
    unroll: int = 2,
    use_pdl: bool | None = None,
    trigger: bool | None = None,
) -> None:
    """Write ``prefix + (routed + shared)`` into ``out`` (all contiguous BF16 2D tensors of one shape; not
    re-checked). ``out`` may alias ``prefix``. Compiles once per variant (vector or element steps) on the first call,
    which must run outside CUDA-graph capture; later calls only launch."""
    n = prefix.numel()
    if n == 0:
        return
    if n >= 2**31:
        raise ValueError("moe_residual_add indexes elements with 32-bit integers")
    vec_ok = (
        n % VEC == 0
        and (prefix.data_ptr() | routed.data_ptr() | shared.data_ptr() | out.data_ptr()) % 16 == 0
    )
    vec = VEC if vec_ok else 1
    if use_pdl is None:
        use_pdl = get_env_enable_pdl()
    if trigger is None:
        trigger = n // vec <= _ONE_WAVE_STEPS * unroll
    _compile(vec, threads, unroll, use_pdl)(
        prefix.view(-1, prefix.shape[-1]),
        routed.view(-1, prefix.shape[-1]),
        shared.view(-1, prefix.shape[-1]),
        out.view(-1, prefix.shape[-1]),
        int(trigger),
    )


def _check(prefix: torch.Tensor, routed: torch.Tensor, shared: torch.Tensor) -> None:
    for name, t in (("prefix", prefix), ("routed", routed), ("shared", shared)):
        if t.dtype is not torch.bfloat16 or t.dim() != 2 or not t.is_contiguous() or not t.is_cuda:
            raise ValueError(f"{name} must be a contiguous 2D BF16 CUDA tensor")
    if routed.shape != prefix.shape or shared.shape != prefix.shape:
        raise ValueError("prefix, routed and shared must have one shape")
    if routed.device != prefix.device or shared.device != prefix.device:
        raise ValueError("prefix, routed and shared must be on one CUDA device")


def k31_moe_residual_add(
    prefix: torch.Tensor, routed: torch.Tensor, shared: torch.Tensor
) -> torch.Tensor:
    """``prefix + (routed + shared)`` (BF16 ``[M, N]``), bit-identical to the two PyTorch BF16 adds."""
    _check(prefix, routed, shared)
    out = torch.empty_like(prefix)
    moe_residual_add_into(prefix, routed, shared, out)
    return out


__all__ = ["MoeResidualAddKernel", "k31_moe_residual_add", "moe_residual_add_into"]
