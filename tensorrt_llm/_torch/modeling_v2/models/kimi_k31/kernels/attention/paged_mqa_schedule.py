# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-SM schedule of the paged MQA logits kernels, written in place (CuTe DSL, one CTA).

Schedule row ``r`` (of ``R``) holds ``n`` query tokens and ``len_r`` keys (``lens[r, col]``); it
is ``ceil(len_r / 256) * n`` units of work, numbered row by row. With ``total`` units, CTA ``s``
(``0 <= s <= num_sms``) of the logits kernel starts at unit ``w = s * (total // num_sms) +
min(s, total % num_sms)``: ``out[s] = (r * n, (w - first unit of r) // n)`` for the row ``r``
holding unit ``w``, and ``(R * n, 0)`` past the last unit. This is DeepGEMM's paged MQA logits
metadata, integer for integer.

One CTA of 1024 threads: each thread sums the units of a contiguous chunk of rows, one block
scan gives every row its inclusive prefix in shared memory, and threads ``0 .. num_sms`` binary
search their start unit. Launched with programmatic dependent launch (the lengths come from the
previous kernel).
"""

import functools

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch
from cutlass import Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

#: Keys per unit of schedule work (two 128-key compute blocks).
SPLIT_KV = 256
#: Largest number of schedule rows served (the shared-memory prefix).
MAX_ROWS = 16384
_THREADS = 1024
_SEARCH_STEPS = (MAX_ROWS + 1).bit_length()


def _asm(res, operands, text, constraints, *, loc=None, ip=None):
    return llvm.inline_asm(
        res,
        operands,
        text,
        constraints,
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def _iv(value, *, loc=None, ip=None):
    return Int32(value).ir_value(loc=loc, ip=ip)


def _lv(value, *, loc=None, ip=None):
    return Int64(value).ir_value(loc=loc, ip=ip)


@dsl_user_op
def _ldg_b32(addr, *, loc=None, ip=None):
    return Int32(
        _asm(
            T.i32(), [_lv(addr, loc=loc, ip=ip)], "ld.global.b32 $0, [$1];", "=r,l", loc=loc, ip=ip
        )
    )


@dsl_user_op
def _stg_v2_b32(addr, v0, v1, *, loc=None, ip=None):
    _asm(
        None,
        [_lv(addr, loc=loc, ip=ip), _iv(v0, loc=loc, ip=ip), _iv(v1, loc=loc, ip=ip)],
        "st.global.v2.b32 [$0], {$1, $2};",
        "l,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _sts_b32(addr, value, *, loc=None, ip=None):
    _asm(
        None,
        [_iv(addr, loc=loc, ip=ip), _iv(value, loc=loc, ip=ip)],
        "st.shared.b32 [$0], $1;",
        "r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _lds_b32(addr, *, loc=None, ip=None):
    return Int32(
        _asm(
            T.i32(), [_iv(addr, loc=loc, ip=ip)], "ld.shared.b32 $0, [$1];", "=r,r", loc=loc, ip=ip
        )
    )


@dsl_user_op
def _warp_inclusive_sum(value, *, loc=None, ip=None):
    body = "".join(
        f"shfl.sync.up.b32 t|p, x, {off}, 0, 0xffffffff;\n@p add.s32 x, x, t;\n"
        for off in (1, 2, 4, 8, 16)
    )
    return Int32(
        _asm(
            T.i32(),
            [_iv(value, loc=loc, ip=ip)],
            "{\n.reg .b32 x, t;\n.reg .pred p;\nmov.b32 x, $1;\n" + body + "mov.b32 $0, x;\n}",
            "=r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _griddepcontrol_wait(*, loc=None, ip=None):
    _asm(None, [], "griddepcontrol.wait;", "", loc=loc, ip=ip)


@dsl_user_op
def _griddepcontrol_launch_dependents(*, loc=None, ip=None):
    _asm(None, [], "griddepcontrol.launch_dependents;", "", loc=loc, ip=ip)


class PagedMqaScheduleKernel:
    """``out[s]`` for ``s = 0 .. num_sms`` (see the module docstring). A PDL launch: the length
    loads wait for the previous grid, then the dependents are released."""

    def __init__(self):
        @cute.struct
        class SharedStorage:
            prefix: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, MAX_ROWS], 16]
            warp_sum: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, 32], 16]

        self.shared_storage = SharedStorage

    @cute.jit
    def __call__(
        self,
        lens_addr: Int64,
        lens_stride: Int32,
        col: Int32,
        rows: Int32,
        next_n: Int32,
        num_sms: Int32,
        out_addr: Int64,
        stream: cuda.CUstream,
    ):
        self.kernel(lens_addr, lens_stride, col, rows, next_n, num_sms, out_addr).launch(
            grid=(1, 1, 1),
            block=(_THREADS, 1, 1),
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self,
        lens_addr: Int64,
        lens_stride: Int32,
        col: Int32,
        rows: Int32,
        next_n: Int32,
        num_sms: Int32,
        out_addr: Int64,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        storage = utils.SmemAllocator().allocate(self.shared_storage)
        pf = storage.prefix.data_ptr().toint()
        ws = storage.warp_sum.data_ptr().toint()
        _griddepcontrol_wait()
        chunk = (rows + _THREADS - 1) // _THREADS
        r0 = tidx * chunk
        r1 = r0 + chunk
        if r1 > rows:
            r1 = rows
        mine = Int32(0)
        for r in cutlass.range(r0, r1, unroll=1):
            length = _ldg_b32(lens_addr + (Int64(r) * Int64(lens_stride) + Int64(col)) * Int64(4))
            mine = mine + ((length + (SPLIT_KV - 1)) >> 8) * next_n
        _griddepcontrol_launch_dependents()
        incl = _warp_inclusive_sum(mine)
        if lane == 31:
            _sts_b32(ws + warp * 4, incl)
        cute.arch.barrier()
        run = incl - mine
        total = Int32(0)
        for w in cutlass.range_constexpr(_THREADS // 32):
            c = _lds_b32(ws + w * 4)
            if w < warp:
                run = run + c
            total = total + c
        for r in cutlass.range(r0, r1, unroll=1):
            length = _ldg_b32(lens_addr + (Int64(r) * Int64(lens_stride) + Int64(col)) * Int64(4))
            run = run + ((length + (SPLIT_KV - 1)) >> 8) * next_n
            _sts_b32(pf + r * 4, run)
        cute.arch.barrier()
        if tidx <= num_sms:
            quota = total // num_sms
            rem = total - quota * num_sms
            extra = tidx
            if extra > rem:
                extra = rem
            w = tidx * quota + extra
            # First row whose inclusive prefix exceeds w.
            lo = Int32(0)
            hi = rows
            for _ in cutlass.range_constexpr(_SEARCH_STEPS):
                if lo < hi:
                    mid = (lo + hi) >> 1
                    if _lds_b32(pf + mid * 4) <= w:
                        lo = mid + 1
                    else:
                        hi = mid
            first_tok = rows * next_n
            split = Int32(0)
            if lo < rows:
                first = Int32(0)
                if lo > 0:
                    first = _lds_b32(pf + (lo - 1) * 4)
                first_tok = lo * next_n
                split = (w - first) // next_n
            _stg_v2_b32(out_addr + Int64(tidx) * Int64(8), first_tok, split)


@functools.cache
def _compiled():
    kernel = PagedMqaScheduleKernel()
    a = Int64(1 << 20)
    return cute.compile(
        kernel,
        a,
        Int32(1),
        Int32(0),
        Int32(1),
        Int32(1),
        Int32(148),
        a,
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )


def paged_mqa_schedule_into(
    context_lens: torch.Tensor, num_sms: int, out: torch.Tensor
) -> torch.Tensor:
    """Write the schedule of int32 ``context_lens`` [R, n] (unit inner stride; row ``r``'s length
    in column ``n - 1``) into ``out`` (int32 contiguous [num_sms + 1, 2]) on the current stream.
    The first call compiles and must run outside CUDA-graph capture."""
    rows, next_n = context_lens.shape
    if rows > MAX_ROWS:
        raise ValueError(f"at most {MAX_ROWS} schedule rows, got {rows}")
    _compiled()(
        Int64(context_lens.data_ptr()),
        Int32(context_lens.stride(0) if rows > 1 else next_n),
        Int32(next_n - 1),
        Int32(rows),
        Int32(next_n),
        Int32(num_sms),
        Int64(out.data_ptr()),
        cuda.CUstream(torch.cuda.current_stream(context_lens.device).cuda_stream),
    )
    return out


__all__ = ["MAX_ROWS", "SPLIT_KV", "PagedMqaScheduleKernel", "paged_mqa_schedule_into"]
