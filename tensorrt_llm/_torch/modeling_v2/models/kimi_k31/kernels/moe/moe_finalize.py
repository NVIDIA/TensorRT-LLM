# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 routed experts: the top-k weighted sum of the experts' unfinalized output rows (CuTe DSL).

``out[t] = bf16(sum_k w[t, k] * y[e2p[t * K + k]])`` over the slots ``k`` whose expert this rank owns
(``e2p >= 0``), FP32 in slot order, the routing weight applied to the BF16 expert output (the
TRTLLM-Gen runner's finalize arithmetic class). One CTA per :data:`_ROWS_PER_CTA` rows; each thread
owns 8 consecutive columns (one 16-byte load per slot). One thread per row compacts the row's local
slots (position, weight) into shared memory in slot order; every thread then issues the loads of the
first :data:`_PREFETCH` local slots of all the CTA's rows before the first add, adds them in slot
order, and walks any further slots :data:`_GROUP` loads at a time. Programmatic dependent launch:
waits for the expert GEMMs' grid, releases its dependents after its stores.
"""

import functools

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch
from cutlass import Float32, Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

TOP_K = 16
#: Rows per CTA.
_ROWS_PER_CTA = 2
#: Local slots of every row loaded before the first add.
_PREFETCH = 2
#: Loads in flight per step over the further slots.
_GROUP = 4


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


@dsl_user_op
def _ldg_v4(addr, *, loc=None, ip=None):
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [Int64(addr).ir_value(loc=loc, ip=ip)],
        "ld.global.nc.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        loc=loc,
        ip=ip,
    )
    return tuple(Int32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip)) for i in range(4))


@dsl_user_op
def _ldg_v4_if(addr, pred, *, loc=None, ip=None):
    """16 bytes from global memory when ``pred`` != 0, else zeros."""
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [Int64(addr).ir_value(loc=loc, ip=ip), Int32(pred).ir_value(loc=loc, ip=ip)],
        "{\n.reg .pred q;\nsetp.ne.s32 q, $5, 0;\nmov.b32 $0, 0;\nmov.b32 $1, 0;\nmov.b32 $2, 0;\n"
        "mov.b32 $3, 0;\n@q ld.global.nc.v4.b32 {$0, $1, $2, $3}, [$4];\n}",
        "=&r,=&r,=&r,=&r,l,r",
        loc=loc,
        ip=ip,
    )
    return tuple(Int32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip)) for i in range(4))


@dsl_user_op
def _lds_v4(addr, *, loc=None, ip=None):
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [Int32(addr).ir_value(loc=loc, ip=ip)],
        "ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        loc=loc,
        ip=ip,
    )
    return tuple(Int32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip)) for i in range(4))


@dsl_user_op
def _lds_b32(addr, *, loc=None, ip=None):
    return Int32(
        _asm(
            T.i32(),
            [Int32(addr).ir_value(loc=loc, ip=ip)],
            "ld.shared.b32 $0, [$1];",
            "=r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _sts_b32(addr, value, *, loc=None, ip=None):
    _asm(
        None,
        [Int32(addr).ir_value(loc=loc, ip=ip), Int32(value).ir_value(loc=loc, ip=ip)],
        "st.shared.b32 [$0], $1;",
        "r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _f32_bits(v, *, loc=None, ip=None):
    """The FP32 whose bits are ``v``."""
    return Float32(
        _asm(
            T.f32(), [Int32(v).ir_value(loc=loc, ip=ip)], "mov.b32 $0, $1;", "=f,r", loc=loc, ip=ip
        )
    )


@dsl_user_op
def _stg_v4(addr, a, b, c, d, *, loc=None, ip=None):
    _asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip)]
        + [Int32(v).ir_value(loc=loc, ip=ip) for v in (a, b, c, d)],
        "st.global.v4.b32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _bf16x2(a, b, *, loc=None, ip=None):
    return Int32(
        _asm(
            T.i32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            "cvt.rn.bf16x2.f32 $0, $2, $1;",
            "=r,f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _lo_f32(v, *, loc=None, ip=None):
    """The low BF16 of a packed pair as FP32."""
    return Float32(
        _asm(
            T.f32(),
            [Int32(v).ir_value(loc=loc, ip=ip)],
            "shl.b32 $0, $1, 16;",
            "=f,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _hi_f32(v, *, loc=None, ip=None):
    """The high BF16 of a packed pair as FP32."""
    return Float32(
        _asm(
            T.f32(),
            [Int32(v).ir_value(loc=loc, ip=ip)],
            "and.b32 $0, $1, 0xFFFF0000;",
            "=f,r",
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


class MoeFinalizeKernel:
    """``out [M, hidden]`` BF16 from ``y [perm rows, hidden]`` BF16, ``e2p [M * top_k]`` int32 and
    ``w [M, top_k]`` BF16; see the module docstring."""

    def __init__(self, hidden: int, top_k: int = TOP_K):
        assert hidden % 8 == 0 and top_k % 4 == 0
        self.hidden = hidden
        self.top_k = top_k
        self.threads = hidden // 8
        self.stride = (
            top_k + _GROUP
        )  # shared entries per row (group loads may read past the row's count)
        R = _ROWS_PER_CTA

        @cute.struct
        class SharedStorage:
            pos: cute.struct.Align[cute.struct.MemRange[Int32, R * (top_k + _GROUP)], 16]
            wgt: cute.struct.Align[cute.struct.MemRange[Int32, R * (top_k + _GROUP)], 16]
            cnt: cute.struct.Align[cute.struct.MemRange[Int32, max(4, R)], 16]

        self.shared_storage = SharedStorage

    @cute.jit
    def __call__(self, rows: Int32, y_addr: Int64, e2p_addr: Int64, w_addr: Int64, out_addr: Int64,
                 stream: cuda.CUstream):  # fmt: skip
        self.kernel(rows, y_addr, e2p_addr, w_addr, out_addr).launch(
            grid=((rows + _ROWS_PER_CTA - 1) // _ROWS_PER_CTA, 1, 1),
            block=[self.threads, 1, 1],
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(self, rows: Int32, y_addr: Int64, e2p_addr: Int64, w_addr: Int64, out_addr: Int64):
        tidx, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        R, S, P, G = _ROWS_PER_CTA, self.stride, _PREFETCH, _GROUP
        row0 = Int32(bid) * R
        storage = utils.SmemAllocator().allocate(self.shared_storage)
        pos_s = storage.pos.data_ptr().toint()
        wgt_s = storage.wgt.data_ptr().toint()
        cnt_s = storage.cnt.data_ptr().toint()
        _griddepcontrol_wait()
        if tidx < R:
            # row row0 + tidx: its local slots in slot order, position and weight (FP32 bits of the BF16 weight)
            row = row0 + tidx
            cnt = Int32(0)
            if row < rows:
                ent = []
                for q in cutlass.range_constexpr(self.top_k // 4):
                    ent.extend(_ldg_v4(e2p_addr + (Int64(row) * self.top_k + q * 4) * 4))
                wpk = []
                for q in cutlass.range_constexpr(self.top_k // 8):
                    wpk.extend(_ldg_v4(w_addr + (Int64(row) * self.top_k + q * 8) * 2))
                base = tidx * (S * 4)
                for k in cutlass.range_constexpr(self.top_k):
                    if ent[k] >= 0:
                        _sts_b32(pos_s + base + cnt * 4, ent[k])
                        if cutlass.const_expr(k % 2 == 0):
                            _sts_b32(wgt_s + base + cnt * 4, wpk[k // 2] << 16)
                        else:
                            _sts_b32(wgt_s + base + cnt * 4, wpk[k // 2] & Int32(-65536))
                        cnt = cnt + 1
            _sts_b32(cnt_s + tidx * 4, cnt)
        cute.arch.barrier()
        col = Int64(tidx * 8)
        cnts = []
        for q in cutlass.range_constexpr((R + 3) // 4):
            cnts.extend(_lds_v4(cnt_s + q * 16))
        # the first P local slots of every row in flight at once
        pre = []
        for r in cutlass.range_constexpr(R):
            ppos = [_lds_b32(pos_s + r * (S * 4) + i * 4) for i in range(P)]
            for i in cutlass.range_constexpr(P):
                ok = Int32(0)
                if i < cnts[r]:
                    ok = Int32(1)
                pre.append(_ldg_v4_if(y_addr + (Int64(ppos[i]) * self.hidden + col) * 2, ok))
        for r in cutlass.range_constexpr(R):
            row = row0 + r
            if row < rows:
                n = cnts[r]
                acc = cute.make_rmem_tensor((8,), Float32)
                for i in cutlass.range_constexpr(8):
                    acc[i] = Float32(0.0)
                for i in cutlass.range_constexpr(P):
                    if i < n:
                        w = _f32_bits(_lds_b32(wgt_s + r * (S * 4) + i * 4))
                        for q in cutlass.range_constexpr(4):
                            acc[2 * q] = acc[2 * q] + w * _lo_f32(pre[r * P + i][q])
                            acc[2 * q + 1] = acc[2 * q + 1] + w * _hi_f32(pre[r * P + i][q])
                for g in cutlass.range(P, n, G, unroll=1):
                    gpos = []
                    for i in cutlass.range_constexpr(G):
                        gpos.append(_lds_b32(pos_s + r * (S * 4) + (g + i) * 4))
                    vals = []
                    for i in cutlass.range_constexpr(G):
                        ok = Int32(0)
                        if g + i < n:
                            ok = Int32(1)
                        vals.append(
                            _ldg_v4_if(y_addr + (Int64(gpos[i]) * self.hidden + col) * 2, ok)
                        )
                    for i in cutlass.range_constexpr(G):
                        if g + i < n:
                            w = _f32_bits(_lds_b32(wgt_s + r * (S * 4) + (g + i) * 4))
                            for q in cutlass.range_constexpr(4):
                                acc[2 * q] = acc[2 * q] + w * _lo_f32(vals[i][q])
                                acc[2 * q + 1] = acc[2 * q + 1] + w * _hi_f32(vals[i][q])
                _stg_v4(
                    out_addr + (Int64(row) * self.hidden + col) * 2,
                    _bf16x2(acc[0], acc[1]),
                    _bf16x2(acc[2], acc[3]),
                    _bf16x2(acc[4], acc[5]),
                    _bf16x2(acc[6], acc[7]),
                )
        _griddepcontrol_launch_dependents()


@functools.cache
def _compiled(hidden: int, top_k: int):
    a = Int64(1 << 20)
    return cute.compile(
        MoeFinalizeKernel(hidden, top_k),
        Int32(1),
        a,
        a,
        a,
        a,
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )


def run_moe_finalize(
    y: torch.Tensor, e2p: torch.Tensor, w: torch.Tensor, out: torch.Tensor
) -> None:
    """``out [M, hidden]`` = top-k weighted sum of ``y`` rows (see the module docstring); on the current stream.
    ``e2p`` and ``w`` (BF16) rows are read with 16-byte loads (16-byte aligned base pointers)."""
    rows, hidden = out.shape
    top_k = w.shape[1]
    if rows == 0:
        return
    assert w.dtype == torch.bfloat16 and w.is_contiguous() and e2p.is_contiguous()
    assert e2p.data_ptr() % 16 == 0 and w.data_ptr() % 16 == 0 and y.data_ptr() % 16 == 0
    _compiled(int(hidden), int(top_k))(
        Int32(rows),
        Int64(y.data_ptr()),
        Int64(e2p.data_ptr()),
        Int64(w.data_ptr()),
        Int64(out.data_ptr()),
        cuda.CUstream(torch.cuda.current_stream(y.device).cuda_stream),
    )


__all__ = ["MoeFinalizeKernel", "run_moe_finalize"]
