# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hyper-connection post-block injection fused with the next site's grouped
gemma RMS norm: one Triton row kernel writes the updated multi-stream residual
and its normalized copy in the same pass. After a MoE block it also forms the
block output itself, from the routed experts' output, the shared expert's
output and its gate logit (the shared-expert combine).

Tier 5 (Triton written for the unified target). No ``torch.ops.trtllm.*`` op or
FlashInfer callable accumulates a block output into ``hc_count`` streams with
per-stream gates and normalizes each stream with its own RMS in one launch; the
two-launch composition (``qwen4_exp_post_inject`` + four
``flashinfer_gemma_rmsnorm`` calls and a concat) is what this replaces, and
with ``shared`` / ``gate_logits`` also the ``fused_sigmoid_gate_mul_add``
launch before it -- and with ``topk_weights`` / ``topk_rows`` the MoE runner's
finalize launch as well.

The decode MoE sites (the finalize fold below `_FEW_WARPS_MIN_ROWS` rows) run a
CuTe DSL kernel instead (Tier 6, written for the unified target), bitwise the Triton
kernel's outputs there.
"""

from __future__ import annotations

import struct
from typing import Optional, Tuple

import cutlass
import cutlass.cute as cute
import torch
import triton
import triton.language as tl
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import T, dsl_user_op

from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl
from tensorrt_llm._torch.kernels.blackwell.utils import make_ptr

from .._op import Arch, Cell, OpWrapper

#: Trigger dependents at kernel start (`griddepcontrol.launch_dependents`); a no-op
#: unless the next kernel is launched with programmatic dependent launch.
_PDL_TRIGGER = get_env_enable_pdl()

#: Rows from which a wide (`hidden >= 2048`) launch runs two warps a program
#: instead of eight: prefill-sized grids are bandwidth-bound and run faster
#: with fewer, longer threads; decode-sized ones keep eight.
_FEW_WARPS_MIN_ROWS = 512


@triton.jit
def _hc_inject_norm_kernel(
    hc_ptr,
    block_ptr,
    inject_ptr,
    shared_ptr,
    gate_ptr,
    topk_w_ptr,
    topk_row_ptr,
    weight_ptr,
    hc_out_ptr,
    normed_ptr,
    stride_hc,
    stride_block,
    stride_inject,
    stride_shared,
    stride_gate,
    stride_topk_w,
    stride_topk_row,
    stride_out,
    stride_normed,
    eps,
    HC_COUNT: tl.constexpr,
    HIDDEN: tl.constexpr,
    BLOCK: tl.constexpr,
    HAS_INJECT: tl.constexpr,
    HAS_SHARED: tl.constexpr,
    TOPK: tl.constexpr,
    STREAMS_PER_PROGRAM: tl.constexpr,
    PDL_TRIGGER: tl.constexpr,
):
    if PDL_TRIGGER:
        # Release a programmatic-dependent-launch consumer (the decode mixer's
        # first projection) at once: it streams its static weights while this
        # kernel runs and waits for the grid to finish before reading `normed`.
        tl.extra.cuda.gdc_launch_dependents()
    # One program per `STREAMS_PER_PROGRAM` streams of a row; a stream's `HIDDEN`
    # columns are one RMS group. A row's programs are adjacent in launch order, so
    # the row's block output (and the combine's operands) come from HBM once and
    # from L2 for the others; with several streams a program forms it once for them.
    pid = tl.program_id(0).to(tl.int64)
    row = pid // (HC_COUNT // STREAMS_PER_PROGRAM)
    first_stream = ((pid % (HC_COUNT // STREAMS_PER_PROGRAM)) * STREAMS_PER_PROGRAM).to(tl.int32)
    cols = tl.arange(0, BLOCK)
    mask = cols < HIDDEN
    if HAS_INJECT:
        if TOPK > 0:
            # The routed experts' finalize: this row's TOPK permuted FC2 rows weighted
            # by their routing weights, fp32 FMAs in slot order from zero (a slot mapped
            # to -1 is skipped), rounded once to the rows' dtype.
            b = tl.zeros([BLOCK], dtype=tl.float32)
            for k in tl.static_range(TOPK):
                prow = tl.load(topk_row_ptr + row * stride_topk_row + k)
                wk = tl.load(topk_w_ptr + row * stride_topk_w + k).to(tl.float32)
                r = tl.load(
                    block_ptr + tl.maximum(prow, 0).to(tl.int64) * stride_block + cols,
                    mask=mask,
                    other=0.0,
                ).to(tl.float32)
                b = tl.where(prow >= 0, tl.fma(r, wk, b), b)
            b = b.to(block_ptr.dtype.element_ty).to(tl.float32)
        else:
            b = tl.load(block_ptr + row * stride_block + cols, mask=mask, other=0.0).to(tl.float32)
        if HAS_SHARED:
            # The MoE block output, `routed + sigmoid(gate) * shared` in fp32, rounded
            # once to the routed output's dtype -- the value that block used to store.
            sg = tl.sigmoid(tl.load(gate_ptr + row * stride_gate).to(tl.float32))
            sh = tl.load(shared_ptr + row * stride_shared + cols, mask=mask, other=0.0)
            b = (b + sg * sh.to(tl.float32)).to(block_ptr.dtype.element_ty).to(tl.float32)
    for j in tl.static_range(STREAMS_PER_PROGRAM):
        stream = first_stream + j
        offs = stream * HIDDEN + cols
        x = tl.load(hc_ptr + row * stride_hc + offs, mask=mask, other=0.0).to(tl.float32)
        if HAS_INJECT:
            g = tl.load(inject_ptr + row * stride_inject + stream).to(tl.float32)
            # Rounded once on the store, like `addcmul`; the norm reads the rounded
            # residual, which is the value every later site sees.
            xr = (x + b * g).to(hc_out_ptr.dtype.element_ty)
            tl.store(hc_out_ptr + row * stride_out + offs, xr, mask=mask)
            x = xr.to(tl.float32)
        rstd = tl.rsqrt(tl.sum(x * x, axis=0) / HIDDEN + eps)
        w = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        y = x * rstd * (1.0 + w)
        tl.store(
            normed_ptr + row * stride_normed + offs, y.to(normed_ptr.dtype.element_ty), mask=mask
        )


# --------------------------------------------------------------------------------------
# The decode MoE sites in CuTe DSL. There the Triton kernel runs two programs a row, each
# gathering all `k` permuted FC2 rows and normalizing its two streams one after the other
# (6.5 us at 128 rows in the con128 decode step on GB300). Here the row's two CTAs are a
# cluster: CTA j forms the block output's columns [j * H / 2, (j + 1) * H / 2) -- the
# finalize's slot-order FMA chain and the combine, the Triton kernel's instructions -- and
# stores them into both CTAs' shared memory; after a cluster barrier each CTA reads the
# whole block row locally and accumulates and normalizes its two streams together. A CTA
# is 256 threads in the Triton kernel's layout at a 4096 block and eight warps (thread t:
# columns [8t, 8t + 8) and, below the hidden, [2048 + 8t, 2048 + 8t + 8)), and the sum of
# squares is its reduction tree: `hc_out` and `normed` are bitwise the Triton kernel's.
# --------------------------------------------------------------------------------------

_CUTE_THREADS = 256
_CUTE_WARPS = _CUTE_THREADS // 32
_CUTE_VEC = 8
#: The Triton kernel's block at these hiddens is twice this: a thread's second chunk starts here.
_CUTE_SPLIT = _CUTE_THREADS * _CUTE_VEC
#: Streams a CTA, as the Triton fold's program; a row's `hc_count // 2` CTAs are one cluster.
_CUTE_SPP = 2


def _iv(x, loc, ip):
    return x.ir_value(loc=loc, ip=ip)


def _asm(ret, args, text, constraints, loc, ip, side_effects=False):
    return llvm.inline_asm(
        ret,
        [_iv(a, loc, ip) for a in args],
        text,
        constraints,
        has_side_effects=side_effects,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def _asm4(args, text, constraints, loc, ip, side_effects=False):
    ty = T.i32()
    res = _asm(
        llvm.StructType.get_literal([ty] * 4), args, text, constraints, loc, ip, side_effects
    )
    return [cutlass.Int32(llvm.extractvalue(ty, res, [i], loc=loc, ip=ip)) for i in range(4)]


@dsl_user_op
def _ldg_v4(addr, pred, *, loc=None, ip=None):
    """16 bytes, or zeros where `pred` is 0 (nothing read)."""
    return _asm4(
        [cutlass.Int64(addr), cutlass.Int32(pred)],
        "{ .reg .pred p; setp.ne.b32 p, $5, 0; mov.b32 $0, 0; mov.b32 $1, 0; mov.b32 $2, 0; "
        "mov.b32 $3, 0; @p ld.global.v4.b32 {$0, $1, $2, $3}, [$4]; }",
        "=r,=r,=r,=r,l,r",
        loc,
        ip,
        side_effects=True,
    )


@dsl_user_op
def _stg_v4(addr, pred, a, b, c, d, *, loc=None, ip=None):
    """16 bytes where `pred` is not 0."""
    _asm(
        T.i32(),
        [cutlass.Int64(addr), cutlass.Int32(pred)] + [cutlass.Int32(v) for v in (a, b, c, d)],
        "{ .reg .pred p; setp.ne.b32 p, $2, 0; @p st.global.v4.b32 [$1], {$3, $4, $5, $6}; "
        "mov.u32 $0, 0; }",
        "=r,l,r,r,r,r,r",
        loc,
        ip,
        side_effects=True,
    )


@dsl_user_op
def _mapa(addr, rank, *, loc=None, ip=None):
    """This CTA's shared-memory address `addr` in cluster CTA `rank`'s window."""
    return cutlass.Int32(
        _asm(
            T.i32(),
            [cutlass.Int32(addr), cutlass.Int32(rank)],
            "mapa.shared::cluster.u32 $0, $1, $2;",
            "=r,r,r",
            loc,
            ip,
        )
    )


@dsl_user_op
def _st_dsmem_v4(addr, pred, a, b, c, d, *, loc=None, ip=None):
    """16 bytes into a cluster CTA's shared memory where `pred` is not 0."""
    _asm(
        T.i32(),
        [cutlass.Int32(addr), cutlass.Int32(pred)] + [cutlass.Int32(v) for v in (a, b, c, d)],
        "{ .reg .pred p; setp.ne.b32 p, $2, 0; @p st.shared::cluster.v4.b32 [$1], "
        "{$3, $4, $5, $6}; mov.u32 $0, 0; }",
        "=r,r,r,r,r,r,r",
        loc,
        ip,
        side_effects=True,
    )


@dsl_user_op
def _lds_v4(addr, pred, *, loc=None, ip=None):
    """16 bytes of this CTA's shared memory, or zeros where `pred` is 0."""
    return _asm4(
        [cutlass.Int32(addr), cutlass.Int32(pred)],
        "{ .reg .pred p; setp.ne.b32 p, $5, 0; mov.b32 $0, 0; mov.b32 $1, 0; mov.b32 $2, 0; "
        "mov.b32 $3, 0; @p ld.shared.v4.b32 {$0, $1, $2, $3}, [$4]; }",
        "=r,=r,=r,=r,r,r",
        loc,
        ip,
        side_effects=True,
    )


@dsl_user_op
def _bf16_lo(w, *, loc=None, ip=None):
    """The low bf16 of a word as fp32 (exact)."""
    return cutlass.Float32(
        _asm(
            T.f32(),
            [cutlass.Int32(w)],
            "{ .reg .b32 t; shl.b32 t, $1, 16; mov.b32 $0, t; }",
            "=f,r",
            loc,
            ip,
        )
    )


@dsl_user_op
def _bf16_hi(w, *, loc=None, ip=None):
    """The high bf16 of a word as fp32 (exact)."""
    return cutlass.Float32(
        _asm(
            T.f32(),
            [cutlass.Int32(w)],
            "{ .reg .b32 t; and.b32 t, $1, 0xFFFF0000; mov.b32 $0, t; }",
            "=f,r",
            loc,
            ip,
        )
    )


@dsl_user_op
def _bf16x2(lo, hi, *, loc=None, ip=None):
    """Two fp32 values rounded (RN) into one bf16x2 word, `lo` low."""
    return cutlass.Int32(
        _asm(
            T.i32(),
            [cutlass.Float32(hi), cutlass.Float32(lo)],
            "cvt.rn.bf16x2.f32 $0, $1, $2;",
            "=r,f,f",
            loc,
            ip,
        )
    )


@dsl_user_op
def _round_bf16(x, *, loc=None, ip=None):
    """fp32 -> bf16 (RN) -> fp32."""
    return cutlass.Float32(
        _asm(
            T.f32(),
            [cutlass.Float32(x)],
            "{ .reg .b16 h; cvt.rn.bf16.f32 h, $1; cvt.f32.bf16 $0, h; }",
            "=f,f",
            loc,
            ip,
        )
    )


@dsl_user_op
def _fma(a, b, c, *, loc=None, ip=None):
    return cutlass.Float32(
        _asm(
            T.f32(),
            [cutlass.Float32(a), cutlass.Float32(b), cutlass.Float32(c)],
            "fma.rn.f32 $0, $1, $2, $3;",
            "=f,f,f,f",
            loc,
            ip,
        )
    )


@dsl_user_op
def _fma_if(slot, a, b, c, *, loc=None, ip=None):
    """`slot >= 0 ? fma(a, b, c) : c` (the finalize skips a slot mapped to -1)."""
    return cutlass.Float32(
        _asm(
            T.f32(),
            [cutlass.Int32(slot), cutlass.Float32(a), cutlass.Float32(b), cutlass.Float32(c)],
            "{ .reg .pred p; .reg .f32 t; fma.rn.f32 t, $2, $3, $4; setp.ge.s32 p, $1, 0; "
            "selp.f32 $0, t, $4, p; }",
            "=f,r,f,f,f",
            loc,
            ip,
        )
    )


@dsl_user_op
def _mul(a, b, *, loc=None, ip=None):
    return cutlass.Float32(
        _asm(
            T.f32(),
            [cutlass.Float32(a), cutlass.Float32(b)],
            "mul.rn.f32 $0, $1, $2;",
            "=f,f,f",
            loc,
            ip,
        )
    )


@dsl_user_op
def _add(a, b, *, loc=None, ip=None):
    return cutlass.Float32(
        _asm(
            T.f32(),
            [cutlass.Float32(a), cutlass.Float32(b)],
            "add.rn.f32 $0, $1, $2;",
            "=f,f,f",
            loc,
            ip,
        )
    )


@dsl_user_op
def _sigmoid(x, *, loc=None, ip=None):
    """`tl.sigmoid` as the Triton kernel compiles it: `mul` by -log2(e), `ex2.approx`, `add`
    1, `div.full`."""
    return cutlass.Float32(
        _asm(
            T.f32(),
            [cutlass.Float32(x)],
            "{ .reg .f32 t; mul.f32 t, $1, 0fBFB8AA3B; ex2.approx.f32 t, t; "
            "add.f32 t, t, 0f3F800000; div.full.f32 $0, 0f3F800000, t; }",
            "=f,f",
            loc,
            ip,
        )
    )


@dsl_user_op
def _rstd(ss, eps, hidden_hex: str, *, loc=None, ip=None):
    """`rsqrt(ss / hidden + eps)` as the Triton kernel's `div.full`, `add`, `rsqrt.approx.ftz`:
    the `add` has no rounding mode there, so ptxas fuses it with the division (a multiply by
    the reciprocal) into one FFMA -- here the same."""
    return cutlass.Float32(
        _asm(
            T.f32(),
            [cutlass.Float32(ss), cutlass.Float32(eps)],
            "{ .reg .f32 t; div.full.f32 t, $1, " + hidden_hex + "; add.f32 t, t, $2; "
            "rsqrt.approx.ftz.f32 $0, t; }",
            "=f,f,f",
            loc,
            ip,
        )
    )


def _vals(words):
    """Four bf16x2 words -> their eight values as fp32, in column order."""
    out = []
    for w in words:
        out += [_bf16_lo(w), _bf16_hi(w)]
    return out


class _FoldNormKernel:
    """CTA (j, row): the row's streams 2j and 2j + 1; the row's CTAs one cluster."""

    def __init__(self, hc_count: int, hidden: int, top_k: int, pdl_trigger: bool):
        self.hc, self.hidden, self.topk, self.pdl = hc_count, hidden, top_k, pdl_trigger
        self.ctas = hc_count // _CUTE_SPP
        # The block-output columns a CTA forms, one 16-byte vector a thread.
        self.per = hidden // self.ctas
        self.n_fold = self.per // _CUTE_VEC
        # Threads with a second chunk (columns [2048, hidden)).
        self.n_b = (hidden - _CUTE_SPLIT) // _CUTE_VEC
        self.hidden_hex = "0f%08X" % struct.unpack("<I", struct.pack("<f", float(hidden)))[0]

    @cute.jit
    def __call__(
        self,
        hc_t: cute.Tensor,
        block_ptr: cute.Pointer,
        inj_ptr: cute.Pointer,
        sh_ptr: cute.Pointer,
        gate_ptr: cute.Pointer,
        tw_ptr: cute.Pointer,
        tr_ptr: cute.Pointer,
        w_ptr: cute.Pointer,
        out_ptr: cute.Pointer,
        nrm_ptr: cute.Pointer,
        rows: cutlass.Int32,
        s_hc: cutlass.Int32,
        s_block: cutlass.Int32,
        s_inj: cutlass.Int32,
        s_sh: cutlass.Int32,
        s_gate: cutlass.Int32,
        s_tw: cutlass.Int32,
        s_tr: cutlass.Int32,
        eps: cutlass.Float32,
        stream,
    ):
        inj = cute.make_tensor(inj_ptr, cute.make_layout((rows, self.hc), stride=(s_inj, 1)))
        gate = cute.make_tensor(gate_ptr, cute.make_layout((rows,), stride=(s_gate,)))
        tw = cute.make_tensor(tw_ptr, cute.make_layout((rows, self.topk), stride=(s_tw, 1)))
        tr = cute.make_tensor(tr_ptr, cute.make_layout((rows, self.topk), stride=(s_tr, 1)))
        # `hyper_input` arrives as a tensor: the TVM-FFI entry takes its stream from a GPU tensor.
        self.kernel(
            hc_t.iterator.toint(),
            block_ptr.toint(),
            inj,
            sh_ptr.toint(),
            gate,
            tw,
            tr,
            w_ptr.toint(),
            out_ptr.toint(),
            nrm_ptr.toint(),
            s_hc,
            s_block,
            s_sh,
            eps,
        ).launch(
            grid=[self.ctas, rows, 1],
            block=[_CUTE_THREADS, 1, 1],
            cluster=[self.ctas, 1, 1],
            smem=_CUTE_SPP * _CUTE_WARPS * 4 + self.hidden * 2 + 64,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        hc_base: cutlass.Int64,
        block_base: cutlass.Int64,
        mInj: cute.Tensor,
        sh_base: cutlass.Int64,
        mGate: cute.Tensor,
        mTw: cute.Tensor,
        mTr: cute.Tensor,
        w_base: cutlass.Int64,
        out_base: cutlass.Int64,
        nrm_base: cutlass.Int64,
        s_hc: cutlass.Int32,
        s_block: cutlass.Int32,
        s_sh: cutlass.Int32,
        eps: cutlass.Float32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        rank, row, _ = cute.arch.block_idx()
        if cutlass.const_expr(self.pdl):
            cute.arch.griddepcontrol_launch_dependents()
        # Phase 0 of the cluster barrier: no CTA writes a peer's shared memory before the
        # peer has started.
        cute.arch.cluster_arrive_relaxed()
        width = self.hc * self.hidden
        smem = cutlass.utils.SmemAllocator()
        red = smem.allocate_tensor(
            cutlass.Float32,
            cute.make_layout((_CUTE_SPP, _CUTE_WARPS), stride=(_CUTE_WARPS, 1)),
            byte_alignment=16,
        )
        bsm = smem.allocate_tensor(
            cutlass.BFloat16, cute.make_layout((self.hidden,), stride=(1,)), byte_alignment=16
        )
        bsm_addr = cutlass.Int32(bsm.iterator.toint())
        row64 = cutlass.Int64(row)
        one = cutlass.Int32(1)
        c_a = tidx * _CUTE_VEC
        c_b = _CUTE_SPLIT + tidx * _CUTE_VEC
        has_b = cutlass.Int32(0)
        if tidx < self.n_b:
            has_b = one
        is_f = cutlass.Int32(0)
        if tidx < self.n_fold:
            is_f = one

        # The routing first (the gathers wait on it), then the streams' residual, the norm
        # weights and the inject gates.
        rows_k = [cutlass.Int32(mTr[row, k]) for k in range(self.topk)]
        wts = [mTw[row, k].to(cutlass.Float32) for k in range(self.topk)]
        gate_v = mGate[row].to(cutlass.Float32)
        xa, xb, wa, wb, gs = [], [], [], [], []
        for s in cutlass.range_constexpr(_CUTE_SPP):
            stream = rank * _CUTE_SPP + s
            hrow = hc_base + (row64 * s_hc + stream * self.hidden) * 2
            xa.append(_ldg_v4(hrow + c_a * 2, one))
            xb.append(_ldg_v4(hrow + c_b * 2, has_b))
            wrow = w_base + cutlass.Int64(stream * self.hidden) * 2
            wa.append(_ldg_v4(wrow + c_a * 2, one))
            wb.append(_ldg_v4(wrow + c_b * 2, has_b))
            gs.append(mInj[row, stream].to(cutlass.Float32))

        # This CTA's columns of the block output: the finalize (fp32 FMAs in slot order
        # from zero, a slot mapped to -1 skipped) rounded once, then the combine
        # `fma(sigmoid(gate), shared, routed)` rounded once.
        fc = rank * self.per + cutlass.min(tidx, cutlass.Int32(self.n_fold - 1)) * _CUTE_VEC
        sg = _sigmoid(gate_v)
        shw = _ldg_v4(sh_base + (row64 * s_sh + fc) * 2, is_f)
        gath = [
            _ldg_v4(
                block_base
                + (cutlass.Int64(cutlass.max(rows_k[k], cutlass.Int32(0))) * s_block + fc) * 2,
                is_f,
            )
            for k in range(self.topk)
        ]
        acc = [cutlass.Float32(0.0)] * _CUTE_VEC
        for k in cutlass.range_constexpr(self.topk):
            vals = _vals(gath[k])
            for e in cutlass.range_constexpr(_CUTE_VEC):
                acc[e] = _fma_if(rows_k[k], vals[e], wts[k], acc[e])
        sh = _vals(shw)
        bv = [_round_bf16(_fma(sg, sh[e], _round_bf16(acc[e]))) for e in range(_CUTE_VEC)]
        words = [_bf16x2(bv[2 * i], bv[2 * i + 1]) for i in range(4)]
        cute.arch.cluster_wait()
        for r in cutlass.range_constexpr(self.ctas):
            _st_dsmem_v4(_mapa(bsm_addr + fc * 2, r), is_f, *words)
        # Release / acquire at cluster scope: every CTA then holds the whole block row.
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        ba = _vals(_lds_v4(bsm_addr + c_a * 2, one))
        bb = _vals(
            _lds_v4(bsm_addr + cutlass.min(c_b, cutlass.Int32(self.hidden - _CUTE_VEC)) * 2, has_b)
        )

        # The streams: `fma(inject, block, x)` rounded once into `hc_out`, and the sum of
        # squares of the rounded residual in the Triton kernel's tree.
        warp = tidx // 32
        xrs = []
        for s in cutlass.range_constexpr(_CUTE_SPP):
            stream = rank * _CUTE_SPP + s
            orow = out_base + (row64 * width + stream * self.hidden) * 2
            xa_v, xb_v = _vals(xa[s]), _vals(xb[s])
            ra = [_round_bf16(_fma(gs[s], ba[e], xa_v[e])) for e in range(_CUTE_VEC)]
            rb = [_round_bf16(_fma(gs[s], bb[e], xb_v[e])) for e in range(_CUTE_VEC)]
            _stg_v4(orow + c_a * 2, one, *[_bf16x2(ra[2 * i], ra[2 * i + 1]) for i in range(4)])
            _stg_v4(orow + c_b * 2, has_b, *[_bf16x2(rb[2 * i], rb[2 * i + 1]) for i in range(4)])
            xrs.append((ra, rb))
            # Per lane L of the bf16 pairs (L = 0 the even columns): (fma(a0, a0, a1 * a1) +
            # fma(a2, a2, a3 * a3)) + (the same over the second chunk), then the two lanes.
            lanes = []
            for lane in cutlass.range_constexpr(2):
                a = [ra[2 * k + lane] for k in range(4)]
                b = [rb[2 * k + lane] for k in range(4)]
                lanes.append(
                    _add(
                        _add(
                            _fma(a[0], a[0], _mul(a[1], a[1])), _fma(a[2], a[2], _mul(a[3], a[3]))
                        ),
                        _add(
                            _fma(b[0], b[0], _mul(b[1], b[1])), _fma(b[2], b[2], _mul(b[3], b[3]))
                        ),
                    )
                )
            v = _add(lanes[0], lanes[1])
            for off in cutlass.range_constexpr(5):
                v = _add(v, cute.arch.shuffle_sync_bfly(v, offset=16 >> off))
            if tidx % 32 == 0:
                red[s, warp] = v
        cute.arch.sync_threads()
        for s in cutlass.range_constexpr(_CUTE_SPP):
            stream = rank * _CUTE_SPP + s
            v = red[s, tidx % _CUTE_WARPS]
            for off in cutlass.range_constexpr(3):
                v = _add(v, cute.arch.shuffle_sync_bfly(v, offset=4 >> off))
            rstd = _rstd(v, eps, self.hidden_hex)
            ra, rb = xrs[s]
            wa_v, wb_v = _vals(wa[s]), _vals(wb[s])
            ya = [
                _mul(_mul(rstd, ra[e]), _add(wa_v[e], cutlass.Float32(1.0)))
                for e in range(_CUTE_VEC)
            ]
            yb = [
                _mul(_mul(rstd, rb[e]), _add(wb_v[e], cutlass.Float32(1.0)))
                for e in range(_CUTE_VEC)
            ]
            nrow = nrm_base + (row64 * width + stream * self.hidden) * 2
            _stg_v4(nrow + c_a * 2, one, *[_bf16x2(ya[2 * i], ya[2 * i + 1]) for i in range(4)])
            _stg_v4(nrow + c_b * 2, has_b, *[_bf16x2(yb[2 * i], yb[2 * i + 1]) for i in range(4)])


_FOLD_NORM: dict = {}


def _fold_cute_ok(
    hyper_input,
    block_out,
    inject,
    norm_weight,
    hc_count,
    shared,
    gate_logits,
    topk_weights,
    topk_rows,
) -> bool:
    """The decode MoE site this kernel certifies (the Triton kernel's eight-warp form with
    16-byte vectors); anything else runs the Triton kernel."""
    rows, width = hyper_input.shape
    hidden = width // hc_count
    bf = torch.bfloat16
    return (
        0 < rows < _FEW_WARPS_MIN_ROWS
        and hc_count == 4
        and _CUTE_SPLIT < hidden <= 2 * _CUTE_SPLIT
        and hidden % (_CUTE_SPP * _CUTE_VEC) == 0
        and hyper_input.dtype == block_out.dtype == shared.dtype == bf
        and inject.dtype == gate_logits.dtype == topk_weights.dtype == norm_weight.dtype == bf
        and topk_rows.dtype == torch.int32
        and all(
            t.data_ptr() % 16 == 0 and t.stride(0) % 16 == 0
            for t in (hyper_input, block_out, shared)
        )
        and norm_weight.data_ptr() % 16 == 0
    )


def _fold_cute(
    hyper_input,
    block_out,
    inject,
    norm_weight,
    eps,
    hc_count,
    shared,
    gate_logits,
    topk_weights,
    topk_rows,
    hc_out,
    normed,
) -> None:
    rows, width = hyper_input.shape
    hidden = width // hc_count
    top_k = topk_rows.shape[1]
    ptrs = (
        hyper_input,
        block_out.data_ptr(),
        inject.data_ptr(),
        shared.data_ptr(),
        gate_logits.data_ptr(),
        topk_weights.data_ptr(),
        topk_rows.data_ptr(),
        norm_weight.data_ptr(),
        hc_out.data_ptr(),
        normed.data_ptr(),
    )
    strides = (
        hyper_input.stride(0),
        block_out.stride(0),
        inject.stride(0),
        shared.stride(0),
        gate_logits.stride(0),
        topk_weights.stride(0),
        topk_rows.stride(0),
    )
    key = (hc_count, hidden, top_k, _PDL_TRIGGER)
    fn = _FOLD_NORM.get(key)
    if fn is None:
        gm = cute.AddressSpace.gmem
        bf = cutlass.BFloat16
        cargs = (
            from_dlpack(hyper_input, assumed_align=16).mark_layout_dynamic(leading_dim=1),
            make_ptr(bf, ptrs[1], gm, assumed_align=16),
            make_ptr(bf, ptrs[2], gm, assumed_align=2),
            make_ptr(bf, ptrs[3], gm, assumed_align=16),
            make_ptr(bf, ptrs[4], gm, assumed_align=2),
            make_ptr(bf, ptrs[5], gm, assumed_align=2),
            make_ptr(cutlass.Int32, ptrs[6], gm, assumed_align=4),
            make_ptr(bf, ptrs[7], gm, assumed_align=16),
            make_ptr(bf, ptrs[8], gm, assumed_align=16),
            make_ptr(bf, ptrs[9], gm, assumed_align=16),
        )
        fn = cute.compile(
            _FoldNormKernel(hc_count, hidden, top_k, _PDL_TRIGGER),
            *cargs,
            *(cutlass.Int32(1),) * 8,
            cutlass.Float32(1e-6),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--opt-level 2 --enable-tvm-ffi",
        )
        _FOLD_NORM[key] = fn
    fn(*ptrs, rows, *strides, eps)


class Qwen4ExpHcInjectNorm(OpWrapper):
    """With `C = hc_count`, `H = width // C`, every tensor 2-D `[T, ...]`:

        hc_out[t, s*H + i] = hyper_input[t, s*H + i] + block_out[t, i] * inject[t, s]
        normed[t, s*H + i] = hc_out[t, s*H + i] * rstd(t, s) * (1 + norm_weight[s*H + i])
        rstd(t, s)         = 1 / sqrt(mean_i(hc_out[t, s*H + i]^2) + eps)

    `hc_out` is rounded once to the activation dtype and `normed` is computed
    from that rounded value (fp32 inside, one rounding on the store). Returns
    `(hc_out, normed)`, both fresh `[T, C*H]` tensors in `hyper_input`'s dtype.

    `block_out=None, inject=None` is the norm alone: `hc_out` *is*
    `hyper_input` (not a copy) and only `normed` is written -- the stack's first
    site and the site after the PLE add.

    With `shared` and `gate_logits` (after a MoE block), `block_out` is the
    routed experts' output and the block output injected above is

        block[t, i] = block_out[t, i] + sigmoid(gate_logits[t]) * shared[t, i]

    in fp32, rounded once to `block_out`'s dtype before the injection reads it:
    bitwise what `fused_sigmoid_gate_mul_add` stores. `gate_logits` is one
    pre-sigmoid scalar per row, `[T, 1]` or `[T]`; none of the inputs is written.

    With `topk_weights` / `topk_rows` as well (the MoE runner left unfinalized),
    `block_out` is its permuted FC2 output `[P, H]` and the routed row is the
    runner's finalize, formed here: `routed[t, i] = sum_k topk_weights[t, k] *
    block_out[topk_rows[t, k], i]` as fp32 FMAs in slot order from zero (a slot
    whose row is -1 is skipped), rounded once to `block_out`'s dtype -- bitwise
    what the runner's finalize stores. `topk_rows` is the runner's
    expanded->permuted index map `[T, k]` int32, `topk_weights` its `[T, k]`
    routing weights. The fold needs the combine (`shared`, `gate_logits`).

    Fusion boundary: the routed experts' finalize (optional), the
    shared-expert combine (optional), the stream-broadcast
    gated accumulate (the old `qwen4_exp_post_inject`) and the grouped gemma
    RMS norm of the next site (the old `_grouped_norm`). The routed and shared
    experts, the gate projection and the mixer that consumes `normed` are not
    inside.
    """

    ARCHS = frozenset({Arch.SM_103})

    # spec: rows, inject (True), moe (False: shared + gate_logits [T, 1] too), fold
    # (0: top-k of the finalize folded in), drop (False: some slots -1), site
    # ("trunk": the norm weight a trunk site's; "stack_final": the stack-final mixer's).
    # The checkpoint's geometry throughout: hc_count 4, hidden 2560, bf16, contiguous rows.
    # The first four reach the tp1 target's four distinct launches -- the Triton kernel's
    # eight-warp form (decode), its two-warp form (prefill, here with the combine),
    # the CuTe fold kernel (decode MoE sites; the -1 slots ride it) and the norm-only
    # variant -- none of which another covers. The rest are the unified target's: one
    # form at every row count, its MoE sites the combine over the routed rows already
    # finalized and reduced across ranks (no fold), so the Triton kernel with the
    # combine at decode sizes, and every form from 1 row up to max_num_tokens.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="checkpoint 4x2560 bf16, inject, decode batch 128 (eight-warp form)",
            spec=dict(rows=128),
        ),
        Cell(
            why="MoE combine, [T, 1] gate, prefill chunk 16384 (the ladder's max_num_tokens)",
            spec=dict(rows=16384, moe=True),
        ),
        Cell(
            why="MoE combine + finalize fold (top-10, CuTe kernel), three slots -1, decode 128",
            spec=dict(rows=128, moe=True, fold=10, drop=True),
        ),
        Cell(
            why="norm only (no inject: stack start / after PLE), decode 128",
            spec=dict(rows=128, inject=False),
        ),
        Cell(
            why="1 row, attention site: inject of the reduced block output, eight-warp form",
            spec=dict(rows=1, moe=False, inject=True),
        ),
        Cell(
            why="1 row, MoE site: combine over the reduced routed rows, no fold",
            spec=dict(rows=1, moe=True, inject=True),
        ),
        Cell(
            why="32 rows, MoE site: combine over the reduced routed rows, no fold",
            spec=dict(rows=32, moe=True, inject=True),
        ),
        Cell(
            why="128 rows, MoE site: combine without the fold (the con-128 decode batch)",
            spec=dict(rows=128, moe=True, inject=True),
        ),
        Cell(
            why="1024 rows, attention site: two-warp form (an MTP-7 verify of 128 requests)",
            spec=dict(rows=1024, moe=False, inject=True),
        ),
        Cell(
            why="8192 rows, MoE site: combine, two-warp form (max_num_tokens)",
            spec=dict(rows=8192, moe=True, inject=True),
        ),
        Cell(
            why="1 row, stack-final site: last MoE combine, the final mixer's norm weight",
            spec=dict(rows=1, moe=True, inject=True, site="stack_final"),
        ),
        Cell(
            why="1 row, norm only: the stack's first site at decode",
            spec=dict(rows=1, moe=False, inject=False),
        ),
        Cell(
            why="8192 rows, norm only: the stack's first site at max_num_tokens (two-warp)",
            spec=dict(rows=8192, moe=False, inject=False),
        ),
    )

    note = """
    Metadata consumed: none; every (row, stream) independent. Call sites: in
    the trunk forward of both targets, once after each block (attention / GDN
    and MoE) of each layer, normalizing for the next site -- the layer's MoE
    mixer, the next layer's attention mixer, or the stack-final mixer -- plus
    norm-only calls at the stack's first site and after the PLE add. 97 calls
    per forward on the 48-layer trunk; the 48 after a MoE block pass `shared`
    and `gate_logits` (in `DecodeTarget` all but the one before the PLE add,
    which keeps the unfused combine, and those also `topk_weights` /
    `topk_rows`: the runner's finalize folded in). The MTP layer and the PLE
    block keep the unfused ops.

    In the unified (tp4 / tep4 / dep4) target HC is replicated and runs on the
    rank's own rows, one form at every row count (1 to max_num_tokens 8192, an
    MTP-7 verify 1024): the attention site injects the block output after its
    all-reduce (none at dep4); the MoE site passes the routed rows already
    finalized and all-reduced / reduce-scattered with this rank's replicated
    shared expert output and gate logit -- the combine, never the fold; the
    last layer's MoE site normalizes with the stack-final mixer's weight
    (`hyper_connection_mixer.hc_norm`); the stack's first site is norm only.

    Preconditions: 2-D operands, last-dim stride 1 (row strides free); the
    width divisible by `hc_count`; `inject` `[T, hc_count]`, `block_out` and
    `shared` `[T, hidden]` of one dtype (with the fold `block_out` is `[P,
    hidden]`, any `P`, read through int32 `topk_rows` `[T, k]`); `gate_logits` `T` elements along its
    first axis (`[T, 1]` or `[T]`, any row stride); `norm_weight` contiguous
    `[width]`. All on one device. `shared` and `gate_logits` come together and
    only with `block_out`.

    Numerics: the combine is fp32 with one rounding to `block_out`'s dtype
    (bitwise `fused_sigmoid_gate_mul_add`: the same `tl.sigmoid` and fma); the
    accumulate is fp32 with one rounding (bitwise what `qwen4_exp_post_inject`
    stores for 16-bit operands up to fma contraction); the norm is fp32 with
    `rsqrt`, one rounding. Against the float64 reference the combine can land
    one ulp of the block output away (its rounding midpoint), which the
    accumulate carries scaled by `inject`; the test's band states that.
    Graph-capturable by construction (no host sync, outputs allocated per
    call); not certified.

    Launch: one program per (row, stream), stream-minor in a 1-D grid, so a
    row's `block_out` / `shared` are read from HBM once and from L2 by the
    other `hc_count - 1` programs; with the fold, one program per two streams
    of a row (the gathered rows are read and summed half as often). Every program releases programmatic
    dependents at its start (`griddepcontrol.launch_dependents`; off with
    TRTLLM_ENABLE_PDL=0): a consumer launched with PDL (the decode mixer's
    `qwen4_exp_hc_decode_tc_first`) streams its weights meanwhile and waits for
    this grid before reading `normed`. A no-op for any other consumer. At a
    hidden of 2048 or more a program runs eight warps below
    `_FEW_WARPS_MIN_ROWS` rows and two from it; the RMS sum's fp32 reduction
    order follows the warp count.

    The decode MoE sites -- the fold below `_FEW_WARPS_MIN_ROWS` rows, bf16
    operands, `hc_count` 4, a hidden in (2048, 4096], 16-byte aligned rows with
    row strides a multiple of 16 -- run `_FoldNormKernel` (CuTe DSL) instead:
    a row's two CTAs one cluster, CTA j forming half the block output (the
    finalize and the combine) into both CTAs' shared memory, then each
    accumulating and normalizing two streams, in the Triton kernel's layout and
    reduction tree -- bitwise its `hc_out` and `normed` (tested at 511 rows, the
    last before the switch). GB300, graph replay,
    16 rotating sites, us: 128 rows 7.00 -> 5.97, 64 rows 6.63 -> 5.25, 8 rows
    3.59 -> 3.09; it releases dependents at its start as the Triton kernel does.

    Certified on sm_103 (GB300), bf16, the checkpoint's 4 x 2560, PDL trigger
    on (the default): the Triton kernel and the CuTe fold kernel are JIT-
    compiled at first call for the running GPU, and the cells ran that path.
      tp1: the injection at 128 rows (eight-warp form), the combine at 16384
        rows (two-warp form), the combine with the fold (three slots -1) at 128
        rows (CuTe kernel), norm-only at 128 rows; the CuTe fold bitwise the
        Triton kernel at 511 rows.
      unified: the injection at 1 row (eight-warp) and 1024 rows (two-warp);
        the combine without the fold at 1, 32 and 128 rows (eight-warp) and
        8192 rows (two-warp); the combine at 1 row under the stack-final
        weight (RMS 3.1, -0.37 to 12.4, so `1 + w` up to 13.4); norm-only at 1
        and 8192 rows (two-warp).
    Measured against the float64 reference rounded once: the decode cells (1,
    32 rows) bitwise or one ulp on 4 of 327680 `normed` elements; at 128 rows
    one `hc_out` element and 5 `normed` elements of 1310720 differ; at 1024
    rows `hc_out` bitwise, 74 of 10485760 `normed` one ulp; norm-only at 8192
    rows 602 of 83886080 one ulp. The combine at 8192 / 16384 rows moves 131 /
    192 `hc_out` elements, each inside one ulp of `hc_out` plus one ulp of the
    combine times `|inject|` (worst 0.95 of that band; up to 31 ulp of a
    near-zero residual), and `normed` within one ulp of the norm of the
    stored `hc_out`. Relative L2 at most 1.8e-5, cosine 1.0 (9 digits) on both
    outputs at every cell.

    Not certified: hc_count other than 4, hidden other than 2560, float16
    operands, a flat `[T]` gate, row strides other than the width, zero rows,
    TRTLLM_ENABLE_PDL=0, CUDA-graph replay (the kernels take them; no cell
    drives them).

    Hard limits: row offsets are int64 (`program_id(0)` widened); the grid is
    `rows * hc_count` programs along x (2^31 - 1); the column offset
    `stream * hidden + i` is int32, bounded by the width.
    """

    def __call__(
        self,
        hyper_input: torch.Tensor,
        block_out: Optional[torch.Tensor],
        inject: Optional[torch.Tensor],
        norm_weight: torch.Tensor,
        eps: float,
        hc_count: int,
        shared: Optional[torch.Tensor] = None,
        gate_logits: Optional[torch.Tensor] = None,
        topk_weights: Optional[torch.Tensor] = None,
        topk_rows: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        rows, width = hyper_input.shape
        hidden = width // hc_count
        normed = torch.empty((rows, width), dtype=hyper_input.dtype, device=hyper_input.device)
        has_inject = block_out is not None
        has_shared = shared is not None
        top_k = topk_rows.shape[1] if topk_rows is not None else 0
        hc_out = (
            torch.empty((rows, width), dtype=hyper_input.dtype, device=hyper_input.device)
            if has_inject
            else hyper_input
        )
        if (
            rows > 0
            and top_k
            and _fold_cute_ok(
                hyper_input,
                block_out,
                inject,
                norm_weight,
                hc_count,
                shared,
                gate_logits,
                topk_weights,
                topk_rows,
            )
        ):
            _fold_cute(
                hyper_input,
                block_out,
                inject,
                norm_weight,
                eps,
                hc_count,
                shared,
                gate_logits,
                topk_weights,
                topk_rows,
                hc_out,
                normed,
            )
        elif rows > 0:
            block = triton.next_power_of_2(hidden)
            # The fold gathers TOPK rows to form a row's block output: two streams a
            # program form it half as often (when hc_count is even).
            spp = 2 if top_k and hc_count % 2 == 0 else 1
            _hc_inject_norm_kernel[(rows * hc_count // spp,)](
                hyper_input,
                block_out if has_inject else hyper_input,
                inject if has_inject else hyper_input,
                shared if has_shared else hyper_input,
                gate_logits if has_shared else hyper_input,
                topk_weights if top_k else hyper_input,
                topk_rows if top_k else hyper_input,
                norm_weight,
                hc_out,
                normed,
                hyper_input.stride(0),
                block_out.stride(0) if has_inject else 0,
                inject.stride(0) if has_inject else 0,
                shared.stride(0) if has_shared else 0,
                gate_logits.stride(0) if has_shared else 0,
                topk_weights.stride(0) if top_k else 0,
                topk_rows.stride(0) if top_k else 0,
                hc_out.stride(0),
                normed.stride(0),
                eps,
                HC_COUNT=hc_count,
                HIDDEN=hidden,
                BLOCK=block,
                HAS_INJECT=has_inject,
                HAS_SHARED=has_shared,
                TOPK=top_k,
                STREAMS_PER_PROGRAM=spp,
                PDL_TRIGGER=_PDL_TRIGGER,
                num_warps=(2 if rows >= _FEW_WARPS_MIN_ROWS else 8) if block >= 2048 else 4,
            )
        return hc_out, normed

    def reference(
        self,
        hyper_input: torch.Tensor,
        block_out: Optional[torch.Tensor],
        inject: Optional[torch.Tensor],
        norm_weight: torch.Tensor,
        eps: float,
        hc_count: int,
        shared: Optional[torch.Tensor] = None,
        gate_logits: Optional[torch.Tensor] = None,
        topk_weights: Optional[torch.Tensor] = None,
        topk_rows: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """float64 throughout. The finalize (with `topk_rows`) and the combine
        (with `shared`) are each rounded once to `block_out`'s dtype, as the op
        states; `hc_out` is the float64
        accumulate rounded once; `normed` is the float64 norm of that rounded
        residual, rounded once -- the same dataflow the op states, with no
        other intermediate rounding."""
        dtype = hyper_input.dtype
        rows, width = hyper_input.shape
        hidden = width // hc_count
        if block_out is None:
            hc_out = hyper_input
        else:
            if topk_rows is not None:
                valid = (topk_rows >= 0).unsqueeze(-1)
                picked = block_out.double()[topk_rows.clamp(min=0).long()]
                w = topk_weights.double().unsqueeze(-1)
                block_out = torch.where(valid, picked * w, 0.0).sum(1).to(block_out.dtype)
            if shared is not None:
                s = torch.sigmoid(gate_logits.double().reshape(rows, 1))
                block_out = (block_out.double() + s * shared.double()).to(block_out.dtype)
            streams = hyper_input.double().unflatten(-1, (hc_count, hidden))
            gated = block_out.double().unsqueeze(-2) * inject.double().unsqueeze(-1)
            hc_out = (streams + gated).flatten(-2).to(dtype)
        x = hc_out.double().unflatten(-1, (hc_count, hidden))
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
        normed = (x.flatten(-2) * (1.0 + norm_weight.double())).to(dtype)
        return hc_out, normed

    def is_valid(
        self,
        hyper_input: torch.Tensor,
        block_out: Optional[torch.Tensor],
        inject: Optional[torch.Tensor],
        norm_weight: torch.Tensor,
        eps: float,
        hc_count: int,
        shared: Optional[torch.Tensor] = None,
        gate_logits: Optional[torch.Tensor] = None,
        topk_weights: Optional[torch.Tensor] = None,
        topk_rows: Optional[torch.Tensor] = None,
    ) -> None:
        # The kernel indexes with raw row strides and unit column stride, and
        # reads `inject` / `block_out` / `shared` / the gate per row: a broadcast
        # (stride-0) or a wrong-width operand would be read out of bounds or
        # silently reused.
        assert hyper_input.dim() == 2 and hyper_input.stride(1) == 1, "hyper_input [T, width]"
        assert (block_out is None) == (inject is None), "block_out and inject come together"
        assert (shared is None) == (gate_logits is None), "shared and gate_logits come together"
        assert shared is None or block_out is not None, "the combine needs block_out"
        assert (topk_weights is None) == (topk_rows is None), (
            "topk_weights and topk_rows come together"
        )
        assert topk_rows is None or shared is not None, "the finalize fold needs the combine"
        rows, width = hyper_input.shape
        assert width % hc_count == 0, f"width {width} not divisible by hc_count {hc_count}"
        assert norm_weight.shape == (width,) and norm_weight.is_contiguous(), "norm_weight [width]"
        if block_out is not None:
            if topk_rows is None:
                assert block_out.shape == (rows, width // hc_count) and block_out.stride(1) == 1
            else:
                # The permuted FC2 rows: any row count, read through topk_rows.
                assert block_out.dim() == 2 and block_out.shape[1] == width // hc_count
                assert block_out.stride(1) == 1, "permuted rows [P, hidden]"
                assert topk_rows.dtype == torch.int32, "topk_rows int32 (the runner's index map)"
                assert topk_rows.dim() == 2 and topk_rows.shape[0] == rows, "topk_rows [T, k]"
                assert topk_rows.stride(1) == 1 and topk_weights.stride(1) == 1
                assert topk_weights.shape == topk_rows.shape, "topk_weights [T, k]"
            assert inject.shape == (rows, hc_count) and inject.stride(1) == 1
        if shared is not None:
            assert shared.shape == (rows, width // hc_count), "shared [T, hidden]"
            assert shared.stride(1) == 1, "shared [T, hidden]"
            assert shared.dtype == block_out.dtype, "shared in block_out's dtype"
            assert gate_logits.shape in ((rows,), (rows, 1)), "gate_logits [T, 1] or [T]"
            assert rows <= 1 or gate_logits.stride(0) != 0, "one gate logit per row"


qwen4_exp_hc_inject_norm = Qwen4ExpHcInjectNorm()
