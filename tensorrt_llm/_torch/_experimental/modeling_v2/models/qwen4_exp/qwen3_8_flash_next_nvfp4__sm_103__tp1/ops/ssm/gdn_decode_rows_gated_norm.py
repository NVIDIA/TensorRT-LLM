# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One Gated-DeltaNet decode step per row and the gated RMS norm on its output, streamed
over the fp32 or bf16 state pool: the decode step at batch sizes past the cluster kernel's
over an fp32 pool, and at every batch size over a bf16 pool.

Tier 6 (CuTe DSL written for this target; the recurrence is the gated delta rule of
FlashInfer's `gdn_decode.gated_delta_rule_decode_pretranspose`, reimplemented here). At a
decode batch the step is the pool's traffic -- each (row, v-head) state, 64 KB fp32 or
32 KB bf16, read and written once -- so a CTA owns one whole state and streams it: warp w
the value rows [w * RPW, (w + 1) * RPW), lane l the key columns [4l, 4l + 4) of each, so
a warp's access is one contiguous state row. A bf16 state is widened to fp32 on the load
and rounded back (nearest even) on the store. The state is loaded before
`griddepcontrol.wait` (nothing in the step writes it ahead of this kernel). The two dot
products a value row takes over K are reduced across the warp by a transposed
reduction (RPW rows in RPW - 1 + log2(32 / RPW) shuffles, not 5 RPW), which leaves row
j's sum in lanes [j * 32 / RPW, (j + 1) * 32 / RPW); `u` goes back to every lane by
index shuffles. The CTA holds the head's whole output, so the gated RMS norm runs in
place.
"""

from __future__ import annotations

import math

import cutlass
import cutlass.cute as cute
import torch
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import T, dsl_user_op

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl
from tensorrt_llm._torch.kernels.blackwell.utils import make_ptr

from .gdn_decode_gated_norm_pdl import _recurrence_reference
from .gdn_decode_gated_norm_pdl import gdn_decode_gated_norm_pdl as _one_launch

#: Key columns a lane (one 16-byte fp32 or 8-byte bf16 vector of a state row).
_PL = 4
#: Warps a CTA; a warp owns `V // _WARPS` value rows of the head's state.
_WARPS = 8
#: Largest recurrent-pool slot count the cache manager was seen to allocate for this
#: checkpoint; the kernel addresses a slot by its index alone, any pool size alike.
ENGINE_SLOTS = 461


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


def _asm_n(n, ty, args, text, constraints, loc, ip, side_effects=False):
    res = _asm(
        llvm.StructType.get_literal([ty] * n), args, text, constraints, loc, ip, side_effects
    )
    return [llvm.extractvalue(ty, res, [i], loc=loc, ip=ip) for i in range(n)]


@dsl_user_op
def _ld_state(addr, pred, *, loc=None, ip=None):
    """Four fp32 of a state row (streaming), or zeros where `pred` is 0."""
    r = _asm_n(
        4,
        T.f32(),
        [cutlass.Int64(addr), cutlass.Int32(pred)],
        "{ .reg .pred p; setp.ne.b32 p, $5, 0; mov.b32 $0, 0; mov.b32 $1, 0; mov.b32 $2, 0; "
        "mov.b32 $3, 0; @p ld.global.cs.v4.f32 {$0, $1, $2, $3}, [$4]; }",
        "=f,=f,=f,=f,l,r",
        loc,
        ip,
        side_effects=True,
    )
    return [cutlass.Float32(v) for v in r]


@dsl_user_op
def _st_state(addr, pred, a, b, c, d, *, loc=None, ip=None):
    """Four fp32 of a state row (streaming) where `pred` is not 0."""
    _asm(
        T.i32(),
        [cutlass.Int64(addr), cutlass.Int32(pred)] + [cutlass.Float32(v) for v in (a, b, c, d)],
        "{ .reg .pred p; setp.ne.b32 p, $2, 0; @p st.global.cs.v4.f32 [$1], {$3, $4, $5, $6}; "
        "mov.u32 $0, 0; }",
        "=r,l,r,f,f,f,f",
        loc,
        ip,
        side_effects=True,
    )


@dsl_user_op
def _ld_state_bf16(addr, pred, *, loc=None, ip=None):
    """Four bf16 of a state row (streaming) as fp32, or zeros where `pred` is 0."""
    r = _asm_n(
        4,
        T.f32(),
        [cutlass.Int64(addr), cutlass.Int32(pred)],
        "{ .reg .pred p; .reg .b32 w0, w1; setp.ne.b32 p, $5, 0; mov.b32 w0, 0; "
        "mov.b32 w1, 0; @p ld.global.cs.v2.b32 {w0, w1}, [$4]; shl.b32 $0, w0, 16; "
        "and.b32 $1, w0, 0xFFFF0000; shl.b32 $2, w1, 16; and.b32 $3, w1, 0xFFFF0000; }",
        "=f,=f,=f,=f,l,r",
        loc,
        ip,
        side_effects=True,
    )
    return [cutlass.Float32(v) for v in r]


@dsl_user_op
def _st_state_bf16(addr, pred, a, b, c, d, *, loc=None, ip=None):
    """Four fp32 rounded to bf16 into a state row (streaming) where `pred` is not 0."""
    _asm(
        T.i32(),
        [cutlass.Int64(addr), cutlass.Int32(pred)] + [cutlass.Float32(v) for v in (a, b, c, d)],
        "{ .reg .pred p; .reg .b32 r0, r1; setp.ne.b32 p, $2, 0; "
        "cvt.rn.bf16x2.f32 r0, $4, $3; cvt.rn.bf16x2.f32 r1, $6, $5; "
        "@p st.global.cs.v2.b32 [$1], {r0, r1}; mov.u32 $0, 0; }",
        "=r,l,r,f,f,f,f",
        loc,
        ip,
        side_effects=True,
    )


@dsl_user_op
def _ld_bf16x4(addr, *, loc=None, ip=None):
    """Four bf16 (8 bytes) as fp32."""
    r = _asm_n(
        4,
        T.f32(),
        [cutlass.Int64(addr)],
        "{ .reg .b32 w0, w1; ld.global.v2.b32 {w0, w1}, [$4]; shl.b32 $0, w0, 16; "
        "and.b32 $1, w0, 0xFFFF0000; shl.b32 $2, w1, 16; and.b32 $3, w1, 0xFFFF0000; }",
        "=f,=f,=f,=f,l",
        loc,
        ip,
        side_effects=True,
    )
    return [cutlass.Float32(v) for v in r]


@dsl_user_op
def _ld_bf16(addr, *, loc=None, ip=None):
    return cutlass.Float32(
        _asm(
            T.f32(),
            [cutlass.Int64(addr)],
            "{ .reg .b16 h; ld.global.b16 h, [$1]; cvt.f32.bf16 $0, h; }",
            "=f,l",
            loc,
            ip,
            side_effects=True,
        )
    )


@dsl_user_op
def _st_bf16(addr, pred, x, *, loc=None, ip=None):
    """`x` rounded to bf16 where `pred` is not 0."""
    _asm(
        T.i32(),
        [cutlass.Int64(addr), cutlass.Int32(pred), cutlass.Float32(x)],
        "{ .reg .pred p; .reg .b16 h; setp.ne.b32 p, $2, 0; cvt.rn.bf16.f32 h, $3; "
        "@p st.global.b16 [$1], h; mov.u32 $0, 0; }",
        "=r,l,r,f",
        loc,
        ip,
        side_effects=True,
    )


@dsl_user_op
def _round_bf16(x, *, loc=None, ip=None):
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


def _rows_reduce(vals, lane):
    """Sum each of `vals` (a lane's partial of one row each) over the warp. Each halving
    level (offset 16 first) trades half the rows with the partner lane, so row j's sum
    lands in lanes [j * 32 / n, (j + 1) * 32 / n); full butterflies finish those lanes."""
    vals = list(vals)
    off = 16
    while len(vals) > 1:
        half = len(vals) // 2
        upper = (lane & off) != 0
        kept = []
        for i in range(half):
            lo, hi = vals[i], vals[i + half]
            recv = cute.arch.shuffle_sync_bfly(cutlass.select_(upper, lo, hi), offset=off)
            kept.append(cutlass.select_(upper, hi, lo) + recv)
        vals = kept
        off //= 2
    v = vals[0]
    while off >= 1:
        v = v + cute.arch.shuffle_sync_bfly(v, offset=off)
        off //= 2
    return v


class _GdnRowsKernel:
    """CTA (v-head, row): the row's whole state of that head. `norm`: the gated RMS norm on
    the head's output; else the recurrence's output, rounded to bf16, is stored as is."""

    def __init__(self, k_heads, v_heads, kdim, vdim, pdl, norm=True, state_bf16=False):
        assert kdim == 32 * _PL and vdim % _WARPS == 0
        self.h, self.hv, self.k, self.v, self.pdl = k_heads, v_heads, kdim, vdim, pdl
        self.norm = norm
        # The pool's element size and the lane's 4-column load and store of a state row.
        self.sb = 2 if state_bf16 else 4
        self.ld_state, self.st_state = (
            (_ld_state_bf16, _st_state_bf16) if state_bf16 else (_ld_state, _st_state)
        )
        self.rpw = vdim // _WARPS
        assert self.rpw in (4, 8, 16, 32)
        # Lanes that hold one row's reduced sums.
        self.lpr = 32 // self.rpw
        self.k_off = k_heads * kdim
        self.v_off = 2 * k_heads * kdim
        self.z_off = self.v_off + v_heads * vdim
        self.a_off = self.z_off + v_heads * vdim
        self.b_off = self.a_off + v_heads

    @cute.jit
    def __call__(
        self,
        proj_t: cute.Tensor,
        pool_ptr: cute.Pointer,
        slot_ptr: cute.Pointer,
        a_log_ptr: cute.Pointer,
        dt_ptr: cute.Pointer,
        w_ptr: cute.Pointer,
        out_ptr: cute.Pointer,
        rows: cutlass.Int32,
        s_proj: cutlass.Int32,
        s_slot: cutlass.Int64,
        s_head: cutlass.Int32,
        s_row: cutlass.Int32,
        s_out: cutlass.Int32,
        eps: cutlass.Float32,
        stream,
    ):
        slots = cute.make_tensor(slot_ptr, cute.make_layout((rows,), stride=(1,)))
        a_log_t = cute.make_tensor(a_log_ptr, cute.make_layout((self.hv,), stride=(1,)))
        dt = cute.make_tensor(dt_ptr, cute.make_layout((self.hv,), stride=(1,)))
        # `proj` arrives as a tensor: the TVM-FFI entry takes its stream from a GPU tensor.
        self.kernel(
            proj_t.iterator.toint(),
            pool_ptr.toint(),
            slots,
            a_log_t,
            dt,
            w_ptr.toint(),
            out_ptr.toint(),
            s_proj,
            s_slot,
            s_head,
            s_row,
            s_out,
            eps,
        ).launch(
            grid=[self.hv, rows, 1],
            block=[32 * _WARPS, 1, 1],
            smem=_WARPS * 4 + 64,
            stream=stream,
            use_pdl=self.pdl,
        )

    @cute.kernel
    def kernel(
        self,
        proj_base: cutlass.Int64,
        pool_base: cutlass.Int64,
        mSlot: cute.Tensor,
        mALog: cute.Tensor,
        mDt: cute.Tensor,
        w_base: cutlass.Int64,
        out_base: cutlass.Int64,
        s_proj: cutlass.Int32,
        s_slot: cutlass.Int64,
        s_head: cutlass.Int32,
        s_row: cutlass.Int32,
        s_out: cutlass.Int32,
        eps: cutlass.Float32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        hv, row, _ = cute.arch.block_idx()
        warp = tidx // 32
        lane = tidx % 32
        kh = hv // (self.hv // self.h)
        v0 = warp * self.rpw
        jl = lane // self.lpr  # the value row whose sums this lane ends up holding
        red = cutlass.utils.SmemAllocator().allocate_tensor(
            cutlass.Float32, cute.make_layout((_WARPS,), stride=(1,)), byte_alignment=16
        )

        # 1. Before the wait: the warp's state rows (a negative slot reads none: zero state),
        # the gate constants and the norm weight of this lane's row.
        slot = mSlot[row]
        live = cutlass.Int32(0)
        if slot >= 0:
            live = cutlass.Int32(1)
        # 64-bit: a coalesced cache page puts every GDN layer's state of a slot in one
        # page, so `slot * s_slot` passes int32.
        base = (
            pool_base
            + (
                cutlass.Int64(cutlass.max(slot, cutlass.Int32(0))) * s_slot
                + cutlass.Int64(hv * s_head + v0 * s_row + lane * _PL)
            )
            * self.sb
        )
        h = [
            self.ld_state(base + cutlass.Int64(j * s_row * self.sb), live) for j in range(self.rpw)
        ]
        a_log = mALog[hv]
        dtb = mDt[hv]
        w_l = cutlass.Float32(0.0)
        if cutlass.const_expr(self.norm):
            w_l = _ld_bf16(w_base + cutlass.Int64(v0 + jl) * 2)
        if cutlass.const_expr(self.pdl):
            cute.arch.griddepcontrol_launch_dependents()
            cute.arch.griddepcontrol_wait()

        # 2. The row's q and k (this lane's four columns), v and z (this lane's value
        # row), a and b; the q/k L2 norms (eps 1e-6 inside the root, 1/sqrt(K) on q) and
        # the gates (softplus beta 1, threshold 20).
        prow = proj_base + cutlass.Int64(row) * s_proj * 2
        q4 = _ld_bf16x4(prow + cutlass.Int64(kh * self.k + lane * _PL) * 2)
        k4 = _ld_bf16x4(prow + cutlass.Int64(self.k_off + kh * self.k + lane * _PL) * 2)
        v_l = _ld_bf16(prow + cutlass.Int64(self.v_off + hv * self.v + v0 + jl) * 2)
        z_l = cutlass.Float32(0.0)
        if cutlass.const_expr(self.norm):
            z_l = _ld_bf16(prow + cutlass.Int64(self.z_off + hv * self.v + v0 + jl) * 2)
        a = _ld_bf16(prow + cutlass.Int64(self.a_off + hv) * 2)
        b = _ld_bf16(prow + cutlass.Int64(self.b_off + hv) * 2)
        sq = cutlass.Float32(0.0)
        sk = cutlass.Float32(0.0)
        for e in cutlass.range_constexpr(_PL):
            sq = q4[e] * q4[e] + sq
            sk = k4[e] * k4[e] + sk
        sq = cute.arch.warp_reduction_sum(sq)
        sk = cute.arch.warp_reduction_sum(sk)
        rq = cute.math.rsqrt(sq + 1e-6) * (1.0 / math.sqrt(self.k))
        rk = cute.math.rsqrt(sk + 1e-6)
        qn = [q4[e] * rq for e in range(_PL)]
        kn = [k4[e] * rk for e in range(_PL)]
        x = a + dtb
        sp = x
        if x <= 20.0:
            sp = cute.math.log1p(cute.math.exp(x))
        decay = cute.math.exp(-cute.math.exp(a_log) * sp)
        beta = 1.0 / (1.0 + cute.math.exp(-b))

        # 3. Per value row: hk = h.k, u = beta (v - decay hk); h = decay h + u k; o = h.q.
        part = []
        for j in cutlass.range_constexpr(self.rpw):
            acc = cutlass.Float32(0.0)
            for e in cutlass.range_constexpr(_PL):
                acc = h[j][e] * kn[e] + acc
            part.append(acc)
        hk = _rows_reduce(part, lane)
        u_l = beta * (v_l - decay * hk)
        part = []
        for j in cutlass.range_constexpr(self.rpw):
            u = cute.arch.shuffle_sync(u_l, j * self.lpr)
            acc = cutlass.Float32(0.0)
            for e in cutlass.range_constexpr(_PL):
                h[j][e] = u * kn[e] + decay * h[j][e]
                acc = h[j][e] * qn[e] + acc
            part.append(acc)
            self.st_state(base + cutlass.Int64(j * s_row * self.sb), live, *h[j])
        o = _rows_reduce(part, lane)

        # 4. The gated RMS norm over the head's V outputs, each rounded to bf16 first (the
        # recurrence's output dtype in the composition): y = o rstd w sigmoid(z).
        o_b = _round_bf16(o) * cutlass.Float32(live)
        own = cutlass.Int32(0)
        if lane % self.lpr == 0:
            own = cutlass.Int32(1)
        y = o_b
        if cutlass.const_expr(self.norm):
            ss = cute.arch.warp_reduction_sum(o_b * o_b * cutlass.Float32(own))
            if lane == 0:
                red[warp] = ss
            cute.arch.sync_threads()
            tot = red[0]
            for i in cutlass.range_constexpr(1, _WARPS):
                tot = tot + red[i]
            rstd = cute.math.rsqrt(tot / self.v + eps)
            y = o_b * rstd * w_l * (1.0 / (1.0 + cute.math.exp(-z_l)))
        _st_bf16(out_base + (cutlass.Int64(row) * s_out + hv * self.v + v0 + jl) * 2, own, y)


_COMPILED: dict = {}


def launch(proj, pool, slots, A_log, dt_bias, norm_weight, eps, k_heads, kdim, pdl=True, out=None):
    """The launch behind `GdnDecodeRowsGatedNorm.__call__` and, with `norm_weight=None`,
    `GdnDecodeRows.__call__`; updates `pool` in place and returns the gated-normed output
    (or the recurrence's) `[rows, HV * V]` bf16, written into `out` when given."""
    rows = proj.shape[0]
    v_heads, vdim = pool.shape[1], pool.shape[2]
    norm = norm_weight is not None
    if out is None:
        out = torch.empty((rows, v_heads * vdim), dtype=torch.bfloat16, device=proj.device)
    if rows == 0:
        return out
    pdl = bool(pdl and get_env_enable_pdl())
    state_bf16 = pool.dtype == torch.bfloat16
    key = (k_heads, v_heads, kdim, vdim, pdl, norm, state_bf16)
    ptrs = (
        proj,
        pool.data_ptr(),
        slots.data_ptr(),
        A_log.data_ptr(),
        dt_bias.data_ptr(),
        (norm_weight if norm else A_log).data_ptr(),
        out.data_ptr(),
    )
    fn = _COMPILED.get(key)
    if fn is None:
        gm = cute.AddressSpace.gmem
        cargs = (
            from_dlpack(proj, assumed_align=8).mark_layout_dynamic(leading_dim=1),
            make_ptr(
                cutlass.BFloat16 if state_bf16 else cutlass.Float32,
                ptrs[1],
                gm,
                assumed_align=8 if state_bf16 else 16,
            ),
            make_ptr(cutlass.Int32, ptrs[2], gm, assumed_align=4),
            make_ptr(cutlass.Float32, ptrs[3], gm, assumed_align=4),
            make_ptr(cutlass.Float32, ptrs[4], gm, assumed_align=4),
            make_ptr(cutlass.BFloat16, ptrs[5], gm, assumed_align=2),
            make_ptr(cutlass.BFloat16, ptrs[6], gm, assumed_align=2),
        )
        fn = cute.compile(
            _GdnRowsKernel(k_heads, v_heads, kdim, vdim, pdl, norm, state_bf16),
            *cargs,
            cutlass.Int32(1),
            cutlass.Int32(1),
            cutlass.Int64(1),
            cutlass.Int32(1),
            cutlass.Int32(1),
            cutlass.Int32(1),
            cutlass.Float32(1e-6),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--opt-level 2 --enable-tvm-ffi",
        )
        _COMPILED[key] = fn
    fn(
        *ptrs,
        rows,
        proj.stride(0),
        pool.stride(0),
        pool.stride(1),
        pool.stride(2),
        out.stride(0),
        float(eps) if norm else 0.0,
    )
    return out


class GdnDecodeRowsGatedNorm(OpWrapper):
    """A single-token Gated-DeltaNet step per row over the fp32 or bf16 slot pool, then
    the sigmoid-gated RMS norm of its output per v-head. With `proj` the decode input
    projection's output row `[q (H*K) | k (H*K) | v (HV*V) | z (HV*V) | a (HV) | b (HV)]`
    (after the conv step: q, k, v are the conv's SiLU outputs)::

        o, pool = gated_delta_rule_decode_pretranspose(q, k, v, A_log, a, dt_bias, b,
                                                       use_qk_l2norm=True, pool, slots)
        out     = rms_norm_gated_token_major(o (bf16), z, norm_weight, eps, "sigmoid")

    Returns `out` `[rows, HV * V]` bf16 and updates `pool[slots[n]]` in place; a
    negative slot skips the row's state (no read or write) and its output is zero.
    `o` is rounded to bf16 between the two, as the composition stores it.

    Fusion boundary: both gates, the q/k L2 norm, the recurrence, the indexed state
    read and write, the gated RMS norm. The projection and the conv are the caller's.
    """

    ARCHS = frozenset({Arch.SM_103})

    # spec: rows; pool (default a contiguous 461-slot pool | "coalesced": the
    # cache manager's pool, slot stride 36 layers' states, slots 100 and 101);
    # pad (row 1 given slot -1).
    CELLS: tuple[Cell, ...] = (
        Cell(why="qwen3.8 decode, 128 rows (con 128), 461-slot pool", spec=dict(rows=128)),
        Cell(why="qwen3.8 decode, 3 rows (past the conv-fused path)", spec=dict(rows=3)),
        Cell(
            why="2 rows, coalesced pool (slot stride 36 layers, slot 100), row 1 a pad slot (-1)",
            spec=dict(rows=2, pool="coalesced", pad=True),
        ),
        Cell(why="bf16 pool (mamba_ssm_cache_dtype), 128 rows", spec=dict(rows=128, state="bf16")),
        Cell(why="bf16 pool, 1 row (con 1)", spec=dict(rows=1, state="bf16")),
        Cell(
            why="bf16 pool, 33 rows, one pad slot (-1)", spec=dict(rows=33, pad=True, state="bf16")
        ),
        Cell(
            why="bf16 pool, 2 rows, coalesced: slot stride 36 layers, slot 100",
            spec=dict(rows=2, pool="coalesced", state="bf16"),
        ),
    )

    note = """
    Metadata consumed: `slots` (the cache manager's state indices for the step).

    Preconditions: bf16 `proj` `[rows, 2*H*K + 2*HV*V + 2*HV]` with unit column stride
    and 8-byte aligned rows; the pool fp32 or bf16 `[slots, HV, V, K]` with K innermost
    (any outer strides, multiples of 4) and aligned to 4 elements; `K == 128`, `V % 8 == 0` with
    `V / 8` in {4, 8, 16, 32}; `A_log`, `dt_bias` fp32 `[HV]`; `norm_weight` bf16
    `[V]`; int32 `slots` `[rows]`. **The pool rows and `slots` must not be written by
    any kernel of the same step queued ahead of this one**: both are read before
    `griddepcontrol.wait` -- as `gdn_decode_gated_norm_pdl` states, this target's decode
    steps keep that (a layer's pool is written only by its own recurrence, `slots` is
    filled before the step's graph launches), and verify and prefill steps do not use
    the entry.

    Launch: one CTA of 8 warps a (v-head, row), `HV * rows` CTAs, PDL (off with
    TRTLLM_ENABLE_PDL=0). GB300, graph replay, 461-slot pool, us (FlashInfer's
    recurrence + the gated norm entry -> this): 128 rows 132.2 -> 118.8 (6.78 TB/s
    against an in-place `mul_` over the same bytes at 6.83), 64 rows 70.7 -> 62.7, 16
    rows 19.1 -> 14.9, 4 rows 8.5 -> 8.4.

    Numerics: fp32 inside; each dot product over K is a lane's 4-term FMA chain and then
    the transposed reduction's tree; `o` rounded to bf16 before the norm, the output
    rounded once. A bf16 pool's state is rounded once per step, at the store, and `o`
    comes from the state before that rounding. Inside torch's default band for bf16 and
    the pool's dtype of the composition's float64 reference, not bitwise FlashInfer's
    kernel. Batch-invariant: a row's output and state do not depend on the other rows.

    Certified: 128 and 3 rows (the fewest a decode step sends here once the conv-fused
    path stops) over a contiguous 461-slot pool, and 2 rows over the cache manager's
    coalesced pool (slot stride 36 layers' states, slot 100 ~2.8G elements past the
    base) with the second row a pad slot (-1); the same over a bf16 pool. Under graph
    replay the op reads the projection after `griddepcontrol.wait` (16 rows). Not
    certified: `pdl=False`.
    """

    def __call__(
        self,
        proj: torch.Tensor,
        pool: torch.Tensor,
        slots: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        norm_weight: torch.Tensor,
        eps: float,
        k_heads: int,
        kdim: int,
        pdl: bool = True,
    ) -> torch.Tensor:
        return launch(proj, pool, slots, A_log, dt_bias, norm_weight, eps, k_heads, kdim, pdl)

    def reference(
        self,
        proj: torch.Tensor,
        pool: torch.Tensor,
        slots: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        norm_weight: torch.Tensor,
        eps: float,
        k_heads: int,
        kdim: int,
        pdl: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`(out, pool_after)`: `gdn_decode_gated_norm_pdl`'s reference, the same contract --
        the fp64 recurrence (o rounded to bf16, the pool to its dtype), then the gated norm
        entry's fp32 reference."""
        return _one_launch.reference(
            proj, pool, slots, A_log, dt_bias, norm_weight, eps, k_heads, kdim, pdl
        )

    def is_valid(
        self,
        proj: torch.Tensor,
        pool: torch.Tensor,
        slots: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        norm_weight: torch.Tensor,
        eps: float,
        k_heads: int,
        kdim: int,
        pdl: bool = True,
    ) -> None:
        rows = proj.shape[0]
        v_heads, vdim, kd = pool.shape[1], pool.shape[2], pool.shape[3]
        assert proj.dtype == torch.bfloat16 and proj.stride(1) == 1, "bf16 proj, unit column stride"
        assert proj.shape[1] == 2 * k_heads * kdim + 2 * v_heads * vdim + 2 * v_heads, "proj width"
        assert proj.data_ptr() % 8 == 0 and proj.stride(0) % 4 == 0, "8-byte aligned proj rows"
        assert pool.dtype in (torch.float32, torch.bfloat16), "fp32 or bf16 pool"
        assert pool.stride(3) == 1 and kd == kdim, "pool [S, HV, V, K]"
        assert pool.data_ptr() % (_PL * pool.element_size()) == 0, "pool aligned to 4 elements"
        assert all(s % _PL == 0 for s in pool.stride()[:3]), "pool strides multiples of 4"
        assert kdim == 32 * _PL, f"K == {32 * _PL}"
        assert vdim % _WARPS == 0 and vdim // _WARPS in (4, 8, 16, 32), "V / 8 in {4, 8, 16, 32}"
        assert v_heads % k_heads == 0, "HV a multiple of H"
        assert slots.dtype == torch.int32 and slots.shape == (rows,), "int32 slots [rows]"
        assert slots.is_contiguous(), "contiguous slots"
        assert A_log.dtype == torch.float32 and dt_bias.dtype == torch.float32, (
            "fp32 gate constants"
        )
        assert A_log.is_contiguous() and dt_bias.is_contiguous(), "contiguous gate constants"
        assert norm_weight.dtype == torch.bfloat16 and norm_weight.shape == (vdim,), (
            "bf16 norm weight [V]"
        )


class GdnDecodeRows(OpWrapper):
    """`GdnDecodeRowsGatedNorm` without the gated norm: the recurrence's output `o` (rounded
    to bf16) written into `out` `[rows, HV * V]` (unit column stride, any row stride) --
    a mixed step's generation rows, whose gated norm runs with the context rows'. The same
    kernel, operands and preconditions otherwise (`proj`'s z columns are not read).

    Fusion boundary: both gates, the q/k L2 norm, the recurrence, the indexed state read and
    write. Returns `out`.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="mixed step: 121 generation rows into a row range of the step output",
            spec=dict(rows=121, out="row_range"),
        ),
        Cell(why="3 rows, one pad slot (-1)", spec=dict(rows=3, pad=True, out="row_range")),
        Cell(
            why="2 rows, the cache manager's coalesced pool: slot stride 36 layers",
            spec=dict(rows=2, pool="coalesced", out="row_range"),
        ),
        Cell(
            why="bf16 pool: 121 generation rows into a row range of the step output",
            spec=dict(rows=121, out="row_range", state="bf16"),
        ),
        Cell(
            why="bf16 pool: 3 rows, one pad slot (-1)",
            spec=dict(rows=3, pad=True, out="row_range", state="bf16"),
        ),
    )

    note = """
    Metadata consumed: `slots`. Preconditions and launch: `GdnDecodeRowsGatedNorm`'s, plus
    a bf16 `out` `[rows, HV * V]` with unit column stride. The pool rows and `slots` are
    read before `griddepcontrol.wait`: in a mixed step nothing queued ahead of the call
    writes the generation rows' slots (the context rows' reset and chunked recurrence touch
    only theirs). Numerics: the recurrence of `GdnDecodeRowsGatedNorm` (the same kernel).

    Certified: a mixed step's 121 generation rows, 3 rows with a pad slot (-1), and 2 rows
    over the cache manager's coalesced pool, each into a row range of the step output.
    """

    def __call__(
        self,
        proj: torch.Tensor,
        pool: torch.Tensor,
        slots: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        k_heads: int,
        kdim: int,
        out: torch.Tensor,
        pdl: bool = True,
    ) -> torch.Tensor:
        return launch(proj, pool, slots, A_log, dt_bias, None, 0.0, k_heads, kdim, pdl, out=out)

    def reference(
        self,
        proj: torch.Tensor,
        pool: torch.Tensor,
        slots: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        k_heads: int,
        kdim: int,
        out: torch.Tensor,
        pdl: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`(o, pool_after)`: the fp64 recurrence, o rounded to bf16 (the pool to its dtype)."""
        rows = proj.shape[0]
        v_heads, vdim = pool.shape[1], pool.shape[2]
        q, k, v, _, a, b = torch.split(
            proj,
            [k_heads * kdim, k_heads * kdim, v_heads * vdim, v_heads * vdim, v_heads, v_heads],
            1,
        )
        o, pool_after = _recurrence_reference(
            q.reshape(rows, 1, k_heads, kdim),
            k.reshape(rows, 1, k_heads, kdim),
            v.reshape(rows, 1, v_heads, vdim),
            A_log,
            a.reshape(rows, 1, v_heads),
            dt_bias,
            b.reshape(rows, 1, v_heads),
            pool,
            slots,
        )
        return o.reshape(rows, v_heads * vdim), pool_after

    def is_valid(
        self,
        proj: torch.Tensor,
        pool: torch.Tensor,
        slots: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        k_heads: int,
        kdim: int,
        out: torch.Tensor,
        pdl: bool = True,
    ) -> None:
        v_heads, vdim = pool.shape[1], pool.shape[2]
        gdn_decode_rows_gated_norm.is_valid(
            proj,
            pool,
            slots,
            A_log,
            dt_bias,
            torch.empty(vdim, dtype=torch.bfloat16),
            0.0,
            k_heads,
            kdim,
        )
        assert out.dtype == torch.bfloat16 and out.dim() == 2, "bf16 out [rows, HV * V]"
        assert out.shape == (proj.shape[0], v_heads * vdim) and out.stride(1) == 1, (
            "out [rows, HV * V], unit column stride"
        )


gdn_decode_rows_gated_norm = GdnDecodeRowsGatedNorm()
gdn_decode_rows = GdnDecodeRows()
