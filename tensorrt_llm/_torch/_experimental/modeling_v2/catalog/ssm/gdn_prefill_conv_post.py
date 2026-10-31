# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gated-DeltaNet prefill preparation in one pass: the causal depthwise conv1d
of the context rows, read token-major straight off the input projection, with
the SiLU, the q/k split and L2 norm, the v copy and the decay/beta gates fused
in, and each sequence's conv window published to the paged pool.

The conv pass is CuTe DSL, the gates and the window publish Triton, all three
written for the Qwen3.8-Flash-Next tp1 target (its `ops/ssm/gdn_prefill_conv_post`)
and reused here at the unified target's per-rank geometries, the conv pass reading a pool
slot through a 64-bit base address (on a dep4 rank's stock conv page a slot's offset
passes 2^31 elements from slot 1942). The three-call
composition it replaces -- `extract_transpose_prefill_slice` (token-major ->
channel-major copy), `causal_conv1d_fwd` (channel-major conv, in place) and
`fused_gdn_post_conv` (split + norm + gates) -- moves the `[T, 10240]`
activation through HBM three times; no ``torch.ops.trtllm.*`` op, in-tree
callable or installed FlashInfer API fuses the conv with the norm, and the
channel-last `causal_conv1d_fwd` kernel alone is slower than this whole pass.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
import triton
import triton.language as tl
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import T, dsl_user_op
from triton.language.extra.cuda import libdevice

from tensorrt_llm._torch.kernels.blackwell.utils import make_ptr

from .._op import Arch, Cell, OpWrapper

#: This checkpoint's linear-attention geometry at tp1.
NUM_K_HEADS, HEAD_K_DIM, NUM_V_HEADS, HEAD_V_DIM = 16, 128, 48, 128
QKV_DIM = 2 * NUM_K_HEADS * HEAD_K_DIM + NUM_V_HEADS * HEAD_V_DIM  # 10240
#: The target's one GDN input-projection output: [q|k|v | z | a | b].
IN_PROJ_WIDTH = QKV_DIM + NUM_V_HEADS * HEAD_V_DIM + 2 * NUM_V_HEADS  # 16480
CONV_WIDTH = 4
#: Largest conv-pool slot count the cache manager was seen to allocate for this
#: checkpoint (the `causal_conv1d_fwd` entry's ENGINE_SLOTS).
ENGINE_SLOTS = 461
#: One tp4/tep4 rank's share: the heads split four ways, its own `[q|k|v|z|a|b]`
#: input projection 2560 + 1536 + 2 * 12 = 4120 wide.
RANK4_K_HEADS, RANK4_V_HEADS = NUM_K_HEADS // 4, NUM_V_HEADS // 4
RANK4_IN_PROJ_WIDTH = (2 * RANK4_K_HEADS + 2 * RANK4_V_HEADS) * HEAD_K_DIM + 2 * RANK4_V_HEADS
#: Conv-pool slots the stock engine bound per rank at core's config (n-gram table
#: host-resident): 1874 at tep4, 850 at dep4. tp4 has no engine to count it; its
#: rank holds tep4's GDN state, so tep4's count bounds it.
TEP4_SLOTS, DEP4_SLOTS = 1874, 850
TP4_SLOTS = TEP4_SLOTS
#: The stock manager's conv page: one slot holds all 36 GDN layers' `[C, 3]` states,
#: so a layer's view has slot stride `36 * C * 3` and the last layer sits `35 * C * 3`
#: elements into the page.
PAGE_GDN_LAYERS = 36
A_LOG_RANGE = (-5.5625, 3.5625)
DT_BIAS_RANGE = (-8.9375, 19.25)
SOFTPLUS_THRESHOLD = 20.0

#: Channels per program of the window publish.
_PUBLISH_BLOCK_C = 512
#: Tokens per program of the gates: at 16384 tokens on GB300 4.4 us (6.1 at 64, 5.3 at
#: 16 or 32).
_GATES_BLOCK_T = 8


@triton.jit
def _gdn_prefill_conv_publish_kernel(
    x_ptr,
    stride_xt,
    state_ptr,
    stride_ss,
    stride_sc,
    stride_sw,
    qsl_ptr,
    slot_ptr,
    init_ptr,
    conv_dim,
    pad_slot_id,
    BLOCK_C: tl.constexpr,
):
    """One program: one sequence x BLOCK_C channels of its new 3-column window.

    New column j holds position `e - 3 + j`: the raw input when it lies inside
    the sequence, else the carried column `L + j` (or 0 without a carry). All
    three are loaded before any is stored, and each channel's three columns
    live in one thread, so a short sequence's shift reads the old window."""
    n = tl.program_id(0)
    c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c < conv_dim
    slot = tl.load(slot_ptr + n)
    if slot != pad_slot_id:
        s0 = tl.load(qsl_ptr + n)
        e = tl.load(qsl_ptr + n + 1)
        carry = tl.load(init_ptr + n) != 0
        base = state_ptr + slot.to(tl.int64) * stride_ss + c.to(tl.int64) * stride_sc
        p0 = e - 3
        p1 = e - 2
        p2 = e - 1
        x0 = tl.load(x_ptr + p0.to(tl.int64) * stride_xt + c, mask=c_mask & (p0 >= s0), other=0.0)
        x1 = tl.load(x_ptr + p1.to(tl.int64) * stride_xt + c, mask=c_mask & (p1 >= s0), other=0.0)
        x2 = tl.load(x_ptr + p2.to(tl.int64) * stride_xt + c, mask=c_mask & (p2 >= s0), other=0.0)
        s_0 = tl.load(base + (3 + p0 - s0) * stride_sw, mask=c_mask & (p0 < s0) & carry, other=0.0)
        s_1 = tl.load(base + (3 + p1 - s0) * stride_sw, mask=c_mask & (p1 < s0) & carry, other=0.0)
        s_2 = tl.load(base + (3 + p2 - s0) * stride_sw, mask=c_mask & (p2 < s0) & carry, other=0.0)
        n0 = tl.where(p0 >= s0, x0, s_0)
        n1 = tl.where(p1 >= s0, x1, s_1)
        n2 = tl.where(p2 >= s0, x2, s_2)
        tl.store(base, n0, mask=c_mask)
        tl.store(base + stride_sw, n1, mask=c_mask)
        tl.store(base + 2 * stride_sw, n2, mask=c_mask)


# --------------------------------------------------------------------------------------
# The conv pass in CuTe DSL. A CTA is one 128-channel head x `_TT` tokens, 128 threads:
# it first copies its `_TT + 3`-row tile (3 history rows) into shared memory with
# 16-byte `cp.async` (rows outside `[0, T)` zero-filled), so the bytes in flight cost no
# registers; then a thread (4 channels x `_TPT` consecutive tokens) slides the width-4
# window over shared-memory rows, every element converted to fp32 once. The conv runs
# as `fma.rn.f32x2` on channel pairs (first tap from zero, then in tap order: per element
# an `fma.rn.f32` chain); the SiLU is `acc * rcp(1 + 2^(-acc * log2e))`
# with `ex2.approx.ftz` and `rcp.approx.ftz`, rounded in pairs by `cvt.rn.bf16x2.f32`.
# --------------------------------------------------------------------------------------

_VEC = 4
_TPT = 16
_RG = 4
_TT = _TPT * _RG
_TILE_ROWS = _TT + 3
_THREADS = 128
_CHUNKS = _TILE_ROWS * 16


def _iv(x, loc, ip):
    return x.ir_value(loc=loc, ip=ip)


def _asm1(ret, args, text, constraints, loc, ip, side_effects=False):
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


def _asm2(ty, args, text, constraints, loc, ip, side_effects=False):
    res = _asm1(
        llvm.StructType.get_literal([ty, ty]), args, text, constraints, loc, ip, side_effects
    )
    return (
        llvm.extractvalue(ty, res, [0], loc=loc, ip=ip),
        llvm.extractvalue(ty, res, [1], loc=loc, ip=ip),
    )


@dsl_user_op
def _ffma2(a0, a1, b0, b1, c0, c1, *, loc=None, ip=None):
    """`fma.rn.f32x2`: two fp32 FMAs, each rounded once."""
    r = _asm2(
        T.f32(),
        [cutlass.Float32(v) for v in (a0, a1, b0, b1, c0, c1)],
        "{ .reg .b64 a, b, c, d; mov.b64 a, {$2, $3}; mov.b64 b, {$4, $5}; "
        "mov.b64 c, {$6, $7}; fma.rn.f32x2 d, a, b, c; mov.b64 {$0, $1}, d; }",
        "=f,=f,f,f,f,f,f,f",
        loc,
        ip,
    )
    return cutlass.Float32(r[0]), cutlass.Float32(r[1])


@dsl_user_op
def _fmul2(a0, a1, b0, b1, *, loc=None, ip=None):
    """`mul.rn.f32x2`."""
    r = _asm2(
        T.f32(),
        [cutlass.Float32(v) for v in (a0, a1, b0, b1)],
        "{ .reg .b64 a, b, d; mov.b64 a, {$2, $3}; mov.b64 b, {$4, $5}; "
        "mul.rn.f32x2 d, a, b; mov.b64 {$0, $1}, d; }",
        "=f,=f,f,f,f,f",
        loc,
        ip,
    )
    return cutlass.Float32(r[0]), cutlass.Float32(r[1])


@dsl_user_op
def _silu2(a0, a1, *, loc=None, ip=None):
    """`acc * rcp(1 + 2^(acc * -log2e))` for two values: `mul.rn.f32x2`, `ex2.approx.ftz`,
    `add.rn.f32x2`, `rcp.approx.ftz`, `mul.rn`. For `1 + e` in [1, 2^126] this equals a
    Triton `acc / (1 + exp(-acc))` (`div.full.f32`); past it (acc < -87.3) it gives -0
    where that gives a value below 1e-36."""
    r = _asm2(
        T.f32(),
        [cutlass.Float32(a0), cutlass.Float32(a1)],
        "{ .reg .b64 a, k, t, o, d; .reg .f32 c, t0, t1, d0, d1; "
        "mov.f32 c, 0fBFB8AA3B; mov.b64 k, {c, c}; mov.f32 c, 0f3F800000; mov.b64 o, {c, c}; "
        "mov.b64 a, {$2, $3}; mul.rn.f32x2 t, a, k; mov.b64 {t0, t1}, t; "
        "ex2.approx.ftz.f32 t0, t0; ex2.approx.ftz.f32 t1, t1; mov.b64 t, {t0, t1}; "
        "add.rn.f32x2 d, t, o; mov.b64 {d0, d1}, d; "
        "rcp.approx.ftz.f32 d0, d0; rcp.approx.ftz.f32 d1, d1; "
        "mul.rn.f32 $0, $2, d0; mul.rn.f32 $1, $3, d1; }",
        "=f,=f,f,f",
        loc,
        ip,
    )
    return cutlass.Float32(r[0]), cutlass.Float32(r[1])


@dsl_user_op
def _fma_rn(a, b, c, *, loc=None, ip=None):
    return cutlass.Float32(
        _asm1(
            T.f32(),
            [cutlass.Float32(a), cutlass.Float32(b), cutlass.Float32(c)],
            "fma.rn.f32 $0, $1, $2, $3;",
            "=f,f,f,f",
            loc,
            ip,
        )
    )


@dsl_user_op
def _inv_norm(ss, eps, *, loc=None, ip=None):
    """1 / sqrt(ss + eps): `add.f32`, `sqrt.approx.ftz.f32`, `div.full.f32` (a Triton
    `1.0 / tl.sqrt(...)`)."""
    return cutlass.Float32(
        _asm1(
            T.f32(),
            [cutlass.Float32(ss), cutlass.Float32(eps)],
            "{ .reg .f32 t; add.f32 t, $1, $2; sqrt.approx.ftz.f32 t, t; "
            "div.full.f32 $0, 0f3F800000, t; }",
            "=f,f,f",
            loc,
            ip,
        )
    )


@dsl_user_op
def _bf16x2(lo, hi, *, loc=None, ip=None):
    """`cvt.rn.bf16x2.f32`: `lo` in the low half (the lower address)."""
    return cutlass.Int32(
        _asm1(
            T.i32(),
            [cutlass.Float32(hi), cutlass.Float32(lo)],
            "cvt.rn.bf16x2.f32 $0, $1, $2;",
            "=r,f,f",
            loc,
            ip,
        )
    )


@dsl_user_op
def _unpack2(w, *, loc=None, ip=None):
    """A bf16x2 word as `(low, high)` fp32."""
    r = _asm2(
        T.f32(),
        [cutlass.Int32(w)],
        "{ .reg .b32 l, h; shl.b32 l, $2, 16; and.b32 h, $2, 0xFFFF0000; "
        "mov.b32 $0, l; mov.b32 $1, h; }",
        "=f,=f,r",
        loc,
        ip,
    )
    return cutlass.Float32(r[0]), cutlass.Float32(r[1])


@dsl_user_op
def _st_v2(addr, a, b, *, loc=None, ip=None):
    _asm1(
        T.i32(),
        [cutlass.Int64(addr), cutlass.Int32(a), cutlass.Int32(b)],
        "{ st.global.v2.b32 [$1], {$2, $3}; mov.u32 $0, 0; }",
        "=r,l,r,r",
        loc,
        ip,
        side_effects=True,
    )


@dsl_user_op
def _cp_async16(smem_addr, gmem_addr, src_bytes, *, loc=None, ip=None):
    """16 bytes global -> shared, the rest of the 16 zero-filled past `src_bytes`."""
    _asm1(
        T.i32(),
        [cutlass.Int32(smem_addr), cutlass.Int64(gmem_addr), cutlass.Int32(src_bytes)],
        "{ cp.async.cg.shared.global [$1], [$2], 16, $3; mov.u32 $0, 0; }",
        "=r,r,l,r",
        loc,
        ip,
        side_effects=True,
    )


@dsl_user_op
def _ld_bf16(addr, *, loc=None, ip=None):
    """One bf16 from global memory at a 64-bit byte address, as fp32."""
    return cutlass.Float32(
        _asm1(
            T.f32(),
            [cutlass.Int64(addr)],
            "{ .reg .b32 r; ld.global.u16 r, [$1]; shl.b32 r, r, 16; mov.b32 $0, r; }",
            "=f,l",
            loc,
            ip,
            side_effects=True,
        )
    )


@dsl_user_op
def _cp_async_wait_all(*, loc=None, ip=None):
    _asm1(T.i32(), [], "{ cp.async.wait_all; mov.u32 $0, 0; }", "=r", loc, ip, side_effects=True)


@dsl_user_op
def _lds_v2(addr, *, loc=None, ip=None):
    r = _asm2(
        T.i32(),
        [cutlass.Int32(addr)],
        "ld.shared.v2.b32 {$0, $1}, [$2];",
        "=r,=r,r",
        loc,
        ip,
        side_effects=True,
    )
    return cutlass.Int32(r[0]), cutlass.Int32(r[1])


class _ConvPassKernel:
    """q / k / v of the conv pass; the gates and the publish are separate launches."""

    def __init__(self, num_k_heads: int, num_v_heads: int, head_dim: int, x_stride: int):
        assert head_dim == _VEC * 32
        self.hk, self.hv, self.d = num_k_heads, num_v_heads, head_dim
        self.heads = 2 * num_k_heads + num_v_heads
        self.qkv = self.heads * head_dim
        self.sx = x_stride

    @cute.jit
    def __call__(
        self,
        x_ptr: cute.Pointer,
        w_ptr: cute.Pointer,
        st_ptr: cute.Pointer,
        qsl_ptr: cute.Pointer,
        slot_ptr: cute.Pointer,
        init_ptr: cute.Pointer,
        q_t: cute.Tensor,
        k_ptr: cute.Pointer,
        v_ptr: cute.Pointer,
        num_tokens: cutlass.Int32,
        num_seqs: cutlass.Int32,
        search_steps: cutlass.Int32,
        pad_slot_id: cutlass.Int32,
        eps: cutlass.Float32,
        w_s0: cutlass.Int32,
        w_s1: cutlass.Int32,
        st_s0: cutlass.Int32,
        st_s1: cutlass.Int32,
        st_s2: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        weight = cute.make_tensor(w_ptr, cute.make_layout((self.qkv, 4), stride=(w_s0, w_s1)))
        # The pool goes in as its base address and strides: a carried column is read at
        # `base + slot * st_s0` in 64 bits (on the stock page that passes 2^31 elements
        # from slot 1942 at 16 / 48 heads, 7768 at 4 / 12) plus its in-slot offset in 32.
        qsl = cute.make_tensor(qsl_ptr, cute.make_layout((num_seqs + 1,), stride=(1,)))
        slots = cute.make_tensor(slot_ptr, cute.make_layout((num_seqs,), stride=(1,)))
        init = cute.make_tensor(init_ptr, cute.make_layout((num_seqs,), stride=(1,)))
        blocks = (num_tokens + _TT - 1) // _TT
        # `q` arrives as a tensor: the TVM-FFI entry takes its stream from a GPU tensor.
        self.kernel(
            x_ptr.toint(),
            weight,
            st_ptr.toint(),
            st_s0,
            st_s1,
            st_s2,
            qsl,
            slots,
            init,
            q_t.iterator.toint(),
            k_ptr.toint(),
            v_ptr.toint(),
            num_tokens,
            num_seqs,
            search_steps,
            pad_slot_id,
            eps,
        ).launch(
            grid=[self.heads, blocks, 1],
            block=[_THREADS, 1, 1],
            smem=_TILE_ROWS * 128 * 2 + 128,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x_base: cutlass.Int64,
        mW: cute.Tensor,
        st_base: cutlass.Int64,
        st_s0: cutlass.Int32,
        st_s1: cutlass.Int32,
        st_s2: cutlass.Int32,
        mQsl: cute.Tensor,
        mSlot: cute.Tensor,
        mInit: cute.Tensor,
        q_base: cutlass.Int64,
        k_base: cutlass.Int64,
        v_base: cutlass.Int64,
        num_tokens: cutlass.Int32,
        num_seqs: cutlass.Int32,
        search_steps: cutlass.Int32,
        pad_slot_id: cutlass.Int32,
        eps: cutlass.Float32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        head, tb, _ = cute.arch.block_idx()
        lane = tidx % 32
        rg = tidx // 32
        t0 = tb * _TT

        smem = cutlass.utils.SmemAllocator()
        tile = smem.allocate_tensor(
            cutlass.BFloat16, cute.make_layout((_TILE_ROWS * 128,), stride=(1,)), byte_alignment=16
        )
        tile_base = cutlass.Int32(tile.iterator.toint())

        # 1. The tile: rows t0-3 .. t0+TT-1 of this head's 128 channels, zero-filled outside [0, T).
        for it in cutlass.range_constexpr((_CHUNKS + _THREADS - 1) // _THREADS):
            chunk = tidx + it * _THREADS
            if chunk < _CHUNKS:
                r = chunk // 16
                cc = chunk - r * 16
                row = t0 - 3 + r
                valid = cutlass.Int32(0)
                if row >= 0:
                    if row < num_tokens:
                        valid = cutlass.Int32(16)
                rowc = cutlass.min(cutlass.max(row, cutlass.Int32(0)), num_tokens - 1)
                _cp_async16(
                    tile_base + (r * 128 + cc * 8) * 2,
                    x_base + (cutlass.Int64(rowc) * self.sx + head * 128 + cc * 8) * 2,
                    valid,
                )

        # 2. While the copies land: the weights, the first token's sequence, the output row.
        c0 = head * 128 + lane * _VEC
        w = cute.make_rmem_tensor(cute.make_layout((_VEC, 4), stride=(4, 1)), cutlass.Float32)
        for i in cutlass.range_constexpr(_VEC * 4):
            w[i // 4, i % 4] = mW[c0 + i // 4, i % 4].to(cutlass.Float32)
        tf = t0 + rg * _TPT
        lo = cutlass.Int32(0)
        hi = num_seqs
        for _ in cutlass.range(search_steps, unroll=1):
            mid = (lo + hi) // 2
            if mQsl[mid] <= tf:
                lo = mid
            else:
                hi = mid
        s0f = mQsl[lo]
        ef = mQsl[lo + 1]
        # One uniform branch a thread: its tokens and their windows inside one sequence.
        fast = cutlass.Int32(0)
        if tf - s0f >= 3:
            if tf + _TPT <= ef:
                if tf + _TPT <= num_tokens:
                    fast = cutlass.Int32(1)
        is_qk = head < 2 * self.hk
        o_base = v_base + cutlass.Int64((head - 2 * self.hk) * 128 + lane * _VEC) * 2
        o_stride = cutlass.Int64(self.hv * 128 * 2)
        if head < self.hk:
            o_base = q_base + cutlass.Int64(head * 128 + lane * _VEC) * 2
            o_stride = cutlass.Int64(self.hk * 128 * 2)
        else:
            if is_qk:
                o_base = k_base + cutlass.Int64((head - self.hk) * 128 + lane * _VEC) * 2
                o_stride = cutlass.Int64(self.hk * 128 * 2)

        _cp_async_wait_all()
        cute.arch.sync_threads()

        # 3. The window over tile rows rg*TPT .. (tile row r is token t0 - 3 + r).
        win = cute.make_rmem_tensor(cute.make_layout((4, _VEC), stride=(_VEC, 1)), cutlass.Float32)
        for i in cutlass.range_constexpr(3):
            lw, hw = _lds_v2(tile_base + ((rg * _TPT + i) * 128 + lane * _VEC) * 2)
            win[i, 0], win[i, 1] = _unpack2(lw)
            win[i, 2], win[i, 3] = _unpack2(hw)
        acc = cute.make_rmem_tensor(cute.make_layout((_VEC,), stride=(1,)), cutlass.Float32)
        for j in cutlass.range_constexpr(_TPT):
            t = tf + j
            lw, hw = _lds_v2(tile_base + ((rg * _TPT + j + 3) * 128 + lane * _VEC) * 2)
            win[(j + 3) % 4, 0], win[(j + 3) % 4, 1] = _unpack2(lw)
            win[(j + 3) % 4, 2], win[(j + 3) % 4, 3] = _unpack2(hw)
            if fast == 1:
                for p in cutlass.range_constexpr(_VEC // 2):
                    a0, a1 = _ffma2(
                        win[j % 4, 2 * p],
                        win[j % 4, 2 * p + 1],
                        w[2 * p, 0],
                        w[2 * p + 1, 0],
                        cutlass.Float32(0.0),
                        cutlass.Float32(0.0),
                    )
                    for kk in cutlass.range_constexpr(1, 4):
                        a0, a1 = _ffma2(
                            win[(j + kk) % 4, 2 * p],
                            win[(j + kk) % 4, 2 * p + 1],
                            w[2 * p, kk],
                            w[2 * p + 1, kk],
                            a0,
                            a1,
                        )
                    acc[2 * p] = a0
                    acc[2 * p + 1] = a1
            else:
                # The token's sequence; a tap before its start reads the carried
                # column W-1-(s0-pos), or 0 without a carry.
                sq = lo
                s0 = s0f
                if t >= ef:
                    lo2 = cutlass.Int32(0)
                    hi2 = num_seqs
                    for _ in cutlass.range(search_steps, unroll=1):
                        mid = (lo2 + hi2) // 2
                        if mQsl[mid] <= t:
                            lo2 = mid
                        else:
                            hi2 = mid
                    sq = lo2
                    s0 = mQsl[lo2]
                slot = mSlot[sq]
                init = cutlass.Int32(mInit[sq])
                slot_base = st_base + cutlass.Int64(slot) * cutlass.Int64(st_s0) * 2
                for v in cutlass.range_constexpr(_VEC):
                    a = cutlass.Float32(0.0)
                    for kk in cutlass.range_constexpr(4):
                        pos = t - 3 + kk
                        xv = win[(j + kk) % 4, v]
                        if pos < s0:
                            xv = cutlass.Float32(0.0)
                            if init != 0:
                                if slot != pad_slot_id:
                                    xv = _ld_bf16(
                                        slot_base
                                        + cutlass.Int64((c0 + v) * st_s1 + (3 + pos - s0) * st_s2)
                                        * 2
                                    )
                        a = _fma_rn(xv, w[v, kk], a)
                    acc[v] = a
            y0, y1 = _silu2(acc[0], acc[1])
            y2, y3 = _silu2(acc[2], acc[3])
            p01 = _bf16x2(y0, y1)
            p23 = _bf16x2(y2, y3)
            if is_qk:
                # L2 norm of the rounded head row: fp32 squares, a 32-lane butterfly.
                f0, f1 = _unpack2(p01)
                f2, f3 = _unpack2(p23)
                s_a, s_b = _ffma2(f0, f1, f0, f1, cutlass.Float32(0.0), cutlass.Float32(0.0))
                s_a, s_b = _ffma2(f2, f3, f2, f3, s_a, s_b)
                ss = s_a + s_b
                for off in cutlass.range_constexpr(5):
                    ss = ss + cute.arch.shuffle_sync_bfly(ss, offset=1 << off)
                inv = _inv_norm(ss, eps)
                n0, n1 = _fmul2(f0, f1, inv, inv)
                n2, n3 = _fmul2(f2, f3, inv, inv)
                p01 = _bf16x2(n0, n1)
                p23 = _bf16x2(n2, n3)
            if t < num_tokens:
                _st_v2(o_base + cutlass.Int64(t) * o_stride, p01, p23)


@triton.jit
def _gdn_gates_kernel(
    a_ptr,
    stride_at,
    stride_ah,
    b_ptr,
    stride_bt,
    stride_bh,
    A_log_ptr,
    dt_bias_ptr,
    g_ptr,
    beta_ptr,
    num_tokens,
    softplus_threshold,
    NUM_V_HEADS: tl.constexpr,
    HP: tl.constexpr,
    BLOCK_T: tl.constexpr,
    LINEAR_GATE: tl.constexpr,
):
    """The decay/beta gates over `[T, NUM_V_HEADS]`, one row-coalesced tile a program;
    with `LINEAR_GATE` the decay stored as `exp(g)` (libdevice `expf`, as `torch.exp`)."""
    t = (tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T))[:, None]
    h = tl.arange(0, HP)[None, :]
    m = (t < num_tokens) & (h < NUM_V_HEADS)
    t64 = t.to(tl.int64)
    a_val = tl.load(a_ptr + t64 * stride_at + h * stride_ah, mask=m, other=0.0)
    b_val = tl.load(b_ptr + t64 * stride_bt + h * stride_bh, mask=m, other=0.0)
    A_log = tl.load(A_log_ptr + h, mask=h < NUM_V_HEADS, other=0.0).to(tl.float32)
    dt_bias = tl.load(dt_bias_ptr + h, mask=h < NUM_V_HEADS, other=0.0).to(tl.float32)
    gate_input = a_val.to(tl.float32) + dt_bias
    softplus = tl.where(
        gate_input <= softplus_threshold, tl.log(1.0 + tl.exp(gate_input)), gate_input
    )
    g_off = t64 * NUM_V_HEADS + h
    g = -tl.exp(A_log) * softplus
    if LINEAR_GATE:
        g = libdevice.exp(g)
    tl.store(g_ptr + g_off, g, mask=m)
    tl.store(beta_ptr + g_off, tl.sigmoid(b_val.to(tl.float32)), mask=m)


_CONV_PASS: dict = {}


def _cute_conv_pass_ok(x, weight, conv_states, num_k_heads, head_k_dim, num_v_heads) -> bool:
    """The CuTe conv pass's preconditions: bf16, 128-channel heads, 16-byte aligned rows
    of `x` (its `cp.async` copies), int32 element strides and an int32 offset inside a
    pool slot (a slot's base is 64-bit)."""
    in_slot = (conv_states.shape[1] - 1) * conv_states.stride(1) + 2 * conv_states.stride(2)
    return (
        x.dtype == torch.bfloat16
        and head_k_dim == _VEC * 32
        and x.data_ptr() % 16 == 0
        and (x.stride(0) * 2) % 16 == 0
        and max(*weight.stride(), *conv_states.stride(), in_slot) < 2**31
    )


def _cute_conv_pass(
    x, weight, conv_states, query_start_loc, cache_indices, has_initial_state, q, k, v,
    num_k_heads, num_v_heads, head_k_dim, pad_slot_id, l2_norm_eps,
):  # fmt: skip
    num_tokens = x.shape[0]
    num_seqs = query_start_loc.shape[0] - 1
    key = (num_k_heads, num_v_heads, head_k_dim, x.stride(0))
    q2 = q.view(num_tokens, -1)
    ptrs = (
        x.data_ptr(),
        weight.data_ptr(),
        conv_states.data_ptr(),
        query_start_loc.data_ptr(),
        cache_indices.data_ptr(),
        has_initial_state.data_ptr(),
        q2,
        k.data_ptr(),
        v.data_ptr(),
    )
    fn = _CONV_PASS.get(key)
    if fn is None:
        gm = cute.AddressSpace.gmem
        types = (
            (cutlass.BFloat16, 16),
            (cutlass.BFloat16, 2),
            (cutlass.BFloat16, 2),
            (cutlass.Int32, 4),
            (cutlass.Int32, 4),
            (cutlass.Uint8, 1),
            None,
            (cutlass.BFloat16, 16),
            (cutlass.BFloat16, 16),
        )
        cptrs = [
            from_dlpack(p, assumed_align=16).mark_layout_dynamic(leading_dim=1)
            if spec is None
            else make_ptr(spec[0], p, gm, assumed_align=spec[1])
            for spec, p in zip(types, ptrs)
        ]
        fn = cute.compile(
            _ConvPassKernel(num_k_heads, num_v_heads, head_k_dim, x.stride(0)),
            *cptrs,
            cutlass.Int32(1),
            cutlass.Int32(1),
            cutlass.Int32(1),
            cutlass.Int32(-1),
            cutlass.Float32(1e-6),
            *(cutlass.Int32(1),) * 5,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--opt-level 2 --enable-tvm-ffi",
        )
        _CONV_PASS[key] = fn
    fn(
        *ptrs,
        num_tokens,
        num_seqs,
        max(num_seqs - 1, 0).bit_length() + 1,
        pad_slot_id,
        l2_norm_eps,
        *weight.stride(),
        *conv_states.stride(),
    )


def _launch(
    x,
    weight,
    conv_states,
    query_start_loc,
    cache_indices,
    has_initial_state,
    a,
    b,
    A_log,
    dt_bias,
    num_k_heads,
    head_k_dim,
    num_v_heads,
    head_v_dim,
    pad_slot_id,
    l2_norm_eps,
    softplus_threshold,
    linear_gate=False,
):
    assert head_k_dim == head_v_dim and weight.shape[-1] == CONV_WIDTH, (
        "one program per 128-channel head: head_k_dim must equal head_v_dim, and the "
        f"conv width must be {CONV_WIDTH}"
    )
    num_tokens = x.shape[0]
    num_seqs = query_start_loc.shape[0] - 1
    conv_dim = 2 * num_k_heads * head_k_dim + num_v_heads * head_v_dim
    dev, dt = x.device, x.dtype
    q = torch.empty((1, num_tokens, num_k_heads, head_k_dim), dtype=dt, device=dev)
    k = torch.empty_like(q)
    v = torch.empty((1, num_tokens, num_v_heads, head_v_dim), dtype=dt, device=dev)
    g = torch.empty((1, num_tokens, num_v_heads), dtype=torch.float32, device=dev)
    beta = torch.empty((1, num_tokens, num_v_heads), dtype=torch.float32, device=dev)
    if num_seqs == 0:
        return q, k, v, g, beta
    if num_tokens > 0:
        assert _cute_conv_pass_ok(x, weight, conv_states, num_k_heads, head_k_dim, num_v_heads), (
            "the conv pass takes bf16 x with 128-channel heads, 16-byte aligned rows and "
            f"int32 weight/pool strides; got {x.dtype}, head_k_dim {head_k_dim}, x at "
            f"{x.data_ptr() % 16} mod 16 with row stride {x.stride(0)}"
        )
        _cute_conv_pass(
            x, weight, conv_states, query_start_loc, cache_indices, has_initial_state, q, k, v,
            num_k_heads, num_v_heads, head_k_dim, pad_slot_id, l2_norm_eps,
        )  # fmt: skip
        _gdn_gates_kernel[(triton.cdiv(num_tokens, _GATES_BLOCK_T),)](
            a,
            a.stride(0),
            a.stride(1),
            b,
            b.stride(0),
            b.stride(1),
            A_log,
            dt_bias,
            g,
            beta,
            num_tokens,
            softplus_threshold,
            NUM_V_HEADS=num_v_heads,
            HP=triton.next_power_of_2(num_v_heads),
            BLOCK_T=_GATES_BLOCK_T,
            LINEAR_GATE=linear_gate,
            num_warps=4,
        )
    _gdn_prefill_conv_publish_kernel[(num_seqs, triton.cdiv(conv_dim, _PUBLISH_BLOCK_C))](
        x,
        x.stride(0),
        conv_states,
        *conv_states.stride(),
        query_start_loc,
        cache_indices,
        has_initial_state,
        conv_dim,
        pad_slot_id,
        BLOCK_C=_PUBLISH_BLOCK_C,
        num_warps=4,
    )
    return q, k, v, g, beta


def _engine_slots(num_seqs: int) -> tuple[int, ...]:
    """`num_seqs` distinct slots of the engine pool, the last and slot 0 first."""
    rest = tuple((i * 37 + 4) % (ENGINE_SLOTS - 2) + 1 for i in range(max(num_seqs - 2, 0)))
    slots = ((ENGINE_SLOTS - 1, 0) + rest)[:num_seqs]
    assert len(set(slots)) == num_seqs
    return slots


def _rank_cell(rank4: bool, lengths, initial, pool_slots, slots, linear_gate) -> dict:
    """A rank's call: tp4/tep4 (`rank4`, 4 k / 12 v heads over its 4120-wide projection) or
    dep4 (all 16 / 48 heads, 16480), over the stock conv page at `pool_slots`."""
    heads = (RANK4_K_HEADS, RANK4_V_HEADS) if rank4 else (NUM_K_HEADS, NUM_V_HEADS)
    return dict(
        num_k_heads=heads[0],
        num_v_heads=heads[1],
        head_dim=HEAD_K_DIM,
        x_row_stride=RANK4_IN_PROJ_WIDTH if rank4 else IN_PROJ_WIDTH,
        lengths=lengths,
        initial=initial,
        pool_slots=pool_slots,
        page_layers=PAGE_GDN_LAYERS,
        slots=slots,
        linear_gate=linear_gate,
    )


class GdnPrefillConvPost(OpWrapper):
    """Causal conv + SiLU of the context rows, then the GDN q/k/v/g/beta.

    `x` is `[T, >= qkv_dim]` token-major with unit channel stride (any row
    stride: the target passes the q|k|v column view of its `[T, 16480]` input
    projection), packed `[q | k | v]`, `qkv_dim = 2 * num_k_heads * head_k_dim
    + num_v_heads * head_v_dim`. For sequence `s` over rows `[b, e)` of
    `query_start_loc` and `slot = cache_indices[s]`:

        xpad = cat([conv_states[slot] if has_initial_state[s] else 0, x[b:e]])
        y[t] = bf16(silu(sum_k weight[:, k] * xpad[t + k]))   (fp32 taps, in order)
        q, k = y / sqrt(sum(y^2) + l2_norm_eps)               (per head, fp32, rounded once)
        v    = y                                              (bit-exact)
        g    = -exp(A_log) * softplus(a + dt_bias)            (fp32; exp(g) with linear_gate)
        beta = sigmoid(b)                                     (fp32)
        conv_states[slot] = xpad[e - b : e - b + W - 1]       (raw input, exact)

    -- the composition `extract_transpose_prefill_slice` -> `causal_conv1d_fwd`
    (SiLU, no bias) -> `fused_gdn_post_conv(prefill, None, ...)`, in one pass
    over the activation plus a per-sequence window publish.

    Fusion boundary: the conv-state read, conv, SiLU, split, q/k norm, gates
    and the state publish are inside. The projection, the generation rows'
    conv and the recurrence are the caller's (`use_qk_l2norm_in_kernel=False`
    downstream).

    Returns `(q, k, v, g, beta)`: fresh contiguous `[1, T, num_k_heads,
    head_k_dim]` x2, `[1, T, num_v_heads, head_v_dim]` in `x.dtype`, and
    `[1, T, num_v_heads]` fp32 twice. `conv_states` is advanced in place for
    every sequence whose slot is not `pad_slot_id`; nothing else is written.
    """

    ARCHS = frozenset({Arch.SM_103})

    # spec: lengths; slots (pool rows used, None -> _engine_slots); pool_slots;
    # initial; layout "in_proj" (x and a/b column views of one [T, 16480]
    # buffer, the target's call) | "rows" (x/a/b the first rows of a longer
    # projection, as a mixed step's context half), extra_rows. A rank's cell
    # (`_rank_cell`) also names num_k_heads, num_v_heads, head_dim, x_row_stride
    # (its own projection's width) and page_layers (the pool is the last layer's
    # view of the stock page); without them, the tp1 geometry over a two-layer buffer.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="target call, engine 461-slot pool: 37+1+128 tokens at slots 460, 0, 230",
            spec=dict(
                lengths=(37, 1, 128),
                pool_slots=ENGINE_SLOTS,
                slots=(ENGINE_SLOTS - 1, 0, ENGINE_SLOTS // 2),
                initial=(True, False, True),
            ),
        ),
        Cell(
            why="mixed step's context half: first 8065 rows of a 8192-row projection",
            spec=dict(
                lengths=(8065,),
                pool_slots=ENGINE_SLOTS,
                initial=(False,),
                layout="rows",
                extra_rows=127,
            ),
        ),
        Cell(
            why="target call with linear_gate: 3000+8192+5000 tokens, the middle one fresh",
            spec=dict(
                lengths=(3000, 8192, 5000),
                pool_slots=ENGINE_SLOTS,
                initial=(True, False, True),
                linear_gate=True,
            ),
        ),
        Cell(
            why="128 short contexts (the batch ceiling), some shorter than the window",
            spec=dict(
                lengths=tuple(1 + (i * 7) % 64 for i in range(128)),
                pool_slots=ENGINE_SLOTS,
                initial=tuple(i % 3 != 0 for i in range(128)),
            ),
        ),
        Cell(
            why="tep4 rank, 4/12 heads at stride 4120: fresh 8192-token context, last of 1874 slots",
            spec=_rank_cell(True, (8192,), (False,), TEP4_SLOTS, (TEP4_SLOTS - 1,), True),
        ),
        Cell(
            why="tep4 rank: 37+1+128 tokens, carry/fresh/carry, slots 1873, 0, 937 of 1874",
            spec=_rank_cell(
                True,
                (37, 1, 128),
                (True, False, True),
                TEP4_SLOTS,
                (TEP4_SLOTS - 1, 0, TEP4_SLOTS // 2),
                False,
            ),
        ),
        Cell(
            why="tp4 rank (tep4's 4/12 heads and 1874-slot bound): fresh 8192 tokens at slot 0",
            spec=_rank_cell(True, (8192,), (False,), TP4_SLOTS, (0,), True),
        ),
        Cell(
            why="dep4 rank, 16/48 heads: two 512-token chunks, the first carrying, 850 slots",
            spec=_rank_cell(
                False,
                (512, 512),
                (True, False),
                DEP4_SLOTS,
                (DEP4_SLOTS - 1, DEP4_SLOTS // 2),
                True,
            ),
        ),
        Cell(
            why="dep4 rank at max_num_tokens: one carrying 8192-token chunk, last of 850 slots",
            spec=_rank_cell(False, (8192,), (True,), DEP4_SLOTS, (DEP4_SLOTS - 1,), True),
        ),
        # Pools past the int32 slot offset (slot 1942 at 16 / 48 heads, 7768 at 4 / 12):
        # larger dep4 pools than core's (kv_nvfp4, reuse, the tune's 0.85 fraction).
        Cell(
            why="dep4 rank, 2048-slot pool: 37+1+128 tokens at slots 2047, 1942, 0, past int32",
            spec=_rank_cell(False, (37, 1, 128), (True, False, True), 2048, (2047, 1942, 0), True),
        ),
        Cell(
            why="dep4 rank, 4096-slot pool: a carrying 8192-token chunk at slot 4095, past 2^32",
            spec=_rank_cell(False, (8192,), (True,), 4096, (4095,), True),
        ),
        Cell(
            why="tp4/tep4 rank, 8192-slot pool: a fresh 8192-token context at slot 8191",
            spec=_rank_cell(True, (8192,), (False,), 8192, (8191,), True),
        ),
        Cell(
            why="tp4/tep4 rank, 8192-slot pool: carrying 8192 tokens at slot 8191, past int32",
            spec=_rank_cell(True, (8192,), (True,), 8192, (8191,), True),
        ),
    )

    note = """
    Metadata consumed: none. Batch layout (`query_start_loc`, int32 `[N + 1]`,
    `query_start_loc[0] == 0`, `query_start_loc[N] == x.shape[0]`), slot map
    (`cache_indices`, int32 `[N]`) and carry flags (`has_initial_state`, bool
    `[N]`) are explicit.

    Call sites (`PrefillTarget._gdn_prefill_prep`, from `_gdn_block`): the
    context rows of every prefill-carrying step -- prefill-only `(mixed_qkv,
    a, b, qsl_all, si_all, hi_all)`, mixed and MTP-verify `(mq_p, a_p, b_p,
    qsl_p, si_p, hi_p)` -- with the layer's `[10240, 4]` conv weight, its
    whole conv pool unsliced, `pad_slot_id` -1, `l2_norm_eps` 1e-6 and
    `softplus_threshold` 20. `x`, `a`, `b` are column (and row) views, row
    stride 16480, of the one input-projection output; the outputs feed
    `flashinfer_chunk_gated_delta_rule` with `use_qk_l2norm_in_kernel=False`,
    and `linear_gate=True` on both: the decay leaves this pass as `exp(g)`,
    the factor FlashInfer's kernel takes, bitwise the `torch.exp` the chunk
    op would otherwise launch.

    The unified target (tp4, tep4, dep4) makes the same call per rank, from its
    prefill GDN core: a tp4 or tep4 rank over its own `[T, 4120]` projection
    (4 key / 12 value heads, conv dim 2560: its conv weight rows, its
    `A_log`/`dt_bias` entries and its conv pool are the rank's shares), a dep4
    rank over the whole `[T, 16480]` projection. The pool is the layer's view
    of the stock manager's conv page, measured (`get_conv_states` at tp4,
    tep4 and dep4 ranks, core config): one slot holds all 36 GDN layers'
    `[C, 3]` states, so the view has strides `(36 * C * 3, 3, 1)` --
    (276480, 3, 1) at C 2560, (1105920, 3, 1) at C 10240 -- and layer `i`
    starts `i * C * 3` elements into the page. The target's forward does
    not call this variant yet; the rank cells mirror tp1's call at the
    rank's widths.

    Three launches. The conv pass (CuTe DSL, `_ConvPassKernel`): grid
    `heads x cdiv(T, 64)` (80 heads at tp1 and dep4, 20 on a tp4/tep4
    rank), 128 threads; a CTA copies its 67-row tile
    (3 history rows) of the head's 128 channels into shared memory with
    16-byte `cp.async`, then a thread slides the width-4 window over 16
    tokens of 4 channels, each element converted once; each thread finds its
    first token's sequence by a binary search of `query_start_loc`
    (sync-free, capture-safe), and a thread whose 16 tokens and windows lie
    inside one sequence takes a straight-line branch, the rest read the pool
    token by token (a carried column at the slot's 64-bit base address plus
    its 32-bit in-slot offset, one `ld.global.u16`). The gates (Triton): one
    row-coalesced `[8, 48]` tile a program (`exp(g)` with `linear_gate`,
    libdevice `expf`). Then a
    per-sequence window publish (Triton), after both, so no
    program reads a window another has already advanced.

    Speed (one layer, 461-slot pool, in_proj views, us, graph replay over 4
    layers' inputs): 16384 tokens / 3 contexts 136 (the three-call chain it
    replaced 528).

    Numerics: the conv result is rounded to bf16 before the norm reads it, as
    the unfused path stores it. Conv taps: fp32 FMAs from zero in tap order
    (`fma.rn.f32x2` on channel pairs). SiLU: `acc * rcp(1 + 2^(-acc *
    log2e))` with `ex2.approx.ftz` and `rcp.approx.ftz` -- a Triton
    `acc / (1 + exp(-acc))`'s `ex2.approx` / `div.full` values wherever
    `1 + e <= 2^126`; below acc = -87.3 it returns -0 where those return a
    value under 1e-36. q/k norm: the 128-channel fp32 sum of squares as each
    thread's pairs plus a 32-lane butterfly, `sqrt.approx.ftz`, `div.full`.
    Against the unfused path (measured at the target-call cell): q/k/v within
    one bf16 ulp on 1e-6..1e-4 of elements, g/beta within the fp32 band, conv
    pool bitwise.

    Certified over the 461-slot pool, seen as one layer of a two-layer buffer:
    three contexts of 37, 1 and 128 tokens; a mixed step's 8065-token context
    half, the first rows of an 8192-row projection; 128 short contexts (1 to
    64 tokens, some shorter than the window); and the target's call with
    `linear_gate` (3000 + 8192 + 5000 tokens, the middle one fresh).

    Certified per rank, over the stock conv page's last-layer view at the
    slot counts the stock engine bound at core's config (`TEP4_SLOTS` 1874,
    `DEP4_SLOTS` 850; `TP4_SLOTS` is tep4's by bound), each also checked to
    leave the page's other 35 layers untouched:
      - 4 / 12 heads, stride 4120 (a tp4/tep4 rank, its own compile): a
        fresh 8192-token context at the last slot and at slot 0 (with
        `linear_gate`); 37 + 1 + 128 tokens carrying / fresh / carrying at
        slots 1873, 0 and 937 (without).
      - 16 / 48 heads, stride 16480 (a dep4 rank): two 512-token chunks, the
        first carrying, at slots 849 and 425; one carrying 8192-token chunk
        (max_num_tokens) at slot 849; both with `linear_gate`.
    Against the reference at these cells: q/k/v differ by one bf16 ulp on
    1.2e-5 to 1.3e-4 of elements (largest 1.6e-2, on v near 4), within
    torch's default bf16 band; g and beta inside the fp32 band; conv pool
    bitwise.

    Certified past the int32 slot offset, on the same page (larger dep4
    pools than core's: kv_fp8 binds 1217 slots, kv_nvfp4 1498; reuse and the
    tune's 0.85-fraction fp8 engine are uncounted), all with `linear_gate`:
      - 16 / 48 heads, a 2048-slot pool: 37 + 1 + 128 tokens carrying /
        fresh / carrying at slots 2047, 1942 and 0; a 4096-slot pool: one
        carrying 8192-token chunk at slot 4095 (offset past 2^32).
      - 4 / 12 heads, an 8192-slot pool: an 8192-token context at slot 8191,
        fresh and carrying.
    Same bands as above (q/k/v one-ulp flips on 2.9e-6 to 2.5e-5 of
    elements). The int32 slot offset this replaced fails the carrying ones:
    slots 2047 (16 / 48) and 8191 (4 / 12) an illegal memory access; slot
    4095 a wrapped offset inside the page, so a silently wrong q/k/v (max
    abs 0.70 on q, 5.6 on v, 3.8e-4 of elements). The fresh 8192-slot cell
    reads no carry and passes under either. Peak device memory 12.7 GiB
    (2048 and 8192 slots) and 25.6 GiB (4096), under 3 s a cell. On every
    cell below 1942 slots the 64-bit base gives the int32 form's q, k, v, g,
    beta and pool bitwise; graph replay on GB300 moved 134.2 -> 135.9 us at
    3000 + 8192 + 5000 tokens, 132.5 -> 133.8 at the 128 short contexts,
    18.4 -> 18.5 at tep4's 37 + 1 + 128 (Int64 strides on the whole pool
    layout cost 151 at the first).

    A `pad_slot_id` context reads no carry and publishes nothing; its outputs
    are computed from a zero window (the unfused conv leaves those columns
    unconvolved instead -- nothing consumes either). The engine gives every
    context row a real slot; a pad context is not certified.

    Preconditions (`is_valid`): `x` 2-D, unit channel stride, at least
    `qkv_dim` wide; `a`/`b` at least `T` rows (read through both strides);
    A_log/dt_bias contiguous with `num_v_heads` entries; `weight` `[qkv_dim,
    4]` in `x`'s dtype (any strides); `conv_states` `[slots, qkv_dim, 3]` in
    `x`'s dtype (any strides); int32 `query_start_loc`/`cache_indices`, bool
    `has_initial_state`. Loud (`__call__`): `head_k_dim != head_v_dim`, conv
    width other than 4, and what the conv pass cannot take: an `x` not bf16,
    heads other than 128 channels, a row of `x` not 16-byte aligned, a weight
    or pool stride, or an offset inside one pool slot, from 2^31 elements.

    Not checked: `cache_indices` within the pool and distinct;
    `query_start_loc` non-decreasing and ending at `T`.

    Compiled at first call, per `(num_k_heads, num_v_heads, head_k_dim, row
    stride of x)` (`_CONV_PASS`): the conv pass by CuTe DSL (TVM-FFI entry,
    `--opt-level 2`), the gates and publish by Triton. The cells ran both JIT
    paths on sm_103 (GB300).

    Not certified: fp16/fp32 activations; head geometries other than 16 / 48
    and 4 / 12; bias; a row stride of `x` other than the projections' 16480
    and 4120 (the conv pass compiles one kernel per row stride); zero-token
    contexts; `a + dt_bias` past the softplus threshold on every head; pools
    past 4096 slots at 16 / 48 heads or 8192 at 4 / 12; tp4's own pool
    (never booted).

    Hard limits
      element offsets  64-bit for the rows of x, the outputs and a pool
        slot's base (`slot * stride(0)`, in the conv pass and the publish);
        weight and pool strides and the offset inside a slot int32 (refused
        from 2^31)  not probed beyond 8192 tokens.
      pool slot  `cache_indices` is int32, so a slot index below 2^31; driven
        to offsets past 2^32 elements (slot 4095 at 16 / 48 heads, 4.5e9).
    """

    def __call__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        conv_states: torch.Tensor,
        query_start_loc: torch.Tensor,
        cache_indices: torch.Tensor,
        has_initial_state: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        num_k_heads: int,
        head_k_dim: int,
        num_v_heads: int,
        head_v_dim: int,
        pad_slot_id: int = -1,
        l2_norm_eps: float = 1e-6,
        softplus_threshold: float = 20.0,
        linear_gate: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return _launch(
            x,
            weight,
            conv_states,
            query_start_loc,
            cache_indices,
            has_initial_state,
            a,
            b,
            A_log,
            dt_bias,
            num_k_heads,
            head_k_dim,
            num_v_heads,
            head_v_dim,
            pad_slot_id,
            l2_norm_eps,
            softplus_threshold,
            linear_gate,
        )

    def reference(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        conv_states: torch.Tensor,
        query_start_loc: torch.Tensor,
        cache_indices: torch.Tensor,
        has_initial_state: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        num_k_heads: int,
        head_k_dim: int,
        num_v_heads: int,
        head_v_dim: int,
        pad_slot_id: int = -1,
        l2_norm_eps: float = 1e-6,
        softplus_threshold: float = 20.0,
        linear_gate: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        """`(q, k, v, g, beta, conv_states as the call leaves them)`; mutates
        nothing. Conv taps in fp32 in tap order and the SiLU in fp32, rounded
        to `x.dtype` once; the q/k norm in fp32 from that rounded value,
        rounded once; `g` in fp64 with the stabilised softplus, rounded to
        fp32 (with `linear_gate`, `exp` of that fp32 value in fp64, rounded to
        fp32); `beta` the fp32 sigmoid."""
        qkv_dim = 2 * num_k_heads * head_k_dim + num_v_heads * head_v_dim
        width = weight.shape[-1]
        n_tok = x.shape[0]
        w = weight.float()
        y = torch.zeros(n_tok, qkv_dim, dtype=x.dtype, device=x.device)
        states = conv_states.clone()
        bounds = query_start_loc.tolist()
        slots = cache_indices.tolist()
        carry = has_initial_state.tolist()
        for s in range(len(bounds) - 1):
            lo, hi, slot = bounds[s], bounds[s + 1], slots[s]
            tokens = x[lo:hi, :qkv_dim].float()
            if slot != pad_slot_id and carry[s]:
                prefix = conv_states[slot].float().t()
            else:
                prefix = tokens.new_zeros(width - 1, qkv_dim)
            padded = torch.cat([prefix, tokens], dim=0)
            length = hi - lo
            acc = torch.zeros_like(tokens)
            for kk in range(width):
                acc = acc + padded[kk : kk + length] * w[:, kk]
            y[lo:hi] = torch.nn.functional.silu(acc).to(x.dtype)
            if slot != pad_slot_id:
                states[slot] = padded[length : length + width - 1].t().to(states.dtype)
        kd = num_k_heads * head_k_dim

        def l2norm(part):
            part = part.reshape(n_tok, num_k_heads, head_k_dim).float()
            return (part / torch.sqrt(part.pow(2).sum(-1, keepdim=True) + l2_norm_eps)).to(x.dtype)

        q, k = l2norm(y[:, :kd]), l2norm(y[:, kd : 2 * kd])
        v = y[:, 2 * kd :].reshape(n_tok, num_v_heads, head_v_dim)
        gx = a[:n_tok, :num_v_heads].double() + dt_bias[:num_v_heads].double()
        softplus = torch.where(gx <= softplus_threshold, torch.log1p(torch.exp(gx)), gx)
        g = (-torch.exp(A_log[:num_v_heads].double()) * softplus).float()
        if linear_gate:
            g = torch.exp(g.double()).float()
        beta = torch.sigmoid(b[:n_tok, :num_v_heads].float())
        return (*(t.unsqueeze(0) for t in (q, k, v, g, beta)), states)

    def is_valid(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        conv_states: torch.Tensor,
        query_start_loc: torch.Tensor,
        cache_indices: torch.Tensor,
        has_initial_state: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        num_k_heads: int,
        head_k_dim: int,
        num_v_heads: int,
        head_v_dim: int,
        pad_slot_id: int = -1,
        l2_norm_eps: float = 1e-6,
        softplus_threshold: float = 20.0,
        linear_gate: bool = False,
    ) -> None:
        # Every address below is computed from the arguments' own extents and
        # strides only where stated; the rest is `ptr + index`, unmasked.
        qkv_dim = 2 * num_k_heads * head_k_dim + num_v_heads * head_v_dim
        n_tok = x.shape[0]
        assert x.dim() == 2 and x.stride(1) == 1 and x.shape[1] >= qkv_dim, (
            f"x must be [T, >= {qkv_dim}] with unit channel stride; got {tuple(x.shape)} "
            f"strides {x.stride()} -- the channel axis is read as `ptr + c`"
        )
        assert weight.dtype == x.dtype and weight.shape[0] == qkv_dim, (
            f"weight must be [{qkv_dim}, 4] in x's dtype; got {tuple(weight.shape)} {weight.dtype}"
        )
        assert conv_states.dtype == x.dtype and tuple(conv_states.shape[1:]) == (qkv_dim, 3), (
            f"conv_states must be [slots, {qkv_dim}, 3] in x's dtype; got "
            f"{tuple(conv_states.shape)} {conv_states.dtype}"
        )
        assert a.shape[0] >= n_tok and b.shape[0] >= n_tok, (
            f"a and b must have at least T={n_tok} rows; the token axis is unmasked"
        )
        assert A_log.stride(0) == 1 and dt_bias.stride(0) == 1, (
            "A_log and dt_bias must be contiguous 1-D; they are read as `ptr + head`"
        )
        assert A_log.numel() >= num_v_heads and dt_bias.numel() >= num_v_heads, (
            f"A_log and dt_bias must hold num_v_heads={num_v_heads} entries; the load is unmasked"
        )
        assert query_start_loc.dtype == torch.int32 and cache_indices.dtype == torch.int32, (
            "query_start_loc and cache_indices must be int32"
        )
        assert has_initial_state.dtype == torch.bool, "has_initial_state must be bool"
        n_seq = query_start_loc.shape[0] - 1
        assert cache_indices.shape[0] >= n_seq and has_initial_state.shape[0] >= n_seq, (
            f"cache_indices and has_initial_state must cover the {n_seq} sequences"
        )


gdn_prefill_conv_post = GdnPrefillConvPost()
