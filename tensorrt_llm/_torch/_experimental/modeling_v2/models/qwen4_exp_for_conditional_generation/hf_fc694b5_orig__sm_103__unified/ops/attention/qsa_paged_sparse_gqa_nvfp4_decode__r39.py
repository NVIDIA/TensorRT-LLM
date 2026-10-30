# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""QSA sparse GQA over an NVFP4 paged K/V pool, decode rows: CuTe DSL split-K + merge.

This target's local variant of the tp1 target's `ops/attention/qsa_paged_sparse_gqa_nvfp4_decode`,
the kernels unchanged, carrying dep4's `kv_nvfp4` decode cells until the promotion merges the
two into the catalog.

CuTe DSL written for the tp1 target, both launches. One CTA (eight warps) serves one
(row, KV head, split) and walks the split's tiles of 64 selected tokens with an online
softmax; the split count is chosen by row count so the grid stays about two waves deep
(33 splits of one tile at 1 row, one split of 33 tiles from 128 rows up).

Per tile:

1. raw NVFP4 bytes land in shared memory with cp.async, one tile ahead (two buffers):
   each token's 128 K bytes, its 16 K scale bytes (token-major), its 128 V bytes and the
   64-byte four-token V-scale block it sits in; a token that does not resolve (padding,
   foreign request, unmapped page) gets zeros instead;
2. the V tile is dequantized into an f16 tile (`cvt.rn.f16x2.e2m1x2` x `cvt.rn.f16x2.e4m3x2`,
   exact) for `ldmatrix.trans`;
3. S = q K^T on `mma.sync.m16n8k16` (f16 in, f32 out), a warp eight tokens over the whole
   head: K's B fragments are decoded straight from the raw bytes in registers. The head
   dimension is permuted identically in q and K (lane quad c of k-step pair m reads the
   bytes 16m + 4c .. 16m + 4c + 3, dims 32m + 8c .. 32m + 8c + 7, inside one scale block),
   which leaves the dot product unchanged;
4. online softmax in the log2 domain across the warps (two shared-memory exchanges), P
   rounded to f16 as the Triton kernel does;
5. O += P V on `mma.sync` (a warp 32 head dims), P from shared memory with `ldmatrix`,
   V with `ldmatrix.trans`.

The second-level scales act on the scores (K) and the normalized output (V), as in the
Triton op. With one split the CTA stores the output itself (optionally times the
sigmoid of the output gate, applied to the bf16-rounded attention value); otherwise it
stores an fp32 partial and its log2-sum-exp for the merge kernel.
"""

from __future__ import annotations

import math
from typing import Optional

import cutlass
import cutlass.cute as cute
import torch
from cutlass import const_expr
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import T, dsl_user_op

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl
from tensorrt_llm._torch.kernels.blackwell.utils import make_ptr

from .qsa_paged_sparse_gqa_nvfp4__r38 import qsa_paged_sparse_gqa_nvfp4 as _composition

TILE = 64
WARPS = 8
THREADS = WARPS * 32
HEAD_DIM = 256
#: Bytes of a raw K row in shared memory (128 + 16 of padding: conflict-free fragment reads).
RAW_LD = 144
#: Bytes of a raw V row (unpadded: the dequant pass reads 32 bytes a thread).
RAW_LDV = 128
#: Raw tiles in flight (cp.async ring).
STAGES = 3
#: Bytes of an f16 V row in shared memory (256 + 8 elements: conflict-free ldmatrix).
SV_LD = 528
#: Bytes of an f16 P row (64 + 8 elements).
SP_LD = 144
LOG2E = 1.4426950408889634
#: Selected tokens a merge CTA covers per head (four adjacent ones a lane of each warp).
CHUNK = 128
MAX_SPLITS = 64


# ---------------------------------------------------------------------------------------
# inline PTX helpers
# ---------------------------------------------------------------------------------------


def _ir(x, cls, loc, ip):
    return cls(x).ir_value(loc=loc, ip=ip)


def _asm(ret, args, text, cons, loc, ip, side=False):
    return llvm.inline_asm(
        ret,
        args,
        text,
        cons,
        has_side_effects=side,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def _unpack(res, n, ty, loc, ip):
    return [
        ty(llvm.extractvalue(T.i32() if ty is cutlass.Int32 else T.f32(), res, [i], loc=loc, ip=ip))
        for i in range(n)
    ]


@dsl_user_op
def _cp16(saddr, gaddr, *, loc=None, ip=None):
    _asm(
        T.i32(),
        [_ir(saddr, cutlass.Int32, loc, ip), _ir(gaddr, cutlass.Int64, loc, ip)],
        "{ cp.async.cg.shared.global [$1], [$2], 16; mov.u32 $0, 0; }",
        "=r,r,l",
        loc,
        ip,
        side=True,
    )


@dsl_user_op
def _cp16z(saddr, gaddr, nbytes, *, loc=None, ip=None):
    """cp.async of 16 bytes, zero-filled past `nbytes` (0 or 16): no branch for a dead token."""
    _asm(
        T.i32(),
        [
            _ir(saddr, cutlass.Int32, loc, ip),
            _ir(gaddr, cutlass.Int64, loc, ip),
            _ir(nbytes, cutlass.Int32, loc, ip),
        ],
        "{ cp.async.cg.shared.global [$1], [$2], 16, $3; mov.u32 $0, 0; }",
        "=r,r,l,r",
        loc,
        ip,
        side=True,
    )


@dsl_user_op
def _sts4(saddr, a, b, c, d, *, loc=None, ip=None):
    _asm(
        T.i32(),
        [_ir(saddr, cutlass.Int32, loc, ip)]
        + [_ir(v, cutlass.Int32, loc, ip) for v in (a, b, c, d)],
        "{ st.shared.v4.b32 [$1], {$2, $3, $4, $5}; mov.u32 $0, 0; }",
        "=r,r,r,r,r,r",
        loc,
        ip,
        side=True,
    )


@dsl_user_op
def _sts1(saddr, a, *, loc=None, ip=None):
    _asm(
        T.i32(),
        [_ir(saddr, cutlass.Int32, loc, ip), _ir(a, cutlass.Int32, loc, ip)],
        "{ st.shared.b32 [$1], $2; mov.u32 $0, 0; }",
        "=r,r,r",
        loc,
        ip,
        side=True,
    )


@dsl_user_op
def _lds1(saddr, *, loc=None, ip=None):
    return cutlass.Int32(
        _asm(
            T.i32(),
            [_ir(saddr, cutlass.Int32, loc, ip)],
            "ld.shared.b32 $0, [$1];",
            "=r,r",
            loc,
            ip,
            side=True,
        )
    )


@dsl_user_op
def _lds4(saddr, *, loc=None, ip=None):
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [_ir(saddr, cutlass.Int32, loc, ip)],
        "ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        loc,
        ip,
        side=True,
    )
    return _unpack(res, 4, cutlass.Int32, loc, ip)


@dsl_user_op
def _ldg4(gaddr, *, loc=None, ip=None):
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [_ir(gaddr, cutlass.Int64, loc, ip)],
        "ld.global.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        loc,
        ip,
        side=True,
    )
    return _unpack(res, 4, cutlass.Int32, loc, ip)


@dsl_user_op
def _ldsm4(saddr, *, loc=None, ip=None):
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [_ir(saddr, cutlass.Int32, loc, ip)],
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        loc,
        ip,
        side=True,
    )
    return _unpack(res, 4, cutlass.Int32, loc, ip)


@dsl_user_op
def _ldsm4t(saddr, *, loc=None, ip=None):
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [_ir(saddr, cutlass.Int32, loc, ip)],
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        loc,
        ip,
        side=True,
    )
    return _unpack(res, 4, cutlass.Int32, loc, ip)


@dsl_user_op
def _mma(a, b, c, *, loc=None, ip=None):
    """D = A B + C on m16n8k16 (f16 A/B, f32 C/D); a: 4 regs, b: 2 regs, c: 4 f32."""
    res = _asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        [_ir(v, cutlass.Int32, loc, ip) for v in a]
        + [_ir(v, cutlass.Int32, loc, ip) for v in b]
        + [_ir(v, cutlass.Float32, loc, ip) for v in c],
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, {$10, $11, $12, $13};",
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f",
        loc,
        ip,
    )
    return _unpack(res, 4, cutlass.Float32, loc, ip)


@dsl_user_op
def _dq_word(word, scale2, *, loc=None, ip=None):
    """Four e2m1 bytes (8 values, low nibble the even one) times an f16x2 scale -> 4 f16x2."""
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [_ir(word, cutlass.Int32, loc, ip), _ir(scale2, cutlass.Int32, loc, ip)],
        "{ .reg .b8 b0, b1, b2, b3; .reg .b32 h0, h1, h2, h3; "
        "mov.b32 {b0, b1, b2, b3}, $4; "
        "cvt.rn.f16x2.e2m1x2 h0, b0; cvt.rn.f16x2.e2m1x2 h1, b1; "
        "cvt.rn.f16x2.e2m1x2 h2, b2; cvt.rn.f16x2.e2m1x2 h3, b3; "
        "mul.rn.f16x2 $0, h0, $5; mul.rn.f16x2 $1, h1, $5; "
        "mul.rn.f16x2 $2, h2, $5; mul.rn.f16x2 $3, h3, $5; }",
        "=r,=r,=r,=r,r,r",
        loc,
        ip,
    )
    return _unpack(res, 4, cutlass.Int32, loc, ip)


@dsl_user_op
def _e2m1_half(half, *, loc=None, ip=None):
    """Two e2m1 bytes (the low 16 bits: 4 values, low nibble the even one) -> 2 f16x2
    (values 0-1, values 2-3), unscaled."""
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 2),
        [_ir(half, cutlass.Int32, loc, ip)],
        "{ .reg .b8 b0, b1, b2, b3; mov.b32 {b0, b1, b2, b3}, $2; "
        "cvt.rn.f16x2.e2m1x2 $0, b0; cvt.rn.f16x2.e2m1x2 $1, b1; }",
        "=r,=r,r",
        loc,
        ip,
    )
    return _unpack(res, 2, cutlass.Int32, loc, ip)


@dsl_user_op
def _e4m3x2(half, *, loc=None, ip=None):
    """Two e4m3 bytes (the low 16 bits) -> f16x2, the low byte in the low half."""
    return cutlass.Int32(
        _asm(
            T.i32(),
            [_ir(half, cutlass.Int32, loc, ip)],
            "{ .reg .b16 h; cvt.u16.u32 h, $1; cvt.rn.f16x2.e4m3x2 $0, h; }",
            "=r,r",
            loc,
            ip,
        )
    )


@dsl_user_op
def _hmul2(a, b, *, loc=None, ip=None):
    return cutlass.Int32(
        _asm(
            T.i32(),
            [_ir(a, cutlass.Int32, loc, ip), _ir(b, cutlass.Int32, loc, ip)],
            "mul.rn.f16x2 $0, $1, $2;",
            "=r,r,r",
            loc,
            ip,
        )
    )


@dsl_user_op
def _prmt(a, b, sel, *, loc=None, ip=None):
    return cutlass.Int32(
        _asm(
            T.i32(),
            [
                _ir(a, cutlass.Int32, loc, ip),
                _ir(b, cutlass.Int32, loc, ip),
                _ir(sel, cutlass.Int32, loc, ip),
            ],  # fmt: skip
            "prmt.b32 $0, $1, $2, $3;",
            "=r,r,r,r",
            loc,
            ip,
        )
    )


@dsl_user_op
def _lds_u16(saddr, *, loc=None, ip=None):
    return cutlass.Int32(
        _asm(
            T.i32(),
            [_ir(saddr, cutlass.Int32, loc, ip)],
            "ld.shared.u16 $0, [$1];",
            "=r,r",
            loc,
            ip,
            side=True,
        )
    )


@dsl_user_op
def _stg4(gaddr, a, b, c, d, *, loc=None, ip=None):
    _asm(
        T.i32(),
        [_ir(gaddr, cutlass.Int64, loc, ip)]
        + [_ir(v, cutlass.Int32, loc, ip) for v in (a, b, c, d)],
        "{ st.global.v4.b32 [$1], {$2, $3, $4, $5}; mov.u32 $0, 0; }",
        "=r,l,r,r,r,r",
        loc,
        ip,
        side=True,
    )


@dsl_user_op
def _scale2(sbyte, *, loc=None, ip=None):
    """An e4m3 byte (in the low 8 bits) -> f16x2 {s, s}."""
    return cutlass.Int32(
        _asm(
            T.i32(),
            [_ir(sbyte, cutlass.Int32, loc, ip)],
            "{ .reg .b32 t; .reg .b16 h; and.b32 t, $1, 255; mul.lo.u32 t, t, 257; "
            "cvt.u16.u32 h, t; cvt.rn.f16x2.e4m3x2 $0, h; }",
            "=r,r",
            loc,
            ip,
        )
    )


@dsl_user_op
def _scale2s(word, sel, *, loc=None, ip=None):
    """The e4m3 byte `word.byte[sel & 3]` (sel = k * 0x11) -> f16x2 {s, s}: one prmt + one cvt."""
    return cutlass.Int32(
        _asm(
            T.i32(),
            [_ir(word, cutlass.Int32, loc, ip), _ir(sel, cutlass.Int32, loc, ip)],
            "{ .reg .b32 t; .reg .b16 h; prmt.b32 t, $1, 0, $2; cvt.u16.u32 h, t; "
            "cvt.rn.f16x2.e4m3x2 $0, h; }",
            "=r,r,r",
            loc,
            ip,
        )
    )


@dsl_user_op
def _ex2(x, *, loc=None, ip=None):
    return cutlass.Float32(
        _asm(
            T.f32(),
            [_ir(x, cutlass.Float32, loc, ip)],
            "ex2.approx.ftz.f32 $0, $1;",
            "=f,f",
            loc,
            ip,
        )
    )


@dsl_user_op
def _bf2_to_f2(w, *, loc=None, ip=None):
    """bf16x2 -> f16x2 (through f32, round to nearest)."""
    return cutlass.Int32(
        _asm(
            T.i32(),
            [_ir(w, cutlass.Int32, loc, ip)],
            "{ .reg .b16 lo, hi; .reg .f32 fl, fh; mov.b32 {lo, hi}, $1; "
            "cvt.f32.bf16 fl, lo; cvt.f32.bf16 fh, hi; cvt.rn.f16x2.f32 $0, fh, fl; }",
            "=r,r",
            loc,
            ip,
        )
    )


@dsl_user_op
def _f2_pack(lo, hi, *, loc=None, ip=None):
    """Two f32 -> f16x2 (lo in the low half)."""
    return cutlass.Int32(
        _asm(
            T.i32(),
            [_ir(lo, cutlass.Float32, loc, ip), _ir(hi, cutlass.Float32, loc, ip)],
            "cvt.rn.f16x2.f32 $0, $2, $1;",
            "=r,f,f",
            loc,
            ip,
        )
    )


@dsl_user_op
def _bf2_pack(lo, hi, *, loc=None, ip=None):
    """Two f32 -> bf16x2 (lo in the low half)."""
    return cutlass.Int32(
        _asm(
            T.i32(),
            [_ir(lo, cutlass.Float32, loc, ip), _ir(hi, cutlass.Float32, loc, ip)],
            "cvt.rn.bf16x2.f32 $0, $2, $1;",
            "=r,f,f",
            loc,
            ip,
        )
    )


@dsl_user_op
def _stg1(gaddr, a, *, loc=None, ip=None):
    _asm(
        T.i32(),
        [_ir(gaddr, cutlass.Int64, loc, ip), _ir(a, cutlass.Int32, loc, ip)],
        "{ st.global.b32 [$1], $2; mov.u32 $0, 0; }",
        "=r,l,r",
        loc,
        ip,
        side=True,
    )


@dsl_user_op
def _stg2f(gaddr, a, b, *, loc=None, ip=None):
    _asm(
        T.i32(),
        [
            _ir(gaddr, cutlass.Int64, loc, ip),
            _ir(a, cutlass.Float32, loc, ip),
            _ir(b, cutlass.Float32, loc, ip),
        ],
        "{ st.global.v2.f32 [$1], {$2, $3}; mov.u32 $0, 0; }",
        "=r,l,f,f",
        loc,
        ip,
        side=True,
    )


@dsl_user_op
def _ldg1(gaddr, *, loc=None, ip=None):
    return cutlass.Int32(
        _asm(
            T.i32(),
            [_ir(gaddr, cutlass.Int64, loc, ip)],
            "ld.global.b32 $0, [$1];",
            "=r,l",
            loc,
            ip,
            side=True,
        )
    )


@dsl_user_op
def _bf2_lo(w, *, loc=None, ip=None):
    return cutlass.Float32(
        _asm(
            T.f32(),
            [_ir(w, cutlass.Int32, loc, ip)],
            "{ .reg .b16 lo, hi; mov.b32 {lo, hi}, $1; cvt.f32.bf16 $0, lo; }",
            "=f,r",
            loc,
            ip,
        )
    )


@dsl_user_op
def _bf2_hi(w, *, loc=None, ip=None):
    return cutlass.Float32(
        _asm(
            T.f32(),
            [_ir(w, cutlass.Int32, loc, ip)],
            "{ .reg .b16 lo, hi; mov.b32 {lo, hi}, $1; cvt.f32.bf16 $0, hi; }",
            "=f,r",
            loc,
            ip,
        )
    )


# ---------------------------------------------------------------------------------------
# split kernel
# ---------------------------------------------------------------------------------------


class _SplitKernel:
    def __init__(
        self,
        *,
        group,
        kv_heads,
        width,
        tpb,
        splits,
        tps,
        scale,
        num_pages,
        bt_width,
        num_requests,
        k_strides,
        v_strides,
        ks_strides,
        vs_strides,
        q_strides,
        req64,
        gated,
        pdl,
    ):
        assert group <= 16 and HEAD_DIM == 256 and tpb % 4 == 0
        self.group, self.kvh, self.width, self.tpb = group, kv_heads, width, tpb
        self.splits, self.tps = splits, tps
        self.n_tiles = -(-width // TILE)
        self.scale = scale * LOG2E
        self.num_pages, self.bt_width, self.num_requests = num_pages, bt_width, num_requests
        self.ks, self.vs, self.kss, self.vss = k_strides, v_strides, ks_strides, vs_strides
        self.qs = q_strides
        self.req64, self.gated, self.pdl = req64, gated, pdl
        self.heads = group * kv_heads
        # shared memory carve-out (bytes)
        self.o_tok = 0
        self.tok_bytes = -(-tps * TILE * 4 // 128) * 128
        self.raw_ldv = RAW_LDV
        self.raw_vsld = 64
        self.raw_k = TILE * RAW_LD
        self.raw_v = TILE * self.raw_ldv
        self.raw_ks = TILE * 16
        self.raw_vs = TILE * self.raw_vsld
        self.raw_buf = self.raw_k + self.raw_v + self.raw_ks + self.raw_vs
        self.stages = min(STAGES, tps)
        self.o_raw = self.o_tok + self.tok_bytes
        self.o_sv = self.o_raw + self.stages * self.raw_buf
        self.o_sp = self.o_sv + TILE * SV_LD
        self.o_red = self.o_sp + 16 * SP_LD
        self.smem = self.o_red + 2 * WARPS * 16 * 4 + 128
        assert k_strides == v_strides and ks_strides == vs_strides, (
            "K and V planes share one layout"
        )
        assert tpb & (tpb - 1) == 0

    @cute.jit
    def __call__(
        self,
        sel_t: cute.Tensor,
        q_ptr: cute.Pointer,
        k_ptr: cute.Pointer,
        v_ptr: cute.Pointer,
        ks_ptr: cute.Pointer,
        vs_ptr: cute.Pointer,
        gs_ptr: cute.Pointer,
        bt_ptr: cute.Pointer,
        req_ptr: cute.Pointer,
        out_ptr: cute.Pointer,
        lse_ptr: cute.Pointer,
        gate_ptr: cute.Pointer,
        rows: cutlass.Int32,
        gate_stride: cutlass.Int32,
        stream,
    ):
        sel = cute.make_tensor(
            sel_t.iterator, cute.make_layout((rows, self.width), stride=(self.width, 1))
        )
        bt = cute.make_tensor(
            bt_ptr, cute.make_layout((self.num_requests, self.bt_width), stride=(self.bt_width, 1))
        )
        req = cute.make_tensor(req_ptr, cute.make_layout((rows,), stride=(1,)))
        gs = cute.make_tensor(gs_ptr, cute.make_layout((3,), stride=(1,)))
        self.kernel(
            sel,
            q_ptr.toint(),
            k_ptr.toint(),
            v_ptr.toint(),
            ks_ptr.toint(),
            vs_ptr.toint(),
            gs,
            bt,
            req,
            out_ptr.toint(),
            lse_ptr.toint(),
            gate_ptr.toint(),
            rows,
            gate_stride,
        ).launch(
            grid=[rows * self.kvh * self.splits, 1, 1],
            block=[THREADS, 1, 1],
            smem=self.smem,
            stream=stream,
            use_pdl=self.pdl,
            min_blocks_per_mp=2,
        )

    @cute.jit
    def _issue(self, j, buf, tidx, kvh, s_tok, s_raw, k_base, v_base, ks_base, vs_base):
        """cp.async of tile `j`'s raw K, K-scale, V and V-scale-block rows into ring slot `buf`,
        committed as one group: thread t serves token t // 4 (a quarter of its rows); a token
        that does not resolve gets zeros (src-size 0)."""
        tpb: cutlass.Constexpr = self.tpb
        lt: cutlass.Constexpr = tpb.bit_length() - 1
        ks0, ks1, ks2 = self.ks
        kss0, kss1 = self.kss
        rb = s_raw + buf * self.raw_buf
        rk = rb
        rv = rb + self.raw_k
        rks = rv + self.raw_v
        rvs = rks + self.raw_ks
        tt = tidx // 4
        q4 = tidx % 4
        ent = _lds1(s_tok + (j * TILE + tt) * 4)
        ok = ent >= 0
        e = cutlass.max(ent, 0)
        pg = cutlass.Int64(e >> lt)
        sl = e & (tpb - 1)
        n = cutlass.Int32(cutlass.select_(ok, cutlass.Int32(16), cutlass.Int32(0)))
        off = pg * ks0 + cutlass.Int64(kvh * ks1 + sl * ks2 + 32 * q4)
        offs = pg * kss0 + cutlass.Int64(kvh * kss1)
        _cp16z(rk + tt * RAW_LD + 32 * q4, k_base + off, n)
        _cp16z(rk + tt * RAW_LD + 32 * q4 + 16, k_base + off + 16, n)
        _cp16z(rv + tt * self.raw_ldv + 32 * q4, v_base + off, n)
        _cp16z(rv + tt * self.raw_ldv + 32 * q4 + 16, v_base + off + 16, n)
        _cp16z(
            rvs + tt * self.raw_vsld + 16 * q4,
            vs_base + offs + cutlass.Int64((sl >> 2) * 64 + 16 * q4),
            n,
        )
        if q4 == 0:
            _cp16z(rks + tt * 16, ks_base + offs + cutlass.Int64(sl * 16), n)
        cute.arch.cp_async_commit_group()

    @cute.kernel
    def kernel(
        self,
        mSel: cute.Tensor,
        q_base: cutlass.Int64,
        k_base: cutlass.Int64,
        v_base: cutlass.Int64,
        ks_base: cutlass.Int64,
        vs_base: cutlass.Int64,
        mGs: cute.Tensor,
        mBT: cute.Tensor,
        mReq: cute.Tensor,
        out_base: cutlass.Int64,
        lse_base: cutlass.Int64,
        gate_base: cutlass.Int64,
        rows: cutlass.Int32,
        gate_stride: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        warp = cute.arch.warp_idx()
        lane = cute.arch.lane_idx()
        g = lane // 4
        c = lane % 4
        group: cutlass.Constexpr = self.group
        kvh_n: cutlass.Constexpr = self.kvh
        splits: cutlass.Constexpr = self.splits
        tps: cutlass.Constexpr = self.tps
        n_tiles_all: cutlass.Constexpr = self.n_tiles
        width: cutlass.Constexpr = self.width
        tpb: cutlass.Constexpr = self.tpb
        heads: cutlass.Constexpr = self.heads
        row = bidx // (kvh_n * splits)
        kvh = (bidx // splits) % kvh_n
        split = bidx % splits
        t0 = split * tps
        n_t = cutlass.min(tps, n_tiles_all - t0)

        smem = cutlass.utils.SmemAllocator()
        base_t = smem.allocate_tensor(
            cutlass.Int8, cute.make_layout((self.smem - 128,), stride=(1,)), byte_alignment=128
        )
        sbase = cutlass.Int32(base_t.iterator.toint())
        s_tok = sbase + self.o_tok
        s_raw = sbase + self.o_raw
        s_v = sbase + self.o_sv
        s_p = sbase + self.o_sp
        s_red = sbase + self.o_red

        if const_expr(self.pdl):
            cute.arch.griddepcontrol_launch_dependents()
            cute.arch.griddepcontrol_wait()

        # ---- the token table of this split: page * tpb + slot, or -1 ---------------------
        rq = cutlass.Int64(mReq[row])
        req_ok = (rq >= 0) & (rq < self.num_requests)
        safe_req = cutlass.Int32(
            cutlass.min(cutlass.max(rq, cutlass.Int64(0)), cutlass.Int64(self.num_requests - 1))
        )
        for i in cutlass.range_constexpr(-(-(tps * TILE) // THREADS)):
            cl = tidx + i * THREADS
            col = t0 * TILE + cl
            colc = cutlass.min(col, width - 1)
            tok = mSel[row, colc]
            ok = req_ok & (col < width) & (tok >= 0)
            safe_tok = cutlass.max(tok, 0)
            lp = safe_tok // tpb
            ok = ok & (lp < self.bt_width)
            page = mBT[safe_req, cutlass.min(lp, self.bt_width - 1)]
            ok = ok & (page >= 0) & (page < self.num_pages)
            ent = cutlass.Int32(cutlass.select_(ok, page * tpb + safe_tok % tpb, cutlass.Int32(-1)))
            if cl < tps * TILE:
                _sts1(s_tok + cl * 4, ent)
        cute.arch.sync_threads()

        # ---- q of this KV head: rows g and g + 8 of the m16 tile, as permuted A fragments --
        # thread (g, c) holds dims 32m + 8c .. 32m + 8c + 7 of heads g and g + 8, m = 0..7
        qa = cute.make_rmem_tensor(cute.make_layout((8, 4), stride=(4, 1)), cutlass.Int32)
        qb = cute.make_rmem_tensor(cute.make_layout((8, 4), stride=(4, 1)), cutlass.Int32)
        qs0, qs1 = self.qs
        q_row = q_base + (cutlass.Int64(row) * qs0 + cutlass.Int64(kvh * group) * qs1) * 2
        for m in cutlass.range_constexpr(8):
            off = cutlass.Int64((32 * m + 8 * c) * 2)
            w4 = _ldg4(q_row + cutlass.Int64(g) * (qs1 * 2) + off)
            for e in cutlass.range_constexpr(4):
                qa[m, e] = _bf2_to_f2(w4[e])
            hb = g + 8
            hbc = cutlass.min(hb, group - 1)
            w4b = _ldg4(q_row + cutlass.Int64(hbc) * (qs1 * 2) + off)
            for e in cutlass.range_constexpr(4):
                qb[m, e] = cutlass.Int32(
                    cutlass.select_(hb < group, _bf2_to_f2(w4b[e]), cutlass.Int32(0))
                )

        stages: cutlass.Constexpr = self.stages
        self._issue(0, 0, tidx, kvh, s_tok, s_raw, k_base, v_base, ks_base, vs_base)
        if const_expr(stages > 2):
            if n_t > 1:
                self._issue(1, 1, tidx, kvh, s_tok, s_raw, k_base, v_base, ks_base, vs_base)

        neg_inf = cutlass.Float32(-math.inf)
        st = cute.make_rmem_tensor(cute.make_layout((4,), stride=(1,)), cutlass.Float32)
        st[0] = neg_inf  # running max, head g
        st[1] = neg_inf  # running max, head g + 8
        st[2] = cutlass.Float32(0.0)  # this thread's share of the running sum, head g
        st[3] = cutlass.Float32(0.0)  # ... head g + 8 (reduced over the CTA at the end)
        acc = cute.make_rmem_tensor(cute.make_layout((4, 4), stride=(4, 1)), cutlass.Float32)
        acc.fill(0.0)
        score_scale = cutlass.Float32(self.scale) * mGs[1]

        # loop-invariant per-thread offsets
        vt = tidx // 4
        vq = tidx % 4
        v_src = vt * RAW_LDV + 32 * vq
        vs_src = vt * 64 + 16 * vq
        v_dst = s_v + vt * SV_LD + 128 * vq
        sel_v = (vt % 4) * 0x11
        ktok = warp * 8 + g
        k_src = ktok * RAW_LD + 4 * c
        ks_src = ktok * 16
        sel_k0 = (c // 2) * 0x11
        sel_k1 = (2 + c // 2) * 0x11
        ta = warp * 8 + 2 * c
        p_st_g = s_p + g * SP_LD + ta * 2
        p_st_h = p_st_g + 8 * SP_LD
        p_ld = s_p + ((lane % 8) + 8 * ((lane // 8) % 2)) * SP_LD + 16 * (lane // 16)
        v_ld_row = s_v + ((lane % 8) + 8 * ((lane // 8) % 2)) * SV_LD
        ci0 = 4 * warp + lane // 16
        ci1 = ci0 + 2
        v_ld0 = v_ld_row + 16 * ((ci0 & ~7) | ((ci0 & 7) ^ (2 * ((ci0 >> 3) & 3))))
        v_ld1 = v_ld_row + 16 * ((ci1 & ~7) | ((ci1 & 7) ^ (2 * ((ci1 >> 3) & 3))))
        red_g = s_red + g * WARPS * 4
        red_h = s_red + (g + 8) * WARPS * 4

        for j in cutlass.range(n_t):
            buf = j % stages
            if const_expr(stages == 1):
                cute.arch.cp_async_wait_group(0)
            else:
                ahead: cutlass.Constexpr = stages - 1
                if j + ahead < n_t:
                    self._issue(
                        j + ahead,
                        (j + ahead) % stages,
                        tidx,
                        kvh,
                        s_tok,
                        s_raw,
                        k_base,
                        v_base,
                        ks_base,
                        vs_base,
                    )
                    cute.arch.cp_async_wait_group(ahead)
                else:
                    if const_expr(stages > 2):
                        if j + 1 < n_t:
                            cute.arch.cp_async_wait_group(1)
                        else:
                            cute.arch.cp_async_wait_group(0)
                    else:
                        cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()  # [A] tile j landed; previous tile's PV done
            rb = s_raw + buf * self.raw_buf
            rk = rb
            rv = rb + self.raw_k
            rks = rv + self.raw_v
            rvs = rks + self.raw_ks

            # -- V tile -> f16 (thread: token tidx // 4, quarter tidx % 4 = dims 64q .. 64q+63)
            vw0 = _lds4(rv + v_src)
            vw1 = _lds4(rv + v_src + 16)
            vsw = _lds4(rvs + vs_src)
            for gg in cutlass.range_constexpr(4):
                sc = _scale2s(vsw[gg], sel_v)
                wa = vw0[2 * gg] if gg < 2 else vw1[2 * gg - 4]
                wb = vw0[2 * gg + 1] if gg < 2 else vw1[2 * gg - 3]
                da = _dq_word(wa, sc)
                db = _dq_word(wb, sc)
                # logical 16-byte chunk 8 vq + k of the row sits at 8 vq + (k ^ 2 vq): the four
                # threads of a token store to different banks; ldmatrix applies the same map
                _sts4(v_dst + 16 * ((2 * gg) ^ (2 * vq)), da[0], da[1], da[2], da[3])
                _sts4(v_dst + 16 * ((2 * gg + 1) ^ (2 * vq)), db[0], db[1], db[2], db[3])

            # -- S = q K^T for this warp's eight tokens (two accumulator chains)
            ksw = _lds4(rks + ks_src)
            kws = [_lds1(rk + k_src + 16 * m) for m in range(8)]
            zero = cutlass.Float32(0.0)
            sa = [zero, zero, zero, zero]
            sb = [zero, zero, zero, zero]
            for m in cutlass.range_constexpr(8):
                # scale block 2m + c // 2: word m // 2, byte 2 (m % 2) + c // 2
                sc = _scale2s(ksw[m // 2], sel_k0 if m % 2 == 0 else sel_k1)
                b = _dq_word(kws[m], sc)
                if const_expr(m % 2 == 0):
                    sa = _mma((qa[m, 0], qb[m, 0], qa[m, 1], qb[m, 1]), (b[0], b[1]), sa)
                    sa = _mma((qa[m, 2], qb[m, 2], qa[m, 3], qb[m, 3]), (b[2], b[3]), sa)
                else:
                    sb = _mma((qa[m, 0], qb[m, 0], qa[m, 1], qb[m, 1]), (b[0], b[1]), sb)
                    sb = _mma((qa[m, 2], qb[m, 2], qa[m, 3], qb[m, 3]), (b[2], b[3]), sb)

            # mask + scale; tokens 8w + 2c, 8w + 2c + 1
            ea = _lds1(s_tok + (j * TILE + ta) * 4)
            eb = _lds1(s_tok + (j * TILE + ta + 1) * 4)
            s0 = cutlass.Float32(cutlass.select_(ea >= 0, (sa[0] + sb[0]) * score_scale, neg_inf))
            s1 = cutlass.Float32(cutlass.select_(eb >= 0, (sa[1] + sb[1]) * score_scale, neg_inf))
            s2 = cutlass.Float32(cutlass.select_(ea >= 0, (sa[2] + sb[2]) * score_scale, neg_inf))
            s3 = cutlass.Float32(cutlass.select_(eb >= 0, (sa[3] + sb[3]) * score_scale, neg_inf))
            mg = cute.arch.fmax(s0, s1)
            mh = cute.arch.fmax(s2, s3)
            mg = cute.arch.fmax(mg, cute.arch.shuffle_sync_bfly(mg, offset=1))
            mh = cute.arch.fmax(mh, cute.arch.shuffle_sync_bfly(mh, offset=1))
            mg = cute.arch.fmax(mg, cute.arch.shuffle_sync_bfly(mg, offset=2))
            mh = cute.arch.fmax(mh, cute.arch.shuffle_sync_bfly(mh, offset=2))
            if c == 0:
                _sts1(red_g + warp * 4, _f32_bits(mg))
                _sts1(red_h + warp * 4, _f32_bits(mh))
            cute.arch.sync_threads()  # [B] maxes; V tile in shared memory
            rg0 = _lds4(red_g)
            rg1 = _lds4(red_g + 16)
            rh0 = _lds4(red_h)
            rh1 = _lds4(red_h + 16)
            tmg = st[0]
            tmh = st[1]
            for e in cutlass.range_constexpr(4):
                tmg = cute.arch.fmax(tmg, cute.arch.fmax(_bits_f32(rg0[e]), _bits_f32(rg1[e])))
                tmh = cute.arch.fmax(tmh, cute.arch.fmax(_bits_f32(rh0[e]), _bits_f32(rh1[e])))
            safe_g = cutlass.Float32(cutlass.select_(tmg == neg_inf, cutlass.Float32(0.0), tmg))
            safe_h = cutlass.Float32(cutlass.select_(tmh == neg_inf, cutlass.Float32(0.0), tmh))
            corr_g = _ex2(st[0] - safe_g)
            corr_h = _ex2(st[1] - safe_h)
            p0 = _ex2(s0 - safe_g)
            p1 = _ex2(s1 - safe_g)
            p2 = _ex2(s2 - safe_h)
            p3 = _ex2(s3 - safe_h)
            _sts1(p_st_g, _f2_pack(p0, p1))
            _sts1(p_st_h, _f2_pack(p2, p3))
            st[0] = tmg
            st[1] = tmh
            st[2] = st[2] * corr_g + (p0 + p1)
            st[3] = st[3] * corr_h + (p2 + p3)
            cute.arch.sync_threads()  # [C] P

            # -- O += P V: this warp's dims 32w .. 32w + 31 (n-tiles of 8)
            if cute.arch.vote_any_sync((corr_g != 1.0) | (corr_h != 1.0)):
                for nt in cutlass.range_constexpr(4):
                    acc[nt, 0] = acc[nt, 0] * corr_g
                    acc[nt, 1] = acc[nt, 1] * corr_g
                    acc[nt, 2] = acc[nt, 2] * corr_h
                    acc[nt, 3] = acc[nt, 3] * corr_h
            for kk in cutlass.range_constexpr(4):
                a = _ldsm4(p_ld + 32 * kk)
                b0 = _ldsm4t(v_ld0 + 16 * kk * SV_LD)
                b1 = _ldsm4t(v_ld1 + 16 * kk * SV_LD)
                r0 = _mma(a, (b0[0], b0[1]), (acc[0, 0], acc[0, 1], acc[0, 2], acc[0, 3]))
                r1 = _mma(a, (b0[2], b0[3]), (acc[1, 0], acc[1, 1], acc[1, 2], acc[1, 3]))
                r2 = _mma(a, (b1[0], b1[1]), (acc[2, 0], acc[2, 1], acc[2, 2], acc[2, 3]))
                r3 = _mma(a, (b1[2], b1[3]), (acc[3, 0], acc[3, 1], acc[3, 2], acc[3, 3]))
                for e in cutlass.range_constexpr(4):
                    acc[0, e] = r0[e]
                    acc[1, e] = r1[e]
                    acc[2, e] = r2[e]
                    acc[3, e] = r3[e]

        # ---- the running sums over the quad and the warps -----------------------------------
        lsg = st[2]
        lsh = st[3]
        lsg = lsg + cute.arch.shuffle_sync_bfly(lsg, offset=1)
        lsh = lsh + cute.arch.shuffle_sync_bfly(lsh, offset=1)
        lsg = lsg + cute.arch.shuffle_sync_bfly(lsg, offset=2)
        lsh = lsh + cute.arch.shuffle_sync_bfly(lsh, offset=2)
        cute.arch.sync_threads()
        if c == 0:
            _sts1(red_g + warp * 4, _f32_bits(lsg))
            _sts1(red_h + warp * 4, _f32_bits(lsh))
        cute.arch.sync_threads()
        ug0 = _lds4(red_g)
        ug1 = _lds4(red_g + 16)
        uh0 = _lds4(red_h)
        uh1 = _lds4(red_h + 16)
        tsg = cutlass.Float32(0.0)
        tsh = cutlass.Float32(0.0)
        for e in cutlass.range_constexpr(4):
            tsg = tsg + _bits_f32(ug0[e])
            tsh = tsh + _bits_f32(uh0[e])
        for e in cutlass.range_constexpr(4):
            tsg = tsg + _bits_f32(ug1[e])
            tsh = tsh + _bits_f32(uh1[e])
        st[2] = tsg
        st[3] = tsh

        # ---- epilogue -----------------------------------------------------------------------
        gv = mGs[2]
        lg = st[2]
        lh = st[3]
        ig = cutlass.Float32(0.0)
        ih = cutlass.Float32(0.0)
        if lg > 0.0:
            ig = gv / lg
        if lh > 0.0:
            ih = gv / lh
        hg = kvh * group + g
        hh = kvh * group + g + 8
        okh = (g + 8) < group
        for nt in cutlass.range_constexpr(4):
            d = 32 * warp + 8 * nt + 2 * c
            og0 = acc[nt, 0] * ig
            og1 = acc[nt, 1] * ig
            oh0 = acc[nt, 2] * ih
            oh1 = acc[nt, 3] * ih
            if const_expr(splits == 1):
                ag = out_base + (cutlass.Int64(row) * heads + hg) * (HEAD_DIM * 2) + d * 2
                ah = out_base + (cutlass.Int64(row) * heads + hh) * (HEAD_DIM * 2) + d * 2
                if const_expr(self.gated):
                    gw = _ldg1(
                        gate_base + (cutlass.Int64(row) * gate_stride + hg * HEAD_DIM + d) * 2
                    )
                    og0 = _round_bf16(og0) * _sigmoid(_bf2_lo(gw))
                    og1 = _round_bf16(og1) * _sigmoid(_bf2_hi(gw))
                _stg1(ag, _bf2_pack(og0, og1))
                if okh:
                    if const_expr(self.gated):
                        gw2 = _ldg1(
                            gate_base + (cutlass.Int64(row) * gate_stride + hh * HEAD_DIM + d) * 2
                        )
                        oh0 = _round_bf16(oh0) * _sigmoid(_bf2_lo(gw2))
                        oh1 = _round_bf16(oh1) * _sigmoid(_bf2_hi(gw2))
                    _stg1(ah, _bf2_pack(oh0, oh1))
            else:
                pg_ = (
                    out_base
                    + ((cutlass.Int64(split) * rows + row) * heads + hg) * (HEAD_DIM * 4)
                    + d * 4
                )
                ph_ = (
                    out_base
                    + ((cutlass.Int64(split) * rows + row) * heads + hh) * (HEAD_DIM * 4)
                    + d * 4
                )
                _stg2f(pg_, og0, og1)
                if okh:
                    _stg2f(ph_, oh0, oh1)
        if const_expr(splits > 1):
            if warp == 0:
                if c == 0:
                    lse_g = neg_inf
                    if lg > 0.0:
                        lse_g = st[0] + cute.math.log2(lg)
                    _stg1(
                        lse_base + ((cutlass.Int64(split) * rows + row) * heads + hg) * 4,
                        _f32_bits(lse_g),
                    )
                    if okh:
                        lse_h = neg_inf
                        if lh > 0.0:
                            lse_h = st[1] + cute.math.log2(lh)
                        _stg1(
                            lse_base + ((cutlass.Int64(split) * rows + row) * heads + hh) * 4,
                            _f32_bits(lse_h),
                        )


@dsl_user_op
def _f32_bits(x, *, loc=None, ip=None):
    return cutlass.Int32(
        _asm(T.i32(), [_ir(x, cutlass.Float32, loc, ip)], "mov.b32 $0, $1;", "=r,f", loc, ip)
    )


@dsl_user_op
def _bits_f32(x, *, loc=None, ip=None):
    return cutlass.Float32(
        _asm(T.f32(), [_ir(x, cutlass.Int32, loc, ip)], "mov.b32 $0, $1;", "=f,r", loc, ip)
    )


@dsl_user_op
def _round_bf16(x, *, loc=None, ip=None):
    return cutlass.Float32(
        _asm(
            T.f32(),
            [_ir(x, cutlass.Float32, loc, ip)],
            "{ .reg .b16 h; cvt.rn.bf16.f32 h, $1; cvt.f32.bf16 $0, h; }",
            "=f,f",
            loc,
            ip,
        )
    )


def _sigmoid(x):
    """1 / (1 + exp(-x)) as the bf16 merge computes it."""
    return 1.0 / (1.0 + cute.math.exp(-cutlass.Float32(x)))


# ---------------------------------------------------------------------------------------
# merge kernel (splits > 1)
# ---------------------------------------------------------------------------------------


class _MergeKernel:
    def __init__(self, *, heads, splits, gated, pdl):
        assert HEAD_DIM % CHUNK == 0 and 1 < splits <= MAX_SPLITS
        self.heads, self.splits, self.gated, self.pdl = heads, splits, gated, pdl
        self.chunks = HEAD_DIM // CHUNK
        # at least four warps: the final pass needs a thread per element of the chunk
        self.warps = max(4, min(8, splits))
        self.spw = -(-splits // self.warps)

    @cute.jit
    def __call__(
        self,
        gate_t: cute.Tensor,
        part_ptr: cute.Pointer,
        lse_ptr: cute.Pointer,
        out_ptr: cute.Pointer,
        rows: cutlass.Int32,
        gate_stride: cutlass.Int32,
        stream,
    ):
        lse = cute.make_tensor(
            lse_ptr,
            cute.make_layout(
                (self.splits, rows, self.heads), stride=(rows * self.heads, self.heads, 1)
            ),
        )
        out = cute.make_tensor(
            out_ptr,
            cute.make_layout(
                (rows, self.heads, HEAD_DIM), stride=(self.heads * HEAD_DIM, HEAD_DIM, 1)
            ),
        )
        self.kernel(gate_t.iterator.toint(), part_ptr.toint(), lse, out, rows, gate_stride).launch(
            grid=[rows * self.heads * self.chunks, 1, 1],
            block=[self.warps * 32, 1, 1],
            smem=self.warps * CHUNK * 4 + 64,
            stream=stream,
            use_pdl=self.pdl,
        )

    @cute.kernel
    def kernel(
        self,
        gate_base: cutlass.Int64,
        part_base: cutlass.Int64,
        mLse,
        mOut,
        rows: cutlass.Int32,
        gate_stride: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        warp_i = cute.arch.warp_idx()
        lane = cute.arch.lane_idx()
        per_row: cutlass.Constexpr = self.heads * self.chunks
        spw: cutlass.Constexpr = self.spw
        row = bidx // per_row
        head = (bidx % per_row) // self.chunks
        c0 = (bidx % self.chunks) * CHUNK
        red = cutlass.utils.SmemAllocator().allocate_tensor(
            cutlass.Float32,
            cute.make_layout((self.warps, CHUNK), stride=(CHUNK, 1)),
            byte_alignment=16,
        )
        gval = cutlass.Float32(0.0)
        if const_expr(self.gated):
            if tidx < CHUNK:
                gw = _ldg1(
                    gate_base
                    + (cutlass.Int64(row) * gate_stride + head * HEAD_DIM + c0 + (tidx & ~1)) * 2
                )
                gval = cutlass.Float32(cutlass.select_((tidx & 1) == 0, _bf2_lo(gw), _bf2_hi(gw)))
        if const_expr(self.pdl):
            cute.arch.griddepcontrol_launch_dependents()
            cute.arch.griddepcontrol_wait()
        neg_inf = cutlass.Float32(-math.inf)
        l0 = neg_inf
        l1 = neg_inf
        if lane < self.splits:
            l0 = mLse[lane, row, head]
        if lane + 32 < self.splits:
            l1 = mLse[lane + 32, row, head]
        p = cute.make_rmem_tensor(cute.make_layout((spw, 4), stride=(4, 1)), cutlass.Float32)
        for jj in cutlass.range_constexpr(spw):
            sidx = cutlass.min(warp_i * spw + jj, self.splits - 1)
            w4 = _ldg4(
                part_base
                + ((cutlass.Int64(sidx) * rows + row) * self.heads + head) * (HEAD_DIM * 4)
                + (c0 + 4 * lane) * 4
            )
            for e in cutlass.range_constexpr(4):
                p[jj, e] = _bits_f32(w4[e])
        mx = cute.arch.warp_reduction_max(cute.arch.fmax(l0, l1))
        w0 = cutlass.Float32(0.0)
        w1 = cutlass.Float32(0.0)
        if mx > neg_inf:
            w0 = cute.math.exp2(l0 - mx)
            w1 = cute.math.exp2(l1 - mx)
        den = cute.arch.warp_reduction_sum(w0 + w1)
        acc = cute.make_rmem_tensor(cute.make_layout((4,), stride=(1,)), cutlass.Float32)
        acc.fill(0.0)
        for jj in cutlass.range_constexpr(spw):
            sidx = warp_i * spw + jj
            wa = cute.arch.shuffle_sync(w0, sidx % 32)
            wb = cute.arch.shuffle_sync(w1, sidx % 32)
            w = cutlass.Float32(cutlass.select_(sidx < 32, wa, wb))
            w = cutlass.Float32(cutlass.select_(sidx < self.splits, w, cutlass.Float32(0.0)))
            for e in cutlass.range_constexpr(4):
                acc[e] = w * p[jj, e] + acc[e]
        for e in cutlass.range_constexpr(4):
            red[warp_i, 4 * lane + e] = acc[e]
        cute.arch.sync_threads()
        if tidx < CHUNK:
            tot = cutlass.Float32(0.0)
            for w_ in cutlass.range_constexpr(self.warps):
                tot = tot + red[w_, tidx]
            merged = cutlass.Float32(0.0)
            if den > 0.0:
                merged = tot / den
            if const_expr(self.gated):
                att = merged.to(cutlass.BFloat16).to(cutlass.Float32)
                merged = att * (1.0 / (1.0 + cute.math.exp(-gval)))
            mOut[row, head, c0 + tidx] = merged.to(cutlass.BFloat16)


# ---------------------------------------------------------------------------------------
# launch
# ---------------------------------------------------------------------------------------

_COMPILED: dict = {}


#: Splits by (rows x KV heads), measured on GB300 at 2051 selected tokens (sweep_splits.py):
#: about 250-290 CTAs where the merge is cheap, a single split from 128 rows up.
_SPLIT_TABLE = ((4, 33), (8, 11), (16, 17), (32, 9), (64, 4), (128, 2))


def plan_splits(rows: int, kv_heads: int, width: int) -> tuple[int, int]:
    """(splits, tiles per split) for this many (row, KV head) pairs."""
    n_tiles = -(-width // TILE)
    pairs = rows * kv_heads
    want = 1
    for bound, sp in _SPLIT_TABLE:
        if pairs <= bound:
            want = sp
            break
    want = max(1, min(n_tiles, want, MAX_SPLITS))
    tps = -(-n_tiles // want)
    return -(-n_tiles // tps), tps


def _ptr(t: torch.Tensor, dtype, align=16):
    return make_ptr(dtype, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=align)


def nvfp4_sparse_gqa_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scale_cache: torch.Tensor,
    v_scale_cache: torch.Tensor,
    kv_scale_quant_orig: torch.Tensor,
    block_table: torch.Tensor,
    selected_tokens: torch.Tensor,
    request_indices: torch.Tensor,
    tokens_per_block: int,
    softmax_scale: float,
    output_gate: Optional[torch.Tensor] = None,
    pdl: bool = True,
    splits: Optional[int] = None,
) -> torch.Tensor:
    """Decode-path NVFP4 sparse GQA. Returns `[rows, Hq, D]` bf16 (contiguous); with
    `output_gate` (`[rows, Hq * D]` bf16, unit column stride) the result is
    `bf16(attn) * sigmoid(gate)` rounded to bf16, as `qsa_paged_sparse_gqa_decode` fuses it."""
    rows, num_q_heads, head_dim = q.shape
    num_pages, num_kv_heads, tpb, packed = k_cache.shape
    assert head_dim == HEAD_DIM and packed == HEAD_DIM // 2 and tpb == tokens_per_block
    assert q.dtype == torch.bfloat16 and q.stride(2) == 1
    assert k_cache.dtype == torch.uint8 and v_cache.shape == k_cache.shape
    assert k_cache.stride(3) == 1 and v_cache.stride(3) == 1
    assert k_scale_cache.stride(2) == HEAD_DIM // 16 and v_scale_cache.stride(2) == HEAD_DIM // 16
    assert block_table.dtype == torch.int32 and block_table.stride(1) == 1
    assert block_table.stride(0) == block_table.shape[1], "row-contiguous block table"
    assert selected_tokens.dtype == torch.int32 and selected_tokens.is_contiguous()
    assert request_indices.is_contiguous() and request_indices.dtype in (torch.int32, torch.int64)
    if selected_tokens.data_ptr() % 16:
        # The kernel reads the selection with 16-byte loads. A row slice of a 2051-wide
        # selection (a mixed step's generation rows after its context rows) starts 16-byte
        # aligned only at every fourth row, so an unaligned one is copied (rows x 8 KB).
        selected_tokens = selected_tokens.clone()
    group = num_q_heads // num_kv_heads
    width = selected_tokens.shape[1]
    out = torch.empty((rows, num_q_heads, head_dim), dtype=torch.bfloat16, device=q.device)
    if rows == 0:
        return out
    if splits is None:
        n_splits, tps = plan_splits(rows, num_kv_heads, width)
    else:
        n_tiles = -(-width // TILE)
        tps = -(-n_tiles // splits)
        n_splits = -(-n_tiles // tps)
    use_pdl = bool(pdl and get_env_enable_pdl())
    gated = output_gate is not None
    if gated:
        assert output_gate.shape == (rows, num_q_heads * head_dim) and output_gate.stride(1) == 1
        assert output_gate.dtype == torch.bfloat16
    req64 = request_indices.dtype == torch.int64
    key = (
        group,
        num_kv_heads,
        width,
        tpb,
        n_splits,
        tps,
        float(softmax_scale),
        num_pages,
        tuple(block_table.shape),
        tuple(k_cache.stride()[:3]),
        tuple(v_cache.stride()[:3]),
        tuple(k_scale_cache.stride()[:2]),
        tuple(v_scale_cache.stride()[:2]),
        tuple(q.stride()[:2]),
        req64,
        gated and n_splits == 1,
        use_pdl,
    )
    if n_splits > 1:
        part = torch.empty(
            (n_splits, rows, num_q_heads, head_dim), dtype=torch.float32, device=q.device
        )
        lse = torch.empty((n_splits, rows, num_q_heads), dtype=torch.float32, device=q.device)
        o_t, l_t = part, lse
    else:
        o_t, l_t = out, out
    gate_t = output_gate if gated else q
    gate_stride = output_gate.stride(0) if gated else 0
    fn = _COMPILED.get(key)
    if fn is None:
        kern = _SplitKernel(
            group=group,
            kv_heads=num_kv_heads,
            width=width,
            tpb=tpb,
            splits=n_splits,
            tps=tps,
            scale=float(softmax_scale),
            num_pages=num_pages,
            bt_width=block_table.shape[1],
            num_requests=block_table.shape[0],
            k_strides=tuple(k_cache.stride()[:3]),
            v_strides=tuple(v_cache.stride()[:3]),
            ks_strides=tuple(k_scale_cache.stride()[:2]),
            vs_strides=tuple(v_scale_cache.stride()[:2]),
            q_strides=tuple(q.stride()[:2]),
            req64=req64,
            gated=gated and n_splits == 1,
            pdl=use_pdl,
        )
        fn = cute.compile(
            kern,
            from_dlpack(selected_tokens, assumed_align=16).mark_layout_dynamic(leading_dim=1),
            _ptr(q, cutlass.BFloat16),
            _ptr(k_cache, cutlass.Uint8),
            _ptr(v_cache, cutlass.Uint8),
            _ptr(k_scale_cache, cutlass.Uint8),
            _ptr(v_scale_cache, cutlass.Uint8),
            _ptr(kv_scale_quant_orig, cutlass.Float32, 4),
            _ptr(block_table, cutlass.Int32, 4),
            _ptr(request_indices, cutlass.Int64 if req64 else cutlass.Int32, 8 if req64 else 4),
            _ptr(o_t, cutlass.Float32),
            _ptr(l_t, cutlass.Float32),
            _ptr(gate_t, cutlass.BFloat16),
            cutlass.Int32(rows),
            cutlass.Int32(gate_stride),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--opt-level 2 --enable-tvm-ffi",
        )
        _COMPILED[key] = fn
    fn(
        selected_tokens,
        q.data_ptr(),
        k_cache.data_ptr(),
        v_cache.data_ptr(),
        k_scale_cache.data_ptr(),
        v_scale_cache.data_ptr(),
        kv_scale_quant_orig.data_ptr(),
        block_table.data_ptr(),
        request_indices.data_ptr(),
        o_t.data_ptr(),
        l_t.data_ptr(),
        gate_t.data_ptr(),
        rows,
        gate_stride,
    )
    if n_splits == 1:
        return out
    mkey = ("merge", num_q_heads, n_splits, gated, use_pdl)
    mfn = _COMPILED.get(mkey)
    if mfn is None:
        mfn = cute.compile(
            _MergeKernel(heads=num_q_heads, splits=n_splits, gated=gated, pdl=use_pdl),
            from_dlpack(gate_t, assumed_align=16).mark_layout_dynamic(leading_dim=gate_t.dim() - 1),
            _ptr(part, cutlass.Float32),
            _ptr(lse, cutlass.Float32, 4),
            _ptr(out, cutlass.BFloat16),
            cutlass.Int32(rows),
            cutlass.Int32(gate_stride),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--opt-level 2 --enable-tvm-ffi",
        )
        _COMPILED[mkey] = mfn
    mfn(gate_t, part.data_ptr(), lse.data_ptr(), out.data_ptr(), rows, gate_stride)
    return out


def _decode(segments, **overrides) -> dict:
    """A cell spec: `(first position, rows)` per request in row order, this checkpoint's
    geometry (24 q heads over 2 KV heads of 256, 32 tokens a page, 2051 selected tokens)."""
    return dict(segments=segments, **overrides)


class QsaPagedSparseGqaNvfp4Decode(OpWrapper):
    """`qsa_paged_sparse_gqa_nvfp4` for decode rows (no prefill pair): sparse GQA over the
    explicitly selected tokens of an NVFP4 paged pool, `[rows, Hq, D]` bf16. With
    `output_gate` (`[rows, Hq * D]` bf16, unit column stride) the result is
    `bf16(attn) * sigmoid(gate)` rounded once more to bf16, the composition
    `fused_sigmoid_mul_inplace` makes.

    Fusion boundary: the block-table gather, the NVFP4 dequantize, scale, online softmax,
    P.V, the split-K merge and (optionally) the output gate. Selection, the K/V append and
    q's projection, norm and RoPE are not inside.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(why="decode: 1 row, 33-way split + merge", spec=_decode(((4000, 1),))),
        Cell(
            why="MTP3 verify: 4 requests x 4 rows, gate fused into the merge",
            spec=_decode(tuple((4000 + 37 * i, 4) for i in range(4)), gated=True),
        ),
        Cell(
            why="128 rows (16 x 8), one split: gate fused into the split kernel, trunk layout",
            spec=_decode(
                tuple((100 + (499 * g) % 8000, 8) for g in range(16)),
                gated=True,
                slot_pages=24,
                layer=11,
                pool_pages=2048,
            ),
        ),
    )

    note = """
    Metadata consumed: none -- every input is an argument (as `qsa_paged_sparse_gqa_nvfp4`).
    Call sites (the target's `modeling.py`): `_sparse_gqa` over an NVFP4 pool for rows without
    the prefill pair -- the decode step's attention layers (with the output gate), the MTP
    draft layer and a mixed step's generation rows.

    Launch shapes: splits by (rows x KV heads) from `_SPLIT_TABLE` (33 splits of one 64-token
    tile at one row, a single split of 33 tiles from 128 rows up), each split followed by the
    merge. GB300, 2051 selected tokens a row, one layer: 1 row 6.7 us, 4 rows 8.5, 32 rows 22.3,
    128 rows 67.8, 256 rows 133, 512 rows 231, 1024 rows 449 -- against 9.0 / 18.8 / 49.5 /
    158 / 292 / 557 / 1036 us for the Triton op. A one-split launch (65 rows and up) runs the
    kernel without the f16 V tile: 72 rows 65.1 -> 61.7 us, 128 rows 71.6 -> 66.6, 192 rows
    111.4 -> 101.6, 256 rows 137.5 -> 128.3 (12 chained calls, gated), bitwise the standard
    kernel.

    Numerics: dequantized K/V values are exact in f16; scores on f16 MMAs with fp32
    accumulation; P rounded to f16 (as the Triton op); the second-level scales act on the
    scores and the normalized output. Within the Triton op's band (4 ulp element, 2 ulp RMS)
    of the float64 reference; not bitwise the Triton op. Deterministic; graph-capturable
    (row count is a runtime value; one compile per split variant).

    Preconditions: bf16 q with head_dim 256; K and V planes (and their scale planes) share
    strides; power-of-two tokens per block; int32 row-contiguous block table; contiguous
    int32 selections.
    """

    def __call__(
        self,
        q,
        k_cache,
        v_cache,
        k_scale_cache,
        v_scale_cache,
        kv_scale_quant_orig,
        block_table,
        selected_tokens,
        request_indices,
        tokens_per_block: int,
        softmax_scale: float,
        output_gate=None,
    ) -> torch.Tensor:
        return nvfp4_sparse_gqa_decode(
            q,
            k_cache,
            v_cache,
            k_scale_cache,
            v_scale_cache,
            kv_scale_quant_orig,
            block_table,
            selected_tokens,
            request_indices,
            tokens_per_block,
            softmax_scale,
            output_gate=output_gate,
        )

    def reference(
        self,
        q,
        k_cache,
        v_cache,
        k_scale_cache,
        v_scale_cache,
        kv_scale_quant_orig,
        block_table,
        selected_tokens,
        request_indices,
        tokens_per_block: int,
        softmax_scale: float,
        output_gate=None,
    ) -> torch.Tensor:
        """`qsa_paged_sparse_gqa_nvfp4`'s float64 reference, then the gate in fp32 over its
        bf16 rounding, rounded once."""
        out = _composition.reference(
            q,
            k_cache,
            v_cache,
            k_scale_cache,
            v_scale_cache,
            kv_scale_quant_orig,
            block_table,
            selected_tokens,
            request_indices,
            tokens_per_block,
            softmax_scale,
        )
        if output_gate is None:
            return out
        rows, heads, dim = out.shape
        gate = torch.sigmoid(output_gate.float().view(rows, heads, dim))
        return (out.to(torch.bfloat16).float() * gate).to(torch.bfloat16)

    def is_valid(
        self,
        q,
        k_cache,
        v_cache,
        k_scale_cache,
        v_scale_cache,
        kv_scale_quant_orig,
        block_table,
        selected_tokens,
        request_indices,
        tokens_per_block: int,
        softmax_scale: float,
        output_gate=None,
    ) -> None:
        assert q.dtype == torch.bfloat16 and q.dim() == 3 and q.shape[2] == HEAD_DIM
        assert q.stride(2) == 1, "q unit last stride"
        assert k_cache.dtype == torch.uint8 and v_cache.shape == k_cache.shape
        assert k_cache.stride() == v_cache.stride(), "K and V planes share strides"
        assert k_scale_cache.stride() == v_scale_cache.stride(), "scale planes share strides"
        assert tokens_per_block & (tokens_per_block - 1) == 0, "power-of-two tokens per block"
        assert block_table.dtype == torch.int32 and block_table.is_contiguous()
        assert selected_tokens.dtype == torch.int32 and selected_tokens.is_contiguous()
        assert output_gate is None or (
            output_gate.dtype == torch.bfloat16
            and output_gate.shape == (q.shape[0], q.shape[1] * q.shape[2])
            and output_gate.stride(1) == 1
        )


qsa_paged_sparse_gqa_nvfp4_decode = QsaPagedSparseGqaNvfp4Decode()
