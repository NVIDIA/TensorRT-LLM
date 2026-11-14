# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused Helix context-parallel all-to-all and combine (SM100 / SM103).

Helix runs the MLA decode attention of every head over a ``1 / P`` slice of the KV cache
(``P`` = CP size). Each rank then needs, for its ``H`` heads, the partial outputs and softmax
statistics of all ``P`` ranks and combines them with a log-sum-exp weighted sum. One launch of
:class:`HelixA2AFusedKernel` does the exchange and the combine:

* Sender CTAs stage chunks of ``(token, destination rank)`` records of this rank in shared memory
  -- the BF16 partial output of the destination's ``H`` heads (``H * D`` values) followed by
  their FP32 ``(max, sum)`` pairs -- and send each record with one bulk copy (TMA) straight into
  slot ``(token, source rank)`` of the destination's receive buffer (MNNVL fabric memory).
  Rows this rank owns no KV for (``zero_kv_mask``) are replaced on the way by the no-op
  contribution (zeros, ``(-inf, 0)``). The bulk copies keep the remote stores out of the SM's
  load/store pipeline, which the receivers on the same SMs poll through.
* Receiver CTAs own tokens; one thread owns one 16-byte column (8 output values) of one head.
  Lane ``l`` of a 16-lane head group polls the statistics of source rank ``l``; the group
  computes the combine weights with the shuffles of the reference kernel; every lane then polls
  its column's ``P`` vectors, accumulates them with FMAs in source order, rounds to BF16, stores
  the output and re-arms the words it read.

Receive buffers are Lamport buffers: every word starts as the sentinel ``0xffffffff``, a NaN
pattern the sender never stores (it rewrites such words to ``0x7fffffff``, also NaN for BF16
pairs and FP32 alike, and NaN inputs combine to the same canonical NaN). A word that is not the
sentinel has therefore arrived; no flag, fence or counter sits between a sender's stores and
the receiver's loads. The buffer has two halves used by alternate launches. A launch reads its
half and re-arms what it read before it exits. The next stores into that half come two launches
later, from a rank that has finished the launch in between, which needed this rank's records of
that launch, which this rank sends only after this launch has finished.

The half is the parity of a launch counter: every CTA takes a ticket from a packed 64-bit word
(high half: launch number, low half: CTAs of the launch that have arrived; the last arrival
advances the launch number), so the grid can vary between launches and the kernel replays
inside CUDA graphs. Launches that share a workspace must be ordered (one stream). A CTA takes
its ticket after ``griddepcontrol.wait``, when the previous grid has completed, and only then
lets the next grid launch, so the tickets of consecutive launches never interleave (a dependent
grid can start before every CTA of a grid larger than the GPU has started).

The output is bitwise equal to ``helixAllToAllKernel`` followed by
``helix_postprocess_kernel_native_v2``: same ``expf`` and reciprocal instruction sequences,
same max / sum butterflies, FMA accumulation in source order, round-to-nearest BF16. That holds
for ``P <= 8`` and ``P = 16`` (other sizes pad the reference's source loop with zero terms).
"""

from __future__ import annotations

import functools

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch
from cutlass import Float32, Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

#: The empty-word sentinel of the receive buffers (and its replacement in sent data).
SENTINEL = -1
#: FP32 bits of -inf (the max of a no-op contribution).
_NEG_INF_BITS = -8388608  # 0xff800000
#: Largest supported CP size (one 16-lane shuffle group holds the statistics of all sources).
MAX_CP = 16
#: Records per sender chunk (one bulk copy each) and shared-memory stages per sender CTA.
_CHUNK_RECORDS = 8
_STAGES = 3
#: Threads per CTA (receivers: whole tokens of ``H * D / 8`` threads each).
_TARGET_THREADS = 256


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


# ---------------------------------------------------------------------------
# Inline PTX helpers
# ---------------------------------------------------------------------------


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


def _fv(value, *, loc=None, ip=None):
    return Float32(value).ir_value(loc=loc, ip=ip)


def _unpack(res, ty, cls, n, *, loc=None, ip=None):
    return tuple(cls(llvm.extractvalue(ty, res, [i], loc=loc, ip=ip)) for i in range(n))


@dsl_user_op
def _atom_add_u64(addr, value, *, loc=None, ip=None):
    return Int64(
        _asm(
            T.i64(),
            [_lv(addr, loc=loc, ip=ip), _lv(value, loc=loc, ip=ip)],
            "atom.relaxed.gpu.global.add.u64 $0, [$1], $2;",
            "=l,l,l",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _sts_b64(addr, value, *, loc=None, ip=None):
    _asm(
        None,
        [_iv(addr, loc=loc, ip=ip), _lv(value, loc=loc, ip=ip)],
        "st.shared.b64 [$0], $1;",
        "r,l",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _lds_b64(addr, *, loc=None, ip=None):
    return Int64(
        _asm(
            T.i64(), [_iv(addr, loc=loc, ip=ip)], "ld.shared.b64 $0, [$1];", "=l,r", loc=loc, ip=ip
        )
    )


@dsl_user_op
def _ld_nc_v4(addr, *, loc=None, ip=None):
    """16 bytes of a tensor the kernel does not write."""
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [_lv(addr, loc=loc, ip=ip)],
        "ld.global.nc.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        loc=loc,
        ip=ip,
    )
    return _unpack(res, T.i32(), Int32, 4, loc=loc, ip=ip)


@dsl_user_op
def _ld_vol_v4(addr, *, loc=None, ip=None):
    """16 bytes of the receive buffer, read past L1 (peers write it over NVLink)."""
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [_lv(addr, loc=loc, ip=ip)],
        "ld.volatile.global.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        loc=loc,
        ip=ip,
    )
    return _unpack(res, T.i32(), Int32, 4, loc=loc, ip=ip)


@dsl_user_op
def _ld_vol_v2(addr, *, loc=None, ip=None):
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 2),
        [_lv(addr, loc=loc, ip=ip)],
        "ld.volatile.global.v2.b32 {$0, $1}, [$2];",
        "=r,=r,l",
        loc=loc,
        ip=ip,
    )
    return _unpack(res, T.i32(), Int32, 2, loc=loc, ip=ip)


@dsl_user_op
def _ld_nc_u8(addr, *, loc=None, ip=None):
    return Int32(
        _asm(
            T.i32(),
            [_lv(addr, loc=loc, ip=ip)],
            "ld.global.nc.u8 $0, [$1];",
            "=r,l",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _st_sentinel_v4(addr, *, loc=None, ip=None):
    _asm(
        None,
        [_lv(addr, loc=loc, ip=ip)],
        "{\n.reg .b32 t;\nmov.b32 t, -1;\nst.global.v4.b32 [$0], {t, t, t, t};\n}",
        "l",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _st_v4(addr, v0, v1, v2, v3, *, loc=None, ip=None):
    _asm(
        None,
        [_lv(addr, loc=loc, ip=ip)] + [_iv(v, loc=loc, ip=ip) for v in (v0, v1, v2, v3)],
        "st.global.v4.b32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _all_valid2(w0, w1, *, loc=None, ip=None):
    return Int32(
        _asm(
            T.i32(),
            [_iv(w0, loc=loc, ip=ip), _iv(w1, loc=loc, ip=ip)],
            "{\n.reg .pred p;\nsetp.ne.b32 p, $1, -1;\nsetp.ne.and.b32 p, $2, -1, p;\n"
            "selp.u32 $0, 1, 0, p;\n}",
            "=r,r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _prep4(w0, w1, w2, w3, zero_kv, is_stats, *, loc=None, ip=None):
    """Words of one 16-byte vector as sent: the no-op contribution on zero-KV rows (zeros, or
    ``(-inf, 0, -inf, 0)`` for a statistics vector), and no word equal to the sentinel."""
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [_iv(w, loc=loc, ip=ip) for w in (w0, w1, w2, w3)]
        + [_iv(zero_kv, loc=loc, ip=ip), _iv(is_stats, loc=loc, ip=ip)],
        "{\n.reg .pred pz, ps, p;\n.reg .b32 m, x;\n"
        "setp.ne.u32 pz, $8, 0;\nsetp.ne.u32 ps, $9, 0;\n"
        "selp.b32 m, 0xff800000, 0, ps;\n"
        "selp.b32 x, m, $4, pz;\nsetp.eq.b32 p, x, -1;\nselp.b32 $0, 0x7fffffff, x, p;\n"
        "selp.b32 x, 0, $5, pz;\nsetp.eq.b32 p, x, -1;\nselp.b32 $1, 0x7fffffff, x, p;\n"
        "selp.b32 x, m, $6, pz;\nsetp.eq.b32 p, x, -1;\nselp.b32 $2, 0x7fffffff, x, p;\n"
        "selp.b32 x, 0, $7, pz;\nsetp.eq.b32 p, x, -1;\nselp.b32 $3, 0x7fffffff, x, p;\n}",
        "=r,=r,=r,=r,r,r,r,r,r,r",
        loc=loc,
        ip=ip,
    )
    return _unpack(res, T.i32(), Int32, 4, loc=loc, ip=ip)


@dsl_user_op
def _all_valid4(w0, w1, w2, w3, *, loc=None, ip=None):
    """1 if none of the four words is the sentinel."""
    return Int32(
        _asm(
            T.i32(),
            [_iv(w, loc=loc, ip=ip) for w in (w0, w1, w2, w3)],
            "{\n.reg .pred p;\n"
            "setp.ne.b32 p, $1, -1;\nsetp.ne.and.b32 p, $2, -1, p;\n"
            "setp.ne.and.b32 p, $3, -1, p;\nsetp.ne.and.b32 p, $4, -1, p;\n"
            "selp.u32 $0, 1, 0, p;\n}",
            "=r,r,r,r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _st_sentinel_v2(addr, *, loc=None, ip=None):
    _asm(
        None,
        [_lv(addr, loc=loc, ip=ip)],
        "{\n.reg .b32 t;\nmov.b32 t, -1;\nst.global.v2.b32 [$0], {t, t};\n}",
        "l",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _as_f32(bits, *, loc=None, ip=None):
    return Float32(
        _asm(T.f32(), [_iv(bits, loc=loc, ip=ip)], "mov.b32 $0, $1;", "=f,r", loc=loc, ip=ip)
    )


@dsl_user_op
def _fmax(a, b, *, loc=None, ip=None):
    return Float32(
        _asm(
            T.f32(),
            [_fv(a, loc=loc, ip=ip), _fv(b, loc=loc, ip=ip)],
            "max.f32 $0, $1, $2;",
            "=f,f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _fadd(a, b, *, loc=None, ip=None):
    return Float32(
        _asm(
            T.f32(),
            [_fv(a, loc=loc, ip=ip), _fv(b, loc=loc, ip=ip)],
            "add.rn.f32 $0, $1, $2;",
            "=f,f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _fmul(a, b, *, loc=None, ip=None):
    return Float32(
        _asm(
            T.f32(),
            [_fv(a, loc=loc, ip=ip), _fv(b, loc=loc, ip=ip)],
            "mul.rn.f32 $0, $1, $2;",
            "=f,f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _rcp(a, *, loc=None, ip=None):
    """IEEE reciprocal (what ``1.F / x`` compiles to without fast math)."""
    return Float32(
        _asm(T.f32(), [_fv(a, loc=loc, ip=ip)], "rcp.rn.f32 $0, $1;", "=f,f", loc=loc, ip=ip)
    )


@dsl_user_op
def _corrected_exp(mx, gmax, total, *, loc=None, ip=None):
    """``total * expf(mx - gmax)`` with CUDA's (non fast-math) ``expf`` instruction sequence."""
    return Float32(
        _asm(
            T.f32(),
            [_fv(mx, loc=loc, ip=ip), _fv(gmax, loc=loc, ip=ip), _fv(total, loc=loc, ip=ip)],
            "{\n.reg .f32 d, t, j, f, e, s;\n.reg .b32 b;\n"
            "sub.rn.f32 d, $1, $2;\n"
            "fma.rn.sat.f32 t, d, 0f3BBB989D, 0f3F000000;\n"
            "fma.rm.f32 t, t, 0f437C0000, 0f4B400001;\n"
            "add.rn.f32 j, t, 0fCB40007F;\n"
            "neg.f32 j, j;\n"
            "fma.rn.f32 f, d, 0f3FB8AA3B, j;\n"
            "fma.rn.f32 f, d, 0f32A57060, f;\n"
            "ex2.approx.ftz.f32 e, f;\n"
            "mov.b32 b, t;\nshl.b32 b, b, 23;\nmov.b32 s, b;\n"
            "mul.rn.f32 e, s, e;\n"
            "mul.rn.f32 $0, e, $3;\n}",
            "=f,f,f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _shfl_bfly(value, offset: int, *, loc=None, ip=None):
    return Float32(
        _asm(
            T.f32(),
            [_fv(value, loc=loc, ip=ip)],
            f"shfl.sync.bfly.b32 $0, $1, {offset}, 31, -1;",
            "=f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _shfl_idx(value, lane, *, loc=None, ip=None):
    return Float32(
        _asm(
            T.f32(),
            [_fv(value, loc=loc, ip=ip), _iv(lane, loc=loc, ip=ip)],
            "shfl.sync.idx.b32 $0, $1, $2, 31, -1;",
            "=f,f,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _bf16x2_fma(word, scale, acc_lo, acc_hi, *, loc=None, ip=None):
    """``(fma(lo, scale, acc_lo), fma(hi, scale, acc_hi))`` of the BF16 pair in ``word``."""
    res = _asm(
        llvm.StructType.get_literal([T.f32()] * 2),
        [
            _iv(word, loc=loc, ip=ip),
            _fv(scale, loc=loc, ip=ip),
            _fv(acc_lo, loc=loc, ip=ip),
            _fv(acc_hi, loc=loc, ip=ip),
        ],
        "{\n.reg .b32 l, h;\n.reg .f32 fl, fh;\n"
        "shl.b32 l, $2, 16;\nand.b32 h, $2, 0xffff0000;\n"
        "mov.b32 fl, l;\nmov.b32 fh, h;\n"
        "fma.rn.f32 $0, fl, $3, $4;\nfma.rn.f32 $1, fh, $3, $5;\n}",
        "=f,=f,r,f,f,f",
        loc=loc,
        ip=ip,
    )
    return _unpack(res, T.f32(), Float32, 2, loc=loc, ip=ip)


@dsl_user_op
def _pack_bf16x2(lo, hi, *, loc=None, ip=None):
    """Round two FP32 values to BF16 (nearest even), ``lo`` in the low half."""
    return Int32(
        _asm(
            T.i32(),
            [_fv(lo, loc=loc, ip=ip), _fv(hi, loc=loc, ip=ip)],
            "cvt.rn.bf16x2.f32 $0, $2, $1;",
            "=r,f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _sts_v4(addr, v0, v1, v2, v3, *, loc=None, ip=None):
    _asm(
        None,
        [_iv(addr, loc=loc, ip=ip)] + [_iv(v, loc=loc, ip=ip) for v in (v0, v1, v2, v3)],
        "st.shared.v4.b32 [$0], {$1, $2, $3, $4};",
        "r,r,r,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _fence_proxy_async_shared(*, loc=None, ip=None):
    _asm(None, [], "fence.proxy.async.shared::cta;", "", loc=loc, ip=ip)


@dsl_user_op
def _bulk_s2g(dst, src, nbytes: int, *, loc=None, ip=None):
    """Bulk copy (TMA) of ``nbytes`` from shared ``src`` to global ``dst`` (local or peer)."""
    _asm(
        None,
        [_lv(dst, loc=loc, ip=ip), _iv(src, loc=loc, ip=ip)],
        f"cp.async.bulk.global.shared::cta.bulk_group [$0], [$1], {nbytes};",
        "l,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _bulk_commit(*, loc=None, ip=None):
    _asm(None, [], "cp.async.bulk.commit_group;", "", loc=loc, ip=ip)


@dsl_user_op
def _bulk_wait_read(n: int, *, loc=None, ip=None):
    _asm(None, [], f"cp.async.bulk.wait_group.read {n};", "", loc=loc, ip=ip)


@dsl_user_op
def _griddepcontrol_wait(*, loc=None, ip=None):
    _asm(None, [], "griddepcontrol.wait;", "", loc=loc, ip=ip)


@dsl_user_op
def _griddepcontrol_launch_dependents(*, loc=None, ip=None):
    _asm(None, [], "griddepcontrol.launch_dependents;", "", loc=loc, ip=ip)


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------


class HelixA2AFusedKernel:
    """Exchange + combine of one Helix MLA layer (see the module docstring).

    Args:
        cp_size: ``P`` (at most 8, or 16).
        num_heads: ``H``, heads per rank after the exchange (even).
        head_dim: ``D``, values per head (128 or 256).
        has_mask: Apply a per-token ``zero_kv_mask`` (uint8) on the sender side.
        use_pdl: Launch with programmatic dependent launch.
    """

    def __init__(self, cp_size: int, num_heads: int, head_dim: int, has_mask: bool, use_pdl: bool):
        if not supports(cp_size, num_heads, head_dim):
            raise ValueError(f"cp_size={cp_size}, num_heads={num_heads}, head_dim={head_dim}")
        self.P = cp_size
        self.has_mask = has_mask
        self.use_pdl = use_pdl
        # 16-byte vectors of a record: partial output, then statistics.
        self.DV = num_heads * head_dim // 8
        self.SV = num_heads // 2
        self.RV = self.DV + self.SV
        # Columns (16-byte vectors) per head; a 16-lane group of a warp is one head.
        self.CPH = head_dim // 8
        self.tok_threads = _ceil_div(self.DV, 32) * 32
        self.tpb = max(1, _TARGET_THREADS // self.tok_threads)
        self.threads = self.tpb * self.tok_threads

    @cute.jit
    def __call__(
        self,
        o_addr: Int64,
        s_addr: Int64,
        mask_addr: Int64,
        out_addr: Int64,
        ws_addr: Int64,
        state_addr: Int64,
        rank_stride: Int64,
        half_bytes: Int64,
        num_tokens: Int32,
        rank: Int32,
        n_send: Int32,
        n_recv: Int32,
        stream: cuda.CUstream,
    ):
        @cute.struct
        class SharedStorage:
            half: cute.struct.Align[cute.struct.MemRange[cutlass.Int64, 2], 16]
            stage: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, _STAGES * _CHUNK_RECORDS * self.RV * 4], 128
            ]

        self.shared_storage = SharedStorage
        self.kernel(
            o_addr,
            s_addr,
            mask_addr,
            out_addr,
            ws_addr,
            state_addr,
            rank_stride,
            half_bytes,
            num_tokens,
            rank,
            n_send,
            n_recv,
        ).launch(
            grid=(n_send + n_recv, 1, 1),
            block=(self.threads, 1, 1),
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        o_addr: Int64,
        s_addr: Int64,
        mask_addr: Int64,
        out_addr: Int64,
        ws_addr: Int64,
        state_addr: Int64,
        rank_stride: Int64,
        half_bytes: Int64,
        num_tokens: Int32,
        rank: Int32,
        n_send: Int32,
        n_recv: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        storage = utils.SmemAllocator().allocate(self.shared_storage)
        half_slot = storage.half.data_ptr().toint()
        if cutlass.const_expr(self.use_pdl):
            _griddepcontrol_wait()
        # The ticket is taken after the wait for the previous grid: the previous launch of this
        # kernel has then completed, so all of its CTAs have their tickets (a dependent grid may
        # start before every CTA of a grid larger than the GPU has started). The senders' first
        # loads overlap the atomic; only thread 0 needs its result before the first bulk copy.
        if tidx == 0:
            n_ctas = Int64(n_send + n_recv)
            ticket = _atom_add_u64(state_addr, Int64(1))
            if (ticket & Int64(0xFFFFFFFF)) == n_ctas - Int64(1):
                # Last arrival: clear the arrivals and advance the launch number in one add.
                _atom_add_u64(state_addr, Int64(1 << 32) - n_ctas)
            _sts_b64(half_slot, ((ticket >> 32) & Int64(1)) * half_bytes)
            if cutlass.const_expr(self.use_pdl):
                _griddepcontrol_launch_dependents()
        if bidx < n_send:
            # Only thread 0, which took the ticket, issues the bulk copies and needs the half.
            self.send(
                tidx, bidx, o_addr, s_addr, mask_addr, ws_addr, half_slot, rank_stride,
                num_tokens, rank, n_send, storage.stage.data_ptr().toint(),
            )  # fmt: skip
        else:
            cute.arch.barrier()
            self.receive(
                tidx,
                bidx - n_send,
                out_addr,
                ws_addr + Int64(rank) * rank_stride + _lds_b64(half_slot),
                num_tokens,
                n_recv,
            )

    @cute.jit
    def send(
        self,
        tidx: Int32,
        bidx: Int32,
        o_addr: Int64,
        s_addr: Int64,
        mask_addr: Int64,
        ws_addr: Int64,
        half_slot: Int32,
        rank_stride: Int64,
        num_tokens: Int32,
        rank: Int32,
        n_send: Int32,
        stage_base: Int32,
    ):
        """Stage chunks of records in shared memory; thread 0 sends each record with one bulk
        copy (record ``t * P + d`` of this rank goes to slot ``t * P + rank`` of rank ``d``)."""
        P, RV, DV, SV = self.P, self.RV, self.DV, self.SV
        R, NS = _CHUNK_RECORDS, _STAGES
        VPC = R * RV
        UL = _ceil_div(VPC, self.threads)
        n_rec = num_tokens * P
        n_chunks = (n_rec + (R - 1)) // R
        it = Int32(0)
        for c in cutlass.range(bidx, n_chunks, n_send, unroll=1):
            sbase = stage_base + (it % NS) * (VPC * 16)
            if tidx == 0:
                # The stage's previous bulk copies (NS chunks ago) have read shared memory.
                _bulk_wait_read(NS - 1)
            cute.arch.barrier()
            rec0 = c * R
            # All loads of the chunk (data and zero-KV flags) are in flight together.
            words = cute.make_rmem_tensor((UL * 4,), Int32)
            zero_kv = cute.make_rmem_tensor((UL,), Int32)
            for u in cutlass.range_constexpr(UL):
                j = tidx + u * self.threads
                if j < VPC:
                    rl = j // RV
                    k = j - rl * RV
                    rec = rec0 + rl
                    if rec < n_rec:
                        src = o_addr + Int64(rec * DV + k) * 16
                        if k >= DV:
                            src = s_addr + Int64(rec * SV + k - DV) * 16
                        w = _ld_nc_v4(src)
                        for e in cutlass.range_constexpr(4):
                            words[u * 4 + e] = w[e]
                        if cutlass.const_expr(self.has_mask):
                            zero_kv[u] = _ld_nc_u8(mask_addr + Int64(rec // P))
            for u in cutlass.range_constexpr(UL):
                j = tidx + u * self.threads
                if j < VPC:
                    rl = j // RV
                    k = j - rl * RV
                    if rec0 + rl < n_rec:
                        zk = Int32(0)
                        if cutlass.const_expr(self.has_mask):
                            zk = zero_kv[u]
                        w = _prep4(
                            words[u * 4],
                            words[u * 4 + 1],
                            words[u * 4 + 2],
                            words[u * 4 + 3],
                            zk,
                            Int32(k >= DV),
                        )
                        _sts_v4(sbase + j * 16, *w)
            _fence_proxy_async_shared()
            cute.arch.barrier()
            if tidx == 0:
                dst_base = ws_addr + _lds_b64(half_slot)
                for r in cutlass.range_constexpr(R):
                    rec = rec0 + r
                    if rec < n_rec:
                        t = rec // P
                        dst = rec - t * P
                        _bulk_s2g(
                            dst_base + Int64(dst) * rank_stride + Int64((t * P + rank) * RV) * 16,
                            sbase + r * (RV * 16),
                            RV * 16,
                        )
                _bulk_commit()
            it = it + 1
        if tidx == 0:
            _bulk_wait_read(0)

    @cute.jit
    def receive(
        self,
        tidx: Int32,
        rcta: Int32,
        out_addr: Int64,
        recv_base: Int64,
        num_tokens: Int32,
        n_recv: Int32,
    ):
        """Combine the records of this CTA's tokens (see the module docstring)."""
        P, RV, DV, CPH = self.P, self.RV, self.DV, self.CPH
        slot_t = tidx // self.tok_threads
        c = tidx - slot_t * self.tok_threads
        lane = tidx % 32
        group = lane - lane % 16
        h = c // CPH
        col = c - h * CPH
        src_l = col % 16
        valid = Int32(c < DV)
        has_stats = valid * Int32(src_l < P)
        # With 32 columns per head both 16-lane groups of a warp read the statistics; the first
        # re-arms them (after the shuffles, so after both have read them).
        owns_stats = has_stats * Int32(col < 16)
        for t in cutlass.range(rcta * self.tpb + slot_t, num_tokens, n_recv * self.tpb, unroll=1):
            rec0 = recv_base + Int64(t * (P * RV)) * 16
            st_addr = rec0 + Int64(src_l * RV + DV) * 16 + Int64(h * 8)
            d_addr = rec0 + Int64(c) * 16
            stats = cute.make_rmem_tensor((2,), Int32)
            stats[0] = Int32(_NEG_INF_BITS)
            stats[1] = Int32(0)
            vals = cute.make_rmem_tensor((P * 4,), Int32)
            if valid != 0:
                # Poll the statistics word and the P vectors of this column (all loads of a
                # round in flight together) until no word holds the sentinel.
                ready = Int32(0)
                while ready == 0:
                    ready = Int32(1)
                    if has_stats != 0:
                        sv = _ld_vol_v2(st_addr)
                        stats[0] = sv[0]
                        stats[1] = sv[1]
                        ready = _all_valid2(sv[0], sv[1])
                    for s in cutlass.range_constexpr(P):
                        w = _ld_vol_v4(d_addr + Int64(s * RV * 16))
                        for e in cutlass.range_constexpr(4):
                            vals[s * 4 + e] = w[e]
                        ready = ready & _all_valid4(*w)

            # Combine weights of the group's head, in the reference kernel's order: lane l holds
            # source l; max and sum butterflies over the 16 lanes (the reference's other 16
            # lanes hold -inf / +0 and change nothing).
            mx = _as_f32(stats[0])
            gmax = mx
            for off in (1, 2, 4, 8):
                gmax = _fmax(gmax, _shfl_bfly(gmax, off))
            corrected = _corrected_exp(mx, gmax, _as_f32(stats[1]))
            total = corrected
            for off in (1, 2, 4, 8):
                total = _fadd(total, _shfl_bfly(total, off))
            weight = _fmul(corrected, _rcp(total))

            acc = [Float32(0.0)] * 8
            for s in cutlass.range_constexpr(P):
                ws = _shfl_idx(weight, group + s)
                for q in cutlass.range_constexpr(4):
                    acc[2 * q], acc[2 * q + 1] = _bf16x2_fma(
                        vals[s * 4 + q], ws, acc[2 * q], acc[2 * q + 1]
                    )
            if valid != 0:
                o = [_pack_bf16x2(acc[2 * q], acc[2 * q + 1]) for q in range(4)]
                _st_v4(out_addr + Int64(t * DV + c) * 16, *o)
                for s in cutlass.range_constexpr(P):
                    _st_sentinel_v4(d_addr + Int64(s * RV * 16))
            if owns_stats != 0:
                _st_sentinel_v2(st_addr)


# ---------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------


def supports(cp_size: int, num_heads: int, head_dim: int) -> bool:
    """Whether :func:`helix_a2a_fused` handles this shape bitwise like the reference path."""
    return (
        (1 <= cp_size <= 8 or cp_size == MAX_CP) and num_heads % 2 == 0 and head_dim in (128, 256)
    )


def record_bytes(num_heads: int, head_dim: int) -> int:
    """Bytes of one ``(token, source rank)`` record in a receive buffer."""
    return num_heads * head_dim * 2 + num_heads * 8


@functools.cache
def _compiled(cp_size: int, num_heads: int, head_dim: int, has_mask: bool, use_pdl: bool):
    kernel = HelixA2AFusedKernel(cp_size, num_heads, head_dim, has_mask, use_pdl)
    addr = Int64(1 << 20)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled = cute.compile(
        kernel,
        addr,
        addr,
        addr,
        addr,
        addr,
        addr,
        Int64(1 << 29),
        Int64(1 << 24),
        Int32(1),
        Int32(0),
        Int32(1),
        Int32(1),
        stream,
    )
    return kernel, compiled


def helix_a2a_fused(
    partial_o: torch.Tensor,
    softmax_stats: torch.Tensor,
    zero_kv_mask: torch.Tensor | None,
    out: torch.Tensor,
    *,
    cp_rank: int,
    cp_size: int,
    num_heads: int,
    head_dim: int,
    ws_addr: int,
    rank_stride: int,
    half_bytes: int,
    state: torch.Tensor,
    max_send_ctas: int,
    max_recv_ctas: int,
    use_pdl: bool = True,
) -> None:
    """Launch the exchange + combine of ``num_tokens = partial_o.shape[0]`` tokens into ``out``.

    ``partial_o``: BF16 ``[T, P * H * D]`` (destination-rank major), ``softmax_stats``: FP32
    ``[T, P * H * 2]``, ``zero_kv_mask``: bool ``[T]`` or None, ``out``: BF16 ``[T, H * D]``.
    ``ws_addr`` is rank 0's receive buffer in this process's address space, rank ``r``'s at
    ``ws_addr + r * rank_stride``; each holds two halves of ``half_bytes``, every word armed
    with the sentinel. ``state`` is the int64 launch counter shared by all launches on the
    workspace. ``T`` records of ``record_bytes(H, D)`` bytes per source must fit a half.
    """
    num_tokens = partial_o.shape[0]
    if num_tokens == 0:
        return
    kernel, compiled = _compiled(cp_size, num_heads, head_dim, zero_kv_mask is not None, use_pdl)
    # Up to max_send_ctas senders once there are that many chunks; fewer, larger senders below
    # (half the cap) send small messages faster.
    n_chunks = _ceil_div(num_tokens * cp_size, _CHUNK_RECORDS)
    n_send = (
        max_send_ctas if n_chunks >= max_send_ctas else min(n_chunks, max(1, max_send_ctas // 2))
    )
    n_recv = min(max_recv_ctas, _ceil_div(num_tokens, kernel.tpb))
    mask_addr = zero_kv_mask.data_ptr() if zero_kv_mask is not None else 0
    compiled(
        Int64(partial_o.data_ptr()),
        Int64(softmax_stats.data_ptr()),
        Int64(mask_addr),
        Int64(out.data_ptr()),
        Int64(ws_addr),
        Int64(state.data_ptr()),
        Int64(rank_stride),
        Int64(half_bytes),
        Int32(num_tokens),
        Int32(cp_rank),
        Int32(n_send),
        Int32(n_recv),
        cuda.CUstream(torch.cuda.current_stream(partial_o.device).cuda_stream),
    )
