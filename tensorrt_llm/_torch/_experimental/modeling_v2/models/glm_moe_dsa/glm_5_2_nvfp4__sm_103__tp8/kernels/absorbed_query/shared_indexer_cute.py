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
"""shared_indexer_absorbed_query: CuTe-DSL cluster kernel for the shared-indexer-layer query region.

Same contract as ``shared_indexer.shared_query_kernel`` (the preabsorbed
query_front packet of a GLM-5.x DSA layer without its own indexer, bit for
bit equal to the native ``forward_dsa_proj`` + absorption BMM), computed
without exchanging data through global memory:

* grid of 9 thread-block clusters of 8 CTAs (256 threads each); cluster
  ``h < 8`` serves local attention head ``h``, CTA ``j`` of it the 32
  ``q_b`` output columns ``[h * 256 + 32 j, +32)`` (``j < 6``: no-RoPE,
  ``j = 6, 7``: RoPE) and the 64 absorbed latent columns ``[64 j, +64)``;
* before ``griddepcontrol.wait`` a CTA copies its ``q_b`` weight tile (32 x
  2048 BF16), its ``W_UK`` slice (64 x 192 BF16) and the q_a norm gain
  into shared memory with ``cp.async`` (weights only; under programmatic
  dependent launch this overlaps the preceding kv_a GEMM);
* after the wait every CTA reproduces the native ``q_a_layernorm``
  (FlashInfer CuTe-DSL ``RMSNormKernel``: one warp per row, lane ``t`` owns
  columns ``256 b + 8 t + v``, sequential fold over ``b`` then ``v``,
  butterfly over lanes, ``rsqrt(sum / H + eps)`` (fast-math), ``bf16((x *
  rstd) * w)``) for all rows into its own shared memory -- no cross-CTA
  flag;
* two warps run the ``q_b`` tile as ``mma.sync.m16n8k16`` chains over K =
  2048 in ascending order (weights as the A operand, the token rows as the
  8-wide B operand: the same products and order as the Triton kernels);
* the no-RoPE CTAs push their ``q`` tile into every cluster peer's shared
  memory with ``st.async`` completing on the receiver's mbarrier (each CTA
  waits only for the bytes it receives; the barriers are initialized and
  made cluster-visible before the dependency wait); every CTA then absorbs
  its 64 latent columns with a 12-step ``mma.sync`` chain over d = 0..191
  and stores them;
* cluster 8 reproduces the native ``kv_a_layernorm`` (H = 512) into
  ``latent[:, :512]`` and copies ``k_pe`` (one row per CTA).

Support domain of this kernel: 1 <= R <= 8 token rows (one 8-wide MMA
tile), the GLM-5.2 TP8 shapes, BF16, sm100 / sm103.
"""

from typing import Tuple

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.utils.smem_allocator import SmemAllocator

_Q_LORA = 2048
_KV_LORA = 512
_ROPE = 64
_NOPE = 192
_HEAD_DIM = 256
_H = 8
_KVA_COLS = _Q_LORA + _KV_LORA + _ROPE  # 2624
_MAX_ROWS = 8

# Shared-memory row strides (bytes) padded by 16 B so that the 8 rows an
# ldmatrix / fragment load touches fall into distinct banks.
_W_STRIDE = _Q_LORA * 2 + 16  # q_b weight tile rows and normalized q_a rows
_UK_STRIDE = _NOPE * 2 + 16  # W_UK slice rows and gathered q_nope rows
_TILE_COLS = 32  # q_b columns per CTA
_ABS_COLS = 64  # latent columns per CTA

# --------------------------------------------------------------------------
# PTX helpers.
# --------------------------------------------------------------------------


def _asm(res, args, text, cons, side_effects=True, loc=None, ip=None):
    return llvm.inline_asm(
        res,
        args,
        text,
        cons,
        has_side_effects=side_effects,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _gaddr(t: cute.Tensor, *, loc=None, ip=None) -> Int64:
    """64-bit global address of a tensor's first element."""
    return Int64(llvm.ptrtoint(T.i64(), t.iterator.llvm_ptr, loc=loc, ip=ip))


@dsl_user_op
def _saddr(p: cute.Pointer, *, loc=None, ip=None) -> Int32:
    """32-bit shared-window address of a shared-memory pointer."""
    return Int32(p.toint(loc=loc, ip=ip))


@dsl_user_op
def _cp_async16(saddr: Int32, gaddr: Int64, *, loc=None, ip=None):
    _asm(
        None,
        [saddr.ir_value(), gaddr.ir_value()],
        "cp.async.cg.shared.global [$0], [$1], 16;",
        "r,l",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _cp_async_commit(*, loc=None, ip=None):
    _asm(None, [], "cp.async.commit_group;", "", loc=loc, ip=ip)


@dsl_user_op
def _cp_async_wait_group(n: int, *, loc=None, ip=None):
    """Wait until at most ``n`` committed cp.async groups are pending."""
    _asm(None, [], f"cp.async.wait_group {n};", "", loc=loc, ip=ip)


@dsl_user_op
def _cp_async_commit_wait_all(*, loc=None, ip=None):
    _asm(None, [], "cp.async.commit_group;\n\tcp.async.wait_group 0;", "", loc=loc, ip=ip)


@dsl_user_op
def _ld_global_v4(gaddr: Int64, *, loc=None, ip=None):
    out = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [gaddr.ir_value()],
        "ld.global.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        loc=loc,
        ip=ip,
    )
    return tuple(Int32(llvm.extractvalue(T.i32(), out, [i], loc=loc, ip=ip)) for i in range(4))


@dsl_user_op
def _st_global_b16(gaddr: Int64, v: Int32, *, loc=None, ip=None):
    _asm(
        None,
        [gaddr.ir_value(), v.ir_value()],
        "{\n\t.reg .b16 lo;\n\tcvt.u16.u32 lo, $1;\n\tst.global.b16 [$0], lo;\n\t}",
        "l,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _st_global_v4(gaddr: Int64, a: Int32, b: Int32, c: Int32, d: Int32, *, loc=None, ip=None):
    _asm(
        None,
        [gaddr.ir_value(), a.ir_value(), b.ir_value(), c.ir_value(), d.ir_value()],
        "st.global.v4.b32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _st_global_b64(gaddr: Int64, v: Int64, *, loc=None, ip=None):
    _asm(None, [gaddr.ir_value(), v.ir_value()], "st.global.b64 [$0], $1;", "l,l", loc=loc, ip=ip)


@dsl_user_op
def _st_shared_v4(saddr: Int32, a: Int32, b: Int32, c: Int32, d: Int32, *, loc=None, ip=None):
    _asm(
        None,
        [saddr.ir_value(), a.ir_value(), b.ir_value(), c.ir_value(), d.ir_value()],
        "st.shared.v4.b32 [$0], {$1, $2, $3, $4};",
        "r,r,r,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _st_shared_b16(saddr: Int32, v: Int32, *, loc=None, ip=None):
    _asm(
        None,
        [saddr.ir_value(), v.ir_value()],
        "{\n\t.reg .b16 lo;\n\tcvt.u16.u32 lo, $1;\n\tst.shared.b16 [$0], lo;\n\t}",
        "r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ld_shared_b32(saddr: Int32, *, loc=None, ip=None) -> Int32:
    return Int32(
        _asm(T.i32(), [saddr.ir_value()], "ld.shared.b32 $0, [$1];", "=r,r", loc=loc, ip=ip)
    )


@dsl_user_op
def _ld_shared_v4(saddr: Int32, *, loc=None, ip=None):
    out = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [saddr.ir_value()],
        "ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        loc=loc,
        ip=ip,
    )
    return tuple(Int32(llvm.extractvalue(T.i32(), out, [i], loc=loc, ip=ip)) for i in range(4))


@dsl_user_op
def _st_cluster_v4(
    saddr: Int32, rank: Int32, a: Int32, b: Int32, c: Int32, d: Int32, *, loc=None, ip=None
):
    """Store 16 bytes to the same shared-memory offset of cluster CTA
    ``rank``."""
    _asm(
        None,
        [saddr.ir_value(), rank.ir_value(), a.ir_value(), b.ir_value(), c.ir_value(), d.ir_value()],
        "{\n\t.reg .b32 ra;\n\tmapa.shared::cluster.u32 ra, $0, $1;\n\t"
        "st.shared::cluster.v4.b32 [ra], {$2, $3, $4, $5};\n\t}",
        "r,r,r,r,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _st_async_cluster_v4(
    saddr: Int32,
    mbar: Int32,
    rank: Int32,
    a: Int32,
    b: Int32,
    c: Int32,
    d: Int32,
    *,
    loc=None,
    ip=None,
):
    """``st.async`` of 16 bytes to the same shared-memory offset of cluster
    CTA ``rank``, completing transaction bytes on that CTA's mbarrier at the
    same offset as the local ``mbar``."""
    _asm(
        None,
        [
            saddr.ir_value(),
            mbar.ir_value(),
            rank.ir_value(),
            a.ir_value(),
            b.ir_value(),
            c.ir_value(),
            d.ir_value(),
        ],
        "{\n\t.reg .b32 ra, rb;\n\t"
        "mapa.shared::cluster.u32 ra, $0, $2;\n\t"
        "mapa.shared::cluster.u32 rb, $1, $2;\n\t"
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 "
        "[ra], {$3, $4, $5, $6}, [rb];\n\t}",
        "r,r,r,r,r,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _mbar_wait_cluster(mbar: Int32, phase: Int32, *, loc=None, ip=None):
    """Wait for phase ``phase`` of the local mbarrier with acquire semantics
    at cluster scope (its transaction bytes come from peer CTAs)."""
    _asm(
        None,
        [mbar.ir_value(), phase.ir_value()],
        "{\n\t.reg .pred p;\n\tINDEXER_Q_MBAR_WAIT:\n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 "
        "p, [$0], $1;\n\t@!p bra INDEXER_Q_MBAR_WAIT;\n\t}",
        "r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ldsm_x4(saddr: Int32, *, loc=None, ip=None):
    out = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [saddr.ir_value()],
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        loc=loc,
        ip=ip,
    )
    return tuple(Int32(llvm.extractvalue(T.i32(), out, [i], loc=loc, ip=ip)) for i in range(4))


@dsl_user_op
def _mma_bf16(
    a0: Int32,
    a1: Int32,
    a2: Int32,
    a3: Int32,
    b0: Int32,
    b1: Int32,
    c0: Float32,
    c1: Float32,
    c2: Float32,
    c3: Float32,
    *,
    loc=None,
    ip=None,
):
    out = _asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        [
            a0.ir_value(),
            a1.ir_value(),
            a2.ir_value(),
            a3.ir_value(),
            b0.ir_value(),
            b1.ir_value(),
            c0.ir_value(),
            c1.ir_value(),
            c2.ir_value(),
            c3.ir_value(),
        ],
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, "
        "{$10, $11, $12, $13};",
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f",
        side_effects=False,
        loc=loc,
        ip=ip,
    )
    return tuple(Float32(llvm.extractvalue(T.f32(), out, [i], loc=loc, ip=ip)) for i in range(4))


@dsl_user_op
def _bf16x2_lo(u: Int32, *, loc=None, ip=None) -> Float32:
    v = _asm(
        T.i32(), [u.ir_value()], "shl.b32 $0, $1, 16;", "=r,r", side_effects=False, loc=loc, ip=ip
    )
    return Float32(llvm.bitcast(T.f32(), v, loc=loc, ip=ip))


@dsl_user_op
def _bf16x2_hi(u: Int32, *, loc=None, ip=None) -> Float32:
    v = _asm(
        T.i32(),
        [u.ir_value()],
        "and.b32 $0, $1, 0xffff0000;",
        "=r,r",
        side_effects=False,
        loc=loc,
        ip=ip,
    )
    return Float32(llvm.bitcast(T.f32(), v, loc=loc, ip=ip))


@dsl_user_op
def _pack_bf16x2(lo: Float32, hi: Float32, *, loc=None, ip=None) -> Int32:
    """Round-to-nearest-even of two FP32 values into one BF16x2 word
    (``lo`` in the low half)."""
    return Int32(
        _asm(
            T.i32(),
            [hi.ir_value(), lo.ir_value()],
            "cvt.rn.bf16x2.f32 $0, $1, $2;",
            "=r,f,f",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _bf16_bits(x: Float32, *, loc=None, ip=None) -> Int32:
    """Round-to-nearest-even of one FP32 value to BF16 (in the low half)."""
    return Int32(
        _asm(
            T.i32(),
            [x.ir_value()],
            "{\n\t.reg .b16 h;\n\tcvt.rn.bf16.f32 h, $1;\n\tcvt.u32.u16 $0, h;\n\t}",
            "=r,f",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _cluster_sync(*, loc=None, ip=None):
    _asm(
        None,
        [],
        "barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;",
        "",
        loc=loc,
        ip=ip,
    )


# --------------------------------------------------------------------------
# Native RMSNorm reproduction (one warp per row).
# --------------------------------------------------------------------------


@cute.jit
def _norm_row_to_smem(
    src: Int64,
    gw: Int64,
    sgw: Int32,
    dst_smem: Int32,
    lane: Int32,
    eps: Float32,
    NB: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    GAIN_SMEM: cutlass.Constexpr[bool],
):
    """Normalize one BF16 row of ``H = 256 * NB`` columns at global address
    ``src`` with gain ``gw`` (global) or ``sgw`` (shared memory, with
    ``GAIN_SMEM``) and store the BF16 result to shared memory at
    ``dst_smem`` (contiguous row), as the native FlashInfer kernel does:
    lane ``t`` owns columns ``256 b + 8 t + v``, sequential fold over ``b``
    then ``v``, butterfly over the lanes, ``rsqrt(sum / H + eps)``
    (fast-math), ``bf16((x * rstd) * (w + 0))``. All loads (row and gain)
    are issued before the arithmetic."""
    xu = []
    for b in cutlass.range_constexpr(NB):
        xu.append(_ld_global_v4(src + Int64((256 * b + 8 * lane) * 2)))
    wu = []
    for b in cutlass.range_constexpr(NB):
        if cutlass.const_expr(GAIN_SMEM):
            wu.append(_ld_shared_v4(sgw + Int32((256 * b + 8 * lane) * 2)))
        else:
            wu.append(_ld_global_v4(gw + Int64((256 * b + 8 * lane) * 2)))
    xs = cute.make_rmem_tensor((NB, 8), Float32)
    for b in cutlass.range_constexpr(NB):
        u = xu[b]
        for i in cutlass.range_constexpr(4):
            xs[b, 2 * i] = _bf16x2_lo(u[i])
            xs[b, 2 * i + 1] = _bf16x2_hi(u[i])
    s = Float32(0.0)
    for b in cutlass.range_constexpr(NB):
        for v in cutlass.range_constexpr(8):
            s = s + xs[b, v] * xs[b, v]
    for i in cutlass.range_constexpr(5):
        s = s + cute.arch.shuffle_sync_bfly(s, offset=1 << i)
    mean = s / Float32(H)
    rstd = cute.math.rsqrt(mean + eps, fastmath=True)
    for b in cutlass.range_constexpr(NB):
        w = wu[b]
        packed = cute.make_rmem_tensor(4, Int32)
        for i in cutlass.range_constexpr(4):
            y0 = (xs[b, 2 * i] * rstd) * (_bf16x2_lo(w[i]) + Float32(0.0))
            y1 = (xs[b, 2 * i + 1] * rstd) * (_bf16x2_hi(w[i]) + Float32(0.0))
            packed[i] = _pack_bf16x2(y0, y1)
        _st_shared_v4(
            dst_smem + Int32((256 * b + 8 * lane) * 2), packed[0], packed[1], packed[2], packed[3]
        )


@cute.jit
def _stamp(
    enabled: cutlass.Constexpr[bool], a_ts: Int64, bid: Int32, tid: Int32, k: cutlass.Constexpr[int]
):
    """Tests only: ``%globaltimer`` of CTA ``bid`` at point ``k``."""
    if cutlass.const_expr(enabled):
        if tid == 0:
            _st_global_b64(a_ts + Int64((bid * 8 + k) * 8), cute.arch.globaltimer())


@cute.jit
def _mma_chain(
    a_row: Int32, b_row: Int32, STEPS: cutlass.Constexpr[int], DIST: cutlass.Constexpr[int]
):
    """``sum_s A_s B_s`` over ``STEPS`` ascending 16-wide K steps of one
    m16n8 tile (A via ldmatrix.x4 at ``a_row + 32 s``, B via two 32-bit
    loads at ``b_row + 32 s`` and ``+ 16``), with the fragments of step
    ``s + DIST`` requested before the MMA of step ``s``."""
    frags = []
    for s in cutlass.range_constexpr(DIST):
        frags.append(
            (
                _ldsm_x4(a_row + s * 32),
                _ld_shared_b32(b_row + s * 32),
                _ld_shared_b32(b_row + s * 32 + 16),
            )
        )
    c0 = Float32(0.0)
    c1 = Float32(0.0)
    c2 = Float32(0.0)
    c3 = Float32(0.0)
    for s in cutlass.range_constexpr(STEPS):
        if cutlass.const_expr(s + DIST < STEPS):
            n = s + DIST
            frags.append(
                (
                    _ldsm_x4(a_row + n * 32),
                    _ld_shared_b32(b_row + n * 32),
                    _ld_shared_b32(b_row + n * 32 + 16),
                )
            )
        a, b0, b1 = frags[s]
        c0, c1, c2, c3 = _mma_bf16(a[0], a[1], a[2], a[3], b0, b1, c0, c1, c2, c3)
    return c0, c1, c2, c3


@cute.jit
def _copy_weights(
    sW: Int32,
    sUK: Int32,
    a_wqb: Int64,
    a_wuk: Int64,
    col0: Int32,
    uk0: Int32,
    tid: Int32,
    NT: cutlass.Constexpr[int],
):
    """``cp.async`` of a CTA's q_b weight tile (32 x 2048) and W_UK slice
    (64 x 192) into shared memory (padded row strides)."""
    for i in cutlass.range_constexpr(32 * _Q_LORA // 8 // NT):
        c = tid + NT * i
        r = c // 256
        k16 = c % 256
        _cp_async16(
            sW + r * _W_STRIDE + k16 * 16, a_wqb + Int64(((col0 + r) * _Q_LORA + k16 * 8) * 2)
        )
    for i in cutlass.range_constexpr(_ABS_COLS * _NOPE // 8 // NT):
        c = tid + NT * i
        r = c // 24
        k16 = c % 24
        _cp_async16(sUK + r * _UK_STRIDE + k16 * 16, a_wuk + Int64((uk0 + r * _NOPE + k16 * 8) * 2))


class SharedQueryClusterKernel:
    """The CuTe-DSL shared_indexer_absorbed_query kernel (see the module docstring)."""

    # Fragment prefetch distance of the MMA chains (steps).
    DIST = 4
    # Threads per CTA (one warp per normalized row).
    NT = 256

    def __init__(
        self,
        use_pdl: bool,
        early_trigger: bool = True,
        debug_qn: bool = False,
        debug_ts: bool = False,
        weights_before_wait: bool = True,
        push_async: bool = True,
    ):
        self.use_pdl = use_pdl
        self.early_trigger = early_trigger
        self.debug_qn = debug_qn
        self.debug_ts = debug_ts
        self.weights_before_wait = weights_before_wait
        # True: q_nope tiles are pushed with st.async and every CTA waits on
        # its own mbarrier for the bytes it receives; False: plain
        # st.shared::cluster stores and one cluster barrier.
        self.push_async = push_async

    @cute.jit
    def __call__(
        self,
        kva: cute.Tensor,
        gq: cute.Tensor,
        gkv: cute.Tensor,
        wqb: cute.Tensor,
        wuk: cute.Tensor,
        q_out: cute.Tensor,
        latent: cute.Tensor,
        fq: cute.Tensor,
        qn_dbg: cute.Tensor,
        ts_dbg: cute.Tensor,
        rows: Int32,
        eps_q: Float32,
        eps_kv: Float32,
        stream,
    ):
        self.kernel(
            kva, gq, gkv, wqb, wuk, q_out, latent, fq, qn_dbg, ts_dbg, rows, eps_q, eps_kv
        ).launch(
            grid=[(_H + 1) * 8, 1, 1],
            block=[self.NT, 1, 1],
            cluster=[8, 1, 1],
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        kva: cute.Tensor,
        gq: cute.Tensor,
        gkv: cute.Tensor,
        wqb: cute.Tensor,
        wuk: cute.Tensor,
        q_out: cute.Tensor,
        latent: cute.Tensor,
        fq: cute.Tensor,
        qn_dbg: cute.Tensor,
        ts_dbg: cute.Tensor,
        rows: Int32,
        eps_q: Float32,
        eps_kv: Float32,
    ):
        """Tests: ``debug_ts`` writes ``%globaltimer`` stamps per CTA into
        ``ts_dbg [72, 8]`` (int64): 0 entry, 1 after the grid-dependency
        wait, 2 normalized rows and weights in shared memory, 3 q_b tile
        done, 4 after the q exchange, 5 absorbed query stored, 6 exit, 7
        warp 0's normalized row in shared memory (the kv cluster stamps 0,
        1, 2 and 6)."""
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        head = bid // 8
        j = bid % 8

        smem = SmemAllocator()
        sW = _saddr(smem.allocate(32 * _W_STRIDE, byte_alignment=128))
        sUK = _saddr(smem.allocate(_ABS_COLS * _UK_STRIDE, byte_alignment=128))
        sQn = _saddr(smem.allocate(_MAX_ROWS * _W_STRIDE, byte_alignment=128))
        sStage = _saddr(smem.allocate(_MAX_ROWS * _TILE_COLS * 2, byte_alignment=128))
        sNope = _saddr(smem.allocate(_MAX_ROWS * _UK_STRIDE, byte_alignment=128))
        sGq = _saddr(smem.allocate(_Q_LORA * 2, byte_alignment=128))
        mbar = smem.allocate_array(Int64, num_elems=1, byte_alignment=8)
        sMbar = _saddr(mbar)

        a_kva = _gaddr(kva)
        a_gq = _gaddr(gq)
        a_gkv = _gaddr(gkv)
        a_wqb = _gaddr(wqb)
        a_wuk = _gaddr(wuk)
        a_q = _gaddr(q_out)
        a_lat = _gaddr(latent)
        a_fq = _gaddr(fq)
        a_qn = _gaddr(qn_dbg)
        a_ts = _gaddr(ts_dbg)
        _stamp(self.debug_ts, a_ts, bid, tid, 0)

        if head < _H:
            col0 = head * _HEAD_DIM + j * _TILE_COLS
            uk0 = (head * _KV_LORA + j * _ABS_COLS) * _NOPE
            # 1. Weights -> shared memory: they do not depend on earlier
            #    grids, so they are requested before the dependency wait.
            #    The q_a norm gain (4 KB, one 16-byte chunk per thread) is
            #    its own cp.async group and is complete and visible to all
            #    warps before the wait.
            _cp_async16(sGq + tid * 16, a_gq + Int64(tid * 16))
            _cp_async_commit()
            if cutlass.const_expr(self.weights_before_wait):
                _copy_weights(sW, sUK, a_wqb, a_wuk, col0, uk0, tid, self.NT)
                _cp_async_commit()
                _cp_async_wait_group(1)
            else:
                _cp_async_wait_group(0)
            cute.arch.sync_threads()
            if cutlass.const_expr(self.push_async):
                # Receive buffer barrier: expects the 6 no-RoPE tiles (R_max
                # rows x 32 columns of BF16 each).
                if tid == 0:
                    cute.arch.mbarrier_init(mbar, 1)
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        mbar, (_NOPE // _TILE_COLS) * _MAX_ROWS * _TILE_COLS * 2
                    )
                cute.arch.mbarrier_init_fence()
            # Every CTA of the cluster has started (and initialized its
            # barrier) before any CTA writes to a peer's shared memory.
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()
            if cutlass.const_expr(self.use_pdl):
                cute.arch.griddepcontrol_wait()
                if cutlass.const_expr(self.early_trigger):
                    cute.arch.griddepcontrol_launch_dependents()
            if cutlass.const_expr(not self.weights_before_wait):
                _copy_weights(sW, sUK, a_wqb, a_wuk, col0, uk0, tid, self.NT)
            _stamp(self.debug_ts, a_ts, bid, tid, 1)

            # 2. Normalized q_a rows -> shared memory, one warp per row;
            #    rows >= R are zero.
            r = warp
            if r < rows:
                _norm_row_to_smem(
                    a_kva + Int64(r * _KVA_COLS * 2),
                    a_gq,
                    sGq,
                    sQn + r * _W_STRIDE,
                    lane,
                    eps_q,
                    8,
                    _Q_LORA,
                    True,
                )
            else:
                for b in cutlass.range_constexpr(8):
                    _st_shared_v4(
                        sQn + r * _W_STRIDE + (256 * b + 8 * lane) * 2,
                        Int32(0),
                        Int32(0),
                        Int32(0),
                        Int32(0),
                    )
            _stamp(self.debug_ts, a_ts, bid, tid, 7)
            _cp_async_commit_wait_all()
            cute.arch.sync_threads()
            _stamp(self.debug_ts, a_ts, bid, tid, 2)
            if cutlass.const_expr(self.debug_qn):
                # Tests: the normalized rows of CTA 0 -> qn_dbg [R, 2048].
                if bid == 0:
                    for i in cutlass.range_constexpr(_MAX_ROWS * 256 // self.NT):
                        c = tid + self.NT * i
                        rr = c // 256
                        k16 = c % 256
                        if rr < rows:
                            u = _ld_shared_v4(sQn + rr * _W_STRIDE + k16 * 16)
                            _st_global_v4(
                                a_qn + Int64(rr * _Q_LORA * 2 + k16 * 16), u[0], u[1], u[2], u[3]
                            )

            # 3. q_b tile: warps 0 and 1, 16 weight rows each, K ascending.
            g = lane // 4
            t4 = lane % 4
            if warp < 2:
                acc = _mma_chain(
                    sW + (16 * warp + lane % 16) * _W_STRIDE + (lane // 16) * 16,
                    sQn + g * _W_STRIDE + t4 * 4,
                    _Q_LORA // 16,
                    self.DIST,
                )
                _stamp(self.debug_ts, a_ts, bid, tid, 3)
                # C[m][n]: m = weight row (16 warp + g, +8), n = token
                # (2 t4, 2 t4 + 1); staged as bf16 [token, 32 columns].
                for hi in cutlass.range_constexpr(2):
                    m = 16 * warp + g + 8 * hi
                    for e in cutlass.range_constexpr(2):
                        n = 2 * t4 + e
                        _st_shared_b16(
                            sStage + (n * _TILE_COLS + m) * 2, _bf16_bits(acc[2 * hi + e])
                        )
            cute.arch.sync_threads()

            # 4. No-RoPE tiles -> every cluster peer's q_nope buffer
            #    (32 chunks of 16 B to 8 peers: one per thread), the cluster
            #    barrier, then the q tile to global memory (after the
            #    barrier, so that its release does not wait for them).
            if j < _NOPE // _TILE_COLS:
                peer = tid // 32
                ch = tid % 32
                n = ch // 4
                part = ch % 4
                u = _ld_shared_v4(sStage + n * _TILE_COLS * 2 + part * 16)
                dst = sNope + n * _UK_STRIDE + (j * _TILE_COLS) * 2 + part * 16
                if cutlass.const_expr(self.push_async):
                    _st_async_cluster_v4(dst, sMbar, peer, u[0], u[1], u[2], u[3])
                else:
                    _st_cluster_v4(dst, peer, u[0], u[1], u[2], u[3])
            if cutlass.const_expr(self.push_async):
                _mbar_wait_cluster(sMbar, Int32(0))
            else:
                _cluster_sync()
            _stamp(self.debug_ts, a_ts, bid, tid, 4)
            if warp >= 4:
                if tid < 128 + _MAX_ROWS * 4:
                    n = (tid - 128) // 4
                    part = tid % 4
                    if n < rows:
                        u = _ld_shared_v4(sStage + n * _TILE_COLS * 2 + part * 16)
                        _st_global_v4(
                            a_q + Int64((n * _H * _HEAD_DIM + col0) * 2 + part * 16),
                            u[0],
                            u[1],
                            u[2],
                            u[3],
                        )

            # 5. Absorption: 4 warps x 16 latent columns, d ascending.
            if warp < 4:
                acc2 = _mma_chain(
                    sUK + (16 * warp + lane % 16) * _UK_STRIDE + (lane // 16) * 16,
                    sNope + g * _UK_STRIDE + t4 * 4,
                    _NOPE // 16,
                    self.DIST,
                )
                for hi in cutlass.range_constexpr(2):
                    cl = j * _ABS_COLS + 16 * warp + g + 8 * hi
                    for e in cutlass.range_constexpr(2):
                        n = 2 * t4 + e
                        if n < rows:
                            _st_global_b16(
                                a_fq + Int64(((n * _H + head) * (_KV_LORA + _ROPE) + cl) * 2),
                                _bf16_bits(acc2[2 * hi + e]),
                            )
            _stamp(self.debug_ts, a_ts, bid, tid, 5)
        else:
            # kv_a RMSNorm -> latent[:, :512]; k_pe -> latent[:, 512:].
            if cutlass.const_expr(self.use_pdl):
                cute.arch.griddepcontrol_wait()
                if cutlass.const_expr(self.early_trigger):
                    cute.arch.griddepcontrol_launch_dependents()
            _stamp(self.debug_ts, a_ts, bid, tid, 1)
            r = j
            if r < rows:
                if warp == 0:
                    _norm_row_to_smem(
                        a_kva + Int64((r * _KVA_COLS + _Q_LORA) * 2),
                        a_gkv,
                        Int32(0),
                        sQn,
                        lane,
                        eps_kv,
                        2,
                        _KV_LORA,
                        False,
                    )
                cute.arch.sync_threads()
                _stamp(self.debug_ts, a_ts, bid, tid, 2)
                if tid < 64:
                    u = _ld_shared_v4(sQn + tid * 16)
                    _st_global_v4(
                        a_lat + Int64((r * (_KV_LORA + _ROPE) * 2) + tid * 16),
                        u[0],
                        u[1],
                        u[2],
                        u[3],
                    )
                elif tid < 72:
                    k = tid - 64
                    u = _ld_global_v4(
                        a_kva + Int64((r * _KVA_COLS + _Q_LORA + _KV_LORA) * 2 + k * 16)
                    )
                    _st_global_v4(
                        a_lat + Int64((r * (_KV_LORA + _ROPE) + _KV_LORA) * 2 + k * 16),
                        u[0],
                        u[1],
                        u[2],
                        u[3],
                    )
        _stamp(self.debug_ts, a_ts, bid, tid, 6)


_COMPILED = {}


def _fake_tensors():
    sym_r = cute.sym_int()
    bf16 = cutlass.BFloat16
    mk = cute.runtime.make_fake_compact_tensor
    return [
        mk(bf16, (sym_r, _KVA_COLS), stride_order=(1, 0)),
        mk(bf16, (_Q_LORA,), stride_order=(0,)),
        mk(bf16, (_KV_LORA,), stride_order=(0,)),
        mk(bf16, (_H * _HEAD_DIM, _Q_LORA), stride_order=(1, 0)),
        mk(bf16, (_H, _KV_LORA, _NOPE), stride_order=(2, 1, 0)),
        mk(bf16, (sym_r, _H * _HEAD_DIM), stride_order=(1, 0)),
        mk(bf16, (sym_r, _KV_LORA + _ROPE), stride_order=(1, 0)),
        mk(bf16, (sym_r, _H, _KV_LORA + _ROPE), stride_order=(2, 1, 0)),
        mk(bf16, (cute.sym_int(), _Q_LORA), stride_order=(1, 0)),
        mk(cutlass.Int64, (cute.sym_int(), 8), stride_order=(1, 0)),
    ]


def compiled_kernel(
    use_pdl: bool,
    early_trigger: bool = True,
    debug_qn: bool = False,
    debug_ts: bool = False,
    weights_before_wait: bool = True,
    push_async: bool = True,
):
    """Compile (once per process and flag set) and return the kernel. The
    compile allocates, so the first call must not be inside a CUDA-graph
    capture (the runner's eager warm-up forwards come first)."""
    key = (use_pdl, early_trigger, debug_qn, debug_ts, weights_before_wait, push_async)
    if key not in _COMPILED:
        kernel = SharedQueryClusterKernel(
            use_pdl, early_trigger, debug_qn, debug_ts, weights_before_wait, push_async
        )
        fakes = _fake_tensors()
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        _COMPILED[key] = cute.compile(
            kernel,
            *fakes,
            Int32(1),
            Float32(1e-5),
            Float32(1e-5),
            stream,
            options="--enable-tvm-ffi",
        )
    return _COMPILED[key]


def launch(
    mla,
    kva: torch.Tensor,
    use_pdl: bool,
    early_trigger: bool = True,
    qn_dbg: torch.Tensor = None,
    ts_dbg: torch.Tensor = None,
    weights_before_wait: bool = True,
    push_async: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the cluster kernel on a native kv_a output ``[R <= 8, 2624]``;
    with ``qn_dbg [R, 2048]`` the normalized q_a rows are also written,
    with ``ts_dbg [72, 8]`` (int64) the per-CTA timer stamps."""
    rows = kva.shape[0]
    assert 1 <= rows <= _MAX_ROWS and kva.is_contiguous()
    q = kva.new_empty(rows, _H * _HEAD_DIM)
    latent = kva.new_empty(rows, _KV_LORA + _ROPE)
    fused_q = kva.new_empty(rows, _H, _KV_LORA + _ROPE)
    debug_qn = qn_dbg is not None
    debug_ts = ts_dbg is not None
    if qn_dbg is None:
        qn_dbg = kva.new_empty(1, _Q_LORA)
    if ts_dbg is None:
        ts_dbg = torch.empty(1, 8, dtype=torch.int64, device=kva.device)
    compiled_kernel(use_pdl, early_trigger, debug_qn, debug_ts, weights_before_wait, push_async)(
        kva,
        mla.q_a_layernorm.weight,
        mla.kv_a_layernorm.weight,
        mla.q_b_proj.weight,
        mla.k_b_proj_trans,
        q,
        latent,
        fused_q,
        qn_dbg,
        ts_dbg,
        rows,
        mla.q_a_layernorm.variance_epsilon,
        mla.kv_a_layernorm.variance_epsilon,
    )
    return q, latent, fused_q
