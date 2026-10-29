# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused Kimi-K3 MoE routing for decode batches (SM100/SM103/SM107).

One launch replaces ``dsv3_router_gemm_op`` -> ``noaux_tc_op`` (-> ``moe_sort``):

* ``logits[M, 896] = h[M, 7168] @ W[896, 7168]^T`` with FP32 accumulation;
* top-16 experts per token by ``sigmoid(logit) + bias``, weights ``sigmoid(logit)``
  renormalized to sum 1 and scaled by ``routed_scaling_factor``, bit-identical to
  ``noaux_tc_op`` applied to the same logits (selection, order and weights);
* optionally the six ``moe_sort`` outputs of one expert-parallel rank.

Design (``1 <= M <= 256``):

* GEMM, ``M <= 64``. The grid has ``8 * S`` CTAs (``S = num_SMs // 8``, one CTA per SM).
  CTA ``(g, s)`` owns 112 experts (seven MMA warps x 16 rows) and the ``s``-th slice of K in
  64-wide chunks. One thread TMA-loads the CTA's whole weight slice (<= 7 x 14 KB,
  128B-swizzled) before ``griddepcontrol.wait``, so the weight stream can overlap the previous
  kernel, then the matching activation chunks. Warps run ``mma.sync`` m16n8k16 (experts on the
  MMA M side, tokens on N).
* GEMM, ``64 < M <= 256``. ``7 * S`` CTAs (``S = num_SMs // 7``) of 128 experts each run
  ``tcgen05`` UMMA (M = 128 experts, N = padded tokens, accumulator in tensor memory) over
  32-wide K chunks: the weight is again prefetched whole before ``griddepcontrol.wait``, the
  activation chunks stream through a TMA ring, and four warps copy the accumulator out.
* Either way the FP32 partial logits go into sentinel-initialized slots ``[S, 256, 896]`` of a
  persistent workspace.
* Roles. When its GEMM is done, each CTA takes a ticket from an arrival counter. The last
  ``T = min(M, CTAs - 1)`` CTAs become top-k workers (token ``t`` goes to worker ``t % T``);
  all others exit after storing their partials. A worker waits on at most ``T - 1`` CTAs that
  have not arrived yet, and every CTA that arrived earlier has exited or is a worker, so
  progress needs only ``T + 1`` co-resident CTAs (no grid-wide co-residency assumption). The
  last CTA to arrive resets the counter.
* Top-k. A worker polls the ``S`` partial slots of its token until none holds the sentinel
  (the data is the signal: no fences or flags on the critical path), sums them in split order
  and forms ``noaux_tc_op``'s packed 64-bit keys (order-preserving score bits, then ``65535 -
  expert``). Each warp sorts its 112 keys with register bitonic networks into its top 16; the
  global top 16 is then placed by a rank merge (binary searches over the eight sorted lists).
  Keys are unique, so this is exactly ``noaux_tc_op``'s selection. The weights use FP32
  division, which equals the reference's FP64 expression whenever the scale is 1 and the sum
  is >= 2^-12, and the FP64 expression otherwise.
* Sort (optional). Workers also publish their expert ids into sentinel slots; worker 0 polls
  them and builds ``moe_sort``'s metadata from per-expert token bitmasks: ``ceil(count / 128)``
  tiles per active local expert in local-expert order, rows in ascending token order (what
  ``moe_sort``'s single-CTA kernels produce; its cluster kernel orders rows by atomics).
* Re-arm. Consumers refill the slots they read with the sentinel (TMA bulk copies from a
  shared-memory sentinel block), so every launch leaves the workspace as it found it and a
  captured CUDA graph can be replayed. Launches sharing a workspace must not overlap, which
  stream order guarantees.

The module also provides :func:`k3_bf16_lowm_gemv`, the ``M <= 64`` GEMM phase as a standalone
BF16 GEMV with a deterministic last-arriver split-K reduction.
"""

from __future__ import annotations

import functools
import struct

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass import Float32, Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, tcgen05
from cutlass.cute.runtime import make_ptr
from cutlass.cutlass_dsl import T, dsl_user_op

#: Kimi-K3 routed experts.
NUM_EXPERTS = 896
#: Kimi-K3 hidden size.
HIDDEN_SIZE = 7168
#: Experts selected per token.
TOP_K = 16
#: Largest token count served by the router.
MAX_M = 256
#: Largest token count of the ``mma.sync`` GEMM path; larger M use ``tcgen05`` UMMA.
SMALL_M = 64
#: Expert-id hand-off slots (sentinel-armed), first in the workspace.
_IDS_SLOT_BYTES = MAX_M * TOP_K * 4

_K_CHUNK = 64
_NUM_K_CHUNKS = HIDDEN_SIZE // _K_CHUNK
_EXPERT_GROUPS = 8
_EXPERTS_PER_CTA = NUM_EXPERTS // _EXPERT_GROUPS
_M_TILES = _EXPERTS_PER_CTA // 16
_THREADS = 256
_W_CHUNK_BYTES = _EXPERTS_PER_CTA * _K_CHUNK * 2
# Large-M (tcgen05) GEMM: 7 CTAs of 128 experts (UMMA M) per K slice, 32-wide K chunks.
_UMMA_GROUPS = 7
_UMMA_ROWS = NUM_EXPERTS // _UMMA_GROUPS
_UMMA_K_CHUNK = 32
_UMMA_NUM_K_CHUNKS = HIDDEN_SIZE // _UMMA_K_CHUNK
_UMMA_A_CHUNK_BYTES = _UMMA_ROWS * _UMMA_K_CHUNK * 2
_SMEM_CAPACITY = utils.get_smem_capacity_in_bytes("sm_100")
#: L2 eviction hint for the streamed weight (CacheHintSm90::EVICT_FIRST).
_EVICT_FIRST = 0x12F0000000000000
#: ``1e-20`` as a PTX double literal (the renormalization epsilon of ``noaux_tc_op``).
_EPS_F64 = "0d" + struct.pack(">d", 1e-20).hex().upper()

_ROW_BYTES = NUM_EXPERTS * 4

# Shared-memory scratch of the top-k / sort phases (aliases the then-dead weight ring).
_SMEM_SIG = 0  # fp32 [896] sigmoid scores
_SMEM_CAND = _SMEM_SIG + NUM_EXPERTS * 4  # u64 [8][16] per-warp sorted top 16
_SMEM_TOP = _SMEM_CAND + _EXPERT_GROUPS * TOP_K * 8  # u64 [16] final top 16
_SMEM_MASK = 0  # sort: u32 [MAX_M / 32][896] token bitmask word per local expert
# sort: i32 [MAX_M / 32][896] first permuted row of the tokens of each mask word
_SMEM_WORD_BASE = _SMEM_MASK + (MAX_M // 32) * NUM_EXPERTS * 4
_SMEM_WARP_TOT = _SMEM_WORD_BASE + (MAX_M // 32) * NUM_EXPERTS * 4  # sort: per-warp tile counts
_SMEM_SENTINEL = 65536  # 4 KB of sentinel bytes, the source of the re-arm bulk copies
_SENTINEL_BYTES = 4096
#: Shared-memory scratch the tail phases need from the GEMM's weight ring.
_SCRATCH_BYTES = _SMEM_SENTINEL + _SENTINEL_BYTES

# Shared-memory cache of the scalar kernel parameters (slot order).
(
    _P_M,
    _P_BIAS,
    _P_WS,
    _P_IDS,
    _P_VALS,
    _P_LOGITS,
    _P_WRITE_LOGITS,
    _P_SCALE,
    _P_E2P,
    _P_P2E,
    _P_TILE_EXPERT,
    _P_TILE_LIMIT,
    _P_TOTAL,
    _P_NUM_TILES,
    _P_LOCAL_OFFSET,
    _P_LOCAL_NUM,
) = range(16)
_NUM_PARAMS = 16

# Worker layout: warp w handles 112 experts (28 lanes x 4); the last warp is free for arrival.
assert _EXPERTS_PER_CTA % 4 == 0 and _EXPERTS_PER_CTA // 4 <= 32
assert _EXPERT_GROUPS == _THREADS // 32 and _M_TILES < _THREADS // 32
assert _SMEM_TOP + TOP_K * 8 <= _SMEM_SENTINEL and _SMEM_WARP_TOT + 64 <= _SMEM_SENTINEL
assert _SENTINEL_BYTES >= _ROW_BYTES and (MAX_M * TOP_K * 4) % _SENTINEL_BYTES == 0


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _slot_bytes(num_splits: int, rows: int) -> int:
    return num_splits * rows * _ROW_BYTES


def workspace_bytes(small_splits: int, umma_splits: int) -> int:
    """Hand-off workspace: ``int32 [256 * 16]`` expert-id slots, the ``M <= 64`` path's
    ``fp32 [S, 64, 896]`` partial-logit slots, then the large-M path's ``fp32 [S, 256, 896]``
    ones (all sentinel-armed)."""
    return _IDS_SLOT_BYTES + _slot_bytes(small_splits, SMALL_M) + _slot_bytes(umma_splits, MAX_M)


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
def _ldsm_x4(addr, *, loc=None, ip=None):
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [_iv(addr, loc=loc, ip=ip)],
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        loc=loc,
        ip=ip,
    )
    return _unpack(res, T.i32(), Int32, 4, loc=loc, ip=ip)


@dsl_user_op
def _mma_bf16(a, b0, b1, c, *, loc=None, ip=None):
    """``d[16x8] = a[16x16] @ b[16x8] + c`` (BF16 inputs, FP32 accumulate)."""
    res = _asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        [_iv(x, loc=loc, ip=ip) for x in (*a, b0, b1)] + [_fv(x, loc=loc, ip=ip) for x in c],
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, {$10, $11, $12, $13};",
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f",
        loc=loc,
        ip=ip,
    )
    return _unpack(res, T.f32(), Float32, 4, loc=loc, ip=ip)


@dsl_user_op
def _st_global_f32(addr, value, *, loc=None, ip=None):
    _asm(
        None,
        [_lv(addr, loc=loc, ip=ip), _fv(value, loc=loc, ip=ip)],
        "st.global.f32 [$0], $1;",
        "l,f",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _st_global_v4_f32(addr, v0, v1, v2, v3, *, loc=None, ip=None):
    _asm(
        None,
        [_lv(addr, loc=loc, ip=ip)] + [_fv(v, loc=loc, ip=ip) for v in (v0, v1, v2, v3)],
        "st.global.v4.f32 [$0], {$1, $2, $3, $4};",
        "l,f,f,f,f",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _st_global_b32(addr, value, *, loc=None, ip=None):
    _asm(
        None,
        [_lv(addr, loc=loc, ip=ip), _iv(value, loc=loc, ip=ip)],
        "st.global.b32 [$0], $1;",
        "l,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ld_nc_v4_f32(addr, *, loc=None, ip=None):
    res = _asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        [_lv(addr, loc=loc, ip=ip)],
        "ld.global.nc.v4.f32 {$0, $1, $2, $3}, [$4];",
        "=f,=f,=f,=f,l",
        loc=loc,
        ip=ip,
    )
    return _unpack(res, T.f32(), Float32, 4, loc=loc, ip=ip)


@dsl_user_op
def _atom_add_relaxed_gpu(addr, value, *, loc=None, ip=None):
    """Relaxed atomic add; callers order it with explicit fences."""
    return Int32(
        _asm(
            T.i32(),
            [_lv(addr, loc=loc, ip=ip), _iv(value, loc=loc, ip=ip)],
            "atom.relaxed.gpu.global.add.u32 $0, [$1], $2;",
            "=r,l,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _st_relaxed_gpu(addr, value, *, loc=None, ip=None):
    _asm(
        None,
        [_lv(addr, loc=loc, ip=ip), _iv(value, loc=loc, ip=ip)],
        "st.relaxed.gpu.global.b32 [$0], $1;",
        "l,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _st_relaxed_f32(addr, value, *, loc=None, ip=None):
    _asm(
        None,
        [_lv(addr, loc=loc, ip=ip), _fv(value, loc=loc, ip=ip)],
        "st.relaxed.gpu.global.f32 [$0], $1;",
        "l,f",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _st_relaxed_b32(addr, value, *, loc=None, ip=None):
    _asm(
        None,
        [_lv(addr, loc=loc, ip=ip), _iv(value, loc=loc, ip=ip)],
        "st.relaxed.gpu.global.b32 [$0], $1;",
        "l,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ld_relaxed_v4_f32(addr, *, loc=None, ip=None):
    res = _asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        [_lv(addr, loc=loc, ip=ip)],
        "ld.relaxed.gpu.global.v4.f32 {$0, $1, $2, $3}, [$4];",
        "=f,=f,=f,=f,l",
        loc=loc,
        ip=ip,
    )
    return _unpack(res, T.f32(), Float32, 4, loc=loc, ip=ip)


@dsl_user_op
def _ld_relaxed_b32(addr, *, loc=None, ip=None):
    return Int32(
        _asm(
            T.i32(),
            [_lv(addr, loc=loc, ip=ip)],
            "ld.relaxed.gpu.global.b32 $0, [$1];",
            "=r,l",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _all_valid4(v0, v1, v2, v3, *, loc=None, ip=None):
    """1 if none of the four FP32 words is the empty-slot sentinel ``0xffffffff``."""
    return Int32(
        _asm(
            T.i32(),
            [_fv(v, loc=loc, ip=ip) for v in (v0, v1, v2, v3)],
            "{\n.reg .pred p, q;\n.reg .b32 t;\n"
            "mov.b32 t, $1;\nsetp.ne.b32 p, t, -1;\n"
            "mov.b32 t, $2;\nsetp.ne.and.b32 p, t, -1, p;\n"
            "mov.b32 t, $3;\nsetp.ne.and.b32 p, t, -1, p;\n"
            "mov.b32 t, $4;\nsetp.ne.and.b32 p, t, -1, p;\n"
            "selp.u32 $0, 1, 0, p;\n}",
            "=r,f,f,f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _bulk_s2g(dst, src, nbytes, *, loc=None, ip=None):
    """Bulk copy ``nbytes`` (multiple of 16) from shared ``src`` to global ``dst``."""
    _asm(
        None,
        [_lv(dst, loc=loc, ip=ip), _iv(src, loc=loc, ip=ip), _iv(nbytes, loc=loc, ip=ip)],
        "cp.async.bulk.global.shared::cta.bulk_group [$0], [$1], $2;",
        "l,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _bulk_commit_and_wait_read(*, loc=None, ip=None):
    """Commit the issued bulk copies and wait until they have read their shared sources."""
    _asm(
        None,
        [],
        "cp.async.bulk.commit_group;\ncp.async.bulk.wait_group.read 0;",
        "",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _fence_proxy_async_shared(*, loc=None, ip=None):
    _asm(None, [], "fence.proxy.async.shared::cta;", "", loc=loc, ip=ip)


@dsl_user_op
def _sts_v4_sentinel(addr, *, loc=None, ip=None):
    _asm(
        None,
        [_iv(addr, loc=loc, ip=ip)],
        "{\n.reg .b32 t;\nmov.b32 t, -1;\nst.shared.v4.b32 [$0], {t, t, t, t};\n}",
        "r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _griddepcontrol_wait(*, loc=None, ip=None):
    _asm(None, [], "griddepcontrol.wait;", "", loc=loc, ip=ip)


@dsl_user_op
def _griddepcontrol_launch_dependents(*, loc=None, ip=None):
    _asm(None, [], "griddepcontrol.launch_dependents;", "", loc=loc, ip=ip)


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
def _sts_f32(addr, value, *, loc=None, ip=None):
    _asm(
        None,
        [_iv(addr, loc=loc, ip=ip), _fv(value, loc=loc, ip=ip)],
        "st.shared.f32 [$0], $1;",
        "r,f",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _lds_f32(addr, *, loc=None, ip=None):
    return Float32(
        _asm(
            T.f32(), [_iv(addr, loc=loc, ip=ip)], "ld.shared.f32 $0, [$1];", "=f,r", loc=loc, ip=ip
        )
    )


@dsl_user_op
def _sts_v4_f32(addr, v0, v1, v2, v3, *, loc=None, ip=None):
    _asm(
        None,
        [_iv(addr, loc=loc, ip=ip)] + [_fv(v, loc=loc, ip=ip) for v in (v0, v1, v2, v3)],
        "st.shared.v4.f32 [$0], {$1, $2, $3, $4};",
        "r,f,f,f,f",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _lds_v4_b32(addr, *, loc=None, ip=None):
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [_iv(addr, loc=loc, ip=ip)],
        "ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        loc=loc,
        ip=ip,
    )
    return _unpack(res, T.i32(), Int32, 4, loc=loc, ip=ip)


@dsl_user_op
def _keep_key(key, keep, *, loc=None, ip=None):
    """``key`` if ``keep != 0`` else 0 (below every real key)."""
    return Int64(
        _asm(
            T.i64(),
            [_lv(key, loc=loc, ip=ip), _iv(keep, loc=loc, ip=ip)],
            "{\n.reg .pred p;\n.reg .b64 z;\nmov.b64 z, 0;\nsetp.ne.u32 p, $2, 0;\n"
            "selp.b64 $0, $1, z, p;\n}",
            "=l,l,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _nonzero(value, *, loc=None, ip=None):
    return Int32(
        _asm(
            T.i32(),
            [_iv(value, loc=loc, ip=ip)],
            "{\n.reg .pred p;\nsetp.ne.u32 p, $1, 0;\nselp.u32 $0, 1, 0, p;\n}",
            "=r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _is_neg(value, *, loc=None, ip=None):
    return Int32(
        _asm(
            T.i32(),
            [_iv(value, loc=loc, ip=ip)],
            "shr.u32 $0, $1, 31;",
            "=r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _below_mask(a, b, *, loc=None, ip=None):
    """All ones if ``a < b`` (signed) else 0."""
    return Int32(
        _asm(
            T.i32(),
            [_iv(a, loc=loc, ip=ip), _iv(b, loc=loc, ip=ip)],
            "{\n.reg .pred p;\nsetp.lt.s32 p, $1, $2;\nselp.b32 $0, -1, 0, p;\n}",
            "=r,r,r",
            loc=loc,
            ip=ip,
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
def _lds_b64_if(lane, limit: int, addr, *, loc=None, ip=None):
    """``*addr`` if ``lane < limit`` else 0."""
    return Int64(
        _asm(
            T.i64(),
            [_iv(lane, loc=loc, ip=ip), _iv(addr, loc=loc, ip=ip)],
            "{\n.reg .pred p;\n"
            f"setp.lt.u32 p, $1, {limit};\n"
            "mov.b64 $0, 0;\n@p ld.shared.b64 $0, [$2];\n}",
            "=l,r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _red_shared_or(addr, value, *, loc=None, ip=None):
    _asm(
        None,
        [_iv(addr, loc=loc, ip=ip), _iv(value, loc=loc, ip=ip)],
        "red.shared.or.b32 [$0], $1;",
        "r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _make_key(score, expert, *, loc=None, ip=None):
    """Packed ``noaux_tc`` selection key: order-preserving FP32 bits, then ``65535 - expert``.

    Larger keys win; equal scores fall back to the smaller expert index.
    """
    return Int64(
        _asm(
            T.i64(),
            [_fv(score, loc=loc, ip=ip), _iv(expert, loc=loc, ip=ip)],
            "{\n"
            ".reg .b32 b, m, hi, lo;\n"
            "mov.b32 b, $1;\n"
            "shr.s32 m, b, 31;\n"
            "or.b32 m, m, 0x80000000;\n"
            "xor.b32 hi, b, m;\n"
            "mov.u32 lo, 65535;\n"
            "sub.u32 lo, lo, $2;\n"
            "and.b32 lo, lo, 0xffff;\n"
            "mov.b64 $0, {lo, hi};\n"
            "}",
            "=l,f,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _key_expert(key, *, loc=None, ip=None):
    return Int32(
        _asm(
            T.i32(),
            [_lv(key, loc=loc, ip=ip)],
            "{\n"
            ".reg .b32 lo, hi;\n"
            "mov.b64 {lo, hi}, $1;\n"
            "and.b32 lo, lo, 0xffff;\n"
            "sub.u32 $0, 65535, lo;\n"
            "}",
            "=r,l",
            loc=loc,
            ip=ip,
        )
    )


def _shfl_b64(src: str, dst: str, lane_xor: int) -> str:
    """PTX: ``dst = src`` of lane ``lane ^ lane_xor`` (64-bit, two 32-bit shuffles)."""
    return (
        f"mov.b64 {{lo, hi}}, {src};\n"
        f"shfl.sync.bfly.b32 lo, lo, {lane_xor}, 0x1f, 0xffffffff;\n"
        f"shfl.sync.bfly.b32 hi, hi, {lane_xor}, 0x1f, 0xffffffff;\n"
        f"mov.b64 {dst}, {{lo, hi}};\n"
    )


@dsl_user_op
def _bitonic_shfl4(xs, lane_xor: int, keep_max, *, loc=None, ip=None):
    """Bitonic compare-exchange of each of four key slots with lane ``lane ^ lane_xor``.

    Lanes with ``keep_max != 0`` keep the larger key, the others the smaller one.
    """
    body = "{\n.reg .b32 lo, hi;\n.reg .b64 o;\n.reg .pred km, p;\nsetp.eq.u32 km, $8, 0;\n"
    for i in range(4):
        body += _shfl_b64(f"${4 + i}", "o", lane_xor)
        # Keep the own key iff (own > other) == keep_max.
        body += f"setp.gt.u64 p, ${4 + i}, o;\nxor.pred p, p, km;\nselp.b64 ${i}, ${4 + i}, o, p;\n"
    res = _asm(
        llvm.StructType.get_literal([T.i64()] * 4),
        [_lv(x, loc=loc, ip=ip) for x in xs] + [_iv(keep_max, loc=loc, ip=ip)],
        body + "}",
        "=l,=l,=l,=l,l,l,l,l,r",
        loc=loc,
        ip=ip,
    )
    return list(_unpack(res, T.i64(), Int64, 4, loc=loc, ip=ip))


@dsl_user_op
def _merge_rev4(xs, lane_xor: int, *, loc=None, ip=None):
    """``x[s] = max(x[s], x'[3 - s])`` with ``x'`` the slots of lane ``lane ^ lane_xor``.

    With ``lane_xor = G | 3`` this pairs element ``i`` of a sorted 16-key group (lanes
    ``4g..4g+3`` x four slots, element ``(lane & 3) * 4 + slot``) with element ``15 - i`` of
    the group ``G / 4`` groups away: the result holds the top 16 of both as a bitonic sequence.
    """
    body = "{\n.reg .b32 lo, hi;\n.reg .b64 o;\n.reg .pred p;\n"
    for i in range(4):
        body += _shfl_b64(f"${4 + 3 - i}", "o", lane_xor)
        body += f"setp.gt.u64 p, ${4 + i}, o;\nselp.b64 ${i}, ${4 + i}, o, p;\n"
    res = _asm(
        llvm.StructType.get_literal([T.i64()] * 4),
        [_lv(x, loc=loc, ip=ip) for x in xs],
        body + "}",
        "=l,=l,=l,=l,l,l,l,l",
        loc=loc,
        ip=ip,
    )
    return list(_unpack(res, T.i64(), Int64, 4, loc=loc, ip=ip))


@dsl_user_op
def _count_greater_sorted(base, key, *, loc=None, ip=None):
    """Number of keys larger than ``key`` in eight descending 16-key lists at ``base``."""
    lists = _EXPERT_GROUPS
    body = "{\n.reg .b64 v;\n.reg .b32 a, t;\n.reg .pred p;\n"
    body += "".join(f".reg .b32 i{j};\nmov.u32 i{j}, 0;\n" for j in range(lists))
    for step in (8, 4, 2, 1, 0):
        for j in range(lists):
            # Probe index i + step - 1 (the final round probes i itself).
            off = (step - 1) if step else 0
            body += (
                f"add.u32 t, i{j}, {off};\nshl.b32 a, t, 3;\nadd.u32 a, a, $1;\n"
                f"ld.shared.b64 v, [a+{j * TOP_K * 8}];\nsetp.gt.u64 p, v, $2;\n"
                f"@p add.u32 i{j}, i{j}, {step if step else 1};\n"
            )
    body += "mov.u32 $0, 0;\n" + "".join(f"add.u32 $0, $0, i{j};\n" for j in range(lists)) + "}"
    return Int32(
        _asm(
            T.i32(),
            [_iv(base, loc=loc, ip=ip), _lv(key, loc=loc, ip=ip)],
            body,
            "=r,r,l",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _cx(a, b, desc, *, loc=None, ip=None):
    """In-thread compare-exchange: ``(max, min)`` if ``desc != 0`` else ``(min, max)``."""
    res = _asm(
        llvm.StructType.get_literal([T.i64()] * 2),
        [_lv(a, loc=loc, ip=ip), _lv(b, loc=loc, ip=ip), _iv(desc, loc=loc, ip=ip)],
        "{\n.reg .pred d, p;\nsetp.eq.u32 d, $4, 0;\n"
        "setp.gt.u64 p, $2, $3;\nxor.pred p, p, d;\n"
        "selp.b64 $0, $2, $3, p;\nselp.b64 $1, $3, $2, p;\n}",
        "=l,=l,l,l,r",
        loc=loc,
        ip=ip,
    )
    return _unpack(res, T.i64(), Int64, 2, loc=loc, ip=ip)


def _clean16(x, nb0, nb1):
    """Sort each bitonic 16-key group descending (lane bits 1, 0 then slot bits 1, 0)."""
    x = _bitonic_shfl4(x, 2, nb1)
    x = _bitonic_shfl4(x, 1, nb0)
    one = Int32(1)
    x[0], x[2] = _cx(x[0], x[2], one)
    x[1], x[3] = _cx(x[1], x[3], one)
    x[0], x[1] = _cx(x[0], x[1], one)
    x[2], x[3] = _cx(x[2], x[3], one)
    return x


def _sort16(x, lane):
    """Bitonic sort of each 16-key group (4 lanes x 4 slots) into descending order."""
    lb0 = lane & 1
    lb1 = (lane >> 1) & 1
    nb0 = 1 - lb0
    nb1 = 1 - lb1
    one = Int32(1)
    zero = Int32(0)
    # k = 2: slot pairs (0, 1) descending, (2, 3) ascending.
    x[0], x[1] = _cx(x[0], x[1], one)
    x[2], x[3] = _cx(x[2], x[3], zero)
    # k = 4: blocks of one lane; descending iff lane bit 0 == 0.
    x[0], x[2] = _cx(x[0], x[2], nb0)
    x[1], x[3] = _cx(x[1], x[3], nb0)
    x[0], x[1] = _cx(x[0], x[1], nb0)
    x[2], x[3] = _cx(x[2], x[3], nb0)
    # k = 8: blocks of two lanes; descending iff lane bit 1 == 0.
    x = _bitonic_shfl4(x, 1, 1 - (lb0 ^ lb1))
    x[0], x[2] = _cx(x[0], x[2], nb1)
    x[1], x[3] = _cx(x[1], x[3], nb1)
    x[0], x[1] = _cx(x[0], x[1], nb1)
    x[2], x[3] = _cx(x[2], x[3], nb1)
    # k = 16: whole group descending.
    return _clean16(x, nb0, nb1)


def _top16_of_groups(x, lane):
    """Top 16 of eight sorted 16-key groups; every group ends up holding it, sorted."""
    nb0 = 1 - (lane & 1)
    nb1 = 1 - ((lane >> 1) & 1)
    for lane_xor in (4 | 3, 8 | 3, 16 | 3):
        x = _merge_rev4(x, lane_xor)
        x = _clean16(x, nb0, nb1)
    return x


@dsl_user_op
def _warp_sum_f32(value, *, loc=None, ip=None):
    """Butterfly warp sum, same order as ``cooperative_groups::reduce(plus)``."""
    body = "".join(
        f"shfl.sync.bfly.b32 t, x, {off}, 0x1f, 0xffffffff;\nadd.rn.f32 x, x, t;\n"
        for off in (16, 8, 4, 2, 1)
    )
    return Float32(
        _asm(
            T.f32(),
            [_fv(value, loc=loc, ip=ip)],
            "{\n.reg .f32 x, t;\nmov.b32 x, $1;\n" + body + "mov.b32 $0, x;\n}",
            "=f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _renormalize(score, total, scale_bits, *, loc=None, ip=None):
    """``float(double(score) * scale / (double(total) + 1e-20))`` like ``noaux_tc_op``."""
    return Float32(
        _asm(
            T.f32(),
            [
                _fv(score, loc=loc, ip=ip),
                _fv(total, loc=loc, ip=ip),
                _lv(scale_bits, loc=loc, ip=ip),
            ],
            "{\n"
            ".reg .f64 a, b, s;\n"
            "mov.b64 s, $3;\n"
            "cvt.f64.f32 a, $1;\n"
            "mul.rn.f64 a, a, s;\n"
            "cvt.f64.f32 b, $2;\n"
            f"add.rn.f64 b, b, {_EPS_F64};\n"
            "div.rn.f64 a, a, b;\n"
            "cvt.rn.f32.f64 $0, a;\n"
            "}",
            "=f,f,f,l",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _unit_scale_fast_path(scale_bits, total, *, loc=None, ip=None):
    """1 if ``scale == 1.0`` and ``total >= 2^-12`` (see :func:`_renormalize`)."""
    return Int32(
        _asm(
            T.i32(),
            [_lv(scale_bits, loc=loc, ip=ip), _fv(total, loc=loc, ip=ip)],
            "{\n.reg .pred p, q;\n"
            "setp.eq.u64 p, $1, 0x3FF0000000000000;\n"
            "setp.ge.f32 q, $2, 0f39800000;\n"
            "and.pred p, p, q;\n"
            "selp.u32 $0, 1, 0, p;\n}",
            "=r,l,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _div_rn_f32(a, b, *, loc=None, ip=None):
    return Float32(
        _asm(
            T.f32(),
            [_fv(a, loc=loc, ip=ip), _fv(b, loc=loc, ip=ip)],
            "div.rn.f32 $0, $1, $2;",
            "=f,f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _popc(value, *, loc=None, ip=None):
    return Int32(
        _asm(T.i32(), [_iv(value, loc=loc, ip=ip)], "popc.b32 $0, $1;", "=r,r", loc=loc, ip=ip)
    )


@dsl_user_op
def _tmem_ld16(taddr, *, loc=None, ip=None):
    """16 consecutive FP32 accumulator columns of this thread's TMEM lane."""
    regs = ", ".join(f"${i}" for i in range(16))
    res = _asm(
        llvm.StructType.get_literal([T.f32()] * 16),
        [_iv(taddr, loc=loc, ip=ip)],
        "tcgen05.fence::after_thread_sync;\n"
        f"tcgen05.ld.sync.aligned.32x32b.x16.b32 {{{regs}}}, [$16];\n"
        "tcgen05.wait::ld.sync.aligned;",
        ",".join(["=f"] * 16) + ",r",
        loc=loc,
        ip=ip,
    )
    return _unpack(res, T.f32(), Float32, 16, loc=loc, ip=ip)


@dsl_user_op
def _tmem_fence_before_sync(*, loc=None, ip=None):
    _asm(None, [], "tcgen05.fence::before_thread_sync;", "", loc=loc, ip=ip)


@dsl_user_op
def _fence_acq_rel_gpu(*, loc=None, ip=None):
    _asm(None, [], "fence.acq_rel.gpu;", "", loc=loc, ip=ip)


@dsl_user_op
def _ld_relaxed_f32(addr, *, loc=None, ip=None):
    return Float32(
        _asm(
            T.f32(),
            [_lv(addr, loc=loc, ip=ip)],
            "ld.relaxed.gpu.global.f32 $0, [$1];",
            "=f,l",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _st_global_bf16(addr, value, *, loc=None, ip=None):
    _asm(
        None,
        [_lv(addr, loc=loc, ip=ip), _fv(value, loc=loc, ip=ip)],
        "{\n.reg .b16 h;\ncvt.rn.bf16.f32 h, $1;\nst.global.b16 [$0], h;\n}",
        "l,f",
        loc=loc,
        ip=ip,
    )


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------


class _RouterTail:
    """Arrival, top-k worker and ``moe_sort`` phases shared by both GEMM paths.

    Subclasses set ``num_splits``, ``num_ctas``, ``max_m`` (token capacity of the path),
    ``with_sort`` and ``use_pdl`` and call :meth:`route` once the CTA's partial logits are
    stored and its ticket is in shared memory.
    """

    def _init_tail(
        self,
        num_splits: int,
        num_ctas: int,
        max_m: int,
        with_sort: bool,
        ids_scratch_offset: int,
    ):
        self.num_splits = num_splits
        self.num_ctas = num_ctas
        #: Token rows per K split in this path's partial-logit slots.
        self.slot_rows = max_m
        self.max_m = max_m
        self.with_sort = with_sort
        self.mask_words = max_m // 32
        self.max_tiles_per_expert = _ceil_div(max_m, 128)
        self.ids_scratch_offset = ids_scratch_offset

    @cute.jit
    def route(
        self,
        smem: Int32,
        params: Int32,
        ticket_addr: Int32,
        tidx: Int32,
        warp: Int32,
        lane: Int32,
        m: Int32,
    ):
        """The last ``T = min(M, CTAs - 1)`` CTAs to arrive are top-k workers (tokens dealt
        round-robin); worker 0 then also builds the ``moe_sort`` metadata."""
        ticket = _lds_b32(ticket_addr)
        num_workers = m
        if num_workers > self.num_ctas - 1:
            num_workers = Int32(self.num_ctas - 1)
        first_worker = self.num_ctas - num_workers
        if ticket >= first_worker:
            worker = ticket - first_worker
            ws_addr = _lds_b64(params + _P_WS * 8)
            self.topk_worker(
                smem,
                tidx,
                warp,
                lane,
                worker,
                num_workers,
                m,
                _lds_b64(params + _P_BIAS * 8),
                ws_addr,
                _lds_b64(params + _P_IDS * 8),
                _lds_b64(params + _P_VALS * 8),
                _lds_b64(params + _P_LOGITS * 8),
                Int32(_lds_b64(params + _P_WRITE_LOGITS * 8)),
                _lds_b64(params + _P_SCALE * 8),
            )
            if cutlass.const_expr(self.with_sort):
                if worker == 0:
                    self.sort_tokens(
                        smem,
                        tidx,
                        warp,
                        lane,
                        m,
                        ws_addr + self.ids_scratch_offset,
                        _lds_b64(params + _P_E2P * 8),
                        _lds_b64(params + _P_P2E * 8),
                        _lds_b64(params + _P_TILE_EXPERT * 8),
                        _lds_b64(params + _P_TILE_LIMIT * 8),
                        _lds_b64(params + _P_TOTAL * 8),
                        _lds_b64(params + _P_NUM_TILES * 8),
                        Int32(_lds_b64(params + _P_LOCAL_OFFSET * 8)),
                        Int32(_lds_b64(params + _P_LOCAL_NUM * 8)),
                    )
            if tidx == _THREADS - 32:
                # The bulk copies read the sentinel block; it must outlive them.
                _bulk_commit_and_wait_read()

    @cute.jit
    def topk_worker(
        self,
        smem: Int32,
        tidx: Int32,
        warp: Int32,
        lane: Int32,
        worker: Int32,
        num_workers: Int32,
        m: Int32,
        bias_addr: Int64,
        ws_addr: Int64,
        ids_addr: Int64,
        vals_addr: Int64,
        logits_addr: Int64,
        write_logits: Int32,
        scale_bits: Int64,
    ):
        """Top-16 of the tokens ``worker, worker + num_workers, ...``.

        Warp ``w`` owns experts ``[112 w, 112 w + 112)``; lane ``l < 28`` polls the ``S``
        partial logits of experts ``112 w + 4 l + {0..3}`` until no slot holds the sentinel,
        sums them in split order and keeps the packed keys in registers. Each warp sorts its
        keys in 16-key groups (4 lanes x 4 slots) and merges them into its top 16 with bitonic
        networks; a rank merge over the eight sorted lists then places the global top 16. Keys
        are unique, so this selects exactly what ``noaux_tc_op``'s arg-max rounds select, in
        the same order. The consumed slots are re-armed with bulk copies.
        """
        S = self.num_splits
        stride = self.slot_rows * _ROW_BYTES
        lanes_used = _EXPERTS_PER_CTA // 4
        loader = lane < lanes_used
        e_lane = lane
        if not loader:
            e_lane = Int32(lanes_used - 1)
        e4 = warp * _EXPERTS_PER_CTA + e_lane * 4
        bias = _ld_nc_v4_f32(bias_addr + Int64(e4 * 4))
        parts = cute.make_rmem_tensor((S * 4,), Float32)
        _sts_v4_sentinel(smem + _SMEM_SENTINEL + tidx * 16)
        _fence_proxy_async_shared()
        # Consumed slots are re-armed right after level 1 when only top-k is computed (the
        # copies then finish with the kernel). With the sort, they are re-armed after the
        # output: their traffic would otherwise slow the sort, which runs last (measured on
        # GB300; a full TMA queue also stalls the issuing warp and every barrier behind it).
        rearm_early = not self.with_sort

        # Workers get several tokens only when M exceeds the worker count (large-M path); there
        # each token's first polling round is issued ahead, during the previous token.
        prefetch = self.slot_rows > SMALL_M
        if cutlass.const_expr(prefetch):
            if loader:
                self.load_rows(parts, ws_addr + Int64((worker * NUM_EXPERTS + e4) * 4), stride)
        for token in cutlass.range(worker, m, num_workers, unroll=1):
            # Poll the S partial rows (all loads in flight at once) until every slot is set.
            row0 = ws_addr + Int64((token * NUM_EXPERTS + e4) * 4)
            ready = Int32(1)
            if loader:
                ready = Int32(0)
                if cutlass.const_expr(prefetch):
                    ready = Int32(1)
                    for sp in cutlass.range_constexpr(S):
                        ready = ready & _all_valid4(*(parts[4 * sp + i] for i in range(4)))
            while ready == 0:
                ready = Int32(1)
                for sp in cutlass.range_constexpr(S):
                    v = _ld_relaxed_v4_f32(row0 + Int64(sp * stride))
                    for i in cutlass.range_constexpr(4):
                        parts[4 * sp + i] = v[i]
                    ready = ready & _all_valid4(*v)
            # Logits = sum of the S partials in split order; sigmoid; packed keys.
            logit = [parts[i] for i in range(4)]
            for sp in cutlass.range_constexpr(1, S):
                for i in cutlass.range_constexpr(4):
                    logit[i] = logit[i] + parts[4 * sp + i]
            sig = [
                cute.math.tanh(logit[i] * Float32(0.5)) * Float32(0.5) + Float32(0.5)
                for i in range(4)
            ]
            keys = [_keep_key(_make_key(sig[i] + bias[i], e4 + i), loader) for i in range(4)]
            if cutlass.const_expr(prefetch):
                # The next token's rows: their latency hides behind this token's top-k.
                if loader and token + num_workers < m:
                    self.load_rows(parts, row0 + Int64(num_workers * _ROW_BYTES), stride)
            if loader:
                _sts_v4_f32(smem + _SMEM_SIG + e4 * 4, *sig)
                if write_logits != 0:
                    _st_global_v4_f32(logits_addr + Int64((token * NUM_EXPERTS + e4) * 4), *logit)

            # Per-warp top 16 (group 0 = lanes 0-3 holds it, element lane * 4 + slot).
            keys = _top16_of_groups(_sort16(keys, lane), lane)
            if lane < 4:
                for i in cutlass.range_constexpr(4):
                    _sts_b64(smem + _SMEM_CAND + (warp * TOP_K + lane * 4 + i) * 8, keys[i])
            cute.arch.barrier()
            if cutlass.const_expr(rearm_early):
                # Every thread has read its partial slots.
                self.rearm_row(smem, tidx, ws_addr, stride, token)

            # Rank merge of the eight sorted lists: candidate c's global rank is the number of
            # larger keys in every list (binary searches); the 16 lanes with rank < 16 place
            # their keys, then lane k of warp 0 takes the k-th largest.
            if warp < _EXPERT_GROUPS * TOP_K // 32:
                cand = _lds_b64(smem + _SMEM_CAND + (warp * 32 + lane) * 8)
                rank = _count_greater_sorted(smem + _SMEM_CAND, cand)
                if rank < TOP_K:
                    _sts_b64(smem + _SMEM_TOP + rank * 8, cand)
            cute.arch.barrier()
            if warp == 0:
                mine = _lds_b64_if(lane, TOP_K, smem + _SMEM_TOP + lane * 8)
                expert = Int32(0)
                score = Float32(0.0)
                if lane < TOP_K:
                    expert = _key_expert(mine)
                    score = _lds_f32(smem + _SMEM_SIG + expert * 4)
                total = _warp_sum_f32(score)
                # Exact FP32 fast path of the FP64 expression (FP64 is slow on SM103): for a
                # unit scale and total >= 2^-12, ``double(total) + 1e-20 == double(total)``
                # and rounding the double quotient to FP32 equals FP32 division.
                weight = Float32(0.0)
                if _unit_scale_fast_path(scale_bits, total) != 0:
                    weight = _div_rn_f32(score, total)
                else:
                    weight = _renormalize(score, total, scale_bits)
                if lane < TOP_K:
                    out = (token * TOP_K + lane) * 4
                    _st_global_b32(ids_addr + Int64(out), expert)
                    _st_global_f32(vals_addr + Int64(out), weight)
                    if cutlass.const_expr(self.with_sort):
                        _st_relaxed_b32(ws_addr + self.ids_scratch_offset + Int64(out), expert)
            cute.arch.barrier()
            if cutlass.const_expr(not rearm_early):
                self.rearm_row(smem, tidx, ws_addr, stride, token)

    @cute.jit
    def load_rows(self, parts: cute.Tensor, row0: Int64, stride: int):
        """Issue the loads of one token's ``S`` partial rows (4 experts) into ``parts``."""
        for sp in cutlass.range_constexpr(self.num_splits):
            v = _ld_relaxed_v4_f32(row0 + Int64(sp * stride))
            for i in cutlass.range_constexpr(4):
                parts[4 * sp + i] = v[i]

    @cute.jit
    def rearm_row(self, smem: Int32, tidx: Int32, ws_addr: Int64, stride: int, token: Int32):
        """Refill a token's ``S`` consumed partial rows with the sentinel (TMA bulk copies)."""
        if tidx == _THREADS - 32:
            for sp in cutlass.range_constexpr(self.num_splits):
                _bulk_s2g(
                    ws_addr + Int64(sp * stride + token * _ROW_BYTES),
                    smem + _SMEM_SENTINEL,
                    _ROW_BYTES,
                )

    @cute.jit
    def sort_tokens(
        self,
        smem: Int32,
        tidx: Int32,
        warp: Int32,
        lane: Int32,
        m: Int32,
        scratch_addr: Int64,
        e2p_addr: Int64,
        p2e_addr: Int64,
        tile_expert_addr: Int64,
        tile_limit_addr: Int64,
        total_addr: Int64,
        num_tiles_addr: Int64,
        local_offset: Int32,
        local_num: Int32,
    ):
        """``moe_sort`` metadata of the local experts.

        A bitmask of the selecting tokens is built per local expert in shared memory. Experts
        get ``ceil(count / 128)`` tiles in local-expert order, rows follow the token order,
        and each tile's ``mn_limit`` is ``min(end of tile, first row of expert + count)``.
        """
        words = self.mask_words
        for i in cutlass.range_constexpr(_ceil_div(NUM_EXPERTS, _THREADS)):
            e = tidx + i * _THREADS
            if e < local_num:
                for w in cutlass.range_constexpr(words):
                    _sts_b32(smem + _SMEM_MASK + (w * NUM_EXPERTS + e) * 4, 0)
        cute.arch.barrier()

        n_entries = m * TOP_K
        per_thread = self.max_m * TOP_K // _THREADS
        # Poll the published expert ids (-1 = not yet written).
        ids = cute.make_rmem_tensor((per_thread,), Int32)
        ready = Int32(0)
        while ready == 0:
            ready = Int32(1)
            for q in cutlass.range_constexpr(per_thread):
                idx = tidx + q * _THREADS
                e = Int32(0)
                if idx < n_entries:
                    e = _ld_relaxed_b32(scratch_addr + Int64(idx * 4))
                ids[q] = e
                ready = ready & (1 - _is_neg(e))
        local_idx = []
        for q in cutlass.range_constexpr(per_thread):
            idx = tidx + q * _THREADS
            jl = Int32(-1)
            if idx < n_entries:
                jl = ids[q] - local_offset
                token = idx >> 4
                if jl >= 0 and jl < local_num:
                    word = smem + _SMEM_MASK + ((token >> 5) * NUM_EXPERTS + jl) * 4
                    _red_shared_or(word, Int32(1) << (token & 31))
            local_idx.append(jl)
        cute.arch.barrier()
        # Thread t owns local experts 4t..4t+3; exclusive scan of their tile counts.
        e0 = tidx * 4
        word_counts = []
        counts = [Int32(0)] * 4
        for w in cutlass.range_constexpr(words):
            v = _lds_v4_b32(smem + _SMEM_MASK + (w * NUM_EXPERTS + e0) * 4)
            word_counts.append([_popc(v[i]) for i in range(4)])
            counts = [counts[i] + word_counts[w][i] for i in range(4)]
        counts = [counts[i] & _below_mask(e0 + i, local_num) for i in range(4)]
        tiles = [(counts[i] + 127) >> 7 for i in range(4)]
        thread_tiles = tiles[0] + tiles[1] + tiles[2] + tiles[3]
        inclusive = _warp_inclusive_sum(thread_tiles)
        if lane == 31:
            _sts_b32(smem + _SMEM_WARP_TOT + warp * 4, inclusive)
        cute.arch.barrier()
        tile = inclusive - thread_tiles
        num_tiles = Int32(0)
        for w in cutlass.range_constexpr(_THREADS // 32):
            cnt_w = _lds_b32(smem + _SMEM_WARP_TOT + w * 4)
            if w < warp:
                tile = tile + cnt_w
            num_tiles = num_tiles + cnt_w
        for i in cutlass.range_constexpr(4):
            if tiles[i] != 0:
                # First permuted row of each mask word's tokens: tile start + earlier words.
                row = tile * 128
                for w in cutlass.range_constexpr(words):
                    _sts_b32(smem + _SMEM_WORD_BASE + (w * NUM_EXPERTS + e0 + i) * 4, row)
                    row = row + word_counts[w][i]
                for j in cutlass.range_constexpr(self.max_tiles_per_expert):
                    if j < tiles[i]:
                        limit = tile * 128 + counts[i]
                        if limit > (tile + j + 1) * 128:
                            limit = (tile + j + 1) * 128
                        _st_global_b32(tile_expert_addr + Int64((tile + j) * 4), e0 + i)
                        _st_global_b32(tile_limit_addr + Int64((tile + j) * 4), limit)
            tile = tile + tiles[i]
        if tidx == 0:
            _st_global_b32(total_addr, num_tiles * 128)
            _st_global_b32(num_tiles_addr, num_tiles)
        cute.arch.barrier()

        for q in cutlass.range_constexpr(per_thread):
            idx = tidx + q * _THREADS
            jl = local_idx[q]
            if idx < n_entries:
                perm = Int32(-1)
                if jl >= 0 and jl < local_num:
                    token = idx >> 4
                    slot = ((token >> 5) * NUM_EXPERTS + jl) * 4
                    word = _lds_b32(smem + _SMEM_MASK + slot)
                    perm = _lds_b32(smem + _SMEM_WORD_BASE + slot) + _popc(
                        word & ((Int32(1) << (token & 31)) - 1)
                    )
                    _st_global_b32(p2e_addr + Int64(perm * 4), idx)
                _st_global_b32(e2p_addr + Int64(idx * 4), perm)
        if tidx == _THREADS - 32:
            # Re-arm the expert-id slots for the next launch (last: a full TMA queue would
            # stall this warp and every barrier behind it).
            n_bytes = n_entries * 4
            for piece in cutlass.range_constexpr(self.max_m * TOP_K * 4 // _SENTINEL_BYTES):
                off = piece * _SENTINEL_BYTES
                if off < n_bytes:
                    size = n_bytes - off
                    if size > _SENTINEL_BYTES:
                        size = Int32(_SENTINEL_BYTES)
                    _bulk_s2g(scratch_addr + Int64(off), smem + _SMEM_SENTINEL, size)


class K3FusedRouterKernel(_RouterTail):
    """Router GEMM + sigmoid/bias top-16 (+ ``moe_sort`` metadata) in one launch.

    Args:
        n_tiles: Token tiles of 8 (``ceil(M / 8)``, 1..8).
        num_splits: K splits ``S``; the grid is ``8 * S`` CTAs.
        with_sort: Also produce the ``moe_sort`` outputs.
        use_pdl: Launch with programmatic dependent launch.
    """

    def __init__(
        self, n_tiles: int, num_splits: int, with_sort: bool, use_pdl: bool, ids_scratch_offset: int
    ):
        if not 1 <= n_tiles <= SMALL_M // 8:
            raise ValueError(f"n_tiles={n_tiles} must be in [1, {SMALL_M // 8}]")
        if not 1 <= num_splits <= _NUM_K_CHUNKS:
            raise ValueError(f"num_splits={num_splits} must be in [1, {_NUM_K_CHUNKS}]")
        self._init_tail(
            num_splits, _EXPERT_GROUPS * num_splits, SMALL_M, with_sort, ids_scratch_offset
        )
        self.n_tiles = n_tiles
        self.m_pad = 8 * n_tiles
        self.max_chunks = _ceil_div(_NUM_K_CHUNKS, num_splits)
        self.use_pdl = use_pdl
        self.h_chunk_bytes = self.m_pad * _K_CHUNK * 2
        if self.max_chunks * _W_CHUNK_BYTES < _SCRATCH_BYTES:
            raise ValueError(f"num_splits={num_splits} leaves too little shared memory")

    @cute.jit
    def __call__(
        self,
        w_ptr: cute.Pointer,
        h_ptr: cute.Pointer,
        m: Int32,
        bias_addr: Int64,
        ws_addr: Int64,
        counter_addr: Int64,
        ids_addr: Int64,
        vals_addr: Int64,
        logits_addr: Int64,
        e2p_addr: Int64,
        p2e_addr: Int64,
        tile_expert_addr: Int64,
        tile_limit_addr: Int64,
        total_addr: Int64,
        num_tiles_addr: Int64,
        local_offset: Int32,
        local_num: Int32,
        write_logits: Int32,
        scale_bits: Int64,
        stream: cuda.CUstream,
    ):
        mW = cute.make_tensor(
            w_ptr, cute.make_layout((NUM_EXPERTS, HIDDEN_SIZE), stride=(HIDDEN_SIZE, 1))
        )
        mH = cute.make_tensor(h_ptr, cute.make_layout((m, HIDDEN_SIZE), stride=(HIDDEN_SIZE, 1)))
        atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_SW128, cutlass.BFloat16
        )
        w_layout = cute.tile_to_shape(
            atom, (_EXPERTS_PER_CTA, _K_CHUNK, self.max_chunks), order=(0, 1, 2)
        )
        h_layout = cute.tile_to_shape(
            atom, (self.m_pad, _K_CHUNK, self.max_chunks), order=(0, 1, 2)
        )
        tma_atom_w, tma_w = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mW,
            cute.tile_to_shape(atom, (_EXPERTS_PER_CTA, _K_CHUNK), order=(0, 1)),
            (_EXPERTS_PER_CTA, _K_CHUNK),
        )
        tma_atom_h, tma_h = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mH,
            cute.tile_to_shape(atom, (self.m_pad, _K_CHUNK), order=(0, 1)),
            (self.m_pad, _K_CHUNK),
        )

        @cute.struct
        class SharedStorage:
            mbar: cute.struct.MemRange[cutlass.Int64, self.max_chunks]
            ticket: cute.struct.MemRange[cutlass.Int32, 1]
            params: cute.struct.Align[cute.struct.MemRange[cutlass.Int64, _NUM_PARAMS], 16]
            sW: cute.struct.Align[
                cute.struct.MemRange[cutlass.BFloat16, cute.cosize(w_layout.outer)], 1024
            ]
            sH: cute.struct.Align[
                cute.struct.MemRange[cutlass.BFloat16, cute.cosize(h_layout.outer)], 1024
            ]

        self.shared_storage = SharedStorage

        self.kernel(
            tma_atom_w,
            tma_w,
            tma_atom_h,
            tma_h,
            w_layout,
            h_layout,
            m,
            bias_addr,
            ws_addr,
            counter_addr,
            ids_addr,
            vals_addr,
            logits_addr,
            e2p_addr,
            p2e_addr,
            tile_expert_addr,
            tile_limit_addr,
            total_addr,
            num_tiles_addr,
            local_offset,
            local_num,
            write_logits,
            scale_bits,
        ).launch(
            grid=(self.num_ctas, 1, 1),
            block=(_THREADS, 1, 1),
            smem=self.shared_storage.size_in_bytes(),
            min_blocks_per_mp=1,
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        tma_atom_w: cute.CopyAtom,
        mWt: cute.Tensor,
        tma_atom_h: cute.CopyAtom,
        mHt: cute.Tensor,
        w_layout: cute.ComposedLayout,
        h_layout: cute.ComposedLayout,
        m: Int32,
        bias_addr: Int64,
        ws_addr: Int64,
        counter_addr: Int64,
        ids_addr: Int64,
        vals_addr: Int64,
        logits_addr: Int64,
        e2p_addr: Int64,
        p2e_addr: Int64,
        tile_expert_addr: Int64,
        tile_limit_addr: Int64,
        total_addr: Int64,
        num_tiles_addr: Int64,
        local_offset: Int32,
        local_num: Int32,
        write_logits: Int32,
        scale_bits: Int64,
    ):
        S = self.num_splits
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()

        group = bidx % _EXPERT_GROUPS
        split = bidx // _EXPERT_GROUPS
        base_chunks = _NUM_K_CHUNKS // S
        rem_chunks = _NUM_K_CHUNKS % S
        chunk_count = Int32(base_chunks)
        chunk_start = split * base_chunks
        if cutlass.const_expr(rem_chunks > 0):
            if split < rem_chunks:
                chunk_count = chunk_count + 1
                chunk_start = chunk_start + split
            else:
                chunk_start = chunk_start + rem_chunks

        storage = utils.SmemAllocator().allocate(self.shared_storage)
        mbar = storage.mbar.data_ptr()
        # Scalar kernel parameters are re-read from constant memory at every use, and
        # constant-cache misses in the latency-critical tail cost ~1k cycles each. Cache them
        # in shared memory once, while the weight stream is in flight.
        params = storage.params.data_ptr().toint()
        if warp == 0:
            with cute.arch.elect_one():
                for i in cutlass.range_constexpr(self.max_chunks):
                    cute.arch.mbarrier_init(mbar + i, 1)
        elif warp == 1:
            if lane == 0:
                cached = (
                    Int64(m),
                    bias_addr,
                    ws_addr,
                    ids_addr,
                    vals_addr,
                    logits_addr,
                    Int64(write_logits),
                    scale_bits,
                    e2p_addr,
                    p2e_addr,
                    tile_expert_addr,
                    tile_limit_addr,
                    total_addr,
                    num_tiles_addr,
                    Int64(local_offset),
                    Int64(local_num),
                )
                for i in cutlass.range_constexpr(_NUM_PARAMS):
                    _sts_b64(params + i * 8, cached[i])
        cute.arch.mbarrier_init_fence()
        cute.arch.barrier()
        m = Int32(_lds_b64(params + _P_M * 8))

        # ---- Loads: the weight never depends on the previous grid. ----
        if warp == 0:
            if cutlass.const_expr(self.use_pdl):
                _griddepcontrol_launch_dependents()
            cpasync.prefetch_descriptor(tma_atom_w)
            cpasync.prefetch_descriptor(tma_atom_h)
            sW = storage.sW.get_tensor(w_layout.outer, swizzle=w_layout.inner)
            sH = storage.sH.get_tensor(h_layout.outer, swizzle=h_layout.inner)
            gW = cute.local_tile(mWt, (_EXPERTS_PER_CTA, _K_CHUNK), (None, None))
            gH = cute.local_tile(mHt, (self.m_pad, _K_CHUNK), (None, None))
            tWs, tWg = cpasync.tma_partition(
                tma_atom_w,
                0,
                cute.make_layout(1),
                cute.group_modes(sW, 0, 2),
                cute.group_modes(gW, 0, 2),
            )
            tHs, tHg = cpasync.tma_partition(
                tma_atom_h,
                0,
                cute.make_layout(1),
                cute.group_modes(sH, 0, 2),
                cute.group_modes(gH, 0, 2),
            )
            for c in cutlass.range(chunk_count, unroll=1):
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        mbar + c, _W_CHUNK_BYTES + self.h_chunk_bytes
                    )
                cute.copy(
                    tma_atom_w,
                    tWg[(None, group, chunk_start + c)],
                    tWs[(None, c)],
                    tma_bar_ptr=mbar + c,
                    cache_policy=Int64(_EVICT_FIRST),
                )
            if cutlass.const_expr(self.use_pdl):
                _griddepcontrol_wait()
            for c in cutlass.range(chunk_count, unroll=1):
                cute.copy(
                    tma_atom_h,
                    tHg[(None, 0, chunk_start + c)],
                    tHs[(None, c)],
                    tma_bar_ptr=mbar + c,
                )

        # ---- GEMM: warp w < 7 owns experts [16 w, 16 w + 16) of the CTA's 112. ----
        ticket_addr = storage.ticket.data_ptr().toint()
        if warp < _M_TILES:
            self.mma_and_store(
                storage,
                mbar,
                warp,
                lane,
                group,
                split,
                chunk_count,
                m,
                _lds_b64(params + _P_WS * 8),
            )
        elif warp == _THREADS // 32 - 1:
            # Arrival: once its last operands have landed, the CTA takes a ticket; the last T
            # CTAs become top-k workers. Partial logits are handed over through
            # sentinel-initialized slots (consumers poll the data itself), so the counter only
            # assigns roles, and its last user resets it.
            cute.arch.mbarrier_wait(mbar + chunk_count - 1, 0)
            if lane == 0:
                arrived = _atom_add_relaxed_gpu(counter_addr, 1)
                if arrived == self.num_ctas - 1:
                    _st_relaxed_gpu(counter_addr, 0)
                _sts_b32(ticket_addr, arrived)
        if cutlass.const_expr(self.use_pdl):
            _griddepcontrol_wait()
        cute.arch.barrier()
        self.route(storage.sW.data_ptr().toint(), params, ticket_addr, tidx, warp, lane, m)

    @cute.jit
    def mma_and_store(
        self,
        storage,
        mbar,
        warp: Int32,
        lane: Int32,
        group: Int32,
        split: Int32,
        chunk_count: Int32,
        m: Int32,
        ws_addr: Int64,
    ):
        n_tiles = self.n_tiles
        acc = cute.make_rmem_tensor((n_tiles * 4,), Float32)
        acc.fill(0.0)
        w_base = storage.sW.data_ptr().toint()
        h_base = storage.sH.data_ptr().toint()
        mat = lane >> 3
        r = lane & 7
        # ldmatrix row addresses: A matrices (rows 0-7 | 8-15) x (k 0-7 | 8-15); the 128B
        # swizzle XORs the 16-byte unit index with (row & 7) == r.
        a_row = w_base + (warp * 16 + (mat & 1) * 8 + r) * 128
        a_unit = mat >> 1
        b_unit = mat & 1
        for c in cutlass.range(chunk_count, unroll=1):
            cute.arch.mbarrier_wait(mbar + c, 0)
            a_chunk = a_row + c * _W_CHUNK_BYTES
            h_chunk = h_base + c * self.h_chunk_bytes
            for ks in cutlass.range_constexpr(_K_CHUNK // 16):
                a = _ldsm_x4(a_chunk + (((ks * 2 + a_unit) ^ r) << 4))
                for p in cutlass.range_constexpr(_ceil_div(n_tiles, 2)):
                    row = (2 * p + (mat >> 1)) * 8 + r
                    if cutlass.const_expr(2 * p + 1 == n_tiles):
                        # Odd last tile: the upper two matrices repeat the lower two.
                        row = (2 * p) * 8 + r
                    b = _ldsm_x4(h_chunk + row * 128 + (((ks * 2 + b_unit) ^ r) << 4))
                    for q in cutlass.range_constexpr(min(2, n_tiles - 2 * p)):
                        base = 4 * (2 * p + q)
                        d = _mma_bf16(
                            a,
                            b[2 * q],
                            b[2 * q + 1],
                            (acc[base], acc[base + 1], acc[base + 2], acc[base + 3]),
                        )
                        for i in cutlass.range_constexpr(4):
                            acc[base + i] = d[i]

        if cutlass.const_expr(self.use_pdl):
            _griddepcontrol_wait()
        # Partial logits: ws[split, token, expert]; d0/d1 at (row lane/4, cols 2(lane%4)+{0,1}),
        # d2/d3 eight rows below.
        expert = group * _EXPERTS_PER_CTA + warp * 16 + (lane >> 2)
        split_base = split * self.slot_rows
        for nt in cutlass.range_constexpr(n_tiles):
            for i in cutlass.range_constexpr(4):
                token = nt * 8 + 2 * (lane & 3) + (i & 1)
                if token < m:
                    off = ((split_base + token) * NUM_EXPERTS + expert + (i >> 1) * 8) * 4
                    _st_relaxed_f32(ws_addr + Int64(off), acc[4 * nt + i])


class K3FusedRouterUmmaKernel(_RouterTail):
    """``64 < M <= 256``: the GEMM phase on ``tcgen05`` UMMA, the rest shared with the small path.

    CTA ``(g, s)`` owns experts ``[128 g, 128 g + 128)`` (the UMMA M side; tokens are N) and the
    ``s``-th slice of K in 32-wide chunks. Warp 0 TMA-loads all of the CTA's weight chunks before
    ``griddepcontrol.wait`` and then streams activation chunks through a ring; warp 1 allocates
    tensor memory and issues the UMMAs; warp 2 takes the arrival ticket; warps 4-7 copy the
    accumulator (one expert row per thread) into the partial-logit slots.

    Args:
        n_pad: Padded token count (UMMA N), a multiple of 16 in ``(64, 256]``.
        num_splits: K splits ``S``; the grid is ``7 * S`` CTAs.
        with_sort: Also produce the ``moe_sort`` outputs.
        use_pdl: Launch with programmatic dependent launch.
    """

    def __init__(
        self, n_pad: int, num_splits: int, with_sort: bool, use_pdl: bool, ids_scratch_offset: int
    ):
        if n_pad % 16 or not SMALL_M < n_pad <= MAX_M:
            raise ValueError(f"n_pad={n_pad} must be a multiple of 16 in ({SMALL_M}, {MAX_M}]")
        if not 1 <= num_splits <= _UMMA_NUM_K_CHUNKS:
            raise ValueError(f"num_splits={num_splits} must be in [1, {_UMMA_NUM_K_CHUNKS}]")
        num_ctas = _UMMA_GROUPS * num_splits
        self._init_tail(
            num_splits,
            num_ctas,
            MAX_M,
            with_sort,
            ids_scratch_offset,
        )
        self.n_pad = n_pad
        self.use_pdl = use_pdl
        self.max_chunks = _ceil_div(_UMMA_NUM_K_CHUNKS, num_splits)
        self.b_chunk_bytes = n_pad * _UMMA_K_CHUNK * 2
        self.tmem_cols = max(32, 1 << (n_pad - 1).bit_length())
        a_bytes = self.max_chunks * _UMMA_A_CHUNK_BYTES
        if a_bytes < _SCRATCH_BYTES:
            raise ValueError(f"num_splits={num_splits} leaves too little shared memory")
        budget = _SMEM_CAPACITY - a_bytes - 4096  # barriers, parameters, alignment
        self.b_stages = min(self.max_chunks, budget // self.b_chunk_bytes)
        if self.b_stages < 2:
            raise ValueError(f"num_splits={num_splits} leaves no room for the activation ring")
        self.mma_tiler = (_UMMA_ROWS, n_pad, _UMMA_K_CHUNK)

    @cute.jit
    def __call__(
        self,
        w_ptr: cute.Pointer,
        h_ptr: cute.Pointer,
        m: Int32,
        bias_addr: Int64,
        ws_addr: Int64,
        counter_addr: Int64,
        ids_addr: Int64,
        vals_addr: Int64,
        logits_addr: Int64,
        e2p_addr: Int64,
        p2e_addr: Int64,
        tile_expert_addr: Int64,
        tile_limit_addr: Int64,
        total_addr: Int64,
        num_tiles_addr: Int64,
        local_offset: Int32,
        local_num: Int32,
        write_logits: Int32,
        scale_bits: Int64,
        stream: cuda.CUstream,
    ):
        mW = cute.make_tensor(
            w_ptr,
            cute.make_layout(
                (NUM_EXPERTS, HIDDEN_SIZE, 1), stride=(HIDDEN_SIZE, 1, NUM_EXPERTS * HIDDEN_SIZE)
            ),
        )
        mH = cute.make_tensor(
            h_ptr, cute.make_layout((m, HIDDEN_SIZE, 1), stride=(HIDDEN_SIZE, 1, m * HIDDEN_SIZE))
        )
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            cutlass.BFloat16,
            cutlass.BFloat16,
            OperandMajorMode.K,
            OperandMajorMode.K,
            cutlass.Float32,
            tcgen05.CtaGroup.ONE,
            self.mma_tiler[:2],
        )
        a_layout = sm100_utils.make_smem_layout_a(
            tiled_mma, self.mma_tiler, cutlass.BFloat16, self.max_chunks
        )
        b_layout = sm100_utils.make_smem_layout_b(
            tiled_mma, self.mma_tiler, cutlass.BFloat16, self.b_stages
        )
        cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((1, 1, 1)), (tiled_mma.thr_id.shape,)
        )
        tma_atom_a, tma_a = cute.nvgpu.make_tiled_tma_atom_A(
            cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
            mW,
            cute.slice_(a_layout, (None, None, None, 0)),
            self.mma_tiler,
            tiled_mma,
            cluster_layout_vmnk.shape,
        )
        tma_atom_b, tma_b = cute.nvgpu.make_tiled_tma_atom_B(
            cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
            mH,
            cute.slice_(b_layout, (None, None, None, 0)),
            self.mma_tiler,
            tiled_mma,
            cluster_layout_vmnk.shape,
        )

        @cute.struct
        class SharedStorage:
            a_full: cute.struct.MemRange[cutlass.Int64, self.max_chunks]
            b_full: cute.struct.MemRange[cutlass.Int64, self.b_stages]
            b_empty: cute.struct.MemRange[cutlass.Int64, self.b_stages]
            acc_full: cute.struct.MemRange[cutlass.Int64, 1]
            tmem_ready: cute.struct.MemRange[cutlass.Int64, 1]
            tmem_free: cute.struct.MemRange[cutlass.Int64, 1]
            tmem_holding: cute.struct.MemRange[cutlass.Int32, 1]
            ticket: cute.struct.MemRange[cutlass.Int32, 1]
            params: cute.struct.Align[cute.struct.MemRange[cutlass.Int64, _NUM_PARAMS], 16]
            sA: cute.struct.Align[
                cute.struct.MemRange[cutlass.BFloat16, cute.cosize(a_layout.outer)], 1024
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[cutlass.BFloat16, cute.cosize(b_layout.outer)], 1024
            ]

        self.shared_storage = SharedStorage
        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_a,
            tma_atom_b,
            tma_b,
            a_layout,
            b_layout,
            m,
            bias_addr,
            ws_addr,
            counter_addr,
            ids_addr,
            vals_addr,
            logits_addr,
            e2p_addr,
            p2e_addr,
            tile_expert_addr,
            tile_limit_addr,
            total_addr,
            num_tiles_addr,
            local_offset,
            local_num,
            write_logits,
            scale_bits,
        ).launch(
            grid=(self.num_ctas, 1, 1),
            block=(_THREADS, 1, 1),
            smem=self.shared_storage.size_in_bytes(),
            min_blocks_per_mp=1,
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB: cute.Tensor,
        a_layout: cute.ComposedLayout,
        b_layout: cute.ComposedLayout,
        m: Int32,
        bias_addr: Int64,
        ws_addr: Int64,
        counter_addr: Int64,
        ids_addr: Int64,
        vals_addr: Int64,
        logits_addr: Int64,
        e2p_addr: Int64,
        p2e_addr: Int64,
        tile_expert_addr: Int64,
        tile_limit_addr: Int64,
        total_addr: Int64,
        num_tiles_addr: Int64,
        local_offset: Int32,
        local_num: Int32,
        write_logits: Int32,
        scale_bits: Int64,
    ):
        S = self.num_splits
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        group = bidx % _UMMA_GROUPS
        split = bidx // _UMMA_GROUPS
        base_chunks = _UMMA_NUM_K_CHUNKS // S
        rem_chunks = _UMMA_NUM_K_CHUNKS % S
        chunk_count = Int32(base_chunks)
        chunk_start = split * base_chunks
        if cutlass.const_expr(rem_chunks > 0):
            if split < rem_chunks:
                chunk_count = chunk_count + 1
                chunk_start = chunk_start + split
            else:
                chunk_start = chunk_start + rem_chunks

        storage = utils.SmemAllocator().allocate(self.shared_storage)
        a_full = storage.a_full.data_ptr()
        b_full = storage.b_full.data_ptr()
        b_empty = storage.b_empty.data_ptr()
        acc_full = storage.acc_full.data_ptr()
        tmem_ready = storage.tmem_ready.data_ptr()
        tmem_free = storage.tmem_free.data_ptr()
        tmem_holding = storage.tmem_holding.data_ptr()
        params = storage.params.data_ptr().toint()
        if warp == 0:
            with cute.arch.elect_one():
                for i in cutlass.range_constexpr(self.max_chunks):
                    cute.arch.mbarrier_init(a_full + i, 1)
                for i in cutlass.range_constexpr(self.b_stages):
                    cute.arch.mbarrier_init(b_full + i, 1)
                    cute.arch.mbarrier_init(b_empty + i, 1)
                cute.arch.mbarrier_init(acc_full, 1)
                cute.arch.mbarrier_init(tmem_ready, 1)
                cute.arch.mbarrier_init(tmem_free, 4)
        elif warp == 3:
            if lane == 0:
                cached = (
                    Int64(m),
                    bias_addr,
                    ws_addr,
                    ids_addr,
                    vals_addr,
                    logits_addr,
                    Int64(write_logits),
                    scale_bits,
                    e2p_addr,
                    p2e_addr,
                    tile_expert_addr,
                    tile_limit_addr,
                    total_addr,
                    num_tiles_addr,
                    Int64(local_offset),
                    Int64(local_num),
                )
                for i in cutlass.range_constexpr(_NUM_PARAMS):
                    _sts_b64(params + i * 8, cached[i])
        cute.arch.mbarrier_init_fence()
        cute.arch.barrier()
        m = Int32(_lds_b64(params + _P_M * 8))
        ticket_addr = storage.ticket.data_ptr().toint()

        if warp == 0:
            # Weight first (independent of the previous grid), then the activation ring.
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            sA = storage.sA.get_tensor(a_layout.outer, swizzle=a_layout.inner)
            sB = storage.sB.get_tensor(b_layout.outer, swizzle=b_layout.inner)
            gA = cute.local_tile(
                mA, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
            )
            gB = cute.local_tile(
                mB, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
            )
            thr_mma = tiled_mma.get_slice(0)
            tAsA, tAgA = cpasync.tma_partition(
                tma_atom_a,
                0,
                cute.make_layout(1),
                cute.group_modes(sA, 0, 3),
                cute.group_modes(thr_mma.partition_A(gA), 0, 3),
            )
            tBsB, tBgB = cpasync.tma_partition(
                tma_atom_b,
                0,
                cute.make_layout(1),
                cute.group_modes(sB, 0, 3),
                cute.group_modes(thr_mma.partition_B(gB), 0, 3),
            )
            for c in cutlass.range(chunk_count, unroll=1):
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(a_full + c, _UMMA_A_CHUNK_BYTES)
                cute.copy(
                    tma_atom_a,
                    tAgA[(None, group, chunk_start + c, 0)],
                    tAsA[(None, c)],
                    tma_bar_ptr=a_full + c,
                    cache_policy=Int64(_EVICT_FIRST),
                )
            # Dependents may launch only once this CTA holds its tensor memory: a dependent CTA
            # that took the columns first and then waited on this grid would deadlock it.
            cute.arch.mbarrier_wait(tmem_ready, 0)
            if cutlass.const_expr(self.use_pdl):
                _griddepcontrol_launch_dependents()
                _griddepcontrol_wait()
            for c in cutlass.range(chunk_count, unroll=1):
                stage = c % self.b_stages
                if c >= self.b_stages:
                    cute.arch.mbarrier_wait(b_empty + stage, ((c // self.b_stages) & 1) ^ 1)
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(b_full + stage, self.b_chunk_bytes)
                cute.copy(
                    tma_atom_b,
                    tBgB[(None, 0, chunk_start + c, 0)],
                    tBsB[(None, stage)],
                    tma_bar_ptr=b_full + stage,
                )
        elif warp == 1:
            cute.arch.alloc_tmem(self.tmem_cols, tmem_holding)
            cute.arch.relinquish_tmem_alloc_permit()
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive(tmem_ready)
            acc_tmem_ptr = cute.arch.retrieve_tmem_ptr(
                cutlass.Float32, alignment=16, ptr_to_buffer_holding_addr=tmem_holding
            )
            acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
            tCtAcc = cute.make_tensor(acc_tmem_ptr, tiled_mma.make_fragment_C(acc_shape).layout)
            sA = storage.sA.get_tensor(a_layout.outer, swizzle=a_layout.inner)
            sB = storage.sB.get_tensor(b_layout.outer, swizzle=b_layout.inner)
            tCrA = tiled_mma.make_fragment_A(sA)
            tCrB = tiled_mma.make_fragment_B(sB)
            num_kphases = cute.size(tCrA, mode=[2])
            tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
            for c in cutlass.range(chunk_count, unroll=1):
                stage = c % self.b_stages
                cute.arch.mbarrier_wait(a_full + c, 0)
                cute.arch.mbarrier_wait(b_full + stage, (c // self.b_stages) & 1)
                for kp in cutlass.range_constexpr(num_kphases):
                    cute.gemm(
                        tiled_mma,
                        tCtAcc,
                        tCrA[(None, None, kp, c)],
                        tCrB[(None, None, kp, stage)],
                        tCtAcc,
                    )
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                with cute.arch.elect_one():
                    tcgen05.commit(b_empty + stage)
            with cute.arch.elect_one():
                tcgen05.commit(acc_full)
            cute.arch.mbarrier_wait(tmem_free, 0)
            cute.arch.dealloc_tmem(acc_tmem_ptr, self.tmem_cols)
        elif warp == 2:
            # Arrival once the accumulator is final (see the small path).
            cute.arch.mbarrier_wait(acc_full, 0)
            if lane == 0:
                arrived = _atom_add_relaxed_gpu(counter_addr, 1)
                if arrived == self.num_ctas - 1:
                    _st_relaxed_gpu(counter_addr, 0)
                _sts_b32(ticket_addr, arrived)
        elif warp >= 4:
            # TMEM lane = expert row: warp 4 + i reads lanes [32 i, 32 i + 32).
            cute.arch.mbarrier_wait(tmem_ready, 0)
            tmem_base = _lds_b32(tmem_holding.toint())
            if cutlass.const_expr(self.use_pdl):
                _griddepcontrol_wait()
            row = tidx - 128
            row_addr = _lds_b64(params + _P_WS * 8) + Int64(
                ((split * self.slot_rows) * NUM_EXPERTS + group * _UMMA_ROWS + row) * 4
            )
            lane_base = tmem_base + (((warp - 4) * 32) << 16)
            cute.arch.mbarrier_wait(acc_full, 0)
            for j in cutlass.range_constexpr(self.n_pad // 16):
                v = _tmem_ld16(lane_base + j * 16)
                for i in cutlass.range_constexpr(16):
                    token = j * 16 + i
                    if token < m:
                        _st_relaxed_f32(row_addr + Int64(token * _ROW_BYTES), v[i])
            _tmem_fence_before_sync()
            cute.arch.sync_warp()
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive(tmem_free)
        if cutlass.const_expr(self.use_pdl):
            _griddepcontrol_wait()
        cute.arch.barrier()
        self.route(storage.sA.data_ptr().toint(), params, ticket_addr, tidx, warp, lane, m)


# ---------------------------------------------------------------------------
# Standalone low-M BF16 GEMV (the router's GEMM phase, generalized)
# ---------------------------------------------------------------------------

#: Largest token count of the standalone GEMV.
GEMV_MAX_M = 64
_GEMV_ROWS = 16
_GEMV_WARPS = 4
_GEMV_THREADS = 32 * _GEMV_WARPS
#: Shared-memory budget of the weight + activation chunk rings.
_GEMV_SMEM_BUDGET = 192 * 1024


class K3LowMGemvKernel:
    """``out[M, N] = x[M, K] @ w[N, K]^T`` for ``M <= 64`` (BF16 in, FP32 accumulate).

    CTA ``(g, s)`` owns weight rows ``[16 g, 16 g + 16)`` and the ``s``-th slice of K in
    64-wide chunks, all of which it TMA-loads up front (the weight before
    ``griddepcontrol.wait``). Its four warps take the chunks round-robin (``mma.sync``
    m16n8k16, rows = MMA M) and are reduced through shared memory in warp order. With
    ``num_splits > 1`` the CTAs of a row tile store FP32 partials and the last one to arrive
    sums them in split order (deterministic) and resets its counter.

    Args:
        n: Rows of ``w`` (output features), a multiple of 8.
        k: Reduction size, a multiple of 128.
        n_tiles: Token tiles of 8 (``ceil(M / 8)``).
        num_splits: K splits ``S`` over CTAs.
        out_fp32: FP32 output (otherwise BF16).
        use_pdl: Launch with programmatic dependent launch.
    """

    def __init__(
        self, n: int, k: int, n_tiles: int, num_splits: int, out_fp32: bool, use_pdl: bool
    ):
        if n % 8 or k % 128:
            raise ValueError(f"N={n} must be a multiple of 8 and K={k} a multiple of 128")
        if not 1 <= n_tiles <= GEMV_MAX_M // 8:
            raise ValueError(f"n_tiles={n_tiles} must be in [1, {GEMV_MAX_M // 8}]")
        self.n = n
        self.k = k
        self.n_tiles = n_tiles
        self.m_pad = 8 * n_tiles
        self.num_groups = _ceil_div(n, _GEMV_ROWS)
        self.num_k_chunks = k // _K_CHUNK
        if not 1 <= num_splits <= self.num_k_chunks:
            raise ValueError(f"num_splits={num_splits} invalid for K={k}")
        self.num_splits = num_splits
        self.max_chunks = _ceil_div(self.num_k_chunks, num_splits)
        self.w_chunk_bytes = _GEMV_ROWS * _K_CHUNK * 2
        self.x_chunk_bytes = self.m_pad * _K_CHUNK * 2
        self.tile_elems = _GEMV_ROWS * self.m_pad
        if self.max_chunks * (self.w_chunk_bytes + self.x_chunk_bytes) > _GEMV_SMEM_BUDGET:
            raise ValueError(f"num_splits={num_splits} leaves too many K chunks per CTA")
        self.out_fp32 = out_fp32
        self.use_pdl = use_pdl

    @cute.jit
    def __call__(
        self,
        w_ptr: cute.Pointer,
        x_ptr: cute.Pointer,
        m: Int32,
        out_addr: Int64,
        ws_addr: Int64,
        counter_addr: Int64,
        stream: cuda.CUstream,
    ):
        mW = cute.make_tensor(w_ptr, cute.make_layout((self.n, self.k), stride=(self.k, 1)))
        mX = cute.make_tensor(x_ptr, cute.make_layout((m, self.k), stride=(self.k, 1)))
        atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_SW128, cutlass.BFloat16
        )
        w_layout = cute.tile_to_shape(atom, (_GEMV_ROWS, _K_CHUNK, self.max_chunks), (0, 1, 2))
        x_layout = cute.tile_to_shape(atom, (self.m_pad, _K_CHUNK, self.max_chunks), (0, 1, 2))
        tma_atom_w, tma_w = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mW,
            cute.tile_to_shape(atom, (_GEMV_ROWS, _K_CHUNK), order=(0, 1)),
            (_GEMV_ROWS, _K_CHUNK),
        )
        tma_atom_x, tma_x = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mX,
            cute.tile_to_shape(atom, (self.m_pad, _K_CHUNK), order=(0, 1)),
            (self.m_pad, _K_CHUNK),
        )

        @cute.struct
        class SharedStorage:
            mbar: cute.struct.MemRange[cutlass.Int64, self.max_chunks]
            ticket: cute.struct.MemRange[cutlass.Int32, 1]
            red: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, _GEMV_WARPS * self.tile_elems], 16
            ]
            sW: cute.struct.Align[
                cute.struct.MemRange[cutlass.BFloat16, cute.cosize(w_layout.outer)], 1024
            ]
            sX: cute.struct.Align[
                cute.struct.MemRange[cutlass.BFloat16, cute.cosize(x_layout.outer)], 1024
            ]

        self.shared_storage = SharedStorage
        self.kernel(
            tma_atom_w,
            tma_w,
            tma_atom_x,
            tma_x,
            w_layout,
            x_layout,
            m,
            out_addr,
            ws_addr,
            counter_addr,
        ).launch(
            grid=(self.num_groups * self.num_splits, 1, 1),
            block=(_GEMV_THREADS, 1, 1),
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        tma_atom_w: cute.CopyAtom,
        mWt: cute.Tensor,
        tma_atom_x: cute.CopyAtom,
        mXt: cute.Tensor,
        w_layout: cute.ComposedLayout,
        x_layout: cute.ComposedLayout,
        m: Int32,
        out_addr: Int64,
        ws_addr: Int64,
        counter_addr: Int64,
    ):
        S = self.num_splits
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        group = bidx % self.num_groups
        split = bidx // self.num_groups
        base_chunks = self.num_k_chunks // S
        rem_chunks = self.num_k_chunks % S
        chunk_count = Int32(base_chunks)
        chunk_start = split * base_chunks
        if cutlass.const_expr(rem_chunks > 0):
            if split < rem_chunks:
                chunk_count = chunk_count + 1
                chunk_start = chunk_start + split
            else:
                chunk_start = chunk_start + rem_chunks

        storage = utils.SmemAllocator().allocate(self.shared_storage)
        mbar = storage.mbar.data_ptr()
        if warp == 0:
            with cute.arch.elect_one():
                for i in cutlass.range_constexpr(self.max_chunks):
                    cute.arch.mbarrier_init(mbar + i, 1)
        cute.arch.mbarrier_init_fence()
        cute.arch.barrier()

        if warp == 0:
            if cutlass.const_expr(self.use_pdl):
                _griddepcontrol_launch_dependents()
            cpasync.prefetch_descriptor(tma_atom_w)
            cpasync.prefetch_descriptor(tma_atom_x)
            sW = storage.sW.get_tensor(w_layout.outer, swizzle=w_layout.inner)
            sX = storage.sX.get_tensor(x_layout.outer, swizzle=x_layout.inner)
            gW = cute.local_tile(mWt, (_GEMV_ROWS, _K_CHUNK), (None, None))
            gX = cute.local_tile(mXt, (self.m_pad, _K_CHUNK), (None, None))
            tWs, tWg = cpasync.tma_partition(
                tma_atom_w,
                0,
                cute.make_layout(1),
                cute.group_modes(sW, 0, 2),
                cute.group_modes(gW, 0, 2),
            )
            tXs, tXg = cpasync.tma_partition(
                tma_atom_x,
                0,
                cute.make_layout(1),
                cute.group_modes(sX, 0, 2),
                cute.group_modes(gX, 0, 2),
            )
            for c in cutlass.range(chunk_count, unroll=1):
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        mbar + c, self.w_chunk_bytes + self.x_chunk_bytes
                    )
                cute.copy(
                    tma_atom_w,
                    tWg[(None, group, chunk_start + c)],
                    tWs[(None, c)],
                    tma_bar_ptr=mbar + c,
                    cache_policy=Int64(_EVICT_FIRST),
                )
            if cutlass.const_expr(self.use_pdl):
                _griddepcontrol_wait()
            for c in cutlass.range(chunk_count, unroll=1):
                cute.copy(
                    tma_atom_x,
                    tXg[(None, 0, chunk_start + c)],
                    tXs[(None, c)],
                    tma_bar_ptr=mbar + c,
                )

        # ---- MMA: warp w takes chunks w, w + 4, ...; all warps share the 16 weight rows. ----
        n_tiles = self.n_tiles
        acc = cute.make_rmem_tensor((n_tiles * 4,), Float32)
        acc.fill(0.0)
        w_base = storage.sW.data_ptr().toint()
        x_base = storage.sX.data_ptr().toint()
        mat = lane >> 3
        r = lane & 7
        a_row = w_base + ((mat & 1) * 8 + r) * 128
        a_unit = mat >> 1
        b_unit = mat & 1
        for c in cutlass.range(warp, chunk_count, _GEMV_WARPS, unroll=1):
            cute.arch.mbarrier_wait(mbar + c, 0)
            a_chunk = a_row + c * self.w_chunk_bytes
            x_chunk = x_base + c * self.x_chunk_bytes
            for ks in cutlass.range_constexpr(_K_CHUNK // 16):
                a = _ldsm_x4(a_chunk + (((ks * 2 + a_unit) ^ r) << 4))
                for p in cutlass.range_constexpr(_ceil_div(n_tiles, 2)):
                    rows = (2 * p + (mat >> 1)) * 8 + r
                    if cutlass.const_expr(2 * p + 1 == n_tiles):
                        rows = (2 * p) * 8 + r
                    b = _ldsm_x4(x_chunk + rows * 128 + (((ks * 2 + b_unit) ^ r) << 4))
                    for q in cutlass.range_constexpr(min(2, n_tiles - 2 * p)):
                        nt = 2 * p + q
                        d = _mma_bf16(
                            a,
                            b[2 * q],
                            b[2 * q + 1],
                            (acc[4 * nt], acc[4 * nt + 1], acc[4 * nt + 2], acc[4 * nt + 3]),
                        )
                        for i in cutlass.range_constexpr(4):
                            acc[4 * nt + i] = d[i]

        # ---- Reduce the four warps (fixed order) into tile element o = token * 16 + row. ----
        red = storage.red.data_ptr().toint() + warp * (self.tile_elems * 4)
        for nt in cutlass.range_constexpr(n_tiles):
            for i in cutlass.range_constexpr(4):
                token = nt * 8 + 2 * (lane & 3) + (i & 1)
                row = (lane >> 2) + (i >> 1) * 8
                _sts_f32(red + (token * _GEMV_ROWS + row) * 4, acc[4 * nt + i])
        if cutlass.const_expr(self.use_pdl):
            _griddepcontrol_wait()
        cute.arch.barrier()
        red0 = storage.red.data_ptr().toint()
        per_thread = _ceil_div(self.tile_elems, _GEMV_THREADS)
        tile = []
        for j in cutlass.range_constexpr(per_thread):
            o = tidx + j * _GEMV_THREADS
            v = _lds_f32(red0 + (o % self.tile_elems) * 4)
            for w in cutlass.range_constexpr(1, _GEMV_WARPS):
                v = v + _lds_f32(red0 + (w * self.tile_elems + o % self.tile_elems) * 4)
            tile.append(v)

        if cutlass.const_expr(S == 1):
            self.store_tile(tile, tidx, group, m, out_addr)
        else:
            part = ws_addr + Int64(((group * S + split) * self.tile_elems) * 4)
            for j in cutlass.range_constexpr(per_thread):
                o = tidx + j * _GEMV_THREADS
                if o < self.tile_elems:
                    _st_global_f32(part + Int64(o * 4), tile[j])
            cute.arch.barrier()
            ticket_addr = storage.ticket.data_ptr().toint()
            counter = counter_addr + Int64(group * 4)
            if tidx == 0:
                _fence_acq_rel_gpu()
                arrived = _atom_add_relaxed_gpu(counter, 1)
                if arrived == S - 1:
                    # Last split of this row tile: acquire the others' partials, reset counter.
                    _fence_acq_rel_gpu()
                    _st_relaxed_gpu(counter, 0)
                _sts_b32(ticket_addr, arrived)
            cute.arch.barrier()
            if _lds_b32(ticket_addr) == S - 1:
                parts = ws_addr + Int64((group * S * self.tile_elems) * 4)
                total = []
                for j in cutlass.range_constexpr(per_thread):
                    o = (tidx + j * _GEMV_THREADS) % self.tile_elems
                    v = _ld_relaxed_f32(parts + Int64(o * 4))
                    for sp in cutlass.range_constexpr(1, S):
                        v = v + _ld_relaxed_f32(parts + Int64((sp * self.tile_elems + o) * 4))
                    total.append(v)
                self.store_tile(total, tidx, group, m, out_addr)

    @cute.jit
    def store_tile(self, tile, tidx: Int32, group: Int32, m: Int32, out_addr: Int64):
        for j in cutlass.range_constexpr(_ceil_div(self.tile_elems, _GEMV_THREADS)):
            o = tidx + j * _GEMV_THREADS
            token = o // _GEMV_ROWS
            row = group * _GEMV_ROWS + o % _GEMV_ROWS
            if o < self.tile_elems and token < m and row < self.n:
                idx = token * self.n + row
                if cutlass.const_expr(self.out_fp32):
                    _st_global_f32(out_addr + Int64(idx * 4), tile[j])
                else:
                    _st_global_bf16(out_addr + Int64(idx * 2), tile[j])


# ---------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------


@functools.cache
def _num_sms(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


@functools.cache
def _is_supported_device(device_index: int) -> bool:
    # SM100 / SM103 (Blackwell) and SM107 (Rubin).
    return torch.cuda.get_device_capability(device_index) in ((10, 0), (10, 3), (10, 7))


def num_splits_for(num_sms: int) -> int:
    """K splits of the ``M <= 64`` path: ``8 * S`` CTAs fill the GPU, one CTA per SM."""
    return max(1, min(num_sms // _EXPERT_GROUPS, _NUM_K_CHUNKS))


def umma_num_splits_for(num_sms: int) -> int:
    """K splits of the ``M > 64`` path: ``7 * S`` CTAs, one per SM."""
    return max(1, min(num_sms // _UMMA_GROUPS, _UMMA_NUM_K_CHUNKS))


def _compile_args(m: int):
    dummy = 1 << 20
    addr = Int64(dummy)
    return (
        make_ptr(cutlass.BFloat16, dummy, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.BFloat16, dummy, cute.AddressSpace.gmem, assumed_align=16),
        Int32(m),
        *([addr] * 12),
        Int32(0),
        Int32(NUM_EXPERTS),
        Int32(0),
        Int64(0),
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )


def _workspace_offsets(num_sms: int) -> tuple[int, int]:
    """Byte offsets of the two paths' partial-logit slots in the workspace."""
    small = _IDS_SLOT_BYTES
    return small, small + _slot_bytes(num_splits_for(num_sms), SMALL_M)


@functools.cache
def _compiled_kernel(n_tiles: int, num_sms: int, with_sort: bool, use_pdl: bool):
    offset = _workspace_offsets(num_sms)[0]
    kernel = K3FusedRouterKernel(n_tiles, num_splits_for(num_sms), with_sort, use_pdl, -offset)
    return cute.compile(kernel, *_compile_args(8 * n_tiles))


@functools.cache
def _compiled_umma_kernel(n_pad: int, num_sms: int, with_sort: bool, use_pdl: bool):
    offset = _workspace_offsets(num_sms)[1]
    kernel = K3FusedRouterUmmaKernel(
        n_pad, umma_num_splits_for(num_sms), with_sort, use_pdl, -offset
    )
    return cute.compile(kernel, *_compile_args(n_pad))


def _compiled_router(m: int, num_sms: int, with_sort: bool, use_pdl: bool):
    """``(compiled kernel, byte offset of its partial-logit slots in the workspace)``."""
    if m <= SMALL_M:
        compiled = _compiled_kernel(_ceil_div(m, 8), num_sms, with_sort, use_pdl)
        return compiled, _workspace_offsets(num_sms)[0]
    compiled = _compiled_umma_kernel(16 * _ceil_div(m, 16), num_sms, with_sort, use_pdl)
    return compiled, _workspace_offsets(num_sms)[1]


_WORKSPACES: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}


def _workspace(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Persistent per-device state: ``(arrival state, hand-off slots)``.

    Both GEMM paths share it. Between launches the arrival counter is zero and the
    hand-off slots hold the sentinel (all ones); each launch restores that state, so a
    captured graph can be replayed. Launches on one device must not overlap (they share this state); stream
    order provides that.
    """
    index = device.index if device.index is not None else torch.cuda.current_device()
    state = _WORKSPACES.get(index)
    if state is None:
        num_sms = _num_sms(index)
        size = workspace_bytes(num_splits_for(num_sms), umma_num_splits_for(num_sms))
        counter = torch.zeros(32, dtype=torch.int32, device=device)
        slots = torch.full((size // 4,), -1, dtype=torch.int32, device=device)
        state = (counter, slots)
        _WORKSPACES[index] = state
    return state


def workspace_ready(device: torch.device) -> bool:
    """Whether this device's persistent router state already exists.

    The state must be created outside CUDA graph capture (a capture would allocate it from
    the graph's private pool), so callers that may be capturing check this first.
    """
    index = device.index if device.index is not None else torch.cuda.current_device()
    return index in _WORKSPACES


def sort_buffer_sizes(num_tokens: int, local_num_experts: int, tile_tokens_dim: int = 128):
    """``(max_num_tiles, max_num_permuted_tokens)`` exactly as ``moe_sort`` allocates them."""
    expanded = num_tokens * TOP_K
    filled = min(local_num_experts, expanded)
    max_tiles = filled + (expanded - filled) // tile_tokens_dim
    return max_tiles, max_tiles * tile_tokens_dim


def _check_inputs(
    hidden_states: torch.Tensor, weight: torch.Tensor, e_score_correction_bias: torch.Tensor
) -> None:
    if hidden_states.dtype != torch.bfloat16 or hidden_states.dim() != 2:
        raise ValueError("hidden_states must be a 2D bfloat16 tensor")
    if not hidden_states.is_contiguous() or hidden_states.shape[1] != HIDDEN_SIZE:
        raise ValueError(f"hidden_states must be contiguous [M, {HIDDEN_SIZE}]")
    if not 1 <= hidden_states.shape[0] <= MAX_M:
        raise ValueError(f"M={hidden_states.shape[0]} must be in [1, {MAX_M}]")
    if (
        weight.dtype != torch.bfloat16
        or tuple(weight.shape) != (NUM_EXPERTS, HIDDEN_SIZE)
        or not weight.is_contiguous()
    ):
        raise ValueError(f"weight must be a contiguous bfloat16 [{NUM_EXPERTS}, {HIDDEN_SIZE}]")
    if (
        e_score_correction_bias.dtype != torch.float32
        or e_score_correction_bias.numel() != NUM_EXPERTS
        or not e_score_correction_bias.is_contiguous()
    ):
        raise ValueError(f"e_score_correction_bias must be a contiguous float32 [{NUM_EXPERTS}]")
    for t in (hidden_states, weight, e_score_correction_bias):
        if not t.is_cuda or t.device != hidden_states.device:
            raise ValueError("all inputs must be on the same CUDA device")
        if t.data_ptr() % 16:
            raise ValueError("all inputs must be 16-byte aligned")


def can_use_k3_fused_router(
    hidden_states: torch.Tensor, weight: torch.Tensor, e_score_correction_bias: torch.Tensor
) -> bool:
    """Whether the fused router accepts these inputs (non-raising check)."""
    if not hidden_states.is_cuda or not _is_supported_device(hidden_states.device.index):
        return False
    try:
        _check_inputs(hidden_states, weight, e_score_correction_bias)
    except ValueError:
        return False
    return True


def run_k3_fused_router(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    e_score_correction_bias: torch.Tensor,
    routed_scaling_factor: float,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    logits: torch.Tensor | None = None,
    sort_outputs: tuple[torch.Tensor, ...] | None = None,
    local_expert_offset: int = 0,
    local_num_experts: int = NUM_EXPERTS,
    use_pdl: bool = True,
) -> None:
    """Launch the fused router on the current stream (inputs are not re-validated).

    Args:
        hidden_states: ``bf16 [M, 7168]``, ``1 <= M <= 256``.
        weight: Router weight ``bf16 [896, 7168]`` (``nn.Linear`` layout).
        e_score_correction_bias: ``fp32 [896]``.
        routed_scaling_factor: Multiplier of the renormalized weights.
        topk_ids: Output ``int32 [M, 16]``.
        topk_weights: Output ``fp32 [M, 16]``.
        logits: Optional output ``fp32 [M, 896]``.
        sort_outputs: Optional ``moe_sort`` outputs ``(tile_idx_to_expert_idx,
            tile_idx_to_mn_limit, expanded_idx_to_permuted_idx, permuted_idx_to_expanded_idx,
            total_num_padded_tokens, num_non_exiting_tiles)``, all ``int32``.
        local_expert_offset: First expert of this rank (sort only).
        local_num_experts: Experts of this rank (sort only).
        use_pdl: Launch with programmatic dependent launch.
    """
    m = hidden_states.shape[0]
    device = hidden_states.device
    with_sort = sort_outputs is not None
    compiled, slot_offset = _compiled_router(m, _num_sms(device.index), with_sort, use_pdl)
    counter, slots = _workspace(device)
    if with_sort:
        tile_expert, tile_limit, e2p, p2e, total, num_tiles = (t.data_ptr() for t in sort_outputs)
    else:
        tile_expert = tile_limit = e2p = p2e = total = num_tiles = 0
    scale_bits = struct.unpack("<q", struct.pack("<d", float(routed_scaling_factor)))[0]
    stream = cuda.CUstream(torch.cuda.current_stream(device).cuda_stream)
    compiled(
        make_ptr(cutlass.BFloat16, weight.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(
            cutlass.BFloat16, hidden_states.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
        ),
        Int32(m),
        Int64(e_score_correction_bias.data_ptr()),
        Int64(slots.data_ptr() + slot_offset),
        Int64(counter.data_ptr()),
        Int64(topk_ids.data_ptr()),
        Int64(topk_weights.data_ptr()),
        Int64(logits.data_ptr() if logits is not None else 0),
        Int64(e2p),
        Int64(p2e),
        Int64(tile_expert),
        Int64(tile_limit),
        Int64(total),
        Int64(num_tiles),
        Int32(local_expert_offset),
        Int32(local_num_experts),
        Int32(1 if logits is not None else 0),
        Int64(scale_bits),
        stream,
    )


@torch.library.custom_op("trtllm::k3_fused_router_topk", mutates_args=())
def k3_fused_router_topk(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    e_score_correction_bias: torch.Tensor,
    routed_scaling_factor: float,
    return_logits: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Router GEMM + ``noaux_tc`` top-16 in one launch.

    Args:
        hidden_states: ``bf16 [M, 7168]``, ``1 <= M <= 256``.
        weight: Router weight ``bf16 [896, 7168]``.
        e_score_correction_bias: ``fp32 [896]``.
        routed_scaling_factor: Multiplier of the renormalized weights.
        return_logits: Also return the FP32 router logits.

    Returns:
        ``(topk_ids int32 [M, 16], topk_weights fp32 [M, 16], logits)`` where ``logits`` is
        ``fp32 [M, 896]`` if requested and ``fp32 [0, 896]`` otherwise. Experts are ordered by
        decreasing ``sigmoid(logit) + bias`` like ``noaux_tc_op``.
    """
    _check_inputs(hidden_states, weight, e_score_correction_bias)
    m = hidden_states.shape[0]
    device = hidden_states.device
    topk_ids = torch.empty((m, TOP_K), dtype=torch.int32, device=device)
    topk_weights = torch.empty((m, TOP_K), dtype=torch.float32, device=device)
    logits = torch.empty(
        (m if return_logits else 0, NUM_EXPERTS), dtype=torch.float32, device=device
    )
    run_k3_fused_router(
        hidden_states,
        weight,
        e_score_correction_bias,
        routed_scaling_factor,
        topk_ids,
        topk_weights,
        logits=logits if return_logits else None,
    )
    return topk_ids, topk_weights, logits


@k3_fused_router_topk.register_fake
def _(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    e_score_correction_bias: torch.Tensor,
    routed_scaling_factor: float,
    return_logits: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m = hidden_states.shape[0]
    return (
        hidden_states.new_empty((m, TOP_K), dtype=torch.int32),
        hidden_states.new_empty((m, TOP_K), dtype=torch.float32),
        hidden_states.new_empty((m if return_logits else 0, NUM_EXPERTS), dtype=torch.float32),
    )


@torch.library.custom_op("trtllm::k3_fused_router_sort", mutates_args=())
def k3_fused_router_sort(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    e_score_correction_bias: torch.Tensor,
    routed_scaling_factor: float,
    local_expert_offset: int,
    local_num_experts: int,
    tile_tokens_dim: int = 128,
    return_logits: bool = False,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Router GEMM + ``noaux_tc`` top-16 + ``moe_sort`` in one launch.

    Args:
        hidden_states: ``bf16 [M, 7168]``, ``1 <= M <= 256``.
        weight: Router weight ``bf16 [896, 7168]``.
        e_score_correction_bias: ``fp32 [896]``.
        routed_scaling_factor: Multiplier of the renormalized weights.
        local_expert_offset: First expert of this rank.
        local_num_experts: Experts of this rank.
        tile_tokens_dim: Must be 128.
        return_logits: Also return the FP32 router logits.

    Returns:
        ``(topk_ids, topk_weights, tile_idx_to_expert_idx, tile_idx_to_mn_limit,
        expanded_idx_to_permuted_idx, permuted_idx_to_expanded_idx, total_num_padded_tokens,
        num_non_exiting_tiles, logits)``. The six sort outputs have ``moe_sort``'s shapes and
        dtypes; like ``moe_sort`` only valid entries are written (tile arrays up to
        ``num_non_exiting_tiles``, permuted rows below each tile's ``mn_limit``). Tokens of
        an expert are in ascending token order.
    """
    _check_inputs(hidden_states, weight, e_score_correction_bias)
    if tile_tokens_dim != 128:
        raise ValueError("tile_tokens_dim must be 128")
    if not (
        0 <= local_expert_offset
        and 1 <= local_num_experts
        and local_expert_offset + local_num_experts <= NUM_EXPERTS
    ):
        raise ValueError("invalid local expert range")
    m = hidden_states.shape[0]
    device = hidden_states.device
    max_tiles, max_permuted = sort_buffer_sizes(m, local_num_experts, tile_tokens_dim)
    i32 = dict(dtype=torch.int32, device=device)
    topk_ids = torch.empty((m, TOP_K), **i32)
    topk_weights = torch.empty((m, TOP_K), dtype=torch.float32, device=device)
    tile_expert = torch.empty((max_tiles,), **i32)
    tile_limit = torch.empty((max_tiles,), **i32)
    e2p = torch.empty((m, TOP_K), **i32)
    p2e = torch.empty((max_permuted,), **i32)
    total = torch.empty((1,), **i32)
    num_tiles = torch.empty((1,), **i32)
    logits = torch.empty(
        (m if return_logits else 0, NUM_EXPERTS), dtype=torch.float32, device=device
    )
    run_k3_fused_router(
        hidden_states,
        weight,
        e_score_correction_bias,
        routed_scaling_factor,
        topk_ids,
        topk_weights,
        logits=logits if return_logits else None,
        sort_outputs=(tile_expert, tile_limit, e2p, p2e, total, num_tiles),
        local_expert_offset=local_expert_offset,
        local_num_experts=local_num_experts,
    )
    return topk_ids, topk_weights, tile_expert, tile_limit, e2p, p2e, total, num_tiles, logits


@k3_fused_router_sort.register_fake
def _(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    e_score_correction_bias: torch.Tensor,
    routed_scaling_factor: float,
    local_expert_offset: int,
    local_num_experts: int,
    tile_tokens_dim: int = 128,
    return_logits: bool = False,
):
    m = hidden_states.shape[0]
    max_tiles, max_permuted = sort_buffer_sizes(m, local_num_experts, tile_tokens_dim)

    def i32(*shape):
        return hidden_states.new_empty(shape, dtype=torch.int32)

    return (
        i32(m, TOP_K),
        hidden_states.new_empty((m, TOP_K), dtype=torch.float32),
        i32(max_tiles),
        i32(max_tiles),
        i32(m, TOP_K),
        i32(max_permuted),
        i32(1),
        i32(1),
        hidden_states.new_empty((m if return_logits else 0, NUM_EXPERTS), dtype=torch.float32),
    )


# ---------------------------------------------------------------------------
# Standalone GEMV host side
# ---------------------------------------------------------------------------

#: A row tile is split over CTAs only if one CTA would otherwise stream more weight than this.
_GEMV_SPLIT_MIN_BYTES = 16 * 1024
_GEMV_MAX_GROUPS = 4096


def gemv_num_splits(n: int, k: int, m: int, num_sms: int) -> int:
    """Deterministic K split for :func:`k3_bf16_lowm_gemv` (one wave of CTAs)."""
    groups = _ceil_div(n, _GEMV_ROWS)
    chunks = k // _K_CHUNK
    m_pad = 8 * _ceil_div(m, 8)
    per_chunk = _GEMV_ROWS * _K_CHUNK * 2 + m_pad * _K_CHUNK * 2
    min_splits = _ceil_div(chunks, max(1, _GEMV_SMEM_BUDGET // per_chunk))
    splits = 1
    if chunks * _GEMV_ROWS * _K_CHUNK * 2 > _GEMV_SPLIT_MIN_BYTES:
        # Every split re-reads its K slice of x and adds a partial tile to the fixup; past
        # ~256 / m_pad splits that costs more than the extra weight bandwidth buys (GB300,
        # N=136, K=7168: M=7 is best at 16 splits, M=28/56 at 8).
        splits = max(1, min(num_sms // groups, max(8, 256 // m_pad)))
    return min(chunks, max(splits, min_splits))


@functools.cache
def _compiled_gemv(n: int, k: int, n_tiles: int, num_splits: int, out_fp32: bool, use_pdl: bool):
    kernel = K3LowMGemvKernel(n, k, n_tiles, num_splits, out_fp32, use_pdl)
    dummy = 1 << 20
    w_ptr = make_ptr(cutlass.BFloat16, dummy, cute.AddressSpace.gmem, assumed_align=16)
    x_ptr = make_ptr(cutlass.BFloat16, dummy, cute.AddressSpace.gmem, assumed_align=16)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    return cute.compile(
        kernel, w_ptr, x_ptr, Int32(8 * n_tiles), Int64(dummy), Int64(dummy), Int64(dummy), stream
    )


_GEMV_COUNTERS: dict[int, torch.Tensor] = {}


def _gemv_counters(device: torch.device) -> torch.Tensor:
    """Per-device split-K arrival counters (zero between launches; single-stream use)."""
    index = device.index if device.index is not None else torch.cuda.current_device()
    counters = _GEMV_COUNTERS.get(index)
    if counters is None:
        counters = torch.zeros(_GEMV_MAX_GROUPS, dtype=torch.int32, device=device)
        _GEMV_COUNTERS[index] = counters
    return counters


def _check_gemv_inputs(x: torch.Tensor, w: torch.Tensor) -> None:
    if x.dtype != torch.bfloat16 or x.dim() != 2 or not x.is_contiguous():
        raise ValueError("x must be a contiguous 2D bfloat16 tensor")
    if w.dtype != torch.bfloat16 or w.dim() != 2 or not w.is_contiguous():
        raise ValueError("w must be a contiguous 2D bfloat16 tensor")
    n, k = w.shape
    if x.shape[1] != k:
        raise ValueError(f"x has K={x.shape[1]}, w has K={k}")
    if n % 8 or k % 128 or _ceil_div(n, _GEMV_ROWS) > _GEMV_MAX_GROUPS:
        raise ValueError(f"N={n} must be a multiple of 8 (<= 65536) and K={k} of 128")
    if not 1 <= x.shape[0] <= GEMV_MAX_M:
        raise ValueError(f"M={x.shape[0]} must be in [1, {GEMV_MAX_M}]")
    if not x.is_cuda or w.device != x.device or x.data_ptr() % 16 or w.data_ptr() % 16:
        raise ValueError("x and w must be 16-byte aligned tensors on the same CUDA device")


def run_k3_bf16_lowm_gemv(
    x: torch.Tensor, w: torch.Tensor, out: torch.Tensor, *, use_pdl: bool = True
) -> torch.Tensor:
    """Launch ``out[M, N] = x[M, K] @ w[N, K]^T`` into ``out`` (bf16 or fp32) on the current stream."""
    m, k = x.shape
    n = w.shape[0]
    device = x.device
    splits = gemv_num_splits(n, k, m, _num_sms(device.index))
    n_tiles = _ceil_div(m, 8)
    compiled = _compiled_gemv(n, k, n_tiles, splits, out.dtype == torch.float32, use_pdl)
    ws = None
    if splits > 1:
        ws = torch.empty(
            (_ceil_div(n, _GEMV_ROWS) * splits * _GEMV_ROWS * 8 * n_tiles,),
            dtype=torch.float32,
            device=device,
        )
    compiled(
        make_ptr(cutlass.BFloat16, w.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.BFloat16, x.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
        Int32(m),
        Int64(out.data_ptr()),
        Int64(ws.data_ptr() if ws is not None else 0),
        Int64(_gemv_counters(device).data_ptr()),
        cuda.CUstream(torch.cuda.current_stream(device).cuda_stream),
    )
    return out


@torch.library.custom_op("trtllm::k3_bf16_lowm_gemv", mutates_args=())
def k3_bf16_lowm_gemv(
    x: torch.Tensor, w: torch.Tensor, out_dtype: torch.dtype = torch.bfloat16
) -> torch.Tensor:
    """``out[M, N] = x[M, K] @ w[N, K]^T`` for small ``M`` (FP32 accumulation).

    Args:
        x: ``bf16 [M, K]``, ``1 <= M <= 64``, ``K`` a multiple of 128.
        w: ``bf16 [N, K]`` (``nn.Linear`` layout), ``N`` a multiple of 8.
        out_dtype: ``torch.bfloat16`` or ``torch.float32``.

    Returns:
        ``[M, N]`` in ``out_dtype``.
    """
    _check_gemv_inputs(x, w)
    if out_dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("out_dtype must be bfloat16 or float32")
    out = torch.empty((x.shape[0], w.shape[0]), dtype=out_dtype, device=x.device)
    return run_k3_bf16_lowm_gemv(x, w, out)


@k3_bf16_lowm_gemv.register_fake
def _(x: torch.Tensor, w: torch.Tensor, out_dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    return x.new_empty((x.shape[0], w.shape[0]), dtype=out_dtype)


def can_use_k3_bf16_lowm_gemv(x: torch.Tensor, w: torch.Tensor) -> bool:
    """Whether :func:`k3_bf16_lowm_gemv` accepts these inputs (non-raising check)."""
    if not x.is_cuda or not _is_supported_device(x.device.index):
        return False
    try:
        _check_gemv_inputs(x, w)
    except ValueError:
        return False
    return True


__all__ = [
    "HIDDEN_SIZE",
    "K3LowMGemvKernel",
    "can_use_k3_bf16_lowm_gemv",
    "gemv_num_splits",
    "k3_bf16_lowm_gemv",
    "run_k3_bf16_lowm_gemv",
    "MAX_M",
    "NUM_EXPERTS",
    "TOP_K",
    "K3FusedRouterKernel",
    "can_use_k3_fused_router",
    "k3_fused_router_sort",
    "k3_fused_router_topk",
    "num_splits_for",
    "run_k3_fused_router",
    "sort_buffer_sizes",
]
