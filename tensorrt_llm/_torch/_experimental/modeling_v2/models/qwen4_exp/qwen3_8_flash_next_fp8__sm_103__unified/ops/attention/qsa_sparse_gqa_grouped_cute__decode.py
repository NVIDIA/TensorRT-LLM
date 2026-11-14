# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""QSA sparse GQA over an e4m3 paged K/V pool for MTP verify rows, request-grouped: CuTe DSL
(tcgen05) split-K over the union of a request's rows' selections, then a row merge.

Derived from `qsa_paged_sparse_gqa_nvfp4_verify` (same warp roles, tcgen05 pipeline, softmax,
epilogue and row merge) with an exact e4m3 -> f16 widening, and the union built in a shared hash
table instead of a token bitmap, so any block-table width is served.

One CTA serves one (group of T = `rows_per_request` rows, KV head, split): 128 MMA rows, row
m = t * G + h (verify row t, q head h of the KV head's G; rows past T * G idle), over the group's
union selection, so each selected K/V token is loaded and widened once for the group's rows. The
union lists the first row's four-token groups in column order, then each later row's new ones
(deterministic whatever order the hash atomics ran in); per (tile, verify row) a 64-bit selection
word masks positions the row did not select to -inf, so the softmax offset is the row's own max
and causality is the selection's. Splits take interleaved tiles of the union.

Roles (11 warps): 0-3 softmax and epilogue (one thread an MMA row), 4 the tcgen05 MMAs, 5 / 6 the
K / V bulk copies into the raw ring, 7-10 the e4m3 -> f16 widening. Tensor memory: O in columns
0-255, S/P buffers at 256 and 320, Q at 384. The hash lives in raw slot 0 and the staged q rows
and the union list in raw slot 1 until the tiles start: one CTA an SM.
"""

from __future__ import annotations

import functools
from typing import NamedTuple, Optional

import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.tcgen05 as tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass import const_expr
from cutlass._mlir.dialects import llvm
from cutlass._mlir.dialects import nvvm as _nvvm
from cutlass.cute.nvgpu import OperandMajorMode
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.typing import Float32, Int32, Int64
from cutlass.cutlass_dsl import T, dsl_user_op

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

from .qsa_sparse_gqa_splitk_cute__decode import (
    MAX_SPLITS,
    _asm,
    _bf2_hi,
    _bf2_lo,
    _bf2_pack,
    _bf2_to_f2,
    _bits_f32,
    _cp16z,
    _e4m3x4,
    _f32_bits,
    _ir,
    _ldg1,
    _ldg4,
    _lds1,
    _lds4,
    _ptr,
    _round_bf16,
    _stg1,
    _sts1,
    _sts4,
    qsa_sparse_gqa_splitk_cute,
)

try:
    from cutlass.memory import SmemAllocator
except ImportError:  # DSL releases before the memory namespace split
    from cutlass.utils import SmemAllocator

F16 = cutlass.Float16
LOG2E = 1.4426950408889634
HEAD_DIM = 256
#: MMA rows a CTA (T verify rows x the KV head's q heads) and positions a tile.
BM = 128
BN = 64
#: Four-token groups a tile.
GPT = BN // 4
#: Bytes of a raw e4m3 K or V row in shared memory: unpadded (a four-token group lands as one
#: 1 KB bulk copy; the widening rotates its reads instead).
RAW_LD = 256
RAW_K = BN * RAW_LD
#: One raw slot: K rows, V rows.
RAW_BUF = 2 * RAW_K
#: Bytes of one f16 K or V slot (64 positions x 256 dims).
F16_SLOT = BN * HEAD_DIM * 2
#: The rings: raw e4m3 tiles in flight, f16 K slots, f16 V slots (32 KB each, 192 KB in all).
#: The gather is bound by the bytes in flight an SM, so the raw ring takes the room; one f16 slot
#: each suffices once the widening runs a tile ahead of the MMA (fastest split measured on GB300).
RAW_SLOTS = 4
K_SLOTS = 1
V_SLOTS = 1
#: Shared memory a CTA may claim (227 KB), less barriers, scan scratch and alignment.
SMEM_BUDGET = 232448 - 4096
#: A row's reference max moves only past this many log2 units (P <= 2^8, exact in f16 range).
RESCALE_TH = 8.0
SOFT_WARPS = 4
MMA_WARP = 4
LOAD_WARP0 = 5
#: Warps issuing the bulk copies: warp 5 the K groups, warp 6 the V groups (lanes 0-15, one group
#: a lane); a warp's bulk copies issue one lane after another, so two warps halve the issue time.
LOAD_WARPS = 2
DQ_WARP0 = 7
DQ_WARPS = 4
THREADS = 32 * (DQ_WARP0 + DQ_WARPS)
TMEM_O = 0
TMEM_S0 = 256
TMEM_Q = 384
TMEM_COLS = 512
CHUNK_COLS = 32
#: Bytes of a staged q row (bf16, 512 + 16 of padding).
Q_LD = 528
#: Hash slots of the union (keys, then masks, in raw slot 0): twice the most groups T = 4 rows
#: name, so a probe chain stays short.
HASH = 4096
#: Most rows a group (the hash mask holds four bits a row).
MAX_RPR = 4
#: Most splits of one (group, KV head) union.
MAX_PLAN_SPLITS = 32
#: Most splits of a last-wave unit (tail mode).
MAX_TAIL_SPLITS = 4
#: Fewer request groups than this take the per-row kernel.
MIN_GROUPS = 8
#: Scan scratch words: (row, column round) x warp counts.
SCAN_WORDS = 256
# --- inline PTX beyond the per-row kernel's ---


@dsl_user_op
def _popc(x, *, loc=None, ip=None):
    return cutlass.Int32(
        _asm(T.i32(), [_ir(x, cutlass.Int32, loc, ip)], "popc.b32 $0, $1;", "=r,r", loc, ip)
    )


@dsl_user_op
def _ctz(x, *, loc=None, ip=None):
    """Trailing zeros of a nonzero word."""
    return cutlass.Int32(
        _asm(
            T.i32(),
            [_ir(x, cutlass.Int32, loc, ip)],
            "{ .reg .b32 t; brev.b32 t, $1; clz.b32 $0, t; }",
            "=r,r",
            loc,
            ip,
        )
    )


@dsl_user_op
def _ballot(x, *, loc=None, ip=None):
    """`vote.sync.ballot` of `x != 0` over the full warp."""
    return cutlass.Int32(
        _asm(
            T.i32(),
            [_ir(x, cutlass.Int32, loc, ip)],
            "{ .reg .pred p; setp.ne.u32 p, $1, 0; vote.sync.ballot.b32 $0, p, 0xffffffff; }",
            "=r,r",
            loc,
            ip,
            side=True,
        )
    )


@dsl_user_op
def _sts_u16(saddr, a, *, loc=None, ip=None):
    _asm(
        T.i32(),
        [_ir(saddr, cutlass.Int32, loc, ip), _ir(a, cutlass.Int32, loc, ip)],
        "{ .reg .b16 h; cvt.u16.u32 h, $2; st.shared.u16 [$1], h; mov.u32 $0, 0; }",
        "=r,r,r",
        loc,
        ip,
        side=True,
    )


@dsl_user_op
def _lds_u16(saddr, *, loc=None, ip=None):
    return cutlass.Int32(
        _asm(
            T.i32(),
            [_ir(saddr, cutlass.Int32, loc, ip)],
            "{ .reg .b16 h; ld.shared.u16 h, [$1]; cvt.u32.u16 $0, h; }",
            "=r,r",
            loc,
            ip,
            side=True,
        )
    )


@dsl_user_op
def _cas_shared(saddr, cmp, val, *, loc=None, ip=None):
    """`atom.shared.cas.b32`: the old word at `saddr`."""
    return cutlass.Int32(
        _asm(
            T.i32(),
            [
                _ir(saddr, cutlass.Int32, loc, ip),
                _ir(cmp, cutlass.Int32, loc, ip),
                _ir(val, cutlass.Int32, loc, ip),
            ],
            "atom.shared.cas.b32 $0, [$1], $2, $3;",
            "=r,r,r,r",
            loc,
            ip,
            side=True,
        )
    )


@dsl_user_op
def _red_or_shared(saddr, v, *, loc=None, ip=None):
    _asm(
        T.i32(),
        [_ir(saddr, cutlass.Int32, loc, ip), _ir(v, cutlass.Int32, loc, ip)],
        "{ red.shared.or.b32 [$1], $2; mov.u32 $0, 0; }",
        "=r,r,r",
        loc,
        ip,
        side=True,
    )


@dsl_user_op
def _bulk_g2s(saddr, gaddr, nbytes, mbar, *, loc=None, ip=None):
    """`cp.async.bulk` global -> this CTA's shared memory, completing `nbytes` on `mbar`."""
    _asm(
        T.i32(),
        [
            _ir(saddr, cutlass.Int32, loc, ip),
            _ir(gaddr, cutlass.Int64, loc, ip),
            _ir(nbytes, cutlass.Int32, loc, ip),
            _ir(mbar, cutlass.Int32, loc, ip),
        ],
        "{ cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [$1], [$2], $3, [$4]; "
        "mov.u32 $0, 0; }",
        "=r,r,l,r,r",
        loc,
        ip,
        side=True,
    )


@dsl_user_op
def _arrive_expect_tx(mbar, nbytes, *, loc=None, ip=None):
    _asm(
        T.i32(),
        [_ir(mbar, cutlass.Int32, loc, ip), _ir(nbytes, cutlass.Int32, loc, ip)],
        "{ mbarrier.arrive.expect_tx.shared::cta.b64 _, [$1], $2; mov.u32 $0, 0; }",
        "=r,r,r",
        loc,
        ip,
        side=True,
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
def _stg4f(gaddr, a, b, c, d, *, loc=None, ip=None):
    _asm(
        T.i32(),
        [_ir(gaddr, cutlass.Int64, loc, ip)]
        + [_ir(v, cutlass.Float32, loc, ip) for v in (a, b, c, d)],
        "{ st.global.v4.f32 [$1], {$2, $3, $4, $5}; mov.u32 $0, 0; }",
        "=r,l,f,f,f,f",
        loc,
        ip,
        side=True,
    )


@dsl_user_op
def _ldg2(gaddr, *, loc=None, ip=None):
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 2),
        [_ir(gaddr, cutlass.Int64, loc, ip)],
        "ld.global.v2.b32 {$0, $1}, [$2];",
        "=r,=r,l",
        loc,
        ip,
        side=True,
    )
    return [cutlass.Int32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip)) for i in range(2)]


@dsl_user_op
def _stg2(gaddr, a, b, *, loc=None, ip=None):
    _asm(
        T.i32(),
        [
            _ir(gaddr, cutlass.Int64, loc, ip),
            _ir(a, cutlass.Int32, loc, ip),
            _ir(b, cutlass.Int32, loc, ip),
        ],
        "{ st.global.v2.b32 [$1], {$2, $3}; mov.u32 $0, 0; }",
        "=r,l,r,r",
        loc,
        ip,
        side=True,
    )


@dsl_user_op
def _red_shared(op, saddr, v, *, loc=None, ip=None):
    """red.shared.<op>.s32 (op: add, min, max)."""
    _asm(
        T.i32(),
        [_ir(saddr, cutlass.Int32, loc, ip), _ir(v, cutlass.Int32, loc, ip)],
        "{ red.shared.%s.s32 [$1], $2; mov.u32 $0, 0; }" % op,
        "=r,r,r",
        loc,
        ip,
        side=True,
    )


@dsl_user_op
def _ldg1cg(gaddr, *, loc=None, ip=None):
    """4-byte global load at L2 (`.cg`): data another CTA of this launch wrote."""
    return cutlass.Int32(
        _asm(
            T.i32(),
            [_ir(gaddr, cutlass.Int64, loc, ip)],
            "ld.global.cg.b32 $0, [$1];",
            "=r,l",
            loc,
            ip,
            side=True,
        )
    )


@dsl_user_op
def _ldg4cg(gaddr, *, loc=None, ip=None):
    """16-byte global load at L2 (`.cg`): data another CTA of this launch wrote."""
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [_ir(gaddr, cutlass.Int64, loc, ip)],
        "ld.global.cg.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        loc,
        ip,
        side=True,
    )
    return [cutlass.Int32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip)) for i in range(4)]


@dsl_user_op
def _atom_add_acqrel_g(gaddr, v, *, loc=None, ip=None):
    """atomicAdd on a global int32, acq_rel at GPU scope (after a CTA barrier it releases every
    thread's prior stores, CUTLASS's generic-barrier pattern); returns the old value."""
    return cutlass.Int32(
        _asm(
            T.i32(),
            [_ir(gaddr, cutlass.Int64, loc, ip), _ir(v, cutlass.Int32, loc, ip)],
            "atom.acq_rel.gpu.global.add.u32 $0, [$1], $2;",
            "=r,l,r",
            loc,
            ip,
            side=True,
        )
    )


def _spread4(nib):
    """Bits 0..3 of `nib` to bits 0, 8, 16, 24."""
    return (nib & 1) | ((nib & 2) << 7) | ((nib & 4) << 14) | ((nib & 8) << 21)


@dsl_user_op
def _tc_fence_before(*, loc=None, ip=None):
    _nvvm.tcgen05_fence(_nvvm.Tcgen05FenceKind.BEFORE_THREAD_SYNC, loc=loc, ip=ip)


@dsl_user_op
def _tc_fence_after(*, loc=None, ip=None):
    _nvvm.tcgen05_fence(_nvvm.Tcgen05FenceKind.AFTER_THREAD_SYNC, loc=loc, ip=ip)


#: Swizzle<B, 4, 3>: B -> the descriptor's layout type (none, 32B, 64B, 128B).
_SW_TYPES = {0: 0, 1: 6, 2: 4, 3: 2}
#: Layout type -> swizzle atom size along MN (16-byte units).
_SW_ATOM_MN = {0: 1, 6: 2, 4: 4, 2: 8}


def _instr_desc(op) -> int:
    """The 32-bit instruction descriptor of an f16/bf16 x f16/bf16 -> f32 tcgen05 MMA op."""
    fmt = {cutlass.Float16: 0, cutlass.BFloat16: 1}
    a_major = 0 if op.a_major_mode == tcgen05.OperandMajorMode.K else 1
    b_major = 0 if op.b_major_mode == tcgen05.OperandMajorMode.K else 1
    m, n = op.shape_mnk[0], op.shape_mnk[1]
    assert op.acc_dtype is cutlass.Float32 and m in (64, 128) and n % 8 == 0 and 8 <= n <= 256
    d = 0
    d |= 1 << 4  # c_format f32
    d |= fmt[op.a_dtype] << 7
    d |= fmt[op.b_dtype] << 10
    d |= a_major << 15
    d |= b_major << 16
    d |= (n >> 3) << 17
    d |= (m >> 4) << 24
    return d


def _smem_desc_base(layout_u128, swizzle, k_major: bool) -> int:
    """The 64-bit shared-memory descriptor without its start address, for an operand whose
    (MN, K) layout `layout_u128` is in 16-byte units (CUTLASS `mma_sm100_desc.hpp`)."""
    assert swizzle.num_base == 4 and swizzle.num_shift == 3
    ltype = _SW_TYPES[swizzle.num_bits]
    atom_mn = _SW_ATOM_MN[ltype]
    if not k_major:
        canon = cute.logical_divide(layout_u128, (atom_mn, 8))
        s00 = canon.stride[0][0]
        s10 = canon.stride[1][0]
        assert (ltype == 0 or s00 == 1) and s10 == atom_mn, "not a canonical UMMA MN layout"
        s01, s11 = canon.stride[0][1], canon.stride[1][1]
        sbo, lbo = (s01, s11) if ltype == 0 else (s11, s01)
    else:
        canon = cute.logical_divide(layout_u128, (8, 2))
        s00 = canon.stride[0][0]
        s10 = canon.stride[1][0]
        assert s00 == atom_mn and (ltype == 0 or s10 == 1), "not a canonical UMMA K layout"
        sbo, lbo = canon.stride[0][1], s10
    d = 0
    d |= (lbo & 0x3FFF) << 16
    d |= (sbo & 0x3FFF) << 32
    d |= 1 << 46  # descriptor version
    d |= (ltype & 0x7) << 61
    return d


def _gemm_ts(op, acc_tmem, a_tmem, tCrA, tCrB, sB, zero_init: bool):
    """acc (+)= A B over every K step: A in tensor memory at `a_tmem` (fragment `tCrA`), B the
    shared-memory operand `sB` (fragment `tCrB`, one stage). Issued by one elected thread of
    the calling warp; the K loop unrolls into one PTX block (precomputed descriptors)."""
    idesc = _instr_desc(op)
    k_major_b = op.b_major_mode == tcgen05.OperandMajorMode.K
    base_b = _smem_desc_base(
        cute.recast_layout(128, op.b_dtype.width, sB.layout[0]),
        sB.iterator.type.swizzle_type,
        k_major_b,
    )
    b_lo = base_b & 0xFFFFFFFF
    b_hi = (base_b >> 32) & 0xFFFFFFFF
    a_layout = cute.recast_layout(32, tCrA.element_type.width, tCrA.layout)
    n_k = cute.size(tCrA.shape[2])
    off_a = [cute.crd2idx((0, 0, k), a_layout) for k in range(n_k)]
    off_b = [cute.crd2idx((0, 0, k), tCrB.layout) for k in range(n_k)]
    start_b = Int32(b_lo) | Int32((sB[None, None, 0].iterator.toint() & 0x3FFFF) >> 4)
    body = "{\n\t.reg .pred leader;\n\t.reg .b32 db_lo;\n\t.reg .b64 db;\n\telect.sync _|leader, -1;\n\t"
    for k in range(n_k):
        acc = "0" if (zero_init and k == 0) else "1"
        body += (
            f"add.u32 db_lo, $2, {hex(off_b[k])};\n\t"
            f"mov.b64 db, {{db_lo, {hex(b_hi)}}};\n\t"
            f"@leader tcgen05.mma.cta_group::1.kind::f16 [$0], [$1 + {hex(off_a[k])}], db, "
            f"{hex(idesc)}, {acc};\n\t"
        )
    body += "}\n"
    llvm.inline_asm(
        None,
        [
            Int32(cute.arch.make_warp_uniform(acc_tmem)).ir_value(),
            Int32(cute.arch.make_warp_uniform(a_tmem)).ir_value(),
            Int32(cute.arch.make_warp_uniform(start_b)).ir_value(),
        ],
        body,
        "r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


# --- the grouped kernel ---


class _Layout(NamedTuple):
    """The shared-memory plan of one compile: the union's group capacity, local tiles a CTA may
    own in one pass, the union-list offset in raw slot 1 and the table size in int32 words."""

    ug_max: int
    tmax: int
    u_ug: int
    tab_words: int


def _layout(rpr: int, group: int, width: int, splits: int, rings=None):
    """The kernel's shared-memory plan for `rpr` rows a group over a `width`-column expanded
    selection, or None where it does not fit: the hash (keys and masks) fills raw slot 0, the
    staged q rows and the union list share raw slot 1."""
    raw_n, ks_n, vs_n = rings or (RAW_SLOTS, K_SLOTS, V_SLOTS)
    if rpr < 1 or rpr > MAX_RPR or rpr * group > BM:
        return None
    # an expanded selection names at most ceil(width / 4) four-token groups a row
    ug_max = rpr * -(-width // 4)
    tmax = -(-(-(-ug_max // GPT)) // splits)
    # the union list: raw slot 2 where the ring has one, else after the q rows in raw slot 1
    u_ug = RAW_BUF if raw_n >= 3 else -(-(rpr * group * Q_LD) // 16) * 16
    tab_words = 2 * tmax * GPT + 2 * tmax * rpr
    fits = (
        2 * HASH * 4 <= RAW_BUF
        and 5 * ug_max <= 3 * HASH
        and rpr * group * Q_LD <= RAW_BUF
        and u_ug + ug_max * 2 <= RAW_BUF * (2 if raw_n >= 3 else 1)
        and rpr * -(-(-(-width // 4)) // THREADS) <= SCAN_WORDS // (THREADS // 32)
        and 4 * tab_words + 4 * SCAN_WORDS + raw_n * RAW_BUF + (ks_n + vs_n) * F16_SLOT
        <= SMEM_BUDGET
        and raw_n >= 2
        and rpr * group * Q_LD <= ks_n * F16_SLOT
    )
    return _Layout(ug_max, tmax, u_ug, tab_words) if fits else None


def _sigmoid_fast(x):
    """1 / (1 + 2^(-x log2 e)) on ex2.approx and rcp.approx (relative error ~1e-6)."""
    return cute.arch.rcp_approx(1.0 + cute.math.exp2(-x * LOG2E, fastmath=True))


def _tree_fmax(vals):
    """Max of a short list, pairwise (traced as Python)."""
    while len(vals) > 1:
        vals = [
            cute.arch.fmax(vals[j], vals[j + 1]) if j + 1 < len(vals) else vals[j]
            for j in range(0, len(vals), 2)
        ]
    return vals[0]


def _tree_sum(vals):
    """Sum of a short list, pairwise (traced as Python)."""
    while len(vals) > 1:
        vals = [
            vals[j] + vals[j + 1] if j + 1 < len(vals) else vals[j] for j in range(0, len(vals), 2)
        ]
    return vals[0]


def _phase(g, n):
    """Parity of use `g` of a ring of `n` slots."""
    return (g // n) % 2


class _GroupedKernel:
    def __init__(
        self,
        *,
        rpr,
        group,
        kv_heads,
        width,
        tpb,
        splits,
        tail,
        scale,
        num_pages,
        bt_width,
        num_requests,
        k_strides,
        q_strides,
        req64,
        gated,
        pdl,
        bal=False,
        bal_ctas=0,
    ):
        self.RS, self.KS, self.VS = RAW_SLOTS, K_SLOTS, V_SLOTS
        lay = _layout(
            rpr, group, width, 1 if (tail or bal) else splits, (self.RS, self.KS, self.VS)
        )
        assert lay is not None, "no shared-memory plan for this geometry (see `_layout`)"
        assert tpb & (tpb - 1) == 0 and tpb % 4 == 0, "a power-of-two page of whole groups"
        self.T = rpr
        self.group, self.kvh, self.width, self.tpb = group, kv_heads, width, tpb
        self.splits = splits
        #: tail mode: CTAs below `n_whole` serve whole units, the rest `splits`-way splits
        self.tail = tail
        #: balanced mode: each CTA's unit and split from `_bal_schedule`; split units merge in
        #: `fixup`
        self.bal = bal
        self.bal_ctas = bal_ctas
        #: histogram bins of the schedule: a unit's union tiles are at most rpr * ceil(width / 4)
        #: groups / GPT
        self.hmax = -(-(-(-(rpr * -(-width // 4)) // GPT) + 1) // 32) * 32
        self.scale_log2 = scale * LOG2E
        self.num_pages, self.bt_width, self.num_requests = num_pages, bt_width, num_requests
        self.ks, self.qs = k_strides, q_strides
        self.req64, self.gated, self.pdl = req64, gated, pdl
        self.heads = group * kv_heads
        self.rows_m = rpr * group
        #: softmax warps holding live MMA rows (the rest only keep the barriers' counts)
        self.live_warps = -(-self.rows_m // 32)
        self.ug_max, self.tmax = lay.ug_max, lay.tmax
        self.raw_bytes = self.RS * RAW_BUF
        # union scratch: the hash in raw slot 0, the q rows then the union list in raw slot 1
        self.u_hk = 0
        self.u_hm = HASH * 4
        self.u_q = RAW_BUF
        self.u_ug = RAW_BUF + lay.u_ug
        # tables (int32 words). Per group: its first token's pool row (page * tpb + slot, -1
        # when it does not resolve) and its (token, row) selection bytes; per (tile, row): the
        # 64-bit selection word.
        self.t_tok = 0
        self.t_gmask = self.tmax * GPT
        self.t_bits = 2 * self.tmax * GPT
        self.tab_words = lay.tab_words

    @cute.jit
    def __call__(
        self,
        sel_t: cute.Tensor,
        q_ptr: cute.Pointer,
        k_ptr: cute.Pointer,
        v_ptr: cute.Pointer,
        bt_ptr: cute.Pointer,
        req_ptr: cute.Pointer,
        out_ptr: cute.Pointer,
        part_ptr: cute.Pointer,
        lse_ptr: cute.Pointer,
        gate_ptr: cute.Pointer,
        rows: cutlass.Int32,
        sel_stride: cutlass.Int32,
        gate_stride: cutlass.Int32,
        gate_hs: cutlass.Int32,
        n_whole: cutlass.Int32,
        tail_row0: cutlass.Int32,
        n_ctas: cutlass.Int32,
        tab_ptr: cute.Pointer,
        cnt_ptr: cute.Pointer,
        est_ptr: cute.Pointer,
        stream,
    ):
        def qk():
            return sm100_utils.make_trivial_tiled_mma(
                F16,
                F16,
                OperandMajorMode.K,
                OperandMajorMode.K,
                Float32,
                tcgen05.CtaGroup.ONE,
                (BM, BN),
                tcgen05.OperandSource.TMEM,
            )

        def pv():
            return sm100_utils.make_trivial_tiled_mma(
                F16,
                F16,
                OperandMajorMode.K,
                OperandMajorMode.MN,
                Float32,
                tcgen05.CtaGroup.ONE,
                (BM, HEAD_DIM),
                tcgen05.OperandSource.TMEM,
            )

        qk_mma, qk_mma_acc, pv_mma_acc = qk(), qk(), pv()
        q_layout = sm100_utils.make_smem_layout_a(qk_mma, (BM, BN, HEAD_DIM), F16, 1)
        k_layout = sm100_utils.make_smem_layout_b(qk_mma, (BM, BN, HEAD_DIM), F16, self.KS)
        vk_layout = sm100_utils.make_smem_layout_b(qk_mma, (BM, BN, HEAD_DIM), F16, self.VS)
        v_layout = sm100_utils.make_smem_layout_b(pv_mma_acc, (BM, HEAD_DIM, BN), F16, self.VS)
        p_layout = sm100_utils.make_smem_layout_a(pv_mma_acc, (BM, HEAD_DIM, BN), F16, 1)

        @cute.struct
        class SharedStorage:
            q_full: cute.struct.MemRange[Int64, 1]
            raw_full: cute.struct.MemRange[Int64, self.RS]
            raw_empty: cute.struct.MemRange[Int64, self.RS]
            k_full: cute.struct.MemRange[Int64, self.KS]
            k_empty: cute.struct.MemRange[Int64, self.KS]
            v_full: cute.struct.MemRange[Int64, self.VS]
            v_empty: cute.struct.MemRange[Int64, self.VS]
            s_full: cute.struct.MemRange[Int64, 2]
            p_full: cute.struct.MemRange[Int64, 2]
            o_full: cute.struct.MemRange[Int64, 1]
            o_final: cute.struct.MemRange[Int64, 1]
            tmem_holding: Int32
            scan: cute.struct.MemRange[Int32, SCAN_WORDS]
            tab: cute.struct.Align[cute.struct.MemRange[Int32, self.tab_words], 16]
            raw: cute.struct.Align[cute.struct.MemRange[cutlass.Int8, self.raw_bytes], 128]
            sK: cute.struct.Align[cute.struct.MemRange[F16, cute.cosize(k_layout)], 1024]
            sV: cute.struct.Align[cute.struct.MemRange[F16, cute.cosize(vk_layout)], 1024]

        self.shared_storage = SharedStorage
        self.vk_layout = vk_layout
        bt = cute.make_tensor(
            bt_ptr, cute.make_layout((self.num_requests, self.bt_width), stride=(self.bt_width, 1))
        )
        req = cute.make_tensor(req_ptr, cute.make_layout((rows,), stride=(1,)))
        self.kernel(
            qk_mma,
            qk_mma_acc,
            pv_mma_acc,
            q_layout,
            k_layout,
            vk_layout,
            v_layout,
            p_layout,
            sel_t.iterator.toint(),
            q_ptr.toint(),
            k_ptr.toint(),
            v_ptr.toint(),
            bt,
            req,
            out_ptr.toint(),
            part_ptr.toint(),
            lse_ptr.toint(),
            gate_ptr.toint(),
            rows,
            sel_stride,
            gate_stride,
            gate_hs,
            n_whole,
            tail_row0,
            tab_ptr.toint(),
            cnt_ptr.toint(),
            est_ptr.toint(),
        ).launch(
            grid=[n_ctas, 1, 1],
            block=[THREADS, 1, 1],
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=self.pdl,
        )

    @cute.jit
    def _bal_schedule(self, tidx, warp, lane, bidx, rows, est_base, tab_base, s_w):
        """Balanced mode, every thread: this CTA's (unit, split, unit's CTAs, unit's first CTA):
        unit u takes ceil(e_u / L) CTAs, L the least load in [ceil(max e / MAX_BAL_SPLITS), max e]
        whose counts fit `bal_ctas`, units on consecutive CTAs; unit -1 past the plan. Scratch
        `s_w` is raw slot 0 (free until the first union)."""
        rpr: cutlass.Constexpr = self.T
        kvh_n: cutlass.Constexpr = self.kvh
        n_ctas: cutlass.Constexpr = self.bal_ctas
        hmax: cutlass.Constexpr = self.hmax
        nw: cutlass.Constexpr = -(-n_ctas // 32)
        s_res = s_w  # 16-byte aligned (raw slot 0)
        s_emax = s_w + 16
        s_l = s_w + 20
        s_tot = s_w + 32
        s_h = s_w + 32 + 32 * 4
        units = ((rows + rpr - 1) // rpr) * kvh_n
        for i in cutlass.range_constexpr(-(-hmax // THREADS)):
            k = tidx + i * THREADS
            if k < hmax:
                _sts1(s_h + k * 4, cutlass.Int32(0))
        if tidx == 0:
            _sts1(s_emax, cutlass.Int32(1))
            _sts1(s_l, cutlass.Int32(hmax))
            _sts4(s_res, cutlass.Int32(-1), cutlass.Int32(0), cutlass.Int32(1), cutlass.Int32(0))
        live_u = tidx < units
        e = cutlass.Int32(0)
        if live_u:
            e = cutlass.min(
                cutlass.max(_ldg1(est_base + cutlass.Int64(tidx // kvh_n) * 4), cutlass.Int32(1)),
                cutlass.Int32(hmax - 1),
            )
        cute.arch.sync_threads()
        if live_u:
            _red_shared("add", s_h + e * 4, cutlass.Int32(1))
            _red_shared("max", s_emax, e)
        cute.arch.sync_threads()
        emax = _lds1(s_emax)
        lo = cutlass.max((emax + MAX_BAL_SPLITS - 1) // MAX_BAL_SPLITS, cutlass.Int32(1))
        # thread t tests L = t + 1: sum over the histogram of h[x] * ceil(x / L), ceil by a
        # 16-bit reciprocal (exact: numerators below 2^9, L below 2^8)
        cand = tidx + 1
        recip = cutlass.Int32(65536) // cand + 1
        cnt = cutlass.Int32(0)
        for x4 in cutlass.range_constexpr(hmax // 4):
            h4 = _lds4(s_h + x4 * 16)
            for k in cutlass.range_constexpr(4):
                x = 4 * x4 + k
                if const_expr(x > 0):
                    cnt = cnt + h4[k] * (((x + cand - 1) * recip) >> 16)
        if (cand >= lo) & (cand <= emax) & (cnt <= n_ctas):
            _red_shared("min", s_l, cand)
        cute.arch.sync_threads()
        load = _lds1(s_l)
        n = cutlass.Int32(0)
        if live_u:
            n = (e + load - 1) // load
        # exclusive scan of n over units (threads 0 .. n_ctas - 1, warp scans + warp totals)
        inc = n
        for i in cutlass.range_constexpr(5):
            up = cute.arch.shuffle_sync_up(inc, offset=1 << i)
            if lane >= (1 << i):
                inc = inc + up
        if lane == 31:
            _sts1(s_tot + warp * 4, inc)
        cute.arch.sync_threads()
        before = cutlass.Int32(0)
        for w in cutlass.range_constexpr(nw):
            if warp > w:
                before = before + _lds1(s_tot + w * 4)
        base = before + inc - n
        if live_u & (bidx >= base) & (bidx < base + n):
            _sts4(s_res, tidx, bidx - base, n, base)
        cute.arch.sync_threads()
        res = _lds4(s_res)
        if tidx == 0:
            _stg4(tab_base + cutlass.Int64(bidx) * 16, res[0], res[1], res[2], res[3])
        cute.arch.sync_threads()
        return res

    # ---- pieces every role runs (every thread of the CTA, barrier 0) -----------------------

    @cute.jit
    def _pass_rows(self, rq: cute.Tensor, handled):
        lead = cutlass.Int32(-1)
        for t in tuple(range(self.T - 1, -1, -1)):
            lead = cutlass.Int32(cutlass.select_(((handled >> t) & 1) == 0, rq[t], lead))
        act = cutlass.Int32(0)
        for t in cutlass.range_constexpr(self.T):
            act = act | cutlass.Int32(
                cutlass.select_(
                    (((handled >> t) & 1) == 0) & (rq[t] == lead),
                    cutlass.Int32(1 << t),
                    cutlass.Int32(0),
                )
            )
        return lead, act

    @cute.jit
    def _hash_insert(self, s_hk, s_hm, gid, bits):
        """Insert group `gid` (linear probing over `HASH` slots; a key never moves once
        placed) and OR `bits` into its mask; returns the slot."""
        slot = ((gid * cutlass.Int32(-1640531535)) >> 20) & (HASH - 1)
        probing = cutlass.Int32(1)
        while probing != 0:
            old = _cas_shared(s_hk + slot * 4, cutlass.Int32(-1), gid)
            if (old == -1) | (old == gid):
                probing = cutlass.Int32(0)
            else:
                slot = (slot + 1) & (HASH - 1)
        _red_or_shared(s_hm + slot * 4, bits)
        return slot

    @cute.jit
    def _union(
        self, act, r0, rows, sel_base, sel_stride, tidx, warp, lane, s_hk, s_hm, s_scan, s_ug
    ):
        """The union of the pass's rows' groups: hash slots of the owned groups, in (row,
        column) order, at `s_ug`; returns their count."""
        rpr: cutlass.Constexpr = self.T
        width: cutlass.Constexpr = self.width
        n_q: cutlass.Constexpr = -(-width // 4)  # four-column quads a row
        n_i: cutlass.Constexpr = -(-n_q // THREADS)
        nwarps: cutlass.Constexpr = THREADS // 32
        # clear the hash: keys -1, masks 0
        for i in cutlass.range_constexpr(-(-(HASH // 4) // THREADS)):
            idx = tidx + i * THREADS
            if idx < HASH // 4:
                _sts4(s_hk + idx * 16, -1, -1, -1, -1)
                _sts4(s_hm + idx * 16, 0, 0, 0, 0)
        # thread: columns 4q .. 4q + 3 of each row, every load first
        toks = []
        for t in cutlass.range_constexpr(rpr):
            row_sel = sel_base + cutlass.Int64(cutlass.min(r0 + t, rows - 1)) * sel_stride * 4
            for i in cutlass.range_constexpr(n_i):
                for u in cutlass.range_constexpr(4):
                    colc = cutlass.min(4 * (tidx + i * THREADS) + u, width - 1)
                    toks.append(_ldg1(row_sel + cutlass.Int64(colc * 4)))
        cute.arch.sync_threads()
        gids = []
        slots = []
        for t in cutlass.range_constexpr(rpr):
            live_t = ((act >> t) & 1) != 0
            for i in cutlass.range_constexpr(n_i):
                q4 = tidx + i * THREADS
                gid = cutlass.Int32(-1)
                bits = cutlass.Int32(0)
                for u in cutlass.range_constexpr(4):
                    tok = toks[(t * n_i + i) * 4 + u]
                    ok = live_t & (4 * q4 + u < width) & (tok >= 0)
                    gid = cutlass.Int32(cutlass.select_(ok & (gid < 0), tok >> 2, gid))
                    bits = bits | cutlass.Int32(
                        cutlass.select_(ok, cutlass.Int32(1) << (tok & 3), cutlass.Int32(0))
                    )
                slot = cutlass.Int32(-1)
                if gid >= 0:
                    slot = self._hash_insert(s_hk, s_hm, gid, bits << (4 * t))
                gids.append(gid)
                slots.append(slot)
        cute.arch.sync_threads()
        # owner: the lowest row naming the group; counts per (row, round) and warp
        owns = []
        for t in cutlass.range_constexpr(rpr):
            for i in cutlass.range_constexpr(n_i):
                k = t * n_i + i
                own = cutlass.Int32(0)
                if slots[k] >= 0:
                    msk = _lds1(s_hm + slots[k] * 4)
                    own = cutlass.Int32(
                        cutlass.select_((_ctz(msk) >> 2) == t, cutlass.Int32(1), cutlass.Int32(0))
                    )
                owns.append(own)
                ball = _ballot(own)
                if lane == 0:
                    _sts1(s_scan + (k * nwarps + warp) * 4, _popc(ball))
        cute.arch.sync_threads()
        # place: rounds in (row, round) order, warps within a round, lanes within a warp
        lt_mask = (cutlass.Int32(1) << lane) - 1
        total = cutlass.Int32(0)
        for k in cutlass.range_constexpr(rpr * n_i):
            cnt_lane = cutlass.Int32(0)
            if lane < nwarps:
                cnt_lane = _lds1(s_scan + (k * nwarps + lane) * 4)
            before_w = cute.arch.warp_reduction_sum(
                cutlass.Int32(cutlass.select_(lane < warp, cnt_lane, cutlass.Int32(0)))
            )
            round_tot = cute.arch.warp_reduction_sum(cnt_lane)
            ball = _ballot(owns[k])
            pos = total + before_w + _popc(ball & lt_mask)
            if owns[k] != 0:
                if pos < self.ug_max:
                    _sts_u16(s_ug + pos * 2, slots[k])
            total = total + round_tot
        cute.arch.sync_threads()
        return cutlass.min(total, self.ug_max)

    @cute.jit
    def _resolve(self, lead, split, nsplit, n_loc, n_ug, tidx, s_hk, s_hm, s_ug, s_tab, mBT):
        """Every local tile's groups (pool row of the group's first token, (token, row)
        selection bytes), then each (tile, verify row)'s 64-bit selection word."""
        rpr: cutlass.Constexpr = self.T
        tmax: cutlass.Constexpr = self.tmax
        tpb: cutlass.Constexpr = self.tpb
        n_rs: cutlass.Constexpr = -(-(tmax * GPT) // THREADS)
        s_tok = s_tab + self.t_tok * 4
        s_gmask = s_tab + self.t_gmask * 4
        s_bits = s_tab + self.t_bits * 4
        pages = []
        gids = []
        msks = []
        for i in cutlass.range_constexpr(n_rs):
            slot = tidx + i * THREADS
            kl = slot // GPT
            ugi = (split + kl * nsplit) * GPT + slot % GPT
            ok = (kl < n_loc) & (ugi < n_ug)
            hs = cutlass.Int32(0)
            if ok:
                hs = _lds_u16(s_ug + cutlass.min(ugi, self.ug_max - 1) * 2)
            gid = cutlass.Int32(cutlass.select_(ok, _lds1(s_hk + hs * 4), cutlass.Int32(0)))
            msk = cutlass.Int32(cutlass.select_(ok, _lds1(s_hm + hs * 4), cutlass.Int32(0)))
            col = cutlass.min(cutlass.max((gid * 4) // tpb, 0), self.bt_width - 1)
            pg = mBT[cutlass.max(lead, 0), col]
            pages.append(cutlass.Int32(cutlass.select_(ok, pg, cutlass.Int32(-1))))
            gids.append(gid)
            msks.append(msk)
        for i in cutlass.range_constexpr(n_rs):
            slot = tidx + i * THREADS
            page = pages[i]
            gid = gids[i]
            msk = msks[i]
            mw = cutlass.Int32(0)
            for t in cutlass.range_constexpr(rpr):
                mw = mw | (_spread4((msk >> (4 * t)) & 0xF) << t)
            okp = (page >= 0) & (page < self.num_pages) & (msk != 0)
            tok = page * tpb + (gid * 4) % tpb
            tok = cutlass.Int32(cutlass.select_(okp, tok, cutlass.Int32(-1)))
            mw = cutlass.Int32(cutlass.select_(okp, mw, cutlass.Int32(0)))
            if slot < tmax * GPT:
                _sts1(s_tok + slot * 4, tok)
                _sts1(s_gmask + slot * 4, mw)
        cute.arch.sync_threads()
        for i in cutlass.range_constexpr(-(-(tmax * rpr) // THREADS)):
            w = tidx + i * THREADS
            if w < tmax * rpr:
                kl = w // rpr
                t = w % rpr
                lo = cutlass.Int32(0)
                hi = cutlass.Int32(0)
                for gq in cutlass.range_constexpr(GPT):
                    nb = (_lds1(s_gmask + (kl * GPT + gq) * 4) >> t) & 0x01010101
                    nb = (nb | (nb >> 7) | (nb >> 14) | (nb >> 21)) & 0xF
                    if const_expr(gq < 8):
                        lo = lo | (nb << (4 * gq))
                    else:
                        hi = hi | (nb << (4 * (gq - 8)))
                _sts1(s_bits + w * 8, lo)
                _sts1(s_bits + w * 8 + 4, hi)
        cute.arch.sync_threads()

    @cute.jit
    def _pass_tiles(
        self,
        act,
        lead,
        r0,
        rows,
        sel_base,
        sel_stride,
        split,
        nsplit,
        tidx,
        warp,
        lane,
        s_raw,
        s_scan,
        s_tab,
        mBT,
    ):
        """Union + resolve of one pass; returns the CTA's local tile count."""
        s_hk = s_raw + self.u_hk
        s_hm = s_raw + self.u_hm
        s_ug = s_raw + self.u_ug
        n_ug = self._union(
            act, r0, rows, sel_base, sel_stride, tidx, warp, lane, s_hk, s_hm, s_scan, s_ug
        )
        n_tiles = (n_ug + GPT - 1) // GPT
        n_loc = cutlass.Int32(
            cutlass.select_(
                split < n_tiles,
                (n_tiles - split + nsplit - 1) // nsplit,
                cutlass.Int32(0),
            )
        )
        self._resolve(lead, split, nsplit, n_loc, n_ug, tidx, s_hk, s_hm, s_ug, s_tab, mBT)
        return n_loc

    @cute.kernel
    def kernel(
        self,
        qk_mma: cute.TiledMma,
        qk_mma_acc: cute.TiledMma,
        pv_mma_acc: cute.TiledMma,
        q_layout: cute.ComposedLayout,
        k_layout: cute.ComposedLayout,
        vk_layout: cute.ComposedLayout,
        v_layout: cute.ComposedLayout,
        p_layout: cute.ComposedLayout,
        sel_base: cutlass.Int64,
        q_base: cutlass.Int64,
        k_base: cutlass.Int64,
        v_base: cutlass.Int64,
        mBT: cute.Tensor,
        mReq: cute.Tensor,
        out_base: cutlass.Int64,
        part_base: cutlass.Int64,
        lse_base: cutlass.Int64,
        gate_base: cutlass.Int64,
        rows: cutlass.Int32,
        sel_stride: cutlass.Int32,
        gate_stride: cutlass.Int32,
        gate_hs: cutlass.Int32,
        n_whole: cutlass.Int32,
        tail_row0: cutlass.Int32,
        tab_base: cutlass.Int64,
        cnt_base: cutlass.Int64,
        est_base: cutlass.Int64,
    ):
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        lane = cute.arch.lane_idx()
        qk_mma.set(tcgen05.Field.ACCUMULATE, False)
        qk_mma_acc.set(tcgen05.Field.ACCUMULATE, True)
        pv_mma_acc.set(tcgen05.Field.ACCUMULATE, True)
        rpr: cutlass.Constexpr = self.T
        group: cutlass.Constexpr = self.group
        kvh_n: cutlass.Constexpr = self.kvh
        splits: cutlass.Constexpr = self.splits
        if const_expr(self.tail):
            # units (group, KV head) below n_whole: one CTA each; the rest: `splits` CTAs each
            is_whole = bidx < n_whole
            t_cta = bidx - n_whole
            unit = cutlass.Int32(cutlass.select_(is_whole, bidx, n_whole + t_cta // splits))
            split = cutlass.Int32(cutlass.select_(is_whole, cutlass.Int32(0), t_cta % splits))
            nsplit = cutlass.Int32(
                cutlass.select_(is_whole, cutlass.Int32(1), cutlass.Int32(splits))
            )
        elif const_expr(self.bal):
            # read from the plan table after the dependency wait
            unit = cutlass.Int32(0)
            split = cutlass.Int32(0)
            nsplit = cutlass.Int32(1)
        else:
            unit = bidx // splits
            split = bidx % splits
            nsplit = cutlass.Int32(splits)
        gi = unit // kvh_n
        kvh = unit % kvh_n
        r0 = gi * rpr

        smem = SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        q_full = storage.q_full.data_ptr()
        raw_full = storage.raw_full.data_ptr()
        raw_empty = storage.raw_empty.data_ptr()
        k_full = storage.k_full.data_ptr()
        k_empty = storage.k_empty.data_ptr()
        v_full = storage.v_full.data_ptr()
        v_empty = storage.v_empty.data_ptr()
        s_full = storage.s_full.data_ptr()
        p_full = storage.p_full.data_ptr()
        o_full = storage.o_full.data_ptr()
        o_final = storage.o_final.data_ptr()
        s_scan = cutlass.Int32(storage.scan.data_ptr().toint())
        s_tab = cutlass.Int32(storage.tab.data_ptr().toint())
        s_raw = cutlass.Int32(storage.raw.data_ptr().toint())
        if tidx == 0:
            cute.arch.mbarrier_init(q_full, 32 * SOFT_WARPS)
            for i in cutlass.range_constexpr(self.RS):
                cute.arch.mbarrier_init(raw_full + i, LOAD_WARPS)
                cute.arch.mbarrier_init(raw_empty + i, 32 * DQ_WARPS)
            for i in cutlass.range_constexpr(self.KS):
                cute.arch.mbarrier_init(k_full + i, BN)
                cute.arch.mbarrier_init(k_empty + i, 1)
            for i in cutlass.range_constexpr(self.VS):
                cute.arch.mbarrier_init(v_full + i, BN)
                cute.arch.mbarrier_init(v_empty + i, 1)
            for i in cutlass.range_constexpr(2):
                cute.arch.mbarrier_init(s_full + i, 1)
                cute.arch.mbarrier_init(p_full + i, 32 * SOFT_WARPS)
            cute.arch.mbarrier_init(o_full, 1)
            cute.arch.mbarrier_init(o_final, 1)
        cute.arch.mbarrier_init_fence()
        if warp == MMA_WARP:
            cute.arch.alloc_tmem(TMEM_COLS, storage.tmem_holding.ptr)
            cute.arch.relinquish_tmem_alloc_permit()
        cute.arch.sync_threads()
        tmem_ptr = cute.arch.retrieve_tmem_ptr(Float32, 16, storage.tmem_holding.ptr)

        qk_thr = qk_mma.get_slice(0)
        pv_thr = pv_mma_acc.get_slice(0)
        sK = storage.sK.get_tensor(k_layout.outer, swizzle=k_layout.inner)
        sVr = storage.sV.get_tensor(vk_layout.outer, swizzle=vk_layout.inner)
        sV = cute.make_tensor(cute.recast_ptr(sVr.iterator, v_layout.inner), v_layout.outer)
        s_k16 = cutlass.Int32(storage.sK.data_ptr().toint())
        s_v16 = cutlass.Int32(storage.sV.data_ptr().toint())

        # ---- softmax-side tensor-memory views (warps 0-3: one thread an MMA row) -------------
        st = tidx
        tStS = qk_thr.make_fragment_C(qk_thr.partition_shape_C((BM, BN)))
        tStS0 = cute.make_tensor(tmem_ptr + TMEM_S0, tStS.layout)
        tOtO_f = pv_thr.make_fragment_C(pv_thr.partition_shape_C((BM, HEAD_DIM)))
        tOtO = cute.make_tensor(tmem_ptr + TMEM_O, tOtO_f.layout)
        tScS = qk_thr.partition_C(cute.make_identity_tensor((BM, BN)))
        load_s = tcgen05.make_tmem_copy(
            cute.make_copy_atom(tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), Float32),
            tStS0,
        )
        thr_load_s = load_s.get_slice(st)
        tLtS0 = thr_load_s.partition_S(tStS0)
        tLcS = thr_load_s.partition_D(tScS)
        p_cols = BN // 2
        tStP0 = cute.make_tensor(
            tStS0.iterator, cute.composition(tStS0.layout, cute.make_layout((BM, p_cols)))
        )
        tScP = cute.make_tensor(
            tScS.iterator, cute.composition(tScS.layout, cute.make_layout((BM, p_cols)))
        )
        store_p = tcgen05.make_tmem_copy(
            cute.make_copy_atom(tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(p_cols)), Float32),
            tStP0,
        )
        thr_store_p = store_p.get_slice(st)
        tPtP0 = thr_store_p.partition_D(tStP0)
        tPcP = thr_store_p.partition_S(tScP)
        tOcO = pv_thr.partition_C(cute.make_identity_tensor((BM, HEAD_DIM)))
        tOtO_c = cute.make_tensor(
            tOtO.iterator, cute.composition(tOtO.layout, cute.make_layout((BM, CHUNK_COLS)))
        )
        tOcO_c = cute.make_tensor(
            tOcO.iterator, cute.composition(tOcO.layout, cute.make_layout((BM, CHUNK_COLS)))
        )
        load_o = tcgen05.make_tmem_copy(
            cute.make_copy_atom(
                tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(CHUNK_COLS)), Float32
            ),
            tOtO_c,
        )
        store_o = tcgen05.make_tmem_copy(
            cute.make_copy_atom(
                tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(CHUNK_COLS)), Float32
            ),
            tOtO_c,
        )
        thr_load_o = load_o.get_slice(st)
        thr_store_o = store_o.get_slice(st)
        tLtO = thr_load_o.partition_S(tOtO_c)
        tLcO = thr_load_o.partition_D(tOcO_c)
        tStO = thr_store_o.partition_D(tOtO_c)
        tview = (load_s, tLtS0, tLcS, store_p, tPtP0, tPcP, load_o, store_o, tLtO, tLcO, tStO)

        if const_expr(self.pdl):
            cute.arch.griddepcontrol_launch_dependents()
            cute.arch.griddepcontrol_wait()

        idle = cutlass.Int32(0)
        bal_base = cutlass.Int32(0)
        if const_expr(self.bal):
            # this CTA's (unit, split, unit's CTAs, unit's first CTA); unit -1: past the plan
            ent = self._bal_schedule(tidx, warp, lane, bidx, rows, est_base, tab_base, s_raw)
            idle = cutlass.Int32(cutlass.select_(ent[0] < 0, cutlass.Int32(1), cutlass.Int32(0)))
            unit = cutlass.max(ent[0], cutlass.Int32(0))
            split = ent[1]
            nsplit = ent[2]
            bal_base = ent[3]
            gi = unit // kvh_n
            kvh = unit % kvh_n
            r0 = gi * rpr
        m = tidx
        t_row = m // group
        h = m % group
        live = (m < self.rows_m) & (r0 + t_row < rows)
        if const_expr(self.bal):
            live = live & (idle == 0)
        safe_row = cutlass.min(r0 + t_row, rows - 1)
        qs0, qs1 = self.qs
        q_row = q_base + (cutlass.Int64(safe_row) * qs0 + cutlass.Int64(kvh * group + h) * qs1) * 2
        q_s = s_raw + self.u_q + cutlass.min(m, self.rows_m - 1) * Q_LD

        if warp < SOFT_WARPS:
            # O = 0; this row's q copied into raw slot 1 now, into tensor memory (f16) after the
            # first pass's resolve. Dead rows stage zeros.
            rZ = cute.make_rmem_tensor(tLcO.shape, Float32)
            rZ.fill(0.0)
            for c in cutlass.range_constexpr(HEAD_DIM // CHUNK_COLS):
                cute.copy(
                    store_o, rZ, cute.make_tensor(tStO.iterator + c * CHUNK_COLS, tStO.layout)
                )
            cute.arch.fence_view_async_tmem_store()
            if m < self.rows_m:
                qn = cutlass.Int32(cutlass.select_(live, cutlass.Int32(16), cutlass.Int32(0)))
                for i in cutlass.range_constexpr(HEAD_DIM // 8):
                    _cp16z(q_s + 16 * i, q_row + cutlass.Int64(16 * i), qn)
            cute.arch.cp_async_commit_group()

        rq = cute.make_rmem_tensor(cute.make_layout((rpr,), stride=(1,)), cutlass.Int32)
        handled = cutlass.Int32(0)
        for t in cutlass.range_constexpr(rpr):
            row = r0 + t
            xr = cutlass.Int64(mReq[cutlass.min(row, rows - 1)])
            okr = (row < rows) & (xr >= 0) & (xr < self.num_requests)
            rq[t] = cutlass.Int32(cutlass.select_(okr, cutlass.Int32(xr), cutlass.Int32(-1)))
            handled = handled | cutlass.Int32(
                cutlass.select_(okr, cutlass.Int32(0), cutlass.Int32(1 << t))
            )
        full = (1 << rpr) - 1
        if const_expr(self.bal):
            handled = cutlass.Int32(cutlass.select_(idle != 0, cutlass.Int32(full), handled))
        scale_log2 = Float32(self.scale_log2)
        neg_inf = -Float32.inf
        row_max = neg_inf
        row_sum = Float32(0.0)
        q_pending = cutlass.Int32(1)
        g = cutlass.Int32(0)
        for _pass in cutlass.range(rpr):
            if handled != full:
                lead, act = self._pass_rows(rq, handled)
                n_loc = self._pass_tiles(
                    act,
                    lead,
                    r0,
                    rows,
                    sel_base,
                    sel_stride,
                    split,
                    nsplit,
                    tidx,
                    warp,
                    lane,
                    s_raw,
                    s_scan,
                    s_tab,
                    mBT,
                )
                if warp >= DQ_WARP0:
                    self.widen_tiles(
                        tidx - 32 * DQ_WARP0,
                        g,
                        n_loc,
                        s_raw,
                        s_tab,
                        s_k16,
                        s_v16,
                        raw_full,
                        raw_empty,
                        k_full,
                        k_empty,
                        v_full,
                        v_empty,
                    )
                elif warp >= LOAD_WARP0:
                    self.load_tiles(
                        tidx - 32 * LOAD_WARP0,
                        g,
                        n_loc,
                        kvh,
                        s_raw,
                        s_tab,
                        k_base,
                        v_base,
                        raw_full,
                        raw_empty,
                        q_full,
                    )
                elif warp == MMA_WARP:
                    self.mma_tiles(
                        g,
                        n_loc,
                        qk_mma,
                        pv_mma_acc,
                        qk_thr,
                        pv_thr,
                        q_layout,
                        p_layout,
                        sK,
                        sV,
                        tmem_ptr,
                        q_full,
                        k_full,
                        k_empty,
                        v_full,
                        v_empty,
                        s_full,
                        p_full,
                        o_full,
                    )
                else:
                    if q_pending != 0:
                        self.stage_q(q_s, live, tview, q_full)
                        q_pending = cutlass.Int32(0)
                    row_max, row_sum = self.softmax_tiles(
                        g,
                        n_loc,
                        row_max,
                        row_sum,
                        live,
                        t_row,
                        warp,
                        s_tab,
                        scale_log2,
                        tview,
                        s_full,
                        p_full,
                        o_full,
                    )
                g = g + n_loc
                cute.arch.sync_threads()  # [P] end of the pass
                handled = handled | act

        if warp == MMA_WARP:
            with cute.arch.elect_one():
                tcgen05.commit(o_final)
        if warp < SOFT_WARPS:
            cute.arch.cp_async_wait_group(0)
            epi = (
                row_max,
                row_sum,
                live,
                warp,
                m,
                r0,
                rows,
                kvh,
                bidx if self.bal else split,  # balanced: the partial slot is the CTA
                nsplit,
                tail_row0,
                scale_log2,
                out_base,
                part_base,
                lse_base,
                gate_base,
                gate_stride,
                gate_hs,
                tview,
                o_final,
                tidx,
                s_k16,
            )
            if const_expr(self.bal):
                if idle == 0:
                    self.epilogue(*epi)
            else:
                self.epilogue(*epi)
        # every role done with tensor memory (the softmax warps read O last)
        cute.arch.sync_threads()
        if warp == MMA_WARP:
            cute.arch.dealloc_tmem(tmem_ptr, TMEM_COLS)
        if const_expr(self.bal):
            if (idle == 0) & (nsplit > 1):
                self.fixup(
                    tidx,
                    unit,
                    kvh,
                    r0,
                    rows,
                    nsplit,
                    bal_base,
                    cnt_base,
                    part_base,
                    lse_base,
                    out_base,
                    gate_base,
                    gate_stride,
                    gate_hs,
                    s_scan,
                    s_raw,
                )

    # ------------------------------------------------------------------ loaders (warps 5-6)
    @cute.jit
    def load_tiles(
        self, lt, g, n_loc, kvh, s_raw, s_tab, k_base, v_base, raw_full, raw_empty, q_full
    ):
        """Raw tile copies: lane l < 16 of warp 5 (K) and of warp 6 (V) bulk-copies group l's four
        rows (1 KB, contiguous in the pool); each warp's lane 0 first arms the slot's mbarrier with
        its half of the tile's bytes. A group that does not resolve is not copied (the widening
        zeroes it)."""
        s_tok = s_tab + self.t_tok * 4
        s_gmask = s_tab + self.t_gmask * 4
        tpb: cutlass.Constexpr = self.tpb
        ks0, ks1, ks2 = self.ks
        if lt < 32 * LOAD_WARPS:
            gi = lt % 16
            is_v = lt >= 32
            base = cutlass.select_(is_v, v_base, k_base) + cutlass.Int64(kvh * ks1)
            dst0 = (
                s_raw
                + cutlass.Int32(cutlass.select_(is_v, cutlass.Int32(RAW_K), cutlass.Int32(0)))
                + gi * (4 * RAW_LD)
            )
            rf0 = cutlass.Int32(raw_full.toint())
            for kl in cutlass.range(n_loc):
                gg = g + kl
                r = gg % self.RS
                cute.arch.mbarrier_wait(raw_empty + r, _phase(gg, self.RS) ^ 1)
                if r == 1:
                    cute.arch.mbarrier_wait(q_full, 0)
                slot = kl * GPT + gi
                tok0 = _lds1(s_tok + slot * 4)
                mw = _lds1(s_gmask + slot * 4)
                ok = (tok0 >= 0) & (mw != 0) & ((lt % 32) < 16)
                nb = cutlass.Int32(
                    cutlass.select_(ok, cutlass.Int32(4 * HEAD_DIM), cutlass.Int32(0))
                )
                total = cute.arch.warp_reduction_sum(nb)
                if (lt % 32) == 0:
                    _arrive_expect_tx(rf0 + r * 8, total)
                cute.arch.sync_warp()
                if ok:
                    off = cutlass.Int64(tok0 // tpb) * ks0 + cutlass.Int64((tok0 % tpb) * ks2)
                    _bulk_g2s(dst0 + r * RAW_BUF, base + off, nb, rf0 + r * 8)

    # ------------------------------------------------------------------ widen (warps 7-10)
    @cute.jit
    def widen_tiles(
        self,
        dt,
        g,
        n_loc,
        s_raw,
        s_tab,
        s_k16,
        s_v16,
        raw_full,
        raw_empty,
        k_full,
        k_empty,
        v_full,
        v_empty,
    ):
        """Thread `dt` < 64 widens K row `dt` of each tile, the others V row `dt - 64`: the raw
        row into registers (the raw slot is then free), then into the f16 SW128 slot once the
        MMA has released it."""
        is_v = dt >= BN
        n = dt % BN
        s_tok = s_tab + self.t_tok * 4
        s_gmask = s_tab + self.t_gmask * 4
        # bulk copies land unpadded rows: chunk c of the read order is the row's chunk (c + n) % 16
        rot = n % 16
        for kl in cutlass.range(n_loc):
            gg = g + kl
            r = gg % self.RS
            slot = kl * GPT + n // 4
            tok0 = _lds1(s_tok + slot * 4)
            mw = _lds1(s_gmask + slot * 4)
            live_row = (tok0 >= 0) & (((mw >> (8 * (n % 4))) & 0xFF) != 0)
            cute.arch.mbarrier_wait(raw_full + r, _phase(gg, self.RS))
            row_off = (
                s_raw
                + r * RAW_BUF
                + cutlass.Int32(cutlass.select_(is_v, cutlass.Int32(RAW_K), cutlass.Int32(0)))
                + n * RAW_LD
            )
            w = [_lds4(row_off + 16 * ((c + rot) % 16)) for c in range(16)]
            cute.arch.mbarrier_arrive(raw_empty + r)
            # a row no verify row selected (not copied) widens to zeros: P = 0 times a stale NaN
            # would poison P V
            w = [
                [cutlass.Int32(cutlass.select_(live_row, x, cutlass.Int32(0))) for x in wc]
                for wc in w
            ]
            if is_v:
                b = gg % self.VS
                cute.arch.mbarrier_wait(v_empty + b, _phase(gg, self.VS) ^ 1)
                self._widen_row(w, s_v16 + b * F16_SLOT, n, rot)
                cute.arch.fence_view_async_shared()
                cute.arch.mbarrier_arrive(v_full + b)
            else:
                b = gg % self.KS
                cute.arch.mbarrier_wait(k_empty + b, _phase(gg, self.KS) ^ 1)
                self._widen_row(w, s_k16 + b * F16_SLOT, n, rot)
                cute.arch.fence_view_async_shared()
                cute.arch.mbarrier_arrive(k_full + b)

    @cute.jit
    def _widen_row(self, w, dst, n, rot):
        """Raw e4m3 row words `w` (16 x 4) -> f16 SW128 row `n` at `dst`: 16-byte chunk ci (dims
        8 ci .. 8 ci + 7) of the row's 64-dim block ci // 8 at 8192 (ci // 8) + 128 n + 16 ((ci %
        8) ^ (n % 8))."""
        for c in cutlass.range_constexpr(16):
            for half in cutlass.range_constexpr(2):
                a = _e4m3x4(w[c][2 * half])
                b = _e4m3x4(w[c][2 * half + 1])
                ci = 2 * ((c + rot) % 16) + half
                _sts4(
                    dst + 8192 * (ci // 8) + 128 * n + 16 * ((ci % 8) ^ (n % 8)),
                    a[0],
                    a[1],
                    b[0],
                    b[1],
                )

    # ------------------------------------------------------------------ MMA (warp 4)
    @cute.jit
    def mma_tiles(
        self,
        g,
        n_loc,
        qk_mma,
        pv_mma_acc,
        qk_thr,
        pv_thr,
        q_layout,
        p_layout,
        sK,
        sV,
        tmem_ptr,
        q_full,
        k_full,
        k_empty,
        v_full,
        v_empty,
        s_full,
        p_full,
        o_full,
    ):
        """QK(g), then QK(j) / PV(j - 1) pairs, then the last PV: S/P buffer j % 2 at TMEM column
        TMEM_S0 + 64 (j % 2); QK(j) overwrites it only after PV(j - 2) was issued (tcgen05 MMAs of
        one thread execute in order)."""
        tQ = cute.make_tensor(tmem_ptr + TMEM_Q, q_layout.outer)
        tSrQ = qk_thr.make_fragment_A(tQ)[None, None, None, 0]
        tSrQ = cute.make_tensor(cute.recast_ptr(tmem_ptr + TMEM_Q, dtype=F16), tSrQ.layout)
        tSrK = qk_thr.make_fragment_B(sK)
        tOrV = pv_thr.make_fragment_B(sV)
        tP = cute.make_tensor(tmem_ptr + TMEM_S0, p_layout.outer)
        tOrP = pv_thr.make_fragment_A(tP)[None, None, None, 0]
        tOrP = cute.make_tensor(cute.recast_ptr(tmem_ptr + TMEM_S0, dtype=F16), tOrP.layout)
        tbase = cutlass.Int32(tmem_ptr.toint())
        qk_op = qk_mma.op
        pv_op = pv_mma_acc.op
        if n_loc > 0:
            cute.arch.mbarrier_wait(q_full, 0)
            cute.arch.mbarrier_wait(k_full + g % self.KS, _phase(g, self.KS))
            _tc_fence_after()
            self.issue_qk(qk_op, tbase, tSrQ, tSrK, sK, g % 2, g % self.KS)
            with cute.arch.elect_one():
                tcgen05.commit(s_full + g % 2)
                tcgen05.commit(k_empty + g % self.KS)
            for j in cutlass.range(1, n_loc + 1, 1, unroll=1):
                gj = g + j
                if j < n_loc:
                    cute.arch.mbarrier_wait(k_full + gj % self.KS, _phase(gj, self.KS))
                    _tc_fence_after()
                    self.issue_qk(qk_op, tbase, tSrQ, tSrK, sK, gj % 2, gj % self.KS)
                    with cute.arch.elect_one():
                        tcgen05.commit(s_full + gj % 2)
                        tcgen05.commit(k_empty + gj % self.KS)
                gp = gj - 1
                cute.arch.mbarrier_wait(v_full + gp % self.VS, _phase(gp, self.VS))
                cute.arch.mbarrier_wait(p_full + gp % 2, _phase(gp, 2))
                _tc_fence_after()
                self._issue_pv(pv_op, tbase, tOrP, tOrV, sV, gp)
                with cute.arch.elect_one():
                    tcgen05.commit(v_empty + gp % self.VS)
                    tcgen05.commit(o_full)

    @cute.jit
    def _issue_pv(self, pv_op, tbase, tOrP, tOrV, sV, gp):
        """O += P(gp) V(gp): P from tensor memory over the S it came from, V from its slot."""
        _gemm_ts(
            pv_op,
            tbase + TMEM_O,
            tbase + TMEM_S0 + (gp % 2) * BN,
            tOrP,
            tOrV[None, None, None, gp % self.VS],
            sV[None, None, None, gp % self.VS],
            False,
        )

    @cute.jit
    def issue_qk(self, qk_op, tbase, tSrQ, tSrK, sK, sbuf, kslot):
        """S[sbuf] = Q K(kslot)^T: Q from tensor memory, K from SW128 slot `kslot`."""
        _gemm_ts(
            qk_op,
            tbase + TMEM_S0 + sbuf * BN,
            tbase + TMEM_Q,
            tSrQ,
            tSrK[None, None, None, kslot],
            sK[None, None, None, kslot],
            True,
        )

    # ------------------------------------------------------------- softmax + epilogue (warps 0-3)
    @cute.jit
    def stage_q(self, q_s, live, tview, q_full):
        """This MMA row's q (bf16 in raw slot 1) -> f16 into tensor memory (the QK A operand)."""
        load_s, tLtS0, tLcS, store_p, tPtP0, tPcP, load_o, store_o, tLtO, tLcO, tStO = tview
        p_cols: cutlass.Constexpr = BN // 2
        cute.arch.cp_async_wait_group(0)
        for c in cutlass.range_constexpr(HEAD_DIM // (2 * p_cols)):
            rQ = cute.make_rmem_tensor(tPcP.shape, Float32)
            for i in cutlass.range_constexpr(p_cols // 4):
                w4 = _lds4(q_s + 16 * (c * (p_cols // 4) + i))
                for e in cutlass.range_constexpr(4):
                    rQ[4 * i + e] = _bits_f32(
                        cutlass.Int32(cutlass.select_(live, _bf2_to_f2(w4[e]), cutlass.Int32(0)))
                    )
            cute.copy(
                store_p,
                rQ,
                cute.make_tensor(tPtP0.iterator + (TMEM_Q - TMEM_S0 + c * p_cols), tPtP0.layout),
            )
        cute.arch.fence_view_async_tmem_store()
        _tc_fence_before()
        cute.arch.mbarrier_arrive(q_full)

    @cute.jit
    def softmax_tiles(
        self,
        g,
        n_loc,
        row_max,
        row_sum,
        live,
        t_row,
        warp,
        s_tab,
        scale_log2,
        tview,
        s_full,
        p_full,
        o_full,
    ):
        """Per tile: S from tensor memory, the max over the row's own entries, the reference max
        moved (and O rescaled) only past RESCALE_TH, P = 2^(s - max) as f16 over the S."""
        load_s, tLtS0, tLcS, store_p, tPtP0, tPcP, load_o, store_o, tLtO, tLcO, tStO = tview
        rpr: cutlass.Constexpr = self.T
        s_bits = s_tab + self.t_bits * 4
        tau = Float32(RESCALE_TH)
        neg_inf = -Float32.inf
        live_warp = warp < self.live_warps
        for kl in cutlass.range(n_loc, unroll=1):
            gg = g + kl
            buf = gg % 2
            cute.arch.mbarrier_wait(s_full + buf, _phase(gg, 2))
            _tc_fence_after()
            if live_warp:
                lo = cutlass.Int32(0)
                hi = cutlass.Int32(0)
                if live:
                    lo = _lds1(s_bits + (kl * rpr + t_row) * 8)
                    hi = _lds1(s_bits + (kl * rpr + t_row) * 8 + 4)
                rS = cute.make_rmem_tensor(tLcS.shape, Float32)
                tLtS_b = cute.make_tensor(tLtS0.iterator + buf * BN, tLtS0.layout)
                for c in cutlass.range_constexpr(cute.size(rS, mode=[1])):
                    cute.copy(load_s, tLtS_b[None, c, 0, 0], rS[None, c, 0, 0])
                cute.arch.fence_view_async_tmem_load()
                tile_max = neg_inf
                for i in cutlass.range_constexpr(BN):
                    word = lo if i < 32 else hi
                    x = rS[i]
                    x = x if (word & (1 << (i % 32))) != 0 else neg_inf
                    rS[i] = x
                    tile_max = cute.arch.fmax(tile_max, x)
                new_max = cute.arch.fmax(row_max, tile_max)
                use_max = row_max
                factor = Float32(1.0)
                if row_max == neg_inf:
                    use_max = new_max
                elif (new_max - row_max) * scale_log2 > tau:
                    use_max = new_max
                    factor = cute.math.exp2((row_max - new_max) * scale_log2, fastmath=True)
                need = factor != Float32(1.0)
                if cute.arch.vote_any_sync(need):
                    # O holds PV(..gg - 1): wait for it, then rescale this warp's rows
                    cute.arch.mbarrier_wait(o_full, (gg - 1) % 2)
                    _tc_fence_after()
                    for c in cutlass.range_constexpr(HEAD_DIM // CHUNK_COLS):
                        rO = cute.make_rmem_tensor(tLcO.shape, Float32)
                        cute.copy(
                            load_o,
                            cute.make_tensor(tLtO.iterator + c * CHUNK_COLS, tLtO.layout),
                            rO,
                        )
                        for i in cutlass.range_constexpr(cute.size(rO)):
                            rO[i] = rO[i] * factor
                        cute.copy(
                            store_o,
                            rO,
                            cute.make_tensor(tStO.iterator + c * CHUNK_COLS, tStO.layout),
                        )
                    cute.arch.fence_view_async_tmem_store()
                row_max = use_max
                safe = use_max
                if use_max == neg_inf:
                    safe = Float32(0.0)
                neg = -safe * scale_log2
                rP = cute.make_rmem_tensor(tPcP.shape, Float32)
                rP16 = cute.make_tensor(cute.recast_ptr(rP.iterator, dtype=F16), rS.layout)
                acc0 = (Float32(0.0), Float32(0.0))
                acc1 = (Float32(0.0), Float32(0.0))
                for i in cutlass.range_constexpr(0, BN, 4):
                    e = cute.make_rmem_tensor((4,), Float32)
                    e[0], e[1] = cute.arch.fma_packed_f32x2(
                        (rS[i], rS[i + 1]), (scale_log2, scale_log2), (neg, neg)
                    )
                    e[2], e[3] = cute.arch.fma_packed_f32x2(
                        (rS[i + 2], rS[i + 3]), (scale_log2, scale_log2), (neg, neg)
                    )
                    for u in cutlass.range_constexpr(4):
                        rS[i + u] = cute.math.exp2(e[u], fastmath=True)
                    acc0 = cute.arch.add_packed_f32x2(acc0, (rS[i], rS[i + 1]))
                    acc1 = cute.arch.add_packed_f32x2(acc1, (rS[i + 2], rS[i + 3]))
                row_sum = row_sum * factor + ((acc0[0] + acc0[1]) + (acc1[0] + acc1[1]))
                rP16.store(rS.load().to(F16))
                cute.copy(store_p, rP, cute.make_tensor(tPtP0.iterator + buf * BN, tPtP0.layout))
                cute.arch.fence_view_async_tmem_store()
            _tc_fence_before()
            cute.arch.mbarrier_arrive(p_full + buf)
        return row_max, row_sum

    @cute.jit
    def epilogue(
        self,
        row_max,
        row_sum,
        live,
        warp,
        m,
        r0,
        rows,
        kvh,
        split,
        nsplit,
        tail_row0,
        scale_log2,
        out_base,
        part_base,
        lse_base,
        gate_base,
        gate_stride,
        gate_hs,
        tview,
        o_final,
        st,
        s_stage,
    ):
        """O / l: a whole unit's rows as bf16, a split's as an fp32 partial and its log2-sum-exp for
        the merge (rows from `tail_row0` on). Gated whole units read the gate before the last P V
        completes, stage bf16(O / l) in the free K slots, then apply the gate in coalesced rows."""
        load_s, tLtS0, tLcS, store_p, tPtP0, tPcP, load_o, store_o, tLtO, tLcO, tStO = tview
        group: cutlass.Constexpr = self.group
        heads: cutlass.Constexpr = self.heads
        splits: cutlass.Constexpr = self.splits
        tailm: cutlass.Constexpr = self.tail or self.bal
        has_whole: cutlass.Constexpr = tailm or splits == 1
        has_split: cutlass.Constexpr = tailm or splits > 1
        gated_whole: cutlass.Constexpr = has_whole and self.gated
        rows_m: cutlass.Constexpr = self.rows_m
        cpr: cutlass.Constexpr = HEAD_DIM // 8  # 16-byte chunks a row
        n_gc: cutlass.Constexpr = -(-(rows_m * cpr) // (32 * SOFT_WARPS))
        n_chunks: cutlass.Constexpr = HEAD_DIM // CHUNK_COLS
        neg_inf = -Float32.inf
        row = r0 + m // group
        h = m % group
        head = kvh * group + h
        prow = cutlass.Int64(row - tail_row0)
        prows = cutlass.Int64(rows - tail_row0)
        if const_expr(self.bal):
            # a CTA's partial slot holds its unit's T rows
            prow = cutlass.Int64(m // group)
            prows = cutlass.Int64(self.T)
        whole = nsplit == 1
        gregs = []
        if const_expr(gated_whole):
            for j in cutlass.range_constexpr(n_gc):
                k = st + 32 * SOFT_WARPS * j
                gm = cutlass.min(k // cpr, rows_m - 1)
                grow = cutlass.min(r0 + gm // group, rows - 1)
                ghead = kvh * group + gm % group
                gregs.append(
                    _ldg4(
                        gate_base
                        + (
                            cutlass.Int64(grow) * gate_stride
                            + cutlass.Int64(ghead) * gate_hs
                            + (k % cpr) * 8
                        )
                        * 2
                    )
                )
        cute.arch.mbarrier_wait(o_final, 0)
        _tc_fence_after()
        if warp < self.live_warps:
            inv = Float32(0.0)
            if row_sum > Float32(0.0):
                inv = Float32(1.0) / row_sum
            for c in cutlass.range_constexpr(n_chunks):
                rO = cute.make_rmem_tensor(tLcO.shape, Float32)
                cute.copy(load_o, cute.make_tensor(tLtO.iterator + c * CHUNK_COLS, tLtO.layout), rO)
                for v8 in cutlass.range_constexpr(CHUNK_COLS // 8):
                    vals = [rO[8 * v8 + i] * inv for i in range(8)]
                    d0 = c * CHUNK_COLS + 8 * v8
                    oaddr = out_base + ((cutlass.Int64(row) * heads + head) * HEAD_DIM + d0) * 2
                    paddr = (
                        part_base
                        + (((cutlass.Int64(split) * prows + prow) * heads + head) * HEAD_DIM + d0)
                        * 4
                    )
                    packed = [_bf2_pack(vals[2 * e], vals[2 * e + 1]) for e in range(4)]
                    saddr = s_stage + cutlass.min(m, rows_m - 1) * Q_LD + d0 * 2
                    if const_expr(has_whole):
                        if whole:
                            if const_expr(gated_whole):
                                if m < rows_m:
                                    _sts4(saddr, packed[0], packed[1], packed[2], packed[3])
                            else:
                                if live:
                                    _stg4(oaddr, packed[0], packed[1], packed[2], packed[3])
                    if const_expr(has_split):
                        if live & (nsplit > 1):
                            _stg4f(paddr, vals[0], vals[1], vals[2], vals[3])
                            _stg4f(paddr + 16, vals[4], vals[5], vals[6], vals[7])
            if const_expr(has_split):
                if live & (nsplit > 1):
                    lse = neg_inf
                    if row_sum > Float32(0.0):
                        lse = row_max * scale_log2 + cute.math.log2(row_sum)
                    _stg1(
                        lse_base + ((cutlass.Int64(split) * prows + prow) * heads + head) * 4,
                        _f32_bits(lse),
                    )
        if const_expr(gated_whole):
            if whole:
                cute.arch.barrier(barrier_id=1, number_of_threads=32 * SOFT_WARPS)
                for j in cutlass.range_constexpr(n_gc):
                    k = st + 32 * SOFT_WARPS * j
                    gm = cutlass.min(k // cpr, rows_m - 1)
                    grow = r0 + gm // group
                    ghead = kvh * group + gm % group
                    a4 = _lds4(s_stage + gm * Q_LD + (k % cpr) * 16)
                    g4 = gregs[j]
                    res = [
                        _bf2_pack(
                            _bf2_lo(a4[e]) * _sigmoid_fast(_bf2_lo(g4[e])),
                            _bf2_hi(a4[e]) * _sigmoid_fast(_bf2_hi(g4[e])),
                        )
                        for e in range(4)
                    ]
                    if (k < rows_m * cpr) & (grow < rows):
                        _stg4(
                            out_base
                            + ((cutlass.Int64(grow) * heads + ghead) * HEAD_DIM + (k % cpr) * 8)
                            * 2,
                            res[0],
                            res[1],
                            res[2],
                            res[3],
                        )

    @cute.jit
    def fixup(
        self,
        tidx,
        unit,
        kvh,
        r0,
        rows,
        nsplit,
        bal_base,
        cnt_base,
        part_base,
        lse_base,
        out_base,
        gate_base,
        gate_stride,
        gate_hs,
        s_flag,
        s_w,
    ):
        """Balanced mode, a unit of several CTAs: the unit's last CTA to finish (a ticket a unit)
        merges its partials in slot order as `_RowMergeKernel` does, applies the gate, stores the
        rows and resets the ticket; the result does not depend on which CTA merges."""
        group: cutlass.Constexpr = self.group
        heads: cutlass.Constexpr = self.heads
        rows_m: cutlass.Constexpr = self.rows_m
        rpr: cutlass.Constexpr = self.T
        d4: cutlass.Constexpr = HEAD_DIM // 4
        n_c: cutlass.Constexpr = rows_m * d4
        per: cutlass.Constexpr = -(-n_c // THREADS)
        nsp: cutlass.Constexpr = MAX_BAL_SPLITS
        neg_inf = -Float32.inf
        # the CTA's partials and log-sum-exps (stored before the barrier) released by thread 0's
        # acq_rel ticket; the last CTA's thread 0 acquires the others', the barrier passes them on
        cute.arch.sync_threads()
        if tidx == 0:
            _sts1(s_flag, _atom_add_acqrel_g(cnt_base + cutlass.Int64(unit) * 4, cutlass.Int32(1)))
        cute.arch.sync_threads()
        if _lds1(s_flag) == nsplit - 1:
            if tidx == 0:
                _stg1(cnt_base + cutlass.Int64(unit) * 4, cutlass.Int32(0))
            # weights of each (verify row, head) of the unit: slots past `nsplit` weigh 0
            for i in cutlass.range_constexpr(-(-rows_m // THREADS)):
                k = tidx + i * THREADS
                if k < rows_m:
                    head = kvh * group + k % group
                    ls = []
                    for s_ in cutlass.range_constexpr(nsp):
                        slot = bal_base + cutlass.min(cutlass.Int32(s_), nsplit - 1)
                        lv = _bits_f32(
                            _ldg1cg(
                                lse_base
                                + ((cutlass.Int64(slot) * rpr + k // group) * heads + head) * 4
                            )
                        )
                        ls.append(Float32(cutlass.select_(nsplit > s_, lv, neg_inf)))
                    mx = _tree_fmax(ls)
                    ws = [
                        Float32(
                            cutlass.select_(mx > neg_inf, cute.math.exp2(lv - mx), Float32(0.0))
                        )
                        for lv in ls
                    ]
                    for s_ in cutlass.range_constexpr(nsp):
                        _sts1(s_w + (s_ * rows_m + k) * 4, _f32_bits(ws[s_]))
                    _sts1(s_w + (nsp * rows_m + k) * 4, _f32_bits(_tree_sum(ws)))
            cute.arch.sync_threads()
            for j in cutlass.range_constexpr(per):
                c = tidx + THREADS * j
                k = cutlass.min(c // d4, rows_m - 1)
                d = (c % d4) * 4
                t = k // group
                head = kvh * group + k % group
                row = r0 + t
                ps = [
                    _ldg4cg(
                        part_base
                        + (
                            (
                                (
                                    cutlass.Int64(
                                        bal_base + cutlass.min(cutlass.Int32(s_), nsplit - 1)
                                    )
                                    * rpr
                                    + t
                                )
                                * heads
                                + head
                            )
                            * HEAD_DIM
                            + d
                        )
                        * 4
                    )
                    for s_ in range(nsp)
                ]
                gw = None
                if const_expr(self.gated):
                    gw = _ldg2(
                        gate_base
                        + (
                            cutlass.Int64(cutlass.min(row, rows - 1)) * gate_stride
                            + cutlass.Int64(head) * gate_hs
                            + d
                        )
                        * 2
                    )
                ws = [_bits_f32(_lds1(s_w + (s_ * rows_m + k) * 4)) for s_ in range(nsp)]
                den = _bits_f32(_lds1(s_w + (nsp * rows_m + k) * 4))
                res = []
                for e in cutlass.range_constexpr(4):
                    acc = _tree_sum([ws[s_] * _bits_f32(ps[s_][e]) for s_ in range(nsp)])
                    merged = Float32(cutlass.select_(den > Float32(0.0), acc / den, Float32(0.0)))
                    if const_expr(self.gated):
                        if const_expr(e % 2 == 0):
                            gx = _bf2_lo(gw[e // 2])
                        else:
                            gx = _bf2_hi(gw[e // 2])
                        merged = _round_bf16(merged) * _sigmoid_fast(gx)
                    res.append(merged)
                if (c < n_c) & (row < rows):
                    _stg2(
                        out_base + ((cutlass.Int64(row) * heads + head) * HEAD_DIM + d) * 2,
                        _bf2_pack(res[0], res[1]),
                        _bf2_pack(res[2], res[3]),
                    )


#: Threads of a merge CTA (four values a thread at a time).
MERGE_THREADS = 256
#: Merge CTAs a row, each a contiguous share of the row's heads x head_dim values.
MERGE_PARTS = 8


class _RowMergeKernel:
    """The split partials of a row range merged as the per-row kernel's merge does
    (log2-sum-exp weights, fp32), the gate applied to the bf16-rounded result: `parts` CTAs a
    row, every load of a thread issued before the first use."""

    def __init__(self, *, heads, splits, gated, pdl, parts):
        self.heads, self.splits, self.gated, self.pdl = heads, splits, gated, pdl
        self.parts = parts

    @cute.jit
    def __call__(
        self,
        gate_t: cute.Tensor,
        part_ptr: cute.Pointer,
        lse_ptr: cute.Pointer,
        out_ptr: cute.Pointer,
        rows: cutlass.Int32,
        gate_stride: cutlass.Int32,
        gate_hs: cutlass.Int32,
        stream,
    ):
        self.kernel(
            part_ptr.toint(),
            lse_ptr.toint(),
            out_ptr.toint(),
            gate_t.iterator.toint(),
            rows,
            gate_stride,
            gate_hs,
        ).launch(
            grid=[rows * self.parts, 1, 1],
            block=[MERGE_THREADS, 1, 1],
            stream=stream,
            use_pdl=self.pdl,
        )

    @cute.kernel
    def kernel(
        self,
        part_base: cutlass.Int64,
        lse_base: cutlass.Int64,
        out_base: cutlass.Int64,
        gate_base: cutlass.Int64,
        rows: cutlass.Int32,
        gate_stride: cutlass.Int32,
        gate_hs: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        heads: cutlass.Constexpr = self.heads
        sp: cutlass.Constexpr = self.splits
        parts: cutlass.Constexpr = self.parts
        n_all: cutlass.Constexpr = heads * HEAD_DIM // 4
        n_c: cutlass.Constexpr = -(-n_all // parts)
        per: cutlass.Constexpr = -(-n_c // MERGE_THREADS)
        neg_inf = -Float32.inf
        row = bidx // parts
        c0 = (bidx % parts) * n_c
        r64 = cutlass.Int64(row)
        cs = [
            cutlass.min(c0 + cutlass.min(tidx + MERGE_THREADS * j, n_c - 1), n_all - 1)
            for j in range(per)
        ]
        gws = []
        if const_expr(self.gated):
            # the gate is an input of the attention call, ready before the split kernel ran
            for c in cs:
                head = c // (HEAD_DIM // 4)
                d = (c % (HEAD_DIM // 4)) * 4
                gws.append(
                    _ldg2(gate_base + (r64 * gate_stride + cutlass.Int64(head) * gate_hs + d) * 2)
                )
        if const_expr(self.pdl):
            cute.arch.griddepcontrol_launch_dependents()
            cute.arch.griddepcontrol_wait()
        lss = []
        pss = []
        for j in cutlass.range_constexpr(per):
            head = cs[j] // (HEAD_DIM // 4)
            lss.append(
                [
                    _bits_f32(
                        _ldg1(lse_base + ((cutlass.Int64(s_) * rows + r64) * heads + head) * 4)
                    )
                    for s_ in range(sp)
                ]
            )
            pss.append(
                [
                    _ldg4(
                        part_base
                        + ((cutlass.Int64(s_) * rows + r64) * (heads * HEAD_DIM) + 4 * cs[j]) * 4
                    )
                    for s_ in range(sp)
                ]
            )
        for j in cutlass.range_constexpr(per):
            ls = lss[j]
            mx = _tree_fmax(ls)
            ws = [
                Float32(cutlass.select_(mx > neg_inf, cute.math.exp2(lv - mx), Float32(0.0)))
                for lv in ls
            ]
            den = _tree_sum(ws)
            res = []
            for e in cutlass.range_constexpr(4):
                acc = _tree_sum([ws[s_] * _bits_f32(pss[j][s_][e]) for s_ in range(sp)])
                merged = Float32(0.0)
                if den > Float32(0.0):
                    merged = acc / den
                if const_expr(self.gated):
                    if const_expr(e % 2 == 0):
                        gx = _bf2_lo(gws[j][e // 2])
                    else:
                        gx = _bf2_hi(gws[j][e // 2])
                    merged = _round_bf16(merged) * _sigmoid_fast(gx)
                res.append(merged)
            if (tidx + MERGE_THREADS * j < n_c) & (c0 + tidx + MERGE_THREADS * j < n_all):
                _stg2(
                    out_base + (r64 * (heads * HEAD_DIM) + 4 * cs[j]) * 2,
                    _bf2_pack(res[0], res[1]),
                    _bf2_pack(res[2], res[3]),
                )


#: Balanced launch: most CTAs of one unit (the last CTA's merge reads this many partials).
MAX_BAL_SPLITS = 4
#: The plan kernel's threads a CTA, and tokens read a four-column quad (1: the quad's first token,
#: which names the group in the expansion layout, where a quad's valid tokens are a prefix).
PLAN_THREADS, PLAN_QUAD = 512, 1
#: Most bitmap words (32 four-token groups each) of a plan CTA: a 262144-token block table.
MAX_PLAN_WORDS = 4096


class _BalPlanKernel:
    """Balanced launch, step 1, one CTA a request group: the group's union tiles (distinct
    four-token groups its valid rows select, / `GPT`, at least 1) to `est`, for
    `_GroupedKernel._bal_schedule`."""

    def __init__(self, *, rpr, width, words, num_requests, pdl):
        self.T, self.width, self.words = rpr, width, words
        self.num_requests, self.pdl = num_requests, pdl

    @cute.jit
    def __call__(
        self,
        sel_t: cute.Tensor,
        req_ptr: cute.Pointer,
        est_ptr: cute.Pointer,
        rows: cutlass.Int32,
        sel_stride: cutlass.Int32,
        n_groups: cutlass.Int32,
        stream,
    ):
        @cute.struct
        class SharedStorage:
            red: cute.struct.MemRange[Int32, 32]
            bm: cute.struct.Align[cute.struct.MemRange[Int32, self.words], 16]

        self.shared_storage = SharedStorage
        req = cute.make_tensor(req_ptr, cute.make_layout((rows,), stride=(1,)))
        self.kernel(sel_t.iterator.toint(), req, est_ptr.toint(), rows, sel_stride).launch(
            grid=[n_groups, 1, 1],
            block=[PLAN_THREADS, 1, 1],
            stream=stream,
            use_pdl=self.pdl,
        )

    @cute.kernel
    def kernel(
        self,
        sel_base: cutlass.Int64,
        mReq: cute.Tensor,
        est_base: cutlass.Int64,
        rows: cutlass.Int32,
        sel_stride: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        rpr: cutlass.Constexpr = self.T
        width: cutlass.Constexpr = self.width
        words: cutlass.Constexpr = self.words
        n_q: cutlass.Constexpr = -(-width // 4)
        n_i: cutlass.Constexpr = -(-n_q // PLAN_THREADS)
        nwarps: cutlass.Constexpr = PLAN_THREADS // 32
        smem = SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        s_red = cutlass.Int32(storage.red.data_ptr().toint())
        s_bm = cutlass.Int32(storage.bm.data_ptr().toint())
        r0 = bidx * rpr
        for i in cutlass.range_constexpr(-(-words // PLAN_THREADS)):
            idx = tidx + i * PLAN_THREADS
            if idx < words:
                _sts1(s_bm + idx * 4, cutlass.Int32(0))
        if const_expr(self.pdl):
            cute.arch.griddepcontrol_wait()
            cute.arch.griddepcontrol_launch_dependents()
        # the first valid token of a quad names its four-token group
        nu: cutlass.Constexpr = PLAN_QUAD
        toks = []
        oks = []
        for t in cutlass.range_constexpr(rpr):
            row = r0 + t
            xr = cutlass.Int64(mReq[cutlass.min(row, rows - 1)])
            oks.append((row < rows) & (xr >= 0) & (xr < self.num_requests))
            row_sel = sel_base + cutlass.Int64(cutlass.min(row, rows - 1)) * sel_stride * 4
            for i in cutlass.range_constexpr(n_i):
                for u in cutlass.range_constexpr(nu):
                    colc = cutlass.min(4 * (tidx + i * PLAN_THREADS) + u, width - 1)
                    toks.append(_ldg1(row_sel + cutlass.Int64(colc * 4)))
        cute.arch.sync_threads()
        for t in cutlass.range_constexpr(rpr):
            for i in cutlass.range_constexpr(n_i):
                q4 = tidx + i * PLAN_THREADS
                gid = cutlass.Int32(-1)
                for u in cutlass.range_constexpr(nu):
                    tok = toks[(t * n_i + i) * nu + u]
                    ok = oks[t] & (4 * q4 + u < width) & (tok >= 0)
                    gid = cutlass.Int32(cutlass.select_(ok & (gid < 0), tok >> 2, gid))
                if (gid >= 0) & (gid < words * 32):
                    _red_or_shared(s_bm + (gid >> 5) * 4, cutlass.Int32(1) << (gid & 31))
        cute.arch.sync_threads()
        cnt = cutlass.Int32(0)
        for i in cutlass.range_constexpr(-(-words // PLAN_THREADS)):
            idx = tidx + i * PLAN_THREADS
            if idx < words:
                cnt = cnt + _popc(_lds1(s_bm + idx * 4))
        cnt = cute.arch.warp_reduction_sum(cnt)
        if lane == 0:
            _sts1(s_red + warp * 4, cnt)
        cute.arch.sync_threads()
        if tidx == 0:
            n = cutlass.Int32(0)
            for w in cutlass.range_constexpr(nwarps):
                n = n + _lds1(s_red + w * 4)
            _stg1(
                est_base + cutlass.Int64(bidx) * 4,
                cutlass.max((n + GPT - 1) // GPT, cutlass.Int32(1)),
            )


# --- launch ---

_COMPILED: dict = {}
#: Geometries whose every reachable launch plan is compiled.
_PLANNED: set = set()


@functools.lru_cache(maxsize=None)
def _num_sms(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def plan_splits(n_groups: int, kv_heads: int, width: int, num_sms: int) -> int:
    """Splits so the grid fills the GPU once (one CTA an SM): `num_sms // units` CTAs a unit, and
    three splits from half to three quarters of a wave while they fit in two waves
    (3 * units <= 2 * num_sms). Outputs of different splits differ in the last bits."""
    est_tiles = -(-int(width * 1.25) // BN)
    units = n_groups * kv_heads
    band = num_sms // 2 < units <= (num_sms * 3) // 4
    if band and 3 * units <= 2 * num_sms:
        return min(3, est_tiles)
    return max(1, min(MAX_PLAN_SPLITS, est_tiles, num_sms // units))


def plan_launch(n_groups: int, kv_heads: int, width: int, num_sms: int):
    """(splits, n_whole) of a launch: n_whole None = every unit (group, KV head) takes `splits`
    CTAs (one wave: `plan_splits`); else units below n_whole take one CTA and the units of the
    last, partial wave (whole groups) `splits` CTAs each, merged afterwards."""
    units = n_groups * kv_heads
    if units <= num_sms:
        return plan_splits(n_groups, kv_heads, width, num_sms), None
    waves = -(-units // num_sms)
    tail = units - (waves - 1) * num_sms
    tail = -(-tail // kv_heads) * kv_heads
    tail_splits = min(MAX_TAIL_SPLITS, num_sms // tail)
    if tail_splits < 2:
        return 1, None
    return tail_splits, units - tail


def reachable_plans(max_groups: int, kv_heads: int, width: int, num_sms: int) -> list:
    """Every `(splits, tail)` pair `plan_launch` returns for `MIN_GROUPS..max_groups` groups."""
    plans = set()
    for n_groups in range(MIN_GROUPS, max(MIN_GROUPS, max_groups) + 1):
        splits, n_whole = plan_launch(n_groups, kv_heads, width, num_sms)
        plans.add((max(1, min(MAX_SPLITS, int(splits))), n_whole is not None))
    return sorted(plans)


#: Launches from this many units (request groups x KV heads) to one wave take the balanced plan
#: (`_BalPlanKernel`, then the grouped kernel's balanced mode) instead of `plan_launch`'s. A unit's
#: split count depends on the batch, so a row's last bits depend on the other requests' unions.
BALANCED_MIN_UNITS = 88


def takes_balanced(units: int, num_sms: int) -> bool:
    """Whether a launch of `units` (group, KV head) units takes the balanced plan: from
    `BALANCED_MIN_UNITS` units to one wave."""
    return BALANCED_MIN_UNITS <= units <= num_sms


def _plan_words(bt_width: int, tpb: int) -> int:
    """Bitmap words (32 four-token groups each) of a plan CTA over a block table."""
    return -(-(bt_width * tpb // 4) // 32)


#: The balanced launch's persistent int32 words a device (zeros once; the kernels leave every
#: ticket at zero): one launch at a time on a device.
_BAL_BUFS: dict = {}


def _bal_buffers(device: torch.device, n: int) -> torch.Tensor:
    """`[8 n + 16]` int32 on `device`: each CTA's plan entry (4 words a CTA, words 0..4n), the
    units' merge tickets (4n..5n), the groups' union tiles (5n + 4 .. 6n + 4)."""
    key = (device.index if device.index is not None else torch.cuda.current_device(), n)
    buf = _BAL_BUFS.get(key)
    if buf is None:
        buf = torch.zeros(8 * n + 16, dtype=torch.int32, device=device)
        _BAL_BUFS[key] = buf
    return buf


def _compile_plan(pkey: tuple, plan_kw: dict, plan_args: tuple):
    """The balanced launch's plan kernel of `pkey`, compiled once."""
    fn = _COMPILED.get(pkey)
    if fn is None:
        fn = cute.compile(
            _BalPlanKernel(**plan_kw), *plan_args, options="--opt-level 2 --enable-tvm-ffi"
        )
        _COMPILED[pkey] = fn
    return fn


def takes_grouped(q, k_cache, selected_tokens, rows_per_request) -> bool:
    """Whether `sparse_gqa_grouped` serves this call (on shapes only, so a per-capture constant):
    an e4m3 pool on SM103, 2..4 rows a request, `MIN_GROUPS` groups, a shared-memory plan."""
    if q.dim() != 3 or k_cache.dim() != 4 or k_cache.dtype != torch.float8_e4m3fn:
        return False
    rows, num_q_heads, head_dim = q.shape
    rpr = int(rows_per_request)
    if rpr < 2 or rpr > MAX_RPR or -(-rows // rpr) < MIN_GROUPS or head_dim != HEAD_DIM:
        return False
    if k_cache.shape[3] != HEAD_DIM or num_q_heads % k_cache.shape[1]:
        return False
    if k_cache.stride(3) != 1 or k_cache.stride(2) != HEAD_DIM:
        return False
    if torch.cuda.get_device_capability(q.device) != (10, 3):
        return False
    group = num_q_heads // k_cache.shape[1]
    return _layout(rpr, group, selected_tokens.shape[1], 1) is not None


def _compile_grouped(key: tuple, kern_kw: dict, compile_args: tuple):
    """The grouped kernel of `key` (its `splits` and launch mode -- False, True for the tail
    mode, "bal" for the balanced one -- at positions 5 and 6), compiled once."""
    fn = _COMPILED.get(key)
    if fn is None:
        kern = _GroupedKernel(splits=key[5], tail=key[6] is True, bal=key[6] == "bal", **kern_kw)
        fn = cute.compile(kern, *compile_args, options="--opt-level 2 --enable-tvm-ffi")
        _COMPILED[key] = fn
    return fn


def _compile_merge(heads: int, splits: int, gated: bool, pdl: bool, merge_args: tuple, gdim: int):
    """The row merge of `splits` partials, compiled once per gate tensor rank `gdim` (the
    trunk's `[rows, Hq * D]` gate, the draft layer's `[rows, Hq, D]` view, q when ungated)."""
    mkey = ("rowmerge", heads, splits, gated, pdl, MERGE_PARTS, gdim)
    mfn = _COMPILED.get(mkey)
    if mfn is None:
        mfn = cute.compile(
            _RowMergeKernel(heads=heads, splits=splits, gated=gated, pdl=pdl, parts=MERGE_PARTS),
            *merge_args,
            options="--opt-level 2 --enable-tvm-ffi",
        )
        _COMPILED[mkey] = mfn
    return mfn


def sparse_gqa_grouped(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    selected_tokens: torch.Tensor,
    request_indices: torch.Tensor,
    tokens_per_block: int,
    softmax_scale: float,
    rows_per_request: int,
    output_gate: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """The grouped kernel (and the row merge with more than one split) over an e4m3 pool; see
    `QsaSparseGqaGroupedCute` for the contract."""
    rows, num_q_heads, head_dim = q.shape
    num_pages, num_kv_heads, tpb, dim = k_cache.shape
    rpr = int(rows_per_request)
    assert head_dim == HEAD_DIM and dim == HEAD_DIM and tpb == tokens_per_block
    assert k_cache.dtype == torch.float8_e4m3fn and v_cache.dtype == k_cache.dtype
    assert q.dtype == torch.bfloat16 and q.stride(2) == 1
    assert (q.stride(0) * 2) % 16 == 0 and (q.stride(1) * 2) % 16 == 0 and q.data_ptr() % 16 == 0
    assert k_cache.stride() == v_cache.stride() and k_cache.stride(3) == 1
    assert k_cache.stride(2) == HEAD_DIM, "a page's token rows contiguous (a group: one bulk copy)"
    assert v_cache.shape == k_cache.shape
    assert k_cache.data_ptr() % 16 == 0 and v_cache.data_ptr() % 16 == 0
    assert all(s % 16 == 0 for s in k_cache.stride()[:3]), "16-byte aligned K/V rows"
    assert block_table.dtype == torch.int32 and block_table.stride(1) == 1
    assert block_table.stride(0) == block_table.shape[1], "row-contiguous block table"
    assert selected_tokens.dtype == torch.int32 and selected_tokens.stride(1) == 1
    assert request_indices.is_contiguous() and request_indices.dtype in (torch.int32, torch.int64)
    group = num_q_heads // num_kv_heads
    width = selected_tokens.shape[1]
    out = torch.empty((rows, num_q_heads, head_dim), dtype=torch.bfloat16, device=q.device)
    if rows == 0:
        return out
    n_groups = -(-rows // rpr)
    units = n_groups * num_kv_heads
    n_whole = None
    n_sms = _num_sms(q.device.index or 0)
    words = _plan_words(block_table.shape[1], tpb)
    bal_fits = (
        words <= MAX_PLAN_WORDS and units <= n_sms <= THREADS and _layout(rpr, group, width, 1)
    )
    bal = bool(bal_fits and takes_balanced(units, n_sms))
    if bal:
        splits = 1
    else:
        splits, n_whole = plan_launch(n_groups, num_kv_heads, width, n_sms)
    n_splits = max(1, min(MAX_SPLITS, int(splits)))
    tail = n_whole is not None
    assert _layout(rpr, group, width, 1 if (tail or bal) else n_splits) is not None, (
        "no shared-memory plan for this geometry: see `takes_grouped`"
    )
    use_pdl = bool(get_env_enable_pdl())
    gated = output_gate is not None
    gate_hs = head_dim
    if gated:
        assert output_gate.dtype == torch.bfloat16 and output_gate.stride(-1) == 1
        if output_gate.dim() == 3:
            assert output_gate.shape == q.shape
            gate_hs = output_gate.stride(1)
        else:
            assert output_gate.shape == (rows, num_q_heads * head_dim)
        assert (output_gate.stride(0) * 2) % 16 == 0 and (gate_hs * 2) % 16 == 0
        assert output_gate.data_ptr() % 16 == 0, "16-byte aligned gate rows"
    req64 = request_indices.dtype == torch.int64
    bbuf = _bal_buffers(q.device, n_sms)
    if bal:
        n_whole, tail_row0, n_ctas = 0, rows, n_sms
    elif tail:
        tail_row0 = (n_whole // num_kv_heads) * rpr
        n_ctas = n_whole + (units - n_whole) * n_splits
    elif n_splits > 1:
        n_whole, tail_row0, n_ctas = 0, 0, units * n_splits
    else:
        n_whole, tail_row0, n_ctas = units, rows, units
    merged = rows - tail_row0
    key = (
        rpr,
        group,
        num_kv_heads,
        width,
        tpb,
        n_splits,
        "bal" if bal else tail,
        float(softmax_scale),
        num_pages,
        tuple(block_table.shape),
        tuple(k_cache.stride()[:3]),
        tuple(q.stride()[:2]),
        req64,
        gated,
        use_pdl,
    )
    if bal:
        # a partial slot a CTA: its unit's T rows
        part = torch.empty(
            (n_ctas, rpr, num_q_heads, head_dim), dtype=torch.float32, device=q.device
        )
        lse = torch.empty((n_ctas, rpr, num_q_heads), dtype=torch.float32, device=q.device)
    elif merged > 0:
        part = torch.empty(
            (n_splits, merged, num_q_heads, head_dim), dtype=torch.float32, device=q.device
        )
        lse = torch.empty((n_splits, merged, num_q_heads), dtype=torch.float32, device=q.device)
    else:
        part, lse = out, out
    gate_t = output_gate if gated else q
    gate_stride = output_gate.stride(0) if gated else 0
    compile_args = (
        # read with scalar loads: a row slice of the selection needs no copy
        from_dlpack(selected_tokens, assumed_align=4).mark_layout_dynamic(leading_dim=1),
        _ptr(q, cutlass.BFloat16),
        _ptr(k_cache, cutlass.Uint8),
        _ptr(v_cache, cutlass.Uint8),
        _ptr(block_table, cutlass.Int32, 4),
        _ptr(request_indices, cutlass.Int64 if req64 else cutlass.Int32, 8 if req64 else 4),
        _ptr(out, cutlass.BFloat16),
        _ptr(part, cutlass.Float32),
        _ptr(lse, cutlass.Float32),
        _ptr(gate_t, cutlass.BFloat16),
        cutlass.Int32(rows),
        cutlass.Int32(selected_tokens.stride(0)),
        cutlass.Int32(gate_stride),
        cutlass.Int32(gate_hs),
        cutlass.Int32(n_whole),
        cutlass.Int32(tail_row0),
        cutlass.Int32(n_ctas),
        _ptr(bbuf, cutlass.Int32, 16),
        _ptr(bbuf, cutlass.Int32, 4),
        _ptr(bbuf, cutlass.Int32, 4),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
    )
    tab_addr = bbuf.data_ptr()
    cnt_addr = tab_addr + 4 * (4 * n_sms)
    est_addr = tab_addr + 4 * (5 * n_sms + 4)
    plan_kw = dict(
        rpr=rpr, width=width, words=words, num_requests=block_table.shape[0], pdl=use_pdl
    )
    pkey = ("balplan", rpr, width, words, block_table.shape[0], req64, use_pdl)
    plan_args = (
        from_dlpack(selected_tokens, assumed_align=4).mark_layout_dynamic(leading_dim=1),
        _ptr(request_indices, cutlass.Int64 if req64 else cutlass.Int32, 8 if req64 else 4),
        _ptr(bbuf, cutlass.Int32, 4),
        cutlass.Int32(rows),
        cutlass.Int32(selected_tokens.stride(0)),
        cutlass.Int32(n_groups),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
    )
    merge_args = (
        from_dlpack(gate_t, assumed_align=16).mark_layout_dynamic(leading_dim=gate_t.dim() - 1),
        _ptr(part, cutlass.Float32),
        _ptr(lse, cutlass.Float32, 4),
        _ptr(out, cutlass.BFloat16, 8),
        cutlass.Int32(merged),
        cutlass.Int32(gate_stride),
        cutlass.Int32(gate_hs),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
    )
    kern_kw = dict(
        rpr=rpr,
        group=group,
        kv_heads=num_kv_heads,
        width=width,
        tpb=tpb,
        scale=float(softmax_scale),
        num_pages=num_pages,
        bt_width=block_table.shape[1],
        num_requests=block_table.shape[0],
        k_strides=tuple(k_cache.stride()[:3]),
        q_strides=tuple(q.stride()[:2]),
        req64=req64,
        gated=gated,
        pdl=use_pdl,
        bal_ctas=n_sms,
    )
    # A launch plan depends on the call's row count: the first call of a geometry compiles every
    # plan its row counts can reach, so no serving step compiles one.
    if key[:5] + key[7:] not in _PLANNED:
        _PLANNED.add(key[:5] + key[7:])
        max_groups = block_table.shape[0]
        for plan_splits_, plan_tail in reachable_plans(
            max_groups, num_kv_heads, width, _num_sms(q.device.index or 0)
        ):
            if _layout(rpr, group, width, 1 if plan_tail else plan_splits_) is None:
                continue
            _compile_grouped(key[:5] + (plan_splits_, plan_tail) + key[7:], kern_kw, compile_args)
            if plan_tail or plan_splits_ > 1:
                _compile_merge(num_q_heads, plan_splits_, gated, use_pdl, merge_args, gate_t.dim())
        if bal_fits:
            _compile_grouped(key[:5] + (1, "bal") + key[7:], kern_kw, compile_args)
            _compile_plan(pkey, plan_kw, plan_args)
    fn = _compile_grouped(key, kern_kw, compile_args)
    if bal:
        pfn = _compile_plan(pkey, plan_kw, plan_args)
        pfn(
            selected_tokens,
            request_indices.data_ptr(),
            est_addr,
            rows,
            selected_tokens.stride(0),
            n_groups,
        )
    fn(
        selected_tokens,
        q.data_ptr(),
        k_cache.data_ptr(),
        v_cache.data_ptr(),
        block_table.data_ptr(),
        request_indices.data_ptr(),
        out.data_ptr(),
        part.data_ptr(),
        lse.data_ptr(),
        gate_t.data_ptr(),
        rows,
        selected_tokens.stride(0),
        gate_stride,
        gate_hs,
        n_whole,
        tail_row0,
        n_ctas,
        tab_addr,
        cnt_addr,
        est_addr,
    )
    if bal or merged <= 0:
        return out
    mfn = _compile_merge(num_q_heads, n_splits, gated, use_pdl, merge_args, gate_t.dim())
    gate_tail = gate_t[tail_row0:] if gated else gate_t
    mfn(
        gate_tail,
        part.data_ptr(),
        lse.data_ptr(),
        out[tail_row0:].data_ptr(),
        merged,
        gate_stride,
        gate_hs,
    )
    return out


# --- catalog entry ---

#: The checkpoint's QSA geometry: 32-token pages, 2051 selected tokens a row, 256 head dims, a
#: `max_seq_len` 262144 table.
_QSA = dict(tokens_per_block=32, compress_ratio=4, token_topk=2048, head_dim=256, pages=8196)
_TEP2 = dict(
    _QSA, q_heads=12, kv_heads=1, slots=128, slot_pages=24, layer=11, kv=torch.float8_e4m3fn
)
_DEP4 = dict(
    _QSA, q_heads=24, kv_heads=2, slots=128, slot_pages=26, layer=11, kv=torch.float8_e4m3fn
)
_TEP2_ROW = dict(gate="row", row_heads=29)
_DEP4_ROW = dict(gate="row", row_heads=55)


def _grouped(member: dict, requests: int, rows: int = 3, last=60000, step=900, **kw) -> dict:
    """A verify step of `requests` requests of `rows` rows each, request r's rows ending at
    `last - step * r`; `overlap` spreads a request's rows' block scores (None: independent rows)."""
    segments = tuple((last + 1 - rows - step * r, rows) for r in range(requests))
    return dict(member, segments=segments, rows_per_request=rows, **{"overlap": 0.05, **kw})


class QsaSparseGqaGroupedCute(OpWrapper):
    """`qsa_paged_sparse_gqa` for decode rows that come `rows_per_request` consecutive rows a
    request (an MTP verify step) over an e4m3 paged pool: `[rows, Hq, D]` bf16. With `output_gate`
    (`[rows, Hq * D]`, or `[rows, Hq, D]` with any head stride; bf16, unit inner stride) the
    result is `bf16(attn) * sigmoid(gate)` rounded once more to bf16.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="ht_tep2 MTP-2 verify: 128 x 3 = 384 rows, one CTA a group (gate fused into its "
            "store), captured",
            spec=_grouped(_TEP2, 128, step=400, capture=True, **_TEP2_ROW),
        ),
        Cell(
            why="ht_tep2 MTP-2 verify: 160 x 3 = 480 rows, one CTA a group past the first wave, "
            "the last wave's groups split 4 ways and merged, gated",
            spec=_grouped(dict(_TEP2, slots=160), 160, step=300, **_TEP2_ROW),
        ),
        Cell(
            why="8 requests x 3 rows deep in long sequences (196k..167k tokens): the fewest "
            "groups the kernel takes, 18 splits, gated",
            spec=_grouped(_TEP2, 8, last=196607, step=4099, **_TEP2_ROW),
        ),
        Cell(
            why="16 x 3 rows whose selections are independent (union ~3x a row's), ungated",
            spec=_grouped(_TEP2, 16, overlap=None),
        ),
        Cell(
            why="ht_dep4 MTP-3 verify: 8 x 4 = 32 rows, 24 q / 2 KV heads over the e4m3 trunk "
            "(26 role pages a slot), gated, captured",
            spec=_grouped(_DEP4, 8, rows=4, capture=True, **_DEP4_ROW),
        ),
        Cell(
            why="draft layer: 16 x 3 rows, the gate half of the interleaved q projection (head "
            "stride 512)",
            spec=_grouped(_TEP2, 16, gate="interleaved"),
        ),
        Cell(
            why="groups spanning requests (3, 3, 1, 2, 3 x 6, 2 rows: misaligned and partial "
            "groups), short sequences, one foreign row, a row slice of a wider selection, "
            "ungated",
            spec=dict(
                _TEP2,
                segments=((77, 3), (2047, 3), (9000, 1), (30000, 2))
                + tuple((40000 + 1000 * r, 3) for r in range(6))
                + ((5, 2),),
                rows_per_request=3,
                overlap=0.05,
                row_offset=1,
                foreign=True,
            ),
        ),
    )

    note = """
    Source: this target's CuTe DSL kernels (`_GroupedKernel`, `_RowMergeKernel`, `_BalPlanKernel`),
    `cute.compile`d per (T, splits, launch mode); every reachable plan of a geometry is compiled on
    its first call. Called by `modeling._sparse_gqa` for verify / draft-extend rows over an e4m3
    pool with `rows_per_request` > 1 where `serves` (an e4m3 pool on SM103, 2..4 rows a request,
    at least `MIN_GROUPS` groups, T x q heads a KV head <= 128); the per-row
    `qsa_sparse_gqa_splitk_cute` otherwise. Graph-capturable (the row count is a runtime value).

    Numerics: K/V widened exactly to f16, q rounded to f16, f16 MMAs with fp32 accumulation, P
    rounded to f16, the softmax offset moved only past 2^8; within the decode op's band of the
    fp32 reference, not bitwise the per-row kernel. Deterministic for one batch.

    Preconditions: as `qsa_sparse_gqa_splitk_cute`, plus expanded selections: each four-column
    quad of a row names one four-token group at its own offsets (column % 4 == token % 4), no
    group twice in a row. A NaN in a token one row selects reaches the group's other rows of that
    request (P = 0 times NaN). A group's pool row `page * tokens_per_block + slot` is 32-bit
    (pools below 2^26 pages).

    Not certified: the three-split band of `plan_splits` (77..87 units on 152 SMs), T = 2.
    """

    serves = staticmethod(takes_grouped)

    def __call__(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        block_table: torch.Tensor,
        selected_tokens: torch.Tensor,
        request_indices: torch.Tensor,
        tokens_per_block: int,
        softmax_scale: float,
        rows_per_request: int,
        output_gate: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return sparse_gqa_grouped(
            q,
            k_cache,
            v_cache,
            block_table,
            selected_tokens,
            request_indices,
            tokens_per_block,
            softmax_scale,
            rows_per_request,
            output_gate=output_gate,
        )

    def reference(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        block_table: torch.Tensor,
        selected_tokens: torch.Tensor,
        request_indices: torch.Tensor,
        tokens_per_block: int,
        softmax_scale: float,
        rows_per_request: int,
        output_gate: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """The per-row entry's reference (fp32 sparse GQA over every resident selected token,
        rounded once; gated in fp32 over that rounding and rounded again). `rows_per_request`
        is ignored: the grouping changes which loads serve a row, not what it attends."""
        return qsa_sparse_gqa_splitk_cute.reference(
            q,
            k_cache,
            v_cache,
            block_table,
            selected_tokens,
            request_indices,
            tokens_per_block,
            softmax_scale,
            output_gate,
        )


qsa_sparse_gqa_grouped_cute = QsaSparseGqaGroupedCute()
