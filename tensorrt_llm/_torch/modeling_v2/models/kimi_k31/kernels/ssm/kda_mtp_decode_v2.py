# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 KDA multi-token verify, packed rows + fused gated RMSNorm: :func:`k31_kda_mtp_verify`.

``kda_decode_mtp_v2_kernel`` is the verify kernel for one runtime
contract: packed token rows (``1 + num_spec`` new rows per request, accepted counts indexed by
state-pool slot), the gated RMSNorm fused into the epilogue with a bf16 output, the 3xTF32
tcgen05 chunk recurrence, an fp32 state, K = V = 128, conv width 4, ``num_spec <= MAX_NUM_SPEC``.
It computes exactly what the engine's ``kda_decode_mtp_kernel`` (``trtllm::kda_mtp_decode``)
computes on that contract, with the same arithmetic in the same order, so the output, the
committed state, the conv windows and the replay caches are bitwise identical. See that module
for the math (replay of accepted drafts, chunk WY form, GEMM1 / GEMM3 / forward substitution /
GEMM2, in-place state commit after the golden token, the cluster-pair ``SPLIT_V == 2`` form).
The difference is the schedule of the latency-bound CTA:

* **Load phase.** The state slot is the first load. The slot-independent operands (new-token
  projections, conv weights, per-head constants, output-norm operands) are issued under it; the
  slot-dependent ones (the S0 TMA, the committed conv windows and every pending replay entry,
  accepted or not) as soon as it arrives. Each lane moves four consecutive channels of a row
  (8 B / 16 B vector loads) and keeps the raw words in registers until the accepted count fixes
  the row's smem position, so the CTA waits for one round trip per dependency level (slot, then
  operands) instead of one per staging step.
* **Precompute** works in smem only, with the q and k conv / SiLU / L2-norm chains of a token
  interleaved in one warp.
* **Tensor-core ordering.** The GEMM3 operand rows lie clear of the S0 boxes, so GEMM3 is issued
  before S0 moves into TMEM and runs under that copy; GEMM1 follows once its rows are written over
  the read boxes. While the tensor cores run, warps 1..15 stage Kbar, seed the GEMM2
  accumulator, build the B2 tiles and write the new tokens' replay-cache rows and conv-window
  commits (vectorized, from smem).

One CTA per (head, request) tile, or a cluster pair while ``HV * N <= SPLIT_MAX_TILES``; the grid
has no co-residency requirement beyond a cluster pair, so any number of tiles runs (in waves when
there are more tiles than SMs). SM100 / SM103 / SM107.

The host side is ``_verify_mma_atoms``, ``_state_tma_atoms``, ``_run_kda_decode_mtp_v2`` and
``_verify_operands_ok``; the generic stride checks, DLPack bridging and layout keys are imported
from the engine op module (``cute_dsl_kimi_k3_kda_mtp_ops``).
"""

from __future__ import annotations

import functools
from typing import NamedTuple, Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, tcgen05
from cutlass.cute.typing import Int64
from cutlass.cutlass_dsl import T, dsl_user_op

from tensorrt_llm._torch.attention.kernels.blackwell.kimi_k3_kda.kda_mtp_decode import (
    MAX_NUM_SPEC,
    NUM_THREADS,
    SCRATCH_FLOATS,
    SPLIT_MAX_TILES,
    T_PAD,
    TILE_K,
    TMEM_COLS,
    TP_STRIDE,
    cvt_bf16x2,
    round_tf32,
    swz64_k16,
    tcgen05_fence_after_thread_sync,
    tcgen05_fence_before_thread_sync,
)
from tensorrt_llm._torch.custom_ops.cute_dsl_kimi_k3_kda_mtp_ops import (
    _beta_cache_assumed_align,
    _dlpack_arg,
    _layout_key,
    _require_stride_layout,
    _try_flatten_args,
)

# Offsets (in floats) of the operand tiles in kda_mtp_decode's SCRATCH_FLOATS region; all
# multiples of 256 (1024 B) for the 128B-swizzle atoms.
OFF_B4HI = 10240  # 16 x 16    Bcoef hi, immediately followed by Kbar hi -> N=80 GEMM2 half-0 tile
OFF_B4LO = 12544  # 16 x 16    Bcoef lo, immediately followed by Kbar lo
OFF_B1K = 6144  # 16 x 128   K~hi (duplicate, N=16 tile for the S0_lo sweep; A2 reuses it)
OFF_B1 = 0  # 48 x 128   [K~hi; Q~; K~lo]
OFF_A2HI = 6144  # 128 x 16   V'^T hi
OFF_A2LO = 8192  # 128 x 16   V'^T lo
OFF_B2HI = 10496  # 128 x 16   Kbar hi
OFF_B2LO = 12800  # 128 x 16   Kbar lo
OFF_A3 = 16384  # 64 x 128   [Kf; Qf hi; Qf lo; -] (clear of the S0 boxes)
OFF_B3HI = 24576  # 16 x 128   Kb hi
OFF_B3LO = 26624  # 16 x 128   Kb lo


def asm_words(n, asm, constraints, operands, *, loc, ip):
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * n),
        operands,
        asm,
        constraints,
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return [cutlass.Int32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip)) for i in range(n)]


@dsl_user_op
def ld_global_b32x2(addr: Int64, *, loc=None, ip=None):
    """Two 32-bit words (8 B, 8-byte aligned) from global memory, unconverted: the load has no
    consumer until its value is placed, so a warp keeps all of its loads in flight."""
    return asm_words(
        2,
        "ld.global.v2.b32 {$0, $1}, [$2];",
        "=r,=r,l",
        [Int64(addr).ir_value(loc=loc, ip=ip)],
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def ld_global_b32x4(addr: Int64, *, loc=None, ip=None):
    """Four 32-bit words (16 B, 16-byte aligned) from global memory, unconverted."""
    return asm_words(
        4,
        "ld.global.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        [Int64(addr).ir_value(loc=loc, ip=ip)],
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def ld_shared_f32x4(addr: cutlass.Int32, *, loc=None, ip=None):
    """Four consecutive fp32 (16-byte aligned) from shared memory."""
    words = asm_words(
        4,
        "ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        [cutlass.Int32(addr).ir_value(loc=loc, ip=ip)],
        loc=loc,
        ip=ip,
    )
    return [w.bitcast(cutlass.Float32) for w in words]


@dsl_user_op
def st_asm(asm, constraints, operands, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        operands,
        asm,
        constraints,
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def st_global_b32x2(addr, a, b):
    st_asm(
        "st.global.v2.b32 [$0], {$1, $2};",
        "l,r,r",
        [Int64(addr).ir_value(), cutlass.Int32(a).ir_value(), cutlass.Int32(b).ir_value()],
    )


def st_global_b32x4(addr, a, b, c, d):
    st_asm(
        "st.global.v4.b32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        [Int64(addr).ir_value()] + [cutlass.Int32(x).ir_value() for x in (a, b, c, d)],
    )


def st_shared_f32x4(addr, a, b, c, d):
    st_asm(
        "st.shared.v4.f32 [$0], {$1, $2, $3, $4};",
        "r,f,f,f,f",
        [cutlass.Int32(addr).ir_value()] + [cutlass.Float32(x).ir_value() for x in (a, b, c, d)],
    )


@dsl_user_op
def griddepcontrol_wait(*, loc=None, ip=None):
    st_asm("griddepcontrol.wait;", "", [], loc=loc, ip=ip)


@dsl_user_op
def griddepcontrol_launch_dependents(*, loc=None, ip=None):
    st_asm("griddepcontrol.launch_dependents;", "", [], loc=loc, ip=ip)


@dsl_user_op
def st_async_f32(addr, rank, mbar_addr, value, *, loc=None, ip=None):
    """Store ``value`` into CTA ``rank``'s copy of the shared-memory word at ``addr`` (this CTA's
    address of it) and count its 4 bytes on that CTA's copy of the mbarrier at ``mbar_addr``."""
    st_asm(
        "{\n.reg .b32 ra, rm;\nmapa.shared::cluster.u32 ra, $0, $1;\n"
        "mapa.shared::cluster.u32 rm, $2, $1;\n"
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [ra], $3, [rm];\n}",
        "r,r,r,f",
        [
            cutlass.Int32(addr).ir_value(loc=loc, ip=ip),
            cutlass.Int32(rank).ir_value(loc=loc, ip=ip),
            cutlass.Int32(mbar_addr).ir_value(loc=loc, ip=ip),
            cutlass.Float32(value).ir_value(loc=loc, ip=ip),
        ],
        loc=loc,
        ip=ip,
    )


_MBAR_WAIT_LABELS = [0]


@dsl_user_op
def mbarrier_wait_cluster(mbar_addr, *, loc=None, ip=None):
    """Wait for phase 0 of a local mbarrier completed by ``st.async`` from the cluster (acquire at
    cluster scope, so the stored words are visible afterwards)."""
    _MBAR_WAIT_LABELS[0] += 1
    label = f"KDA_DV_WAIT{_MBAR_WAIT_LABELS[0]}"
    st_asm(
        "{\n.reg .pred p;\n"
        f"{label}:\n"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [$0], 0;\n"
        f"@!p bra {label};\n}}",
        "r",
        [cutlass.Int32(mbar_addr).ir_value(loc=loc, ip=ip)],
        loc=loc,
        ip=ip,
    )


def _load_new_token_row(
    x_q,
    x_k,
    x_v,
    g,
    onorm_g,
    beta,
    tok_row,
    i_h,
    i_hv,
    hk_off,
    hv_off,
    k4,
    r_new,
    r_newb,
    use_flat_layout,
):
    """Issue one lane's loads of a new token's projections: q / k / v / gate / output gate four
    channels each into ``r_new``, beta into ``r_newb``."""
    if cutlass.const_expr(use_flat_layout):
        ld_row4(x_q, (0, tok_row, hk_off + k4), r_new, 0)
        ld_row4(x_k, (0, tok_row, hk_off + k4), r_new, 4)
        ld_row4(x_v, (0, tok_row, hv_off + k4), r_new, 8)
    else:
        ld_row4(x_q, (0, tok_row, i_h, k4), r_new, 0)
        ld_row4(x_k, (0, tok_row, i_h, k4), r_new, 4)
        ld_row4(x_v, (0, tok_row, i_hv, k4), r_new, 8)
    ld_row4(g, (0, tok_row, i_hv, k4), r_new, 12)
    ld_row4(onorm_g, (0, tok_row, i_hv, k4), r_new, 16)
    r_newb[0] = cutlass.Float32(beta[0, tok_row, i_hv])


def gaddr(tensor, coord):
    """Global address of `tensor[coord]`."""
    return (tensor.iterator + cute.crd2idx(coord, tensor.layout)).toint()


def saddr(tensor, offset):
    """Shared-memory address of element `offset` of an smem tensor (row-major offset)."""
    return (tensor.iterator + offset).toint()


def ld_row4(tensor, coord, words, base):
    """Issue the load of four consecutive elements of `tensor` (16- or 32-bit) at `coord` into the
    raw-word buffer `words[base : base + 4]` (two packed words for a 16-bit type)."""
    if cutlass.const_expr(tensor.element_type.width == 16):
        w = ld_global_b32x2(gaddr(tensor, coord))
        words[base] = w[0]
        words[base + 1] = w[1]
    else:
        w = ld_global_b32x4(gaddr(tensor, coord))
        for i in range(4):
            words[base + i] = w[i]


def unpack4(dtype, words, base):
    """Four fp32 values from the raw words of `ld_row4` (bf16 -> fp32 is exact)."""
    if cutlass.const_expr(dtype.width == 16):
        assert dtype == cutlass.BFloat16, "16-bit operands are bf16"
        w0 = words[base]
        w1 = words[base + 1]
        return [
            (w0 << 16).bitcast(cutlass.Float32),
            (w0 & cutlass.Int32(-65536)).bitcast(cutlass.Float32),
            (w1 << 16).bitcast(cutlass.Float32),
            (w1 & cutlass.Int32(-65536)).bitcast(cutlass.Float32),
        ]
    return [words[base + i].bitcast(cutlass.Float32) for i in range(4)]


def st_row4(tensor, coord, v):
    """Store four fp32 values `v` as consecutive elements of `tensor` (rounded to nearest even
    for bf16, like a scalar `.to(BFloat16)`)."""
    if cutlass.const_expr(tensor.element_type.width == 16):
        assert tensor.element_type == cutlass.BFloat16, "16-bit caches are bf16"
        st_global_b32x2(
            gaddr(tensor, coord),
            cvt_bf16x2(v[1], v[0]).bitcast(cutlass.Int32),
            cvt_bf16x2(v[3], v[2]).bitcast(cutlass.Int32),
        )
    else:
        st_global_b32x4(gaddr(tensor, coord), *[x.bitcast(cutlass.Int32) for x in v])


@cute.kernel
def kda_decode_mtp_v2_kernel(
    h0: cute.Tensor,
    x_q: cute.Tensor,
    x_k: cute.Tensor,
    x_v: cute.Tensor,
    w_q: cute.Tensor,
    w_k: cute.Tensor,
    w_v: cute.Tensor,
    cs_q: cute.Tensor,
    cs_k: cute.Tensor,
    cs_v: cute.Tensor,
    A_log: cute.Tensor,
    g: cute.Tensor,
    dt_bias: cute.Tensor,
    beta: cute.Tensor,
    onorm_g: cute.Tensor,
    onorm_weight: cute.Tensor,
    o: cute.Tensor,
    k_cache: cute.Tensor,
    g_cache: cute.Tensor,
    v_cache: cute.Tensor,
    beta_cache: cute.Tensor,
    ssm_state_indices: cute.Tensor,
    num_accepted_tokens: cute.Tensor,
    smem_qk_layout: cute.Layout,
    tiled_mma1: cute.TiledMma,
    tiled_mma3: cute.TiledMma,
    tiled_mma1k: cute.TiledMma,
    smem_layout_b1: cute.ComposedLayout,
    smem_layout_a2: cute.ComposedLayout,
    smem_layout_a3: cute.ComposedLayout,
    smem_layout_b3: cute.ComposedLayout,
    smem_layout_b1k: cute.ComposedLayout,
    tiled_mma4: cute.TiledMma,
    tma_atom_t: cute.CopyAtom,
    mT_tma: cute.Tensor,
    box_layout: cute.ComposedLayout,
    tma_atom_l: cute.CopyAtom,
    mS_tma: cute.Tensor,
    tiled_mma2h: cute.TiledMma,
    smem_layout_b2h: cute.ComposedLayout,
    box_layout_t: cute.ComposedLayout,
    tiled_mma2q: cute.TiledMma,
    smem_layout_b2q: cute.ComposedLayout,
    scale: cutlass.Constexpr[float],
    HV: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int],
    NUM_SPEC: cutlass.Constexpr[int],
    KERNEL_WIDTH: cutlass.Constexpr[int],
    lower_bound: cutlass.Constexpr[float],
    onorm_eps: cutlass.Constexpr[float],
    USE_FLAT_LAYOUT: cutlass.Constexpr[bool],
    SPLIT_V: cutlass.Constexpr[int],
):
    """KDA MTP verify, packed-token / fused-norm contract (see the module docstring).

    Grid ``(HV, N, SPLIT_V)``: one (head, request) tile per CTA, or per cluster pair of CTAs
    with ``SPLIT_V == 2`` (each CTA seeds and stores half of the state rows).
    """
    tidx, _, _ = cute.arch.thread_idx()
    in_warp_tid = tidx % 32
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    i_hv, i_n, vh = cute.arch.block_idx()
    i_h = i_hv
    # The request's state slot is the first load of the kernel; its round trip overlaps the
    # slot-independent loads issued below.
    slot32 = ssm_state_indices[i_n]
    hk_off = i_h * K
    hv_off = i_hv * V
    # Recurrent-state element type (fp32). The TMA boxes have 128 B swizzled rows,
    # i.e. BOX_COLS = 32 fp32 columns; the state tile is NUM_SLABS such boxes.
    STATE_DTYPE = h0.element_type
    # 3xTF32 recurrence (hi/lo splits on every product that feeds the state, S0 included).
    N1 = 3 * T_PAD  # GEMM1 N: [K~hi; Q~; K~lo]
    CONV_DTYPE = cs_q.element_type
    K_CACHE_DTYPE = k_cache.element_type
    V_CACHE_DTYPE = v_cache.element_type
    BETA_CACHE_DTYPE = beta_cache.element_type
    BOX_COLS = 128 // (STATE_DTYPE.width // 8)
    NUM_SLABS = K // BOX_COLS
    t_max = 2 * NUM_SPEC + 1
    NEW_TOK = 1 + NUM_SPEC
    vec_size = TILE_K // 32
    # Load-phase warp roles: warp j < NEW_TOK stages new token j, warp REPLAY_WARP0 + r the
    # pending draft r (all NUM_SPEC entries are loaded, the accepted ones placed), the last warp
    # the committed conv windows and the S0 TMA.
    REPLAY_WARP0 = NUM_THREADS // 64
    TMA_WARP = NUM_THREADS // 32 - 1
    assert NUM_SPEC <= 7 and t_max <= T_PAD, "chunk form pads the token axis to 16"
    assert NEW_TOK <= REPLAY_WARP0 and REPLAY_WARP0 + NUM_SPEC <= TMA_WARP
    assert K == 128 and V == 128 and KERNEL_WIDTH == 4 and NUM_THREADS == 512, (
        "the thread mappings assume K = V = 128, conv width 4 and 512 threads"
    )
    # Rows of this CTA's committed-state store boxes.
    ST_ROWS = V // SPLIT_V
    k_weight_base = KERNEL_WIDTH * K
    v_weight_base = 2 * KERNEL_WIDTH * K
    conv_weight_elems = 2 * KERNEL_WIDTH * K + KERNEL_WIDTH * V
    smem = cutlass.utils.SmemAllocator()
    sQ = smem.allocate_tensor(cutlass.Float32, smem_qk_layout, 16)
    sK = smem.allocate_tensor(cutlass.Float32, smem_qk_layout, 16)
    sG = smem.allocate_tensor(cutlass.Float32, smem_qk_layout, 16)
    sBeta = smem.allocate_tensor(cutlass.Float32, cute.make_layout((t_max,)), 16)
    sVall = smem.allocate_tensor(cutlass.Float32, cute.make_layout((t_max * V,)), 16)
    sConvW = smem.allocate_tensor(cutlass.Float32, cute.make_layout((conv_weight_elems,)), 16)
    sOnG = smem.allocate_tensor(cutlass.Float32, cute.make_layout((NEW_TOK * V,)), 16)
    sOnW = smem.allocate_tensor(cutlass.Float32, cute.make_layout((V,)), 16)
    # Staged per-token inputs (raw conv sequence with the committed window
    # prepended in rows 0..W-2, raw gates, raw betas).
    sXQ = smem.allocate_tensor(cutlass.Float32, cute.make_layout(((t_max + 3) * K,)), 16)
    sXK = smem.allocate_tensor(cutlass.Float32, cute.make_layout(((t_max + 3) * K,)), 16)
    sXV = smem.allocate_tensor(cutlass.Float32, cute.make_layout(((t_max + 3) * V,)), 16)
    sGin = smem.allocate_tensor(cutlass.Float32, cute.make_layout((t_max * K,)), 16)
    sBetaRaw = smem.allocate_tensor(cutlass.Float32, cute.make_layout((t_max,)), 16)
    sDtBias = smem.allocate_tensor(cutlass.Float32, cute.make_layout((K,)), 16)
    sAlog = smem.allocate_tensor(cutlass.Float32, cute.make_layout((32,)), 16)
    tmem_holder = smem.allocate_array(cutlass.Int32, 1)
    mbar1 = smem.allocate_array(cutlass.Int64, 1)
    mbar2 = smem.allocate_array(cutlass.Int64, 1)
    mbar_ld = smem.allocate_array(cutlass.Int64, 1)
    mbar2b = smem.allocate_array(cutlass.Int64, 1)
    sScratch = smem.allocate_tensor(cutlass.Float32, cute.make_layout((SCRATCH_FLOATS,)), 1024)
    sB1 = cute.make_tensor(
        cute.recast_ptr(sScratch.iterator + OFF_B1, smem_layout_b1.inner, dtype=cutlass.Float32),
        smem_layout_b1.outer,
    )
    sA2hi = cute.make_tensor(
        cute.recast_ptr(sScratch.iterator + OFF_A2HI, smem_layout_a2.inner, dtype=cutlass.Float32),
        smem_layout_a2.outer,
    )
    sA2lo = cute.make_tensor(
        cute.recast_ptr(sScratch.iterator + OFF_A2LO, smem_layout_a2.inner, dtype=cutlass.Float32),
        smem_layout_a2.outer,
    )
    sA3 = cute.make_tensor(
        cute.recast_ptr(sScratch.iterator + OFF_A3, smem_layout_a3.inner, dtype=cutlass.Float32),
        smem_layout_a3.outer,
    )
    sB3hi = cute.make_tensor(
        cute.recast_ptr(sScratch.iterator + OFF_B3HI, smem_layout_b3.inner, dtype=cutlass.Float32),
        smem_layout_b3.outer,
    )
    sB3lo = cute.make_tensor(
        cute.recast_ptr(sScratch.iterator + OFF_B3LO, smem_layout_b3.inner, dtype=cutlass.Float32),
        smem_layout_b3.outer,
    )
    sB1k = cute.make_tensor(
        cute.recast_ptr(sScratch.iterator + OFF_B1K, smem_layout_b1k.inner, dtype=cutlass.Float32),
        smem_layout_b1k.outer,
    )
    sB1kf = cute.make_tensor(sScratch.iterator + OFF_B1K, cute.make_layout((T_PAD * 128,)))
    # Flat (unswizzled-pointer) views for element writes with closed-form swizzle offsets.
    sB1f = cute.make_tensor(sScratch.iterator + OFF_B1, cute.make_layout((48 * 128,)))
    sA2hif = cute.make_tensor(sScratch.iterator + OFF_A2HI, cute.make_layout((128 * 16,)))
    sA2lof = cute.make_tensor(sScratch.iterator + OFF_A2LO, cute.make_layout((128 * 16,)))
    sB2hif = cute.make_tensor(sScratch.iterator + OFF_B2HI, cute.make_layout((128 * 16,)))
    sB2lof = cute.make_tensor(sScratch.iterator + OFF_B2LO, cute.make_layout((128 * 16,)))
    sB4hif = cute.make_tensor(sScratch.iterator + OFF_B4HI, cute.make_layout((T_PAD * T_PAD,)))
    sB4lof = cute.make_tensor(sScratch.iterator + OFF_B4LO, cute.make_layout((T_PAD * T_PAD,)))
    sA3f = cute.make_tensor(sScratch.iterator + OFF_A3, cute.make_layout((64 * 128,)))
    sB3hif = cute.make_tensor(sScratch.iterator + OFF_B3HI, cute.make_layout((16 * 128,)))
    sB3lof = cute.make_tensor(sScratch.iterator + OFF_B3LO, cute.make_layout((16 * 128,)))
    # S0 / committed-state TMA boxes at the start of the scratch region
    sBoxf = cute.make_tensor(sScratch.iterator, cute.make_layout((V * K,)))
    # GEMM3 result [L; B hi; B lo; -] as a plain (64 x 16) row-major matrix.
    sLB = smem.allocate_tensor(
        cutlass.Float32, cute.make_layout((4 * T_PAD, T_PAD), stride=(T_PAD, 1)), 16
    )
    sEc = smem.allocate_tensor(cutlass.Float32, cute.make_layout((K,)), 16)
    # V'^T (v, s) and Kbar^T (k, s) staged with a padded row stride (conflict-free for both
    # the row-per-thread producers and the 16-lanes-per-row tile writers).
    sVpT = smem.allocate_tensor(cutlass.Float32, cute.make_layout((V * TP_STRIDE,)), 16)
    sKbT = smem.allocate_tensor(cutlass.Float32, cute.make_layout((K * TP_STRIDE,)), 16)
    # Cumulative gates G_t (per k) reuse the raw-gate staging buffer (no longer needed after the
    # precompute).
    sE = cute.make_tensor(sGin.iterator, cute.make_layout((t_max * K,)))

    # ---- load phase. Every load is issued before any of its consumers, so the CTA waits for its
    #      round trips in parallel: the slot-independent operands (new-token projections, conv
    #      weights, per-head constants, output-norm operands) under the slot load, the
    #      slot-dependent ones (S0 TMA, committed conv windows, every replay entry) as soon as the
    #      slot arrives; only the smem placement of the new tokens and accepted drafts waits for
    #      the accepted count. Each lane moves four consecutive channels (8 B for bf16, 16 B for
    #      fp32) per row, held as raw words until the row's smem position is known.
    ch = tidx % 128
    quad = tidx // 128
    k4 = 4 * in_warp_tid
    # (1) slot-independent: conv weights ([ch][w] rows of 16 B: threads [0, 384) one row each),
    #     dt_bias / output-norm weight / A_log (threads [384, 512)), new token j = warp
    #     (warps < NEW_TOK)
    r_w = cute.make_rmem_tensor(cute.make_layout((4,)), cutlass.Int32)
    if quad == 0:
        ld_row4(w_q, (hk_off + ch, 0), r_w, 0)
    if quad == 1:
        ld_row4(w_k, (hk_off + ch, 0), r_w, 0)
    if quad == 2:
        ld_row4(w_v, (hv_off + ch, 0), r_w, 0)
    r_c = cute.make_rmem_tensor(cute.make_layout((3,)), cutlass.Float32)
    if quad == 3:
        r_c[0] = cutlass.Float32(dt_bias[hk_off + ch])
        r_c[1] = cutlass.Float32(onorm_weight[ch])
        r_c[2] = cutlass.Float32(A_log[i_h])
    r_new = cute.make_rmem_tensor(cute.make_layout((5 * 4,)), cutlass.Int32)
    r_newb = cute.make_rmem_tensor(cute.make_layout((1,)), cutlass.Float32)
    tok_row = i_n * NEW_TOK + warp_idx
    new_rows = (x_q, x_k, x_v, g, onorm_g, beta)
    if warp_idx < NEW_TOK:
        _load_new_token_row(
            *new_rows,
            tok_row,
            i_h,
            i_hv,
            hk_off,
            hv_off,
            k4,
            r_new,
            r_newb,
            USE_FLAT_LAYOUT,
        )
    if warp_idx == 0:
        with cute.arch.elect_one():
            cute.arch.mbarrier_init(mbar1, 1)
            cute.arch.mbarrier_init(mbar2, 1)
            cute.arch.mbarrier_init(mbar2b, 1)
            cute.arch.mbarrier_init(mbar_ld, 1)
    cute.arch.mbarrier_init_fence()
    cute.arch.barrier()
    # V2 state views may place many layer pages between adjacent slots. Widen
    # the slot before indexing: slot * state stride can exceed INT32_MAX.
    slot = Int64(slot32)
    h0_idx = slot * HV + i_hv
    # (2) slot-dependent: replay entry r = warp - REPLAY_WARP0 (every pending entry, accepted or
    #     not), the committed conv windows (TMA warp: three columns of q, k and v)
    rep = warp_idx - REPLAY_WARP0
    r_rep = cute.make_rmem_tensor(cute.make_layout((6 * 4,)), cutlass.Int32)
    r_repb = cute.make_rmem_tensor(cute.make_layout((1,)), cutlass.Float32)
    if (warp_idx >= REPLAY_WARP0) & (rep < NUM_SPEC):
        ld_row4(k_cache, (slot, rep, hk_off + k4), r_rep, 0)
        ld_row4(g_cache, (slot, rep, hk_off + k4), r_rep, 4)
        ld_row4(v_cache, (slot, rep, hv_off + k4), r_rep, 8)
        ld_row4(cs_q, (slot, hk_off + k4, KERNEL_WIDTH - 1 + rep), r_rep, 12)
        ld_row4(cs_k, (slot, hk_off + k4, KERNEL_WIDTH - 1 + rep), r_rep, 16)
        ld_row4(cs_v, (slot, hv_off + k4, KERNEL_WIDTH - 1 + rep), r_rep, 20)
        r_repb[0] = cutlass.Float32(beta_cache[slot, rep, i_hv])
    r_win = cute.make_rmem_tensor(cute.make_layout((3 * (KERNEL_WIDTH - 1) * 4,)), cutlass.Int32)
    if warp_idx == TMA_WARP:
        for w in cutlass.range_constexpr(KERNEL_WIDTH - 1):
            ld_row4(cs_q, (slot, hk_off + k4, w), r_win, 4 * w)
            ld_row4(cs_k, (slot, hk_off + k4, w), r_win, 4 * (KERNEL_WIDTH - 1 + w))
            ld_row4(cs_v, (slot, hv_off + k4, w), r_win, 4 * (2 * (KERNEL_WIDTH - 1) + w))
    commit_len = num_accepted_tokens[slot]
    # ---- TMA bulk load of S0 into the boxes; completes under the token staging /
    #      precompute and is consumed (mbar_ld) at the start of the chunk stage.
    # Issued by the whole of one warp (a TMA load issued from a single thread never
    # completes on this DSL build; the copy elects its issuing lane internally).
    if warp_idx == TMA_WARP:
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(mbar_ld, V * K * (STATE_DTYPE.width // 8))
    for sl_h in cutlass.range_constexpr(NUM_SLABS):
        sTl_h = cute.make_tensor(
            cute.recast_ptr(
                sScratch.iterator + sl_h * (V * 32), box_layout.inner, dtype=STATE_DTYPE
            ),
            box_layout.outer,
        )
        gS_h_all = cute.local_tile(mS_tma, (V, BOX_COLS, 1), (0, sl_h, None))
        if cutlass.const_expr(USE_FLAT_LAYOUT):
            gS_h = gS_h_all[(None, None, 0, h0_idx)]
        else:
            gS_h = gS_h_all[(None, None, 0, (i_hv, slot))]
        tSl_h, tGl_h = cpasync.tma_partition(
            tma_atom_l,
            0,
            cute.make_layout(1),
            cute.group_modes(sTl_h, 0, 2),
            cute.group_modes(gS_h, 0, 2),
        )
        if warp_idx == TMA_WARP:
            cute.copy(tma_atom_l, tGl_h, tSl_h, tma_bar_ptr=mbar_ld)
    # TMEM allocation overlaps the in-flight loads; the pointer is retrieved after the
    # precompute barrier (first TMEM use is the S0 staging in the chunk stage).
    if warp_idx == 0:
        cute.arch.alloc_tmem(TMEM_COLS, tmem_holder)
    # (3) placement: the slot-independent operands first, the token rows once the accepted count
    #     is in. Smem rows keep the [row][channel] fp32 layout of the precompute.
    if quad < 3:
        wv = unpack4(w_q.element_type, r_w, 0)
        for w in cutlass.range_constexpr(KERNEL_WIDTH):
            sConvW[quad * (KERNEL_WIDTH * K) + w * K + ch] = wv[w]
    if quad == 3:
        sDtBias[ch] = r_c[0]
        sOnW[ch] = r_c[1]
        if ch == 0:
            sAlog[0] = r_c[2]
    if warp_idx < NEW_TOK:
        st_shared_f32x4(saddr(sOnG, warp_idx * V + k4), *unpack4(onorm_g.element_type, r_new, 16))
    if warp_idx == TMA_WARP:
        for w in cutlass.range_constexpr(KERNEL_WIDTH - 1):
            st_shared_f32x4(saddr(sXQ, w * K + k4), *unpack4(CONV_DTYPE, r_win, 4 * w))
            st_shared_f32x4(
                saddr(sXK, w * K + k4), *unpack4(CONV_DTYPE, r_win, 4 * (KERNEL_WIDTH - 1 + w))
            )
            st_shared_f32x4(
                saddr(sXV, w * V + k4),
                *unpack4(CONV_DTYPE, r_win, 4 * (2 * (KERNEL_WIDTH - 1) + w)),
            )
    # Only NUM_SPEC drafts can be pending from the previous round. Clamp
    # so a malformed count cannot drive T_loop past the t_max-sized SMEM
    # buffers or the num_spec extents of the replay caches.
    if commit_len > NUM_SPEC:
        commit_len = cutlass.Int32(NUM_SPEC)
    T_loop = commit_len + NEW_TOK
    if warp_idx < NEW_TOK:
        t_new = commit_len + warp_idx
        st_shared_f32x4(saddr(sXQ, (t_new + 3) * K + k4), *unpack4(x_q.element_type, r_new, 0))
        st_shared_f32x4(saddr(sXK, (t_new + 3) * K + k4), *unpack4(x_k.element_type, r_new, 4))
        st_shared_f32x4(saddr(sXV, (t_new + 3) * V + k4), *unpack4(x_v.element_type, r_new, 8))
        st_shared_f32x4(saddr(sGin, t_new * K + k4), *unpack4(g.element_type, r_new, 12))
        if in_warp_tid == 0:
            sBetaRaw[t_new] = r_newb[0]
    if (warp_idx >= REPLAY_WARP0) & (rep < commit_len):
        # replayed drafts produce no output: their q rows are unused
        z = cutlass.Float32(0.0)
        st_shared_f32x4(saddr(sQ, rep * K + k4), z, z, z, z)
        st_shared_f32x4(saddr(sK, rep * K + k4), *unpack4(K_CACHE_DTYPE, r_rep, 0))
        st_shared_f32x4(saddr(sG, rep * K + k4), *unpack4(g_cache.element_type, r_rep, 4))
        st_shared_f32x4(saddr(sVall, rep * V + k4), *unpack4(V_CACHE_DTYPE, r_rep, 8))
        st_shared_f32x4(saddr(sXQ, (rep + 3) * K + k4), *unpack4(CONV_DTYPE, r_rep, 12))
        st_shared_f32x4(saddr(sXK, (rep + 3) * K + k4), *unpack4(CONV_DTYPE, r_rep, 16))
        st_shared_f32x4(saddr(sXV, (rep + 3) * V + k4), *unpack4(CONV_DTYPE, r_rep, 20))
        if in_warp_tid == 0:
            sBeta[rep] = r_repb[0]
    # Padding rows [T_loop, t_max) of the gates, values and betas are zeroed so the chunk
    # stage's r_Y / cumulative-gate / V' loops are branch-free.
    if tidx < K:
        for _z in cutlass.range(T_loop, t_max, 1):
            sG[_z, tidx] = cutlass.Float32(0.0)
            sVall[_z * V + tidx] = cutlass.Float32(0.0)
            if tidx == 0:
                sBeta[_z] = cutlass.Float32(0.0)
    wg = warp_idx // 4
    wg_tidx = tidx % 128
    r_q = cute.make_rmem_tensor(cute.make_layout((vec_size,)), cutlass.Float32)
    r_k = cute.make_rmem_tensor(cute.make_layout((vec_size,)), cutlass.Float32)
    r_state = cute.make_rmem_tensor(cute.make_layout((32,)), cutlass.Float32)
    cute.arch.barrier()

    # ---- per-token-parallel precompute (smem only; the new tokens' cache and conv-window writes
    #      are issued from smem while the tensor cores run GEMM1/GEMM3): warp j < NEW_TOK computes
    #      the q and k conv + SiLU + L2 norm of new token j (the two chains interleaved), warp 8 + j
    #      its gate, beta and v conv.
    half = NUM_THREADS // 64
    role = warp_idx // half
    j = warp_idx % half
    if j < NEW_TOK:
        t = commit_len + j
        if role == 0:
            for i in cutlass.range(vec_size, unroll_full=True):
                k_idx = i * 32 + in_warp_tid
                r_conv_0 = cutlass.Float32(0.0)
                for w in cutlass.range(KERNEL_WIDTH, unroll_full=True):
                    r_conv_0 += sXQ[(t + w) * K + k_idx] * sConvW[w * K + k_idx]
                r_conv = sXK[t * K + k_idx] * sConvW[k_weight_base + 0 * K + k_idx]
                for w in cutlass.range(1, KERNEL_WIDTH, unroll_full=True):
                    r_conv += sXK[(t + w) * K + k_idx] * sConvW[k_weight_base + w * K + k_idx]
                e0 = cute.math.exp(-r_conv_0, fastmath=True)
                r_q[i] = r_conv_0 * cute.arch.rcp_approx(cutlass.Float32(1.0) + e0)
                r_conv = r_conv * cute.arch.rcp_approx(
                    cutlass.Float32(1.0) + cute.math.exp(-r_conv, fastmath=True)
                )
                r_k[i] = r_conv
            sum_q = 0.0
            sum_k = 0.0
            for i in cutlass.range(vec_size, unroll_full=True):
                sum_q += r_q[i] * r_q[i]
                sum_k += r_k[i] * r_k[i]
            for offset in [16, 8, 4, 2, 1]:
                sum_q += cute.arch.shuffle_sync_bfly(
                    sum_q, offset=offset, mask=-1, mask_and_clamp=31
                )
                sum_k += cute.arch.shuffle_sync_bfly(
                    sum_k, offset=offset, mask=-1, mask_and_clamp=31
                )
            rnorm_q_scaled = cute.math.rsqrt(sum_q + 1e-06, fastmath=True) * scale
            rnorm_k = cute.math.rsqrt(sum_k + 1e-06, fastmath=True)
            for i in cutlass.range(vec_size, unroll_full=True):
                k_idx = i * 32 + in_warp_tid
                sQ[t, k_idx] = r_q[i] * rnorm_q_scaled
                sK[t, k_idx] = r_k[i] * rnorm_k
        else:
            r_exp_A = cute.math.exp(sAlog[0], fastmath=True)
            for i in cutlass.range(vec_size, unroll_full=True):
                k_idx = i * 32 + in_warp_tid
                r_g_raw = sGin[t * K + k_idx]
                r_g_raw = r_g_raw + sDtBias[k_idx]
                exp_A_x = r_exp_A * r_g_raw
                sigmoid_val = cute.arch.rcp_approx(
                    cutlass.Float32(1.0) + cute.math.exp(-exp_A_x, fastmath=True)
                )
                sG[t, k_idx] = lower_bound * sigmoid_val
            r_beta_t = cute.arch.rcp_approx(
                cutlass.Float32(1.0) + cute.math.exp(-sBetaRaw[t], fastmath=True)
            )
            if in_warp_tid == 0:
                sBeta[t] = r_beta_t
            for i in cutlass.range(vec_size, unroll_full=True):
                v_idx = i * 32 + in_warp_tid
                _v_conv = 0.0
                for w in cutlass.range(KERNEL_WIDTH, unroll_full=True):
                    _v_conv += sXV[(t + w) * V + v_idx] * sConvW[v_weight_base + w * V + v_idx]
                _v_conv = _v_conv * cute.arch.rcp_approx(
                    cutlass.Float32(1.0) + cute.math.exp(-_v_conv, fastmath=True)
                )
                sVall[t * V + v_idx] = _v_conv
    cute.arch.barrier()

    # =================================================================
    # Chunk (WY-form) recurrence on tensor cores.
    # =================================================================
    tmem_ptr = cute.arch.retrieve_tmem_ptr(cutlass.Float32, 16, tmem_holder)
    pool = cutlass.utils.TmemBufferPool(tmem_ptr, TMEM_COLS)
    tD4_frag = tiled_mma4.make_fragment_C(tiled_mma4.partition_shape_C((V, T_PAD)))
    tD4 = pool.allocate_tensor(tD4_frag.layout, cutlass.Float32)  # outputs (columns 0..15)
    tD2h_frag = tiled_mma2h.make_fragment_C(tiled_mma2h.partition_shape_C((V, K // 2)))
    tD2h0 = pool.allocate_tensor(tD2h_frag.layout, cutlass.Float32)  # GEMM2 half 0: k cols 0..63
    tD2h1 = pool.allocate_tensor(tD2h_frag.layout, cutlass.Float32)  # GEMM2 half 1: k cols 64..127
    tD2q_frag = tiled_mma2q.make_fragment_C(tiled_mma2q.partition_shape_C((V, T_PAD + K // 2)))
    tD2q = cute.make_tensor(tD4.iterator, tD2q_frag.layout)  # [D4 | dS half 0] accumulator view
    tA_frag = tiled_mma1.make_fragment_A(tiled_mma1.partition_shape_A((V, K)))
    tAhi = pool.allocate_tensor(tA_frag.layout, cutlass.Float32)
    tAlo = pool.allocate_tensor(tA_frag.layout, cutlass.Float32)
    tD1_frag = tiled_mma1.make_fragment_C(tiled_mma1.partition_shape_C((V, N1)))
    tD1 = pool.allocate_tensor(tD1_frag.layout, cutlass.Float32)
    tD3_frag = tiled_mma3.make_fragment_C(tiled_mma3.partition_shape_C((4 * T_PAD, T_PAD)))
    tD3 = pool.allocate_tensor(tD3_frag.layout, cutlass.Float32)
    tD2 = cute.make_tensor(tD2h0.iterator, tA_frag.layout)  # A-view layout (all 128 columns)

    st32 = cute.make_copy_atom(tcgen05.St32x32bOp(tcgen05.Repetition.x32), cutlass.Float32)
    ld16 = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition.x16), cutlass.Float32)
    ld32 = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition.x32), cutlass.Float32)
    thr_stA = tcgen05.make_tmem_copy(st32, tD2).get_slice(wg_tidx)
    tD2_st = thr_stA.partition_D(tD2)
    rA_shape = thr_stA.partition_S(tD2).shape
    tAhi_st = thr_stA.partition_D(tAhi)
    tAlo_st = thr_stA.partition_D(tAlo)
    thr_ldD2 = tcgen05.make_tmem_copy(ld32, tD2).get_slice(wg_tidx)
    tD2_ld = thr_ldD2.partition_S(tD2)
    rS = cute.make_rmem_tensor(thr_ldD2.partition_D(tD2).shape[:-1], cutlass.Float32)
    rS_flat = cute.make_tensor(rS.iterator, cute.make_layout((32,)))
    # TMA store partitions for the NUM_SLABS boxes of this CTA's committed-state tile.
    tS_list = []
    tG_list = []
    for sl_h in cutlass.range_constexpr(NUM_SLABS):
        sT_h = cute.make_tensor(
            cute.recast_ptr(
                sScratch.iterator + sl_h * (ST_ROWS * 32),
                box_layout_t.inner,
                dtype=STATE_DTYPE,
            ),
            box_layout_t.outer,
        )
        gT_h_all = cute.local_tile(mT_tma, (ST_ROWS, BOX_COLS, 1), (vh, sl_h, None))
        if cutlass.const_expr(USE_FLAT_LAYOUT):
            gT_h = gT_h_all[(None, None, 0, h0_idx)]
        else:
            gT_h = gT_h_all[(None, None, 0, (i_hv, slot))]
        tS_h, tG_h = cpasync.tma_partition(
            tma_atom_t,
            0,
            cute.make_layout(1),
            cute.group_modes(sT_h, 0, 2),
            cute.group_modes(gT_h, 0, 2),
        )
        tS_list.append(tS_h)
        tG_list.append(tG_h)
    r_hi = cute.make_rmem_tensor(rA_shape[:-1], cutlass.Float32)
    r_lo = cute.make_rmem_tensor(rA_shape[:-1], cutlass.Float32)
    r_hi_flat = cute.make_tensor(r_hi.iterator, cute.make_layout((32,)))
    r_lo_flat = cute.make_tensor(r_lo.iterator, cute.make_layout((32,)))

    # ---- cumulative gates per k column (thread kq, quarter q owns tokens t = q mod 4):
    #      G_t -> sE, the T x T coupling factors, and the operand rows written with closed-form
    #      128B-swizzle offsets (lanes = consecutive k: conflict-free). The GEMM3 rows (A3 =
    #      [Kf; Qf hi; Qf lo], B3 = Kb hi / lo) lie clear of the S0 boxes and are written first,
    #      so GEMM3 runs while S0 moves into TMEM; the GEMM1 rows (K~ hi / lo, Q~) stay in
    #      registers until the S0 boxes have been read.
    kq = tidx % K
    quarter = tidx // K
    t_mid = T_loop // 2
    r_G = cute.make_rmem_tensor(cute.make_layout((T_PAD,)), cutlass.Float32)
    g_run = cutlass.Float32(0.0)
    for t in cutlass.range(T_PAD, unroll_full=True):
        if t < t_max:
            g_run = g_run + sG[t, kq]
        r_G[t] = g_run
    g_mid = cutlass.Float32(0.0)
    for t in cutlass.range(T_PAD, unroll_full=True):
        if t == t_mid:
            g_mid = r_G[t]
    # Factors of the T x T coupling: e^{G_t - G_mid} (t side) and e^{G_mid - G_s} (s side),
    # clamped at e^87 so a chunk whose decay exceeds the fp32 range yields ~0 instead of
    # inf/NaN (the affected products are < e^{-|g|} and negligible either way).
    exp_clamp = cutlass.Float32(87.0)
    ka = kq // 32
    kc = kq % 32
    b1_base = ka * (3 * T_PAD * 32)
    a3_base = ka * (4 * T_PAD * 32)
    b3_base = ka * (T_PAD * 32)
    b1k_base = ka * (T_PAD * 32)
    # GEMM1 rows of this thread's tokens t = quarter + 4 i: (K~ hi, Q~, K~ lo) per token
    r_b1 = cute.make_rmem_tensor(cute.make_layout((3 * (T_PAD // 4),)), cutlass.Float32)
    for t in cutlass.range(T_PAD, unroll_full=True):
        if quarter == t % 4:
            kt_hi = cutlass.Float32(0.0)
            kt_lo = cutlass.Float32(0.0)
            qt_r = cutlass.Float32(0.0)
            kf = cutlass.Float32(0.0)
            qf_hi = cutlass.Float32(0.0)
            qf_lo = cutlass.Float32(0.0)
            kb_hi = cutlass.Float32(0.0)
            kb_lo = cutlass.Float32(0.0)
            if t < T_loop:
                d_t = r_G[t] - g_mid
                e_t = cute.math.exp(-cute.arch.fmax(-d_t, -exp_clamp), fastmath=True)
                e_bt = cute.math.exp(-cute.arch.fmax(d_t, -exp_clamp), fastmath=True)
                eg_t = cute.math.exp(r_G[t], fastmath=True)
                sE[t * K + kq] = r_G[t]
                if t == commit_len:
                    sEc[kq] = eg_t
                k_t = sK[t, kq]
                q_t = sQ[t, kq]
                kt = k_t * eg_t
                kt_hi = round_tf32(kt)
                qt_r = round_tf32(q_t * eg_t)
                kf = round_tf32(k_t * e_t)
                qf = q_t * e_t
                qf_hi = round_tf32(qf)
                kb = k_t * e_bt
                kb_hi = round_tf32(kb)
                kt_lo = kt - kt_hi
                qf_lo = qf - qf_hi
                kb_lo = kb - kb_hi
            # rows t, 16+t, 32+t share t % 8 -> the same 16-byte-chunk XOR
            xc = kc ^ (4 * (t % 8))
            sA3f[a3_base + 32 * t + xc] = kf
            sA3f[a3_base + 32 * (T_PAD + t) + xc] = qf_hi
            sA3f[a3_base + 32 * (2 * T_PAD + t) + xc] = qf_lo
            sB3hif[b3_base + 32 * t + xc] = kb_hi
            sB3lof[b3_base + 32 * t + xc] = kb_lo
            r_b1[3 * (t // 4)] = kt_hi
            r_b1[3 * (t // 4) + 1] = qt_r
            r_b1[3 * (t // 4) + 2] = kt_lo
    cute.arch.fence_view_async_shared()
    tcgen05_fence_before_thread_sync()
    cute.arch.barrier()
    tcgen05_fence_after_thread_sync()
    # ---- GEMM3: D3 = A3 Kb_hi^T + A3 Kb_lo^T (warp 0; completion is tracked with GEMM1's)
    tCrA3 = tiled_mma3.make_fragment_A(
        cute.make_tensor(
            cute.recast_ptr(sA3.iterator, smem_layout_a3.inner, dtype=cutlass.TFloat32),
            sA3.layout,
        )
    )
    tCrB3hi = tiled_mma3.make_fragment_B(
        cute.make_tensor(
            cute.recast_ptr(sB3hi.iterator, smem_layout_b3.inner, dtype=cutlass.TFloat32),
            sB3hi.layout,
        )
    )
    tCrB3lo = tiled_mma3.make_fragment_B(
        cute.make_tensor(
            cute.recast_ptr(sB3lo.iterator, smem_layout_b3.inner, dtype=cutlass.TFloat32),
            sB3lo.layout,
        )
    )
    if warp_idx == 0:
        tiled_mma3.set(tcgen05.Field.ACCUMULATE, False)
        for kb_i in cutlass.range_constexpr(K // 8):
            cute.gemm(
                tiled_mma3, tD3, tCrA3[(None, None, kb_i, 0)], tCrB3hi[(None, None, kb_i, 0)], tD3
            )
            tiled_mma3.set(tcgen05.Field.ACCUMULATE, True)
        for kb_i in cutlass.range_constexpr(K // 8):
            cute.gemm(
                tiled_mma3, tD3, tCrA3[(None, None, kb_i, 0)], tCrB3lo[(None, None, kb_i, 0)], tD3
            )

    # ---- S0 row v = wg_tidx, columns [32 wg, +32), from the TMA-loaded swizzled box
    #      (16 B chunks: conflict-free LDS.128 under the 128B swizzle) into TMEM (hi / lo)
    cute.arch.mbarrier_wait(mbar_ld, 0)
    # In-place update of a cluster pair: the peer CTA must not store its state half before our
    # S0 load completed, and CTA 0 must not overwrite the replay entries and conv windows before
    # the peer's load phase read them (those values have been consumed by now) -> arrive here;
    # CTA 0 waits before its replay-cache writes, CTA 1 before its state store.
    if cutlass.const_expr(SPLIT_V == 2):
        cute.arch.cluster_arrive_relaxed()
    vx = 4 * (wg_tidx % 8)
    # Box row of S0 row v = wg_tidx.
    s0_row = wg_tidx
    for c in cutlass.range(8, unroll_full=True):
        off_l = wg * (V * 32) + 32 * s0_row + ((4 * c) ^ vx)
        for i in cutlass.range(4, unroll_full=True):
            r_state[4 * c + i] = sBoxf[off_l + i]
    for i in cutlass.range(32, unroll_full=True):
        s_hi = round_tf32(r_state[i])
        r_hi_flat[i] = s_hi
        r_lo_flat[i] = r_state[i] - s_hi
    for jj in cutlass.range_constexpr(4):
        if wg == jj:
            cute.copy(st32, r_hi, tAhi_st[(None, None, None, jj)])
            cute.copy(st32, r_lo, tAlo_st[(None, None, None, jj)])
    cute.arch.fence_view_async_tmem_store()
    # the S0 boxes have been read: the GEMM1 operand rows go over them
    cute.arch.barrier()
    for t in cutlass.range(T_PAD, unroll_full=True):
        if quarter == t % 4:
            xc = kc ^ (4 * (t % 8))
            sB1f[b1_base + 32 * t + xc] = r_b1[3 * (t // 4)]
            sB1kf[b1k_base + 32 * t + xc] = r_b1[3 * (t // 4)]
            sB1f[b1_base + 32 * (T_PAD + t) + xc] = r_b1[3 * (t // 4) + 1]
            sB1f[b1_base + 32 * (2 * T_PAD + t) + xc] = r_b1[3 * (t // 4) + 2]
    cute.arch.fence_view_async_shared()
    tcgen05_fence_before_thread_sync()
    cute.arch.barrier()
    tcgen05_fence_after_thread_sync()

    # ---- operand fragments of GEMM1 (D1 = S0_hi B1^T + S0_lo K~hi^T)
    tCrB1 = tiled_mma1.make_fragment_B(
        cute.make_tensor(
            cute.recast_ptr(sB1.iterator, smem_layout_b1.inner, dtype=cutlass.TFloat32),
            sB1.layout,
        )
    )
    tCrB1k = tiled_mma1k.make_fragment_B(
        cute.make_tensor(
            cute.recast_ptr(sB1k.iterator, smem_layout_b1k.inner, dtype=cutlass.TFloat32),
            sB1k.layout,
        )
    )
    tD1k = cute.make_tensor(
        tD1.iterator,
        tiled_mma1k.make_fragment_C(tiled_mma1k.partition_shape_C((V, T_PAD))).layout,
    )
    tAhi_tf = cute.make_tensor(cute.recast_ptr(tAhi.iterator, dtype=cutlass.TFloat32), tAhi.layout)
    tAlo_tf = cute.make_tensor(cute.recast_ptr(tAlo.iterator, dtype=cutlass.TFloat32), tAlo.layout)
    # ---- warp 0 issues GEMM1 right away; warps 1..15 stage Kbar^T (sKbT[k, s], lanes =
    #      consecutive k: conflict-free), seed the GEMM2 accumulator, build the B2 tiles and write
    #      the new tokens' cache rows while the tensor cores run.
    if warp_idx > 0:
        for i in cutlass.range(
            (K * T_PAD + NUM_THREADS - 33) // (NUM_THREADS - 32), unroll_full=True
        ):
            e_kb = i * (NUM_THREADS - 32) + (tidx - 32)
            if e_kb < K * T_PAD:
                k_kb = e_kb % K
                s_idx = e_kb // K
                b_val = cutlass.Float32(0.0)
                if s_idx <= commit_len:
                    b_val = sK[s_idx, k_kb] * cute.math.exp(
                        sE[commit_len * K + k_kb] - sE[s_idx * K + k_kb], fastmath=True
                    )
                sKbT[k_kb * TP_STRIDE + s_idx] = b_val
    if warp_idx == 0:
        tiled_mma1.set(tcgen05.Field.ACCUMULATE, False)
        for kb_i in cutlass.range_constexpr(K // 8):
            cute.gemm(
                tiled_mma1, tD1, tAhi_tf[(None, None, kb_i)], tCrB1[(None, None, kb_i, 0)], tD1
            )
            tiled_mma1.set(tcgen05.Field.ACCUMULATE, True)
        tiled_mma1k.set(tcgen05.Field.ACCUMULATE, True)
        for kb_i in cutlass.range_constexpr(K // 8):
            cute.gemm(
                tiled_mma1k,
                tD1k,
                tAlo_tf[(None, None, kb_i)],
                tCrB1k[(None, None, kb_i, 0)],
                tD1k,
            )
        # tracks GEMM3 and GEMM1
        with cute.arch.elect_one():
            tcgen05.commit(mbar1)
    else:
        # sKbT complete before the B2 tiles read it
        cute.arch.barrier(barrier_id=5, number_of_threads=NUM_THREADS - 32)
    # ---- GEMM2 accumulator seed (overlaps the GEMM1/GEMM3 execution)
    # D4 (output-combination) columns start from zero: GEMM2 half 0 accumulates into [D4 | dS]
    st16 = cute.make_copy_atom(tcgen05.St32x32bOp(tcgen05.Repetition.x16), cutlass.Float32)
    thr_stD4 = tcgen05.make_tmem_copy(st16, tD4).get_slice(wg_tidx)
    tD4_st = thr_stD4.partition_D(tD4)
    r_zero = cute.make_rmem_tensor(thr_stD4.partition_S(tD4).shape, cutlass.Float32)
    for i in cutlass.range(cute.size(r_zero), unroll_full=True):
        r_zero[i] = cutlass.Float32(0.0)
    if wg == 0:
        cute.copy(st16, r_zero, tD4_st)
    # only this CTA's state rows (TMEM lanes [vh * V/2, +V/2)) are seeded and stored
    if cutlass.const_expr(SPLIT_V == 1) or ((warp_idx % 4) // 2 == vh):
        for i in cutlass.range(32, unroll_full=True):
            r_hi_flat[i] = r_state[i] * sEc[wg * 32 + i]
        for jj in cutlass.range_constexpr(4):
            if wg == jj:
                cute.copy(st32, r_hi, tD2_st[(None, None, None, jj)])
    cute.arch.fence_view_async_tmem_store()
    # ---- B2 tiles (k rows x s cols, K-major SW64) from sKbT while the MMAs run:
    #      lanes 0..15 -> row 2w, lanes 16..31 -> row 2w+1, four 32-row rounds
    s_col = in_warp_tid % T_PAD
    if warp_idx > 0:
        for r in cutlass.range(4, unroll_full=True):
            k_row = 32 * r + 2 * warp_idx + in_warp_tid // T_PAD
            b_val2 = sKbT[k_row * TP_STRIDE + s_col]
            b_hi = round_tf32(b_val2)
            o_b2 = swz64_k16(k_row, s_col)
            sB2hif[o_b2] = b_hi
            sB2lof[o_b2] = b_val2 - b_hi
        # warp 0 is issuing the MMAs: warps 1..4 cover its rows 32 r + {0, 1}
        if warp_idx <= 4:
            k_row0 = 32 * (warp_idx - 1) + in_warp_tid // T_PAD
            b_val0 = sKbT[k_row0 * TP_STRIDE + s_col]
            b_hi0 = round_tf32(b_val0)
            o_b20 = swz64_k16(k_row0, s_col)
            sB2hif[o_b20] = b_hi0
            sB2lof[o_b20] = b_val0 - b_hi0
    # ---- the new tokens' replay-cache rows and conv-window commits, from smem: warp 1 + p
    #      writes draft p's k / gate / v cache rows, its raw conv columns and beta, warps
    #      REPLAY_WARP0 + {0, 1, 2} commit the q / k / v conv windows (the raw inputs after the
    #      golden token). One CTA of a pair writes them.
    if vh == 0:
        if cutlass.const_expr(SPLIT_V == 2):
            cute.arch.cluster_wait()
        if (warp_idx >= 1) & (warp_idx <= NUM_SPEC):
            p = warp_idx - 1
            t_c = commit_len + 1 + p
            st_row4(k_cache, (slot, p, hk_off + k4), ld_shared_f32x4(saddr(sK, t_c * K + k4)))
            st_row4(g_cache, (slot, p, hk_off + k4), ld_shared_f32x4(saddr(sG, t_c * K + k4)))
            st_row4(v_cache, (slot, p, hv_off + k4), ld_shared_f32x4(saddr(sVall, t_c * V + k4)))
            st_row4(
                cs_q,
                (slot, hk_off + k4, KERNEL_WIDTH - 1 + p),
                ld_shared_f32x4(saddr(sXQ, (t_c + 3) * K + k4)),
            )
            st_row4(
                cs_k,
                (slot, hk_off + k4, KERNEL_WIDTH - 1 + p),
                ld_shared_f32x4(saddr(sXK, (t_c + 3) * K + k4)),
            )
            st_row4(
                cs_v,
                (slot, hv_off + k4, KERNEL_WIDTH - 1 + p),
                ld_shared_f32x4(saddr(sXV, (t_c + 3) * V + k4)),
            )
            if in_warp_tid == 0:
                beta_cache[slot, p, i_hv] = sBeta[t_c].to(BETA_CACHE_DTYPE)
        if warp_idx == REPLAY_WARP0:
            for w in cutlass.range_constexpr(KERNEL_WIDTH - 1):
                st_row4(
                    cs_q,
                    (slot, hk_off + k4, w),
                    ld_shared_f32x4(saddr(sXQ, (commit_len + 1 + w) * K + k4)),
                )
        if warp_idx == REPLAY_WARP0 + 1:
            for w in cutlass.range_constexpr(KERNEL_WIDTH - 1):
                st_row4(
                    cs_k,
                    (slot, hk_off + k4, w),
                    ld_shared_f32x4(saddr(sXK, (commit_len + 1 + w) * K + k4)),
                )
        if warp_idx == REPLAY_WARP0 + 2:
            for w in cutlass.range_constexpr(KERNEL_WIDTH - 1):
                st_row4(
                    cs_v,
                    (slot, hv_off + k4, w),
                    ld_shared_f32x4(saddr(sXV, (commit_len + 1 + w) * V + k4)),
                )
    # ---- wait for GEMM1/GEMM3
    cute.arch.mbarrier_wait(mbar1, 0)
    tcgen05_fence_after_thread_sync()
    thr_ld1 = tcgen05.make_tmem_copy(ld16, tD1).get_slice(wg_tidx)
    rD1 = cute.make_rmem_tensor(thr_ld1.partition_D(tD1).shape, cutlass.Float32)
    cute.copy(ld16, thr_ld1.partition_S(tD1), rD1)
    cute.arch.fence_view_async_tmem_load()
    rD1_flat = cute.make_tensor(rD1.iterator, cute.make_layout((N1,)))
    v_row = wg_tidx
    # warpgroup 0: D3 (64 x 16) -> smem [L; B_hi; B_lo; -]
    ld3 = cute.make_copy_atom(
        tcgen05.Ld16x128bOp(tcgen05.Repetition(T_PAD * 32 // 128)), cutlass.Float32
    )
    thr_ld3 = tcgen05.make_tmem_copy(ld3, tD3).get_slice(wg_tidx)
    tD3_ld = thr_ld3.partition_S(tD3)
    sLB3 = cute.make_tensor(
        sLB.iterator,
        cute.make_layout(((4 * T_PAD, T_PAD), 1, 1), stride=((T_PAD, 1), 0, 0)),
    )
    tD3_sm = thr_ld3.partition_D(sLB3)
    rD3 = cute.make_rmem_tensor(tD3_sm.shape, cutlass.Float32)
    if wg == 0:
        cute.copy(ld3, tD3_ld, rD3)
        cute.arch.fence_view_async_tmem_load()
        cute.autovec_copy(rD3, tD3_sm)
    cute.arch.barrier()
    # ---- warpgroup 0: V' by forward substitution (row v per thread, four accumulators);
    #      warpgroups 1-2: output-combination operand Bcoef[t, s] = B_hi + B_lo (rows t < T_loop,
    #      s <= t; zero elsewhere), one element per thread, K-major SW64 16 x 16 tiles (hi / lo)
    #      that head the GEMM2 half-0 B tile
    if wg == 0:
        r_Y = cute.make_rmem_tensor(cute.make_layout((T_PAD,)), cutlass.Float32)
        for s in cutlass.range(T_PAD, unroll_full=True):
            y = cutlass.Float32(0.0)
            if s < t_max:
                u0 = rD1_flat[s] + rD1_flat[2 * T_PAD + s]
                y = sBeta[s] * (sVall[s * V + v_row] - u0)
            r_Y[s] = y
        r_V = cute.make_rmem_tensor(cute.make_layout((T_PAD,)), cutlass.Float32)
        for t in cutlass.range(T_PAD, unroll_full=True):
            vp = cutlass.Float32(0.0)
            if t < t_max:
                acc0 = cutlass.Float32(0.0)
                acc1 = cutlass.Float32(0.0)
                acc2 = cutlass.Float32(0.0)
                acc3 = cutlass.Float32(0.0)
                for s in range(t):
                    if s % 4 == 0:
                        acc0 = acc0 + sLB[t, s] * r_V[s]
                    elif s % 4 == 1:
                        acc1 = acc1 + sLB[t, s] * r_V[s]
                    elif s % 4 == 2:
                        acc2 = acc2 + sLB[t, s] * r_V[s]
                    else:
                        acc3 = acc3 + sLB[t, s] * r_V[s]
                vp = r_Y[t] - sBeta[t] * ((acc0 + acc1) + (acc2 + acc3))
            r_V[t] = vp
        for t in cutlass.range(T_PAD, unroll_full=True):
            sVpT[v_row * TP_STRIDE + t] = r_V[t]
    else:
        if tidx < 128 + T_PAD * T_PAD:
            t4 = (tidx - 128) // T_PAD
            s4 = (tidx - 128) % T_PAD
            bc = cutlass.Float32(0.0)
            if (t4 < T_loop) & (s4 <= t4):
                bc = sLB[T_PAD + t4, s4] + sLB[2 * T_PAD + t4, s4]
            bc_hi = round_tf32(bc)
            o_b4 = swz64_k16(t4, s4)
            sB4hif[o_b4] = bc_hi
            sB4lof[o_b4] = bc - bc_hi
    cute.arch.barrier()
    # ---- A2 tiles (v rows x s cols) from sVpT, same 16-lanes-per-row mapping; s > c -> 0
    for r in cutlass.range(4, unroll_full=True):
        v_row2 = 32 * r + 2 * warp_idx + in_warp_tid // T_PAD
        a_val = sVpT[v_row2 * TP_STRIDE + s_col]
        a_hi = round_tf32(a_val)
        o_a2 = swz64_k16(v_row2, s_col)
        sA2hif[o_a2] = a_hi
        sA2lof[o_a2] = a_val - a_hi
    cute.arch.fence_view_async_shared()
    tcgen05_fence_before_thread_sync()
    cute.arch.barrier()
    tcgen05_fence_after_thread_sync()
    sA2hi_tf = cute.make_tensor(
        cute.recast_ptr(sA2hi.iterator, smem_layout_a2.inner, dtype=cutlass.TFloat32),
        sA2hi.layout,
    )
    sA2lo_tf = cute.make_tensor(
        cute.recast_ptr(sA2lo.iterator, smem_layout_a2.inner, dtype=cutlass.TFloat32),
        sA2lo.layout,
    )
    tCrA2hi = tiled_mma2h.make_fragment_A(sA2hi_tf)
    tCrA2lo = tiled_mma2h.make_fragment_A(sA2lo_tf)
    # B2 halves: k rows 0..63 / 64..127 of the SW64 K-major tiles (a 4 KB shift keeps the swizzle)
    tCrB2hi_h1 = tiled_mma2h.make_fragment_B(
        cute.make_tensor(
            cute.recast_ptr(
                sScratch.iterator + OFF_B2HI + 64 * T_PAD,
                smem_layout_b2h.inner,
                dtype=cutlass.TFloat32,
            ),
            smem_layout_b2h.outer,
        )
    )
    tCrB2lo_h1 = tiled_mma2h.make_fragment_B(
        cute.make_tensor(
            cute.recast_ptr(
                sScratch.iterator + OFF_B2LO + 64 * T_PAD,
                smem_layout_b2h.inner,
                dtype=cutlass.TFloat32,
            ),
            smem_layout_b2h.outer,
        )
    )
    tCrA2hi_q = tiled_mma2q.make_fragment_A(sA2hi_tf)
    tCrA2lo_q = tiled_mma2q.make_fragment_A(sA2lo_tf)
    tCrB2q_hi = tiled_mma2q.make_fragment_B(
        cute.make_tensor(
            cute.recast_ptr(
                sScratch.iterator + OFF_B4HI, smem_layout_b2q.inner, dtype=cutlass.TFloat32
            ),
            smem_layout_b2q.outer,
        )
    )
    tCrB2q_lo = tiled_mma2q.make_fragment_B(
        cute.make_tensor(
            cute.recast_ptr(
                sScratch.iterator + OFF_B4LO, smem_layout_b2q.inner, dtype=cutlass.TFloat32
            ),
            smem_layout_b2q.outer,
        )
    )
    if warp_idx == 0:
        # GEMM2 half 0 fused with the output combination: [D4 | dS_0..63] += A2 x [Bcoef; Kbar_0..63]
        tiled_mma2q.set(tcgen05.Field.ACCUMULATE, True)
        for kb_i in cutlass.range_constexpr(T_PAD // 8):
            cute.gemm(
                tiled_mma2q,
                tD2q,
                tCrA2hi_q[(None, None, kb_i, 0)],
                tCrB2q_hi[(None, None, kb_i, 0)],
                tD2q,
            )
        for kb_i in cutlass.range_constexpr(T_PAD // 8):
            cute.gemm(
                tiled_mma2q,
                tD2q,
                tCrA2hi_q[(None, None, kb_i, 0)],
                tCrB2q_lo[(None, None, kb_i, 0)],
                tD2q,
            )
        for kb_i in cutlass.range_constexpr(T_PAD // 8):
            cute.gemm(
                tiled_mma2q,
                tD2q,
                tCrA2lo_q[(None, None, kb_i, 0)],
                tCrB2q_hi[(None, None, kb_i, 0)],
                tD2q,
            )
        with cute.arch.elect_one():
            tcgen05.commit(mbar2)
        tiled_mma2h.set(tcgen05.Field.ACCUMULATE, True)
        for kb_i in cutlass.range_constexpr(T_PAD // 8):
            cute.gemm(
                tiled_mma2h,
                tD2h1,
                tCrA2hi[(None, None, kb_i, 0)],
                tCrB2hi_h1[(None, None, kb_i, 0)],
                tD2h1,
            )
        for kb_i in cutlass.range_constexpr(T_PAD // 8):
            cute.gemm(
                tiled_mma2h,
                tD2h1,
                tCrA2hi[(None, None, kb_i, 0)],
                tCrB2lo_h1[(None, None, kb_i, 0)],
                tD2h1,
            )
        for kb_i in cutlass.range_constexpr(T_PAD // 8):
            cute.gemm(
                tiled_mma2h,
                tD2h1,
                tCrA2lo[(None, None, kb_i, 0)],
                tCrB2hi_h1[(None, None, kb_i, 0)],
                tD2h1,
            )
        with cute.arch.elect_one():
            tcgen05.commit(mbar2b)

    # ---- outputs of the new tokens: o_t = O0_t + D4[v, t] (columns 0..15 of GEMM2 half 0),
    #      warpgroup w writes new tokens j = w mod 4, bf16-rounded, into sVall rows j
    cute.arch.mbarrier_wait(mbar2, 0)
    tcgen05_fence_after_thread_sync()
    thr_ld4 = tcgen05.make_tmem_copy(ld16, tD4).get_slice(wg_tidx)
    rD4 = cute.make_rmem_tensor(thr_ld4.partition_D(tD4).shape, cutlass.Float32)
    cute.copy(ld16, thr_ld4.partition_S(tD4), rD4)
    cute.arch.fence_view_async_tmem_load()
    rD4_flat = cute.make_tensor(rD4.iterator, cute.make_layout((T_PAD,)))
    for j_new in cutlass.range(NEW_TOK, unroll_full=True):
        if wg == j_new % 4:
            t_out = commit_len + j_new
            o_val = cutlass.Float32(0.0)
            for tt in cutlass.range(T_PAD, unroll_full=True):
                if tt == t_out:
                    o_val = rD1_flat[T_PAD + tt] + rD4_flat[tt]
            sVall[j_new * V + v_row] = cutlass.Float32(cutlass.BFloat16(o_val))
    # ---- committed-state store through TMA: D2 slab -> swizzled smem box (row per thread,
    #      16 B chunks land in distinct banks under the 128B swizzle) -> one bulk tensor store
    if cutlass.const_expr(SPLIT_V == 2):
        if vh == 1:
            cute.arch.cluster_wait()
    # All 16 warps produced disjoint value rows above; the barrier makes them visible to the
    # norm epilogue, which assigns one complete token to each warp.
    cute.arch.barrier()
    # The state-owning warps of warpgroups 0/1 stage k columns 0..63 (GEMM2 half 0, mbar2), those
    # of warpgroups 2/3 columns 64..127 (half 1, mbar2b) into the swizzled smem boxes; the first
    # staging thread of each pair issues the TMA stores once its half is visible (named barriers
    # 6 / 7) and waits for the bulk groups at the very end. The warps without state rows run the
    # norm epilogue.
    if wg < 2:
        cute.arch.mbarrier_wait(mbar2, 0)
    else:
        cute.arch.mbarrier_wait(mbar2b, 0)
    tcgen05_fence_after_thread_sync()
    # Warps whose TMEM lanes hold rows this CTA stores (all warps without a split).
    if cutlass.const_expr(SPLIT_V == 2):
        st_warp = (warp_idx % 4) // 2 == vh
    for sl in cutlass.range_constexpr(4):
        if wg == sl:
            if cutlass.const_expr(SPLIT_V == 1) or st_warp:
                cute.copy(ld32, tD2_ld[(None, None, None, sl)], rS)
                cute.arch.fence_view_async_tmem_load()
                for c in cutlass.range(8, unroll_full=True):
                    off_b = sl * (ST_ROWS * 32) + 32 * (wg_tidx - ST_ROWS * vh) + ((4 * c) ^ vx)
                    for i in cutlass.range(4, unroll_full=True):
                        sBoxf[off_b + i] = rS_flat[4 * c + i]
    # last TMEM reads of the kernel: warps 1..15 signal barrier 8, warp 0 frees TMEM below
    tcgen05_fence_before_thread_sync()
    if warp_idx > 0:
        cute.arch.barrier_arrive(barrier_id=8, number_of_threads=NUM_THREADS)
    cute.arch.fence_view_async_shared()
    st_tid0 = (V // 2) * vh
    st_threads = 256 // SPLIT_V
    if cutlass.const_expr(SPLIT_V == 1) or st_warp:
        if wg < 2:
            cute.arch.barrier(barrier_id=6, number_of_threads=st_threads)
            if tidx == st_tid0:
                for s in cutlass.range_constexpr(NUM_SLABS // 2):
                    cute.copy(tma_atom_t, tS_list[s], tG_list[s])
                cute.arch.cp_async_bulk_commit_group()
        else:
            cute.arch.barrier(barrier_id=7, number_of_threads=st_threads)
            if tidx == 256 + st_tid0:
                for s in cutlass.range_constexpr(NUM_SLABS // 2, NUM_SLABS):
                    cute.copy(tma_atom_t, tS_list[s], tG_list[s])
                cute.arch.cp_async_bulk_commit_group()
    if warp_idx == 0:
        # all TMEM reads are done (barrier 8): free the columns while the store drains
        cute.arch.barrier(barrier_id=8, number_of_threads=NUM_THREADS)
        tcgen05_fence_after_thread_sync()
        cute.arch.relinquish_tmem_alloc_permit()
        cute.arch.dealloc_tmem(tmem_ptr, TMEM_COLS)
    # ---- gated RMSNorm epilogue (CTA 0 of a pair) on the warps (warp % 4) in {2, 3}: one token
    #      per warp, p_v = o_v * w_v * sigmoid(g_v), out = bf16(p_v * rsqrt(mean(o_v^2) + eps)).
    token_idx = (warp_idx // 4) * 2 + (warp_idx % 4 - 2)
    norm_warp = (vh == 0) & ((warp_idx % 4) // 2 == 1)
    token_step = NUM_THREADS // 32
    if norm_warp:
        while token_idx < NEW_TOK:
            pre_values = cute.make_rmem_tensor(cute.make_layout((V // 32,)), cutlass.Float32)
            rms_partial = cutlass.Float32(0.0)
            for i in cutlass.range(V // 32, unroll_full=True):
                v_idx = i * 32 + in_warp_tid
                value = sVall[token_idx * V + v_idx]
                gate = sOnG[token_idx * V + v_idx]
                sigmoid_gate = cute.arch.rcp_approx(
                    cutlass.Float32(1.0) + cute.math.exp(-gate, fastmath=True)
                )
                pre_values[i] = value * sOnW[v_idx] * sigmoid_gate
                rms_partial += value * value
            for offset in [16, 8, 4, 2, 1]:
                rms_partial += cute.arch.shuffle_sync_bfly(
                    rms_partial, offset=offset, mask=-1, mask_and_clamp=31
                )
            inv_rms = cute.math.rsqrt(
                rms_partial / cutlass.Float32(V) + cutlass.Float32(onorm_eps),
                fastmath=True,
            )
            output_row = i_n * NEW_TOK + token_idx
            for i in cutlass.range(V // 32, unroll_full=True):
                v_idx = i * 32 + in_warp_tid
                o[0, output_row, i_hv, v_idx] = cutlass.BFloat16(pre_values[i] * inv_rms)
            token_idx = token_idx + token_step
    # the state stores must have read their smem boxes before the CTA exits
    if (tidx == st_tid0) | (tidx == 256 + st_tid0):
        cute.arch.cp_async_bulk_wait_group(0, read=True)


# ---------------------------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------------------------


class _VerifyMmaAtoms(NamedTuple):
    """tcgen05 MMA atoms and operand smem layouts shared by the verify kernels."""

    tiled_mma1: "cute.TiledMma"
    smem_layout_b1: "cute.ComposedLayout"
    tiled_mma_f32a: "cute.TiledMma"
    tiled_mma1k: "cute.TiledMma"
    smem_layout_b1k: "cute.ComposedLayout"
    tiled_mma2h: "cute.TiledMma"
    smem_layout_b2h: "cute.ComposedLayout"
    smem_layout_a2: "cute.ComposedLayout"
    tiled_mma2q: "cute.TiledMma"
    smem_layout_b2q: "cute.ComposedLayout"
    tiled_mma3: "cute.TiledMma"
    smem_layout_a3: "cute.ComposedLayout"
    smem_layout_b3: "cute.ComposedLayout"
    tiled_mma4: "cute.TiledMma"


def _verify_mma_atoms(op_dtype, V: int, K: int, n1: int) -> _VerifyMmaAtoms:
    """The verify MMAs for fp32 ``op_dtype`` operands (3xTF32) with fp32 accumulators.

    GEMM1: M = V rows of the state (TMEM), N = ``n1`` operand rows ([K~; Q~; K~lo] for
    tf32). ``tiled_mma_f32a`` is the fp32 A-fragment layout of the (V x K) state in TMEM
    (the GEMM2 accumulator seed / state store views); for 3xTF32 it is GEMM1's atom.
    GEMM1b: the S0_lo sweep only needs K~hi (N = T_PAD). GEMM2: (V x T_PAD) x (T_PAD x K),
    both operands in smem, issued as two N = K/2 column halves so the state store can start
    on half 0; half 0 also carries the output-combination GEMM (N = T_PAD (Bcoef) + K/2).
    GEMM3: T x T coupling matrices, A3 = [Kf; Qf] padded to 64 rows, B3 = Kb (16 rows).
    ``tiled_mma4`` only provides the accumulator layout of the output combination
    D4[v, t] = sum_s V'[s, v] Bcoef[t, s], the first T_PAD columns of GEMM2 half 0.
    """
    # The only atoms built: 3xTF32 operands, GEMM1 over the [K~; Q~; K~lo] rows (host-side, at trace time).
    assert op_dtype is cutlass.Float32 and n1 == 3 * T_PAD, (op_dtype, n1)
    f32 = cutlass.Float32

    def mma(dtype, shape_mn, source):
        return sm100_utils.make_trivial_tiled_mma(
            dtype,
            OperandMajorMode.K,
            OperandMajorMode.K,
            f32,
            tcgen05.CtaGroup.ONE,
            shape_mn,
            source,
        )

    tmem, smem = tcgen05.OperandSource.TMEM, tcgen05.OperandSource.SMEM
    tiled_mma_f32a = mma(f32, (V, 3 * T_PAD), tmem)
    tiled_mma1 = tiled_mma_f32a
    tiled_mma1k = mma(f32, (V, T_PAD), tmem)
    tiled_mma2h = mma(op_dtype, (V, K // 2), smem)
    tiled_mma2q = mma(op_dtype, (V, T_PAD + K // 2), smem)
    tiled_mma3 = mma(op_dtype, (4 * T_PAD, T_PAD), smem)
    return _VerifyMmaAtoms(
        tiled_mma1=tiled_mma1,
        smem_layout_b1=sm100_utils.make_smem_layout_b(tiled_mma1, (V, n1, K), op_dtype, 1),
        tiled_mma_f32a=tiled_mma_f32a,
        tiled_mma1k=tiled_mma1k,
        smem_layout_b1k=sm100_utils.make_smem_layout_b(tiled_mma1k, (V, T_PAD, K), f32, 1),
        tiled_mma2h=tiled_mma2h,
        smem_layout_b2h=sm100_utils.make_smem_layout_b(
            tiled_mma2h, (V, K // 2, T_PAD), op_dtype, 1
        ),
        smem_layout_a2=sm100_utils.make_smem_layout_a(tiled_mma2h, (V, K // 2, T_PAD), op_dtype, 1),
        tiled_mma2q=tiled_mma2q,
        smem_layout_b2q=sm100_utils.make_smem_layout_b(
            tiled_mma2q, (V, T_PAD + K // 2, T_PAD), op_dtype, 1
        ),
        tiled_mma3=tiled_mma3,
        smem_layout_a3=sm100_utils.make_smem_layout_a(
            tiled_mma3, (4 * T_PAD, T_PAD, K), op_dtype, 1
        ),
        smem_layout_b3=sm100_utils.make_smem_layout_b(
            tiled_mma3, (4 * T_PAD, T_PAD, K), op_dtype, 1
        ),
        tiled_mma4=mma(f32, (V, T_PAD), smem),
    )


class _StateTmaAtoms(NamedTuple):
    """TMA load of the initial state S0 and bulk store of the committed state."""

    tma_atom_l: "cpasync.CopyAtom"
    mS_tma: "cute.Tensor"
    box_layout: "cute.ComposedLayout"
    tma_atom_t: "cpasync.CopyAtom"
    mT_tma: "cute.Tensor"
    box_layout_t: "cute.ComposedLayout"


def _state_tiles(state: cute.Tensor, V: int, K: int) -> cute.Tensor:
    """(V, K, tiles) view of a V-first state pool with its own strides (the per-layer state
    cache may be a strided slice of a larger buffer); the tile mode is the pool index (flat
    ``[pool, V, K]`` layout) or the (head, slot) pair (``[pool, H, V, K]``)."""
    stride = state.layout.stride
    if cute.rank(state.shape) == 3:
        return cute.make_tensor(
            state.iterator,
            cute.make_layout((V, K, state.shape[0]), stride=(stride[1], stride[2], stride[0])),
        )
    return cute.make_tensor(
        state.iterator,
        cute.make_layout(
            (V, K, (state.shape[1], state.shape[0])),
            stride=(stride[2], stride[3], (stride[1], stride[0])),
        ),
    )


def _state_tma_atoms(
    h0: cute.Tensor, ht: cute.Tensor, V: int, K: int, split_v: int
) -> _StateTmaAtoms:
    """S0 is loaded from ``h0`` and the committed state stored to ``ht`` in 128B-swizzled
    boxes of 32 fp32 state columns; each CTA of a ``split_v`` group stores its
    V / split_v rows."""
    box_cols = 128 // (h0.element_type.width // 8)
    box_atom = sm100_utils.make_smem_layout_atom(
        tcgen05.SmemLayoutAtomKind.K_SW128, h0.element_type
    )
    box_layout = cute.tile_to_shape(box_atom, (V, box_cols), order=(0, 1))
    box_layout_t = cute.tile_to_shape(box_atom, (V // split_v, box_cols), order=(0, 1))
    tma_atom_l, mS_tma = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(),
        _state_tiles(h0, V, K),
        box_layout,
        (V, box_cols),
    )
    tma_atom_t, mT_tma = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileS2GOp(),
        _state_tiles(ht, V, K),
        box_layout_t,
        (V // split_v, box_cols),
    )
    return _StateTmaAtoms(tma_atom_l, mS_tma, box_layout, tma_atom_t, mT_tma, box_layout_t)


def _launch_verify(launched, HV: int, N, split_v: int, stream, **launch_kwargs) -> None:
    if split_v == 2:
        # cluster pair: the two CTAs of a (head, request) must be co-scheduled (in-place state)
        launched.launch(
            grid=(HV, N, 2),
            block=[NUM_THREADS, 1, 1],
            cluster=(1, 1, 2),
            stream=stream,
            **launch_kwargs,
        )
    else:
        launched.launch(grid=(HV, N, 1), block=[NUM_THREADS, 1, 1], stream=stream, **launch_kwargs)


@cute.jit
def _run_kda_decode_mtp_v2(
    h0: cute.Tensor,
    x_q: cute.Tensor,
    x_k: cute.Tensor,
    x_v: cute.Tensor,
    w_q: cute.Tensor,
    w_k: cute.Tensor,
    w_v: cute.Tensor,
    cs_q: cute.Tensor,
    cs_k: cute.Tensor,
    cs_v: cute.Tensor,
    A_log: cute.Tensor,
    g: cute.Tensor,
    dt_bias: cute.Tensor,
    beta: cute.Tensor,
    onorm_g: cute.Tensor,
    onorm_weight: cute.Tensor,
    o: cute.Tensor,
    k_cache: cute.Tensor,
    g_cache: cute.Tensor,
    v_cache: cute.Tensor,
    beta_cache: cute.Tensor,
    ssm_state_indices: cute.Tensor,
    num_accepted_tokens: cute.Tensor,
    N: cutlass.Int32,
    scale: cutlass.Constexpr[float],
    HV: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int],
    NUM_SPEC: cutlass.Constexpr[int],
    KERNEL_WIDTH: cutlass.Constexpr[int],
    lower_bound: cutlass.Constexpr[float],
    onorm_eps: cutlass.Constexpr[float],
    USE_FLAT_LAYOUT: cutlass.Constexpr[bool],
    SPLIT_V: cutlass.Constexpr[int],
    stream: cuda.CUstream,
):
    """Host side of ``kda_decode_mtp_v2_kernel`` (packed rows, fused gated RMSNorm, bf16
    output, 3xTF32): the tf32 MMA atoms, operand smem layouts and state TMA atoms of
    ``_run_kda_decode_mtp``, then the launch."""
    mma = _verify_mma_atoms(cutlass.Float32, V, K, 3 * T_PAD)
    state_tma = _state_tma_atoms(h0, h0, V, K, SPLIT_V)
    launched = kda_decode_mtp_v2_kernel(
        h0,
        x_q,
        x_k,
        x_v,
        w_q,
        w_k,
        w_v,
        cs_q,
        cs_k,
        cs_v,
        A_log,
        g,
        dt_bias,
        beta,
        onorm_g,
        onorm_weight,
        o,
        k_cache,
        g_cache,
        v_cache,
        beta_cache,
        ssm_state_indices,
        num_accepted_tokens,
        cute.make_layout((2 * NUM_SPEC + 1, K), stride=(K, 1)),
        mma.tiled_mma1,
        mma.tiled_mma3,
        mma.tiled_mma1k,
        mma.smem_layout_b1,
        mma.smem_layout_a2,
        mma.smem_layout_a3,
        mma.smem_layout_b3,
        mma.smem_layout_b1k,
        mma.tiled_mma4,
        state_tma.tma_atom_t,
        state_tma.mT_tma,
        state_tma.box_layout,
        state_tma.tma_atom_l,
        state_tma.mS_tma,
        mma.tiled_mma2h,
        mma.smem_layout_b2h,
        state_tma.box_layout_t,
        mma.tiled_mma2q,
        mma.smem_layout_b2q,
        scale,
        HV,
        K,
        V,
        NUM_SPEC,
        KERNEL_WIDTH,
        lower_bound,
        onorm_eps,
        USE_FLAT_LAYOUT,
        SPLIT_V,
    )
    _launch_verify(launched, HV, N, SPLIT_V, stream)


def _vec4_rows(tensor: torch.Tensor, dim: int = -1) -> bool:
    """True when every group of four consecutive elements along the unit-stride dimension `dim`
    that starts at a multiple of four is one aligned 8 B (bf16) or 16 B (fp32) access: the v2
    kernel moves its per-token rows with such vector loads and stores."""
    if tensor.dtype not in (torch.bfloat16, torch.float32) or tensor.stride(dim) != 1:
        return False
    if tensor.data_ptr() % (4 * tensor.element_size()):
        return False
    return all(
        stride % 4 == 0 for d, stride in enumerate(tensor.stride()) if d != dim % tensor.ndim
    )


def _verify_operands_ok(*, w_q, w_k, w_v, state, projections, rows, conv_caches, g_cache) -> bool:
    """Operands of the runtime verify contract that ``kda_decode_mtp_v2_kernel`` and
    ``kda_decode_mtp_2cta_kernel`` take.

    Layouts for their vector accesses (see `_vec4_rows`): token rows and replay caches
    vectorized along their last dimension, the dim-contiguous conv caches along the channel
    dimension, conv weights as contiguous [channel][4] rows. Dtypes: fp32 state, bf16
    new-token projections and conv caches, fp32 replay gates.
    """
    weights_ok = all(
        w.dtype == w_q.dtype and w.is_contiguous() and w.shape[1] == 4 and _vec4_rows(w)
        for w in (w_q, w_k, w_v)
    )
    return (
        weights_ok
        and all(_vec4_rows(t) for t in rows)
        and all(_vec4_rows(t, dim=1) for t in conv_caches)
        and state.dtype == torch.float32
        and all(t.dtype == torch.bfloat16 for t in (*projections, *conv_caches))
        and g_cache.dtype == torch.float32
    )


@functools.cache
def _is_supported_capability(device_index: int) -> bool:
    return torch.cuda.get_device_capability(device_index) in ((10, 0), (10, 3), (10, 7))


def supports_device(device: torch.device) -> bool:
    """Whether the kernel runs on ``device`` (SM100, SM103 or SM107)."""
    if device.type != "cuda":
        return False
    index = device.index if device.index is not None else torch.cuda.current_device()
    return _is_supported_capability(index)


# Compiled launchers keyed by the kernel, operand layouts and the kernel's constexprs (no data
# pointers).
_compiled_cache: dict = {}


@functools.cache
def _num_sms_of(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def _num_sms(device: torch.device) -> int:
    return _num_sms_of(device.index if device.index is not None else torch.cuda.current_device())


def k31_kda_mtp_verify_impl(
    x_q: torch.Tensor,
    x_k: torch.Tensor,
    x_v: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    cs_q: torch.Tensor,
    cs_k: torch.Tensor,
    cs_v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    recurrent_state: torch.Tensor,
    k_cache: torch.Tensor,
    g_cache: torch.Tensor,
    v_cache: torch.Tensor,
    beta_cache: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    num_spec: int,
    num_accepted_tokens: torch.Tensor,
    lower_bound: float,
    scale: Optional[float],
    onorm_g: torch.Tensor,
    onorm_weight: torch.Tensor,
    onorm_eps: float,
    out: Optional[torch.Tensor] = None,
    kernel: Optional[str] = None,
    use_pdl: bool = False,
    state_evict_last: bool = False,
) -> torch.Tensor:
    """Launch the verify on the current stream; see ``k31_kda_mtp_verify``.

    ``kernel``: ``"v2"`` (``kda_decode_mtp_v2_kernel``, one CTA or a cluster pair per tile, in
    waves) or ``"two_cta"`` (``kda_decode_mtp_2cta_kernel``, two tiles per SM); by default the
    two-CTA kernel when the grid has more (head, request) tiles than the GPU has SMs. Both give
    the same bits. ``use_pdl``: launch the two-CTA
    kernel as a programmatic dependent that reads only the new-token rows and gates after
    ``griddepcontrol.wait`` and releases its dependent right after it (same bits); the v2 kernel
    launches plainly. ``state_evict_last``: the two-CTA kernel loads the recurrent state with an L2
    evict-last cache policy (same bits; ignored by the v2 kernel).
    """
    _, T_total, _, D = x_q.shape
    H = A_log.shape[0]
    HV = g.shape[2]
    K = D
    V_dim = x_v.shape[-1]
    W = w_q.shape[1]
    if K != TILE_K or V_dim != 128 or W != 4:
        raise ValueError("the verify kernel expects K = 128, V = 128 and conv width 4")
    if HV != H:
        raise ValueError("the verify kernel expects HV == H")
    if not 1 <= num_spec <= MAX_NUM_SPEC:
        raise ValueError(f"num_spec={num_spec} must be in [1, {MAX_NUM_SPEC}]")
    if not supports_device(x_q.device):
        raise RuntimeError("k31_kda_mtp_verify runs on SM100, SM103 and SM107 only")
    N = ssm_state_indices.shape[0]
    if T_total != N * (num_spec + 1):
        raise ValueError(
            f"packed rows: {T_total} new-token rows for {N} requests of {num_spec + 1} rows"
        )
    if recurrent_state.dtype != torch.float32:
        raise ValueError("k31_kda_mtp_verify takes an fp32 recurrent state")
    if scale is None:
        scale = K**-0.5
    if out is None:
        out = torch.empty((1, T_total, HV, V_dim), dtype=x_q.dtype, device=x_q.device)
    if num_accepted_tokens.dtype != torch.int32:
        num_accepted_tokens = num_accepted_tokens.to(torch.int32)
    _require_stride_layout(
        x_q=x_q,
        x_k=x_k,
        x_v=x_v,
        w_q=w_q,
        w_k=w_k,
        w_v=w_v,
        cs_q=cs_q,
        cs_k=cs_k,
        cs_v=cs_v,
        g=g,
        beta=beta,
        A_log=A_log,
        dt_bias=dt_bias,
        recurrent_state=recurrent_state,
        k_cache=k_cache,
        g_cache=g_cache,
        v_cache=v_cache,
        beta_cache=beta_cache,
        ssm_state_indices=ssm_state_indices,
        # The replay caches share the recurrent-state slot (ops/ssm/kda_cache_manager).
        replay_state_indices=None,
        cu_seqlens=None,
        num_accepted_tokens=num_accepted_tokens,
        out=out,
        onorm_g=onorm_g,
        onorm_weight=onorm_weight,
        H=H,
        HV=HV,
        K=K,
        V=V_dim,
        W=W,
        num_spec=num_spec,
        T_total=T_total,
        packed_token_layout=True,
        fuse_output_norm=True,
        quantize_output=False,
    )
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    use_flat_layout, h0_arg, x_q_arg, x_k_arg, x_v_arg = _try_flatten_args(
        recurrent_state=recurrent_state,
        x_q=x_q,
        x_k=x_k,
        x_v=x_v,
        T_total=T_total,
        H=H,
        HV=HV,
        K=K,
        V=V_dim,
    )
    if not _verify_operands_ok(
        w_q=w_q,
        w_k=w_k,
        w_v=w_v,
        state=h0_arg,
        projections=(x_q, x_k, x_v),
        rows=(x_q_arg, x_k_arg, x_v_arg, g, onorm_g, k_cache, g_cache, v_cache),
        conv_caches=(cs_q, cs_k, cs_v),
        g_cache=g_cache,
    ):
        raise ValueError(
            "KDA MTP verify: the packed, fused-norm verify takes 16-byte-vectorizable bf16 "
            "projections and conv caches, contiguous conv weights, fp32 replay gates and an fp32 "
            "state"
        )
    beta_cache_assumed_align = _beta_cache_assumed_align(beta_cache)
    if kernel is None:
        kernel = "two_cta" if HV * N > _num_sms(x_q.device) else "v2"
    if kernel == "two_cta":
        from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kernels.ssm.kda_mtp_decode_2cta import (
            run_kda_decode_mtp_2cta as runner,
        )

        variant = {"PDL": bool(use_pdl)}
        if state_evict_last:
            variant["S0_EVICT_LAST"] = True
    elif kernel == "v2":
        runner = _run_kda_decode_mtp_v2
        # Two CTAs per (head, request) while they fit in one wave (see SPLIT_MAX_TILES).
        variant = {"SPLIT_V": 2 if HV * N <= SPLIT_MAX_TILES else 1}
    else:
        raise ValueError(f"unknown verify kernel {kernel!r}")
    args = [
        (h0_arg, 16),
        (x_q_arg, 16),
        (x_k_arg, 16),
        (x_v_arg, 16),
        (w_q, 16),
        (w_k, 16),
        (w_v, 16),
        (cs_q, 16),
        (cs_k, 16),
        (cs_v, 16),
        (A_log, 16),
        (g, 16),
        (dt_bias, 16),
        (beta, 16),
        (onorm_g, 16),
        (onorm_weight, 16),
        (out, 16),
        (k_cache, 16),
        (g_cache, 16),
        (v_cache, 16),
        (beta_cache, beta_cache_assumed_align),
        (ssm_state_indices, 4),
        (num_accepted_tokens, 4),
    ]
    constexprs = dict(
        scale=scale,
        HV=HV,
        K=K,
        V=V_dim,
        NUM_SPEC=num_spec,
        KERNEL_WIDTH=W,
        lower_bound=float(lower_bound),
        onorm_eps=float(onorm_eps),
        USE_FLAT_LAYOUT=use_flat_layout,
        **variant,
    )
    key = (
        kernel,
        tuple(_layout_key(t, dynamic_layout=True, assumed_align=a) for t, a in args),
        tuple(sorted(constexprs.items())),
    )
    if key not in _compiled_cache:
        _compiled_cache[key] = cute.compile(
            runner,
            *[_dlpack_arg(t, assumed_align=a) for t, a in args],
            N,
            **constexprs,
            stream=stream,
        )
    _compiled_cache[key](*[_dlpack_arg(t, assumed_align=a) for t, a in args], N, stream)
    return out


def k31_kda_mtp_verify(
    x_q: torch.Tensor,
    x_k: torch.Tensor,
    x_v: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    cs_q: torch.Tensor,
    cs_k: torch.Tensor,
    cs_v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    recurrent_state: torch.Tensor,
    k_cache: torch.Tensor,
    g_cache: torch.Tensor,
    v_cache: torch.Tensor,
    beta_cache: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    num_spec: int,
    num_accepted_tokens: torch.Tensor,
    lower_bound: float,
    scale: Optional[float],
    onorm_g: torch.Tensor,
    onorm_weight: torch.Tensor,
    onorm_eps: float,
    use_pdl: bool = False,
    state_evict_last: bool = False,
) -> torch.Tensor:
    """Fused KDA multi-token verify with in-place state commit and the gated RMSNorm epilogue.

    The engine ``trtllm::kda_mtp_decode`` contract with ``packed_token_layout=True``,
    ``fuse_output_norm=True`` and ``cu_seqlens=None`` (see its module docstring for the state
    management): replays each request's ``num_accepted_tokens[slot]`` pending drafts from the
    replay caches, runs its ``1 + num_spec`` new rows, commits state and conv windows after the
    golden row and caches the new drafts. Returns the normed core, bf16 ``[1, T_total, H, V]``.
    Grids with more (head, request) tiles than SMs run the two-CTA kernel, which ``use_pdl``
    launches as a programmatic dependent and ``state_evict_last`` makes load the state with an L2
    evict-last cache policy (same bits).
    """
    return k31_kda_mtp_verify_impl(
        x_q=x_q,
        x_k=x_k,
        x_v=x_v,
        w_q=w_q,
        w_k=w_k,
        w_v=w_v,
        cs_q=cs_q,
        cs_k=cs_k,
        cs_v=cs_v,
        g=g,
        beta=beta,
        A_log=A_log,
        dt_bias=dt_bias,
        recurrent_state=recurrent_state,
        k_cache=k_cache,
        g_cache=g_cache,
        v_cache=v_cache,
        beta_cache=beta_cache,
        ssm_state_indices=ssm_state_indices,
        num_spec=num_spec,
        num_accepted_tokens=num_accepted_tokens,
        lower_bound=lower_bound,
        scale=scale,
        onorm_g=onorm_g,
        onorm_weight=onorm_weight,
        onorm_eps=onorm_eps,
        use_pdl=use_pdl,
        state_evict_last=state_evict_last,
    )
