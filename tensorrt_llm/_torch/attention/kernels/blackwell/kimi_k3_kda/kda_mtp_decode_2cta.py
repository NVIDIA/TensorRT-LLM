# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""KDA multi-token verify sized for two CTAs per SM (large batches).

Computes exactly what ``kda_mtp_decode_v2.kda_decode_mtp_v2_kernel`` computes on its runtime
contract (packed rows, fused gated RMSNorm with a bf16 output, 3xTF32 chunk recurrence, fp32 or
bf16 state, bf16 projections and conv caches, fp32 replay gates, one CTA per (head, request)
tile, forget gate from the projection) with the same arithmetic in the same order, so outputs,
committed state, conv windows and replay caches are bitwise identical. A tile still takes one
512-thread CTA; the footprint shrinks so that two tiles share an SM when the grid has more tiles
than SMs:

* **Tensor memory: 256 columns.** S0 enters GEMM1 as two sequential A operands in the same 128
  columns (S0 hi for GEMM1, then S0 lo for the K~ sweep, issued in v2's order), and the GEMM2
  accumulator (S0 * decay) is seeded into those columns once GEMM1 is done. D4 | D2 | D1 | D3
  = 208 columns. A bf16 state is exact in tf32, so (as in v2) it has no lo part and GEMM1 is
  the hi pass alone.
* **Shared memory: <= 114 KB** (113.7 KB at seven drafts). One 64 KB region holds, in turn, the
  S0 TMA boxes (S0 is moved into registers and its hi part into tensor memory right after the
  precompute), the GEMM3 tiles with the cumulative gates and e^{G_c}, the GEMM1 / Kbar / Bcoef
  tiles once GEMM3 is done, V' over the GEMM1 tiles once GEMM1 is done, and finally the
  committed-state TMA boxes. The raw conv inputs stay bf16 (the committed window and replay
  rows land through ``cp.async``; the new tokens are indexed by token, not position, so their
  placement does not wait for the accepted count), q is staged for the new tokens only, the
  gate warps keep their token's raw gate, beta, dt_bias and A_log in registers (gate of four
  consecutive channels per lane), the epilogue keeps its output-norm operands in registers, and
  the GEMM3 result reuses the conv-weight words. Kbar and V' go straight into their swizzled
  tiles from the row-per-thread producers.
* **Registers: 64 per thread** (``min_blocks_per_mp=2``); D1 is read from tensor memory in the
  16-column pieces each consumer needs, and a bf16 S0 row stays in registers as packed bf16x2
  words. The tensor-memory allocation permit is released right after the allocation so that the
  SM's other CTA can allocate.
* **Named barriers: ids <= 4, all compile-time constants** (see ``BAR_WG0``), which SM107 needs
  to co-schedule two CTAs.

The epilogue writes the BF16 output or, with ``MXFP8_OUT``, its MXFP8 quantization with the
scale layout and the arithmetic of ``kda_decode_mtp_kernel``'s ``QUANTIZE_OUTPUT`` epilogue (the
operand of the Rubin o_proj GEMM).

A tile alone on an SM is in general not faster than in the v2 kernel (the precompute reads bf16
and GEMM1's two A operands are serialized), while two tiles per SM finish sooner than two waves
of v2 tiles, so the kernel pays off for grids with more tiles than SMs. With the MXFP8 output a
tile alone on an SM also finishes sooner than in ``kda_decode_mtp_kernel`` (one CTA or a cluster
pair).
"""

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.typing import Int64
from cutlass.cutlass_dsl import T, dsl_user_op

from .kda_mtp_decode import (
    NUM_THREADS,
    T_PAD,
    TILE_K,
    cvt_bf16x2,
    round_tf32,
    swz64_k16,
    tcgen05_fence_after_thread_sync,
    tcgen05_fence_before_thread_sync,
)
from .kda_mtp_decode_v2 import (
    asm_words,
    gaddr,
    ld_row4,
    saddr,
    st_asm,
    st_global_b32x2,
    st_row4,
    st_shared_f32x4,
    unpack4,
)
from .kda_mtp_decode_v2 import ld_shared_f32x4 as _ld_shared_f32x4

TMEM_COLS = 256
# The 64 KB operand region (floats), used in turn as
#   P0: the S0 TMA boxes (128 B swizzled rows: four 128 x 32 fp32 or two 128 x 64 bf16 boxes),
#       until S0 is in registers
#   P1: A3 = [Kf; Qf hi; Qf lo; -] 64 x 128, B3 hi / lo 16 x 128 (GEMM3), the cumulative gates
#       and e^{G_c} (both kept through P2 / P3)
#   P2 (once GEMM3 is done): B1 = [K~hi; Q~; K~lo] 48 x 128, B1k = K~hi 16 x 128, then
#      [Bcoef hi; Kbar hi] and [Bcoef lo; Kbar lo] (16 + 128) x 16 each (GEMM2 B tiles)
#   P3 (once GEMM1 is done): A2 hi / lo = V'^T 128 x 16 over B1
#   P4 (once GEMM2 is done): the committed-state TMA boxes
OFF_A3 = 0
OFF_B3HI = 8192
OFF_B3LO = 10240
OFF_B1 = 0
OFF_B1K = 6144
OFF_B4HI = 8192
OFF_B2HI = 8448
OFF_B4LO = 10496
OFF_B2LO = 10752
OFF_E = 12800
OFF_EC = 14720
OFF_A2HI = 0
OFF_A2LO = 2048
REGION_FLOATS = 16384
# Named barrier ids (0 is the CTA-wide barrier), all compile-time constants. An SM reserves
# (max id + 1) named barriers per CTA and SM107 co-schedules two CTAs only up to id 4, so the ids
# are reused across phases that CTA-wide barriers separate: BAR_WG0 + wg (1..4) are the
# per-warpgroup GEMM3 barriers, the store / TMEM-release barriers of the epilogue take ids 1..3
# again.
BAR_WG0 = 1
BAR_STORE_LO = 1  # committed-state boxes staged, threads 0..255
BAR_STORE_HI = 2  # committed-state boxes staged, threads 256..511
BAR_TMEM_FREE = 3  # last TMEM reads done -> warp 0 frees the columns


@dsl_user_op
def _cp_async_8(smem_addr, gmem_addr, *, loc=None, ip=None):
    st_asm(
        "cp.async.ca.shared.global [$0], [$1], 8;",
        "r,l",
        [
            cutlass.Int32(smem_addr).ir_value(loc=loc, ip=ip),
            Int64(gmem_addr).ir_value(loc=loc, ip=ip),
        ],
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _cp_async_16(smem_addr, gmem_addr, *, loc=None, ip=None):
    st_asm(
        "cp.async.cg.shared.global [$0], [$1], 16;",
        "r,l",
        [
            cutlass.Int32(smem_addr).ir_value(loc=loc, ip=ip),
            Int64(gmem_addr).ir_value(loc=loc, ip=ip),
        ],
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _cp_async_commit(*, loc=None, ip=None):
    st_asm("cp.async.commit_group;", "", [], loc=loc, ip=ip)


@dsl_user_op
def _cp_async_wait_all(*, loc=None, ip=None):
    st_asm("cp.async.wait_group 0;", "", [], loc=loc, ip=ip)


@dsl_user_op
def _tmem_ld_x1(taddr, *, loc=None, ip=None) -> cutlass.Float32:
    """One 32-bit tensor-memory word per lane (``tcgen05.ld 32x32b.x1``) at ``taddr``."""
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [cutlass.Int32(taddr).ir_value(loc=loc, ip=ip)],
            "tcgen05.ld.sync.aligned.32x32b.x1.b32 {$0}, [$1];",
            "=f,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


def _ld_shared_b32x2(addr):
    return asm_words(
        2,
        "ld.shared.v2.b32 {$0, $1}, [$2];",
        "=r,=r,r",
        [cutlass.Int32(addr).ir_value()],
        loc=None,
        ip=None,
    )


def _st_shared_b32x2(addr, a, b):
    st_asm(
        "st.shared.v2.b32 [$0], {$1, $2};",
        "r,r,r",
        [cutlass.Int32(addr).ir_value(), cutlass.Int32(a).ir_value(), cutlass.Int32(b).ir_value()],
    )


def _s0_value(r_state, r_state_w, idx, packed):
    """fp32 value of S0 column ``idx`` of this thread's row: ``r_state[idx]``, or for a packed bf16
    state the bf16 value ``idx`` of the bf16x2 words ``r_state_w`` (even values in the low half)."""
    if cutlass.const_expr(packed):
        w = r_state_w[idx // 2]
        if cutlass.const_expr(idx % 2 == 0):
            w = w << 16
        else:
            w = w & cutlass.Int32(-65536)
        return w.bitcast(cutlass.Float32)
    return r_state[idx]


def _load_new_token_qkv(
    x_q, x_k, x_v, tok_row, hk_off, hv_off, k4, r_new, use_flat_layout, i_h, i_hv
):
    """v2's new-token q / k / v loads (raw bf16 words, four channels per lane)."""
    if cutlass.const_expr(use_flat_layout):
        ld_row4(x_q, (0, tok_row, hk_off + k4), r_new, 0)
        ld_row4(x_k, (0, tok_row, hk_off + k4), r_new, 4)
        ld_row4(x_v, (0, tok_row, hv_off + k4), r_new, 8)
    else:
        ld_row4(x_q, (0, tok_row, i_h, k4), r_new, 0)
        ld_row4(x_k, (0, tok_row, i_h, k4), r_new, 4)
        ld_row4(x_v, (0, tok_row, i_hv, k4), r_new, 8)


@cute.kernel
def kda_decode_mtp_2cta_kernel(
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
    o8: cute.Tensor,
    osf: cute.Tensor,
    tiled_mma1: cute.TiledMma,
    tiled_mma3: cute.TiledMma,
    tiled_mma1k: cute.TiledMma,
    smem_layout_b1: cute.ComposedLayout,
    smem_layout_a2: cute.ComposedLayout,
    smem_layout_a3: cute.ComposedLayout,
    smem_layout_b3: cute.ComposedLayout,
    smem_layout_b1k: cute.ComposedLayout,
    tiled_mma4: cute.TiledMma,
    tiled_mma2h: cute.TiledMma,
    smem_layout_b2h: cute.ComposedLayout,
    tiled_mma2q: cute.TiledMma,
    smem_layout_b2q: cute.ComposedLayout,
    tma_atom_l: cute.CopyAtom,
    mS_tma: cute.Tensor,
    box_layout: cute.ComposedLayout,
    tma_atom_t: cute.CopyAtom,
    mT_tma: cute.Tensor,
    scale: cutlass.Constexpr[float],
    HV: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int],
    NUM_SPEC: cutlass.Constexpr[int],
    KERNEL_WIDTH: cutlass.Constexpr[int],
    lower_bound: cutlass.Constexpr[float],
    onorm_eps: cutlass.Constexpr[float],
    USE_FLAT_LAYOUT: cutlass.Constexpr[bool],
    MXFP8_OUT: cutlass.Constexpr[bool],
):
    """See the module docstring. Grid ``(HV, N, 1)``, one (head, request) tile per CTA.

    ``MXFP8_OUT``: instead of the BF16 output ``o``, write the MXFP8 operand of the Rubin CuTe
    o_proj GEMM with the arithmetic of ``kda_decode_mtp_kernel``'s ``QUANTIZE_OUTPUT`` epilogue:
    ``o8`` (E4M3 ``[T, HV * V]``) and ``osf`` (uint8, UE8M0 1x128 scales in the R128c4 layout:
    four copies per (row, head)). Without it ``o8`` and ``osf`` are not accessed."""
    tidx, _, _ = cute.arch.thread_idx()
    in_warp_tid = tidx % 32
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    i_hv, i_n, _ = cute.arch.block_idx()
    i_h = i_hv
    slot32 = ssm_state_indices[i_n]
    hk_off = i_h * K
    hv_off = i_hv * V
    N1 = 3 * T_PAD
    K_CACHE_DTYPE = k_cache.element_type
    V_CACHE_DTYPE = v_cache.element_type
    BETA_CACHE_DTYPE = beta_cache.element_type
    t_max = 2 * NUM_SPEC + 1
    NEW_TOK = 1 + NUM_SPEC
    NUM_OLD = KERNEL_WIDTH - 1 + NUM_SPEC  # committed window + replay rows of the raw staging
    vec_size = TILE_K // 32
    REPLAY_WARP0 = NUM_THREADS // 64
    TMA_WARP = NUM_THREADS // 32 - 1
    assert NUM_SPEC <= 7 and t_max <= T_PAD
    assert NEW_TOK <= REPLAY_WARP0 and REPLAY_WARP0 + NUM_SPEC <= TMA_WARP
    assert K == 128 and V == 128 and KERNEL_WIDTH == 4 and NUM_THREADS == 512
    STATE_DTYPE = h0.element_type
    STATE_BF16 = STATE_DTYPE == cutlass.BFloat16
    assert STATE_BF16 or STATE_DTYPE == cutlass.Float32, "fp32 or bf16 recurrent state"
    # a bf16 state is exact in tf32: no S0 lo part (v2's S0_LO)
    S0_LO = not STATE_BF16
    BOX_COLS = 128 // (STATE_DTYPE.width // 8)  # state columns per 128 B swizzled box row
    NUM_SLABS = K // BOX_COLS
    assert g_cache.element_type == cutlass.Float32, "fp32 replay gates (cp.async into sG)"
    assert cs_q.element_type == cutlass.BFloat16 and x_q.element_type == cutlass.BFloat16
    k_weight_base = KERNEL_WIDTH * K
    v_weight_base = 2 * KERNEL_WIDTH * K
    conv_weight_elems = 2 * KERNEL_WIDTH * K + KERNEL_WIDTH * V

    smem = cutlass.utils.SmemAllocator()
    sQn = smem.allocate_tensor(cutlass.Float32, cute.make_layout((NEW_TOK * K,)), 16)
    sK = smem.allocate_tensor(cutlass.Float32, cute.make_layout((t_max, K), stride=(K, 1)), 16)
    sG = smem.allocate_tensor(cutlass.Float32, cute.make_layout((t_max, K), stride=(K, 1)), 16)
    sVall = smem.allocate_tensor(cutlass.Float32, cute.make_layout((t_max * V,)), 16)
    # Conv weights during the precompute; the GEMM3 result [L; B hi; B lo; -] (64 x 16) after.
    sConvW = smem.allocate_tensor(cutlass.Float32, cute.make_layout((conv_weight_elems,)), 16)
    sLB = cute.make_tensor(sConvW.iterator, cute.make_layout((4 * T_PAD, T_PAD), stride=(T_PAD, 1)))
    # Raw conv inputs (bf16): [window; replayed drafts] and the new tokens, per q / k / v.
    # rows [0, NUM_OLD): window + replay rows, then the new tokens' rows
    sXq = smem.allocate_tensor(cutlass.BFloat16, cute.make_layout(((NUM_OLD + NEW_TOK) * K,)), 16)
    sXk = smem.allocate_tensor(cutlass.BFloat16, cute.make_layout(((NUM_OLD + NEW_TOK) * K,)), 16)
    sXv = smem.allocate_tensor(cutlass.BFloat16, cute.make_layout(((NUM_OLD + NEW_TOK) * V,)), 16)
    sXrq, sXrk, sXrv = sXq, sXk, sXv
    sXnq = cute.make_tensor(sXq.iterator + NUM_OLD * K, cute.make_layout((NEW_TOK * K,)))
    sXnk = cute.make_tensor(sXk.iterator + NUM_OLD * K, cute.make_layout((NEW_TOK * K,)))
    sXnv = cute.make_tensor(sXv.iterator + NUM_OLD * V, cute.make_layout((NEW_TOK * V,)))
    sBeta = smem.allocate_tensor(cutlass.Float32, cute.make_layout((t_max,)), 16)
    tmem_holder = smem.allocate_array(cutlass.Int32, 1)
    mbar1a = smem.allocate_array(cutlass.Int64, 1)
    mbar1 = smem.allocate_array(cutlass.Int64, 1)
    mbar3 = smem.allocate_array(cutlass.Int64, 1)
    mbar2 = smem.allocate_array(cutlass.Int64, 1)
    mbar2b = smem.allocate_array(cutlass.Int64, 1)
    mbar_ld = smem.allocate_array(cutlass.Int64, 1)
    sRegion = smem.allocate_tensor(cutlass.Float32, cute.make_layout((REGION_FLOATS,)), 1024)

    def region_tile(off, layout):
        return cute.make_tensor(
            cute.recast_ptr(sRegion.iterator + off, layout.inner, dtype=cutlass.Float32),
            layout.outer,
        )

    sA3 = region_tile(OFF_A3, smem_layout_a3)
    sB3hi = region_tile(OFF_B3HI, smem_layout_b3)
    sB3lo = region_tile(OFF_B3LO, smem_layout_b3)
    sB1 = region_tile(OFF_B1, smem_layout_b1)
    sB1k = region_tile(OFF_B1K, smem_layout_b1k)
    sA2hi = region_tile(OFF_A2HI, smem_layout_a2)
    sA2lo = region_tile(OFF_A2LO, smem_layout_a2)
    # Flat views for closed-form swizzled element writes.
    sA3f = cute.make_tensor(sRegion.iterator + OFF_A3, cute.make_layout((64 * 128,)))
    sB3hif = cute.make_tensor(sRegion.iterator + OFF_B3HI, cute.make_layout((16 * 128,)))
    sB3lof = cute.make_tensor(sRegion.iterator + OFF_B3LO, cute.make_layout((16 * 128,)))
    sB1f = cute.make_tensor(sRegion.iterator + OFF_B1, cute.make_layout((48 * 128,)))
    sB1kf = cute.make_tensor(sRegion.iterator + OFF_B1K, cute.make_layout((T_PAD * 128,)))
    sB4hif = cute.make_tensor(sRegion.iterator + OFF_B4HI, cute.make_layout((T_PAD * T_PAD,)))
    sB4lof = cute.make_tensor(sRegion.iterator + OFF_B4LO, cute.make_layout((T_PAD * T_PAD,)))
    # Cumulative gates G_t (P1 / P2) and the S0 / committed-state boxes (P0 / P4).
    sE = cute.make_tensor(sRegion.iterator + OFF_E, cute.make_layout((t_max * K,)))
    sEc = cute.make_tensor(sRegion.iterator + OFF_EC, cute.make_layout((K,)))
    sBoxf = cute.make_tensor(sRegion.iterator, cute.make_layout((V * K,)))
    sBoxh = cute.make_tensor(
        cute.recast_ptr(sRegion.iterator, dtype=cutlass.BFloat16), cute.make_layout((V * K,))
    )

    # ---- load phase (v2's schedule): slot-independent loads under the slot load; then the
    #      slot-dependent ones. Window and replay raw conv rows and replay gates go to smem through
    #      cp.async (no registers); replay k / v / beta (converted on placement) and the new
    #      tokens through registers.
    ch = tidx % 128
    quad = tidx // 128
    k4 = 4 * in_warp_tid
    r_w = cute.make_rmem_tensor(cute.make_layout((4,)), cutlass.Int32)
    if quad == 0:
        ld_row4(w_q, (hk_off + ch, 0), r_w, 0)
    if quad == 1:
        ld_row4(w_k, (hk_off + ch, 0), r_w, 0)
    if quad == 2:
        ld_row4(w_v, (hv_off + ch, 0), r_w, 0)
    # gate warps (REPLAY_WARP0 + j): this lane's dt_bias channels [4 l, 4 l + 4) and A_log
    j_gate = warp_idx - REPLAY_WARP0
    tok_row_g = i_n * NEW_TOK + j_gate
    r_dtb = cute.make_rmem_tensor(cute.make_layout((4,)), cutlass.Int32)
    r_gate = cute.make_rmem_tensor(cute.make_layout((4,)), cutlass.Int32)
    r_gb = cute.make_rmem_tensor(cute.make_layout((2,)), cutlass.Float32)
    if (warp_idx >= REPLAY_WARP0) & (j_gate < NEW_TOK):
        ld_row4(dt_bias, (hk_off + k4,), r_dtb, 0)
        r_gb[1] = cutlass.Float32(A_log[i_h])
    r_new = cute.make_rmem_tensor(cute.make_layout((3 * 4,)), cutlass.Int32)
    tok_row = i_n * NEW_TOK + warp_idx
    if warp_idx < NEW_TOK:
        _load_new_token_qkv(
            x_q, x_k, x_v, tok_row, hk_off, hv_off, k4, r_new, USE_FLAT_LAYOUT, i_h, i_hv
        )
    if (warp_idx >= REPLAY_WARP0) & (j_gate < NEW_TOK):
        ld_row4(g, (0, tok_row_g, i_hv, k4), r_gate, 0)
        r_gb[0] = cutlass.Float32(beta[0, tok_row_g, i_hv])
    # The TMA warp, the first to use an mbarrier, initializes them; every other warp first touches
    # one after the placement barrier.
    if warp_idx == TMA_WARP:
        with cute.arch.elect_one():
            cute.arch.mbarrier_init(mbar1a, 1)
            cute.arch.mbarrier_init(mbar1, 1)
            cute.arch.mbarrier_init(mbar3, 1)
            cute.arch.mbarrier_init(mbar2, 1)
            cute.arch.mbarrier_init(mbar2b, 1)
            cute.arch.mbarrier_init(mbar_ld, 1)
        cute.arch.mbarrier_init_fence()
        cute.arch.fence_view_async_shared()
    slot = Int64(slot32)
    h0_idx = slot * HV + i_hv
    # byte step between consecutive window positions (mode 2) of the conv states
    cs_step = [cute.crd2idx((0, 0, 1), t.layout) * 2 for t in (cs_q, cs_k, cs_v)]
    rep = warp_idx - REPLAY_WARP0
    r_rep = cute.make_rmem_tensor(cute.make_layout((2 * 4,)), cutlass.Int32)
    r_repb = cute.make_rmem_tensor(cute.make_layout((1,)), cutlass.Float32)
    if (warp_idx >= REPLAY_WARP0) & (rep < NUM_SPEC):
        ld_row4(k_cache, (slot, rep, hk_off + k4), r_rep, 0)
        ld_row4(v_cache, (slot, rep, hv_off + k4), r_rep, 4)
        r_repb[0] = cutlass.Float32(beta_cache[slot, rep, i_hv])
        # every pending entry lands in its row; rows from the accepted count on are overwritten
        # by the new tokens' precompute before they are read
        _cp_async_16(saddr(sG, rep * K + k4), gaddr(g_cache, (slot, rep, hk_off + k4)))
        rr = KERNEL_WIDTH - 1 + rep
        _cp_async_8(saddr(sXrq, rr * K + k4), gaddr(cs_q, (slot, hk_off + k4, rr)))
        _cp_async_8(saddr(sXrk, rr * K + k4), gaddr(cs_k, (slot, hk_off + k4, rr)))
        _cp_async_8(saddr(sXrv, rr * V + k4), gaddr(cs_v, (slot, hv_off + k4, rr)))
        _cp_async_commit()
    if warp_idx == TMA_WARP:
        # committed window rows 0 .. KERNEL_WIDTH - 2: one base per conv state
        wq_base = gaddr(cs_q, (slot, hk_off + k4, 0))
        wk_base = gaddr(cs_k, (slot, hk_off + k4, 0))
        wv_base = gaddr(cs_v, (slot, hv_off + k4, 0))
        for w in cutlass.range_constexpr(KERNEL_WIDTH - 1):
            _cp_async_8(saddr(sXrq, w * K + k4), wq_base + w * cs_step[0])
            _cp_async_8(saddr(sXrk, w * K + k4), wk_base + w * cs_step[1])
            _cp_async_8(saddr(sXrv, w * V + k4), wv_base + w * cs_step[2])
        _cp_async_commit()
    commit_len = num_accepted_tokens[slot]
    # S0 into the region's boxes (consumed right after the precompute)
    if warp_idx == TMA_WARP:
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(mbar_ld, V * K * (STATE_DTYPE.width // 8))
    for sl_h in cutlass.range_constexpr(NUM_SLABS):
        sTl_h = cute.make_tensor(
            cute.recast_ptr(
                sRegion.iterator + sl_h * (V * 32), box_layout.inner, dtype=STATE_DTYPE
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
    if warp_idx == 0:
        cute.arch.alloc_tmem(TMEM_COLS, tmem_holder)
        # the SM's other CTA allocates its 256 columns as soon as we hold ours
        cute.arch.relinquish_tmem_alloc_permit()
    # ---- placement
    if quad < 3:
        wv = unpack4(w_q.element_type, r_w, 0)
        for w in cutlass.range_constexpr(KERNEL_WIDTH):
            sConvW[quad * (KERNEL_WIDTH * K) + w * K + ch] = wv[w]
    if commit_len > NUM_SPEC:
        commit_len = cutlass.Int32(NUM_SPEC)
    T_loop = commit_len + NEW_TOK
    if warp_idx < NEW_TOK:
        # raw rows of new token j stay bf16 and are indexed by j
        _st_shared_b32x2(saddr(sXnq, warp_idx * K + k4), r_new[0], r_new[1])
        _st_shared_b32x2(saddr(sXnk, warp_idx * K + k4), r_new[4], r_new[5])
        _st_shared_b32x2(saddr(sXnv, warp_idx * V + k4), r_new[8], r_new[9])
    if (warp_idx >= REPLAY_WARP0) & (rep < commit_len):
        st_shared_f32x4(saddr(sK, rep * K + k4), *unpack4(K_CACHE_DTYPE, r_rep, 0))
        st_shared_f32x4(saddr(sVall, rep * V + k4), *unpack4(V_CACHE_DTYPE, r_rep, 4))
        if in_warp_tid == 0:
            sBeta[rep] = r_repb[0]
    if ((warp_idx >= REPLAY_WARP0) & (rep < NUM_SPEC)) | (warp_idx == TMA_WARP):
        _cp_async_wait_all()
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
    cute.arch.barrier()

    # ---- precompute (v2's arithmetic; raw rows read as bf16 from the split staging)
    half = NUM_THREADS // 64
    role = warp_idx // half
    j = warp_idx % half
    if j < NEW_TOK:
        t = commit_len + j
        # raw row of tap w of token j: j + w - 3 >= 0 -> new token row, else window / replay row
        x_row = []
        for w in cutlass.range_constexpr(KERNEL_WIDTH):
            r_new_idx = j + w - (KERNEL_WIDTH - 1)
            old_row = commit_len + j + w
            is_new = cutlass.Int32(r_new_idx >= 0)
            x_row.append(old_row + is_new * (NUM_OLD + r_new_idx - old_row))
        if role == 0:
            for i in cutlass.range(vec_size, unroll_full=True):
                k_idx = i * 32 + in_warp_tid
                r_conv_0 = cutlass.Float32(0.0)
                for w in cutlass.range_constexpr(KERNEL_WIDTH):
                    r_conv_0 += cutlass.Float32(sXq[x_row[w] * K + k_idx]) * sConvW[w * K + k_idx]
                r_conv = (
                    cutlass.Float32(sXk[x_row[0] * K + k_idx])
                    * sConvW[k_weight_base + 0 * K + k_idx]
                )
                for w in cutlass.range_constexpr(1, KERNEL_WIDTH):
                    r_conv += (
                        cutlass.Float32(sXk[x_row[w] * K + k_idx])
                        * sConvW[k_weight_base + w * K + k_idx]
                    )
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
                sQn[j * K + k_idx] = r_q[i] * rnorm_q_scaled
                sK[t, k_idx] = r_k[i] * rnorm_k
        else:
            # gate of channels [4 l, 4 l + 4) from this warp's registers (v2's per-channel math)
            r_exp_A = cute.math.exp(r_gb[1], fastmath=True)
            g_raw = unpack4(g.element_type, r_gate, 0)
            dtb = unpack4(dt_bias.element_type, r_dtb, 0)
            gates = []
            for i in cutlass.range_constexpr(4):
                r_g_raw = g_raw[i]
                r_g_raw = r_g_raw + dtb[i]
                exp_A_x = r_exp_A * r_g_raw
                sigmoid_val = cute.arch.rcp_approx(
                    cutlass.Float32(1.0) + cute.math.exp(-exp_A_x, fastmath=True)
                )
                gates.append(lower_bound * sigmoid_val)
            st_shared_f32x4(saddr(sG, t * K + k4), *gates)
            r_beta_t = cute.arch.rcp_approx(
                cutlass.Float32(1.0) + cute.math.exp(-r_gb[0], fastmath=True)
            )
            if in_warp_tid == 0:
                sBeta[t] = r_beta_t
            for i in cutlass.range(vec_size, unroll_full=True):
                v_idx = i * 32 + in_warp_tid
                _v_conv = 0.0
                for w in cutlass.range_constexpr(KERNEL_WIDTH):
                    _v_conv += (
                        cutlass.Float32(sXv[x_row[w] * V + v_idx])
                        * sConvW[v_weight_base + w * V + v_idx]
                    )
                _v_conv = _v_conv * cute.arch.rcp_approx(
                    cutlass.Float32(1.0) + cute.math.exp(-_v_conv, fastmath=True)
                )
                sVall[t * V + v_idx] = _v_conv

    # =================================================================
    # Chunk (WY-form) recurrence on tensor cores.
    # =================================================================
    tmem_ptr = cute.arch.retrieve_tmem_ptr(cutlass.Float32, 16, tmem_holder)
    pool = cutlass.utils.TmemBufferPool(tmem_ptr, TMEM_COLS)
    tD4_frag = tiled_mma4.make_fragment_C(tiled_mma4.partition_shape_C((V, T_PAD)))
    tD4 = pool.allocate_tensor(tD4_frag.layout, cutlass.Float32)
    tD2h_frag = tiled_mma2h.make_fragment_C(tiled_mma2h.partition_shape_C((V, K // 2)))
    tD2h0 = pool.allocate_tensor(tD2h_frag.layout, cutlass.Float32)
    tD2h1 = pool.allocate_tensor(tD2h_frag.layout, cutlass.Float32)
    tD2q_frag = tiled_mma2q.make_fragment_C(tiled_mma2q.partition_shape_C((V, T_PAD + K // 2)))
    tD2q = cute.make_tensor(tD4.iterator, tD2q_frag.layout)
    tA_frag = tiled_mma1.make_fragment_A(tiled_mma1.partition_shape_A((V, K)))
    # S0 hi, then S0 lo, then the GEMM2 accumulator: the same 128 columns
    tX = cute.make_tensor(tD2h0.iterator, tA_frag.layout)
    tD1_frag = tiled_mma1.make_fragment_C(tiled_mma1.partition_shape_C((V, N1)))
    tD1 = pool.allocate_tensor(tD1_frag.layout, cutlass.Float32)
    tD3_frag = tiled_mma3.make_fragment_C(tiled_mma3.partition_shape_C((4 * T_PAD, T_PAD)))
    tD3 = pool.allocate_tensor(tD3_frag.layout, cutlass.Float32)

    st8 = cute.make_copy_atom(tcgen05.St32x32bOp(tcgen05.Repetition.x8), cutlass.Float32)
    ld8 = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition.x8), cutlass.Float32)
    ld16 = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition.x16), cutlass.Float32)
    thr_st8 = tcgen05.make_tmem_copy(st8, tX).get_slice(wg_tidx)
    tX_st8 = thr_st8.partition_D(tX)
    r8 = cute.make_rmem_tensor(thr_st8.partition_S(tX).shape[:-1], cutlass.Float32)
    thr_ld8 = tcgen05.make_tmem_copy(ld8, tX).get_slice(wg_tidx)
    tX_ld8 = thr_ld8.partition_S(tX)
    tS_list = []
    tG_list = []
    for sl_h in cutlass.range_constexpr(NUM_SLABS):
        sT_h = cute.make_tensor(
            cute.recast_ptr(
                sRegion.iterator + sl_h * (V * 32), box_layout.inner, dtype=STATE_DTYPE
            ),
            box_layout.outer,
        )
        gT_h_all = cute.local_tile(mT_tma, (V, BOX_COLS, 1), (0, sl_h, None))
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

    # ---- S0 row v = wg_tidx, columns [32 wg, +32): from its TMA box (v2's conflict-free 16 B
    #      chunks) into registers, its hi part into X; the boxes are free after the barrier, which
    #      also publishes the precompute
    v_row = wg_tidx
    vx = 4 * (wg_tidx % 8)
    r_state = cute.make_rmem_tensor(cute.make_layout((32,)), cutlass.Float32)
    # a bf16 state stays packed: bf16x2 words, value 2 i in the low half of word i
    r_state_w = cute.make_rmem_tensor(cute.make_layout((16,)), cutlass.Int32)
    cute.arch.mbarrier_wait(mbar_ld, 0)
    if cutlass.const_expr(STATE_BF16):
        # bf16 box: 64 columns per 128 B row, so this warpgroup's 32 columns are the 16 B
        # chunks [4 (wg % 2), +4) of box wg // 2 (v2), kept as raw bf16x2 words
        for c in cutlass.range_constexpr(4):
            chunk = (4 * (wg % 2) + c) ^ (wg_tidx % 8)
            off_l = (wg // 2) * (V * 64) + 64 * wg_tidx + 8 * chunk
            words = _ld_shared_f32x4(saddr(sBoxh, off_l))
            for i in cutlass.range_constexpr(4):
                r_state_w[4 * c + i] = words[i].bitcast(cutlass.Int32)
    else:
        for c in cutlass.range(8, unroll_full=True):
            off_l = wg * (V * 32) + 32 * wg_tidx + ((4 * c) ^ vx)
            for i in cutlass.range(4, unroll_full=True):
                r_state[4 * c + i] = sBoxf[off_l + i]
    for jj in cutlass.range_constexpr(4):
        if wg == jj:
            for c in cutlass.range_constexpr(4):
                for i in cutlass.range_constexpr(8):
                    if cutlass.const_expr(S0_LO):
                        r8[i] = round_tf32(r_state[8 * c + i])
                    else:
                        r8[i] = _s0_value(r_state, r_state_w, 8 * c + i, STATE_BF16)
                cute.copy(st8, r8, tX_st8[(None, None, None, 4 * jj + c)])
    cute.arch.fence_view_async_tmem_store()
    cute.arch.barrier()

    # ---- cumulative gates (v2), q of the replayed drafts is zero
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
    exp_clamp = cutlass.Float32(87.0)
    ka = kq // 32
    kc = kq % 32
    b1_base = ka * (3 * T_PAD * 32)
    a3_base = ka * (4 * T_PAD * 32)
    b3_base = ka * (T_PAD * 32)
    b1k_base = ka * (T_PAD * 32)
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
                q_t = cutlass.Float32(0.0)
                if t >= commit_len:
                    q_t = sQn[(t - commit_len) * K + kq]
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
        with cute.arch.elect_one():
            tcgen05.commit(mbar3)

    # GEMM1 operand rows over the GEMM3 tiles once GEMM3 has read them; D3 -> smem. One warp per
    # warpgroup polls the mbarrier, the others wait at the warpgroup's barrier.
    if warp_idx % 4 == 1:
        cute.arch.mbarrier_wait(mbar3, 0)
    # one compile-time barrier id per warpgroup: a barrier id held in a register makes the CTA
    # reserve all 16 named barriers, which leaves room for one CTA per SM on SM107
    for jj in cutlass.range_constexpr(4):
        if wg == jj:
            cute.arch.barrier(barrier_id=BAR_WG0 + jj, number_of_threads=128)
    tcgen05_fence_after_thread_sync()
    for t in cutlass.range(T_PAD, unroll_full=True):
        if quarter == t % 4:
            xc = kc ^ (4 * (t % 8))
            sB1f[b1_base + 32 * t + xc] = r_b1[3 * (t // 4)]
            if cutlass.const_expr(S0_LO):
                sB1kf[b1k_base + 32 * t + xc] = r_b1[3 * (t // 4)]
            sB1f[b1_base + 32 * (T_PAD + t) + xc] = r_b1[3 * (t // 4) + 1]
            sB1f[b1_base + 32 * (2 * T_PAD + t) + xc] = r_b1[3 * (t // 4) + 2]
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
    cute.arch.fence_view_async_shared()
    tcgen05_fence_before_thread_sync()
    cute.arch.barrier()
    tcgen05_fence_after_thread_sync()
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
    tX_tf = cute.make_tensor(cute.recast_ptr(tX.iterator, dtype=cutlass.TFloat32), tX.layout)
    if warp_idx == 0:
        tiled_mma1.set(tcgen05.Field.ACCUMULATE, False)
        for kb_i in cutlass.range_constexpr(K // 8):
            cute.gemm(tiled_mma1, tD1, tX_tf[(None, None, kb_i)], tCrB1[(None, None, kb_i, 0)], tD1)
            tiled_mma1.set(tcgen05.Field.ACCUMULATE, True)
        with cute.arch.elect_one():
            if cutlass.const_expr(S0_LO):
                tcgen05.commit(mbar1a)
            else:
                tcgen05.commit(mbar1)
    # ---- under GEMM1: D4 zero, Bcoef tiles (warpgroups 1..2, from sLB), Kbar tiles, the new
    #      tokens' replay-cache rows and conv-window commits
    st16 = cute.make_copy_atom(tcgen05.St32x32bOp(tcgen05.Repetition.x16), cutlass.Float32)
    thr_stD4 = tcgen05.make_tmem_copy(st16, tD4).get_slice(wg_tidx)
    tD4_st = thr_stD4.partition_D(tD4)
    r_zero = cute.make_rmem_tensor(thr_stD4.partition_S(tD4).shape, cutlass.Float32)
    for i in cutlass.range(cute.size(r_zero), unroll_full=True):
        r_zero[i] = cutlass.Float32(0.0)
    if wg == 0:
        cute.copy(st16, r_zero, tD4_st)
    if (tidx >= 128) & (tidx < 128 + T_PAD * T_PAD):
        t4 = (tidx - 128) // T_PAD
        s4 = (tidx - 128) % T_PAD
        bc = cutlass.Float32(0.0)
        if (t4 < T_loop) & (s4 <= t4):
            bc = sLB[T_PAD + t4, s4] + sLB[2 * T_PAD + t4, s4]
        bc_hi = round_tf32(bc)
        o_b4 = swz64_k16(t4, s4)
        sB4hif[o_b4] = bc_hi
        sB4lof[o_b4] = bc - bc_hi
    # Kbar[k, s] = K[s, k] e^{G_c[k] - G_s[k]} (s <= c, else 0): thread (k = kq, quarter) owns
    # columns s = 4 quarter + {0..3}, i.e. one 16 B chunk of row k of the SW64 hi / lo tiles
    sB2hi_row = saddr(sRegion, OFF_B2HI)
    sB2lo_row = saddr(sRegion, OFF_B2LO)
    # (quarters whose first column is past the committed count hold zeros only)
    kb_0 = cutlass.Float32(0.0)
    kb_1 = cutlass.Float32(0.0)
    kb_2 = cutlass.Float32(0.0)
    kb_3 = cutlass.Float32(0.0)
    s_q = 4 * quarter
    if s_q <= commit_len:
        e_c = sE[commit_len * K + kq]
        kb_0 = sK[s_q, kq] * cute.math.exp(e_c - sE[s_q * K + kq], fastmath=True)
        if s_q + 1 <= commit_len:
            kb_1 = sK[s_q + 1, kq] * cute.math.exp(e_c - sE[(s_q + 1) * K + kq], fastmath=True)
        if s_q + 2 <= commit_len:
            kb_2 = sK[s_q + 2, kq] * cute.math.exp(e_c - sE[(s_q + 2) * K + kq], fastmath=True)
        if s_q + 3 <= commit_len:
            kb_3 = sK[s_q + 3, kq] * cute.math.exp(e_c - sE[(s_q + 3) * K + kq], fastmath=True)
    kb_vals = [kb_0, kb_1, kb_2, kb_3]
    kb_his = [round_tf32(v) for v in kb_vals]
    o_b2 = swz64_k16(kq, 4 * quarter)
    st_shared_f32x4(sB2hi_row + 4 * o_b2, *kb_his)
    st_shared_f32x4(sB2lo_row + 4 * o_b2, *[kb_vals[i] - kb_his[i] for i in range(4)])
    if (warp_idx >= 1) & (warp_idx <= NUM_SPEC):
        p = warp_idx - 1
        t_c = commit_len + 1 + p
        st_row4(k_cache, (slot, p, hk_off + k4), _ld_shared_f32x4(saddr(sK, t_c * K + k4)))
        st_row4(g_cache, (slot, p, hk_off + k4), _ld_shared_f32x4(saddr(sG, t_c * K + k4)))
        st_row4(v_cache, (slot, p, hv_off + k4), _ld_shared_f32x4(saddr(sVall, t_c * V + k4)))
        # draft p is new token 1 + p
        wq = _ld_shared_b32x2(saddr(sXnq, (1 + p) * K + k4))
        st_global_b32x2(gaddr(cs_q, (slot, hk_off + k4, KERNEL_WIDTH - 1 + p)), wq[0], wq[1])
        wk = _ld_shared_b32x2(saddr(sXnk, (1 + p) * K + k4))
        st_global_b32x2(gaddr(cs_k, (slot, hk_off + k4, KERNEL_WIDTH - 1 + p)), wk[0], wk[1])
        wvv = _ld_shared_b32x2(saddr(sXnv, (1 + p) * V + k4))
        st_global_b32x2(gaddr(cs_v, (slot, hv_off + k4, KERNEL_WIDTH - 1 + p)), wvv[0], wvv[1])
        if in_warp_tid == 0:
            beta_cache[slot, p, i_hv] = sBeta[t_c].to(BETA_CACHE_DTYPE)
    # committed window: raw rows commit_len + 1 + w (w = 2 is new token 0)
    for sec in cutlass.range_constexpr(3):
        if warp_idx == REPLAY_WARP0 + sec:
            cs_sec = (cs_q, cs_k, cs_v)[sec]
            old_sec = (sXrq, sXrk, sXrv)[sec]
            new_sec = (sXnq, sXnk, sXnv)[sec]
            off_sec = hv_off if sec == 2 else hk_off
            sec_base = gaddr(cs_sec, (slot, off_sec + k4, 0))
            for w in cutlass.range_constexpr(KERNEL_WIDTH - 1):
                if cutlass.const_expr(w + 1 >= KERNEL_WIDTH - 1):
                    src = saddr(new_sec, (w + 1 - (KERNEL_WIDTH - 1)) * K + k4)
                else:
                    src = saddr(old_sec, (commit_len + 1 + w) * K + k4)
                ww = _ld_shared_b32x2(src)
                st_global_b32x2(sec_base + w * cs_step[sec], ww[0], ww[1])
    # ---- GEMM1's S0 lo sweep once the hi pass has read X
    if cutlass.const_expr(S0_LO):
        cute.arch.mbarrier_wait(mbar1a, 0)
        tcgen05_fence_after_thread_sync()
        for jj in cutlass.range_constexpr(4):
            if wg == jj:
                for c in cutlass.range_constexpr(4):
                    for i in cutlass.range_constexpr(8):
                        s_hi = round_tf32(r_state[8 * c + i])
                        r8[i] = r_state[8 * c + i] - s_hi
                    cute.copy(st8, r8, tX_st8[(None, None, None, 4 * jj + c)])
        cute.arch.fence_view_async_tmem_store()
        cute.arch.fence_view_async_shared()
        tcgen05_fence_before_thread_sync()
        cute.arch.barrier()
        tcgen05_fence_after_thread_sync()
        if warp_idx == 0:
            tiled_mma1k.set(tcgen05.Field.ACCUMULATE, True)
            for kb_i in cutlass.range_constexpr(K // 8):
                cute.gemm(
                    tiled_mma1k,
                    tD1k,
                    tX_tf[(None, None, kb_i)],
                    tCrB1k[(None, None, kb_i, 0)],
                    tD1k,
                )
            with cute.arch.elect_one():
                tcgen05.commit(mbar1)
    # ---- GEMM2 accumulator seed S0 * e^{G_c} into X once GEMM1 is done
    cute.arch.mbarrier_wait(mbar1, 0)
    tcgen05_fence_after_thread_sync()
    for jj in cutlass.range_constexpr(4):
        if wg == jj:
            for c in cutlass.range_constexpr(4):
                for i in cutlass.range_constexpr(8):
                    r8[i] = (
                        _s0_value(r_state, r_state_w, 8 * c + i, STATE_BF16)
                        * sEc[jj * 32 + 8 * c + i]
                    )
                cute.copy(st8, r8, tX_st8[(None, None, None, 4 * jj + c)])
    cute.arch.fence_view_async_tmem_store()
    # ---- warpgroup 0: V' by forward substitution (row v per thread) straight into the A2 tiles
    thr_ld1 = tcgen05.make_tmem_copy(ld16, tD1).get_slice(wg_tidx)
    tD1_ld = thr_ld1.partition_S(tD1)
    # 16-column chunks of D1 are mode 1 of the ld16 partition: [K~hi | Q~ | K~lo]
    rD1_shape = thr_ld1.partition_D(tD1)[(None, 0, None, None)].shape
    rD1p = cute.make_rmem_tensor(rD1_shape, cutlass.Float32)
    rD1q = cute.make_rmem_tensor(rD1_shape, cutlass.Float32)
    rD1p_flat = cute.make_tensor(rD1p.iterator, cute.make_layout((T_PAD,)))
    rD1q_flat = cute.make_tensor(rD1q.iterator, cute.make_layout((T_PAD,)))
    if wg == 0:
        cute.copy(ld16, tD1_ld[(None, 0, None, None)], rD1p)
        cute.copy(ld16, tD1_ld[(None, 2, None, None)], rD1q)
        cute.arch.fence_view_async_tmem_load()
        r_Y = cute.make_rmem_tensor(cute.make_layout((T_PAD,)), cutlass.Float32)
        for s in cutlass.range(T_PAD, unroll_full=True):
            y = cutlass.Float32(0.0)
            if s < t_max:
                u0 = rD1p_flat[s] + rD1q_flat[s]
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
        sA2hi_row = saddr(sRegion, OFF_A2HI)
        sA2lo_row = saddr(sRegion, OFF_A2LO)
        for c in cutlass.range_constexpr(4):
            his = [round_tf32(r_V[4 * c + i]) for i in range(4)]
            o_a2 = swz64_k16(v_row, 4 * c)
            st_shared_f32x4(sA2hi_row + 4 * o_a2, *his)
            st_shared_f32x4(sA2lo_row + 4 * o_a2, *[r_V[4 * c + i] - his[i] for i in range(4)])
    cute.arch.fence_view_async_shared()
    tcgen05_fence_before_thread_sync()
    cute.arch.barrier()
    tcgen05_fence_after_thread_sync()
    sA2hi_tf = cute.make_tensor(
        cute.recast_ptr(sA2hi.iterator, smem_layout_a2.inner, dtype=cutlass.TFloat32), sA2hi.layout
    )
    sA2lo_tf = cute.make_tensor(
        cute.recast_ptr(sA2lo.iterator, smem_layout_a2.inner, dtype=cutlass.TFloat32), sA2lo.layout
    )
    tCrA2hi = tiled_mma2h.make_fragment_A(sA2hi_tf)
    tCrA2lo = tiled_mma2h.make_fragment_A(sA2lo_tf)
    tCrB2hi_h1 = tiled_mma2h.make_fragment_B(
        cute.make_tensor(
            cute.recast_ptr(
                sRegion.iterator + OFF_B2HI + 64 * T_PAD,
                smem_layout_b2h.inner,
                dtype=cutlass.TFloat32,
            ),
            smem_layout_b2h.outer,
        )
    )
    tCrB2lo_h1 = tiled_mma2h.make_fragment_B(
        cute.make_tensor(
            cute.recast_ptr(
                sRegion.iterator + OFF_B2LO + 64 * T_PAD,
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
                sRegion.iterator + OFF_B4HI, smem_layout_b2q.inner, dtype=cutlass.TFloat32
            ),
            smem_layout_b2q.outer,
        )
    )
    tCrB2q_lo = tiled_mma2q.make_fragment_B(
        cute.make_tensor(
            cute.recast_ptr(
                sRegion.iterator + OFF_B4LO, smem_layout_b2q.inner, dtype=cutlass.TFloat32
            ),
            smem_layout_b2q.outer,
        )
    )
    if warp_idx == 0:
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
    # output-gate operand of the epilogue (warps 2, 3 of each warpgroup: one token each)
    token_idx = (warp_idx // 4) * 2 + (warp_idx % 4 - 2)
    r_og = cute.make_rmem_tensor(cute.make_layout((V // 32,)), cutlass.Float32)
    r_ow = cute.make_rmem_tensor(cute.make_layout((V // 32,)), cutlass.Float32)
    if ((warp_idx % 4) // 2 == 1) & (token_idx < NEW_TOK):
        for i in cutlass.range(V // 32, unroll_full=True):
            r_og[i] = cutlass.Float32(
                onorm_g[0, i_n * NEW_TOK + token_idx, i_hv, i * 32 + in_warp_tid]
            )
            r_ow[i] = cutlass.Float32(onorm_weight[i * 32 + in_warp_tid])
    # ---- outputs of the new tokens: o_t = Q-part of D1 + D4 (v2), bf16-rounded into sVall rows;
    #      warpgroup wg reads column t of both for its tokens wg and wg + 4 (lane = row v)
    cute.arch.mbarrier_wait(mbar2, 0)
    tcgen05_fence_after_thread_sync()
    lane_hi = (warp_idx % 4) << 21
    d4_col = cutlass.Int32(tD4.iterator.toint()) + lane_hi
    d1q_col = cutlass.Int32(tD1.iterator.toint()) + lane_hi + T_PAD
    for jw in cutlass.range_constexpr((NEW_TOK + 3) // 4):
        j_out = wg + 4 * jw
        if j_out < NEW_TOK:
            t_col = commit_len + j_out
            o_q = _tmem_ld_x1(d1q_col + t_col)
            o_d = _tmem_ld_x1(d4_col + t_col)
            cute.arch.fence_view_async_tmem_load()
            sVall[j_out * V + v_row] = cutlass.Float32(cutlass.BFloat16(o_q + o_d))
    cute.arch.barrier()
    # ---- committed state: warpgroup wg stages columns [32 wg, +32) into box wg of the region
    #      (free once GEMM2 is done), then one thread per half issues the TMA stores (v2)
    cute.arch.mbarrier_wait(mbar2b, 0)
    tcgen05_fence_after_thread_sync()
    for jj in cutlass.range_constexpr(4):
        if wg == jj:
            for cc in cutlass.range_constexpr(4):
                cute.copy(ld8, tX_ld8[(None, None, None, 4 * jj + cc)], r8)
                cute.arch.fence_view_async_tmem_load()
                if cutlass.const_expr(STATE_BF16):
                    # bf16 box jj // 2, 16 B chunk 4 (jj % 2) + cc, round to nearest (v2's
                    # cvt.rn per value, two values per cvt.rn.bf16x2)
                    chunk = (4 * (jj % 2) + cc) ^ (wg_tidx % 8)
                    off_b = (jj // 2) * (V * 64) + 64 * wg_tidx + 8 * chunk
                    st_shared_f32x4(
                        saddr(sBoxh, off_b),
                        *[cvt_bf16x2(r8[2 * i + 1], r8[2 * i]) for i in range(4)],
                    )
                else:
                    for h in cutlass.range_constexpr(2):
                        c = 2 * cc + h
                        off_b = jj * (V * 32) + 32 * wg_tidx + ((4 * c) ^ vx)
                        for i in cutlass.range_constexpr(4):
                            sBoxf[off_b + i] = r8[4 * h + i]
    tcgen05_fence_before_thread_sync()
    if warp_idx > 0:
        cute.arch.barrier_arrive(barrier_id=BAR_TMEM_FREE, number_of_threads=NUM_THREADS)
    cute.arch.fence_view_async_shared()
    if wg < 2:
        cute.arch.barrier(barrier_id=BAR_STORE_LO, number_of_threads=256)
        if tidx == 0:
            for sl in cutlass.range_constexpr(NUM_SLABS // 2):
                cute.copy(tma_atom_t, tS_list[sl], tG_list[sl])
            cute.arch.cp_async_bulk_commit_group()
    else:
        cute.arch.barrier(barrier_id=BAR_STORE_HI, number_of_threads=256)
        if tidx == 256:
            for sl in cutlass.range_constexpr(NUM_SLABS // 2, NUM_SLABS):
                cute.copy(tma_atom_t, tS_list[sl], tG_list[sl])
            cute.arch.cp_async_bulk_commit_group()
    if warp_idx == 0:
        cute.arch.barrier(barrier_id=BAR_TMEM_FREE, number_of_threads=NUM_THREADS)
        tcgen05_fence_after_thread_sync()
        cute.arch.dealloc_tmem(tmem_ptr, TMEM_COLS)
    # ---- gated RMSNorm epilogue (v2)
    if ((warp_idx % 4) // 2 == 1) & (token_idx < NEW_TOK):
        pre_values = cute.make_rmem_tensor(cute.make_layout((V // 32,)), cutlass.Float32)
        rms_partial = cutlass.Float32(0.0)
        pre_max = cutlass.Float32(0.0)
        for i in cutlass.range(V // 32, unroll_full=True):
            v_idx = i * 32 + in_warp_tid
            value = sVall[token_idx * V + v_idx]
            gate = r_og[i]
            sigmoid_gate = cute.arch.rcp_approx(
                cutlass.Float32(1.0) + cute.math.exp(-gate, fastmath=True)
            )
            p_v = value * r_ow[i] * sigmoid_gate
            pre_values[i] = p_v
            rms_partial += value * value
            if cutlass.const_expr(MXFP8_OUT):
                pre_max = cute.arch.fmax(pre_max, cute.arch.fmax(p_v, -p_v))
        for offset in [16, 8, 4, 2, 1]:
            rms_partial += cute.arch.shuffle_sync_bfly(
                rms_partial, offset=offset, mask=-1, mask_and_clamp=31
            )
            if cutlass.const_expr(MXFP8_OUT):
                pre_max = cute.arch.fmax(
                    pre_max,
                    cute.arch.shuffle_sync_bfly(pre_max, offset=offset, mask=-1, mask_and_clamp=31),
                )
        inv_rms = cute.math.rsqrt(
            rms_partial / cutlass.Float32(V) + cutlass.Float32(onorm_eps), fastmath=True
        )
        output_row = i_n * NEW_TOK + token_idx
        if cutlass.const_expr(MXFP8_OUT):
            # kda_decode_mtp_kernel's QUANTIZE_OUTPUT epilogue: BF16-rounded normalized values,
            # UE8M0 scale of bf16(max |p_v| * inv_rms) / 448 (bf16 rounding is monotonic), E4M3
            # values, the scale byte replicated four times in the R128c4 layout
            output_amax = cutlass.Float32(cutlass.BFloat16(pre_max * inv_rms))
            output_amax = cute.arch.fmax(output_amax, cutlass.Float32(1.0e-10))
            output_sf = (output_amax * cutlass.Float32(1.0 / 448.0)).to(cutlass.Float8E8M0FNU)
            quant_scale = cute.arch.rcp_approx(cutlass.Float32(output_sf))
            for i in cutlass.range(V // 32, unroll_full=True):
                normalized = cutlass.Float32(cutlass.BFloat16(pre_values[i] * inv_rms))
                o8[output_row, hv_off + i * 32 + in_warp_tid] = cutlass.Float8E4M3FN(
                    normalized * quant_scale
                )
            if in_warp_tid == 0:
                scale_bytes = cute.make_tensor(
                    cute.recast_ptr(osf.iterator, dtype=cutlass.Float8E8M0FNU),
                    cute.make_layout(osf.shape),
                )
                scale_offset = (
                    (output_row // 128) * HV * 512
                    + i_hv * 512
                    + (output_row % 32) * 16
                    + ((output_row % 128) // 32) * 4
                )
                for replica in cutlass.range_constexpr(4):
                    scale_bytes[scale_offset + replica] = output_sf
        else:
            for i in cutlass.range(V // 32, unroll_full=True):
                v_idx = i * 32 + in_warp_tid
                o[0, output_row, i_hv, v_idx] = cutlass.BFloat16(pre_values[i] * inv_rms)
    # the state stores must have read their smem boxes before the CTA exits
    if (tidx == 0) | (tidx == 256):
        cute.arch.cp_async_bulk_wait_group(0, read=True)
