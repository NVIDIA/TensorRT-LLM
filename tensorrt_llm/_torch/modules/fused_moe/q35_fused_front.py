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
"""Fused decode front end for the CuteDSL NVFP4 MoE (Qwen3.5 low-latency, opt-in).

For decode-sized batches (M <= 32 tokens) this replaces the chain that runs
between the router GEMM and the FC1 grouped GEMM of ``CuteDslFusedMoE``:

    renorm_moe_routing_op (customMoeRoutingKernel: top-k + softmax)
    -> fp4_quantize (quantize_with_block_size, linear scale layout)
    -> moe_sort (routingIndicesClusterKernel: tile / permutation tables)
    -> moe_output memset (aux stream, joined by two events)

with ONE Triton launch:

* program 0: per-token top-k over the router logits (ties -> smaller expert
  index, as ``reduceTopK``), softmax over the k selected logits in the
  ``calcSoftmax`` order (expf, xor-butterfly sum, IEEE division), then the
  ``moe_sort`` tables for this rank's local experts (one tile per active
  expert, since M <= tile size);
* programs 1..M: NVFP4 quantization of one hidden row (the
  ``cvt_warp_fp16_to_fp4`` recipe: rcp.approx reciprocals, e4m3 scale,
  cvt.rn.satfinite.e2m1x2) and zeroing of that row of the MoE output.

The kernel is a PDL dependent of the router GEMM (nothing is read before
``griddepcontrol.wait``) and releases its dependents only after a CTA-wide
barrier that follows its last store.

Within one expert the order of the permuted rows comes from atomics, as in
``routingIndicesClusterKernel``; the GEMM rows are independent and the
finalize kernel scatter-adds with async bulk reductions, so this does not
change what the stock path guarantees.

Enabled by ``TRTLLM_Q35_MOE_FUSED_FRONT=1`` (default off).
"""

import os
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from ...autotuner import AutoTuner

Q35_MOE_FUSED_FRONT = os.environ.get("TRTLLM_Q35_MOE_FUSED_FRONT", "0") == "1"
# Largest decode batch (tokens) the fused front end handles; larger batches
# (prefill, chunked context) keep the stock path.
Q35_MOE_FUSED_FRONT_MAX_TOKENS = 32


SIGN_BIT = tl.constexpr(-2147483648)
MIN_KEY = tl.constexpr(-2147483648)


@triton.jit
def _gdc_wait_with_memory_clobber():
    tl.inline_asm_elementwise(
        "griddepcontrol.wait; // dummy $0",
        "=r,~{memory}",
        [],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _rcp_approx(x):
    return tl.inline_asm_elementwise("rcp.approx.ftz.f32 $0, $1;",
                                     "=f,f", [x],
                                     dtype=tl.float32,
                                     is_pure=True,
                                     pack=1)


@triton.jit
def _pick_col(x, cols, k: tl.constexpr):
    # Exact column extraction (adds zeros only).
    return tl.sum(tl.where(cols[None, :] == k, x, 0.0), axis=1)


@triton.jit
def _nvfp4_quant_row(x_ptr, x_row_stride, gs_ptr, xq_ptr, xsf_ptr, r,
                     H: tl.constexpr):
    """NVFP4-quantize hidden row r (cvt_warp_fp16_to_fp4 recipe, linear SF)."""
    cols = tl.arange(0, H)
    v = tl.load(x_ptr + r * x_row_stride + cols).to(tl.float32)
    v16 = tl.reshape(v, (H // 16, 16))
    vmax = tl.max(tl.abs(v16), axis=1)
    gs = tl.load(gs_ptr)
    rcp6 = _rcp_approx(tl.full((H // 16, ), 6.0, tl.float32))
    sf8 = (gs * (vmax * rcp6)).to(tl.float8e4nv)
    rcpgs = _rcp_approx(tl.full((H // 16, ), 0.0, tl.float32) + gs)
    osc = _rcp_approx(sf8.to(tl.float32) * rcpgs)
    osc = tl.where(vmax != 0.0, osc, 0.0)
    tl.store(xsf_ptr + r * (H // 16) + tl.arange(0, H // 16),
             sf8.to(tl.uint8, bitcast=True))
    y = tl.reshape(v16 * osc[:, None], (H, ))
    lo, hi = tl.split(tl.reshape(y, (H // 2, 2)))
    byte = tl.inline_asm_elementwise(
        "{ .reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, $2, $1; cvt.u16.u8 $0, t; }",
        "=h,f,f",
        [lo, hi],
        dtype=tl.uint16,
        is_pure=True,
        pack=1,
    ).to(tl.uint8)
    tl.store(xq_ptr + r * (H // 2) + tl.arange(0, H // 2), byte)


@triton.jit
def _q35_moe_front_kernel(
    logits_ptr,  # [M, E] router logits (bf16)
    x_ptr,  # [M, H] bf16 hidden states
    x_row_stride,
    gs_ptr,  # [1] fp32 FC1 input global scale (fc31_input_scale)
    ids_ptr,  # out [M, K] int32 selected experts
    w_ptr,  # out [M, K] fp32 renormalized weights
    xq_ptr,  # out [M, H // 2] uint8 e2m1x2 (element 2j in the low nibble)
    xsf_ptr,  # out [M, H // 16] uint8 e4m3 scales (linear layout)
    tile_expert_ptr,  # out [max_tiles] int32 local expert of each tile
    tile_mnlimit_ptr,  # out [max_tiles] int32
    exp2perm_ptr,  # out [M * K] int32 (-1 when the expert is not local)
    perm2exp_ptr,  # out [max_permuted] int32 (valid rows only)
    total_padded_ptr,  # out [1] int32
    num_tiles_ptr,  # out [1] int32
    out_ptr,  # out [M, H] bf16 MoE output, zeroed here
    scratch_ptr,  # [2 * NLP] int32 per-layer scratch
    M,
    local_start,
    E: tl.constexpr,
    K: tl.constexpr,
    KP: tl.constexpr,
    H: tl.constexpr,
    NL: tl.constexpr,
    NLP: tl.constexpr,
    TILE: tl.constexpr,
    BM: tl.constexpr,
    QUANT: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    pid = tl.program_id(0)
    if USE_PDL:
        _gdc_wait_with_memory_clobber()

    if pid == 0:
        # ---------------- routing: top-k + softmax ----------------
        rows = tl.arange(0, BM)
        rmask = rows < M
        ecols = tl.arange(0, E)
        lmask = rmask[:, None] & (ecols[None, :] < E)
        lg = tl.load(logits_ptr + rows[:, None] * E + ecols[None, :],
                     mask=lmask,
                     other=0.0)
        bits = lg.to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
        # cub TwiddleIn for bf16 (negative -> ~bits, else bits | sign), then
        # the reduce_topk::TopKRedType packing: twiddled value in the upper
        # 16 bits, (65535 - idx) in the lower 16 -> the max picks the larger
        # value, then the smaller index. XOR with the sign bit makes the
        # unsigned order a signed int32 order.
        tw = tl.where((bits & 0x8000) != 0, bits ^ 0xFFFF, bits | 0x8000)
        key = ((tw << 16) | (65535 - ecols[None, :])) ^ SIGN_BIT
        key = tl.where(lmask, key, MIN_KEY)

        kcols = tl.arange(0, KP)
        ids = tl.zeros([BM, KP], dtype=tl.int32)
        vals = tl.zeros([BM, KP], dtype=tl.float32)
        for k in tl.static_range(K):
            mx = tl.max(key, axis=1)
            ukey = mx ^ SIGN_BIT
            sel = 65535 - (ukey & 0xFFFF)
            twv = (ukey >> 16) & 0xFFFF
            vb = tl.where((twv & 0x8000) != 0, twv ^ 0x8000, twv ^ 0xFFFF)
            v = vb.to(tl.int16).to(tl.bfloat16, bitcast=True).to(tl.float32)
            ids = tl.where(kcols[None, :] == k, sel[:, None], ids)
            vals = tl.where(kcols[None, :] == k, v[:, None], vals)
            key = tl.where(key == mx[:, None], MIN_KEY, key)

        # calcSoftmax: max over the k values (the first one), expf of the
        # shifted values, xor-butterfly warp sum (masks 16, 8, 4, 2, 1 over
        # lanes 0..31 with lanes >= K holding 0), IEEE division.
        kvalid = kcols[None, :] < K
        vmax = _pick_col(vals, kcols, 0)
        ex = tl.extra.cuda.libdevice.exp(vals - vmax[:, None])
        ex = tl.where(kvalid, ex, 0.0)
        e0 = _pick_col(ex, kcols, 0)
        e1 = _pick_col(ex, kcols, 1)
        e2 = _pick_col(ex, kcols, 2)
        e3 = _pick_col(ex, kcols, 3)
        e4 = _pick_col(ex, kcols, 4)
        e5 = _pick_col(ex, kcols, 5)
        e6 = _pick_col(ex, kcols, 6)
        e7 = _pick_col(ex, kcols, 7)
        e8 = _pick_col(ex, kcols, 8)
        e9 = _pick_col(ex, kcols, 9)
        s0 = ((e0 + e8) + e4) + (e2 + e6)
        s1 = ((e1 + e9) + e5) + (e3 + e7)
        ssum = s0 + s1
        wts = tl.extra.cuda.libdevice.div_rn(ex, ssum[:, None])

        omask = rmask[:, None] & kvalid
        exp_idx = rows[:, None] * K + kcols[None, :]
        tl.store(ids_ptr + exp_idx, ids, mask=omask)
        tl.store(w_ptr + exp_idx, wts, mask=omask)

        # ---------------- sort tables (moe_sort semantics) ----------------
        lid = ids - local_start
        is_local = omask & (lid >= 0) & (lid < NL)
        lid = tl.where(is_local, lid, 0)
        ecnt = tl.arange(0, NLP)
        emask = ecnt < NL
        tl.store(scratch_ptr + ecnt, tl.zeros([NLP], dtype=tl.int32),
                 mask=emask)
        tl.debug_barrier()
        offs = tl.atomic_add(scratch_ptr + lid,
                             tl.full([BM, KP], 1, tl.int32),
                             mask=is_local,
                             sem="relaxed")
        tl.debug_barrier()
        cnt = tl.load(scratch_ptr + ecnt, mask=emask, other=0, volatile=True)
        active = cnt > 0
        act_i = active.to(tl.int32)
        tile_of_e = tl.cumsum(act_i, axis=0) - act_i
        num_tiles = tl.sum(act_i, axis=0)
        tl.store(tile_expert_ptr + tile_of_e, ecnt, mask=active)
        tl.store(tile_mnlimit_ptr + tile_of_e, tile_of_e * TILE + cnt,
                 mask=active)
        tl.store(scratch_ptr + NLP + ecnt, tile_of_e * TILE, mask=active)
        tl.debug_barrier()
        base = tl.load(scratch_ptr + NLP + lid,
                       mask=is_local,
                       other=0,
                       volatile=True)
        perm = base + offs
        tl.store(exp2perm_ptr + exp_idx, tl.where(is_local, perm, -1),
                 mask=omask)
        tl.store(perm2exp_ptr + perm, exp_idx, mask=is_local)
        tl.store(total_padded_ptr, num_tiles * TILE)
        tl.store(num_tiles_ptr, num_tiles)
    else:
        # ------ NVFP4 quantization of one row (unless done upstream) ------
        r = (pid - 1).to(tl.int64)
        if QUANT:
            _nvfp4_quant_row(x_ptr, x_row_stride, gs_ptr, xq_ptr, xsf_ptr, r,
                             H)
        tl.store(out_ptr + r * H + tl.arange(0, H),
                 tl.zeros([H], dtype=tl.bfloat16))

    if USE_PDL:
        tl.debug_barrier()
        tl.extra.cuda.gdc_launch_dependents()


# TRTLLM_Q35_MOE_ROUTER_GEMM: "1"/"triton" = single Triton launch (split-K GEMM + NVFP4 quant CTAs);
# "lowm" = the low-M BF16 GEMM (modules/lowm_linear.py) for the logits, with x quantized by a
# separate Triton launch issued first (the low-M kernel prefetches its weights before griddepcontrol.wait, so it
# overlaps the quant launch).
# TRTLLM_Q35_MOE_EARLY_TRIGGER (default off): the Q35 router launch and / or the per-token route + tables launch release
# their PDL dependents (trtllm-gen routing kernel, route launch, FC1) at kernel entry instead of at exit, so the
# dependent's launch overlaps this kernel; the dependent still reads its inputs only after griddepcontrol.wait.
# 1 / all = both, router = only the router launch, route = only the route launch. (Both is slower than PREROUTE
# alone at TEP16: FC1's early CTAs wait on SMs the aux-stream shared expert needs.)
_Q35_EARLY_MODE = os.environ.get("TRTLLM_Q35_MOE_EARLY_TRIGGER", "0").lower()
Q35_MOE_EARLY_TRIGGER = _Q35_EARLY_MODE in ("1", "all", "router")
Q35_MOE_EARLY_TRIGGER_ROUTE = _Q35_EARLY_MODE in ("1", "all", "route")
_Q35_ROUTER_MODE = os.environ.get("TRTLLM_Q35_MOE_ROUTER_GEMM", "0").lower()
Q35_MOE_ROUTER_GEMM = _Q35_ROUTER_MODE in ("1", "triton", "lowm")
Q35_MOE_ROUTER_LOWM = _Q35_ROUTER_MODE == "lowm"
# TRTLLM_Q35_MOE_GATE_LOWM=1 (with TRTLLM_Q35_MOE_ROUTER_GEMM=1, default off): router calls without the NVFP4 quant
# (the BF16 MTP layer's gate) run on the low-M BF16 GEMM (1.4-1.7 us vs ~4 us at M <= 7). Other logits rounding than
# the Triton router: draft routing near-ties may differ (AL gate).
Q35_MOE_GATE_LOWM = os.environ.get("TRTLLM_Q35_MOE_GATE_LOWM", "0") == "1"


@triton.jit
def _fence_acq_rel_gpu():
    tl.inline_asm_elementwise("fence.acq_rel.gpu; // dummy $0",
                              "=r,~{memory}", [],
                              dtype=tl.int32,
                              is_pure=False,
                              pack=1)


@triton.jit
def _q35_top1(key):
    """One reduceTopK step on packed keys [R, E]: the row maxima's expert ids and
    bf16 values, and the keys with the picked entries knocked out. Everything
    stays in the reduction's row layout (no per-step layout conversions)."""
    mx = tl.max(key, axis=1)
    ukey = mx ^ SIGN_BIT
    sel = 65535 - (ukey & 0xFFFF)
    twv = (ukey >> 16) & 0xFFFF
    vb = tl.where((twv & 0x8000) != 0, twv ^ 0x8000, twv ^ 0xFFFF)
    v = vb.to(tl.int16).to(tl.bfloat16, bitcast=True).to(tl.float32)
    return tl.where(key == mx[:, None], MIN_KEY, key), sel, v


@triton.jit
def _q35_store_pick(ids_ptr, w_ptr, rows, rmask, K: tl.constexpr,
                    j: tl.constexpr, sel, e, ssum):
    tl.store(ids_ptr + rows * K + j, sel, mask=rmask)
    tl.store(w_ptr + rows * K + j,
             tl.extra.cuda.libdevice.div_rn(e, ssum).to(w_ptr.dtype.element_ty),
             mask=rmask)


@triton.jit
def _q35_route_core(logits_ptr, ids_ptr, w_ptr, e2p_ptr, p2t_ptr,
                    cta_expert_ptr, cta_limit_ptr, num_ctas_ptr,
                    num_padded_ptr, M, local_start, n_groups, my_group,
                    E: tl.constexpr, K: tl.constexpr, KP: tl.constexpr,
                    MP: tl.constexpr, RB: tl.constexpr, PE: tl.constexpr,
                    TILE: tl.constexpr, NL: tl.constexpr,
                    NLP: tl.constexpr, MAX_CTAS_PER_EXPERT: tl.constexpr,
                    DEA: tl.constexpr, PHASES: tl.constexpr = 3,
                    SORT: tl.constexpr = True):
    """Routing of all M (<= MP) tokens in ONE program, no atomics (RB = rows
    per top-K block = min(8, MP), PE = MP * KP).

    Phase 1, blocks of RB rows: top-K of each row of the [M, E] bf16 logits
    (reduceTopK order: packed twiddled value << 16 | 65535 - index) and the
    calcSoftmax renormalization (expf, butterfly sum, IEEE division) -> ids,
    weights. Phase 2, over the M * K expanded picks: trtllm-gen's routing
    tables for TILE over the local experts [local_start, local_start + NL):
    rows of an expert in expanded-index order (stable sort), CTAs in expert
    order, -1 for picks outside. DEA: NL = E, local_start = 0; the step's
    distinct experts are dealt round-robin in id order over n_groups and only
    my_group's are kept. NLP > NL (bin NL collects the other picks)."""
    cols = tl.arange(0, E)
    slots = tl.arange(0, KP)
    kmask = slots[None, :] < K
    # PHASES (bench diagnostics only): 1 = top-K, 2 = tables from the ids in ids_ptr, 3 = both
    for rb in range(0, M if PHASES != 2 else 0, RB):
        rows = rb + tl.arange(0, RB)
        rmask = rows < M
        lg = tl.load(logits_ptr + rows[:, None] * E + cols[None, :],
                     mask=rmask[:, None],
                     other=0.0,
                     cache_modifier=".cg")
        bits = lg.to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
        tw = tl.where((bits & 0x8000) != 0, bits ^ 0xFFFF, bits | 0x8000)
        key = ((tw << 16) | (65535 - cols[None, :])) ^ SIGN_BIT
        if K == 10:
            # Unrolled by hand: the ten picks stay [RB] tensors in the row layout of the
            # reduction (a [RB, KP] accumulator costs a layout conversion per pick); the
            # row indices are taken through the same reduction to share that layout.
            rows_r = tl.max(rows[:, None] + tl.zeros([RB, E], dtype=tl.int32), axis=1)
            rmask_r = rows_r < M
            key, s0, v0 = _q35_top1(key)
            key, s1, v1 = _q35_top1(key)
            key, s2, v2 = _q35_top1(key)
            key, s3, v3 = _q35_top1(key)
            key, s4, v4 = _q35_top1(key)
            key, s5, v5 = _q35_top1(key)
            key, s6, v6 = _q35_top1(key)
            key, s7, v7 = _q35_top1(key)
            key, s8, v8 = _q35_top1(key)
            key, s9, v9 = _q35_top1(key)
            vmax = tl.maximum(
                tl.maximum(tl.maximum(tl.maximum(v0, v1), tl.maximum(v2, v3)),
                           tl.maximum(tl.maximum(v4, v5), tl.maximum(v6, v7))),
                tl.maximum(v8, v9))
            e0 = tl.extra.cuda.libdevice.exp(v0 - vmax)
            e1 = tl.extra.cuda.libdevice.exp(v1 - vmax)
            e2 = tl.extra.cuda.libdevice.exp(v2 - vmax)
            e3 = tl.extra.cuda.libdevice.exp(v3 - vmax)
            e4 = tl.extra.cuda.libdevice.exp(v4 - vmax)
            e5 = tl.extra.cuda.libdevice.exp(v5 - vmax)
            e6 = tl.extra.cuda.libdevice.exp(v6 - vmax)
            e7 = tl.extra.cuda.libdevice.exp(v7 - vmax)
            e8 = tl.extra.cuda.libdevice.exp(v8 - vmax)
            e9 = tl.extra.cuda.libdevice.exp(v9 - vmax)
            ssum = (((e0 + e8) + e4) + (e2 + e6)) + (((e1 + e9) + e5) +
                                                    (e3 + e7))
            _q35_store_pick(ids_ptr, w_ptr, rows_r, rmask_r, K, 0, s0, e0, ssum)
            _q35_store_pick(ids_ptr, w_ptr, rows_r, rmask_r, K, 1, s1, e1, ssum)
            _q35_store_pick(ids_ptr, w_ptr, rows_r, rmask_r, K, 2, s2, e2, ssum)
            _q35_store_pick(ids_ptr, w_ptr, rows_r, rmask_r, K, 3, s3, e3, ssum)
            _q35_store_pick(ids_ptr, w_ptr, rows_r, rmask_r, K, 4, s4, e4, ssum)
            _q35_store_pick(ids_ptr, w_ptr, rows_r, rmask_r, K, 5, s5, e5, ssum)
            _q35_store_pick(ids_ptr, w_ptr, rows_r, rmask_r, K, 6, s6, e6, ssum)
            _q35_store_pick(ids_ptr, w_ptr, rows_r, rmask_r, K, 7, s7, e7, ssum)
            _q35_store_pick(ids_ptr, w_ptr, rows_r, rmask_r, K, 8, s8, e8, ssum)
            _q35_store_pick(ids_ptr, w_ptr, rows_r, rmask_r, K, 9, s9, e9, ssum)
        else:
            picked = tl.zeros([RB, KP], dtype=tl.int32)
            vals = tl.zeros([RB, KP], dtype=tl.float32)
            for k in tl.static_range(K):
                key, sel, v = _q35_top1(key)
                picked = tl.where(slots[None, :] == k, sel[:, None], picked)
                vals = tl.where(slots[None, :] == k, v[:, None], vals)
            vmax = tl.max(tl.where(kmask, vals, float("-inf")), axis=1)
            ex = tl.where(kmask,
                          tl.extra.cuda.libdevice.exp(vals - vmax[:, None]), 0.0)
            e0 = tl.sum(tl.where(slots[None, :] == 0, ex, 0.0), axis=1)
            e1 = tl.sum(tl.where(slots[None, :] == 1, ex, 0.0), axis=1)
            e2 = tl.sum(tl.where(slots[None, :] == 2, ex, 0.0), axis=1)
            e3 = tl.sum(tl.where(slots[None, :] == 3, ex, 0.0), axis=1)
            e4 = tl.sum(tl.where(slots[None, :] == 4, ex, 0.0), axis=1)
            e5 = tl.sum(tl.where(slots[None, :] == 5, ex, 0.0), axis=1)
            e6 = tl.sum(tl.where(slots[None, :] == 6, ex, 0.0), axis=1)
            e7 = tl.sum(tl.where(slots[None, :] == 7, ex, 0.0), axis=1)
            e8 = tl.sum(tl.where(slots[None, :] == 8, ex, 0.0), axis=1)
            e9 = tl.sum(tl.where(slots[None, :] == 9, ex, 0.0), axis=1)
            ssum = (((e0 + e8) + e4) + (e2 + e6)) + (((e1 + e9) + e5) +
                                                    (e3 + e7))
            wts = tl.extra.cuda.libdevice.div_rn(ex, ssum[:, None])
            m2 = rmask[:, None] & kmask
            tl.store(ids_ptr + rows[:, None] * K + slots[None, :], picked,
                     mask=m2)
            tl.store(w_ptr + rows[:, None] * K + slots[None, :],
                     wts.to(w_ptr.dtype.element_ty),
                     mask=m2)
    tl.debug_barrier()  # phase 2 reads the ids other threads of this program wrote
    if PHASES != 1:
        _q35_route_tables_phase(ids_ptr, e2p_ptr, p2t_ptr, cta_expert_ptr, cta_limit_ptr, num_ctas_ptr,
                                num_padded_ptr, M, local_start, n_groups, my_group, K, PE, TILE, NL, NLP,
                                MAX_CTAS_PER_EXPERT, DEA, SORT)


@triton.jit
def _q35_route_tables_phase(ids_ptr, e2p_ptr, p2t_ptr, cta_expert_ptr, cta_limit_ptr, num_ctas_ptr,
                            num_padded_ptr, M, local_start, n_groups, my_group, K: tl.constexpr,
                            PE: tl.constexpr, TILE: tl.constexpr, NL: tl.constexpr, NLP: tl.constexpr,
                            MAX_CTAS_PER_EXPERT: tl.constexpr, DEA: tl.constexpr, SORT: tl.constexpr = True):
    """Phase 2 of _q35_route_core: the routing tables over the M * K expanded picks.
    SORT=False: each pick's row inside its expert block (expanded-index order, as the stable sort gives) is
    counted directly over chunks of the other picks instead of tl.sort; same tables."""
    ei = tl.arange(0, PE)
    valid = ei < M * K
    gid = tl.load(ids_ptr + ei, mask=valid, other=0, cache_modifier=".cg")
    lid = gid - local_start
    local = valid & (lid >= 0) & (lid < NL)
    count = tl.histogram(tl.where(local, lid, NL), NLP)
    e = tl.arange(0, NLP)
    expert = e < NL
    count = tl.where(expert, count, 0)
    if DEA:
        present = (count > 0).to(tl.int32)
        ordinal = tl.cumsum(present, axis=0) - present
        count = tl.where((present > 0) & ((ordinal % n_groups) == my_group),
                         count, 0)
        local = local & (tl.gather(count, tl.where(local, lid, 0), axis=0) > 0)
    num_cta = (count + TILE - 1) // TILE
    offs = tl.cumsum(num_cta, axis=0) - num_cta
    if SORT:
        row_start = tl.cumsum(count, axis=0) - count
        # stable order inside each expert: sort (local expert, expanded index)
        sk = tl.sort(tl.where(local, (lid << 12) | ei, (NLP << 12) | ei))
        s_lid = sk >> 12
        s_ei = sk & 4095
        s_local = s_lid < NL
        s_lid0 = tl.where(s_local, s_lid, 0)
        s_perm = (tl.gather(offs, s_lid0, axis=0) * TILE + ei -
                  tl.gather(row_start, s_lid0, axis=0))
        tl.store(e2p_ptr + s_ei, s_perm, mask=s_local)
        tl.store(e2p_ptr + ei, tl.full([PE], -1, tl.int32),
                 mask=valid & (local == 0))
        tl.store(p2t_ptr + s_perm, s_ei // K, mask=s_local)
    else:
        CH: tl.constexpr = 32 if PE >= 32 else PE
        lid_l = tl.where(local, lid, -1)
        rank = tl.zeros([PE], dtype=tl.int32)
        for c0 in tl.static_range(0, PE, CH):
            cj = c0 + tl.arange(0, CH)
            vj = cj < M * K
            lj = tl.load(ids_ptr + cj, mask=vj, other=0, cache_modifier=".cg") - local_start
            lj = tl.where(vj & (lj >= 0) & (lj < NL), lj, -2)
            same = (lid_l[:, None] == lj[None, :]) & (cj[None, :] < ei[:, None])
            rank += tl.sum(same.to(tl.int32), axis=1)
        perm = tl.gather(offs, tl.where(local, lid, 0), axis=0) * TILE + rank
        tl.store(e2p_ptr + ei, tl.where(local, perm, -1), mask=valid)
        tl.store(p2t_ptr + perm, ei // K, mask=local)
    for c in tl.static_range(MAX_CTAS_PER_EXPERT):
        has = expert & (c < num_cta)
        cta = offs + c
        tl.store(cta_expert_ptr + cta, e, mask=has)
        tl.store(cta_limit_ptr + cta,
                 tl.minimum((cta + 1) * TILE, offs * TILE + count),
                 mask=has)
    total = tl.sum(num_cta, axis=0)
    tl.store(num_ctas_ptr, total)
    tl.store(num_padded_ptr, total * TILE)


@triton.jit
def _q35_route_cta_kernel(logits_ptr, ids_ptr, w_ptr, e2p_ptr, p2t_ptr,
                          cta_expert_ptr, cta_limit_ptr, num_ctas_ptr,
                          num_padded_ptr, M, local_start, n_groups, my_group,
                          E: tl.constexpr, K: tl.constexpr, KP: tl.constexpr,
                          MP: tl.constexpr, RB: tl.constexpr, PE: tl.constexpr,
                          TILE: tl.constexpr,
                          NL: tl.constexpr, NLP: tl.constexpr,
                          MAX_CTAS_PER_EXPERT: tl.constexpr, DEA: tl.constexpr,
                          USE_PDL: tl.constexpr, PHASES: tl.constexpr = 3,
                          SORT: tl.constexpr = True):
    """Standalone single-program route + tables (PDL dependent of the router)."""
    if USE_PDL:
        _gdc_wait_with_memory_clobber()
    _q35_route_core(logits_ptr, ids_ptr, w_ptr, e2p_ptr, p2t_ptr,
                    cta_expert_ptr, cta_limit_ptr, num_ctas_ptr,
                    num_padded_ptr, M, local_start, n_groups, my_group, E, K,
                    KP, MP, RB, PE, TILE, NL, NLP, MAX_CTAS_PER_EXPERT, DEA,
                    PHASES, SORT)
    if USE_PDL:
        tl.debug_barrier()
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def _q35_router_gemm_kernel(
    x_ptr,  # [M, K] bf16 activations
    x_row_stride,
    w_ptr,  # [N, K] bf16 router weight (row-major, as nn.Linear)
    out_ptr,  # [M, N] bf16 logits
    part_ptr,  # [SPLITK, BM, N] fp32 split-K partials (scratch)
    cnt_ptr,  # [N // BN] int32 arrival counters (zero between launches)
    gs_ptr,  # [1] fp32 NVFP4 global scale (QUANT only)
    xq_ptr,  # out [M, K // 2] uint8 (QUANT only)
    xsf_ptr,  # out [M, K // 16] uint8 linear scales (QUANT only)
    rcnt_ptr,  # [1] int32 N-block arrival ticket (ROUTE only; zero between launches)
    ids_ptr,  # out [M, TOPK] int32 (ROUTE only)
    wts_ptr,  # out [M, TOPK] bf16 (ROUTE only)
    e2p_ptr,  # out routing tables (ROUTE only, see _q35_route_core)
    p2t_ptr,
    cta_expert_ptr,
    cta_limit_ptr,
    num_ctas_ptr,
    num_padded_ptr,
    hist_ptr,  # [NLP] int32 per-local-expert counts (ROUTE == 2; zero between launches)
    rank_ptr,  # [M * TOPK] int32 scratch (ROUTE == 2)
    counter_ptr,  # [1] int32 token-program ticket (ROUTE == 2; zero between launches)
    M,
    local_start,
    n_groups,
    my_group,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    SPLITK: tl.constexpr,
    QUANT: tl.constexpr,
    ROUTE: tl.constexpr,
    TOPK: tl.constexpr,
    KP: tl.constexpr,
    MP: tl.constexpr,
    RB: tl.constexpr,
    PE: tl.constexpr,
    TILE: tl.constexpr,
    NL: tl.constexpr,
    NLP: tl.constexpr,
    MAX_CTAS_PER_EXPERT: tl.constexpr,
    DEA: tl.constexpr,
    EXPANDED_PAD: tl.constexpr,
    USE_PDL: tl.constexpr,
    EARLY_TRIGGER: tl.constexpr = False,
):
    """logits = x @ w^T, one launch: (N/BN) x SPLITK CTAs, each a [BM, BN]
    partial over K/SPLITK; the last CTA of each N block sums the partials in
    split order (deterministic) and writes bf16. The weight tile does not
    depend on the producer, so it is loaded before griddepcontrol.wait.
    With QUANT, M extra CTAs NVFP4-quantize the activation rows. ROUTE (the MoE
    op then needs no routing launch at all, TRTLLM_Q35_MOE_PREROUTE):
      1: the last N block to finish routes all tokens and writes the trtllm-gen
         routing tables in one program (_q35_route_core);
      2: each quant program, after its row, waits until every N block has
         written its logits and then routes its token as the per-token route
         kernel does (_q35_route_token: histogram atomics, and the last token
         program writes the tables). Needs QUANT; the waiting programs spin on
         the N-block ticket (at most 32 of them, next to 8 x N/BN GEMM CTAs)."""
    KS: tl.constexpr = K // SPLITK
    NB: tl.constexpr = N // BN
    if USE_PDL and EARLY_TRIGGER:
        # TRTLLM_Q35_MOE_EARLY_TRIGGER: release the dependent (routing kernel / route launch / FC1) at entry;
        # it reads this launch's outputs only after griddepcontrol.wait (= this grid's completion).
        tl.extra.cuda.gdc_launch_dependents()
    pid = tl.program_id(0)
    if pid >= NB * SPLITK:
        if QUANT:
            if USE_PDL:
                _gdc_wait_with_memory_clobber()
            _nvfp4_quant_row(x_ptr, x_row_stride, gs_ptr, xq_ptr, xsf_ptr,
                             (pid - NB * SPLITK).to(tl.int64), K)
            if ROUTE == 2:
                # acquire: every N block's logits (released before its ticket)
                done = tl.atomic_add(rcnt_ptr, 0, sem="acquire", scope="gpu")
                while done < NB:
                    done = tl.atomic_add(rcnt_ptr, 0, sem="acquire", scope="gpu")
                _fence_acq_rel_gpu()
                _q35_route_token(pid - NB * SPLITK, out_ptr, ids_ptr, wts_ptr,
                                 hist_ptr, rank_ptr, counter_ptr, e2p_ptr,
                                 p2t_ptr, cta_expert_ptr, cta_limit_ptr,
                                 num_ctas_ptr, num_padded_ptr, rcnt_ptr, M,
                                 local_start, n_groups, my_group, N, TOPK, KP,
                                 TILE, NL, NLP, EXPANDED_PAD,
                                 MAX_CTAS_PER_EXPERT, DEA, True, True)
    else:
        pid_n = pid % NB
        pid_k = pid // NB
        rn = pid_n * BN + tl.arange(0, BN)
        rk = pid_k * KS + tl.arange(0, KS)
        rm = tl.arange(0, BM)
        w = tl.load(w_ptr + rn[:, None] * K + rk[None, :])  # [BN, KS]
        if USE_PDL:
            _gdc_wait_with_memory_clobber()
        xt = tl.load(x_ptr + rm[:, None] * x_row_stride + rk[None, :],
                     mask=(rm < M)[:, None],
                     other=0.0)  # [BM, KS]
        acc = tl.dot(xt, tl.trans(w), out_dtype=tl.float32)  # [BM, BN]
        tl.store(part_ptr + (pid_k * BM + rm[:, None]) * N + rn[None, :],
                 acc)
        _fence_acq_rel_gpu()
        tl.debug_barrier()
        prev = tl.atomic_add(cnt_ptr + pid_n, 1, sem="acq_rel", scope="gpu")
        if prev == SPLITK - 1:
            _fence_acq_rel_gpu()
            tot = tl.zeros([BM, BN], dtype=tl.float32)
            for s in tl.static_range(SPLITK):
                tot += tl.load(part_ptr + (s * BM + rm[:, None]) * N +
                               rn[None, :],
                               volatile=True)
            tl.store(out_ptr + rm[:, None] * N + rn[None, :],
                     tot.to(tl.bfloat16),
                     mask=(rm < M)[:, None])
            tl.atomic_xchg(cnt_ptr + pid_n, 0, sem="relaxed", scope="gpu")
            if ROUTE == 2:
                # release this N block's logits to the token programs (the last of them
                # zeroes the ticket once every one has seen NB)
                _fence_acq_rel_gpu()
                tl.debug_barrier()
                tl.atomic_add(rcnt_ptr, 1, sem="release", scope="gpu")
            if ROUTE == 1:
                _fence_acq_rel_gpu()
                tl.debug_barrier()
                done = tl.atomic_add(rcnt_ptr, 1, sem="acq_rel", scope="gpu")
                if done == NB - 1:
                    _fence_acq_rel_gpu()
                    _q35_route_core(out_ptr, ids_ptr, wts_ptr, e2p_ptr,
                                    p2t_ptr, cta_expert_ptr, cta_limit_ptr,
                                    num_ctas_ptr, num_padded_ptr, M,
                                    local_start, n_groups, my_group, N, TOPK,
                                    KP, MP, RB, PE, TILE, NL, NLP,
                                    MAX_CTAS_PER_EXPERT, DEA)
                    tl.atomic_xchg(rcnt_ptr, 0, sem="relaxed", scope="gpu")
    if USE_PDL and not EARLY_TRIGGER:
        tl.debug_barrier()
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def _q35_quant_rows_kernel(x_ptr, x_row_stride, gs_ptr, xq_ptr, xsf_ptr,
                           H: tl.constexpr, USE_PDL: tl.constexpr):
    if USE_PDL:
        _gdc_wait_with_memory_clobber()
    _nvfp4_quant_row(x_ptr, x_row_stride, gs_ptr, xq_ptr, xsf_ptr,
                     tl.program_id(0).to(tl.int64), H)
    if USE_PDL:
        tl.debug_barrier()
        tl.extra.cuda.gdc_launch_dependents()


def q35_nvfp4_quant(x: torch.Tensor, global_scale: torch.Tensor):
    """Standalone NVFP4 quantization of x [M, H] bf16 (linear scales), byte-
    identical to trtllm::fp4_quantize(x, gs, 16, False, False)."""
    m, h = x.shape
    assert x.dtype == torch.bfloat16 and x.stride(1) == 1 and (h & (h - 1)) == 0
    xq = torch.empty((m, h // 2), dtype=torch.uint8, device=x.device)
    xsf = torch.empty((m, h // 16), dtype=torch.uint8, device=x.device)
    use_pdl = _pdl_enabled()
    _q35_quant_rows_kernel[(m, )](x,
                                  x.stride(0),
                                  global_scale,
                                  xq,
                                  xsf,
                                  H=h,
                                  USE_PDL=use_pdl,
                                  num_warps=8,
                                  launch_pdl=use_pdl)
    return xq, xsf


def q35_lowm_router(x: torch.Tensor,
                    weight: torch.Tensor,
                    global_scale: Optional[torch.Tensor] = None):
    """Router logits on the low-M BF16 GEMM (modules/lowm_linear.py), NVFP4 quant first.
    Returns None when the low-M kernel does not take the shape."""
    from ..lowm_linear import lowm_bf16_gemm
    m = x.shape[0]
    tactic = None if m <= 16 else (16, 1)
    q = q35_nvfp4_quant(x, global_scale) if global_scale is not None else None
    logits = lowm_bf16_gemm(x, weight, out_dtype=torch.bfloat16, tactic=tactic)
    if logits is None:
        return None
    if q is not None:
        return logits, q[0], q[1]
    return logits


class Q35RouterGemm:
    """Per-module scratch + launcher for the single-launch router GEMM."""

    BN = 32
    SPLITK = 8

    def __init__(self):
        self.part = None
        self.cnt = None
        self.rcnt = None
        self.hist = None
        self.counter = None

    def __call__(self,
                 x: torch.Tensor,
                 weight: torch.Tensor,
                 global_scale: Optional[torch.Tensor] = None,
                 route: Optional[tuple] = None,
                 route_mode: Optional[int] = None):
        """logits [M, N] bf16; with ``global_scale`` also returns the NVFP4
        quantization of x: (logits, xq [M, K/2] u8, xsf [M, K/16] u8). With
        ``route`` = (top_k, local_start, num_local, tile, flat tables,
        (groups, group) or None) the same launch also routes the tokens and
        fills the tables (q35_prerouted_views layout) and the call returns
        (logits, xq, xsf, ids, bf16 weights). route_mode (default
        TRTLLM_Q35_MOE_PREROUTE_FOLD): 1 = one program routes all tokens,
        2 = one program per token (see _q35_router_gemm_kernel)."""
        m, k = x.shape
        n = weight.shape[0]
        assert x.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16
        assert weight.is_contiguous() and x.stride(1) == 1
        assert m <= Q35_MOE_FUSED_FRONT_MAX_TOKENS
        assert n % self.BN == 0 and k % (self.SPLITK * 16) == 0
        bm = 16 if m <= 16 else 32
        dev = x.device
        if self.part is None or self.part.device != dev:
            self.part = torch.empty((self.SPLITK, 32, n),
                                    dtype=torch.float32,
                                    device=dev)
            self.cnt = torch.zeros((n // self.BN, ),
                                   dtype=torch.int32,
                                   device=dev)
        out = torch.empty((m, n), dtype=torch.bfloat16, device=dev)
        quant = global_scale is not None
        if quant:
            assert (k & (k - 1)) == 0
            xq = torch.empty((m, k // 2), dtype=torch.uint8, device=dev)
            xsf = torch.empty((m, k // 16), dtype=torch.uint8, device=dev)
        else:
            xq = xsf = out
            global_scale = self.cnt
        use_pdl = _pdl_enabled()
        grid = (n // self.BN) * self.SPLITK + (m if quant else 0)
        mode = 0
        if route is not None:
            mode = route_mode if route_mode is not None else (Q35_MOE_PREROUTE_FOLD_MODE or 1)
            assert mode in (1, 2)
            top_k, local_start, num_local, tile, flat, dea = route
            assert quant and top_k <= 10 and (n & (n - 1)) == 0
            if self.rcnt is None or self.rcnt.device != dev:
                self.rcnt = torch.zeros((1, ), dtype=torch.int32, device=dev)
            ids = torch.empty((m, top_k), dtype=torch.int32, device=dev)
            wts = torch.empty((m, top_k), dtype=torch.bfloat16, device=dev)
            tables = q35_prerouted_views(flat, m, top_k, num_local, tile)
            mp = max(triton.next_power_of_2(m), 2)
            if mode == 2:
                nlp = triton.next_power_of_2(num_local)
                if self.hist is None or self.hist.numel() < nlp or self.hist.device != dev:
                    self.hist = torch.zeros((nlp, ), dtype=torch.int32, device=dev)
                    self.counter = torch.zeros((1, ), dtype=torch.int32, device=dev)
                rank = torch.empty((m * top_k, ), dtype=torch.int32, device=dev)
                scratch = (self.hist, rank, self.counter)
            else:
                nlp = triton.next_power_of_2(num_local + 1)
                scratch = (self.cnt, ) * 3
            rkw = dict(TOPK=top_k,
                       KP=16,
                       MP=mp,
                       RB=min(8, mp),
                       PE=mp * 16,
                       TILE=tile,
                       NL=num_local,
                       NLP=nlp,
                       MAX_CTAS_PER_EXPERT=triton.cdiv(
                           Q35_MOE_FUSED_FRONT_MAX_TOKENS, tile),
                       DEA=dea is not None,
                       EXPANDED_PAD=triton.next_power_of_2(
                           Q35_MOE_FUSED_FRONT_MAX_TOKENS * top_k))
            rargs = (self.rcnt, ids, wts, *tables, *scratch, m, local_start,
                     dea[0] if dea else 1, dea[1] if dea else 0)
        else:
            rkw = dict(TOPK=1, KP=16, MP=2, RB=2, PE=32, TILE=8, NL=1, NLP=2,
                       MAX_CTAS_PER_EXPERT=1, DEA=False, EXPANDED_PAD=16)
            rargs = (self.cnt, ) * 12 + (m, 0, 1, 0)
        _q35_router_gemm_kernel[(grid, )](
            x,
            x.stride(0),
            weight,
            out,
            self.part,
            self.cnt,
            global_scale,
            xq,
            xsf,
            *rargs,
            N=n,
            K=k,
            BM=bm,
            BN=self.BN,
            SPLITK=self.SPLITK,
            QUANT=quant,
            ROUTE=mode,
            **rkw,
            USE_PDL=use_pdl,
            # only the NVFP4 path (router + quant): its dependents are the trtllm-gen routing kernel, the
            # route launch or FC1, which all wait; the plain router (MTP gate) keeps the late trigger
            EARLY_TRIGGER=Q35_MOE_EARLY_TRIGGER and quant,
            num_warps=4,
            launch_pdl=use_pdl,
        )
        if route is not None:
            return out, xq, xsf, ids, wts
        if quant:
            return out, xq, xsf
        return out


# TRTLLM_Q35_MOE_PREROUTE=1 (default off; with TRTLLM_Q35_MOE_ROUTER_GEMM, TRTLLM-gen NVFP4 MoE, and a
# libth_common that has FP4BlockScaleMoERunner.run_moe_prerouted): one small Triton launch after the router
# turns the logits into the top-k ids / weights and trtllm-gen's routing tables for this rank's experts, and
# the MoE op skips its routing kernel (routingIndicesCluster) whenever its tactic uses the tables' tile.
# TRTLLM_Q35_MOE_PREROUTE_TILE pins the tile (default: 8, then the tile of the tactic the op reports for the
# shape, learned in the eager warmup before CUDA graph capture).
Q35_MOE_PREROUTE = os.environ.get("TRTLLM_Q35_MOE_PREROUTE", "0") == "1"
Q35_MOE_PREROUTE_TILE = int(os.environ.get("TRTLLM_Q35_MOE_PREROUTE_TILE", "0"))
# TRTLLM_Q35_MOE_PREROUTE_FOLD: route + tables in the router launch instead of a separate route + tables launch.
#   1: one program routes all tokens (single-program core; slower in a unit microbenchmark: router + quant
#      4.1 us -> 15.6 us with the fold at m=7, 43.8 us at m=28).
#   2: one program per token (the M quant programs, after their row, wait for every N block's logits, then do the
#      per-token route kernel's work; the last token program writes the tables).
Q35_MOE_PREROUTE_FOLD_MODE = int(os.environ.get("TRTLLM_Q35_MOE_PREROUTE_FOLD", "0") or 0)
# TRTLLM_Q35_MOE_PREROUTE_LOWM=1 (with PREROUTE, default off): the router is the plain BF16 GEMM on the low-M
# kernel (1.4-3.4 us vs 4.0-4.8 us for the Triton router + fused quant, bench), and the NVFP4 quantization of x
# moves into the route + tables launch (M extra programs). Same launch count; the logits come from another GEMM
# (within 1 bf16 ulp of cuBLAS like the Triton router's), so top-k ties can differ: validate accuracy and
# acceptance length.
Q35_MOE_PREROUTE_FOLD = Q35_MOE_PREROUTE_FOLD_MODE in (1, 2)
Q35_MOE_PREROUTE_LOWM = os.environ.get("TRTLLM_Q35_MOE_PREROUTE_LOWM", "0") == "1"
# Warps per program of the per-token route + tables kernel: 4 (microbenchmark: DEA tables 11.2-12.2 -> 5.2-5.8 us,
# EP 4.9-6.4 -> 3.9-5.1 us, vs trtllm-gen routing 3.3-5.1 us; 1 warp leaves the last program's table phase serial).
_Q35_ROUTE_WARPS = int(os.environ.get("TRTLLM_Q35_ROUTE_WARPS", "4"))
# TRTLLM_Q35_MOE_DEA=1 (default off; needs the pre-routed path above): dynamic expert assignment. Every EP rank
# holds its MoE-TP shard of ALL experts (full, data-independent replication); per decode step the route + tables
# launch deals the step's distinct experts round-robin over the EP groups and each rank runs the all-local MoE
# problem over the experts dealt to it, so the busiest rank reads ceil(D / G) experts' weights instead of those of
# the busiest contiguous block. Everything else (prefill, autotuning, M > 32, no tuned tactic of the tables' tile)
# runs contiguous EP on the rank's home block of the replicated weights, exactly as without the flag.
Q35_MOE_DEA = os.environ.get("TRTLLM_Q35_MOE_DEA", "0") == "1"


@triton.jit
def _q35_route_tables_kernel(
    logits_ptr,  # [M, E] bf16 router logits
    ids_ptr,  # out [M, K] int32 global expert ids
    w_ptr,  # out [M, K] renormalized weights (dtype of the buffer, bf16 for trtllm-gen)
    hist_ptr,  # [NLP] int32 per-local-expert counts (zero between launches)
    rank_ptr,  # [M * K] int32 scratch: row inside the expert block
    counter_ptr,  # [1] int32 arrival ticket (zero between launches)
    e2p_ptr,  # out [M * K] int32 expanded -> permuted (-1: expert on another rank)
    p2t_ptr,  # out [>= max padded] int32 permuted -> token
    cta_expert_ptr,  # out [>= max CTAs] int32 local expert of each CTA
    cta_limit_ptr,  # out [>= max CTAs] int32 row limit of each CTA
    num_ctas_ptr,  # out [1] int32
    num_padded_ptr,  # out [1] int32
    M,
    local_start,
    n_groups,
    my_group,
    x_ptr,  # [M, H] bf16 MoE input (QUANT_H > 0)
    x_row_stride,
    gs_ptr,  # [1] fp32 NVFP4 global scale (QUANT_H > 0)
    xq_ptr,  # out [M, H / 2] uint8 (QUANT_H > 0)
    xsf_ptr,  # out [M, H / 16] uint8 linear scales (QUANT_H > 0)
    E: tl.constexpr,
    K: tl.constexpr,
    KP: tl.constexpr,
    TILE: tl.constexpr,
    NL: tl.constexpr,
    NLP: tl.constexpr,
    EXPANDED_PAD: tl.constexpr,
    MAX_CTAS_PER_EXPERT: tl.constexpr,
    DEA: tl.constexpr,
    USE_PDL: tl.constexpr,
    EARLY_TRIGGER: tl.constexpr = False,
    QUANT_H: tl.constexpr = 0,
):
    """One program per token: top-k of its router logits (reduceTopK order),
    the calcSoftmax renormalization, and the token's rows in the local experts'
    blocks (histogram atomics). The last token program to arrive lays out
    trtllm-gen's routing tables for TILE (what its routing kernel writes for
    precomputed ids), so the MoE op can skip its routing launch.

    DEA (dynamic expert assignment over replicated experts; NL = E, local_start
    = 0): the last program deals this step's distinct experts round-robin in id
    order over n_groups groups and keeps only the experts of my_group in the
    tables (picks of other groups' experts map to -1). Every rank computes the
    same assignment from the same ids, and each pick belongs to one group.

    QUANT_H > 0 (TRTLLM_Q35_MOE_PREROUTE_LOWM: the router is a plain GEMM): M more
    programs NVFP4-quantize the H-wide rows of x for the experts, beside the
    routing programs (they take no ticket)."""
    if USE_PDL and EARLY_TRIGGER:
        tl.extra.cuda.gdc_launch_dependents()
    if USE_PDL:
        _gdc_wait_with_memory_clobber()
    pid = tl.program_id(0)
    if QUANT_H > 0 and pid >= M:
        _nvfp4_quant_row(x_ptr, x_row_stride, gs_ptr, xq_ptr, xsf_ptr,
                         (pid - M).to(tl.int64), QUANT_H)
    else:
        _q35_route_token(pid, logits_ptr, ids_ptr, w_ptr, hist_ptr, rank_ptr,
                         counter_ptr, e2p_ptr, p2t_ptr, cta_expert_ptr,
                         cta_limit_ptr, num_ctas_ptr, num_padded_ptr,
                         counter_ptr, M, local_start, n_groups, my_group, E, K,
                         KP, TILE, NL, NLP, EXPANDED_PAD, MAX_CTAS_PER_EXPERT,
                         DEA, False, False)
    if USE_PDL and not EARLY_TRIGGER:
        tl.debug_barrier()
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def _q35_route_token(row, logits_ptr, ids_ptr, w_ptr, hist_ptr, rank_ptr,
                     counter_ptr, e2p_ptr, p2t_ptr, cta_expert_ptr,
                     cta_limit_ptr, num_ctas_ptr, num_padded_ptr, reset_ptr,
                     M, local_start, n_groups, my_group, E: tl.constexpr,
                     K: tl.constexpr, KP: tl.constexpr, TILE: tl.constexpr,
                     NL: tl.constexpr, NLP: tl.constexpr,
                     EXPANDED_PAD: tl.constexpr,
                     MAX_CTAS_PER_EXPERT: tl.constexpr, DEA: tl.constexpr,
                     LOGITS_L2: tl.constexpr, RESET: tl.constexpr):
    """The per-token route + tables work of one program (see
    _q35_route_tables_kernel). LOGITS_L2: the logits were written by other
    programs of this launch (L2 loads after the caller's acquire). RESET: the
    last program also zeroes reset_ptr (the router launch's N-block ticket)."""
    ecols = tl.arange(0, E)
    if LOGITS_L2:
        lg = tl.load(logits_ptr + row * E + ecols, cache_modifier=".cg")
    else:
        lg = tl.load(logits_ptr + row * E + ecols)
    bits = lg.to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
    tw = tl.where((bits & 0x8000) != 0, bits ^ 0xFFFF, bits | 0x8000)
    key = ((tw << 16) | (65535 - ecols)) ^ SIGN_BIT
    slots = tl.arange(0, KP)
    picked = tl.zeros([KP], dtype=tl.int32)
    vals = tl.zeros([KP], dtype=tl.float32)
    for k in tl.static_range(K):
        mx = tl.max(key, axis=0)
        ukey = mx ^ SIGN_BIT
        sel = 65535 - (ukey & 0xFFFF)
        twv = (ukey >> 16) & 0xFFFF
        vb = tl.where((twv & 0x8000) != 0, twv ^ 0x8000, twv ^ 0xFFFF)
        v = vb.to(tl.int16).to(tl.bfloat16, bitcast=True).to(tl.float32)
        picked = tl.where(slots == k, sel, picked)
        vals = tl.where(slots == k, v, vals)
        key = tl.where(key == mx, MIN_KEY, key)
    kmask = slots < K
    # The local picks take their row in their expert's block first: the atomics'
    # round trip overlaps the softmax.
    lid = picked - local_start
    local = kmask & (lid >= 0) & (lid < NL)
    rank = tl.atomic_add(hist_ptr + tl.where(local, lid, 0),
                         tl.full([KP], 1, tl.int32),
                         mask=local,
                         sem="relaxed",
                         scope="gpu")
    vmax = tl.max(tl.where(kmask, vals, float("-inf")), axis=0)
    ex = tl.where(kmask, tl.extra.cuda.libdevice.exp(vals - vmax), 0.0)
    e0 = tl.sum(tl.where(slots == 0, ex, 0.0), axis=0)
    e1 = tl.sum(tl.where(slots == 1, ex, 0.0), axis=0)
    e2 = tl.sum(tl.where(slots == 2, ex, 0.0), axis=0)
    e3 = tl.sum(tl.where(slots == 3, ex, 0.0), axis=0)
    e4 = tl.sum(tl.where(slots == 4, ex, 0.0), axis=0)
    e5 = tl.sum(tl.where(slots == 5, ex, 0.0), axis=0)
    e6 = tl.sum(tl.where(slots == 6, ex, 0.0), axis=0)
    e7 = tl.sum(tl.where(slots == 7, ex, 0.0), axis=0)
    e8 = tl.sum(tl.where(slots == 8, ex, 0.0), axis=0)
    e9 = tl.sum(tl.where(slots == 9, ex, 0.0), axis=0)
    ssum = (((e0 + e8) + e4) + (e2 + e6)) + (((e1 + e9) + e5) + (e3 + e7))
    wts = tl.extra.cuda.libdevice.div_rn(ex, ssum)
    tl.store(ids_ptr + row * K + slots, picked, mask=kmask)
    tl.store(w_ptr + row * K + slots, wts.to(w_ptr.dtype.element_ty), mask=kmask)
    tl.store(rank_ptr + row * K + slots, rank, mask=kmask)
    _fence_acq_rel_gpu()
    tl.debug_barrier()
    ticket = tl.atomic_add(counter_ptr, 1, sem="acq_rel", scope="gpu")
    if ticket == M - 1:
        _fence_acq_rel_gpu()
        i = tl.arange(0, EXPANDED_PAD)
        valid = i < M * K
        e = tl.arange(0, NLP)
        expert = e < NL
        # L2 loads (.cg) after the acquire: data other programs released before the ticket.
        # (volatile loads here were serialized: 11 us for the 512-bin DEA tables.)
        gids = tl.load(ids_ptr + i, mask=valid, other=0, cache_modifier=".cg")
        grank = tl.load(rank_ptr + i, mask=valid, other=0, cache_modifier=".cg")
        count = tl.load(hist_ptr + e, mask=expert, other=0, cache_modifier=".cg")
        tl.store(hist_ptr + e, tl.zeros([NLP], dtype=tl.int32), mask=expert)
        if DEA:
            present = (count > 0).to(tl.int32)
            ordinal = tl.cumsum(present, axis=0) - present
            count = tl.where((present > 0) & ((ordinal % n_groups) == my_group),
                             count, 0)
        num_cta = (count + TILE - 1) // TILE
        offs = tl.cumsum(num_cta, axis=0) - num_cta
        for c in tl.static_range(MAX_CTAS_PER_EXPERT):
            has = expert & (c < num_cta)
            cta = offs + c
            tl.store(cta_expert_ptr + cta, e, mask=has)
            tl.store(cta_limit_ptr + cta,
                     tl.minimum((cta + 1) * TILE, offs * TILE + count),
                     mask=has)
        total = tl.sum(num_cta, axis=0)
        tl.store(num_ctas_ptr, total)
        tl.store(num_padded_ptr, total * TILE)
        glid = gids - local_start
        glocal = valid & (glid >= 0) & (glid < NL)
        if DEA:
            # only picks of this group's experts (they have >= 1 CTA)
            glocal = glocal & (tl.gather(num_cta, tl.where(glocal, glid, 0),
                                         axis=0) > 0)
        permuted = tl.gather(offs, tl.where(glocal, glid, 0), axis=0) * TILE + grank
        tl.store(e2p_ptr + i, tl.where(glocal, permuted, -1), mask=valid)
        tl.store(p2t_ptr + permuted, i // K, mask=glocal)
        tl.store(counter_ptr, 0)
        if RESET:
            tl.store(reset_ptr, 0)


def trtllm_gen_max_padded(num_tokens: int, top_k: int, num_experts: int,
                          tile: int) -> int:
    """getMaxPermutedPaddedCount (trtllmGen blockScaleMoe runner.h)."""
    return max_num_tiles(num_tokens, top_k, num_experts, tile) * tile


def q35_prerouted_sizes(num_tokens: int, top_k: int, num_local: int,
                        tile: int):
    """int32 sizes of the routing tables packed in one flat buffer, in the order
    run_moe_prerouted takes them: expanded -> permuted index, permuted index ->
    token (one tile of slack past the route map), CTA -> local expert, CTA -> row
    limit, CTA count, padded row count. Sized for an all-local problem of
    num_local experts, as the op runs a pre-routed expert-parallel layer."""
    max_ctas = max_num_tiles(num_tokens, top_k, num_local, tile)
    return [num_tokens * top_k, (max_ctas + 1) * tile, max_ctas, max_ctas, 1, 1]


def q35_prerouted_views(flat: torch.Tensor, num_tokens: int, top_k: int,
                        num_local: int, tile: int):
    sizes = q35_prerouted_sizes(num_tokens, top_k, num_local, tile)
    return list(torch.split(flat[:sum(sizes)], sizes))


# Tile of the tactic the MoE op last ran a table buffer's shape with (by data pointer;
# table buffers live as long as the process).
_Q35_PREROUTED_TILE_CHOSEN = {}


def q35_record_prerouted_tile(flat: torch.Tensor, tile: int) -> None:
    _Q35_PREROUTED_TILE_CHOSEN[flat.data_ptr()] = tile


def q35_prerouted_tile_chosen(flat: torch.Tensor) -> Optional[int]:
    return _Q35_PREROUTED_TILE_CHOSEN.get(flat.data_ptr())


class Q35RouteTables:
    """Per-layer scratch + launcher for the trtllm-gen routing tables built
    from the router logits (TRTLLM_Q35_MOE_PREROUTE)."""

    def __init__(self):
        self.hist = None
        self.counter = None

    def __call__(self,
                 logits: torch.Tensor,
                 top_k: int,
                 local_start: int,
                 num_local: int,
                 tile: int,
                 weight_dtype=torch.bfloat16,
                 flat: Optional[torch.Tensor] = None,
                 dea_groups: int = 0,
                 dea_group: int = 0,
                 single_program: bool = True,
                 phases: int = 3,
                 num_warps: int = 4,
                 rb: Optional[int] = None,
                 sort: bool = True,
                 ids_in: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                 x: Optional[torch.Tensor] = None,
                 global_scale: Optional[torch.Tensor] = None):
        """dea_groups > 0: dynamic expert assignment (all experts local:
        local_start 0, num_local E), tables for group dea_group of dea_groups.
        single_program: one program routes all tokens (_q35_route_cta_kernel,
        no atomics); else one program per token + histogram atomics + ticket.
        phases / num_warps / rb / sort / ids_in: single-program bench
        diagnostics (phases 1 = top-K only, 2 = tables only from ids_in).
        x / global_scale (per-token launch only): the same launch also
        NVFP4-quantizes x [m, H] (bf16) for the experts; the call then returns
        (ids, wts, *tables, xq [m, H/2] u8, xsf [m, H/16] u8)."""
        m, e = logits.shape
        if dea_groups > 0:
            assert local_start == 0 and num_local == e and 0 <= dea_group < dea_groups
        assert logits.dtype == torch.bfloat16 and logits.is_contiguous()
        assert 1 <= m <= Q35_MOE_FUSED_FRONT_MAX_TOKENS and top_k <= 10
        assert (e & (e - 1)) == 0 and (tile & (tile - 1)) == 0
        dev = logits.device
        nlp = triton.next_power_of_2(num_local)
        if self.hist is None or self.hist.numel() < nlp or self.hist.device != dev:
            self.hist = torch.zeros((nlp, ), dtype=torch.int32, device=dev)
            self.counter = torch.zeros((1, ), dtype=torch.int32, device=dev)
        if phases == 2:
            ids, wts = ids_in  # tables only (bench diagnostics)
        else:
            ids = torch.empty((m, top_k), dtype=torch.int32, device=dev)
            wts = torch.empty((m, top_k), dtype=weight_dtype, device=dev)
        rank = torch.empty((m * top_k, ), dtype=torch.int32, device=dev)
        if flat is None:
            flat = torch.empty(
                (sum(q35_prerouted_sizes(m, top_k, num_local, tile)), ),
                dtype=torch.int32,
                device=dev)
        e2p, p2t, cta_expert, cta_limit, num_ctas, num_padded = q35_prerouted_views(
            flat, m, top_k, num_local, tile)
        use_pdl = _pdl_enabled()
        if single_program:
            _q35_route_cta_kernel[(1, )](
                logits,
                ids,
                wts,
                e2p,
                p2t,
                cta_expert,
                cta_limit,
                num_ctas,
                num_padded,
                m,
                local_start,
                max(dea_groups, 1),
                dea_group,
                E=e,
                K=top_k,
                KP=16,
                MP=max(triton.next_power_of_2(m), 2),
                RB=rb or min(8, max(triton.next_power_of_2(m), 2)),
                PE=max(triton.next_power_of_2(m), 2) * 16,
                TILE=tile,
                NL=num_local,
                NLP=triton.next_power_of_2(num_local + 1),
                MAX_CTAS_PER_EXPERT=triton.cdiv(
                    Q35_MOE_FUSED_FRONT_MAX_TOKENS, tile),
                DEA=dea_groups > 0,
                USE_PDL=use_pdl,
                PHASES=phases,
                SORT=sort,
                num_warps=num_warps,
                launch_pdl=use_pdl,
            )
            return ids, wts, e2p, p2t, cta_expert, cta_limit, num_ctas, num_padded
        quant = x is not None
        if quant:
            assert global_scale is not None and x.dtype == torch.bfloat16 and x.shape[0] == m
            assert x.stride(1) == 1 and (x.shape[1] & (x.shape[1] - 1)) == 0
            xq = torch.empty((m, x.shape[1] // 2), dtype=torch.uint8, device=dev)
            xsf = torch.empty((m, x.shape[1] // 16), dtype=torch.uint8, device=dev)
            qargs = (x, x.stride(0), global_scale, xq, xsf)
        else:
            qargs = (logits, 0, self.counter, self.counter, self.counter)
        self._compiled = _q35_route_tables_kernel[(2 * m if quant else m, )](
            logits,
            ids,
            wts,
            self.hist,
            rank,
            self.counter,
            e2p,
            p2t,
            cta_expert,
            cta_limit,
            num_ctas,
            num_padded,
            m,
            local_start,
            max(dea_groups, 1),
            dea_group,
            *qargs,
            E=e,
            K=top_k,
            KP=16,
            TILE=tile,
            NL=num_local,
            NLP=nlp,
            EXPANDED_PAD=triton.next_power_of_2(
                Q35_MOE_FUSED_FRONT_MAX_TOKENS * top_k),
            MAX_CTAS_PER_EXPERT=triton.cdiv(Q35_MOE_FUSED_FRONT_MAX_TOKENS,
                                            tile),
            DEA=dea_groups > 0,
            USE_PDL=use_pdl,
            EARLY_TRIGGER=Q35_MOE_EARLY_TRIGGER_ROUTE,
            QUANT_H=x.shape[1] if quant else 0,
            num_warps=_Q35_ROUTE_WARPS,
            launch_pdl=use_pdl,
        )
        if quant:
            return ids, wts, e2p, p2t, cta_expert, cta_limit, num_ctas, num_padded, xq, xsf
        return ids, wts, e2p, p2t, cta_expert, cta_limit, num_ctas, num_padded


class Q35PreRoute:
    """Per MoE layer state of TRTLLM_Q35_MOE_PREROUTE: the route + tables
    launcher and, per decode shape, the flat table buffer and its tile (the tile
    of the tactic the op reports for the shape; buffers are made in the eager
    warmup that precedes CUDA graph capture)."""

    def __init__(self):
        self.route = Q35RouteTables()
        self.buffers = {}
        self.tiles = {}
        self.fused = None  # Q35FusedFront (TRTLLM_Q35_MOE_FRONT_FUSED), made on first use
        self.big = None  # Q35FusedFrontBig (TRTLLM_Q35_MOE_FRONT_BIG=gemm), made on first use
        self.route_big = None  # Q35RouteBig (TRTLLM_Q35_MOE_FRONT_BIG=1), made on first use

    def plan(self,
             m: int,
             top_k: int,
             num_local: int,
             device,
             tile_hint: Optional[int] = None):
        """(flat table buffer, tile) for an m-token step. tile_hint: the tile of
        the tactic the autotuner holds for the shape (tables for any other tile
        would make the op fall back to its own routing). A buffer first needed
        during CUDA graph capture comes from the graph's pool and is kept with
        the layer (every rank must pre-route the same steps under DEA)."""
        tile = (Q35_MOE_PREROUTE_TILE or tile_hint or self.tiles.get(m) or 8)
        flat = self.buffers.get((m, tile))
        chosen = q35_prerouted_tile_chosen(flat) if flat is not None else None
        if (chosen is not None and chosen != tile and not Q35_MOE_PREROUTE_TILE
                and tile_hint is None):
            self.tiles[m] = tile = chosen
            flat = self.buffers.get((m, tile))
        if flat is None:
            flat = torch.empty(
                (sum(q35_prerouted_sizes(m, top_k, num_local, tile)), ),
                dtype=torch.int32,
                device=device)
            self.buffers[(m, tile)] = flat
        return flat, tile

    def __call__(self,
                 logits: torch.Tensor,
                 top_k: int,
                 local_start: int,
                 num_local: int,
                 dea: Optional[Tuple[int, int]] = None,
                 tile_hint: Optional[int] = None,
                 x: Optional[torch.Tensor] = None,
                 global_scale: Optional[torch.Tensor] = None):
        """(flat tables, tile, ids, bf16 weights, is_dea), or None. dea =
        (groups, my group) builds dynamic-expert-assignment tables. Separate
        launch of the per-token route + tables kernel (faster than the
        single-program core). With x / global_scale the launch also
        NVFP4-quantizes x and the result ends with (xq, xsf)."""
        m = logits.shape[0]
        plan = self.plan(m, top_k, num_local, logits.device, tile_hint)
        if plan is None:
            return None
        flat, tile = plan
        out = self.route(logits,
                         top_k,
                         local_start,
                         num_local,
                         tile,
                         flat=flat,
                         dea_groups=dea[0] if dea else 0,
                         dea_group=dea[1] if dea else 0,
                         single_program=False,
                         x=x,
                         global_scale=global_scale)
        res = (flat, tile, out[0], out[1], dea is not None)
        return res + (out[8], out[9]) if x is not None else res


# TRTLLM_Q35_MOE_FRONT_FUSED=1 (default off; acts only together with TRTLLM_Q35_MOE_PREROUTE=1 and
# TRTLLM_Q35_MOE_PREROUTE_LOWM=1, i.e. on the two-launch front): the low-M router GEMM, the per-token route +
# tables launch and the NVFP4 quantization of x become ONE CuTe DSL launch (cute_dsl_kernels/blackwell/
# q35_moe_front.py: the router CTAs count their arrival on a ticket and the last one routes every token and writes
# the tables). Its logits, top-k ids / weights, xq / xsf and tables are bitwise those of the two-launch front (the
# tables up to the row order inside an expert's block, which the per-token kernel leaves to atomics; the MoE output
# does not depend on it). Not for DEA (falls back to the two-launch front); falls back too when a shape was not
# compiled before CUDA graph capture.
Q35_MOE_FRONT_FUSED = os.environ.get("TRTLLM_Q35_MOE_FRONT_FUSED", "0") == "1"
# Hand-off of the router logits to the routing CTA (q35_moe_front.py): 1 = staging buffer polled by a dedicated
# router CTA (default), 0 = arrival ticket (the last router CTA routes). Top-k: 1 = sorted lanes + pop (default),
# 0 = per-round re-max of each lane's keys (ticket hand-off only).
Q35_MOE_FRONT_SYNC = int(os.environ.get("TRTLLM_Q35_MOE_FRONT_SYNC", "1"))
Q35_MOE_FRONT_TOPK = int(os.environ.get("TRTLLM_Q35_MOE_FRONT_TOPK", "1"))
# Largest token count (M = 7 x batch at the target verify) the model path sends through the fused front; larger
# steps keep the two-launch front. Default 16 = b1 + b2, where the fused front is faster in-model at TEP8 and TEP16.
# b4 (M = 28) stays on the two-launch front: the fused kernel's routing CTA is slower there in a microbenchmark
# (+0.9..+1.6 us per layer).
Q35_MOE_FRONT_MAX_M = int(os.environ.get("TRTLLM_Q35_MOE_FRONT_MAX_M", "16"))
_Q35_FRONT_COMPILED = {}


def q35_front_fused_k_warps(m: int, n: int, k: int) -> Optional[int]:
    """Warps per router CTA of the low-M BF16 router GEMM at m rows (q35_lowm_router's tactic: the default
    tactic up to 16 rows, (16, 1) above), or None when the fused kernel cannot reproduce that GEMM (another
    tile shape or cluster split-K, e.g. through TRTLLM_Q35_LOWM_TACTIC)."""
    from ..lowm_linear import _tactic3, default_tactic
    try:
        tactic = default_tactic(m, n, k, False, False, "bf16") if m <= 16 else (16, 1)
    except ValueError:
        return None
    k_warps, n_tiles, ksplit, _, _ = _tactic3(tactic)
    if n_tiles != 1 or ksplit != 1 or k % (64 * k_warps):
        return None
    return k_warps


def q35_front_fused_supported(x: torch.Tensor, weight: torch.Tensor, global_scale: torch.Tensor, num_local: int,
                              tile: int, top_k: int) -> Optional[int]:
    """k_warps of the fused front for this call, or None when it does not apply."""
    from ..lowm_linear import _device_supported
    if not _device_supported() or x.dim() != 2 or x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        return None
    m, k = x.shape
    n = weight.shape[0]
    if (not 1 <= m <= Q35_MOE_FUSED_FRONT_MAX_TOKENS or n != 512 or top_k != 10 or weight.shape[1] != k
            or not weight.is_contiguous() or x.stride(1) != 1 or x.stride(0) % 8 or x.data_ptr() % 16
            or k % 16 or global_scale.dtype != torch.float32 or global_scale.numel() != 1):
        return None
    if num_local % 32 or not 32 <= num_local <= n or tile & (tile - 1) or not 8 <= tile <= 256:
        return None
    k_warps = q35_front_fused_k_warps(m, n, k)
    if k_warps is None or num_local > 32 * k_warps or m * top_k > 32 * k_warps:
        return None
    return k_warps


def _q35_front_compile(m: int, k: int, n: int, k_warps: int, num_local: int, tile: int, top_k: int, a_stride: int,
                       use_pdl: bool, sync_mode: int, topk_mode: int, trace: bool):
    key = (m, k, n, k_warps, num_local, tile, top_k, a_stride, use_pdl, sync_mode, topk_mode, trace)
    fn = _Q35_FRONT_COMPILED.get(key)
    if fn is None:
        if torch.cuda.is_current_stream_capturing():
            return None  # compile outside graph capture only (eager warmup precedes it)
        import cuda.bindings.driver as cuda
        import cutlass
        import cutlass.cute as cute

        from ...cute_dsl_kernels.blackwell.q35_moe_front import Q35MoeFrontKernel
        from ..lowm_linear import _cute, _row_view
        kernel = Q35MoeFrontKernel(num_rows=m, k_extent=k, k_warps=k_warps, use_pdl=use_pdl, num_experts=n,
                                   top_k=top_k, num_local=num_local, tile=tile, a_stride=a_stride,
                                   sync_mode=sync_mode, topk_mode=topk_mode, trace=trace)
        dev = torch.device("cuda")
        a16 = _row_view(m, k, a_stride, torch.bfloat16, dev)
        w = torch.empty((n, k), dtype=torch.bfloat16, device=dev)
        c = torch.empty((m, n), dtype=torch.bfloat16, device=dev)
        z = cutlass.Int64(0)
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        fn = (cute.compile(kernel, _cute(a16), _cute(w), _cute(c), z, z, z, z, z, z, z, z, cutlass.Int32(0),
                           stream), kernel.ctas)
        _Q35_FRONT_COMPILED[key] = fn
    return fn


class Q35FusedFront:
    """Launcher of the one-launch MoE front (TRTLLM_Q35_MOE_FRONT_FUSED) with its per-layer synchronization state:
    the staging buffer [32, E] int32 (SYNC_STAGE; the router CTA zeroes what it consumed) or the arrival ticket
    (SYNC_TICKET; reset by the last router CTA). Both are zero between launches."""

    def __init__(self):
        self.ticket = None
        self.stage = None
        self.trace_buf = None  # last trace build's stamps (debug)

    def __call__(self, x: torch.Tensor, weight: torch.Tensor, global_scale: torch.Tensor, flat: torch.Tensor,
                 tile: int, local_start: int, num_local: int, top_k: int, sync_mode: Optional[int] = None,
                 topk_mode: Optional[int] = None, trace: bool = False):
        """(logits [m, E] bf16, xq [m, H/2] u8, xsf [m, H/16] u8, ids [m, K] int32, bf16 weights [m, K]) with the
        tables written into ``flat`` (q35_prerouted_views layout for (m, top_k, num_local, tile)); None when the
        fused kernel does not apply. sync_mode / topk_mode: TRTLLM_Q35_MOE_FRONT_SYNC / _TOPK by default."""
        k_warps = q35_front_fused_supported(x, weight, global_scale, num_local, tile, top_k)
        if k_warps is None:
            return None
        m, k = x.shape
        n = weight.shape[0]
        sync_mode = Q35_MOE_FRONT_SYNC if sync_mode is None else sync_mode
        topk_mode = Q35_MOE_FRONT_TOPK if topk_mode is None else topk_mode
        if topk_mode == 0:
            sync_mode = 0
        assert flat.dtype == torch.int32 and flat.is_contiguous()
        assert flat.numel() >= sum(q35_prerouted_sizes(m, top_k, num_local, tile))
        compiled = _q35_front_compile(m, k, n, k_warps, num_local, tile, top_k, x.stride(0), _pdl_enabled(),
                                      sync_mode, topk_mode, trace)
        if compiled is None:
            return None
        fn, ctas = compiled
        import cuda.bindings.driver as cuda
        import cutlass

        from ..lowm_linear import _cute
        dev = x.device
        if sync_mode == 1:
            if self.stage is None or self.stage.device != dev or self.stage.numel() < 32 * n:
                self.stage = torch.zeros((32 * n, ), dtype=torch.int32, device=dev)
            sync = self.stage
        else:
            if self.ticket is None or self.ticket.device != dev:
                self.ticket = torch.zeros((1, ), dtype=torch.int32, device=dev)
            sync = self.ticket
        trace_ptr = 0
        if trace:
            self.trace_buf = torch.zeros(((ctas + 1) * 8, ), dtype=torch.int64, device=dev)
            trace_ptr = self.trace_buf.data_ptr()
        logits = torch.empty((m, n), dtype=torch.bfloat16, device=dev)
        xq = torch.empty((m, k // 2), dtype=torch.uint8, device=dev)
        xsf = torch.empty((m, k // 16), dtype=torch.uint8, device=dev)
        ids = torch.empty((m, top_k), dtype=torch.int32, device=dev)
        wts = torch.empty((m, top_k), dtype=torch.bfloat16, device=dev)
        p64 = lambda t: cutlass.Int64(t.data_ptr())  # noqa: E731
        fn(_cute(x), _cute(weight), _cute(logits), p64(global_scale), p64(xq), p64(xsf), p64(ids), p64(wts),
           p64(flat), p64(sync), cutlass.Int64(trace_ptr), cutlass.Int32(local_start),
           cuda.CUstream(torch.cuda.current_stream().cuda_stream))
        return logits, xq, xsf, ids, wts


def _q35_front_fused(moe, weight: torch.Tensor, x: torch.Tensor, global_scale: torch.Tensor):
    """TRTLLM_Q35_MOE_FRONT_FUSED: router GEMM + routing + tables + x quant in one launch. Hands the pre-routed
    state to the backend (as _q35_preroute does) and returns (logits, xq, xsf), or None (the caller then runs
    the two-launch front)."""
    state = _q35_preroute_state(moe)
    if state is None:
        return None
    top_k, local_start, num_local, dea = _q35_route_spec(moe)
    if dea is not None:
        return None
    backend = moe.backend
    m = x.shape[0]
    if m > Q35_MOE_FRONT_MAX_M:
        return None
    tactic = q35_tuned_tactic(backend, m)
    flat, tile = state.plan(m, top_k, num_local, x.device, tile_hint=tactic[0] if tactic else None)
    if state.fused is None:
        state.fused = Q35FusedFront()
    out = state.fused(x, weight, global_scale, flat, tile, local_start, num_local, top_k)
    if out is None:
        return None
    logits, xq, xsf, ids, wts = out
    backend._q35_prerouted_pending = (flat, tile, ids, wts, False, tactic)
    return logits, xq, xsf


# TRTLLM_Q35_MOE_FRONT_BIG (default off; TRTLLM-gen NVFP4 MoE whose C++ runner has run_moe_prerouted): the MoE front
# above 32 tokens, i.e. the MTP-6 target verify at b16 / b32 (112 / 224 tokens), where the stock path runs the
# router GEMM (cuBLAS), fp4_quantize and the op's routing kernel. Not for DEA; falls back to the stock path when a
# shape was not compiled before CUDA graph capture.
#   1 / route: the stock router GEMM (logits bitwise the stock path's), then the routing, the pre-routed tables
#      and the NVFP4 quantization of x in ONE CuTe DSL launch (q35_moe_front_big.Q35MoeRouteBigKernel: clusters of
#      8, the routing CTAs combine their per-expert counts through distributed shared memory).
#   gemm: the router GEMM in the same launch too (q35_moe_front_big.Q35MoeFrontBigKernel: logits of the low-M BF16
#      router GEMM, tactic (16, 1), not cuBLAS's: top-k near-ties can flip against the stock path). On GB300 its
#      split-K GEMM (mma.sync, partials through L2) is slower than cuBLAS, so it is the A/B variant, not the
#      default.
_Q35_MOE_FRONT_BIG_MODE = os.environ.get("TRTLLM_Q35_MOE_FRONT_BIG", "0").lower()
Q35_MOE_FRONT_BIG = _Q35_MOE_FRONT_BIG_MODE in ("1", "route", "gemm")
Q35_MOE_FRONT_BIG_GEMM = _Q35_MOE_FRONT_BIG_MODE == "gemm"
# Token counts that take it (default 112 and 224: the target verify of CUDA-graph batches 16 and 32).
Q35_MOE_FRONT_BIG_MS = frozenset(
    int(v) for v in os.environ.get("TRTLLM_Q35_MOE_FRONT_BIG_MS", "112,224").split(",") if v.strip())
# Router n-tiles (8 columns) per MMA warp of =gemm (0: one per 64 columns of the CTA tile). A/B knob.
Q35_MOE_FRONT_BIG_TN = int(os.environ.get("TRTLLM_Q35_MOE_FRONT_BIG_TN", "0"))
# Threads per CTA of the route-only launch (1024: one warp per token up to 256 tokens). A/B knob.
Q35_MOE_FRONT_BIG_THREADS = int(os.environ.get("TRTLLM_Q35_MOE_FRONT_BIG_THREADS", "1024"))
_Q35_FRONT_BIG_COMPILED = {}
_Q35_FRONT_BIG_SCRATCH = {}  # device -> FP32 slice-partial scratch (one launch at a time per device: stream order)


def _num_sms(device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


def _q35_front_big_compile(m: int, k: int, n: int, num_local: int, tile: int, top_k: int, a_stride: int,
                           use_pdl: bool, row_group: int, col_block: int, n_per_warp: int, trace: bool = False):
    key = ("gemm", m, k, n, num_local, tile, top_k, a_stride, use_pdl, row_group, col_block, n_per_warp, trace)
    if key in _Q35_FRONT_BIG_COMPILED:
        return _Q35_FRONT_BIG_COMPILED[key]  # None: the kernel does not take this shape
    if torch.cuda.is_current_stream_capturing():
        return None  # compile outside graph capture only (eager warmup precedes it)
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute

    from ...cute_dsl_kernels.blackwell.q35_moe_front_big import Q35MoeFrontBigKernel
    try:
        kernel = Q35MoeFrontBigKernel(num_rows=m, k_extent=k, num_experts=n, top_k=top_k, num_local=num_local,
                                      tile=tile, a_stride=a_stride, row_group=row_group, col_block=col_block,
                                      use_pdl=use_pdl, n_per_warp=n_per_warp, trace=trace)
    except ValueError:
        _Q35_FRONT_BIG_COMPILED[key] = None
        return None
    z = cutlass.Int64(0)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    fn = (cute.compile(kernel, z, z, z, z, z, z, z, z, z, z, z, z, cutlass.Int32(0), stream), kernel.ctas)
    _Q35_FRONT_BIG_COMPILED[key] = fn
    return fn


def _q35_route_big_compile(m: int, k: int, n: int, num_local: int, tile: int, top_k: int, a_stride: int,
                           use_pdl: bool, threads: int, trace: bool = False):
    key = ("route", m, k, n, num_local, tile, top_k, a_stride, use_pdl, threads, trace)
    if key in _Q35_FRONT_BIG_COMPILED:
        return _Q35_FRONT_BIG_COMPILED[key]  # None: the kernel does not take this shape
    if torch.cuda.is_current_stream_capturing():
        return None  # compile outside graph capture only (eager warmup precedes it)
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute

    from ...cute_dsl_kernels.blackwell.q35_moe_front_big import Q35MoeRouteBigKernel
    try:
        kernel = Q35MoeRouteBigKernel(num_rows=m, k_extent=k, num_experts=n, top_k=top_k, num_local=num_local,
                                      tile=tile, a_stride=a_stride, use_pdl=use_pdl, threads=threads,
                                      trace=trace)
    except ValueError:
        _Q35_FRONT_BIG_COMPILED[key] = None
        return None
    z = cutlass.Int64(0)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    fn = (cute.compile(kernel, z, z, z, z, z, z, z, z, z, cutlass.Int32(0), stream), kernel.ctas)
    _Q35_FRONT_BIG_COMPILED[key] = fn
    return fn


class Q35RouteBig:
    """Launcher of the route-only big front (TRTLLM_Q35_MOE_FRONT_BIG=1): routing + pre-routed tables + NVFP4 x in one
    launch after the stock router GEMM (no device state carried between launches)."""

    def __init__(self):
        self.trace_buf = None  # last trace build's stamps (debug)

    def compiled(self, x: torch.Tensor, n: int, global_scale: torch.Tensor, tile: int, num_local: int, top_k: int,
                 threads: Optional[int] = None, trace: bool = False):
        """The compiled launch for this call, or None when the kernel does not apply (checked before the router
        GEMM runs, so the caller can still take the stock path)."""
        from ..lowm_linear import _device_supported
        if not _device_supported() or x.dim() != 2 or x.dtype != torch.bfloat16:
            return None
        m, k = x.shape
        if (m > 256 or n != 512 or top_k != 10 or k % 16 or x.stride(1) != 1 or x.stride(0) % 8
                or x.data_ptr() % 16 or global_scale.dtype != torch.float32 or global_scale.numel() != 1):
            return None
        if num_local % 32 or not 32 <= num_local <= n or tile & (tile - 1) or not 8 <= tile <= 256:
            return None
        return _q35_route_big_compile(m, k, n, num_local, tile, top_k, x.stride(0), _pdl_enabled(),
                                      threads or Q35_MOE_FRONT_BIG_THREADS, trace)

    def __call__(self, fn, logits: torch.Tensor, x: torch.Tensor, global_scale: torch.Tensor, flat: torch.Tensor,
                 tile: int, local_start: int, num_local: int, top_k: int, trace: bool = False):
        """(xq [m, H/2] u8, xsf [m, H/16] u8, ids [m, K] int32, bf16 weights [m, K]) with the tables written into
        ``flat`` (q35_prerouted_views layout for (m, top_k, num_local, tile))."""
        import cuda.bindings.driver as cuda
        import cutlass
        m, k = x.shape
        assert logits.dtype == torch.bfloat16 and logits.is_contiguous() and logits.shape == (m, 512)
        assert flat.dtype == torch.int32 and flat.is_contiguous()
        assert flat.numel() >= sum(q35_prerouted_sizes(m, top_k, num_local, tile))
        dev = x.device
        xq = torch.empty((m, k // 2), dtype=torch.uint8, device=dev)
        xsf = torch.empty((m, k // 16), dtype=torch.uint8, device=dev)
        ids = torch.empty((m, top_k), dtype=torch.int32, device=dev)
        wts = torch.empty((m, top_k), dtype=torch.bfloat16, device=dev)
        fn, ctas = fn
        trace_ptr = 0
        if trace:
            self.trace_buf = torch.zeros((ctas * 8, ), dtype=torch.int64, device=dev)
            trace_ptr = self.trace_buf.data_ptr()
        p64 = lambda t: cutlass.Int64(t.data_ptr())  # noqa: E731
        fn(p64(logits), p64(x), p64(global_scale), p64(xq), p64(xsf), p64(ids), p64(wts), p64(flat),
           cutlass.Int64(trace_ptr), cutlass.Int32(local_start), cuda.CUstream(torch.cuda.current_stream().cuda_stream))
        return xq, xsf, ids, wts


class Q35FusedFrontBig:
    """Launcher of the big one-launch MoE front (TRTLLM_Q35_MOE_FRONT_BIG) with its per-layer state (epoch, ticket,
    slice flags, token masks, staging; zero-initialized, see q35_moe_front_big)."""

    def __init__(self):
        self.state = None
        self.trace_buf = None  # last trace build's stamps (debug)

    def __call__(self, x: torch.Tensor, weight: torch.Tensor, global_scale: torch.Tensor, flat: torch.Tensor,
                 tile: int, local_start: int, num_local: int, top_k: int, n_per_warp: Optional[int] = None,
                 trace: bool = False):
        """(logits [m, E] bf16, xq [m, H/2] u8, xsf [m, H/16] u8, ids [m, K] int32, bf16 weights [m, K]) with the
        tables written into ``flat`` (q35_prerouted_views layout for (m, top_k, num_local, tile)); None when the
        kernel does not apply."""
        from ...cute_dsl_kernels.blackwell.q35_moe_front_big import MAX_ROWS, STATE_WORDS, big_front_config
        from ..lowm_linear import _device_supported
        if not _device_supported() or x.dim() != 2 or x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
            return None
        m, k = x.shape
        n = weight.shape[0]
        if (m > MAX_ROWS or k != 4096 or n != 512 or top_k != 10 or weight.shape[1] != k
                or not weight.is_contiguous() or x.stride(1) != 1 or x.stride(0) % 8 or x.data_ptr() % 16
                or weight.data_ptr() % 16 or global_scale.dtype != torch.float32 or global_scale.numel() != 1):
            return None
        if num_local % 32 or not 32 <= num_local <= n or tile & (tile - 1) or not 8 <= tile <= 256:
            return None
        dev = x.device
        cfg = big_front_config(m, n, _num_sms(dev))
        if cfg is None:
            return None
        row_group, col_block = cfg
        tn = Q35_MOE_FRONT_BIG_TN if n_per_warp is None else n_per_warp
        assert flat.dtype == torch.int32 and flat.is_contiguous()
        assert flat.numel() >= sum(q35_prerouted_sizes(m, top_k, num_local, tile))
        compiled = _q35_front_big_compile(m, k, n, num_local, tile, top_k, x.stride(0), _pdl_enabled(), row_group,
                                          col_block, tn, trace)
        if compiled is None:
            return None
        fn, ctas = compiled
        if ctas > _num_sms(dev):
            return None  # the spin waits need every CTA resident
        import cuda.bindings.driver as cuda
        import cutlass
        if self.state is None or self.state.device != dev:
            self.state = torch.zeros((STATE_WORDS, ), dtype=torch.int32, device=dev)
        scratch = _Q35_FRONT_BIG_SCRATCH.get(dev)
        if scratch is None:
            scratch = torch.zeros((MAX_ROWS * n * 16, ), dtype=torch.float32, device=dev)
            _Q35_FRONT_BIG_SCRATCH[dev] = scratch
        logits = torch.empty((m, n), dtype=torch.bfloat16, device=dev)
        xq = torch.empty((m, k // 2), dtype=torch.uint8, device=dev)
        xsf = torch.empty((m, k // 16), dtype=torch.uint8, device=dev)
        ids = torch.empty((m, top_k), dtype=torch.int32, device=dev)
        wts = torch.empty((m, top_k), dtype=torch.bfloat16, device=dev)
        trace_ptr = 0
        if trace:
            from ...cute_dsl_kernels.blackwell.q35_moe_front_big import TRACE_SLOTS
            self.trace_buf = torch.zeros((ctas * TRACE_SLOTS, ), dtype=torch.int64, device=dev)
            trace_ptr = self.trace_buf.data_ptr()
        p64 = lambda t: cutlass.Int64(t.data_ptr())  # noqa: E731
        fn(p64(x), p64(weight), p64(logits), p64(global_scale), p64(xq), p64(xsf), p64(ids), p64(wts), p64(flat),
           p64(self.state), p64(scratch), cutlass.Int64(trace_ptr), cutlass.Int32(local_start),
           cuda.CUstream(torch.cuda.current_stream().cuda_stream))
        return logits, xq, xsf, ids, wts


def _q35_front_big(moe, weight: torch.Tensor, x: torch.Tensor, global_scale: torch.Tensor):
    """TRTLLM_Q35_MOE_FRONT_BIG: router GEMM + routing + tables + x quant in one launch above 32 tokens. Hands the
    pre-routed state to the backend (as _q35_front_fused does) and returns (logits, xq, xsf), or None (the caller
    then runs the stock path)."""
    state = _q35_preroute_state(moe)
    if state is None:
        return None
    top_k, local_start, num_local, dea = _q35_route_spec(moe)
    if dea is not None:
        return None
    backend = moe.backend
    m = x.shape[0]
    tactic = q35_tuned_tactic(backend, m)
    flat, tile = state.plan(m, top_k, num_local, x.device, tile_hint=tactic[0] if tactic else None)
    if state.big is None:
        state.big = Q35FusedFrontBig()
    out = state.big(x, weight, global_scale, flat, tile, local_start, num_local, top_k)
    if out is None:
        return None
    logits, xq, xsf, ids, wts = out
    backend._q35_prerouted_pending = (flat, tile, ids, wts, False, tactic)
    return logits, xq, xsf


def _q35_route_big(moe, gate, x: torch.Tensor, global_scale: torch.Tensor):
    """TRTLLM_Q35_MOE_FRONT_BIG=1: the stock router GEMM, then routing + tables + x quant in one launch above 32
    tokens. Hands the pre-routed state to the backend and returns (logits, xq, xsf), or None (nothing ran; the
    caller then runs the stock path)."""
    state = _q35_preroute_state(moe)
    if state is None:
        return None
    top_k, local_start, num_local, dea = _q35_route_spec(moe)
    if dea is not None:
        return None
    backend = moe.backend
    m = x.shape[0]
    tactic = q35_tuned_tactic(backend, m)
    flat, tile = state.plan(m, top_k, num_local, x.device, tile_hint=tactic[0] if tactic else None)
    if state.route_big is None:
        state.route_big = Q35RouteBig()
    fn = state.route_big.compiled(x, gate.weight.shape[0], global_scale, tile, num_local, top_k)
    if fn is None:
        return None
    logits = gate(x)
    xq, xsf, ids, wts = state.route_big(fn, logits, x, global_scale, flat, tile, local_start, num_local, top_k)
    backend._q35_prerouted_pending = (flat, tile, ids, wts, False, tactic)
    return logits, xq, xsf


def _q35_stock_fp4_inputs(backend, m: int):
    """(FP4BlockScaleMoERunner, inputs) of the stock trtllm-gen FP4 MoE op call
    TRTLLMGenFusedMoE.run_moe makes for m tokens with router logits (contiguous
    EP; the home block of replicated weights under DEA), with meta tensors for
    the activations: enough for an autotuner cache lookup with the same key."""
    from ...custom_ops.trtllm_gen_custom_ops import FP4BlockScaleMoERunner
    from ...utils import ActType_TrtllmGen
    act_type = backend._to_trtllm_gen_activation_type(backend.activation_type)
    factor = 1 if act_type in [ActType_TrtllmGen.Relu2, ActType_TrtllmGen.Silu
                               ] else 2
    inter = backend.w3_w1_weight.shape[-2] // factor
    g1a = backend.situ_alpha if backend.is_situ_activation else backend.swiglu_alpha
    g1b = backend.situ_beta if backend.is_situ_activation else backend.swiglu_beta
    data = lambda name: (getattr(backend, name).data
                         if getattr(backend, name, None) is not None else None)
    per_expert = [
        backend.w3_w1_weight, backend.w3_w1_weight_scale,
        backend.w3_w1_bias if backend.bias else None, g1a, g1b,
        backend.swiglu_limit, backend.w2_weight, backend.w2_weight_scale,
        backend.w2_bias if backend.bias else None,
        data("fc31_scale_c"), data("fc31_alpha"), data("fc2_alpha")
    ]
    if backend.__dict__.get("_q35_dea_home") is not None:
        per_expert = q35_dea_home(backend, per_expert)
        lo, n = backend._q35_dea_home
    else:
        lo, n = backend.slot_start, backend.expert_size_per_partition
    rp = backend._extract_routing_params()
    runner = FP4BlockScaleMoERunner(backend.num_slots,
                                    rp.top_k,
                                    rp.n_group,
                                    rp.topk_group,
                                    inter,
                                    lo,
                                    n,
                                    rp.routed_scaling_factor,
                                    backend.routing_method.routing_method_type,
                                    True,
                                    act_type,
                                    tune_max_num_tokens=backend.max_num_tokens,
                                    use_dp=backend.use_dp)
    h2 = backend.w3_w1_weight.shape[-1]  # packed fp4: hidden / 2
    meta = lambda *shape: torch.empty(shape, device="meta")
    inputs = [
        meta(m, backend.num_slots), rp.routing_bias, meta(m, h2),
        meta(m * h2 // 8), *per_expert, None, None
    ]
    return runner, inputs


def q35_tuned_tactic(backend, m: int):
    """[tile, cfg] the autotuner holds for the stock FP4 MoE op call of this
    layer at m tokens (the tactic the op will run), or None."""
    from ...custom_ops.trtllm_gen_custom_ops import _q35_moegemm_pinned_tactic
    runner, inputs = _q35_stock_fp4_inputs(backend, m)
    _, tactic = AutoTuner.get().choose_one("trtllm::fp4_block_scale_moe_runner",
                                           [runner], runner.tuning_config,
                                           inputs)
    # TRTLLM_Q35_MOEGEMM_PIN: the op runs the pinned tactic instead of the
    # tuner's pick, the same on every rank; build the tables for its tile.
    pinned = _q35_moegemm_pinned_tactic(runner, m, inputs[2].shape[1] * 2)
    if pinned is not None:
        tactic = pinned
    if isinstance(tactic, (list, tuple)) and len(tactic) == 2 and tactic[0] > 0:
        return [int(tactic[0]), int(tactic[1])]
    return None


def _q35_preroute_state(moe):
    """The layer's Q35PreRoute, or None when pre-routing does not apply."""
    from ...custom_ops.trtllm_gen_custom_ops import FP4BlockScaleMoERunner
    from .routing import RoutingMethodType
    backend = moe.backend
    if (moe.routing_method.routing_method_type != RoutingMethodType.Renormalize
            or backend.num_slots != backend.num_experts
            or not FP4BlockScaleMoERunner.supports_prerouted()):
        return None
    state = backend.__dict__.get("_q35_preroute")
    if state is None:
        state = backend._q35_preroute = Q35PreRoute()
    return state


def _q35_route_spec(moe):
    """(top_k, local_start, num_local, dea) of the layer's pre-routed tables."""
    backend = moe.backend
    dea = backend.__dict__.get("_q35_dea_groups")
    if dea is not None:
        return moe.routing_method.experts_per_token, 0, backend.num_experts, dea
    return (moe.routing_method.experts_per_token, backend.slot_start,
            backend.expert_size_per_partition, None)


def _q35_preroute(moe, logits: torch.Tensor, x: Optional[torch.Tensor] = None,
                  global_scale: Optional[torch.Tensor] = None):
    """Hands the TRTLLM-gen backend the routing of these logits (pre-routed
    tables), with a separate route + tables launch (router not folded). With x
    the launch also NVFP4-quantizes it: returns (xq, xsf), or None when the
    launch did not happen (the caller must then quantize x itself)."""
    state = _q35_preroute_state(moe)
    if state is None:
        return None
    backend = moe.backend
    tactic = q35_tuned_tactic(backend, logits.shape[0])
    top_k, local_start, num_local, dea = _q35_route_spec(moe)
    res = state(logits, top_k, local_start, num_local, dea=dea,
                tile_hint=tactic[0] if tactic else None, x=x,
                global_scale=global_scale)
    if res is None:
        return None
    backend._q35_prerouted_pending = res[:5] + (tactic, )
    return res[5:] if x is not None else None


def q35_dea_setup(moe, backend) -> None:
    """TRTLLM_Q35_MOE_DEA: called by ConfigurableMoE after mirroring the expert
    layout onto the backend and before create_weights: the TRTLLM-gen NVFP4
    backend of a plain EP layer (no load balancer, no attention DP) allocates and
    loads its MoE-TP shard of all experts and remembers its home block."""
    if not Q35_MOE_DEA:
        return
    from .fused_moe_trtllm_gen import TRTLLMGenFusedMoE
    qc = getattr(backend, "quant_config", None)
    if (type(backend) is not TRTLLMGenFusedMoE or qc is None
            or not qc.layer_quant_mode.has_nvfp4()
            or getattr(moe, "layer_load_balancer", None) is not None
            or getattr(moe, "use_dp", False) or moe.ep_size <= 1
            or backend.num_slots != backend.num_experts):
        return
    backend._q35_dea_home = (backend.slot_start, backend.expert_size_per_partition)
    backend._q35_dea_groups = (moe.ep_size, moe.ep_rank)
    backend.slot_start, backend.slot_end = 0, backend.num_experts
    backend.expert_size_per_partition = backend.num_experts
    backend.initial_local_expert_ids = list(range(backend.num_experts))


def q35_dea_home(backend, tensors):
    """The home-block slices of per-expert tensors (dim 0 = all experts)."""
    lo, n = backend._q35_dea_home
    e = backend.num_experts
    return [
        t[lo:lo + n] if isinstance(t, torch.Tensor) and t.dim() >= 1
        and t.shape[0] == e else t for t in tensors
    ]


def q35_dea_run(backend, inputs_full, prerouted, *, top_k, n_group, topk_group,
                intermediate_size, routed_scaling_factor, routing_method_type,
                do_finalize, act_type, output):
    """TRTLLM_Q35_MOE_DEA: run this rank's dealt experts as an all-local
    problem over the replicated weights, with the tactic the op tuned for the
    contiguous-EP problem of the same shape (same tuner key as the stock call on
    the home block). Returns the op outputs, or None when no tuned tactic of the
    tables' tile is available (the caller then runs contiguous EP).

    inputs_full: the FP4BlockScaleMoERunner inputs over all experts.

    Never falls back: every rank must run its dealt share in a dealt step (a
    rank running contiguous EP instead would duplicate / drop experts). Without
    a tuned tactic of the tables' tile it runs the first valid config of that
    tile for the all-local problem."""
    from ...custom_ops.trtllm_gen_custom_ops import FP4BlockScaleMoERunner
    flat, tile, ids, wts = prerouted[:4]
    tactic = prerouted[5] if len(prerouted) > 5 else None
    e = backend.num_experts
    m = inputs_full[2].shape[0]
    hidden = inputs_full[2].shape[1] * 2
    kw = dict(tune_max_num_tokens=backend.max_num_tokens, use_dp=backend.use_dp)
    if tactic is not None:
        q35_record_prerouted_tile(flat, int(tactic[0]))
    r_all = FP4BlockScaleMoERunner(e, top_k, n_group, topk_group,
                                   intermediate_size, 0, e,
                                   routed_scaling_factor, routing_method_type,
                                   do_finalize, act_type, **kw)
    cache = backend.__dict__.setdefault("_q35_dea_cfg", {})
    key = (m, tile, tuple(tactic) if tactic else None)
    if key not in cache:
        valid = sorted(
            int(b) for a, b in r_all.get_runner().get_valid_configs(
                top_k, hidden, intermediate_size, e, m) if int(a) == tile)
        assert valid, f"no valid trtllm-gen config of tile {tile} for the all-local problem"
        tuned = tactic is not None and tactic[0] == tile and tactic[1] in valid
        cache[key] = tactic[1] if tuned else valid[0]
    tables = q35_prerouted_views(flat, m, top_k, e, tile)
    out = r_all(inputs_full,
                tactic=[tile, cache[key]],
                output=output,
                prerouted=(tables, ids, wts,
                           q35_dea_max_ctas(m, top_k, e,
                                            backend._q35_dea_groups[0], tile)),
                prerouted_tile=tile)
    if not do_finalize:
        out = q35_prerouted_unfinalized(out, flat, top_k, wts)
    return out


def q35_prerouted_unfinalized(outputs, flat: torch.Tensor, top_k: int,
                              wts: torch.Tensor):
    """do_finalize=False outputs of a pre-routed MoE run, shaped like the stock
    op's: [fc2 rows, [m, K] top-k weights, [m, K] int32 expanded -> permuted
    index (-1: another rank's pick)]. The weights and the index table are the
    caller's own tensors (the route kernel's output and the head of the flat
    table buffer; a custom op cannot return them, see
    fp4_block_scale_moe_runner). TRTLLM_Q35_FIN_FOLD hands these to the CuTe
    all-reduce, which finalizes in-kernel."""
    m = wts.shape[0]
    return [
        outputs[0],
        wts.view(m, top_k),
        flat[:m * top_k].view(m, top_k)
    ]


def q35_dea_max_ctas(m: int, top_k: int, num_experts: int, n_groups: int,
                     tile: int) -> int:
    """Bound on the CTAs of one group's DEA share (any routing): the group holds
    A <= ceil(min(E, m*K) / G) experts, each with 1..m picks, so at most
    P = min(A*m, m*K) picks; its tiles sum ceil(c_e / tile) <= A + (P - A) // tile.
    The generic all-local bound min(E, m*K) + ... is several times larger."""
    a = -(-min(num_experts, m * top_k) // n_groups)
    p = min(a * m, m * top_k)
    return a + (p - a) // tile


def max_num_tiles(num_tokens: int, top_k: int, num_local_experts: int,
                  tile: int) -> int:
    """Mirror of getMaxNumCgasInBatchDim (trtllmGen blockScaleMoe runner.h)."""
    remaining = num_tokens * top_k
    filled = min(num_local_experts, remaining)
    remaining -= filled
    tiles = filled
    if remaining > 0:
        tiles += remaining // tile
    return tiles


def _pdl_enabled() -> bool:
    return os.environ.get("TRTLLM_ENABLE_PDL", "1") == "1"


def q35_moe_front(
    router_logits: torch.Tensor,
    x: torch.Tensor,
    global_scale: torch.Tensor,
    top_k: int,
    local_expert_offset: int,
    num_local_experts: int,
    tile_size: int,
    scratch: torch.Tensor,
    moe_output: Optional[torch.Tensor] = None,
    xq: Optional[torch.Tensor] = None,
    xsf: Optional[torch.Tensor] = None,
    token_selected_experts: Optional[torch.Tensor] = None,
    token_final_scales: Optional[torch.Tensor] = None,
    quantize: bool = True,
) -> Tuple[torch.Tensor, ...]:
    """Run the fused front end (``quantize=False``: xq / xsf were already
    produced upstream, e.g. by the router GEMM, and are passed through).

    Returns (token_selected_experts [M,K] i32, token_final_scales [M,K] f32,
    xq [M,H/2] u8, xsf [M,H/16] u8, tile_idx_to_expert_idx,
    tile_idx_to_mn_limit, expanded_idx_to_permuted_idx [M,K],
    permuted_idx_to_expanded_idx, total_num_padded_tokens [1],
    num_non_exiting_tiles [1], moe_output [M,H] (zeroed)) with the shapes and
    dtypes of the stock ops.
    """
    m, num_experts = router_logits.shape
    h = x.shape[1]
    assert x.dtype == torch.bfloat16 and x.stride(1) == 1
    assert router_logits.dtype == torch.bfloat16
    assert router_logits.is_contiguous()
    assert 1 <= m <= Q35_MOE_FUSED_FRONT_MAX_TOKENS and m <= tile_size
    assert top_k <= 10, "softmax sum order is written out for top_k <= 10"
    assert h % 16 == 0 and (h & (h - 1)) == 0
    assert (num_experts & (num_experts - 1)) == 0
    dev = x.device
    mt = max_num_tiles(m, top_k, num_local_experts, tile_size)
    if token_selected_experts is None:
        token_selected_experts = torch.empty((m, top_k),
                                             dtype=torch.int32,
                                             device=dev)
    if token_final_scales is None:
        token_final_scales = torch.empty((m, top_k),
                                         dtype=torch.float32,
                                         device=dev)
    if xq is None:
        xq = torch.empty((m, h // 2), dtype=torch.uint8, device=dev)
    if xsf is None:
        xsf = torch.empty((m, h // 16), dtype=torch.uint8, device=dev)
    if moe_output is None:
        moe_output = torch.empty((m, h), dtype=torch.bfloat16, device=dev)
    tile_expert = torch.empty((mt, ), dtype=torch.int32, device=dev)
    tile_mnlimit = torch.empty((mt, ), dtype=torch.int32, device=dev)
    exp2perm = torch.empty((m, top_k), dtype=torch.int32, device=dev)
    perm2exp = torch.empty((mt * tile_size, ), dtype=torch.int32, device=dev)
    total_padded = torch.empty((1, ), dtype=torch.int32, device=dev)
    num_tiles = torch.empty((1, ), dtype=torch.int32, device=dev)
    nlp = triton.next_power_of_2(num_local_experts)
    assert scratch.numel() >= 2 * nlp and scratch.dtype == torch.int32
    use_pdl = _pdl_enabled()
    _q35_moe_front_kernel[(1 + m, )](
        router_logits,
        x,
        x.stride(0),
        global_scale,
        token_selected_experts,
        token_final_scales,
        xq,
        xsf,
        tile_expert,
        tile_mnlimit,
        exp2perm,
        perm2exp,
        total_padded,
        num_tiles,
        moe_output,
        scratch,
        m,
        local_expert_offset,
        E=num_experts,
        K=top_k,
        KP=16,
        H=h,
        NL=num_local_experts,
        NLP=nlp,
        TILE=tile_size,
        BM=32,
        QUANT=quantize,
        USE_PDL=use_pdl,
        num_warps=8,
        launch_pdl=use_pdl,
    )
    return (token_selected_experts, token_final_scales, xq, xsf, tile_expert,
            tile_mnlimit, exp2perm, perm2exp, total_padded, num_tiles,
            moe_output)


def q35_fused_front_eligible(moe, x: torch.Tensor) -> bool:
    """Whether ``moe`` (a ConfigurableMoE) can take the fused decode path."""
    from .configurable_moe import ConfigurableMoE
    from .fused_moe_cute_dsl import CuteDslFusedMoE
    from .routing import RenormalizeMoeRoutingMethod
    if not isinstance(moe, ConfigurableMoE):
        return False
    backend = moe.backend
    if type(backend) is not CuteDslFusedMoE or not backend.has_nvfp4:
        return False
    if moe.comm is not None or moe.enable_dwdp or moe.use_dp:
        return False
    if moe.layer_load_balancer is not None or moe.apply_router_weight_on_input:
        return False
    if not backend.use_fused_finalize:
        return False
    rm = moe.routing_method
    if type(rm) is not RenormalizeMoeRoutingMethod or rm.top_k > 10:
        return False
    if rm.output_dtype != torch.float32 or rm.force_enable_pytorch_op:
        return False
    if not isinstance(x, torch.Tensor) or x.dtype != torch.bfloat16:
        return False
    if x.dim() != 2 or x.shape[0] < 1 or x.shape[0] > Q35_MOE_FUSED_FRONT_MAX_TOKENS:
        return False
    if AutoTuner.get().is_tuning_mode:
        return False
    return True


def q35_fused_moe_forward(moe,
                          x: torch.Tensor,
                          router_logits: torch.Tensor,
                          xq: Optional[torch.Tensor] = None,
                          xsf: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Routed-expert output of ``moe`` (ConfigurableMoE + CuteDslFusedMoE)
    for decode-sized ``x`` through the fused front end. ``xq``/``xsf``: the
    NVFP4 quantization of x when the router GEMM already produced it."""
    from .fused_moe_cute_dsl import CuteDslFusedMoENvfp4Runner
    backend = moe.backend
    m, h = x.shape
    top_k = moe.routing_method.top_k
    wv = backend._build_local_weight_view()
    esp = wv.expert_size_per_partition
    slot_start = wv.slot_start
    dev = x.device

    scratch = getattr(backend, "_q35_front_scratch", None)
    nlp = triton.next_power_of_2(esp)
    if scratch is None or scratch.numel() < 2 * nlp:
        scratch = torch.zeros((2 * nlp, ), dtype=torch.int32, device=dev)
        backend._q35_front_scratch = scratch

    # Output buffers first: the tile-size tactic lookup only needs shapes.
    tse = torch.empty((m, top_k), dtype=torch.int32, device=dev)
    tfs = torch.empty((m, top_k), dtype=torch.float32, device=dev)
    quantize = xq is None
    if quantize:
        xq = torch.empty((m, h // 2), dtype=torch.uint8, device=dev)
        xsf = torch.empty((m, h // 16), dtype=torch.uint8, device=dev)
    moe_output = torch.empty((m, backend.hidden_size),
                             dtype=torch.bfloat16,
                             device=dev)
    runner = CuteDslFusedMoENvfp4Runner(
        forward_impl=backend.run_moe_nvfp4_impl,
        num_experts=backend.num_slots,
        top_k=top_k,
        num_local_experts=esp,
        local_expert_offset=slot_start,
        enable_finalize_fusion=backend.use_fused_finalize,
        enable_alltoall=False,
    )
    _, best_tactic = AutoTuner.get().choose_one(
        "CuteDslFusedMoE::run_moe_nvfp4",
        [runner],
        runner.get_tuning_config(),
        [xq, tse, tfs, xsf, moe_output, wv],
    )
    tile_size = best_tactic if isinstance(best_tactic,
                                          int) and best_tactic > 0 else 128

    (tse, tfs, xq, xsf, tile_expert, tile_mnlimit, exp2perm, perm2exp,
     _total_padded, num_tiles, moe_output) = q35_moe_front(
         router_logits,
         x,
         backend.fc31_input_scale,
         top_k,
         slot_start,
         esp,
         tile_size,
         scratch,
         moe_output=moe_output,
         xq=xq,
         xsf=xsf,
         token_selected_experts=tse,
         token_final_scales=tfs,
         quantize=quantize)
    del exp2perm  # consumed only by the (replaced) memset kernel

    xa, xa_sf = torch.ops.trtllm.cute_dsl_nvfp4_gather_grouped_gemm_act_fusion_blackwell(
        input=xq.view(torch.float4_e2m1fn_x2),
        weight=wv.w3_w1_weight.view(torch.float4_e2m1fn_x2),
        input_scale=xsf.view(torch.uint8),
        weight_scale=wv.fc1_weight_scale.view(torch.uint8),
        alpha=wv.fc1_global_scale,
        tile_idx_to_group_idx=tile_expert,
        tile_idx_to_mn_limit=tile_mnlimit,
        permuted_idx_to_expanded_idx=perm2exp,
        num_non_exiting_tiles=num_tiles,
        global_sf=backend.fc2_input_scale,
        num_experts=backend.num_slots,
        top_k=top_k,
        num_local_experts=esp,
        local_expert_offset=slot_start,
        tile_size=tile_size,
        activation_type=backend.activation_type,
        swiglu_limit_scalar=backend.swiglu_limit_scalar,
    )
    torch.ops.trtllm.cute_dsl_nvfp4_grouped_gemm_finalize_inplace_blackwell(
        input=xa.view(torch.float4_e2m1fn_x2),
        weight=wv.w2_weight.view(torch.float4_e2m1fn_x2),
        input_scale=xa_sf.view(torch.uint8),
        weight_scale=wv.fc2_weight_scale.view(torch.uint8),
        alpha=wv.fc2_global_scale,
        output=moe_output,
        tile_idx_to_group_idx=tile_expert,
        tile_idx_to_mn_limit=tile_mnlimit,
        permuted_idx_to_expanded_idx=perm2exp,
        num_non_exiting_tiles=num_tiles,
        token_final_scales=tfs,
        num_experts=backend.num_slots,
        top_k=top_k,
        num_local_experts=esp,
        local_expert_offset=slot_start,
        tile_size=tile_size,
        output_dtype=torch.bfloat16,
    )
    moe.repeat_idx = (moe.repeat_idx + 1) % moe.repeat_count
    return moe_output


def _trtllm_quant_eligible(moe, x: torch.Tensor,
                           max_tokens: int = Q35_MOE_FUSED_FRONT_MAX_TOKENS) -> bool:
    from .configurable_moe import ConfigurableMoE
    from .fused_moe_trtllm_gen import TRTLLMGenFusedMoE
    if not isinstance(moe, ConfigurableMoE):
        return False
    backend = moe.backend
    if type(backend) is not TRTLLMGenFusedMoE or not backend.has_nvfp4:
        return False
    if moe.comm is not None or moe.enable_dwdp or moe.use_dp:
        return False
    if moe.layer_load_balancer is not None or moe.apply_router_weight_on_input:
        return False
    if getattr(backend, "fc31_act_scale", None) is not None:
        return False
    if backend.w3_w1_weight.shape[-1] * 2 != x.shape[-1]:
        return False
    if not isinstance(x, torch.Tensor) or x.dtype != torch.bfloat16:
        return False
    if x.dim() != 2 or x.shape[0] < 1 or x.shape[0] > max_tokens:
        return False
    if AutoTuner.get().is_tuning_mode:
        return False
    return True


def q35_routed_forward(gate, moe, x: torch.Tensor, **moe_kwargs):
    """Q35 decode fast paths for Qwen3NextSparseMoeBlock's routed experts.

    Returns the routed output, or None when no fast path applies (the caller
    then runs the stock path).
      * TRTLLM_Q35_MOE_FUSED_FRONT=1, CuteDSL NVFP4: router GEMM (stock cuBLAS
        or, with TRTLLM_Q35_MOE_ROUTER_GEMM=1, the Triton GEMM that also
        quantizes x) -> fused route/sort/quant front end -> FC1 -> FC2.
      * TRTLLM_Q35_MOE_ROUTER_GEMM=1, TRTLLM-gen NVFP4: one launch for router
        GEMM + NVFP4 quantization of x, then the trtllm-gen MoE op on the
        pre-quantized input (drops the separate fp4_quantize kernel); with
        TRTLLM_Q35_MOE_PREROUTE=1 also the route + tables launch that lets the
        op skip its routing kernel.
    """
    from ...utils import Fp4QuantizedTensor
    if (Q35_MOE_FRONT_BIG and x.dim() == 2 and x.shape[0] in Q35_MOE_FRONT_BIG_MS
            and gate.out_dtype == torch.bfloat16 and _trtllm_quant_eligible(moe, x, max_tokens=256)):
        # routing + tables + x quant (and with =gemm the router GEMM) in ONE launch above 32 tokens
        if Q35_MOE_FRONT_BIG_GEMM:
            big = _q35_front_big(moe, gate.weight, x, moe.backend.fc31_input_scale)
        else:
            big = _q35_route_big(moe, gate, x, moe.backend.fc31_input_scale)
        if big is not None:
            logits, xq, xsf = big
            x_in = Fp4QuantizedTensor(xq, xsf, is_sf_swizzled=False)
            return moe(x_in, logits, output_dtype=x.dtype, **moe_kwargs)
    if Q35_MOE_FUSED_FRONT and q35_fused_front_eligible(moe, x):
        if Q35_MOE_ROUTER_GEMM and gate.q35_router_ok(x):
            logits, xq, xsf = gate.q35_router(x, moe.backend.fc31_input_scale)
            return q35_fused_moe_forward(moe, x, logits, xq, xsf)
        return q35_fused_moe_forward(moe, x, gate(x))
    if Q35_MOE_ROUTER_GEMM and _trtllm_quant_eligible(
            moe, x) and gate.q35_router_ok(x):
        gs = moe.backend.fc31_input_scale
        state = _q35_preroute_state(moe) if Q35_MOE_PREROUTE else None
        plan = None
        if state is not None and Q35_MOE_PREROUTE_FOLD and not Q35_MOE_ROUTER_LOWM:
            top_k, local_start, num_local, dea = _q35_route_spec(moe)
            tactic = q35_tuned_tactic(moe.backend, x.shape[0])
            plan = state.plan(x.shape[0], top_k, num_local, x.device,
                              tile_hint=tactic[0] if tactic else None)
        quant = None
        if (plan is None and state is not None and Q35_MOE_PREROUTE_LOWM
                and not Q35_MOE_ROUTER_LOWM):
            if Q35_MOE_FRONT_FUSED:
                # the same router GEMM, routing, tables and x quant in ONE launch
                fused = _q35_front_fused(moe, gate.weight, x, gs)
                if fused is not None:
                    logits, xq, xsf = fused
                    quant = (xq, xsf)
            if quant is None:
                # router = plain BF16 GEMM on the low-M kernel; the route + tables launch
                # also quantizes x (its extra programs), so no launch is added
                logits = q35_lowm_router(x, gate.weight)
                if logits is not None:
                    quant = _q35_preroute(moe, logits, x=x, global_scale=gs)
        if quant is not None:
            xq, xsf = quant
        elif plan is not None:
            # router GEMM + NVFP4 quant + routing + tables in ONE launch
            flat, tile = plan
            logits, xq, xsf, ids, wts = gate.q35_router(
                x, gs, route=(top_k, local_start, num_local, tile, flat, dea))
            moe.backend._q35_prerouted_pending = (flat, tile, ids, wts,
                                                  dea is not None, tactic)
        else:
            moe.backend.__dict__.pop("_q35_prerouted_pending", None)
            logits, xq, xsf = gate.q35_router(x, gs)
            if state is not None:
                _q35_preroute(moe, logits)
        x_in = Fp4QuantizedTensor(xq, xsf, is_sf_swizzled=False)
        return moe(x_in, logits, output_dtype=x.dtype, **moe_kwargs)
    return None


# ---------------------------------------------------------------------------
# Debug only (TRTLLM_Q35_MOE_STATS=1, default off): per-layer routing statistics
# of decode-sized batches, recorded with graph-capturable torch ops into
# preallocated device buffers and dumped as JSON at process exit
# (TRTLLM_Q35_MOE_STATS_PATH, "{rank}" is replaced by the global rank).
# For EP sizes 1..32: per-rank distinct experts (mean over ranks and max over
# ranks), i.e. the expert bytes the average / busiest GPU must read.
# ---------------------------------------------------------------------------
Q35_MOE_STATS = os.environ.get("TRTLLM_Q35_MOE_STATS", "0") == "1"
# TRTLLM_Q35_MOE_STATS_IDS=<n> (with TRTLLM_Q35_MOE_STATS=1): also keep the top-k expert ids of the last n
# decode-sized forwards of every MoE layer (device ring buffer, graph-capturable), saved by every rank next to the
# JSON as *_ids.pt (with a {rank} path) for offline expert-placement / assignment simulation and for checking that
# all ranks route identically (per-step expert assignment relies on it).
Q35_MOE_STATS_IDS = int(os.environ.get("TRTLLM_Q35_MOE_STATS_IDS", "0"))
_Q35_STATS_EP = (1, 2, 4, 8, 16, 32)
_q35_stats = {}


def _q35_stats_buffers(device, num_experts):
    st = _q35_stats.get(device)
    if st is None:
        nl = 64  # layer slots (60 target + MTP)
        mm = Q35_MOE_FUSED_FRONT_MAX_TOKENS + 1
        st = dict(
            steps=torch.zeros((nl, mm), dtype=torch.float64, device=device),
            mean=torch.zeros((nl, mm, len(_Q35_STATS_EP)),
                             dtype=torch.float64,
                             device=device),
            busiest=torch.zeros((nl, mm, len(_Q35_STATS_EP)),
                                dtype=torch.float64,
                                device=device),
            num_experts=num_experts,
        )
        if Q35_MOE_STATS_IDS > 0:
            n = Q35_MOE_STATS_IDS
            st["ids"] = torch.full((nl, n, mm - 1, 16),
                                   -1,
                                   dtype=torch.int16,
                                   device=device)
            st["ids_m"] = torch.zeros((nl, n), dtype=torch.int16, device=device)
            st["ids_cnt"] = torch.zeros((nl, ), dtype=torch.int64, device=device)
        _q35_stats[device] = st
        if not _q35_stats.get("_atexit"):
            import atexit
            atexit.register(_q35_stats_dump)
            _q35_stats["_atexit"] = True
    return st


def q35_moe_stats_record(layer_idx: Optional[int], router_logits: torch.Tensor,
                         top_k: int) -> None:
    if layer_idx is None or router_logits.dim() != 2:
        return
    m, e = router_logits.shape
    if m < 1 or m > Q35_MOE_FUSED_FRONT_MAX_TOKENS or layer_idx >= 64:
        return
    st = _q35_stats_buffers(router_logits.device, e)
    idx = torch.topk(router_logits.float(), top_k, dim=-1).indices
    used = torch.zeros((m, e), dtype=torch.float32,
                       device=router_logits.device).scatter_(1, idx, 1.0)
    used = used.amax(dim=0)
    means, maxes = [], []
    for ep in _Q35_STATS_EP:
        per_rank = used.view(ep, e // ep).sum(dim=1)
        means.append(per_rank.mean())
        maxes.append(per_rank.max())
    st["steps"][layer_idx, m] += 1
    st["mean"][layer_idx, m] += torch.stack(means).double()
    st["busiest"][layer_idx, m] += torch.stack(maxes).double()
    if "ids" in st and top_k <= 16:
        # ring slot on the device (the counter advances on CUDA graph replay too)
        cnt = st["ids_cnt"][layer_idx:layer_idx + 1]
        pos = torch.remainder(cnt, st["ids"].shape[1])
        row = torch.full((1, st["ids"].shape[2], 16),
                         -1,
                         dtype=torch.int16,
                         device=router_logits.device)
        row[0, :m, :top_k] = idx.to(torch.int16)
        st["ids"][layer_idx].index_copy_(0, pos, row)
        st["ids_m"][layer_idx].index_fill_(0, pos, m)
        cnt.add_(1)


def _q35_stats_dump():
    import json
    rank = os.environ.get("OMPI_COMM_WORLD_RANK",
                          os.environ.get("SLURM_PROCID", "0"))
    path = os.environ.get("TRTLLM_Q35_MOE_STATS_PATH",
                          "/tmp/q35_moe_stats_rank{rank}.json").replace(
                              "{rank}", str(rank))
    out = {"ep_sizes": list(_Q35_STATS_EP), "layers": {}}
    for dev, st in _q35_stats.items():
        if not isinstance(st, dict):
            continue
        steps = st["steps"].cpu()
        mean = st["mean"].cpu()
        busiest = st["busiest"].cpu()
        for layer in range(steps.shape[0]):
            for m in range(steps.shape[1]):
                n = steps[layer, m].item()
                if n <= 0:
                    continue
                out["layers"].setdefault(str(layer), {})[str(m)] = {
                    "steps": n,
                    "mean_distinct": (mean[layer, m] / n).tolist(),
                    "busiest_distinct": (busiest[layer, m] / n).tolist(),
                }
    try:
        with open(path, "w") as f:
            json.dump(out, f)
    except OSError:
        pass
    if Q35_MOE_STATS_IDS > 0:
        for dev, st in _q35_stats.items():
            if isinstance(st, dict) and "ids" in st:
                try:
                    torch.save(
                        {
                            "ids": st["ids"].cpu(),
                            "m": st["ids_m"].cpu(),
                            "count": st["ids_cnt"].cpu(),
                            "num_experts": st["num_experts"],
                        },
                        path.rsplit(".", 1)[0] + "_ids.pt")
                except (OSError, RuntimeError):
                    pass
                break
