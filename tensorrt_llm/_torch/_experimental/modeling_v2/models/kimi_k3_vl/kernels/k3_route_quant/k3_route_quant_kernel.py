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
"""Kimi K3 MoE routing and MXFP8 input quantization for decode, in CuTe DSL.

Computes exactly what ``trtllm::kimi_k3_noaux_tc_mxfp8_quant`` computes, bit for bit:

* routing: ``sigmoid(logit) = 0.5 * tanhf(0.5 * logit) + 0.5``; the 16 experts with the largest
  ``sigmoid + bias``, in descending order, ties to the lower expert id; weights
  ``bf16(sigmoid * scale / (sum of the 16 sigmoids + 1e-20))`` evaluated in fp64, the sum being the
  xor-butterfly warp reduction ``cg::reduce`` performs over lanes 0..15;
* quantization: MXFP8 (e4m3) with one UE8M0 scale per 32 elements in the linear [M, 112] layout,
  the recipe of ``cvt_warp_fp16_to_mxfp8`` (scale rounded up from ``amax / 448``).

The selection differs from the C++ kernel's: one warp holds 28 keys per lane, each lane sorts its
six largest, and the 16 winners come out of 16 rounds of ``redux.sync.max`` over the lanes' heads
plus ``redux.sync.min`` over the candidate ids of the lanes holding that maximum (the tie-break);
the winner's lane shifts its list. The C++ path sorts 32 packed 64-bit keys per lane and runs 16
rounds of 64-bit shuffle arg-max instead.

``top16_warp`` (a ``cute.jit`` function) and ``mxfp8_quant_vec8`` (a trace-time helper) can be
inlined into other kernels, e.g. a fused MoE kernel routing in its prologue.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
from cutlass import dsl_user_op
from cutlass.experimental import primitives as prims

NUM_EXPERTS = 896
TOP_K = 16
HIDDEN_SIZE = 3584
SF_VEC_SIZE = 32
THREADS = HIDDEN_SIZE // 8  # one 8-element vector per thread in a quantization CTA
KEYS_PER_LANE = NUM_EXPERTS // 32
FULL_MASK = 0xFFFFFFFF
REMOVED_KEY = -(2**31)  # below the key of every float
NO_CANDIDATE = 0x7FFFFFFF
LANE_LIST = 6  # sorted keys each lane keeps for the fast selection rounds

assert NUM_EXPERTS % 32 == 0 and NUM_EXPERTS == 2 * THREADS and TOP_K <= 32


# =============================================================================
# PTX the C++ reference uses, emitted verbatim so the results match bit for bit.
# =============================================================================
def _asm(result_type, operands, text, constraints, *, loc, ip):
    from cutlass._mlir.dialects import llvm as _llvm

    return _llvm.inline_asm(
        result_type,
        [op.ir_value(loc=loc, ip=ip) for op in operands],
        text,
        constraints,
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=_llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def rcp_approx_ftz(a, *, loc=None, ip=None):
    """``rcp.approx.ftz.f32`` (``reciprocal_approximate_ftz``); never constant-folded."""
    from cutlass._mlir.extras import types as _T

    return cutlass.Float32(
        _asm(_T.f32(), [cutlass.Float32(a)], "rcp.approx.ftz.f32 $0, $1;", "=f,f", loc=loc, ip=ip)
    )


@dsl_user_op
def cvt_rp_satfinite_ue8m0(a, *, loc=None, ip=None):
    """UE8M0 byte of ``a`` rounded toward +inf, saturated to finite (``__nv_cvt_float_to_e8m0``)."""
    from cutlass._mlir.extras import types as _T

    pair = cutlass.Int16(
        _asm(
            _T.i16(),
            [cutlass.Float32(a), cutlass.Float32(a)],
            "cvt.rp.satfinite.ue8m0x2.f32 $0, $1, $2;",
            "=h,f,f",
            loc=loc,
            ip=ip,
        )
    )
    return cutlass.Int32(pair) & cutlass.Int32(0xFF)


@dsl_user_op
def cvt_rn_satfinite_e4m3x2(hi, lo, *, loc=None, ip=None):
    """Two e4m3 bytes, ``lo`` in bits 0-7 and ``hi`` in bits 8-15 (``__nv_fp8x2_e4m3(float2)``)."""
    from cutlass._mlir.extras import types as _T

    pair = cutlass.Int16(
        _asm(
            _T.i16(),
            [cutlass.Float32(hi), cutlass.Float32(lo)],
            "cvt.rn.satfinite.e4m3x2.f32 $0, $1, $2;",
            "=h,f,f",
            loc=loc,
            ip=ip,
        )
    )
    return cutlass.Int32(pair) & cutlass.Int32(0xFFFF)


@dsl_user_op
def cvt_rn_bf16_f64(a, *, loc=None, ip=None):
    """bf16 bits of an fp64 value, one rounding (``__double2bfloat16`` on sm_90+)."""
    from cutlass._mlir.extras import types as _T

    return cutlass.Int16(
        _asm(_T.i16(), [cutlass.Float64(a)], "cvt.rn.bf16.f64 $0, $1;", "=h,d", loc=loc, ip=ip)
    )


# =============================================================================
# Routing
# =============================================================================
def sigmoid_accurate(x):
    """``0.5f * tanhf(0.5f * x) + 0.5f`` with the accurate libdevice ``tanhf``. Both scalings by
    0.5 are exact, so contracting the outer multiply-add into an FMA cannot change the result."""
    half = cutlass.Float32(0.5)
    return half * cute.math.tanh(half * x, fastmath=False) + half


def selection_key(score):
    """Int32 whose signed order is the order ``cub::Traits<float>::TwiddleIn`` gives the bits of
    ``score`` as unsigned integers (the C++ top-k's comparison key)."""
    bits = score.bitcast(cutlass.Int32)
    return bits ^ ((bits >> cutlass.Int32(31)) & cutlass.Int32(0x7FFFFFFF))


def _argmax_lowest_index(values):
    """(max, index of its first occurrence) over a list of Int32, as a balanced tree."""
    items = [(v, cutlass.Int32(j)) for j, v in enumerate(values)]
    while len(items) > 1:
        paired = []
        for i in range(0, len(items) - 1, 2):
            (va, ja), (vb, jb) = items[i], items[i + 1]
            take_b = vb > va  # the left item has the lower indices, so it keeps ties
            paired.append(
                (
                    cutlass.Int32(cutlass.select_(take_b, vb, va)),
                    cutlass.Int32(cutlass.select_(take_b, jb, ja)),
                )
            )
        if len(items) % 2 == 1:
            paired.append(items[-1])
        items = paired
    return items[0]


def _select_i32(pred, a, b):
    return cutlass.Int32(cutlass.select_(pred, a, b))


def _round_winner(head, head_slot, lane):
    """One selection round: the largest head over the warp, the lowest expert id among equal heads.
    Returns (winner id, this lane's candidate id)."""
    top = prims.redux_sync(head, prims.ReductionKind.MAX, FULL_MASK)
    candidate = _select_i32(
        head == top, head_slot * cutlass.Int32(32) + lane, cutlass.Int32(NO_CANDIDATE)
    )
    return prims.redux_sync(candidate, prims.ReductionKind.MIN, FULL_MASK), candidate


def _load_lane_keys(s_key, lane):
    return [s_key.load(idx=lane + cutlass.Int32(32 * slot)) for slot in range(KEYS_PER_LANE)]


def _rounds_exact(lane_keys, lane):
    """The 16 rounds over all 28 keys of every lane, the lane's argmax recomputed after each pop.
    Returns the round-r winner in lane r."""
    keys = list(lane_keys)
    best, best_slot = _argmax_lowest_index(keys)
    expert = cutlass.Int32(0)
    for rank in range(TOP_K):
        winner, candidate = _round_winner(best, best_slot, lane)
        expert = _select_i32(lane == cutlass.Int32(rank), winner, expert)
        if rank < TOP_K - 1:
            popped = candidate == winner
            for slot in range(KEYS_PER_LANE):
                keys[slot] = _select_i32(
                    popped & (best_slot == cutlass.Int32(slot)),
                    cutlass.Int32(REMOVED_KEY),
                    keys[slot],
                )
            best, best_slot = _argmax_lowest_index(keys)
    return expert


def _lane_list(lane_keys):
    """The lane's LANE_LIST largest keys and their slots, descending, the lower slot first among equal
    keys (insertion with a strict comparison)."""
    values = [cutlass.Int32(REMOVED_KEY)] * LANE_LIST
    slots = [cutlass.Int32(0)] * LANE_LIST
    for slot, key in enumerate(lane_keys):
        above = [key > v for v in values]
        new_values = [_select_i32(above[0], key, values[0])]
        new_slots = [_select_i32(above[0], cutlass.Int32(slot), slots[0])]
        for i in range(1, LANE_LIST):
            new_values.append(
                _select_i32(above[i - 1], values[i - 1], _select_i32(above[i], key, values[i]))
            )
            new_slots.append(
                _select_i32(
                    above[i - 1], slots[i - 1], _select_i32(above[i], cutlass.Int32(slot), slots[i])
                )
            )
        values, slots = new_values, new_slots
    return values, slots


def _rounds_from_lists(lane_keys, lane):
    """The 16 rounds over each lane's sorted list of its LANE_LIST largest keys, a pop being a shift.
    Returns (the round-r winner in lane r, nonzero in a lane whose list ran empty before a round)."""
    values, slots = _lane_list(lane_keys)
    expert = cutlass.Int32(0)
    emptied = cutlass.Int32(0)
    for rank in range(TOP_K):
        if rank > 0:
            emptied = emptied | _select_i32(
                values[0] == cutlass.Int32(REMOVED_KEY), cutlass.Int32(1), cutlass.Int32(0)
            )
        winner, candidate = _round_winner(values[0], slots[0], lane)
        expert = _select_i32(lane == cutlass.Int32(rank), winner, expert)
        if rank < TOP_K - 1:
            popped = candidate == winner
            for i in range(LANE_LIST - 1):
                values[i] = _select_i32(popped, values[i + 1], values[i])
                slots[i] = _select_i32(popped, slots[i + 1], slots[i])
            values[-1] = _select_i32(popped, cutlass.Int32(REMOVED_KEY), values[-1])
    return expert, emptied


def _routing_weight(s_sigmoid, expert, lane, routed_scaling_factor):
    """bf16 bits of ``sigmoid * scale / (sum of the 16 sigmoids + 1e-20)`` in fp64, the sum being
    ``cg::reduce``'s xor butterfly over the warp with lanes 16-31 contributing 0."""
    selected = lane < cutlass.Int32(TOP_K)
    sig = cutlass.Float32(
        cutlass.select_(selected, s_sigmoid.load(idx=expert), cutlass.Float32(0.0))
    )
    total = sig
    for offset in (16, 8, 4, 2, 1):
        total = total + cute.arch.shuffle_sync_bfly(total, offset=offset)
    weight = (cutlass.Float64(sig) * routed_scaling_factor) / (
        cutlass.Float64(total) + cutlass.Float64(1e-20)
    )
    return cvt_rn_bf16_f64(weight)


@cute.jit
def top16_warp(s_key, s_sigmoid, lane, routed_scaling_factor):
    """Top-16 experts of one token and their routing weights, computed by one whole warp.

    ``s_key`` (Int32 [896], from ``selection_key(sigmoid + bias)``) and ``s_sigmoid`` (Float32
    [896]) hold the token in shared memory. Returns ``(expert_id, weight_bf16_bits)``; lane r < 16
    holds the rank-r expert, the other lanes hold garbage. All 32 lanes must call it together.

    Each lane first sorts its LANE_LIST largest keys, so a round costs two ``redux.sync`` and a
    shift. A lane can hold more of the winners than that (P ~ 1.4e-5 per lane for random keys); the
    warp then redoes the rounds over all 28 keys per lane, so the result is always exact.
    """
    lane_keys = _load_lane_keys(s_key, lane)
    expert, emptied = _rounds_from_lists(lane_keys, lane)
    if prims.vote_sync(FULL_MASK, emptied != cutlass.Int32(0), prims.VoteSync.ANY):
        expert = _rounds_exact(lane_keys, lane)
    return expert, _routing_weight(s_sigmoid, expert, lane, routed_scaling_factor)


# =============================================================================
# MXFP8 quantization
# =============================================================================
def mxfp8_quant_vec8(words):
    """MXFP8 of eight bf16 values given as four Int32 words (element 2i in the low half of word i).

    The four consecutive lanes that share one 32-element scale must call it together. Returns
    ``(q_lo, q_hi, sf_byte)``: the e4m3 bytes of elements 0-3 and 4-7 (element order, little
    endian) and the UE8M0 scale byte of the lane's group.
    """
    values = []
    for w in words:
        values.append((w << cutlass.Int32(16)).bitcast(cutlass.Float32))
        values.append((w & cutlass.Int32(-65536)).bitcast(cutlass.Float32))
    amax = cute.arch.fmax(cute.math.abs(values[0]), cute.math.abs(values[1]))
    for v in values[2:]:
        amax = cute.arch.fmax(amax, cute.math.abs(v))
    for offset in (1, 2):
        amax = cute.arch.fmax(cute.arch.shuffle_sync_bfly(amax, offset=offset), amax)

    sf_byte = cvt_rp_satfinite_ue8m0(amax * rcp_approx_ftz(cutlass.Float32(448.0)))
    # static_cast<float>(__nv_fp8_e8m0): 2^(byte - 127), with byte 0 the fp32 denormal 2^-127.
    sf_bits = cutlass.Int32(
        cutlass.select_(
            sf_byte == cutlass.Int32(0), cutlass.Int32(0x00400000), sf_byte << cutlass.Int32(23)
        )
    )
    sf_bits = cutlass.Int32(
        cutlass.select_(sf_byte == cutlass.Int32(0xFF), cutlass.Int32(0x7FFFFFFF), sf_bits)
    )
    out_scale = cutlass.Float32(
        cutlass.select_(
            amax != cutlass.Float32(0.0),
            rcp_approx_ftz(sf_bits.bitcast(cutlass.Float32)),
            cutlass.Float32(0.0),
        )
    )
    pairs = [
        cvt_rn_satfinite_e4m3x2(values[2 * i + 1] * out_scale, values[2 * i] * out_scale)
        for i in range(4)
    ]
    q_lo = pairs[0] | (pairs[1] << cutlass.Int32(16))
    q_hi = pairs[2] | (pairs[3] << cutlass.Int32(16))
    return q_lo, q_hi, sf_byte


# =============================================================================
# Kernel: CTAs [0, M) route token b, CTAs [M, 2M) quantize row b - M (the C++ kernel's split).
# =============================================================================
@cute.kernel
def k3_route_quant_kernel(
    scores: cutlass.Array,  # fp32 [M * 896] router logits
    bias: cutlass.Array,  # fp32 [896]
    hidden_words: cutlass.Array,  # int32 view of bf16 [M, 3584]: [M * 1792]
    topk_ids: cutlass.Array,  # int32 [M * 16]
    topk_weight_bits: cutlass.Array,  # int16 view of bf16 [M * 16]
    quant_words: cutlass.Array,  # int32 view of e4m3 [M, 3584]: [M * 896]
    scales: cutlass.Array,  # uint8 [M * 112]
    num_tokens: cutlass.Int32,
    routed_scaling_factor: cutlass.Float64,
    early_trigger: cutlass.Constexpr[bool],
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    s_key = cutlass.Array(cutlass.Int32, NUM_EXPERTS, space=cutlass.AddressSpace.smem, alignment=16)
    s_sigmoid = cutlass.Array(
        cutlass.Float32, NUM_EXPERTS, space=cutlass.AddressSpace.smem, alignment=16
    )
    routes = bidx < num_tokens

    # The bias is a weight: read it before waiting for the producer of the logits (every CTA
    # reads it, the index is valid for all).
    bias_lo = bias.load(idx=tidx)
    bias_hi = bias.load(idx=tidx + cutlass.Int32(THREADS))

    prims.griddepcontrol(prims.GridDepAction.WAIT)
    if cutlass.const_expr(early_trigger):
        prims.griddepcontrol(prims.GridDepAction.LAUNCH_DEPENDENTS)

    if routes:
        row_base = bidx * cutlass.Int32(NUM_EXPERTS)
        sig_lo = sigmoid_accurate(scores.load(idx=row_base + tidx))
        sig_hi = sigmoid_accurate(scores.load(idx=row_base + tidx + cutlass.Int32(THREADS)))
        s_sigmoid.store(sig_lo, idx=tidx)
        s_sigmoid.store(sig_hi, idx=tidx + cutlass.Int32(THREADS))
        s_key.store(selection_key(sig_lo + bias_lo), idx=tidx)
        s_key.store(selection_key(sig_hi + bias_hi), idx=tidx + cutlass.Int32(THREADS))
        cute.arch.barrier()
        if tidx < cutlass.Int32(32):
            expert, weight_bits = top16_warp(s_key, s_sigmoid, tidx, routed_scaling_factor)
            if tidx < cutlass.Int32(TOP_K):
                out = bidx * cutlass.Int32(TOP_K) + tidx
                topk_ids.store(expert, idx=out)
                topk_weight_bits.store(weight_bits, idx=out)
    else:
        row = bidx - num_tokens
        words = hidden_words.load(
            idx=row * cutlass.Int32(HIDDEN_SIZE // 2) + tidx * cutlass.Int32(4),
            vector_size=4,
            alignment=16,
        )
        q_lo, q_hi, sf_byte = mxfp8_quant_vec8([words[0], words[1], words[2], words[3]])
        quant_words.store(
            (q_lo, q_hi),
            idx=row * cutlass.Int32(HIDDEN_SIZE // 4) + tidx * cutlass.Int32(2),
            alignment=8,
        )
        if tidx % cutlass.Int32(SF_VEC_SIZE // 8) == cutlass.Int32(0):
            scales.store(
                cutlass.Uint8(sf_byte),
                idx=row * cutlass.Int32(HIDDEN_SIZE // SF_VEC_SIZE)
                + tidx // cutlass.Int32(SF_VEC_SIZE // 8),
            )

    if cutlass.const_expr(not early_trigger):
        cute.arch.fence_acq_rel_gpu()
        cute.arch.barrier()
        prims.griddepcontrol(prims.GridDepAction.LAUNCH_DEPENDENTS)


@cute.jit
def k3_route_quant(
    scores: cute.Tensor,
    bias: cute.Tensor,
    hidden_words: cute.Tensor,
    topk_ids: cute.Tensor,
    topk_weight_bits: cute.Tensor,
    quant_words: cute.Tensor,
    scales: cute.Tensor,
    num_tokens: cutlass.Int32,
    routed_scaling_factor: cutlass.Float64,
    early_trigger: cutlass.Constexpr[bool],
    use_pdl: cutlass.Constexpr[bool],
    stream: cuda_driver.CUstream,
) -> None:
    k3_route_quant_kernel(
        scores, bias, hidden_words, topk_ids, topk_weight_bits, quant_words, scales, num_tokens,
        routed_scaling_factor, early_trigger,
    ).launch(
        grid=[num_tokens * 2, 1, 1],
        block=[THREADS, 1, 1],
        stream=stream,
        use_pdl=use_pdl,
    )  # fmt: skip
