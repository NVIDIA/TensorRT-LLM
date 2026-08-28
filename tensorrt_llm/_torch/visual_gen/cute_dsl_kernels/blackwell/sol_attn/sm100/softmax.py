# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
#
# Vendored from https://github.com/NVlabs/Sana (Apache-2.0); see
# THIRD_PARTY_NOTICES.md in this directory for the pin and scope.
#
# Portions derive from the FlashAttention project
# (https://github.com/Dao-AILab/flash-attention), BSD-3-Clause; its license
# text is vendored at sol_attn/sm100/LICENSE.flash-attention.
"""Online-softmax helpers for the Blackwell mainloop."""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32
from cutlass.cute.nvgpu import tcgen05
from flash_attn.cute import utils as fa_utils

from .tmem import _add_physical_tmem_base, _zero_based_tmem_tensor, tcgen05_wait_ld, tcgen05_wait_st

M = 64
N_HALF = 128
DV = 128
LOG2E = 1.4426950408889634


@cute.jit
def _load_m64_n128_score(
    score_template: cute.Tensor,
    thr_mma_qk: cute.ThrMma,
    tmem_base: Int32,
    score_offset: Int32,
    owner_tidx: Int32,
):
    """Load one M64xN128 FP32 score tile from TMEM."""

    relative_score = _zero_based_tmem_tensor(Float32, score_template.layout)
    load_atom = cute.make_copy_atom(
        tcgen05.copy.Ld16x64bOp(tcgen05.copy.Repetition(64)),
        Float32,
    )
    tiled_load = tcgen05.make_tmem_copy(load_atom, relative_score)
    thread_load = tiled_load.get_slice(owner_tidx)
    source_relative = thread_load.partition_S(relative_score)
    source = _add_physical_tmem_base(source_relative, tmem_base + score_offset)
    coordinates = thread_load.partition_D(
        thr_mma_qk.partition_C(cute.make_identity_tensor((M, N_HALF)))
    )
    scores = cute.make_rmem_tensor(coordinates.shape, Float32)
    cute.copy(tiled_load, source, scores)
    tcgen05_wait_ld()
    cute.arch.fence_view_async_tmem_load()
    return scores, coordinates


@cute.jit
def _rescale_m64_partial_o(
    o_template: cute.Tensor,
    thr_mma_pv: cute.ThrMma,
    tmem_base: Int32,
    o_offset: Int32,
    owner_tidx: Int32,
    alpha: Float32,
):
    """Rescale the prior M64 output accumulator before its next PV update."""

    relative_o = _zero_based_tmem_tensor(Float32, o_template.layout)
    correction_width = 16
    relative_fragment = cute.composition(relative_o, cute.make_layout((M, correction_width)))
    load_atom = cute.make_copy_atom(tcgen05.copy.Ld16x64bOp(tcgen05.copy.Repetition(8)), Float32)
    store_atom = cute.make_copy_atom(tcgen05.copy.St16x64bOp(tcgen05.copy.Repetition(8)), Float32)
    thread_load = tcgen05.make_tmem_copy(load_atom, relative_fragment).get_slice(owner_tidx)
    thread_store = tcgen05.make_tmem_copy(store_atom, relative_fragment).get_slice(owner_tidx)
    source = _add_physical_tmem_base(
        thread_load.partition_S(relative_fragment), tmem_base + o_offset
    )
    destination = _add_physical_tmem_base(
        thread_store.partition_D(relative_fragment), tmem_base + o_offset
    )
    for fragment_idx in cutlass.range_constexpr(DV // correction_width):
        registers = cute.make_rmem_tensor(thread_load.partition_D(relative_fragment).shape, Float32)
        source_i = cute.make_tensor(
            source.iterator + fragment_idx * correction_width, source.layout
        )
        cute.copy(thread_load, source_i, registers)
        tcgen05_wait_ld()
        cute.arch.fence_view_async_tmem_load()
        for i in cutlass.range(cute.size(registers), unroll_full=True):
            registers[i] = Float32(registers[i]) * Float32(alpha)
        destination_i = cute.make_tensor(
            destination.iterator + fragment_idx * correction_width,
            destination.layout,
        )
        cute.copy(thread_store, registers, destination_i)
        tcgen05_wait_st()
    cute.arch.fence_view_async_tmem_store()


@cute.jit
def _online_update_one_half(
    scores: cute.Tensor,
    running_max: Float32,
    running_sum: Float32,
    softmax_scale: Float32,
):
    """Apply one FP32 online-softmax update to an M64xN128 score tile."""

    local_max = fa_utils.fmax_reduce(scores.load(), arch=100)
    local_max = Float32(local_max) * softmax_scale
    peer_max = cute.arch.shuffle_sync_bfly(local_max, offset=2)
    transaction_max = local_max
    if peer_max > transaction_max:
        transaction_max = peer_max
    new_max = running_max
    if running_max == -Float32.inf or transaction_max > running_max:
        new_max = transaction_max
    alpha = Float32(0.0)
    if running_max != -Float32.inf:
        alpha = cute.math.exp2((running_max - new_max) * Float32(LOG2E), fastmath=True)
    probabilities = cute.make_rmem_tensor(scores.shape, Float32)
    for i in cutlass.range(cute.size(scores), unroll_full=True):
        probabilities[i] = cute.math.exp2(
            Float32(scores[i]) * softmax_scale * Float32(LOG2E) - new_max * Float32(LOG2E),
            fastmath=True,
        )
    transaction_sum = fa_utils.fadd_reduce(probabilities.load(), arch=100)
    transaction_sum += cute.arch.shuffle_sync_bfly(transaction_sum, offset=2)
    new_sum = running_sum * alpha + transaction_sum
    return probabilities, new_max, new_sum, alpha


__all__ = [
    "_load_m64_n128_score",
    "_online_update_one_half",
    "_rescale_m64_partial_o",
]
