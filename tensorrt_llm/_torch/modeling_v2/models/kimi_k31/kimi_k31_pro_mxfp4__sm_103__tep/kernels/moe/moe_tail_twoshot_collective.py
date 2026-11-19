# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Two-shot Lamport collective of the fused latent-MoE tail.

Same contract and numerics as ``CollectiveKernel`` (routed AllReduce +
RMSNorm, shared ReduceScatter; rank-order FP32 sums), but each token's routed
row is reduced once, on its owner rank ``token % tp``, instead of on every
rank: per-rank ingress is about ``2 * M * latent`` (the owners' inputs plus
the normalized rows) instead of ``tp * M * latent``. One CTA per token,
``latent / 8`` threads:

1. push this rank's routed row to the owner's slot and each shared column
   shard to that shard's rank (unicast, BF16 -0 is the empty marker);
2. clear this rank's buffers of the previous launch;
3. owner: wait for every rank's routed row, sum, RMSNorm, multicast the row
   into every rank's latent mailbox;
4. sum the shared slices pushed to this rank;
5. wait for the token's normalized row in the local mailbox and copy it out.

Buffers rotate over three generations: a launch clears the previous launch's
generation, which no rank writes again until two launches later. A CTA only
depends on the same token's CTAs on the other ranks, so the grid needs no
co-residency.

With ``shared_pushed`` the shared column shards of step 1 were stored earlier by
:class:`TwoShotSharedPush` (on the shared expert's stream, while the routed experts
run): the stores and the sums are the same, only step 1's shared half leaves the
critical path. The push targets the generation of the next collective call, which
its rotation already cleared on every rank (see :class:`TwoShotSharedPush`).

With ``top_k > 0`` the routed input is the routed experts' unfinalized output
(``UnfinalizedRoutedOutput``): step 1 computes this rank's row as the
routing-weighted sum of the token's local FC2 rows (FP32, slot order, one BF16
rounding: the routed-expert finalize's numerics) instead of reading a
finalized row.

Above 16 ranks (attention TP 32) steps 3 and 4 wait for and sum the ranks' slices
16 ranks at a time (bounded registers, as the MNNVL all-reduce polls), with the
sum orders fixed at construction from the rows the instance serves (``min_m``):
with an MNNVL plan the routed rows and the shared slices both reduce in the MNNVL
two-shot all-reduce's rank chunks (the unfused path's two all-reduces at those
rows), without one in rank order.
"""

from __future__ import annotations

from typing import Any

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64, Uint32

from .moe_tail_mnnvl_norm import MnnvlNormPlan, rank_sum, row_square_sum, square_sum_slot
from .moe_tail_primitives import (
    VEC_BF16,
    UnfinalizedRoutedOutput,
    atom_add_acq_rel_gpu_u32,
    bf16x8_to_packed_u32x4,
    finalize_top16_bf16,
    fragment_is_dirty,
    load_global_u32x4,
    load_u32_addr,
    load_u32x4_addr,
    packed_u32x4_to_bf16x8,
    sanitize_negative_zero,
    stmc_bf16x8,
    store_global_u32x4,
    store_u32_addr,
    to_cute,
    to_cute_dynamic_m,
    warp_sum_specialized,
)

NUM_GENERATIONS = 3
NEG_ZERO_F32_BITS = 0x80000000
# Ranks whose slices one wait loop holds in registers (the MNNVL two-shot all-reduce's rank chunk).
POLL_RANKS = 16


def owned_rows(max_m: int, tp_size: int) -> int:
    return (max_m + tp_size - 1) // tp_size


def rank_sum_chunks(
    *, tp_size: int, latent_dim: int, hidden_dim: int, min_m: int, norm_plan: MnnvlNormPlan | None
) -> tuple[int, int]:
    """(routed, shared) rank chunks of the polled sums above :data:`POLL_RANKS` ranks: each chunk of ranks summed
    from zero in rank order, the chunk sums added in order. With ``norm_plan`` both follow the MNNVL two-shot
    all-reduce, which reduces every row the instance serves (``min_m`` and up) for the routed latent and the shared
    output alike; without one, plain rank order. Up to :data:`POLL_RANKS` ranks the kernel keeps its all-rank wait."""
    if tp_size <= POLL_RANKS:
        return tp_size, tp_size
    if tp_size % POLL_RANKS:
        raise ValueError(
            f"the two-shot tail collective polls {POLL_RANKS} ranks at a time, got {tp_size}"
        )
    if norm_plan is None:
        return tp_size, tp_size
    from .moe_tail_mnnvl_norm import mnnvl_one_shot_threshold_bytes

    threshold = mnnvl_one_shot_threshold_bytes()
    for dim, rows_name in ((latent_dim, "routed latent"), (hidden_dim, "shared output")):
        if min_m * dim * tp_size * 2 <= threshold:
            raise ValueError(
                f"{tp_size} ranks: the {rows_name} all-reduce of {min_m} rows is one-shot; the polled sum follows "
                "the two-shot order only"
            )
    return norm_plan.twoshot_rank_chunk, norm_plan.twoshot_rank_chunk


class TwoShotAllReduceRMSNormReduceScatter:
    def __init__(
        self,
        *,
        rank: int,
        tp_size: int,
        latent_dim: int,
        hidden_dim: int,
        max_m: int,
        top_k: int = 0,
        shared_pushed: bool = False,
        norm_plan: MnnvlNormPlan | None = None,
        min_m: int = 1,
    ) -> None:
        if latent_dim % (VEC_BF16 * 32) or hidden_dim % (tp_size * VEC_BF16):
            raise ValueError("unsupported two-shot tail collective shape")
        if top_k not in (0, 16) or (top_k and latent_dim != 3584):
            raise ValueError("the top-k finalize sums 16 routes of a 3584-wide latent")
        self.top_k = top_k
        self.shared_pushed = shared_pushed
        self.rank = rank
        self.tp_size = tp_size
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.max_m = max_m
        self.owned_rows = owned_rows(max_m, tp_size)
        self.threads = latent_dim // VEC_BF16
        if not 32 <= self.threads <= 1024:
            raise ValueError("latent_dim / 8 must be in [32, 1024]")
        self.warps = self.threads // 32
        self.shard_dim = hidden_dim // tp_size
        self.shard_vectors = self.shard_dim // VEC_BF16
        self.hidden_vectors = hidden_dim // VEC_BF16
        if self.shard_vectors > self.threads:
            raise ValueError("shared shard wider than one CTA")
        # Elements per generation of each symmetric region.
        self.rs_generation = self.owned_rows * tp_size * latent_dim
        self.ss_generation = max_m * hidden_dim
        self.lat_generation = max_m * latent_dim
        # With a plan, the routed rank sum and RMSNorm reproduce the MNNVL all-reduce RMSNorm
        # fusion bit for bit (see ``mnnvl_norm``); the row layout follows the row count.
        self.norm_plan = norm_plan
        # Above POLL_RANKS ranks: the rank chunks of the polled routed and shared sums.
        self.poll_ranks = min(tp_size, POLL_RANKS)
        self.routed_chunk, self.shared_chunk = rank_sum_chunks(
            tp_size=tp_size,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            min_m=min_m,
            norm_plan=norm_plan,
        )

    @cute.jit
    def __call__(
        self,
        routed: cute.Tensor,
        shared: cute.Tensor,
        gamma: cute.Tensor,
        latent_out: cute.Tensor,
        shared_out: cute.Tensor,
        ctrl: cute.Tensor,
        rs_peers: cute.Tensor,
        ss_peers: cute.Tensor,
        lat_uc: Int64,
        lat_mc: Int64,
        m: Int32,
        epsilon: Float32,
        expert_weights: cute.Tensor,
        expanded_idx_to_permuted_idx: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.kernel(
            routed,
            shared,
            gamma,
            latent_out,
            shared_out,
            ctrl,
            rs_peers,
            ss_peers,
            lat_uc,
            lat_mc,
            m,
            epsilon,
            expert_weights,
            expanded_idx_to_permuted_idx,
        ).launch(
            grid=(m, 1, 1),
            block=(self.threads, 1, 1),
            smem=(self.norm_plan.slots if self.norm_plan is not None else self.warps) * 4,
            stream=stream,
            use_pdl=True,
        )

    @cute.jit
    def _routed_fragment(
        self,
        routed: cute.Tensor,
        token: Int32,
        vector: Int32,
        expert_weights: cute.Tensor,
        expanded_idx_to_permuted_idx: cute.Tensor,
    ):
        """This rank's sanitized 8-wide BF16 fragment ``vector`` of the token's routed row."""
        packed = None
        if cutlass.const_expr(self.top_k > 0):
            packed = finalize_top16_bf16(
                cute.make_ptr(
                    BFloat16,
                    (routed.iterator + Int64(vector) * VEC_BF16).llvm_ptr,
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                ),
                cute.make_ptr(
                    Int32,
                    (expanded_idx_to_permuted_idx.iterator + Int64(token) * self.top_k).llvm_ptr,
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                ),
                cute.make_ptr(
                    BFloat16,
                    (expert_weights.iterator + Int64(token) * self.top_k).llvm_ptr,
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                ),
            )
        else:
            packed = load_global_u32x4(
                cute.make_ptr(
                    BFloat16,
                    (
                        routed.iterator + Int64(token) * self.latent_dim + Int64(vector) * VEC_BF16
                    ).llvm_ptr,
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                ),
                volatile=False,
            )
        return sanitize_negative_zero(packed)

    @cute.jit
    def _clear(self, address: Int64):
        sentinel = cute.make_rmem_tensor(cute.make_layout((4,)), Uint32)
        for word in cutlass.range_constexpr(4):
            sentinel[word] = Uint32(NEG_ZERO_F32_BITS)
        store_global_u32x4(address, sentinel.load())

    @cute.jit
    def _wait_rank_sum(
        self,
        accum: cute.Tensor,
        base: Int64,
        rank_stride: cutlass.Constexpr,
        chunk: cutlass.Constexpr,
    ):
        """``accum`` = the FP32 sum of every rank's BF16 fragment at ``base + rank * rank_stride`` (bytes), each
        chunk of ``chunk`` ranks summed from zero in rank order and the chunk sums added in order; waits for
        ``poll_ranks`` ranks' fragments at a time."""
        part = cute.make_rmem_tensor(cute.make_layout((VEC_BF16,)), Float32)
        for batch in cutlass.range_constexpr(self.tp_size // self.poll_ranks):
            words = cute.make_rmem_tensor(
                cute.make_layout((self.poll_ranks, 4), stride=(4, 1)), Uint32
            )
            valid = False
            while not valid:
                valid = True
                for i in cutlass.range_constexpr(self.poll_ranks):
                    packed = load_u32x4_addr(
                        base + (batch * self.poll_ranks + i) * rank_stride, volatile=True
                    )
                    for word in cutlass.range_constexpr(4):
                        words[i, word] = packed[word]
                    valid = valid & (not fragment_is_dirty(packed))
            for i in cutlass.range_constexpr(self.poll_ranks):
                source = batch * self.poll_ranks + i
                values = packed_u32x4_to_bf16x8(words[i, None].load()).to(Float32)
                if cutlass.const_expr(source % chunk == 0):
                    for element in cutlass.range_constexpr(VEC_BF16):
                        part[element] = Float32(0.0)
                for element in cutlass.range_constexpr(VEC_BF16):
                    part[element] = part[element] + values[element]
                if cutlass.const_expr(source % chunk == chunk - 1):
                    for element in cutlass.range_constexpr(VEC_BF16):
                        if cutlass.const_expr(source == chunk - 1):
                            accum[element] = part[element]
                        else:
                            accum[element] = accum[element] + part[element]

    @cute.kernel
    def kernel(
        self,
        routed: cute.Tensor,
        shared: cute.Tensor,
        gamma: cute.Tensor,
        latent_out: cute.Tensor,
        shared_out: cute.Tensor,
        ctrl: cute.Tensor,
        rs_peers: cute.Tensor,
        ss_peers: cute.Tensor,
        lat_uc: Int64,
        lat_mc: Int64,
        m: Int32,
        epsilon: Float32,
        expert_weights: cute.Tensor,
        expanded_idx_to_permuted_idx: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        token, _, _ = cute.arch.block_idx()
        lane = cute.arch.lane_idx()
        warp_idx = cute.arch.warp_idx()
        L = self.latent_dim
        H = self.hidden_dim
        S = self.shard_dim
        tp = self.tp_size
        smem = cutlass.utils.SmemAllocator()
        sums_size = self.norm_plan.slots if self.norm_plan is not None else self.warps
        sums = smem.allocate_tensor(Float32, cute.make_layout((sums_size,)), byte_alignment=16)

        cute.arch.griddepcontrol_wait()
        ctrl_addr = Int64(ctrl.iterator.toint())
        epoch = load_u32_addr(ctrl_addr)
        prev_m = Int32(load_u32_addr(ctrl_addr + 4))
        generation = Int64(epoch % Uint32(NUM_GENERATIONS))
        prev_generation = Int64((epoch + Uint32(NUM_GENERATIONS - 1)) % Uint32(NUM_GENERATIONS))
        owner = token % Int32(tp)
        owned_row = Int64(token // Int32(tp))
        my_rs = cute.arch.load((rs_peers.iterator + self.rank).llvm_ptr, Int64)
        my_ss = cute.arch.load((ss_peers.iterator + self.rank).llvm_ptr, Int64)
        my_lat = lat_uc

        # 1. push: routed row -> owner's slot [gen, owned_row, rank]
        owner_rs = cute.arch.load((rs_peers.iterator + owner).llvm_ptr, Int64)
        rs_slot = (
            generation * self.rs_generation
            + (owned_row * tp + self.rank) * L
            + Int64(tidx) * VEC_BF16
        )
        store_global_u32x4(
            owner_rs + rs_slot * 2,
            self._routed_fragment(
                routed, token, tidx, expert_weights, expanded_idx_to_permuted_idx
            ),
        )
        #    shared column shard d of the row -> rank d's slot [gen, token, rank] (unless pushed earlier)
        vector = tidx
        if cutlass.const_expr(self.shared_pushed):
            vector = Int32(self.hidden_vectors)
        while vector < self.hidden_vectors:
            destination = vector // Int32(self.shard_vectors)
            column = vector - destination * Int32(self.shard_vectors)
            shared_ptr = cute.make_ptr(
                BFloat16,
                (shared.iterator + Int64(token) * H + Int64(vector) * VEC_BF16).llvm_ptr,
                cute.AddressSpace.gmem,
                assumed_align=16,
            )
            peer_ss = cute.arch.load((ss_peers.iterator + destination).llvm_ptr, Int64)
            ss_slot = (
                generation * self.ss_generation
                + Int64(token) * H
                + Int64(self.rank) * S
                + Int64(column) * VEC_BF16
            )
            store_global_u32x4(
                peer_ss + ss_slot * 2,
                sanitize_negative_zero(load_global_u32x4(shared_ptr, volatile=False)),
            )
            vector = vector + self.threads
        cute.arch.griddepcontrol_launch_dependents()

        # 2. clear this rank's previous generation (rows < prev_m)
        row = token
        while row < prev_m:
            lat_prev = (prev_generation * self.lat_generation + Int64(row) * L) * 2
            self._clear(my_lat + lat_prev + Int64(tidx) * 16)
            vector = tidx
            while vector < self.hidden_vectors:
                ss_prev = (prev_generation * self.ss_generation + Int64(row) * H) * 2
                self._clear(my_ss + ss_prev + Int64(vector) * 16)
                vector = vector + self.threads
            if row % Int32(tp) == Int32(self.rank):
                rs_prev = (
                    prev_generation * self.rs_generation + Int64(row // Int32(tp)) * tp * L
                ) * 2
                for source in cutlass.range_constexpr(tp):
                    self._clear(my_rs + rs_prev + (Int64(source) * L + Int64(tidx) * VEC_BF16) * 2)
            row = row + m

        # 3. owner: routed AllReduce (rank order, FP32) + RMSNorm, multicast
        if owner == Int32(self.rank):
            accum = cute.make_rmem_tensor(cute.make_layout((VEC_BF16,)), Float32)
            if cutlass.const_expr(self.poll_ranks < tp):
                self._wait_rank_sum(
                    accum,
                    my_rs
                    + (
                        generation * self.rs_generation
                        + owned_row * tp * L
                        + Int64(tidx) * VEC_BF16
                    )
                    * 2,
                    L * 2,
                    self.routed_chunk,
                )
            else:
                words = cute.make_rmem_tensor(cute.make_layout((tp, 4), stride=(4, 1)), Uint32)
                valid = False
                while not valid:
                    valid = True
                    for source in cutlass.range_constexpr(tp):
                        slot = (
                            generation * self.rs_generation
                            + (owned_row * tp + source) * L
                            + Int64(tidx) * VEC_BF16
                        )
                        packed = load_u32x4_addr(my_rs + slot * 2, volatile=True)
                        for word in cutlass.range_constexpr(4):
                            words[source, word] = packed[word]
                        valid = valid & (not fragment_is_dirty(packed))
                if cutlass.const_expr(self.norm_plan is not None):
                    plan = self.norm_plan
                    if cutlass.const_expr(plan.oneshot_rank_chunk == plan.twoshot_rank_chunk):
                        rank_sum(accum, words, tp, plan.oneshot_rank_chunk)
                    elif m <= Int32(plan.oneshot_max_rows):
                        rank_sum(accum, words, tp, plan.oneshot_rank_chunk)
                    else:
                        rank_sum(accum, words, tp, plan.twoshot_rank_chunk)
                else:
                    for element in cutlass.range_constexpr(VEC_BF16):
                        accum[element] = Float32(0.0)
                    for source in cutlass.range_constexpr(tp):
                        values = packed_u32x4_to_bf16x8(words[source, None].load()).to(Float32)
                        for element in cutlass.range_constexpr(VEC_BF16):
                            accum[element] = accum[element] + values[element]
            norm_input_bf16 = accum.load().to(BFloat16)
            norm_input = norm_input_bf16.to(Float32)
            # The MNNVL RMSNorm fusion squares the BF16-rounded sum in BF16.
            norm_square = (norm_input_bf16 * norm_input_bf16).to(Float32)
            full_sum = Float32(0.0)
            if cutlass.const_expr(self.norm_plan is not None):
                # Zero every slot (the layout's padding lanes stay zero), stage the thread's square
                # sum at its MNNVL (CTA, warp, lane) slot for this row count's layout, and let each
                # warp reduce the row in the MNNVL kernel's order.
                slot = tidx
                while slot < self.norm_plan.slots:
                    sums[slot] = Float32(0.0)
                    slot = slot + self.threads
                thread_sum = Float32(0.0)
                for element in cutlass.range_constexpr(VEC_BF16):
                    thread_sum = thread_sum + norm_square[element]
                cute.arch.barrier()
                for layout in cutlass.range_constexpr(len(self.norm_plan.geometries)):
                    first = 1 if layout == 0 else self.norm_plan.geometries[layout - 1][0] + 1
                    last, block, cluster = self.norm_plan.geometries[layout]
                    if m >= Int32(first) and m <= Int32(last):
                        sums[square_sum_slot(tidx, block, cluster)] = thread_sum
                        cute.arch.barrier()
                        full_sum = row_square_sum(sums, Int32(0), lane, block, cluster)
            else:
                thread_sum = norm_square.reduce(
                    cute.ReductionOp.ADD, init_val=Float32(0.0), reduction_profile=0
                )
                warp_sum = warp_sum_specialized(thread_sum, warp_idx, lane, self.warps, 32, -1)
                if lane == 0:
                    sums[warp_idx] = warp_sum
                cute.arch.barrier()
                for w in cutlass.range_constexpr(self.warps):
                    full_sum = full_sum + sums[w]
            inv_rms = cute.math.rsqrt(full_sum / Float32(L) + epsilon, fastmath=True)
            gamma_ptr = cute.make_ptr(
                BFloat16,
                (gamma.iterator + Int64(tidx) * VEC_BF16).llvm_ptr,
                cute.AddressSpace.gmem,
                assumed_align=16,
            )
            gamma_values = packed_u32x4_to_bf16x8(load_global_u32x4(gamma_ptr, volatile=False))
            if cutlass.const_expr(self.norm_plan is None):
                result = (norm_input * inv_rms * gamma_values.to(Float32)).to(BFloat16)
            else:
                # One-shot rows: (x * rstd) * w; two-shot rows: (w * x) * rstd.
                normalized = cute.make_rmem_tensor(cute.make_layout((VEC_BF16,)), BFloat16)
                if m <= Int32(self.norm_plan.oneshot_max_rows):
                    normalized.store((norm_input * inv_rms * gamma_values.to(Float32)).to(BFloat16))
                else:
                    normalized.store((gamma_values.to(Float32) * norm_input * inv_rms).to(BFloat16))
                result = normalized.load()
            stmc_bf16x8(
                lat_mc
                + (generation * self.lat_generation + Int64(token) * L + Int64(tidx) * VEC_BF16)
                * 2,
                sanitize_negative_zero(bf16x8_to_packed_u32x4(result)),
            )

        # 4. this rank's shared columns: sum the slices of every rank
        if tidx < self.shard_vectors:
            accum = cute.make_rmem_tensor(cute.make_layout((VEC_BF16,)), Float32)
            if cutlass.const_expr(self.poll_ranks < tp):
                self._wait_rank_sum(
                    accum,
                    my_ss
                    + (generation * self.ss_generation + Int64(token) * H + Int64(tidx) * VEC_BF16)
                    * 2,
                    S * 2,
                    self.shared_chunk,
                )
            else:
                words = cute.make_rmem_tensor(cute.make_layout((tp, 4), stride=(4, 1)), Uint32)
                valid = False
                while not valid:
                    valid = True
                    for source in cutlass.range_constexpr(tp):
                        slot = (
                            generation * self.ss_generation
                            + Int64(token) * H
                            + Int64(source) * S
                            + Int64(tidx) * VEC_BF16
                        )
                        packed = load_u32x4_addr(my_ss + slot * 2, volatile=True)
                        for word in cutlass.range_constexpr(4):
                            words[source, word] = packed[word]
                        valid = valid & (not fragment_is_dirty(packed))
                for element in cutlass.range_constexpr(VEC_BF16):
                    accum[element] = Float32(0.0)
                for source in cutlass.range_constexpr(tp):
                    values = packed_u32x4_to_bf16x8(words[source, None].load()).to(Float32)
                    for element in cutlass.range_constexpr(VEC_BF16):
                        accum[element] = accum[element] + values[element]
            store_global_u32x4(
                Int64(
                    (
                        shared_out.iterator
                        + Int64(token) * H
                        + Int64(self.rank) * S
                        + Int64(tidx) * VEC_BF16
                    ).toint()
                ),
                bf16x8_to_packed_u32x4(accum.load().to(BFloat16)),
            )

        # 5. the normalized row from its owner
        mailbox = my_lat + (generation * self.lat_generation + Int64(token) * L) * 2
        mailbox = mailbox + Int64(tidx) * 16
        packed = load_u32x4_addr(mailbox, volatile=True)
        while fragment_is_dirty(packed):
            packed = load_u32x4_addr(mailbox, volatile=True)
        store_global_u32x4(
            Int64((latent_out.iterator + Int64(token) * L + Int64(tidx) * VEC_BF16).toint()),
            packed,
        )

        # 6. the last CTA advances the generation (every CTA read ctrl at its start). Every warp has read the
        # epoch and prev_m before thread 0 counts the CTA, so none reads the next call's control words.
        cute.arch.barrier()
        if tidx == 0:
            arrived = atom_add_acq_rel_gpu_u32(ctrl_addr + 8, Uint32(1))
            if arrived == Uint32(m) - Uint32(1):
                store_u32_addr(ctrl_addr + 8, Uint32(0))
                store_u32_addr(ctrl_addr + 4, Uint32(m))
                store_u32_addr(ctrl_addr, epoch + Uint32(1))


class TwoShotSharedPush:
    """Step 1's shared half of :class:`TwoShotAllReduceRMSNormReduceScatter` as its own kernel.

    One CTA per token stores this rank's shared row, column shard ``d`` into rank ``d``'s slot
    ``[gen, token, rank]`` (sanitized, the same stores as the collective's step 1), where ``gen`` is
    the generation of the NEXT collective call on the workspace (read from ``ctrl``, which only that
    call advances). Launched on the shared expert's stream after its down projection, so the
    shared reduce-scatter's bytes travel while the routed experts run; that collective call then
    runs with ``shared_pushed`` (sums the slots, pushes nothing).

    Ordering: the stream must order this kernel after the previous collective call on this rank
    (its ``ctrl`` update) and before the next one. Every rank's slot of generation ``gen`` was
    cleared two calls ago: this rank reached the next call only after the previous call received
    every rank's pushes, so every rank completed the call before it, which cleared ``gen``.
    """

    def __init__(self, *, rank: int, tp_size: int, hidden_dim: int, max_m: int) -> None:
        if hidden_dim % (tp_size * VEC_BF16):
            raise ValueError("unsupported two-shot shared push shape")
        self.rank = rank
        self.tp_size = tp_size
        self.hidden_dim = hidden_dim
        self.max_m = max_m
        self.shard_dim = hidden_dim // tp_size
        self.shard_vectors = self.shard_dim // VEC_BF16
        self.hidden_vectors = hidden_dim // VEC_BF16
        self.threads = min(self.hidden_vectors, 512)
        while self.hidden_vectors % self.threads:
            self.threads -= 32
        self.ss_generation = max_m * hidden_dim

    @cute.jit
    def __call__(
        self,
        shared: cute.Tensor,
        ctrl: cute.Tensor,
        ss_peers: cute.Tensor,
        m: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(shared, ctrl, ss_peers).launch(
            grid=(m, 1, 1),
            block=(self.threads, 1, 1),
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(self, shared: cute.Tensor, ctrl: cute.Tensor, ss_peers: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        token, _, _ = cute.arch.block_idx()
        H = self.hidden_dim
        S = self.shard_dim
        cute.arch.griddepcontrol_wait()
        epoch = load_u32_addr(Int64(ctrl.iterator.toint()))
        generation = Int64(epoch % Uint32(NUM_GENERATIONS))
        vector = tidx
        while vector < self.hidden_vectors:
            destination = vector // Int32(self.shard_vectors)
            column = vector - destination * Int32(self.shard_vectors)
            shared_ptr = cute.make_ptr(
                BFloat16,
                (shared.iterator + Int64(token) * H + Int64(vector) * VEC_BF16).llvm_ptr,
                cute.AddressSpace.gmem,
                assumed_align=16,
            )
            peer_ss = cute.arch.load((ss_peers.iterator + destination).llvm_ptr, Int64)
            ss_slot = (
                generation * self.ss_generation
                + Int64(token) * H
                + Int64(self.rank) * S
                + Int64(column) * VEC_BF16
            )
            store_global_u32x4(
                peer_ss + ss_slot * 2,
                sanitize_negative_zero(load_global_u32x4(shared_ptr, volatile=False)),
            )
            vector = vector + self.threads
        cute.arch.griddepcontrol_launch_dependents()


_COMPILED: dict[tuple[object, ...], Any] = {}


class TwoShotCollectiveKernel:
    """Owns the compiled two-shot collective; the caller owns the symmetric memory.

    ``rs_peers`` is an int64 tensor ``[tp]`` of every rank's address of the
    routed slot region ``[3, ceil(max_m / tp), tp, latent]``, ``ss_peers``
    ``[tp]`` of the shared slot region ``[3, max_m, hidden]``; ``lat_uc`` /
    ``lat_mc`` are this rank's and the multicast address of the latent mailbox
    ``[3, max_m, latent]``. All regions BF16 filled with the Lamport sentinel;
    ``ctrl`` a local ``[3]`` uint32 zero tensor.
    """

    def __init__(
        self,
        *,
        rank: int,
        tp_size: int,
        latent_dim: int,
        hidden_dim: int,
        max_m: int,
        rms_eps: float,
        rs_peers: torch.Tensor,
        ss_peers: torch.Tensor,
        lat_uc: int,
        lat_mc: int,
        ctrl: torch.Tensor,
        top_k: int = 0,
        norm_plan: MnnvlNormPlan | None = None,
        min_m: int = 1,
    ) -> None:
        """``top_k > 0``: the routed input is an ``UnfinalizedRoutedOutput``
        (finalized on load, see ``TwoShotAllReduceRMSNormReduceScatter``). With ``norm_plan`` the
        routed sum and RMSNorm reproduce the MNNVL all-reduce RMSNorm fusion's numerics. ``min_m``: the
        fewest rows a call takes (above 16 ranks it fixes the sum orders, see ``rank_sum_chunks``)."""
        if not 1 <= min_m <= max_m:
            raise ValueError(f"min_m={min_m} must be in [1, {max_m}]")
        self.top_k = top_k
        self.norm_plan = norm_plan
        self.min_m = min_m
        self.rank = rank
        self.tp_size = tp_size
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.shard_dim = hidden_dim // tp_size
        self.max_m = max_m
        self.rms_eps = float(rms_eps)
        self._rs_peers = rs_peers
        self._ss_peers = ss_peers
        self._lat_uc = int(lat_uc)
        self._lat_mc = int(lat_mc)
        self._ctrl = ctrl
        device = ctrl.device
        self._latent_output = torch.empty((max_m, latent_dim), dtype=torch.bfloat16, device=device)
        self._shared_output = torch.empty((max_m, hidden_dim), dtype=torch.bfloat16, device=device)
        start = rank * self.shard_dim
        self._shared_shard = self._shared_output[:, start : start + self.shard_dim]
        routes = max(top_k, 1)
        self._dummy_expert_weights = torch.zeros(
            (max_m, routes), dtype=torch.bfloat16, device=device
        )
        self._dummy_expanded_idx = torch.zeros((max_m, routes), dtype=torch.int32, device=device)
        self._compiled = self._compile(shared_pushed=False)
        # The collective of a call whose shared shards were pushed earlier, and the push (compiled here: a call
        # may come first under CUDA-graph capture).
        self._compiled_shared_pushed = self._compile(shared_pushed=True)
        self._compiled_push = self._compile_push()

    def _compile(self, *, shared_pushed: bool):
        key = (
            torch.cuda.current_device(),
            self.rank,
            self.tp_size,
            self.latent_dim,
            self.hidden_dim,
            self.max_m,
            self.top_k,
            shared_pushed,
            self.norm_plan,
            self.min_m,
            POLL_RANKS,
        )
        if key not in _COMPILED:
            kernel = TwoShotAllReduceRMSNormReduceScatter(
                rank=self.rank,
                tp_size=self.tp_size,
                latent_dim=self.latent_dim,
                hidden_dim=self.hidden_dim,
                max_m=self.max_m,
                top_k=self.top_k,
                shared_pushed=shared_pushed,
                norm_plan=self.norm_plan,
                min_m=self.min_m,
            )
            device = self._ctrl.device
            routed = torch.empty((self.max_m, self.latent_dim), dtype=torch.bfloat16, device=device)
            shared = torch.empty((self.max_m, self.hidden_dim), dtype=torch.bfloat16, device=device)
            gamma = torch.empty((self.latent_dim,), dtype=torch.bfloat16, device=device)
            _COMPILED[key] = cute.compile(
                kernel,
                *self._args(
                    routed,
                    shared,
                    gamma,
                    1,
                    self._dummy_expert_weights,
                    self._dummy_expanded_idx,
                ),
            )
        return _COMPILED[key]

    def _compile_push(self):
        key = (
            "push",
            torch.cuda.current_device(),
            self.rank,
            self.tp_size,
            self.hidden_dim,
            self.max_m,
        )
        if key not in _COMPILED:
            kernel = TwoShotSharedPush(
                rank=self.rank, tp_size=self.tp_size, hidden_dim=self.hidden_dim, max_m=self.max_m
            )
            probe = torch.empty(
                (self.max_m, self.hidden_dim), dtype=torch.bfloat16, device=self._ctrl.device
            )
            _COMPILED[key] = cute.compile(
                kernel,
                to_cute_dynamic_m(probe, mode=0, assumed_align=16),
                to_cute(self._ctrl, 16),
                to_cute(self._ss_peers, 16),
                Int32(1),
                cuda.CUstream(torch.cuda.current_stream().cuda_stream),
            )
        return _COMPILED[key]

    def push_shared(self, shared: torch.Tensor) -> None:
        """Store this rank's ``[m, hidden]`` shared partial rows into the shard owners' slots of the next call
        (:class:`TwoShotSharedPush`); that call must then pass ``shared_pushed=True``."""
        m = shared.shape[0]
        if not 1 <= m <= self.max_m or shared.dtype != torch.bfloat16 or not shared.is_contiguous():
            raise ValueError("the shared push takes contiguous BF16 [1..max_m, hidden] rows")
        self._compiled_push(
            to_cute_dynamic_m(shared, mode=0, assumed_align=16),
            to_cute(self._ctrl, 16),
            to_cute(self._ss_peers, 16),
            Int32(m),
            cuda.CUstream(torch.cuda.current_stream().cuda_stream),
        )

    @staticmethod
    def region_bytes(*, max_m: int, tp_size: int, latent_dim: int, hidden_dim: int) -> dict:
        rows = owned_rows(max_m, tp_size)
        return {
            "rs": NUM_GENERATIONS * rows * tp_size * latent_dim * 2,
            "ss": NUM_GENERATIONS * max_m * hidden_dim * 2,
            "lat": NUM_GENERATIONS * max_m * latent_dim * 2,
        }

    def _args(self, routed, shared, gamma, m, expert_weights, expanded_idx):
        return (
            to_cute_dynamic_m(routed, mode=0, assumed_align=16),
            to_cute_dynamic_m(shared, mode=0, assumed_align=16),
            to_cute(gamma, 16),
            to_cute(self._latent_output, 16),
            to_cute(self._shared_output, 16),
            to_cute(self._ctrl, 16),
            to_cute(self._rs_peers, 16),
            to_cute(self._ss_peers, 16),
            Int64(self._lat_uc),
            Int64(self._lat_mc),
            Int32(m),
            Float32(self.rms_eps),
            to_cute_dynamic_m(expert_weights, mode=0, assumed_align=16),
            to_cute_dynamic_m(expanded_idx, mode=0, assumed_align=16),
            cuda.CUstream(torch.cuda.current_stream().cuda_stream),
        )

    def __call__(
        self,
        routed: torch.Tensor | UnfinalizedRoutedOutput,
        shared: torch.Tensor,
        gamma: torch.Tensor,
        shared_pushed: bool = False,
    ):
        """``shared_pushed``: :meth:`push_shared` already stored this call's shared rows (``shared`` is then only
        read for its shape)."""
        if isinstance(routed, UnfinalizedRoutedOutput):
            if self.top_k <= 0:
                raise ValueError("two-shot collective was not configured for top-k finalize")
            expert_weights = routed.expert_weights
            expanded_idx = routed.expanded_idx_to_permuted_idx
            m = expanded_idx.shape[0]
            routed = routed.gemm2_permuted
        else:
            if self.top_k > 0:
                raise ValueError("top-k two-shot collective requires an unfinalized output")
            m = routed.shape[0]
            expert_weights = self._dummy_expert_weights
            expanded_idx = self._dummy_expanded_idx
        if not self.min_m <= m <= self.max_m:
            raise ValueError(f"runtime M={m} must be in [{self.min_m}, {self.max_m}]")
        compiled = self._compiled_shared_pushed if shared_pushed else self._compiled
        compiled(*self._args(routed, shared, gamma, m, expert_weights, expanded_idx))
        return self._latent_output[:m], self._shared_shard
