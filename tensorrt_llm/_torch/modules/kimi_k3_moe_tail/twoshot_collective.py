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

With ``top_k > 0`` the routed input is the routed experts' unfinalized output
(``UnfinalizedRoutedOutput``): step 1 computes this rank's row as the
routing-weighted sum of the token's local FC2 rows (FP32, slot order, one BF16
rounding: the routed-expert finalize's numerics) instead of reading a
finalized row.
"""

from __future__ import annotations

from typing import Any

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64, Uint32

from .primitives import (
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


def owned_rows(max_m: int, tp_size: int) -> int:
    return (max_m + tp_size - 1) // tp_size


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
    ) -> None:
        if latent_dim % (VEC_BF16 * 32) or hidden_dim % (tp_size * VEC_BF16):
            raise ValueError("unsupported two-shot tail collective shape")
        self.top_k = top_k
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
            smem=self.warps * 4,
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
        if cutlass.const_expr(self.top_k == 16 and self.latent_dim == 3584):
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
        elif cutlass.const_expr(self.top_k > 0):
            values = cute.make_rmem_tensor(cute.make_layout((VEC_BF16,)), Float32)
            for element in cutlass.range_constexpr(VEC_BF16):
                values[element] = Float32(0.0)
            for slot in cutlass.range_constexpr(self.top_k):
                permuted_idx = expanded_idx_to_permuted_idx[token, slot]
                if permuted_idx >= Int32(0):
                    row_ptr = cute.make_ptr(
                        BFloat16,
                        (
                            routed.iterator
                            + Int64(permuted_idx) * self.latent_dim
                            + Int64(vector) * VEC_BF16
                        ).llvm_ptr,
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    )
                    row = packed_u32x4_to_bf16x8(load_global_u32x4(row_ptr, volatile=False))
                    weight = expert_weights[token, slot].to(Float32)
                    for element in cutlass.range_constexpr(VEC_BF16):
                        values[element] = values[element] + row[element].to(Float32) * weight
            packed = bf16x8_to_packed_u32x4(values.load().to(BFloat16))
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
        sums = smem.allocate_tensor(Float32, cute.make_layout((self.warps,)), byte_alignment=16)

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
        #    shared column shard d of the row -> rank d's slot [gen, token, rank]
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
            accum = cute.make_rmem_tensor(cute.make_layout((VEC_BF16,)), Float32)
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
            thread_sum = norm_square.reduce(
                cute.ReductionOp.ADD, init_val=Float32(0.0), reduction_profile=0
            )
            warp_sum = warp_sum_specialized(thread_sum, warp_idx, lane, self.warps, 32, -1)
            if lane == 0:
                sums[warp_idx] = warp_sum
            cute.arch.barrier()
            full_sum = Float32(0.0)
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
            result = (norm_input * inv_rms * gamma_values.to(Float32)).to(BFloat16)
            stmc_bf16x8(
                lat_mc
                + (generation * self.lat_generation + Int64(token) * L + Int64(tidx) * VEC_BF16)
                * 2,
                sanitize_negative_zero(bf16x8_to_packed_u32x4(result)),
            )

        # 4. this rank's shared columns: sum the slices of every rank
        if tidx < self.shard_vectors:
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
            accum = cute.make_rmem_tensor(cute.make_layout((VEC_BF16,)), Float32)
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

        # 6. the last CTA advances the generation (every CTA read ctrl at its start)
        if tidx == 0:
            arrived = atom_add_acq_rel_gpu_u32(ctrl_addr + 8, Uint32(1))
            if arrived == Uint32(m) - Uint32(1):
                store_u32_addr(ctrl_addr + 8, Uint32(0))
                store_u32_addr(ctrl_addr + 4, Uint32(m))
                store_u32_addr(ctrl_addr, epoch + Uint32(1))


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
    ) -> None:
        """``top_k > 0``: the routed input is an ``UnfinalizedRoutedOutput``
        (finalized on load, see ``TwoShotAllReduceRMSNormReduceScatter``)."""
        self.top_k = top_k
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
        self._kernel = TwoShotAllReduceRMSNormReduceScatter(
            rank=rank,
            tp_size=tp_size,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            max_m=max_m,
            top_k=top_k,
        )
        routes = max(top_k, 1)
        self._dummy_expert_weights = torch.zeros(
            (max_m, routes), dtype=torch.bfloat16, device=device
        )
        self._dummy_expanded_idx = torch.zeros((max_m, routes), dtype=torch.int32, device=device)
        key = (torch.cuda.current_device(), rank, tp_size, latent_dim, hidden_dim, max_m, top_k)
        if key not in _COMPILED:
            routed = torch.empty((max_m, latent_dim), dtype=torch.bfloat16, device=device)
            shared = torch.empty((max_m, hidden_dim), dtype=torch.bfloat16, device=device)
            gamma = torch.empty((latent_dim,), dtype=torch.bfloat16, device=device)
            _COMPILED[key] = cute.compile(
                self._kernel,
                *self._args(
                    routed,
                    shared,
                    gamma,
                    1,
                    self._dummy_expert_weights,
                    self._dummy_expanded_idx,
                ),
            )
        self._compiled = _COMPILED[key]

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
    ):
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
        if not 1 <= m <= self.max_m:
            raise ValueError(f"runtime M={m} must be in [1, {self.max_m}]")
        self._compiled(*self._args(routed, shared, gamma, m, expert_weights, expanded_idx))
        return self._latent_output[:m], self._shared_shard
