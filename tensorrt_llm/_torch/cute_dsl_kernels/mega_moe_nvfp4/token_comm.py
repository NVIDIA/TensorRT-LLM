# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Token communication implementations for MegaMoE-style kernels.

Current implementation: token-in pull with token-back push.  The standalone
``dispatch_kernel`` uses the same object methods as the fused MegaMoE kernel.
"""

from typing import Any, Dict, List

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass.cutlass_dsl import (Float32, Int32, Int64, Uint8, Uint32,
                                 extract_mlir_values, new_from_mlir_values)

# Keep these as separate handlers (NOT a tuple `except (A, B)`): CuteDSL's
# preprocessor import-walker (cutlass-dsl 4.5.0) raises AttributeError on
# tuple except types, which silently disables AST preprocessing for this
# module and breaks dynamic `if` control flow in the kernel.
try:
    from cutlass.cute import iket as _iket  # type: ignore
except ImportError:  # pragma: no cover
    from .iket_compat import iket as _iket
except NotImplementedError:  # pragma: no cover
    from .iket_compat import iket as _iket

from cutlass._mlir import ir

from .grid_sync import software_grid_sync
from .moe_utils import spin_wait
from .ptx_helpers import (fns_b32, ldg_b32_raw, ldg_f32_raw,
                          red_add_release_sys_s32_raw,
                          red_add_release_sys_u64_raw, stg_b32_raw, stg_b64_raw,
                          tma_load_1d_raw, tma_store_1d)
from .sf_swizzle import sf_atom_int32_offset


def _store_token_src_metadata_u32x3(
    token_src_metadata,
    pool_token_idx,
    src_rank: Uint32,
    src_token: Uint32,
    src_topk: Uint32,
) -> None:
    """Store `{src_rank, src_token, src_topk}` as three 32-bit fields."""
    base_ptr = token_src_metadata.iterator + (pool_token_idx * Int32(12))
    cute.arch.store(base_ptr, src_rank, scope="gpu")
    cute.arch.store(base_ptr + Int32(4), src_token, scope="gpu")
    cute.arch.store(base_ptr + Int32(8), src_topk, scope="gpu")


_MLIR_VALUE_FIELDS = (
    "input_token_buffer",
    "input_sf_buffer",
    "topk_idx",
    "input_topk_weights_buffer",
    "expert_send_count",
    "expert_recv_count",
    "expert_recv_count_sum",
    "src_token_topk_idx",
    "fc1_input_token_buffer",
    "fc1_input_sf_buffer",
    "fc1_input_topk_weights_buffer",
    "fc1_ready_counter",
    "token_src_metadata",
    "combine_output",
    "fc2_output_workspace",
    "fc2_done_counter",
    "nvlink_barrier_signal",
    "nvlink_barrier_counter",
    "grid_sync_counter",
    "peer_rank_ptr_mapper",
)

_CONST_FIELDS = (
    "world_size",
    "local_rank",
    "num_total_experts",
    "num_experts_per_rank",
    "num_topk",
    "hidden_bytes",
    "sf_uint32_per_token",
    "token_padding_block",
    "sf_padding_block",
    "sm_count",
)


class TokenCommArgs:
    """MegaMoE token communication argument bundle."""

    def __init__(
        self,
        *,
        input_token_buffer: cute.Tensor,
        input_sf_buffer: cute.Tensor,
        topk_idx: cute.Tensor,
        input_topk_weights_buffer: cute.Tensor,
        expert_send_count: cute.Tensor,
        expert_recv_count: cute.Tensor,
        expert_recv_count_sum: cute.Tensor,
        src_token_topk_idx: cute.Tensor,
        fc1_input_token_buffer: cute.Tensor,
        fc1_input_sf_buffer: cute.Tensor,
        fc1_input_topk_weights_buffer: cute.Tensor,
        fc1_ready_counter: cute.Tensor,
        token_src_metadata: cute.Tensor,
        combine_output: cute.Tensor,
        nvlink_barrier_signal: cute.Tensor,
        nvlink_barrier_counter: cute.Tensor,
        grid_sync_counter: cute.Tensor,
        peer_rank_ptr_mapper: Any,
        world_size: int,
        local_rank: int,
        num_total_experts: int,
        num_experts_per_rank: int,
        num_topk: int,
        hidden_bytes: int,
        sf_uint32_per_token: int,
        token_padding_block: int,
        sf_padding_block: int,
        sm_count: int,
        fc2_output_workspace: cute.Tensor = None,
        fc2_done_counter: cute.Tensor = None,
    ):
        self.input_token_buffer = input_token_buffer
        self.input_sf_buffer = input_sf_buffer
        self.topk_idx = topk_idx
        self.input_topk_weights_buffer = input_topk_weights_buffer
        self.expert_send_count = expert_send_count
        self.expert_recv_count = expert_recv_count
        self.expert_recv_count_sum = expert_recv_count_sum
        self.src_token_topk_idx = src_token_topk_idx
        self.fc1_input_token_buffer = fc1_input_token_buffer
        self.fc1_input_sf_buffer = fc1_input_sf_buffer
        self.fc1_input_topk_weights_buffer = fc1_input_topk_weights_buffer
        self.fc1_ready_counter = fc1_ready_counter
        self.token_src_metadata = token_src_metadata
        self.combine_output = combine_output
        self.fc2_output_workspace = fc2_output_workspace
        self.fc2_done_counter = fc2_done_counter
        self.nvlink_barrier_signal = nvlink_barrier_signal
        self.nvlink_barrier_counter = nvlink_barrier_counter
        self.grid_sync_counter = grid_sync_counter
        self.peer_rank_ptr_mapper = peer_rank_ptr_mapper
        self.world_size = world_size
        self.local_rank = local_rank
        self.num_total_experts = num_total_experts
        self.num_experts_per_rank = num_experts_per_rank
        self.num_topk = num_topk
        self.hidden_bytes = hidden_bytes
        self.sf_uint32_per_token = sf_uint32_per_token
        self.token_padding_block = token_padding_block
        self.sf_padding_block = sf_padding_block
        self.sm_count = sm_count

    def __extract_mlir_values__(self) -> List[ir.Value]:
        values: List[ir.Value] = []
        for name in _MLIR_VALUE_FIELDS:
            attr = getattr(self, name)
            if attr is None:
                continue
            values.extend(extract_mlir_values(attr))
        return values

    def __new_from_mlir_values__(self,
                                 values: List[ir.Value]) -> "TokenCommArgs":
        idx = 0
        rebuilt: Dict[str, Any] = {}
        for name in _MLIR_VALUE_FIELDS:
            proto = getattr(self, name)
            if proto is None:
                rebuilt[name] = None
                continue
            n = len(extract_mlir_values(proto))
            rebuilt[name] = new_from_mlir_values(proto, values[idx:idx + n])
            idx += n
        assert idx == len(values), (
            f"TokenCommArgs serialization mismatch: consumed={idx} provided={len(values)}"
        )
        const_kwargs = {name: getattr(self, name) for name in _CONST_FIELDS}
        return TokenCommArgs(**rebuilt, **const_kwargs)


class TokenInPullTokenBackPush:
    """Current implementation: token-in pull, token-back push."""

    num_dispatch_warps: int = 4
    warp_threads: int = 32
    num_dispatch_threads: int = num_dispatch_warps * warp_threads
    dispatch_intra_cta_bar_id: int = 10
    kernel_tail_named_barrier_id: int = 8
    dispatch_to_sched_named_barrier_id: int = 9
    dispatch_to_sched_threads: int = (num_dispatch_warps + 1) * warp_threads
    experts_per_dispatch_pass: int = num_dispatch_threads

    def __init__(
        self,
        *,
        world_size: int,
        local_rank: int,
        num_topk: int,
        num_experts_per_rank: int,
        num_total_experts: int,
        hidden: int,
        fc1_token_dtype,
        sf_uint32_per_token: int,
        token_padding_block: int,
        sf_padding_block: int,
        cluster_tile_tokens: int,
        cluster_shape_mn,
        dispatch_warp_start: int,
        num_other_warps: int,
        fc2_output_dtype=None,
        fc2_publishes_per_token_cluster_tile: int = 0,
    ) -> None:
        self.world_size = world_size
        self.local_rank = local_rank
        self.num_topk = num_topk
        self.num_experts_per_rank = num_experts_per_rank
        self.num_total_experts = num_total_experts
        self.hidden = hidden
        self.fc1_token_dtype = fc1_token_dtype
        self.hidden_bytes = hidden * int(fc1_token_dtype.width) // 8
        self.sf_uint32_per_token = sf_uint32_per_token
        self.token_padding_block = token_padding_block
        self.sf_padding_block = sf_padding_block
        self.cluster_tile_tokens = cluster_tile_tokens
        self.cluster_shape_mn = cluster_shape_mn
        self.dispatch_warp_start = dispatch_warp_start
        # Warps that share this CTA with the dispatch group but are not part
        # of it. They participate in kernel-tail / dispatch-with-other
        # rendezvous and determine `number_of_threads` for those barriers.
        # Pure standalone dispatch passes 0 (no cohabitants -> barriers
        # collapse to dispatch-only).
        self.num_other_warps = num_other_warps
        self.num_other_threads = num_other_warps * self.warp_threads
        self.num_total_threads = self.num_dispatch_threads + self.num_other_threads
        self.kernel_tail_threads = self.num_total_threads

        self.fc2_output_dtype = fc2_output_dtype
        if fc2_output_dtype is not None:
            self.fc2_token_bytes = hidden * int(fc2_output_dtype.width) // 8
            if self.fc2_token_bytes % self.hidden_bytes != 0:
                raise ValueError(
                    f"fc2_token_bytes={self.fc2_token_bytes} must be a "
                    f"multiple of hidden_bytes={self.hidden_bytes} so the "
                    f"per-warp pull buffer can be reused chunk-by-chunk.")
            self.fc2_num_chunks = self.fc2_token_bytes // self.hidden_bytes
            if fc2_publishes_per_token_cluster_tile <= 0:
                raise ValueError(
                    "fc2_publishes_per_token_cluster_tile must be > 0 when "
                    "fc2_output_dtype is set (token_back_by_push enabled).")
            self.fc2_publishes_per_token_cluster_tile = fc2_publishes_per_token_cluster_tile
        else:
            self.fc2_token_bytes = 0
            self.fc2_num_chunks = 0
            self.fc2_publishes_per_token_cluster_tile = 0

    @property
    def enable_token_back(self) -> bool:
        return self.fc2_output_dtype is not None

    def extra_smem_storage_class(self) -> type:
        hidden_bytes = self.hidden_bytes
        num_total_experts = self.num_total_experts

        @cute.struct
        class TokenCommStorage:
            pull_mbar: cute.struct.MemRange[Int64, self.num_dispatch_warps]
            smem_expert_count: cute.struct.MemRange[Int32, num_total_experts]
            pull_buffer: cute.struct.Align[cute.struct.MemRange[
                Uint8, self.num_dispatch_warps * hidden_bytes], 16]

        return TokenCommStorage

    def fc1_ready_counter_ptr(self, token_comm_args):
        return token_comm_args.fc1_ready_counter.iterator

    @cute.jit
    def sched_warp_pre_init_wait(self, token_comm_args):
        nb = pipeline.NamedBarrier(
            barrier_id=self.dispatch_to_sched_named_barrier_id,
            num_threads=self.dispatch_to_sched_threads,
        )
        nb.arrive_and_wait()

    @cute.jit
    def fc1_tma_b_predispatch_spin(self, token_comm_args, work_tile_info):
        counter_slot = work_tile_info.cumulative_token_block_count + work_tile_info.tile_n_idx
        counter_ptr = token_comm_args.fc1_ready_counter.iterator + counter_slot
        if not work_tile_info.peek_ready:
            _iket.range_push("tma_token_fc1_wait")
            spin_wait(
                counter_ptr,
                lambda v: v >= work_tile_info.valid_tokens_in_tile,
                fail_sleep_cycles=20,
            )
            _iket.range_pop()

    @cute.jit
    def dispatch_prep(
        self,
        token_comm_storage,
        topk_idx,
        expert_send_count,
        src_token_topk_idx,
        peer_rank_ptr_mapper,
        sm_idx,
        warp_idx,
        lane_idx,
        *,
        num_tokens,
        num_sms,
    ):
        thread_idx_in_dispatch = Int32(warp_idx * self.warp_threads + lane_idx)
        smem_count_ptr = token_comm_storage.smem_expert_count.data_ptr()
        i = thread_idx_in_dispatch
        while i < Int32(self.num_total_experts):
            (smem_count_ptr + i).store(Int32(0))
            i = i + Int32(self.num_dispatch_threads)
        cute.arch.barrier(
            barrier_id=self.dispatch_intra_cta_bar_id,
            number_of_threads=self.num_dispatch_threads,
        )

        tokens_per_warp: cutlass.Constexpr[int] = 32 // self.num_topk
        active_lanes: cutlass.Constexpr[int] = tokens_per_warp * self.num_topk
        num_dispatch_warps_per_grid: cutlass.Constexpr[
            int] = num_sms * self.num_dispatch_warps

        base_token_for_warp = (sm_idx * self.num_dispatch_warps +
                               warp_idx) * tokens_per_warp
        grid_token_stride = num_dispatch_warps_per_grid * tokens_per_warp

        t = base_token_for_warp
        while t < num_tokens:
            token_offset_in_warp = lane_idx // self.num_topk
            token_global = t + token_offset_in_warp
            if lane_idx < active_lanes and token_global < num_tokens:
                topk_slot = lane_idx % self.num_topk
                expert_id = Int32(topk_idx[token_global, topk_slot])
                if expert_id >= Int32(0):
                    cute.arch.atomic_add(
                        smem_count_ptr + expert_id,
                        Int32(1),
                        sem="relaxed",
                        scope="cta",
                    )
            cute.arch.sync_warp()
            t += grid_token_stride

        cute.arch.barrier(
            barrier_id=self.dispatch_intra_cta_bar_id,
            number_of_threads=self.num_dispatch_threads,
        )

        for offset in cutlass.range_constexpr(
                0,
                self.num_total_experts,
                self.experts_per_dispatch_pass,
        ):
            expert_id = Int32(offset + warp_idx * self.warp_threads + lane_idx)
            if expert_id < Int32(self.num_total_experts):
                slot_ptr = smem_count_ptr + expert_id
                local_count = (slot_ptr).load()
                delta = (Int64(1) << Int64(32)) | (Int64(local_count)
                                                   & Int64(0xFFFFFFFF))
                old_packed = cute.arch.atomic_add(
                    expert_send_count.iterator + expert_id,
                    delta,
                    sem="relaxed",
                    scope="gpu",
                )
                base_slot = Int32(old_packed & Int64(0xFFFFFFFF))
                (slot_ptr).store(base_slot)
        cute.arch.barrier(
            barrier_id=self.dispatch_intra_cta_bar_id,
            number_of_threads=self.num_dispatch_threads,
        )

        t = base_token_for_warp
        while t < num_tokens:
            token_offset_in_warp = lane_idx // self.num_topk
            token_global = t + token_offset_in_warp
            if lane_idx < active_lanes and token_global < num_tokens:
                topk_slot = lane_idx % self.num_topk
                expert_id = Int32(topk_idx[token_global, topk_slot])
                if expert_id >= Int32(0):
                    dst_rank = expert_id // Int32(self.num_experts_per_rank)
                    local_expert = expert_id % Int32(self.num_experts_per_rank)
                    slot = cute.arch.atomic_add(
                        smem_count_ptr + expert_id,
                        Int32(1),
                        sem="relaxed",
                        scope="cta",
                    )
                    token_topk_word = Int32(token_global * self.num_topk +
                                            topk_slot)
                    MAX_SLOT_C: cutlass.Constexpr[
                        int] = num_tokens * self.num_topk
                    elem_off = ((local_expert * Int32(self.world_size) + Int32(
                        self.local_rank)) * Int32(MAX_SLOT_C) + slot) * Int32(4)
                    peer_addr = peer_rank_ptr_mapper.map(
                        src_token_topk_idx.iterator.toint(),
                        dst_rank,
                        Int64(elem_off),
                    )
                    stg_b32_raw(peer_addr, token_topk_word)
            cute.arch.sync_warp()
            t += grid_token_stride

    @cute.jit
    def dispatch_barrier(
        self,
        expert_send_count,
        expert_recv_count,
        expert_recv_count_sum,
        nvlink_barrier_signal,
        grid_sync_counter,
        peer_rank_ptr_mapper,
        sm_idx,
        warp_idx,
        lane_idx,
        *,
        num_sms,
        nvlink_barrier_counter=None,
    ):
        # software_grid_sync expects a dispatch-group-relative thread id.
        tid_in_group = warp_idx * Int32(self.warp_threads) + lane_idx

        software_grid_sync(grid_sync_counter,
                           sm_idx,
                           num_sms,
                           tid_in_group,
                           num_threads=self.num_dispatch_threads)

        if sm_idx == 0:
            for offset in cutlass.range_constexpr(
                    0,
                    self.num_total_experts,
                    self.experts_per_dispatch_pass,
            ):
                expert_id = Int32(offset + warp_idx * self.warp_threads +
                                  lane_idx)
                if expert_id < Int32(self.num_total_experts):
                    dst_rank = expert_id // Int32(self.num_experts_per_rank)
                    dst_local_expert = expert_id % Int32(
                        self.num_experts_per_rank)
                    status_u64 = cute.arch.load(
                        expert_send_count.iterator + expert_id,
                        Int64,
                        sem="relaxed",
                        scope="gpu",
                    )
                    token_count_u32 = Int32(status_u64 & Int64(0xFFFFFFFF))
                    erc_local_base = expert_recv_count.iterator.toint()
                    erc_elem_off = (Int32(self.local_rank) *
                                    Int32(self.num_experts_per_rank) +
                                    dst_local_expert) * Int32(8)
                    erc_peer_addr = peer_rank_ptr_mapper.map(
                        erc_local_base,
                        dst_rank,
                        Int64(erc_elem_off),
                    )
                    stg_b64_raw(erc_peer_addr, Int64(token_count_u32))
                    ercs_local_base = expert_recv_count_sum.iterator.toint()
                    ercs_peer_addr = peer_rank_ptr_mapper.map(
                        ercs_local_base,
                        dst_rank,
                        Int64(dst_local_expert * Int32(8)),
                    )
                    red_add_release_sys_u64_raw(ercs_peer_addr, status_u64)
        cute.arch.barrier(
            barrier_id=self.dispatch_intra_cta_bar_id,
            number_of_threads=self.num_dispatch_threads,
        )

        self.nvlink_barrier(
            nvlink_barrier_signal,
            nvlink_barrier_counter,
            grid_sync_counter,
            peer_rank_ptr_mapper,
            sm_idx,
            warp_idx,
            lane_idx,
            slot=0,
            num_sms=num_sms,
            prologue_grid_sync=False,
            epilogue_grid_sync=True,
        )

    @cute.jit
    def dispatch_pull(
        self,
        token_comm_storage,
        input_token_buffer,
        input_sf_buffer,
        input_topk_weights_buffer,
        src_token_topk_idx,
        expert_recv_count,
        expert_recv_count_sum,
        fc1_input_token_buffer,
        fc1_input_sf_buffer,
        fc1_input_topk_weights_buffer,
        fc1_ready_counter,
        token_src_metadata,
        peer_rank_ptr_mapper,
        sm_idx,
        warp_idx,
        lane_idx,
        *,
        num_sms,
    ):
        # MemRange does not support dynamic indexing here; use raw pointers.
        pull_mbar_ptr = token_comm_storage.pull_mbar.data_ptr()
        pull_buffer_ptr = token_comm_storage.pull_buffer.data_ptr()
        if lane_idx == Int32(0):
            cute.arch.mbarrier_init(pull_mbar_ptr + warp_idx, 1)
        cute.arch.sync_warp()

        phase_bit = Int32(0)

        current_expert_idx = Int32(-1)
        expert_start_idx = Int32(0)
        expert_end_idx = Int32(0)
        expert_pool_block_offset = Int32(0)
        expert_task_tile_offset = Int32(0)
        # SF rows use their own padding; token and SF pool offsets can diverge.
        expert_sf_pool_block_offset = Int32(0)

        stored_rank_count_lane = Int32(0)

        NUM_EXPERTS_PER_LANE: cutlass.Constexpr[int] = (
            self.num_experts_per_rank + 31) // 32
        stored_num_tokens_per_expert = []
        for _ in cutlass.range_constexpr(0, NUM_EXPERTS_PER_LANE, 1):
            stored_num_tokens_per_expert.append(Int32(0))
        for i in cutlass.range_constexpr(0, NUM_EXPERTS_PER_LANE, 1):
            e_idx_for_lane = Int32(i * self.warp_threads) + lane_idx
            if e_idx_for_lane < Int32(self.num_experts_per_rank):
                sum_packed_init = expert_recv_count_sum[e_idx_for_lane]
                stored_num_tokens_per_expert[i] = Int32(
                    Int64(sum_packed_init) & Int64(0xFFFFFFFF))
        cute.arch.sync_warp()

        num_global_warps: cutlass.Constexpr[
            int] = num_sms * self.num_dispatch_warps
        token_idx = sm_idx * Int32(self.num_dispatch_warps) + warp_idx

        _iket_pull_emit = (sm_idx == Int32(0)) and (warp_idx == Int32(0)) and (
            lane_idx == Int32(0))

        while current_expert_idx < Int32(self.num_experts_per_rank):
            if _iket_pull_emit:
                _iket.range_push("Pull.ChooseToken")
            old_expert_idx = current_expert_idx
            while (token_idx >= expert_end_idx) and (current_expert_idx < Int32(
                    self.num_experts_per_rank)):
                prev_valid_count = expert_end_idx - expert_start_idx
                prev_block_count = (prev_valid_count +
                                    Int32(self.token_padding_block) -
                                    Int32(1)) // Int32(self.token_padding_block)
                expert_pool_block_offset = expert_pool_block_offset + prev_block_count
                # Mirror cumul for the release-counter granularity (self.cluster_tile_tokens).
                prev_task_tile_count = (
                    prev_valid_count + Int32(self.cluster_tile_tokens) -
                    Int32(1)) // Int32(self.cluster_tile_tokens)
                expert_task_tile_offset = expert_task_tile_offset + prev_task_tile_count
                # Mirror cumul for the SF axis granularity (self.sf_padding_block).
                prev_sf_block_count = (prev_valid_count +
                                       Int32(self.sf_padding_block) -
                                       Int32(1)) // Int32(self.sf_padding_block)
                expert_sf_pool_block_offset = expert_sf_pool_block_offset + prev_sf_block_count
                current_expert_idx = current_expert_idx + Int32(1)
                if current_expert_idx < Int32(self.num_experts_per_rank):
                    expert_start_idx = expert_end_idx
                    valid_value = Int32(0)
                    for i in cutlass.range_constexpr(0, NUM_EXPERTS_PER_LANE,
                                                     1):
                        if current_expert_idx == Int32(
                                i * self.warp_threads) + lane_idx:
                            valid_value = stored_num_tokens_per_expert[i]
                    total_for_expert = cute.arch.shuffle_sync(
                        valid_value,
                        current_expert_idx % Int32(self.warp_threads))
                    expert_end_idx = expert_end_idx + total_for_expert

            if current_expert_idx < Int32(self.num_experts_per_rank):
                if old_expert_idx != current_expert_idx:
                    if lane_idx < Int32(self.world_size):
                        stored_rank_count_lane = Int32(
                            expert_recv_count[lane_idx, current_expert_idx])
                    else:
                        stored_rank_count_lane = Int32(0)

                token_idx_in_expert = token_idx - expert_start_idx
                slot_idx = token_idx_in_expert
                offset = Int32(0)
                remaining_lane = stored_rank_count_lane

                current_rank_in_expert_idx = Int32(0)
                token_idx_in_rank = Int32(0)

                decided = Int32(0)
                for _round in cutlass.range_constexpr(0, self.world_size + 1,
                                                      1):
                    if decided == Int32(0):
                        active = remaining_lane > Int32(0)
                        mask = cute.arch.vote_ballot_sync(active)
                        num_active_ranks = Int32(cute.arch.popc(Int32(mask)))
                        v_for_min = Int32(0x7FFFFFFF)
                        if active:
                            v_for_min = remaining_lane
                        length = Int32(
                            cute.arch.warp_redux_sync(v_for_min, "min"))

                        if num_active_ranks > Int32(0):
                            num_round_tokens = length * num_active_ranks
                            if slot_idx < num_round_tokens:
                                slot_idx_in_round = slot_idx % num_active_ranks
                                current_rank_in_expert_idx = fns_b32(
                                    Int32(mask),
                                    Int32(0),
                                    slot_idx_in_round + Int32(1),
                                )
                                token_idx_in_rank = offset + (slot_idx //
                                                              num_active_ranks)
                                decided = Int32(1)
                            else:
                                slot_idx = slot_idx - num_round_tokens
                                offset = offset + length
                                if remaining_lane > length:
                                    remaining_lane = remaining_lane - length
                                else:
                                    remaining_lane = Int32(0)
                        else:
                            decided = Int32(1)

                if _iket_pull_emit:
                    _iket.range_pop()  # Pull.ChooseToken
                    _iket.range_push("Pull.TMA_NVLink_Roundtrip")

                src_token_topk = Uint32(src_token_topk_idx[
                    current_expert_idx,
                    current_rank_in_expert_idx,
                    token_idx_in_rank,
                ])
                src_token = Int32(src_token_topk // Uint32(self.num_topk))
                src_topk = Int32(src_token_topk % Uint32(self.num_topk))

                cur_peer_offset = peer_rank_ptr_mapper.map(
                    Int64(0), current_rank_in_expert_idx, Int64(0))
                inp_tok_local_base = input_token_buffer.iterator.toint()
                inp_sf_local_base = input_sf_buffer.iterator.toint()
                inp_w_local_base = input_topk_weights_buffer.iterator.toint()

                with cute.arch.elect_one():
                    pull_buffer_warp_ptr = pull_buffer_ptr + (
                        warp_idx * Int32(self.hidden_bytes))
                    tma_src_addr = (inp_tok_local_base + cur_peer_offset +
                                    Int64(src_token * Int32(self.hidden_bytes)))
                    tma_load_1d_raw(
                        pull_buffer_warp_ptr,
                        tma_src_addr,
                        pull_mbar_ptr + warp_idx,
                        Int32(self.hidden_bytes),
                    )
                cute.arch.sync_warp()

                if _iket_pull_emit:
                    _iket.range_push("Pull.SF_LDG_STG")

                sf_token_in_pool_axis = (
                    expert_sf_pool_block_offset * Int32(self.sf_padding_block) +
                    token_idx_in_expert)
                pool_token_idx = (
                    expert_pool_block_offset * Int32(self.token_padding_block) +
                    token_idx_in_expert)
                sf_passes: cutlass.Constexpr[int] = (self.sf_uint32_per_token +
                                                     31) // 32

                sf_vals = []
                for _ in cutlass.range_constexpr(0, sf_passes, 1):
                    sf_vals.append(Int32(0))

                for i in cutlass.range_constexpr(0, sf_passes, 1):
                    j = Int32(i * self.warp_threads) + lane_idx
                    if j < Int32(self.sf_uint32_per_token):
                        sf_addr = (inp_sf_local_base + cur_peer_offset + Int64(
                            (src_token * Int32(self.sf_uint32_per_token) + j) *
                            Int32(4)))
                        sf_vals[i] = ldg_b32_raw(sf_addr)

                weight = Float32(0.0)
                if lane_idx == Int32(0):
                    weight_addr = (inp_w_local_base + cur_peer_offset + Int64(
                        (src_token * Int32(self.num_topk) + src_topk) *
                        Int32(4)))
                    weight = ldg_f32_raw(weight_addr)

                if _iket_pull_emit:
                    _iket.range_pop()  # Pull.SF_LDG_STG  (= LD phase)
                    _iket.range_push("Pull.Weight_LDG")  # (= ST phase)

                for i in cutlass.range_constexpr(0, sf_passes, 1):
                    j = Int32(i * self.warp_threads) + lane_idx
                    if j < Int32(self.sf_uint32_per_token):
                        sf_int32_pos = sf_atom_int32_offset(
                            sf_token_in_pool_axis,
                            j,
                            num_k_atoms=self.sf_uint32_per_token,
                        )
                        fc1_input_sf_buffer[sf_int32_pos] = sf_vals[i]
                cute.arch.sync_warp()

                if lane_idx == Int32(0):
                    fc1_input_topk_weights_buffer[pool_token_idx] = weight

                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        pull_mbar_ptr + warp_idx, Int32(self.hidden_bytes))
                    cute.arch.mbarrier_wait(
                        pull_mbar_ptr + warp_idx,
                        phase_bit,
                    )

                if _iket_pull_emit:
                    _iket.range_pop()  # Pull.Weight_LDG (ST phase)
                    _iket.range_pop()  # Pull.TMA_NVLink_Roundtrip (outer)
                    _iket.range_push("Pull.TMA_Store")

                with cute.arch.elect_one():
                    pull_buffer_warp_ptr = pull_buffer_ptr + (
                        warp_idx * Int32(self.hidden_bytes))
                    tma_store_1d(
                        fc1_input_token_buffer.iterator
                        # T=128k) × self.hidden_bytes overflows int32 (max 2.1 G).
                        # 64-bit address math is required for large token pools.
                        + (Int64(pool_token_idx) * Int64(self.hidden_bytes)),
                        pull_buffer_warp_ptr,
                        Int32(self.hidden_bytes),
                    )

                with cute.arch.elect_one():
                    _store_token_src_metadata_u32x3(
                        token_src_metadata,
                        pool_token_idx,
                        Uint32(current_rank_in_expert_idx),
                        Uint32(src_token),
                        Uint32(src_topk),
                    )

                with cute.arch.elect_one():
                    cute.arch.cp_async_bulk_commit_group()
                    cute.arch.cp_async_bulk_wait_group(0)

                if _iket_pull_emit:
                    _iket.range_pop()  # Pull.TMA_Store
                    _iket.range_push("Pull.Arrival_Atomic")

                with cute.arch.elect_one():
                    task_tile_idx = expert_task_tile_offset + (
                        token_idx_in_expert // Int32(self.cluster_tile_tokens))
                    cute.arch.atomic_add(
                        fc1_ready_counter.iterator + task_tile_idx,
                        Int32(1),
                        sem="release",
                        scope="gpu",
                    )
                cute.arch.sync_warp()

                if _iket_pull_emit:
                    _iket.range_pop()  # Pull.Arrival_Atomic

                phase_bit = phase_bit ^ Int32(1)

                token_idx = token_idx + Int32(num_global_warps)

        return phase_bit, stored_num_tokens_per_expert

    @cute.jit
    def token_back_by_push(
        self,
        token_comm_storage,
        fc2_output_workspace,
        fc2_done_counter,
        token_src_metadata,
        combine_output,
        peer_rank_ptr_mapper,
        phase_bit,
        stored_num_tokens_per_expert,
        sm_idx,
        warp_idx,
        lane_idx,
        *,
        num_sms,
    ):
        _iket_emit = (sm_idx == Int32(0)) and (warp_idx == Int32(0))

        chunk_bytes: cutlass.Constexpr[int] = self.hidden_bytes
        num_chunks: cutlass.Constexpr[int] = self.fc2_num_chunks
        fc2_token_bytes: cutlass.Constexpr[int] = self.fc2_token_bytes

        pull_buffer_ptr = token_comm_storage.pull_buffer.data_ptr()
        pull_mbar_ptr = token_comm_storage.pull_mbar.data_ptr()

        num_experts_per_lane: cutlass.Constexpr[int] = (
            self.num_experts_per_rank + 31) // 32
        num_global_warps: cutlass.Constexpr[
            int] = num_sms * self.num_dispatch_warps

        token_idx = sm_idx * Int32(self.num_dispatch_warps) + warp_idx

        current_expert_idx = Int32(-1)
        expert_start_idx = Int32(0)
        expert_end_idx = Int32(0)
        expert_pool_block_offset = Int32(0)

        while current_expert_idx < Int32(self.num_experts_per_rank):
            while (token_idx >= expert_end_idx) and (current_expert_idx < Int32(
                    self.num_experts_per_rank)):
                prev_valid_count = expert_end_idx - expert_start_idx
                prev_block_count = (prev_valid_count +
                                    Int32(self.token_padding_block) -
                                    Int32(1)) // Int32(self.token_padding_block)
                expert_pool_block_offset = expert_pool_block_offset + prev_block_count

                current_expert_idx = current_expert_idx + Int32(1)
                if current_expert_idx < Int32(self.num_experts_per_rank):
                    expert_start_idx = expert_end_idx
                    valid_value = Int32(0)
                    for i in cutlass.range_constexpr(0, num_experts_per_lane,
                                                     1):
                        if current_expert_idx == Int32(
                                i * self.warp_threads) + lane_idx:
                            valid_value = stored_num_tokens_per_expert[i]
                    total_for_expert = cute.arch.shuffle_sync(
                        valid_value,
                        current_expert_idx % Int32(self.warp_threads),
                    )
                    expert_end_idx = expert_end_idx + total_for_expert

                    cluster_tile_cnt = (
                        total_for_expert + Int32(self.cluster_tile_tokens) -
                        Int32(1)) // Int32(self.cluster_tile_tokens)
                    expected = cluster_tile_cnt * Int32(
                        self.fc2_publishes_per_token_cluster_tile)
                    spin_wait(
                        fc2_done_counter.iterator + current_expert_idx,
                        lambda v: v >= expected,
                        fail_sleep_cycles=500,
                    )

            if current_expert_idx < Int32(self.num_experts_per_rank):
                token_idx_in_expert = token_idx - expert_start_idx
                pool_token_idx = (
                    expert_pool_block_offset * Int32(self.token_padding_block) +
                    token_idx_in_expert)

                md_base = token_src_metadata.iterator + (pool_token_idx *
                                                         Int32(12))
                src_rank = Int32(
                    cute.arch.load(md_base + Int32(0), Int32, scope="gpu"))
                src_token = Int32(
                    cute.arch.load(md_base + Int32(4), Int32, scope="gpu"))
                src_topk = Int32(
                    cute.arch.load(md_base + Int32(8), Int32, scope="gpu"))

                local_token_addr = fc2_output_workspace.iterator.toint(
                ) + Int64(pool_token_idx) * Int64(fc2_token_bytes)
                peer_combine_ptr = peer_rank_ptr_mapper.ptr_map_to_rank(
                    combine_output.iterator,
                    src_rank,
                )
                peer_token_ptr = peer_combine_ptr + (
                    Int64(src_token * Int32(self.num_topk) + src_topk) *
                    Int64(fc2_token_bytes))

                smem_ptr_warp = pull_buffer_ptr + warp_idx * Int32(chunk_bytes)
                mbar_ptr_warp = pull_mbar_ptr + warp_idx

                if _iket_emit:
                    _iket.range_push("token_back")

                for chunk in cutlass.range_constexpr(0, num_chunks, 1):
                    chunk_off = Int64(chunk * chunk_bytes)
                    # chunk_t0 = read_clock64()

                    with cute.arch.elect_one():
                        tma_load_1d_raw(
                            smem_ptr_warp,
                            local_token_addr + chunk_off,
                            mbar_ptr_warp,
                            Int32(chunk_bytes),
                        )
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            mbar_ptr_warp,
                            Int32(chunk_bytes),
                        )
                        cute.arch.mbarrier_wait(mbar_ptr_warp, phase_bit)
                    cute.arch.sync_warp()

                    with cute.arch.elect_one():
                        tma_store_1d(
                            peer_token_ptr + chunk_off,
                            smem_ptr_warp,
                            Int32(chunk_bytes),
                        )
                        cute.arch.cp_async_bulk_commit_group()
                        cute.arch.cp_async_bulk_wait_group(0)
                    cute.arch.sync_warp()

                    # if read_clock64() - chunk_t0 < Int64(600):
                    #     _nanosleep(100)

                    phase_bit = phase_bit ^ Int32(1)
                if _iket_emit:
                    _iket.range_pop()

                token_idx = token_idx + Int32(num_global_warps)

        cute.arch.fence_acq_rel_sys()

    @cute.jit
    def nvlink_barrier(
        self,
        nvlink_barrier_signal,
        nvlink_barrier_counter,
        grid_sync_counter,
        peer_rank_ptr_mapper,
        sm_idx,
        warp_idx,
        lane_idx,
        *,
        slot: cutlass.Constexpr[int],
        num_sms,
        prologue_grid_sync: cutlass.Constexpr[bool],
        epilogue_grid_sync: cutlass.Constexpr[bool],
    ):
        # software_grid_sync expects a dispatch-group-relative thread id.
        tid_in_group = warp_idx * Int32(self.warp_threads) + lane_idx

        if prologue_grid_sync:
            software_grid_sync(
                grid_sync_counter,
                sm_idx,
                num_sms,
                tid_in_group,
                num_threads=self.num_dispatch_threads,
            )

        if sm_idx == 0:
            if warp_idx == 0:
                signal_phase = Int32(slot)
                signal_delta = Int32(1)
                target = Int32(self.world_size)
                if cutlass.const_expr(nvlink_barrier_counter is not None):
                    status = nvlink_barrier_counter[0] & Int32(3)
                    signal_phase = status & Int32(1)
                    signal_sign = status >> Int32(1)
                    if signal_sign != Int32(0):
                        signal_delta = Int32(-1)
                        target = Int32(0)

                nbs_local_base = nvlink_barrier_signal.iterator.toint()
                if lane_idx < Int32(self.world_size):
                    lane_peer_addr = peer_rank_ptr_mapper.map(
                        nbs_local_base,
                        lane_idx,
                        Int64(signal_phase * Int32(4)),
                    )
                    red_add_release_sys_s32_raw(lane_peer_addr, signal_delta)
                cute.arch.sync_warp()

                if lane_idx == 0:
                    if cutlass.const_expr(nvlink_barrier_counter is not None):
                        cute.arch.atomic_add(
                            nvlink_barrier_counter.iterator,
                            Int32(1),
                            sem="relaxed",
                            scope="gpu",
                        )
                    local_signal_ptr = nvlink_barrier_signal.iterator + signal_phase
                    if cutlass.const_expr(nvlink_barrier_counter is None):
                        while (cute.arch.load(local_signal_ptr,
                                              Int32,
                                              sem="acquire",
                                              scope="sys") < target):
                            pass
                    else:
                        while (cute.arch.load(local_signal_ptr,
                                              Int32,
                                              sem="acquire",
                                              scope="sys") != target):
                            pass

        if epilogue_grid_sync:
            software_grid_sync(
                grid_sync_counter,
                sm_idx,
                num_sms,
                tid_in_group,
                num_threads=self.num_dispatch_threads,
            )

    @cute.jit
    def dispatch_warp_body(
        self,
        token_comm_args,
        token_comm_storage,
        *,
        warp_idx,
        lane_idx,
        tidx,
    ):
        bidx, bidy, bidz = cute.arch.block_idx()
        cta_linear_id = (
            Int32(bidx) + Int32(self.cluster_shape_mn[1]) * Int32(bidy) +
            Int32(self.cluster_shape_mn[1] * self.cluster_shape_mn[0]) *
            Int32(bidz))
        local_warp_idx = Int32(warp_idx) - Int32(self.dispatch_warp_start)

        iket_active = (cta_linear_id == Int32(0)) and (local_warp_idx
                                                       == Int32(0))
        if iket_active:
            _iket.range_push("Dispatch_Prep")

        self.dispatch_prep(
            token_comm_storage,
            token_comm_args.topk_idx,
            token_comm_args.expert_send_count,
            token_comm_args.src_token_topk_idx,
            token_comm_args.peer_rank_ptr_mapper,
            cta_linear_id,
            local_warp_idx,
            lane_idx,
            num_tokens=token_comm_args.input_token_buffer.shape[0],
            num_sms=token_comm_args.sm_count,
        )

        if iket_active:
            _iket.range_pop()
            _iket.range_push("Dispatch_Barrier")

        self.dispatch_barrier(
            token_comm_args.expert_send_count,
            token_comm_args.expert_recv_count,
            token_comm_args.expert_recv_count_sum,
            token_comm_args.nvlink_barrier_signal,
            token_comm_args.grid_sync_counter,
            token_comm_args.peer_rank_ptr_mapper,
            cta_linear_id,
            local_warp_idx,
            lane_idx,
            num_sms=token_comm_args.sm_count,
            nvlink_barrier_counter=token_comm_args.nvlink_barrier_counter,
        )

        nb_dispatch_to_sched = pipeline.NamedBarrier(
            barrier_id=self.dispatch_to_sched_named_barrier_id,
            num_threads=self.dispatch_to_sched_threads,
        )
        nb_dispatch_to_sched.arrive()

        if iket_active:
            _iket.range_pop()
            _iket.range_push("Dispatch_Pull")

        phase_bit, stored_num_tokens_per_expert = self.dispatch_pull(
            token_comm_storage,
            token_comm_args.input_token_buffer,
            token_comm_args.input_sf_buffer,
            token_comm_args.input_topk_weights_buffer,
            token_comm_args.src_token_topk_idx,
            token_comm_args.expert_recv_count,
            token_comm_args.expert_recv_count_sum,
            token_comm_args.fc1_input_token_buffer,
            token_comm_args.fc1_input_sf_buffer,
            token_comm_args.fc1_input_topk_weights_buffer,
            token_comm_args.fc1_ready_counter,
            token_comm_args.token_src_metadata,
            token_comm_args.peer_rank_ptr_mapper,
            cta_linear_id,
            local_warp_idx,
            lane_idx,
            num_sms=token_comm_args.sm_count,
        )

        if iket_active:
            _iket.range_pop()

        if cutlass.const_expr(self.enable_token_back):
            if iket_active:
                _iket.range_push("Token_Back_By_Push")

            self.token_back_by_push(
                token_comm_storage,
                token_comm_args.fc2_output_workspace,
                token_comm_args.fc2_done_counter,
                token_comm_args.token_src_metadata,
                token_comm_args.combine_output,
                token_comm_args.peer_rank_ptr_mapper,
                phase_bit,
                stored_num_tokens_per_expert,
                cta_linear_id,
                local_warp_idx,
                lane_idx,
                num_sms=token_comm_args.sm_count,
            )

            if iket_active:
                _iket.range_pop()

    @cute.jit
    def tail_reset_shared_counters(
        self,
        token_comm_args,
        *,
        cta_linear_id,
        local_warp_idx,
        lane_idx,
    ):
        thread_linear = (cta_linear_id * Int32(self.num_dispatch_warps) +
                         local_warp_idx) * Int32(self.warp_threads) + lane_idx
        stride = Int32(token_comm_args.sm_count * self.num_dispatch_threads)

        recv_total: cutlass.Constexpr[
            int] = self.world_size * self.num_experts_per_rank
        i = thread_linear
        while i < Int32(recv_total):
            rank_idx = i // Int32(self.num_experts_per_rank)
            expert_idx = i % Int32(self.num_experts_per_rank)
            token_comm_args.expert_recv_count[rank_idx, expert_idx] = Int64(0)
            i = i + stride

        i = thread_linear
        while i < Int32(self.num_experts_per_rank):
            token_comm_args.expert_recv_count_sum[i] = Int64(0)
            i = i + stride

        if cutlass.const_expr(self.enable_token_back):
            i = thread_linear
            while i < Int32(self.num_experts_per_rank):
                token_comm_args.fc2_done_counter[i] = Int32(0)
                i = i + stride

    @cute.jit
    def kernel_tail(
        self,
        token_comm_args,
        *,
        warp_idx,
        lane_idx,
        tidx,
    ):
        nb_kernel_tail = pipeline.NamedBarrier(
            barrier_id=self.kernel_tail_named_barrier_id,
            num_threads=self.kernel_tail_threads,
        )
        nb_kernel_tail.arrive_and_wait()

        if warp_idx >= self.dispatch_warp_start:
            bidx, bidy, bidz = cute.arch.block_idx()
            cta_linear_id = (
                Int32(bidx) + Int32(self.cluster_shape_mn[1]) * Int32(bidy) +
                Int32(self.cluster_shape_mn[1] * self.cluster_shape_mn[0]) *
                Int32(bidz))
            local_warp_idx = Int32(warp_idx) - Int32(self.dispatch_warp_start)
            self.nvlink_barrier(
                token_comm_args.nvlink_barrier_signal,
                token_comm_args.nvlink_barrier_counter,
                token_comm_args.grid_sync_counter,
                token_comm_args.peer_rank_ptr_mapper,
                cta_linear_id,
                local_warp_idx,
                lane_idx,
                slot=1,
                num_sms=token_comm_args.sm_count,
                prologue_grid_sync=True,
                epilogue_grid_sync=True,
            )
            self.nvlink_barrier(
                token_comm_args.nvlink_barrier_signal,
                token_comm_args.nvlink_barrier_counter,
                token_comm_args.grid_sync_counter,
                token_comm_args.peer_rank_ptr_mapper,
                cta_linear_id,
                local_warp_idx,
                lane_idx,
                slot=1,
                num_sms=token_comm_args.sm_count,
                prologue_grid_sync=True,
                epilogue_grid_sync=True,
            )
            self.tail_reset_shared_counters(
                token_comm_args,
                cta_linear_id=cta_linear_id,
                local_warp_idx=local_warp_idx,
                lane_idx=lane_idx,
            )
            self.nvlink_barrier(
                token_comm_args.nvlink_barrier_signal,
                token_comm_args.nvlink_barrier_counter,
                token_comm_args.grid_sync_counter,
                token_comm_args.peer_rank_ptr_mapper,
                cta_linear_id,
                local_warp_idx,
                lane_idx,
                slot=0,
                num_sms=token_comm_args.sm_count,
                prologue_grid_sync=True,
                epilogue_grid_sync=True,
            )
