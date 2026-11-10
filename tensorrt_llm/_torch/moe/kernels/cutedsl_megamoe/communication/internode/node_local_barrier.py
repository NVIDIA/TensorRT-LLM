"""Sense-reversing barrier over the ranks of one NVLink domain (node), not the whole world."""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import Int32, Int64

from ...helpers.ptx_helpers import red_add_relaxed_sys_s32
from ...helpers.software_sync import NvlinkBarrier


class NodeLocalNvlinkBarrier(NvlinkBarrier):
    """``NvlinkBarrier`` whose peers are the ``ranks_per_node`` global ranks of this rank's node.

    The NVLink-domain protocol signals ranks ``0 .. world_size-1``; with more than one node the
    peers a rank can reach over NVLink (and through an LSA mapper, which traps on a cross-node
    rank) are ``node_base .. node_base + ranks_per_node - 1`` where ``node_base`` is derived from
    the local rank assigned in the kernel prologue. With one node this is the parent barrier.
    """

    def __init__(self, *, world_size: int, ranks_per_node: int, barrier_id: int) -> None:
        if ranks_per_node <= 0 or world_size % ranks_per_node != 0:
            raise ValueError(f"ranks_per_node={ranks_per_node} must divide world_size={world_size}.")
        # The parent's ``world_size`` is the signal target: arrivals per barrier = ranks per node.
        super().__init__(world_size=ranks_per_node, barrier_id=barrier_id)
        self.ranks_per_node = ranks_per_node
        self.total_world_size = world_size
        self._local_rank = None

    def assign_local_rank(self, local_rank: Int32) -> None:
        self._local_rank = local_rank

    def remove_device_members(self) -> None:
        super().remove_device_members()
        self._local_rank = None

    @cute.jit
    def arrive_and_wait(
        self,
        participating_threads: int,
        actual_cta_count: Int32,
        linear_cta_idx: Int32,
        thread_idx_in_group: Int32,
        *,
        prologue_grid_sync: bool,
        epilogue_grid_sync: bool,
    ) -> None:
        if cutlass.const_expr(prologue_grid_sync):
            self.sync(participating_threads, actual_cta_count, linear_cta_idx, thread_idx_in_group)
        if linear_cta_idx == Int32(0):
            status = cute.arch.load(self._phase_counter, Int32, sem="relaxed", scope="gpu") & Int32(3)
            signal_phase = status & Int32(1)
            signal_direction = status >> Int32(1)
            signal_delta = Int32(1)
            signal_target = Int32(self.ranks_per_node)
            if signal_direction != Int32(0):
                signal_delta = Int32(-1)
                signal_target = Int32(0)
            self._cta_rendezvous(participating_threads)
            if thread_idx_in_group == Int32(0):
                cute.arch.fence_acq_rel_sys()
            self._cta_rendezvous(participating_threads)
            node_base = (self._local_rank // Int32(self.ranks_per_node)) * Int32(self.ranks_per_node)
            rank_round_count = (self.ranks_per_node + participating_threads - 1) // participating_threads
            signal_base_address = self._signal.toint()
            signal_byte_offset = Int64(signal_phase * Int32(4))
            for rank_round in cutlass.range_constexpr(rank_round_count):
                peer_local_index = Int32(rank_round * participating_threads) + thread_idx_in_group
                if peer_local_index < Int32(self.ranks_per_node):
                    destination_address = self._peer_rank_ptr_mapper.map(
                        signal_base_address, node_base + peer_local_index, signal_byte_offset
                    )
                    red_add_relaxed_sys_s32(destination_address, signal_delta)
            self._cta_rendezvous(participating_threads)
            if thread_idx_in_group == Int32(0):
                cute.arch.atomic_add(self._phase_counter, Int32(1), sem="relaxed", scope="gpu")
                local_signal = self._signal + signal_phase
                while cute.arch.load(local_signal, Int32, sem="acquire", scope="sys") != signal_target:
                    pass
        if cutlass.const_expr(epilogue_grid_sync):
            self.sync(participating_threads, actual_cta_count, linear_cta_idx, thread_idx_in_group)


__all__ = ["NodeLocalNvlinkBarrier"]
