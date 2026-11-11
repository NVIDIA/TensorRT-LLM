"""Software grid and NVLink synchronization for persistent kernels."""

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import Int32, Int64

from .device_workspace import DeviceWorkspace
from .ptx_helpers import red_add_relaxed_sys_s32


class SoftwareGridSync:
    """Reusable device-local barrier over a phase-flipping GMEM counter."""

    finish_sum_tag = 0x80000000
    grid_counter_region = "software_grid_sync.counter"

    def __init__(self, *, barrier_id: int) -> None:
        self.barrier_id = barrier_id
        self._grid_counter = None

    def __extract_mlir_values__(self) -> list:
        return []

    def __new_from_mlir_values__(self, values: list) -> "SoftwareGridSync":
        if values:
            raise ValueError(f"SoftwareGridSync expected no MLIR values, got {len(values)}.")
        return self

    def register_device_workspace(self, workspace: DeviceWorkspace) -> None:
        workspace.register(
            self.grid_counter_region,
            cutlass.Int32,
            (1,),
            buffer_space="local",
            byte_alignment=16,
            reset="zero_on_first_allocate",
        )

    @cute.jit
    def assign_device_members(self, workspace: DeviceWorkspace) -> None:
        self._grid_counter = workspace.ptr(self.grid_counter_region)

    def remove_device_members(self) -> None:
        self._grid_counter = None

    @cute.jit
    def _cta_rendezvous(self, participating_threads: int) -> None:
        cute.arch.barrier(barrier_id=self.barrier_id, number_of_threads=participating_threads)

    @cute.jit
    def sync(
        self,
        participating_threads: int,
        actual_cta_count: Int32,
        linear_cta_idx: Int32,
        thread_idx_in_group: Int32,
    ) -> None:
        self._cta_rendezvous(participating_threads)
        leader_delta = Int32(-self.finish_sum_tag) - (actual_cta_count - Int32(1))
        _inline_grid_sync(
            self._grid_counter,
            linear_cta_idx,
            leader_delta,
            Int32(1),
            thread_idx_in_group,
        )
        self._cta_rendezvous(participating_threads)


class NvlinkBarrier(SoftwareGridSync):
    """Sense-reversing all-rank barrier layered over software grid sync."""

    period = 4
    grid_counter_region = "nvlink.token_comm.grid_sync_counter"
    phase_counter_region = "nvlink.token_comm.nvlink_phase_counter"
    signal_region = "nvlink.token_comm.nvlink_signal"

    def __init__(
        self,
        *,
        world_size: int,
        barrier_id: int,
    ) -> None:
        super().__init__(barrier_id=barrier_id)
        self.world_size = world_size
        self._phase_counter = None
        self._signal = None
        self._peer_rank_ptr_mapper = None

    def register_device_workspace(self, workspace: DeviceWorkspace) -> None:
        super().register_device_workspace(workspace)
        workspace.register(
            self.phase_counter_region,
            cutlass.Int32,
            (1,),
            buffer_space="local",
            byte_alignment=16,
            reset="zero_on_first_allocate",
        )
        workspace.register(
            self.signal_region,
            cutlass.Int32,
            (2,),
            buffer_space="shared",
            byte_alignment=16,
            reset="zero_on_first_allocate",
        )

    @cute.jit
    def assign_device_members(
        self,
        workspace: DeviceWorkspace,
        peer_rank_ptr_mapper,
    ) -> None:
        super().assign_device_members(workspace)
        self._phase_counter = workspace.ptr(self.phase_counter_region)
        self._signal = workspace.ptr(self.signal_region)
        self._peer_rank_ptr_mapper = peer_rank_ptr_mapper

    def remove_device_members(self) -> None:
        super().remove_device_members()
        self._phase_counter = None
        self._signal = None
        self._peer_rank_ptr_mapper = None

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
            self.sync(
                participating_threads,
                actual_cta_count,
                linear_cta_idx,
                thread_idx_in_group,
            )

        if linear_cta_idx == Int32(0):
            status = cute.arch.load(
                self._phase_counter,
                Int32,
                sem="relaxed",
                scope="gpu",
            ) & Int32(3)
            signal_phase = status & Int32(1)
            signal_direction = status >> Int32(1)
            signal_delta = Int32(1)
            signal_target = Int32(self.world_size)
            if signal_direction != Int32(0):
                signal_delta = Int32(-1)
                signal_target = Int32(0)

            self._cta_rendezvous(participating_threads)
            if thread_idx_in_group == Int32(0):
                cute.arch.fence_acq_rel_sys()
            self._cta_rendezvous(participating_threads)

            rank_round_count = (
                self.world_size + participating_threads - 1
            ) // participating_threads
            signal_base_address = self._signal.toint()
            signal_byte_offset = Int64(signal_phase * Int32(4))
            for rank_round in cutlass.range_constexpr(rank_round_count):
                destination_rank = (
                    Int32(rank_round * participating_threads)
                    + thread_idx_in_group
                )
                if destination_rank < Int32(self.world_size):
                    destination_address = self._peer_rank_ptr_mapper.map(
                        signal_base_address,
                        destination_rank,
                        signal_byte_offset,
                    )
                    red_add_relaxed_sys_s32(
                        destination_address,
                        signal_delta,
                    )

            self._cta_rendezvous(participating_threads)
            if thread_idx_in_group == Int32(0):
                cute.arch.atomic_add(
                    self._phase_counter,
                    Int32(1),
                    sem="relaxed",
                    scope="gpu",
                )
                local_signal = self._signal + signal_phase
                while cute.arch.load(
                    local_signal,
                    Int32,
                    sem="acquire",
                    scope="sys",
                ) != signal_target:
                    pass

        if cutlass.const_expr(epilogue_grid_sync):
            self.sync(
                participating_threads,
                actual_cta_count,
                linear_cta_idx,
                thread_idx_in_group,
            )

    @cute.jit
    def finalize(
        self,
        completed_calls: int,
        participating_threads: int,
        actual_cta_count: Int32,
        linear_cta_idx: Int32,
        thread_idx_in_group: Int32,
    ) -> None:
        padding_calls = (-completed_calls) % self.period
        for _ in cutlass.range_constexpr(padding_calls):
            self.arrive_and_wait(
                participating_threads,
                actual_cta_count,
                linear_cta_idx,
                thread_idx_in_group,
                prologue_grid_sync=True,
                epilogue_grid_sync=True,
            )


@cute.jit
def _inline_grid_sync(
    counter,
    linear_cta_idx,
    leader_delta,
    other_delta,
    thread_idx_in_group,
) -> None:
    llvm.inline_asm(
        None,
        [
            counter.toint().ir_value(),
            Int32(linear_cta_idx).ir_value(),
            leader_delta.ir_value(),
            other_delta.ir_value(),
            Int32(thread_idx_in_group).ir_value(),
        ],
        (
            "{\n\t"
            ".reg .b32 %delta; .reg .b32 %old; .reg .b32 %current;\n\t"
            ".reg .pred %not_leader; .reg .pred %is_cta0; "
            ".reg .pred %waiting;\n\t"
            "setp.ne.u32 %not_leader, $4, 0;\n\t"
            "@%not_leader bra DONE;\n\t"
            "setp.eq.u32 %is_cta0, $1, 0;\n\t"
            "selp.b32 %delta, $2, $3, %is_cta0;\n\t"
            "atom.release.gpu.global.add.u32 %old, [$0], %delta;\n\t"
            "SPIN:\n\t"
            "ld.relaxed.gpu.global.b32 %current, [$0];\n\t"
            "xor.b32 %current, %current, %old;\n\t"
            "and.b32 %current, %current, 0x80000000;\n\t"
            "setp.eq.u32 %waiting, %current, 0;\n\t"
            "@%waiting bra SPIN;\n\t"
            "fence.acq_rel.gpu;\n\t"
            "DONE:\n\t"
            "}"
        ),
        "l,r,r,r,r",
        has_side_effects=True,
        asm_dialect=0,
    )


__all__ = ["NvlinkBarrier", "SoftwareGridSync"]
