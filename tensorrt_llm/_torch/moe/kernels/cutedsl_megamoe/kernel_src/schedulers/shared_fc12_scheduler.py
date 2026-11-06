# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Work tiles of the shared expert fused into the swap-AB FC12 kernel."""

import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import Int32

from ...helpers.device_workspace import DeviceWorkspace
from ...helpers.utils import ceil_div
from .base import SchedulerBase
from .fc12_mapping import BlockPhase, SwapAbFc12WorkTileInfo


class SharedExpertTileSchedule:
    """Statically spread shared-expert tiles over the canonical cluster slots.

    Every CTA of a cluster walks the same work IDs (slot, slot + slots, ...), so TMA multicast and
    2-CTA MMA see identical tile sequences without a claim protocol. A work ID is
    ``token_block * output_blocks + output_block``.
    """

    done_counter_region = "fc12.shared_expert.fc1_done_counter"

    def __init__(
        self,
        *,
        fc1_output_blocks: int,
        fc2_output_blocks: int,
        tokens_per_tile: int,
        cluster_m: int,
        slot_count: int,
        max_tokens: int,
    ) -> None:
        self.fc1_output_blocks = fc1_output_blocks
        self.fc2_output_blocks = fc2_output_blocks
        self.tokens_per_tile = tokens_per_tile
        self.cluster_m = cluster_m
        self.slot_count = slot_count
        self.done_counter_count = ceil_div(max_tokens, tokens_per_tile)

    def register_device_workspace(self, device_workspace: DeviceWorkspace) -> None:
        """One completion counter per token tile of the gate/up projection."""
        device_workspace.register(
            self.done_counter_region,
            cutlass.Int32,
            (self.done_counter_count,),
            buffer_space="local",
            byte_alignment=16,
            reset="tail_reset",
        )

    @cute.jit
    def publish(
        self,
        scheduler: SchedulerBase,
        phase: int,
        num_tokens: Int32,
        slot: Int32,
        cta_m: Int32,
        slots: str = "all",
    ) -> SchedulerBase:
        """Publish this cluster slot's tiles of one projection; returns the advanced scheduler.

        ``slots`` keeps the tiles of slots "with_gate_up" or "without_gate_up" tiles only (default: "all")."""
        output_blocks = (
            self.fc1_output_blocks if phase == BlockPhase.SharedLinear1 else self.fc2_output_blocks
        )
        token_blocks = (num_tokens + Int32(self.tokens_per_tile - 1)) // Int32(self.tokens_per_tile)
        total = token_blocks * Int32(output_blocks)
        # Down-projection tiles start from the other end, away from the slots that ran gate/up tiles.
        work_id = slot if phase == BlockPhase.SharedLinear1 else Int32(self.slot_count - 1) - slot
        if cutlass.const_expr(slots != "all"):
            skip = slot < token_blocks * Int32(self.fc1_output_blocks)
            if cutlass.const_expr(slots == "with_gate_up"):
                skip = not skip
            if skip:
                work_id = total
        while work_id < total:
            token_block = work_id // Int32(output_blocks)
            output_block = work_id - token_block * Int32(output_blocks)
            valid_tokens = cutlass.min(
                num_tokens - token_block * Int32(self.tokens_per_tile), Int32(self.tokens_per_tile)
            )
            scheduler.publish_work(
                SwapAbFc12WorkTileInfo(
                    expert_idx=Int32(0),
                    tile_m_idx=output_block * Int32(self.cluster_m) + cta_m,
                    tile_n_idx=token_block,
                    cumulative_data_physical_row=Int32(0),
                    cumulative_sf_physical_row=Int32(0),
                    cumulative_token_block_count=Int32(0),
                    valid_tokens_in_cta_tile=valid_tokens,
                    phase_and_flags=Int32(phase),
                )
            )
            work_id = work_id + Int32(self.slot_count)
        return scheduler


__all__ = ["SharedExpertTileSchedule"]
