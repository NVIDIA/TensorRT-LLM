# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sched extension for fused fc1+fc2 work-tile enrichment and GMEM slicing."""

from typing import List, Optional, Tuple, Union

import cutlass
import cutlass.cute as cute
from cutlass._mlir import ir
from cutlass.cute.typing import Pointer
from cutlass.cutlass_dsl import Int32, extract_mlir_values, new_from_mlir_values
from cutlass.utils.blockscaled_layout import tile_atom_to_shape_SF

from .fc1_fc2_fuse_sched import BlockPhase
from .moe_persistent_scheduler import MoESchedExtension, MoEWorkTileInfo
from .moe_utils import rewrite_tensor_shape, spin_wait

PhaseBits = 16
PhaseMask = (1 << PhaseBits) - 1
PeekReadyBit = 1 << PhaseBits

# =============================================================================
# Fused FC1+FC2 WorkTileInfo
# =============================================================================


class SwapABSwigluFp4Fc12WorkTileInfo(MoEWorkTileInfo):
    """8-field fc12 work tile; slot 3 aliases base k_tile_cnt."""

    TotalFields = 8  # 4 base + 4 extra (4 new fields beyond the alias)

    def __init__(
        self,
        expert_idx: Int32,
        tile_m_idx: Int32,
        tile_n_idx: Int32,
        cumulative_data_physical_row: Int32,
        cumulative_sf_physical_row: Int32,
        cumulative_token_block_count: Int32,
        valid_tokens_in_tile: Int32,
        phase_and_peek: Int32,
    ):
        # Slot 3 reuses base k_tile_cnt storage.
        super().__init__(
            expert_idx,
            tile_m_idx,
            tile_n_idx,
            cumulative_data_physical_row,
        )
        self.cumulative_data_physical_row = self.k_tile_cnt
        self.cumulative_sf_physical_row = cumulative_sf_physical_row
        self.cumulative_token_block_count = cumulative_token_block_count
        self.valid_tokens_in_tile = valid_tokens_in_tile
        # Slot 7 is the packed (BlockPhase | (peek_ready << 16)) field.
        # The ``.phase`` and ``.peek_ready`` properties below unpack it;
        # consumers call them directly so the codebase reads as if the
        # two pieces were separate fields.
        self.phase_and_peek = phase_and_peek

    @property
    def phase(self) -> Int32:
        """Decode the BlockPhase from slot 7's low 16 bits."""
        return self.phase_and_peek & Int32(PhaseMask)

    @property
    def peek_ready(self):
        """Decode the sched-warp counter peek result from slot 7's bit 16.

        Returns a Boolean SSA: True iff the sched-warp's enrich-time
        peek of the fc1_done_counter (for this fc2 work tile) observed
        saturation, allowing the TMA-B warp to skip its own spin_wait.
        For fc1 tiles or when peek wasn't done, returns False (fc12 sched
        ext only sets the bit on Linear2 phase work tiles).
        """
        return ((self.phase_and_peek >> Int32(PhaseBits))
                & Int32(1)) != Int32(0)

    def __extract_mlir_values__(self) -> List[ir.Value]:
        # Base's __extract_mlir_values__ already emits the first 4 slots
        # (slot 3 = self.k_tile_cnt = cumulative_data_physical_row).
        values = super().__extract_mlir_values__()
        values.extend(extract_mlir_values(self.cumulative_sf_physical_row))
        values.extend(extract_mlir_values(self.cumulative_token_block_count))
        values.extend(extract_mlir_values(self.valid_tokens_in_tile))
        values.extend(extract_mlir_values(self.phase_and_peek))
        return values

    def __new_from_mlir_values__(
            self, values: List[ir.Value]) -> "SwapABSwigluFp4Fc12WorkTileInfo":
        assert len(values) == 8
        return SwapABSwigluFp4Fc12WorkTileInfo(
            expert_idx=new_from_mlir_values(self.expert_idx, [values[0]]),
            tile_m_idx=new_from_mlir_values(self.tile_m_idx, [values[1]]),
            tile_n_idx=new_from_mlir_values(self.tile_n_idx, [values[2]]),
            cumulative_data_physical_row=new_from_mlir_values(
                self.cumulative_data_physical_row, [values[3]]),
            cumulative_sf_physical_row=new_from_mlir_values(
                self.cumulative_sf_physical_row, [values[4]]),
            cumulative_token_block_count=new_from_mlir_values(
                self.cumulative_token_block_count, [values[5]]),
            valid_tokens_in_tile=new_from_mlir_values(self.valid_tokens_in_tile,
                                                      [values[6]]),
            phase_and_peek=new_from_mlir_values(self.phase_and_peek,
                                                [values[7]]),
        )

    def to_rmem(self) -> cute.Tensor:
        rmem = cute.make_rmem_tensor((self.TotalFields, ), Int32)
        rmem[0] = self.expert_idx
        rmem[1] = self.tile_m_idx
        rmem[2] = self.tile_n_idx
        rmem[3] = self.k_tile_cnt  # = cumulative_data_physical_row
        rmem[4] = self.cumulative_sf_physical_row
        rmem[5] = self.cumulative_token_block_count
        rmem[6] = self.valid_tokens_in_tile
        rmem[7] = self.phase_and_peek
        return rmem

    @classmethod
    def from_rmem(cls, rmem: cute.Tensor) -> "SwapABSwigluFp4Fc12WorkTileInfo":
        return cls(
            expert_idx=rmem[0],  # type: ignore[arg-type]
            tile_m_idx=rmem[1],  # type: ignore[arg-type]
            tile_n_idx=rmem[2],  # type: ignore[arg-type]
            cumulative_data_physical_row=rmem[3],  # type: ignore[arg-type]
            cumulative_sf_physical_row=rmem[4],  # type: ignore[arg-type]
            cumulative_token_block_count=rmem[5],  # type: ignore[arg-type]
            valid_tokens_in_tile=rmem[6],  # type: ignore[arg-type]
            phase_and_peek=rmem[7],  # type: ignore[arg-type]
        )


# =============================================================================
# Fused FC1+FC2 SchedExtension
# =============================================================================


class SwapABSwigluFp4Fc12SchedExtension(MoESchedExtension):
    """Sched extension for the fused fc1+fc2 swap-AB SwiGLU NVFP4 kernel.

    ``WorkTileInfo = SwapABSwigluFp4Fc12WorkTileInfo``.  The 8th slot stores
    ``phase_and_peek`` (low 16 bit BlockPhase, bit 16 sched-warp peek result);
    consumers read it through ``.phase`` and ``.peek_ready``.

    `enrich_work_tile_info` packs a sched-warp counter peek for fc2 tiles.
    `get_gmem_tensor` is phase-invariant; the caller supplies the phase-specific
    physical tensor.
    """

    WorkTileInfo = SwapABSwigluFp4Fc12WorkTileInfo

    def __init__(
        self,
        sf_vec_size: int,
        fc1_done_counter_ptr: Pointer,
        fc2_spin_threshold: Union[int, Int32],
        # MegaMoE-only: when set, ``enrich_work_tile_info`` also peeks the
        # dispatch->fc1 release-counter for fc1 phase tiles, so the fc1
        # TMA-B warp can skip its blocking spin when the counter already
        # shows enough arrivals.  Mirrors ``fc1_done_counter_ptr`` for the
        # fc1->fc2 link: this side is "fc1 input ready", that side is
        # "fc1 output done".  The threshold per-tile is the tile's
        # ``valid_tokens_in_tile`` (dispatch does not pull padding
        # tokens), read straight off the base work tile -- no separate
        # threshold field needed.  ``None`` in the lean fc1+fc2 path keeps
        # ``enrich_work_tile_info`` to its existing fc2-only peek shape and
        # this pointer is not carried through MLIR.
        fc1_ready_counter_ptr: Optional[Pointer] = None,
    ):
        super().__init__(workspace=None)
        if sf_vec_size <= 0:
            raise ValueError(
                f"sf_vec_size must be positive, got {sf_vec_size}.")
        self.sf_vec_size = sf_vec_size
        self.fc1_done_counter_ptr = fc1_done_counter_ptr
        # Coerce to Int32 SSA so downstream serialization / arithmetic
        # never has to type-discriminate.  Python-int callers (the
        # ``static_expert_shape`` path) get a constant SSA op which IR
        # canonicalize folds to an immediate; runtime-Int32 callers
        # passthrough.  Net IR is identical.
        self.fc2_spin_threshold = Int32(fc2_spin_threshold)
        self.fc1_ready_counter_ptr = fc1_ready_counter_ptr

    def __extract_mlir_values__(self) -> List[ir.Value]:
        values: List[ir.Value] = []
        values.extend(extract_mlir_values(self.fc1_done_counter_ptr))
        values.extend(extract_mlir_values(self.fc2_spin_threshold))
        if self.fc1_ready_counter_ptr is not None:
            values.extend(extract_mlir_values(self.fc1_ready_counter_ptr))
        return values

    def __new_from_mlir_values__(
            self,
            values: List[ir.Value]) -> "SwapABSwigluFp4Fc12SchedExtension":
        ptr_len = len(extract_mlir_values(self.fc1_done_counter_ptr))
        thresh_len = len(extract_mlir_values(self.fc2_spin_threshold))
        idx = 0
        new_ptr = new_from_mlir_values(self.fc1_done_counter_ptr,
                                       values[idx:idx + ptr_len])
        idx += ptr_len
        new_threshold = new_from_mlir_values(self.fc2_spin_threshold,
                                             values[idx:idx + thresh_len])
        idx += thresh_len
        # fc1_ready_counter_ptr: prototype tells us whether it is carried.
        if self.fc1_ready_counter_ptr is not None:
            ready_ptr_len = len(extract_mlir_values(self.fc1_ready_counter_ptr))
            new_ready_ptr = new_from_mlir_values(
                self.fc1_ready_counter_ptr, values[idx:idx + ready_ptr_len])
            idx += ready_ptr_len
        else:
            new_ready_ptr = None
        assert idx == len(values), (
            f"SwapABSwigluFp4Fc12SchedExtension serialization mismatch: "
            f"idx={idx} len(values)={len(values)}")
        result = SwapABSwigluFp4Fc12SchedExtension.__new__(
            SwapABSwigluFp4Fc12SchedExtension)
        result.workspace = None
        result.sf_vec_size = self.sf_vec_size  # codegen const passthrough
        result.fc1_done_counter_ptr = new_ptr
        result.fc2_spin_threshold = new_threshold
        result.fc1_ready_counter_ptr = new_ready_ptr
        return result

    # --------------------------------------------------------------
    # enrich_work_tile_info — sched-warp fc2 counter peek + pack
    # --------------------------------------------------------------

    @cute.jit
    def enrich_work_tile_info(
        self,
        base_work: SwapABSwigluFp4Fc12WorkTileInfo,
    ) -> SwapABSwigluFp4Fc12WorkTileInfo:
        """Pack a non-blocking counter peek into ``phase_and_peek``.

        - fc2 tiles always peek the fc1->fc2 ``fc1_done_counter`` at
          ``cumulative_token_block_count + tile_n_idx`` against
          ``self.fc2_spin_threshold`` (work-tile-invariant const).
        - fc1 tiles peek the dispatch->fc1 ``fc1_ready_counter`` at the
          same slot index but with ``valid_tokens_in_tile`` as threshold
          (per-tile dynamic).  This branch only emits when
          ``self.fc1_ready_counter_ptr is not None`` (MegaMoE mode).
        """
        # Invalid tiles keep (None_ | 0); do not index an arbitrary counter slot.
        is_valid = base_work.is_valid_tile

        new_phase_and_peek = base_work.phase_and_peek
        if is_valid:
            # Same slot index for both phases -- fc1 release-add (dispatch
            # pull) and fc2 release-add (fc1 epi) target the per-task-tile
            # counter slot indexed by ``cumulative_token_block_count +
            # tile_n_idx``.
            counter_slot = (base_work.cumulative_token_block_count +
                            base_work.tile_n_idx)
            is_fc1 = base_work.phase == Int32(int(BlockPhase.Linear1))
            is_fc2 = base_work.phase == Int32(int(BlockPhase.Linear2))

            # MegaMoE-only: fc1 phase peek on fc1_ready_counter.  Threshold
            # is dynamic (per-tile valid count) because dispatch does not
            # pull padding tokens, so the counter's terminal value matches
            # the tile's valid_tokens_in_tile (cluster_tile_m for full
            # tiles, less for an expert's last partial tile).
            if cutlass.const_expr(self.fc1_ready_counter_ptr is not None):
                if is_fc1:
                    counter_ptr = self.fc1_ready_counter_ptr + counter_slot
                    peek_ready = spin_wait(
                        counter_ptr,
                        lambda v: v >= base_work.valid_tokens_in_tile,
                        peek_only=True,
                    )
                    peek_bit = Int32(0)
                    if peek_ready:
                        peek_bit = Int32(PeekReadyBit)
                    new_phase_and_peek = base_work.phase_and_peek | peek_bit

            # fc2 tiles can skip the later TMA-B spin (existing path).
            if is_fc2:
                counter_ptr = self.fc1_done_counter_ptr + counter_slot
                # peek_only=True: single ld.cg + cmp, returns Boolean.
                # ``self.fc2_spin_threshold`` was Int32-coerced in __init__.
                peek_ready = spin_wait(
                    counter_ptr,
                    lambda v: v >= self.fc2_spin_threshold,
                    peek_only=True,
                )
                # Pack peek bit into slot 7's bit 16.  Use the runtime-if
                # assign-an-iter-arg-int idiom (same pattern as
                # ``_advance_expert_within_phase`` for phase-aware tile
                # count selection); avoids relying on Boolean->Int32
                # implicit casts whose presence is dialect-version-dependent.
                peek_bit = Int32(0)
                if peek_ready:
                    peek_bit = Int32(PeekReadyBit)
                new_phase_and_peek = base_work.phase_and_peek | peek_bit

        return SwapABSwigluFp4Fc12WorkTileInfo(
            expert_idx=base_work.expert_idx,
            tile_m_idx=base_work.tile_m_idx,
            tile_n_idx=base_work.tile_n_idx,
            cumulative_data_physical_row=base_work.cumulative_data_physical_row,
            cumulative_sf_physical_row=base_work.cumulative_sf_physical_row,
            cumulative_token_block_count=base_work.cumulative_token_block_count,
            valid_tokens_in_tile=base_work.valid_tokens_in_tile,
            phase_and_peek=new_phase_and_peek,
        )

    # --------------------------------------------------------------
    # get_gmem_tensor — phase-aware
    # --------------------------------------------------------------

    @cute.jit
    def get_gmem_tensor(
        self,
        tensor_name: str,
        gmem_tensor_in_moe_view: cute.Tensor,
        work_tile_info: SwapABSwigluFp4Fc12WorkTileInfo,
    ) -> Tuple[cute.Tensor, Optional[Pointer]]:
        """Phase-invariant GMEM slice for the 6 operands.

        Weight operands anchor at expert; data and SF operands use separate
        token offsets because their padding granularities differ.  Caller
        passes the phase-specific physical tensor.  Desc-ptr is always None.
        """
        expert_idx = work_tile_info.expert_idx
        data_token_offset = work_tile_info.cumulative_data_physical_row
        sf_token_offset = work_tile_info.cumulative_sf_physical_row

        shape = gmem_tensor_in_moe_view.shape
        stride = gmem_tensor_in_moe_view.stride
        c1 = cutlass.Int32(1)
        sf_vec_size = self.sf_vec_size

        if cutlass.const_expr(tensor_name == "a"):
            real = cute.domain_offset((0, 0, expert_idx),
                                      gmem_tensor_in_moe_view)
            real = rewrite_tensor_shape(
                real, (shape[0], shape[1], c1))  # type: ignore[index]
            return (real, None)

        elif cutlass.const_expr(tensor_name == "b"):
            real = cute.domain_offset((data_token_offset, 0, 0),
                                      gmem_tensor_in_moe_view)
            real = rewrite_tensor_shape(
                real, (shape[0], shape[1], c1))  # type: ignore[index]
            return (real, None)

        elif cutlass.const_expr(tensor_name == "sfa"):
            real = cute.domain_offset((0, 0, expert_idx),
                                      gmem_tensor_in_moe_view)
            per_expert_shape = (shape[0], shape[1], c1)  # type: ignore[index]
            sf_layout = tile_atom_to_shape_SF(per_expert_shape, sf_vec_size)
            real = cute.make_tensor(
                real.iterator, cute.make_layout(sf_layout.shape, stride=stride))
            return (real, None)

        elif cutlass.const_expr(tensor_name == "sfb"):
            real = cute.domain_offset((sf_token_offset, 0, 0),
                                      gmem_tensor_in_moe_view)
            per_expert_shape = (shape[0], shape[1], c1)  # type: ignore[index]
            sf_layout = tile_atom_to_shape_SF(per_expert_shape, sf_vec_size)
            real = cute.make_tensor(
                real.iterator, cute.make_layout(sf_layout.shape, stride=stride))
            return (real, None)

        elif cutlass.const_expr(tensor_name == "c"):
            real = cute.domain_offset((data_token_offset, 0, 0),
                                      gmem_tensor_in_moe_view)
            real = rewrite_tensor_shape(
                real, (shape[0], shape[1], c1))  # type: ignore[index]
            return (real, None)

        elif cutlass.const_expr(tensor_name == "sfc"):
            # Linear1 phase only — fc2 has no output SF.  Caller must not
            # invoke this branch with ``work_tile_info.phase == Linear2``.
            real = cute.domain_offset((sf_token_offset, 0, 0),
                                      gmem_tensor_in_moe_view)
            per_expert_shape = (shape[0], shape[1], c1)  # type: ignore[index]
            sf_layout = tile_atom_to_shape_SF(per_expert_shape, sf_vec_size)
            real = cute.make_tensor(
                real.iterator, cute.make_layout(sf_layout.shape, stride=stride))
            return (real, None)

        elif cutlass.const_expr(tensor_name == "topk"):
            # Linear1 phase only — fc2 doesn't consume topk weights (Path A:
            # topk weight is pre-multiplied into fc1's swiglu fp32 output
            # before NVFP4 quantize, so fc2 mainloop already reads the
            # weight-scaled values from fc1's output buffer).
            #
            # ``gmem_tensor_in_moe_view`` here is the global per-token
            # ``topk_scores`` 1D tensor of shape ``(data_total_rows,)``.
            # Caller passes the global tensor; we shift to the current
            # expert's slice via ``data_token_offset`` (data-side physical
            # row offset, same shift used by ``b`` / ``c`` operands).
            #
            # Returned view shape is ``(this_expert_padded_rows,)`` (same
            # length as the input but offset to the right slice).  The
            # epilogue then indexes it with the **expert-local** token
            # coord — symmetric with the SFC write pattern.
            real = cute.domain_offset((data_token_offset, ),
                                      gmem_tensor_in_moe_view)
            return (real, None)

        raise ValueError(f"Unknown tensor_name: {tensor_name!r}.")

    # --------------------------------------------------------------
    # prefetch_for_expert
    # --------------------------------------------------------------

    @cute.jit
    def prefetch_for_expert(self, expert_idx: Int32) -> None:
        """No-op: swap-AB makes every TMA desc tile-invariant in both phases."""
