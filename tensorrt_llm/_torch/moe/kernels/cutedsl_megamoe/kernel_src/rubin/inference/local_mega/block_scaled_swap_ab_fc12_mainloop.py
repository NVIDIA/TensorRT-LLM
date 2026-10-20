# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mainloop component for the Rubin block-scaled swap-AB FC12 kernel."""

from typing import ClassVar, Tuple

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
import cutlass.utils.rubin_helpers as sm107_utils
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, tcgen05
from cutlass.cutlass_dsl import Int32

from .....api import ImplDesc, KernelComponent, ProblemDesc, StaticOrRuntimeIntegerType
from .....helpers.cute_py_helpers import (
    Tcgen05MmaInstruction,
    make_smem_layouts,
    tcgen05_block_scaled_acc_dtype,
    tcgen05_smem_alloc_type,
)
from .....helpers.dsl_helpers import spin_wait, tma_multicast_mask
from .....helpers.iket_compat import iket
from .....helpers.smem_workspace import SmemWorkspace
from .....helpers.utils import ceil_div, round_up, strides_equal_ignoring_singletons
from .....quant_def import QuantKind
from ....schedulers.base import SchedulerConsumer
from ....schedulers.fc12_mapping import BlockPhase
from ..mega.block_scaled_swap_ab_fc12_extension import BlockScaledSwapAbFc12Extension
from .local_routing import RoutingMetadata
from .tma_gather import sm107_tma_gather4_load


class BlockScaledSwapAbFc12Mainloop(KernelComponent):
    """Own all Rubin load, MMA, pipeline, SMEM, and TMEM mainloop state."""

    ab_pipeline_mbarriers_region: ClassVar[str] = (
        "rubin.swap_ab_fc12.mainloop.ab_pipeline_mbarriers"
    )
    a_smem_tensor_region: ClassVar[str] = "rubin.swap_ab_fc12.mainloop.a_smem_tensor"
    b_smem_tensor_region: ClassVar[str] = "rubin.swap_ab_fc12.mainloop.b_smem_tensor"
    sfa_smem_tensor_region: ClassVar[str] = "rubin.swap_ab_fc12.mainloop.sfa_smem_tensor"
    sfb_smem_tensor_region: ClassVar[str] = "rubin.swap_ab_fc12.mainloop.sfb_smem_tensor"
    acc_pipeline_mbarriers_region: ClassVar[str] = (
        "rubin.swap_ab_fc12.mainloop.acc_pipeline_mbarriers"
    )
    tmem_holding_buffer_region: ClassVar[str] = "rubin.swap_ab_fc12.mainloop.tmem_holding_buffer"
    tmem_deallocation_mbarrier_region: ClassVar[str] = (
        "rubin.swap_ab_fc12.mainloop.tmem_deallocation_mbarrier"
    )
    tmem_allocation_barrier_id: ClassVar[int] = 2
    fc1_gather_sync_barrier_id: ClassVar[int] = 3

    @classmethod
    def problem_desc_require(cls) -> dict[str, type]:
        return {
            "quant_kind": str,
            "a_major_mode": OperandMajorMode,
            "b_major_mode": OperandMajorMode,
            "hidden_size": StaticOrRuntimeIntegerType,
            "intermediate_gateup_size": StaticOrRuntimeIntegerType,
        }

    @classmethod
    def impl_desc_require(cls) -> dict[str, type]:
        return {
            "mma_instruction_mnk": tuple,
            "mma_tiler_mnk": tuple,
            "mma_k_mode": str,
            "cluster_shape_mn": tuple,
            "use_2cta_instrs": bool,
            "mainloop_smem_budget_bytes": int,
            "num_accumulator_consumer_warps_per_cta": int,
        }

    def __init__(self, problem_desc: ProblemDesc, impl_desc: ImplDesc) -> None:
        self._validate_desc_inputs(problem_desc, impl_desc)

        self.quant_kind = QuantKind(problem_desc["quant_kind"])
        self.a_dtype = self.quant_kind.weight_dtype
        self.b_dtype = self.quant_kind.activation_dtype
        self.sf_dtype = self.quant_kind.sf_dtype
        self.sf_vec_size = self.quant_kind.sf_vec_size
        self.acc_dtype = tcgen05_block_scaled_acc_dtype
        self.a_major_mode = problem_desc["a_major_mode"]
        self.b_major_mode = problem_desc["b_major_mode"]
        self.hidden_size = problem_desc["hidden_size"]
        self.intermediate_gateup_size = problem_desc["intermediate_gateup_size"]
        self.mma_instruction_mnk = impl_desc["mma_instruction_mnk"]
        self.mma_tiler_mnk = impl_desc["mma_tiler_mnk"]
        self.mma_k_mode = impl_desc["mma_k_mode"]
        self.cluster_shape_mn = impl_desc["cluster_shape_mn"]
        self.use_2cta_instrs = impl_desc["use_2cta_instrs"]
        self.mainloop_smem_budget_bytes = impl_desc["mainloop_smem_budget_bytes"]
        self.num_accumulator_consumer_warps_per_cta = impl_desc[
            "num_accumulator_consumer_warps_per_cta"
        ]

        self.architecture = "sm_107"
        self.mma_cta_count = 2 if self.use_2cta_instrs else 1
        self.instruction_k = self.mma_instruction_mnk[2]
        self.num_mma_instructions_per_ab_stage = self.mma_tiler_mnk[2] // self.instruction_k
        self._validate_configuration()

        self.cta_tile_m = self.mma_tiler_mnk[0] // self.mma_cta_count
        self.cta_tile_n = self.mma_tiler_mnk[1]
        self.mma_tile_k = self.mma_tiler_mnk[2]
        tmem_plan = impl_desc["tmem_plan"]
        self.num_sfa_tmem_cols = tmem_plan.sfa_columns
        self.num_sfb_tmem_cols = tmem_plan.sfb_columns
        self.num_sf_tmem_cols = tmem_plan.sfa_columns + tmem_plan.sfb_columns
        self.num_tmem_alloc_cols = tmem_plan.allocation_columns
        self.num_accumulator_stages = tmem_plan.accumulator_stage_count
        self.num_accumulator_pipeline_stages = tmem_plan.accumulator_pipeline_stages
        if tmem_plan.accumulator_stage_stride_columns != tmem_plan.accumulator_stage_columns:
            raise ValueError("Rubin Local MegaMoE does not support overlapping accumulator stages.")
        if tmem_plan.accumulator_pipeline_stages != tmem_plan.accumulator_stage_count:
            raise ValueError(
                "Rubin Local MegaMoE requires one pipeline stage per disjoint accumulator stage."
            )
        self.num_accumulator_tmem_cols = tmem_plan.accumulator_columns
        self.accumulator_shape = (
            self.cta_tile_m,
            self.cta_tile_n,
            tmem_plan.accumulator_stage_count,
        )
        self.accumulator_stride = (1 << 16, 1, tmem_plan.accumulator_stage_stride_columns)

        self.mma_instruction = Tcgen05MmaInstruction(
            a_type=self.a_dtype,
            b_type=self.b_dtype,
            acc_type=self.acc_dtype,
            instruction_mnk=self.mma_instruction_mnk,
            participates=self.mma_cta_count,
            sfa_type=self.sf_dtype,
            sfb_type=self.sf_dtype,
            sf_vec_size=self.sf_vec_size,
        )
        self.a_smem_alloc_dtype = tcgen05_smem_alloc_type(
            self.a_dtype, self.b_dtype, self.architecture
        )
        self.b_smem_alloc_dtype = tcgen05_smem_alloc_type(
            self.b_dtype, self.a_dtype, self.architecture
        )
        self.sfb_instruction_shape_mnk = (
            self.mma_instruction_mnk[0] // self.mma_cta_count,
            round_up(self.mma_instruction_mnk[1], 128),
            self.instruction_k,
        )
        self.sfb_mma_instruction = Tcgen05MmaInstruction(
            a_type=self.a_dtype,
            b_type=self.b_dtype,
            acc_type=self.acc_dtype,
            instruction_mnk=self.sfb_instruction_shape_mnk,
            participates=1,
            sfa_type=self.sf_dtype,
            sfb_type=self.sf_dtype,
            sf_vec_size=self.sf_vec_size,
        )
        self.mma_tiler_sfb = (
            self.sfb_instruction_shape_mnk[0],
            self.sfb_instruction_shape_mnk[1],
            self.mma_tile_k,
        )
        self.cluster_layout_shape_vmnk = (
            (self.mma_cta_count,),
            self.cluster_shape_mn[0] // self.mma_cta_count,
            self.cluster_shape_mn[1],
            1,
        )
        self.cluster_layout_sfb_shape_vmnk = ((1,), *self.cluster_shape_mn, 1)
        self.num_mcast_ctas_a = self.cluster_layout_shape_vmnk[2]
        self.num_mcast_ctas_b = self.cluster_layout_shape_vmnk[1]
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        self._select_ab_stage_plan()
        self.num_tma_load_bytes = self.ab_stage_tma_bytes * self.mma_cta_count
        if self.num_tma_load_bytes != (
            self.a_stage_tma_bytes
            + self.b_stage_tma_bytes
            + self.sfa_stage_tma_bytes
            + self.sfb_stage_tma_bytes
        ):
            raise ValueError("Local MegaMoE AB stage transaction byte accounting is inconsistent.")
        if self.num_tma_load_bytes % 2 != 0:
            raise ValueError(
                "Local MegaMoE total AB transaction bytes must split evenly across two producer arrivals."
            )
        self.ab_pipeline_tx_count_per_producer = self.num_tma_load_bytes // 2
        self.num_gather_b_warps = 4
        self.gather_b_rows_per_cta = self.cta_tile_n // self.mma_cta_count
        self.gather_b_rows_per_warp = self.gather_b_rows_per_cta // self.num_gather_b_warps
        if self.gather_b_rows_per_warp % 4 != 0:
            raise ValueError(
                "Local MegaMoE gather-B rows per warp must be divisible by gather4 width."
            )
        self.gather_b_calls_per_warp = self.gather_b_rows_per_warp // 4
        self.gather_b_stage_bytes = self.cta_tile_n * self.mma_tile_k * int(self.b_dtype.width) // 8
        if self.gather_b_stage_bytes != self.b_stage_tma_bytes:
            raise ValueError(
                "Local MegaMoE FC1 Gather-B and FC2 TMA-B must transfer identical B operand bytes per AB stage."
            )
        if self.gather_b_stage_bytes % self.num_gather_b_warps != 0:
            raise ValueError("Local MegaMoE gather-B bytes must split evenly across four warps.")
        self.gather_b_stage_bytes_per_warp = self.gather_b_stage_bytes // self.num_gather_b_warps
        self.fc1_sfb_stage_bytes = self.sfb_stage_tma_bytes
        fc1_b_side_stage_bytes = self.gather_b_stage_bytes + self.fc1_sfb_stage_bytes
        fc2_b_side_stage_bytes = self.b_stage_tma_bytes + self.sfb_stage_tma_bytes
        if fc1_b_side_stage_bytes != fc2_b_side_stage_bytes:
            raise ValueError(
                "Local MegaMoE FC1 and FC2 B-side transaction bytes must be identical."
            )
        if self.fc1_sfb_stage_bytes <= 0:
            raise ValueError("Local MegaMoE FC1 SFB transaction byte count must be positive.")
        self.fc1_gather_sync_barrier = pipeline.NamedBarrier(
            barrier_id=self.fc1_gather_sync_barrier_id,
            num_threads=32 * (1 + self.num_gather_b_warps),
        )

    def _validate_configuration(self) -> None:
        if self.mma_k_mode != "2x":
            raise ValueError(
                f"Rubin Local MegaMoE only supports mma_k_mode='2x', got {self.mma_k_mode!r}."
            )
        for field_name, dimensions, expected_rank in (
            ("mma_instruction_mnk", self.mma_instruction_mnk, 3),
            ("mma_tiler_mnk", self.mma_tiler_mnk, 3),
            ("cluster_shape_mn", self.cluster_shape_mn, 2),
        ):
            if len(dimensions) != expected_rank:
                raise ValueError(f"{field_name} must contain {expected_rank} dimensions.")
            if not all(
                isinstance(dimension, int) and not isinstance(dimension, bool)
                for dimension in dimensions
            ):
                raise TypeError(f"{field_name} dimensions must be Python integers.")
            if any(dimension <= 0 for dimension in dimensions):
                raise ValueError(f"{field_name} dimensions must be positive.")

        expected_instruction_k = self.quant_kind.instruction_k(self.mma_k_mode)
        if self.instruction_k != expected_instruction_k:
            raise ValueError(
                f"{self.quant_kind} requires Rubin 2x instruction K={expected_instruction_k}, got {self.instruction_k}."
            )
        expected_instruction_m = 256 if self.use_2cta_instrs else 128
        if self.mma_instruction_mnk[0] != expected_instruction_m:
            raise ValueError(
                f"{self.mma_cta_count}-CTA Rubin MMA requires instruction M={expected_instruction_m}, "
                f"got {self.mma_instruction_mnk[0]}."
            )
        if self.mma_instruction_mnk[:2] != self.mma_tiler_mnk[:2]:
            raise NotImplementedError(
                "Rubin Local MegaMoE does not implement M/N instruction repetition or B-reuse."
            )
        if self.mma_tiler_mnk[1] not in (64, 128, 256):
            raise ValueError("Rubin Local MegaMoE supports tile N in (64, 128, 256).")
        if self.mma_tiler_mnk[2] % self.instruction_k != 0:
            raise ValueError("mma_tiler K must be divisible by instruction K.")
        if self.cluster_shape_mn[0] % self.mma_cta_count != 0:
            raise ValueError("cluster M must be divisible by the MMA CTA count.")
        if self.mainloop_smem_budget_bytes <= 0:
            raise ValueError("mainloop_smem_budget_bytes must be positive.")
        if self.num_accumulator_consumer_warps_per_cta <= 0:
            raise ValueError("num_accumulator_consumer_warps_per_cta must be positive.")
        if isinstance(self.hidden_size, int) and self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive.")
        if isinstance(self.intermediate_gateup_size, int):
            if self.intermediate_gateup_size <= 0:
                raise ValueError("intermediate_gateup_size must be positive.")
            if self.intermediate_gateup_size % 2 != 0:
                raise ValueError("intermediate_gateup_size must be even.")

    def register_smem_regions(self, smem_workspace: SmemWorkspace) -> None:
        """Register the selected Rubin mainloop SMEM plan."""
        smem_workspace.register_mbarrier(
            self.ab_pipeline_mbarriers_region, self.num_ab_pipeline_stages * 2
        )
        for region_name, region in (
            (self.a_smem_tensor_region, self.a_smem_region),
            (self.b_smem_tensor_region, self.b_smem_region),
            (self.sfa_smem_tensor_region, self.sfa_smem_region),
            (self.sfb_smem_tensor_region, self.sfb_smem_region),
        ):
            smem_workspace.register_tensor(
                region_name,
                region.dtype,
                region.shape,
                stride=region.stride,
                swizzle=region.swizzle,
                byte_alignment=region.byte_alignment,
            )
        smem_workspace.register_mbarrier(
            self.acc_pipeline_mbarriers_region, self.num_accumulator_pipeline_stages * 2
        )
        smem_workspace.register_mbarrier(self.tmem_deallocation_mbarrier_region, 1)
        smem_workspace.register_tensor(
            self.tmem_holding_buffer_region, cutlass.Int32, (1,), byte_alignment=8
        )

    def _select_ab_stage_plan(self) -> None:
        stage_one_regions = make_smem_layouts(
            self.mma_instruction,
            self.mma_tiler_mnk,
            1,
            (self.a_major_mode, self.b_major_mode),
            self.architecture,
        )
        independent_sfb_region = make_smem_layouts(
            self.sfb_mma_instruction,
            self.mma_tiler_sfb,
            1,
            (self.a_major_mode, self.b_major_mode),
            self.architecture,
        )[3]
        if (
            independent_sfb_region.shape != stage_one_regions[3].shape
            or independent_sfb_region.stride != stage_one_regions[3].stride
            or independent_sfb_region.nbytes != stage_one_regions[3].nbytes
        ):
            raise ValueError(
                "The Rubin independent SFB plan does not match the primary MMA SFB operand."
            )

        def packed_bytes(region, dtype) -> int:
            return (region.cosize * int(dtype.width) + 7) // 8

        # Native mixed FP4 halves its SMEM region, so the budget can select more AB stages.
        # TMA completion still counts packed source bytes, independent of the SMEM allocation type.
        self.ab_stage_payload_bytes = sum(region.nbytes for region in stage_one_regions)
        stage_operand_tma_bytes = (
            packed_bytes(stage_one_regions[0], self.a_dtype),
            packed_bytes(stage_one_regions[1], self.b_dtype),
            stage_one_regions[2].nbytes,
            stage_one_regions[3].nbytes,
        )
        self.ab_stage_tma_bytes = sum(stage_operand_tma_bytes)
        (
            self.a_stage_tma_bytes,
            self.b_stage_tma_bytes,
            self.sfa_stage_tma_bytes,
            self.sfb_stage_tma_bytes,
        ) = tuple(stage_bytes * self.mma_cta_count for stage_bytes in stage_operand_tma_bytes)
        mbarrier_bytes = int(cutlass.Int64.width) // 8
        ab_stage_cost_bytes = self.ab_stage_payload_bytes + 2 * mbarrier_bytes
        plan_tail_bytes = (2 * self.num_accumulator_pipeline_stages + 2) * mbarrier_bytes
        self.num_ab_pipeline_stages = (
            self.mainloop_smem_budget_bytes - plan_tail_bytes
        ) // ab_stage_cost_bytes
        if self.num_ab_pipeline_stages < 1:
            raise ValueError(
                f"One AB stage needs {ab_stage_cost_bytes + plan_tail_bytes} bytes, exceeding the "
                f"{self.mainloop_smem_budget_bytes}-byte mainloop budget."
            )
        (self.a_smem_region, self.b_smem_region, self.sfa_smem_region, self.sfb_smem_region) = (
            make_smem_layouts(
                self.mma_instruction,
                self.mma_tiler_mnk,
                self.num_ab_pipeline_stages,
                (self.a_major_mode, self.b_major_mode),
                self.architecture,
            )
        )
        self.selected_smem_bytes = sum(
            region.nbytes
            for region in (
                self.a_smem_region,
                self.b_smem_region,
                self.sfa_smem_region,
                self.sfb_smem_region,
            )
        )

    def make_tiled_mma(self) -> cute.TiledMma:
        """Create a context-local SM107 block-scaled MMA object."""
        cta_group = tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE
        return sm107_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            cta_group,
            self.mma_instruction_mnk,
        )

    def materialize_codegen_members(self) -> None:
        """Materialize IR-backed Rubin MMA and layout objects."""
        common_mma_arguments = (
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
        )
        self.tiled_mma = self.make_tiled_mma()
        self.tiled_mma_sfb = sm107_utils.make_blockscaled_trivial_tiled_mma(
            *common_mma_arguments, tcgen05.CtaGroup.ONE, self.sfb_instruction_shape_mnk
        )
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)), (self.tiled_mma.thr_id.shape,)
        )
        self.cluster_layout_sfb_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)), (self.tiled_mma_sfb.thr_id.shape,)
        )
        if self.cluster_layout_vmnk.shape != self.cluster_layout_shape_vmnk:
            raise ValueError(
                f"Main cluster layout mismatch: Python {self.cluster_layout_shape_vmnk}, "
                f"CuTe {self.cluster_layout_vmnk.shape}."
            )
        if self.cluster_layout_sfb_vmnk.shape != self.cluster_layout_sfb_shape_vmnk:
            raise ValueError(
                f"SFB cluster layout mismatch: Python {self.cluster_layout_sfb_shape_vmnk}, "
                f"CuTe {self.cluster_layout_sfb_vmnk.shape}."
            )

        self.a_smem_composed_layout_staged = sm100_utils.make_smem_layout_a(
            self.tiled_mma, self.mma_tiler_mnk, self.a_smem_alloc_dtype, self.num_ab_pipeline_stages
        )
        self.b_smem_composed_layout_staged = sm100_utils.make_smem_layout_b(
            self.tiled_mma, self.mma_tiler_mnk, self.b_smem_alloc_dtype, self.num_ab_pipeline_stages
        )
        self.a_smem_layout_staged = self.a_smem_composed_layout_staged.outer
        self.b_smem_layout_staged = self.b_smem_composed_layout_staged.outer
        self.sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
            self.tiled_mma, self.mma_tiler_mnk, self.sf_vec_size, self.num_ab_pipeline_stages
        )
        self.sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
            self.tiled_mma, self.mma_tiler_mnk, self.sf_vec_size, self.num_ab_pipeline_stages
        )
        for tensor_name, materialized_layout, planned_region in (
            ("A", self.a_smem_layout_staged, self.a_smem_region),
            ("B", self.b_smem_layout_staged, self.b_smem_region),
            ("SFA", self.sfa_smem_layout_staged, self.sfa_smem_region),
            ("SFB", self.sfb_smem_layout_staged, self.sfb_smem_region),
        ):
            if (
                materialized_layout.shape != planned_region.shape
                or not strides_equal_ignoring_singletons(
                    planned_region.shape, materialized_layout.stride, planned_region.stride
                )
            ):
                raise ValueError(
                    f"{tensor_name} SMEM plan mismatch: Python "
                    f"{planned_region.shape}:{planned_region.stride}, CuTe "
                    f"{materialized_layout.shape}:{materialized_layout.stride}."
                )

    def prepare_tma_load_params(
        self,
        *,
        fc1_a: cute.Tensor,
        fc1_b: cute.Tensor,
        fc1_sfa: cute.Tensor,
        fc1_sfb: cute.Tensor,
        fc2_a: cute.Tensor,
        fc2_b: cute.Tensor,
        fc2_sfa: cute.Tensor,
        fc2_sfb: cute.Tensor,
    ) -> Tuple:
        """Return TMA tensor/atom pairs ordered for run_tma_a then run_tma_b."""
        a_stage_layout = cute.slice_(self.a_smem_composed_layout_staged, (None, None, None, 0))
        b_stage_layout = cute.slice_(self.b_smem_composed_layout_staged, (None, None, None, 0))
        sfa_stage_layout = cute.slice_(self.sfa_smem_layout_staged, (None, None, None, 0))
        sfb_stage_layout = cute.slice_(self.sfb_smem_layout_staged, (None, None, None, 0))

        a_operation = sm100_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, self.tiled_mma.thr_id
        )
        b_operation = sm100_utils.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mn, self.tiled_mma.thr_id
        )
        sfa_operation = sm100_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, self.tiled_mma.thr_id
        )
        sfb_operation = sm100_utils.cluster_shape_to_tma_atom_SFB(
            self.cluster_shape_mn, self.tiled_mma.thr_id
        )
        a_internal_type = (
            self.a_smem_alloc_dtype if self.a_smem_alloc_dtype is not self.a_dtype else None
        )
        b_internal_type = (
            self.b_smem_alloc_dtype if self.b_smem_alloc_dtype is not self.b_dtype else None
        )

        fc1_a_atom, fc1_a_tensor = cute.nvgpu.make_tiled_tma_atom_A(
            a_operation,
            fc1_a,
            a_stage_layout,
            self.mma_tiler_mnk,
            self.tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=a_internal_type,
        )
        # FC1 B is read from the original token-major activation.  A one-row
        # descriptor lets four producer warps provide arbitrary source row IDs
        # through TMA TILE_GATHER4 while landing in the native K_SW128 B layout.
        fc1_b_2d = fc1_b[(None, None, 0)]
        gather_b_base = cute.make_layout((1, self.mma_tile_k), stride=(self.mma_tile_k, 1))
        gather_b_smem_layout = cute.make_composed_layout(
            cute.make_swizzle(3, 4, 3), 0, gather_b_base
        )
        fc1_b_atom, fc1_b_tensor = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            fc1_b_2d,
            gather_b_smem_layout,
            (1, self.mma_tile_k),
        )
        fc1_sfa_atom, fc1_sfa_tensor = cute.nvgpu.make_tiled_tma_atom_A(
            sfa_operation,
            fc1_sfa,
            sfa_stage_layout,
            self.mma_tiler_mnk,
            self.tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=cutlass.Uint64,
        )
        fc1_sfb_atom, fc1_sfb_tensor = cute.nvgpu.make_tiled_tma_atom_B(
            sfb_operation,
            fc1_sfb,
            sfb_stage_layout,
            self.mma_tiler_sfb,
            self.tiled_mma_sfb,
            self.cluster_layout_sfb_vmnk.shape,
            internal_type=cutlass.Uint64,
        )
        fc2_a_atom, fc2_a_tensor = cute.nvgpu.make_tiled_tma_atom_A(
            a_operation,
            fc2_a,
            a_stage_layout,
            self.mma_tiler_mnk,
            self.tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=a_internal_type,
        )
        fc2_b_atom, fc2_b_tensor = cute.nvgpu.make_tiled_tma_atom_B(
            b_operation,
            fc2_b,
            b_stage_layout,
            self.mma_tiler_mnk,
            self.tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=b_internal_type,
        )
        fc2_sfa_atom, fc2_sfa_tensor = cute.nvgpu.make_tiled_tma_atom_A(
            sfa_operation,
            fc2_sfa,
            sfa_stage_layout,
            self.mma_tiler_mnk,
            self.tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=cutlass.Uint64,
        )
        fc2_sfb_atom, fc2_sfb_tensor = cute.nvgpu.make_tiled_tma_atom_B(
            sfb_operation,
            fc2_sfb,
            sfb_stage_layout,
            self.mma_tiler_sfb,
            self.tiled_mma_sfb,
            self.cluster_layout_sfb_vmnk.shape,
            internal_type=cutlass.Uint64,
        )
        return (
            fc1_a_tensor,
            fc1_a_atom,
            fc1_sfa_tensor,
            fc1_sfa_atom,
            fc2_a_tensor,
            fc2_a_atom,
            fc2_sfa_tensor,
            fc2_sfa_atom,
            fc1_b_tensor,
            fc1_b_atom,
            fc1_sfb_tensor,
            fc1_sfb_atom,
            fc2_b_tensor,
            fc2_b_atom,
            fc2_sfb_tensor,
            fc2_sfb_atom,
        )

    @cute.jit
    def assign_device_members(
        self,
        smem_workspace: SmemWorkspace,
        smem_base: cute.Pointer,
        cta_coord_in_cluster: cute.Coord,
        hidden_size,
        intermediate_gateup_size,
    ) -> None:
        """Root device-side Rubin SMEM tensors and CTA coordinates."""
        self.a_smem_tensor = smem_workspace.tensor(self.a_smem_tensor_region, smem_base)
        self.b_smem_tensor = smem_workspace.tensor(self.b_smem_tensor_region, smem_base)
        self.sfa_smem_tensor = smem_workspace.tensor(self.sfa_smem_tensor_region, smem_base)
        self.sfb_smem_tensor = smem_workspace.tensor(self.sfb_smem_tensor_region, smem_base)
        self.cta_coord_in_cluster = cta_coord_in_cluster
        self.mma_cta_index = cta_coord_in_cluster[0] % self.mma_cta_count
        self.is_leader_cta = self.mma_cta_index == 0
        self.main_vmnk_coord = (
            self.mma_cta_index,
            cta_coord_in_cluster[0] // self.mma_cta_count,
            cta_coord_in_cluster[1],
            cta_coord_in_cluster[2],
        )
        bound_hidden_size = self.hidden_size if isinstance(self.hidden_size, int) else hidden_size
        bound_intermediate_gateup_size = (
            self.intermediate_gateup_size
            if isinstance(self.intermediate_gateup_size, int)
            else intermediate_gateup_size
        )
        self.fc1_k_tile_count = ceil_div(bound_hidden_size, self.mma_tile_k)
        self.fc2_k_tile_count = ceil_div(bound_intermediate_gateup_size // 2, self.mma_tile_k)
        self.sfb_vmnk_coord = (
            0,
            cta_coord_in_cluster[0],
            cta_coord_in_cluster[1],
            cta_coord_in_cluster[2],
        )

    @cute.jit
    def create_ab_pipeline(
        self, smem_workspace: SmemWorkspace, smem_base: cute.Pointer
    ) -> pipeline.PipelineAsync:
        """Create the shared two-producer TMA-to-UMMA pipeline.

        Each producer arrival contributes the same accounting share.  The two
        arrivals together register the combined A, B, SFA, and SFB bytes; an
        individual share does not need to match the bytes issued by that warp.
        """
        producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 2)
        num_tma_consumers = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, num_tma_consumers)
        return pipeline.PipelineTmaUmma.create(
            barrier_storage=smem_workspace.ptr(self.ab_pipeline_mbarriers_region, smem_base),
            num_stages=self.num_ab_pipeline_stages,
            producer_group=producer_group,
            consumer_group=consumer_group,
            tx_count=self.ab_pipeline_tx_count_per_producer,
            cta_layout_vmnk=self.cluster_layout_vmnk,
            defer_sync=True,
        )

    @cute.jit
    def advance_pipeline_state(
        self,
        state: pipeline.PipelineState,
        iterations: Int32,
    ) -> None:
        """Advance software pipeline state without touching its mbarriers."""
        advanced_index = state.index + iterations
        wrap_count = advanced_index // Int32(state.stages)
        state._index = advanced_index % Int32(state.stages)
        state._phase = state.phase ^ (wrap_count & Int32(1))
        state._count += iterations

    @cute.jit
    def create_acc_pipeline(self, smem_workspace: SmemWorkspace, smem_base: cute.Pointer):
        """Create the UMMA-to-epilogue accumulator pipeline."""
        producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        consumer_thread_count = (
            self.num_accumulator_consumer_warps_per_cta * 32 * self.mma_cta_count
        )
        consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, consumer_thread_count)
        return pipeline.PipelineUmmaAsync.create(
            barrier_storage=smem_workspace.ptr(self.acc_pipeline_mbarriers_region, smem_base),
            num_stages=self.num_accumulator_pipeline_stages,
            producer_group=producer_group,
            consumer_group=consumer_group,
            cta_layout_vmnk=self.cluster_layout_vmnk,
            defer_sync=True,
        )

    @cute.jit
    def create_tmem_allocator(
        self, smem_workspace: SmemWorkspace, smem_base: cute.Pointer, *, allocator_warp_id: int
    ):
        """Bind the SM107 TMEM allocator to finalized SMEM regions."""
        allocation_barrier = pipeline.NamedBarrier(
            barrier_id=self.tmem_allocation_barrier_id,
            num_threads=32 * (1 + self.num_accumulator_consumer_warps_per_cta),
        )
        return utils.TmemAllocator(
            smem_workspace.ptr(self.tmem_holding_buffer_region, smem_base),
            barrier_for_retrieve=allocation_barrier,
            allocator_warp_id=allocator_warp_id,
            is_two_cta=self.use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=smem_workspace.ptr(
                self.tmem_deallocation_mbarrier_region, smem_base
            ),
            arch=self.architecture,
        )

    def _s2t_copy_and_partition(self, smem_tensor: cute.Tensor, tmem_tensor: cute.Tensor):
        compact_smem_tensor = cute.filter_zeros(smem_tensor)
        compact_tmem_tensor = cute.filter_zeros(tmem_tensor)
        cta_group = tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE
        copy_atom = cute.make_copy_atom(tcgen05.Cp4x32x128bOp(cta_group), self.sf_dtype)
        tiled_copy = tcgen05.make_s2t_copy(copy_atom, compact_tmem_tensor)
        thread_copy = tiled_copy.get_slice(0)

        mn_mode = cute.get(compact_smem_tensor.layout, mode=[0, 0])
        mn_mode = cute.append(mn_mode, cute.make_layout((4,), stride=(0,)))
        broadcast_layout = cute.append(
            cute.group_modes(mn_mode, 0), cute.get(compact_smem_tensor.layout, mode=[0, 1])
        )
        broadcast_layout = cute.append(
            cute.group_modes(broadcast_layout, 0), cute.get(compact_smem_tensor.layout, mode=[1])
        )
        broadcast_layout = cute.append(
            broadcast_layout, cute.get(compact_smem_tensor.layout, mode=[2])
        )
        broadcast_layout = cute.append(
            broadcast_layout, cute.get(compact_smem_tensor.layout, mode=[3])
        )
        broadcast_smem_tensor = cute.make_tensor(compact_smem_tensor.iterator, broadcast_layout)

        partitioned_smem = thread_copy.partition_S(broadcast_smem_tensor)
        partitioned_smem = tcgen05.get_s2t_smem_desc_tensor(tiled_copy, partitioned_smem)
        partitioned_tmem = thread_copy.partition_D(compact_tmem_tensor)
        return tiled_copy, partitioned_smem, partitioned_tmem

    @cute.jit
    def run_tma_a(
        self,
        *,
        fc1_tma_a_tensor: cute.Tensor,
        fc1_tma_a_atom: cute.CopyAtom,
        fc1_tma_sfa_tensor: cute.Tensor,
        fc1_tma_sfa_atom: cute.CopyAtom,
        fc2_tma_a_tensor: cute.Tensor,
        fc2_tma_a_atom: cute.CopyAtom,
        fc2_tma_sfa_tensor: cute.Tensor,
        fc2_tma_sfa_atom: cute.CopyAtom,
        ab_pipeline: pipeline.PipelineAsync,
        ab_pipeline_state: pipeline.PipelineState,
        sched_consumer: SchedulerConsumer,
        kernel_extension: BlockScaledSwapAbFc12Extension,
    ) -> None:
        """Run the TMA-A warp that loads weights and their scale factors."""
        multicast_mask = None
        if cutlass.const_expr(self.is_a_mcast or self.use_2cta_instrs):
            multicast_mask = tma_multicast_mask(
                self.cluster_shape_mn,
                None,
                self.cta_coord_in_cluster,
                None,
                self.use_2cta_instrs,
                "a",
            )

        a_cta_layout = cute.make_layout(
            cute.slice_(self.cluster_layout_vmnk, (0, 0, None, 0)).shape
        )
        sfa_cta_layout = a_cta_layout
        thread_mma = self.tiled_mma.get_slice(self.mma_cta_index)

        work_tile = sched_consumer.consume_work()
        while work_tile.is_valid_tile:
            is_fc1 = work_tile.phase == Int32(BlockPhase.Linear1)
            if is_fc1:
                iket.range_push("tma_weight_fc1")
                k_tile_count = self.fc1_k_tile_count
                real_a, a_descriptor_ptr = kernel_extension.get_gmem_tensor(
                    "a", fc1_tma_a_tensor, work_tile
                )
                real_sfa, sfa_descriptor_ptr = kernel_extension.get_gmem_tensor(
                    "sfa", fc1_tma_sfa_tensor, work_tile
                )

                global_a = cute.local_tile(
                    real_a, cute.slice_(self.mma_tiler_mnk, (None, 0, None)), (None, None, None)
                )
                global_sfa = cute.local_tile(
                    real_sfa, cute.slice_(self.mma_tiler_mnk, (None, 0, None)), (None, None, None)
                )
                partitioned_global_a = thread_mma.partition_A(global_a)
                partitioned_global_sfa = thread_mma.partition_A(global_sfa)

                partitioned_smem_a, partitioned_global_a = cpasync.tma_partition(
                    fc1_tma_a_atom,
                    self.main_vmnk_coord[2],
                    a_cta_layout,
                    cute.group_modes(self.a_smem_tensor, 0, 3),
                    cute.group_modes(partitioned_global_a, 0, 3),
                )
                partitioned_smem_sfa, partitioned_global_sfa = cpasync.tma_partition(
                    fc1_tma_sfa_atom,
                    self.main_vmnk_coord[2],
                    sfa_cta_layout,
                    cute.group_modes(self.sfa_smem_tensor, 0, 3),
                    cute.group_modes(partitioned_global_sfa, 0, 3),
                )
                partitioned_smem_sfa = cute.filter_zeros(partitioned_smem_sfa)
                partitioned_global_sfa = cute.filter_zeros(partitioned_global_sfa)

                mma_tile_m = work_tile.tile_m_idx // self.mma_cta_count
                global_a_slice = partitioned_global_a[(None, mma_tile_m, None, 0)]
                global_sfa_slice = partitioned_global_sfa[(None, mma_tile_m, None, 0)]

                empty_status = ab_pipeline.producer_try_acquire(ab_pipeline_state)
                for k_tile_idx in cutlass.range(k_tile_count, unroll=1):
                    ab_pipeline.producer_acquire(ab_pipeline_state, empty_status)
                    tma_barrier = ab_pipeline.producer_get_barrier(ab_pipeline_state)
                    stage_index = ab_pipeline_state.index
                    ab_pipeline_state.advance()
                    if k_tile_idx + 1 < k_tile_count:
                        empty_status = ab_pipeline.producer_try_acquire(ab_pipeline_state)
                    cute.copy(
                        fc1_tma_a_atom,
                        global_a_slice[(None, k_tile_idx)],
                        partitioned_smem_a[(None, stage_index)],
                        tma_bar_ptr=tma_barrier,
                        tma_desc_ptr=a_descriptor_ptr,
                        mcast_mask=multicast_mask,
                    )
                    cute.copy(
                        fc1_tma_sfa_atom,
                        global_sfa_slice[(None, k_tile_idx)],
                        partitioned_smem_sfa[(None, stage_index)],
                        tma_bar_ptr=tma_barrier,
                        tma_desc_ptr=sfa_descriptor_ptr,
                        mcast_mask=multicast_mask,
                    )
            else:
                iket.range_push("tma_weight_fc2")
                k_tile_count = self.fc2_k_tile_count
                real_a, a_descriptor_ptr = kernel_extension.get_gmem_tensor(
                    "a", fc2_tma_a_tensor, work_tile
                )
                real_sfa, sfa_descriptor_ptr = kernel_extension.get_gmem_tensor(
                    "sfa", fc2_tma_sfa_tensor, work_tile
                )

                global_a = cute.local_tile(
                    real_a, cute.slice_(self.mma_tiler_mnk, (None, 0, None)), (None, None, None)
                )
                global_sfa = cute.local_tile(
                    real_sfa, cute.slice_(self.mma_tiler_mnk, (None, 0, None)), (None, None, None)
                )
                partitioned_global_a = thread_mma.partition_A(global_a)
                partitioned_global_sfa = thread_mma.partition_A(global_sfa)

                partitioned_smem_a, partitioned_global_a = cpasync.tma_partition(
                    fc2_tma_a_atom,
                    self.main_vmnk_coord[2],
                    a_cta_layout,
                    cute.group_modes(self.a_smem_tensor, 0, 3),
                    cute.group_modes(partitioned_global_a, 0, 3),
                )
                partitioned_smem_sfa, partitioned_global_sfa = cpasync.tma_partition(
                    fc2_tma_sfa_atom,
                    self.main_vmnk_coord[2],
                    sfa_cta_layout,
                    cute.group_modes(self.sfa_smem_tensor, 0, 3),
                    cute.group_modes(partitioned_global_sfa, 0, 3),
                )
                partitioned_smem_sfa = cute.filter_zeros(partitioned_smem_sfa)
                partitioned_global_sfa = cute.filter_zeros(partitioned_global_sfa)

                mma_tile_m = work_tile.tile_m_idx // self.mma_cta_count
                global_a_slice = partitioned_global_a[(None, mma_tile_m, None, 0)]
                global_sfa_slice = partitioned_global_sfa[(None, mma_tile_m, None, 0)]

                empty_status = ab_pipeline.producer_try_acquire(ab_pipeline_state)
                for k_tile_idx in cutlass.range(k_tile_count, unroll=1):
                    ab_pipeline.producer_acquire(ab_pipeline_state, empty_status)
                    tma_barrier = ab_pipeline.producer_get_barrier(ab_pipeline_state)
                    stage_index = ab_pipeline_state.index
                    ab_pipeline_state.advance()
                    if k_tile_idx + 1 < k_tile_count:
                        empty_status = ab_pipeline.producer_try_acquire(ab_pipeline_state)
                    cute.copy(
                        fc2_tma_a_atom,
                        global_a_slice[(None, k_tile_idx)],
                        partitioned_smem_a[(None, stage_index)],
                        tma_bar_ptr=tma_barrier,
                        tma_desc_ptr=a_descriptor_ptr,
                        mcast_mask=multicast_mask,
                    )
                    cute.copy(
                        fc2_tma_sfa_atom,
                        global_sfa_slice[(None, k_tile_idx)],
                        partitioned_smem_sfa[(None, stage_index)],
                        tma_bar_ptr=tma_barrier,
                        tma_desc_ptr=sfa_descriptor_ptr,
                        mcast_mask=multicast_mask,
                    )

            iket.range_pop()
            work_tile = sched_consumer.consume_work()

        ab_pipeline.producer_tail(ab_pipeline_state)

    @cute.jit
    def run_tma_b(
        self,
        *,
        fc1_tma_b_tensor: cute.Tensor,
        fc1_tma_b_atom: cute.CopyAtom,
        fc1_tma_sfb_tensor: cute.Tensor,
        fc1_tma_sfb_atom: cute.CopyAtom,
        fc2_tma_b_tensor: cute.Tensor,
        fc2_tma_b_atom: cute.CopyAtom,
        fc2_tma_sfb_tensor: cute.Tensor,
        fc2_tma_sfb_atom: cute.CopyAtom,
        ab_pipeline: pipeline.PipelineAsync,
        ab_pipeline_state: pipeline.PipelineState,
        sched_consumer: SchedulerConsumer,
        kernel_extension: BlockScaledSwapAbFc12Extension,
        fc1_done_counter_pointer: cute.Pointer,
        fc2_spin_threshold: Int32,
    ) -> None:
        """Run the TMA-B warp that loads token data and scale factors."""
        b_multicast_mask = None
        sfb_multicast_mask = None
        if cutlass.const_expr(self.is_b_mcast or self.use_2cta_instrs):
            b_multicast_mask = tma_multicast_mask(
                self.cluster_shape_mn,
                None,
                self.cta_coord_in_cluster,
                None,
                self.use_2cta_instrs,
                "b",
            )
            sfb_multicast_mask = tma_multicast_mask(
                self.cluster_shape_mn,
                None,
                self.cta_coord_in_cluster,
                None,
                self.use_2cta_instrs,
                "sfb",
            )

        b_cta_layout = cute.make_layout(
            cute.slice_(self.cluster_layout_vmnk, (0, None, 0, 0)).shape
        )
        sfb_cta_layout = cute.make_layout(
            cute.slice_(self.cluster_layout_sfb_vmnk, (0, None, 0, 0)).shape
        )
        thread_mma = self.tiled_mma.get_slice(self.mma_cta_index)
        thread_mma_sfb = self.tiled_mma_sfb.get_slice(self.mma_cta_index)

        work_tile = sched_consumer.consume_work()
        while work_tile.is_valid_tile:
            is_fc1 = work_tile.phase == Int32(BlockPhase.Linear1)
            if is_fc1:
                iket.range_push("tma_token_fc1")
                iket.range_push("tma_token_fc1_wait")
                kernel_extension.wait_for_input(work_tile)
                iket.range_pop()
                k_tile_count = self.fc1_k_tile_count
                real_sfb, sfb_descriptor_ptr = kernel_extension.get_gmem_tensor(
                    "sfb", fc1_tma_sfb_tensor, work_tile
                )

                global_sfb = cute.local_tile(
                    real_sfb, cute.slice_(self.mma_tiler_sfb, (0, None, None)), (None, None, None)
                )
                partitioned_global_sfb = thread_mma_sfb.partition_B(global_sfb)

                partitioned_smem_sfb, partitioned_global_sfb = cpasync.tma_partition(
                    fc1_tma_sfb_atom,
                    self.sfb_vmnk_coord[1],
                    sfb_cta_layout,
                    cute.group_modes(self.sfb_smem_tensor, 0, 3),
                    cute.group_modes(partitioned_global_sfb, 0, 3),
                )
                partitioned_smem_sfb = cute.filter_zeros(partitioned_smem_sfb)
                partitioned_global_sfb = cute.filter_zeros(partitioned_global_sfb)

                sfb_tile_n_index = work_tile.tile_n_idx
                if cutlass.const_expr(self.cta_tile_n == 64):
                    sfb_tile_n_index = work_tile.tile_n_idx // Int32(2)
                global_sfb_slice = partitioned_global_sfb[(None, sfb_tile_n_index, None, 0)]

                empty_status = ab_pipeline.producer_try_acquire(ab_pipeline_state)
                for k_tile_idx in cutlass.range(k_tile_count, unroll=1):
                    ab_pipeline.producer_acquire(ab_pipeline_state, empty_status)
                    tma_barrier = ab_pipeline.producer_get_barrier(ab_pipeline_state)
                    stage_index = ab_pipeline_state.index
                    # TMA-B has waited for this AB stage to become empty and
                    # contributed one of the two static producer transaction
                    # expectations. Rendezvous with the four Gather-B warps
                    # before any of them issues Gather4 against this barrier.
                    self.fc1_gather_sync_barrier.arrive_and_wait()
                    ab_pipeline_state.advance()
                    if k_tile_idx + 1 < k_tile_count:
                        empty_status = ab_pipeline.producer_try_acquire(ab_pipeline_state)
                    cute.copy(
                        fc1_tma_sfb_atom,
                        global_sfb_slice[(None, k_tile_idx)],
                        partitioned_smem_sfb[(None, stage_index)],
                        tma_bar_ptr=tma_barrier,
                        tma_desc_ptr=sfb_descriptor_ptr,
                        mcast_mask=sfb_multicast_mask,
                    )
            else:
                iket.range_push("tma_token_fc2")
                counter_slot = work_tile.cumulative_token_block_count + work_tile.tile_n_idx
                counter_pointer = fc1_done_counter_pointer + counter_slot
                iket.range_push("tma_token_fc2_wait")
                spin_wait(
                    counter_pointer,
                    lambda value: value >= fc2_spin_threshold,
                    sleep_cycles=500,
                    peek_status=work_tile.peek_ready,
                )
                iket.range_pop()

                k_tile_count = self.fc2_k_tile_count
                real_b, b_descriptor_ptr = kernel_extension.get_gmem_tensor(
                    "b", fc2_tma_b_tensor, work_tile
                )
                real_sfb, sfb_descriptor_ptr = kernel_extension.get_gmem_tensor(
                    "sfb", fc2_tma_sfb_tensor, work_tile
                )

                global_b = cute.local_tile(
                    real_b, cute.slice_(self.mma_tiler_mnk, (0, None, None)), (None, None, None)
                )
                global_sfb = cute.local_tile(
                    real_sfb, cute.slice_(self.mma_tiler_sfb, (0, None, None)), (None, None, None)
                )
                partitioned_global_b = thread_mma.partition_B(global_b)
                partitioned_global_sfb = thread_mma_sfb.partition_B(global_sfb)

                partitioned_smem_b, partitioned_global_b = cpasync.tma_partition(
                    fc2_tma_b_atom,
                    self.main_vmnk_coord[1],
                    b_cta_layout,
                    cute.group_modes(self.b_smem_tensor, 0, 3),
                    cute.group_modes(partitioned_global_b, 0, 3),
                )
                partitioned_smem_sfb, partitioned_global_sfb = cpasync.tma_partition(
                    fc2_tma_sfb_atom,
                    self.sfb_vmnk_coord[1],
                    sfb_cta_layout,
                    cute.group_modes(self.sfb_smem_tensor, 0, 3),
                    cute.group_modes(partitioned_global_sfb, 0, 3),
                )
                partitioned_smem_sfb = cute.filter_zeros(partitioned_smem_sfb)
                partitioned_global_sfb = cute.filter_zeros(partitioned_global_sfb)

                global_b_slice = partitioned_global_b[(None, work_tile.tile_n_idx, None, 0)]
                sfb_tile_n_index = work_tile.tile_n_idx
                if cutlass.const_expr(self.cta_tile_n == 64):
                    sfb_tile_n_index = work_tile.tile_n_idx // Int32(2)
                global_sfb_slice = partitioned_global_sfb[(None, sfb_tile_n_index, None, 0)]

                empty_status = ab_pipeline.producer_try_acquire(ab_pipeline_state)
                for k_tile_idx in cutlass.range(k_tile_count, unroll=1):
                    ab_pipeline.producer_acquire(ab_pipeline_state, empty_status)
                    tma_barrier = ab_pipeline.producer_get_barrier(ab_pipeline_state)
                    stage_index = ab_pipeline_state.index
                    ab_pipeline_state.advance()
                    if k_tile_idx + 1 < k_tile_count:
                        empty_status = ab_pipeline.producer_try_acquire(ab_pipeline_state)
                    cute.copy(
                        fc2_tma_b_atom,
                        global_b_slice[(None, k_tile_idx)],
                        partitioned_smem_b[(None, stage_index)],
                        tma_bar_ptr=tma_barrier,
                        tma_desc_ptr=b_descriptor_ptr,
                        mcast_mask=b_multicast_mask,
                    )
                    cute.copy(
                        fc2_tma_sfb_atom,
                        global_sfb_slice[(None, k_tile_idx)],
                        partitioned_smem_sfb[(None, stage_index)],
                        tma_bar_ptr=tma_barrier,
                        tma_desc_ptr=sfb_descriptor_ptr,
                        mcast_mask=sfb_multicast_mask,
                    )
            iket.range_pop()
            work_tile = sched_consumer.consume_work()

        ab_pipeline.producer_tail(ab_pipeline_state)

    @cute.jit
    def run_tma_gather_b1(
        self,
        *,
        fc1_tma_b_atom: cute.CopyAtom,
        token_src_metadata: cute.Tensor,
        ab_pipeline: pipeline.PipelineAsync,
        ab_pipeline_state: pipeline.PipelineState,
        sched_consumer: SchedulerConsumer,
        kernel_extension: BlockScaledSwapAbFc12Extension,
        gather_warp_idx: Int32,
    ) -> None:
        """Gather original-order FC1 activations directly into B SMEM.

        Four logical producer warps split the tile N dimension.  In 2CTA mode
        each physical CTA owns one contiguous N half, so matching warp IDs
        across the CTA pair jointly cover the user-visible N/4 rows.

        This implementation assumes every scheduled FC1 tile is full. TMA-B
        contributes its static producer transaction expectation and uses
        a CTA-local named barrier to release these Gather-B warps. Gather-B
        contributes transaction completions only; it is not an AB producer.
        FC2 consumes the identical scheduler stream and advances only the
        register-held AB state.
        """
        rows_per_warp = self.gather_b_rows_per_warp
        row_ids = cute.make_rmem_tensor(cute.make_layout((rows_per_warp,)), cutlass.Int32)
        gather_b_multicast_mask = None
        if cutlass.const_expr(self.is_b_mcast or self.use_2cta_instrs):
            gather_b_multicast_mask = tma_multicast_mask(
                self.cluster_shape_mn,
                None,
                self.cta_coord_in_cluster,
                None,
                self.use_2cta_instrs,
                "b",
            )
        is_gather_b_mcast_leader = (not self.is_b_mcast) or (self.main_vmnk_coord[1] == Int32(0))

        work_tile = sched_consumer.consume_work()
        while work_tile.is_valid_tile:
            is_fc1 = work_tile.phase == Int32(BlockPhase.Linear1)
            if is_fc1:
                iket.range_push("tma_gather_b1_fc1")
                kernel_extension.wait_for_input(work_tile)
                tile_pool_base = (
                    work_tile.cumulative_data_physical_row
                    + work_tile.tile_n_idx * Int32(self.cta_tile_n)
                )
                cta_row_base = self.mma_cta_index * Int32(self.gather_b_rows_per_cta)
                warp_row_base = gather_warp_idx * Int32(rows_per_warp)
                for row in cutlass.range_constexpr(rows_per_warp):
                    pool_row = tile_pool_base + cta_row_base + warp_row_base + Int32(row)
                    metadata = RoutingMetadata.load(token_src_metadata.iterator + pool_row)
                    row_ids[row] = metadata.token

                for k_tile_idx in cutlass.range(self.fc1_k_tile_count, unroll=1):
                    # TMA-B has already acquired this stage and contributed
                    # its static transaction expectation. The named barrier
                    # orders that arrive-and-expect-tx before Gather4.
                    self.fc1_gather_sync_barrier.arrive_and_wait()
                    tma_barrier = ab_pipeline.producer_get_barrier(ab_pipeline_state)
                    stage_index = ab_pipeline_state.index

                    if is_gather_b_mcast_leader:
                        with cute.arch.elect_one():
                            col_k = k_tile_idx * Int32(self.mma_tile_k)
                            stage_base_elements = (
                                stage_index * self.gather_b_rows_per_cta * self.mma_tile_k
                            )
                            warp_base_elements = gather_warp_idx * Int32(
                                rows_per_warp * self.mma_tile_k
                            )
                            for gather_idx in cutlass.range_constexpr(self.gather_b_calls_per_warp):
                                row_start = gather_idx * 4
                                destination_offset = (
                                    stage_base_elements
                                    + warp_base_elements
                                    + row_start * self.mma_tile_k
                                )
                                sm107_tma_gather4_load(
                                    fc1_tma_b_atom,
                                    self.b_smem_tensor.iterator + destination_offset,
                                    tma_barrier,
                                    col_k,
                                    row_ids[row_start],
                                    row_ids[row_start + 1],
                                    row_ids[row_start + 2],
                                    row_ids[row_start + 3],
                                    use_cta_group_2=self.use_2cta_instrs,
                                    mcast_mask=gather_b_multicast_mask,
                                )
                    ab_pipeline_state.advance()
                iket.range_pop()
            else:
                iket.range_push("tma_gather_b1_fc2_state_advance")
                # FC2 is fully produced by TMA-A and TMA-B.  Gather-B only
                # keeps its register-held state aligned for the next FC1.
                self.advance_pipeline_state(
                    ab_pipeline_state,
                    Int32(self.fc2_k_tile_count),
                )
                iket.range_pop()

            work_tile = sched_consumer.consume_work()

    @cute.jit
    def run_mma(
        self,
        *,
        tmem_allocator,
        ab_pipeline: pipeline.PipelineAsync,
        ab_pipeline_state: pipeline.PipelineState,
        acc_pipeline: pipeline.PipelineAsync,
        sched_consumer: SchedulerConsumer,
    ) -> None:
        """Execute full-tile SM107 MMA across K tiles."""
        tiled_mma = self.make_tiled_mma()
        fragment_a = tiled_mma.make_fragment_A(self.a_smem_tensor)
        fragment_b = tiled_mma.make_fragment_B(self.b_smem_tensor)

        tmem_allocator.wait_for_alloc()
        accumulator_pointer = tmem_allocator.retrieve_ptr(self.acc_dtype)
        accumulator_layout = cute.make_layout(
            self.accumulator_shape, stride=self.accumulator_stride
        )
        accumulator_base = cute.make_tensor(accumulator_pointer, accumulator_layout)

        sfa_stage_smem_layout = cute.slice_(
            blockscaled_utils.make_smem_layout_sfa(
                tiled_mma, self.mma_tiler_mnk, self.sf_vec_size, 1
            ),
            (None, None, None, 0),
        )
        sfb_stage_smem_layout = cute.slice_(
            blockscaled_utils.make_smem_layout_sfb(
                tiled_mma, self.mma_tiler_mnk, self.sf_vec_size, 1
            ),
            (None, None, None, 0),
        )
        sfa_pointer = cute.recast_ptr(
            accumulator_pointer + self.num_accumulator_tmem_cols, dtype=self.sf_dtype
        )
        sfa_layout = blockscaled_utils.make_tmem_layout_sfa(
            tiled_mma, self.mma_tiler_mnk, self.sf_vec_size, sfa_stage_smem_layout
        )
        tmem_sfa = cute.make_tensor(sfa_pointer, sfa_layout)
        sfb_pointer = cute.recast_ptr(
            accumulator_pointer + self.num_accumulator_tmem_cols + self.num_sfa_tmem_cols,
            dtype=self.sf_dtype,
        )
        sfb_layout = blockscaled_utils.make_tmem_layout_sfb(
            tiled_mma, self.mma_tiler_mnk, self.sf_vec_size, sfb_stage_smem_layout
        )
        tmem_sfb = cute.make_tensor(sfb_pointer, sfb_layout)
        tiled_copy_sfa, partitioned_smem_sfa, partitioned_tmem_sfa = self._s2t_copy_and_partition(
            self.sfa_smem_tensor, tmem_sfa
        )
        tiled_copy_sfb, partitioned_smem_sfb, partitioned_tmem_sfb = self._s2t_copy_and_partition(
            self.sfb_smem_tensor, tmem_sfb
        )

        acc_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.num_accumulator_pipeline_stages
        )
        work_tile = sched_consumer.consume_work()
        while work_tile.is_valid_tile:
            is_fc1 = work_tile.phase == Int32(BlockPhase.Linear1)
            k_tile_count = Int32(0)
            if is_fc1:
                k_tile_count = self.fc1_k_tile_count
            else:
                k_tile_count = self.fc2_k_tile_count
            accumulator_stage_index = acc_producer_state.index

            if self.is_leader_cta:
                accumulator = accumulator_base[(None, None, accumulator_stage_index)]
                accumulator = cute.tiled_divide(accumulator, accumulator.shape)
                mma_tmem_sfb = tmem_sfb
                if cutlass.const_expr(self.cta_tile_n == 64):
                    sfb_shift = (work_tile.tile_n_idx % Int32(2)) * Int32(2)
                    shifted_sfb_pointer = cute.recast_ptr(
                        accumulator_pointer
                        + self.num_accumulator_tmem_cols
                        + self.num_sfa_tmem_cols
                        + sfb_shift,
                        dtype=self.sf_dtype,
                    )
                    mma_tmem_sfb = cute.make_tensor(shifted_sfb_pointer, sfb_layout)
                next_ab_pipeline_state = ab_pipeline_state.clone()
                next_ab_pipeline_state.advance()
                full_status = ab_pipeline.consumer_try_wait(ab_pipeline_state)
                acc_pipeline.producer_acquire(acc_producer_state)
                for k_tile_idx in cutlass.range(k_tile_count, unroll=1):
                    ab_pipeline.consumer_wait(ab_pipeline_state, full_status)
                    next_full_status = cutlass.Boolean(1)
                    if k_tile_idx + 1 < k_tile_count:
                        next_full_status = ab_pipeline.consumer_try_wait(next_ab_pipeline_state)

                    s2t_stage_coord = (None, None, None, None, ab_pipeline_state.index)
                    cute.copy(
                        tiled_copy_sfa, partitioned_smem_sfa[s2t_stage_coord], partitioned_tmem_sfa
                    )
                    cute.copy(
                        tiled_copy_sfb, partitioned_smem_sfb[s2t_stage_coord], partitioned_tmem_sfb
                    )

                    a_stage_frag = fragment_a[(None, None, None, ab_pipeline_state.index)]
                    b_stage_frag = fragment_b[(None, None, None, ab_pipeline_state.index)]
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, k_tile_idx != 0)
                    cute.gemm(
                        tiled_mma,
                        accumulator,
                        [a_stage_frag, tmem_sfa],
                        [b_stage_frag, mma_tmem_sfb],
                        accumulator,
                    )

                    ab_pipeline.consumer_release(ab_pipeline_state)
                    ab_pipeline_state.advance()
                    next_ab_pipeline_state.advance()
                    full_status = next_full_status
                acc_pipeline.producer_commit(acc_producer_state)

            acc_producer_state.advance()
            work_tile = sched_consumer.consume_work()
        acc_pipeline.producer_tail(acc_producer_state)


__all__ = ["BlockScaledSwapAbFc12Mainloop"]
