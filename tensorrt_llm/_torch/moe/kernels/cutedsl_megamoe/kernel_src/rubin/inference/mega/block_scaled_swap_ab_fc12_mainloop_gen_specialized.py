# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Mainloop component for the Rubin generation-phase block-scaled swap-AB FC12 kernel.

Three properties separate this from the two mainloops it descends from.

FC1's token operand arrives by ``TILE_GATHER4`` straight into B SMEM, as in
``local_mega``, but the row identifiers come from the communication component's
``gather_index`` (a flat ``src_rank * max_tokens_per_rank + src_token`` per pool
row) rather than from a packed routing record, and the gather work is split
across every CTA of the B multicast group instead of being issued by one leader.
Each CTA therefore issues ``cta_tile_n / cluster_m`` rows -- the MMA CTA count
cancels out of that expression, so the split is the same under one-CTA and
two-CTA instructions.

The MMA issues a dynamic instruction N, as in ``mega``. A generation-phase
expert commonly holds a few tens of tokens while the token tile is 128 or 256
wide, so a static extent would spend most of the tensor-core time on padding.
The consequence for the gather is that under two-CTA instructions the row window
a CTA owns starts at a runtime offset (see ``run_tma_gather_b1``).

Input readiness is mostly whole-rank: the communication kernel publishes counters
for the whole grid and the kernel waits on them once before the work loop. The one
exception is the FC1 scale-factor pool, which is counted per expert in refine
blocks, so ``run_tma_b`` gates each FC1 tile on its own expert.
"""

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
    Tcgen05TmemPlan,
    make_smem_layouts,
    tcgen05_block_scaled_acc_dtype,
    tcgen05_smem_alloc_type,
)
from .....helpers.dsl_helpers import mark_alignment, spin_wait, tma_multicast_mask
from .....helpers.iket_compat import iket
from .....helpers.smem_workspace import SmemWorkspace
from .....helpers.utils import ceil_div, round_up, strides_equal_ignoring_singletons
from .....quant_def import QuantKind
from ....schedulers.base import SchedulerConsumer
from ....schedulers.fc12_mapping import BlockPhase
from ..local_mega.tma_gather import sm107_tma_gather4_load
from . import dynamic_mainloop
from .block_scaled_swap_ab_fc12_extension import BlockScaledSwapAbFc12Extension


class BlockScaledSwapAbFc12MainloopGenSpecialized(KernelComponent):
    """Own all Rubin load, MMA, pipeline, SMEM, and TMEM mainloop state."""

    ab_pipeline_mbarriers_region: ClassVar[str] = (
        "rubin.genphase_fc12.mainloop.ab_pipeline_mbarriers"
    )
    token_pipeline_mbarriers_region: ClassVar[str] = (
        "rubin.genphase_fc12.mainloop.token_pipeline_mbarriers"
    )
    a_smem_tensor_region: ClassVar[str] = "rubin.genphase_fc12.mainloop.a_smem_tensor"
    b_smem_tensor_region: ClassVar[str] = "rubin.genphase_fc12.mainloop.b_smem_tensor"
    sfa_smem_tensor_region: ClassVar[str] = "rubin.genphase_fc12.mainloop.sfa_smem_tensor"
    sfb_smem_tensor_region: ClassVar[str] = "rubin.genphase_fc12.mainloop.sfb_smem_tensor"
    acc_pipeline_mbarriers_region: ClassVar[str] = (
        "rubin.genphase_fc12.mainloop.acc_pipeline_mbarriers"
    )
    tmem_holding_buffer_region: ClassVar[str] = "rubin.genphase_fc12.mainloop.tmem_holding_buffer"
    tmem_deallocation_mbarrier_region: ClassVar[str] = (
        "rubin.genphase_fc12.mainloop.tmem_deallocation_mbarrier"
    )
    tmem_allocation_barrier_id: ClassVar[int] = 2
    fc1_gather_sync_barrier_id: ClassVar[int] = 3
    # Four warps rather than one because the gather4 row operands live in uniform
    # registers and stay live across the whole K-tile loop: the per-warp row count
    # is the uniform register pressure, so splitting it is what keeps it bounded.
    num_gather_b_warps: ClassVar[int] = 4
    gather4_width: ClassVar[int] = 4
    # One B row must occupy exactly one swizzle atom of this size; see
    # _plan_gather_geometry for what breaks otherwise.
    gather_atom_bytes: ClassVar[int] = 128
    # Row identifiers are loaded eight at a time, which is one 256-bit access and
    # two gather4 groups. Ownership is therefore granular in eight-row runs.
    gather_index_load_rows: ClassVar[int] = 8
    # Fixed rather than configurable: the four-way gather split of §5 is the point
    # of this kernel, and cluster N carries no tokens under swap-AB.
    cluster_shape_mn: ClassVar[Tuple[int, int]] = (4, 1)
    # Rubin's block-scaled instruction K is always the doubled one here.
    mma_k_mode: ClassVar[str] = "2x"
    # Both operands are K-major: an fp4 TCGen05 operand has no other option, and
    # nothing in this kernel wants a different major mode for the wider dtypes.
    a_major_mode: ClassVar[OperandMajorMode] = OperandMajorMode.K
    b_major_mode: ClassVar[OperandMajorMode] = OperandMajorMode.K
    # Token-side depth once the operands stop sharing one. Only the token tile
    # scales B, so a 256-wide tile drags the weight depth down with it even though
    # A's per-stage bytes never change: the shared plan drops from ten stages at
    # 128 to seven at 256. Pinning the token side here and spending the remainder
    # on the weight side is what recovers weight depth.
    #
    # Six leaves the weights nine stages, two more than the shared plan's seven
    # and one short of what the 128 tile gets. The trade runs the other way too:
    # five would give the weights ten, four eleven, three thirteen. Six is where
    # profiling put it -- the token operand is read out of L2, so a handful of
    # stages covers its latency, but not as few as the L2 latency alone suggests.
    asymmetric_token_stages: ClassVar[int] = 6
    # The token tile this specialization applies to; other tiles use the shared plan.
    asymmetric_token_tile: ClassVar[int] = 256

    @classmethod
    def problem_desc_require(cls) -> dict[str, type]:
        return {
            "quant_kind": str,
            "hidden_size": StaticOrRuntimeIntegerType,
            "intermediate_gateup_size": StaticOrRuntimeIntegerType,
        }

    @classmethod
    def impl_desc_require(cls) -> dict[str, object]:
        return {
            "mma_instruction_mnk": tuple,
            "mma_tiler_mnk": tuple,
            "use_2cta_instrs": bool,
            "tmem_plan": Tcgen05TmemPlan,
            "mainloop_smem_budget_bytes": int,
            "num_accumulator_consumer_warps_per_cta": int,
            "sf_padding_block": int,
        }

    def __init__(self, problem_desc: ProblemDesc, impl_desc: ImplDesc) -> None:
        self._validate_desc_inputs(problem_desc, impl_desc)

        self.quant_kind = QuantKind(problem_desc["quant_kind"])
        self.a_dtype = self.quant_kind.weight_dtype
        self.b_dtype = self.quant_kind.activation_dtype
        self.sf_dtype = self.quant_kind.sf_dtype
        self.sf_vec_size = self.quant_kind.sf_vec_size
        self.acc_dtype = tcgen05_block_scaled_acc_dtype
        self.hidden_size = problem_desc["hidden_size"]
        self.intermediate_gateup_size = problem_desc["intermediate_gateup_size"]
        self.sf_padding_block = impl_desc["sf_padding_block"]
        self.mma_instruction_mnk = impl_desc["mma_instruction_mnk"]
        self.mma_tiler_mnk = impl_desc["mma_tiler_mnk"]
        self.use_2cta_instrs = impl_desc["use_2cta_instrs"]
        self.tmem_plan = impl_desc["tmem_plan"]
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
        tmem_plan = self.tmem_plan
        self.num_sfa_tmem_cols = tmem_plan.sfa_columns
        self.num_sfb_tmem_cols = tmem_plan.sfb_columns
        self.num_sf_tmem_cols = tmem_plan.sfa_columns + tmem_plan.sfb_columns
        self.num_tmem_alloc_cols = tmem_plan.allocation_columns
        self.num_accumulator_stages = tmem_plan.accumulator_stage_count
        self.num_accumulator_pipeline_stages = tmem_plan.accumulator_pipeline_stages
        if tmem_plan.accumulator_stage_stride_columns != tmem_plan.accumulator_stage_columns:
            raise ValueError(
                "Rubin Genphase MegaMoE does not support overlapping accumulator stages."
            )
        if tmem_plan.accumulator_pipeline_stages != tmem_plan.accumulator_stage_count:
            raise ValueError(
                "Rubin Genphase MegaMoE requires one pipeline stage per disjoint accumulator stage."
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

        # Only the selected token tile uses asymmetric operand stages.
        self.uses_asymmetric_ab_stages = self.cta_tile_n == self.asymmetric_token_tile
        self._select_ab_stage_plan()
        self.num_tma_load_bytes = self.ab_stage_tma_bytes * self.mma_cta_count
        if self.num_tma_load_bytes != (
            self.a_stage_tma_bytes
            + self.b_stage_tma_bytes
            + self.sfa_stage_tma_bytes
            + self.sfb_stage_tma_bytes
        ):
            raise ValueError(
                "Genphase MegaMoE AB stage transaction byte accounting is inconsistent."
            )
        # Under the split plan each pipeline has a single producer whose expectation
        # is exactly the bytes it issues. The shared plan instead has two producers
        # splitting one combined expectation, so neither share matches the bytes
        # that warp actually sends -- which is why it needs the even split.
        self.weight_pipeline_tx_count = self.a_stage_tma_bytes + self.sfa_stage_tma_bytes
        self.token_pipeline_tx_count = self.b_stage_tma_bytes + self.sfb_stage_tma_bytes
        if not self.uses_asymmetric_ab_stages:
            if self.num_tma_load_bytes % 2 != 0:
                raise ValueError(
                    "Genphase MegaMoE total AB transaction bytes must split evenly across two arrivals."
                )
        self.ab_pipeline_tx_count_per_producer = self.num_tma_load_bytes // 2
        self._plan_gather_geometry()
        self.fc1_gather_sync_barrier = pipeline.NamedBarrier(
            barrier_id=self.fc1_gather_sync_barrier_id,
            num_threads=32 * (1 + self.num_gather_b_warps),
        )

    def _plan_gather_geometry(self) -> None:
        """Derive the interleaved gather4 ownership and check its preconditions.

        Rows a CTA's B SMEM holds is ``cta_tile_n / mma_cta_count``; the CTAs that
        share a B multicast group split that between them. The MMA CTA count
        cancels, so every CTA issues ``cta_tile_n / cluster_m`` rows whichever
        instruction width is in use.

        Ownership is granular in eight-row runs, interleaved across issuing units
        rather than block-partitioned. Eight rows is what one 256-bit access covers,
        so a run's identifiers are one load and its two gather4 destinations are
        adjacent. Interleaving spreads a short row window over every issuing unit
        instead of piling it onto unit zero, and it avoids block partitioning's
        requirement that the live row count be a multiple of the whole span --
        which is what keeps the door open for a dynamic transaction count later.
        """
        self.gather_b_rows_per_cta = self.cta_tile_n // self.mma_cta_count
        self.gather_issuing_units = self.num_mcast_ctas_b * self.num_gather_b_warps
        self.gather_b_groups_per_run = self.gather_index_load_rows // self.gather4_width
        run_span_rows = self.gather_issuing_units * self.gather_index_load_rows
        if self.gather_b_rows_per_cta % run_span_rows != 0:
            # The MMA CTA count cancels out of both sides, so the requirement is
            # purely on tile N and cluster M.
            required_multiple = (
                self.gather_index_load_rows * self.num_gather_b_warps * self.cluster_shape_mn[0]
            )
            raise ValueError(
                f"Genphase MegaMoE gather runs must divide evenly across {self.gather_issuing_units} issuing "
                f"units: tile N {self.cta_tile_n} must be a multiple of {required_multiple}."
            )
        self.gather_b_runs_per_unit = self.gather_b_rows_per_cta // run_span_rows
        self.gather_b_rows_per_unit = self.gather_b_runs_per_unit * self.gather_index_load_rows

        # The gather writes four consecutive box images of a (1, mma_tile_k) box
        # under a 128-byte swizzle. That matches the planned B stage only when one
        # row occupies exactly one swizzle atom: above 128 bytes `make_smem_layouts`
        # splits the K mode into interleaved chunks that no single base address can
        # address, and below it the region picks a narrower swizzle than the
        # descriptor's. Both make the flat destination offset silently wrong.
        gather_row_bytes = self.mma_tile_k * int(self.b_smem_alloc_dtype.width) // 8
        if gather_row_bytes != self.gather_atom_bytes:
            raise ValueError(
                f"{self.quant_kind} FC1 gather requires mma_tiler K == "
                f"{self.gather_atom_bytes * 8 // int(self.b_smem_alloc_dtype.width)}, got {self.mma_tile_k}: one B "
                f"row must occupy exactly one {self.gather_atom_bytes}-byte swizzle atom, and this one occupies "
                f"{gather_row_bytes}."
            )

        # The AB pipeline registers one static transaction expectation that serves
        # both phases, so FC1's gather has to move exactly as many B bytes as FC2's
        # tiled TMA. The two sides of this are computed independently -- the left
        # from the tile geometry here, the right by `make_smem_layouts` -- so the
        # comparison catches a drift between them rather than restating one of them.
        self.gather_b_stage_bytes = self.cta_tile_n * self.mma_tile_k * int(self.b_dtype.width) // 8
        if self.gather_b_stage_bytes != self.b_stage_tma_bytes:
            raise ValueError(
                f"Genphase MegaMoE FC1 Gather-B moves {self.gather_b_stage_bytes} B bytes per AB stage but FC2's "
                f"TMA-B moves {self.b_stage_tma_bytes}; one static transaction expectation cannot serve both."
            )

    def _validate_configuration(self) -> None:
        for field_name, dimensions, expected_rank in (
            ("mma_instruction_mnk", self.mma_instruction_mnk, 3),
            ("mma_tiler_mnk", self.mma_tiler_mnk, 3),
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
                "Rubin Genphase MegaMoE does not implement M/N instruction repetition."
            )
        # Tile N 64 is excluded: it forces the halved SFB tile index and the SFB
        # TMEM shift, two special paths whose interaction with a dynamic
        # instruction N is not worth carrying for a token extent this small.
        if self.mma_tiler_mnk[1] not in (128, 256):
            raise ValueError("Rubin Genphase MegaMoE supports tile N in (128, 256).")
        if self.mma_tiler_mnk[2] % self.instruction_k != 0:
            raise ValueError("mma_tiler K must be divisible by instruction K.")
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
        # Under the shared plan this one array serves all four operands. Under the
        # split plan it becomes the weight side's and the token side gets its own,
        # because the two depths no longer index the same stage.
        smem_workspace.register_mbarrier(
            self.ab_pipeline_mbarriers_region, self.num_weight_ab_stages * 2
        )
        if self.uses_asymmetric_ab_stages:
            smem_workspace.register_mbarrier(
                self.token_pipeline_mbarriers_region, self.num_token_ab_stages * 2
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
        plan_tail_bytes = (2 * self.num_accumulator_pipeline_stages + 2) * mbarrier_bytes
        stage_budget_bytes = self.mainloop_smem_budget_bytes - plan_tail_bytes
        # Each pipeline spends two mbarriers per stage, full and empty. The shared
        # plan pays that once for all four operands; the split plan pays it twice,
        # which is the only overhead the split introduces.
        weight_stage_cost_bytes = (
            stage_one_regions[0].nbytes + stage_one_regions[2].nbytes + 2 * mbarrier_bytes
        )
        token_stage_cost_bytes = (
            stage_one_regions[1].nbytes + stage_one_regions[3].nbytes + 2 * mbarrier_bytes
        )

        if self.uses_asymmetric_ab_stages:
            self.num_token_ab_stages = self.asymmetric_token_stages
            self.num_weight_ab_stages = (
                stage_budget_bytes - self.num_token_ab_stages * token_stage_cost_bytes
            ) // weight_stage_cost_bytes
            if self.num_weight_ab_stages < 1:
                raise ValueError(
                    f"{self.num_token_ab_stages} token stages leave no room for a weight stage in the "
                    f"{self.mainloop_smem_budget_bytes}-byte mainloop budget."
                )
            # Splitting is pointless if it does not buy depth, and a plan that
            # bought none would silently pay the second mbarrier set for nothing.
            shared_stages = stage_budget_bytes // (weight_stage_cost_bytes + token_stage_cost_bytes)
            if self.num_weight_ab_stages <= shared_stages:
                raise ValueError(
                    f"Asymmetric stages give the weight side {self.num_weight_ab_stages} stages, no more than the "
                    f"{shared_stages} a shared plan reaches; the token depth of {self.num_token_ab_stages} is too deep "
                    f"to be worth splitting."
                )
        else:
            shared_stages = stage_budget_bytes // (weight_stage_cost_bytes + token_stage_cost_bytes)
            self.num_weight_ab_stages = shared_stages
            self.num_token_ab_stages = shared_stages
            if shared_stages < 1:
                raise ValueError(
                    f"One AB stage needs {weight_stage_cost_bytes + token_stage_cost_bytes + plan_tail_bytes} bytes, "
                    f"exceeding the {self.mainloop_smem_budget_bytes}-byte mainloop budget."
                )
        # Retained for the callers that size one pipeline's state: under the shared
        # plan both depths are this, and under the split plan the weight side is
        # the one the MMA's primary state follows.
        self.num_ab_pipeline_stages = self.num_weight_ab_stages

        (self.a_smem_region, _, self.sfa_smem_region, _) = make_smem_layouts(
            self.mma_instruction,
            self.mma_tiler_mnk,
            self.num_weight_ab_stages,
            (self.a_major_mode, self.b_major_mode),
            self.architecture,
        )
        (_, self.b_smem_region, _, self.sfb_smem_region) = make_smem_layouts(
            self.mma_instruction,
            self.mma_tiler_mnk,
            self.num_token_ab_stages,
            (self.a_major_mode, self.b_major_mode),
            self.architecture,
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

        # Each operand is staged to its own side's depth; under the shared plan the
        # two depths are the same number and this is the previous behaviour.
        self.a_smem_composed_layout_staged = sm100_utils.make_smem_layout_a(
            self.tiled_mma, self.mma_tiler_mnk, self.a_smem_alloc_dtype, self.num_weight_ab_stages
        )
        self.b_smem_composed_layout_staged = sm100_utils.make_smem_layout_b(
            self.tiled_mma, self.mma_tiler_mnk, self.b_smem_alloc_dtype, self.num_token_ab_stages
        )
        self.a_smem_layout_staged = self.a_smem_composed_layout_staged.outer
        self.b_smem_layout_staged = self.b_smem_composed_layout_staged.outer
        self.sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
            self.tiled_mma, self.mma_tiler_mnk, self.sf_vec_size, self.num_weight_ab_stages
        )
        self.sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
            self.tiled_mma, self.mma_tiler_mnk, self.sf_vec_size, self.num_token_ab_stages
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
        # FC1 B is read from the dense (source rank, source token) activation plane
        # the communication kernel filled. A one-row descriptor lets the producer
        # warps supply arbitrary source row IDs through TMA TILE_GATHER4 while
        # landing in the native K_SW128 B layout. Only the atom is returned: the
        # gather supplies coordinates directly and never partitions a tensor view.
        fc1_b_2d = fc1_b[(None, None, 0)]
        gather_b_base = cute.make_layout((1, self.mma_tile_k), stride=(self.mma_tile_k, 1))
        gather_b_smem_layout = cute.make_composed_layout(
            cute.make_swizzle(3, 4, 3), 0, gather_b_base
        )
        fc1_b_atom, _ = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), fc1_b_2d, gather_b_smem_layout, (1, self.mma_tile_k)
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
        Gather-B never arrives: it only lets its completions decrement the byte
        count that TMA-B already registered.
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
    def create_weight_pipeline(
        self, smem_workspace: SmemWorkspace, smem_base: cute.Pointer
    ) -> pipeline.PipelineAsync:
        """Create the weight half of the split TMA-to-UMMA pipeline.

        One producer, TMA-A, and its expectation is exactly the A and SFA bytes it
        issues -- unlike the shared pipeline, where two producers each register
        half of a combined count that matches neither warp's traffic.
        """
        producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        num_tma_consumers = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, num_tma_consumers)
        return pipeline.PipelineTmaUmma.create(
            barrier_storage=smem_workspace.ptr(self.ab_pipeline_mbarriers_region, smem_base),
            num_stages=self.num_weight_ab_stages,
            producer_group=producer_group,
            consumer_group=consumer_group,
            tx_count=self.weight_pipeline_tx_count,
            cta_layout_vmnk=self.cluster_layout_vmnk,
            defer_sync=True,
        )

    @cute.jit
    def create_token_pipeline(
        self, smem_workspace: SmemWorkspace, smem_base: cute.Pointer
    ) -> pipeline.PipelineAsync:
        """Create the token half of the split TMA-to-UMMA pipeline.

        One producer, TMA-B, registering the B and SFB bytes. Gather-B still never
        arrives: on FC1 it only lets its completions decrement the count TMA-B
        registered, which is why the two share this pipeline rather than getting
        one each.
        """
        producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        num_tma_consumers = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, num_tma_consumers)
        return pipeline.PipelineTmaUmma.create(
            barrier_storage=smem_workspace.ptr(self.token_pipeline_mbarriers_region, smem_base),
            num_stages=self.num_token_ab_stages,
            producer_group=producer_group,
            consumer_group=consumer_group,
            tx_count=self.token_pipeline_tx_count,
            cta_layout_vmnk=self.cluster_layout_vmnk,
            defer_sync=True,
        )

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

        # The two phases keep separate loop bodies rather than selecting atoms from
        # a shared one: `is_fc1` is a runtime predicate, so the atoms and the
        # per-phase tensor views cannot be chosen by a Python conditional.
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
        fc1_sf_ready_pointer: cute.Pointer,
        expert_sizes: cute.Tensor,
    ) -> None:
        """Run the TMA-B warp: FC1 scale factors only, FC2 token data and scale factors.

        FC1's token payload is Gather-B's job, so this warp contributes only the
        SFB bytes there. It still owns the whole static transaction expectation
        for the stage, which is why the named barrier below has to release the
        Gather-B warps only after ``producer_acquire`` has registered it.

        The FC1 scale factors are gated per expert rather than once for the whole
        pool: the communication kernel counts refine blocks into
        ``fc1_sf_ready_pointer``, and both sides walk experts in the same order, so
        this tile only has to outlast its own expert's blocks.
        """
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
                iket.range_push("tma_b.wait_expert_sf_ready")
                expert_idx = work_tile.expert_idx
                expert_sf_blocks = ceil_div(expert_sizes[expert_idx], Int32(self.sf_padding_block))
                spin_wait(
                    fc1_sf_ready_pointer + expert_idx,
                    lambda value: value == expert_sf_blocks,
                    scope="gpu",
                )
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

                global_sfb_slice = partitioned_global_sfb[(None, work_tile.tile_n_idx, None, 0)]

                empty_status = ab_pipeline.producer_try_acquire(ab_pipeline_state)
                for k_tile_idx in cutlass.range(k_tile_count, unroll=1):
                    ab_pipeline.producer_acquire(ab_pipeline_state, empty_status)
                    tma_barrier = ab_pipeline.producer_get_barrier(ab_pipeline_state)
                    stage_index = ab_pipeline_state.index
                    # TMA-B has waited for this AB stage to become empty and
                    # contributed one of the two static producer transaction
                    # expectations. Rendezvous with the Gather-B warps before any
                    # of them issues Gather4 against this barrier.
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

                # Under two-CTA instructions the dynamic N window is split across
                # the pair, so the non-leader's rows start at the runtime midpoint
                # rather than at half the static tile.
                if cutlass.const_expr(self.use_2cta_instrs):
                    if not self.is_leader_cta:
                        load_shift = dynamic_mainloop.compute_non_leader_cta_load_shift(
                            valid_tokens_in_tile=work_tile.valid_tokens_in_cta_tile,
                            mma_tiler_n=self.mma_tiler_mnk[1],
                        )
                        real_b = cute.domain_offset((load_shift, 0, 0), real_b)

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
                global_sfb_slice = partitioned_global_sfb[(None, work_tile.tile_n_idx, None, 0)]

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
        gather_index: cute.Tensor,
        ab_pipeline: pipeline.PipelineAsync,
        ab_pipeline_state: pipeline.PipelineState,
        sched_consumer: SchedulerConsumer,
        gather_warp_idx: Int32,
    ) -> None:
        """Gather dense-order FC1 activations directly into B SMEM.

        Every CTA of the B multicast group issues its share and multicasts it to
        the whole group, so the group mask is the ordinary ``"b"`` mask each CTA
        already computes for itself -- no leader election. Ownership is
        interleaved by gather4 group so that a short row window spreads over all
        issuing units rather than landing entirely on group zero.

        Gather-B contributes transaction completions only; it is not an AB
        producer. FC2 is fully produced by TMA-A and TMA-B, so on those tiles this
        warp only keeps its register-held pipeline state aligned.
        """
        row_ids = cute.make_rmem_tensor(
            cute.make_layout((self.gather_b_rows_per_unit,)), cutlass.Int32
        )
        row_id_runs = cute.zipped_divide(row_ids, (self.gather_index_load_rows,))
        index_load_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            cutlass.Int32,
            num_bits_per_copy=self.gather_index_load_rows * cutlass.Int32.width,
        )
        index_load_bytes = self.gather_index_load_rows * cutlass.Int32.width // 8
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
        issuing_unit = self.main_vmnk_coord[1] * Int32(self.num_gather_b_warps) + gather_warp_idx

        work_tile = sched_consumer.consume_work()
        while work_tile.is_valid_tile:
            is_fc1 = work_tile.phase == Int32(BlockPhase.Linear1)
            if is_fc1:
                iket.range_push("tma_gather_b1_fc1")
                tile_pool_base = (
                    work_tile.cumulative_data_physical_row
                    + work_tile.tile_n_idx * Int32(self.cta_tile_n)
                )
                # The MMA issues a dynamic instruction N, so under two-CTA
                # instructions the pair splits that runtime extent rather than the
                # static tile: the non-leader's local row zero is the midpoint of
                # the aligned token count. This mirrors what
                # `compute_non_leader_cta_load_shift` does for the tiled TMA legs.
                # Rounding to sixteen keeps that midpoint a multiple of eight, so
                # every run stays aligned to its 256-bit index load.
                cta_row_base = self.mma_cta_index * (
                    round_up(work_tile.valid_tokens_in_cta_tile, 16) // Int32(2)
                )
                index_runs = cute.zipped_divide(
                    cute.domain_offset((tile_pool_base + cta_row_base,), gather_index),
                    (self.gather_index_load_rows,),
                )
                for local_run in cutlass.range_constexpr(self.gather_b_runs_per_unit):
                    run_index = Int32(local_run * self.gather_issuing_units) + issuing_unit
                    cute.copy(
                        index_load_atom,
                        mark_alignment(index_runs[(None, run_index)], index_load_bytes),
                        row_id_runs[(None, local_run)],
                    )

                for k_tile_idx in cutlass.range(self.fc1_k_tile_count, unroll=1):
                    # TMA-B has already acquired this stage and contributed its
                    # static transaction expectation. The named barrier orders that
                    # arrive-and-expect-tx before Gather4.
                    self.fc1_gather_sync_barrier.arrive_and_wait()
                    tma_barrier = ab_pipeline.producer_get_barrier(ab_pipeline_state)
                    stage_index = ab_pipeline_state.index
                    col_k = k_tile_idx * Int32(self.mma_tile_k)
                    stage_base_elements = stage_index * Int32(
                        self.gather_b_rows_per_cta * self.mma_tile_k
                    )

                    for local_run in cutlass.range_constexpr(self.gather_b_runs_per_unit):
                        run_index = Int32(local_run * self.gather_issuing_units) + issuing_unit
                        run_base_elements = stage_base_elements + run_index * Int32(
                            self.gather_index_load_rows * self.mma_tile_k
                        )
                        for group_in_run in cutlass.range_constexpr(self.gather_b_groups_per_run):
                            destination_offset = run_base_elements + Int32(
                                group_in_run * self.gather4_width * self.mma_tile_k
                            )
                            row_start = (
                                local_run * self.gather_index_load_rows
                                + group_in_run * self.gather4_width
                            )
                            with cute.arch.elect_one():
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
                advanced_index = ab_pipeline_state.index + Int32(self.fc2_k_tile_count)
                stage_count = Int32(ab_pipeline_state.stages)
                wrap_count = advanced_index // stage_count
                ab_pipeline_state._index = advanced_index % stage_count
                ab_pipeline_state._phase = ab_pipeline_state.phase ^ (wrap_count & Int32(1))
                ab_pipeline_state._count += Int32(self.fc2_k_tile_count)
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
        """Execute SM107 MMA with an instruction N that follows the live token count."""
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
                # At tile N 256 a dynamic extent of 128 or less makes the SFB TMEM
                # window compact, so its two K halves come from a different pair of
                # SMEM rest coordinates than the full-width case.
                use_n128_sfb_mapping = False
                if cutlass.const_expr(
                    self.quant_kind == QuantKind.nvfp4 and self.cta_tile_n == 256
                ):
                    use_n128_sfb_mapping = round_up(
                        work_tile.valid_tokens_in_cta_tile, 16
                    ) <= Int32(128)

                acc_pipeline.producer_acquire(acc_producer_state)
                if use_n128_sfb_mapping:
                    next_ab_pipeline_state = ab_pipeline_state.clone()
                    next_ab_pipeline_state.advance()
                    full_status = ab_pipeline.consumer_try_wait(ab_pipeline_state)
                    for k_tile_idx in cutlass.range(k_tile_count, unroll=1):
                        ab_pipeline.consumer_wait(ab_pipeline_state, full_status)
                        next_full_status = cutlass.Boolean(1)
                        if k_tile_idx + 1 < k_tile_count:
                            next_full_status = ab_pipeline.consumer_try_wait(next_ab_pipeline_state)

                        s2t_stage_coord = (None, None, None, None, ab_pipeline_state.index)
                        cute.copy(
                            tiled_copy_sfa,
                            partitioned_smem_sfa[s2t_stage_coord],
                            partitioned_tmem_sfa,
                        )
                        for instruction_index in cutlass.range_constexpr(
                            self.num_mma_instructions_per_ab_stage
                        ):
                            # Rest_Tiler order: N0K0, N128K0, N0K64, N128K64.
                            lower_n_lower_k_source_coord = (
                                None,
                                0,
                                None,
                                instruction_index,
                                ab_pipeline_state.index,
                            )
                            lower_n_lower_k_destination_coord = (None, 0, None, instruction_index)
                            cute.copy(
                                tiled_copy_sfb,
                                partitioned_smem_sfb[lower_n_lower_k_source_coord],
                                partitioned_tmem_sfb[lower_n_lower_k_destination_coord],
                            )
                            lower_n_upper_k_source_coord = (
                                None,
                                2,
                                None,
                                instruction_index,
                                ab_pipeline_state.index,
                            )
                            compact_upper_k_destination_coord = (None, 1, None, instruction_index)
                            cute.copy(
                                tiled_copy_sfb,
                                partitioned_smem_sfb[lower_n_upper_k_source_coord],
                                partitioned_tmem_sfb[compact_upper_k_destination_coord],
                            )

                        a_stage_frag = fragment_a[(None, None, None, ab_pipeline_state.index)]
                        b_stage_frag = fragment_b[(None, None, None, ab_pipeline_state.index)]
                        dynamic_mainloop.issue_dynamic_block_scaled_mma_window(
                            quant_kind=self.quant_kind,
                            acc_tensor=accumulator,
                            a_window_frag=a_stage_frag,
                            b_window_frag=b_stage_frag,
                            sfa_window_tensor=tmem_sfa,
                            sfb_window_tensor=tmem_sfb,
                            valid_tokens_in_tile=work_tile.valid_tokens_in_cta_tile,
                            mma_instruction_mnk=self.mma_instruction_mnk,
                            window_instruction_offset=0,
                            window_instruction_count=self.num_mma_instructions_per_ab_stage,
                            first_instruction_accumulate=k_tile_idx != 0,
                        )

                        ab_pipeline.consumer_release(ab_pipeline_state)
                        ab_pipeline_state.advance()
                        next_ab_pipeline_state.advance()
                        full_status = next_full_status
                else:
                    next_ab_pipeline_state = ab_pipeline_state.clone()
                    next_ab_pipeline_state.advance()
                    full_status = ab_pipeline.consumer_try_wait(ab_pipeline_state)
                    for k_tile_idx in cutlass.range(k_tile_count, unroll=1):
                        ab_pipeline.consumer_wait(ab_pipeline_state, full_status)
                        next_full_status = cutlass.Boolean(1)
                        if k_tile_idx + 1 < k_tile_count:
                            next_full_status = ab_pipeline.consumer_try_wait(next_ab_pipeline_state)

                        s2t_stage_coord = (None, None, None, None, ab_pipeline_state.index)
                        cute.copy(
                            tiled_copy_sfa,
                            partitioned_smem_sfa[s2t_stage_coord],
                            partitioned_tmem_sfa,
                        )
                        cute.copy(
                            tiled_copy_sfb,
                            partitioned_smem_sfb[s2t_stage_coord],
                            partitioned_tmem_sfb,
                        )

                        a_stage_frag = fragment_a[(None, None, None, ab_pipeline_state.index)]
                        b_stage_frag = fragment_b[(None, None, None, ab_pipeline_state.index)]
                        dynamic_mainloop.issue_dynamic_block_scaled_mma_window(
                            quant_kind=self.quant_kind,
                            acc_tensor=accumulator,
                            a_window_frag=a_stage_frag,
                            b_window_frag=b_stage_frag,
                            sfa_window_tensor=tmem_sfa,
                            sfb_window_tensor=tmem_sfb,
                            valid_tokens_in_tile=work_tile.valid_tokens_in_cta_tile,
                            mma_instruction_mnk=self.mma_instruction_mnk,
                            window_instruction_offset=0,
                            window_instruction_count=self.num_mma_instructions_per_ab_stage,
                            first_instruction_accumulate=k_tile_idx != 0,
                        )

                        ab_pipeline.consumer_release(ab_pipeline_state)
                        ab_pipeline_state.advance()
                        next_ab_pipeline_state.advance()
                        full_status = next_full_status
                acc_pipeline.producer_commit(acc_producer_state)

            acc_producer_state.advance()
            work_tile = sched_consumer.consume_work()
        acc_pipeline.producer_tail(acc_producer_state)

    @cute.jit
    def run_mma_asymmetric(
        self,
        *,
        tmem_allocator,
        weight_pipeline: pipeline.PipelineAsync,
        weight_pipeline_state: pipeline.PipelineState,
        token_pipeline: pipeline.PipelineAsync,
        token_pipeline_state: pipeline.PipelineState,
        acc_pipeline: pipeline.PipelineAsync,
        sched_consumer: SchedulerConsumer,
    ) -> None:
        """``run_mma`` for the split-depth plan, where the two operands no longer share a stage index.

        The view construction below duplicates ``run_mma``'s; the two have to be
        kept in step. What differs is only the stage bookkeeping: A and SFA are
        indexed by the weight state, B and SFB by the token state, and each stage
        is waited on and released through its own pipeline.

        No ``consumer_try_wait`` here, unlike ``run_mma``. Prefetching the next
        stage's readiness overlaps a barrier wait with MMA issue, which is worth a
        four-variable state machine when there is one pipeline -- but with two it
        doubles into eight, and this shape is weight-bound anyway: the MMA warp is
        waiting on DRAM, not on the barrier. A plain wait is also far easier for
        the compiler to schedule than two predicated ones.
        """
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
                # At tile N 256 a dynamic extent of 128 or less makes the SFB TMEM
                # window compact, so its two K halves come from a different pair of
                # SMEM rest coordinates than the full-width case. Tested inside the
                # loop rather than around it, which ``run_mma`` does to keep its
                # state machine out of the branch: the predicate is warp-uniform
                # and this loop has no state machine to duplicate.
                use_n128_sfb_mapping = False
                if cutlass.const_expr(
                    self.quant_kind == QuantKind.nvfp4 and self.cta_tile_n == 256
                ):
                    use_n128_sfb_mapping = round_up(
                        work_tile.valid_tokens_in_cta_tile, 16
                    ) <= Int32(128)

                acc_pipeline.producer_acquire(acc_producer_state)
                for k_tile_idx in cutlass.range(k_tile_count, unroll=1):
                    weight_pipeline.consumer_wait(weight_pipeline_state)
                    token_pipeline.consumer_wait(token_pipeline_state)

                    weight_stage_coord = (None, None, None, None, weight_pipeline_state.index)
                    cute.copy(
                        tiled_copy_sfa,
                        partitioned_smem_sfa[weight_stage_coord],
                        partitioned_tmem_sfa,
                    )
                    if use_n128_sfb_mapping:
                        for instruction_index in cutlass.range_constexpr(
                            self.num_mma_instructions_per_ab_stage
                        ):
                            # Rest_Tiler order: N0K0, N128K0, N0K64, N128K64.
                            lower_n_lower_k_source_coord = (
                                None,
                                0,
                                None,
                                instruction_index,
                                token_pipeline_state.index,
                            )
                            cute.copy(
                                tiled_copy_sfb,
                                partitioned_smem_sfb[lower_n_lower_k_source_coord],
                                partitioned_tmem_sfb[(None, 0, None, instruction_index)],
                            )
                            lower_n_upper_k_source_coord = (
                                None,
                                2,
                                None,
                                instruction_index,
                                token_pipeline_state.index,
                            )
                            cute.copy(
                                tiled_copy_sfb,
                                partitioned_smem_sfb[lower_n_upper_k_source_coord],
                                partitioned_tmem_sfb[(None, 1, None, instruction_index)],
                            )
                    else:
                        token_stage_coord = (None, None, None, None, token_pipeline_state.index)
                        cute.copy(
                            tiled_copy_sfb,
                            partitioned_smem_sfb[token_stage_coord],
                            partitioned_tmem_sfb,
                        )

                    a_stage_frag = fragment_a[(None, None, None, weight_pipeline_state.index)]
                    b_stage_frag = fragment_b[(None, None, None, token_pipeline_state.index)]
                    dynamic_mainloop.issue_dynamic_block_scaled_mma_window(
                        quant_kind=self.quant_kind,
                        acc_tensor=accumulator,
                        a_window_frag=a_stage_frag,
                        b_window_frag=b_stage_frag,
                        sfa_window_tensor=tmem_sfa,
                        sfb_window_tensor=tmem_sfb,
                        valid_tokens_in_tile=work_tile.valid_tokens_in_cta_tile,
                        mma_instruction_mnk=self.mma_instruction_mnk,
                        window_instruction_offset=0,
                        window_instruction_count=self.num_mma_instructions_per_ab_stage,
                        first_instruction_accumulate=k_tile_idx != 0,
                    )

                    weight_pipeline.consumer_release(weight_pipeline_state)
                    token_pipeline.consumer_release(token_pipeline_state)
                    weight_pipeline_state.advance()
                    token_pipeline_state.advance()
                acc_pipeline.producer_commit(acc_producer_state)

            acc_producer_state.advance()
            work_tile = sched_consumer.consume_work()
        acc_pipeline.producer_tail(acc_producer_state)


__all__ = ["BlockScaledSwapAbFc12MainloopGenSpecialized"]
