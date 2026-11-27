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
for the whole grid and the kernel waits on them once (on the first routed FC1 tile
when shared-expert tiles run ahead). The one exception is the FC1 scale-factor pool,
which is counted per expert in refine blocks, so ``run_tma_b`` gates each FC1 tile
on its own expert.
"""

from typing import ClassVar, Optional, Tuple

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
import cutlass.utils.rubin_helpers as sm107_utils
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, tcgen05
from cutlass.cute.nvgpu.cpasync.copy import CopyBulkTensorTileG2SMulticastTrait
from cutlass.cute.typing import AddressSpace
from cutlass.cutlass_dsl import Int32

from .....api import ImplDesc, KernelComponent, OptionalRequirement, ProblemDesc, StaticOrRuntimeIntegerType
from .....helpers.ptx_helpers import barrier_sync_unaligned
from .....helpers.cute_py_helpers import (
    Tcgen05MmaInstruction,
    Tcgen05TmemPlan,
    make_smem_layouts,
    tcgen05_block_scaled_acc_dtype,
    tcgen05_smem_alloc_type,
)
from .....helpers.dsl_helpers import ReadyFlag, wait_ready, mark_alignment, spin_wait, tma_multicast_mask
from .....helpers.iket_compat import iket
from .....helpers.smem_workspace import SmemWorkspace
from .....helpers.utils import ceil_div, round_up
from .....quant_def import QuantKind
from ....schedulers.base import SchedulerConsumer
from ....schedulers.fc12_mapping import BlockPhase
from . import dynamic_mainloop
from .block_scaled_swap_ab_fc12_extension import BlockScaledSwapAbFc12Extension
from .shared_expert_mainloop import SharedKindMainloopMixin
from ..local_mega.tma_gather import sm107_tma_gather4_load
from ....blackwell.custom_mix_cga_helpers import bind_executable_tma_load_fields, make_executable_tma_atom_or_pair


def _bind_tma_issue(exec_atom: cute.CopyAtom, tma_barrier: cute.Pointer, mcast_mask) -> cute.CopyAtom:
    """Bind one stage's barrier (and, on multicast atoms, the mask) to an executable TMA load atom."""
    if not isinstance(exec_atom._trait, CopyBulkTensorTileG2SMulticastTrait):
        mcast_mask = None
    return bind_executable_tma_load_fields(exec_atom, tma_bar_ptr=tma_barrier, tma_desc_ptr=None, mcast_mask=mcast_mask)

class BlockScaledSwapAbFc12MainloopGenSpecialized(SharedKindMainloopMixin, KernelComponent):
    """Own all Rubin load, MMA, pipeline, SMEM, and TMEM mainloop state."""

    ab_pipeline_mbarriers_region: ClassVar[str] = "rubin.genphase_fc12.mainloop.ab_pipeline_mbarriers"
    token_pipeline_mbarriers_region: ClassVar[str] = "rubin.genphase_fc12.mainloop.token_pipeline_mbarriers"
    a_smem_tensor_region: ClassVar[str] = "rubin.genphase_fc12.mainloop.a_smem_tensor"
    b_smem_tensor_region: ClassVar[str] = "rubin.genphase_fc12.mainloop.b_smem_tensor"
    sfa_smem_tensor_region: ClassVar[str] = "rubin.genphase_fc12.mainloop.sfa_smem_tensor"
    sfb_smem_tensor_region: ClassVar[str] = "rubin.genphase_fc12.mainloop.sfb_smem_tensor"
    # Heterogeneous shared expert: the shared kind's views of the same operand stages.
    shared_a_smem_tensor_region: ClassVar[str] = "rubin.genphase_fc12.mainloop.shared_a_smem_tensor"
    shared_b_smem_tensor_region: ClassVar[str] = "rubin.genphase_fc12.mainloop.shared_b_smem_tensor"
    shared_sfa_smem_tensor_region: ClassVar[str] = "rubin.genphase_fc12.mainloop.shared_sfa_smem_tensor"
    shared_sfb_smem_tensor_region: ClassVar[str] = "rubin.genphase_fc12.mainloop.shared_sfb_smem_tensor"
    acc_pipeline_mbarriers_region: ClassVar[str] = "rubin.genphase_fc12.mainloop.acc_pipeline_mbarriers"
    tmem_holding_buffer_region: ClassVar[str] = "rubin.genphase_fc12.mainloop.tmem_holding_buffer"
    tmem_deallocation_mbarrier_region: ClassVar[str] = "rubin.genphase_fc12.mainloop.tmem_deallocation_mbarrier"
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
    # The token tile this specialization applies to. At 128 the shared plan
    # already reaches 93% of DRAM peak and is left alone.
    asymmetric_token_tile: ClassVar[int] = 256

    @classmethod
    def problem_desc_require(cls) -> dict[str, type]:
        return {
            "quant_kind": str,
            "hidden_size": StaticOrRuntimeIntegerType,
            "intermediate_gateup_size": StaticOrRuntimeIntegerType,
            # Shared expert (optional): FC1 K, FC1 N (gate-up), FC2 N and quant kind (mxfp8_e4m3 or mxfp8_e5m2).
            "shared_hidden_size": OptionalRequirement(Optional[int]),
            "shared_intermediate_gateup_size": OptionalRequirement(Optional[int]),
            "shared_output_size": OptionalRequirement(Optional[int]),
            "shared_quant_kind": OptionalRequirement(Optional[str]),
        }

    @classmethod
    def impl_desc_require(cls) -> dict[str, object]:
        return {
            "mma_instruction_mnk": tuple,
            "mma_tiler_mnk": tuple,
            "use_2cta_instrs": bool,
            "cluster_shape_mn": tuple,
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
        self._parse_shared_problem(problem_desc)
        self.sf_padding_block = impl_desc["sf_padding_block"]
        self.mma_instruction_mnk = impl_desc["mma_instruction_mnk"]
        self.mma_tiler_mnk = impl_desc["mma_tiler_mnk"]
        self.use_2cta_instrs = impl_desc["use_2cta_instrs"]
        self.cluster_shape_mn = tuple(impl_desc["cluster_shape_mn"])
        self.num_gather_b_warps = 2 if self.mma_tiler_mnk[1] == 64 and self.cluster_shape_mn == (4, 1) else 4
        self.tmem_plan = impl_desc["tmem_plan"]
        self.mainloop_smem_budget_bytes = impl_desc["mainloop_smem_budget_bytes"]
        self.num_accumulator_consumer_warps_per_cta = impl_desc["num_accumulator_consumer_warps_per_cta"]

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
            raise ValueError("Rubin Genphase MegaMoE does not support overlapping accumulator stages.")
        if tmem_plan.accumulator_pipeline_stages != tmem_plan.accumulator_stage_count:
            raise ValueError("Rubin Genphase MegaMoE requires one pipeline stage per disjoint accumulator stage.")
        self.num_accumulator_tmem_cols = tmem_plan.accumulator_columns
        self.accumulator_shape = (self.cta_tile_m, self.cta_tile_n, tmem_plan.accumulator_stage_count)
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
        self.a_smem_alloc_dtype = tcgen05_smem_alloc_type(self.a_dtype, self.b_dtype, self.architecture)
        self.b_smem_alloc_dtype = tcgen05_smem_alloc_type(self.b_dtype, self.a_dtype, self.architecture)
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
        self.mma_tiler_sfb = (self.sfb_instruction_shape_mnk[0], self.sfb_instruction_shape_mnk[1], self.mma_tile_k)
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

        self._configure_shared_kind()

        # Only the 256-wide token tile is specialized: at 128 the shared plan
        # already reaches 93% of DRAM peak, so it stays byte-for-byte as it was.
        self.uses_asymmetric_ab_stages = self.cta_tile_n == self.asymmetric_token_tile
        # Decode MMA reads only a 16-row-aligned valid token window. Keep the
        # allocated layout, but omit padding-row B transfers in the qualified
        # N64 geometry and register their exact completion bytes per stage.
        self.dynamic_gather = (
            self.mma_tiler_mnk == self.n64_decode_tile
            and self.cluster_shape_mn == (4, 1)
            and self.use_2cta_instrs
            and not self.uses_asymmetric_ab_stages
        )
        self._select_ab_stage_plan()
        self.num_tma_load_bytes = self.ab_stage_tma_bytes * self.mma_cta_count
        if self.num_tma_load_bytes != (
            self.a_stage_tma_bytes + self.b_stage_tma_bytes + self.sfa_stage_tma_bytes + self.sfb_stage_tma_bytes
        ):
            raise ValueError("Genphase MegaMoE AB stage transaction byte accounting is inconsistent.")
        # Under the split plan each pipeline has a single producer whose expectation
        # is exactly the bytes it issues. The shared plan instead has two producers
        # splitting one combined expectation, so neither share matches the bytes
        # that warp actually sends -- which is why it needs the even split.
        self.weight_pipeline_tx_count = self.a_stage_tma_bytes + self.sfa_stage_tma_bytes
        self.token_pipeline_tx_count = self.b_stage_tma_bytes + self.sfb_stage_tma_bytes
        if not self.uses_asymmetric_ab_stages:
            if self.num_tma_load_bytes % 2 != 0:
                raise ValueError("Genphase MegaMoE total AB transaction bytes must split evenly across two arrivals.")
        self.ab_pipeline_tx_count_per_producer = self.num_tma_load_bytes // 2
        if self.heterogeneous_shared:
            # Shared tiles override the barrier's static expectation per producer (``expected_tx``).
            self.shared_weight_tx_count = self.shared_a_stage_tma_bytes + self.shared_sfa_stage_tma_bytes
            self.shared_token_tx_count = self.shared_b_stage_tma_bytes + self.shared_sfb_stage_tma_bytes
            if self.shared_weight_tx_count + self.shared_token_tx_count != (
                self.shared_ab_stage_tma_bytes * self.mma_cta_count
            ):
                raise ValueError("Shared-expert AB stage transaction byte accounting is inconsistent.")
        self._plan_gather_geometry()
        self.fc1_gather_sync_barrier = pipeline.NamedBarrier(
            barrier_id=self.fc1_gather_sync_barrier_id, num_threads=32 * (1 + self.num_gather_b_warps)
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
            required_multiple = self.gather_index_load_rows * self.num_gather_b_warps * self.cluster_shape_mn[0]
            raise ValueError(
                f"Genphase MegaMoE gather runs must divide evenly across {self.gather_issuing_units} issuing "
                f"units: tile N {self.cta_tile_n} must be a multiple of {required_multiple}."
            )
        self.gather_b_runs_per_unit = self.gather_b_rows_per_cta // run_span_rows
        self.gather_b_rows_per_unit = self.gather_b_runs_per_unit * self.gather_index_load_rows
        # Strides of the padding row ids, so every (CTA, warp, row) gets its own: 32 per warp and tile M per CTA.
        self.gather_padding_warp_stride = max(32, self.gather_b_rows_per_unit)
        self.gather_padding_cta_stride = max(
            self.mma_tiler_mnk[0], self.num_gather_b_warps * self.gather_padding_warp_stride
        )

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

        # Full-tile Gather4 capacity must match FC2's tiled TMA layout. Compare
        # independently derived geometry and SMEM byte counts before allowing
        # FC1 to subtract omitted padding bytes from its stage expectation.
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
            if not all(isinstance(dimension, int) and not isinstance(dimension, bool) for dimension in dimensions):
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
            raise NotImplementedError("Rubin Genphase MegaMoE does not implement M/N instruction repetition.")
        # The additional decode geometry is deliberately bounded: tile N 64 with the K that
        # makes one B row a 128-byte swizzle atom (128 for 8-bit tokens, 256 for 4-bit),
        # two-CTA MMA, cluster 4x1 (two gather warps, B multicast across both CTA pairs). The
        # (2, 1) decode cluster keeps the N128 / N256 tiles.
        if self.mma_tiler_mnk[1] == 64 and self.cluster_shape_mn != (4, 1):
            raise ValueError(f"GenPhase N64 runs on 4x1 clusters, got {self.cluster_shape_mn}.")
        if self.mma_tiler_mnk[1] == 64 and not (self.mma_tiler_mnk == self.n64_decode_tile and self.use_2cta_instrs):
            raise ValueError(
                f"GenPhase N64 requires tile {self.n64_decode_tile} for {self.quant_kind.value} with two-CTA MMA."
            )
        if self.mma_tiler_mnk[1] not in (64, 128, 256):
            raise ValueError("Rubin Genphase MegaMoE supports tile N in (64, 128, 256).")
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
        if self.shared_enabled:
            for name, value in (
                ("shared_hidden_size", self.shared_hidden_size),
                ("shared_intermediate_gateup_size", self.shared_intermediate_gateup_size),
                ("shared_output_size", self.shared_output_size),
            ):
                if type(value) is not int or value <= 0:
                    raise ValueError(f"{name} must be a positive static int, got {value!r}.")
            sf_atom = self.shared_quant_kind.sf_vec_size * 4
            if self.shared_hidden_size % sf_atom != 0:
                raise ValueError(f"shared_hidden_size must be a multiple of {sf_atom} (four scale-factor vectors).")
            if self.shared_intermediate_gateup_size % 2 != 0:
                raise ValueError("shared_intermediate_gateup_size must be even.")
            if (self.shared_intermediate_gateup_size // 2) % sf_atom != 0:
                raise ValueError(
                    f"shared_intermediate_gateup_size // 2 must be a multiple of {sf_atom} (four scale-factor vectors)."
                )

    def _check_shared_tile_rows(self) -> None:
        """Tile K keeps one 128-byte row for the shared kind too, so A/B stages coincide with the gathered routed ones."""
        for operand_name, alloc_dtype in (("A", self.shared_a_smem_alloc_dtype), ("B", self.shared_b_smem_alloc_dtype)):
            row_bytes = self.shared_mma_tile_k * int(alloc_dtype.width) // 8
            if row_bytes != self.gather_atom_bytes:
                raise ValueError(
                    f"shared_quant_kind {self.shared_quant_kind}: one {operand_name} row occupies {row_bytes} bytes at "
                    f"tile K {self.shared_mma_tile_k}, but the shared tiles use the routed stages and need "
                    f"{self.gather_atom_bytes}."
                )

    def register_smem_regions(self, smem_workspace: SmemWorkspace) -> None:
        """Register the selected Rubin mainloop SMEM plan."""
        # Under the shared plan this one array serves all four operands. Under the
        # split plan it becomes the weight side's and the token side gets its own,
        # because the two depths no longer index the same stage.
        smem_workspace.register_mbarrier(self.ab_pipeline_mbarriers_region, self.num_weight_ab_stages * 2)
        if self.uses_asymmetric_ab_stages:
            smem_workspace.register_mbarrier(self.token_pipeline_mbarriers_region, self.num_token_ab_stages * 2)

        # Heterogeneous shared expert: routed and shared regions are two views (overlay lifetimes) of one ring.
        self._register_ab_regions(smem_workspace)
        smem_workspace.register_mbarrier(self.acc_pipeline_mbarriers_region, self.num_accumulator_pipeline_stages * 2)
        smem_workspace.register_mbarrier(self.tmem_deallocation_mbarrier_region, 1)
        smem_workspace.register_tensor(self.tmem_holding_buffer_region, cutlass.Int32, (1,), byte_alignment=8)

    @property
    def n64_decode_tile(self) -> Tuple[int, int, int]:
        """The one tile-N-64 geometry: K sized so one token row is a 128-byte swizzle atom."""
        return (256, 64, 128 * 8 // int(self.b_dtype.width))

    def _select_ab_stage_plan(self) -> None:
        stage_one_regions = make_smem_layouts(
            self.mma_instruction, self.mma_tiler_mnk, 1, (self.a_major_mode, self.b_major_mode), self.architecture
        )
        independent_sfb_region = make_smem_layouts(
            self.sfb_mma_instruction, self.mma_tiler_sfb, 1, (self.a_major_mode, self.b_major_mode), self.architecture
        )[3]
        if (
            independent_sfb_region.shape != stage_one_regions[3].shape
            or independent_sfb_region.stride != stage_one_regions[3].stride
            or independent_sfb_region.nbytes != stage_one_regions[3].nbytes
        ):
            raise ValueError("The Rubin independent SFB plan does not match the primary MMA SFB operand.")

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
        (self.a_stage_tma_bytes, self.b_stage_tma_bytes, self.sfa_stage_tma_bytes, self.sfb_stage_tma_bytes) = tuple(
            stage_bytes * self.mma_cta_count for stage_bytes in stage_operand_tma_bytes
        )
        mbarrier_bytes = int(cutlass.Int64.width) // 8
        plan_tail_bytes = (2 * self.num_accumulator_pipeline_stages + 2) * mbarrier_bytes
        # Each pipeline spends two mbarriers per stage, full and empty. The shared
        # plan pays that once for all four operands; the split plan pays it twice,
        # which is the only overhead the split introduces.
        a_stage_bytes = stage_one_regions[0].nbytes
        b_stage_bytes = stage_one_regions[1].nbytes
        sfa_stage_bytes = stage_one_regions[2].nbytes
        sfb_stage_bytes = stage_one_regions[3].nbytes
        shared_stage_one_regions = None
        if self.heterogeneous_shared:
            shared_stage_one_regions = make_smem_layouts(
                self.shared_mma_instruction,
                self.shared_mma_tiler_mnk,
                1,
                (self.a_major_mode, self.b_major_mode),
                self.architecture,
            )
            # One ring serves both kinds: every stage is as wide as the larger kind's, each kind keeps its own swizzle.
            a_stage_bytes = max(a_stage_bytes, shared_stage_one_regions[0].nbytes)
            b_stage_bytes = max(b_stage_bytes, shared_stage_one_regions[1].nbytes)
            shared_stage_operand_tma_bytes = (
                packed_bytes(shared_stage_one_regions[0], self.shared_a_dtype),
                packed_bytes(shared_stage_one_regions[1], self.shared_b_dtype),
                shared_stage_one_regions[2].nbytes,
                shared_stage_one_regions[3].nbytes,
            )
            self.shared_ab_stage_tma_bytes = sum(shared_stage_operand_tma_bytes)
            (
                self.shared_a_stage_tma_bytes,
                self.shared_b_stage_tma_bytes,
                self.shared_sfa_stage_tma_bytes,
                self.shared_sfb_stage_tma_bytes,
            ) = tuple(stage_bytes * self.mma_cta_count for stage_bytes in shared_stage_operand_tma_bytes)
            sfa_stage_bytes = max(sfa_stage_bytes, shared_stage_one_regions[2].nbytes)
            sfb_stage_bytes = max(sfb_stage_bytes, shared_stage_one_regions[3].nbytes)
        weight_stage_cost_bytes = a_stage_bytes + sfa_stage_bytes + 2 * mbarrier_bytes
        token_stage_cost_bytes = b_stage_bytes + sfb_stage_bytes + 2 * mbarrier_bytes

        def plan_stages() -> Tuple[int, int]:
            stage_budget_bytes = self.mainloop_smem_budget_bytes - plan_tail_bytes
            shared_stages = stage_budget_bytes // (weight_stage_cost_bytes + token_stage_cost_bytes)
            if shared_stages < 1:
                raise ValueError(
                    f"One AB stage needs {weight_stage_cost_bytes + token_stage_cost_bytes + plan_tail_bytes} "
                    f"bytes, exceeding the {self.mainloop_smem_budget_bytes}-byte mainloop budget."
                )
            if not self.uses_asymmetric_ab_stages:
                return shared_stages, shared_stages
            token_stages = self.asymmetric_token_stages
            weight_stages = (stage_budget_bytes - token_stages * token_stage_cost_bytes) // weight_stage_cost_bytes
            if weight_stages <= shared_stages:
                if not self.shared_enabled:
                    # Routed-only builds keep rejecting a split that buys no depth, so the solver picks another kernel.
                    if weight_stages < 1:
                        raise ValueError(
                            f"{token_stages} token stages leave no room for a weight stage in the "
                            f"{self.mainloop_smem_budget_bytes}-byte mainloop budget."
                        )
                    raise ValueError(
                        f"Asymmetric stages give the weight side {weight_stages} stages, no more than the "
                        f"{shared_stages} a shared plan reaches; the token depth of {token_stages} is too deep "
                        f"to be worth splitting."
                    )
                # A shared build can void a tuned split (wider stages, more epilogue scratch): degrade to the shared plan.
                self.uses_asymmetric_ab_stages = False
                return shared_stages, shared_stages
            return weight_stages, token_stages

        self.num_weight_ab_stages, self.num_token_ab_stages = plan_stages()
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
        self.shared_a_smem_region = None
        self.shared_b_smem_region = None
        self.shared_sfa_smem_region = None
        self.shared_sfb_smem_region = None
        if self.heterogeneous_shared:
            (self.shared_a_smem_region, _, self.shared_sfa_smem_region, _) = make_smem_layouts(
                self.shared_mma_instruction,
                self.shared_mma_tiler_mnk,
                self.num_weight_ab_stages,
                (self.a_major_mode, self.b_major_mode),
                self.architecture,
            )
            (_, self.shared_b_smem_region, _, self.shared_sfb_smem_region) = make_smem_layouts(
                self.shared_mma_instruction,
                self.shared_mma_tiler_mnk,
                self.num_token_ab_stages,
                (self.a_major_mode, self.b_major_mode),
                self.architecture,
            )
            for operand in ("a", "b", "sfa", "sfb"):
                routed_region, shared_region = self._align_stage_stride(
                    getattr(self, f"{operand}_smem_region"), getattr(self, f"shared_{operand}_smem_region")
                )
                self._check_overlay_stage_stride(operand.upper(), routed_region, shared_region)
                setattr(self, f"{operand}_smem_region", routed_region)
                setattr(self, f"shared_{operand}_smem_region", shared_region)
        self.selected_smem_bytes = sum(
            max(region.nbytes, 0 if shared_region is None else shared_region.nbytes)
            for region, shared_region in (
                (self.a_smem_region, self.shared_a_smem_region),
                (self.b_smem_region, self.shared_b_smem_region),
                (self.sfa_smem_region, self.shared_sfa_smem_region),
                (self.sfb_smem_region, self.shared_sfb_smem_region),
            )
        )

    def make_tiled_mma(self, *, shared: bool = False) -> cute.TiledMma:
        """Create a context-local SM107 block-scaled MMA object (the shared expert's kind with ``shared``)."""
        cta_group = tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE
        if shared:
            return sm107_utils.make_blockscaled_trivial_tiled_mma(
                self.shared_a_dtype,
                self.shared_b_dtype,
                self.a_major_mode,
                self.b_major_mode,
                self.shared_sf_dtype,
                self.shared_sf_vec_size,
                cta_group,
                self.shared_mma_instruction_mnk,
            )
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
        self.sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
            self.tiled_mma, self.mma_tiler_mnk, self.sf_vec_size, self.num_weight_ab_stages
        )
        self.sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
            self.tiled_mma, self.mma_tiler_mnk, self.sf_vec_size, self.num_token_ab_stages
        )

        self.shared_tiled_mma = self.tiled_mma
        self.shared_tiled_mma_sfb = self.tiled_mma_sfb
        self.shared_a_smem_composed_layout_staged = self.a_smem_composed_layout_staged
        self.shared_b_smem_composed_layout_staged = self.b_smem_composed_layout_staged
        self.shared_sfa_smem_layout_staged = self.sfa_smem_layout_staged
        self.shared_sfb_smem_layout_staged = self.sfb_smem_layout_staged
        if self.heterogeneous_shared:
            self.shared_tiled_mma = self.make_tiled_mma(shared=True)
            self.shared_tiled_mma_sfb = sm107_utils.make_blockscaled_trivial_tiled_mma(
                self.shared_a_dtype,
                self.shared_b_dtype,
                self.a_major_mode,
                self.b_major_mode,
                self.shared_sf_dtype,
                self.shared_sf_vec_size,
                tcgen05.CtaGroup.ONE,
                self.shared_sfb_instruction_shape_mnk,
            )
            if self.shared_tiled_mma.thr_id.shape != self.tiled_mma.thr_id.shape:
                raise ValueError("The shared expert's tiled MMA must span the same CTA group as the routed one.")
            self.shared_a_smem_composed_layout_staged = sm100_utils.make_smem_layout_a(
                self.shared_tiled_mma,
                self.shared_mma_tiler_mnk,
                self.shared_a_smem_alloc_dtype,
                self.num_weight_ab_stages,
            )
            self.shared_b_smem_composed_layout_staged = sm100_utils.make_smem_layout_b(
                self.shared_tiled_mma,
                self.shared_mma_tiler_mnk,
                self.shared_b_smem_alloc_dtype,
                self.num_token_ab_stages,
            )
            self.shared_sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
                self.shared_tiled_mma, self.shared_mma_tiler_mnk, self.shared_sf_vec_size, self.num_weight_ab_stages
            )
            self.shared_sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
                self.shared_tiled_mma, self.shared_mma_tiler_mnk, self.shared_sf_vec_size, self.num_token_ab_stages
            )
            for layout_name, routed_dtype, shared_dtype in (
                ("a_smem_composed_layout_staged", self.a_smem_alloc_dtype, self.shared_a_smem_alloc_dtype),
                ("b_smem_composed_layout_staged", self.b_smem_alloc_dtype, self.shared_b_smem_alloc_dtype),
                ("sfa_smem_layout_staged", self.sf_dtype, self.shared_sf_dtype),
                ("sfb_smem_layout_staged", self.sf_dtype, self.shared_sf_dtype),
            ):
                routed_layout, shared_layout = self._align_layout_stage_stride(
                    getattr(self, layout_name), routed_dtype, getattr(self, "shared_" + layout_name), shared_dtype
                )
                setattr(self, layout_name, routed_layout)
                setattr(self, "shared_" + layout_name, shared_layout)
            self._check_smem_plan(
                (
                    ("shared A", self.shared_a_smem_composed_layout_staged.outer, self.shared_a_smem_region),
                    ("shared B", self.shared_b_smem_composed_layout_staged.outer, self.shared_b_smem_region),
                    ("shared SFA", self.shared_sfa_smem_layout_staged, self.shared_sfa_smem_region),
                    ("shared SFB", self.shared_sfb_smem_layout_staged, self.shared_sfb_smem_region),
                )
            )
        self.a_smem_layout_staged = self.a_smem_composed_layout_staged.outer
        self.b_smem_layout_staged = self.b_smem_composed_layout_staged.outer
        self._check_smem_plan(
            (
                ("A", self.a_smem_layout_staged, self.a_smem_region),
                ("B", self.b_smem_layout_staged, self.b_smem_region),
                ("SFA", self.sfa_smem_layout_staged, self.sfa_smem_region),
                ("SFB", self.sfb_smem_layout_staged, self.sfb_smem_region),
            )
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

        a_operation = sm100_utils.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, self.tiled_mma.thr_id)
        b_operation = sm100_utils.cluster_shape_to_tma_atom_B(self.cluster_shape_mn, self.tiled_mma.thr_id)
        sfa_operation = sm100_utils.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, self.tiled_mma.thr_id)
        sfb_operation = sm100_utils.cluster_shape_to_tma_atom_SFB(self.cluster_shape_mn, self.tiled_mma.thr_id)
        a_internal_type = self.a_smem_alloc_dtype if self.a_smem_alloc_dtype is not self.a_dtype else None
        b_internal_type = self.b_smem_alloc_dtype if self.b_smem_alloc_dtype is not self.b_dtype else None

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
        gather_b_smem_layout = cute.make_composed_layout(cute.make_swizzle(3, 4, 3), 0, gather_b_base)
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

    def prepare_shared_tma_load_params(
        self,
        *,
        shared_fc1_a: cute.Tensor,
        shared_fc1_b: cute.Tensor,
        shared_fc1_sfa: cute.Tensor,
        shared_fc1_sfb: cute.Tensor,
        shared_fc2_a: cute.Tensor,
        shared_fc2_b: cute.Tensor,
        shared_fc2_sfa: cute.Tensor,
        shared_fc2_sfb: cute.Tensor,
    ) -> Tuple:
        """Shared expert TMA tensor/atom pairs (FC1 A, SFA, B, SFB, then FC2 likewise); all plain tiled legs."""
        if not self.shared_enabled:
            raise ValueError("prepare_shared_tma_load_params requires the shared expert to be configured.")
        tiled_mma = self.shared_tiled_mma
        a_stage_layout = cute.slice_(self.shared_a_smem_composed_layout_staged, (None, None, None, 0))
        b_stage_layout = cute.slice_(self.shared_b_smem_composed_layout_staged, (None, None, None, 0))
        sfa_stage_layout = cute.slice_(self.shared_sfa_smem_layout_staged, (None, None, None, 0))
        sfb_stage_layout = cute.slice_(self.shared_sfb_smem_layout_staged, (None, None, None, 0))

        a_operation = sm100_utils.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma.thr_id)
        b_operation = sm100_utils.cluster_shape_to_tma_atom_B(self.cluster_shape_mn, tiled_mma.thr_id)
        sfa_operation = sm100_utils.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma.thr_id)
        sfb_operation = sm100_utils.cluster_shape_to_tma_atom_SFB(self.cluster_shape_mn, tiled_mma.thr_id)
        a_internal_type = (
            self.shared_a_smem_alloc_dtype if self.shared_a_smem_alloc_dtype is not self.shared_a_dtype else None
        )
        b_internal_type = (
            self.shared_b_smem_alloc_dtype if self.shared_b_smem_alloc_dtype is not self.shared_b_dtype else None
        )

        results = []
        for operand_a, operand_sfa, operand_b, operand_sfb in (
            (shared_fc1_a, shared_fc1_sfa, shared_fc1_b, shared_fc1_sfb),
            (shared_fc2_a, shared_fc2_sfa, shared_fc2_b, shared_fc2_sfb),
        ):
            a_atom, a_tensor = cute.nvgpu.make_tiled_tma_atom_A(
                a_operation,
                operand_a,
                a_stage_layout,
                self.shared_mma_tiler_mnk,
                tiled_mma,
                self.cluster_layout_vmnk.shape,
                internal_type=a_internal_type,
            )
            sfa_atom, sfa_tensor = cute.nvgpu.make_tiled_tma_atom_A(
                sfa_operation,
                operand_sfa,
                sfa_stage_layout,
                self.shared_mma_tiler_mnk,
                tiled_mma,
                self.cluster_layout_vmnk.shape,
                internal_type=cutlass.Uint64,
            )
            b_atom, b_tensor = cute.nvgpu.make_tiled_tma_atom_B(
                b_operation,
                operand_b,
                b_stage_layout,
                self.shared_mma_tiler_mnk,
                tiled_mma,
                self.cluster_layout_vmnk.shape,
                internal_type=b_internal_type,
            )
            sfb_atom, sfb_tensor = cute.nvgpu.make_tiled_tma_atom_B(
                sfb_operation,
                operand_sfb,
                sfb_stage_layout,
                self.shared_mma_tiler_sfb,
                self.shared_tiled_mma_sfb,
                self.cluster_layout_sfb_vmnk.shape,
                internal_type=cutlass.Uint64,
            )
            results.extend((a_tensor, a_atom, sfa_tensor, sfa_atom, b_tensor, b_atom, sfb_tensor, sfb_atom))
        return tuple(results)

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
        self.shared_a_smem_tensor = self.a_smem_tensor
        self.shared_b_smem_tensor = self.b_smem_tensor
        self.shared_sfa_smem_tensor = self.sfa_smem_tensor
        self.shared_sfb_smem_tensor = self.sfb_smem_tensor
        if cutlass.const_expr(self.heterogeneous_shared):
            self.shared_a_smem_tensor = smem_workspace.tensor(self.shared_a_smem_tensor_region, smem_base)
            self.shared_b_smem_tensor = smem_workspace.tensor(self.shared_b_smem_tensor_region, smem_base)
            self.shared_sfa_smem_tensor = smem_workspace.tensor(self.shared_sfa_smem_tensor_region, smem_base)
            self.shared_sfb_smem_tensor = smem_workspace.tensor(self.shared_sfb_smem_tensor_region, smem_base)
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
        self._set_shared_k_tile_counts()
        self.sfb_vmnk_coord = (0, cta_coord_in_cluster[0], cta_coord_in_cluster[1], cta_coord_in_cluster[2])

    @cute.jit
    def create_ab_pipeline(self, smem_workspace: SmemWorkspace, smem_base: cute.Pointer) -> pipeline.PipelineAsync:
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
    def create_weight_pipeline(self, smem_workspace: SmemWorkspace, smem_base: cute.Pointer) -> pipeline.PipelineAsync:
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
    def create_token_pipeline(self, smem_workspace: SmemWorkspace, smem_base: cute.Pointer) -> pipeline.PipelineAsync:
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
        consumer_thread_count = self.num_accumulator_consumer_warps_per_cta * 32 * self.mma_cta_count
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
    def create_tmem_allocator(self, smem_workspace: SmemWorkspace, smem_base: cute.Pointer, *, allocator_warp_id: int):
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
            two_cta_tmem_dealloc_mbar_ptr=smem_workspace.ptr(self.tmem_deallocation_mbarrier_region, smem_base),
            arch=self.architecture,
        )

    def _s2t_copy_and_partition(self, smem_tensor: cute.Tensor, tmem_tensor: cute.Tensor, sf_dtype=None):
        compact_smem_tensor = cute.filter_zeros(smem_tensor)
        compact_tmem_tensor = cute.filter_zeros(tmem_tensor)
        cta_group = tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE
        copy_atom = cute.make_copy_atom(
            tcgen05.Cp4x32x128bOp(cta_group), self.sf_dtype if sf_dtype is None else sf_dtype
        )
        tiled_copy = tcgen05.make_s2t_copy(copy_atom, compact_tmem_tensor)
        thread_copy = tiled_copy.get_slice(0)

        mn_mode = cute.get(compact_smem_tensor.layout, mode=[0, 0])
        mn_mode = cute.append(mn_mode, cute.make_layout((4,), stride=(0,)))
        broadcast_layout = cute.append(cute.group_modes(mn_mode, 0), cute.get(compact_smem_tensor.layout, mode=[0, 1]))
        broadcast_layout = cute.append(
            cute.group_modes(broadcast_layout, 0), cute.get(compact_smem_tensor.layout, mode=[1])
        )
        broadcast_layout = cute.append(broadcast_layout, cute.get(compact_smem_tensor.layout, mode=[2]))
        broadcast_layout = cute.append(broadcast_layout, cute.get(compact_smem_tensor.layout, mode=[3]))
        broadcast_smem_tensor = cute.make_tensor(compact_smem_tensor.iterator, broadcast_layout)

        partitioned_smem = thread_copy.partition_S(broadcast_smem_tensor)
        partitioned_smem = tcgen05.get_s2t_smem_desc_tensor(tiled_copy, partitioned_smem)
        partitioned_tmem = thread_copy.partition_D(compact_tmem_tensor)
        return tiled_copy, partitioned_smem, partitioned_tmem

    @staticmethod
    def _check_same_tma_type(role: str, atoms) -> None:
        """Atoms a warp value-selects between must have one MLIR type (box, CTA group, tensor-map basis)."""
        types = [atom._trait.value.type for atom in atoms if atom is not None]
        if not types:
            raise ValueError(f"No {role} tensor map.")
        if any(atom_type != types[0] for atom_type in types[1:]):
            raise ValueError(f"{role} TMA atoms differ in type: {types[0]} vs {[t for t in types if t != types[0]][0]}.")

    def _check_phase_tma_types(self, role: str, routed_atoms, shared_atoms) -> None:
        """One select group with equal kinds; separate routed and shared groups otherwise."""
        if self.heterogeneous_shared:
            self._check_same_tma_type(role, routed_atoms)
            self._check_same_tma_type(f"shared {role}", shared_atoms)
        else:
            self._check_same_tma_type(role, (*routed_atoms, *shared_atoms))

    @cute.jit
    def _select_by_phase(self, phase: Int32, fc1_value, fc2_value, shared_fc1_value=None, shared_fc2_value=None):
        """Pick a per-phase value (k count, executable TMA atom); shared values only when given."""
        value = fc2_value
        if phase == Int32(BlockPhase.Linear1):
            value = fc1_value
        if cutlass.const_expr(shared_fc1_value is not None):
            if phase == Int32(BlockPhase.SharedLinear1):
                value = shared_fc1_value
            if phase == Int32(BlockPhase.SharedLinear2):
                value = shared_fc2_value
        return value

    @cute.jit
    def _select_shared_phase(self, phase: Int32, shared_fc1_value, shared_fc2_value):
        value = shared_fc2_value
        if phase == Int32(BlockPhase.SharedLinear1):
            value = shared_fc1_value
        return value

    @cute.jit
    def _is_fc1_gemm(self, phase: Int32) -> cutlass.Boolean:
        """FC1 of either expert (the routed or the shared one)."""
        is_fc1 = phase == Int32(BlockPhase.Linear1)
        if cutlass.const_expr(self.shared_enabled):
            is_fc1 = is_fc1 | (phase == Int32(BlockPhase.SharedLinear1))
        return is_fc1

    @cute.jit
    def _issue_tma_a_tile(
        self,
        kind: tuple,
        a_exec_atom: cute.CopyAtom,
        sfa_exec_atom: cute.CopyAtom,
        a_cta_layout: cute.Layout,
        multicast_mask,
        work_tile,
        k_tile_count: Int32,
        ab_pipeline: pipeline.PipelineAsync,
        ab_pipeline_state: pipeline.PipelineState,
        expected_tx: Optional[Int32] = None,
        progress_mbarrier=None,
        claim_gate_k_tile: Optional[Int32] = None,
    ) -> pipeline.PipelineState:
        """Load one tile's A/SFA k tiles; ``kind`` bundles MMA, tiler, template views, SMEM tensors and extension.

        With ``claim_gate_k_tile``, TMA-A arrives on ``progress_mbarrier`` (the scheduler warp's claim gate) right
        before it loads that k tile of the unit."""
        (
            tiled_mma,
            mma_tiler_mnk,
            template_a_tensor,
            template_a_atom,
            template_sfa_tensor,
            template_sfa_atom,
            a_smem_tensor,
            sfa_smem_tensor,
            kernel_extension,
        ) = kind
        thread_mma = tiled_mma.get_slice(self.mma_cta_index)
        real_a, _ = kernel_extension.get_gmem_tensor("a", template_a_tensor, work_tile)
        real_sfa, _ = kernel_extension.get_gmem_tensor("sfa", template_sfa_tensor, work_tile)
        global_a = cute.local_tile(real_a, cute.slice_(mma_tiler_mnk, (None, 0, None)), (None, None, None))
        global_sfa = cute.local_tile(real_sfa, cute.slice_(mma_tiler_mnk, (None, 0, None)), (None, None, None))
        partitioned_global_a = thread_mma.partition_A(global_a)
        partitioned_global_sfa = thread_mma.partition_A(global_sfa)
        partitioned_smem_a, partitioned_global_a = cpasync.tma_partition(
            template_a_atom,
            self.main_vmnk_coord[2],
            a_cta_layout,
            cute.group_modes(a_smem_tensor, 0, 3),
            cute.group_modes(partitioned_global_a, 0, 3),
        )
        partitioned_smem_sfa, partitioned_global_sfa = cpasync.tma_partition(
            template_sfa_atom,
            self.main_vmnk_coord[2],
            a_cta_layout,
            cute.group_modes(sfa_smem_tensor, 0, 3),
            cute.group_modes(partitioned_global_sfa, 0, 3),
        )
        partitioned_smem_sfa = cute.filter_zeros(partitioned_smem_sfa)
        partitioned_global_sfa = cute.filter_zeros(partitioned_global_sfa)

        mma_tile_m = kernel_extension.local_tile_m(work_tile) // self.mma_cta_count
        global_a_slice = partitioned_global_a[(None, mma_tile_m, None, 0)]
        global_sfa_slice = partitioned_global_sfa[(None, mma_tile_m, None, 0)]

        empty_status = ab_pipeline.producer_try_acquire(ab_pipeline_state)
        for k_tile_idx in cutlass.range(k_tile_count, unroll=1):
            if cutlass.const_expr(progress_mbarrier is not None and claim_gate_k_tile is not None):
                if k_tile_idx == claim_gate_k_tile:
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive(progress_mbarrier)
            ab_pipeline.producer_acquire(ab_pipeline_state, empty_status, expected_tx=expected_tx)
            tma_barrier = ab_pipeline.producer_get_barrier(ab_pipeline_state)
            stage_index = ab_pipeline_state.index
            ab_pipeline_state.advance()
            if k_tile_idx + 1 < k_tile_count:
                empty_status = ab_pipeline.producer_try_acquire(ab_pipeline_state)
            cute.copy(
                _bind_tma_issue(a_exec_atom, tma_barrier, multicast_mask),
                global_a_slice[(None, k_tile_idx)],
                partitioned_smem_a[(None, stage_index)],
            )
            cute.copy(
                _bind_tma_issue(sfa_exec_atom, tma_barrier, multicast_mask),
                global_sfa_slice[(None, k_tile_idx)],
                partitioned_smem_sfa[(None, stage_index)],
            )
        return ab_pipeline_state

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
        trigger_dependents_when_drained: bool = False,
        progress_mbarrier=None,
        late_claim_gate_k_tiles: Optional[Tuple[int, int]] = None,
        shared_fc1_tma_a_tensor: Optional[cute.Tensor] = None,
        shared_fc1_tma_a_atom: Optional[cute.CopyAtom] = None,
        shared_fc1_tma_sfa_tensor: Optional[cute.Tensor] = None,
        shared_fc1_tma_sfa_atom: Optional[cute.CopyAtom] = None,
        shared_fc2_tma_a_atom: Optional[cute.CopyAtom] = None,
        shared_fc2_tma_sfa_atom: Optional[cute.CopyAtom] = None,
        shared_kernel_extension: Optional[BlockScaledSwapAbFc12Extension] = None,
    ) -> None:
        """Run the TMA-A warp for every phase; the tile's phase selects the executable atoms."""
        self._check_phase_tma_types(
            "A", (fc1_tma_a_atom, fc2_tma_a_atom), (shared_fc1_tma_a_atom, shared_fc2_tma_a_atom)
        )
        self._check_phase_tma_types(
            "SFA", (fc1_tma_sfa_atom, fc2_tma_sfa_atom), (shared_fc1_tma_sfa_atom, shared_fc2_tma_sfa_atom)
        )
        if cutlass.const_expr(self.heterogeneous_shared):
            if cutlass.const_expr(shared_kernel_extension is None):
                raise ValueError("A heterogeneous shared expert needs the shared kernel extension.")
        multicast_mask = None
        if cutlass.const_expr(self.is_a_mcast or self.use_2cta_instrs):
            multicast_mask = tma_multicast_mask(
                self.cluster_shape_mn, None, self.cta_coord_in_cluster, None, self.use_2cta_instrs, "a"
            )
        a_cta_layout = cute.make_layout(cute.slice_(self.cluster_layout_vmnk, (0, 0, None, 0)).shape)

        fc1_a_exec_atom = make_executable_tma_atom_or_pair(fc1_tma_a_atom)
        fc1_sfa_exec_atom = make_executable_tma_atom_or_pair(fc1_tma_sfa_atom)
        fc2_a_exec_atom = make_executable_tma_atom_or_pair(fc2_tma_a_atom)
        fc2_sfa_exec_atom = make_executable_tma_atom_or_pair(fc2_tma_sfa_atom)
        shared_fc1_a_exec_atom = None
        shared_fc1_sfa_exec_atom = None
        shared_fc2_a_exec_atom = None
        shared_fc2_sfa_exec_atom = None
        if cutlass.const_expr(self.shared_enabled):
            shared_fc1_a_exec_atom = make_executable_tma_atom_or_pair(shared_fc1_tma_a_atom)
            shared_fc1_sfa_exec_atom = make_executable_tma_atom_or_pair(shared_fc1_tma_sfa_atom)
            shared_fc2_a_exec_atom = make_executable_tma_atom_or_pair(shared_fc2_tma_a_atom)
            shared_fc2_sfa_exec_atom = make_executable_tma_atom_or_pair(shared_fc2_tma_sfa_atom)
        routed_kind = (
            self.tiled_mma,
            self.mma_tiler_mnk,
            fc1_tma_a_tensor,
            fc1_tma_a_atom,
            fc1_tma_sfa_tensor,
            fc1_tma_sfa_atom,
            self.a_smem_tensor,
            self.sfa_smem_tensor,
            kernel_extension,
        )
        routed_shared_a = None if self.heterogeneous_shared else shared_fc1_a_exec_atom
        routed_shared_a2 = None if self.heterogeneous_shared else shared_fc2_a_exec_atom
        routed_shared_sfa = None if self.heterogeneous_shared else shared_fc1_sfa_exec_atom
        routed_shared_sfa2 = None if self.heterogeneous_shared else shared_fc2_sfa_exec_atom

        # The claim gate: TMA-A arrives on progress_mbarrier once per unit. By default right after it takes the unit
        # (Done included). With late_claim_gate_k_tiles = (FC1, FC2), an FC1 / FC2 unit arrives right before it loads
        # the k tile that many k tiles before its end (0: before its first k tile), and Done does not arrive.
        late_claim_gate = progress_mbarrier is not None and late_claim_gate_k_tiles is not None
        work_tile = sched_consumer.consume_work()
        if cutlass.const_expr(progress_mbarrier is not None and not late_claim_gate):
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive(progress_mbarrier)
        while work_tile.is_valid_tile:
            phase = work_tile.phase
            # The fixed FC1 / FC2 range names the iket summaries key on (shared tiles count with their GEMM).
            if self._is_fc1_gemm(phase):
                iket.range_push("tma_weight_fc1")
            else:
                iket.range_push("tma_weight_fc2")
            k_tile_count = self._phase_k_tile_count(phase)
            claim_gate_k_tile = None
            if cutlass.const_expr(late_claim_gate):
                claim_gate_k_tile = Int32(0)
                k_tiles_before_end = Int32(late_claim_gate_k_tiles[1])
                if self._is_fc1_gemm(phase):
                    k_tiles_before_end = Int32(late_claim_gate_k_tiles[0])
                if k_tiles_before_end > Int32(0):
                    claim_gate_k_tile = cutlass.max(k_tile_count - k_tiles_before_end, Int32(0))
            if cutlass.const_expr(self.heterogeneous_shared):
                if self._is_shared_phase(phase):
                    shared_kind = (
                        self.shared_tiled_mma,
                        self.shared_mma_tiler_mnk,
                        shared_fc1_tma_a_tensor,
                        shared_fc1_tma_a_atom,
                        shared_fc1_tma_sfa_tensor,
                        shared_fc1_tma_sfa_atom,
                        self.shared_a_smem_tensor,
                        self.shared_sfa_smem_tensor,
                        shared_kernel_extension,
                    )
                    ab_pipeline_state = self._issue_tma_a_tile(
                        shared_kind,
                        self._select_shared_phase(phase, shared_fc1_a_exec_atom, shared_fc2_a_exec_atom),
                        self._select_shared_phase(phase, shared_fc1_sfa_exec_atom, shared_fc2_sfa_exec_atom),
                        a_cta_layout,
                        multicast_mask,
                        work_tile,
                        k_tile_count,
                        ab_pipeline,
                        ab_pipeline_state,
                        Int32(self.shared_weight_tx_count),
                        progress_mbarrier=progress_mbarrier if late_claim_gate else None,
                        claim_gate_k_tile=claim_gate_k_tile,
                    )
                else:
                    ab_pipeline_state = self._issue_tma_a_tile(
                        routed_kind,
                        self._select_by_phase(phase, fc1_a_exec_atom, fc2_a_exec_atom),
                        self._select_by_phase(phase, fc1_sfa_exec_atom, fc2_sfa_exec_atom),
                        a_cta_layout,
                        multicast_mask,
                        work_tile,
                        k_tile_count,
                        ab_pipeline,
                        ab_pipeline_state,
                        progress_mbarrier=progress_mbarrier if late_claim_gate else None,
                        claim_gate_k_tile=claim_gate_k_tile,
                    )
            else:
                ab_pipeline_state = self._issue_tma_a_tile(
                    routed_kind,
                    self._select_by_phase(phase, fc1_a_exec_atom, fc2_a_exec_atom, routed_shared_a, routed_shared_a2),
                    self._select_by_phase(
                        phase, fc1_sfa_exec_atom, fc2_sfa_exec_atom, routed_shared_sfa, routed_shared_sfa2
                    ),
                    a_cta_layout,
                    multicast_mask,
                    work_tile,
                    k_tile_count,
                    ab_pipeline,
                    ab_pipeline_state,
                    progress_mbarrier=progress_mbarrier if late_claim_gate else None,
                    claim_gate_k_tile=claim_gate_k_tile,
                )
            iket.range_pop()
            work_tile = sched_consumer.consume_work()
            if cutlass.const_expr(progress_mbarrier is not None and not late_claim_gate):
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive(progress_mbarrier)

        if cutlass.const_expr(trigger_dependents_when_drained):
            # No work is left to claim: this CTA's last tile is already in flight, so
            # the polling top-k reduce may become resident while it drains.
            cute.arch.griddepcontrol_launch_dependents()
        ab_pipeline.producer_tail(ab_pipeline_state)

    @cute.jit
    def _issue_tma_b_tile(
        self,
        kind: tuple,
        b_exec_atom: cute.CopyAtom,
        sfb_exec_atom: cute.CopyAtom,
        b_cta_layout: cute.Layout,
        sfb_cta_layout: cute.Layout,
        b_multicast_mask,
        sfb_multicast_mask,
        work_tile,
        k_tile_count: Int32,
        ab_pipeline: pipeline.PipelineAsync,
        ab_pipeline_state: pipeline.PipelineState,
        is_fc1=None,
        expected_tx: Optional[Int32] = None,
    ) -> pipeline.PipelineState:
        """Load one tile's B/SFB k tiles; on routed FC1 tiles Gather-B supplies B and only SFB is issued here."""
        (
            tiled_mma,
            tiled_mma_sfb,
            mma_tiler_mnk,
            mma_tiler_sfb,
            template_b_tensor,
            template_b_atom,
            template_sfb_tensor,
            template_sfb_atom,
            b_smem_tensor,
            sfb_smem_tensor,
            kernel_extension,
        ) = kind
        thread_mma = tiled_mma.get_slice(self.mma_cta_index)
        thread_mma_sfb = tiled_mma_sfb.get_slice(self.mma_cta_index)
        real_b, _ = kernel_extension.get_gmem_tensor("b", template_b_tensor, work_tile)
        real_sfb, _ = kernel_extension.get_gmem_tensor("sfb", template_sfb_tensor, work_tile)
        if cutlass.const_expr(self.use_2cta_instrs):
            if not self.is_leader_cta:
                load_shift = dynamic_mainloop.compute_non_leader_cta_load_shift(
                    valid_tokens_in_tile=work_tile.valid_tokens_in_cta_tile, mma_tiler_n=self.mma_tiler_mnk[1]
                )
                real_b = cute.domain_offset((load_shift, 0, 0), real_b)
        global_b = cute.local_tile(real_b, cute.slice_(mma_tiler_mnk, (0, None, None)), (None, None, None))
        global_sfb = cute.local_tile(real_sfb, cute.slice_(mma_tiler_sfb, (0, None, None)), (None, None, None))
        partitioned_global_b = thread_mma.partition_B(global_b)
        partitioned_global_sfb = thread_mma_sfb.partition_B(global_sfb)
        partitioned_smem_b, partitioned_global_b = cpasync.tma_partition(
            template_b_atom,
            self.main_vmnk_coord[1],
            b_cta_layout,
            cute.group_modes(b_smem_tensor, 0, 3),
            cute.group_modes(partitioned_global_b, 0, 3),
        )
        partitioned_smem_sfb, partitioned_global_sfb = cpasync.tma_partition(
            template_sfb_atom,
            self.sfb_vmnk_coord[1],
            sfb_cta_layout,
            cute.group_modes(sfb_smem_tensor, 0, 3),
            cute.group_modes(partitioned_global_sfb, 0, 3),
        )
        partitioned_smem_sfb = cute.filter_zeros(partitioned_smem_sfb)
        partitioned_global_sfb = cute.filter_zeros(partitioned_global_sfb)
        global_b_slice = partitioned_global_b[(None, work_tile.tile_n_idx, None, 0)]
        # SF tiles are 128 tokens: a 64-token tile reads half of one.
        sfb_tile_n_index = work_tile.tile_n_idx
        if cutlass.const_expr(mma_tiler_mnk[1] == 64):
            sfb_tile_n_index = work_tile.tile_n_idx // Int32(2)
        global_sfb_slice = partitioned_global_sfb[(None, sfb_tile_n_index, None, 0)]

        if cutlass.const_expr(self.dynamic_gather and is_fc1 is not None):
            # Weight/SFA and SFB transfers are unchanged. On routed FC1 tiles TMA-B
            # alone subtracts the B bytes Gather4 omits for padding rows, so the two
            # producer arrivals still register the exact combined transaction count.
            expected_tx = Int32(self.ab_pipeline_tx_count_per_producer)
            if is_fc1:
                omitted_rows = Int32(self.cta_tile_n) - round_up(work_tile.valid_tokens_in_cta_tile, 16)
                expected_tx = expected_tx - omitted_rows * Int32(self.mma_tile_k * int(self.b_dtype.width) // 8)

        empty_status = ab_pipeline.producer_try_acquire(ab_pipeline_state)
        for k_tile_idx in cutlass.range(k_tile_count, unroll=1):
            ab_pipeline.producer_acquire(ab_pipeline_state, empty_status, expected_tx=expected_tx)
            tma_barrier = ab_pipeline.producer_get_barrier(ab_pipeline_state)
            stage_index = ab_pipeline_state.index
            if cutlass.const_expr(is_fc1 is not None):
                if is_fc1:
                    # The stage is acquired and its expectation registered; now release the Gather-B warps.
                    barrier_sync_unaligned(self.fc1_gather_sync_barrier_id, 32 * (1 + self.num_gather_b_warps))
            ab_pipeline_state.advance()
            if k_tile_idx + 1 < k_tile_count:
                empty_status = ab_pipeline.producer_try_acquire(ab_pipeline_state)
            if cutlass.const_expr(is_fc1 is None):
                cute.copy(
                    _bind_tma_issue(b_exec_atom, tma_barrier, b_multicast_mask),
                    global_b_slice[(None, k_tile_idx)],
                    partitioned_smem_b[(None, stage_index)],
                )
            else:
                if not is_fc1:
                    cute.copy(
                        _bind_tma_issue(b_exec_atom, tma_barrier, b_multicast_mask),
                        global_b_slice[(None, k_tile_idx)],
                        partitioned_smem_b[(None, stage_index)],
                    )
            cute.copy(
                _bind_tma_issue(sfb_exec_atom, tma_barrier, sfb_multicast_mask),
                global_sfb_slice[(None, k_tile_idx)],
                partitioned_smem_sfb[(None, stage_index)],
            )
        return ab_pipeline_state

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
        fc1_sf_parts_per_block: int = 1,
        shared_fc1_tma_b_tensor: Optional[cute.Tensor] = None,
        shared_fc1_tma_b_atom: Optional[cute.CopyAtom] = None,
        shared_fc1_tma_sfb_tensor: Optional[cute.Tensor] = None,
        shared_fc1_tma_sfb_atom: Optional[cute.CopyAtom] = None,
        shared_fc2_tma_b_atom: Optional[cute.CopyAtom] = None,
        shared_fc2_tma_sfb_atom: Optional[cute.CopyAtom] = None,
        shared_fc1_done_counter_pointer: Optional[cute.Pointer] = None,
        shared_fc2_spin_threshold: Optional[Int32] = None,
        shared_kernel_extension: Optional[BlockScaledSwapAbFc12Extension] = None,
    ) -> None:
        """Run the TMA-B warp: token data and scale factors for every phase, structured like ``run_tma_a``."""
        self._check_phase_tma_types("B", (fc2_tma_b_atom,), (shared_fc1_tma_b_atom, shared_fc2_tma_b_atom))
        self._check_phase_tma_types(
            "SFB", (fc1_tma_sfb_atom, fc2_tma_sfb_atom), (shared_fc1_tma_sfb_atom, shared_fc2_tma_sfb_atom)
        )
        if cutlass.const_expr(self.heterogeneous_shared):
            if cutlass.const_expr(shared_kernel_extension is None):
                raise ValueError("A heterogeneous shared expert needs the shared kernel extension.")
        b_multicast_mask = None
        sfb_multicast_mask = None
        if cutlass.const_expr(self.is_b_mcast or self.use_2cta_instrs):
            b_multicast_mask = tma_multicast_mask(
                self.cluster_shape_mn, None, self.cta_coord_in_cluster, None, self.use_2cta_instrs, "b"
            )
            sfb_multicast_mask = tma_multicast_mask(
                self.cluster_shape_mn, None, self.cta_coord_in_cluster, None, self.use_2cta_instrs, "sfb"
            )

        b_cta_layout = cute.make_layout(cute.slice_(self.cluster_layout_vmnk, (0, None, 0, 0)).shape)
        sfb_cta_layout = cute.make_layout(cute.slice_(self.cluster_layout_sfb_vmnk, (0, None, 0, 0)).shape)

        fc2_b_exec_atom = make_executable_tma_atom_or_pair(fc2_tma_b_atom)
        fc1_sfb_exec_atom = make_executable_tma_atom_or_pair(fc1_tma_sfb_atom)
        fc2_sfb_exec_atom = make_executable_tma_atom_or_pair(fc2_tma_sfb_atom)
        shared_fc1_b_exec_atom = None
        shared_fc1_sfb_exec_atom = None
        shared_fc2_b_exec_atom = None
        shared_fc2_sfb_exec_atom = None
        if cutlass.const_expr(self.shared_enabled):
            shared_fc1_b_exec_atom = make_executable_tma_atom_or_pair(shared_fc1_tma_b_atom)
            shared_fc1_sfb_exec_atom = make_executable_tma_atom_or_pair(shared_fc1_tma_sfb_atom)
            shared_fc2_b_exec_atom = make_executable_tma_atom_or_pair(shared_fc2_tma_b_atom)
            shared_fc2_sfb_exec_atom = make_executable_tma_atom_or_pair(shared_fc2_tma_sfb_atom)
        routed_kind = (
            self.tiled_mma,
            self.tiled_mma_sfb,
            self.mma_tiler_mnk,
            self.mma_tiler_sfb,
            fc2_tma_b_tensor,
            fc2_tma_b_atom,
            fc1_tma_sfb_tensor,
            fc1_tma_sfb_atom,
            self.b_smem_tensor,
            self.sfb_smem_tensor,
            kernel_extension,
        )
        routed_shared_b = None if self.heterogeneous_shared else shared_fc1_b_exec_atom
        routed_shared_b2 = None if self.heterogeneous_shared else shared_fc2_b_exec_atom
        routed_shared_sfb = None if self.heterogeneous_shared else shared_fc1_sfb_exec_atom
        routed_shared_sfb2 = None if self.heterogeneous_shared else shared_fc2_sfb_exec_atom

        work_tile = sched_consumer.consume_work()
        while work_tile.is_valid_tile:
            phase = work_tile.phase
            is_fc1 = phase == Int32(BlockPhase.Linear1)
            if self._is_fc1_gemm(phase):
                iket.range_push("tma_token_fc1")
            else:
                iket.range_push("tma_token_fc2")
            k_tile_count = self._phase_k_tile_count(phase)

            if is_fc1:
                iket.range_push("tma_b.wait_expert_sf_ready")
            else:
                iket.range_push("tma_token_fc2_wait")
            if is_fc1:
                # Refined scale factors are counted per expert; the peek bit tracks token readiness, not this pool.
                expert_idx = work_tile.expert_idx
                # The refine publishes one count per column part of each block.
                expert_sf_blocks = ceil_div(expert_sizes[expert_idx], Int32(self.sf_padding_block)) * Int32(
                    fc1_sf_parts_per_block
                )
                spin_wait(fc1_sf_ready_pointer + expert_idx, lambda value: value == expert_sf_blocks, scope="gpu")
            else:
                # FC2 waits for its token block's FC1 tiles, shared FC2 for the shared ones; shared FC1 has no wait.
                if cutlass.const_expr(not self.shared_enabled):
                    counter_slot = work_tile.cumulative_token_block_count + work_tile.tile_n_idx
                    spin_wait(
                        fc1_done_counter_pointer + counter_slot,
                        lambda value: value >= fc2_spin_threshold,
                        sleep_cycles=500,
                        peek_status=work_tile.peek_ready,
                    )
                else:
                    if phase != Int32(BlockPhase.SharedLinear1):
                        counter_slot = work_tile.cumulative_token_block_count + work_tile.tile_n_idx
                        wait_address = (fc1_done_counter_pointer + counter_slot).toint()
                        wait_target = fc2_spin_threshold
                        if phase == Int32(BlockPhase.SharedLinear2):
                            wait_address = (shared_fc1_done_counter_pointer + work_tile.tile_n_idx).toint()
                            wait_target = shared_fc2_spin_threshold
                        spin_wait(
                            cute.make_ptr(Int32, wait_address, AddressSpace.gmem, assumed_align=4),
                            lambda value: value >= wait_target,
                            sleep_cycles=500,
                            peek_status=work_tile.peek_ready,
                        )
            iket.range_pop()

            if cutlass.const_expr(self.heterogeneous_shared):
                if self._is_shared_phase(phase):
                    shared_kind = (
                        self.shared_tiled_mma,
                        self.shared_tiled_mma_sfb,
                        self.shared_mma_tiler_mnk,
                        self.shared_mma_tiler_sfb,
                        shared_fc1_tma_b_tensor,
                        shared_fc1_tma_b_atom,
                        shared_fc1_tma_sfb_tensor,
                        shared_fc1_tma_sfb_atom,
                        self.shared_b_smem_tensor,
                        self.shared_sfb_smem_tensor,
                        shared_kernel_extension,
                    )
                    ab_pipeline_state = self._issue_tma_b_tile(
                        shared_kind,
                        self._select_shared_phase(phase, shared_fc1_b_exec_atom, shared_fc2_b_exec_atom),
                        self._select_shared_phase(phase, shared_fc1_sfb_exec_atom, shared_fc2_sfb_exec_atom),
                        b_cta_layout,
                        sfb_cta_layout,
                        b_multicast_mask,
                        sfb_multicast_mask,
                        work_tile,
                        k_tile_count,
                        ab_pipeline,
                        ab_pipeline_state,
                        None,
                        Int32(self.shared_token_tx_count),
                    )
                else:
                    ab_pipeline_state = self._issue_tma_b_tile(
                        routed_kind,
                        fc2_b_exec_atom,
                        self._select_by_phase(phase, fc1_sfb_exec_atom, fc2_sfb_exec_atom),
                        b_cta_layout,
                        sfb_cta_layout,
                        b_multicast_mask,
                        sfb_multicast_mask,
                        work_tile,
                        k_tile_count,
                        ab_pipeline,
                        ab_pipeline_state,
                        is_fc1,
                    )
            else:
                # Routed FC1 loads no B here (Gather-B does); the FC2 atom is a placeholder for it.
                ab_pipeline_state = self._issue_tma_b_tile(
                    routed_kind,
                    self._select_by_phase(phase, fc2_b_exec_atom, fc2_b_exec_atom, routed_shared_b, routed_shared_b2),
                    self._select_by_phase(
                        phase, fc1_sfb_exec_atom, fc2_sfb_exec_atom, routed_shared_sfb, routed_shared_sfb2
                    ),
                    b_cta_layout,
                    sfb_cta_layout,
                    b_multicast_mask,
                    sfb_multicast_mask,
                    work_tile,
                    k_tile_count,
                    ab_pipeline,
                    ab_pipeline_state,
                    is_fc1,
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
        routed_input_ready_flag: Optional[ReadyFlag] = None,
        routed_metadata_ready_flag: Optional[ReadyFlag] = None,
    ) -> None:
        """Gather dense-order FC1 activations directly into B SMEM.

        Given the routed ready flags (lazy-wait builds), it waits for payload and metadata on its first FC1 tile.

        Every CTA of the B multicast group issues its share and multicasts it to
        the whole group, so the group mask is the ordinary ``"b"`` mask each CTA
        already computes for itself -- no leader election. Ownership is
        interleaved by gather4 group so that a short row window spreads over all
        issuing units rather than landing entirely on group zero.

        Gather-B contributes transaction completions only; it is not an AB
        producer. FC2 is fully produced by TMA-A and TMA-B, so on those tiles this
        warp only keeps its register-held pipeline state aligned.
        """
        row_ids = cute.make_rmem_tensor(cute.make_layout((self.gather_b_rows_per_unit,)), cutlass.Int32)
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
                self.cluster_shape_mn, None, self.cta_coord_in_cluster, None, self.use_2cta_instrs, "b"
            )
        issuing_unit = self.main_vmnk_coord[1] * Int32(self.num_gather_b_warps) + gather_warp_idx
        # Base of this warp's padding row ids: 1 + cta_id * tile_M + warp_id * 32, negated per row.
        linear_cta_idx = cute.arch.block_idx_in_cluster() + cute.arch.block_idx()[2] * Int32(
            self.cluster_shape_mn[0] * self.cluster_shape_mn[1]
        )
        padding_id_base = (
            Int32(1)
            + linear_cta_idx * Int32(self.gather_padding_cta_stride)
            + gather_warp_idx * Int32(self.gather_padding_warp_stride)
        )

        routed_inputs_ready = Int32(0)
        work_tile = sched_consumer.consume_work()
        while work_tile.is_valid_tile:
            is_fc1 = work_tile.phase == Int32(BlockPhase.Linear1)
            if is_fc1:
                iket.range_push("tma_gather_b1_fc1")
                if cutlass.const_expr(routed_input_ready_flag is not None):
                    if routed_inputs_ready == Int32(0):
                        iket.range_push("gather_b1.wait_input_ready")
                        wait_ready(routed_input_ready_flag)
                        wait_ready(routed_metadata_ready_flag)
                        iket.range_pop()
                        routed_inputs_ready = Int32(1)
                tile_pool_base = work_tile.cumulative_data_physical_row + work_tile.tile_n_idx * Int32(self.cta_tile_n)
                # The MMA issues a dynamic instruction N, so under two-CTA
                # instructions the pair splits that runtime extent rather than the
                # static tile: the non-leader's local row zero is the midpoint of
                # the aligned token count. This mirrors what
                # `compute_non_leader_cta_load_shift` does for the tiled TMA legs.
                # Rounding to sixteen keeps that midpoint a multiple of eight, so
                # every run stays aligned to its 256-bit index load.
                cta_row_base = self.mma_cta_index * (round_up(work_tile.valid_tokens_in_cta_tile, 16) // Int32(2))
                cta_valid_rows = work_tile.valid_tokens_in_cta_tile - cta_row_base
                index_runs = cute.zipped_divide(
                    cute.domain_offset((tile_pool_base + cta_row_base,), gather_index), (self.gather_index_load_rows,)
                )
                for local_run in cutlass.range_constexpr(self.gather_b_runs_per_unit):
                    run_index = Int32(local_run * self.gather_issuing_units) + issuing_unit
                    cute.copy(
                        index_load_atom,
                        mark_alignment(index_runs[(None, run_index)], index_load_bytes),
                        row_id_runs[(None, local_run)],
                    )
                    # Rows past the tile's tokens hold stale ids: give each a distinct out-of-bounds id.
                    # Equal ids, valid or not, make the padding gathers contend on one address.
                    run_row = run_index * Int32(self.gather_index_load_rows)
                    for row_in_run in cutlass.range_constexpr(self.gather_index_load_rows):
                        if run_row + Int32(row_in_run) >= cta_valid_rows:
                            row_ids[local_run * self.gather_index_load_rows + row_in_run] = Int32(0) - (
                                padding_id_base + Int32(local_run * self.gather_index_load_rows + row_in_run)
                            )

                for k_tile_idx in cutlass.range(self.fc1_k_tile_count, unroll=1):
                    # TMA-B has already acquired this stage and contributed its
                    # stage-specific transaction expectation. The named barrier orders that
                    # arrive-and-expect-tx before Gather4.
                    barrier_sync_unaligned(self.fc1_gather_sync_barrier_id, 32 * (1 + self.num_gather_b_warps))
                    tma_barrier = ab_pipeline.producer_get_barrier(ab_pipeline_state)
                    stage_index = ab_pipeline_state.index
                    col_k = k_tile_idx * Int32(self.mma_tile_k)
                    # The planned stage stride, which is wider than one gathered tile when a shared kind pads B.
                    stage_base_elements = stage_index * Int32(self.b_smem_region.stride[-1])

                    for local_run in cutlass.range_constexpr(self.gather_b_runs_per_unit):
                        run_index = Int32(local_run * self.gather_issuing_units) + issuing_unit
                        issue_rows = True
                        if cutlass.const_expr(self.dynamic_gather):
                            rows_per_cta = round_up(work_tile.valid_tokens_in_cta_tile, 16) // Int32(self.mma_cta_count)
                            issue_rows = run_index * Int32(self.gather_index_load_rows) < rows_per_cta
                        if issue_rows:
                            run_base_elements = stage_base_elements + run_index * Int32(
                                self.gather_index_load_rows * self.mma_tile_k
                            )
                            for group_in_run in cutlass.range_constexpr(self.gather_b_groups_per_run):
                                destination_offset = run_base_elements + Int32(
                                    group_in_run * self.gather4_width * self.mma_tile_k
                                )
                                row_start = local_run * self.gather_index_load_rows + group_in_run * self.gather4_width
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
                # FC2 and shared tiles are produced by TMA-A/TMA-B; only keep this state aligned.
                iket.range_push("tma_gather_b1_fc2_state_advance")
                skipped_k_tiles = self._phase_k_tile_count(work_tile.phase)
                advanced_index = ab_pipeline_state.index + skipped_k_tiles
                stage_count = Int32(ab_pipeline_state.stages)
                wrap_count = advanced_index // stage_count
                ab_pipeline_state._index = advanced_index % stage_count
                ab_pipeline_state._phase = ab_pipeline_state.phase ^ (wrap_count & Int32(1))
                ab_pipeline_state._count += skipped_k_tiles
                iket.range_pop()

            work_tile = sched_consumer.consume_work()

    @cute.jit
    def _phase_k_tile_count(self, phase: Int32) -> Int32:
        """K tiles of one work tile: FC1 over hidden, FC2 over the down-projection, shared likewise."""
        shared = self.shared_enabled
        return self._select_by_phase(
            phase,
            Int32(self.fc1_k_tile_count),
            Int32(self.fc2_k_tile_count),
            Int32(self.shared_fc1_k_tile_count) if shared else None,
            Int32(self.shared_fc2_k_tile_count) if shared else None,
        )

    @cute.jit
    def _mma_use_n128_sfb_mapping(self, quant_kind: QuantKind, work_tile):
        """At tile N 256 an NVFP4 dynamic extent of 128 or less makes the SFB TMEM window compact."""
        use_n128_sfb_mapping = False
        if cutlass.const_expr(quant_kind == QuantKind.nvfp4 and self.cta_tile_n == 256):
            use_n128_sfb_mapping = round_up(work_tile.valid_tokens_in_cta_tile, 16) <= Int32(128)
        return use_n128_sfb_mapping

    @cute.jit
    def _mma_tile_single(
        self,
        quant_kind: QuantKind,
        mma_instruction_mnk: tuple,
        views: tuple,
        accumulator: cute.Tensor,
        work_tile,
        k_tile_count: Int32,
        ab_pipeline: pipeline.PipelineAsync,
        ab_pipeline_state: pipeline.PipelineState,
        token_pipeline=None,
        token_pipeline_state=None,
    ) -> Tuple[pipeline.PipelineState, pipeline.PipelineState]:
        """One tile's k loop over a single pipeline, prefetching the next stage's readiness (token arguments unused)."""
        (
            fragment_a,
            fragment_b,
            tmem_sfa,
            tmem_sfb,
            _sfb_layout,
            tiled_copy_sfa,
            partitioned_smem_sfa,
            partitioned_tmem_sfa,
            tiled_copy_sfb,
            partitioned_smem_sfb,
            partitioned_tmem_sfb,
        ) = views
        use_n128_sfb_mapping = self._mma_use_n128_sfb_mapping(quant_kind, work_tile)
        valid_tokens_in_cta_tile = work_tile.valid_tokens_in_cta_tile
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
                cute.copy(tiled_copy_sfa, partitioned_smem_sfa[s2t_stage_coord], partitioned_tmem_sfa)
                for instruction_index in cutlass.range_constexpr(self.num_mma_instructions_per_ab_stage):
                    # Rest_Tiler order: N0K0, N128K0, N0K64, N128K64.
                    lower_n_lower_k_source_coord = (None, 0, None, instruction_index, ab_pipeline_state.index)
                    lower_n_lower_k_destination_coord = (None, 0, None, instruction_index)
                    cute.copy(
                        tiled_copy_sfb,
                        partitioned_smem_sfb[lower_n_lower_k_source_coord],
                        partitioned_tmem_sfb[lower_n_lower_k_destination_coord],
                    )
                    lower_n_upper_k_source_coord = (None, 2, None, instruction_index, ab_pipeline_state.index)
                    compact_upper_k_destination_coord = (None, 1, None, instruction_index)
                    cute.copy(
                        tiled_copy_sfb,
                        partitioned_smem_sfb[lower_n_upper_k_source_coord],
                        partitioned_tmem_sfb[compact_upper_k_destination_coord],
                    )

                a_stage_frag = fragment_a[(None, None, None, ab_pipeline_state.index)]
                b_stage_frag = fragment_b[(None, None, None, ab_pipeline_state.index)]
                dynamic_mainloop.issue_dynamic_block_scaled_mma_window(
                    quant_kind=quant_kind,
                    acc_tensor=accumulator,
                    a_window_frag=a_stage_frag,
                    b_window_frag=b_stage_frag,
                    sfa_window_tensor=tmem_sfa,
                    sfb_window_tensor=tmem_sfb,
                    valid_tokens_in_tile=valid_tokens_in_cta_tile,
                    mma_instruction_mnk=mma_instruction_mnk,
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
                cute.copy(tiled_copy_sfa, partitioned_smem_sfa[s2t_stage_coord], partitioned_tmem_sfa)
                cute.copy(tiled_copy_sfb, partitioned_smem_sfb[s2t_stage_coord], partitioned_tmem_sfb)

                a_stage_frag = fragment_a[(None, None, None, ab_pipeline_state.index)]
                b_stage_frag = fragment_b[(None, None, None, ab_pipeline_state.index)]
                dynamic_mainloop.issue_dynamic_block_scaled_mma_window(
                    quant_kind=quant_kind,
                    acc_tensor=accumulator,
                    a_window_frag=a_stage_frag,
                    b_window_frag=b_stage_frag,
                    sfa_window_tensor=tmem_sfa,
                    sfb_window_tensor=tmem_sfb,
                    valid_tokens_in_tile=valid_tokens_in_cta_tile,
                    mma_instruction_mnk=mma_instruction_mnk,
                    window_instruction_offset=0,
                    window_instruction_count=self.num_mma_instructions_per_ab_stage,
                    first_instruction_accumulate=k_tile_idx != 0,
                )

                ab_pipeline.consumer_release(ab_pipeline_state)
                ab_pipeline_state.advance()
                next_ab_pipeline_state.advance()
                full_status = next_full_status
        return ab_pipeline_state, token_pipeline_state

    @cute.jit
    def _mma_tile_split(
        self,
        quant_kind: QuantKind,
        mma_instruction_mnk: tuple,
        views: tuple,
        accumulator: cute.Tensor,
        work_tile,
        k_tile_count: Int32,
        weight_pipeline: pipeline.PipelineAsync,
        weight_pipeline_state: pipeline.PipelineState,
        token_pipeline: pipeline.PipelineAsync,
        token_pipeline_state: pipeline.PipelineState,
    ) -> Tuple[pipeline.PipelineState, pipeline.PipelineState]:
        """``_mma_tile_single`` for the split-depth plan: A/SFA follow the weight state, B/SFB the token state."""
        (
            fragment_a,
            fragment_b,
            tmem_sfa,
            tmem_sfb,
            _sfb_layout,
            tiled_copy_sfa,
            partitioned_smem_sfa,
            partitioned_tmem_sfa,
            tiled_copy_sfb,
            partitioned_smem_sfb,
            partitioned_tmem_sfb,
        ) = views
        use_n128_sfb_mapping = self._mma_use_n128_sfb_mapping(quant_kind, work_tile)
        valid_tokens_in_cta_tile = work_tile.valid_tokens_in_cta_tile
        for k_tile_idx in cutlass.range(k_tile_count, unroll=1):
            weight_pipeline.consumer_wait(weight_pipeline_state)
            token_pipeline.consumer_wait(token_pipeline_state)

            weight_stage_coord = (None, None, None, None, weight_pipeline_state.index)
            cute.copy(tiled_copy_sfa, partitioned_smem_sfa[weight_stage_coord], partitioned_tmem_sfa)
            if use_n128_sfb_mapping:
                for instruction_index in cutlass.range_constexpr(self.num_mma_instructions_per_ab_stage):
                    # Rest_Tiler order: N0K0, N128K0, N0K64, N128K64.
                    lower_n_lower_k_source_coord = (None, 0, None, instruction_index, token_pipeline_state.index)
                    cute.copy(
                        tiled_copy_sfb,
                        partitioned_smem_sfb[lower_n_lower_k_source_coord],
                        partitioned_tmem_sfb[(None, 0, None, instruction_index)],
                    )
                    lower_n_upper_k_source_coord = (None, 2, None, instruction_index, token_pipeline_state.index)
                    cute.copy(
                        tiled_copy_sfb,
                        partitioned_smem_sfb[lower_n_upper_k_source_coord],
                        partitioned_tmem_sfb[(None, 1, None, instruction_index)],
                    )
            else:
                token_stage_coord = (None, None, None, None, token_pipeline_state.index)
                cute.copy(tiled_copy_sfb, partitioned_smem_sfb[token_stage_coord], partitioned_tmem_sfb)

            a_stage_frag = fragment_a[(None, None, None, weight_pipeline_state.index)]
            b_stage_frag = fragment_b[(None, None, None, token_pipeline_state.index)]
            dynamic_mainloop.issue_dynamic_block_scaled_mma_window(
                quant_kind=quant_kind,
                acc_tensor=accumulator,
                a_window_frag=a_stage_frag,
                b_window_frag=b_stage_frag,
                sfa_window_tensor=tmem_sfa,
                sfb_window_tensor=tmem_sfb,
                valid_tokens_in_tile=valid_tokens_in_cta_tile,
                mma_instruction_mnk=mma_instruction_mnk,
                window_instruction_offset=0,
                window_instruction_count=self.num_mma_instructions_per_ab_stage,
                first_instruction_accumulate=k_tile_idx != 0,
            )

            weight_pipeline.consumer_release(weight_pipeline_state)
            token_pipeline.consumer_release(token_pipeline_state)
            weight_pipeline_state.advance()
            token_pipeline_state.advance()
        return weight_pipeline_state, token_pipeline_state

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
        """Execute SM107 MMA with an instruction N that follows the live token count (shared stage plan)."""
        self._run_mma(tmem_allocator, ab_pipeline, ab_pipeline_state, None, None, acc_pipeline, sched_consumer)

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
        """``run_mma`` for the split-depth plan: A/SFA follow the weight pipeline, B/SFB the token pipeline."""
        self._run_mma(
            tmem_allocator,
            weight_pipeline,
            weight_pipeline_state,
            token_pipeline,
            token_pipeline_state,
            acc_pipeline,
            sched_consumer,
        )

    @cute.jit
    def _run_mma(
        self,
        tmem_allocator,
        ab_pipeline: pipeline.PipelineAsync,
        ab_pipeline_state: pipeline.PipelineState,
        token_pipeline: Optional[pipeline.PipelineAsync],
        token_pipeline_state: Optional[pipeline.PipelineState],
        acc_pipeline: pipeline.PipelineAsync,
        sched_consumer: SchedulerConsumer,
    ) -> None:
        """Common MMA warp body; a heterogeneous shared expert's tiles use the shared views on the same pipelines."""
        mma_tile = self._mma_tile_split if self.uses_asymmetric_ab_stages else self._mma_tile_single
        tiled_mma = self.make_tiled_mma()
        barrier_sync_unaligned(self.tmem_allocation_barrier_id, 32 * (1 + self.num_accumulator_consumer_warps_per_cta))
        accumulator_pointer = tmem_allocator.retrieve_ptr(self.acc_dtype)
        accumulator_layout = cute.make_layout(self.accumulator_shape, stride=self.accumulator_stride)
        accumulator_base = cute.make_tensor(accumulator_pointer, accumulator_layout)
        views = self._mma_operand_views(
            (
                tiled_mma,
                self.mma_tiler_mnk,
                self.sf_vec_size,
                self.sf_dtype,
                self.a_smem_tensor,
                self.b_smem_tensor,
                self.sfa_smem_tensor,
                self.sfb_smem_tensor,
            ),
            accumulator_pointer,
        )
        if cutlass.const_expr(self.heterogeneous_shared):
            shared_views = self._mma_operand_views(
                (
                    self.make_tiled_mma(shared=True),
                    self.shared_mma_tiler_mnk,
                    self.shared_sf_vec_size,
                    self.shared_sf_dtype,
                    self.shared_a_smem_tensor,
                    self.shared_b_smem_tensor,
                    self.shared_sfa_smem_tensor,
                    self.shared_sfb_smem_tensor,
                ),
                accumulator_pointer,
            )

        acc_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.num_accumulator_pipeline_stages
        )
        work_tile = sched_consumer.consume_work()
        while work_tile.is_valid_tile:
            k_tile_count = self._phase_k_tile_count(work_tile.phase)
            accumulator_stage_index = acc_producer_state.index

            if self.is_leader_cta:
                accumulator = accumulator_base[(None, None, accumulator_stage_index)]
                accumulator = cute.tiled_divide(accumulator, accumulator.shape)
                routed_views = views
                if cutlass.const_expr(self.heterogeneous_shared):
                    tile_shared_views = shared_views
                if cutlass.const_expr(self.cta_tile_n == 64):
                    # The scale-factor atom is ((32, 4), (sf_vec, 4)) : ((16, 4), (0, 1)) bytes
                    # (BlockScaledBasicChunk): a 16-byte row holds one token of each of the four
                    # 32-token groups, four K blocks per token, and the S2T copy puts each group
                    # in its own 32-bit TMEM column. Every 4-column chunk (one per four K blocks)
                    # therefore has group g of the 128-token SF tile in column g, so the second
                    # 64 tokens (groups 2 and 3) start 2 columns in, for any K and sf_vec. Only
                    # the MMA's SFB window moves; the S2T copy still fills the unshifted base.
                    # Shared tiles keep the routed tile N (and the SFB TMEM base), so a
                    # heterogeneous shared kind shifts its own window the same way.
                    sfb_shift = (work_tile.tile_n_idx % Int32(2)) * Int32(2)
                    sfb_column = accumulator_pointer + self.num_accumulator_tmem_cols + self.num_sfa_tmem_cols + sfb_shift
                    routed_views = (
                        *views[:3],
                        cute.make_tensor(cute.recast_ptr(sfb_column, dtype=self.sf_dtype), views[4]),
                        *views[4:],
                    )
                    if cutlass.const_expr(self.heterogeneous_shared):
                        tile_shared_views = (
                            *shared_views[:3],
                            cute.make_tensor(cute.recast_ptr(sfb_column, dtype=self.shared_sf_dtype), shared_views[4]),
                            *shared_views[4:],
                        )
                acc_pipeline.producer_acquire(acc_producer_state)
                if cutlass.const_expr(self.heterogeneous_shared):
                    if self._is_shared_phase(work_tile.phase):
                        ab_pipeline_state, token_pipeline_state = mma_tile(
                            self.shared_quant_kind,
                            self.shared_mma_instruction_mnk,
                            tile_shared_views,
                            accumulator,
                            work_tile,
                            k_tile_count,
                            ab_pipeline,
                            ab_pipeline_state,
                            token_pipeline,
                            token_pipeline_state,
                        )
                    else:
                        ab_pipeline_state, token_pipeline_state = mma_tile(
                            self.quant_kind,
                            self.mma_instruction_mnk,
                            routed_views,
                            accumulator,
                            work_tile,
                            k_tile_count,
                            ab_pipeline,
                            ab_pipeline_state,
                            token_pipeline,
                            token_pipeline_state,
                        )
                else:
                    ab_pipeline_state, token_pipeline_state = mma_tile(
                        self.quant_kind,
                        self.mma_instruction_mnk,
                        routed_views,
                        accumulator,
                        work_tile,
                        k_tile_count,
                        ab_pipeline,
                        ab_pipeline_state,
                        token_pipeline,
                        token_pipeline_state,
                    )
                acc_pipeline.producer_commit(acc_producer_state)

            acc_producer_state.advance()
            work_tile = sched_consumer.consume_work()
        acc_pipeline.producer_tail(acc_producer_state)


__all__ = ["BlockScaledSwapAbFc12MainloopGenSpecialized"]
